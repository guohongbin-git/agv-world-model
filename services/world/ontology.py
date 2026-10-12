"""W05 本体与库存事务 — 设计文档第10章实现。

核心约束:
- Pallet 只有一个已确认位置或 UNKNOWN
- GoodsUnit 最多一个活动绑定; pallet 最多一件标准箱
- 装卸在同一事务原子提交绑定+位置+事件
- event_id 唯一, (run_id, robot_id, command_id) 动作唯一
- UNKNOWN 不删除计数; 显式边界事件才离场

G1 状态机 (B3, 故障矩阵 docs/knowledge/paper1_fault_matrix.md):
    PREPARED → EXECUTING → VERIFIED → commit_transfer() → COMMITTED
    任意未闭合阶段可 → RECONCILIATION (结果不确定, 隔离复核) 或
    CANCELLED (复核确认未执行)。任务意图/资源租约/命令 ID 必须在
    产生不可逆物理影响**前**经 prepare_transfer 持久化 (D01/D02);
    重启恢复只信重新观测, 不信故障瞬间的状态写入 (见 twin/server
    recover_open_commands)。

三独立标识 (矩阵「三个独立标识不得混用」, 生成/用途/禁混用):
- command_id  命令标识, **跨重试幂等**。生成于任务受理
  (prepare_transfer); 与 transfers.idempotency_key 同源同值
  (twin 层公式 ``{task_id}:{phase}:{pallet}:{loc}``), 生命周期 =
  命令闭合 (COMMITTED / CANCELLED)。commit_transfer(command_id=…)
  强制校验「已 PREPARED 且与幂等键一致」—— 用 evidence_id 或 run_id
  顶替会被拒绝。
- evidence_id  证据引用 = execution_evidence 主键 =
  EvidenceVerdict.verdict_id (``{run_id}-EVD-xxxx``), 带 valid_until
  有效窗。只作提交凭据, **禁止当幂等键** (过期即作废, S05)。
- run_id  运行级审计标识。事件前缀 / 库文件名 / 重启恢复身份;
  **禁止当命令或证据键**。
"""
from __future__ import annotations

import itertools
import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LocKind(str, Enum):
    SLOT = "SLOT"
    STATION = "STATION"
    ROBOT = "ROBOT"
    GRIPPER = "GRIPPER"
    FEED = "FEED"
    PALLET = "PALLET"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Loc:
    kind: LocKind
    id: str = ""

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.id}" if self.id else self.kind.value

    @staticmethod
    def unknown() -> "Loc":
        return Loc(LocKind.UNKNOWN)


class PalletCargo(str, Enum):
    EMPTY = "EMPTY"
    LOADED = "LOADED"
    TRANSFER_PENDING = "TRANSFER_PENDING"
    UNKNOWN = "UNKNOWN"


class GoodsLifecycle(str, Enum):
    INBOUND = "INBOUND"                # 在入料托台, 不计在库
    IN_WAREHOUSE = "IN_WAREHOUSE"
    OUTBOUND_PENDING = "OUTBOUND_PENDING"
    OFFSITE = "OFFSITE"


class SlotOcc(str, Enum):
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"
    UNKNOWN = "UNKNOWN"


class WorldError(Exception):
    pass


class VersionConflict(WorldError):
    pass


class ConstraintViolation(WorldError):
    pass


@dataclass
class GoodsUnit:
    goods_id: str
    sku: str
    mass_kg: float
    location: Loc
    lifecycle: GoodsLifecycle = GoodsLifecycle.INBOUND
    entity_version: int = 0


@dataclass
class Pallet:
    pallet_id: str
    tare_mass_kg: float
    location: Loc
    cargo_state: PalletCargo = PalletCargo.EMPTY
    entity_version: int = 0


@dataclass
class Slot:
    slot_id: str
    pose_m: tuple[float, float, float]
    occupancy: SlotOcc = SlotOcc.EMPTY
    reservation_id: Optional[str] = None


@dataclass
class Binding:
    goods_id: str
    pallet_id: str
    binding_version: int
    active: bool = True


@dataclass
class Event:
    event_id: str
    task_id: str
    type: str
    payload: dict
    sim_time_s: float


# 乐观版本检查支持的实体 (表名 → 主键列)
_VERSIONED_COLUMNS = {"pallets": "pallet_id", "goods": "goods_id"}


@dataclass(frozen=True)
class TransferOutcome:
    """commit_transfer 的返回 (幂等重放与首次提交同构)。"""
    key: str
    committed: bool
    replayed: bool                     # True = 同 key 重放, 返回首次结果
    event_ids: tuple[str, ...]
    new_versions: dict[str, int]       # 实体 → 提交后版本


# ---------- G1 状态机阶段 (B3) ----------
class TransferStage(str, Enum):
    """prepared_commands.stage 的合法取值 (矩阵 G1 修正版)。

    PREPARED    意图/租约/命令 ID 已持久化, 物理未开始 (D01/D02 边界)
    EXECUTING   物理执行中 (进程存活时于物理开始前推进; 崩溃后此阶段
                只作提示 —— 恢复判定以重新观测为准, D03)
    VERIFIED    EvidenceVerdict 已形成且过门禁 (尚未提交)
    COMMITTED   物理证据与业务状态闭合 (终态)
    RECONCILIATION  结果不确定, 隔离复核 (可转 COMMITTED/CANCELLED)
    CANCELLED   复核确认未执行而安全取消 (终态)
    """
    PREPARED = "PREPARED"
    EXECUTING = "EXECUTING"
    VERIFIED = "VERIFIED"
    COMMITTED = "COMMITTED"
    RECONCILIATION = "RECONCILIATION"
    CANCELLED = "CANCELLED"


# 阶段只进不退; 同阶/降阶调用为 no-op (重试路径安全)
_STAGE_RANK = {TransferStage.PREPARED: 1, TransferStage.EXECUTING: 2,
               TransferStage.VERIFIED: 3, TransferStage.RECONCILIATION: 4,
               TransferStage.COMMITTED: 5, TransferStage.CANCELLED: 5}

_ALLOWED_STAGE_TRANSITIONS = {
    TransferStage.PREPARED: {TransferStage.EXECUTING,
                             TransferStage.COMMITTED,    # 恢复补确认 (D06)
                             TransferStage.CANCELLED},
    TransferStage.EXECUTING: {TransferStage.VERIFIED,
                              TransferStage.COMMITTED,   # 恢复补提交 (D03)
                              TransferStage.RECONCILIATION,
                              TransferStage.CANCELLED},
    TransferStage.VERIFIED: {TransferStage.COMMITTED,
                             TransferStage.RECONCILIATION,
                             TransferStage.CANCELLED},
    TransferStage.RECONCILIATION: {TransferStage.COMMITTED,
                                   TransferStage.CANCELLED},
    TransferStage.COMMITTED: set(),
    TransferStage.CANCELLED: set(),
}

_TERMINAL_STAGES = (TransferStage.COMMITTED, TransferStage.CANCELLED)

@dataclass
class World:
    """单进程世界模型 + SQLite 持久化。所有业务写入只经此事务边界。"""
    run_id: str
    db_path: str = ":memory:"
    _conn: sqlite3.Connection = field(init=False, repr=False)
    _events: dict[str, Event] = field(init=False, repr=False)
    _seq: itertools.count = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # WAL 模式; 多线程访问由调用方串行化 (twin 服务用锁), 读侧短查询
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._events = {}
        # 重启恢复 (B3): 事件序号接续既有库 —— event_id 全局唯一不回绕
        row = self._conn.execute(
            "SELECT MAX(CAST(SUBSTR(event_id, -6) AS INTEGER)) FROM events "
            "WHERE event_id LIKE ?", (f"{self.run_id}-EVT-%",)).fetchone()
        self._seq = itertools.count((row[0] or 0) + 1)

    # ---------- schema ----------
    def _create_tables(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS pallets(pallet_id TEXT PRIMARY KEY, location TEXT,
              cargo_state TEXT, tare_mass_kg REAL, entity_version INTEGER);
            CREATE TABLE IF NOT EXISTS goods(goods_id TEXT PRIMARY KEY, sku TEXT, mass_kg REAL,
              location TEXT, lifecycle TEXT, entity_version INTEGER);
            CREATE TABLE IF NOT EXISTS slots(slot_id TEXT PRIMARY KEY, x_m REAL, y_m REAL, z_m REAL,
              occupancy TEXT, reservation_id TEXT);
            CREATE TABLE IF NOT EXISTS bindings(goods_id TEXT, pallet_id TEXT,
              binding_version INTEGER, active INTEGER,
              PRIMARY KEY(goods_id, pallet_id, binding_version));
            CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, task_id TEXT, type TEXT,
              payload TEXT, sim_time_s REAL);
            CREATE TABLE IF NOT EXISTS command_log(run_id TEXT, robot_id TEXT, command_id TEXT,
              status TEXT, PRIMARY KEY(run_id, robot_id, command_id));
            CREATE TABLE IF NOT EXISTS execution_evidence(evidence_id TEXT PRIMARY KEY,
              schema_version INTEGER, kind TEXT, pallet_id TEXT, vehicle_id TEXT,
              task_id TEXT, payload TEXT, valid_until REAL, sim_time_s REAL);
            CREATE TABLE IF NOT EXISTS transfers(idempotency_key TEXT PRIMARY KEY,
              evidence_id TEXT, result_json TEXT, sim_time_s REAL);
            -- G1 PREPARED 持久化 (B3): 任务意图/资源租约/命令 ID 在任何
            -- 不可逆物理影响前落盘; 重启后由 open_prepared_commands 扫出
            -- 未闭合命令做现场对账 (D02/D03/D06)。
            --   command_id        跨重试幂等命令标识 (= 提交幂等键)
            --   run_id/task_id    审计归属 (恢复时沿用原 run 身份)
            --   stage             TransferStage (状态机阶段)
            --   intent           任务意图 JSON: vehicle/pallet_id/goods_id/
            --                     src/dst/phase + 可选 slot_expect (C02 重验)
            --   resource_lease    资源租约 (如 aisle:F01; 取消时释放)
            --   expected_versions prepare 时刻乐观版本快照 (执行前重验)
            CREATE TABLE IF NOT EXISTS prepared_commands(command_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL, task_id TEXT, stage TEXT NOT NULL,
              intent TEXT NOT NULL, resource_lease TEXT,
              expected_versions TEXT NOT NULL, sim_time_s REAL);
            """
        )
        c.commit()

    # ---------- 注册 ----------
    def register_slot(self, slot: Slot) -> None:
        self._conn.execute(
            "INSERT INTO slots VALUES(?,?,?,?,?,?)",
            (slot.slot_id, *slot.pose_m, slot.occupancy.value, slot.reservation_id))
        self._conn.commit()

    def register_pallet(self, pallet: Pallet) -> None:
        self._conn.execute(
            "INSERT INTO pallets VALUES(?,?,?,?,?)",
            (pallet.pallet_id, str(pallet.location), pallet.cargo_state.value,
             pallet.tare_mass_kg, pallet.entity_version))
        self._conn.commit()

    def register_goods(self, g: GoodsUnit) -> None:
        self._conn.execute(
            "INSERT INTO goods VALUES(?,?,?,?,?,?)",
            (g.goods_id, g.sku, g.mass_kg, str(g.location), g.lifecycle.value, g.entity_version))
        self._conn.commit()

    # ---------- 事件 ----------
    def emit(self, task_id: str, type_: str, payload: dict, sim_time_s: float) -> str:
        event_id = f"{self.run_id}-EVT-{next(self._seq):06d}"
        if event_id in self._events:
            raise ConstraintViolation(f"event_id 重复: {event_id}")
        self._conn.execute("INSERT INTO events VALUES(?,?,?,?,?)",
                           (event_id, task_id, type_, json.dumps(payload), sim_time_s))
        self._conn.commit()
        self._events[event_id] = Event(event_id, task_id, type_, payload, sim_time_s)
        return event_id

    # ---------- 库存事务 ----------
    def bind_and_store(self, goods_id: str, pallet_id: str, station_loc: Loc,
                       task_id: str, sim_time_s: float) -> int:
        """入库装托: 原子 建绑定+G→IN_WAREHOUSE(+位置PAL)+PAL→LOADED。"""
        cur = self._conn.execute("SELECT location, lifecycle, entity_version FROM goods WHERE goods_id=?",
                                 (goods_id,))
        row = cur.fetchone()
        if row is None:
            raise ConstraintViolation(f"货物不存在: {goods_id}")
        loc, lifecycle, gv = row
        if lifecycle != GoodsLifecycle.INBOUND.value:
            raise ConstraintViolation(f"货物生命周期 {lifecycle}, 非INBOUND, 不可装托")
        prow = self._conn.execute(
            "SELECT cargo_state, entity_version FROM pallets WHERE pallet_id=?",
            (pallet_id,)).fetchone()
        if prow is None:
            raise ConstraintViolation(f"托盘不存在: {pallet_id}")
        pstate, pv = prow
        if pstate != PalletCargo.EMPTY.value:
            raise ConstraintViolation(f"托盘状态 {pstate}, 非EMPTY")
        dup = self._conn.execute(
            "SELECT 1 FROM bindings WHERE goods_id=? AND active=1", (goods_id,)).fetchone()
        if dup:
            raise ConstraintViolation("货物已有活动绑定")
        bv = self._next_binding_version(goods_id, pallet_id)
        with self._conn:
            self._conn.execute(
                "INSERT INTO bindings VALUES(?,?,?,1)", (goods_id, pallet_id, bv))
            self._conn.execute(
                "UPDATE goods SET location=?, lifecycle=?, entity_version=entity_version+1 WHERE goods_id=?",
                (f"PALLET:{pallet_id}", GoodsLifecycle.IN_WAREHOUSE.value, goods_id))
            self._conn.execute(
                "UPDATE pallets SET cargo_state=?, location=?, entity_version=entity_version+1 WHERE pallet_id=?",
                (PalletCargo.LOADED.value, str(station_loc), pallet_id))
            if station_loc.kind == LocKind.STATION:
                self._conn.execute(
                    "UPDATE slots SET occupancy=? WHERE slot_id=?",
                    (SlotOcc.OCCUPIED.value, station_loc.id))
        self.emit(task_id, "CargoBindingConfirmed",
                  {"goods_id": goods_id, "pallet_id": pallet_id, "binding_version": bv}, sim_time_s)
        return bv

    def unbind_at_output(self, goods_id: str, pallet_id: str, task_id: str,
                         sim_time_s: float) -> None:
        """卸货接管: 原子 关闭绑定+G→GRIPPER+PAL→EMPTY。"""
        row = self._conn.execute(
            "SELECT binding_version FROM bindings WHERE goods_id=? AND pallet_id=? AND active=1",
            (goods_id, pallet_id)).fetchone()
        if row is None:
            raise ConstraintViolation(f"无活动绑定: {goods_id}/{pallet_id}")
        with self._conn:
            self._conn.execute(
                "UPDATE bindings SET active=0 WHERE goods_id=? AND pallet_id=?",
                (goods_id, pallet_id))
            self._conn.execute(
                "UPDATE goods SET location=?, entity_version=entity_version+1 WHERE goods_id=?",
                ("GRIPPER:M02", goods_id))
            self._conn.execute(
                "UPDATE pallets SET cargo_state=?, entity_version=entity_version+1 WHERE pallet_id=?",
                (PalletCargo.EMPTY.value, pallet_id))
        self.emit(task_id, "CargoUnbound", {"goods_id": goods_id, "pallet_id": pallet_id}, sim_time_s)

    def exit_offsite(self, goods_id: str, task_id: str, sim_time_s: float) -> None:
        """显式离场边界事件: 计数−1。"""
        row = self._conn.execute("SELECT lifecycle FROM goods WHERE goods_id=?",
                                 (goods_id,)).fetchone()
        if row is None or row[0] != GoodsLifecycle.OUTBOUND_PENDING.value:
            raise ConstraintViolation("仅 OUTBOUND_PENDING 可离场")
        with self._conn:
            self._conn.execute(
                "UPDATE goods SET location=?, lifecycle=?, entity_version=entity_version+1 WHERE goods_id=?",
                (Loc(LocKind.FEED, "O-TRAY").__str__(), GoodsLifecycle.OFFSITE.value, goods_id))
        self.emit(task_id, "GoodsExitConfirmed", {"goods_id": goods_id}, sim_time_s)

    # ---------- 入料接管 / 出料放置 / 到出库站 (设计 25.5) ----------
    def take_from_feed(self, goods_id: str, arm_id: str, task_id: str,
                       sim_time_s: float) -> None:
        """入料接管: G 位置 FEED→GRIPPER, 生命周期仍 INBOUND (**不计在库**)。

        设计 25.5 明确: 入料接管后「仍 INBOUND, 不计在库」—— 因此这里
        只改位置, 不动 lifecycle, in_warehouse_count 不变。
        """
        row = self._conn.execute("SELECT lifecycle FROM goods WHERE goods_id=?",
                                 (goods_id,)).fetchone()
        if row is None:
            raise ConstraintViolation(f"货物不存在: {goods_id}")
        if row[0] != GoodsLifecycle.INBOUND.value:
            raise ConstraintViolation(f"货物生命周期 {row[0]}, 非INBOUND")
        with self._conn:
            self._conn.execute(
                "UPDATE goods SET location=?, entity_version=entity_version+1 WHERE goods_id=?",
                (str(Loc(LocKind.GRIPPER, arm_id)), goods_id))
        self.emit(task_id, "FeedPicked",
                  {"goods_id": goods_id, "arm_id": arm_id}, sim_time_s)

    def mark_outbound_pending(self, goods_id: str, task_id: str,
                              sim_time_s: float) -> None:
        """到出库站: G→OUTBOUND_PENDING, **仍计在库** (设计 25.5)。"""
        row = self._conn.execute("SELECT lifecycle FROM goods WHERE goods_id=?",
                                 (goods_id,)).fetchone()
        if row is None or row[0] != GoodsLifecycle.IN_WAREHOUSE.value:
            raise ConstraintViolation("仅 IN_WAREHOUSE 可转 OUTBOUND_PENDING")
        with self._conn:
            self._conn.execute(
                "UPDATE goods SET lifecycle=?, entity_version=entity_version+1 WHERE goods_id=?",
                (GoodsLifecycle.OUTBOUND_PENDING.value, goods_id))
        self.emit(task_id, "OutboundPending", {"goods_id": goods_id}, sim_time_s)

    def place_at_feed(self, goods_id: str, feed_id: str, task_id: str,
                     sim_time_s: float) -> None:
        """出料放置: G 位置 GRIPPER→FEED:O-TRAY, 生命周期保持 OUTBOUND_PENDING。

        设计 25.5: 出料放置「保持 OUTBOUND_PENDING」—— 计数不减,
        直到显式离场边界事件。
        """
        row = self._conn.execute("SELECT lifecycle FROM goods WHERE goods_id=?",
                                 (goods_id,)).fetchone()
        if row is None or row[0] != GoodsLifecycle.OUTBOUND_PENDING.value:
            raise ConstraintViolation("仅 OUTBOUND_PENDING 可放置到出料托台")
        with self._conn:
            self._conn.execute(
                "UPDATE goods SET location=?, entity_version=entity_version+1 WHERE goods_id=?",
                (str(Loc(LocKind.FEED, feed_id)), goods_id))
        self.emit(task_id, "PlacedAtOutputFeed",
                  {"goods_id": goods_id, "feed_id": feed_id}, sim_time_s)

    def move_pallet(self, pallet_id: str, to: Loc, task_id: str, sim_time_s: float) -> None:
        """整托运输: 只动 PAL 位置; 绑定货物的有效位置仍是 PALLET。"""
        with self._conn:
            self._conn.execute(
                "UPDATE pallets SET location=?, entity_version=entity_version+1 WHERE pallet_id=?",
                (str(to), pallet_id))
        self.emit(task_id, "PalletMoved",
                  {"pallet_id": pallet_id, "to_location": str(to)}, sim_time_s)

    # ---------- 查询 ----------
    def in_warehouse_count(self) -> int:
        """在场货物计数 (设计 25.5)。

        含 IN_WAREHOUSE **与** OUTBOUND_PENDING —— 设计「到出库站」行明写
        「G→OUTBOUND_PENDING 仍计在库」, 「卸货接管」与「出料放置」两行
        也都标「计数不变」。只有「货物离场」才 −1。只数 IN_WAREHOUSE 会
        在到出库站那一刻凭空少一件, 后面三段全是错的基线。
        """
        return self._conn.execute(
            "SELECT COUNT(*) FROM goods WHERE lifecycle IN (?,?)",
            (GoodsLifecycle.IN_WAREHOUSE.value,
             GoodsLifecycle.OUTBOUND_PENDING.value)).fetchone()[0]

    def lifecycle_of(self, goods_id: str) -> str:
        """货物当前生命周期 (回放前置判定用)。"""
        row = self._conn.execute("SELECT lifecycle FROM goods WHERE goods_id=?",
                                 (goods_id,)).fetchone()
        if row is None:
            raise KeyError(goods_id)
        return row[0]

    def goods_location(self, goods_id: str) -> str:
        return self._conn.execute(
            "SELECT location FROM goods WHERE goods_id=?", (goods_id,)).fetchone()[0]

    def active_binding(self, goods_id: str) -> Optional[tuple[str, int]]:
        row = self._conn.execute(
            "SELECT pallet_id, binding_version FROM bindings WHERE goods_id=? AND active=1",
            (goods_id,)).fetchone()
        return (row[0], row[1]) if row else None

    def pallet(self, pallet_id: str) -> Optional[tuple[str, str, int]]:
        return self._conn.execute(
            "SELECT location, cargo_state, entity_version FROM pallets WHERE pallet_id=?",
            (pallet_id,)).fetchone()

    def _next_binding_version(self, goods_id: str, pallet_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(binding_version) FROM bindings WHERE goods_id=? AND pallet_id=?",
            (goods_id, pallet_id)).fetchone()
        return (row[0] or 0) + 1

    # ---------- 幂等命令 ----------
    def record_command(self, robot_id: str, command_id: str, status: str) -> str | None:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO command_log VALUES(?,?,?,?)",
                    (self.run_id, robot_id, command_id, status))
            return None
        except sqlite3.IntegrityError:
            prev = self._conn.execute(
                "SELECT status FROM command_log WHERE run_id=? AND robot_id=? AND command_id=?",
                (self.run_id, robot_id, command_id)).fetchone()[0]
            return prev

    # ---------- 原子转移 (GPT 评审 P0-1/P1-4) ----------
    def entity_version_of(self, entity: str) -> int:
        """读取实体当前版本 (乐观并发检查用)。entity 形如 'pallets:PAL-001'。"""
        table, _, eid = entity.partition(":")
        col = _VERSIONED_COLUMNS.get(table)
        if col is None:
            raise ConstraintViolation(f"不支持版本化实体: {entity}")
        row = self._conn.execute(
            f"SELECT entity_version FROM {table} WHERE {col}=?", (eid,)).fetchone()
        if row is None:
            raise ConstraintViolation(f"实体不存在: {entity}")
        return row[0]

    def commit_transfer(self, evidence_id: str, expected_versions: dict[str, int],
                        mutations: list[tuple[str, tuple]],
                        events: list[tuple[str, dict]],
                        idempotency_key: str, *, task_id: str = "",
                        sim_time_s: float = 0.0,
                        evidence_json: str = "",
                        command_id: str = "") -> TransferOutcome:
        """原子提交一次物理转移: 单一 SQLite 事务内完成
        乐观版本检查 → 状态变更 → 事件写入 → 证据快照登记 → 幂等登记。

        - 任何一步失败整体回滚, 库存/事件/登记**零残留**; 调用方在物理
          已成功而提交失败时应进入 RECONCILIATION_REQUIRED (twin/server)。
        - 期望版本不符 → :class:`VersionConflict`, 拒绝且零残留。
        - 同 ``idempotency_key`` 重放直接返回首次结果 (不重复记账、
          不重复发事件) —— 应答丢失/重试安全。
        - ``evidence_json`` 为 :class:`services.world.evidence.EvidenceVerdict`
          序列化; 过期 (sim_time_s > valid_until) → ConstraintViolation。
        - G1 (B3): ``command_id`` 是跨重试幂等的命令标识, **必须与
          ``idempotency_key`` 一致**且已经 :meth:`prepare_transfer`
          PREPARED —— 用 evidence_id / run_id 顶替或未预登记均拒绝
          (三标识分离)。提交成功时在同一事务内把命令阶段推进到
          COMMITTED 并写 CommandStage 审计事件 (状态+事件+幂等记录
          同生共死, I3)。
        """
        if command_id:
            if command_id != idempotency_key:
                raise ConstraintViolation(
                    f"command_id 与幂等键不一致 (三标识混用): "
                    f"{command_id!r} ≠ {idempotency_key!r}")
            prow = self._conn.execute(
                "SELECT stage FROM prepared_commands WHERE command_id=?",
                (command_id,)).fetchone()
            if prow is None:
                raise ConstraintViolation(
                    f"命令未 PREPARED, 拒绝提交: {command_id!r} "
                    "(command_id ≠ evidence_id/run_id, 不得混用)")
            if TransferStage(prow[0]) is TransferStage.CANCELLED:
                raise ConstraintViolation(f"命令已取消, 拒绝提交: {command_id}")
        prev = self._conn.execute(
            "SELECT result_json FROM transfers WHERE idempotency_key=?",
            (idempotency_key,)).fetchone()
        if prev is not None:
            # D06 重试: 返回首次结果, 零重复事件/记账; 顺手闭合尚开放
            # 的命令阶段 (首次提交未携带 command_id 的旧路径) —— 静默,
            # 不产生新事件。
            if command_id:
                self._conn.execute(
                    "UPDATE prepared_commands SET stage=?, sim_time_s=? "
                    "WHERE command_id=? AND stage NOT IN (?,?)",
                    (TransferStage.COMMITTED.value, sim_time_s, command_id,
                     TransferStage.COMMITTED.value,
                     TransferStage.CANCELLED.value))
                self._conn.commit()
            r = json.loads(prev[0])
            return TransferOutcome(key=idempotency_key, committed=True,
                                   replayed=True,
                                   event_ids=tuple(r["event_ids"]),
                                   new_versions=dict(r["new_versions"]))
        if evidence_json:
            try:
                ev = json.loads(evidence_json)
                valid_until = float(ev["valid_until"])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise ConstraintViolation(f"证据不可解析: {e}") from e
            if sim_time_s > valid_until + 1e-9:
                raise ConstraintViolation(
                    f"证据过期: sim_time {sim_time_s:.3f}s > "
                    f"valid_until {valid_until:.3f}s ({evidence_id})")
        expected_versions = expected_versions or {}
        event_ids: list[str] = []
        with self._conn:                       # 原子边界: 异常整体回滚
            for entity, expected in expected_versions.items():
                current = self.entity_version_of(entity)
                if current != expected:
                    raise VersionConflict(
                        f"{entity} 期望 v{expected}, 实际 v{current}")
            for sql, params in mutations:
                self._conn.execute(sql, params)
            for type_, payload in events:
                event_id = f"{self.run_id}-EVT-{next(self._seq):06d}"
                self._conn.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?)",
                    (event_id, task_id, type_, json.dumps(payload), sim_time_s))
                event_ids.append(event_id)
            if evidence_id and evidence_json:
                ev = json.loads(evidence_json)
                self._conn.execute(
                    "INSERT INTO execution_evidence VALUES(?,?,?,?,?,?,?,?,?)",
                    (evidence_id, int(ev.get("schema_version", 0)),
                     str(ev.get("kind", "")), str(ev.get("pallet_id", "")),
                     str(ev.get("vehicle_id", "")), str(ev.get("task_id", "")),
                     evidence_json, float(ev.get("valid_until", 0.0)),
                     sim_time_s))
            new_versions = {entity: self.entity_version_of(entity)
                            for entity in expected_versions}
            self._conn.execute(
                "INSERT INTO transfers VALUES(?,?,?,?)",
                (idempotency_key, evidence_id,
                 json.dumps({"event_ids": event_ids,
                             "new_versions": new_versions}), sim_time_s))
            if command_id:      # G1: 阶段 → COMMITTED 与业务提交同事务
                srow = self._conn.execute(
                    "SELECT stage FROM prepared_commands WHERE command_id=?",
                    (command_id,)).fetchone()
                self._conn.execute(
                    "UPDATE prepared_commands SET stage=?, sim_time_s=? "
                    "WHERE command_id=?",
                    (TransferStage.COMMITTED.value, sim_time_s, command_id))
                stage_eid = f"{self.run_id}-EVT-{next(self._seq):06d}"
                self._conn.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?)",
                    (stage_eid, task_id, "CommandStage",
                     json.dumps({"command_id": command_id,
                                 "from_stage": srow[0] if srow else None,
                                 "to_stage": TransferStage.COMMITTED.value}),
                     sim_time_s))
        for eid, (type_, payload) in zip(event_ids, events):
            self._events[eid] = Event(eid, task_id, type_, payload, sim_time_s)
        return TransferOutcome(key=idempotency_key, committed=True,
                               replayed=False, event_ids=tuple(event_ids),
                               new_versions=new_versions)

    def evidence(self, evidence_id: str) -> Optional[dict]:
        """取回持久化证据快照 (事件 payload 的 evidence_id 可追溯)。"""
        row = self._conn.execute(
            "SELECT payload FROM execution_evidence WHERE evidence_id=?",
            (evidence_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def transfer_recorded(self, idempotency_key: str) -> bool:
        """该幂等键是否已有已提交登记 (恢复路径判定用)。"""
        return self._conn.execute(
            "SELECT 1 FROM transfers WHERE idempotency_key=?",
            (idempotency_key,)).fetchone() is not None

    # ---------- G1 状态机: PREPARED 持久化与重启恢复 (B3) ----------

    @staticmethod
    def _core_intent(intent: dict) -> tuple:
        """意图中定义命令身份的核心字段 (command_id 复用一致性检查)。"""
        return (intent.get("vehicle"), intent.get("pallet_id"),
                intent.get("phase"), intent.get("src"), intent.get("dst"))

    def _stage_event(self, task_id: str, payload: dict,
                     sim_time_s: float) -> None:
        """在已开启的事务内写 CommandStage 审计事件 (调用方持事务)。"""
        event_id = f"{self.run_id}-EVT-{next(self._seq):06d}"
        self._conn.execute(
            "INSERT INTO events VALUES(?,?,?,?,?)",
            (event_id, task_id, "CommandStage", json.dumps(payload), sim_time_s))

    def prepare_transfer(self, command_id: str, *, task_id: str,
                         intent: dict, resource_lease: str,
                         expected_versions: dict[str, int],
                         sim_time_s: float = 0.0) -> dict:
        """G1 PREPARED (矩阵 D01/D02): 任务意图 / 资源租约 / 命令 ID 在
        产生任何不可逆物理影响**前**持久化。

        - 首次: 插入 stage=PREPARED + CommandStage 审计事件 (同一事务)。
        - 幂等重试 (命令未闭合): 核心意图一致时刷新乐观版本快照与租约
          (阶段不回退) —— 重启后重执行的安全前提。
        - 已 COMMITTED: 返回既有记录 (replayed=True), 上层重放提交即
          得首次结果 (D06)。
        - 已 CANCELLED: 拒绝复用 command_id —— 需新命令重新调度。
        - 核心意图不一致: 拒绝 (command_id 不得混用)。
        """
        row = self._conn.execute(
            "SELECT stage, intent FROM prepared_commands WHERE command_id=?",
            (command_id,)).fetchone()
        payload = {"command_id": command_id, "run_id": self.run_id}
        if row is None:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO prepared_commands VALUES(?,?,?,?,?,?,?,?)",
                    (command_id, self.run_id, task_id,
                     TransferStage.PREPARED.value, json.dumps(intent),
                     resource_lease, json.dumps(expected_versions or {}),
                     sim_time_s))
                self._stage_event(task_id,
                                  {**payload, "to_stage":
                                   TransferStage.PREPARED.value,
                                   "intent": intent,
                                   "resource_lease": resource_lease},
                                  sim_time_s)
            return {"command_id": command_id, "stage":
                    TransferStage.PREPARED.value, "replayed": False}
        stage = TransferStage(row[0])
        if stage is TransferStage.CANCELLED:
            raise ConstraintViolation(
                f"命令已取消, 拒绝复用 command_id (需新命令重新调度): "
                f"{command_id}")
        if stage is TransferStage.COMMITTED:
            return {"command_id": command_id,
                    "stage": stage.value, "replayed": True}
        if self._core_intent(json.loads(row[1])) != self._core_intent(intent):
            raise ConstraintViolation(
                f"command_id 复用但核心意图不同 (禁止混用): {command_id}")
        with self._conn:     # 重试: 刷新快照, 阶段保持
            self._conn.execute(
                "UPDATE prepared_commands SET expected_versions=?, "
                "resource_lease=?, sim_time_s=? WHERE command_id=?",
                (json.dumps(expected_versions or {}), resource_lease,
                 sim_time_s, command_id))
        return {"command_id": command_id, "stage": stage.value,
                "replayed": True}

    def advance_stage(self, command_id: str, stage: TransferStage, *,
                      sim_time_s: float = 0.0, note: str = "") -> str:
        """推进命令阶段 (PREPARED→EXECUTING→VERIFIED→…, 只进不退)。

        阶段更新与 CommandStage 审计事件同一事务; 同阶/降阶为 no-op
        (重试路径安全), 非法跃迁 ConstraintViolation。
        """
        row = self._conn.execute(
            "SELECT stage, task_id FROM prepared_commands WHERE command_id=?",
            (command_id,)).fetchone()
        if row is None:
            raise ConstraintViolation(f"命令未 PREPARED: {command_id}")
        cur = TransferStage(row[0])
        if _STAGE_RANK[stage] <= _STAGE_RANK[cur]:
            return cur.value
        if stage not in _ALLOWED_STAGE_TRANSITIONS[cur]:
            raise ConstraintViolation(
                f"非法阶段跃迁 {cur.value} → {stage.value}: {command_id}")
        with self._conn:
            self._conn.execute(
                "UPDATE prepared_commands SET stage=?, sim_time_s=? "
                "WHERE command_id=?", (stage.value, sim_time_s, command_id))
            self._stage_event(row[1],
                              {"command_id": command_id,
                               "from_stage": cur.value,
                               "to_stage": stage.value, "note": note},
                              sim_time_s)
        return stage.value

    def cancel_prepared(self, command_id: str, *, reason: str,
                        sim_time_s: float = 0.0) -> None:
        """安全取消未闭合命令 (D02 恢复: 复核确认物理未执行)。

        账本不动 (未提交过的业务事实零变化), 记 CommandCancelled 事件
        (含取消原因与租约释放信息); 租约的物理侧释放由调用方负责。
        """
        row = self._conn.execute(
            "SELECT stage, task_id, resource_lease FROM prepared_commands "
            "WHERE command_id=?", (command_id,)).fetchone()
        if row is None:
            raise ConstraintViolation(f"命令未 PREPARED: {command_id}")
        cur = TransferStage(row[0])
        if cur in _TERMINAL_STAGES:
            raise ConstraintViolation(
                f"命令已闭合 ({cur.value}), 不可取消: {command_id}")
        with self._conn:
            self._conn.execute(
                "UPDATE prepared_commands SET stage=?, sim_time_s=? "
                "WHERE command_id=?",
                (TransferStage.CANCELLED.value, sim_time_s, command_id))
            self._stage_event(row[1],
                              {"command_id": command_id,
                               "from_stage": cur.value,
                               "to_stage": TransferStage.CANCELLED.value,
                               "reason": reason,
                               "lease_released": row[2]},
                              sim_time_s)

    def revalidate_prepared(self, command_id: str) -> None:
        """执行前重验世界状态 (矩阵 C02): 调度可行 ≠ 执行时可提交。

        对比 prepare 时刻的乐观版本快照与货位占用前置条件
        (intent["slot_expect"]); 不符 → VersionConflict /
        ConstraintViolation —— 拒绝旧计划, 需重新规划。已闭合命令 no-op。
        """
        row = self._conn.execute(
            "SELECT stage, expected_versions, intent FROM prepared_commands "
            "WHERE command_id=?", (command_id,)).fetchone()
        if row is None:
            raise ConstraintViolation(f"命令未 PREPARED: {command_id}")
        if TransferStage(row[0]) in _TERMINAL_STAGES:
            return
        for entity, expected in json.loads(row[1]).items():
            current = self.entity_version_of(entity)
            if current != expected:
                raise VersionConflict(
                    f"旧计划拒绝 (C02): {entity} 计划 v{expected}, "
                    f"当前 v{current} — 世界已变化, 需重新规划 "
                    f"({command_id})")
        for slot, want in ((json.loads(row[2]).get("slot_expect"))
                           or {}).items():
            cur = self._conn.execute(
                "SELECT occupancy FROM slots WHERE slot_id=?",
                (slot,)).fetchone()
            if cur is None or cur[0] != want:
                raise ConstraintViolation(
                    f"旧计划拒绝 (C02): 货位 {slot} 当前 "
                    f"{cur[0] if cur else '不存在'} ≠ 计划 {want} "
                    f"— 执行时重验失败 ({command_id})")

    def open_prepared_commands(self) -> list[dict]:
        """扫描未闭合命令 (I6 可恢复性): 重启恢复的输入, 按创建顺序。"""
        rows = self._conn.execute(
            "SELECT command_id, run_id, task_id, stage, intent, "
            "resource_lease, expected_versions, sim_time_s "
            "FROM prepared_commands WHERE stage NOT IN (?,?) ORDER BY rowid",
            (TransferStage.COMMITTED.value,
             TransferStage.CANCELLED.value)).fetchall()
        return [{"command_id": r[0], "run_id": r[1], "task_id": r[2],
                 "stage": r[3], "intent": json.loads(r[4]),
                 "resource_lease": r[5],
                 "expected_versions": json.loads(r[6]),
                 "sim_time_s": r[7]} for r in rows]
