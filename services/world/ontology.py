"""W05 本体与库存事务 — 设计文档第10章实现。

核心约束:
- Pallet 只有一个已确认位置或 UNKNOWN
- GoodsUnit 最多一个活动绑定; pallet 最多一件标准箱
- 装卸在同一事务原子提交绑定+位置+事件
- event_id 唯一, (run_id, robot_id, command_id) 动作唯一
- UNKNOWN 不删除计数; 显式边界事件才离场
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
        self._seq = itertools.count(1)

    # ---------- schema ----------
    def _create_tables(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE pallets(pallet_id TEXT PRIMARY KEY, location TEXT,
              cargo_state TEXT, tare_mass_kg REAL, entity_version INTEGER);
            CREATE TABLE goods(goods_id TEXT PRIMARY KEY, sku TEXT, mass_kg REAL,
              location TEXT, lifecycle TEXT, entity_version INTEGER);
            CREATE TABLE slots(slot_id TEXT PRIMARY KEY, x_m REAL, y_m REAL, z_m REAL,
              occupancy TEXT, reservation_id TEXT);
            CREATE TABLE bindings(goods_id TEXT, pallet_id TEXT,
              binding_version INTEGER, active INTEGER,
              PRIMARY KEY(goods_id, pallet_id, binding_version));
            CREATE TABLE events(event_id TEXT PRIMARY KEY, task_id TEXT, type TEXT,
              payload TEXT, sim_time_s REAL);
            CREATE TABLE command_log(run_id TEXT, robot_id TEXT, command_id TEXT,
              status TEXT, PRIMARY KEY(run_id, robot_id, command_id));
            CREATE TABLE execution_evidence(evidence_id TEXT PRIMARY KEY,
              schema_version INTEGER, kind TEXT, pallet_id TEXT, vehicle_id TEXT,
              task_id TEXT, payload TEXT, valid_until REAL, sim_time_s REAL);
            CREATE TABLE transfers(idempotency_key TEXT PRIMARY KEY,
              evidence_id TEXT, result_json TEXT, sim_time_s REAL);
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
                        evidence_json: str = "") -> TransferOutcome:
        """原子提交一次物理转移: 单一 SQLite 事务内完成
        乐观版本检查 → 状态变更 → 事件写入 → 证据快照登记 → 幂等登记。

        - 任何一步失败整体回滚, 库存/事件/登记**零残留**; 调用方在物理
          已成功而提交失败时应进入 RECONCILIATION_REQUIRED (twin/server)。
        - 期望版本不符 → :class:`VersionConflict`, 拒绝且零残留。
        - 同 ``idempotency_key`` 重放直接返回首次结果 (不重复记账、
          不重复发事件) —— 应答丢失/重试安全。
        - ``evidence_json`` 为 :class:`services.world.evidence.EvidenceVerdict`
          序列化; 过期 (sim_time_s > valid_until) → ConstraintViolation。
        """
        prev = self._conn.execute(
            "SELECT result_json FROM transfers WHERE idempotency_key=?",
            (idempotency_key,)).fetchone()
        if prev is not None:
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
