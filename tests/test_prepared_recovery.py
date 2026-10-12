"""B3 验收 (G1 状态机修正, GPT 评审): PREPARED 阶段 + 三独立标识 +
重启恢复 (重新观测判定) + 断点进程退出测试 D01/D02/D03/D06 (+C02)。

矩阵全文 docs/knowledge/paper1_fault_matrix.md:
- D01 PREPARED 持久化前写入失败 → 不执行物理动作, 零业务变化;
- D02 PREPARED 成功后物理前进程退出 → 重启识别未闭合命令, 确认
  未执行后安全取消/恢复;
- D03 物理完成、BEGIN 前进程退出 → 重启后隔离资源并现场对账
  (重新观测 → 新证据 → 同 command_id 补提交);
- D06 COMMIT 成功但 ACK 丢失 → 重试返回原结果, 不生成重复事件;
- C02 计划后执行前目标货位被占 → 执行时重新验证世界状态, 拒绝旧计划。

进程级退出用 SystemExit 模拟 (BaseException, 不被 run_mission 的
``except Exception`` 捕获 —— 等价进程死亡, 不同于事务内异常);
「重启」= 新 WorldService 实例 + 复用故障前账本文件与物理场景
(数据库与物理世界都不回滚)。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

import json
from services.world.evidence import EvidenceFrame, build_verdict
from services.world.ontology import (ConstraintViolation, Slot, SlotOcc,
                                     TransferStage, VersionConflict, World,
                                     Loc, LocKind, Pallet, PalletCargo,
                                     GoodsUnit, GoodsLifecycle)

TABLES = ("pallets", "goods", "slots", "bindings", "events", "command_log",
          "execution_evidence", "transfers")


# ---------------------------------------------------------------- 夹具/构造
@pytest.fixture()
def world() -> World:
    w = World(run_id="RUN-B3")
    w.register_slot(Slot("A-L3-C1", (0.0, 0.0, 0.400), SlotOcc.OCCUPIED))
    w.register_slot(Slot("A-L2-C1", (0.0, 0.0, 0.250), SlotOcc.OCCUPIED))
    w.register_slot(Slot("A-L1-C1", (0.0, 0.0, 0.100), SlotOcc.OCCUPIED))
    w.register_slot(Slot("H01", (-0.330, 0.0, 0.045), SlotOcc.EMPTY))
    w.register_pallet(Pallet("PAL-001", 0.022, Loc(LocKind.SLOT, "A-L3-C1"),
                             PalletCargo.LOADED))
    w.register_pallet(Pallet("PAL-002", 0.022, Loc(LocKind.SLOT, "A-L1-C1"),
                             PalletCargo.LOADED))
    w.register_goods(GoodsUnit("G-001", "COKE-500ML×24", 0.023,
                               Loc(LocKind.PALLET, "PAL-001"),
                               GoodsLifecycle.IN_WAREHOUSE))
    w._conn.execute("INSERT INTO bindings VALUES('G-001','PAL-001',1,1)")
    w._conn.commit()
    return w


def _frames(n: int = 11, t0: float = 1.0) -> tuple[EvidenceFrame, ...]:
    return tuple(EvidenceFrame(
        t=t0 + i * 0.05, new_support=1.0, old_support=0.0,
        pallet_x=0.1, pallet_y=0.0, pallet_z=0.42, speed=0.0)
        for i in range(n))


def _verdict(w: World, vid: int = 1) -> object:
    return build_verdict(verdict_id=f"{w.run_id}-EVD-{vid:04d}", kind="PICK",
                         task_id="T1", pallet_id="PAL-001", vehicle_id="F01",
                         lease_id="aisle:F01", frames=_frames())


def _pick_call(w: World, v, key: str, *, command_id: str | None = None,
               sim_time_s: float = 1.5):
    kw = {} if command_id is None else {"command_id": command_id}
    return w.commit_transfer(
        v.verdict_id,
        {"pallets:PAL-001": w.entity_version_of("pallets:PAL-001")},
        [("UPDATE pallets SET location=?, entity_version=entity_version+1 "
          "WHERE pallet_id=?", ("ROBOT:F01", "PAL-001")),
         ("UPDATE slots SET occupancy=? WHERE slot_id=?",
          (SlotOcc.EMPTY.value, "A-L3-C1"))],
        [("PalletMoved", {"pallet_id": "PAL-001",
                          "to_location": "ROBOT:F01",
                          "evidence_id": v.verdict_id})],
        key, task_id="T1", sim_time_s=sim_time_s,
        evidence_json=v.to_json(), **kw)


def _prepare(w: World, cmd: str, **over):
    kw = dict(task_id="T1",
              intent={"vehicle": "F01", "pallet_id": "PAL-001",
                      "goods_id": "G-001", "src": "L3", "dst": "S1",
                      "phase": "PICK"},
              resource_lease="aisle:F01",
              expected_versions={"pallets:PAL-001":
                                 w.entity_version_of("pallets:PAL-001")},
              sim_time_s=1.0)
    kw.update(over)
    return w.prepare_transfer(cmd, **kw)


class _FailConn:
    """sqlite 连接代理: 武装后第一条 DML 写入抛 I/O 故障 (D01)。"""

    def __init__(self, real):
        self._real, self.armed = real, False

    def execute(self, sql, *a):
        if self.armed and not sql.strip().upper().startswith("SELECT"):
            raise sqlite3.OperationalError("simulated disk I/O error (D01)")
        return self._real.execute(sql, *a)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False               # 异常自然传播 (真连接会回滚)

    def __getattr__(self, name):
        return getattr(self._real, name)

# ================================================================ World 层
def test_d01_prepare_write_failure_zero_business_change(world):
    """D01: PREPARED 写入失败 → 异常抛出, 零命令登记、零业务变化。"""
    before = {t: world._conn.execute(f"SELECT * FROM {t}").fetchall()
              for t in TABLES}
    proxy = _FailConn(world._conn)
    world._conn = proxy
    proxy.armed = True
    with pytest.raises(sqlite3.OperationalError):
        _prepare(world, "CMD-D01")
    world._conn = proxy._real
    after = {t: world._conn.execute(f"SELECT * FROM {t}").fetchall()
             for t in TABLES}
    assert before == after
    assert world._conn.execute(
        "SELECT COUNT(*) FROM prepared_commands").fetchone()[0] == 0
    assert world.open_prepared_commands() == []


def test_d06_retry_returns_original_result_zero_duplicate(world):
    """D06: COMMIT 成功但调用方 ACK 丢失 → 同 command_id 重试返回原
    结果, 事件/登记零重复; 阶段保持 COMMITTED。"""
    _prepare(world, "CMD-D06")
    out1 = _pick_call(world, _verdict(world), "CMD-D06", command_id="CMD-D06")
    assert out1.committed and not out1.replayed
    row = world._conn.execute(
        "SELECT stage FROM prepared_commands WHERE command_id='CMD-D06'"
    ).fetchone()
    assert row[0] == TransferStage.COMMITTED.value
    n_events = world._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    # ACK 丢失: 调用方按超时处理, 原样重试 (同一 command_id)
    out2 = _pick_call(world, _verdict(world, vid=2), "CMD-D06",
                      command_id="CMD-D06")
    assert out2.replayed is True
    assert out2.event_ids == out1.event_ids
    assert out2.new_versions == out1.new_versions
    assert world._conn.execute(
        "SELECT COUNT(*) FROM events").fetchone()[0] == n_events   # 零新事件
    assert world._conn.execute(
        "SELECT COUNT(*) FROM transfers").fetchone()[0] == 1       # 零重复登记
    assert world._conn.execute(
        "SELECT COUNT(*) FROM events WHERE type='PalletMoved'"
    ).fetchone()[0] == 1
    assert world.open_prepared_commands() == []


def test_three_identifier_separation(world):
    """三独立标识不得混用: evidence_id / run_id 顶替 command_id 或
    command_id ≠ 幂等键一律拒绝; command_id 复用须意图一致。"""
    v = _verdict(world)
    # evidence_id 当 command_id (也未 PREPARED)
    with pytest.raises(ConstraintViolation, match="未 PREPARED"):
        _pick_call(world, v, v.verdict_id, command_id=v.verdict_id)
    # run_id 当 command_id
    with pytest.raises(ConstraintViolation, match="未 PREPARED"):
        _pick_call(world, v, world.run_id, command_id=world.run_id)
    # command_id 与幂等键不一致
    _prepare(world, "CMD-X")
    with pytest.raises(ConstraintViolation, match="不一致"):
        _pick_call(world, v, "CMD-X", command_id="CMD-Y")
    # command_id 复用但核心意图不同
    _prepare(world, "CMD-Z")
    with pytest.raises(ConstraintViolation, match="意图不同"):
        _prepare(world, "CMD-Z",
                 intent={"vehicle": "F02", "pallet_id": "PAL-001",
                         "goods_id": "G-001", "src": "L3", "dst": "S1",
                         "phase": "PICK"})
    # 取消后的 command_id 拒绝复用 (需新命令重新调度)
    world.cancel_prepared("CMD-Z", reason="test", sim_time_s=1.0)
    with pytest.raises(ConstraintViolation, match="已取消"):
        _prepare(world, "CMD-Z")


def test_stage_machine_transitions_enforced(world):
    """阶段只进不退; 非法跃迁拒绝; 终态不可取消。"""
    _prepare(world, "CMD-ST")
    assert world.advance_stage(
        "CMD-ST", TransferStage.EXECUTING) == "EXECUTING"
    # 降阶/同阶 no-op (重试路径安全)
    assert world.advance_stage(
        "CMD-ST", TransferStage.PREPARED) == "EXECUTING"
    # 非法跃迁: PREPARED 不可直达 RECONCILIATION (须先 EXECUTING)
    _prepare(world, "CMD-ST2")
    with pytest.raises(ConstraintViolation, match="非法阶段跃迁"):
        world.advance_stage("CMD-ST2", TransferStage.RECONCILIATION)
    # VERIFIED 可安全取消; 终态不可再取消
    world.advance_stage("CMD-ST2", TransferStage.EXECUTING)
    world.advance_stage("CMD-ST2", TransferStage.VERIFIED)
    world.cancel_prepared("CMD-ST2", reason="test", sim_time_s=2.0)
    with pytest.raises(ConstraintViolation, match="不可取消"):
        world.cancel_prepared("CMD-ST2", reason="again", sim_time_s=3.0)
    open_ids = {r["command_id"] for r in world.open_prepared_commands()}
    assert open_ids == {"CMD-ST"}


def test_c02_stale_plan_rejected_on_revalidation(world):
    """C02: 计划后执行前世界变化 (版本被别的命令推进 / 目标货位被占)
    → 执行时重验拒绝旧计划; 世界未变则放行。"""
    _prepare(world, "CMD-C02A", expected_versions={"pallets:PAL-001": 0})
    # 别的命令先动了这颗托盘 (版本 +1) —— 旧计划作废
    _pick_call(world, _verdict(world), "CMD-OTHER")
    with pytest.raises(VersionConflict, match="旧计划拒绝"):
        world.revalidate_prepared("CMD-C02A")
    # 目标货位被占: 旧 place 计划要求 H01 EMPTY
    world._conn.execute("UPDATE slots SET occupancy='OCCUPIED' "
                        "WHERE slot_id='H01'")
    world._conn.commit()
    _prepare(world, "CMD-C02B",
             intent={"vehicle": "F01", "pallet_id": "PAL-001",
                     "goods_id": "G-001", "src": "L3", "dst": "STAGE",
                     "phase": "PLACE",
                     "slot_expect": {"H01": SlotOcc.EMPTY.value}},
             expected_versions={})
    with pytest.raises(ConstraintViolation, match="货位 H01"):
        world.revalidate_prepared("CMD-C02B")
    # 世界与计划一致 → 放行; 已闭合命令重验为 no-op
    world._conn.execute("UPDATE slots SET occupancy='EMPTY' "
                        "WHERE slot_id='H01'")
    world._conn.commit()
    world.revalidate_prepared("CMD-C02B")
    world.advance_stage("CMD-C02B", TransferStage.COMMITTED)
    world.revalidate_prepared("CMD-C02B")


# ================================================================ 服务层 E2E
def _svc(**kw):
    from twin.server import WorldService
    s = WorldService(**kw)
    s.gate.speed = 64.0        # 测试加速: 只解除墙钟配速, 物理步进不变
    return s


def test_e2e_full_stage_progression_and_handoff():
    """E2E: PREPARED→EXECUTING→VERIFIED→COMMITTED 全阶段流转 (事件序列
    可审计) + 双车交接终态与 B1 基线一致。"""
    svc = _svc()
    svc.tasks = list(svc.HANDOFF_TASKS)
    svc.mode = "running"
    svc.run_mission()
    assert svc.mode == "done", svc.step_label
    rows, _ = svc.events_since(0)
    stages = [(json.loads(e[2])["command_id"], json.loads(e[2])["to_stage"])
              for e in rows if e[1] == "CommandStage"]
    pick1 = f"{svc.task_id}:pick:PAL-001:A-L3-C1"
    place1 = f"{svc.task_id}:place:PAL-001:S1"
    pick2 = f"{svc.task_id}:pick:PAL-001:S1"
    place2 = f"{svc.task_id}:place:PAL-001:H01"
    for cmd in (pick1, place1, pick2, place2):
        seq = [to for c, to in stages if c == cmd]
        assert seq == ["PREPARED", "EXECUTING", "VERIFIED", "COMMITTED"], \
            (cmd, seq)
    pc = svc.world._conn.execute(
        "SELECT command_id, stage FROM prepared_commands").fetchall()
    assert len(pc) == 4 and all(s == "COMMITTED" for _, s in pc)
    # 终态与 B1 E2E 一致
    chain = [json.loads(e[2])["to_location"] for e in rows
             if e[1] == "PalletMoved"]
    assert chain == ["ROBOT:F01", "STATION:S1", "ROBOT:F02", "SLOT:H01"]
    b = svc.board()
    pallets = {p[0]: (p[1], p[2]) for p in b["pallets"]}
    goods = {g[0]: (g[2], g[3]) for g in b["goods"]}
    assert pallets["PAL-001"] == ("SLOT:H01", "LOADED")
    assert goods["G-001"][1] == "OUTBOUND_PENDING"
    assert dict(b["slots"])["H01"] == "OCCUPIED"
    assert b["count"] == 3
    assert svc.world.open_prepared_commands() == []


def test_d01_service_prepare_failure_aborts_before_physical():
    """D01 服务级: PREPARED 写失败 → mission 失败退出, 零物理动作、
    零命令登记、账本零变化。"""
    svc = _svc()
    svc.tasks = list(svc.HANDOFF_TASKS)
    before = svc.board()

    def boom(*a, **kw):
        raise sqlite3.OperationalError("simulated disk I/O error (D01)")
    orig = World.prepare_transfer
    World.prepare_transfer = boom
    try:
        svc.mode = "running"
        svc.run_mission()
    finally:
        World.prepare_transfer = orig
    assert svc.mode == "failed"
    assert svc.world._conn.execute(
        "SELECT COUNT(*) FROM prepared_commands").fetchone()[0] == 0
    assert svc.board()["pallets"] == before["pallets"]
    rows, _ = svc.events_since(0)
    assert not any(e[1] == "PalletMoved" for e in rows)
    # 物理未动: 车仍在初始车位 (mission 在步骤循环前退出)
    assert svc.cell.vehicle("F01").axle_x == pytest.approx(
        svc.cell.vehicle("F01").start_axle_x, abs=1e-3)


def test_d02_prepared_then_exit_restart_cancels_safely():
    """D02: PREPARED 成功后、物理前进程退出 → 重启识别两条未闭合命令
    (阶段 PREPARED), 重新观测确认未执行 → 安全取消+释放租约, 账本零
    变化、零转移事件。"""
    svc = _svc()
    db, cell = svc.world.db_path, svc.cell
    svc.tasks = list(svc.HANDOFF_TASKS)

    orig = type(svc)._prepare_task

    def crash_after_prepare(self, *a, **kw):
        orig(self, *a, **kw)
        raise SystemExit(2)     # D02: PREPARED 落盘后、物理前进程退出
    type(svc)._prepare_task = crash_after_prepare
    try:
        svc.mode = "running"
        with pytest.raises(SystemExit):
            svc.run_mission()
    finally:
        type(svc)._prepare_task = orig
    # 故障瞬间: 命令已持久化 (PREPARED), 物理未动
    open_before = svc.world.open_prepared_commands()
    assert [r["stage"] for r in open_before] == ["PREPARED", "PREPARED"]
    assert svc.cell.vehicle("F01").axle_x == pytest.approx(
        svc.cell.vehicle("F01").start_axle_x, abs=1e-3)

    # ---- 重启: 新实例 + 复用账本与物理现场, build 内自动恢复
    svc2 = _svc(resume_db=db, resume_cell=cell)
    assert svc2.run_id == svc.run_id          # 审计身份接续
    rep = svc2.last_recovery
    by_phase = {r["phase"]: r for r in rep}
    assert len(rep) == 2
    assert by_phase["PICK"]["decision"] == "cancelled_not_executed"
    assert by_phase["PICK"]["observation"] == "AT_SOURCE"
    assert by_phase["PLACE"]["decision"] == "cancelled_not_executed"
    assert svc2.world.open_prepared_commands() == []
    rows = svc2.world._conn.execute(
        "SELECT command_id, stage FROM prepared_commands").fetchall()
    assert all(s == "CANCELLED" for _, s in rows)
    # 账本零变化 + 零转移事件 + 有取消事件与叙述
    b = svc2.board()
    pallets = {p[0]: (p[1], p[2]) for p in b["pallets"]}
    assert pallets["PAL-001"] == ("SLOT:A-L3-C1", "LOADED")
    assert dict(b["slots"])["A-L3-C1"] == "OCCUPIED"
    rows, _ = svc2.events_since(0)
    assert not any(e[1] == "PalletMoved" for e in rows)
    assert sum(1 for e in rows if e[1] == "CommandCancelled") == 2
    assert any("重启复核" in m["text"] and "安全取消" in m["text"]
               for m in svc2.chat)
    # 幂等: 恢复可重复调用, 已闭合命令不再出现
    assert svc2.recover_open_commands() == []
    # 取消后的命令拒绝复用 (需新命令重新调度)
    with pytest.raises(ConstraintViolation, match="已取消"):
        _prepare(svc2.world, f"{svc2.task_id}:pick:PAL-001:A-L3-C1")


def test_d03_physical_done_exit_restart_recommits_with_new_evidence():
    """D03: 物理动作完成、BEGIN 前进程退出 → 重启隔离资源并现场对账:
    重新观测发现托盘在 F01 叉上 → 采新证据过门禁, 同 command_id 补提交;
    place 未执行 → 取消。账实重新闭合。"""
    svc = _svc()
    db, cell = svc.world.db_path, svc.cell
    svc.tasks = list(svc.HANDOFF_TASKS)

    def crash_before_begin(self, src, pid, vid="F01"):
        raise SystemExit(3)     # D03: 物理完成 (举升离梁), BEGIN 前退出
    orig = type(svc)._commit_pick
    type(svc)._commit_pick = crash_before_begin
    try:
        svc.mode = "running"
        with pytest.raises(SystemExit):
            svc.run_mission()
    finally:
        type(svc)._commit_pick = orig
    # 故障瞬间: 阶段 EXECUTING (物理在飞), 账本未动
    open_before = {r["command_id"]: r for r in svc.world.open_prepared_commands()}
    pick_cmd = f"{svc.task_id}:pick:PAL-001:A-L3-C1"
    place_cmd = f"{svc.task_id}:place:PAL-001:S1"
    assert open_before[pick_cmd]["stage"] == "EXECUTING"
    assert not svc.world.transfer_recorded(pick_cmd)
    # 物理事实: 托盘已被抬起、由 F01 叉臂承托 (物理世界不回滚)
    assert svc.cell.pallet_fork_contact("F01")
    assert svc.cell.pallet_bottom_z > svc.LOC["L3"] + 0.003

    # ---- 重启 + 现场对账
    svc2 = _svc(resume_db=db, resume_cell=cell)
    rep = {r["command_id"]: r for r in svc2.last_recovery}
    assert rep[pick_cmd]["decision"] == "committed_new_evidence"
    assert rep[pick_cmd]["observation"] == "ON_ROBOT"
    assert rep[place_cmd]["decision"] == "cancelled_not_executed"
    assert rep[place_cmd]["observation"] == "ON_ROBOT"
    # 账实闭合: 补提交用的是**新**证据 (独立 evidence_id), 同 command_id
    rows, _ = svc2.events_since(0)
    moved = [e for e in rows if e[1] == "PalletMoved"]
    assert len(moved) == 1
    ev_id = json.loads(moved[0][2])["evidence_id"]
    assert ev_id != open_before[pick_cmd].get("evidence_id", "")
    assert ev_id.startswith(svc2.run_id + "-EVD-")
    snap = svc2.world.evidence(ev_id)
    assert snap and snap["kind"] == "PICK" and len(snap["frames"]) == 11
    assert any(e[1] == "CommandRecovered" for e in rows)
    stages = dict(svc2.world._conn.execute(
        "SELECT command_id, stage FROM prepared_commands").fetchall())
    assert stages[pick_cmd] == "COMMITTED"
    assert stages[place_cmd] == "CANCELLED"
    b = svc2.board()
    pallets = {p[0]: (p[1], p[2]) for p in b["pallets"]}
    assert pallets["PAL-001"] == ("ROBOT:F01", "LOADED")
    assert dict(b["slots"])["A-L3-C1"] == "EMPTY"
    assert svc2.world.open_prepared_commands() == []
    assert svc2.recover_open_commands() == []


def test_d06_service_restart_ackloss_closes_without_duplicate():
    """D06 服务级: COMMIT 已落账但调用方未收到 ACK (阶段未闭合) →
    重启恢复识别「已入账」→ 补确认闭合, 零重复事件/零重复登记。"""
    svc = _svc()
    db, cell = svc.world.db_path, svc.cell
    pick_cmd = f"{svc.task_id}:pick:PAL-001:A-L3-C1"
    # 模拟: 命令已 PREPARED, 提交成功落账, 但 ACK 在途中丢失 ——
    # 调用方进程死亡, 阶段未闭合 (旧路径 commit 未带 command_id)。
    _prepare(svc.world, pick_cmd)
    _pick_call(svc.world, _verdict(svc.world), pick_cmd)
    assert svc.world.transfer_recorded(pick_cmd)
    n_events = svc.world._conn.execute(
        "SELECT COUNT(*) FROM events").fetchone()[0]
    assert svc.world.open_prepared_commands()[0]["command_id"] == pick_cmd

    # ---- 重启: 恢复发现「已入账」→ 补确认, 不重复不重放
    svc2 = _svc(resume_db=db, resume_cell=cell)
    rep = svc2.last_recovery
    assert len(rep) == 1 and rep[0]["decision"] == "already_committed"
    assert rep[0]["command_id"] == pick_cmd
    stages = dict(svc2.world._conn.execute(
        "SELECT command_id, stage FROM prepared_commands").fetchall())
    assert stages[pick_cmd] == "COMMITTED"
    # 零重复: 事件数只增加恢复审计事件, PalletMoved 仍恰 1 条, 登记 1 条
    assert svc2.world._conn.execute(
        "SELECT COUNT(*) FROM events WHERE type='PalletMoved'"
    ).fetchone()[0] == 1
    assert svc2.world._conn.execute(
        "SELECT COUNT(*) FROM transfers").fetchone()[0] == 1
    rows, _ = svc2.events_since(0)
    assert any(e[1] == "CommandRecovered" for e in rows)
    assert svc2.world._conn.execute(
        "SELECT COUNT(*) FROM events").fetchone()[0] >= n_events
    assert svc2.world.open_prepared_commands() == []
    assert svc2.recover_open_commands() == []
