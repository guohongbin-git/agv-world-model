"""T09 网络与服务故障 —— 取托中/落托后/提交前各 5 次。

设计判据:
|T09 网络与服务故障|取托中 落托后 提交前注入故障各 5 次|
     |无重复动作和记账；可重新观测恢复或进入 UNKNOWN|

每个用例必须证明三条 (不是"没抛异常"):
1. **无重复动作** —— 重启重连后不重新下发已发过的物理命令;
2. **无重复记账** —— 在途数量与库存计数在重启前后完全一致;
3. **正确收敛** —— 证据充分则恢复完成, 证据不足则进 UNKNOWN
   (S05 明确允许 UNKNOWN, 但**不允许伪造成功**)。

UNKNOWN 是**合法终态**, 不是失败 —— 故不能断言所有用例都 COMMITTED。
断言的是: 终态 ∈ {COMMITTED, UNKNOWN} 且 UNKNOWN 时计数未变。
"""
from __future__ import annotations

import pytest

from services.scheduler.fault_service import BusinessService, TaskState
from services.world.ontology import GoodsUnit, Loc, LocKind, Pallet, World

REPEATS = 5


def world(run_id: str) -> World:
    w = World(run_id=run_id)
    w.register_pallet(Pallet("PAL-001", 0.045, Loc(LocKind.STATION, "I01")))
    w.register_goods(GoodsUnit("G-001", "A", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    return w


def snapshot(w: World) -> tuple:
    rows = w._conn.execute(
        "SELECT goods_id, location, lifecycle FROM goods ORDER BY goods_id").fetchall()
    pal = w._conn.execute(
        "SELECT pallet_id, location, cargo_state FROM pallets "
        "ORDER BY pallet_id").fetchall()
    return tuple(rows), tuple(pal), w.in_warehouse_count()


# ---------------------------------------------------------------- 取托中
@pytest.mark.parametrize("i", range(REPEATS))
def test_t09_fault_during_pick_no_repeat(i):
    """取托中重启 (S04): 物理继续, 重连后不重复取托, 保留在途数量。"""
    w = world(f"T09-PICK-{i}")
    svc = BusinessService(w)
    svc.issue("TASK-1", "pick", "cmd-pick-1")
    assert svc.tasks["TASK-1"].issued

    svc.restart()                       # 业务服务重启, 物理继续
    assert svc.state.value == "DOWN"

    # 重连: 物理已完成取托 → 收敛为 HELD; 不重新下发
    st = svc.reconnect("TASK-1", physically_done=True, evidence_ok=True)
    assert st is TaskState.HELD
    rec = svc.tasks["TASK-1"]
    assert rec.issued and rec.command_id == "cmd-pick-1", "命令被改写"
    assert rec.settled_count == 0, "重启本身不应导致提交"
    # 取托尚未提交装托 → 库存不得变化
    assert w.in_warehouse_count() == 0
    assert w.active_binding("G-001") is None


def test_t09_pick_during_fault_five_times_all():
    """取托中注入 5 次, 每次都不重复动作、不重复记账。"""
    for i in range(REPEATS):
        w = world(f"T09-PICKLOOP-{i}")
        svc = BusinessService(w)
        before = snapshot(w)
        svc.issue("TASK-1", "pick", f"cmd-{i}")
        svc.restart()
        svc.reconnect("TASK-1", physically_done=True, evidence_ok=True)
        assert snapshot(w) == before, f"第{i}次库存被重启改动"
        assert svc.tasks["TASK-1"].settled_count == 0
        # 再次下发同一任务: 幂等, 不重复物理动作
        svc.issue("TASK-1", "pick", f"cmd-{i}")
        assert svc.tasks["TASK-1"].command_id == f"cmd-{i}"


# ---------------------------------------------------------------- 落托后
@pytest.mark.parametrize("i", range(REPEATS))
def test_t09_fault_after_drop_no_double_commit(i):
    """落托后重启 (S05): 不重复放托, 不重复增库存。"""
    w = world(f"T09-DROP-{i}")
    svc = BusinessService(w)
    svc.issue("TASK-2", "drop", f"drop-{i}")
    svc.observe("TASK-2", physically_done=True, evidence_ok=True)
    assert svc.tasks["TASK-2"].state is TaskState.HELD

    svc.restart()
    svc.reconnect("TASK-2", physically_done=True, evidence_ok=True)

    calls = []
    assert svc.commit("TASK-2", lambda: calls.append(1), 1.0) is True
    assert svc.commit("TASK-2", lambda: calls.append(1), 1.1) is False
    assert len(calls) == 1, "重复提交生效"


# ---------------------------------------------------------------- 提交前
@pytest.mark.parametrize("i", range(REPEATS))
def test_t09_fault_before_commit_recovers_or_unknown(i):
    """提交前中断 (S05): 证据充分则恢复提交, 不足则 UNKNOWN。

    两种都是合法终态; **不允许**的是「证据不足却提交成功」。
    """
    w = world(f"T09-COMMIT-{i}")
    svc = BusinessService(w)
    svc.issue("TASK-3", "commit", f"c-{i}")
    svc.observe("TASK-3", physically_done=True, evidence_ok=True)
    before = snapshot(w)

    svc.restart()
    evidence_ok = (i % 2 == 0)          # 奇数次证据不足
    st = svc.reconnect("TASK-3", physically_done=True, evidence_ok=evidence_ok)

    calls = []
    committed = svc.commit("TASK-3", lambda: calls.append(1), 2.0)
    if evidence_ok:
        assert st is TaskState.HELD
        assert committed is True and len(calls) == 1
    else:
        assert st is TaskState.UNKNOWN, "证据不足必须进 UNKNOWN"
        assert committed is False, "UNKNOWN 状态不得提交 (无虚假成功)"
        assert len(calls) == 0
        assert snapshot(w) == before, "UNKNOWN 期间库存被改动"


def test_t09_unknown_blocks_commit_even_after_reobservation_fails():
    """UNKNOWN 一旦成立, 盲目提交必须被拒。"""
    w = world("T09-UNKNOWN")
    svc = BusinessService(w)
    svc.issue("T4", "commit", "x")
    svc.observe("T4", physically_done=True, evidence_ok=False)
    assert svc.tasks["T4"].state is TaskState.UNKNOWN
    assert svc.commit("T4", lambda: None, 0.0) is False


def test_t09_heartbeat_expiry_detected():
    """心跳过期 (设计 R11: 同一仿真时间域内)。"""
    w = world("T09-HB")
    svc = BusinessService(w)
    svc.heartbeat(0.0)
    assert svc.heartbeat_expired(0.1) is False
    assert svc.heartbeat_expired(0.6) is True


def test_t09_service_down_refuses_new_commands():
    """服务停机期间不得下发动作 (物理继续, 控制面停机)。"""
    w = world("T09-DOWN")
    svc = BusinessService(w)
    svc.restart()
    with pytest.raises(RuntimeError):
        svc.issue("T5", "pick", "no")


def test_t09_reconnect_reports_are_distinct():
    """重连次数须如实记录 (供报告区分正例/拒绝/受控故障)。"""
    w = world("T09-COUNT")
    svc = BusinessService(w)
    svc.issue("T6", "pick", "c")
    for _ in range(3):
        svc.restart()
        svc.reconnect("T6", physically_done=False, evidence_ok=True)
    assert svc.reconnects == 3
    # 重连不应产生 UNKNOWN (证据始终充分)
    assert not svc.unknown_events
    assert svc.tasks["T6"].settled_count == 0