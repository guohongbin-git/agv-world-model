"""P3 Wave2 双车交接 —— 全链验收 (实测)。

范围:
* F02 行车关节 range 放宽 (货架插叉位/交接台可达), 单车构造逐位不变;
* 双车交接 E2E: F01 取 rack→交接台 S1 → 前移让位 → F02 等巷道空闲 →
  S1 取托送 H01; 托盘所有权链 ROBOT:F01→STATION:S1→ROBOT:F02→SLOT:H01,
  G-001 → OUTBOUND_PENDING, 在库计数不变;
* 单车旧任务 (L3→STAGE→L3, 4 元组) 在双车世界回归仍通;
* 巷道互斥锁 (AisleLock) 语义;
* 全程零碰撞 (max_penetration < 5e-4)。
"""
from __future__ import annotations

import json

import pytest

from services.tasks.messages import Action, Command, check_capability
from sim.forklift_cell import ForkliftCell
from twin.server import AISLE_X_MIN, AisleLock, PacedForkliftAdapter, WorldService

PEN_LIMIT = 5e-4


@pytest.fixture(scope="module")
def pen_guard():
    """全程碰撞审计: 类级包装 PacedForkliftAdapter._tick, 每 200 步采样
    max_penetration ( survives WorldService.build()/reset 重建适配器)。"""
    orig = PacedForkliftAdapter._tick
    max_pen = [0.0]

    def _tick(self):
        orig(self)
        if self._n % 200 == 0:
            max_pen[0] = max(max_pen[0], self.cell.max_penetration())

    PacedForkliftAdapter._tick = _tick
    yield max_pen
    PacedForkliftAdapter._tick = orig


@pytest.fixture(scope="module")
def svc(pen_guard):
    s = WorldService()
    s.gate.speed = 64.0        # 测试加速: 只解除墙钟配速, 物理步进不变
    return s


# ---------------------------------------------------------------- cell 层
def test_f02_range_relaxed_single_vehicle_unchanged():
    """F02 行程上限放宽到货架插叉位; 单车默认构造逐位不变。"""
    single = ForkliftCell(timestep_s=0.0005, levels=[0.100, 0.250, 0.400])
    dual = ForkliftCell(timestep_s=0.0005, levels=[0.100, 0.250, 0.400],
                        vehicles=["F01", "F02"])
    m = dual.model
    f01, f02 = dual.vehicle("F01"), dual.vehicle("F02")
    r01, r02 = m.jnt_range[f01.j_x], m.jnt_range[f02.j_x]
    # 首车: 与单车构造逐位一致 (限位未变, 物理行为不变)
    assert list(r01) == list(single.model.jnt_range[single.j_x])
    assert abs(r01[0] + 0.8) < 1e-12 and abs(r01[1] - 0.65) < 1e-12
    # F02: 可达货架插叉位 (-0.095, 由 k["rack_insert_axle_x"] 推导) 与
    # 交接台放托位 (-0.444/-0.424), 上限 = 插叉位 + 5mm, 不多给
    lo_axle, hi_axle = f02.q0_x + r02[0], f02.q0_x + r02[1]
    assert hi_axle >= dual.k["rack_insert_axle_x"]
    assert hi_axle <= dual.k["rack_insert_axle_x"] + 0.010
    assert abs(r02[0] + 0.8) < 1e-12
    for need in (-0.444, -0.424):
        assert lo_axle <= need <= hi_axle


# ---------------------------------------------------------------- 互斥层
def test_aisle_lock_mutual_exclusion():
    """巷道互斥: 一车持锁期间他车不可进入; 非持有者释放是 no-op。"""
    lk = AisleLock()
    assert lk.owner is None
    assert lk.acquire("F01", block=False)
    assert lk.owner == "F01"
    assert not lk.acquire("F02", block=False)      # F01 作业中, F02 被拒
    lk.release("F02")                              # 非持有者释放 → no-op
    assert lk.owner == "F01"
    lk.release("F01")
    assert lk.owner is None
    assert lk.acquire("F02", block=False)
    assert not lk.acquire("F01", block=False)
    lk.release("F02")
    assert AISLE_X_MIN == -1.0


# ---------------------------------------------------------------- agent 层
class _StubSvc:
    """rules() 单元测试桩: 只记录 cmd 调用, 不触物理。"""

    mode = "idle"
    HANDOFF_TASKS = [("F01", "PAL-001", "G-001", "L3", "S1"),
                     ("F02", "PAL-001", "G-001", "S1", "H01")]

    def __init__(self):
        self.calls = []

    def cmd(self, op, value=None):
        self.calls.append((op, value))
        return {"mode": "running"}


def test_agent_routes_handoff_command():
    """"接力/交接/F02" 指令 → 双车交接队列 (5 元组), 不落入单车分支。"""
    from twin.agent import rules
    stub = _StubSvc()
    reply, tools = rules("F01 取货放交接台, F02 接力送 H01", stub)
    assert [t["tool"] for t in tools] == ["task_handoff"]
    assert stub.calls == [("dispatch", list(_StubSvc.HANDOFF_TASKS))]
    assert "STATION:S1" in reply and "SLOT:H01" in reply


def test_agent_legacy_command_unchanged():
    """单车下架指令仍走旧 4 元组 dispatch, 不误触交接分支。"""
    from twin.agent import rules
    stub = _StubSvc()
    _, tools = rules("把 A-L3-C1 那托下架到 H01", stub)
    assert tools[0]["tool"] == "task_dispatch"
    assert stub.calls[0][1] == [("PAL-001", "G-001", "L3", "STAGE")]


# ---------------------------------------------------------------- 消息层
def test_f02_in_message_contract():
    """CAPABILITIES 与 Command.robot_id 契约收录 F02。"""
    check_capability("F02", Action.DOCK_LINEAR)
    check_capability("F02", Action.INSERT_FORKS)
    cmd = Command(run_id="R", command_id="C1", task_id="T",
                  robot_id="F02", action=Action.STOP,
                  params={"reason": "test", "level": "controlled"},
                  timeout_s=1.0, expected_state_version=0)
    assert cmd.robot_id == "F02"


# ---------------------------------------------------------------- E2E
def _run_queue(s):
    s.mode = "running"
    s.run_mission()


def test_handoff_e2e_ledger_chain_and_zero_collision(svc, pen_guard):
    """双车交接 E2E: 完整所有权链 + OutboundPending + 账本一致 + 零碰撞。"""
    pen_guard[0] = 0.0
    svc.tasks = list(svc.HANDOFF_TASKS)
    _run_queue(svc)
    assert svc.mode == "done", f"mission 未完成: {svc.step_label}"
    # 零碰撞
    assert pen_guard[0] < PEN_LIMIT, f"max_pen={pen_guard[0]:.2e} 超限"
    # 账本: 托盘所有权链 ROBOT:F01 → STATION:S1 → ROBOT:F02 → SLOT:H01
    rows, _ = svc.events_since(0)
    chain = [json.loads(e[2]).get("to_location")
             for e in rows if e[1] == "PalletMoved"]
    assert chain == ["ROBOT:F01", "STATION:S1", "ROBOT:F02", "SLOT:H01"]
    started = [json.loads(e[2]).get("vehicle")
               for e in rows if e[1] == "TaskStarted"]
    assert started == ["F01", "F02"]
    b = svc.board()
    pallets = {p[0]: (p[1], p[2]) for p in b["pallets"]}
    assert pallets["PAL-001"] == ("SLOT:H01", "LOADED")
    goods = {g[0]: (g[2], g[3]) for g in b["goods"]}
    assert goods["G-001"] == ("PALLET:PAL-001", "OUTBOUND_PENDING")
    slots = dict(b["slots"])
    assert slots["A-L3-C1"] == "EMPTY"      # 取空
    assert slots["H01"] == "OCCUPIED"       # 落托
    assert b["count"] == 3                  # OUTBOUND_PENDING 仍计在库
    assert sum(1 for e in rows if e[1] == "OutboundPending") == 1
    assert not any(e[1] == "CommitEvidenceMissing" for e in rows)
    # 让位生效: F01 停在待命位, 不挡 F02 的交接台进近
    assert svc.cell.vehicle("F01").axle_x == pytest.approx(0.020, abs=0.010)
    # 物理终态: 托盘落在 H01 支撑面, 双叉均脱离
    assert svc.cell.pallet_bottom_z == pytest.approx(0.045, abs=0.005)
    assert not svc.cell.pallet_fork_contact("F01")
    assert not svc.cell.pallet_fork_contact("F02")


def test_single_vehicle_roundtrip_regression(svc, pen_guard):
    """单车旧任务 (4 元组 L3→STAGE→L3) 在双车世界回归仍通, 语义不变。"""
    pen_guard[0] = 0.0
    svc.cmd("reset")
    svc.gate.speed = 64.0
    r = svc.cmd("dispatch", [("PAL-001", "G-001", "L3", "STAGE"),
                             ("PAL-001", "G-001", "STAGE", "L3")])
    svc._thread.join(timeout=600)
    assert svc.mode == "done", f"回归任务未完成: {svc.step_label}"
    assert pen_guard[0] < PEN_LIMIT, f"max_pen={pen_guard[0]:.2e} 超限"
    rows, _ = svc.events_since(0)
    chain = [json.loads(e[2]).get("to_location")
             for e in rows if e[1] == "PalletMoved"]
    # 旧链语义逐位保持: STAGE 仍提交 STATION:H01 + OutboundPending,
    # 回架提交 SLOT:A-L3-C1 + InboundConfirmed
    assert chain == ["ROBOT:F01", "STATION:H01", "ROBOT:F01", "SLOT:A-L3-C1"]
    assert sum(1 for e in rows if e[1] == "OutboundPending") == 1
    assert sum(1 for e in rows if e[1] == "InboundConfirmed") == 1
    started = [json.loads(e[2]).get("vehicle")
               for e in rows if e[1] == "TaskStarted"]
    assert started == ["F01", "F01"]        # 4 元组默认首车
    b = svc.board()
    pallets = {p[0]: p[1] for p in b["pallets"]}
    assert pallets["PAL-001"] == "SLOT:A-L3-C1"
    goods = {g[0]: g[3] for g in b["goods"]}
    assert goods["G-001"] == "IN_WAREHOUSE"
    assert b["count"] == 3
