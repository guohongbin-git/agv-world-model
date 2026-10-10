"""W05 调度与库存事务 — 七段流程的记账语义 (设计 25.5)。

逐段断言设计表里的「原子业务提交」列, 而非只断言「没抛异常」:
|流程|原子业务提交|
|入料接管|G→GRIPPER, 仍 INBOUND **不计在库**|
|入库装托|建绑定 + PAL→LOADED + G→IN_WAREHOUSE, 计数+1|
|车辆运输|只动 PAL 位置, 绑定 G 有效位置仍是 PALLET, 计数不变|
|到出库站|G→OUTBOUND_PENDING, **仍计在库**|
|卸货接管|关活动绑定 + PAL→EMPTY, 计数不变|
|出料放置|G→FEED:O-TRAY, 保持 OUTBOUND_PENDING, 计数不变|
|货物离场|G→OFFSITE 并计数−1|

每个测试自建双臂与 World: 物理单元有内部状态 (箱件位置/夹持), 复用会
让后一个测试接手前一个的半成品货物 —— 上一轮的 G-001 已在 GRIPPER 里,
下一轮再调 feed_pick 只会报「货物不在 FEED」, 测不到真实路径。
"""
from __future__ import annotations

import numpy as np
import pytest

from services.adapters.manipulator import ManipulatorAdapter
from services.scheduler.reservations import ReservationBoard
from services.scheduler.tasks import StepOutcome, TaskScheduler
from services.world.ontology import GoodsUnit, Loc, LocKind, Pallet, Slot, World
from sim.manipulator_cell import ManipulatorCell

ARM_CAPS = frozenset(["arm_move_joints", "arm_move_tcp", "gripper_close",
                      "gripper_open", "arm_stow", "stop"])


def make_arm(arm_id: str):
    c = ManipulatorCell(timestep_s=0.0005, arm=arm_id)
    for _ in range(3000):
        c.step(c.substeps)
    return c, ManipulatorAdapter(arm_id, c, capabilities=ARM_CAPS)


@pytest.fixture
def rig():
    """双臂 + 全新 World + 调度器 (每测试独立, 避免状态串味)。"""
    c1, a1 = make_arm("M01")
    c2, a2 = make_arm("M02")
    w = World(run_id="FLOW")
    w.register_slot(Slot("A-L1-C1", (0.415, 1.0, 0.4)))
    w.register_pallet(Pallet("PAL-001", 0.045, Loc(LocKind.STATION, "I01")))
    w.register_goods(GoodsUnit("G-001", "A", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    sch = TaskScheduler(w, ReservationBoard(), m01=a1, m02=a2,
                        clock=lambda: float(c1.data.time))
    return w, sch, c1, a1, c2, a2


def approach_grasp(a: ManipulatorAdapter) -> None:
    """把 TCP 走到箱件抓取位 (抬离由调度器自己做)。"""
    c = a.cell
    bx, by, bz = c.box_pos[0], c.box_pos[1], c.box_bottom_z
    pose, _ = c.grasp_target_pose(np.array([bx, by]), bz)
    for k, t in enumerate([np.array([bx, by, 0.22]),
                           pose + np.array([0, 0, 0.05]), pose]):
        a.execute("move_tcp", f"pre{k}", target=t)


def lower_over_pallet(a: ManipulatorAdapter) -> None:
    """把箱件放到托盘承货面上方 55mm。"""
    c = a.cell
    a.execute("move_tcp", "over", target=np.array(
        [c.box_pos[0], c.box_pos[1], c.pallet_support_z + 0.10]))
    a.execute("move_tcp", "low", target=np.array(
        [c.box_pos[0], c.box_pos[1], c.pallet_support_z + 0.055]))


def test_feed_pick_does_not_count_in_warehouse(rig):
    """入料接管: G→GRIPPER, 仍不计在库 (设计 25.5 第一行)。"""
    w, sch, c1, a1, _, _ = rig
    assert w.in_warehouse_count() == 0
    approach_grasp(a1)
    assert sch.feed_pick("G-001") is StepOutcome.OK
    assert w.goods_location("G-001") == "GRIPPER:M01"
    assert w.in_warehouse_count() == 0, "入料接管不计在库"


def test_feed_pick_requires_real_lift(rig):
    """抬离不足 5mm 时不得算接管 (设计 25.3 GRASP_CONFIRMED)。"""
    w, sch, c1, a1, _, _ = rig
    approach_grasp(a1)
    # 用一个抬离步长为 0 的调度器: 只闭爪不抬起 → 抬离量必为 0。
    # (不能靠"预先闭爪"来模拟: 闭爪是调度器自己的动作, 预闭爪会让
    #  幂等路径跳过闭爪, 测到的不是设计要求的那条判据。)
    sch._lift_step_m = 0.0
    assert sch.feed_pick("G-001") is StepOutcome.FAILED
    assert "抬离" in sch.records[-1].detail
    assert w.goods_location("G-001") == "FEED:I-TRAY", "失败不得提交事务"


def test_full_seven_phase_ledger(rig):
    """完整七段: 计数 0→0→1→1→1→1→0, 每段位置与绑定都对。"""
    w, sch, c1, a1, c2, a2 = rig
    approach_grasp(a1)
    assert sch.feed_pick("G-001") is StepOutcome.OK
    assert w.in_warehouse_count() == 0

    lower_over_pallet(a1)
    assert sch.load_pallet("G-001", "PAL-001", "A-L1-C1") is StepOutcome.OK
    assert w.goods_location("G-001") == "PALLET:PAL-001"
    assert w.pallet("PAL-001")[1] == "LOADED"
    assert w.active_binding("G-001") is not None
    assert w.in_warehouse_count() == 1, "装托计数 +1"

    w.move_pallet("PAL-001", Loc(LocKind.STATION, "O01"), "FLOW", c1.data.time)
    assert w.goods_location("G-001") == "PALLET:PAL-001", "绑定货物有效位置仍是 PALLET"
    assert w.in_warehouse_count() == 1, "运输计数不变"

    w.mark_outbound_pending("G-001", "FLOW", c1.data.time)
    assert w.in_warehouse_count() == 1, "到出库站仍计在库"

    # M02 的箱件起始在自己的托盘上, 先走到抓取位
    approach_grasp(a2)
    assert sch.unload_pick("G-001", arm_id="M02") is StepOutcome.OK
    assert w.active_binding("G-001") is None, "活动绑定应关闭"
    assert w.pallet("PAL-001")[1] == "EMPTY", "PAL→EMPTY (仅承载状态)"
    assert w.in_warehouse_count() == 1, "卸货计数不变"

    # 出料放置: M02 放到出料托台 (保持 OUTBOUND_PENDING), 再离场
    assert sch.place_feed("G-001", arm_id="M02") is StepOutcome.OK
    assert w.goods_location("G-001").startswith("FEED:"), "G→出料托台"
    assert w.in_warehouse_count() == 1, "放置后计数不变"

    assert sch.offsite("G-001") is StepOutcome.OK
    assert w.in_warehouse_count() == 0, "离场计数 −1"


def test_load_pallet_rejects_non_empty_pallet(rig):
    """托盘非空时装托必须 FAILED, 且不提交事务。"""
    _, _, c1, a1, _, _ = rig
    w = World(run_id="REJ")
    w.register_pallet(Pallet("PAL-999", 0.045, Loc(LocKind.STATION, "I01")))
    w.register_goods(GoodsUnit("G-9", "A", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    sch = TaskScheduler(w, ReservationBoard(), m01=a1, clock=lambda: float(c1.data.time))
    # 先走完整的「取 → 装」, 让托盘真的变成 LOADED
    approach_grasp(a1)
    assert sch.feed_pick("G-9") is StepOutcome.OK
    lower_over_pallet(a1)
    assert sch.load_pallet("G-9", "PAL-999", "X") is StepOutcome.OK
    assert w.pallet("PAL-999")[1] == "LOADED"
    # 换一件新货物再装同一个托盘 → 必须被拒 (托盘非空)
    w.register_goods(GoodsUnit("G-10", "B", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    assert sch.load_pallet("G-10", "PAL-999", "X") is StepOutcome.FAILED
    assert "非空" in sch.records[-1].detail
    assert w.active_binding("G-10") is None, "被拒后不得有绑定"


def test_offsite_requires_placement_first(rig):
    """未进入 OUTBOUND_PENDING 就不得离场, 且不得抛异常到调用方。"""
    _, _, c1, a1, _, _ = rig
    w = World(run_id="OFF")
    w.register_pallet(Pallet("P", 0.045, Loc(LocKind.STATION, "I01")))
    w.register_goods(GoodsUnit("G-x", "A", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    sch = TaskScheduler(w, ReservationBoard(), clock=lambda: c1.data.time)
    assert sch.offsite("G-x") is StepOutcome.FAILED
    assert w.in_warehouse_count() == 0


def test_feed_pick_commit_is_idempotent(rig):
    """重复接管不重复提交 (设计 R09: 事件 ID 唯一)。"""
    w, sch, c1, a1, _, _ = rig
    approach_grasp(a1)
    assert sch.feed_pick("G-001") is StepOutcome.OK
    committed = len([k for k in sch.committed if k.startswith("feed_pick")])
    sch.feed_pick("G-001")                    # 第二次: 位置已在 GRIPPER
    assert len([k for k in sch.committed if k.startswith("feed_pick")]) == committed
    assert w.in_warehouse_count() == 0


def test_arms_do_not_interlock_when_envelopes_disjoint(rig):
    """M01/M02 包络不相交 → 不互相阻塞 (设计 159 行锁的是相交通道)。"""
    _, sch, c1, a1, _, a2 = rig
    d = float(np.linalg.norm(
        sch._envelope_center(a1) - sch._envelope_center(a2)))
    assert d > 1.0, f"双臂基座应相距 >1m, 实测 {d:.3f}m"
    a2.execute("stow", "init")                 # M02 未收拢也不该卡住 M01
    sch._require_exclusive("M01")              # 不抛异常即通过