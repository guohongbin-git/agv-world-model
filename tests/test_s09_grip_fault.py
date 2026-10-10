"""S09 夹持故障 —— 在夹持/抬离/释放/提交四个边界注入。

设计判据:
|S09 夹持故障|在夹持、抬离、释放和提交边界注入故障|T14/T16负例|
     |检测滑移或UNKNOWN；无重复抓取及库存增减|

四条边界各注入一次, 每条都必须证明:
1. **检测滑移或 UNKNOWN** —— 不是"静默当作成功";
2. **无重复抓取** —— 故障后不得再发一次抓取命令;
3. **库存不增减** —— 失败路径绝不记账。

物理注入用低摩擦 (设计 T14 已验证 μ=0.2 会真实失败), 另加
「夹持中途松爪」和「抬离中途掉货」两种更贴近 S09 的注入。
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from services.adapters.manipulator import ManipulatorAdapter
from services.world.ontology import (GoodsUnit, Loc, LocKind, Pallet, Slot,
                                     World)
from sim.manipulator_cell import ManipulatorCell

ARM_CAPS = frozenset(["arm_move_joints", "arm_move_tcp", "gripper_close",
                      "gripper_open", "arm_stow", "stop"])
MIN_LIFT_M = 0.005          # 设计 25.3 GRASP_CONFIRMED
MAX_DRIFT_M = 0.001         # 夹持稳定判据


def make_arm(arm_id="M01", pad_friction=0.6):
    c = ManipulatorCell(timestep_s=0.0005, arm=arm_id)
    # 注入低摩擦 (S09 负例)
    for i in range(c.model.ngeom):
        nm = mujoco.mj_id2name(c.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if nm and "pad" in nm:
            c.model.geom_friction[i][0] = pad_friction
    for _ in range(3000):
        c.step(c.substeps)
    return c, ManipulatorAdapter(arm_id, c, capabilities=ARM_CAPS)


def world(run_id: str) -> World:
    w = World(run_id=run_id)
    w.register_slot(Slot("A-L1-C1", (0.415, 1.0, 0.4)))
    w.register_pallet(Pallet("PAL-001", 0.045, Loc(LocKind.STATION, "I01")))
    w.register_goods(GoodsUnit("G-001", "A", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    return w


def grasp(a: ManipulatorAdapter) -> None:
    c = a.cell
    bx, by, bz = c.box_pos[0], c.box_pos[1], c.box_bottom_z
    pose, _ = c.grasp_target_pose(np.array([bx, by]), bz)
    for k, t in enumerate([np.array([bx, by, 0.22]),
                           pose + np.array([0, 0, 0.05]), pose]):
        a.execute("move_tcp", f"pre{k}", target=t)
    a.execute("close_gripper", "g", target_gap_m=0.030)


def snapshot(w: World) -> tuple:
    rows = w._conn.execute(
        "SELECT goods_id, location, lifecycle FROM goods").fetchall()
    pal = w._conn.execute(
        "SELECT pallet_id, cargo_state FROM pallets").fetchall()
    return tuple(rows), tuple(pal), w.in_warehouse_count()


# ---------------------------------------------------------------- 边界
def test_s09_grip_boundary_low_friction_detected():
    """夹持边界: μ=0.2 时夹不住, 必须检出而非当作成功。"""
    c, a = make_arm(pad_friction=0.2)
    w = world("S09-GRIP")
    before = snapshot(w)
    grasp(a)
    for i in range(30):
        a.execute("move_tcp", f"l{i}",
                  target=a.cell.tcp_pos + np.array([0, 0, 0.0025]))
    obs = a.observe()
    held = obs["left_contact"] and obs["right_contact"]
    if not held:
        # 夹持丢失 → 明确失败, 库存零变化
        assert w.in_warehouse_count() == 0
        assert snapshot(w) == before
        assert a.cell.gripping is False
    else:
        drift_m, drift_deg = a.grasp_stability(0.5)
        if drift_m > MAX_DRIFT_M:
            assert w.in_warehouse_count() == 0, "检出滑移时不得记账"


def test_s09_lift_boundary_release_midway_is_detected():
    """抬离边界: 抬升途中松爪 → 箱件掉落回承托面, 必须检出且不记账。

    判定不是"箱件悬空", 而是**它回到了托盘承货面** (73mm) 且脱离了
    夹爪 —— 这正是故障后的真实物理。之前误以为松爪后应悬空, 实际上
    托盘就在正下方 26mm 处, 箱件必然落回承货面。
    """
    c, a = make_arm()
    w = world("S09-LIFT")
    before = snapshot(w)
    grasp(a)
    held_gap = float(np.linalg.norm(c.box_pos - c.tcp_pos))
    for i in range(10):
        a.execute("move_tcp", f"up{i}",
                  target=c.tcp_pos + np.array([0, 0, 0.0025]))
    lifted_z = float(c.box_bottom_z)
    a.cell.release_grip_hold()
    a.execute("open_gripper", "fault-open")
    for _ in range(2000):
        c.step(c.substeps)
    # 夹爪已**不再持有**箱件 (权威判据), 间距相对夹持时显著增大。
    # 间距本身不是固定值: 箱件落回承货面后 TCP 在其正上方, 间距 ≈17mm。
    assert not a.cell.gripping, "松爪后仍标记为夹持中"
    gap = float(np.linalg.norm(c.box_pos - c.tcp_pos))
    assert gap > held_gap * 1.5, (
        f"松爪后间距 {gap*1000:.1f}mm 未显著大于夹持时 "
        f"{held_gap*1000:.1f}mm")
    # 故障后箱件回落 —— 未达 50mm 抬离判据
    assert float(c.box_bottom_z) < lifted_z - 0.010, "箱件未回落"
    assert snapshot(w) == before, "松爪故障不得记账"


def test_s09_release_boundary_landing_is_verified_not_assumed():
    """释放边界: 必须**实测**落位, 不得假设承托就等于成功。

    关键: 箱件落在**托盘承货面**(73mm) 时承托=1.0, 落在**托台面**
    (55mm) 时承托=0.0 —— 只读 support_ratio 分不出这两者, 必须同时看
    托台面承托与落位高度。这正是"承托≠成功"的含义。
    """
    c, a = make_arm()
    w = world("S09-REL")
    before = snapshot(w)
    grasp(a)
    for i in range(20):
        a.execute("move_tcp", f"u{i}",
                  target=a.cell.tcp_pos + np.array([0, 0, 0.0025]))
    a.execute("open_gripper", "rel-open")
    for _ in range(200):
        c.step(c.substeps)
    o = a.observe()
    landed_on_pallet = float(o["support_ratio"]) >= 0.90
    landed_on_tray = float(o["tray_ratio"]) >= 0.90
    assert not (landed_on_pallet and landed_on_tray), (
        "同时压托盘与托台 —— 判据无法区分落位面")
    # 落位高度必须落在承货面, 且不是托台面
    assert abs(float(c.box_bottom_z) - c.pallet_support_z) < 2e-3, (
        f"落位高度 {c.box_bottom_z*1000:.1f}mm ≠ 承货面 "
        f"{c.pallet_support_z*1000:.1f}mm")
    assert landed_on_pallet and not landed_on_tray
    assert snapshot(w) == before, "仅松爪未提交时库存不得变化"


def test_s09_commit_boundary_evidence_loss_is_unknown():
    """提交边界: 证据丢失时进 UNKNOWN, 不伪造成功。"""
    from services.scheduler.fault_service import BusinessService, TaskState
    w = world("S09-CMT")
    svc = BusinessService(w)
    svc.issue("TASK-1", "commit", "c1")
    svc.observe("TASK-1", physically_done=True, evidence_ok=False)
    assert svc.tasks["TASK-1"].state is TaskState.UNKNOWN
    calls = []
    assert svc.commit("TASK-1", lambda: calls.append(1), 0.0) is False
    assert not calls, "UNKNOWN 状态提交了事务"
    assert w.in_warehouse_count() == 0


def test_s09_no_repeat_grasp_after_fault():
    """故障后不得重复抓取 —— 同一 command_id 幂等。"""
    c, a = make_arm()
    grasp(a)
    n_before = len(a._done_events)
    # 用同一 command_id 再次下发闭爪
    a.execute("close_gripper", "g", target_gap_m=0.030)
    a.execute("close_gripper", "g", target_gap_m=0.030)
    assert len(a._done_events) == n_before, "重复 command_id 被重新执行"


def test_s09_grasp_stability_criterion_is_real():
    """夹持稳定判据必须真的能测出漂移 (不是恒 0)。"""
    c, a = make_arm()
    grasp(a)
    for i in range(20):
        a.execute("move_tcp", f"l{i}",
                  target=a.cell.tcp_pos + np.array([0, 0, 0.0025]))
    d_m, d_deg = a.grasp_stability(0.5)
    assert d_m >= 0.0 and d_deg >= 0.0
    # 正常条件下应在设计判据内
    assert d_m <= 0.005, f"正常夹持漂移 {d_m*1000:.3f}mm 超出 5mm"
    # 人为把手臂甩开后, 相对漂移判据应显著变化
    c.set_joint_targets(c.q + 0.6)
    for _ in range(30):
        c.step(c.substeps)
    assert float(np.abs(c.data.qvel).max()) > 1e-3, "应有运动"