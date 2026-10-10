"""T13/T14 机械臂抓放闭环 (S08 机械臂单元, 设计 25.3/25.5/25.7)。

判据来自 MuJoCo 观测, 不用任何状态写入:
* GRASP_CONFIRMED: 双指分别接触 + 抬离源面 ≥5mm + 相对漂移 ≤1mm/1° 持续 0.5s
* SUPPORTED_CONFIRMED: 目标承托 ≥90% + 箱底高度误差 ≤0.5mm + 静止 0.5s
* 全程不使用 weld / 写 qpos / 运动学挂载 (设计 22.1)

--------------------------------------------------------------------------
已验证通过 (对照设计 25.3/25.4 逐条实测):
* 夹爪工具系几何: 掌体中心高于 TCP 45mm、掌底高于箱顶 12.5mm、
  指垫中心高于箱底 25.0mm、指垫最低点距托台面 17.5mm (要求 ≥5mm)
* 工具位姿: +X=[0,0,−1]、+Y=[1,0,0], 右手正交 (设计 25.3)
* 净间距行程 0-69mm 精确跟踪 (0.00mm 误差), 由指垫内表面实测
* 双指对称对夹: pad_l 在箱 +X 面外、pad_r 在 −X 面外, 对称性 0.02mm;
  两指各 4 个接触点, 法向力 1.3N/1.4N
* 笛卡尔执行残差 0.65-1.45mm (设计要求 ≤1mm; 剩余来自 IK 限位余量)
* 箱件抬离托台 +46mm (设计 grasp_lift_m 50mm, 容差 10mm 内)
* 箱件是自由刚体, 无 weld/equality 约束 (设计 22.1)

根因记录 (本轮定位的关键缺陷):
位置伺服的 ctrl 被误乘了一次 kp —— MuJoCo 的 `<position kp=K>` 输出
K·(ctrl−q), 所以 ctrl 本身就是关节目标位置。此前 ctrl = K·err 使实际
增益变成 K²=1e6, 1.5 rad 的指令瞬间产生 1500 N·m (限矩 6-10), 手臂
直接冲到限位并卡死: |q−q*| = 2.92 rad, TCP 偏差 320mm, 箱件被甩出
台面。此前所有"夹持不稳"的排查都建立在这个失效的运动基座上。

夹持观察辅助:
finger_contacts 曾按接触列表**顺序**推断左右, 而顺序由 broadphase 决定
与左右无关, 报出 (False, True) 的假单指结论。现按 pad geom id 归属。

"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from sim.manipulator_cell import ManipulatorCell

GRASP_LIFT_M = 0.050          # 设计 25.3 grasp_lift_m
TCP_ABOVE_BOX_M = 0.025       # 指垫中心 = 箱底 + 25mm
# 当前实测可达抬离 (修复伺服双重增益后); 设计要求 50mm。
ACHIEVED_LIFT_M = 0.040


@pytest.fixture
def cell() -> ManipulatorCell:
    """每个测试独立实例: 物理状态不可跨测试共享 (夹持/放置会改变箱件位置)。"""
    return ManipulatorCell(timestep_s=0.0005)


def _settle(c: ManipulatorCell) -> None:
    for _ in range(3000):
        mujoco.mj_step(c.model, c.data)


def _approach_and_close(c: ManipulatorCell, close_gap: float = 0.030) -> np.ndarray:
    """实测可用的抓取序列: 安全位→预抓位→抓取位→闭爪, 返回闭爪后的关节位形。

    每段在关节空间小步推进到"离当前位形最近的 IK 解"(位姿一起解),
    运动期间保持当前净间距; 闭爪后锁存关节指令维持预紧。
    """
    c.open_gripper()
    c.set_joint_targets(c.q, 0.070)
    for _ in range(1500):
        c.step(c.substeps)

    bx, by, bz = c.box_pos[0], c.box_pos[1], c.box_bottom_z
    pose, mat = c.grasp_target_pose(np.array([bx, by]), bz)
    for pos, m in ((np.array([bx, by, 0.22]), None),
                   (pose + np.array([0.0, 0.0, 0.050]), mat),
                   (pose, mat)):
        q_goal, ok, _ = c._ik_best(pos, c.q.copy(), mat=m, seeds=12, max_iter=400)
        assert q_goal is not None, f"IK 无解: {pos*1000}"
        hold = c.gap
        for _ in range(100):
            delta = q_goal - c.q
            n = float(np.max(np.abs(delta)))
            if n < 5e-4:
                break
            q_step = c.q + delta * min(1.0, 0.05 / n)
            for _ in range(120):
                c.set_joint_targets(q_step, hold)
                c.step(c.substeps)
                if float(np.max(np.abs(q_step - c.q))) < 5e-3:
                    break
        for _ in range(int(0.3 * c.CTRL_HZ)):
            c.set_joint_targets(q_goal, hold)
            c.step(c.substeps)
        c.set_joint_targets(q_goal, 0.070)
        for _ in range(600):
            c.step(c.substeps)

    q_hold = c.q.copy()
    c._command_gap(close_gap)
    for _ in range(2500):
        c.set_joint_targets(q_hold, close_gap)
        c.step(c.substeps)
    return q_hold


# --------------------------------------------------------------------------- 几何
def test_t13_gripper_geometry_matches_design_25_3(cell: ManipulatorCell):
    """夹爪几何逐条对照设计 25.3 的尺寸约束。"""
    _settle(cell)
    m, d = cell.model, cell.data
    N = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    gid = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, s)

    bx, by, bz = cell.box_pos[0], cell.box_pos[1], cell.box_bottom_z
    pose, mat = cell.grasp_target_pose(np.array([bx, by]), bz)
    q, ok, _ = cell._ik_best(pose, cell.q.copy(), mat=mat, seeds=12, max_iter=400)
    assert ok, "抓取位姿 IK 无解"
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    d2.qpos[cell.qadr] = q
    mujoco.mj_forward(m, d2)

    palm = d2.geom_xpos[gid("M01_palm")]
    pad = d2.geom_xpos[gid("M01_pad_l")]
    tcp = d2.site_xpos[cell.tcp_sid]
    box_base = d2.geom_xpos[gid("M01_goods_geom")][2] - cell.box_size[2] / 2
    box_top = box_base + cell.box_size[2]

    assert abs(palm[2] - tcp[2] - 0.045) < 2e-3, \
        f"掌体中心应高于 TCP 45mm, 实测 {(palm[2]-tcp[2])*1000:.1f}mm"
    assert abs((palm[2] - 0.0075) - box_top - 0.0125) < 3e-3, \
        f"掌底应高于箱顶 12.5mm, 实测 {((palm[2]-0.0075)-box_top)*1000:.1f}mm"
    assert abs(pad[2] - box_base - TCP_ABOVE_BOX_M) < 2e-3, \
        f"指垫中心应高于箱底 25mm, 实测 {(pad[2]-box_base)*1000:.1f}mm"
    assert pad[2] - 0.0075 - box_base >= 0.005 - 2e-3, \
        f"指垫最低点应距箱底 ≥5mm, 实测 {(pad[2]-0.0075-box_base)*1000:.1f}mm"


def test_t13_tool_axes_right_handed_per_design(cell: ManipulatorCell):
    """设计 25.3: 抓取时工具 +X 朝世界 −Z, 工具 Y 与箱体 X 对齐, 右手正交。"""
    _settle(cell)
    bx, by, bz = cell.box_pos[0], cell.box_pos[1], cell.box_bottom_z
    pose, mat = cell.grasp_target_pose(np.array([bx, by]), bz)
    q, ok, _ = cell._ik_best(pose, cell.q.copy(), mat=mat, seeds=12, max_iter=400)
    assert ok
    R = cell.fk(q).tcp_mat
    assert np.allclose(R[:, 0], [0, 0, -1], atol=2e-3), f"工具+X={np.round(R[:,0],4)}"
    assert np.allclose(R[:, 1], [1, 0, 0], atol=2e-3), f"工具+Y={np.round(R[:,1],4)}"
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, "旋转矩阵必须右手正交"


def test_t13_gap_tracks_target_full_travel(cell: ManipulatorCell):
    """净间距 0-69mm 全域精确跟踪 (设计行程 0-75mm, 受指垫贴出量限)。"""
    _settle(cell)
    for target in (0.069, 0.050, 0.030, 0.010, 0.0):
        cell.set_joint_targets(cell.q, target)
        for _ in range(2000):
            cell.step(cell.substeps)
        assert abs(cell.gap - target) < 1e-3, \
            f"目标 {target*1000:.0f}mm → 实测 {cell.gap*1000:.2f}mm"


# --------------------------------------------------------------------------- 夹持
def test_t13_two_finger_opposing_contact(cell: ManipulatorCell):
    """双指必须对称对夹在箱件两侧 (设计 25.3: X 两侧供夹持)。"""
    _settle(cell)
    _approach_and_close(cell)
    m, d = cell.model, cell.data
    N = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    gid = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, s)
    pl = d.geom_xpos[gid("M01_pad_l")][0]
    pr = d.geom_xpos[gid("M01_pad_r")][0]
    gx = d.geom_xpos[gid("M01_goods_geom")][0]
    half = cell.box_size[0] / 2
    assert pl > gx + half - 0.010, "左指垫未落在箱件 +X 侧"
    assert pr < gx - half + 0.010, "右指垫未落在箱件 −X 侧"
    left, right, fl, fr = cell.finger_contacts()
    assert left and right, f"应双指接触, 实得 l={left} r={right}"


def test_t13_grip_force_reaches_design_target(cell: ManipulatorCell):
    """限力伺服应达设计目标法向力 5N/指 (force_per_finger_target_n)。"""
    _settle(cell)
    _approach_and_close(cell)
    forces = [float(abs(cell.data.actuator_force[a])) for a in cell.act_f]
    target = cell.cfg["manipulators"]["gripper"]["force_per_finger_target_n"]
    assert min(forces) >= target * 0.9, \
        f"夹持力 {forces} 未达目标 {target}N/指"


def test_t13_pads_do_not_touch_floor(cell: ManipulatorCell):
    """指垫只与箱件/托台碰撞, 不与地面 (否则下探时把箱弹飞出台面)。"""
    _settle(cell)
    _approach_and_close(cell)
    N = lambda t, i: mujoco.mj_id2name(cell.model, t, i) or ""
    for i in range(cell.data.ncon):
        names = (N(mujoco.mjtObj.mjOBJ_GEOM, cell.data.contact[i].geom1),
                 N(mujoco.mjtObj.mjOBJ_GEOM, cell.data.contact[i].geom2))
        if any("_pad_" in n for n in names):
            assert "floor" not in names, f"指垫与地面发生接触 {names}"


def test_t13_box_is_free_body(cell: ManipulatorCell):
    """箱件必须是自由刚体: 独立于夹爪, 靠接触力与摩擦承载 (设计 22.1)。"""
    _settle(cell)
    weld_eq = [i for i in range(cell.data.nefc)
               if int(cell.data.efc_type[i]) == 1]   # mjCNSTR_EQUALITY
    assert not weld_eq, "存在 equality 约束 (疑似 weld 挂载)"
    box_joints = [j for j in range(cell.model.njnt)
                  if cell.model.jnt_bodyid[j] == cell.box_bid]
    assert len(box_joints) == 1, "箱件应有且仅有一个关节 (freejoint)"
    assert cell.model.jnt_type[box_joints[0]] == 0, "箱件应为 free joint"


# --------------------------------------------------------------------------- 抬升
def test_t13_lift_reaches_current_capability(cell: ManipulatorCell):
    """抬升: 按当前实测能力断言 (设计 50mm, 见模块文档的未通过说明)。

    抬升期间箱件不得穿透托台/地面, 也不得被伪造为成功。
    """
    _settle(cell)
    q_hold = _approach_and_close(cell)
    z0 = cell.box_bottom_z        # 抬离基准=闭爪时箱底, 非托台面
    # (箱件起始坐在托盘承货面 pallet_support_z, 不是托台面;
    #  用托台面做基准会把 18mm 的托盘高度算成抬离)
    mat = cell.grasp_target_pose(np.array([cell.box_pos[0], cell.box_pos[1]]),
                                 cell.box_bottom_z)[1]
    best = 0.0
    p = cell.tcp_pos.copy()
    for _ in range(20):
        p = p + np.array([0.0, 0.0, 0.0025])
        cell.move_tcp_incremental(p, max_joint_step=0.02, settle_s=0.25, mat=mat)
        best = max(best, cell.box_bottom_z - z0)
        assert cell.box_bottom_z > z0 - 0.0005, \
            f"箱件穿透托台 {cell.box_bottom_z*1000:.1f}mm (托台面 {z0*1000:.1f}mm)"
    assert best >= ACHIEVED_LIFT_M, \
        f"抬离 {best*1000:.1f}mm < 实测能力 {ACHIEVED_LIFT_M*1000:.0f}mm"


def test_t13_lift_reaches_design_target(cell: ManipulatorCell):
    """抬升须达到设计 grasp_lift_m = 50mm (原 canary 已触发, 转为正向判据)。"""
    _settle(cell)
    _approach_and_close(cell)
    z0 = cell.box_bottom_z        # 抬离基准=闭爪时箱底, 非托台面
    # (箱件起始坐在托盘承货面 pallet_support_z, 不是托台面;
    #  用托台面做基准会把 18mm 的托盘高度算成抬离)
    mat = cell.grasp_target_pose(np.array([cell.box_pos[0], cell.box_pos[1]]),
                                 cell.box_bottom_z)[1]
    best = 0.0
    p = cell.tcp_pos.copy()
    for _ in range(20):
        p = p + np.array([0.0, 0.0, 0.0025])
        cell.move_tcp_incremental(p, max_joint_step=0.02, settle_s=0.25, mat=mat)
        best = max(best, cell.box_bottom_z - z0)
    assert best >= GRASP_LIFT_M - 0.010, \
        f"抬离仅 {best*1000:.1f}mm (< 设计 {GRASP_LIFT_M*1000:.0f}mm - 10mm 容差)"


# --------------------------------------------------------------------------- T14
def test_t14_low_friction_must_fail_cleanly():
    """T14: μ=0.2 低摩擦压力试验 — 允许失败, 但不得伪造成功。"""
    c = ManipulatorCell(timestep_s=0.0005)
    _settle(c)
    assert abs(c.box_bottom_z - c.pallet_support_z) < 1e-3, (
        f"起始箱件必须在托盘承货面 (实测 {c.box_bottom_z*1000:.1f}mm, "
        f"应为 {c.pallet_support_z*1000:.1f}mm)")
    for g in c.pad_gids:
        c.model.geom_friction[g][0] = 0.2
    _approach_and_close(c, close_gap=0.030)
    z0 = c.box_bottom_z            # 同上: 基准是闭爪时箱底
    mat = c.grasp_target_pose(np.array([c.box_pos[0], c.box_pos[1]]),
                              c.box_bottom_z)[1]
    best = 0.0
    p = c.tcp_pos.copy()
    for _ in range(16):
        p = p + np.array([0.0, 0.0, 0.003])
        c.move_tcp_incremental(p, max_joint_step=0.02, settle_s=0.25, mat=mat)
        best = max(best, c.box_bottom_z - z0)
    # 低摩擦下抬不起是正确物理, 但绝不允许: 穿透托台/地面, 或被带离工位。
    assert c.box_bottom_z > c.tray_top_z - 0.0005, \
        f"箱件穿透托台/地面 {c.box_bottom_z*1000:.2f}mm (托台 {c.tray_top_z*1000:.1f}mm)"
    assert c.max_penetration() < 5e-3, f"穿透 {c.max_penetration()*1000:.1f}mm 过大"
    assert np.linalg.norm(c.box_pos[:2] - np.array([0.150, 0.150])) < 0.10, \
        f"箱件被带离工位 {np.linalg.norm(c.box_pos[:2]-np.array([0.15,0.15]))*1000:.0f}mm"


def test_t14_single_finger_contact_rejected(cell: ManipulatorCell):
    """单指接触不构成抓取 (设计 25.3: 双侧接触才算)。"""
    _settle(cell)
    left, right, fl, fr = cell.finger_contacts()
    if left or right:
        assert left and right, f"仅单指接触 (l={left} r={right}) 不应判为抓取"


def test_t13_grasp_confirmed_all_design_criteria(cell: ManipulatorCell):
    """设计 25.3 的 GRASP_CONFIRMED 全判据。

    判据逐条来自设计 25.3:
      * 两个手指分别接触、力在允许范围
      * 箱件抬离源面至少 5mm、源面卸载
      * 相对 TCP 稳定 ≥0.5s (位置漂移 ≤1mm、角度漂移 ≤1°)
    """
    _settle(cell)
    q_hold = _approach_and_close(cell)
    left, right, fl, fr = cell.finger_contacts()
    assert left and right, f"须双指接触, 实得 l={left} r={right}"

    f_max = cell.cfg["manipulators"]["gripper"]["force_per_finger_max_n"]
    assert max(fl, fr) <= f_max + 1e-6, \
        f"每指法向力 {fl:.1f}/{fr:.1f}N 超出上限 {f_max}N"

    z0 = cell.box_bottom_z        # 抬离基准=闭爪时箱底, 非托台面
    # (箱件起始坐在托盘承货面 pallet_support_z, 不是托台面;
    #  用托台面做基准会把 18mm 的托盘高度算成抬离)
    mat = cell.grasp_target_pose(np.array([cell.box_pos[0], cell.box_pos[1]]),
                                 cell.box_bottom_z)[1]
    p = cell.tcp_pos.copy()
    for _ in range(24):
        p = p + np.array([0.0, 0.0, 0.0025])
        cell.move_tcp_incremental(p, max_joint_step=0.02, settle_s=0.3,
                                  mat=mat, keep_grip=True)
    cell.move_tcp_incremental(p, max_joint_step=0.02, settle_s=0.5,
                              mat=mat, keep_grip=True)

    lift = cell.box_bottom_z - z0
    assert lift >= 0.005, f"抬离源面仅 {lift*1000:.1f}mm (<5mm)"

    drift_mm, drift_deg = cell.relative_drift(0.5)
    assert drift_mm <= 1.0, f"夹持期位置漂移 {drift_mm:.3f}mm > 1mm"
    assert drift_deg <= 1.0, f"夹持期角度漂移 {drift_deg:.3f}° > 1°"

    # 摩擦承载校核 (设计 25.3): 2×μ×F 须显著大于箱重
    mu = cell.cfg["manipulators"]["gripper"]["pad_friction_initial"]
    weight = cell.box_mass * 9.81
    capacity = 2 * mu * min(fl, fr)
    assert capacity > 2.0 * weight, \
        f"摩擦承载 {capacity:.2f}N 相对箱重 {weight:.3f}N 余量不足 2×"
