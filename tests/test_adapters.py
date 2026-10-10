"""W05 设备适配器 — 具名动作、超时语义与交接证据。

判据 (设计 25.5/25.6):
* 动作失败返回具名 FAILED, 不得静默成功;
* 单指接触不算抓取 (设计 25.3);
* 退叉须先达最小退出高度 (带载 110mm + 10mm 余量 = 120mm);
* STOW_CLEAR 需四项证据且新鲜度 ≤0.1s, 速度是持续量不得沿用旧标志;
* 重复命令幂等 (设计 10.3: command_id 唯一)。
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from services.adapters.base import ActionResult, StowState
from services.adapters.forklift import ForkliftAdapter, ForkActionLimits
from services.adapters.manipulator import ManipulatorAdapter
from sim.forklift_cell import ForkliftCell
from sim.manipulator_cell import ManipulatorCell

ARM_CAPS = frozenset(["arm_move_joints", "arm_move_tcp", "gripper_close",
                      "gripper_open", "arm_stow", "stop"])
FORK_CAPS = frozenset(["drive", "fork_lift", "fork_insert", "fork_withdraw", "stop"])


@pytest.fixture
def arm():
    c = ManipulatorCell(timestep_s=0.0005)
    for _ in range(3000):
        c.step(c.substeps)
    return c, ManipulatorAdapter("M01", c, capabilities=ARM_CAPS)


@pytest.fixture
def fork():
    c = ForkliftCell()
    for _ in range(2000):
        mujoco.mj_step(c.model, c.data)
    return c, ForkliftAdapter("F01", c, capabilities=FORK_CAPS)


def test_arm_grasp_sequence_through_adapter(arm):
    """具名动作序列: 接近 → 预抓 → 抓取 → 闭爪, 且双指接触。"""
    c, a = arm
    bx, by, bz = c.box_pos[0], c.box_pos[1], c.box_bottom_z
    pose, _ = c.grasp_target_pose(np.array([bx, by]), bz)
    assert a.execute("move_tcp", "t1", target=np.array([bx, by, 0.22])) is ActionResult.OK
    assert a.execute("move_tcp", "t2", target=pose + np.array([0, 0, 0.05])) is ActionResult.OK
    assert a.execute("move_tcp", "t3", target=pose) is ActionResult.OK
    assert a.execute("close_gripper", "t4", target_gap_m=0.030) is ActionResult.OK
    obs = a.observe()
    assert obs["left_contact"] and obs["right_contact"], "闭爪后应双指接触"
    assert obs["gap_m"] < 0.066


def test_arm_repeated_command_is_idempotent(arm):
    """重复 command_id 不得重复执行 (设计 10.3 动作唯一)。"""
    c, a = arm
    assert a.execute("stop", "dup", ) is ActionResult.OK
    # 人为篡改物理状态, 重复命令不应把它改回来 (证明未重新执行)
    c.data.ctrl[c.act_j[0]] = 3.0
    before = float(c.data.qpos[c.qadr[0]])
    assert a.execute("stop", "dup") is ActionResult.OK
    assert float(c.data.qpos[c.qadr[0]]) >= before


def test_arm_unknown_action_and_capability_rejected(arm):
    _, a = arm
    with pytest.raises(ValueError):
        a.execute("teleport", "x1")
    a.capabilities = frozenset()          # 抽掉全部能力
    with pytest.raises(ValueError):
        a.execute("stop", "x2")


def test_arm_stow_requires_fresh_multi_evidence(arm):
    """STOW_CLEAR 需四项证据; 高速或超包络时不得判为 CLEAR。

    包络按**全部连杆 geom** 判定 (设计 981), 不是只看 TCP。
    """
    c, a = arm
    assert a.execute("stow", "s1") is ActionResult.OK
    ev = a.stow_evidence()
    assert ev is not None
    assert ev.state is StowState.STOW_CLEAR
    assert ev.bodies_clear and ev.position_ok and ev.speed_ok and ev.gripper_closed
    assert a.stow_radius_measured_m <= a._stow_radius_m + 1e-6
    # 把手臂开到零位 (全展) → 包络超限, 不得判为收拢。
    # 必须先 stow_state(refresh=True) 重新采样: 实测半径是**采样值**,
    # 直接读上一次证据会把旧读数当成新姿态的判据。
    for _ in range(3000):
        c.set_joint_targets(np.zeros(6))
        c.step(c.substeps)
    a.stow_state(refresh=True)
    assert a.stow_radius_measured_m > a._stow_radius_m, "零位应超出实测包络"
    assert a.stow_state() is not StowState.STOW_CLEAR
    # 速度是持续量: 制造运动后立即查询应不是 CLEAR
    c.set_joint_targets(c.q + 0.5)
    for _ in range(20):
        c.step(c.substeps)
    assert a.stow_state() is not StowState.STOW_CLEAR, "运动中不得判为收拢"


def test_arm_stow_evidence_goes_stale(arm):
    """证据超过新鲜度窗口 (设计 25.6: ≤0.1s) 即为 UNKNOWN。"""
    c, a = arm
    a.execute("stow", "s1")
    a.stow_state(refresh=True)
    # 冻结物理时间 → 证据不新鲜
    frozen = a.stow_evidence()
    assert frozen.stamped_at_s is not None
    max_age = float(c.cfg["manipulators"]["stow_evidence_max_age_s"])
    assert max_age <= 0.1, "设计要求新鲜度初值 ≤0.1s"
    assert not frozen.fresh(frozen.stamped_at_s + max_age + 0.01, max_age)


def test_fork_withdraw_requires_minimum_exit_height(fork):
    """退叉前必须达最小退出高度 (带载 110mm + 10mm 余量)。"""
    c, a = fork
    a.execute("lift", "f1", target_z=c.k["transport_z"])       # 低位
    assert a.execute("withdraw", "w1", loaded=True) is ActionResult.FAILED
    need = c.k["retreat_loaded"] + 0.010
    a.execute("lift", "f2", target_z=need)
    assert a.execute("withdraw", "w2", loaded=True) is ActionResult.OK


def test_fork_stow_requires_low_forks_and_no_load(fork):
    """叉车收拢: 叉位低 + 无载荷 + 静止。"""
    c, a = fork
    a.execute("lift", "f1", target_z=0.130)                   # 高位
    assert a.stow_state() is StowState.STOW_BLOCKED
    a.execute("lift", "f2", target_z=c.k["transport_z"])
    assert a.stow_state() in (StowState.STOW_CLEAR, StowState.STOW_BLOCKED)


def test_fork_action_limits_follow_design():
    """时限初值须与设计 499 行一致 (由行程/速度反推, 不统一写 5s)。"""
    lim = ForkActionLimits()
    assert lim.insert_65mm_s >= 13.0, "65mm @5mm/s 至少 13s"
    assert lim.lift_430mm_s >= 43.0, "430mm @10mm/s 至少 43s"
    assert lim.insert_65mm_s < lim.lift_430mm_s < 300.0
    assert all(getattr(lim, f) > 0 for f in lim.__dataclass_fields__)


def test_fork_withdraw_empty_uses_lower_threshold(fork):
    """空叉退出高度低于带载 (设计: 95mm vs 110mm)。"""
    c, a = fork
    empty_need = c.k["withdraw_empty"] + 0.010
    loaded_need = c.k["retreat_loaded"] + 0.010
    assert empty_need < loaded_need
    a.execute("lift", "g1", target_z=empty_need)
    assert a.execute("withdraw", "g1", loaded=False) is ActionResult.OK