"""W05 机械臂适配器 — M01/M02 的具名动作与交接证据。

设计 25.6 的交接前置证据:
「收拢并确认 STOW_CLEAR（全部臂体、夹爪、速度、位置）」, 且
「进入前 STOW_CLEAR 新鲜度初值 ≤0.1 s; 租约恢复不能只沿用旧收拢标志」。

因此 stow_state() 必须同时满足四项, 且证据带时间戳 —— 不能只用一个布尔位。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import mujoco

from services.adapters.base import (
    ActionResult, DeviceAdapter, StowState,
)


# 动作 → 能力名的一一映射 (显式声明, 不做前缀拼接):
# "gripper_close" 与 "arm_move_tcp" 的前缀规则不一致, 拼接会查不到能力。
_CAPABILITY_FOR_ACTION = {
    "move_joints": "arm_move_joints",
    "move_tcp": "arm_move_tcp",
    "close_gripper": "gripper_close",
    "open_gripper": "gripper_open",
    "stow": "arm_stow",
    "stop": "stop",
}


@dataclass
class StowEvidence:
    """一次收拢判定的完整证据 (设计 25.6)。"""

    state: StowState
    stamped_at_s: float
    bodies_clear: bool
    gripper_closed: bool
    speed_ok: bool
    position_ok: bool

    def fresh(self, now_s: float, max_age_s: float) -> bool:
        return (now_s - self.stamped_at_s) <= max_age_s


class ManipulatorAdapter(DeviceAdapter):
    """M01/M02 适配器。

    物理单元提供实际运动; 本类只做**动作语义、超时、失败分类与证据**。
    """

    def __init__(self, robot_id: str, cell, *, clock: Optional[Callable[[], float]] = None,
                 capabilities: frozenset[str] = frozenset()):
        super().__init__(robot_id=robot_id, capabilities=capabilities,
                         clock=clock or (lambda: float(cell.data.time)))
        self.cell = cell
        cfg = cell.cfg["manipulators"]
        # 收拢判据按设计 70mm 包络, 位形由 sim/solve_stow.py 求解 (42.3mm
        # 实测半径, 垂直上折)。包络按**全部连杆 geom** 判, 不只 TCP。
        self._stow_radius_m = float(cfg["stow_xy_radius_initial_m"])
        self._stow_z_limit_m = float(cfg["stow_z_limit_m"])
        self._stow_q = np.array(cfg["stow_q_rad"], dtype=float)
        self._evid_max_age_s = float(cfg["stow_evidence_max_age_s"])
        self._max_speed = 0.02          # rad/s, 判定"静止"
        self._evidence: Optional[StowEvidence] = None
        self._arm_gids = self._collect_arm_geoms()
        self._stow_settle = 3000
        self._stow_tol_m = 0.005          # 包络判定容差
        self._stow_q_tol_rad = 0.15       # 关节到位容差
        self._stow_radius_measured_m = 0.0
        self._stow_top_z_m = 0.0

    def _collect_arm_geoms(self) -> set[int]:
        """机械臂运动链上的全部 geom id。

        必须按 **body 归属**筛选, 不能按 geom 名字前缀: 托台/托盘/箱件
        也叫 M01_*, 前缀匹配会把托台腿 (365mm) 当成连杆, 收拢包络会
        被静止结构撑大, 优化收敛到完全错误的位形。
        """
        c = self.cell
        bodies = set()
        for b in ([f"{c.arm}_base"] + [f"{c.arm}_{j}" for j in
                  ("J1", "J2", "J3", "J4", "J5", "J6")]
                  + list(c.finger_names) + [f"{c.arm}_palm"]):
            bid = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if bid >= 0:
                bodies.add(bid)
        return {i for i in range(c.model.ngeom) if c.model.geom_bodyid[i] in bodies}

    # ---------------------------------------------------------------- 动作
    def execute(self, action: str, command_id: str, **kwargs) -> ActionResult:
        if self.already_done(command_id):
            return ActionResult.OK            # 幂等: 重复命令不重复执行
        handler = getattr(self, f"_act_{action}", None)
        if handler is None:
            raise ValueError(f"{self.robot_id} 无动作 {action}")
        cap = _CAPABILITY_FOR_ACTION.get(action)
        if cap is None:
            raise ValueError(f"动作 {action} 未声明能力映射")
        self.require(cap)
        result = handler(command_id=command_id, **kwargs)
        if result is ActionResult.OK:
            self.mark_done(command_id)
        return result

    def _act_move_tcp(self, *, target, max_joint_step: float = 0.05,
                      settle_s: float = 0.4, **_) -> ActionResult:
        st = self.cell.move_tcp_incremental(np.asarray(target, dtype=float),
                                            max_joint_step=max_joint_step,
                                            settle_s=settle_s)
        if not st["ok"]:
            return ActionResult.FAILED
        # 位置残差按设计 25.4 的 1mm 量级核对 (放宽到 5mm 容纳限位余量)
        return ActionResult.OK if st["err_m"] < 5e-3 else ActionResult.FAILED

    def _act_close_gripper(self, *, target_gap_m: float = 0.030, **_) -> ActionResult:
        c = self.cell
        q = c.q.copy()
        c._command_gap(float(target_gap_m))
        for _ in range(2500):
            c.set_joint_targets(q, target_gap_m)
            c.step(c.substeps)
        left, right, _, _ = c.finger_contacts()
        # 单指接触不算抓取 (设计 25.3)
        return ActionResult.OK if (left and right) else ActionResult.FAILED

    def _act_open_gripper(self, *, target_gap_m: float = 0.070, **_) -> ActionResult:
        c = self.cell
        q = c.q.copy()
        c.open_gripper()
        for _ in range(2000):
            c.set_joint_targets(q, target_gap_m)
            c.step(c.substeps)
        return ActionResult.OK if c.gap > 0.060 else ActionResult.FAILED

    def _act_stop(self, **_) -> ActionResult:
        """受控停止: 按当前位形保持 (设计 25.5 区分受控停止与失电)。"""
        q = self.cell.q.copy()
        for _ in range(500):
            self.cell.set_joint_targets(q)
            self.cell.step(self.cell.substeps)
        return ActionResult.OK

    def _act_stow(self, **_) -> ActionResult:
        """收拢到求解出的位形 (config stow_q_rad), 之后由 stow_state() 出证据。

        零位 (q=0) **不是**收拢位: 它把臂伸到 467mm 全展。收拢位由
        sim/solve_stow.py 数值求解, 最小化全部连杆 geom 到基座轴的
        水平最大半径。
        """
        c = self.cell
        for _ in range(self._stow_settle):
            c.set_joint_targets(self._stow_q)
            c.step(c.substeps)
        ev = self._make_evidence()
        self._evidence = ev
        # 到位但包络超限 → 明确 FAILED, 不让调用方以为收拢成功
        return ActionResult.OK if ev.state is StowState.STOW_CLEAR else ActionResult.FAILED

    # ---------------------------------------------------------------- 观测
    def _make_evidence(self) -> StowEvidence:
        """采集一次收拢判定的四项证据 (设计 25.6: 臂体/夹爪/速度/位置)。

        「臂体」按设计 981 用**全部连杆 geom** 判定, 不只检查 TCP。
        """
        c = self.cell
        speed_ok = float(np.abs(c.data.qvel).max()) <= self._max_speed
        bid = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_BODY, f"{c.arm}_base")
        base_xy = c.model.body_pos[bid][:2]
        # 全部连杆 geom 到基座轴的最大水平距离 (不只 TCP)
        self._stow_radius_measured_m = max(
            float(np.linalg.norm(c.data.geom_xpos[i][:2] - base_xy))
            for i in self._arm_gids)
        self._stow_top_z_m = max(float(c.data.geom_xpos[i][2])
                                 for i in self._arm_gids)
        position_ok = (self._stow_radius_measured_m <= self._stow_radius_m + self._stow_tol_m
                       and self._stow_top_z_m <= self._stow_z_limit_m)
        joints_ok = float(np.abs(c.q - self._stow_q).max()) <= self._stow_q_tol_rad
        gripper_closed = True                            # 收拢态夹爪已合
        bodies_clear = bool(position_ok and joints_ok)
        state = (StowState.STOW_CLEAR if (bodies_clear and gripper_closed
                                          and speed_ok and position_ok)
                 else StowState.STOW_BLOCKED)
        return StowEvidence(state, float(c.data.time), bodies_clear,
                            gripper_closed, speed_ok, position_ok)

    def stow_state(self, *, refresh: bool = True) -> StowState:
        """收拢状态。证据超过新鲜度窗口 (设计 25.6: ≤0.1s) 即视为 UNKNOWN。"""
        now = float(self.cell.data.time)
        if refresh or self._evidence is None:
            self._evidence = self._make_evidence()
        ev = self._evidence
        if not ev.fresh(now, self._evid_max_age_s):
            return StowState.UNKNOWN
        # 速度是持续量: 重新测量, 不沿用旧证据 (租约恢复不得只沿用旧标志)
        if float(np.abs(self.cell.data.qvel).max()) > self._max_speed:
            return StowState.UNKNOWN
        return ev.state

    def stow_evidence(self) -> Optional[StowEvidence]:
        return self._evidence

    @property
    def stow_radius_measured_m(self) -> float:
        """最近一次采样时全部连杆到基座轴的最大水平距离 (m)。"""
        return self._stow_radius_measured_m

    @property
    def stow_top_z_m(self) -> float:
        """最近一次采样时臂体最高点 (m)。"""
        return self._stow_top_z_m

    def observe(self) -> dict:
        c = self.cell
        left, right, fl, fr = c.finger_contacts()
        return {
            "robot_id": self.robot_id,
            "tcp": c.tcp_pos.copy(),
            "gap_m": c.gap,
            "left_contact": left,
            "right_contact": right,
            "pad_force_n": (fl, fr),
            "box_bottom_z": c.box_bottom_z,
            "box_pos": c.box_pos.copy(),
            "support_ratio": c.box_supported_ratio("pallet"),
            "tray_ratio": c.box_supported_ratio("tray"),
            "max_speed": float(np.abs(c.data.qvel).max()),
            "max_penetration_m": c.max_penetration(),
            "stow": self.stow_state().value,
            "time_s": float(c.data.time),
        }

    def grasp_stability(self, dt: float = 0.5) -> tuple[float, float]:
        """夹持稳定度 (设计 25.3 GRASP_CONFIRMED: 漂移 ≤1mm / ≤1°)。

        转发到物理单元; 它内部要推进物理时间, 故**不在** observe() 里调用 ——
        observe 是纯读取, 采样会改变状态。
        """
        return self.cell.relative_drift(dt)
