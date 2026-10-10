"""车辆低层控制器 —— 100 Hz 轮速闭环 (设计 927 行)。

设计 927:
「低层控制按 100 Hz 运行, 物理步长 1 ms, 使用模拟编码器或关节反馈
生成轮速和升降控制。**轮速目标依据差速模型换算, 输出转矩须限幅**;
叉架位置目标经 10 mm/s 速度限制生成平滑参考, 驱动力包含重力补偿、
反馈与限幅, 不通过直接写关节位置跳过执行器。」

关键 (W04 追查中定位到的缺失):
场景里车轮是 `<motor>` 执行器, ctrl 是**转矩**。直接给满扭矩时车辆
只跑到 ~10 mm/s (扭矩不足以加速 1.5kg 车体), 而提高扭矩上限又会让
轮子空转打滑 —— 因为**没有任何环节把速度闭环住**。设计的 100 mm/s
要求由本控制器实现: 轮速 PI → 转矩限幅。

差速模型 (本项目布局, 直行沿本体 +X):
    v_body = (ω_L + ω_R) / 2 · r
    ω_yaw = (ω_R − ω_L) / track · r
反解出达到目标 (v, ω_yaw) 所需的左右轮速。
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class WheelCtrl:
    ctrl_hz: float = 100.0        # 设计 927: 低层 100 Hz
    kp: float = 0.25              # N·m per rad/s
    ki: float = 0.05
    torque_max_nm: float = 0.5    # 设计 wheel_drive_torque_max_nm
    speed_max_mps: float = 0.10   # 设计 robots.*.speed_max_mps
    wheel_radius_m: float = 0.025
    i_limit: float = 2.0
    i_deadband_radps: float = 0.05   # 误差进入此范围时冻结积分(防低速死区极限环)


class DiffDriveController:
    """差速驱动低层控制器 (100 Hz)。

    只用**编码器反馈** (关节角速度), 不读 qpos 目标、不跳过执行器 ——
    符合设计 927「不通过直接写关节位置跳过执行器」。
    """

    def __init__(self, cell, body_name: str, wheel_actuators: list[str],
                 rear_actuators: list[str], lift_actuator: str | None = None,
                 *, ctrl_hz: float = 100.0, wheel_radius_m: float = 0.025,
                 track_m: float = 0.120) -> None:
        self.cell = cell
        m = cell.model
        self.act = {a: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                    for a in wheel_actuators + rear_actuators
                    + ([lift_actuator] if lift_actuator else [])}
        self.left = wheel_actuators
        self.right = rear_actuators
        self.lift_act = self.act[lift_actuator] if lift_actuator else None
        # 传进来的是执行器名, 要经传动找到关节 dof
        self.dof_l = [self._joint_dof(n) for n in wheel_actuators]
        self.dof_r = [self._joint_dof(n) for n in rear_actuators]
        self.root_dof = self._dof(f"{body_name}_root")
        self.body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.wheel_r = wheel_radius_m
        self.track = track_m
        self.ctrl = WheelCtrl(ctrl_hz=ctrl_hz, wheel_radius_m=wheel_radius_m)
        self._integ_l = 0.0
        self._integ_r = 0.0
        self.lift_target_m: float | None = None
        self.lift_speed_mps = 0.010       # 设计 lift_speed_max_mps

    def _dof(self, name: str, base: str | None = None) -> int:
        """名称 → 该关节的 dof 地址。

        传入的是**关节名**。若传执行器名 (`..._drv`) 查不到关节,
        `mj_name2id` 返回 -1, 再取 `jnt_dofadr[-1]` 会静默拿到最后一个
        dof —— 实测左右轮因此都指向 dof 44, 反馈全错, 控制器
        只能靠开环扭矩推车, 速度恒为 10mm/s。
        """
        jid = mujoco.mj_name2id(self.cell.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"关节不存在: {name}")
        return int(self.cell.model.jnt_dofadr[jid])

    def _joint_dof(self, actuator_name: str) -> int:
        """执行器名 → 其传动关节的 dof 地址。"""
        aid = mujoco.mj_name2id(self.cell.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                               actuator_name)
        if aid < 0:
            raise KeyError(f"执行器不存在: {actuator_name}")
        jid = int(self.cell.model.actuator_trnid[aid][0])
        return int(self.cell.model.jnt_dofadr[jid])

    # ------------------------------------------------------------ 观测
    def wheel_speeds(self) -> tuple[float, float]:
        d = self.cell.data
        return (float(np.mean([d.qvel[i] for i in self.dof_l])),
                float(np.mean([d.qvel[i] for i in self.dof_r])))

    def body_velocity(self) -> np.ndarray:
        """车体在世界系的线速度与角速度 (根 freejoint 的 3 个 dof)。"""
        d = self.cell.data
        return d.qvel[self.root_dof:self.root_dof + 3].copy()

    def heading(self) -> float:
        """车体 yaw (rad)。"""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.cell.data.xquat[self.body_id])
        R = R.reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    # ------------------------------------------------------------ 差速反解
    def wheel_targets(self, v_mps: float, yaw_rate: float) -> tuple[float, float]:
        """(v, ω) → (ω_L, ω_R)。差速模型反解。"""
        wl = (v_mps - yaw_rate * self.track / 2.0) / self.wheel_r
        wr = (v_mps + yaw_rate * self.track / 2.0) / self.wheel_r
        return wl, wr

    # ------------------------------------------------------------ 一个控制周期
    def step(self, v_mps: float, yaw_rate: float = 0.0,
             dt: float | None = None) -> dict:
        """跑一个低层周期: 轮速 PI → 转矩限幅 → 下发。"""
        d = self.cell.data
        dt = dt or (1.0 / self.ctrl.ctrl_hz)
        v_cmd = float(np.clip(v_mps, -self.ctrl.speed_max_mps,
                              self.ctrl.speed_max_mps))
        wl, wr = self.wheel_targets(v_cmd, yaw_rate)
        vl, vr = self.wheel_speeds()
        tm = self.ctrl.torque_max_nm
        db = self.ctrl.i_deadband_radps

        # 抗积分饱和 (back-calculation) + 死区冻结。
        #
        # 死区语义: 误差进入 ±db 内时**停止累积**(冻结), 不做任何清零。
        # 曾经的错误实现是在死区内 `min(integ, 0.0)` —— 把任何**正**积分
        # 强制清零。50 mm/s 指令下轮速误差恰好在 ±0.05 rad/s 死区边界上
        # 反复穿越, 于是积分被不断清零又重新累积, 形成 1.8 s 周期的大幅
        # 极限环 (车速 50→90→35 mm/s 往复, 积分 0.19→−1.38)。
        # 冻结只阻止进一步累积, 不改已有值。
        err_l = wl - vl
        if abs(err_l) >= db:
            self._integ_l += err_l * dt
            ul_test = self.ctrl.kp * err_l + self.ctrl.ki * self._integ_l
            if ul_test > tm:
                self._integ_l -= (ul_test - tm) / max(self.ctrl.ki, 1e-9)
            elif ul_test < -tm:
                self._integ_l -= (ul_test + tm) / max(self.ctrl.ki, 1e-9)
            self._integ_l = float(np.clip(self._integ_l,
                                          -self.ctrl.i_limit, self.ctrl.i_limit))

        err_r = wr - vr
        if abs(err_r) >= db:
            self._integ_r += err_r * dt
            ur_test = self.ctrl.kp * err_r + self.ctrl.ki * self._integ_r
            if ur_test > tm:
                self._integ_r -= (ur_test - tm) / max(self.ctrl.ki, 1e-9)
            elif ur_test < -tm:
                self._integ_r -= (ur_test + tm) / max(self.ctrl.ki, 1e-9)
            self._integ_r = float(np.clip(self._integ_r,
                                          -self.ctrl.i_limit, self.ctrl.i_limit))

        ul = float(np.clip(self.ctrl.kp * (wl - vl) + self.ctrl.ki * self._integ_l, -tm, tm))
        ur = float(np.clip(self.ctrl.kp * (wr - vr) + self.ctrl.ki * self._integ_r, -tm, tm))

        for a in self.left:
            d.ctrl[self.act[a]] = ul
        for a in self.right:
            d.ctrl[self.act[a]] = ur
        return {"v_cmd": v_cmd, "wl": wl, "wr": wr, "vl": vl, "vr": vr,
                "torque_l": ul, "torque_r": ur}

    def run(self, v_mps: float, seconds: float, yaw_rate: float = 0.0,
            on_tick=None) -> dict:
        """连续跑 seconds 秒低层闭环。"""
        dt = self.cell.model.opt.timestep
        n_ctrl = int(seconds * self.ctrl.ctrl_hz)
        steps_per_ctrl = max(1, int((1.0 / self.ctrl.ctrl_hz) / dt))
        last = {}
        for _ in range(n_ctrl):
            last = self.step(v_mps, yaw_rate)
            for _ in range(steps_per_ctrl):
                if self.lift_target_m is not None and self.lift_act is not None:
                    self._drive_lift()
                self.cell.step(1)
            if on_tick:
                on_tick(self)
        bv = self.body_velocity()
        return {"v_body": bv, "wheel": self.wheel_speeds(),
                "heading": self.heading(), "last": last}

    # ------------------------------------------------------------ 叉架
    def command_lift(self, target_z: float) -> None:
        self.lift_target_m = float(target_z)

    def _drive_lift(self) -> None:
        """叉架位置环: 速度限制 10 mm/s → 转矩 (设计 927)。"""
        d = self.cell.data
        jid = mujoco.mj_name2id(self.cell.model, mujoco.mjtObj.mjOBJ_JOINT,
                               self._lift_joint_name())
        qadr = int(self.cell.model.jnt_qposadr[jid])
        cur = float(d.qpos[qadr])
        err = self.lift_target_m - cur
        step = float(np.clip(err, -self.lift_speed_mps, self.lift_speed_mps))
        target = cur + step * self.cell.model.opt.timestep
        kp = 20.0
        d.ctrl[self.lift_act] = float(np.clip(kp * (target - cur),
                                              -self._lift_force(), self._lift_force()))

    def _lift_joint_name(self) -> str:
        return next(k for k, v in self.act.items() if v == self.lift_act) \
            .replace("_drv", "")

    def _lift_force(self) -> float:
        return float(self.cell.model.actuator_forcerange[self.lift_act][1])