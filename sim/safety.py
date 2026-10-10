"""车辆安全与故障响应 —— T10「模拟急停与失电」的实现层。

设计判据:
|T10 模拟急停与失电|低位和第三层分别注入停止、驱动失电与制动失效|
     |模拟响应与停止距离有轨迹; 正常制动保持, 制动失效被检测并记为失败;
       实体试验留待 H0/H1|

三种物理模式必须**可区分**, 不能都退化成"停下来":
|模式|物理含义|期望结果|
|---|---|---|
|`estop` 正常急停|驱动器主动制动, 保持当前位置|短距离停住, 载荷不滑, 记为**安全**|
|`drive_power_loss` 驱动失电|驱动器不出力, 制动**仍有效**|惯性滑行后被制动拉住, 距离更大, 仍安全|
|`brake_fail` 制动失效|无力矩约束, 只有残余摩擦|下滑/漂移, **必须记为失败**并冻结资源|

关键: 三者都用同一段轨迹记录, 但 `brake_fail` 无论位移大小都必须
`counted_as_failure=True` —— 设计 S07 要求「检测下滑和风险, 冻结业务
资源, **无虚假成功**」。位移小的失效也仍是失效。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import mujoco
import numpy as np

from sim.forklift_cell import ForkliftCell


class StopMode(str, Enum):
    ESTOP = "estop"                      # 正常急停
    DRIVE_POWER_LOSS = "drive_power_loss"  # 驱动失电
    BRAKE_FAIL = "brake_fail"            # 制动失效


@dataclass
class StopRecord:
    """一次停止试验的轨迹与判定 (设计 R11: 停止轨迹)。"""

    mode: StopMode
    level: str                       # "low" / "third"
    t_inject_s: float
    x_inject_m: float
    z_inject_m: float
    loaded: bool
    x_final_m: float = 0.0
    z_final_m: float = 0.0
    stop_distance_m: float = 0.0
    drop_m: float = 0.0              # 载荷高度变化 (下滑量)
    slip_m: float = 0.0              # 载荷相对叉的滑移
    held: bool = False               # 是否被制动保持
    safe: bool = False
    failure: bool = False
    reason: str = ""
    traj: list[tuple[float, float, float, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"mode": self.mode.value, "level": self.level,
                "t_inject_s": round(self.t_inject_s, 4),
                "x_inject_m": round(self.x_inject_m, 5),
                "z_inject_m": round(self.z_inject_m, 5),
                "loaded": self.loaded,
                "stop_distance_m": round(self.stop_distance_m, 5),
                "drop_m": round(self.drop_m, 5),
                "slip_m": round(self.slip_m, 5),
                "held": self.held, "safe": self.safe,
                "failure": self.failure, "reason": self.reason,
                "trajectory_points": len(self.traj)}


#: 载荷判定阈值 (设计 pallet cargo_mass_max_kg 0.2kg, 托盘自重 0.045kg)
#: 停止后允许的残余速度; 超出即"未被制动保持"
RESIDUAL_SPEED_TOL = 5e-3          # m/s
#: 载荷高度允许的变化 (托盘被叉捕获后应完全随车)
LOAD_DROP_TOL_M = 2e-3
#: 载荷相对叉的允许滑移
LOAD_SLIP_TOL_M = 2e-3
#: ESTOP 减速度。契约 (T10) 要求 0.2m/s 急停 <2mm → 减速度 ≥10 m/s²;
#: 载荷不滑移要求减速度 < μg。叉孔摩擦 1.4 (μg=13.7), 取 12 留 14% 余量,
#: 停止距离 v²/2a = 1.67mm。位置环锁存的瞬时停 (≈500 m/s²) 任何未捆绑
#: 载荷都跟不住, 真实控制器急停按载荷保护斜坡减速。
ESTOP_DECEL_MPS2 = 12.0


class VehicleSafety:
    """在 ForkliftCell 上注入停止模式并判定结果。"""

    def __init__(self, cell: ForkliftCell):
        self.cell = cell
        self.resources_frozen = False
        self.last: Optional[StopRecord] = None
        # 记住标称传动/阻尼, 试验复位时还原 (不靠写死常量)
        self._nominal_damping = cell.model.dof_damping.copy()
        self._nominal_gear = cell.model.actuator_gear.copy()

    # ------------------------------------------------------------ 状态读取
    def _x(self) -> float:
        return float(self.cell.axle_x)

    def _z(self) -> float:
        return float(self.cell.fork_top_z)

    def _loaded(self) -> bool:
        return bool(self.cell.pallet_fork_contact())

    def _pallet_z(self) -> float:
        return float(self.cell.pallet_bottom_z)

    def _pallet_x(self) -> float:
        p = self.cell.data.xpos[self.cell.pallet_bid]
        return float(p[0])

    def _speed(self) -> float:
        jid = self.cell.j_x
        dadr = int(self.cell.model.jnt_dofadr[jid])
        return float(abs(self.cell.data.qvel[dadr]))

    # ------------------------------------------------------------ 模式开关
    def _actuators_on(self) -> None:
        """恢复传动 —— 驱动器有力矩 (急停/正常)。"""
        self.cell.model.actuator_gear[:, 0] = 1.0

    def _actuators_off(self) -> None:
        """切断传动 —— 驱动器不出力矩 (失电/制动失效)。

        `actuator_gear=0` 是真正的"驱动失电": position 执行器的
        kp·(ctrl−q) 整项消失。只把 ctrl 设成当前位置是**主动保持**,
        那是急停, 不是失电 —— 两者物理含义完全不同。
        """
        self.cell.model.actuator_gear[:, 0] = 0.0

    def _brakes_off(self) -> None:
        """制动失效: 关节阻尼归零, 只剩接触摩擦。"""
        self.cell.model.dof_damping[:] = 0.0

    def _hold(self) -> None:
        """把两个执行器切到**保持**目标 (主动制动的前提)。"""
        c = self.cell
        c.data.ctrl[c.a_x] = float(c.data.qpos[c.qadr_x])
        c.data.ctrl[c.a_lift] = float(c.data.qpos[c.qadr_lift])

    # ------------------------------------------------------------ 注入
    def apply(self, mode: StopMode, level: str, duration_s: float = 2.0,
              initial_v_mps: float = 0.0) -> StopRecord:
        """注入停止/失电/制动失效, 记录轨迹并判定。

        三种模式的物理区别 (必须真的不一样, 否则 T10 判据无意义):
          ESTOP           驱动器有电且主动保持 → 强力最短距离停住
          POWER_LOSS      传动断开 (gear=0), 阻尼仍在 → 惯性滑行更远
          BRAKE_FAIL      传动断开 + 阻尼归零   → 只剩摩擦, 下滑最大
        """
        c = self.cell
        self._actuators_on()
        c.model.actuator_gear[:, 0] = self._nominal_gear[:, 0].copy()
        c.model.dof_damping[:] = self._nominal_damping.copy()
        self._hold()
        for _ in range(8000):                    # 先稳定到托盘完全落位
            mujoco.mj_step(c.model, c.data)

        if mode is StopMode.ESTOP:
            self._actuators_on()
            self._hold()
        else:
            self._actuators_off()

        if initial_v_mps:
            dadr = int(c.model.jnt_dofadr[c.j_x])
            c.data.qvel[dadr] = float(initial_v_mps)
            # 托盘同速注入: 只给车体初速会让托盘以 0.2m/s 的相对速度
            # 在叉上摩擦滑行 (v²/2μg ≈ 2.6mm), 把注入伪影记成载荷滑移。
            pj = int(np.argmax(c.model.jnt_bodyid == c.pallet_bid))
            pdadr = int(c.model.jnt_dofadr[pj])
            c.data.qvel[pdadr] = float(initial_v_mps)   # freejoint 首 dof = 平移 x

        # 保持目标在**注入时刻固定一次**, 之后不再更新。
        # 若每步都把 ctrl 重设成当前 qpos, 位置环退化成"跟随当前位置":
        # 任何外力 (重力、载荷) 都会被当成新的目标, 叉会一点点沉下去
        # (实测 2 秒内从 0.4220 落到 0.3735), 载荷随之掉到横梁上。
        # 正确做法是像真实驱动器一样**锁存**注入瞬间的目标。
        self._hold()

        rec = StopRecord(mode=mode, level=level,
                         t_inject_s=float(c.data.time),
                         x_inject_m=self._x(), z_inject_m=self._z(),
                         loaded=self._loaded())
        z0 = self._pallet_z()
        # 滑移 = 托盘**相对叉**的位移。此前用世界坐标位移, 把车体制动
        # 距离 (0.2m/s 急停约 2mm) 也算成"载荷滑移", 载货急停必然超
        # 2mm 容差而误判不安全。
        rel0 = self._pallet_x() - self.cell.axle_x
        t0 = float(c.data.time)
        s = 1.0 if initial_v_mps >= 0 else -1.0
        v0 = abs(float(initial_v_mps))
        t_stop = v0 / ESTOP_DECEL_MPS2
        x_stop = rec.x_inject_m + s * v0 * v0 / (2 * ESTOP_DECEL_MPS2)
        steps = max(1, int(duration_s / c.model.opt.timestep))
        sample_every = max(1, steps // 120)

        for i in range(steps):
            if mode is StopMode.BRAKE_FAIL:
                self._actuators_off()
                self._brakes_off()               # 制动也失效
            elif mode is StopMode.ESTOP and initial_v_mps:
                # 减速度受限的急停斜坡: 载荷摩擦跟得住, 不滑移
                t = c.data.time - t0
                xt = (rec.x_inject_m + s * (v0 * t - 0.5 * ESTOP_DECEL_MPS2 * t * t)
                      if t < t_stop else x_stop)
                c.data.ctrl[c.a_x] = xt - c.q0_x
            # ESTOP / POWER_LOSS: 保持目标保持不变, 只继续步进
            mujoco.mj_step(c.model, c.data)
            if i % sample_every == 0:
                rec.traj.append((float(c.data.time), self._x(), self._z(),
                                 self._speed()))

        rec.x_final_m = self._x()
        rec.z_final_m = self._z()
        rec.stop_distance_m = abs(rec.x_final_m - rec.x_inject_m)
        rec.drop_m = abs(self._pallet_z() - z0)
        rec.slip_m = abs((self._pallet_x() - self.cell.axle_x) - rel0)
        rec.held = self._speed() <= RESIDUAL_SPEED_TOL

        if mode is StopMode.BRAKE_FAIL:
            # 制动失效: **无论位移多小都记为失败** (设计 S07 无虚假成功)
            rec.failure = True
            rec.safe = False
            rec.reason = (f"制动失效: 停止距离 {rec.stop_distance_m*1000:.1f}mm, "
                          f"载荷位移 {rec.slip_m*1000:.1f}mm, "
                          f"下沉 {rec.drop_m*1000:.1f}mm")
            self.resources_frozen = True
        else:
            slip_ok = rec.slip_m <= LOAD_SLIP_TOL_M
            drop_ok = rec.drop_m <= LOAD_DROP_TOL_M
            rec.failure = not (rec.held and slip_ok and drop_ok)
            rec.safe = not rec.failure
            rec.reason = (f"制动保持 ({mode.value}), 载荷无滑移"
                          if rec.safe else
                          f"停止后未保持: v={self._speed():.4f} m/s, "
                          f"滑移={rec.slip_m*1000:.2f}mm, "
                          f"下沉={rec.drop_m*1000:.2f}mm")
            if rec.failure:
                self.resources_frozen = True
        self.last = rec
        return rec

    def thaw(self) -> None:
        """解除资源冻结并恢复驱动/制动 (试验复位, 不是轮内自动纠正)。"""
        self.resources_frozen = False
        self._actuators_on()
        self.cell.model.actuator_gear[:] = self._nominal_gear
        self.cell.model.dof_damping[:] = self._nominal_damping.copy()