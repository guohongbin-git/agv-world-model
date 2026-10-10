"""W03 低层控制器 — 100Hz 位置闭环 + 梯形速度限幅。

物理 1ms, 控制 10ms。速度限幅由目标轨迹生成器保证; 不写 qpos。
"""
from __future__ import annotations

import mujoco
import numpy as np

from sim.forklift_cell import ForkliftCell


class Ramp:
    """梯形速度剖面位置目标 (速度上限 + 加速度上限)。

    at(t) 精确积分: 加速段 (s=½at²) → 匀速段 (s=v·Δt) → 减速段。
    单调、无速度尖峰; 到位后精确停在 target。
    """

    def __init__(self, start: float, target: float, speed_mps: float,
                 accel_mps2: float = 0.30):
        self.start = start
        self.target = target
        self.speed = speed_mps
        self.a = accel_mps2
        self.d = abs(target - start)
        self.sgn = 1.0 if target >= start else -1.0
        # 梯形剖面: 能否达到满速?
        ta = speed_mps / accel_mps2
        d_tri = speed_mps * ta          # 三角形总位移 (0→v→0)
        if self.d >= d_tri:
            self.t1 = ta
            self.t3 = ta
            self.d1 = 0.5 * accel_mps2 * ta ** 2
            self.d3 = self.d1
            self.d2 = self.d - 2 * self.d1
            self.t2 = self.d2 / speed_mps       # 匀速段时长
        else:
            # 三角形剖面: 峰值速度降为 v_peak = sqrt(a·d)
            v_peak = np.sqrt(accel_mps2 * self.d)
            self.t1 = self.t3 = v_peak / accel_mps2
            self.t2 = 0.0
            self.d1 = self.d3 = 0.5 * v_peak * (v_peak / accel_mps2)
            self.d2 = 0.0
            self.speed = v_peak          # 中段速度须同步降为峰值
        self.done_at = self.t1 + self.t2 + self.t3

    def at(self, t_s: float) -> float:
        t = max(0.0, t_s)
        if t <= self.t1:
            s = 0.5 * self.a * t ** 2
        elif t <= self.t1 + self.t2:
            s = self.d1 + self.speed * (t - self.t1)
        else:
            td = min(t - self.t1 - self.t2, self.t3)
            s = self.d1 + self.d2 + self.speed * td - 0.5 * self.a * td ** 2
        return self.start + self.sgn * min(s, self.d)

    @property
    def duration(self) -> float:
        return self.done_at


class CellController:
    """驱动 [chassis_x, lift] 两个位置执行器, 按 Ramp 轨迹跟踪。"""

    CTRL_HZ = 100

    def __init__(self, cell: ForkliftCell):
        self.cell = cell
        self.substeps = max(1, round(0.01 / cell.model.opt.timestep))

    def run_sequence(self, ramps: list[tuple[int, "Ramp | Callable[[], Ramp]"]],
                     settle_s: float = 0.5,
                     max_pen_m: float = 5e-4) -> dict:
        """顺序执行各段 (actuator_idx, ramp); 每段结束再 settle。返回审计数据。

        第二项可以是 `Ramp`, 也可以是**零参可调用对象**返回 `Ramp`。
        后者用于「起点须在该段开始时才取」的轨迹: 一次性构造的 Ramp 会让
        后续各段共用同一个起点值, 轨迹全错 (T04 往返 12 段即此情形)。
        """
        m, d = self.cell.model, self.cell.data
        audit: dict = {"segments": [], "max_pen": 0.0, "violations": []}
        for act_idx, ramp_spec in ramps:
            ramp = ramp_spec() if callable(ramp_spec) else ramp_spec
            n_ctrl = int(ramp.duration * self.CTRL_HZ) + 1
            q0 = self.cell.q0_x if act_idx == self.cell.a_x else 0.0
            for _ in range(n_ctrl):
                target = ramp.at(d.time) - q0   # 世界坐标 → 关节q
                d.ctrl[act_idx] = target
                for _ in range(self.substeps):
                    mujoco.mj_step(m, d)
                    pen = self.cell.max_penetration()
                    audit["max_pen"] = max(audit["max_pen"], pen)
                    if pen < -max_pen_m:
                        audit["violations"].append(
                            f"穿透 {pen*1000:.2f}mm @ t={d.time:.3f}s")
                        return audit
            d.ctrl[act_idx] = ramp.target - (self.cell.q0_x if act_idx == self.cell.a_x else 0.0)
            n_settle = int(settle_s * self.CTRL_HZ)
            for _ in range(n_settle):
                for _ in range(self.substeps):
                    mujoco.mj_step(m, d)
                pen = self.cell.max_penetration()
                audit["max_pen"] = max(audit["max_pen"], pen)
            audit["segments"].append({
                "t_end": d.time, "axle_x": self.cell.axle_x,
                "fork_top_z": self.cell.fork_top_z,
                "pallet_bottom_z": self.cell.pallet_bottom_z,
            })
        return audit
