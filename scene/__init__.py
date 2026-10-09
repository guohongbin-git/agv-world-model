"""W04 全场景 — 18 货位 + 四设备 + 站体的可运行 MuJoCo 世界。

场景构成（全部来自 configs/design.yaml，无硬编码坐标）:
* 桌面 1800×1200mm，原点在左下角（设计 3 节）
* 站体 I01/H01/O01 + 入/出料托台
* 货架 A/B：各 3 列 × 3 层 = 18 货位，支撑面 z=100/250/400mm
* 设备 M01/M02（机械臂）、R01（平板车）、F01（叉车）
* 初始物：空托盘在 I01，入料箱在入料托台（尚未绑定）

本模块只做**场景装配与重力补偿**，不实现任务逻辑：
任务的设备互斥、预约、库存事务在 services/ 下分别实现。
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from cad.build import build_all_models
from cad.mjcf import build_mjcf
from cad.params import derive_parameters, load_params


@dataclass(frozen=True)
class Slot:
    """一个货位（设计 25.x：两货架 × 三列 × 三层 = 18）。"""

    slot_id: str
    group: str          # "A" / "B"
    column: int
    level: int
    x_m: float
    y_m: float
    support_z_m: float

    def pallet_pose(self) -> np.ndarray:
        """托盘几何中心位���（托盘底面贴在支撑面上）。"""
        return np.array([self.x_m, self.y_m, self.support_z_m])


class WarehouseScene:
    """18 货位四设备场景。

    提供重力补偿伺服（设计 25.4「使用限转矩驱动与重力补偿」）与货位查询；
    设备运动、互斥与事务由上层服务驱动。
    """

    def __init__(self, timestep_s: float | None = None, hold_zero_pose: bool = True):
        p = load_params()
        d = derive_parameters(p)
        self.params = p
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(p, d, build_all_models(p, d)))
        if timestep_s is not None:
            self.model.opt.timestep = timestep_s
        self.data = mujoco.MjData(self.model)
        self.slots = self._build_slots(p)
        self._arm_cmd: dict[int, float] = {}
        self._index_actuators()
        mujoco.mj_forward(self.model, self.data)
        if hold_zero_pose:
            self.capture_hold_pose()

    # ------------------------------------------------------------------ 货位
    @staticmethod
    def _build_slots(p) -> list[Slot]:
        rk = p.rack
        cols = {"A": rk.column_centers_group_a_x_m, "B": rk.column_centers_group_b_x_m}
        out: list[Slot] = []
        for group, xs in cols.items():
            for ci, x in enumerate(xs):
                for li, z in enumerate(rk.support_z_m):
                    out.append(Slot(f"{group}-L{li + 1}-C{ci + 1}", group, ci + 1,
                                    li + 1, float(x), float(rk.slot_center_y_m), float(z)))
        return out

    def slot(self, slot_id: str) -> Slot:
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        raise KeyError(f"未知货位 {slot_id}")

    def slot_at(self, x: float, y: float, tol: float = 0.05) -> Slot | None:
        """按坐标反查货位（感知端由标签位置给出世界坐标时用）。"""
        for s in self.slots:
            if abs(s.x_m - x) <= tol and abs(s.y_m - y) <= tol:
                return s
        return None

    # ------------------------------------------------------------ 执行器索引
    def _index_actuators(self) -> None:
        self.act_joint: dict[str, int] = {}
        self.act_arm: dict[str, int] = {}
        self.act_grip: dict[str, int] = {}
        self.act_wheel: dict[str, int] = {}
        m = self.model
        name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
        for a in range(m.nu):
            nm = name(mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if int(m.actuator_trntype[a]) != 1:      # 非 joint 传动
                continue
            jid = int(m.actuator_trnid[a][0])
            jn = name(mujoco.mjtObj.mjOBJ_JOINT, jid)
            if "_spin" in jn or "_lift" in jn or "fork" in jn:
                self.act_wheel[nm] = a
            elif "finger" in jn:
                self.act_grip[nm] = a
            else:
                self.act_arm[nm] = a
                self.act_joint[jn] = a

    # ------------------------------------------------------------ 重力补偿
    def capture_hold_pose(self) -> None:
        """把当前位形记为保持目标，并求出对应的重力补偿力矩。

        设计 25.4 要求「限转矩驱动与重力补偿」；没有补偿时机械臂在
        ctrl=0 下会被重力折倒（实测 J3/J5 转速达 25 rad/s）。

        补偿量来自**零位**的 qfrc_bias（力矩随位形变化, 由 step 里的
        实时 bias 覆盖 —— 见 _refresh_gravity_comp）。
        """
        mujoco.mj_forward(self.model, self.data)
        self._hold_q = self.data.qpos.copy()
        self._gravity_comp = np.zeros(self.model.nu)
        m = self.model
        name = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        self._arm_actuators = [a for a in range(m.nu)
                               if int(m.actuator_trntype[a]) == 1
                               and ("M01" in name(a) or "M02" in name(a))]
        self._gravity_enabled = True
        self._refresh_gravity_comp()

    def _refresh_gravity_comp(self) -> None:
        """用当前 qfrc_bias 覆盖机械臂电机的重力补偿量（实时）。"""
        if not getattr(self, "_gravity_enabled", False):
            return
        m, d = self.model, self.data
        for a in self._arm_actuators:
            jid = int(m.actuator_trnid[a][0])
            dof = int(m.jnt_dofadr[jid])
            self._gravity_comp[a] = d.qfrc_bias[dof]
            self.data.ctrl[a] = self._gravity_comp[a] + self._arm_cmd.get(a, 0.0)

    def set_arm_target(self, actuator_name: str, value: float) -> None:
        """给机械臂关节设定补偿后的目标（value 是额外力矩增量）。"""
        a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if a < 0:
            raise KeyError(actuator_name)
        self._arm_cmd[a] = float(value)
        self.data.ctrl[a] = self._gravity_comp[a] + self._arm_cmd[a]

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._refresh_gravity_comp()
            mujoco.mj_step(self.model, self.data)

    def settle(self, seconds: float) -> None:
        n = int(seconds / self.model.opt.timestep)
        for _ in range(n):
            self.step(1)

    # ------------------------------------------------------------------ 观测
    def geom(self, name: str) -> np.ndarray:
        g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            raise KeyError(name)
        return self.data.geom_xpos[g].copy()

    def body(self, name: str) -> np.ndarray:
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise KeyError(name)
        return self.data.xpos[b].copy()

    def max_speed(self) -> float:
        return float(np.abs(self.data.qvel).max())

    def describe(self) -> str:
        rk = self.params.rack
        return (f"桌面 {self.params.table.length_m*1000:.0f}×{self.params.table.width_m*1000:.0f}mm, "
                f"货位 {len(self.slots)} 个 ({rk.groups}组×{rk.columns}列×{rk.levels}层), "
                f"设备 M01/M02/R01/F01, 站体 I01/H01/O01")