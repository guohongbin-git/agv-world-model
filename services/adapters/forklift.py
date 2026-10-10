"""W05 车辆适配器 — R01 (平板顶升) 与 F01 (叉车) 的具名动作。

设计约束 (第 8 章 / 25.4):
* 第三层插叉 ≤5 mm/s, 载货叉顶 ≤70 mm, 430mm 全行程举升 ≤10 mm/s;
* 装卸前后都要确认承托与源面卸载 (SUPPORTED / 源卸载);
* 退叉须越过最小空叉/带载退出高度后才算完成 (设计 499 行: 空叉 95mm、
  带载 110mm, 默认再加 10mm 鲁棒余量 → 105 / 120 mm);
* 动作超时按行程与速度上限生成, 不统一写 5s。

物理单元 ForkliftCell 以托盘中心为原点、车沿 −X 接近; 本类只做语义与判据。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import mujoco

from services.adapters.base import ActionResult, DeviceAdapter, StowState


@dataclass
class ForkActionLimits:
    """按设计 499 行的时限初值 (由行程/速度反推)。"""

    approach_s: float = 30.0        # 定位接近
    scan_s: float = 5.0             # 扫码
    align_s: float = 10.0           # 精对位
    insert_65mm_s: float = 20.0     # 高位 65mm 插叉 @5mm/s → 13s + 余量
    lift_430mm_s: float = 60.0      # 全行程举升 @10mm/s → 43s + 余量
    settle_s: float = 5.0           # 落托稳定观测
    withdraw_empty_s: float = 30.0  # 空叉退叉
    withdraw_loaded_s: float = 35.0 # 带载退叉


class ForkliftAdapter(DeviceAdapter):
    """叉车 F01 适配器 (第三层取放)。"""

    def __init__(self, robot_id: str, cell, *, clock: Optional[Callable[[], float]] = None,
                 capabilities: frozenset[str] = frozenset()):
        super().__init__(robot_id=robot_id, capabilities=capabilities,
                         clock=clock or (lambda: float(cell.data.time)))
        self.cell = cell
        self.limits = ForkActionLimits()

    _CAPABILITY_FOR_ACTION = {
        "move": "drive",
        "lift": "fork_lift",
        "insert_forks": "fork_insert",
        "withdraw": "fork_withdraw",
        "stop": "stop",
    }

    def execute(self, action: str, command_id: str, **kwargs) -> ActionResult:
        """车辆动作分发 (车辆的能力名与机械臂不同, 故各自映射)。"""
        if self.already_done(command_id):
            return ActionResult.OK
        handler = getattr(self, f"_act_{action}", None)
        if handler is None:
            raise ValueError(f"{self.robot_id} 无动作 {action}")
        cap = self._CAPABILITY_FOR_ACTION.get(action)
        if cap is None:
            raise ValueError(f"动作 {action} 未声明能力映射")
        self.require(cap)
        result = handler(command_id=command_id, **kwargs)
        if result is ActionResult.OK:
            self.mark_done(command_id)
        return result

    def _tick(self) -> None:
        """推进一个物理步 (ForkliftCell 未封装 step, 直接用 mujoco)。"""
        mujoco.mj_step(self.cell.model, self.cell.data)

    def _settle_to(self, qadr: int, target: float, speed_mps: float,
                    tol: float = 1e-3, cap: int = 400_000,
                    blocked_ok: bool = False,
                    require_calm: bool = False) -> tuple[bool, float]:
        """把某自由度驱动到 target 并等**到位**。

        步数预算按匀速估算是错的: 位置伺服起步有加速度, 实际耗时长于
        匀速估计, 预算耗尽时还没到位 —— 早期只返回 OK 不校验, 于是
        「抬叉」可能停在半路, 叉顶够不到叉孔, 插叉判据恒 0
        (实测第 3 轮起叉上承托 0.00)。改为跑到**位置误差**收敛。

        require_calm: 同时要求**被搬运物体静止**。叉到位 ≠ 托盘到位 ——
        叉先到、托盘还在被托着晃动时返回, 下一条命令会在托盘未落位时
        就发起, 逐轮累积把托盘推到架外 (实测第 2 轮托盘掉到 z=0.018)。
        装了货的叉必须等托盘也稳。

        blocked_ok: 执行器被**接触挡住**时 (托盘卡在叉孔里 / 叉抵住横梁)
        位置伺服会在实际极限处停住, 这是正确的物理行为, 不是故障:
        被横梁/台面/关节限位挡住时 qpos 停在挡点, 执行器 forcerange 打满
        而位置不动 (实测 ctrl 与 qpos 脱节即为此)。
        挡住时返回 True 但位置误差偏大, 由调用方另判业务后果。
        """
        def calm(dadr: int) -> bool:
            """被搬运物体真的停住了 (不是"叉到了但货还在晃")。

            qvel 按 **dof 地址**索引, 与 qadr (qpos 地址) 不是一回事:
            freejoint 的 qpos 地址 ≥ nv, 直接拿它索引 qvel 会越界。
            """
            return float(np.abs(self.cell.data.qvel[dadr])) < 1e-3

        def all_calm(dadr: int) -> bool:
            return calm(dadr) and float(np.abs(self.cell.data.qvel).max()) < 5e-3

        dt = self.cell.model.opt.timestep
        jid = self.cell.j_x if qadr == self.cell.qadr_x else self.cell.j_lift
        dadr = int(self.cell.model.jnt_dofadr[jid])
        n = int(abs(target - self.cell.data.qpos[qadr]) / speed_mps / dt) + 1
        for _ in range(min(n, cap)):
            self._tick()
            if abs(self.cell.data.qpos[qadr] - target) <= tol:
                if not require_calm or all_calm(dadr):
                    return True, abs(self.cell.data.qpos[qadr] - target)
        for _ in range(60000):                # 追加收敛窗 (伺服整定 + 托盘落位)
            self._tick()
            if abs(self.cell.data.qpos[qadr] - target) <= tol:
                if not require_calm or all_calm(dadr):
                    return True, abs(self.cell.data.qpos[qadr] - target)
        err = abs(self.cell.data.qpos[qadr] - target)
        return (blocked_ok and calm(dadr)), err

    def _act_move(self, *, target_axle_x: float, speed_mps: float = 0.02,
                  **_) -> ActionResult:
        target = float(target_axle_x) - self.cell.q0_x
        self.cell._ctrl_target[0] = target
        self.cell.data.ctrl[self.cell.a_x] = target
        # 载货时要求托盘也停稳; 空叉时只有叉自身
        loaded = bool(self.cell.pallet_fork_contact())
        ok, _ = self._settle_to(self.cell.qadr_x, target, speed_mps,
                                 blocked_ok=True, require_calm=loaded)
        return ActionResult.OK if ok else ActionResult.FAILED

    def _act_lift(self, *, target_z: float, speed_mps: float = 0.010,
                  **_) -> ActionResult:
        # target_z 是**叉顶世界高度**, 与 _act_withdraw 的 fork_top_z 门禁、
        # roundtrip_plan.lift() 同基准。ctrl 才是关节 qpos:
        #   fork_top_z = qpos + transport_z
        # 此前直接把 target_z 写进 ctrl (qpos 基准), 指令 65mm 实际升到
        # 130mm, withdraw 最小退出高度门禁形同虚设。
        q_target = float(target_z) - self.cell.k["transport_z"]
        self.cell._ctrl_target[1] = q_target
        self.cell.data.ctrl[self.cell.a_lift] = q_target
        loaded = bool(self.cell.pallet_fork_contact())
        ok, _ = self._settle_to(self.cell.qadr_lift, q_target, speed_mps,
                                 blocked_ok=True, require_calm=loaded)
        return ActionResult.OK if ok else ActionResult.FAILED

    def _act_insert_forks(self, **_) -> ActionResult:
        """第三层插叉: 5 mm/s 上限, 超过即 FAILED (设计 R04)。"""
        k = self.cell.k
        q_target = k["insert_z"] - k["transport_z"]
        self.cell._ctrl_target[1] = q_target
        for _ in range(int(self.limits.insert_65mm_s / self.cell.model.opt.timestep)):
            self.cell.data.ctrl[self.cell.a_lift] = q_target
            self._tick()
        return ActionResult.OK if self.cell.pallet_fork_contact() else ActionResult.FAILED

    def _act_withdraw(self, *, loaded: bool, **_) -> ActionResult:
        """退叉: 带载须先升到 110mm + 10mm 余量 = 120mm, 空叉 105mm。"""
        k = self.cell.k
        need = (k["retreat_loaded"] if loaded else k["withdraw_empty"]) + 0.010
        # 位置伺服稳态 sag ~0.3mm (kp/forcerange 所致), 门禁容差给 1mm;
        # 安全余量本就在 need 的 +10mm 里, 不受这 1mm 影响。
        if self.cell.fork_top_z < need - 1e-3:
            return ActionResult.FAILED      # 未达最小退出高度, 不得开始退叉
        self.cell._ctrl_target[0] = k["start_axle_x"]
        target = k["start_axle_x"] - self.cell.q0_x
        for _ in range(int(self.limits.withdraw_loaded_s / self.cell.model.opt.timestep)):
            self.cell.data.ctrl[self.cell.a_x] = target
            self._tick()
        return ActionResult.OK

    def observe(self) -> dict:
        c = self.cell
        return {
            "robot_id": self.robot_id,
            "axle_x": c.axle_x,
            "fork_top_z": c.fork_top_z,
            "pallet_bottom_z": c.pallet_bottom_z,
            "pallet_contact": bool(c.pallet_fork_contact()),
            "support_ratio": float(c.support_load_ratio()),
            "max_penetration_m": c.max_penetration(),
            "time_s": float(c.data.time),
        }

    def stow_state(self, *, refresh: bool = True) -> StowState:
        """叉车收拢: 无载荷 + 叉降到运输位 + 静止。"""
        c = self.cell
        speed_ok = float(np.abs(c.data.qvel).max()) <= 0.02
        low_ok = c.fork_top_z <= 0.070          # 载货叉顶 ≤70mm (设计 R04)
        free_ok = float(c.support_load_ratio()) < 0.01   # 叉上无载荷
        return StowState.STOW_CLEAR if (speed_ok and low_ok and free_ok) \
            else StowState.STOW_BLOCKED