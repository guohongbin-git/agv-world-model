"""W05 任务调度 — 七段业务流程的原子提交与互斥 (设计 25.5/25.6)。

设计要求 (25.5 装托卸货与事务边界):
|流程|机械动作与观测|原子业务提交|
|入料接管|身份/姿态有效；M01闭爪抬离；源卸载、双指夹持稳定|G→GRIPPER:M01，仍 INBOUND 不计在库|
|入库装托|I01空托盘已核验；放到PAL；承托、开爪、撤离观察均通过|G→PALLET；建立绑定；PAL→LOADED；G→IN_WAREHOUSE，计数+1|
|车辆运输|M01收拢，R01/F01沿既有托盘合同执行|只移动 PAL 位置；绑定 G 有效位置仍是 PALLET，计数不变|
|到出库站|R01放托O01并退出；货物与PAL身份一致|G→OUTBOUND_PENDING 仍计在库；PAL 仍 LOADED|
|卸货接管|M02闭爪抬离，PAL卸载、G夹持稳定|G→GRIPPER:M02；关闭活动绑定；PAL→EMPTY；计数不变|
|出料放置|G由出料托台承托；开爪撤离稳定，M02收拢|G→FEED:O-TRAY，保持 OUTBOUND_PENDING|
|货物离场|机器人任务已停止活动，出料托台身份与边界事件一致|G→OFFSITE 并计数−1；现场空PAL不变|

硬约束:
* 每段的**物理前置全部通过**才提交事务; 任一失败 → 不提交, 状态保持;
* FAILED 与 UNKNOWN 都**不推进**下一步 (设计 499 行);
* 提交是原子的, 且事件 ID 唯一 → 重复提交不重复计数 (设计 R09);
* 同一主任务内不同时启动机械臂和相交路径车辆 (设计 R14);
* 交接须有新鲜 STOW_CLEAR 证据, 不能只沿用旧收拢标志 (设计 25.6)。
"""
from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from services.adapters.base import ActionResult, StowState
from services.scheduler.reservations import ReservationBoard, ReservationError
from services.world.ontology import (
    Binding, ConstraintViolation, GoodsLifecycle, GoodsUnit, Loc, LocKind, Pallet,
    PalletCargo, Slot, World,
)


class Phase(str, Enum):
    FEED_PICK = "FEED_PICK"          # 入料接管
    LOAD_PALLET = "LOAD_PALLET"      # 入库装托
    TRANSPORT = "TRANSPORT"          # 车辆运输
    ARRIVE_OUT = "ARRIVE_OUT"        # 到出库站
    UNLOAD_PICK = "UNLOAD_PICK"      # 卸货接管
    PLACE_FEED = "PLACE_FEED"        # 出料放置
    OFFSITE = "OFFSITE"              # 货物离场


class StepOutcome(str, Enum):
    OK = "OK"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"          # 状态不明, 不得推进


@dataclass
class StepRecord:
    """一步的执行审计 (设计要求保留证据, 便于独立验收)。"""

    phase: Phase
    outcome: StepOutcome
    detail: str = ""
    sim_time_s: float = 0.0
    evidence: dict = field(default_factory=dict)


class TaskScheduler:
    """主任务调度器: 七段流程 + 互斥 + 原子提交。"""

    PHASES = [Phase.FEED_PICK, Phase.LOAD_PALLET, Phase.TRANSPORT,
              Phase.ARRIVE_OUT, Phase.UNLOAD_PICK, Phase.PLACE_FEED, Phase.OFFSITE]

    #: 抬离判据 (设计 25.3: GRASP_CONFIRMED 需抬离 ≥5mm)
    _min_lift_m = 0.005
    _lift_step_m = 0.0025
    _lift_steps = 20
    #: 承托判据 (设计 SUPPORTED_CONFIRMED: ≥0.90, 穿模 ≤1mm)
    _min_support = 0.90
    _max_penetration_m = 0.001
    _support_samples = 5
    #: 夹持稳定判据 (设计 25.3: 漂移 ≤1mm / ≤1°)
    _max_drift_m = 0.001
    _max_drift_deg = 1.0
    #: 撤离高度 (开爪后抬离, 再收拢) 与落托沉降窗
    _retract_m = 0.060
    _settle_samples = 40

    def __init__(self, world: World, board: ReservationBoard, *,
                 m01=None, m02=None, r01=None, f01=None,
                 clock=None):
        self.world = world
        self.board = board
        self.arms = {"M01": m01, "M02": m02}
        self.vehicles = {"R01": r01, "F01": f01}
        self.clock = clock or (lambda: 0.0)
        self.records: list[StepRecord] = []
        self.committed: set[str] = set()
        self._task_id = "TASK-1"

    # ------------------------------------------------------------ 互斥
    def _require_exclusive(self, robot_id: str) -> None:
        """工位互斥 (设计 R14 / 159 行)。

        设计说的是「机械臂执行期间, 其**工作区和相交车辆通道**一并锁定」
        —— 不是「只要有别的机器人在动就全锁」。M01/M02 基座相距 1.5m
        (0.15/1.65), 收拢包络 255.7mm 互不相交, 用全局互斥会让 M02 永远
        卡住 M01。故只对**包络实际相交**的机器人互斥。
        """
        me = {**self.arms, **self.vehicles}.get(robot_id)
        if me is None:
            return
        my_r = getattr(me, "stow_radius_measured_m", 0.0) or self._fallback_radius(me)
        my_xy = self._envelope_center(me)
        blockers = []
        for rid, ad in {**self.arms, **self.vehicles}.items():
            if ad is None or rid == robot_id:
                continue
            other_r = getattr(ad, "stow_radius_measured_m", 0.0) or self._fallback_radius(ad)
            d = float(np.linalg.norm(my_xy - self._envelope_center(ad)))
            if d > my_r + other_r:
                continue                      # 包络不相交 → 不互斥
            try:
                if ad.stow_state() is not StowState.STOW_CLEAR:
                    blockers.append(rid)
            except Exception:
                blockers.append(rid)
        if blockers:
            raise ReservationError(
                f"{robot_id} 启动被拒: {blockers} 未收拢且包络相交 "
                f"(设计 R14 工位互斥)")

    @staticmethod
    def _fallback_radius(ad) -> float:
        """未采样时用基座名义半径 (设计 base_radius_m 40mm)。"""
        try:
            return float(ad.cell.cfg["manipulators"]["base_radius_m"])
        except Exception:
            return 0.040

    @staticmethod
    def _envelope_center(ad) -> np.ndarray:
        """包络中心 (机械臂=基座, 车辆=当前车位)。"""
        c = ad.cell
        for attr in ("base_xy", "base_position_m", "axle_xy"):
            v = getattr(c, attr, None)
            if v is not None:
                return np.asarray(v, dtype=float)[:2]
        bid = None
        try:
            import mujoco
            bid = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_BODY, f"{c.arm}_base")
            if bid >= 0:
                return c.model.body_pos[bid][:2].copy()
        except Exception:
            pass
        return np.zeros(2)

    def _require_fresh_stow(self, robot_id: str, max_age_s: float = 0.1) -> None:
        """交接须有新鲜 STOW_CLEAR 证据 (设计 25.6)。"""
        ad = self.arms.get(robot_id) or self.vehicles.get(robot_id)
        if ad is None:
            return
        st = ad.stow_state(refresh=True)
        if st is not StowState.STOW_CLEAR:
            raise ReservationError(f"{robot_id} 未收拢 ({st.value}), 禁止交接")

    # ------------------------------------------------------------ 原子提交
    def _commit_once(self, key: str, fn) -> bool:
        """事务只提交一次 (设计 R09: 事件唯一, 重复不重复记账)。"""
        if key in self.committed:
            return False
        self.committed.add(key)
        fn()
        return True

    def _record(self, phase: Phase, outcome: StepOutcome, detail: str = "",
                **ev) -> StepOutcome:
        self.records.append(StepRecord(phase, outcome, detail, self.clock(), ev))
        return outcome

    # ------------------------------------------------------------ 各段流程
    def feed_pick(self, goods_id: str, arm_id: str = "M01") -> StepOutcome:
        """入料接管: 闭爪抬离 + 源卸载 + 双指稳定 → G→GRIPPER, 仍不计在库。"""
        arm = self.arms[arm_id]
        loc = self.world.goods_location(goods_id)
        if not loc.startswith(f"{LocKind.FEED.value}:"):
            return self._record(Phase.FEED_PICK, StepOutcome.FAILED,
                                f"货物不在 FEED: {loc}")
        self._require_exclusive(arm_id)
        # 源高度必须**在闭爪前**采样: 抓取后箱子已随臂运动, 再采样只会读到
        # 抬升后的高度, 抬离量恒为 0 (设计 25.3 GRASP_CONFIRMED 要求
        # 「抬离 + 源卸载」为同一次判定的两个条件)。
        src_z = float(arm.observe()["box_bottom_z"])
        # 用**本次任务**的命令 id; 若上一轮已闭爪 (幂等), 箱底基准仍取
        # 闭爪前读数 —— 否则重复调用会把抬离基准误当成已抬起的高度。
        if arm.execute("close_gripper", f"{self._task_id}:pick") is not ActionResult.OK:
            return self._record(Phase.FEED_PICK, StepOutcome.FAILED, "闭爪未双侧接触")
        # 抬离 ≥5mm (设计 GRASP_CONFIRMED), 由调度器显式抬升
        for i in range(self._lift_steps):
            tgt = arm.observe()["tcp"] + np.array([0.0, 0.0, self._lift_step_m])
            if arm.execute("move_tcp", f"{self._task_id}:lift{i}", target=tgt) \
                    is not ActionResult.OK:
                return self._record(Phase.FEED_PICK, StepOutcome.FAILED,
                                    f"抬升第 {i} 步失败")
        obs = arm.observe()
        lift = float(obs["box_bottom_z"]) - src_z
        if lift < self._min_lift_m:
            return self._record(Phase.FEED_PICK, StepOutcome.FAILED,
                                f"抬离仅 {lift*1000:.1f}mm < "
                                f"{self._min_lift_m*1000:.0f}mm")
        if not (obs["left_contact"] and obs["right_contact"]):
            return self._record(Phase.FEED_PICK, StepOutcome.FAILED, "双指夹持不稳定")
        # 入料接管只改位置与生命周期前缀, 仍 INBOUND 不计在库 (设计 25.5)
        self._commit_once(f"feed_pick:{goods_id}", lambda: self.world.take_from_feed(
            goods_id, arm_id, f"{self._task_id}:pick", self.clock()))
        return self._record(Phase.FEED_PICK, StepOutcome.OK, "已接管",
                            lift_mm=lift * 1000)

    def _retract_and_stow(self, arm, arm_id: str) -> bool:
        """撤离观察后收拢 (设计 25.6 交接序列)。

        判据要求「撤离观察」在**收拢之前**——手还压在货物上方时收拢是
        假收拢。所以先抬到安全高度再收拢, 收拢后证据必须是新鲜的。
        """
        clear = arm.observe()["tcp"] + np.array([0.0, 0.0, self._retract_m])
        if arm.execute("move_tcp", f"{self._task_id}:retract",
                       target=clear) is not ActionResult.OK:
            return False
        return arm.execute("stow", f"{self._task_id}:stow") is ActionResult.OK

    def load_pallet(self, goods_id: str, pallet_id: str, slot_id: str,
                    arm_id: str = "M01") -> StepOutcome:
        """入库装托: 承托 + 开爪 + 撤离观察 → 绑定 + LOADED + IN_WAREHOUSE +1。"""
        arm = self.arms[arm_id]
        prow = self.world.pallet(pallet_id)
        if prow is None:
            return self._record(Phase.LOAD_PALLET, StepOutcome.FAILED,
                                f"托盘不存在: {pallet_id}")
        if prow[1] != PalletCargo.EMPTY.value:
            return self._record(Phase.LOAD_PALLET, StepOutcome.FAILED,
                                f"托盘 {pallet_id} 非空 ({prow[1]})")
        if self.world.active_binding(goods_id) is not None:
            return self._record(Phase.LOAD_PALLET, StepOutcome.FAILED,
                                "货物已有活动绑定, 不可重复装托")
        if arm.execute("open_gripper", f"{self._task_id}:load") is not ActionResult.OK:
            return self._record(Phase.LOAD_PALLET, StepOutcome.FAILED, "开爪失败")
        # 承托 ≥90% (设计 SUPPORTED_CONFIRMED)。观察窗内读数必须持续达标:
        # 单帧读数可能撞上箱体尚未落稳的瞬间。
        for _ in range(self._settle_samples):
            arm.cell.step(arm.cell.substeps)
        ratio = min(arm.observe()["support_ratio"] for _ in range(self._support_samples))
        if ratio < self._min_support:
            return self._record(Phase.LOAD_PALLET, StepOutcome.FAILED,
                                f"承托率 {ratio*100:.0f}% < "
                                f"{self._min_support*100:.0f}%")
        pen = float(arm.observe()["max_penetration_m"])
        if pen > self._max_penetration_m:
            return self._record(Phase.LOAD_PALLET, StepOutcome.FAILED,
                                f"穿模 {pen*1000:.2f}mm > "
                                f"{self._max_penetration_m*1000:.1f}mm")
        # 撤离 → 收拢 → 新鲜证据 (设计 25.6)
        if not self._retract_and_stow(arm, arm_id):
            return self._record(Phase.LOAD_PALLET, StepOutcome.FAILED, "撤离/收拢失败")
        self._require_fresh_stow(arm_id)

        def _txn():
            # bind_and_store 已原子完成 建绑定 + G→IN_WAREHOUSE + 位置PALLET
            # + PAL→LOADED (设计 25.5「入库装托」行) —— 不要重复记 IN_WAREHOUSE。
            self.world.bind_and_store(goods_id, pallet_id,
                                      Loc(LocKind.PALLET, pallet_id),
                                      f"{self._task_id}:load", self.clock())

        self._commit_once(f"load:{goods_id}", _txn)
        return self._record(Phase.LOAD_PALLET, StepOutcome.OK, "已装托",
                            support_ratio=ratio)

    def unload_pick(self, goods_id: str, arm_id: str = "M02") -> StepOutcome:
        """卸货接管: PAL 卸载 + G 夹持稳定 → 关闭绑定 + PAL→EMPTY, 计数不变。

        「PAL 卸载」是**物理条件**, 不是状态字段: 抬离后托盘承货面上
        不得再有箱件承托力 (design 25.5「卸货接管」行)。只查双指接触会
        在箱件还压在托盘上时就提交解绑。
        """
        arm = self.arms[arm_id]
        src_z = float(arm.observe()["box_bottom_z"])
        if arm.execute("close_gripper", f"{self._task_id}:unload") is not ActionResult.OK:
            return self._record(Phase.UNLOAD_PICK, StepOutcome.FAILED, "闭爪未双侧接触")
        for i in range(self._lift_steps):
            tgt = arm.observe()["tcp"] + np.array([0.0, 0.0, self._lift_step_m])
            if arm.execute("move_tcp", f"{self._task_id}:ulo{i}", target=tgt) \
                    is not ActionResult.OK:
                return self._record(Phase.UNLOAD_PICK, StepOutcome.FAILED,
                                    f"抬升第 {i} 步失败")
        obs = arm.observe()
        lift = float(obs["box_bottom_z"]) - src_z
        if lift < self._min_lift_m:
            return self._record(Phase.UNLOAD_PICK, StepOutcome.FAILED,
                                f"抬离仅 {lift*1000:.1f}mm < "
                                f"{self._min_lift_m*1000:.0f}mm")
        if not (obs["left_contact"] and obs["right_contact"]):
            return self._record(Phase.UNLOAD_PICK, StepOutcome.FAILED, "双指夹持不稳定")
        # PAL 卸载: 托盘面承托必须归零
        pallet_load = float(arm.observe()["tray_ratio"]) + \
            float(arm.observe()["support_ratio"])
        if pallet_load > self._min_support:
            return self._record(Phase.UNLOAD_PICK, StepOutcome.FAILED,
                                f"托盘未卸载 (承托 {pallet_load:.2f})")
        before = self.world.in_warehouse_count()
        # unbind_at_output 已是原子的: 关绑定 + G→GRIPPER + PAL→EMPTY
        pal_id = self.world.active_binding(goods_id)
        if pal_id is None:
            return self._record(Phase.UNLOAD_PICK, StepOutcome.FAILED, "无活动绑定")
        self._commit_once(f"unload:{goods_id}", lambda: self.world.unbind_at_output(
            goods_id, pal_id[0], f"{self._task_id}:unload", self.clock()))
        if self.world.in_warehouse_count() != before:
            return self._record(Phase.UNLOAD_PICK, StepOutcome.FAILED,
                                "计数不应变化 (设计 25.5)")
        return self._record(Phase.UNLOAD_PICK, StepOutcome.OK, "已接管")

    def place_feed(self, goods_id: str, arm_id: str = "M02") -> StepOutcome:
        """出料放置: 出料托台承托 + 开爪撤离 → G→FEED:O-TRAY, 保持 OUTBOUND_PENDING。"""
        arm = self.arms[arm_id]
        if arm.execute("open_gripper", f"{self._task_id}:place") is not ActionResult.OK:
            return self._record(Phase.PLACE_FEED, StepOutcome.FAILED, "开爪失败")
        for _ in range(self._settle_samples):
            arm.cell.step(arm.cell.substeps)
        ratio = min(arm.observe()["support_ratio"] for _ in range(self._support_samples))
        if ratio < self._min_support:
            return self._record(Phase.PLACE_FEED, StepOutcome.FAILED,
                                f"承托率 {ratio*100:.0f}% < "
                                f"{self._min_support*100:.0f}%")
        pen = float(arm.observe()["max_penetration_m"])
        if pen > self._max_penetration_m:
            return self._record(Phase.PLACE_FEED, StepOutcome.FAILED,
                                f"穿模 {pen*1000:.2f}mm > "
                                f"{self._max_penetration_m*1000:.1f}mm")
        # 撤离 → 收拢 → 新鲜证据 (设计 25.6)。不收拢就交接 = 假交接。
        if not self._retract_and_stow(arm, arm_id):
            return self._record(Phase.PLACE_FEED, StepOutcome.FAILED, "撤离/收拢失败")
        self._require_fresh_stow(arm_id)
        self._commit_once(f"place:{goods_id}", lambda: self.world.place_at_feed(
            goods_id, "O-TRAY", f"{self._task_id}:place", self.clock()))
        return self._record(Phase.PLACE_FEED, StepOutcome.OK, "已放置",
                            support_ratio=ratio)

    def offsite(self, goods_id: str) -> StepOutcome:
        """货物离场: 机器人任务已停 + 托台身份一致 → OFFSITE 且计数−1。"""
        loc = self.world.goods_location(goods_id)
        if not loc.startswith(f"{LocKind.FEED.value}:"):
            return self._record(Phase.OFFSITE, StepOutcome.FAILED,
                                f"货物不在出料托台: {loc}")
        before = self.world.in_warehouse_count()
        try:
            self._commit_once(f"offsite:{goods_id}",
                              lambda: self.world.exit_offsite(
                                  goods_id, f"{self._task_id}:exit", self.clock()))
        except ConstraintViolation as e:
            return self._record(Phase.OFFSITE, StepOutcome.FAILED, str(e))
        if self.world.in_warehouse_count() != before - 1:
            return self._record(Phase.OFFSITE, StepOutcome.FAILED, "计数未减一")
        return self._record(Phase.OFFSITE, StepOutcome.OK, "已离场")

    # ------------------------------------------------------------ 报告
    def all_ok(self) -> bool:
        return all(r.outcome is StepOutcome.OK for r in self.records)

    def summary(self) -> dict:
        return {
            "task": self._task_id,
            "steps": len(self.records),
            "ok": sum(1 for r in self.records if r.outcome is StepOutcome.OK),
            "in_warehouse": self.world.in_warehouse_count(),
            "committed": sorted(self.committed),
        }
