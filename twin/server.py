"""T04 世界模型实时服务 — agent 唯一入口, 物理/事务/事件三路流式推送.

    PYTHONPATH=. ./.venv/bin/python twin/server.py     # http://127.0.0.1:8300

架构 (docs/agentic_twin_design.md §1/§4, 对齐设计文档 §8.1 数据流):

    用户 ⇄ /api/chat ⇄ WarehouseAgent (twin/agent.py, 唯一指令入口)
                          │ task.start / pause / speed / query…
                          ▼
    ForkliftAdapter 判据层 (≤5mm/s 插叉、退叉门禁、承托观测)
                          │ 物理步进全部汇于 _tick → PacedForkliftAdapter 配速+采样
                          ▼
    ForkliftCell (MuJoCo 2kHz)          World (SQLite 事务 + Event, 唯一业务写入)
        └── 60Hz 位姿快照 → /ws ──→ three.js 投影 + 库存板 + 事件流

演示任务 (判据链执行, 库存只在物理证据成立后提交):
    单车闭环: A-L3-C1 下架 → 暂存站 H01 → 放回 A-L3-C1 (旧式 4 元组队列);
    双车交接 (P3 Wave2): F01 从货架取 PAL-001 放交接台 S1 (STATION),
    随后前移让位清出巷道; F02 等巷道空闲后接力, S1 取托送出库位 H01
    (SLOT) 并标记 OutboundPending。交接 = 单一托盘所有权转移:
    ROBOT:F01 → STATION:S1 → ROBOT:F02 → SLOT:H01。
    巷道作业区 (x > -1.0) 由 AisleLock 互斥 —— 两车同处 y=0 单巷道,
    并发作业几何上必然碰撞。
"""
from __future__ import annotations

import asyncio
import itertools
import json
import numpy as np
import threading
import time
import uuid
from collections import deque
from pathlib import Path
import mujoco
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from sim.forklift_cell import ForkliftCell
from services.adapters.base import ActionResult
from services.adapters.forklift import ForkliftAdapter
from services.world.evidence import (FRAME_INTERVAL_S, EvidenceFrame,
                                     EvidenceVerdict, build_verdict,
                                     collect_frames)
from services.world.ontology import (GoodsLifecycle, GoodsUnit, Loc, LocKind,
                                     Pallet, PalletCargo, Slot, SlotOcc,
                                     TransferOutcome, World, WorldError)

WEB = Path(__file__).parent / "web"
RUNS = Path(__file__).parent / "runs"
PUB_HZ = 60
FORK_CAPS = frozenset({"drive", "fork_lift", "fork_insert",
                       "fork_withdraw", "stop"})
# run 身份 (GPT 评审 P1-7): 不再是模块加载时一次性常量 —— 每次
# build()/reset() 重新生成全局唯一 uuid, 见 WorldService._new_identity。

# 巷道作业区: 货架 (x≈0) 与暂存台/交接台 (x=-0.340) 所在区段。两车同在
# y=0 单巷道 (底盘只有 x 向自由度), 同一时刻只允许一车在此区作业。
AISLE_X_MIN = -1.0


class AisleLock:
    """巷道互斥锁 (P3 Wave2): 同一时刻只允许一辆车在作业区 (x > AISLE_X_MIN)
    作业。

    两车共用一条 y=0 巷道且底盘只有 x 向自由度 —— 几何上不存在让行路径,
    并发作业必然碰撞; 互斥是安全性约束而非优化。持有者是具体车辆 id,
    任务相同也不共用 (对齐 ReservationBoard 语义); 非持有者释放是 no-op,
    防止异常路径误释放。
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.owner: str | None = None

    def acquire(self, vid: str, block: bool = True) -> bool:
        with self._cond:
            if self.owner == vid:            # 重入 (同一任务重试路径)
                return True
            if not block:
                if self.owner is not None:
                    return False
            else:
                while self.owner is not None:
                    self._cond.wait()
            self.owner = vid
            return True

    def release(self, vid: str) -> None:
        with self._cond:
            if self.owner != vid:
                return
            self.owner = None
            self._cond.notify_all()


class Gate:
    """配速/暂停/采样共享状态 (mission 线程写, ws 线程读)。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.paused = False
        self.speed = 1.0
        self.t0 = time.monotonic()
        self.sim0 = 0.0
        self.snap: dict = {"t": 0.0, "next_t": 0.0}


class PacedForkliftAdapter(ForkliftAdapter):
    """物理步进唯一入口 hook: 暂停门 + 墙钟配速 (速度倍率) + 60Hz 位姿采样。

    P3 Wave2: 每辆车一个实例 (robot_id = 车辆 id), 绑同一 cell 的
    :class:`~sim.forklift_cell.VehicleView` (``self.view``)。物理步进本身
    仍是整车一级的 (``_tick`` 推进共享 ``cell.data``), per-vehicle 的只是
    判据/计划层用的视图与位姿采样键。
    """

    def __init__(self, robot_id, cell, gate: Gate, **kw):
        super().__init__(robot_id=robot_id, cell=cell, **kw)
        self.gate = gate
        self._n = 0
        self.view = cell.vehicle(robot_id)
        # 位姿采样键缓存: <vid>_chassis / <vid>_carriage + 共享 pallet
        m = cell.model
        self._body_ids = {
            **{f"{vid}_{b}": mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY, f"{vid}_{b}")
               for vid in cell.vehicles for b in ("chassis", "carriage")},
            "pallet": cell.pallet_bid,
        }

    def _tick(self) -> None:
        mujoco.mj_step(self.cell.model, self.cell.data)
        self._n += 1
        if self._n % 100 != 0:         # 每 0.05s 仿真检查一次暂停/采样
            return
        g = self.gate
        while True:
            with g.lock:
                paused = g.paused
            if not paused:
                break
            time.sleep(0.05)
        d = self.cell.data
        if d.time >= g.snap.get("next_t", 0.0):
            mujoco.mj_forward(self.cell.model, d)
            snap = {"t": round(d.time, 3), "next_t": d.time + 1.0 / PUB_HZ}
            for name, bid in self._body_ids.items():
                snap[name] = [round(v, 5) for v in d.xpos[bid]] + \
                             [round(v, 5) for v in d.xquat[bid]]
            with g.lock:
                snap["next_t"] = max(snap["next_t"], g.snap.get("next_t", 0))
                g.snap = snap
                speed = g.speed
            # 墙钟配速: sim 时间以 speed 倍推进
            target = g.t0 + (d.time - g.sim0) / max(0.25, speed)
            lead = target - time.monotonic()
            if lead > 0:
                time.sleep(min(lead, 0.05))


class ReconciliationRequired(RuntimeError):
    """物理已成功而账本提交失败 —— 任务挂起, 等待核对后恢复。

    设计 §4 失败矩阵「提交失败但物理已交接」: 保留事实进"待核对",
    禁止依赖该状态的后续动作 (任务线程立即返回); 恢复走同幂等键
    重放 (WorldService.reconcile), 不重复记账。
    """


class WorldService:
    """世界模型 + 物理 + 任务编排; 业务写入只经 World 事务。"""

    def __init__(self) -> None:
        self.gate = Gate()
        self.aisle = AisleLock()
        self.mode = "idle"  # idle|running|paused|blocked|done|failed|reconciliation_required
        self.step = -1
        self.step_label = "待命"
        self.chat: deque = deque(maxlen=200)
        self.chat_seq = 0
        self.db_lock = threading.RLock()   # 串行化 World 跨线程访问
        self._board_cache: tuple | None = None
        self.reconciliation: dict[str, dict] = {}   # 挂起提交 (待核对)
        self.run_id = ""
        self.task_id = ""
        self._verdict_seq = itertools.count(1)
        self.tasks: list | None = None
        self.steps_total = 0
        self.build()

    # ---------- 构建/复位 ----------
    def _new_identity(self) -> None:
        """run 身份重建 (GPT 评审 P1-7): 每次构造/reset 生成全局唯一身份。

        旧实现 RUN_ID 是模块加载时常量, reset() 重建了数据库文件但身份
        不变 —— 「新 run」不是全局唯一的新逻辑运行。现在 DEMO-<时间>-
        <uuid8> 保证同秒多次 reset 也互不相同的身份; 事件流
        (event_id 前缀)、叙述与 SQLite 文件名全部引用当前身份。
        """
        self.run_id = (f"DEMO-{time.strftime('%Y%m%d-%H%M%S')}-"
                       f"{uuid.uuid4().hex[:8]}")
        self.task_id = f"{self.run_id}-T01"

    def build(self) -> None:
        cell = ForkliftCell(timestep_s=0.0005, levels=[0.100, 0.250, 0.400],
                            vehicles=["F01", "F02"])
        d = cell.data
        for vid in cell.vehicles:      # 每车回安全车位 (F02 初始已在自身车位)
            v = cell.vehicle(vid)
            d.ctrl[v.a_x] = v.start_axle_x - v.q0_x
            d.ctrl[v.a_lift] = 0.0
        for _ in range(1000):
            mujoco.mj_step(cell.model, d)
        self.cell = cell
        # 双车各一个配速适配器: robot_id 即车辆 id, view 绑单车视图。
        # 物理步进是整车一级的 (共享 cell.data), 但判定/采样按车分流。
        self.adapters = {
            vid: PacedForkliftAdapter(vid, cell, self.gate,
                                      capabilities=FORK_CAPS)
            for vid in cell.vehicles}
        RUNS.mkdir(exist_ok=True)
        from twin.memory import DecisionMemory
        from twin.observation import ObservationChannel
        # 观测隔离 (P0-3): 决策侧 (避障/恢复/记忆) 只经 ObservationChannel
        # 拿带噪估计; mocap 真值只留在环境操作 (place/remove_cones)。
        self.obs = ObservationChannel(cell)
        self.memory = DecisionMemory(RUNS / "decision_memory.jsonl")
        self._blocked_sig: str | None = None
        self._blocked_t0: float | None = None
        self._blocked_retries = 0
        self.blocked_reason: str | None = None
        self._blocked_dir: float = 1.0    # 受阻时的行车方向 (恢复确认用)
        self._new_identity()
        self._verdict_seq = itertools.count(1)
        self.reconciliation = {}
        self.world = World(run_id=self.run_id,
                           db_path=str(RUNS /
                                       f"{self.run_id}-{time.time_ns()}.sqlite"))
        self._seed_ledger()
        with self.gate.lock:
            self.gate.sim0 = float(d.time)
            self.gate.t0 = time.monotonic()
            self.gate.paused = False
            self.gate.snap = {"t": round(float(d.time), 3), "next_t": 0.0}
        # 初始位姿快照 (mission 未跑时 /api/state 也有完整 per-vehicle 位姿)
        mujoco.mj_forward(cell.model, d)
        init_snap = dict(self.gate.snap)
        for name, bid in self.adapters[self.cell.vehicles[0]]._body_ids.items():
            init_snap[name] = [round(v, 5) for v in d.xpos[bid]] + \
                              [round(v, 5) for v in d.xquat[bid]]
        with self.gate.lock:
            self.gate.snap = init_snap
        self.mode = "idle"
        self.step = -1
        self.step_label = "待命"

    def _seed_ledger(self) -> None:
        """注册本体事实 (与物理初态一致): 3 货位 + H01 出库位 + 3 托盘 + 3 货物。"""
        w = self.world
        w.register_slot(Slot("A-L3-C1", (0.0, 0.0, 0.400), SlotOcc.OCCUPIED))
        w.register_slot(Slot("A-L2-C1", (0.0, 0.0, 0.250), SlotOcc.OCCUPIED))
        w.register_slot(Slot("A-L1-C1", (0.0, 0.0, 0.100), SlotOcc.OCCUPIED))
        w.register_slot(Slot("H01", (self.LOC_X["H01"], 0.0, 0.045),
                             SlotOcc.EMPTY))
        for pid, slot in (("PAL-001", "A-L3-C1"), ("PAL-002", "A-L1-C1"),
                          ("PAL-003", "A-L2-C1")):
            w.register_pallet(Pallet(pid, 0.022, Loc(LocKind.SLOT, slot),
                                     PalletCargo.LOADED))
        for gid, pid, sku in (("G-001", "PAL-001", "COKE-500ML×24"),
                              ("G-002", "PAL-002", "SPARKLING-330ML×24"),
                              ("G-003", "PAL-003", "MIZONE-500ML×12")):
            w.register_goods(GoodsUnit(gid, sku, 0.023,
                                       Loc(LocKind.PALLET, pid),
                                       GoodsLifecycle.IN_WAREHOUSE))
            w._conn.execute("INSERT INTO bindings VALUES(?,?,?,1)",
                            (gid, pid, 1))
        w._conn.commit()

    # ---------- 位置常量与通用取放任务 (P2 多货物/任务队列) ----------
    LOC = {"L1": 0.100, "L2": 0.250, "L3": 0.400, "STAGE": 0.045,
           "S1": 0.045, "H01": 0.045}
    # 暂存台台上托盘中心 x: 台面 x∈[-0.400,-0.280], 托盘半长 40mm →
    # 中心可行域 [-0.350,-0.330]。S1 = F01 的交接放托位 (靠巷道侧),
    # H01 = F02 的最终出库位 (靠货架侧), 相距 20mm —— 交接是真实的位置
    # 转移, 不是原地过户; 单车旧任务键 "STAGE" 统一到 H01 位。
    LOC_X = {"STAGE": -0.330, "S1": -0.350, "H01": -0.330}
    SLOT_OF = {"L1": "A-L1-C1", "L2": "A-L2-C1", "L3": "A-L3-C1",
               "STAGE": "H01", "H01": "H01"}   # S1 是 STATION, 无 slot 行
    # 任务队列: (vid, pallet, goods, src, dst) 或旧式 4 元组 (=首车);
    # agent dispatch 覆盖。交接链 (P3 Wave2): 单托盘所有权转移
    # ROBOT:F01 → STATION:S1 → ROBOT:F02 → SLOT:H01。
    HANDOFF_TASKS = [("F01", "PAL-001", "G-001", "L3", "S1"),
                     ("F02", "PAL-001", "G-001", "S1", "H01")]
    DEFAULT_TASKS = [("PAL-001", "G-001", "L3", "STAGE"),
                     ("PAL-001", "G-001", "STAGE", "L3")]

    def _task_steps(self, src: str, dst: str, pid: str, gid: str,
                    vid: str | None = None,
                    clear_lane: bool = False) -> list:
        """per-vehicle 取放计划 (P3 Wave2 参数化, 不复制 roundtrip_plan)。

        vid 缺省 = 首车 (单车任务与 P3 前逐位一致); clear_lane=True 时在
        计划尾部追加「前移让位」段 —— 双车同巷道 (y=0 同线, 底盘只有 x 向
        自由度): 前车放托交接台后若停在退出位 (axle≈-0.624), 车尾距后车
        车头仅 ~16mm, 后车永远无法进近交接台, 而向后让位会被后车挡死。
        唯一可行让位方向是**前移**: 降叉至叉顶 20mm (< 台面下沿 44mm <
        托盘底 45mm), 从暂存台台下净空与货架底层梁下穿过, 停在货位底层
        下方待命位 (axle=+0.020, 车尾 -0.190, 后车进近最前点 -0.304,
        净空 ≥90mm)。
        """
        k = self.cell.k
        ad = self.adapters[vid or self.cell.vehicles[0]]
        v = ad.view
        x_ins = k["rack_insert_axle_x"]
        x_edge = -0.040 - k["approach_gap"] - k["fork_len"] - k["fork_root_x"]
        x_clear = -0.200 - k["fork_root_x"] - k["fork_len"] - 0.010
        off = 0.094                       # 托盘中心相对轴偏置 (实测值)
        x_src_stage = self.LOC_X[src] - off if src in self.LOC_X else None
        x_dst_stage = self.LOC_X[dst] - off if dst in self.LOC_X else None
        ax, lf = v.a_x, v.a_lift
        qx, ql = v.qadr_x, v.qadr_lift
        q0x, tz = v.q0_x, k["transport_z"]

        def mv(x, sp, calm=False, label="", commit=None):
            direction = float(np.sign(x - v.axle_x)) or 1.0
            return (lambda: self._ramp(ad, ax, qx, x - q0x, sp, calm,
                                       watch=direction),
                    label, commit)

        def up(z, sp, calm=False, label="", commit=None):
            return (lambda: self._ramp(ad, lf, ql, z - tz, sp, calm),
                    label, commit)
        zs, zd = self.LOC[src], self.LOC[dst]
        pick = lambda: self._commit_pick(src, pid, ad.robot_id)
        place = lambda: self._commit_place(dst, pid, gid, ad.robot_id)
        steps = []
        if x_src_stage is not None:
            steps += [
                up(zs + 0.010, 0.010, False, f"[{pid}] 叉降至台面临时高度 (+10)"),
                mv(x_src_stage, 0.020, False, "低位进近暂存站 (≤5mm/s 前插)"),
                up(zs + 0.017, 0.002, True, "抬叉离台 (+17)", pick),
                mv(x_src_stage - k["retreat_loaded"] - 0.060, 0.020, True,
                   "带载退出暂存站"),
                up(tz, 0.010, True, "升至运输高度")]
        else:
            steps += [
                up(zs + 0.010, 0.010, False,
                   f"[{pid}] 升至插叉高度 ({self.SLOT_OF[src]}+10)"),
                mv(x_edge, 0.020, False, "接近货架前沿"),
                mv(x_ins, 0.005, False, "插叉 65mm (≤5mm/s 判据)"),
                up(zs + 0.017, 0.002, True, "举升离梁 (+17)", pick),
                mv(x_clear, 0.020, True, "带载直线退出 (叉尖越过横梁)"),
                up(tz, 0.010, True, "降至运输高度 65mm")]
        if x_dst_stage is not None:
            steps += [
                mv(x_dst_stage, 0.020, True, "低位转运至暂存站上方"),
                up(zd + 0.012, 0.010, True, "举至放托位 (台面+12)"),
                up(zd + 0.010, 0.002, False, "微降脱离叉孔上壁"),
                mv(x_dst_stage - k["retreat_loaded"] - 0.060, 0.020, False,
                   "空叉退出 (越过近边+10mm)", place)]
        else:
            steps += [
                mv(x_edge, 0.020, False, "转运至货架前沿"),
                up(zd + 0.017, 0.010, True, f"升至插入高度 ({self.SLOT_OF[dst]}+17)"),
                mv(x_ins, 0.005, False, "对位插入 (≤5mm/s)"),
                up(zd + 0.012, 0.002, False, "缓降至支撑面 (托盘落梁)"),
                up(zd + 0.010, 0.002, False, "再降 2mm 脱离叉孔上壁"),
                mv(x_edge, 0.020, False, "空叉撤出叉孔"),
                mv(v.start_axle_x, 0.020, False, "返回安全车位", place),
                up(tz, 0.010, False, "叉架复位运输高度")]
        if clear_lane:
            steps += [
                up(0.020, 0.005, False, "让位准备: 降叉至台下净空 (叉顶 20mm)"),
                mv(0.020, 0.020, False,
                   "前移让位: 穿越台下净空, 给 F02 让出交接台进近路径"),
                up(tz, 0.010, False, "待命位叉架复位运输高度")]
        return steps

    # 提交点: 只在物理证据成立时写 World (P3 Wave2: 参数化车辆)。
    # B1: 证据从「接触+高度」单帧读数升级为 EvidenceVerdict 多帧联合
    # 判定 (设计 §4 H_valid), 提交从多次写收敛为单次原子 commit_transfer
    # (GPT 评审 P0-1/P1-4)。
    def _commit_pick(self, src: str, pid: str, vid: str = "F01") -> bool:
        c = self.cell
        if not c.pallet_fork_contact(vid) or \
                c.pallet_bottom_z < self.LOC[src] + 0.003:
            return False
        verdict = self._sample_verdict("PICK", pid, vid)
        if not verdict.passed:
            self.narrate(f"[{pid}] 取托证据未过门禁: "
                         f"{'；'.join(verdict.failures)} — 拒绝提交。")
            return False
        t = float(c.data.time)
        to = Loc(LocKind.ROBOT, vid)
        mutations = [
            ("UPDATE pallets SET location=?, entity_version=entity_version+1 "
             "WHERE pallet_id=?", (str(to), pid))]
        if src in self.SLOT_OF:               # S1 是 STATION, 无 slot 行
            mutations.append(
                ("UPDATE slots SET occupancy=? WHERE slot_id=?",
                 (SlotOcc.EMPTY.value, self.SLOT_OF[src])))
        out = self._transfer(
            f"{self.task_id}:pick:{pid}:{self.SLOT_OF.get(src, src)}",
            verdict=verdict,
            expected={"pallets:" + pid:
                      self.world.entity_version_of("pallets:" + pid)},
            mutations=mutations,
            events=[("PalletMoved", {"pallet_id": pid, "to_location": str(to),
                                     "evidence_id": verdict.verdict_id})],
            sim_time_s=t)
        if out is None:
            return False
        self.narrate(f"物理证据确认: {pid} 已离面、由 {vid} 叉臂承托 "
                     f"(承托率 {verdict.support:.0%}, 旧支撑卸载 "
                     f"{verdict.old_support:.0%}, 稳定窗 "
                     f"{verdict.stable_window_s:.2f}s, 证据 "
                     f"{verdict.verdict_id})。"
                     f"事务提交 {pid}→ROBOT:{vid}"
                     + (f", 货位 {self.SLOT_OF[src]} 释放" if src in self.SLOT_OF
                        else "") + "。在库计数不变。")
        return True

    def _commit_place(self, dst: str, pid: str, gid: str,
                      vid: str = "F01") -> bool:
        c = self.cell
        z = self.LOC[dst]
        if c.pallet_fork_contact(vid) or not (z - 0.002 <= c.pallet_bottom_z
                                              <= z + 0.005):
            return False
        verdict = self._sample_verdict("PLACE", pid, vid)
        if not verdict.passed:
            self.narrate(f"[{pid}] 放托证据未过门禁: "
                         f"{'；'.join(verdict.failures)} — 拒绝提交。")
            return False
        t = float(c.data.time)
        expected = {"pallets:" + pid:
                    self.world.entity_version_of("pallets:" + pid)}
        if dst == "S1":      # 交接台过户: 只转移 pallet 所有权到 STATION
            to = Loc(LocKind.STATION, "S1")
            out = self._transfer(
                f"{self.task_id}:place:{pid}:S1", verdict=verdict,
                expected=expected,
                mutations=[("UPDATE pallets SET location=?, "
                            "entity_version=entity_version+1 WHERE pallet_id=?",
                            (str(to), pid))],
                events=[("PalletMoved", {"pallet_id": pid,
                                         "to_location": str(to),
                                         "evidence_id": verdict.verdict_id})],
                sim_time_s=t)
            if out is None:
                return False
            self.narrate(f"交接验真: 叉臂脱离、{pid} 底面落在交接台 S1 支撑面 "
                         f"{z*1000:.0f}mm (承托率 {verdict.support:.0%}, "
                         f"稳定窗 {verdict.stable_window_s:.2f}s)。"
                         f"事务提交 {pid}→STATION:S1 "
                         f"(ROBOT:{vid} → 交接台), 待 F02 接力。在库计数不变。")
            return True
        kind = LocKind.STATION if dst == "STAGE" else LocKind.SLOT
        slot = self.SLOT_OF[dst]
        to = Loc(kind, slot)
        mutations = [
            ("UPDATE pallets SET location=?, entity_version=entity_version+1 "
             "WHERE pallet_id=?", (str(to), pid))]
        if dst in ("STAGE", "H01"):     # 到出库位: G→OUTBOUND_PENDING 仍计在库
            mutations.append(
                ("UPDATE goods SET lifecycle=?, entity_version=entity_version+1 "
                 "WHERE goods_id=?",
                 (GoodsLifecycle.OUTBOUND_PENDING.value, gid)))
            ev_type, ev_extra = "OutboundPending", {"goods_id": gid}
        else:  # 回架: 出库待命 → 在库 (同一事务)
            mutations.append(
                ("UPDATE goods SET lifecycle=?, entity_version=entity_version+1 "
                 "WHERE goods_id=?",
                 (GoodsLifecycle.IN_WAREHOUSE.value, gid)))
            ev_type, ev_extra = "InboundConfirmed", {"goods_id": gid,
                                                      "slot": slot}
        expected["goods:" + gid] = self.world.entity_version_of("goods:" + gid)
        mutations.append(("UPDATE slots SET occupancy=? WHERE slot_id=?",
                          (SlotOcc.OCCUPIED.value, slot)))
        out = self._transfer(
            f"{self.task_id}:place:{pid}:{dst}", verdict=verdict,
            expected=expected, mutations=mutations,
            events=[("PalletMoved", {"pallet_id": pid, "to_location": str(to),
                                     "evidence_id": verdict.verdict_id}),
                    (ev_type, dict(ev_extra, evidence_id=verdict.verdict_id))],
            sim_time_s=t)
        if out is None:
            return False
        self.narrate(f"放托验真 ({vid}): 叉臂脱离、{pid} 底面落在 "
                     f"{slot} 支撑面 {z*1000:.0f}mm (承托率 "
                     f"{verdict.support:.0%}, 旧支撑卸载 "
                     f"{verdict.old_support:.0%}, 证据 {verdict.verdict_id})。"
                     f"事务提交 {pid}→{slot}。")
        return True

    def _sample_verdict(self, kind: str, pid: str,
                        vid: str) -> EvidenceVerdict:
        """多帧证据采样 (设计 §4 稳定 0.5s): 稳定窗内每 0.05s 一帧。

        采样期间的物理步进仍走 adapter._tick —— 暂停门/墙钟配速/位姿
        快照语义与正常执行完全一致。稳定判据作用于**转移对象** (托盘):
        速度 = 帧间位姿差分 —— 放托后车辆自身仍在整定不应否决托盘
        已稳落的证据; 反之托盘被拖动/滑移会被立即捕获。租约 = 巷道
        互斥当前持有者 (任务步骤在锁内)。
        """
        c = self.cell
        ad = self.adapters[vid]
        per_frame = max(1, int(round(FRAME_INTERVAL_S / c.model.opt.timestep)))
        lease = f"aisle:{self.aisle.owner}" if self.aisle.owner == vid else None
        new_surface = "fork" if kind == "PICK" else "beam"
        old_surface = "beam" if kind == "PICK" else "fork"
        prev: dict = {"pos": None, "t": None}

        def probe() -> EvidenceFrame:
            pos, _ = c.pallet_pose
            t = float(c.data.time)
            if prev["pos"] is None:
                speed = 0.0                # 初帧无前帧, 差分从第 2 帧起
            else:
                dt = max(t - prev["t"], 1e-9)
                speed = float(np.linalg.norm(np.asarray(pos) - prev["pos"]) / dt)
            prev.update(pos=np.asarray(pos), t=t)
            return EvidenceFrame(
                t=t,
                new_support=float(c.support_load_ratio(new_surface, vid)),
                old_support=float(c.support_load_ratio(old_surface, vid)),
                pallet_x=float(pos[0]), pallet_y=float(pos[1]),
                pallet_z=float(pos[2]),
                speed=speed)

        def step() -> None:
            for _ in range(per_frame):
                ad._tick()

        frames = collect_frames(step=step, probe=probe)
        return build_verdict(
            verdict_id=f"{self.run_id}-EVD-{next(self._verdict_seq):04d}",
            kind=kind, task_id=self.task_id, pallet_id=pid, vehicle_id=vid,
            lease_id=lease, frames=tuple(frames))

    def _transfer(self, key: str, *, verdict: EvidenceVerdict,
                  expected: dict[str, int],
                  mutations: list[tuple[str, tuple]],
                  events: list[tuple[str, dict]],
                  sim_time_s: float) -> TransferOutcome | None:
        """单点原子提交 + 故障语义 (GPT 评审 P0-1)。

        - 逻辑层拒绝 (版本冲突/证据过期/约束不符): 返回 None —— 上层按
          「证据不足」失败, 不改事实;
        - 提交通道故障 (sqlite 异常 —— 物理已成功而账本未动): 登记挂起
          提交并抛 :class:`ReconciliationRequired`。任务挂起, 禁止依赖
          该状态的后续动作; 叙述+事件上报, 等待 reconcile 同键重放。
        """
        try:
            out = self.world.commit_transfer(
                verdict.verdict_id, expected, mutations, events, key,
                task_id=self.task_id, sim_time_s=sim_time_s,
                evidence_json=verdict.to_json())
            self._board_cache = None       # 账本已变, 库存板缓存失效
            return out
        except WorldError as e:
            self.narrate(f"提交被拒 ({key}): {e} — 不改事实。")
            return None
        except Exception as e:      # 提交通道故障: 回滚已保证零残留
            self.reconciliation[key] = {
                "key": key, "evidence_id": verdict.verdict_id,
                "evidence_json": verdict.to_json(),
                "expected": expected, "mutations": mutations,
                "events": events, "sim_time_s": sim_time_s,
                "error": repr(e)}
            self.world.emit(self.task_id, "ReconciliationRequired",
                            {"idempotency_key": key, "error": repr(e),
                             "evidence_id": verdict.verdict_id},
                            float(self.cell.data.time))
            self.narrate(f"提交通道故障 ({key}): {e!r} — 事务已整体回滚, "
                         "库存零残留; 但物理已完成而账本未更新, 任务挂起"
                         "进入待核对 (RECONCILIATION_REQUIRED)。请核对后"
                         "执行 reconcile 恢复 (同幂等键重放, 不重复记账)。")
            raise ReconciliationRequired(key) from e

    # ---------- mission 线程 ----------
    def run_mission(self) -> None:
        d = self.cell.data
        try:
            self._run_steps(d)
        except Exception as e:
            self.mode = "failed"
            self.step_label = f"异常: {type(e).__name__}"
            self.narrate(f"mission 线程异常: {e!r} — 未提交的事务保持原状。")

    def _ramp(self, ad: PacedForkliftAdapter, idx: int, qadr: int,
              q_target: float, speed: float, calm: bool = False,
              watch: float = 0.0) -> bool:
        """梯形速度剖面驱动到 q_target (R04 限速语义), 复用 adapter._tick
        (配速/采样/暂停门都在里面)。到位判据 = 位置误差 + 可选全域静止。
        ad 决定配速入口与避障观测原点 (per-vehicle, P3 Wave2)。"""
        from sim.controllers import Ramp
        d = self.cell.data
        q0 = float(d.qpos[qadr])
        ramp = Ramp(q0, q_target, speed)
        t0 = d.time
        tick = ad._tick
        n = int((ramp.duration + 12.0) * 2000)      # 预算: 行程 + 12s 整定
        last_check = 0.0
        for _ in range(n):
            d.ctrl[idx] = ramp.at(min(d.time - t0, ramp.duration))
            tick()
            # P4 行车避障: 每 0.5s 仿真经观测通道检查前方 (带噪扫描估计,
            # 禁上帝视角 —— GPT 评审 P0-3: 不读 mocap 真值)
            if watch and d.time - last_check >= 0.5:
                last_check = d.time
                dd = self.obs.front_obstacle(ad.view.vid, direction=watch)
                if dd.near_dist_est < 0.12:
                    with self.gate.lock:
                        self.blocked_reason = "cone"
                        self._blocked_dir = float(watch)
                    return False
            if abs(float(d.qpos[qadr]) - q_target) <= 1e-3:
                if not calm or float(np.abs(d.qvel).max()) < 5e-3:
                    return True
        return abs(float(d.qpos[qadr]) - q_target) <= 2e-3

    def _run_steps(self, d) -> None:
        tasks = self.tasks or self.DEFAULT_TASKS
        # 任务规范化: 4 元组 = 首车 (P3 前旧队列兼容), 5 元组 = (vid, ...)
        plans = []
        for t in tasks:
            if len(t) == 5:
                vid, pid, gid, src, dst = t
            else:
                vid, (pid, gid, src, dst) = self.cell.vehicles[0], t
            # 单巷道让位: 双车编队时, 前车放托交接台 (S1) 后必须前移清出
            # 进近路径, 后车才可能接力 (几何推导见 _task_steps docstring)。
            clear = (len(self.cell.vehicles) > 1 and dst == "S1"
                     and vid == self.cell.vehicles[0])
            steps = self._task_steps(src, dst, pid, gid, vid,
                                     clear_lane=clear)
            plans.append((vid, pid, gid, src, dst, steps))
        self.steps_total = sum(len(s) for *_, s in plans)
        i = 0
        for n, (vid, pid, gid, src, dst, steps) in enumerate(plans, 1):
            with self.db_lock:
                self.world.emit(self.task_id, "TaskStarted",
                                {"task": n, "vehicle": vid, "pallet": pid,
                                 "from": self.SLOT_OF.get(src, src),
                                 "to": self.SLOT_OF.get(dst, dst)},
                                float(d.time))
            self.narrate(f"— 任务 {n}/{len(plans)}: {vid} 承运 {pid} "
                         f"{self.SLOT_OF.get(src, src)} → "
                         f"{self.SLOT_OF.get(dst, dst)} —")
            # 巷道互斥 (P3 Wave2): 作业区 x > -1.0 同一时刻只允许一车。
            if not self.aisle.acquire(vid, block=False):
                self.narrate(f"{vid} 等待巷道空闲 (当前占用: "
                             f"{self.aisle.owner}) — 作业区 x>{AISLE_X_MIN} "
                             "互斥, 同一时刻只允许一车作业。")
                with self.gate.lock:
                    self.step_label = f"T{n}·等待巷道空闲"
                while not self.aisle.acquire(vid, block=False):
                    if self.mode != "running":
                        return                  # reset 打断
                    time.sleep(0.2)
                self.narrate(f"巷道空闲确认, {vid} 进入作业区。")
            try:
                for fn, label, commit in steps:
                    i += 1
                    attempts = 0
                    while True:
                        with self.gate.lock:
                            self.step = i
                            self.step_label = f"T{n}·{label}"
                        if not fn():
                            br = self.blocked_reason
                            if not br:
                                break                       # 真失败
                            # 避障停车: PATH_INFEASIBLE (巷道无法绕行)
                            with self.db_lock:
                                self.world.emit(self.task_id, "ObstacleDetected",
                                                {"step": i, "label": label,
                                                 "reason": br},
                                                float(d.time))
                            self.mode = "blocked"
                            self.step_label = f"受阻: {label}"
                            # 记忆签名只用观测估计 (P0-3: 不用注入真值坐标)
                            est_xy, _ = self.obs.cone_estimate(vid)
                            obs_x = est_xy[0] if est_xy else None
                            self._blocked_sig = self.memory.signature(
                                br, label, obs_x)
                            self._blocked_t0 = time.monotonic()
                            self._blocked_retries = 0
                            self.obs.reset_clear(vid)   # 恢复连胜清零
                            self.narrate("LiDAR 检测到前方障碍, 车辆已安全停车。"
                                         "原路径 PATH_INFEASIBLE (巷道无法绕行) — "
                                         "请移除障碍, 我会重新感知并恢复执行。")
                            hint = self.memory.hint(self._blocked_sig)
                            if hint:
                                self.narrate(hint)
                            # 恢复确认 = 连续 N 次扫描无障碍 (不因一句指令
                            # 放行); reset 打断 (mode 离开 blocked) 同样退出。
                            while (self.mode == "blocked"
                                   and not self.obs.clear_confirmed(
                                       vid, direction=self._blocked_dir)):
                                time.sleep(0.2)
                            if self.mode != "blocked":
                                return                      # reset 打断
                            with self.gate.lock:
                                self.blocked_reason = None
                            attempts += 1
                            self._blocked_retries = attempts
                            self.narrate(f"重新感知: 通道恢复确认 (第 {attempts} 次"
                                         "重试), 从当前位姿继续执行。")
                            continue
                        # 恢复成功 -> 记录决策经验 (Paper II)
                        if self._blocked_sig and attempts:
                            self.memory.record(
                                self._blocked_sig,
                                ["remove_obstacle(user)", "lidar_recheck"],
                                "recovered",
                                time.monotonic() - (self._blocked_t0 or 0),
                                self._blocked_retries)
                            self._blocked_sig = None
                        break
                    if commit is not None:
                        try:
                            with self.db_lock:
                                ok = commit()
                        except ReconciliationRequired:
                            # 物理已成功、账本提交失败: 任务挂起, 后续
                            # 动作全部禁止 (叙述/事件已在 _transfer 上报)
                            with self.gate.lock:
                                self.step_label = f"待核对: {label}"
                            self.mode = "reconciliation_required"
                            return
                        if not ok:
                            with self.db_lock:
                                self.world.emit(self.task_id,
                                                "CommitEvidenceMissing",
                                                {"step": i, "label": label},
                                                float(d.time))
                            self.mode = "failed"
                            self.step_label = f"证据不足: {label}"
                            self.narrate(f"[{label}] 物理执行完成但承托/脱离"
                                         "证据不满足 — 拒绝提交 (agent 可改"
                                         "计划, 不能改事实)。")
                            return
                    with self.db_lock:
                        self.world.emit(self.task_id, "StepCompleted",
                                        {"step": i, "label": label},
                                        float(d.time))
                with self.db_lock:
                    self.world.emit(self.task_id, "TaskCompleted",
                                    {"task": n, "vehicle": vid,
                                     "pallet": pid}, float(d.time))
            finally:
                self.aisle.release(vid)
                if self.mode in ("running", "done"):
                    self.narrate(f"{vid} 退出作业区, 巷道释放。")
        self.step = i
        self.step_label = "队列完成"
        self.mode = "done"

    # ---------- 命令 (只接受来自 agent 层的调用) ----------
    def cmd(self, op: str, value=None) -> dict:
        if op == "start":
            if self.mode in ("idle", "done", "failed"):
                self.mode = "running"
                with self.gate.lock:
                    self.gate.paused = False
                    self.gate.t0 = time.monotonic()
                    self.gate.sim0 = float(self.cell.data.time)
                self._thread = threading.Thread(target=self.run_mission,
                                                daemon=True)
                self._thread.start()
                q = self.tasks or self.DEFAULT_TASKS
                def _desc(t):
                    vid = t[0] if len(t) == 5 else self.cell.vehicles[0]
                    src, dst = (t[3], t[4]) if len(t) == 5 else (t[2], t[3])
                    return (f"{vid}:{self.SLOT_OF.get(src, src)}→"
                            f"{self.SLOT_OF.get(dst, dst)}")
                self.narrate(f"任务队列已受理 ({len(q)} 个任务: "
                             + "; ".join(_desc(t) for t in q)
                             + "): 判据层限速执行, 库存只在物理证据成立后提交,"
                               " 巷道作业区互斥。")
        elif op == "pause":
            with self.gate.lock:
                self.gate.paused = True
            self.narrate("已暂停: 物理冻结在当前状态, 目标位保持。")
        elif op == "resume":
            with self.gate.lock:
                self.gate.paused = False
                sim_now = float(self.cell.data.time) - self.gate.sim0
                self.gate.t0 = time.monotonic() - sim_now / self.gate.speed
            self.narrate("已恢复执行。")
        elif op == "speed":
            v = max(0.25, min(8.0, float(value or 1)))
            with self.gate.lock:
                sim_now = float(self.cell.data.time) - self.gate.sim0
                self.gate.speed = v
                self.gate.t0 = time.monotonic() - sim_now / v
            self.narrate(f"播放倍率 = {v:g}× (物理仿真不变, 只改墙钟配速)。")
        elif op == "dispatch":
            self.tasks = [tuple(t) for t in (value or [])]
            return self.cmd("start")
        elif op == "inject_cone":
            x = float(value if value is not None else -0.25)
            self.place_cone(x)
            self.narrate(f"模拟现场注入锥桶 @ x={x:+.2f}m (巷道主通道)。"
                         "注意: 这是环境变化, 不是任务指令。")
        elif op == "clear_cone":
            self.remove_cones()
            self.narrate("锥桶已移出场景。等待 LiDAR 连续确认通道恢复后"
                         "才会恢复运动 —— 不会因为一句话就盲目开动。")
        elif op == "reconcile":
            # 待核对恢复 (B1): value = 挂起提交的 idempotency_key
            return {**self.reconcile(str(value or "")),
                    "mode": self.mode, "step": self.step,
                    "label": self.step_label, "speed": self.gate.speed}
        elif op == "reset":
            self.tasks = None
            self.mode = "idle"
            self.build()
            self.narrate("已复位: 新 run、账本重播种、与物理初态一致。")
        return {"mode": self.mode, "step": self.step,
                "label": self.step_label, "speed": self.gate.speed}

    def reconcile(self, key: str) -> dict:
        """人工核对后重放挂起提交 (RECONCILIATION_REQUIRED 恢复路径)。

        用同一 idempotency_key 重放: 首次若实际已提交 (应答丢失) 返回
        首次结果不重复记账; 首次若已回滚未提交则现在真正落账。恢复后
        mode 回 idle, 任务可重新 start 继续队列 (已完成步骤的提交幂等
        跳过)。重放仍失败则保持待核对, 不掩盖故障。
        """
        rec = self.reconciliation.get(key)
        if rec is None:
            return {"ok": False,
                    "error": f"无挂起提交: {key!r} (可核对 reconciliation 键集)"}
        with self.db_lock:
            try:
                out = self.world.commit_transfer(
                    rec["evidence_id"], rec["expected"], rec["mutations"],
                    rec["events"], key, task_id=self.task_id,
                    sim_time_s=rec["sim_time_s"],
                    evidence_json=rec["evidence_json"])
            except Exception as e:
                self.narrate(f"核对重放仍失败 ({key}): {e!r} — 保持待核对。")
                return {"ok": False, "error": repr(e), "key": key}
        del self.reconciliation[key]
        self._board_cache = None           # 账本已变, 库存板缓存失效
        with self.db_lock:
            self.world.emit(self.task_id, "ReconciliationApplied",
                            {"idempotency_key": key, "replayed": out.replayed,
                             "event_ids": list(out.event_ids)},
                            float(self.cell.data.time))
        if self.mode == "reconciliation_required":
            self.mode = "idle"
        self.narrate(("核对完成: 挂起提交已幂等落账"
                      + (" (重放首次结果, 无重复记账)" if out.replayed
                         else " (首次落账)")
                      + f", 事件 {len(out.event_ids)} 条 — "
                        "可重新 start 继续队列。"))
        return {"ok": True, "key": key, "replayed": out.replayed,
                "event_ids": list(out.event_ids)}

    # ---------- 锥桶注入/移除 (mocap; 环境操作, 非决策) ----------
    def place_cone(self, x: float) -> None:
        """模拟现场人员放锥 (环境变化, 非任务指令)。mocap 真值只在此
        与 evaluator (tests) 出现 —— 决策侧经 ObservationChannel 感知。"""
        d = self.cell.data
        for mid in self.cell.cone_mocapid:
            if mid < 0:
                continue
            cur = d.mocap_pos[mid]
            if abs(cur[1]) < 1.0:                     # 场景内已有锥桶
                continue
            d.mocap_pos[mid] = [x, 0.0, 0.045]
            with self.db_lock:
                self.world.emit(self.task_id, "ConeInjected",
                                {"x_m": x}, float(d.time))
            return

    def remove_cones(self) -> None:
        """移锥只改环境; 恢复与否由观测通道连续 N 次扫描确认
        (clear_confirmed), 不在这里直接解除 blocked。"""
        d = self.cell.data
        for mid in self.cell.cone_mocapid:
            if mid >= 0:
                d.mocap_pos[mid] = [0.0, 5.0, 0.045]

    # ---------- 输出 ----------
    def narrate(self, text: str) -> None:
        self.chat_seq += 1
        self.chat.append({"seq": self.chat_seq, "role": "agent",
                          "text": text,
                          "t": round(float(self.cell.data.time), 1)})

    def board(self) -> dict:
        now = time.monotonic()
        if self._board_cache and now - self._board_cache[0] < 1.0:
            return self._board_cache[1]
        with self.db_lock:
            c = self.world._conn
            out = {
                "run_id": self.run_id,
                "count": self.world.in_warehouse_count(),
                "slots": [list(r) for r in c.execute(
                    "SELECT slot_id, occupancy FROM slots")],
                "pallets": [list(r) for r in c.execute(
                    "SELECT pallet_id, location, cargo_state FROM pallets")],
                "goods": [list(r) for r in c.execute(
                    "SELECT goods_id, sku, location, lifecycle FROM goods")],
            }
        self._board_cache = (now, out)
        return out

    def events_since(self, seq: int):
        with self.db_lock:
            rows = self.world._conn.execute(
                "SELECT event_id, type, payload, sim_time_s FROM events "
                "ORDER BY event_id").fetchall()
        out = [{"eid": e[0], "type": e[1],
                "payload": json.loads(e[2]), "t": round(e[3], 1)}
               for e in rows[seq:]]
        return rows, out

    def state(self, ev_cursor: int, chat_cursor: int) -> dict:
        with self.gate.lock:
            snap = dict(self.gate.snap)
            speed = self.gate.speed
            paused = self.gate.paused
        rows, events = self.events_since(ev_cursor)
        chats = [m for m in self.chat if m["seq"] > chat_cursor]
        # per-vehicle 位姿 (P3 Wave2): snap 平铺键 <vid>_chassis → 按车嵌套
        poses = {"pallet": snap.get("pallet")}
        for vid in self.cell.vehicles:
            poses[vid] = {b: snap.get(f"{vid}_{b}")
                          for b in ("chassis", "carriage")}
        return {
            "poses": poses, "sim_t": snap.get("t", 0),
            "vehicles": list(self.cell.vehicles),
            "mode": self.mode, "paused": paused, "speed": speed,
            "step": self.step, "total": self.steps_total,
            "step_label": self.step_label,
            "events": events, "n_events": len(rows),
            # UI 障碍标记走观测侧口径 (最近扫描估计, P0-3); 3D 场景本身
            # 渲染的是物理真值, 这个标记表达「车看到了什么」。
            "cones": self._observed_cones(), "blocked": self.blocked_reason,
            "chat": chats, "board": self.board(),
        }

    def _observed_cones(self) -> list[list[float]]:
        """各车最近一次扫描的锥桶估计 (无扫描/无命中 = 空)。"""
        out = []
        for vid in self.cell.vehicles:
            est, _ = self.obs.cone_estimate(vid)
            if est:
                out.append([est[0], est[1]])
        return out


SVC = WorldService()
app = FastAPI(title="T04 World Model")


@app.middleware("http")
async def _nocache(request, call_next):
    resp = await call_next(request)
    if request.url.path.endswith((".html", "/")):
        resp.headers["cache-control"] = "no-store"
    return resp


@app.post("/api/chat")
async def chat(body: dict):
    from twin.agent import handle
    text = str(body.get("text", ""))
    return handle(text, SVC)


@app.get("/api/state")
async def state():
    return SVC.state(0, 0)


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    ev_cur, chat_cur = 0, SVC.chat_seq
    try:
        while True:
            st = SVC.state(ev_cur, chat_cur)
            ev_cur = st["n_events"]
            if st["chat"]:
                chat_cur = st["chat"][-1]["seq"]
            await ws.send_text(json.dumps(st, ensure_ascii=False))
            await asyncio.sleep(1.0 / PUB_HZ)
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8300, log_level="warning")
