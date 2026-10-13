"""W3b — 矩阵仓库世界服务 (:8400): 决策/物理/监控三路流式推送.

组装既有模块 (不改 sim/、不动 twin/server.py 旧双车场景):
- WarehouseCell (8 车 / 128 架 / 384 货位)   — 物理真值 (MuJoCo 1kHz)
- DemandEngine (需求曲线)                    — demand 驱动 continuous 订单流
- Dispatcher (CP-SAT)                        — 任务分配决策 → 决策事件流
- MissionRunner (相位机)                     — pick/haul/fetch/putaway 物理执行
- TrafficManager (锁矩阵)                    — claim_wait / reroute / release 事件

B1/B2 红线在新服务里的延续:
- B1 事务: dispatcher.commit(task_id, evidence=物理校验) —— 托盘终位姿与
  台账一致 (±0.02m) 才提交, 空证据/非 ASSIGNED 一律拒绝 (R09 幂等)。
- B2 观测隔离: 锥桶注入走 traffic.lane_holder 通道占用 (W3 语义预留位,
  见 test_detour_when_column_held 的 "OCC" 外部占用), 决策侧 (A*/claim)
  只消费锁矩阵, 不读注入真值旁路; 注入本身是环境操作。

物理闭环 (sustainable 24h): 出库 pick→haul 把托盘送到交付台; 需求引擎
库存低于阈值触发补货单 → fetch(交付台→交接台)+putaway(交接台→货位)
把托盘送回货架。托盘全程经物理取放 (无 qpos 写入, 无瞬移)。

线程模型: 单仿真线程拥有 MissionRunner/cell 推进; HTTP/ws 线程只做
C 级快照拷贝 (dict()/list() 原子拷贝 + 只读 numpy 标量), 无锁读取。
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np
import re
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sim.demand import DemandEngine, DemandOrder, OrderKind
from sim.dispatch import (TICK_S, Dispatcher, InfeasibleAssignment, Vehicle)
from sim.mission import Job, MissionRunner
from sim.warehouse_cell import K, LAYOUT, WarehouseCell, pallet_name_for

WEB = Path(__file__).parent / "web"
PUB_HZ = 12                      # ws 状态推送频率
MAX_CHAINS = 16                  # 在飞订单链上限 (bound CP-SAT 批规模/队列)
MAX_OUT_STANDS = 4               # 出库链同时持有的交接台数上限 (防 stand↔dock 环)
EVENT_RING, DEC_RING, CHAT_RING = 2048, 512, 200
MAX_CHAINS = 24                  # 在飞订单链上限 (倍率高时队列不憋死)
BACKLOG_CAP = 24                 # 未提交订单背压上限 (超限暂缓生成)
UTIL_SAMPLE_S = 60.0             # 利用率滑动窗采样周期 (仿真秒)
UTIL_WIN = 5                     # 滑动窗样本数 (~5min 仿真)
# ---- 闲置车预置 (利用率专项: 空驶调度是真实仓库利用率手段) ----
REPOS_UTIL_TH = 0.40             # 滑动窗利用率低于此且空闲 → 触发预置
REPOS_MIN_GAP_S = 60.0           # 同车两次预置最小间隔 (仿真秒, 跟随热点漂移)
REPOS_TICK_S = 12.0              # 预置重试节拍 (仿真秒; 与 60s 采样解耦)
REPOS_MAX_ACTIVE = 4             # 同时在途预置车上限 (限场内扰动)
REPOS_NEAR_M = 0.55              # 距目标小于此值不巡游 (过近挪动无意义)


class _TapDeque(deque):
    """traffic.events 替身: append 时同步推入带 seq 的事件流 (供 ws 消费)。"""

    def __init__(self, maxlen: int, push):
        super().__init__(maxlen=maxlen)
        self._push = push

    def append(self, item):                      # TrafficManager 全部走 append
        super().append(item)
        self._push(item)


@dataclass
class Spec:
    """一条物理任务腿 (mission 层任务) — 调度器决策的物理投影。"""
    uid: str
    task_id: str                 # 调度任务 (= 订单) id
    kind: str                    # pick / haul / fetch / putaway
    vid: str
    task: dict                   # MissionRunner 任务字典
    deps: list[str] = field(default_factory=list)
    state: str = "pending"       # pending | active | done | failed
    t_start: float = 0.0
    t_end: float = 0.0
    err_base: int = 0
    label: str = ""
    elig: list[str] = field(default_factory=list)   # 类型+载重合格车队

@dataclass
class Chain:
    """一张订单的物理执行链 (两腿, 交接链先后)。"""
    task_id: str
    kind: str                    # OUTBOUND / REPLENISH
    desc: str
    specs: list[Spec] = field(default_factory=list)
    state: str = "pending"       # pending | active | done | failed
    reserved: dict = field(default_factory=dict)   # stand/dock/slot → 用途
    pallet: str = ""             # 主角托盘 (OUTBOUND=取自货位; REPLENISH=取自交付台)
    t_created: float = 0.0


def _fleet8() -> tuple[Vehicle, ...]:
    return tuple(
        Vehicle(vid, "forklift" if vid.startswith("F") else "agv",
                LAYOUT["vehicle_class"][vid], LAYOUT["park"][vid][:2])
        for vid in (*("F01", "F02", "F03", "F04"),
                    *("A01", "A02", "A03", "A04")))


class MatrixService:
    """矩阵仓库世界模型: 物理 + 决策 + 事件 + 指标 (headless 可用)。"""

    def __init__(self, *, seed: int = 7, multiplier: float = 1.0,
                 speed: float = 1.0, replenish_threshold: int = 1):
        self.seed = int(seed)
        self.multiplier = float(multiplier)
        self.speed = float(speed)          # 0 = 不限速 (压测)
        self.replenish_threshold = int(replenish_threshold)
        self.mode = "idle"                 # idle|running|paused|stopped|failed
        self._demand_on = True             # continuous 开关 (压测收尾 drain 用)
        self._build()

    # ------------------------------------------------------------ 构建
    def _build(self) -> None:
        """重建世界 (首次构造 / 停后重启 / 发散自恢复)。

        身份纪律 (对齐 B3/P1-7): 重建即新 run —— run_id 必须重生成,
        否则'新 run'只是旧身份下的静默重置, 审计与事件归属全乱。
        """
        prev = getattr(self, "run_id", None)
        self.demand = DemandEngine(seed=self.seed,
                                   global_multiplier=self.multiplier,
                                   replenish_threshold=self.replenish_threshold)
        inv = self.demand.inventory()      # slot 全 id → sku
        seed_slots = [s for s, sku in inv.items() if sku is not None]
        # WarehouseCell 固定播种 R11-S0-L2 (交接链测试位): 计划库存须与
        # 物理一致, 否则补货单可能落进已被物理占用的货位 → putaway 失败
        fixed = "R11-S0-L2"
        if fixed not in seed_slots:
            seed_slots.append(fixed)
            light = min(self.demand.catalog.values(),
                        key=lambda c: c.weight_kg).sku
            self.demand._place(fixed, light)
            inv[fixed] = light
        # 交接台不预载: 所有 H 台留给在飞链 (W3b 连续模式)
        self.cell = WarehouseCell(timestep_s=0.001, seed=self.seed,
                                  seed_slots=seed_slots,
                                  preload_stands=())
        self.cell.warmup(0.5)
        self.mr = MissionRunner(self.cell)
        self.dispatcher = Dispatcher(
            fleet=_fleet8(), event_sink=self._on_dec,
            load_offset_provider=lambda: {
                vid: int(round(self._real_busy_s(vid) / TICK_S))
                for vid in self.busy_s})
        # 托盘 → SKU (决策/展示用; 物理由 mission 台账跟踪)
        self.sku_of: dict[str, str] = {}
        for slot_full, sku in inv.items():
            if sku is not None:
                self.sku_of[pallet_name_for(slot_full)] = sku

        self.chat: deque = deque(maxlen=CHAT_RING)
        self.chat_seq = 0
        self.decisions: deque = deque(maxlen=DEC_RING)
        self._dec_seq = itertools.count(1)
        self.dec_total = 0
        self.feeds: deque = deque(maxlen=EVENT_RING)
        self._feed_seq = itertools.count(1)
        self.feed_total = 0
        self._uid_seq = itertools.count(1)
        self._gen = 0
        self.chains: list[Chain] = []
        self._unsubmitted: deque = deque()          # 已生成未提交调度的订单
        self._unsubmitted_age0: float | None = None  # 队首到达仿真时刻
        self._svc_lock = threading.Lock()  # 调度/需求跨线程互斥
        self._dock_loaded: dict[str, float] = {}    # dock → sim_t (FIFO 取托)

        self._repos: dict[str, dict] = {}       # vid → {job, goal, t0, cancel?}
        self._repos_last: dict[str, float] = {}  # vid → 上次预置发起仿真时刻
        self._repos_tick_t = 0.0                 # 上次预置重试节拍仿真时刻
        self._repos_parked: dict[str, str] = {}  # vid → 驻位节点 (看门狗让位簿记)
        self._repos_started = 0
        self._repos_busy_s: dict[str, float] = {v: 0.0 for v in self.cell.veh}
        self._repos_done = 0

        # traffic 事件分流 (assign/reroute/claim_wait/op_done 全走这里)
        self.mr.traffic.events = _TapDeque(4096, self._on_traffic)

        # 指标累加器
        self.orders_done = 0
        self.tasks_done = 0
        self.tasks_failed = 0
        self.busy_s: dict[str, float] = {v: 0.0 for v in self.cell.veh}
        self.done_marks: deque = deque(maxlen=4096)  # 任务完成 sim_t 截记
        self.stalled = False
        self.stall_s = 0.0
        self._last_progress = time.monotonic()
        self._util_t = float(self.cell.data.time)
        self._util_win: deque = deque(maxlen=UTIL_WIN)
        self._util_sample_t = float(self.cell.data.time)
        self._busy_snap: dict[str, float] = dict(self.busy_s)
        self._events_seen = 0
        self._pace_base = (float(self.cell.data.time), time.monotonic())
        self._thread: threading.Thread | None = None
        self.run_id = (f"MX-{time.strftime('%Y%m%d-%H%M%S')}-"
                       f"{uuid.uuid4().hex[:8]}")
        if prev:
            self._feed("SYS", "run_rebuilt",
                       f"世界重建: {prev} → {self.run_id} (账本重置)")

    # ------------------------------------------------------------ 事件流
    def _on_dec(self, ev: dict) -> None:
        self.dec_total += 1
        self.decisions.append({"seq": next(self._dec_seq), **ev})

    def _feed(self, vid: str, kind: str, detail: str) -> None:
        self.feed_total += 1
        self.feeds.append({"seq": next(self._feed_seq),
                           "t": round(float(self.cell.data.time), 2),
                           "vid": vid, "kind": kind, "detail": detail})

    def _on_traffic(self, item) -> None:
        _, vid, kind, detail = item
        self._feed(vid, kind, detail if isinstance(detail, str)
                   else json.dumps(detail, ensure_ascii=False))

    def narrate(self, text: str, role: str = "agent") -> None:
        self.chat_seq += 1
        self.chat.append({"seq": self.chat_seq, "role": role, "text": text,
                          "t": round(float(self.cell.data.time), 1)})

    # ------------------------------------------------------------ 命令
    def start(self) -> dict:
        if self.mode in ("idle", "stopped", "failed"):
            if self.mode != "idle":
                self._build()               # 新 run: 身份/账本/物理全重建
                self._gen += 1              # 旧 mission 线程随之退役
            self.mode = "running"
            self._demand_on = True
            self._pace_base = (float(self.cell.data.time), time.monotonic())
            self._thread = threading.Thread(
                target=self._mission_loop, args=(self._gen,), daemon=True)
            self._thread.start()
            self.narrate(
                f"矩阵仓库连续运行已启动 (run {self.run_id}): 8 车混合编队 "
                f"(叉车 F01-F04 / AGV A01-A04, L1/L2/L3 异构), 128 架 384 货位, "
                f"需求曲线倍率 {self.multiplier:g}×。调度 = CP-SAT 批次分配 "
                "(候选→选定→理由→求解状态), 物理 = 相位机逐腿执行, "
                "托盘位置提交仅在物理校验通过后进行 (B1)。说「下 10 个订单」"
                "可加压, 「注入锥桶」可制造通道占用。")
        return self.status()

    def pause(self) -> dict:
        if self.mode == "running":
            self.mode = "paused"
            self.narrate("已暂停: 物理冻结在当前相位, 锁矩阵保持。")
        return self.status()

    def resume(self) -> dict:
        if self.mode == "paused":
            self.mode = "running"
            self._pace_base = (float(self.cell.data.time), time.monotonic())
            self.narrate("已恢复。")
        return self.status()

    def stop(self) -> dict:
        if self.mode in ("running", "paused"):
            self.mode = "stopped"
            self.narrate("已停止: 在飞任务冻结 (可 start 重建新 run)。")
        return self.status()

    def set_speed(self, v: float) -> dict:
        self.speed = max(0.0, min(16.0, float(v)))
        self._pace_base = (float(self.cell.data.time), time.monotonic())
        self.narrate(f"播放倍率 = {'max (不限速)' if self.speed == 0 else f'{self.speed:g}×'}"
                     " (墙钟配速, 物理不变)。")
        return self.status()

    def status(self) -> dict:
        return {"mode": self.mode, "speed": self.speed,
                "sim_t": round(float(self.cell.data.time), 2),
                "run_id": self.run_id}

    def manual_orders(self, n: int) -> dict:
        """立即生成 n 张订单 (需求曲线前移; 计划库存同步扣减)。

        从 HTTP 线程调用: 需求时钟推进与调度批次提交都与仿真线程的
        _pump_demand 互斥 (RNG/CP-SAT/任务表非线程安全)。
        """
        n = max(1, min(60, int(n)))
        got: list[DemandOrder] = []
        with self._svc_lock:
            t = self.demand.cursor_s
            while len(got) < n and t < self.demand.cursor_s + 6 * 3600:
                t = min(t + 120.0, self.demand.cursor_s + 6 * 3600)
                got.extend(self.demand.generate_until(t))
            out = [o for o in got if o.kind is OrderKind.OUTBOUND]
            # 手动单直接进调度批次 (需求时钟已前移, 不能再等 _pump_demand)
            self._dispatch_batch(list(got))
        self.narrate(f"已下单 {len(out)} 张出库单 "
                     f"(+{len(got) - len(out)} 张补货单) 已提交调度 "
                     "(计划库存同步扣减)。")
        self._last_progress = time.monotonic()
        return {"queued": len(got), "outbound": len(out)}

    # ------------------------------------------------------------ 雉桶 (B2 延续)
    _LANE_RE = re.compile(r"^(col|row):(\d+)$")

    def inject_cone(self, spec: str | None = None) -> dict:
        """通道占用注入 (环境操作): 对应物理锥桶封路, 决策侧只见锁。"""
        g = self.mr.traffic.graph
        lanes = [f"col:{i}" for i in range(1, len(g.col_xs) - 1)] + \
                [f"row:{j}" for j in range(1, len(g.aisle_ys) - 1)]
        lane = None
        if spec:
            m = self._LANE_RE.match(spec.strip())
            if not m or f"{m.group(1)}:{m.group(2)}" not in (
                    [f"col:{i}" for i in range(len(g.col_xs))]
                    + [f"row:{j}" for j in range(len(g.aisle_ys))]):
                return {"ok": False, "error": f"通道名应为 col:0-4 / row:0-4"}
            lane = f"{m.group(1)}:{m.group(2)}"
        else:
            lane = lanes[int(time.time()) % len(lanes)]
        held = self.mr.traffic.lane_holder
        held[lane] = "CONE"
        xy = self._lane_anchor(lane)
        self._feed("CONE", "cone_injected",
                   f"{lane} @({xy[0]:.2f},{xy[1]:.2f})")
        self.narrate(f"环境注入: 通道 {lane} 被锥桶封闭 (等效占用锁)。"
                     "在飞车辆将按锁矩阵绕行/等待, 移除后自动恢复。")
        return {"ok": True, "lane": lane, "xy": xy}

    def clear_cones(self) -> dict:
        held = self.mr.traffic.lane_holder
        hit = [r for r, v in list(held.items()) if v == "CONE"]
        for r in hit:
            del held[r]
        for vid in list(self.mr.traffic.last_blocked):
            self.mr.traffic.prune_exclusions(vid)
        if hit:
            self._feed("CONE", "cone_cleared", ",".join(sorted(hit)))
            self.narrate(f"锥桶已移除 ({', '.join(sorted(hit))}); "
                         "等待中的车辆将重新申请路由。")
        return {"ok": True, "cleared": hit}

    def _lane_anchor(self, lane: str) -> tuple[float, float]:
        g = self.mr.traffic.graph
        kind, i = lane.split(":")
        i = int(i)
        if kind == "col":
            return (g.col_xs[i], 0.0)
        return (0.0, g.aisle_ys[i])

    def cone_xy(self) -> list[list[float]]:
        held = self.mr.traffic.lane_holder
        return [list(self._lane_anchor(r)) for r, v in held.items()
                if v == "CONE"]

    # ------------------------------------------------------------ 闲置车预置
    def _maybe_reposition(self, t: float) -> None:
        """滑动窗利用率低 (<40%) 且无在飞任务的车 → 预置到任务密集列。

        空驶调度是真实仓库利用率手段: 预置 = 普通 nav 移动 (不占订单链,
        无账本变更), 沿锁矩阵开往热点 stand 列的 staging 节点; 真任务
        gate 时立刻让位 (见 _repos_yield), 等待者被占时看门狗强制让位。
        """
        if not self._demand_on or len(self._repos) >= REPOS_MAX_ACTIVE:
            return
        per = self._util_win[-1]["per"] if self._util_win else {}
        if not per:
            return
        cands = [v for v, u in per.items()
                 if u < REPOS_UTIL_TH and self.mr.jobs.get(v) is None
                 and v not in self._repos
                 and t - self._repos_last.get(v, -1e9) >= REPOS_MIN_GAP_S]
        for vid in sorted(cands, key=lambda v: (per[v], v)):
            if len(self._repos) >= REPOS_MAX_ACTIVE:
                break
            for goal in self._repos_goals(vid):
                if self._start_reposition(vid, goal, per[vid], t):
                    break

    def _repos_goals(self, vid: str) -> list[str]:
        """预置目标序列 (按优先序逐一尝试 claim, 全败则本拍放弃)。

        - 驻位车 (安全 transit 节点): [自身泊位, 其他热点列] —— 巡回
          pk↔热点 或热点列互移, 避开被真任务独占的热点列通道;
        - 泊位/他处车: [热点列 x:i:2 ...] (最近的高占用 stand 列优先)。

        x:i:2 是纯 transit 节点 (行列交点, 非 stand/dock/货位/standstrip
        类作业目标): 驻位其上不卡真任务目标节点, 穿越争用由锁矩阵 +
        A* 硬排除绕行消化 (驻位挡路时看门狗令其回泊)。
        """
        g = self.mr.graph
        cur = self.mr.node_of.get(vid)
        staged = bool(cur and cur.startswith("x:")
                      and not cur.endswith(":0"))
        w: dict[str, int] = {}
        for c in self.chains[-20:]:
            st = c.specs[0].task.get("stand") if c.specs else None
            if st:
                w[st] = w.get(st, 0) + (2 if c.state in ("pending", "active")
                                        else 1)
        goals: list[str] = [g.park_node(vid)] if staged else []
        if not w:
            return goals
        vx, vy = self.cell.veh[vid].xy()

        def dist(stand: str) -> float:
            p = g.pos[f"stand:{stand}"]
            return abs(p.x - vx) + abs(p.y - vy)

        ranked = sorted(w.items(), key=lambda kv: (-kv[1], dist(kv[0]), kv[0]))
        taken = {st["goal"] for st in self._repos.values()}
        for _rank, (stand, _n) in enumerate(ranked[:3]):
            node = g.xnode(g.col_index_of_stand(stand), 2)
            p = g.pos[node]
            if node == cur:
                continue
            if abs(p.x - vx) + abs(p.y - vy) < REPOS_NEAR_M:
                continue                        # 过近挪动无意义
            if node in taken or self.mr.traffic.node_holder.get(node) \
                    not in (None, vid):
                continue                        # 目标被他车/他预置占用
            goals.append(node)
        return goals

    def _start_reposition(self, vid: str, goal: str, util: float | None,
                          t: float, *, why: str = "stage") -> bool:
        """预置 = nav 移动: A* 整路由原子 claim → 逐边 submit (mission 同款)。

        why="stage" (利用率预置) / "yield" (看门狗令驻位车回泊让路)。
        """
        mr = self.mr
        kind = "agv" if vid in self.cell.agv_ids else "forklift"
        mr.traffic.prune_exclusions(vid)
        lanes, nodes = mr.traffic.occupancy(exclude=vid)
        blk = mr.traffic.last_blocked.get(vid, {})
        plan = mr.graph.plan(mr.node_of[vid], goal, kind, mr.quad_of[vid],
                             frozenset(lanes), frozenset(nodes),
                             hard_lanes=frozenset(blk.get("lanes", ())),
                             hard_nodes=frozenset(blk.get("nodes", ())))
        if plan is None:
            return False
        path, edges = plan
        if not edges or not mr.traffic.claim_route(vid, edges, path[1:]):
            return False
        frm = mr.node_of[vid]
        job = Job(vid=vid, task={"kind": "reposition", "goal": goal})
        job.path, job.edges, job.idx = path, edges, 0
        job.mode = "nav"
        self._repos[vid] = {"job": job, "goal": goal, "t0": t,
                            "step_due": False}
        self._repos_last[vid] = t
        self._repos_started += 1
        self._repos_parked.pop(vid, None)
        mr._submit_edge(vid, job, 0)
        reason = (f"滑动窗利用率 {util:.0%} < {REPOS_UTIL_TH:.0%} 且无在飞任务 "
                  f"→ 预置到热点列 staging 节点 (真任务即刻抢占)"
                  if why == "stage" else
                  "驻位挡住等待中的真任务 → 回泊让路 (锁矩阵护卫)")
        self._on_dec({"type": "Reposition", "vid": vid, "ts_s": round(t, 2),
                      "from": frm, "to": goal, "reason": reason,
                      "solver_status": "PLANNING", "solve_ms": 0.0,
                      "makespan_ticks": None})
        self._feed(vid, "reposition", f"预置 {frm} → {goal}")
        return True

    def _advance_repos(self, idle: list[str]) -> None:
        """预置车队列耗尽回调 → 标记待推进并推进 (与 mission nav 同款)。"""
        for vid in idle:
            st = self._repos.get(vid)
            if st is not None:
                st["step_due"] = True
            self._repos_progress(vid)

    def _repos_pump(self) -> None:
        """每拍推进驻留中的预置车 (重定向/续走 claim 失败时重试)。"""
        for vid in list(self._repos):
            self._repos_progress(vid)

    def _repos_progress(self, vid: str) -> None:
        """到节点边界的推进: 滞后释放 → 抢占判定 (安全节点才可驻) → 下一边。"""
        mr = self.mr
        st = self._repos.get(vid)
        if st is None or mr.runner.busy(vid) or not st.get("step_due"):
            return
        st["step_due"] = False
        job = st["job"]
        job.idx += 1
        mr.traffic.notify_progress(vid, job.path, job.edges, job.idx)
        mr.node_of[vid] = job.path[job.idx]
        if job.idx >= len(job.edges):           # 到达: 预置终点必为安全节点
            self._teardown_repos(vid, "staged")
            return
        if st.pop("cancel", False):
            if self._safe_park(vid, mr.node_of[vid]):
                self._teardown_repos(vid, "preempted")
                return
            if self._repos_redirect(vid, st):   # 改向最近安全节点再驻
                return
            if not mr.traffic.claim_route(      # 续走原路由剩余段
                    vid, job.edges[job.idx:], job.path[job.idx + 1:]):
                st["cancel"] = True             # 驻留重试 (泵驱动)
                st["step_due"] = True
                mr.traffic.claim_route(vid, [], [mr.node_of[vid]])
                return
        mr._submit_edge(vid, job, job.idx)

    def _safe_park(self, vid: str, node: str) -> bool:
        """可驻位节点: 自身 pk (仅 owner 的回泊目标) 或列通道 x:i:2..5
        (纯 transit 交点)。stand/dock/sf/sm/stg/x:i:0(standstrip 目标)/
        他人 pk 都是某真任务的目标节点 —— 驻其上 = 永久卡住该任务。
        """
        return node == self.mr.graph.park_node(vid) or (
            node.startswith("x:") and not node.endswith(":0"))

    def _repos_redirect(self, vid: str, st: dict) -> bool:
        """改向最近安全驻位节点 (x:i:2..5 / 自身 pk), 供抢占收尾。"""
        mr = self.mr
        g = mr.graph
        kind = "agv" if vid in self.cell.agv_ids else "forklift"
        vx, vy = self.cell.veh[vid].xy()

        def d(n: str) -> float:
            p = g.pos[n]
            return abs(p.x - vx) + abs(p.y - vy)

        cands = sorted([g.xnode(i, j)
                        for i in range(len(g.col_xs)) for j in (2, 3, 4, 5)]
                       + [g.park_node(vid)], key=d)
        mr.traffic.release_all(vid)
        mr.traffic.prune_exclusions(vid)
        for goal in cands[:6]:
            if goal == mr.node_of[vid]:
                continue
            lanes, nodes = mr.traffic.occupancy(exclude=vid)
            blk = mr.traffic.last_blocked.get(vid, {})
            plan = g.plan(mr.node_of[vid], goal, kind, mr.quad_of[vid],
                          frozenset(lanes), frozenset(nodes),
                          hard_lanes=frozenset(blk.get("lanes", ())),
                          hard_nodes=frozenset(blk.get("nodes", ())))
            if plan is None:
                continue
            path, edges = plan
            if not edges or not mr.traffic.claim_route(vid, edges, path[1:]):
                continue
            job = st["job"]
            job.path, job.edges, job.idx = path, edges, 0
            st["goal"] = goal
            st["step_due"] = False
            mr._submit_edge(vid, job, 0)
            self._feed(vid, "reposition_preempt", f"改向安全驻位 → {goal}")
            return True
        mr.traffic.claim_route(vid, [], [mr.node_of[vid]])
        return False

    def _repos_yield(self, vid: str) -> bool:
        """真任务抢占: 标记让位; 驻留态立即尝试安全收尾/改向, 在途续走。"""
        st = self._repos.get(vid)
        if st is None:
            return True
        if "cancel" not in st:
            st["cancel"] = True
            self._feed(vid, "reposition_preempt", "真任务抢占, 让位")
        if not self.mr.runner.busy(vid):
            self._repos_progress(vid)      # 驻留态: 立即安全收尾或改向
        return vid not in self._repos

    def _teardown_repos(self, vid: str, outcome: str) -> None:
        """收尾: 释放整条路由, 仅驻位节点保持持有 (物理占用事实)。"""
        if self._repos.pop(vid, None) is None:
            return
        mr = self.mr
        mr.traffic.release_all(vid)
        mr.traffic.claim_route(vid, [], [mr.node_of[vid]])
        self._repos_done += 1
        self._repos_parked[vid] = mr.node_of[vid]   # 驻位簿记 (看门狗可令让位)
        zh = {"staged": "已驻位热点列", "preempted": "预置中止 (让位)"}[outcome]
        self._feed(vid, f"reposition_{outcome}", f"{zh} @ {mr.node_of[vid]}")

    def _repos_watchdog(self) -> None:
        """锁矩阵护卫: 等待中的真任务被预置路由/驻位车挡住 → 令其让位。"""
        mr = self.mr
        for vid, job in list(mr.jobs.items()):
            if job is None or job.mode != "wait":
                continue
            blk = mr.traffic.last_blocked.get(vid)
            if not blk:
                continue
            for r in list(blk.get("lanes", ())) + list(blk.get("nodes", ())):
                holder = mr.traffic.holder_of(r)
                if not holder:
                    continue
                st = self._repos.get(holder)
                if st is not None:
                    if "cancel" not in st:
                        st["cancel"] = True
                        self._feed(holder, "reposition_preempt",
                                   f"让位: {vid} 等待资源 {r}")
                    continue
                if (holder in self._repos_parked
                        and mr.jobs.get(holder) is None
                        and holder not in self._repos):
                    # 驻位车挡住等待者 → 令其回自身 pk (repos 机制, 目标安全)
                    t = float(self.cell.data.time)
                    if self._start_reposition(holder, mr.graph.park_node(holder),
                                              None, t, why="yield"):
                        self._repos_parked.pop(holder, None)
                        self._feed(holder, "reposition_preempt",
                                   f"让位: {vid} 等待资源 {r} → 回泊")

    def _pump_demand(self) -> None:
        if not self._demand_on:
            # 收尾 drain: 排队订单全部直送调度 (不再攒批), 让系统可清空
            if self._unsubmitted:
                room = MAX_CHAINS - sum(1 for c in self.chains
                                        if c.state in ("pending", "active"))
                batch = []
                while room > 0 and self._unsubmitted:
                    batch.append(self._unsubmitted.popleft())
                    room -= 1
                with self._svc_lock:
                    self._dispatch_batch(batch)
            return
        now = float(self.cell.data.time)
        if now <= self.demand.cursor_s:
            return                  # 手动单已把需求时钟前移: 不回拨
        if len(self._unsubmitted) > BACKLOG_CAP:
            return                  # 背压: 队列过深暂缓生成 (时钟滞后追赶)
        with self._svc_lock:
            try:
                fresh = self.demand.generate_until(now)
            except RuntimeError as e:                # 无空闲货位等
                self.narrate(f"需求引擎停摆: {e}")
                self._demand_on = False
                return
            if fresh and self._unsubmitted_age0 is None:
                self._unsubmitted_age0 = now
            self._unsubmitted.extend(fresh)
            self._trim_chains()
            # 批缓冲: 攒 3 单或最长等 60s 仿真再送 CP-SAT (决策质量↑)
            age = now - (self._unsubmitted_age0 or now)
            if self._unsubmitted and len(self._unsubmitted) < 3 and age < 60.0:
                return
            room = MAX_CHAINS - sum(1 for c in self.chains
                                    if c.state in ("pending", "active"))
            batch = []
            while room > 0 and self._unsubmitted:
                batch.append(self._unsubmitted.popleft())
                room -= 1
            self._unsubmitted_age0 = None
            self._dispatch_batch(batch)

    def _dispatch_batch(self, batch: list) -> None:
        """订单批 → CP-SAT 调度 → 决策事件 → 执行链 (手动/自动共用)。"""
        if not batch:
            return
        now = float(self.cell.data.time)
        try:
            events = self.dispatcher.submit_orders(batch, now_s=now)
        except InfeasibleAssignment as e:
            self.narrate(f"批次不可行 (CP-SAT 判定): {e} — 订单丢弃。")
            return
        for o, ev in zip(batch, events):
            self._make_chain(o, ev)
        n_ok = sum(1 for e in events if e["type"] == "TaskAssigned")
        st = {}
        for e in events:
            st[e["solver_status"]] = st.get(e["solver_status"], 0) + 1
        sts = " ".join(f"{k}×{v}" for k, v in sorted(st.items()))
        self.narrate(f"调度批次 {len(batch)} 单: {n_ok} 单已分配 "
                     f"[{sts}], 决策已入面板。")
        if n_ok:
            self._last_progress = time.monotonic()   # 活性: 新链入系统

    def _trim_chains(self) -> None:
        if len(self.chains) > 64:
            del self.chains[:len(self.chains) - 64]

    def _make_chain(self, order: DemandOrder, ev: dict) -> None:
        if ev["type"] != "TaskAssigned":
            self.narrate(f"订单 {order.order_id} 无法分配: {ev['reason']}")
            return
        t = self.dispatcher.task(order.order_id)
        chosen = dict(t.assignment)
        if order.kind is OrderKind.OUTBOUND:
            slot = order.src.rsplit("-L", 1)[0]
            level = int(order.src.rsplit("-L", 1)[1])
            handoff, dock = t.handoff, order.dst
            fv = chosen[f"{order.order_id}:fork"]
            av = chosen[f"{order.order_id}:trunk"]
            c = Chain(task_id=order.order_id, kind="OUTBOUND",
                      desc=f"{order.sku} 出库 {order.src}→{dock}",
                      pallet=pallet_name_for(order.src),
                      t_created=float(self.cell.data.time))
            s1 = Spec(uid=f"S{next(self._uid_seq)}", task_id=order.order_id,
                      kind="pick", vid=fv,
                      task={"kind": "pick", "slot": slot, "level": level,
                            "stand": handoff},
                      elig=self._elig("forklift", t.weight_kg),
                      label=f"取 {order.src}→{handoff}")
            s2 = Spec(uid=f"S{next(self._uid_seq)}", task_id=order.order_id,
                      kind="haul", vid=av,
                      task={"kind": "haul", "stand": handoff, "dock": dock},
                      elig=self._elig("agv", t.weight_kg),
                      deps=[s1.uid], label=f"{handoff}→{dock} 干线")
            c.specs = [s1, s2]
        else:                                        # REPLENISH
            slot = order.dst.rsplit("-L", 1)[0]
            level = int(order.dst.rsplit("-L", 1)[1])
            handoff = t.handoff
            av = chosen[f"{order.order_id}:trunk"]
            fv = chosen[f"{order.order_id}:fork"]
            c = Chain(task_id=order.order_id, kind="REPLENISH",
                      desc=f"{order.sku} 补货 →{order.dst}",
                      t_created=float(self.cell.data.time))
            s1 = Spec(uid=f"S{next(self._uid_seq)}", task_id=order.order_id,
                      kind="fetch", vid=av,
                      task={"kind": "fetch", "dock": "", "stand": handoff},
                      elig=self._elig("agv", t.weight_kg),
                      label=f"交付台→{handoff} 取托")
            s2 = Spec(uid=f"S{next(self._uid_seq)}", task_id=order.order_id,
                      kind="putaway", vid=fv,
                      task={"kind": "putaway", "stand": handoff,
                            "slot": slot, "level": level},
                      elig=self._elig("forklift", t.weight_kg),
                      deps=[s1.uid], label=f"{handoff}→{order.dst} 上架")
            c.specs = [s1, s2]
        self.chains.append(c)

    # ------------------------------------------------------------ 执行门控
    _FLEET = _fleet8()

    @classmethod
    def _elig(cls, vtype: str, weight_kg: float) -> list[str]:
        """类型+载重合格车队 (CP-SAT 候选的物理全集)。"""
        return [v.vid for v in cls._FLEET
                if v.vtype == vtype and v.capacity_kg + 1e-12 >= weight_kg]

    def _maybe_reassign(self, spec: Spec) -> None:
        """选定车忙而同类合格车空闲 → 运行时改派 (决策事件可审计)。"""
        if self.mr.jobs.get(spec.vid) is None or len(spec.elig) < 2:
            return
        busy = {v: self._real_busy_s(v) for v in spec.elig
                if v != spec.vid and self.mr.jobs.get(v) is None
                and v not in self._repos}   # 预置在途车不可即时接管
        if not busy:
            return
        alt = min(busy, key=lambda v: (busy[v], v))
        old = spec.vid
        spec.vid = alt
        self._on_dec({
            "type": "TaskReassigned", "task_id": spec.task_id,
            "ts_s": round(float(self.cell.data.time), 2),
            "leg": spec.kind, "from": old, "to": alt,
            "candidates": list(spec.elig),
            "chosen": {spec.kind: alt},
            "reason": (f"chosen {old} occupied ({self.busy_s.get(old, 0):.0f}s busy), "
                       f"idle capable {alt} (least utilized) takes over"),
            "solver_status": "RUNTIME_BALANCE", "solve_ms": 0.0,
            "makespan_ticks": None})

    def _ledger_loc(self) -> dict:
        return dict(self.mr.ledger.loc)              # C 级原子拷贝

    def _stand_holders(self) -> tuple[set[str], int]:
        """(被链持有的交接台集合, 其中出库链持有数)。"""
        held, out_n = set(), 0
        for c in self.chains:
            if c.state in ("pending", "active"):
                s = c.reserved.get("stand")
                if s:
                    held.add(s)
                    if c.kind == "OUTBOUND":
                        out_n += 1
        return held, out_n

    def _gate(self, spec: Spec, c: Chain, loc: dict) -> bool:
        """发射条件: 车辆空闲 (忙则尝试改派) + 目标资源空闲。"""
        self._maybe_reassign(spec)
        if self.mr.jobs.get(spec.vid) is not None:
            return False
        if spec.kind == "pick":
            slot = spec.task["slot"]
            pn_now = self._pallet_at(loc, f"slot:{slot}")
            if pn_now != c.pallet:
                if pn_now is None:
                    # detfix2: 补货开单即计划占位 (demand._place), 物理托盘
                    # 却要到 putaway 落位才存在 —— 同 SKU 后续出库单先于
                    # 落位发射时按名查账必空 (实测 live 5f82cc59: t=312
                    # ORD-000006 R23-S1 / t=557 ORD-000013 R31-S2
                    # chain_failed「无托盘可取」, 同刻补货链仍 pending)。
                    # 同货位在途补货链未死 → 等落位再发射; 链死了才判死。
                    inbound = any(
                        cc is not c and cc.kind == "REPLENISH"
                        and cc.state in ("pending", "active")
                        and len(cc.specs) > 1
                        and cc.specs[1].task.get("slot") == slot
                        for cc in self.chains)
                    if inbound:
                        return False
                    c.state = "failed"
                    self._feed(spec.vid, "chain_failed",
                               f"{c.task_id}: 货位 {slot} 无托盘可取")
                    return False
                # detfix2: 占位者与本链主角异名 —— 补货回填的是交付台
                # 旧托 (名字带原货位, 如 pallet_R23_S4_L1 坐 slot:R32-S1),
                # 按 slot 命名推断的 c.pallet 永不匹配。物理即真值: 重绑
                # 链主角与作业对象 (mission 按 task["pallet"] 闭环作业)。
                c.pallet = pn_now
                spec.task["pallet"] = pn_now
            held, out_n = self._stand_holders()
            stand = spec.task["stand"]
            if (stand in held or self._pallet_at(loc, f"stand:{stand}")
                    or (c.kind == "OUTBOUND" and out_n >= MAX_OUT_STANDS)):
                return False
            c.reserved["stand"] = stand
            return True
        if spec.kind == "haul":
            dock = spec.task["dock"]
            if (dock in c.reserved.values()
                    or self._pallet_at(loc, f"dock:{dock}")
                    or any(dock == cc.reserved.get("dock")
                           for cc in self.chains
                           if cc is not c and cc.state in ("pending", "active"))):
                return False
            c.reserved["dock"] = dock
            return True
        if spec.kind == "fetch":
            # 源: 有托盘且未被他链预约的交付台 (FIFO); 目标: 空交接台
            held, _ = self._stand_holders()
            stand = spec.task["stand"]
            if stand in held or self._pallet_at(loc, f"stand:{stand}"):
                return False
            taken = {cc.reserved.get("dock") for cc in self.chains
                     if cc is not c and cc.state in ("pending", "active")}
            cands = [(t, d) for d, t in self._dock_loaded.items()
                     if self._pallet_at(loc, f"dock:{d}") and d not in taken]
            if not cands:
                return False
            _, dock = min(cands)
            spec.task["dock"] = dock
            c.pallet = self._pallet_at(loc, f"dock:{dock}")
            c.reserved["dock"] = dock
            c.reserved["stand"] = stand
            return True
        if spec.kind == "putaway":
            slot = spec.task["slot"]
            if any(k == f"slot:{slot}" for k in loc.values()) \
                    or slot in c.reserved.values():
                c.state = "failed"
                self._feed(spec.vid, "chain_failed",
                           f"{c.task_id}: 目标货位 {slot} 非空")
                return False
            c.reserved["slot"] = slot
            return True
        return False

    @staticmethod
    def _pallet_at(loc: dict, where: str) -> str | None:
        for p, w in loc.items():
            if w == where:
                return p
        return None

    # ------------------------------------------------- 数值发散看门狗
    def _guard_divergence(self) -> bool:
        """24h continuous 自恢复: qacc 非有限 → 隔离现场并重建世界。

        多体接触在极端并发下仍可能数值发散 (实测 t=1023 一例, DOF 属某
        托盘自由体)。发散后所有运动控制失效, 拖下去只会累积废账 ——
        故: 留痕 (事件+叙述) → 退役本线程 → 重建 (身份/账本/物理全新,
        物理修复到位后此类事件应趋于零)。"""
        d = self.cell.data
        try:
            bad = not bool(np.isfinite(d.qacc).all())
        except Exception:                          # noqa: BLE001
            return False
        if not bad:
            return False
        t = float(d.time)
        self._feed("SYS", "divergence",
                   f"QACC 非有限 @ t={t:.2f}s — 隔离并重建 (自恢复)")
        self.narrate(f"检测到数值发散 (t={t:.1f}s): 按 24h continuous "
                     "自恢复策略隔离现场并重建世界 (账本重置, 事件留痕)。")
        threading.Thread(target=self._recover_after_divergence,
                         daemon=True).start()
        return True

    def _recover_after_divergence(self) -> None:
        time.sleep(0.5)
        self.mode = "failed"                       # start() 见 failed → 重建
        self.start()

    # ------------------------------------------------------------ 主循环
    def _mission_loop(self, gen: int = 0) -> None:
        mr, d = self.mr, self.cell.data
        stall = 0
        try:
            while self.mode in ("running", "paused") and self._gen == gen:
                while self.mode == "paused":
                    time.sleep(0.05)
                if self.mode != "running":
                    break
                self._pump_demand()
                self._issue_specs()
                mr._advance_waiting()
                idle = mr.runner.run_until_any_idle(d.time + 2400.0)
                for vid in idle:
                    mr._on_queue_empty(vid)
                self._advance_repos(idle)
                self._repos_pump()
                self._repos_watchdog()
                t_now = float(d.time)
                if t_now - self._repos_tick_t >= REPOS_TICK_S:
                    self._repos_tick_t = t_now
                    self._maybe_reposition(t_now)
                self._collect_specs()
                self._acc_util()
                self._sample_util()
                if self._guard_divergence():
                    return                         # 线程退役, 守护线程重建
                moving = mr.runner.busy()
                waiting = [v for v, j in dict(mr.jobs).items()
                           if j is not None and j.mode == "wait"]
                n_work = sum(1 for c in self.chains
                             if c.state in ("pending", "active"))
                if idle or moving:
                    stall = 0
                    if moving:
                        self._last_progress = time.monotonic()  # 活性: 有车在动
                elif n_work:
                    # 有任务链但全静止: 等待者按节拍重试 claim (锁矩阵防死锁)
                    stall += 1
                    time.sleep(0.02)                # 让出 CPU
                    if stall > 250:                 # ~5s+ 无任何进展
                        self.stall_s += 5.0
                        self._feed("SYS", "stall",
                                   f"无进展等待 waiting={waiting}")
                        stall = 0
                else:
                    # 无执行链 (订单可在缓冲队列): 推进仿真钟 — 否则时钟
                    # 与需求/批缓冲互锁 (无车移动 → 无时间 → 不出单/不满
                    # 批年龄 → 永不派发)。空场步进对停靠车辆无害。
                    self.cell.step(100)             # 0.1s 仿真
                    stall = 0
                self._pace()
        except Exception as e:                       # noqa: BLE001
            self.mode = "failed"
            tail = traceback.format_exc().strip().splitlines()[-3:]
            self.narrate(f"mission 线程异常: {e!r} — " + " | ".join(tail))
            self._feed("SYS", "fatal", repr(e))

    def _issue_specs(self) -> None:
        loc = self._ledger_loc()
        for c in self.chains:
            if c.state in ("done", "failed"):
                continue
            done = {s.uid: s.state == "done" for s in c.specs}
            for s in c.specs:
                if s.state != "pending":
                    continue
                if not all(done.get(u) for u in s.deps):
                    continue
                if s.vid in self._repos and not self._repos_yield(s.vid):
                    continue    # 预置在途: 已标记让位, 当前边走完再 gate
                if self._gate(s, c, loc):
                    s.state = "active"
                    s.t_start = float(self.cell.data.time)
                    s.err_base = len(self.mr.op_errors)
                    self._repos_parked.pop(s.vid, None)   # 驻位车接真任务
                    self.mr.jobs[s.vid] = Job(vid=s.vid, task=s.task)
                    self._feed(s.vid, "dispatch",
                               f"{c.task_id} {s.label} → {s.vid}")
                    loc = self._ledger_loc()         # 预约已改账面
            if any(s.state == "active" for s in c.specs):
                c.state = "active"

    def _collect_specs(self) -> None:
        mr = self.mr
        for c in self.chains:
            if c.state in ("done", "failed"):
                continue
            for s in c.specs:
                if s.state == "active" and mr.jobs.get(s.vid) is None:
                    s.t_end = float(self.cell.data.time)
                    errs = [e for e in mr.op_errors[s.err_base:]
                            if e.startswith(f"{s.vid}:")]
                    s.state = "failed" if errs else "done"
                    self.tasks_done += 1 if s.state == "done" else 0
                    self.tasks_failed += 1 if s.state == "failed" else 0
                    self.done_marks.append((float(self.cell.data.time),
                                            s.state))
                    if errs:
                        self._feed(s.vid, "task_failed",
                                   f"{c.task_id} {s.label}: {errs[-1]}")
            if any(s.state == "failed" for s in c.specs):
                c.state = "failed"
                self._release(c)
                self.narrate(f"订单 {c.task_id} ({c.desc}) 物理失败, "
                             "不提交台账 — 详见事件流。")
            elif all(s.state == "done" for s in c.specs):
                ok, why = self._commit_chain(c)
                c.state = "done" if ok else "failed"
                self._release(c)
                self.orders_done += 1 if ok else 0
                if ok:
                    self._last_progress = time.monotonic()
                else:
                    self.narrate(f"订单 {c.task_id} ({c.desc}) 提交被拒 "
                                 f"(B1 证据不足: {why}) — 不改台账。")

    def _release(self, c: Chain) -> None:
        c.reserved.clear()

    def _commit_chain(self, c: Chain) -> tuple[bool, str]:
        """B1: 物理证据 (台账+位姿校验) 成立才 dispatcher.commit。"""
        L = self.cell.layout
        loc = self._ledger_loc()
        pn = c.pallet
        if c.kind == "OUTBOUND":
            dock = c.specs[1].task["dock"]
            where = f"dock:{dock}"
            ex, ey, ez = (*L.stand_pose(dock)[:2], L.p["stand_top_z"])
        else:
            slot = c.specs[1].task["slot"]
            level = c.specs[1].task["level"]
            where = f"slot:{slot}"
            ex, ey, _, ez = L.slot_pose(f"{slot}-L{level}")
        if loc.get(pn) != where:
            return False, f"台账位置 {loc.get(pn)} ≠ {where}"
        try:
            pos = self.cell.pallet_pose(pn)[0]
        except KeyError:
            return False, "托盘不在物理场景中"
        err = math.hypot(pos[0] - ex, pos[1] - ey)
        if err > 0.02 or abs(pos[2] - ez) > 0.02:
            return False, f"位姿校验失败 dxy={err:.3f} dz={pos[2] - ez:.3f}"
        t = float(self.cell.data.time)
        try:
            self.dispatcher.commit(
                c.task_id, ts_s=t,
                evidence={"ledger_loc": where, "pose_err_m": round(err, 5),
                          "sku": self.sku_of.get(pn, "?"),
                          "sim_t": round(t, 2)})
        except (ValueError, KeyError) as e:
            return False, f"提交被拒: {e}"
        if c.kind == "REPLENISH":
            # 补货落架: 计划库存同步到物理事实 (回库托盘的真实 SKU);
            # 交付台腾空 (FIFO 记账移除)
            dst = next(o.dst for o in self.demand.orders
                       if o.order_id == c.task_id)
            self.demand._place(dst, self.sku_of.get(pn))
            self._dock_loaded.pop(c.specs[0].task["dock"], None)
        else:
            self._dock_loaded[c.specs[1].task["dock"]] = t
        self._feed(c.specs[-1].vid, "order_committed",
                   f"{c.task_id} {c.desc} ✓")
        return True, ""

    # ------------------------------------------------------------ 指标
    def _acc_util(self) -> None:
        d = self.cell.data
        dt = float(d.time) - self._util_t
        self._util_t = float(d.time)
        if dt <= 0:
            return
        for vid in self.cell.veh:
            if self.mr.jobs.get(vid) is not None or vid in self._repos:
                # 预置移动计入 busy (被派发的生产性移动); 同时单列预置
                # 时长 —— 派遣侧 (CP-SAT 均衡偏置/运行时改派) 只看真实
                # 作业负载, 预置不与真任务抢"低负载"资格。
                self.busy_s[vid] = self.busy_s.get(vid, 0.0) + dt
                if vid in self._repos:
                    self._repos_busy_s[vid] = \
                        self._repos_busy_s.get(vid, 0.0) + dt

    def _real_busy_s(self, vid: str) -> float:
        """真实作业 busy (总 busy − 预置空驶): 派遣侧负载口径。"""
        return max(0.0, self.busy_s.get(vid, 0.0)
                   - self._repos_busy_s.get(vid, 0.0))

    def _sample_util(self) -> None:
        """每 60s 仿真采样利用率 → 5 样本滑动窗; 驱动需求倍率反馈。"""
        t = float(self.cell.data.time)
        if t - self._util_sample_t < UTIL_SAMPLE_S:
            return
        dt = t - self._util_sample_t
        self._util_sample_t = t
        per = {vid: round(min(1.0, (b - self._busy_snap.get(vid, 0.0)) / dt), 3)
               for vid, b in self.busy_s.items()}
        self._busy_snap = dict(self.busy_s)
        self._util_win.append({"t": round(t, 1), "per": per,
                               "fleet": round(sum(per.values())
                                              / max(len(per), 1), 3)})
        self._tune_demand(t)
        self._maybe_reposition(t)

    DEMAND_MULT_MAX = 3.0     # 加压上限: 实测 30min 压测 17.9× 雪崩
                              # (chains 20/24 顶满 → 批次=1 → 队列淹死,
                              #  拥堵剐蹭 32 条穿透采样; 3× 已足够触底)
    def _tune_demand(self, t: float) -> None:
        """利用率反馈: <60% 加压 ×1.3 (≤3×); >85% 减压 ×0.85 (≥1×)。

        队列感知: chains_open ≥ 80% 上限 视为饱和, 只减不增 ——
        利用率低但队列满 = 派发/执行瓶颈, 加压只会淹死队列 (实测)。
        """
        if not self._demand_on or not self._util_win:
            return
        util = self._util_win[-1]["fleet"]
        old = float(self.multiplier)
        new = old
        why = ""
        n_open = sum(1 for c in self.chains
                     if c.state in ("pending", "active"))
        saturated = n_open >= 0.8 * MAX_CHAINS
        if util < 0.60 and not saturated:
            new, why = min(old * 1.3, self.DEMAND_MULT_MAX), \
                "util<60% → ×1.3 加压"
        elif util > 0.85 or (saturated and old > 1.0):
            new, why = max(old * 0.85, 1.0), \
                ("util>85%" if util > 0.85 else f"chains {n_open}/{MAX_CHAINS} 饱和") \
                + " → ×0.85 防过载"
        if abs(new - old) < 1e-9:
            return
        self.multiplier = round(new, 3)
        self.demand.global_multiplier = self.multiplier
        self._on_dec({"type": "DemandTuned", "ts_s": round(t, 2),
                      "util_window": util, "old_multiplier": old,
                      "new_multiplier": self.multiplier,
                      "reason": f"fleet util {util:.0%}: {why}",
                      "solver_status": "FEEDBACK", "solve_ms": 0.0,
                      "makespan_ticks": None})
        self.narrate(f"需求倍率自适应: {old:g}× → {self.multiplier:g}× "
                     f"(滑动窗利用率 {util:.0%}, {why})。")

    def _pace(self) -> None:
        if self.speed <= 0:
            return
        sim0, wall0 = self._pace_base
        lead = (float(self.cell.data.time) - sim0) \
            - (time.monotonic() - wall0) * self.speed
        if lead > 0:
            time.sleep(min(lead / self.speed, 0.25))

    def metrics(self) -> dict:
        win = list(self._util_win)
        per_w = {v: round(sum(s["per"].get(v, 0.0) for s in win) / len(win), 3)
                 for v in self.cell.veh} if win else {}
        util_window = {"fleet": round(sum(s["fleet"] for s in win) / len(win), 3)
                       if win else 0.0, "per_vehicle": per_w,
                       "n_samples": len(win),
                       "window_s": UTIL_WIN * int(UTIL_SAMPLE_S),
                       "multiplier": self.multiplier}
        t = float(self.cell.data.time)
        # 吞吐: 拖尾 600s 仿真窗内完成任务数 → 任务/仿真小时 (只读扫描)
        marks = list(self.done_marks)
        i0 = 0
        while i0 < len(marks) and t - marks[i0][0] > 600.0:
            i0 += 1
        rate = (len(marks) - i0) / 600.0 * 3600.0
        elapsed = max(t - 0.5, 1e-6)
        stats = {v: dict(s) for v, s in dict(self.mr.traffic.stats).items()}
        waits = sum(s.get("waits", 0) for s in stats.values())
        wait_s = sum(s.get("wait_s", 0.0) for s in stats.values())
        violations = [v for v in self.mr.runner.audit["violations"]
                      if str(v).startswith("穿透")]
        chains = self.chains
        n_pending = sum(1 for c in chains if c.state in ("pending", "active"))
        jobs = dict(self.mr.jobs)
        return {
            "sim_t": round(t, 1),
            "wall_progress_s": round(time.monotonic() - self._last_progress, 1),
            "throughput_tasks_h": round(rate, 1),
            "orders_done": self.orders_done,
            "tasks_done": self.tasks_done,
            "tasks_failed": self.tasks_failed,
            "chains_open": n_pending,
            "queue_depth": len(self._unsubmitted),
            "utilization": {v: round(self.busy_s.get(v, 0.0) / elapsed, 3)
                            for v in self.cell.veh},
            "busy_s": {v: round(self.busy_s.get(v, 0.0), 1)
                       for v in self.cell.veh},
            "lock_waits": waits,
            "lock_wait_s": round(wait_s, 1),
            "collisions": len(violations),
            "max_pen_m": round(float(self.mr.runner.audit["max_pen"]), 8),
            "decisions_total": self.dec_total,
            "feed_depth": len(self.feeds),
            "decision_depth": len(self.decisions),
            "op_errors": len(self.mr.op_errors),
            "stall_s": round(self.stall_s, 1),
            "util_window": util_window,
            "jobs_active": sum(1 for j in jobs.values() if j is not None),
            # 预置/背压观测 (利用率专项: 判据口径数据, 不改语义)
            "reposition_moves": self._repos_started,
            "repositions_active": len(self._repos),
            "repositions_parked": len(self._repos_parked),
            "stands_held_out": self._stand_holders()[1],
            "max_out_stands": MAX_OUT_STANDS,
            "chains_cap": MAX_CHAINS,
            "backlog_cap": BACKLOG_CAP,
        }

    # ------------------------------------------------------------ 状态快照
    def layout_payload(self) -> dict:
        L = self.cell.layout
        p = L.p
        slots = []
        for s in L.slot_ids():
            x, y, _, z = L.slot_pose(s)
            base, lvl = s.rsplit("-L", 1)
            slots.append({"id": s, "x": round(x, 4), "y": round(y, 4),
                          "z": z, "level": int(lvl)})
        return {
            "field": {"w": round(L.field_w, 3), "h": round(L.field_h, 3)},
            "matrix": {"w": round(L.matrix_w, 3), "h": round(L.matrix_h, 3)},
            "levels_z": p["levels_z"],
            "group_w": round(L.group_w, 4), "group_h": round(L.group_h, 4),
            "groups": [{"r": r, "c": c, "x": round(L.group_center(r, c)[0], 4),
                        "y": round(L.group_center(r, c)[1], 4)}
                       for r in range(1, 5) for c in range(1, 5)],
            "col_xs": [round(v, 4) for v in L.col_aisle_xs()],
            "row_ys": [round(v, 4) for v in L.row_aisle_ys()],
            "stands": [{"id": sid, "x": round(x, 4), "y": round(y, 4)}
                       for sid, (x, y, _) in L.handoff_poses().items()],
            "docks": [{"id": did, "x": round(x, 4), "y": round(y, 4)}
                      for did, (x, y, _) in L.dock_poses().items()],
            "parks": {vid: [round(px, 3), round(py, 3)]
                      for vid, (px, py, _) in p["park"].items()},
            "stand_top_z": p["stand_top_z"],
            "slots": slots,
            "vehicles": [{"vid": v,
                          "type": "forklift" if v in self.cell.forklift_ids
                          else "agv",
                          "level": p["vehicle_class"][v]}
                         for v in self.cell.veh],
            "k": {"fork_top0": 0.065, "transport_z": K["transport_z"],
                  "pin_travel": K["pin_travel"]},
        }

    def state(self, dec_cur: int = 0, feed_cur: int = 0,
              chat_cur: int = 0) -> dict:
        d = self.cell.data
        jobs = dict(self.mr.jobs)
        vehicles = []
        for vid, view in self.cell.veh.items():
            x, y = view.xy()
            job = jobs.get(vid)
            repos = self._repos.get(vid)
            parked = vid in self._repos_parked
            v = {"vid": vid, "x": round(x, 4), "y": round(y, 4),
                 "yaw": round(view.q("yaw"), 4),
                 "level": self.cell.layout.p["vehicle_class"][vid],
                 "type": "forklift" if vid in self.cell.forklift_ids
                         else "agv",
                 "busy": job is not None or repos is not None,
                 "task": (job.task["kind"] if job
                          else ("reposition" if repos
                                else ("staged" if parked else ""))),
                 "arg": (_arg_of(job) if job
                         else (f"→{repos['goal']}" if repos else ""))}
            if vid in self.cell.forklifts:
                v["lift"] = round(view.q("lift"), 4)
            else:
                v["pin"] = round(view.q("pin"), 4)
            vehicles.append(v)
        pallets = []
        for name, pv in self.cell.pallets.items():
            pos = d.xpos[pv.bid]
            pallets.append({"n": name, "x": round(float(pos[0]), 4),
                            "y": round(float(pos[1]), 4),
                            "z": round(float(pos[2]), 4)})
        loc = self._ledger_loc()
        occ = {"slots": sorted(k[5:] for k in loc.values()
                               if k.startswith("slot:")),
               "stands": sorted(k[6:] for k in loc.values()
                                if k.startswith("stand:")),
               "docks": sorted(k[5:] for k in loc.values()
                               if k.startswith("dock:"))}
        decs = [e for e in self.decisions if e["seq"] > dec_cur]
        feeds = [e for e in self.feeds if e["seq"] > feed_cur]
        chats = [m for m in self.chat if m["seq"] > chat_cur]
        m = self.metrics()
        st = self.status()
        return {**st, "paused": self.mode == "paused",
                "vehicles": vehicles, "pallets": pallets, "occ": occ,
                "decisions": decs, "n_dec": self.dec_total,
                "feeds": feeds, "n_feed": self.feed_total,
                "chat": chats, "chat_seq": self.chat_seq,
                "metrics": m, "cones": self.cone_xy(),
                "inventory": {"stock_pallets": sum(
                    1 for v in loc.values() if v.startswith("slot:")),
                    "sku_kinds": len({s for s in self.sku_of.values()}),
                    "free_slots": self.demand.free_slot_count()},
                "chains": [{"id": c.task_id, "kind": c.kind, "desc": c.desc,
                            "state": c.state,
                            "specs": [{"k": s.kind, "vid": s.vid,
                                       "state": s.state,
                                       "label": s.label}
                                      for s in c.specs]}
                           for c in self.chains[-24:]]}


def _arg_of(job) -> str:
    t = job.task
    return str(t.get("slot") or t.get("stand") or t.get("dock") or "")


# ================================================================ chat
_NUM = re.compile(r"(\d+)")


def _llm_reply(text: str) -> str | None:
    """可选 LLM 叙述 (TWIN_LLM_BASE 指向 OpenAI 兼容端点如 oMLX)。"""
    base = os.environ.get("TWIN_LLM_BASE", "")
    if not base:
        return None
    model = os.environ.get("TWIN_LLM_MODEL", "local")
    sysp = ("你是 8 车矩阵仓库 (128 架/384 货位) 的调度 Agent。可用指令:"
            "开始持续运行/停/暂停/恢复/速度 N/下 N 单/注入锥桶 [col:i|row:j]/"
            "清锥桶/库存/状态。回答要简短, 不要编造数值, 涉及实时数据时"
            "提示用户用「状态」「库存」查询。")
    try:
        import urllib.request
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps({"model": model, "messages": [
                {"role": "system", "content": sysp},
                {"role": "user", "content": text}], "max_tokens": 220,
                "temperature": 0.3}).encode(),
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception:                              # noqa: BLE001
        return None


def chat_handle(text: str, svc: MatrixService) -> dict:
    """确定性规则内核: 意图 → 服务命令; 未命中且配置了 LLM 则转述。"""
    t = text.strip()
    low = t.lower()
    tools = []
    reply = None

    def tool(name, arg=None):
        tools.append({"tool": name, "arg": arg})

    if re.search(r"开始|启动|持续运行|跑起来|^start$", low):
        reply = json.dumps(svc.start(), ensure_ascii=False)
        tool("start")
    elif re.search(r"^停$|停止|^stop$", low):
        reply = json.dumps(svc.stop(), ensure_ascii=False)
        tool("stop")
    elif re.search(r"暂停|pause", low):
        reply = json.dumps(svc.pause(), ensure_ascii=False)
        tool("pause")
    elif re.search(r"恢复|继续|resume", low) and svc.mode == "paused":
        reply = json.dumps(svc.resume(), ensure_ascii=False)
        tool("resume")
    elif re.search(r"速度|倍速|speed", low):
        m = _NUM.search(low)
        v = float(m.group(1)) if m else 1.0
        if "max" in low or "不限" in low:
            v = 0.0
        reply = json.dumps(svc.set_speed(v), ensure_ascii=False)
        tool("speed", v)
    elif re.search(r"下\s*\d+\s*(张个)?单?|下\s*\d+\s*个?订单|下单|订单|来.*单|order", low):
        m = _NUM.search(low)
        n = int(m.group(1)) if m else 10
        reply = json.dumps(svc.manual_orders(n), ensure_ascii=False)
        tool("manual_orders", n)
    elif re.search(r"注入|放.*锥桶|inject", low):
        m = re.search(r"(col|row)\s*[:：]?\s*(\d)", low)
        spec = f"{m.group(1)}:{m.group(2)}" if m else None
        reply = json.dumps(svc.inject_cone(spec), ensure_ascii=False)
        tool("inject_cone", spec)
    elif re.search(r"清除|移除|撤|拿走|清.*锥桶|clear", low) \
            and "锥" in t:
        reply = json.dumps(svc.clear_cones(), ensure_ascii=False)
        tool("clear_cones")
    elif re.search(r"库存|inventory|在库", low):
        inv = svc.state()["inventory"]
        reply = (f"当前计划库存: 货架托盘 {inv['stock_pallets']} / "
                 f"SKU 种类 {inv['sku_kinds']} / 空闲货位 "
                 f"{inv['free_slots']} (384 总货位)。"
                 "台账以物理校验后的提交为准。")
        tool("inventory")
    elif re.search(r"技能|skill|能力", low):
        from sim import robot_skills as _rs
        cat = _rs.catalog()
        reply = ("可用预验证技能 (agent 只选技能+传参, 物理序列由技能库"
                 "展开): " + "; ".join(
                     f"{s['name']} — {s['doc']}" for s in cat))
        tool("skills")
    elif re.search(r"状态|指标|metric|进度", low):
        m = svc.metrics()
        reply = (f"mode={svc.mode} sim_t={m['sim_t']}s | "
                 f"吞吐 {m['throughput_tasks_h']} 任务/h | "
                 f"订单完成 {m['orders_done']} (失败 {m['tasks_failed']}) | "
                 f"锁等待 {m['lock_waits']} 次/{m['lock_wait_s']}s | "
                 f"碰撞 {m['collisions']} (max_pen "
                 f"{m['max_pen_m'] * 1000:.2f}mm) | 在飞链 "
                 f"{m['chains_open']} 排队 {m['queue_depth']}")
        tool("metrics")
    elif re.search(r"复位|重置|reset", low):
        svc.mode = "stopped"
        svc.narrate("已请求复位。")
        reply = json.dumps(svc.start(), ensure_ascii=False)
        tool("reset")
    if reply is None:
        llm = _llm_reply(t)
        reply = llm or ("指令未识别。可用: 开始持续运行 / 停 / 暂停 / 恢复 / "
                        "速度 2 / 下 10 单 / 注入锥桶 col:2 / 清锥桶 / "
                        "库存 / 状态")
    return {"reply": reply, "tools": tools,
            "engine": "llm" if "TWIN_LLM_BASE" in os.environ else "rules"}


# ================================================================ app
SVC = MatrixService()
app = FastAPI(title="W3b Matrix World")


@app.middleware("http")
async def _nocache(request, call_next):
    resp = await call_next(request)
    if request.url.path.endswith((".html", ".js", "/")):
        resp.headers["cache-control"] = "no-store"
    return resp


@app.get("/")
async def index():
    return FileResponse(WEB / "matrix.html")


@app.get("/api/layout")
async def layout():
    return SVC.layout_payload()


@app.post("/api/chat")
async def chat(body: dict):
    text = str(body.get("text", ""))
    SVC.narrate(text, role="user")
    try:
        out = chat_handle(text, SVC)
    except Exception as e:                     # noqa: BLE001
        import traceback as _tb
        out = {"reply": f"命令执行异常: {e!r} — " +
                        " | ".join(_tb.format_exc().strip().splitlines()[-2:]),
               "tools": [], "engine": "error"}
    SVC.narrate(out["reply"])
    return out


@app.get("/api/state")
async def state():
    return SVC.state()


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    dec_cur, feed_cur, chat_cur = 0, 0, SVC.chat_seq
    try:
        while True:
            st = SVC.state(dec_cur, feed_cur, chat_cur)
            dec_cur = st["n_dec"]
            feed_cur = st["n_feed"]
            if st["chat"]:
                chat_cur = st["chat"][-1]["seq"]
            await ws.send_text(json.dumps(st, ensure_ascii=False))
            await asyncio.sleep(1.0 / PUB_HZ)
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=WEB), name="web")

if __name__ == "__main__":
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="墙钟配速倍率; 0 = 不限速")
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    SVC = MatrixService(seed=a.seed, multiplier=a.multiplier,
                        speed=a.speed)
    SVC.start()          # 开机即跑 (此前 start 只挂在聊天指令上,
                         # 服务重启后 mission 线程不创建 → sim 冻结)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
