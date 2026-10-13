"""M2-W2b 任务调度层 —— CP-SAT 任务分配 + 回退贪心 + 决策事件流 (设计 §19.4)。

边界 (§19.4): CP-SAT 只处理任务顺序、开始时间、能力和离散资源时间窗,
不求连续轨迹; 时间单位 = 100ms 整数 tick, 持续时间向上取整; 求解状态:
- OPTIMAL / FEASIBLE → 读方案;
- UNKNOWN (超时)     → 回退最小代价贪心 (距离最近的合格车), 仍可行;
- INFEASIBLE         → 明确失败 (不声称已优化);
- MODEL_INVALID      → 抛 DispatchError, 不允许按「优化失败可忽略」继续。

任务生命周期 (术语对齐 openTCS TransportOrder):
    PENDING   ↔ openTCS DISPATCHABLE
    ASSIGNED  ↔ openTCS BEING_PROCESSED   (本层: 已排程, 各腿含开始 tick)
    COMMITTED ↔ openTCS FINISHED 的**严格化**: 必须携带物理证据 (接触/
                位姿观测) 才原子提交, 对齐 services/scheduler 七段事务语义
                (FAILED/UNKNOWN 不推进, 证据先于提交)。相对工业标准的
                差异化贡献 —— 决策面板/注释按此展示。

与 W2a 的接口契约 (鸭子类型注入, 本层不调用, 由集成执行器消费):
- navigation.plan(vehicle_xy, from, to, lock_penalty) -> list[Segment]
- traffic.LockMatrix.acquire(path_segments, vid, ordered=True) / release(vid)

车辆 8 辆扩编为**打桩常量**: W1 物理层 MJCF 当前仅 F01/F02/A01/A02,
F03/F04/A03/A04 与等级-速度-载重参数化未进 warehouse_cell (红线: 不改
W1 语义), 以 FLEET_STUB 代替; 扩编 MJCF 属后续波次。

评估口径 (对齐 mmmorgandata/agv-warehouse-scheduling):
Dispatcher.metrics() 提供 makespan / 总等待 / 各等级车辆利用率。
"""
from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from sim.demand import DEFAULT_CATALOG_PATH, INBOUND_NODE, OrderKind, SkuInfo, \
    load_catalog

# ---------------------------------------------------------------- 常量单源
TICK_S = 0.1                       # §19.4: 100ms 整数 tick

#: 车辆等级: 载重上限 + 满载巡航速度 (能力强 ⇋ 速度慢)
VEHICLE_LEVELS: dict[str, dict] = {
    "L1": {"capacity_kg": 0.08, "speed_mps": 0.35},
    "L2": {"capacity_kg": 0.15, "speed_mps": 0.25},
    "L3": {"capacity_kg": 0.30, "speed_mps": 0.15},
}

#: 托盘皮重 (configs/design.yaml pallet.tare_mass_kg 设计初值 0.045; W1 实测
#: 0.048 ≤ 0.050 上限, 差异远小于载重档位间隔, 不影响能力判定)
PALLET_MASS_KG = 0.045

#: 固定作业时长 (s): 取/放/对中; 货位层越高越慢
RACK_PICK_BASE_S = 5.0             # 货位端取托 (含门架举升基准)
RACK_PICK_PER_LEVEL_S = 2.0        # 每升高一层附加
STAND_PLACE_S = 3.0                # 交接台/交付台放托
STAND_PICK_S = 2.0                 # 交接台取托 (对中钻入+顶升)
HANDOFF_DWELL_TICKS = 2            # 交接链最小交接间隔 (落托→可接管)

#: 求解预算 (§19.4 初值 1s; 本层单批规模小, 默认 0.5s 保 24h 连续模式)
SOLVE_TIMEOUT_S = 0.5
_MAKESPAN_WEIGHT = 1000            # 字典序: makespan 优先, Σstart 次级 tie-break
#: 负载均衡权重 (利用率专项): 目标追加 W·max_v(load_v), load_v = 批前累计
#: 偏置 + 本批该车腿时长和 (tick)。取 2 的依据: makespan 权重 1000 ⇒ 1 tick
#: makespan ≈ 500 tick 峰值负载, 而单批内换车的峰值摆幅典型 ≪500 tick ——
#: 均衡只在 makespan/Σstart 平局带内起作用, 不撼动主目标; 同时足够压过
#: 个别腿 start 的微小差异, 让低累计负载的合格车 (L1 快车) 稳定分到轻货。
_BALANCE_WEIGHT = 2


@enum.unique
class TaskState(str, enum.Enum):
    PENDING = "PENDING"            # ↔ openTCS DISPATCHABLE
    ASSIGNED = "ASSIGNED"          # ↔ openTCS BEING_PROCESSED
    COMMITTED = "COMMITTED"        # 严格化 FINISHED: 物理证据后提交


class DispatchError(RuntimeError):
    """MODEL_INVALID 等不允许静默忽略的求解器故障 (§19.4)。"""


class InfeasibleAssignment(DispatchError):
    """CP-SAT 判定 INFEASIBLE (§19.4: 返回明确失败, 不回退不声称优化)。"""


# ---------------------------------------------------------------- 车队
@dataclass(frozen=True)
class Vehicle:
    vid: str
    vtype: str                     # "forklift" (货位端) | "agv" (干线段)
    level: str                     # L1 / L2 / L3
    park_xy: tuple[float, float]

    @property
    def capacity_kg(self) -> float:
        return VEHICLE_LEVELS[self.level]["capacity_kg"]

    @property
    def speed_mps(self) -> float:
        return VEHICLE_LEVELS[self.level]["speed_mps"]


#: W1 实体 4 辆 (park 取 LAYOUT["park"]); F03/F04/A03/A04 为扩编打桩位
FLEET_STUB: tuple[Vehicle, ...] = (
    Vehicle("F01", "forklift", "L1", (-2.60, -2.42)),
    Vehicle("F02", "forklift", "L2", (0.00, 1.08)),
    Vehicle("F03", "forklift", "L3", (-1.10, -2.42)),     # 打桩: 南泊位带
    Vehicle("F04", "forklift", "L3", (1.10, -2.42)),      # 打桩: 南泊位带
    Vehicle("A01", "agv", "L1", (-2.22, -2.85)),
    Vehicle("A02", "agv", "L1", (0.00, -2.85)),
    Vehicle("A03", "agv", "L2", (1.10, -2.85)),           # 打桩: 南带外侧
    Vehicle("A04", "agv", "L3", (2.22, -2.85)),           # 打桩: 南带外侧
)

#: 车辆类型 ↔ 任务腿类型 (类型匹配: 叉车=货位端操作, AGV=干线段)。
#: 既有 fork/trunk 为内核语义 (不改); 场景腿类型 (如 dc-twin 的 load)
#: 经 register_leg_type 注册进同一张表 —— 指派/时长逻辑只读表, 不分叉。
LEG_VEHICLE_TYPE = {"fork": "forklift", "trunk": "agv"}
_LEG_HANDLING: dict[str, Callable[["Leg"], float]] = {}


def register_leg_type(leg_type: str, *, vehicle_type: str,
                      handling_s: Callable[["Leg"], float] | float) -> None:
    """注册新腿类型: 合格载具类型 + 腿端固定作业时长 (标量或 fn(leg))。

    重注册拒绝 (内核 fork/trunk 同样不可覆盖); 合法名 = 非空无空白。
    """
    if not leg_type or leg_type.split() != [leg_type]:
        raise ValueError(f"腿类型非法 (非空无空白): {leg_type!r}")
    if leg_type in LEG_VEHICLE_TYPE:
        raise ValueError(f"腿类型已注册: {leg_type!r}")
    LEG_VEHICLE_TYPE[leg_type] = vehicle_type
    if callable(handling_s):
        _LEG_HANDLING[leg_type] = handling_s
    else:
        h = float(handling_s)
        _LEG_HANDLING[leg_type] = lambda leg, _h=h: _h


# ---------------------------------------------------------------- 任务
@dataclass(frozen=True)
class Leg:
    leg_id: str
    task_id: str
    leg_type: str                  # "fork" (货位端) | "trunk" (干线段)
    src: str
    dst: str
    dist_m: float
    level_z: float = 0.0           # fork 腿取托层高 (审计用)

    def handling_s(self) -> float:
        """腿端固定作业时长 (不含行驶)。"""
        if self.leg_type == "fork":
            lvl = max(0, round((self.level_z - 0.10) / 0.15))   # LAYOUT 层高派生
            return (RACK_PICK_BASE_S + RACK_PICK_PER_LEVEL_S * lvl
                    + STAND_PLACE_S)
        if self.leg_type in _LEG_HANDLING:                      # 注册腿类型
            return _LEG_HANDLING[self.leg_type](self)
        return STAND_PICK_S + STAND_PLACE_S


@dataclass
class TransportTask:
    """一张订单的运输任务: 货位端腿 + 干线腿 (交接链)。

    legs[0] 落到交接台 (出库: fork→H; 补货: trunk INBOUND→H),
    legs[1] 从交接台取走 (交接链先后: start(legs[1]) ≥ end(legs[0]) + dwell)。
    """

    task_id: str
    kind: OrderKind
    sku: str
    qty: int
    weight_kg: float               # 托盘 + qty 件 SKU (能力约束输入)
    release_s: float
    handoff: str                   # 交接台 H01..H04 (就近 x 通道)
    legs: tuple[Leg, Leg]
    state: TaskState = TaskState.PENDING
    assignment: dict[str, str] = field(default_factory=dict)    # leg_id -> vid
    schedule: dict[str, tuple[int, int]] = field(default_factory=dict)  # leg -> (start,dur) ticks
    evidence: dict | None = None

    @property
    def release_tick(self) -> int:
        return math.ceil(self.release_s / TICK_S)

    def eligible_vehicles(self, fleet: Iterable[Vehicle]) -> dict[str, list[Vehicle]]:
        """逐腿候选: 车辆类型匹配 + 等级载重 ≥ 货总重 (超载分配禁止)。"""
        out: dict[str, list[Vehicle]] = {}
        for l in self.legs:
            need = LEG_VEHICLE_TYPE[l.leg_type]
            out[l.leg_id] = [
                v for v in fleet
                if v.vtype == need and v.capacity_kg + 1e-12 >= self.weight_kg
            ]
        return out


def build_task(order, *, catalog: dict[str, SkuInfo],
               node_pos: Callable[[str], tuple[float, float, float]],
               handoffs: list[str]) -> TransportTask:
    """订单 → 运输任务 (两条腿, 经 x 通道就近交接台)。node_pos 含层高 z。"""
    w = PALLET_MASS_KG + catalog[order.sku].weight_kg * order.qty
    anchor = order.src if order.kind is OrderKind.OUTBOUND else order.dst
    ax = node_pos(anchor)[0]
    handoff = min(handoffs, key=lambda h: (abs(node_pos(h)[0] - ax), h))
    if order.kind is OrderKind.OUTBOUND:
        legs = (
            Leg(f"{order.order_id}:fork", order.order_id, "fork",
                order.src, handoff,
                _manhattan(node_pos(order.src), node_pos(handoff)),
                node_pos(order.src)[2]),
            Leg(f"{order.order_id}:trunk", order.order_id, "trunk",
                handoff, order.dst,
                _manhattan(node_pos(handoff), node_pos(order.dst))),
        )
    else:
        legs = (
            Leg(f"{order.order_id}:trunk", order.order_id, "trunk",
                order.src, handoff,
                _manhattan(node_pos(order.src), node_pos(handoff))),
            Leg(f"{order.order_id}:fork", order.order_id, "fork",
                handoff, order.dst,
                _manhattan(node_pos(handoff), node_pos(order.dst)),
                node_pos(order.dst)[2]),
        )
    return TransportTask(
        task_id=order.order_id, kind=order.kind, sku=order.sku, qty=order.qty,
        weight_kg=round(w, 6), release_s=order.t_s, handoff=handoff, legs=legs,
    )


def _manhattan(a, b) -> float:
    """通道约束近似: 曼哈顿距离 (车辆沿 x/y 通道直角行驶)。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------- 分配器
@dataclass
class DispatchMetrics:
    """评估口径 (对齐 mmmorgandata/agv-warehouse-scheduling)。"""

    makespan_ticks: int = 0
    total_wait_ticks: int = 0                  # Σ 任务首腿开始 − release
    utilization_by_level: dict = field(default_factory=dict)   # L1/L2/L3 → 0..1
    busy_ticks_by_vehicle: dict = field(default_factory=dict)


class Dispatcher:
    """批次分配器: submit 一批 PENDING 任务 → CP-SAT 排程 (超时回退贪心)。

    - 决策事件: 每任务一条 TaskAssigned / TaskUnassigned; commit 时一条
      TaskCommitted。事件为纯 dict, 直接喂 W3 决策面板。
    - navigator / lockboard: W2a 注入点 (契约见模块 docstring); 本层不调用,
      由集成执行器经 iter_schedule() 逐腿消费。
    """

    def __init__(
        self,
        *,
        fleet: tuple[Vehicle, ...] = FLEET_STUB,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        solve_timeout_s: float = SOLVE_TIMEOUT_S,
        random_seed: int = 0,
        num_workers: int = 1,          # 固定 1: 同批同解, 决策可复现
        event_sink: Callable[[dict], None] | None = None,
        navigator=None,
        lockboard=None,
        load_offset_provider: Callable[[], dict[str, int]] | None = None,
    ):
        #: 批前累计负载 {vid: busy tick} —— CP-SAT 均衡项的跨批记忆
        #: (集成层注入; None = 纯批内均衡)。
        self.load_offset_provider = load_offset_provider
        self.fleet = tuple(fleet)
        self.catalog = load_catalog(catalog_path)
        self.solve_timeout_s = float(solve_timeout_s)
        self.random_seed = int(random_seed)
        self.num_workers = int(num_workers)
        self.navigator = navigator
        self.lockboard = lockboard
        self._events: list[dict] = []
        self._sink = event_sink
        self._committed: set[str] = set()
        self._tasks: dict[str, TransportTask] = {}
        (self._node_pos, self._handoffs,
         self._inbound_pos) = _layout_index()
        self._veh_pos: dict[str, tuple[float, float]] = {
            v.vid: v.park_xy for v in self.fleet}
        self.metrics_acc = DispatchMetrics()

    # ------------------------------------------------------------ 布局
    def node_pos(self, node: str) -> tuple[float, float, float]:
        if node in self._node_pos:
            return self._node_pos[node]
        if node == INBOUND_NODE:
            return self._inbound_pos
        raise KeyError(f"未知节点 {node!r}")

    def node_xy(self, node: str) -> tuple[float, float]:
        return tuple(self.node_pos(node)[:2])

    # ------------------------------------------------------------ 提交
    def submit_orders(self, orders: Iterable, *, now_s: float | None = None,
                      solver: str = "cpsat") -> list[dict]:
        tasks = [build_task(o, catalog=self.catalog, node_pos=self.node_pos,
                            handoffs=self._handoffs) for o in orders]
        return self.submit(tasks, now_s=now_s, solver=solver)

    def submit(self, tasks: list[TransportTask], *,
               now_s: float | None = None, solver: str = "cpsat") -> list[dict]:
        """分配一批任务: 逐腿候选过滤 → CP-SAT → (超时) 贪心 → 决策事件。

        solver="greedy" 直接走贪心 (对比实验/诊断用); "cpsat" 路径在
        UNKNOWN(超时) 时自动回退, INFEASIBLE 抛 InfeasibleAssignment,
        MODEL_INVALID 抛 DispatchError (§19.4, 不允许静默忽略)。
        """
        if solver not in ("cpsat", "greedy"):
            raise ValueError(f"未知 solver: {solver!r}")
        batch_events: list[dict] = []
        for t in tasks:
            if t.task_id in self._tasks:
                raise ValueError(f"任务重复提交: {t.task_id}")
            self._tasks[t.task_id] = t

        elig = {t.task_id: t.eligible_vehicles(self.fleet) for t in tasks}
        assignable, stranded = [], []
        for t in tasks:
            (assignable if all(elig[t.task_id].values())
             else stranded).append(t)

        for t in stranded:
            bad = next(l for l in t.legs if not elig[t.task_id][l.leg_id])
            cap = max((v.capacity_kg for v in self.fleet
                       if v.vtype == LEG_VEHICLE_TYPE[bad.leg_type]),
                      default=0.0)
            ev = {
                "type": "TaskUnassigned", "task_id": t.task_id,
                "ts_s": now_s if now_s is not None else t.release_s,
                "candidates": [], "chosen": None,
                "reason": (f"no_capable_vehicle: {bad.leg_type} leg weight "
                           f"{t.weight_kg:.3f}kg > max {cap:.2f}kg"),
                "solver_status": "INFEASIBLE_FILTER", "solve_ms": 0.0,
                "makespan_ticks": None,
            }
            batch_events.append(ev)
            self._emit(ev)

        if assignable:
            solve_ms = 0.0
            if solver == "cpsat":
                t0 = time.perf_counter()
                plan, status = _solve_cpsat(
                    assignable, elig, self.node_xy, self.fleet,
                    timeout_s=self.solve_timeout_s, seed=self.random_seed,
                    workers=self.num_workers,
                    load_offset=(self.load_offset_provider()
                                 if self.load_offset_provider else None))
                solve_ms = (time.perf_counter() - t0) * 1000.0
            else:
                plan, status = None, "SKIP"
            if plan is None:
                if status == "INFEASIBLE":
                    raise InfeasibleAssignment(
                        "CP-SAT 判定 INFEASIBLE (能力/资源冲突), 按设计不回退")
                plan = _greedy_plan(assignable, elig, self.node_xy, self.fleet)
                status = "FALLBACK_GREEDY"

            self._apply(assignable, plan)
            batch_mk = max(t.schedule[t.legs[-1].leg_id][0]
                           + t.schedule[t.legs[-1].leg_id][1]
                           for t in assignable)
            for t in assignable:
                cands: list[str] = []
                for l in t.legs:
                    for v in elig[t.task_id][l.leg_id]:
                        if v.vid not in cands:
                            cands.append(v.vid)
                ev = {
                    "type": "TaskAssigned",
                    "ts_s": now_s if now_s is not None else t.release_s,
                    "task_id": t.task_id,
                    "candidates": cands,
                    "chosen": dict(t.assignment),
                    "start_ticks": {lid: s for lid, (s, _d) in
                                    t.schedule.items()},
                    "reason": self._reason(t, status),
                    "solver_status": status,
                    "solve_ms": round(solve_ms, 3),
                    "makespan_ticks": batch_mk,
                }
                batch_events.append(ev)
                self._emit(ev)
            self._acc_metrics(assignable, batch_mk)
        return batch_events

    # ------------------------------------------------------------ 事件
    def events(self) -> list[dict]:
        return list(self._events)

    def _emit(self, ev: dict) -> None:
        self._events.append(ev)
        if self._sink is not None:
            self._sink(ev)

    def _reason(self, t: TransportTask, status: str) -> str:
        end = t.schedule[t.legs[-1].leg_id][0] + t.schedule[t.legs[-1].leg_id][1]
        veh = "/".join(t.assignment[l.leg_id] for l in t.legs)
        if status == "FALLBACK_GREEDY":
            return (f"greedy nearest-capable: {veh}, handoff={t.handoff}, "
                    f"end={end}ticks")
        return (f"cp_sat makespan-optimal: {veh}, handoff={t.handoff}, "
                f"end={end}ticks")

    def _apply(self, tasks: list[TransportTask], plan: dict) -> None:
        for t in tasks:
            t.assignment = dict(plan[t.task_id]["veh"])
            t.schedule = dict(plan[t.task_id]["sched"])
            t.state = TaskState.ASSIGNED
            for l in t.legs:               # 记录车辆末位姿 (贪心就近用)
                self._veh_pos[t.assignment[l.leg_id]] = self.node_xy(l.dst)

    def _acc_metrics(self, tasks: list[TransportTask], batch_mk: int) -> None:
        m = self.metrics_acc
        m.makespan_ticks = max(m.makespan_ticks, batch_mk)
        for t in tasks:
            first = t.legs[0]
            m.total_wait_ticks += t.schedule[first.leg_id][0] - t.release_tick
        busy = dict(m.busy_ticks_by_vehicle)
        for t in tasks:
            for l in t.legs:
                busy[t.assignment[l.leg_id]] = (busy.get(t.assignment[l.leg_id], 0)
                                                + t.schedule[l.leg_id][1])
        m.busy_ticks_by_vehicle = busy
        win = max(m.makespan_ticks, 1)
        by_level: dict[str, list[int]] = {}
        for v in self.fleet:
            b, n = by_level.get(v.level, [0, 0])
            by_level[v.level] = [b + busy.get(v.vid, 0), n + 1]
        m.utilization_by_level = {
            lv: round(b / (win * n), 6)
            for lv, (b, n) in sorted(by_level.items())
        }

    # ------------------------------------------------------------ 事务
    def commit(self, task_id: str, *, evidence: dict, ts_s: float = 0.0) -> None:
        """ASSIGNED → COMMITTED: 必须物理证据, 原子且幂等 (R09 事件唯一)。

        对齐 services/scheduler 七段语义: 无证据/状态不对一律不提交;
        FAILED/UNKNOWN 不推进 (证据先于提交, 见模块 docstring openTCS 对照)。
        """
        if task_id in self._committed:
            raise ValueError(f"任务已提交 (R09 幂等): {task_id}")
        t = self._tasks.get(task_id)
        if t is None:
            raise KeyError(f"未知任务 {task_id}")
        if t.state is not TaskState.ASSIGNED:
            raise ValueError(f"任务状态 {t.state.value} ≠ ASSIGNED, 不得提交")
        if not evidence or not isinstance(evidence, dict):
            raise ValueError("COMMITTED 需要物理证据 (接触/位姿观测), 拒绝空证据")
        t.state = TaskState.COMMITTED
        t.evidence = dict(evidence)
        self._committed.add(task_id)
        self._emit({
            "type": "TaskCommitted", "task_id": task_id, "ts_s": ts_s,
            "evidence_keys": sorted(evidence),
            "openTCS_note": "COMMITTED = FINISHED 严格化 (物理证据后提交)",
        })

    # ------------------------------------------------------------ 视图
    def task(self, task_id: str) -> TransportTask:
        return self._tasks[task_id]

    def tasks_by_state(self, state: TaskState) -> list[TransportTask]:
        return [t for t in self._tasks.values() if t.state is state]

    def metrics(self) -> DispatchMetrics:
        return self.metrics_acc

    def iter_schedule(self) -> Iterator[tuple[TransportTask, list[dict]]]:
        """按开始 tick 输出 (task, [{leg, vid, start_tick, dur_ticks, ...}])。

        执行器逐腿调 navigation.plan(vehicle_xy, src, dst, lock_penalty)
        与 LockMatrix.acquire/release (W2a 契约)。
        """
        rows = []
        for t in self._tasks.values():
            if t.state is not TaskState.ASSIGNED:
                continue
            steps = [{
                "leg": l, "vid": t.assignment[l.leg_id],
                "start_tick": t.schedule[l.leg_id][0],
                "dur_ticks": t.schedule[l.leg_id][1],
                "vehicle_xy": self._veh_pos[t.assignment[l.leg_id]],
                "src": l.src, "dst": l.dst,
            } for l in t.legs]
            rows.append((t, steps))
        yield from sorted(rows, key=lambda r: r[1][0]["start_tick"])


# ---------------------------------------------------------------- 布局索引
def _layout_index() -> tuple[dict[str, tuple[float, float, float]], list[str],
                             tuple[float, float, float]]:
    """节点坐标索引 (W1 LAYOUT 单一来源, 惰性导入避免 demand 硬依赖 mujoco)。"""
    from sim.warehouse_cell import MatrixLayout

    lay = MatrixLayout()
    pos: dict[str, tuple[float, float, float]] = {}
    for s in lay.slot_ids():
        pos[s] = tuple(lay.slot_pose(s))
    for h, p in lay.handoff_poses().items():
        pos[h] = (p[0], p[1], 0.0)
    for d, p in lay.dock_poses().items():
        pos[d] = (p[0], p[1], 0.0)
    for n, p in lay.extra_nodes().items():    # 场景注册节点 (如 tk:*)
        pos[n] = (p[0], p[1], p[2])
    # 入库区: 西边缘通道 × 南泊位带 (打桩常量, 不改 W1 布局语义)
    inbound = (lay.p["handoff_x"][0], round(lay.handoff_y - 0.30, 4), 0.0)
    return pos, list(lay.p["handoff_ids"]), inbound


def _dur_ticks(leg: Leg, v: Vehicle) -> int:
    return max(1, math.ceil((leg.dist_m / v.speed_mps + leg.handling_s())
                            / TICK_S))


# ---------------------------------------------------------------- CP-SAT
def _solve_cpsat(tasks, elig, node_xy, fleet, *, timeout_s, seed, workers,
                 load_offset=None):
    """CP-SAT: 逐腿指派 + 开始时间 + 车辆互斥 + 交接链先后 + 交接台时间窗。

    决策变量: 每腿 × 合格车 指派 bool + 腿开始 tick;
    约束: 每腿恰一车 | 车辆同腿区间不重叠 (NoOverlap) | 同任务交接链
    (干线开始 ≥ 货位端结束 + dwell) | 释放期 (start ≥ release) |
    交接台占用窗 [落托, 接管] 同台不重叠 (离散资源时间窗);
    目标: makespan ≫ Σstart + w·max_v(load_v) —— makespan 主导,
    Σstart 消平局, 小权重均衡项把腿摊给低累计负载的合格车
    (load_offset = 批前累计 busy tick, 集成层注入; 纯批内为 0)。

    返回 plan = {task_id: {"veh": leg→vid, "sched": leg→(start,dur)}};
    UNKNOWN(超时) → (None, "UNKNOWN") 由上层回退贪心; INFEASIBLE /
    MODEL_INVALID 按 §19.4 语义上抛。
    """
    from ortools.sat.python import cp_model

    legs = [l for t in tasks for l in t.legs]
    elig_leg = {l.leg_id: elig[t.task_id][l.leg_id]
                for t in tasks for l in t.legs}
    # 贪心解作暖启动 + horizon 上界 (greedy 可行 ⇒ 模型必可行, 求解飞快)
    g = _greedy_plan(tasks, elig, node_xy, fleet)
    g_mk = max(s + d for tid in g for (s, d) in g[tid]["sched"].values())
    horizon = g_mk + max(
        max(_dur_ticks(l, v) for v in elig_leg[l.leg_id]) for l in legs
    ) + HANDOFF_DWELL_TICKS + 10

    m = cp_model.CpModel()
    start = {l.leg_id: m.NewIntVar(0, horizon, f"s_{l.leg_id}") for l in legs}
    end = {l.leg_id: m.NewIntVar(0, horizon, f"e_{l.leg_id}") for l in legs}
    b: dict[tuple[str, str], object] = {}
    for l in legs:
        for v in elig_leg[l.leg_id]:
            b[l.leg_id, v.vid] = m.NewBoolVar(f"b_{l.leg_id}_{v.vid}")
            m.Add(start[l.leg_id] + _dur_ticks(l, v) == end[l.leg_id]) \
                .OnlyEnforceIf(b[l.leg_id, v.vid])
        m.AddExactlyOne([b[l.leg_id, v.vid] for v in elig_leg[l.leg_id]])

    makespan = m.NewIntVar(0, horizon, "makespan")
    for t in tasks:
        m.Add(start[t.legs[0].leg_id] >= t.release_tick)      # 释放期
        # 交接链先后 (N 腿泛化; 两腿任务与原 f,k 约束逐字等价)
        for u, w in zip(t.legs, t.legs[1:]):
            m.Add(start[w.leg_id] >= end[u.leg_id] + HANDOFF_DWELL_TICKS)
        m.Add(makespan >= end[t.legs[-1].leg_id])

    # 车辆互斥: 同车全部腿区间不重叠 (时间窗)
    for v in fleet:
        ivs = [m.NewOptionalIntervalVar(
                   start[l.leg_id], _dur_ticks(l, v), end[l.leg_id],
                   b[l.leg_id, v.vid], f"iv_{l.leg_id}_{v.vid}")
               for l in legs if (l.leg_id, v.vid) in b]
        if len(ivs) > 1:
            m.AddNoOverlap(ivs)

    # 交接台离散资源时间窗: 占用 = [落托结束, 接管开始], 同台不重叠
    # (N 腿泛化: 窗取「dst==交接台腿」与其后继; 纯场景链无台边界则不占窗)
    by_handoff: dict[str, list] = {}
    for t in tasks:
        by_handoff.setdefault(t.handoff, []).append(t)
    for h, ts in by_handoff.items():
        if len(ts) < 2:
            continue
        ivs = []
        for t in ts:
            pair = next(((a, b) for a, b in zip(t.legs, t.legs[1:])
                         if a.dst == t.handoff), None)
            if pair is None:
                continue
            f, k = pair
            occ = m.NewIntVar(HANDOFF_DWELL_TICKS, horizon,
                              f"occs_{t.task_id}_{h}")
            m.Add(occ == start[k.leg_id] - end[f.leg_id])
            ivs.append(m.NewIntervalVar(
                end[f.leg_id], occ, start[k.leg_id], f"occ_{t.task_id}_{h}"))
        if len(ivs) > 1:
            m.AddNoOverlap(ivs)

    # 负载均衡 (利用率专项): load_v = 批前累计偏置 + 本批指派给 v 的腿
    # 时长和; 峰值 max_v load_v 以小权重进目标 —— 同等 makespan/Σstart 下
    # 把腿摊给低累计负载的合格车 (如 L1 快车多接轻货), 而非堆在少数车。
    off = {v.vid: int(load_offset.get(v.vid, 0))
           for v in fleet} if load_offset else {v.vid: 0 for v in fleet}
    work_ub = horizon + max(off.values(), default=0)
    work: dict[str, object] = {}
    for v in fleet:
        terms = [_dur_ticks(l, v) * b[l.leg_id, v.vid] for l in legs
                 if (l.leg_id, v.vid) in b]
        wv = m.NewIntVar(off[v.vid], work_ub, f"work_{v.vid}")
        m.Add(wv == off[v.vid] + sum(terms))
        work[v.vid] = wv
    max_load = m.NewIntVar(0, work_ub, "max_load")
    for v in fleet:
        m.Add(max_load >= work[v.vid])

    m.Minimize(_MAKESPAN_WEIGHT * makespan
               + sum(start[l.leg_id] for l in legs)
               + _BALANCE_WEIGHT * max_load)

    # 暖启动: 贪心解作初始 hint (部分/可行即可, CP-SAT 自行修复)
    for tid, p in g.items():
        for lid, (s0, d0) in p["sched"].items():
            m.AddHint(start[lid], s0)
            m.AddHint(end[lid], s0 + d0)
            for v in elig_leg[lid]:
                m.AddHint(b[lid, v.vid],
                          1 if p["veh"][lid] == v.vid else 0)
    m.AddHint(makespan, g_mk)
    g_load = dict(off)
    for tid, p in g.items():
        for lid, (_s0, d0) in p["sched"].items():
            g_load[p["veh"][lid]] = g_load.get(p["veh"][lid], 0) + d0
    m.AddHint(max_load, max(g_load.values(), default=0))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(timeout_s, 1e-6)
    solver.parameters.num_search_workers = max(1, workers)
    solver.parameters.random_seed = seed
    status = solver.Solve(m)
    name = solver.StatusName(status)
    if name == "MODEL_INVALID":
        raise DispatchError(f"CP-SAT MODEL_INVALID: {solver.SolutionInfo()}")
    if name in ("OPTIMAL", "FEASIBLE"):
        plan: dict[str, dict] = {}
        for t in tasks:
            veh = {l.leg_id: next(v.vid for v in elig_leg[l.leg_id]
                                  if solver.Value(b[l.leg_id, v.vid]))
                   for l in t.legs}
            sched = {l.leg_id: (solver.Value(start[l.leg_id]),
                                solver.Value(end[l.leg_id])
                                - solver.Value(start[l.leg_id]))
                     for l in t.legs}
            plan[t.task_id] = {"veh": veh, "sched": sched}
        return plan, name
    return None, name                 # INFEASIBLE / UNKNOWN


# ---------------------------------------------------------------- 贪心回退
def _greedy_plan(tasks, elig, node_xy, fleet):
    """回退: 最小代价贪心 —— 按释放序, 每腿选「当前可用且距 src 最近」的合格车。

    顺序排程保证物理可行 (同车串行 / 同台串行); 近距离任务被同一辆车
    串住正是 CP-SAT 严格更优的来源 (测试固化该用例)。
    交接台串行近似: 后任务落托结束 ≥ 前任务接管开始 (占用窗不重叠)。
    """
    free_at: dict[str, int] = {v.vid: 0 for v in fleet}
    pos: dict[str, tuple[float, float]] = {v.vid: v.park_xy for v in fleet}
    stand_ready: dict[str, int] = {}           # 交接台: 占用窗结束 tick
    plan: dict[str, dict] = {}
    for t in sorted(tasks, key=lambda t: (t.release_tick, t.task_id)):
        veh: dict[str, str] = {}
        sched: dict[str, tuple[int, int]] = {}
        earliest = t.release_tick
        for i, l in enumerate(t.legs):
            cands = elig[t.task_id][l.leg_id]
            v = min(cands, key=lambda v: (_manhattan(pos[v.vid], node_xy(l.src)),
                                          v.vid))
            st = max(earliest, free_at[v.vid])
            du = _dur_ticks(l, v)
            if l.dst == t.handoff:             # 落托到交接台: 占用窗不重叠
                if st + du < stand_ready.get(t.handoff, 0):
                    st = stand_ready[t.handoff] - du
                stand_ready[t.handoff] = st + du + HANDOFF_DWELL_TICKS
            veh[l.leg_id], sched[l.leg_id] = v.vid, (st, du)
            free_at[v.vid], pos[v.vid] = st + du, node_xy(l.dst)
            earliest = st + du + HANDOFF_DWELL_TICKS     # 交接链先后
        plan[t.task_id] = {"veh": veh, "sched": sched}
    return plan
