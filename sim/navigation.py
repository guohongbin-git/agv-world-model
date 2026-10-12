"""M2 Wave2 — 2-DOF 网格导航: 布局图派生 + A* + 分段运动指令。

图 (全部从 LAYOUT/MATRIX 派生, 零硬编码坐标):
- 节点: 列×行交叉点 / 交接台 stand / 交付台 dock (行通道线上内联) /
  槽位嘴 sm (行通道线上内联) / 槽位接近点 stg (巷道中心线) / 槽位面 sf /
  停车点 pk (南带外排 y=-2.85; F01 直停在横移带西端) / 叉车停车嘴 fm
  (横移带上, 外排停车 stub 的入带点)。
- 边: 列通道沿線 / 行通道沿线 / 停车 stub / 槽位接近 stub。
  行通道线取交付台 y (即 aisle−0.085, 与 dock_poses 同源); stg 取真实巷道
  中心线 (row_aisle_ys)。

转弯安全 (物理包络推演, 见 tests/test_fleet_navigation.py):
- 叉车原地转弯扫掠半径 0.218 m (配重角), AGV 0.12 m。
- 货架立柱距行通道线最近 0.155 m → **叉车禁止**在 sm/sf/stand 原地转弯
  (A* 剪枝 turn_block); AGV 全图可转。
- 内巷道槽位接近走 stg (巷道中心, 双侧 0.233 m 净空): sm→stg 为横移边
  (move="strafe", 不转头平移 ≤0.085 m), stg→sf(前/后) 为行车边。
- 后排槽位取放后原路倒车退出 (W1 leg-8 带载倒车同款)。

A* (Space-Time A* 的软化版): 状态 = (节点, 朝向象限); 代价 = 边长 +
转弯惩罚 + 动态锁惩罚 (他车持有 lane/节点的边与点加罚) + 横移惩罚。
锁惩罚由 sim/traffic.TrafficManager 的当前占用快照提供 —— 被占巷道自动
绕行; 全部可行路径均被占时 A* 仍返回最优 (可能 claim 失败 → 上层等待)。

学术参照 (docs/knowledge/related_work.md):
- Space-Time A* (GavinPHR/MAPF): 本模块把"时空障碍"软化为 lock_penalty,
  严格版=把已占资源当不可通行; openTCS Block 语义 = 本锁矩阵的区域互斥
  对应 (sim/traffic.py)。

分段运动指令: 路径 → [(yaw 原地转 | 单轴平移)] 段序列, 段 = W1 同款
(执行器id, Ramp工厂, 关节偏移, settle) 四元组, 由 FleetRunner 并发执行
(100Hz 位置闭环 + Ramp 梯形限速, 物理 1ms)。
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace

import mujoco

from sim.warehouse_cell import K, WarehouseCell

_QUADS = (0.0, math.pi / 2, math.pi, -math.pi / 2)      # E N W S
_STRAFE_PENALTY = 0.30       # 横移边等效附加里程 (m)
_TURN_PENALTY = 0.15         # 每次原地转弯等效里程 (m)
_LOCK_PENALTY = 3.0          # 他车占用 lane 的等效附加里程 (m)
_NODE_PENALTY = 1.0          # 他车占用节点的等效附加里程 (m)


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    axis: str                 # "x" | "y"
    lane: str                 # 主 lane 锁
    crosses: tuple = ()       # 额外需占用的 lane (stub 穿越的行通道)
    strafe: bool = False      # True: 不转头平移 (转弯禁区边)
    exit_quad: int | None = None   # 平移前在本边起点原地转向的目标象限
    nodes: tuple = ()         # 额外需占用的节点 (内圈槽位 stub 毗邻 dock)


@dataclass
class Node:
    x: float
    y: float
    kind: str                 # x/stand/dock/sm/stg/sf/pk/fm


class NavGraph:
    """布局交通图 —— 节点/边/锁资源全部派生自 MatrixLayout。"""

    def __init__(self, layout):
        L = layout
        self.layout = layout
        self.pos: dict[str, Node] = {}
        self.adj: dict[str, list[tuple[str, Edge]]] = {}
        p = L.p
        self.col_xs = L.col_aisle_xs()
        self.aisle_ys = L.row_aisle_ys()                    # 真实巷道中心
        self.stand_y = L.handoff_y
        self.strip_y = L.handoff_y - p["berth_depth_y"]     # 横移/停车带 (=W1 clear_y)
        # 行通道线 y: 内巷道 3 条 + 北线, 全部取 dock_poses (与 −0.085 同源)
        docks = L.dock_poses()
        self.line_ys = {0: self.strip_y, 1: self.stand_y}
        for r in (1, 2, 3):
            self.line_ys[r + 1] = docks[f"D_G{r}1"][1]
        self.line_ys[5] = docks[f"D_G41"][1]
        self._build()

    # ------------------------------------------------------------ 构建
    def _add(self, nid, x, y, kind):
        self.pos[nid] = Node(x, y, kind)
        self.adj.setdefault(nid, [])

    def _edge(self, a, b, axis, lane, crosses=(), strafe=False,
              exit_quad: int | None = None, nodes=()):
        e = Edge(a, b, axis, lane, crosses, strafe, exit_quad, nodes)
        self.adj[a].append((b, e))
        # 反向副本: 出场取向 (exit_quad) 属于起点侧机动, 不随边复制
        self.adj[b].append((a, replace(e, a=b, b=a, exit_quad=None)))

    def xnode(self, i, j):
        return f"x:{i}:{j}"

    def _build(self):
        L, p = self.layout, self.layout.p
        cols = self.col_xs
        # ---- 停车点节点 (F01 类横移带停车: 节点即 fm, 内联 strip) ----
        for vid, (px, py, _) in p["park"].items():
            kind = "fm" if (vid.startswith("F")
                            and abs(py - self.strip_y) < 1e-6) else "pk"
            self._add(f"pk:{vid}", px, py, kind)
        # ---- 列通道: strip → stand → 内巷道 → 北线 ----
        for i, x in enumerate(cols):
            self._add(self.xnode(i, 0), x, self.strip_y, "x")
            hx, hy, _ = L.handoff_poses()[f"H{i + 1:02d}"]
            assert abs(hx - x) < 1e-6, "交接台必须位于列通道中心"
            self._add(f"stand:H{i + 1:02d}", hx, hy, "stand")
            self._edge(self.xnode(i, 0), f"stand:H{i + 1:02d}", "y", f"col:{i}")
            for j in (2, 3, 4, 5):
                self._add(self.xnode(i, j), x, self.line_ys[j], "x")
                prev = f"stand:H{i + 1:02d}" if j == 2 else self.xnode(i, j - 1)
                self._edge(prev, self.xnode(i, j), "y", f"col:{i}")
        # ---- 交付台 (行通道线上内联节点) ----
        for did, (dx, dy, _) in L.dock_poses().items():
            j = min((jj for jj in self.line_ys if jj != 1),
                    key=lambda jj: abs(self.line_ys[jj] - dy))
            assert abs(self.line_ys[j] - dy) < 1e-6, f"{did} 应在行通道线上"
            self._add(f"dock:{did}", dx, dy, "dock")
        # ---- 停车 stub: AGV 直连列通道; 叉车经横移带嘴 fm (横移带停车者免) ----
        for vid, (px, py, _) in p["park"].items():
            if vid.startswith("A"):
                i = min(range(len(cols)),
                        key=lambda i: abs(cols[i] - px))
                assert abs(cols[i] - px) < 1e-6, \
                    f"AGV 停车点 {vid} 必须对齐列通道 (布局约束)"
                self._edge(f"pk:{vid}", self.xnode(i, 0), "y", f"col:{i}")
            elif abs(py - self.strip_y) >= 1e-6:
                self._add(f"fm:{vid}", px, self.strip_y, "fm")
                self._edge(f"pk:{vid}", f"fm:{vid}", "y", f"park:{vid}",
                           crosses=("row:0",))
        self._link_rows()

    def _slot_face(self, slot: str):
        """槽位面节点 id + 几何 (face/approach 原点)。返回 (id, x, y)。"""
        L = self.layout
        r, c = int(slot[1]), int(slot[2])
        k = int(slot.split("S")[1].split("-")[0])
        cx, _ = L.group_center(r, c)
        sx = cx + (k % 4 - 1.5) * L.p["bay_pitch_x"]
        sy = L.slot_pose(f"{slot}-L0")[1]
        off = K["approach_gap"] + K["fork_len"] + K["fork_root_x"]
        if k < 4:      # 前排: 巷道南侧接近, 面向 +y
            return f"sf:{slot}", sx, sy - 0.040 - off
        return f"sf:{slot}", sx, sy + 0.040 + off

    @staticmethod
    def _line_of_slot(r: int, front: bool) -> int:
        """槽位服务行通道线号: 前排=组南侧线, 后排=组北侧线。

        r=1: front→0(strip), back→2; r=2: 2/3; r=3: 3/4; r=4: front→4, back→5。
        """
        if front:
            return 0 if r == 1 else r
        return r + 1

    def _link_rows(self):
        # 槽位嘴 / 接近点 / 面; 嘴与接近点按 (line, c, k%4) 归一 —— 同一
        # 槽位 x 是前后两排槽位的公共开口 (G1 前排挂在 strip 除外)
        for r in range(1, 5):
            for c in range(1, 5):
                for k in range(8):
                    slot = f"R{r}{c}-S{k}"
                    front = k < 4
                    j = self._line_of_slot(r, front)
                    fid, fx, fy = self._slot_face(slot)
                    self._add(fid, fx, fy, "sf")
                    mid = f"sm:{j}:{c}:{k % 4}"
                    lane = f"stub:{j}:{c}:{k % 4}"
                    if j == 0:
                        # G1 前排: strip 上开口, 直接转弯接近 (strip 净空足)
                        if mid not in self.pos:
                            self._add(mid, fx, self.strip_y, "sm")
                        self._edge(mid, fid, "y", lane, crosses=("row:0",))
                    else:
                        stg = f"stg:{j}:{c}:{k % 4}"
                        if mid not in self.pos:
                            self._add(mid, fx, self.line_ys[j], "sm")
                        if stg not in self.pos:
                            self._add(stg, fx, self.aisle_ys[r], "stg")
                        cr = (f"row:{j}",)
                        # 内圈槽位 (组心 ±0.08): 车体横向包络 ±0.075 与
                        # 毗邻 dock 上 AGV 包络 (r0.12) 重叠 0.115 → stub
                        # 申请须连带该 dock 节点 (dock 墙已视觉化, 互斥
                        # 由锁矩阵保证)
                        dn = (f"dock:D_G{j - 1}{c}",) if k % 4 in (1, 2) \
                            else ()
                        self._edge(mid, stg, "y", lane, crosses=cr,
                                   strafe=True, nodes=dn)
                        # stg→sf: 前排面向 +y, 后排面向 −y (行车边, 转弯在 stg)
                        self._edge(stg, fid, "y", lane, crosses=cr, nodes=dn)
                        # stg→sm 出场: 转弯在 stg (安全点) 完成东/西取向后
                        # 横移回线上 —— 叉车在 sm 处禁转 (扫掠包络)
                        for xq in (0, 2):
                            self._edge(stg, mid, "y", lane, crosses=cr,
                                       strafe=True, exit_quad=xq, nodes=dn)
        for j in (0, 2, 3, 4, 5):
            y = self.line_ys[j]
            ns = sorted((n for n, nd in self.pos.items()
                         if abs(nd.y - y) < 1e-6 and nd.kind in
                         ("x", "dock", "sm", "fm")),
                        key=lambda n: self.pos[n].x)
            for a, b in zip(ns, ns[1:]):
                self._edge(a, b, "x", f"row:{j}")

    # ------------------------------------------------------------ 查询
    def turn_blocked(self, node: str, veh_kind: str) -> bool:
        """该节点是否禁止该类车原地转弯 (扫掠包络)。"""
        if veh_kind == "agv":
            return False
        return node.startswith(("sm:", "sf:", "stand:")) \
            and not node.startswith("sm:0:")

    def anchor_node(self, node: str) -> bool:
        """stub 锚点 (离线进入点): 在回到线上前必须保持持有。"""
        return node.startswith("sm:") and not node.startswith("sm:0:")

    def stub_internal(self, node: str) -> bool:
        return node.startswith(("stg:", "sf:"))

    def node_for(self, what: str) -> str:
        """业务名 → 图节点: 货位 'R23-S5' → sf; 'H02' → stand; 'D_G12' → dock。"""
        if what.startswith("R"):
            return f"sf:{what}"
        if what.startswith("H"):
            return f"stand:{what}"
        if what.startswith("D"):
            return f"dock:{what}"
        return what

    def park_node(self, vid: str) -> str:
        return f"pk:{vid}"

    def col_index_of_stand(self, sid: str) -> int:
        """'H02' / 'stand:H02' → 列号 0-基 (stand 节点按 handoff_x 顺序
        挂在 col:i 上, 见 _add('stand:H(i+1)') 构造)。"""
        tok = sid.rsplit(":", 1)[-1]           # 兼容两种格式
        return int(tok[1:]) - 1

    def plan(self, start: str, goal: str, veh_kind: str, start_quad: int = 0,
             occ_lanes=frozenset(), occ_nodes=frozenset(),
             carrying: bool = False, hard_lanes=frozenset(),
             hard_nodes=frozenset()):
        """A* 规划 (软锁惩罚 + 硬排除重试)。返回 (path, edges) 或 None。"""
        try:
            path, edges, _ = a_star(
                self, start, goal, veh_kind, start_quad,
                occ_lanes, occ_nodes, hard_lanes=hard_lanes,
                hard_nodes=hard_nodes)
        except ValueError:
            return None
        return path, edges

    def nearest_node(self, x: float, y: float) -> str:
        return min(self.pos, key=lambda n: (self.pos[n].x - x) ** 2
                   + (self.pos[n].y - y) ** 2)


# ---------------------------------------------------------------- A*
def a_star(graph: NavGraph, start: str, goal: str, veh_kind: str,
           start_quad: int = 0, occ_lanes: frozenset = frozenset(),
           occ_nodes: frozenset = frozenset(), turn_penalty=_TURN_PENALTY,
           lock_penalty=_LOCK_PENALTY, hard_lanes: frozenset = frozenset(),
           hard_nodes: frozenset = frozenset()) \
        -> tuple[list[str], list[Edge], float]:
    """状态图 A*: (节点, 朝向象限)。返回 (节点路径, 边序列, 代价)。

    occ_lanes/occ_nodes   = 他车持有资源 → 加罚 (软化 Space-Time A*):
                            有更便宜的绕行即绕行。
    hard_lanes/hard_nodes = claim 失败后上层回填的"确定阻塞"资源 → 禁行
                            (保证重试必然换路或确实无路可走 → 等待)。
    """
    goal_xy = graph.pos[goal]

    def h(state):
        n, _ = state
        nd = graph.pos[n]
        return math.hypot(nd.x - goal_xy.x, nd.y - goal_xy.y)

    ctr = 0
    pq = [(h((start, start_quad)), 0, ctr, (start, start_quad), None, None)]
    best = {(start, start_quad): 0.0}
    parent: dict = {}
    while pq:
        f, g, _, state, pedge, pstate = heapq.heappop(pq)
        if g > best.get(state, 1e18) + 1e-12:
            continue
        parent[state] = (pstate, pedge)
        n, q = state
        if n == goal:
            # 回溯
            path, edges = [n], []
            s = state
            while parent[s][0] is not None:
                ps, pe = parent[s]
                edges.append(pe)
                path.append(ps[0])
                s = ps
            path.reverse()
            edges.reverse()
            return path, edges, g
        for nb, e in graph.adj[n]:
            if e.lane in hard_lanes or set(e.crosses) & hard_lanes \
                    or nb in hard_nodes or set(e.nodes) & hard_nodes:
                continue                      # 确定阻塞 → 禁行
            if graph.pos[n].kind == "sf":
                nq = q                        # 面位出场: 倒车, 不转头
            elif e.strafe:
                nq = e.exit_quad if e.exit_quad is not None else q
            else:
                vec = (graph.pos[nb].x - graph.pos[n].x,
                       graph.pos[nb].y - graph.pos[n].y)
                if abs(vec[0]) > abs(vec[1]):
                    nq = 0 if vec[0] > 0 else 2
                else:
                    nq = 1 if vec[1] > 0 else 3
            if q != nq and graph.turn_blocked(n, veh_kind):
                continue                      # 转弯禁区剪枝
            cost = math.hypot(graph.pos[nb].x - graph.pos[n].x,
                              graph.pos[nb].y - graph.pos[n].y)
            if q != nq:
                cost += turn_penalty
            if e.strafe:
                cost += _STRAFE_PENALTY
            if e.lane in occ_lanes or set(e.crosses) & occ_lanes:
                cost += lock_penalty
            if nb in occ_nodes or set(e.nodes) & occ_nodes:
                cost += _NODE_PENALTY
            ng = g + cost
            ns = (nb, nq)
            if ng < best.get(ns, 1e18) - 1e-12:
                best[ns] = ng
                ctr += 1
                heapq.heappush(pq, (ng + h(ns), ng, ctr, ns, e, state))
    raise ValueError(f"A* 无路径: {start} → {goal} ({veh_kind})")


# ---------------------------------------------------------------- 运动规划
@dataclass
class Leg:
    kind: str                 # "yaw" | "move"
    axis: str | None          # yaw: "yaw"; move: "x"/"y"
    target: float             # 世界坐标目标 (yaw 为弧度)
    speed: float
    accel: float


class MotionPlanner:
    """路径 → 分段运动指令 (原地转向 → Ramp 直行/横移)。

    段目标一律世界坐标; 执行时经 view.seg() 换算为关节量 (W1 机制复用)。
    """

    def __init__(self, cell: WarehouseCell, graph: NavGraph):
        self.cell = cell
        self.graph = graph

    def _speeds(self, vid: str, carrying: bool):
        cp = self.cell.vclass(vid)
        f = 0.6 if carrying else 1.0
        return dict(cruise=cp["cruise"], accel=cp["accel"],
                    yaw=cp["yaw_speed"] * f, yaw_acc=cp["yaw_accel"] * f,
                    strafe=cp["strafe"])

    def route_legs(self, vid: str, path: list[str], edges: list[Edge],
                   start_quad: int, carrying: bool = False) -> list[Leg]:
        sp = self._speeds(vid, carrying)
        legs: list[Leg] = []
        q = start_quad
        for e in edges:
            a = self.graph.pos[e.a]
            b = self.graph.pos[e.b]
            axis = "x" if abs(b.x - a.x) >= 1e-9 else "y"
            target = b.x if axis == "x" else b.y
            if a.kind == "sf":
                # 面位出场: 倒车 (W1 leg-8 带载倒退同款), 不转头
                legs.append(Leg("move", axis, target,
                                K["fk_approach"], min(sp["accel"], 0.3)))
                continue
            if e.strafe and e.exit_quad is not None and e.exit_quad != q:
                legs.append(Leg("yaw", "yaw", _QUADS[e.exit_quad],
                                sp["yaw"], sp["yaw_acc"]))
                q = e.exit_quad
            if e.strafe:
                legs.append(Leg("move", axis, target,
                                sp["strafe"], min(sp["accel"], 0.3)))
                continue
            vec = (b.x - a.x, b.y - a.y)
            if abs(vec[0]) >= abs(vec[1]):
                nq = 0 if vec[0] > 0 else 2
            else:
                nq = 1 if vec[1] > 0 else 3
            if nq != q:
                legs.append(Leg("yaw", "yaw", _QUADS[nq], sp["yaw"],
                                sp["yaw_acc"]))
                q = nq
            # 载货减速: 加减速惯性使叉上/顶载托盘沿摩擦面滑移 (haul 实测
            # dxy 0.039), 速度/加速度各降 40%/50%
            f = 0.6 if carrying else 1.0
            legs.append(Leg("move", axis, target, sp["cruise"] * f,
                            sp["accel"] * 0.5 if carrying else sp["accel"]))
        return legs


# ---------------------------------------------------------------- 并发执行器
class FleetRunner:
    """多车并发段执行器: 100Hz 位置闭环, 每车独立段队列 + settle。

    段四元组 (aid, ramp工厂, off, settle_s) 与 W1 FleetController 同构,
    但多车同时推进 —— 同一控制 tick 内各车写各自执行器, 物理同步步进。
    确定性: 单线程, 车辆按 id 排序遍历, 无墙钟。
    """

    CTRL_HZ = 100

    def __init__(self, cell: WarehouseCell, audit_every: int = 25):
        self.cell = cell
        self.substeps = max(1, round(0.01 / cell.model.opt.timestep))
        self.audit_every = audit_every
        self.queues: dict[str, list] = {v: [] for v in cell.veh}
        self.cur: dict[str, tuple | None] = {v: None for v in cell.veh}
        self._was_busy: dict[str, bool] = {v: False for v in cell.veh}
        self.audit = {"max_pen": 0.0, "violations": [],
                      "steps": 0, "done": []}

    def submit(self, vid: str, segs):
        self.queues[vid].extend(segs)

    def busy(self, vid: str | None = None) -> bool:
        if vid is not None:
            return bool(self.queues[vid]) or self.cur[vid] is not None
        return any(self.busy(v) for v in self.queues)

    def _start(self, vid):
        item = self.queues[vid].pop(0)
        aid, build, off = item[0], item[1], item[2]
        settle = item[3] if len(item) > 3 else 0.15
        ramp = build()
        # [aid, ramp, off, settle, t_start, phase, t_settle]
        self.cur[vid] = [aid, ramp, off, settle, self.cell.data.time, 0, 0.0]

    def run_until_any_idle(self, max_s: float = 3600.0) -> list[str]:
        """推进仿真直至任一车队列耗尽变空闲 (或全部完成)。返回刚空闲车辆。

        调用方在此间隙为空闲车申请下一段路由/作业相位并 submit;
        全部完成后返回 []。确定性: 单线程 + 车辆 id 序遍历。
        """
        m, d = self.cell.model, self.cell.data
        step_n = self.audit["steps"]
        while any(self.busy(v) for v in self.queues):
            if d.time > max_s:
                self.audit["violations"].append(f"超时 max_s={max_s}")
                break
            for vid in sorted(self.queues):
                if self.cur[vid] is None:
                    if self.queues[vid]:
                        self._start(vid)
                    else:
                        continue
                aid, ramp, off, settle, t0, phase, t_settle = self.cur[vid]
                te = d.time - t0
                if te < ramp.duration:
                    d.ctrl[aid] = ramp.at(te) - off
                elif phase == 0:
                    d.ctrl[aid] = ramp.target - off
                    self.cur[vid][5] = 1          # 进入 settle
                    self.cur[vid][6] = d.time
                elif d.time - t_settle < settle:
                    pass                          # settle: ctrl 保持 target
                else:
                    self.audit["done"].append((vid, d.time,
                                               ramp.target - off))
                    self.cur[vid] = None
                    continue
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
                step_n += 1
                if step_n % self.audit_every == 0:
                    pen = self.cell.max_penetration()
                    self.audit["max_pen"] = max(self.audit["max_pen"], pen)
                    if pen > 5e-4 and len(self.audit["violations"]) < 32:
                        self.audit["violations"].append(
                            f"穿透 {pen * 1000:.2f}mm @ t={d.time:.2f}s")
            self.audit["steps"] = step_n
            # 有车刚空闲 → 交还调用方喂下一段相位 (claim → 段序列)
            newly = [v for v in sorted(self.queues)
                     if self.cur[v] is None and not self.queues[v]
                     and self._was_busy.get(v)]
            if newly:
                for v in newly:
                    self._was_busy[v] = False
                return newly
            for v in self.queues:
                if self.busy(v):
                    self._was_busy[v] = True
        return []

    def run(self, max_s: float = 3600.0) -> dict:
        """推进直至所有车队列空。返回审计。"""
        while self.run_until_any_idle(max_s):
            pass
        return self.audit
