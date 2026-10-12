"""M2 Wave2 — episodic 任务薄驱动 (并发演示/验收用)。

正式调度 (任务分配/CP-SAT/需求曲线) 由 dispatch 模块承担 —— 本模块只做
"物理接口对接": 任务 → 相位序列 (导航 claim → 分段运动 → 精确作业段 →
导航返回), 全部走 W1 已验证的 seg()+Ramp 机制, 并经 TrafficManager 锁矩阵
串行化巷道/交叉口争用。4 类任务:

- pick    叉车: 货位取托 → 交接台放托        (下架)
- haul    AGV : 交接台顶升 → 交付台放托      (干线运输)
- putaway 叉车: 交接台取托 → 货位放托        (上架)
- fetch   AGV : 交付台顶升 → 交接台放托      (回库干线)

导航相位: A* (软锁惩罚; claim 失败 → 失败资源硬排除重规划 = 绕行, 无路
可走 → 原地等待重试) + 原子整路由 claim。作业段几何与 W1 plan_* 同源
(K 常数), 行车/转向/举升速度按车辆等级 (VEHICLE_CLASSES) 缩放, 精度段
(插叉/对中/落托) 保持 W1 定值。确定性: 单线程, 任务 FIFO, 分配按
(A* 代价, vid) 字典序。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from sim.controllers import Ramp
from sim.navigation import (FleetRunner, Leg, MotionPlanner, NavGraph,
                            a_star)
from sim.pallet_support import ledger_from_cell
from sim.traffic import TrafficManager
from sim.warehouse_cell import K, WarehouseCell, pallet_name_for

NAV_SETTLE, YAW_SETTLE, OP_SETTLE = 0.12, 0.10, 0.30

# 每类任务的相位序列: nav(目标, 载货?) / op_* / reverse (AGV 放托后倒出)
_PHASES = {
    "pick": [("nav", "slot", False), ("op_pick",),
             ("nav", "standstrip", True), ("op_stand_place",),
             ("nav", "park", False)],
    "putaway": [("nav", "standstrip", False), ("op_stand_take",),
                ("nav", "slot", True), ("op_put",), ("nav", "park", False)],
    "haul": [("nav", "stand", False), ("op_pin",), ("nav", "dock", True),
             ("op_unpin", "rev"), ("nav", "park", False)],
    "fetch": [("nav", "dock", False), ("op_pin",), ("nav", "stand", True),
              ("op_unpin", "rev"), ("nav", "park", False)],
}


@dataclass
class Job:
    vid: str
    task: dict
    phase: int = 0
    mode: str = "wait"                 # wait / nav / op
    path: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    idx: int = 0
    carrying: bool = False
    extras: tuple = ()
    needs_rev: bool = False
    micro: bool = False                # 倒出微路由: 不推进 phase
    last_goal: str = ""                # 上次导航目标种类 (reroute 事件判定)
    last_path: list = field(default_factory=list)


class MissionRunner:
    """episodic 任务执行器: 最小 A* 代价分配 + 相位推进 + 库存台账。"""

    FORK_TASKS = ("pick", "putaway")
    AGV_TASKS = ("haul", "fetch")

    def __init__(self, cell: WarehouseCell):
        self.cell = cell
        self.graph = NavGraph(cell.layout)
        self.planner = MotionPlanner(cell, self.graph)
        self.traffic = TrafficManager(self.graph)
        self.runner = FleetRunner(cell)
        self.ledger = ledger_from_cell(cell)
        self.jobs: dict[str, Job | None] = {v: None for v in cell.veh}
        self.node_of = {v: self.graph.park_node(v) for v in cell.veh}
        self.quad_of = {v: 0 for v in cell.veh}
        for v in cell.veh:
            assert self.traffic.claim_route(v, [], [self.node_of[v]])
        self.assignment_log: list[dict] = []
        self.op_errors: list[str] = []

    # ------------------------------------------------------------ 分配
    def assign(self, task) -> str | None:
        goal = self.graph.node_for(task[self._key(task)])
        lanes, nodes = self.traffic.occupancy()
        best, best_cost = None, 1e18
        kind = "agv" if task["kind"] in self.AGV_TASKS else "forklift"
        for vid in self._eligible(task["kind"]):
            if self.jobs[vid] is not None:
                continue
            try:
                _, _, cost = a_star(self.graph, self.node_of[vid], goal,
                                    kind, self.quad_of[vid],
                                    frozenset(lanes), frozenset(nodes))
            except ValueError:
                continue
            if cost < best_cost - 1e-9:
                best, best_cost = vid, cost
        if best is not None:
            self.assignment_log.append(
                {"task": task["kind"],
                 "arg": task.get("slot") or task.get("stand")
                 or task.get("dock"),
                 "vid": best, "cost": round(best_cost, 3)})
            self.traffic.events.append(
                (self.traffic.t, best, "assign",
                 f"{task['kind']}:{task.get('slot') or task.get('stand') or task.get('dock')}"
                 f" cost={best_cost:.2f}"))
        return best

    @staticmethod
    def _key(task):
        return {"pick": "slot", "putaway": "stand",
                "haul": "stand", "fetch": "dock"}[task["kind"]]

    def _eligible(self, kind):
        return self.cell.forklift_ids if kind in self.FORK_TASKS \
            else self.cell.agv_ids

    # ------------------------------------------------------------ 段构造
    def _seg(self, vid, leg, settle=NAV_SETTLE):
        view = self.cell.veh[vid]
        if leg.kind == "yaw":
            return (view.a["yaw"],
                    lambda: Ramp(view.q("yaw"), leg.target,
                                 leg.speed, leg.accel),
                    view.off["yaw"], settle)
        return (view.a[leg.axis],
                lambda: Ramp(view.q(leg.axis) + view.off[leg.axis],
                             leg.target, leg.speed, leg.accel),
                view.off[leg.axis], settle)

    def _slift(self, vid, z_abs, speed):
        view = self.cell.veh[vid]
        return (view.a["lift"],
                lambda: Ramp(view.q("lift"), z_abs - K["transport_z"],
                             speed, 0.10), view.off["lift"], OP_SETTLE)

    def _move_seg(self, vid, axis, target, speed):
        view = self.cell.veh[vid]
        return (view.a[axis],
                lambda: Ramp(view.q(axis) + view.off[axis], target,
                             speed, 0.30), view.off[axis], OP_SETTLE)

    def _yaw_seg(self, vid, target):
        cp = self.cell.vclass(vid)
        return self._seg(vid, Leg("yaw", "yaw", target, cp["yaw_speed"],
                                  cp["yaw_accel"]), YAW_SETTLE)

    def _quad_after(self, job, i, q):
        """按边语义推演穿越边 i 后的朝向 (exit_quad / 面位倒车 / 几何方向)。"""
        e = job.edges[i]
        if e.exit_quad is not None:
            return e.exit_quad
        a, b = self.graph.pos[e.a], self.graph.pos[e.b]
        if a.kind == "sf" or e.strafe:
            return q
        if abs(b.x - a.x) >= abs(b.y - a.y):
            return 0 if b.x > a.x else 2
        return 1 if b.y > a.y else 3

    def _submit_edge(self, vid, job, i):
        legs = self.planner.route_legs(vid, job.path, job.edges[i:i + 1],
                                       self.quad_of[vid], job.carrying)
        for lg in legs:
            self.runner.submit(vid, [self._seg(vid, lg)])
        self.quad_of[vid] = self._quad_after(job, i, self.quad_of[vid])

    # ---- 作业段 (W1 几何同源) ----
    def _op_pick_segs(self, vid, slot, level):
        z = self.cell.layout.p["levels_z"][level]
        fy = self.graph._slot_face(slot)[2]
        sgn = 1.0 if int(slot.split("S")[1]) < 4 else -1.0
        cp = self.cell.vclass(vid)
        # W1 顺序: 先升至插入高度 (层+10), 插叉, 微升离梁 (+17), 带载倒退
        return [
            self._slift(vid, z + 0.010, cp["lift"]),
            self._move_seg(vid, "y", fy + sgn * 0.120, K["fk_insert"]),
            self._slift(vid, z + 0.017, K["lift_slow"]),
            self._move_seg(vid, "y", fy, K["fk_approach"]),
            self._slift(vid, K["transport_z"], cp["lift"]),
        ]

    def _op_put_segs(self, vid, slot, level):
        z = self.cell.layout.p["levels_z"][level]
        fy = self.graph._slot_face(slot)[2]
        sgn = 1.0 if int(slot.split("S")[1]) < 4 else -1.0
        cp = self.cell.vclass(vid)
        return [
            self._slift(vid, z + 0.030, cp["lift"]),
            self._move_seg(vid, "y", fy + sgn * 0.120, K["fk_insert"]),
            self._slift(vid, z + 0.0115, K["place"]),
            self._move_seg(vid, "y", fy, K["fk_approach"]),
            self._slift(vid, K["transport_z"], cp["lift"]),
        ]

    def _op_stand_segs(self, vid, stand, take: bool):
        """交接台放托/取托 (W1 legs 12-18 同源)。"""
        top = self.cell.layout.p["stand_top_z"]
        hy = self.cell.layout.stand_pose(stand)[1]
        place_z, rel_z = top + 0.012, top + 0.0115
        app2 = hy - (K["approach_gap"] + K["fork_len"] + K["fork_root_x"])
        axle = hy - K["pallet_center_off"]
        clear_y = self.graph.strip_y
        cp = self.cell.vclass(vid)
        segs = [self._yaw_seg(vid, math.pi / 2),
                self._slift(vid, place_z + 0.030, K["place"]),
                self._move_seg(vid, "y", app2, K["fk_approach"]),
                self._move_seg(vid, "y", axle, K["fk_insert"])]
        if take:
            segs += [self._slift(vid, place_z + 0.042, K["lift_slow"]),
                     self._move_seg(vid, "y", clear_y, K["fk_approach"]),
                     self._slift(vid, K["transport_z"], cp["lift"])]
        else:
            segs += [self._slift(vid, place_z, K["place"]),
                     self._slift(vid, rel_z, K["lift_slow"]),
                     self._move_seg(vid, "y", clear_y, K["fk_approach"]),
                     self._slift(vid, K["transit_fork_top"], K["lift"])]
        return segs

    def _op_pin_seg(self, vid, up: bool):
        view = self.cell.veh[vid]
        cp = self.cell.vclass(vid)
        return (view.a["pin"],
                lambda: Ramp(view.q("pin"),
                             K["pin_travel"] if up else 0.0,
                             cp["lift"] if up else 0.005, 0.10),
                view.off["pin"], OP_SETTLE)

    # ------------------------------------------------------------ 相位
    def _phase_args(self, job, ph):
        t = job.task
        if ph[1] == "slot":
            return self.graph.node_for(t["slot"])
        if ph[1] == "stand":
            return self.graph.node_for(t["stand"])
        if ph[1] == "standstrip":
            i = self.graph.col_index_of_stand(t["stand"])
            return self.graph.xnode(i, 0)
        if ph[1] == "dock":
            return self.graph.node_for(t["dock"])
        return self.graph.park_node(job.vid)

    def _try_next_phase(self, vid) -> bool:
        job = self.jobs[vid]
        t = job.task
        phs = _PHASES[t["kind"]]
        if job.phase >= len(phs):
            self.jobs[vid] = None              # 任务完成, 车辆回空闲池
            return False
        ph = phs[job.phase]
        if ph[0] == "nav":
            goal = self._phase_args(job, ph)
            kind = "agv" if vid in self.cell.agv_ids else "forklift"
            self.traffic.prune_exclusions(vid)
            lanes, nodes = self.traffic.occupancy(exclude=vid)
            blk = self.traffic.last_blocked.get(vid, {})
            plan = self.graph.plan(
                self.node_of[vid], goal, kind, self.quad_of[vid],
                frozenset(lanes), frozenset(nodes),
                hard_lanes=frozenset(blk.get("lanes", ())),
                hard_nodes=frozenset(blk.get("nodes", ())))
            if plan is None:
                return False
            path, edges = plan
            if not self.traffic.claim_route(vid, edges, path[1:]):
                return False
            self.traffic.last_blocked.pop(vid, None)
            if job.last_goal == ph[1] and job.last_path \
                    and job.last_path != path:
                self.traffic.events.append(
                    (self.traffic.t, vid, "reroute",
                     f"{job.last_path[:3]}…→{path[:3]}…"))
            job.last_goal = ph[1]
            job.last_path = path
            job.path, job.edges, job.idx = path, edges, 0
            job.carrying = bool(ph[2])
            job.mode = "nav"
            self._submit_edge(vid, job, 0)
            return True
        if ph[0] == "op_stand_place" or ph[0] == "op_stand_take":
            stand = t["stand"]
            i = self.graph.col_index_of_stand(stand)
            sn = self.graph.node_for(stand)
            if not self.traffic.claim_route(
                    vid, [], [], extra_lanes=(f"col:{i}",),
                    extra_nodes=(sn,)):
                return False
            job.extras = (sn, f"col:{i}")
            self.runner.submit(vid, self._op_stand_segs(vid, stand,
                                                        ph[0] == "op_stand_take"))
            self.quad_of[vid] = 1
            job.mode = "op"
            return True
        if ph[0] == "op_pick":
            self.runner.submit(vid, self._op_pick_segs(
                vid, t["slot"], t.get("level", 0)))
        elif ph[0] == "op_put":
            self.runner.submit(vid, self._op_put_segs(
                vid, t["slot"], t.get("level", 0)))
        elif ph[0] == "op_pin":
            self.runner.submit(vid, [self._op_pin_seg(vid, True)])
        elif ph[0] == "op_unpin":
            self.runner.submit(vid, [self._op_pin_seg(vid, False)])
            job.needs_rev = "rev" in ph[1:]
        job.mode = "op"
        return True

    # ------------------------------------------------------------ 主循环
    def run(self, tasks: list[dict], max_s: float = 2400.0) -> dict:
        """执行任务表 (after: [i] 声明依赖)。返回审计/台账/事件。"""
        n = len(tasks)
        done = [False] * n
        assigned: dict[int, str] = {}
        stall = 0
        while not all(done):
            for i, t in enumerate(tasks):
                if done[i] or i in assigned:
                    continue
                if not all(done[d] for d in t.get("after", [])):
                    continue
                vid = self.assign(t)
                if vid:
                    assigned[i] = vid
                    self.jobs[vid] = Job(vid=vid, task=t)
            self._advance_waiting()
            idle = self.runner.run_until_any_idle(max_s)
            for vid in idle:
                self._on_queue_empty(vid)
            for i, vid in list(assigned.items()):
                if not done[i] and self.jobs.get(vid) is None:
                    done[i] = True
            moving = any(self.runner.busy(v) for v in self.runner.queues)
            waiting = [v for v, j in self.jobs.items()
                       if j is not None and j.mode == "wait"]
            if not idle and not moving and waiting:
                stall += 1
                if stall > 64:
                    raise RuntimeError(
                        f"调度静止 (疑似无路可走): waiting={waiting}")
            else:
                stall = 0
        return {"audit": self.runner.audit,
                "ledger": self.ledger.snapshot(),
                "assignment": self.assignment_log,
                "events": self.traffic.events_view(),
                "stats": dict(self.traffic.stats),
                "op_errors": self.op_errors}

    def _advance_waiting(self):
        for vid in sorted(self.jobs):
            job = self.jobs[vid]
            if job is not None and job.mode == "wait":
                self._try_next_phase(vid)

    def _on_queue_empty(self, vid):
        job = self.jobs[vid]
        if job is None:
            return
        if job.mode == "nav":
            job.idx += 1
            self.traffic.notify_progress(vid, job.path, job.edges, job.idx)
            self.node_of[vid] = job.path[job.idx]
            if job.idx < len(job.edges):
                self._submit_edge(vid, job, job.idx)
                return
            if job.micro:
                job.micro = False
                job.mode = "wait"
                return
            job.phase += 1
            job.mode = "wait"
        elif job.mode == "op":
            self._finish_op(
                vid, final=job.phase == len(_PHASES[job.task["kind"]]) - 2)
            job.phase += 1
            if job.needs_rev:
                job.needs_rev = False
                if self._setup_reverse(vid):
                    return
            job.mode = "wait"

    def _finish_op(self, vid, final: bool):
        """op 相位收尾: 中间相位仅释放外延资源; 末端相位做台账搬动 +
        物理校验 (台账位置必须与托盘实际位姿一致)。"""
        job = self.jobs[vid]
        t = job.task
        kind = t["kind"]
        if job.extras:
            self.traffic.release_resources(vid, job.extras)
            job.extras = ()
        if not final:
            self.traffic.events.append(
                (self.traffic.t, vid, "op_phase", kind))
            return
        if kind == "pick":
            pn = pallet_name_for(f"{t['slot']}-L{t.get('level', 0)}")
            self.ledger.move(pn, f"stand:{t['stand']}")
        elif kind == "putaway":
            pn = self.ledger.at(f"stand:{t['stand']}")
            if pn is None:
                self.op_errors.append(f"{vid}: 台上无托可上架 {t}")
            else:
                self.ledger.move(pn, f"slot:{t['slot']}")
        elif kind in ("haul", "fetch"):
            src = f"stand:{t['stand']}" if kind == "haul" else f"dock:{t['dock']}"
            dst = f"dock:{t['dock']}" if kind == "haul" else f"stand:{t['stand']}"
            pn = self.ledger.at(src)
            if pn is None:
                self.op_errors.append(f"{vid}: {src} 无托 {t}")
            else:
                self.ledger.move(pn, dst)
        self._verify_physical(vid, t, kind)
        self.traffic.events.append((self.traffic.t, vid, "op_done", kind))

    def _verify_physical(self, vid, t, kind):
        """物理即真值: 作业后台账位置必须与托盘实际位姿一致 (±0.02)。"""
        L = self.cell.layout
        level = t.get("level", 0)
        if kind == "pick":
            pn = pallet_name_for(f"{t['slot']}-L{level}")
            ex, ey, ez = (*L.stand_pose(t["stand"])[:2],
                          L.p["stand_top_z"])
        elif kind == "putaway":
            pn = self.ledger.at(f"slot:{t['slot']}")
            sx, sy, _, sz = L.slot_pose(f"{t['slot']}-L{level}")
            ex, ey, ez = sx, sy, sz
        elif kind == "haul":
            pn = self.ledger.at(f"dock:{t['dock']}")
            ex, ey, ez = (*L.stand_pose(t["dock"])[:2],
                          L.p["stand_top_z"])
        else:  # fetch
            pn = self.ledger.at(f"stand:{t['stand']}")
            ex, ey, ez = (*L.stand_pose(t["stand"])[:2],
                          L.p["stand_top_z"])
        if pn is None:
            return
        pos = self.cell.pallet_pose(pn)[0]
        err = math.hypot(pos[0] - ex, pos[1] - ey)
        if err > 0.02 or abs(pos[2] - ez) > 0.02:
            self.op_errors.append(
                f"{vid}: {kind} 物理校验失败 {pn} "
                f"dxy={err:.3f} dz={pos[2] - ez:.3f}")

    def _setup_reverse(self, vid) -> bool:
        """AGV 放托后倒出作业点 (W1 plan_agv_drop 同款): 沿来路倒回一线节点。"""
        job = self.jobs[vid]
        cur = self.node_of[vid]
        cands = []
        if len(job.path) >= 2 and job.path[-2] != cur:
            prev = job.path[-2]
            for n, e in self.graph.adj[cur]:
                if n == prev:
                    cands.append((n, e))
        for n, e in sorted(self.graph.adj[cur]):
            if self.graph.pos[n].kind in ("x", "sm", "stand", "fm") \
                    and all(n != c[0] for c in cands):
                cands.append((n, e))
        for n, e in cands:
            if not self.traffic.claim_route(vid, [e], [n]):
                continue
            job.path, job.edges, job.idx = [cur, n], [e], 0
            job.carrying = False
            job.micro = True
            job.mode = "nav"
            b = self.graph.pos[n]
            axis = "x" if abs(b.x - self.graph.pos[cur].x) >= 1e-9 \
                else "y"
            tgt = b.x if axis == "x" else b.y
            self.runner.submit(vid, [self._move_seg(
                vid, axis, tgt,
                self.cell.vclass(vid)["cruise"] * 0.6)])
            return True
        return False
