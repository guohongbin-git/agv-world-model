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
from dataclasses import dataclass, field, replace

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
    lift_armed: bool = False           # detfix2: 高位越台进行中 (叉已升)
    lift_back: float = 0.0             # 越台前叉关节高 (还原基准)


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
        e = job.edges[i]
        legs = self.planner.route_legs(vid, job.path, job.edges[i:i + 1],
                                       self.quad_of[vid], job.carrying)
        # detfix: AGV 末段进入 stand/dock (钻入托盘下方, 凸台↔底盘名义
        # 仅 3mm 净空) 必须 W1 plan_agv_dock_lift 同款慢速 —— 巡航速
        # 冲入时任何托盘微倾都会让底盘缘撞凸台 (实测 seed7 H04: 亚稳
        # 托盘被顶离条带, 楔死于底盘/条带之间直至求解发散)。
        if (vid in self.cell.agv_ids and i == len(job.edges) - 1
                and self.graph.pos[job.path[-1]].kind in ("stand", "dock")):
            legs = [lg if lg.kind != "move"
                    else replace(lg, speed=min(lg.speed, K["agv_dock"]))
                    for lg in legs]
        # detfix: 叉车穿「有托 stand 节点」必须 W1 leg-12 高位越台 ——
        # 常规 transit 叉顶 0.1045 与台上托盘叉孔天花底 (~0.108) 仅
        # ~2-3mm 裕度, 叉尖挂天花即把整托铲走 (实测 F01 回程绕行北穿
        # H02, 铲走自己刚放的托盘拖至 D_G11 刮坠)。升至 0.125 (>
        # 托盘顶 0.1115, 13mm 裕度) 越过, 再回原高度。
        # detfix2: (a) dock 同享此 guard —— 交付台坐在行通道线上, 已提交
        # 待取托盘 (顶板 0.115) 对 transit 叉底 0.102 同样裸奔 (实测
        # live 5f82cc59 t≈484: F02 列通道穿越 D_G14, 把 ORD-000002
        # 已提交托盘铲离台位 → REP-000004 fetch 物理校验失败)。
        # (b) 升叉必须提前一节点: 预节点处叉尖已前探 0.12m、恰插到台位
        # 托盘孔口 (节点距 0.16 − 叉长 0.12 = 台面缘), 原地升叉会顺孔
        # 天花把整托吊起携走 (修复版回放 t=487 实录: 托盘随叉 0.224
        # 高位沿行道拖行 26s)。提前到上一节点起步升叉, 叉尖入台位区
        # 前即已过孔口高度; 过台后下一边起步回降。
        if vid in self.cell.forklift_ids:
            view = self.cell.veh[vid]

            def held_hit(k):
                if not 0 <= k < len(job.edges):
                    return False
                e_k = job.edges[k]
                return any(self.graph.pos[n].kind in ("stand", "dock")
                           and self.ledger.at(n) is not None
                           for n in (e_k.a, e_k.b))

            cur_hit, nxt_hit = held_hit(i), held_hit(i + 1)
            if job.lift_armed and not (cur_hit or nxt_hit):
                # 过台后首个未触台边起步回降 (记录臂高, 防高位回拉)
                legs = ([Leg("move", "lift",
                             job.lift_back + K["transport_z"],
                             K["lift"], 0.10)] + legs)
                job.lift_armed = False
            elif not job.lift_armed and (cur_hit or nxt_hit):
                job.lift_back = view.q("lift")
                legs = ([Leg("move", "lift", 0.125, K["lift"], 0.10)]
                        + legs)
                job.lift_armed = True
            # armed 且仍在台位区: 保持高叉, 不重复升
        for lg in legs:
            self.runner.submit(vid, [self._seg(vid, lg)])
        self.quad_of[vid] = self._quad_after(job, i, self.quad_of[vid])

    # ---- 作业段 (W1 几何同源) ----

    def _align_segs(self, vid, goal_x):
        """x 向闭环对中段 (nav 粗到位后、作业前的精对位)。

        nav 只保证到节点 (±0.2m 级), 长距离载运 (L3 慢车) settle 残余
        横移会被 op 段的落托/插叉固化 —— 实测 R12-S6-L1→H05 放托
        dxy 0.210 即此缺陷 (W2a 定位)。对中复用 W1「先对中后作业」
        语义: 偏差 >2mm 先补一段 fk_approach 微调, 不改任何已验证
        段几何。"""
        vx = self.cell.veh[vid].xy()[0]
        if abs(goal_x - vx) > 2e-3:
            return [self._move_seg(vid, "x", goal_x, K["fk_approach"])]
        return []

    def _pallet_align_seg(self, vid, pn, axis="x"):
        """对「托盘实测位姿」的横向闭环对中段 (detfix)。

        机理: 对节点/货位几何对中是开环 —— 托盘在架上/台上的实测
        横移 (沉降漂移、前序接触推挤) 不被吸收。实测 seed7 取托
        R12-S7-L2: 叉对货位几何到位, 但托盘横移 ~13mm, 右叉从中板
        上方滑入 (中板顶↔顶板底之间恰容叉体), 微升段压住「顶板」
        而非叉孔天花 → 托盘骑在叉尖上摇晃, 运输中滑脱坠地
        (t=161 坠落, 后续 haul/AGV 链式全废)。段工厂在段启动时刻
        读托盘实测 x, 把叉送到孔位正中 (孔位游隙 ±4mm 之内)。"""
        view = self.cell.veh[vid]

        def build():
            p = self.cell.pallet_pose(pn)[0]
            i = 0 if axis == "x" else 1
            return Ramp(view.xy()[i], p[i], K["fk_approach"], 0.30)
        return (view.a[axis], build, view.off[axis], OP_SETTLE)

    def _pallet_lift_seg(self, vid, pn, dz, speed):
        """举升到「托盘实测座高 + dz」(detfix 闭环插叉/微升高度)。

        开环层高在托盘沉降/微倾时双向吃裕度: W1 +10mm 叉底↔板顶
        仅 2.5mm (沉降 1mm 即擦板推托, W2a 实测 R12-S7 坠托);
        W2a +12mm 顶板裕度又只剩 1mm (微倾即骑顶)。闭环取叉孔窗心
        (板顶 0.004 / 顶板底 0.013, 窗高 9mm, 叉厚 2.5mm → 上下
        各 3.25mm), 且自动跟随实测沉降。微升 dz=+0.019: 叉顶压叉孔
        天花吊起托盘, 底板离梁 6mm —— W1「稳抓天花」语义不变。"""
        view = self.cell.veh[vid]

        def build():
            pz = self.cell.pallet_pose(pn)[0][2]
            return Ramp(view.q("lift"), pz + dz - K["transport_z"],
                        speed, 0.10)
        return (view.a["lift"], build, view.off["lift"], OP_SETTLE)

    def _recenter_segs(self, vid, pn, stand_x, stand_y, rel_z):
        """落托后「取游隙-重吊-平移-释放」精对中 (detfix)。

        机理: 落托横向残余 (叉孔游隙 ±4mm + 插入期侧推) 超出条带
        支撑窗 (名义每侧仅 3mm) 时托盘单轨悬空 → 亚稳态翘置 (实测
        seed7 H04 放托偏东 +28mm 翘置一角, 29s 后 AGV 钻入台下,
        倾斜凸台撞底盘缘, 位置伺服持续推进 → 托盘楔死于底盘/条带
        之间, t=558.77 QACC NaN)。四步收尾: 叉先平移到托盘正上
        (x/y 各自吃掉孔游隙, 天花全接触) → 微升 6mm 重新吊起 (悬
        吊自回平) → 平移到「台心 − 托心前距」(吊起态随动; y 向托
        心 = 叉心 +0.094, 即叉目标为台心 −0.094) → 释放回 rel_z。
        x/y 残余双双收到 <1mm —— 否则顶销 36mm 盘对 boss 40mm 的
        2mm 全接合余量会被残余吃光, 长途载运滑脱坠地 (实测 head-on
        /8task A02 运 R11-S0-L1 至 D_G11 途中坠托)。|偏差|>10mm
        视为异常, 不强拉 (留给台账校验报错)。"""
        view = self.cell.veh[vid]
        off = K["pallet_center_off"]            # 叉心→托心 (面向 +y)

        def toward(cur, tgt):
            return max(-0.010, min(0.010, tgt - cur))

        def xbuild():                           # 叉 x → 托 x (吃游隙)
            fx = view.xy()[0]
            return Ramp(fx, fx + toward(fx, self.cell.pallet_pose(pn)[0][0]),
                        K["fk_approach"], 0.30)

        def ybuild():                           # 叉 y → 托 y − 前距
            fy = view.xy()[1]
            return Ramp(fy, fy + toward(fy, self.cell.pallet_pose(pn)[0][1]
                                         - off), K["fk_approach"], 0.30)
        return [(view.a["x"], xbuild, view.off["x"], OP_SETTLE),
                self._slift(vid, rel_z + 0.006, K["lift_slow"]),
                self._move_seg(vid, "x", stand_x, K["fk_approach"]),
                (view.a["y"], ybuild, view.off["y"], OP_SETTLE),
                self._move_seg(vid, "y", stand_y - off, K["fk_approach"]),
                self._slift(vid, rel_z, K["lift_slow"])]

    def _op_pick_segs(self, vid, slot, level, pn: str | None = None):
        fy = self.graph._slot_face(slot)[2]
        sgn = 1.0 if int(slot.split("S")[1]) < 4 else -1.0
        cp = self.cell.vclass(vid)
        # W1 顺序: 对中 → 升至插入高度 → 插叉 → 微升离梁 → 带载倒退。
        # 对中与高度均闭环到托盘实测位姿 (机理见 _pallet_align_seg /
        # _pallet_lift_seg docstring; 段工厂延迟到段启动时刻读值)。
        # detfix2: pn 可由调度层显式指定 (补货回填的异名托盘 ——
        # 物理occupant即真值, 不再按 slot+level 命名推断)。
        pn = pn or pallet_name_for(f"{slot}-L{level}")
        return [self._pallet_align_seg(vid, pn),
                self._pallet_lift_seg(vid, pn, 0.0095, cp["lift"]),
                self._move_seg(vid, "y", fy + sgn * 0.120, K["fk_insert"]),
                self._pallet_lift_seg(vid, pn, 0.019, K["lift_slow"]),
                self._move_seg(vid, "y", fy, K["fk_approach"]),
                self._slift(vid, K["transport_z"], cp["lift"])]

    def _op_put_segs(self, vid, slot, level, pn: str | None = None):
        z = self.cell.layout.p["levels_z"][level]
        fy = self.graph._slot_face(slot)[2]
        fx = self.graph._slot_face(slot)[1]
        sgn = 1.0 if int(slot.split("S")[1]) < 4 else -1.0
        cp = self.cell.vclass(vid)
        # detfix: 上架对「叉上实测托盘」横向闭环 —— 托盘在叉孔游隙
        # (±4mm) 内的偏移会让轨条 (6mm) 错出架梁 (6mm) 支撑窗
        # (±3mm), 落架即倾斜卡位 (press putaway L0/L1 失败列)。
        # approach 位悬吊态平移 (slot_x − pallet_x), 托随叉走。
        view = self.cell.veh[vid]

        def xbuild():
            px = self.cell.pallet_pose(pn)[0][0]
            fx0 = view.xy()[0]
            d = max(-0.010, min(0.010, fx - px))
            return Ramp(fx0, fx0 + d, K["fk_approach"], 0.30)
        nudge = ([(view.a["x"], xbuild, view.off["x"], OP_SETTLE)]
                 if pn is not None else [])
        return self._align_segs(vid, fx) + [
            self._slift(vid, z + 0.030, cp["lift"]),
            self._move_seg(vid, "y", fy + sgn * 0.120, K["fk_insert"]),
            *nudge,
            # detfix: 落架释放高度 z+0.010 (原 +0.0115): 托盘吊挂原点 =
            # 叉顶−0.013, +0.0115 时原点 = z−0.0015 → 托盘被叉顶与
            # 架梁双向楔住 1.5mm, 退出时摩擦随机胜出把整托拖走
            # (实测 L01 场景 pallet_H05 上架未释放, 随车拖回 park)。
            # +0.010 → 叉体落回窗心 (板顶 +5.4mm / 天花 −2.5mm),
            # 托盘纯由架梁承载, 退叉零干涉 (W1 插叉窗心高度同源)。
            self._slift(vid, z + 0.010, K["place"]),
            self._move_seg(vid, "y", fy, K["fk_approach"]),
            self._slift(vid, K["transport_z"], cp["lift"]),
        ]

    def _op_stand_segs(self, vid, stand, take: bool, pn: str | None = None):
        """交接台放托/取托 (W1 legs 12-18 同源)。

        pn: 实测闭环对象托盘 —— take=True 为台上待取托 (台账),
        take=False (放托) 为本任务所放托盘 (由 pick 任务推出)。
        detfix: 取托对中/高度闭环到实测托盘 (同 _op_pick_segs 机理);
        放托追加 _recenter_segs 精对中防亚稳态翘置。"""
        top = self.cell.layout.p["stand_top_z"]
        hy = self.cell.layout.stand_pose(stand)[1]
        place_z, rel_z = top + 0.012, top + 0.0115
        app2 = hy - (K["approach_gap"] + K["fork_len"] + K["fork_root_x"])
        axle = hy - K["pallet_center_off"]
        clear_y = self.graph.strip_y
        cp = self.cell.vclass(vid)
        col = self.graph.col_index_of_stand(stand)
        gx = self.graph.pos[self.graph.xnode(col, 0)].x
        align = ([self._pallet_align_seg(vid, pn)] if take and pn
                 else self._align_segs(vid, gx))
        segs = align + [
            self._yaw_seg(vid, math.pi / 2),
            self._slift(vid, place_z + 0.030, K["place"])]   # 高位越条带
        if take:
            # detfix: 取托段原序「高位直接进给到 axle 再举升」从未进入
            # 叉孔窗 (approach 0.125 / lift 0.137 均高于托盘顶板
            # 0.1115), 叉从托盘上方空过 —— 抓取无效, press 补货
            # (putaway) 全列物理失败的主因。修正: 高位越条带 → 台
            # 南 approach 降入窗心 (实测座高 +0.0095, 同 pick 闭环)
            # → 插叉至 axle → 微升吊起 +0.0445 (吊起后托心
            # 0.0935+0.0445−0.013=0.125, 凸台底恰离垫块顶面)
            # → 带载南退。
            win = (self._pallet_lift_seg(vid, pn, 0.0095, K["place"])
                   if pn else self._slift(vid, place_z - 0.0025,
                                          K["place"]))
            hang = (self._pallet_lift_seg(vid, pn, 0.0445, K["lift_slow"])
                    if pn else self._slift(vid, place_z + 0.042,
                                           K["lift_slow"]))
            segs += [self._move_seg(vid, "y", app2, K["fk_approach"]),
                     win,
                     self._move_seg(vid, "y", axle, K["fk_insert"]),
                     hang,
                     self._move_seg(vid, "y", clear_y, K["fk_approach"]),
                     self._slift(vid, K["transport_z"], cp["lift"])]
            return segs
        segs += [self._move_seg(vid, "y", app2, K["fk_approach"]),
                 self._move_seg(vid, "y", axle, K["fk_insert"]),
                 self._slift(vid, place_z, K["place"]),
                 self._slift(vid, rel_z, K["lift_slow"])]
        if pn is not None:
            segs += self._recenter_segs(
                vid, pn, *self.cell.layout.stand_pose(stand)[:2], rel_z)
        segs += [self._move_seg(vid, "y", clear_y, K["fk_approach"]),
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
            # detfix: 载运 AGV 的路由必须绕开「有托 stand/dock」节点
            # —— 列通道图在 stand 节点链式穿行, 空车钻台下是设计
            # 行为 (凸台↔底盘 3mm), 但车顶载着的托盘 (凸台下探
            # 0.092) 会与台上托盘 (顶板 0.1115) 在台位区互撞 (实测
            # seed7 fixture: A02 载 R11-S0-L2 绕行西线穿 H01, 凸台
            # ×H01 预载托盘叠压 → t=607.6 QACC NaN)。终点自身除外
            # (haul/fetch 的目的台就是取/放托位)。
            hard = set(blk.get("nodes", ()))
            if ph[2] and vid in self.cell.agv_ids:
                held = {v for v in self.ledger.loc.values()
                        if v.startswith(("stand:", "dock:"))}
                held.discard(goal)
                hard |= held
            plan = self.graph.plan(
                self.node_of[vid], goal, kind, self.quad_of[vid],
                frozenset(lanes), frozenset(nodes),
                hard_lanes=frozenset(blk.get("lanes", ())),
                hard_nodes=frozenset(hard))
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
            # detfix: 交接台作业对象托盘 → 实测闭环对中/精对中
            # (取托=台上托; 放托=pick 任务所取托盘)
            take = ph[0] == "op_stand_take"
            pn = (self.ledger.at(f"stand:{stand}") if take
                  else self._pick_pallet(t)
                  if t["kind"] == "pick" else None)
            self.runner.submit(vid, self._op_stand_segs(vid, stand,
                                                        take, pn))
            self.quad_of[vid] = 1
            job.mode = "op"
            return True
        if ph[0] == "op_pick":
            # detfix: 槽位作业占线锁 —— 叉车在 sf 面位作业时, 车体
            # (长 0.22 沿巷道向) 横占行通道, 而 nav 到位即释放全部
            # lane → 过路车会从作业叉车底盘上碾过 (实测 seed7: F03
            # 西行穿 R12-S7 作业区, 与 F02 底盘互穿 62mm, 把插叉中
            # 的 F02 推偏 13mm → 叉骑顶板 → 托盘坠落 → 空运/楔墙
            # → 求解发散的整条链)。作业期持有被穿越的 row:j (与
            # op_stand 持 col:i 同款; 无等待依赖, 不破坏无死锁定理)。
            j = self.graph._line_of_slot(
                int(t["slot"][1]), int(t["slot"].split("S")[1]) < 4)
            fid = self.graph._slot_face(t["slot"])[0]
            if not self.traffic.claim_route(
                    vid, [], [], extra_lanes=(f"row:{j}",),
                    extra_nodes=(fid,)):
                return False
            # detfix: 不入 extras —— op 结束即释放 row:j 会重开「无锁
            # 驻巷」窗口 (实测: F04 取 R11-S6 完毕待退, A01 沿 row:2
            # 西行穿其面位, 两车底盘对磨 10-12N, 台上托盘被拍飞)。
            # row:j 由 sf 到位 keep 保留, 持续到退场路由首次
            # notify_progress 释放; claim 仅作入口串行化 (自持幂等)。
            job.extras = ()
            self.runner.submit(vid, self._op_pick_segs(
                vid, t["slot"], t.get("level", 0),
                self._pick_pallet(t)))
        elif ph[0] == "op_put":
            j = self.graph._line_of_slot(
                int(t["slot"][1]), int(t["slot"].split("S")[1]) < 4)
            fid = self.graph._slot_face(t["slot"])[0]
            if not self.traffic.claim_route(
                    vid, [], [], extra_lanes=(f"row:{j}",),
                    extra_nodes=(fid,)):
                return False
            job.extras = ()          # 同 op_pick: row:j 保留至退场 (detfix)
            self.runner.submit(vid, self._op_put_segs(
                vid, t["slot"], t.get("level", 0),
                self.ledger.at(f"stand:{t['stand']}")))
        if ph[0] == "op_pin":
            # 顶升前按托盘实测位姿闭环对中 (物理即真值: 目标=接触动力学
            # 产出的托盘位姿): place 残余 ±20mm 会使 pin(36mm) 对 boss
            # (40mm) 偏载滑脱 —— 先把 AGV x/y 微调到托盘中心 ±2mm。
            # detfix: 台上无托 (上游 pick 失败的空 haul) 时销盘若照常
            # 顶起, 偏置销盘 (r18 vs 孔 24mm 半宽) 会撬条带 → 倒车把
            # 底盘拖进台墙楔死 (实测 seed7 A04 穿墙 70mm, 3e9N 静楔
            # 直至求解器爆) —— 无托也必须对中到台位几何中心再顶销。
            src = t["stand"] if t["kind"] == "haul" else t["dock"]
            pn = self.ledger.at(f"stand:{src}" if t["kind"] == "haul"
                                else f"dock:{src}")
            segs = []
            view = self.cell.veh[vid]
            if pn is not None:
                px, py, _ = self.cell.pallet_pose(pn)[0]
            else:
                px, py, _ = self.cell.layout.stand_pose(src)
            dx, dy = px - view.xy()[0], py - view.xy()[1]
            # detfix: 对中限幅 ±6mm —— 销盘 (r18) 顶起位 z 0.088-0.092
            # 与条带底 (0.091) 有 1mm 重叠, 销心横偏 >6mm (孔半宽 24
            # − 销半径 18) 即挂钩条带 → 原地转向被撬偏楔墙 (实测
            # seed7 三台 AGV 冻死 y≈-1.99)。放托经 _recenter_segs
            # 已 ±1mm 级, 限幅仅兜底异常布局 (H 台墙余量 ±20mm 仍满足)。
            dx = max(-0.006, min(0.006, dx))
            dy = max(-0.006, min(0.006, dy))
            # 死区 1mm (< boss/销盘 2mm 全接合余量): 残余必须收进
            # 余量内, 否则长途载运转弯时托盘从销盘滑脱 (实测坠托)。
            if abs(dx) > 1e-3:
                segs.append(self._move_seg(vid, "x", view.xy()[0] + dx,
                                           K["fk_approach"]))
            if abs(dy) > 1e-3:
                segs.append(self._move_seg(vid, "y", view.xy()[1] + dy,
                                           K["fk_approach"]))
            self.runner.submit(vid, segs + [self._op_pin_seg(vid, True)])
        elif ph[0] == "op_unpin":
            # detfix: 放托前闭环对中 —— nav 到 dock/stand 节点的残余
            # 加长途载运中托盘在销面的微移, 使销上托盘偏出垫块孔
            # (>6mm) → 降销时 boss 挂垫块边缘、托盘被推下条带坠地
            # (实测 seed7 fixture: A02 放 D_G11, strip↔boss 23.5N,
            # 托盘滑落至 2.2m 外)。读「销上托盘实测位姿」, 平移
            # (台心 − 托心): 托随销面摩擦随动, 落点即台心。
            dst = t["dock"] if t["kind"] == "haul" else t["stand"]
            px, py, _ = self.cell.layout.stand_pose(dst)
            src = f"stand:{t['stand']}" if t["kind"] == "haul" \
                else f"dock:{t['dock']}"
            pn = self.ledger.at(src)
            view = self.cell.veh[vid]
            segs = []
            if pn is not None:
                qx, qy, _ = self.cell.pallet_pose(pn)[0]
            else:
                qx, qy = view.xy()
            dx = max(-0.010, min(0.010, px - qx))
            dy = max(-0.010, min(0.010, py - qy))
            if abs(dx) > 1e-3:
                segs.append(self._move_seg(vid, "x", view.xy()[0] + dx,
                                           K["fk_approach"]))
            if abs(dy) > 1e-3:
                segs.append(self._move_seg(vid, "y", view.xy()[1] + dy,
                                           K["fk_approach"]))
            self.runner.submit(vid, segs + [self._op_pin_seg(vid, False)])
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

    @staticmethod
    def _pick_pallet(t):
        """pick 作业对象托盘 (detfix2): 调度层显式重绑 (补货回填的异名
        占位托盘) 优先, 否则按 slot+level 命名推断 (初始播种恒同名)。"""
        return t.get("pallet") or pallet_name_for(
            f"{t['slot']}-L{t.get('level', 0)}")

    def _finish_op(self, vid, final: bool):
        """op 相位收尾: 中间相位仅释放外延资源; 末端相位做台账搬动 +
        物理校验 (台账位置必须与托盘实际位姿一致)。

        detfix2: 物理校验失败即回滚本腿台账搬动 —— 物理即真值, 托盘
        未被证实到达 dst, 账本不得断言 dst (否则幻影占位把 stand/dock
        永久锁死, 后续链按死账取托, 账实失步级联)。"""
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
        pn, prev_loc = None, None
        if kind == "pick":
            pn = self._pick_pallet(t)
            prev_loc = self.ledger.loc.get(pn)
            self.ledger.move(pn, f"stand:{t['stand']}")
        elif kind == "putaway":
            pn = self.ledger.at(f"stand:{t['stand']}")
            if pn is None:
                self.op_errors.append(f"{vid}: 台上无托可上架 {t}")
            else:
                prev_loc = f"stand:{t['stand']}"
                self.ledger.move(pn, f"slot:{t['slot']}")
        elif kind in ("haul", "fetch"):
            src = f"stand:{t['stand']}" if kind == "haul" else f"dock:{t['dock']}"
            dst = f"dock:{t['dock']}" if kind == "haul" else f"stand:{t['stand']}"
            pn = self.ledger.at(src)
            if pn is None:
                self.op_errors.append(f"{vid}: {src} 无托 {t}")
            else:
                prev_loc = src
                self.ledger.move(pn, dst)
        if pn is not None \
                and not self._verify_physical(vid, t, kind) \
                and prev_loc is not None:
            self.ledger.loc[pn] = prev_loc       # detfix2: 失败不认账
        self.traffic.events.append((self.traffic.t, vid, "op_done", kind))

    def _verify_physical(self, vid, t, kind) -> bool:
        """物理即真值: 作业后台账位置必须与托盘实际位姿一致 (±0.02)。

        返回 True=通过; 失败时照旧记 op_errors (spec 失败判据), 并由
        _finish_op 据返回值回滚本腿台账搬动 (detfix2)。"""
        L = self.cell.layout
        level = t.get("level", 0)
        if kind == "pick":
            pn = self._pick_pallet(t)
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
            return False
        pos = self.cell.pallet_pose(pn)[0]
        err = math.hypot(pos[0] - ex, pos[1] - ey)
        if err > 0.02 or abs(pos[2] - ez) > 0.02:
            self.op_errors.append(
                f"{vid}: {kind} 物理校验失败 {pn} "
                f"dxy={err:.3f} dz={pos[2] - ez:.3f}")
            return False
        return True

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
