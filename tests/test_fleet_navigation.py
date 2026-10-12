"""M2 Wave2 — 2-DOF 网格导航 + 巷道/交叉口锁矩阵 + 并发任务驱动 验收。

验收项 (对应规格):
- 图全部从 LAYOUT 派生 (零硬编码坐标), 拓扑/几何断言
- 单车 A* 最短性 sanity (与同图 Dijkstra 对拍) + 转弯惩罚可见
- 动态锁惩罚: 主通道被占 → 自动绕行 (路径改变可断言), 完成时间变长
- 双车对穿/争用: 一车等待, 等待有事件可查, 零碰撞不死锁
- 8 任务并发 (4 pick + 4 haul): 全部完成, 零碰撞 (pen<5e-4), 锁矩阵零死锁
- 确定性: 同 seed 两次 run 结果一致 (台账/事件/托盘终位)
"""
from __future__ import annotations

import math
from collections import Counter

import pytest

from sim.controllers import Ramp
from sim.mission import MissionRunner
from sim.navigation import FleetRunner, MotionPlanner, NavGraph, a_star
from sim.pallet_support import ledger_from_cell
from sim.traffic import TrafficManager
from sim.warehouse_cell import (K, VEHICLE_CLASSES, WarehouseCell,
                                pallet_name_for)

PEN_LIMIT = 5e-4


# ---------------------------------------------------------------- 图派生
@pytest.fixture(scope="module")
def graph():
    return NavGraph(WarehouseCell(timestep_s=0.001, seed=3).layout)


def test_graph_derives_from_layout(graph):
    g = graph
    L = g.layout
    kinds = Counter(n.split(":")[0] for n in g.pos)
    assert kinds["sf"] == 128 and kinds["sm"] == 80 and kinds["stg"] == 64
    assert kinds["dock"] == 16 and kinds["stand"] == 5
    assert kinds["x"] == 25 and kinds["pk"] == 8 and kinds["fm"] == 3
    # 关键坐标全部来自布局派生
    for sid, (x, y, _) in L.handoff_poses().items():
        n = g.pos[f"stand:{sid}"]
        assert (n.x, n.y) == pytest.approx((x, y))
    for did, (x, y, _) in L.dock_poses().items():
        n = g.pos[f"dock:{did}"]
        assert (n.x, n.y) == pytest.approx((x, y))
    # 槽位面 = W1 插叉接近公式 (edge ± (gap+len+root))
    off = K["approach_gap"] + K["fork_len"] + K["fork_root_x"]
    for slot in ("R11-S0", "R23-S3", "R42-S4", "R34-S7"):
        r, c, k = int(slot[1]), int(slot[2]), int(slot.split("S")[1])
        sy = L.slot_pose(f"{slot}-L0")[1]
        fy = g.pos[f"sf:{slot}"].y
        assert fy == pytest.approx(sy - 0.040 - off if k < 4
                                   else sy + 0.040 + off)
    # 接近点 = 真实巷道中心线 (row_aisle_ys)
    stg = g.pos["stg:2:2:0"]
    assert stg.y == pytest.approx(L.row_aisle_ys()[1])
    # 全部边轴对齐且相邻节点距离为正
    for a in g.adj:
        for b, e in g.adj[a]:
            pa, pb = g.pos[a], g.pos[b]
            assert (abs(pa.x - pb.x) < 1e-9) or (abs(pa.y - pb.y) < 1e-9)
            assert math.hypot(pa.x - pb.x, pa.y - pb.y) > 1e-6


def test_vehicles_match_layout_classes():
    c = WarehouseCell(timestep_s=0.001, seed=3)
    assert len(c.veh) == 8 and c.model.nu == 32
    for vid, cls in c.layout.p["vehicle_class"].items():
        cp = c.vclass(vid)
        assert cp is VEHICLE_CLASSES[cls]
    assert [c.vclass(v)["cruise"] for v in ("F01", "F02", "F03")] == \
        [0.35, 0.25, 0.15]


# ---------------------------------------------------------------- A*
def _dijkstra_reference(g, start, goal, veh_kind, start_quad=0):
    """同图同代价规则的 Dijkstra (无启发式) —— A* 最短性对拍基准。"""
    import heapq
    ctr = 0
    pq = [(0.0, 0, (start, start_quad))]
    best = {(start, start_quad): 0.0}
    while pq:
        gcost, _, state = heapq.heappop(pq)
        if gcost > best.get(state, 1e18) + 1e-12:
            continue
        n, q = state
        if n == goal:
            return gcost
        for nb, e in g.adj[n]:
            nq = q if e.strafe else (
                (0 if (abs(g.pos[nb].x - g.pos[n].x) >= abs(g.pos[nb].y - g.pos[n].y)
                       and g.pos[nb].x > g.pos[n].x)
                 else 2 if abs(g.pos[nb].x - g.pos[n].x) >= abs(g.pos[nb].y - g.pos[n].y)
                 else 1 if g.pos[nb].y > g.pos[n].y else 3))
            if q != nq and g.turn_blocked(n, veh_kind):
                continue
            c = math.hypot(g.pos[nb].x - g.pos[n].x, g.pos[nb].y - g.pos[n].y)
            if q != nq:
                c += 0.15
            if e.strafe:
                c += 0.30
            if gcost + c < best.get((nb, nq), 1e18) - 1e-12:
                best[(nb, nq)] = gcost + c
                ctr += 1
                heapq.heappush(pq, (gcost + c, ctr, (nb, nq)))
    raise ValueError("unreachable")


ODS = [("pk:F01", "sf:R11-S0"), ("pk:F02", "sf:R12-S4"),
       ("pk:F03", "sf:R34-S6"), ("pk:F01", "stand:H02"),
       ("pk:A01", "dock:D_G12"), ("pk:A04", "dock:D_G41"),
       ("pk:F04", "sf:R21-S5"), ("pk:A02", "dock:D_G33")]


def test_astar_matches_dijkstra(graph):
    g = graph
    for s, t in ODS:
        kind = "agv" if s.startswith("pk:A") else "forklift"
        _, _, cost = a_star(g, s, t, kind, 0)
        ref = _dijkstra_reference(g, s, t, kind, 0)
        assert cost == pytest.approx(ref, abs=1e-6), (s, t)


def test_turn_penalty_visible(graph):
    """同一 OD: 零转弯罚走纯最短距离路; 带罚选少转弯路 —— 惩罚生效可见。"""
    g = graph
    # pk:A02 → dock:D_G11: 北线绕行 vs 内巷道直穿 (转弯数不同)
    _, _, c_turn = a_star(g, "pk:A02", "dock:D_G11", "agv", 0)
    _, _, c_flat = a_star(g, "pk:A02", "dock:D_G11", "agv", 0,
                          turn_penalty=0.0)
    assert c_turn >= c_flat - 1e-9
    # 找一个两条代价不同的 OD 证明排序可反转
    flips = 0
    for s, t in ODS:
        kind = "agv" if s.startswith("pk:A") else "forklift"
        _, e1, c1 = a_star(g, s, t, kind, 0, turn_penalty=0.0)
        _, e2, c2 = a_star(g, s, t, kind, 0, turn_penalty=1.5)
        turns1 = sum(1 for e in e1 if not e.strafe)
        if c1 < c2 - 1e-6:
            flips += 1
    assert flips >= 1, "转弯惩罚未在任何 OD 上改变最优路径"


def test_lock_penalty_detour_route_change(graph):
    """col:2 被占 → F03 到 R23-S4 的路径必须绕开 col:2 (路径改变可断言)。"""
    g = graph
    base_path, _, base_cost = a_star(g, "pk:F03", "sf:R23-S4", "forklift", 0)
    assert any(n.startswith("x:2:") for n in base_path), "基线路径应走 col:2"
    det_path, det_edges, det_cost = a_star(
        g, "pk:F03", "sf:R23-S4", "forklift", 0,
        occ_lanes=frozenset({"col:2"}))
    assert "col:2" not in {e.lane for e in det_edges} \
        and not any("col:2" in e.crosses for e in det_edges)
    assert det_path != base_path and det_cost > base_cost
    # 硬排除全部行通道 → 无路可走 (等待语义的规划侧证据)
    with pytest.raises(ValueError):
        a_star(g, "pk:F03", "sf:R23-S4", "forklift", 0,
               hard_lanes=frozenset({f"row:{j}" for j in (0, 2, 3, 4, 5)}))


def test_turn_envelope_constraints(graph):
    """叉车路径不得在 sm/sf/stand 原地转弯 (扫掠包络); AGV 不受限。"""
    g = graph
    for s, t in ODS:
        kind = "agv" if s.startswith("pk:A") else "forklift"
        path, edges, _ = a_star(g, s, t, kind, 0)
        q = 0
        for i, e in enumerate(edges):
            nq = 0 if e.strafe else None
            if not e.strafe:
                a, b = g.pos[e.a], g.pos[e.b]
                nq = (0 if b.x > a.x else 2) if abs(b.x - a.x) >= abs(b.y - a.y) \
                    else (1 if b.y > a.y else 3)
            if q != nq and not e.strafe:
                assert not g.turn_blocked(e.a, kind), (s, t, e.a)
            q = nq if nq is not None else q


# ---------------------------------------------------------------- 锁矩阵
def test_claim_atomicity_and_wait_event(graph):
    tm = TrafficManager(graph)
    path, edges, _ = a_star(graph, "pk:A01", "dock:D_G12", "agv", 0)
    assert tm.claim_route("A01", edges, path[1:])
    # 对向申请同列通道 → 整体失败 (一个不取) + 等待事件含持有者
    path_b, edges_b, _ = a_star(graph, "pk:A02", "dock:D_G12", "agv", 0)
    held_before = dict(tm.lane_holder), dict(tm.node_holder)
    assert not tm.claim_route("A02", edges_b, path_b[1:])
    assert dict(tm.lane_holder) == held_before[0] \
        and dict(tm.node_holder) == held_before[1]
    waits = [e for e in tm.events_view() if e[1] == "A02" and e[2] == "claim_wait"]
    assert waits and any("col:0" in str(w[3]) or "col:2" in str(w[3])
                         or "row:2" in str(w[3]) for w in waits)
    # 释放后可取得
    tm.release_all("A01")
    assert tm.claim_route("A02", edges_b, path_b[1:])


def test_retention_release_chain(graph):
    """沿路由逐节点推进: 滞后释放 + 锚点保持 + 终点清 lane。"""
    tm = TrafficManager(graph)
    path, edges, _ = a_star(graph, "pk:A01", "dock:D_G12", "agv", 0)
    assert tm.claim_route("A01", edges, path[1:])
    for i in range(1, len(path)):
        tm.notify_progress("A01", path, edges, i)
    # 终点: lane 全清, 仅持 dock 节点
    assert not [r for r in tm.held["A01"] if r in tm.lane_holder]
    assert tm.held["A01"] == {"dock:D_G12"}
    # 他车此时可走同线 (lane 已释放) 但不可穿 dock 节点
    assert "row:2" not in tm.lane_holder
    path2, edges2, _ = a_star(graph, "pk:A02", "dock:D_G11", "agv", 0)
    assert not tm.claim_route("A02", edges2, path2[1:])   # D_G12 占用


def test_randomized_no_deadlock(graph):
    """随机 claim/推进/释放压力: 原子性不变量 + 全部任务最终完成 (无死锁)。"""
    import random
    rng = random.Random(11)
    tm = TrafficManager(graph)
    vids = [f"V{i}" for i in range(8)]
    ods = [(a, b) for a in graph.pos for b in graph.pos
           if a.startswith(("pk:", "dock:", "x:")) and
           b.startswith(("dock:", "stand:", "sf:"))]
    routes = {}
    progress = {v: 0 for v in vids}
    for step in range(600):
        for v in vids:
            if v not in routes and rng.random() < 0.4:
                s, t = rng.choice(ods)
                kind = "agv" if "A" in v else "forklift"
                try:
                    p, e, _ = a_star(graph, s, t, kind, 0)
                except ValueError:
                    continue
                if tm.claim_route(v, e, p[1:]):
                    routes[v] = (p, e)
            elif v in routes:
                p, e = routes[v]
                progress[v] += rng.randint(1, 2)
                i = min(progress[v], len(p) - 1)
                tm.notify_progress(v, p, e, i)
                if i >= len(p) - 1:
                    tm.release_all(v)
                    routes.pop(v)
                    progress[v] = 0
        # 原子性不变量: 任何时刻持有者唯一
        for r, h in tm.lane_holder.items():
            assert tm.node_holder.get(r) in (None, h)
    # 收尾: 停止新增申请, 纯排空 —— 等待者最终可取得 → 无死锁
    for _ in range(20000):
        if not routes:
            break
        for v in list(routes):
            p, e = routes[v]
            progress[v] += 1
            i = min(progress[v], len(p) - 1)
            tm.notify_progress(v, p, e, i)
            if i >= len(p) - 1:
                tm.release_all(v)
                routes.pop(v)
                progress[v] = 0
    assert not routes, "收尾排空后仍有未完成路由 → 死锁"


# ---------------------------------------------------------------- 物理并发
@pytest.fixture(scope="module")
def fleet8_result():
    """8 任务并发 (4 pick + 4 haul, 轻重货位混合)。"""
    cell = WarehouseCell(timestep_s=0.001, seed=7)
    cell.warmup(0.5)
    led = ledger_from_cell(cell)
    slots = sorted(k for k, v in led.loc.items() if v.startswith("slot:")
                   and "-S" in v)
    # 目的台必须取台账中的空台 (H01 预载占有); 断言防线防空转假通过
    occupied = {v for v in led.loc.values() if v.startswith("stand:")}
    free_nodes = sorted(f"stand:{sid}"
                        for sid in cell.layout.p["handoff_ids"])
    # 任务字段用业务名 ('H02'); node_for/col_index_of_stand 自行转换
    free_stands = sorted(n.split(":", 1)[1]
                         for n in free_nodes if n not in occupied)[:4]
    assert len(free_stands) == 4 and slots
    picks = []
    for i, name in enumerate(slots[:4]):
        slot = led.loc[name][len("slot:"):]
        level = int(name[-1])
        picks.append({"kind": "pick", "slot": slot, "level": level,
                      "stand": free_stands[i]})
    stands = [t["stand"] for t in picks]
    docks = ("D_G11", "D_G22", "D_G33", "D_G44")
    tasks = picks + [
        {"kind": "haul", "stand": stands[i], "dock": docks[i],
         "after": [i]} for i in range(4)]
    mr = MissionRunner(cell)
    res = mr.run(tasks, max_s=2400)
    return cell, mr, res, picks, docks


def test_8task_all_complete(fleet8_result):
    cell, mr, res, picks, docks = fleet8_result
    assert res["op_errors"] == []
    assert res["assignment"], "任务必须真实分配 (防空转假通过)"
    # 全车回位
    for vid in cell.veh:
        px, py, _ = cell.layout.p["park"][vid]
        vx, vy = cell.veh[vid].xy()
        assert math.hypot(vx - px, vy - py) < 0.02, vid
    # 4 托全部落位 dock (对位 ±5mm, 条带支撑 ≥85%, 脱销)
    for i, t in enumerate(picks):
        pn = pallet_name_for(f"{t['slot']}-L{t['level']}")
        dock = docks[i]
        tx, ty, _ = cell.layout.stand_pose(dock)
        pos = cell.pallet_pose(pn)[0]
        assert abs(pos[0] - tx) <= 0.005 and abs(pos[1] - ty) <= 0.005, pn
        assert abs(cell.pallet_bottom_z(pn) - 0.095) <= 0.002
        assert cell.surface_load_ratio(pn, cell.stand_gids[dock]) >= 0.85
        agv = cell.agvs[f"A0{i + 1}"]
        assert not cell.pallet_touches(pn, {agv.pin_gid})


def test_8task_zero_collision_no_deadlock(fleet8_result):
    cell, mr, res, picks, docks = fleet8_result
    assert res["audit"]["max_pen"] < PEN_LIMIT
    assert res["audit"]["violations"] == []
    waits = [e for e in res["events"] if e[2] == "claim_wait"]
    assert len(waits) >= 1, "8 车并发应出现锁等待 (争用被串行化)"
    # 等待必有释放后的成功 (吞吐成立): 每 assign 都有 op_done 收尾
    assert sum(1 for e in res["events"] if e[2] == "op_done") == 8


def test_8task_determinism():
    """同 seed 两次 run: 台账/分配/事件序列/托盘终位全一致。

    与 fleet8 fixture 同任务 (空台交付 + haul 链), 并加非空/穿透防线 ——
    两次崩溃逐位一致也会"相等", 故必须排除空转假通过。
    """
    runs = []
    for _ in range(2):
        cell = WarehouseCell(timestep_s=0.001, seed=7)
        cell.warmup(0.5)
        led = ledger_from_cell(cell)
        slots = sorted(k for k, v in led.loc.items()
                       if v.startswith("slot:") and "-S" in v)
        occupied = {v for v in led.loc.values() if v.startswith("stand:")}
        free_nodes = sorted(f"stand:{sid}"
                            for sid in cell.layout.p["handoff_ids"])
        free_stands = sorted(n.split(":", 1)[1]
                             for n in free_nodes
                             if n not in occupied)[:4]
        assert len(free_stands) == 4 and slots
        # 同格不同层并发取托会共享面位节点且两叉同空间交叉 (物理扰动,
        # 真实仓库亦由库位互斥禁止) —— 与 head-on 用例同款约束: 每格只挑一层
        tasks, used_slots = [], set()
        for name in slots:
            if len(tasks) == 4:
                break
            slot = led.loc[name][len("slot:"):]
            if slot in used_slots:
                continue
            used_slots.add(slot)
            tasks.append({"kind": "pick", "slot": slot,
                          "level": int(name[-1]),
                          "stand": free_stands[len(tasks)]})
        docks = ("D_G11", "D_G22", "D_G33", "D_G44")
        tasks += [{"kind": "haul", "stand": free_stands[i], "dock": docks[i],
                   "after": [i]} for i in range(4)]
        mr = MissionRunner(cell)
        res = mr.run(tasks, max_s=2400)
        assert res["op_errors"] == []
        assert res["assignment"] and res["audit"]["max_pen"] < PEN_LIMIT
        finals = {n: tuple(round(float(x), 6) for x in cell.pallet_pose(n)[0])
                  for n in cell.pallets}
        runs.append((res["ledger"], res["assignment"],
                     [e[:3] + (e[3],) for e in res["events"]],
                     finals, round(res["audit"]["max_pen"], 9),
                     round(cell.data.time, 3)))
    assert runs[0] == runs[1]


# ---------------------------------------------------------------- 双车对穿
def test_two_agv_head_on_wait_and_complete():
    """双车对穿同一行通道 (反向): 一车等待 (事件可查), 零碰撞不死锁。

    A01: H02 → D_G14 (row:2 东行长穿); A02: H04 → D_G11 (row:2 西行长穿)
    —— 两路由在 row:2 完全交叠, 锁矩阵串行化: 后申请者等待先到者放托退场。
    """
    cell = WarehouseCell(timestep_s=0.001, seed=7)
    cell.warmup(0.5)
    led = ledger_from_cell(cell)
    racked = sorted(k for k, v in led.loc.items() if v.startswith("slot:"))
    # 两 pick 取不同货位 (同 slot 双层双 pick 会共享同一面位节点, 串行
    # 依赖面位锁且上下层托盘相互扰动, 不作为对穿场景的载荷)
    s1 = led.loc[racked[0]][len("slot:"):]
    k2 = next(k for k in racked[1:]
              if led.loc[k][len("slot:"):].split("-")[0]
              != s1.split("-")[0])
    s2 = led.loc[k2][len("slot:"):]
    tasks = [
        {"kind": "pick", "slot": s1, "level": int(racked[0][-1]),
         "stand": "H02"},
        {"kind": "pick", "slot": s2, "level": int(k2[-1]),
         "stand": "H04"},
        {"kind": "haul", "stand": "H02", "dock": "D_G14", "after": [0]},
        {"kind": "haul", "stand": "H04", "dock": "D_G11", "after": [1]},
    ]
    mr = MissionRunner(cell)
    res = mr.run(tasks, max_s=1500)
    assert res["op_errors"] == []
    waits = [e for e in res["events"] if e[2] == "claim_wait"]
    assert waits, "对穿必须出现锁等待事件"
    for e in waits:
        assert e[3], "等待事件必须携带阻塞资源@持有者"
    assert any("row:2" in str(e[3]) for e in waits), \
        f"等待应发生在 row:2 巷道: {waits[:3]}"
    assert res["audit"]["max_pen"] < PEN_LIMIT
    # 两托分别落位
    pn1 = f"pallet_{s1.replace('-', '_')}_L{racked[0][-1]}"
    pn2 = f"pallet_{s2.replace('-', '_')}_L{k2[-1]}"
    for pn, dock in ((pn1, "D_G14"), (pn2, "D_G11")):
        tx, ty, _ = cell.layout.stand_pose(dock)
        pos = cell.pallet_pose(pn)[0]
        assert abs(pos[0] - tx) <= 0.005 and abs(pos[1] - ty) <= 0.005, pn
        assert cell.surface_load_ratio(pn, cell.stand_gids[dock]) >= 0.85


# ---------------------------------------------------------------- 绕行物理
def test_detour_when_column_held():
    """col:2 被占 (静态占用模型) → F03 取货任务自动绕行并完成。"""
    cell = WarehouseCell(timestep_s=0.001, seed=7)
    cell.warmup(0.5)
    mr = MissionRunner(cell)
    # 取一个已播种的后排货位 (S4-7, 走 staging 接近); F03 (L3 慢车) 直取
    led = mr.ledger
    cands = [v for k, v in sorted(led.loc.items())
             if v.startswith("slot:") and int(v.split("S")[1].split("-")[0]) >= 4]
    slot = cands[0][len("slot:"):]
    level = int([k for k, v in led.loc.items() if v == cands[0]][0][-1])
    # 基线须经过 col:2; 否则换一个使断言有意义的货位 (同列约束下取 c=2/3 组)
    base = a_star(mr.graph, "pk:F03", f"sf:{slot}", "forklift", 0)
    assert any(n.startswith("x:2:") for n in base[0]), slot
    mr.traffic.lane_holder["col:2"] = "OCC"      # 外部占用 (W3 语义预留位)
    res = mr.run([{"kind": "pick", "slot": slot, "level": level,
                   "stand": "H05"}], max_s=900)
    # 任务完成且路径不含 col:2
    assert res["op_errors"] == []
    pn = f"pallet_{slot.replace('-', '_')}_L{level}"
    assert mr.ledger.loc[pn] == "stand:H05"
    hx, hy, _ = cell.layout.stand_pose("H05")
    pos = cell.pallet_pose(pn)[0]
    assert abs(pos[0] - hx) <= 0.005 and abs(pos[1] - hy) <= 0.005
    assert res["audit"]["max_pen"] < PEN_LIMIT
    # 占用释放后, 规划器对新 OD 不再回避 (占用账本未被任务污染)
    assert mr.traffic.lane_holder.get("col:2") == "OCC"


def test_class_speeds_reflected_in_mission():
    """异构等级: 同 OD 规划, L1 与 L3 的段速度不同 (物理参数化生效)。"""
    cell = WarehouseCell(timestep_s=0.001, seed=3)
    mr = MissionRunner(cell)
    from sim.navigation import NavGraph as _G
    goal = mr.graph.node_for("D_G12")
    legs = {}
    for vid in ("A01", "A03"):        # L1 vs L2
        kind = "agv"
        path, edges, _ = a_star(mr.graph, "pk:A01" if vid == "A01"
                                else "pk:A03", goal, kind, 0)
        pl = MotionPlanner(cell, mr.graph)
        lg = pl.route_legs(vid, path, edges, 0)
        moves = [l.speed for l in lg if l.kind == "move"]
        legs[vid] = max(moves)
    assert legs["A01"] == pytest.approx(0.35)
    assert legs["A03"] == pytest.approx(0.25)
