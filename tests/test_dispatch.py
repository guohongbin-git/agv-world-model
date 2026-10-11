"""M2-W2b 调度层验收 (设计 §19.4 / W2b 契约 / Main 补充口径)。

验收口径:
- 候选过滤: 等级载重 ≥ 货总重 (超载禁止), 类型匹配 (叉车=货位端/AGV=干线);
- CP-SAT: 4 任务 × 8 车求出分配且能力/互斥/交接链约束全满足;
- 强制超时 → 回退贪心仍可行; MODEL_INVALID 不允许静默 (§19.4);
- CP-SAT vs 贪心: 至少一用例 makespan 严格更优;
- 决策事件完整 (TaskAssigned/Unassigned/Committed, solver_status 三态);
- 打桩集成: mock navigation/traffic 跑通 任务→分配→执行→COMMITTED 全链;
- 性能: 单次分配 (24 任务批) 求解 <1s;
- 评估口径: makespan / 总等待 / 各等级利用率 (对齐 agv-warehouse-scheduling)。
"""
from __future__ import annotations

from collections import defaultdict

import pytest

import sim.dispatch as dispatch_mod
from sim.demand import DemandEngine, DemandOrder, OrderKind
from sim.dispatch import (HANDOFF_DWELL_TICKS, LEG_VEHICLE_TYPE, Dispatcher,
                          DispatchError, InfeasibleAssignment, TaskState,
                          Vehicle)

SEED = 20260501

LIGHT, MID, HEAVY = "SKU_20", "SKU_2", "SKU_21"   # light 0.028 / mid 0.087 / heavy 0.155


def order(oid, sku=LIGHT, qty=1, t_s=0.0, kind="OUTBOUND",
          src="R11-S0-L0", dst="D_G11"):
    return DemandOrder(order_id=oid, kind=OrderKind(kind), sku=sku, qty=qty,
                       t_s=t_s, src=src, dst=dst)


@pytest.fixture
def dsp() -> Dispatcher:
    return Dispatcher(random_seed=0)


# ------------------------------------------------------------ 通用校验
def assert_plan_feasible(d: Dispatcher):
    """能力 / 类型 / 释放期 / 交接链 / 车辆互斥 / 交接台占用窗 全量校验。"""
    fleet = {v.vid: v for v in d.fleet}
    by_vehicle: dict[str, list] = defaultdict(list)
    by_stand: dict[str, list] = defaultdict(list)
    for t in d.tasks_by_state(TaskState.ASSIGNED):
        for l in t.legs:
            v = fleet[t.assignment[l.leg_id]]
            assert v.capacity_kg + 1e-9 >= t.weight_kg, \
                f"{t.task_id} 超载: {t.weight_kg:.3f} > {v.capacity_kg}"
            assert v.vtype == LEG_VEHICLE_TYPE[l.leg_type]
            s, du = t.schedule[l.leg_id]
            assert s >= t.release_tick
            by_vehicle[v.vid].append((s, s + du))
        f, k = t.legs
        sf, df = t.schedule[f.leg_id]
        sk, _dk = t.schedule[k.leg_id]
        assert sk >= sf + df + HANDOFF_DWELL_TICKS, "交接链先后被破坏"
        by_stand[t.handoff].append((sf + df, sk))      # 占用窗 [落托, 接管)
    for vid, ivs in by_vehicle.items():
        ivs.sort()
        for (_s0, e0), (s1, _e1) in zip(ivs, ivs[1:]):
            assert s1 >= e0, f"{vid} 时间窗重叠"
    for h, wins in by_stand.items():
        wins.sort()
        for (_a0, b0), (a1, _b1) in zip(wins, wins[1:]):
            assert a1 >= b0, f"交接台 {h} 占用窗重叠"


def batch_events(d: Dispatcher, typ: str) -> list[dict]:
    return [e for e in d.events() if e["type"] == typ]


# ------------------------------------------------------------ 候选过滤
def test_heavy_task_only_L3_candidates(dsp):
    """重货 0.045+0.155=0.200kg > L2 0.15 → 叉车腿只剩 F03/F04, 干线只剩 A04。"""
    evs = dsp.submit_orders([order("ORD-H", sku=HEAVY)])
    ev = evs[0]
    assert ev["type"] == "TaskAssigned"
    assert set(ev["candidates"]) == {"F03", "F04", "A04"}
    assert ev["chosen"]["ORD-H:fork"] in ("F03", "F04")
    assert ev["chosen"]["ORD-H:trunk"] == "A04"
    assert_plan_feasible(dsp)


def test_mid_task_L2_or_L3_candidates(dsp):
    evs = dsp.submit_orders([order("ORD-M", sku=MID)])   # 0.045+0.087=0.132
    ev = evs[0]
    assert {"F02", "F03", "F04"} & set(ev["candidates"])
    assert ev["chosen"]["ORD-M:fork"] not in ("F01",)    # L1 0.08 < 0.132
    assert ev["chosen"]["ORD-M:trunk"] != "A01"
    assert_plan_feasible(dsp)


def test_overload_impossible_task_stays_pending(dsp):
    """3 件重货 0.045+3×0.155=0.509 > L3 0.30 → 无合格车, 明确失败不硬排。"""
    evs = dsp.submit_orders([order("ORD-X", sku=HEAVY, qty=3)])
    ev = evs[0]
    assert ev["type"] == "TaskUnassigned"
    assert ev["candidates"] == [] and ev["chosen"] is None
    assert "no_capable_vehicle" in ev["reason"]
    assert ev["solver_status"] == "INFEASIBLE_FILTER"
    assert dsp.task("ORD-X").state is TaskState.PENDING
    assert dsp.tasks_by_state(TaskState.ASSIGNED) == []


# ------------------------------------------------------------ CP-SAT 主路径
def test_cpsat_batch_4x8_with_mixed_weights(dsp):
    orders = [order("ORD-A", sku=LIGHT, t_s=0.0, src="R11-S0-L0", dst="D_G11"),
              order("ORD-B", sku=HEAVY, t_s=1.5, src="R33-S4-L2", dst="D_G33"),
              order("ORD-C", sku=MID, t_s=3.0, src="R42-S2-L1", dst="D_G42"),
              order("REP-D", sku=LIGHT, t_s=4.5, kind="REPLENISH",
                    src="INBOUND", dst="R24-S5-L0")]
    evs = dsp.submit_orders(orders, now_s=0.0)
    assert [e["type"] for e in evs] == ["TaskAssigned"] * 4
    assert {e["solver_status"] for e in evs} <= {"OPTIMAL", "FEASIBLE"}
    for e in evs:
        assert e["solve_ms"] < 1000.0
        assert set(e["chosen"]) == {f"{e['task_id']}:fork",
                                    f"{e['task_id']}:trunk"}
    assert_plan_feasible(dsp)
    rep = dsp.task("REP-D")
    assert rep.legs[0].leg_type == "trunk" and rep.legs[0].src == "INBOUND"


def test_decision_event_schema_complete(dsp):
    evs = dsp.submit_orders([order("ORD-S1"), order("ORD-X", sku=HEAVY, qty=3)])
    kinds = {e["type"] for e in evs}
    assert kinds == {"TaskAssigned", "TaskUnassigned"}
    for e in dsp.events():
        assert {"type", "task_id", "ts_s", "candidates", "chosen", "reason",
                "solver_status", "solve_ms", "makespan_ticks"} <= set(e)
    assigned = batch_events(dsp, "TaskAssigned")[0]
    assert assigned["solver_status"] in ("OPTIMAL", "FEASIBLE",
                                         "FALLBACK_GREEDY")
    assert assigned["makespan_ticks"] is not None and assigned["solve_ms"] >= 0


def test_cpsat_deterministic_same_seed():
    orders = [order(f"ORD-{i}", sku=(LIGHT, MID, HEAVY)[i % 3], t_s=float(i),
                    src=f"R1{1 + i % 3}-S{i}-L{i % 3}",
                    dst=f"D_G1{1 + i % 3}") for i in range(6)]
    a = Dispatcher(random_seed=7).submit_orders(orders)
    b = Dispatcher(random_seed=7).submit_orders(orders)
    assert [(e["chosen"], e["start_ticks"]) for e in a] == \
           [(e["chosen"], e["start_ticks"]) for e in b]


def test_solve_under_1s_for_24_task_batch():
    """性能: 24 任务 × 8 车 单次分配 <1s (24h 长跑: 批量求解可叠加)。"""
    e = DemandEngine(seed=SEED)
    outs = [o for o in e.generate_until(3600.0)
            if o.kind is OrderKind.OUTBOUND][:24]
    d = Dispatcher(random_seed=0)
    import time
    t0 = time.perf_counter()
    evs = d.submit_orders(outs, now_s=0.0)
    wall_s = time.perf_counter() - t0
    assert len(evs) == 24
    assert wall_s < 1.0, f"24 任务分配耗时 {wall_s:.2f}s"
    assert max(ev["solve_ms"] for ev in evs) < 1000.0
    assert_plan_feasible(d)


# ------------------------------------------------------------ 回退语义
def test_timeout_falls_back_to_feasible_greedy(dsp, monkeypatch):
    """强制 UNKNOWN (超时语义) → 回退贪心, 方案仍满足全部硬约束。"""
    monkeypatch.setattr(dispatch_mod, "_solve_cpsat",
                        lambda *a, **k: (None, "UNKNOWN"))
    evs = dsp.submit_orders([order("ORD-F1"), order("ORD-F2", t_s=1.0,
                                                    src="R22-S1-L0",
                                                    dst="D_G22")])
    assert {e["solver_status"] for e in evs} == {"FALLBACK_GREEDY"}
    assert all("greedy nearest-capable" in e["reason"] for e in evs)
    assert_plan_feasible(dsp)


def test_infeasible_raises_per_design(dsp, monkeypatch):
    """§19.4: INFEASIBLE 明确失败, 不回退不声称已优化。"""
    monkeypatch.setattr(dispatch_mod, "_solve_cpsat",
                        lambda *a, **k: (None, "INFEASIBLE"))
    with pytest.raises(InfeasibleAssignment):
        dsp.submit_orders([order("ORD-I")])


def test_model_invalid_raises_never_ignored(dsp, monkeypatch):
    """§19.4: MODEL_INVALID 不允许按「优化失败可忽略」继续。"""
    def boom(*a, **k):
        raise DispatchError("CP-SAT MODEL_INVALID")
    monkeypatch.setattr(dispatch_mod, "_solve_cpsat", boom)
    with pytest.raises(DispatchError):
        dsp.submit_orders([order("ORD-V")])


def test_explicit_greedy_solver_runnable(dsp):
    evs = dsp.submit_orders([order("ORD-G")], solver="greedy")
    assert evs[0]["solver_status"] == "FALLBACK_GREEDY"
    assert evs[0]["solve_ms"] == 0.0
    assert_plan_feasible(dsp)


# ------------------------------------------------------------ 优化价值
def test_cpsat_makespan_strictly_beats_greedy():
    """贪心陷阱: 两货位都距 Fa 最近 → 串行; CP-SAT 拆给 Fb 并行 → 严格更优。"""
    fleet = (
        Vehicle("Fa", "forklift", "L1", (-1.89, -2.42)),   # 正对两个目标货位
        Vehicle("Fb", "forklift", "L1", (0.50, -2.42)),
        Vehicle("Aa", "agv", "L1", (-2.22, -2.85)),
        Vehicle("Ab", "agv", "L1", (0.00, -2.85)),
    )
    orders = [order("ORD-P", t_s=0.0, src="R11-S0-L0", dst="D_G11"),
              order("ORD-Q", t_s=0.0, src="R11-S3-L0", dst="D_G11")]
    g = Dispatcher(fleet=fleet, random_seed=0).submit_orders(orders,
                                                             solver="greedy")
    c = Dispatcher(fleet=fleet, random_seed=0).submit_orders(orders)
    g_mk = g[0]["makespan_ticks"]
    c_mk = c[0]["makespan_ticks"]
    assert c[0]["solver_status"] in ("OPTIMAL", "FEASIBLE")
    assert c_mk < g_mk, f"CP-SAT 应严格更优: {c_mk} !< {g_mk}"


def test_same_stand_occupancy_windows_disjoint():
    """同交接台多任务: 占用窗 [落托, 接管) 两两不重叠 (离散资源时间窗)。

    G11 的 S0/S1 就近 H01, S2/S3 就近 H02 (x 通道就近) → 两台各 2 窗。
    """
    orders = [order(f"ORD-K{i}", t_s=float(i), src=f"R11-S{i}-L0", dst="D_G11")
              for i in range(4)]
    d = Dispatcher(random_seed=0)
    d.submit_orders(orders)
    stands = defaultdict(list)
    for t in d.tasks_by_state(TaskState.ASSIGNED):
        f, k = t.legs
        stands[t.handoff].append((t.schedule[f.leg_id][0]
                                  + t.schedule[f.leg_id][1],
                                  t.schedule[k.leg_id][0]))
    assert {h: len(w) for h, w in stands.items()} == {"H01": 2, "H02": 2}
    assert_plan_feasible(d)


# ------------------------------------------------------------ 事务与集成
class MockNavigator:
    """W2a navigation桩: plan(vehicle_xy, from, to, lock_penalty)->[Segment]。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def plan(self, vehicle_xy, src, dst, lock_penalty=0.0):
        assert isinstance(vehicle_xy, tuple) and len(vehicle_xy) == 2
        self.calls.append((vehicle_xy, src, dst, lock_penalty))
        return [{"wp": [vehicle_xy, (src, dst)]}]     # Segment 序列占位


class MockLockMatrix:
    """W2a traffic 桩: acquire(path_segments, vid, ordered=True)/release(vid)。"""

    def __init__(self):
        self.acquired: list[tuple] = []
        self.released: list[str] = []

    def acquire(self, path_segments, vid, ordered=True):
        assert isinstance(path_segments, list) and path_segments
        self.acquired.append((vid, len(path_segments), ordered))
        return True

    def release(self, vid):
        self.released.append(vid)


def test_stubbed_navigation_traffic_full_chain():
    """任务→分配→(mock 执行)→COMMITTED 全链; W2a 契约签名逐腿走查。"""
    orders = [order("ORD-C1", t_s=0.0),
              order("ORD-C2", t_s=2.0, sku=HEAVY, src="R32-S6-L1",
                    dst="D_G32"),
              order("REP-C3", t_s=3.0, kind="REPLENISH", src="INBOUND",
                    dst="R13-S2-L1")]
    nav, lock = MockNavigator(), MockLockMatrix()
    d = Dispatcher(random_seed=0, navigator=nav, lockboard=lock)
    d.submit_orders(orders, now_s=0.0)
    assert {t.state for t in d.tasks_by_state(TaskState.ASSIGNED)} == \
        {TaskState.ASSIGNED}

    steps = [(t, s) for t, ss in d.iter_schedule() for s in ss]
    steps.sort(key=lambda x: x[1]["start_tick"])
    for t, s in steps:
        segs = nav.plan(s["vehicle_xy"], s["src"], s["dst"], lock_penalty=0.0)
        assert lock.acquire(segs, s["vid"], ordered=True) is True
        # …物理执行 (W2a)…
        lock.release(s["vid"])
    assert len(nav.calls) == 6 and len(lock.acquired) == 6
    assert len(lock.released) == 6
    assert all(ordered is True for (_v, _n, ordered) in lock.acquired)

    for tid in ("ORD-C1", "ORD-C2", "REP-C3"):
        d.commit(tid, evidence={"pallet_pose_m": (0.0, 0.0, 0.095),
                                "contact_above_threshold": True,
                                "support_ratio": 0.97},
                 ts_s=42.0)
        assert d.task(tid).state is TaskState.COMMITTED
    committed = batch_events(d, "TaskCommitted")
    assert len(committed) == 3
    assert all(e["evidence_keys"] == ["contact_above_threshold",
                                      "pallet_pose_m", "support_ratio"]
               for e in committed)


def test_commit_requires_evidence_and_is_idempotent():
    d = Dispatcher(random_seed=0)
    d.submit_orders([order("ORD-T"), order("ORD-T2", t_s=1.0,
                                           src="R11-S1-L0", dst="D_G11")])
    with pytest.raises(ValueError, match="物理证据"):
        d.commit("ORD-T", evidence={})
    d.commit("ORD-T", evidence={"contact": True})
    with pytest.raises(ValueError, match="幂等"):
        d.commit("ORD-T", evidence={"contact": True})
    with pytest.raises(KeyError):
        d.commit("ORD-NOPE", evidence={"contact": True})
    d.task("ORD-T2").state = TaskState.PENDING      # 篡改模拟: 未执行先提交
    with pytest.raises(ValueError, match="ASSIGNED"):
        d.commit("ORD-T2", evidence={"contact": True})


# ------------------------------------------------------------ 评估口径
def test_metrics_makespan_wait_and_level_utilization():
    e = DemandEngine(seed=SEED)
    outs = [o for o in e.generate_until(1800.0)
            if o.kind is OrderKind.OUTBOUND][:12]
    d = Dispatcher(random_seed=0)
    evs = d.submit_orders(outs, now_s=0.0)
    m = d.metrics()
    mk_ref = max(s + du for t in d.tasks_by_state(TaskState.ASSIGNED)
                 for (_s, du) in [t.schedule[t.legs[-1].leg_id]]
                 for s in [t.schedule[t.legs[-1].leg_id][0]])
    assert m.makespan_ticks == mk_ref == evs[0]["makespan_ticks"]
    assert m.total_wait_ticks >= 0
    assert set(m.utilization_by_level) == {"L1", "L2", "L3"}
    assert all(0.0 <= u <= 1.0 for u in m.utilization_by_level.values())
    assert sum(m.busy_ticks_by_vehicle.values()) > 0
