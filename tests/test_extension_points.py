"""core 可扩展点回归 (dc-twin P0 前置契约, 设计 v0.5.1 §1/§3-P0)。

红线: core 不为场景硬编码 —— 场景载荷 (tk:* 节点类 / load 腿类型 /
LOAD 事件类型 / 外部到达源) 只经注册表进主轴; 既有 fork/trunk 两腿
语义零改动 (本文件全部断言与注册前行为兼容)。
"""
from __future__ import annotations

import uuid

import pytest

from services.world import evidence
from sim import dispatch
from sim import warehouse_cell as wc


def _uid() -> str:
    return uuid.uuid4().hex[:8].upper()


# ---------------------------------------------------------------- 节点注册
def test_node_source_registry_extends_layout_and_dispatch_index():
    tag = _uid()

    @wc.register_node_source
    def _src(lay):
        y = lay.dock_y + lay.p["berth_depth_y"] / 2
        return {f"tk:T{tag}": (-0.55, y, 0.0)}

    lay = wc.MatrixLayout()
    assert f"tk:T{tag}" in lay.extra_nodes()
    # 调度层节点索引含注册节点; 既有节点不受影响
    pos, handoffs, inbound = dispatch._layout_index()
    assert f"tk:T{tag}" in pos
    assert "D_G42" in pos and "H01" in handoffs
    d = dispatch.Dispatcher(fleet=dispatch.FLEET_STUB[:2])
    x, y, z = d.node_pos(f"tk:T{tag}")
    assert z == 0.0


def test_node_source_duplicate_name_rejected():
    tag = _uid()
    saved = list(wc._NODE_SOURCES)                 # 注册表是全局态: 隔离+还原
    wc._NODE_SOURCES.clear()
    try:
        wc.register_node_source(lambda lay: {f"tk:D{tag}": (0.0, 0.0, 0.0)})
        wc.register_node_source(lambda lay: {f"tk:D{tag}": (1.0, 0.0, 0.0)})
        with pytest.raises(ValueError, match="tk:D" + tag):
            wc.MatrixLayout().extra_nodes()
    finally:
        wc._NODE_SOURCES[:] = saved


# ---------------------------------------------------------------- 腿类型注册
def test_leg_type_registry_vehicle_and_handling():
    tag = _uid()
    calls = []

    def _handling(leg):
        calls.append(leg.leg_id)
        return 1.5

    dispatch.register_leg_type(f"load{tag}", vehicle_type="agv",
                               handling_s=_handling)
    leg = dispatch.Leg(f"L{tag}", "T", f"load{tag}", "D_G42", "tk:X", 0.16)
    assert dispatch.LEG_VEHICLE_TYPE[f"load{tag}"] == "agv"
    assert leg.handling_s() == 1.5 and calls == [f"L{tag}"]
    # 标量作业时长也允许
    dispatch.register_leg_type(f"loadS{tag}", vehicle_type="agv",
                               handling_s=2.5)
    assert dispatch.Leg("L2", "T", f"loadS{tag}", "a", "b", 1.0).handling_s() \
        == 2.5
    with pytest.raises(ValueError, match=f"load{tag}"):
        dispatch.register_leg_type(f"load{tag}", vehicle_type="agv",
                                   handling_s=1.0)


def test_leg_registry_preserves_fork_trunk_semantics():
    # 注册不改变既有 fork/trunk 的载具映射与作业时长
    slot = "R11-S0-L0"                                 # 底层: 无层附加
    from sim.warehouse_cell import MatrixLayout

    z = MatrixLayout().slot_pose(slot)[3]              # 层支撑面 z (x,y,yaw,z)
    fork = dispatch.Leg("f", "T", "fork", slot, "H01", 1.0, level_z=z)
    trunk = dispatch.Leg("t", "T", "trunk", "H01", "D_G42", 1.0)
    assert dispatch.LEG_VEHICLE_TYPE["fork"] == "forklift"
    assert dispatch.LEG_VEHICLE_TYPE["trunk"] == "agv"
    assert fork.handling_s() == dispatch.RACK_PICK_BASE_S + \
        dispatch.STAND_PLACE_S
    assert trunk.handling_s() == dispatch.STAND_PICK_S + dispatch.STAND_PLACE_S


# ---------------------------------------------------------------- 事件类型注册
def test_event_kind_registry():
    tag = _uid()
    assert evidence.KIND_PICK in evidence.event_kinds()
    assert evidence.KIND_PLACE in evidence.event_kinds()
    k = evidence.register_event_kind(f"LOAD{tag}", "装车 (dc-twin P0)")
    assert k in evidence.event_kinds()
    assert evidence.event_kinds()[f"LOAD{tag}"] == "装车 (dc-twin P0)"
    with pytest.raises(ValueError):
        evidence.register_event_kind(f"LOAD{tag}")
    with pytest.raises(ValueError):
        evidence.register_event_kind("bad kind")       # 空白不合法


# ---------------------------------------------------------------- 到达源注册
def test_demand_arrival_sources_merged_by_time():
    from sim.demand import DemandEngine

    tag = _uid()

    class _Ext:
        def __init__(self, t_s, oid):
            self.t_s, self.oid = t_s, oid

        def arrivals_in(self, t0_s, t1_s):
            if t0_s <= self.t_s < t1_s:
                return [type("O", (), {"t_s": self.t_s,
                                       "order_id": self.oid})()]
            return []

    eng = DemandEngine(initial_pallets=64,
                       arrival_sources=(_Ext(30.0, f"EXT-A{tag}"),
                                        _Ext(10.0, f"EXT-B{tag}")))
    fresh = eng.generate_until(60.0)
    ids = [o.order_id for o in fresh]
    assert f"EXT-B{tag}" in ids and f"EXT-A{tag}" in ids
    ts = [o.t_s for o in fresh]
    assert ts == sorted(ts)                            # 合并后按时间有序
    ext_ts = [o.t_s for o in fresh if o.order_id.startswith("EXT")]
    assert ext_ts == [10.0, 30.0]
    # 游标推进后同窗口不重复吐出
    fresh2 = eng.generate_until(65.0)
    assert not [o for o in fresh2 if o.order_id.startswith("EXT")]


# ---------------------------------------------------------------- 多腿链
def _three_leg_task(tag: str) -> dispatch.TransportTask:
    wc.register_node_source(
        lambda lay: {f"tk:Y{tag}": (-0.55,
                                    lay.dock_y + lay.p["berth_depth_y"] / 2,
                                    0.0)})
    dispatch.register_leg_type(f"ld{tag}", vehicle_type="agv",
                               handling_s=2.0)
    mk = dispatch._layout_index()[0]
    slot, handoff, dock, tk = "R11-S0-L2", "H01", "D_G42", f"tk:Y{tag}"
    return dispatch.TransportTask(
        task_id=f"T3{tag}", kind=dispatch.OrderKind.OUTBOUND, sku="SKU-001",
        qty=1, weight_kg=0.05, release_s=0.0, handoff=handoff,
        legs=(
            dispatch.Leg(f"T3{tag}:f", f"T3{tag}", "fork", slot, handoff,
                         1.0, mk[slot][3]),
            dispatch.Leg(f"T3{tag}:t", f"T3{tag}", "trunk", handoff, dock,
                         2.0),
            dispatch.Leg(f"T3{tag}:l", f"T3{tag}", f"ld{tag}", dock, tk,
                         0.16),
        ))


@pytest.mark.parametrize("solver", ["greedy", "cpsat"])
def test_dispatcher_schedules_three_leg_chain(solver: str):
    tag = _uid()
    t = _three_leg_task(tag)
    d = dispatch.Dispatcher(fleet=dispatch.FLEET_STUB)
    evs = d.submit([t], now_s=0.0, solver=solver)
    assert evs and evs[-1]["type"] == "TaskAssigned"
    assert set(t.assignment) == {l.leg_id for l in t.legs}   # 三腿全指派
    # 链先后: 后腿开始 ≥ 前腿结束 + dwell; 载具类型: 叉车货位端, AGV 干线/装车
    f, k, l = t.legs
    assert t.assignment[f.leg_id].startswith("F")
    assert t.assignment[k.leg_id].startswith("A")
    assert t.assignment[l.leg_id].startswith("A")
    end_f = t.schedule[f.leg_id][0] + t.schedule[f.leg_id][1]
    start_k = t.schedule[k.leg_id][0]
    end_k = start_k + t.schedule[k.leg_id][1]
    start_l = t.schedule[l.leg_id][0]
    assert start_k >= end_f + dispatch.HANDOFF_DWELL_TICKS
    assert start_l >= end_k + dispatch.HANDOFF_DWELL_TICKS


def test_dispatcher_commit_three_leg_ok():
    tag = _uid()
    t = _three_leg_task(tag)
    d = dispatch.Dispatcher(fleet=dispatch.FLEET_STUB)
    d.submit([t], now_s=0.0, solver="greedy")
    d.commit(t.task_id, evidence={"support": 1.0}, ts_s=1.0)
    assert d.task(t.task_id).state is dispatch.TaskState.COMMITTED
