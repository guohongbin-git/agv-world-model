"""全库测试共享夹具。

core 注册表 (节点源/腿类型/事件类型) 是进程级全局态: 单测注册的条目
不还原则跨文件泄漏 (布局派生节点混入后续 MatrixLayout、重注册误撞)。
autouse 快照-还原让每个测试从同一基准出发, 与执行顺序解耦。
"""
import pytest


@pytest.fixture(autouse=True)
def _reset_core_registries():
    import sim.dispatch as dispatch
    import sim.warehouse_cell as wc
    from services.world import evidence as evidence

    node_sources = list(wc._NODE_SOURCES)
    leg_vehicle = dict(dispatch.LEG_VEHICLE_TYPE)
    leg_handling = dict(dispatch._LEG_HANDLING)
    event_kinds = dict(evidence.EVENT_KINDS)
    yield
    wc._NODE_SOURCES[:] = node_sources
    dispatch.LEG_VEHICLE_TYPE.clear()
    dispatch.LEG_VEHICLE_TYPE.update(leg_vehicle)
    dispatch._LEG_HANDLING.clear()
    dispatch._LEG_HANDLING.update(leg_handling)
    evidence.EVENT_KINDS.clear()
    evidence.EVENT_KINDS.update(event_kinds)
