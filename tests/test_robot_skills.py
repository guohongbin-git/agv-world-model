"""robot_skills 技能库测试 (Voyager 范式的可调用性/断言面)。

物理 E2E (outbound_order 全链) 约 3.5 分钟 —— 与 fleet 同量级。
"""
import pytest

from sim import robot_skills as rs
from sim.mission import MissionRunner
from sim.warehouse_cell import WarehouseCell


@pytest.fixture(scope="module")
def mr():
    cell = WarehouseCell(timestep_s=0.001, seed=7)
    cell.warmup(0.5)
    return MissionRunner(cell)


def test_catalog_lists_validated_skills():
    cat = rs.catalog()
    names = {s["name"] for s in cat}
    assert {"fork_pick_rack", "agv_transport", "outbound_order",
            "replenish_order"} <= names
    for s in cat:
        assert s["doc"] and isinstance(s["params"], dict)


def test_precondition_rejects_missing_pallet(mr):
    r = rs.invoke(mr, "fork_pick_rack",
                  {"slot": "R11-S3", "level": 0, "stand": "H02"})
    assert not r.ok and "前置条件" in r.detail
    assert r.tasks == []                      # 未展开任务 → 零物理动作


def test_precondition_rejects_occupied_stand(mr):
    # H01 由播种预载占据 → 放托目标不可用
    led = mr.ledger.loc
    occupied = [l.split(":")[1] for l in led.values() if l.startswith("stand:")]
    assert occupied, "播种应占用至少一个交接台"
    pn = next(p for p, l in led.items() if l.startswith("slot:"))
    slot, level = led[pn].split(":")[1], int(pn[-1])
    r = rs.invoke(mr, "fork_pick_rack",
                  {"slot": slot, "level": level, "stand": occupied[0]})
    assert not r.ok and "交接台" in r.detail


def test_unknown_skill_is_rejected(mr):
    r = rs.invoke(mr, "no_such_skill", {})
    assert not r.ok and "未知技能" in r.detail


@pytest.mark.slow
def test_outbound_order_full_chain_physical():
    """物理 E2E: 取托 → 交接台 → AGV 干运 → 交付台, 零 op_errors。"""
    cell = WarehouseCell(timestep_s=0.001, seed=7)
    cell.warmup(0.5)
    m = MissionRunner(cell)
    led = m.ledger.loc
    pn = next(p for p, l in led.items()
              if l.startswith("slot:") and p.endswith("_L1"))
    slot, level = led[pn].split(":")[1], int(pn[-1])
    stands = {l.split(":")[1] for l in led.values() if l.startswith("stand:")}
    stand = next(s for s in ("H02", "H03", "H04", "H05") if s not in stands)
    docks = {l for l in led.values() if l.startswith("dock:")}
    dock = next(d for d in ("D_G11", "D_G22", "D_G33", "D_G44")
                if f"dock:{d}" not in docks)
    r = rs.invoke(m, "outbound_order",
                  {"slot": slot, "level": level, "stand": stand, "dock": dock})
    assert r.ok, r.detail
    assert r.evidence["op_errors"] == []
    assert r.evidence["loc"] == f"dock:{dock}"      # 台账终点与期望一致