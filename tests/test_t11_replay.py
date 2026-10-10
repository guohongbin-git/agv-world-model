"""T11 库存回放 / T16 装卸事务恢复 —— 事件回放的幂等、乱序、缺失。

设计判据:
|T11|重复 乱序 缺失事件及快照恢复|重复幂等; 乱序可检测; 缺失不得伪造成功|
|T16|入料夹持后、装托松爪后、卸货夹持后、出料松爪后、离场提交中断各5次,
     |另注入重复/乱序事件|无重复抓放、计数或绑定; 允许证据不足进入UNKNOWN,
     |恢复正例应在证据恢复后自动完成|

每个用例都断言**库存不变量**, 而不只是回放器的返回值:
活动绑定至多一条、在库计数与生命周期一致、托盘承载态自洽。
"""
from __future__ import annotations

import pytest

from services.scheduler.replay import (EventKind, EventReplay, ReplayOutcome)
from services.world.ontology import GoodsUnit, Loc, LocKind, Pallet, World


def make_world(run_id: str, goods=("G-001",), pallets=("PAL-001",)):
    w = World(run_id=run_id)
    for p in pallets:
        w.register_pallet(Pallet(p, 0.045, Loc(LocKind.STATION, "I01")))
    for g in goods:
        w.register_goods(GoodsUnit(g, "A", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    return w


def invariants(w: World, label: str):
    """任何时刻都必须成立的不变量。"""
    for g in ("G-001", "G-002"):
        try:
            loc = w.goods_location(g)
        except TypeError:
            continue
        assert loc is not None
    # 活动绑定至多一条
    rows = w._conn.execute(
        "SELECT goods_id, COUNT(*) FROM bindings WHERE active=1 "
        "GROUP BY goods_id HAVING COUNT(*)>1").fetchall()
    assert not rows, f"{label}: 重复活动绑定 {rows}"
    # 在库计数与生命周期一致
    n = w._conn.execute(
        "SELECT COUNT(*) FROM goods WHERE lifecycle IN ('IN_WAREHOUSE',"
        "'OUTBOUND_PENDING')").fetchone()[0]
    assert w.in_warehouse_count() == n, f"{label}: 计数与生命周期不一致"
    return w


# ---------------------------------------------------------------- T11
def test_t11_duplicate_event_is_idempotent():
    """重复事件不得重复记账 (T11: 重复幂等)。"""
    w = make_world("T11-DUP")
    r = EventReplay(w)
    seq = 0
    for _ in range(20):
        seq += 1
        r.apply(EventKind.BIND_STORE, "G-001", seq, "PAL-001")
    assert w.active_binding("G-001")[0] == "PAL-001"
    assert w.in_warehouse_count() == 1, "重复 20 次后仍只计 1"
    assert w.pallet("PAL-001")[1] == "LOADED"
    invariants(w, "重复绑定")


def test_t11_out_of_order_is_detected_not_applied():
    """乱序事件必须被**检测**, 且不静默应用 (T11)。"""
    w = make_world("T11-ORDER")
    r = EventReplay(w)
    r.apply(EventKind.BIND_STORE, "G-001", 10, "PAL-001")
    out = r.apply(EventKind.UNBIND, "G-001", 5, "PAL-001")   # 序号倒退
    assert out is ReplayOutcome.OUT_OF_ORDER
    assert w.active_binding("G-001") is not None, "乱序不得解绑"
    assert w.pallet("PAL-001")[1] == "LOADED"


def test_t11_missing_event_does_not_fake_success():
    """缺失事件不得伪造成功 (T11) —— 跳过装托直接离场必须被拒。"""
    w = make_world("T11-MISS")
    r = EventReplay(w)
    # 未装托、未解绑, 直接 EXIT → 前置不满足
    assert r.apply(EventKind.EXIT, "G-001", 1) is ReplayOutcome.BLOCKED
    assert w.in_warehouse_count() == 0


def test_t11_snapshot_recovery_after_interrupted_commit():
    """提交中断 → UNKNOWN; 证据恢复后重放应自动完成 (T11/T16)。"""
    w = make_world("T11-RECOVER")
    r = EventReplay(w)
    # 提交前中断: 不产生任何状态变化
    out = r.apply(EventKind.BIND_STORE, "G-001", 1, "PAL-001",
                  fault_at="before_commit")
    assert out is ReplayOutcome.UNKNOWN
    assert w.active_binding("G-001") is None, "中断不得留下半成品绑定"
    assert w.in_warehouse_count() == 0
    invariants(w, "提交中断")
    # 证据恢复 → 重放自动完成
    assert r.apply(EventKind.BIND_STORE, "G-001", 2, "PAL-001") \
        is ReplayOutcome.APPLIED
    assert w.in_warehouse_count() == 1
    invariants(w, "恢复后")


def test_t11_lost_ack_after_commit_is_deduplicated():
    """提交后回执丢失 → 重放必须靠事务键去重, 不重复绑定。"""
    w = make_world("T11-ACK")
    r = EventReplay(w)
    assert r.apply(EventKind.BIND_STORE, "G-001", 1, "PAL-001",
                   fault_at="after_commit") is ReplayOutcome.APPLIED
    # 调用方没收到回执, 重发同一事件
    assert r.apply(EventKind.BIND_STORE, "G-001", 2, "PAL-001") \
        is ReplayOutcome.DUPLICATE
    assert w.active_binding("G-001")[0] == "PAL-001"
    assert w.in_warehouse_count() == 1
    invariants(w, "回执丢失")


# ---------------------------------------------------------------- T16
@pytest.mark.parametrize("stage,kind,pre_events", [
    # 每个中断点前, 必须先把生命周期/绑定推到该阶段真实的前置态,
    # 否则测到的 BLOCKED 只是「前置没到位」, 测不到「中断不提交」。
    ("入料夹持后", EventKind.FEED_PICK, []),
    ("装托松爪后", EventKind.BIND_STORE, []),
    ("卸货夹持后", EventKind.UNBIND,
     [(EventKind.BIND_STORE, "PAL-001")]),
    ("出料松爪后", EventKind.PLACE_FEED,
     [(EventKind.BIND_STORE, "PAL-001"), (EventKind.UNBIND, "PAL-001"),
      (EventKind.OUTBOUND, "PAL-001")]),
])
def test_t16_interrupted_stage_five_times(stage, kind, pre_events):
    """T16: 五个中断点各 5 次 —— 无重复抓放、计数或绑定。"""
    w = make_world(f"T16-{kind.value}")
    r = EventReplay(w)
    seq = 0
    for ekind, pal in pre_events:
        seq += 1
        r.apply(ekind, "G-001", seq, pal)
    before_count = w.in_warehouse_count()
    before_bind = w.active_binding("G-001")
    for _ in range(5):
        seq += 1
        assert r.apply(kind, "G-001", seq, fault_at="before_commit") \
            is ReplayOutcome.UNKNOWN
    assert w.in_warehouse_count() == before_count, f"{stage}: 计数被改动"
    assert w.active_binding("G-001") == before_bind, f"{stage}: 绑定被改动"
    invariants(w, f"{stage}×5")


def test_t16_offsite_commit_interrupt():
    """T16: 离场提交中断 5 次 —— 计数不得错减。"""
    w = make_world("T16-EXIT")
    r = EventReplay(w)
    r.apply(EventKind.BIND_STORE, "G-001", 1, "PAL-001")
    r.apply(EventKind.UNBIND, "G-001", 2, "PAL-001")
    r.apply(EventKind.OUTBOUND, "G-001", 3)          # 到出库站 → 才允许离场
    before = w.in_warehouse_count()
    assert before == 1, "离场前应仍计在库"
    for i in range(5):
        assert r.apply(EventKind.EXIT, "G-001", 4 + i,
                       fault_at="before_commit") is ReplayOutcome.UNKNOWN
    assert w.in_warehouse_count() == before, "中断不得减计数"
    assert r.apply(EventKind.EXIT, "G-001", 20) is ReplayOutcome.APPLIED
    assert w.in_warehouse_count() == 0, "恢复后才减一"


def test_t16_full_lifecycle_with_mixed_faults():
    """T16 综合: 正常 + 重复 + 乱序 + 中断混合, 终态必须正确。"""
    w = make_world("T16-MIX")
    r = EventReplay(w)
    r.apply(EventKind.BIND_STORE, "G-001", 1, "PAL-001")          # 装托
    r.apply(EventKind.BIND_STORE, "G-001", 2, "PAL-001")          # 重复
    r.apply(EventKind.BIND_STORE, "G-001", 3, "PAL-001")          # 重复
    r.apply(EventKind.UNBIND, "G-001", 0, "PAL-001")              # 乱序
    r.apply(EventKind.OUTBOUND, "G-001", 4)                      # 到出库站
    assert w.in_warehouse_count() == 1, "到出库站仍计在库"
    r.apply(EventKind.UNBIND, "G-001", 5, "PAL-001")              # 卸货
    assert w.in_warehouse_count() == 1, "卸货计数不变"
    assert w.pallet("PAL-001")[1] == "EMPTY"
    r.apply(EventKind.PLACE_FEED, "G-001", 6)                     # 出料放置
    assert w.goods_location("G-001").startswith("FEED:")
    r.apply(EventKind.EXIT, "G-001", 7)                          # 离场
    assert w.in_warehouse_count() == 0
    invariants(w, "综合")