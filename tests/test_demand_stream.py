"""M2-W2b 需求引擎验收 (设计 §19.4 / M2-W2b 接口契约)。

验收口径:
- 同 seed 严格可复现; 异 seed 不同;
- 1h 订单流统计 (订单数 / SKU 分布) 与 profile 一致性 sanity;
- 周内峰谷与 weekday_factor 方向一致 (周中峰, 周末谷);
- 补货单在库存击穿阈值时生成, 计划库存守恒;
- stream() 与 generate_until() 等价 (24h continuous 可行)。
"""
from __future__ import annotations

from collections import Counter

import pytest

from sim.demand import (DEFAULT_CATALOG_PATH, DemandEngine, DemandOrder,
                        OrderKind, load_catalog, one_hour_sanity)

SEED = 20260501
SKUS = sorted(load_catalog(DEFAULT_CATALOG_PATH))


def test_same_seed_reproduces_order_stream():
    a = DemandEngine(seed=SEED).generate_until(7200.0)
    b = DemandEngine(seed=SEED).generate_until(7200.0)
    assert len(a) > 100
    assert a == b                     # frozen dataclass → 逐字段相等


def test_different_seed_diverges():
    a = DemandEngine(seed=SEED).generate_until(7200.0)
    b = DemandEngine(seed=SEED + 1).generate_until(7200.0)
    assert [(o.sku, o.t_s) for o in a] != [(o.sku, o.t_s) for o in b]


def test_one_hour_order_count_matches_profile():
    """1h 订单数 vs ∫λ (seed 固定 → 确定性, 容差 20% ≈ 3σ@λ≈50)。"""
    st = one_hour_sanity(DemandEngine(seed=SEED))
    assert st["orders"] > 0
    assert st["rel_err_total"] < 0.20


def test_sku_distribution_matches_profile_weights():
    """高倍率下 (≈5000 单/h) 逐 SKU 相对误差收敛 (前 10 大 SKU ≤ 25%)。"""
    e = DemandEngine(seed=SEED, global_multiplier=100.0)
    st = one_hour_sanity(e)
    top = sorted(st["sku_expected"], key=st["sku_expected"].get,
                 reverse=True)[:10]
    assert st["sku_expected"][top[0]] > 50      # 期望量级足以判分布
    for sku in top:
        got, exp = st["sku_counts"][sku], st["sku_expected"][sku]
        assert abs(got - exp) / exp < 0.25, f"{sku}: {got} vs {exp:.1f}"


def test_weekday_peak_and_valley_follow_profile():
    """7 天日单量: 峰在周二/三 (≈1.2), 谷在周六/日 (≈0.8)。"""
    e = DemandEngine(seed=SEED, global_multiplier=10.0)
    daily = [0.0] * 7
    for d in range(7):
        fresh = e.generate_until((d + 1) * 86400.0)
        daily[d] = sum(1 for o in fresh if o.kind is OrderKind.OUTBOUND)
    wd = (e.start_weekday + 0) % 7              # 天 0 的星期
    order_by_day = sorted(range(7), key=lambda i: -daily[i])
    peak, valley = order_by_day[0], order_by_day[-1]
    assert (wd + peak) % 7 in (1, 2), f"峰应在周中: {daily}"
    assert (wd + valley) % 7 in (5, 6), f"谷应在周末: {daily}"
    assert max(daily) / max(min(daily), 1) > 1.15


def test_replenish_triggers_on_stock_breakthrough():
    """每 SKU 恰 1 托起步 → 首次出库即击穿阈值 → 同时刻补货单生成。"""
    e = DemandEngine(seed=SEED, initial_pallets=len(SKUS),
                     replenish_threshold=1)
    fresh = e.generate_until(600.0)
    out = [o for o in fresh if o.kind is OrderKind.OUTBOUND]
    rep = [o for o in fresh if o.kind is OrderKind.REPLENISH]
    assert out and rep, "应既有出库又有补货"
    # 击穿即补: 补货单时刻 = 对应出库单时刻 (同一次到达内联触发)
    out_t = {o.t_s: o.sku for o in out}
    for r in rep:
        assert out_t.get(r.t_s) == r.sku, f"补货 {r} 未与击穿出库同时刻"
    # 补过的 SKU 计划库存回到阈值之上
    for r in rep:
        assert e.stock_by_sku(r.sku) >= 1


def test_inventory_conservation_after_generation():
    e = DemandEngine(seed=SEED)
    e.generate_until(3600.0)
    inv = e.inventory()
    filled = sum(1 for v in inv.values() if v is not None)
    n_out = sum(1 for o in e.orders if o.kind is OrderKind.OUTBOUND)
    n_rep = sum(1 for o in e.orders if o.kind is OrderKind.REPLENISH)
    # 计划库存守恒: 出库开单 −1, 补货开单 +1 → 在库 = 48 − (出库−补货)
    # (差值 = 阈值之上未被击穿的 SKU, 语义正确: 2→1 不触发补货)
    assert filled + e.free_slot_count() == len(inv) == 384
    assert filled == 48 - (n_out - n_rep)
    assert n_out >= 1 and n_rep >= 1 and 0 <= n_out - n_rep <= len(SKUS)


def test_outbound_dst_is_group_dock_of_src_slot():
    e = DemandEngine(seed=SEED)
    fresh = [o for o in e.generate_until(600.0)
             if o.kind is OrderKind.OUTBOUND]
    for o in fresh:
        assert o.dst == f"D_G{o.src[1:3]}", f"{o.src} → {o.dst}"


def test_stream_equals_batch_generation():
    """stream() 逐单产出与 generate_until() 同批推进完全一致 (24h 长跑)。"""
    a = DemandEngine(seed=SEED)
    b = DemandEngine(seed=SEED)
    via_stream = list(a.stream(86400.0))
    via_batch = b.generate_until(86400.0)
    assert a.cursor_s == b.cursor_s == 86400.0
    assert via_stream == via_batch
    assert len(via_stream) > 800          # λ≈1200 单/天 量级 sanity
    assert all(o.t_s <= 86400.0 for o in via_stream)


def test_multiplier_scales_rate_linearly():
    e1 = DemandEngine(seed=SEED)
    e2 = DemandEngine(seed=SEED, global_multiplier=3.0)
    n1 = sum(1 for o in e1.generate_until(3600.0)
             if o.kind is OrderKind.OUTBOUND)
    n2 = sum(1 for o in e2.generate_until(3600.0)
             if o.kind is OrderKind.OUTBOUND)
    assert abs(n2 - 3 * n1) <= max(6, 0.25 * 3 * n1)


def test_qty_sampling_respects_max_qty():
    e = DemandEngine(seed=SEED, max_qty=4)
    fresh = [o for o in e.generate_until(3600.0)
             if o.kind is OrderKind.OUTBOUND]
    assert all(1 <= o.qty <= 4 for o in fresh)
    assert any(o.qty > 1 for o in fresh)


def test_daily_noise_is_seed_stable():
    a = DemandEngine(seed=SEED, daily_noise=True)
    b = DemandEngine(seed=SEED, daily_noise=True)
    assert a.generate_until(2 * 86400.0) == b.generate_until(2 * 86400.0)


def test_initial_pallets_below_sku_count_rejected():
    with pytest.raises(ValueError):
        DemandEngine(seed=SEED, initial_pallets=10)
