"""T12 完整演示 —— 同一冻结版本连续 20 轮闭环。

设计判据:
|T12 完整演示|同一冻结版本连续20轮|每轮 M01装托、R01/F01上架下架、M02卸货
     |及货物离场；错码支线正确拒绝，画面与库存一致|

统计必须**分开** (设计 621 行: 正例、预期拒绝和受控故障各自统计,
不能合并成一个含义不明的成功率)。
"""
from __future__ import annotations

import pytest

from services.ui.scenario import ROUNDS, DemoScenario


@pytest.fixture(scope="module")
def run():
    sc = DemoScenario(rounds=ROUNDS)
    sc.run()
    return sc, sc.report()


def test_t12_twenty_rounds_all_complete(run):
    """20 轮全部完成入库→上架→下架→卸货→离场。"""
    sc, rep = run
    assert rep["rounds"] == ROUNDS
    assert rep["ok"] == ROUNDS, (
        f"仅 {rep['ok']}/{ROUNDS} 轮成功, 失败: {rep['failures'][:5]}")
    assert not rep["failures"]


def test_t12_each_round_reaches_offsite(run):
    """每轮都必须走到货物离场 (OFFSITE)。"""
    sc, _ = run
    assert len(sc.outcomes) == ROUNDS
    for oc in sc.outcomes:
        assert oc.phase_reached == "OFFSITE", (
            f"第{oc.round_no}轮止步于 {oc.phase_reached}")


def test_t12_inventory_and_bindings_drain_each_round(run):
    """每轮离场后在库 0、活动绑定 0 —— 无残留。"""
    sc, rep = run
    for oc in sc.outcomes:
        assert oc.in_warehouse_after == 0, (
            f"第{oc.round_no}轮残留 {oc.in_warehouse_after} 件")
        assert oc.active_bindings_after == 0, (
            f"第{oc.round_no}轮残留 {oc.active_bindings_after} 条活动绑定")
    assert rep["in_warehouse_final"] == 0
    assert rep["active_bindings_final"] == 0


def test_t12_wrong_code_branch_rejected_every_round(run):
    """错码支线 20/20 正确拒绝, 且库存零变化。"""
    sc, rep = run
    assert rep["reject_branch_ok"] == ROUNDS
    for oc in sc.outcomes:
        assert oc.reject_branch_ok, f"第{oc.round_no}轮错码未被拒绝"
        assert oc.reject_reason == "wrong_pallet", (
            f"第{oc.round_no}轮拒绝原因 {oc.reject_reason}")


def test_t12_frame_matches_ledger(run):
    """「画面与库存一致」—— 逐帧恒等式校验。"""
    sc, rep = run
    cons = rep["consistency"]
    assert cons["frames"] > 0
    assert cons["consistent"], f"画面与账本不一致: {cons['anomalies'][:5]}"


def test_t12_distinct_ids_per_round(run):
    """每轮使用不同 goods_id/pallet_id (设计 1065 行)。"""
    sc, _ = run
    gids = [o.goods_id for o in sc.outcomes]
    pids = [o.pallet_id for o in sc.outcomes]
    assert len(set(gids)) == ROUNDS, "goods_id 有重复"
    assert len(set(pids)) == ROUNDS, "pallet_id 有重复"


def test_t12_boundary_events_recorded(run):
    """轮边界供料/回收须显式记录 (不掩盖空托盘物流未实现)。"""
    sc, rep = run
    assert rep["boundary_events"] >= ROUNDS
    kinds = {e.kind for e in sc.boundary_events}
    assert "supply" in kinds, "缺供料事件记录"
    assert "recycle" in kinds, "缺回收事件记录"


def test_t12_statistics_are_separated(run):
    """正例与错码拒绝分开统计, 不合并成单一成功率。"""
    _, rep = run
    assert set(rep) >= {"ok", "reject_branch_ok"}
    assert rep["ok"] == rep["reject_branch_ok"] == ROUNDS
