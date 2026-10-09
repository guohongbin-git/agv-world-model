"""W04 货位标签 — 18 货位的渲染-解码回路不变量。

设计 21.2 / 25.4 的可测子集:
* 感知只吃渲染帧, 令牌↔货位映射冻结且无冲突
* 45° 斜视预算内仍可解 (标签在货架斜向取景下仍可识别)
* 失败必须给出具名原因, 不得静默放行
"""
from __future__ import annotations

import pytest

from scene import WarehouseScene
from services.perception.labels import RegistryConflictError, parse_token, verify_crc
from services.perception.slots import (
    SLOT_KIND, SlotLabelRenderer, build_slot_registry,
)


@pytest.fixture(scope="module")
def slot_ids() -> list[str]:
    return [s.slot_id for s in WarehouseScene().slots]


@pytest.fixture(scope="module")
def registry(slot_ids):
    return build_slot_registry(slot_ids)


def test_registry_covers_every_slot_one_to_one(slot_ids, registry):
    assert len(registry) == len(slot_ids) == 18
    for sid in slot_ids:
        token = registry.token_for_entity(sid)
        assert token is not None, f"货位 {sid} 未注册"
        kind, number, crc = parse_token(token)
        assert kind == SLOT_KIND
        assert 1 <= number <= 9999
        assert verify_crc(token)
    numbers = {parse_token(registry.token_for_entity(s))[1] for s in slot_ids}
    assert len(numbers) == len(slot_ids), "令牌编号必须唯一"


def test_duplicate_slot_id_rejected(registry, slot_ids):
    """同一货位不能持有两枚令牌 (设计 R09 身份唯一)。"""
    with pytest.raises(RegistryConflictError):
        build_slot_registry([slot_ids[0], slot_ids[0]])


def test_render_decode_roundtrip_all_slots(registry, slot_ids):
    """正对拍摄: 18 个货位全部渲染→解码成功且身份正确。"""
    renderer = SlotLabelRenderer(registry, view_deg=0.0)
    results = renderer.self_check(slot_ids)
    for sid, res in results.items():
        assert res.ok, f"货位 {sid} 解码失败: {res.failure}"
        assert res.entity_id == sid, f"身份错配: 期望 {sid}, 实得 {res.entity_id}"
        assert res.token == registry.token_for_entity(sid)
        assert res.px_per_module and res.px_per_module > 1.0, \
            f"货位 {sid} 每模块像素 {res.px_per_module} 过低, 不足以解码"


def test_oblique_view_budget_up_to_45_deg(registry, slot_ids):
    """设计 21.2: 斜视 45° 预算核对 —— 45° 内仍须可解。"""
    renderer = SlotLabelRenderer(registry, view_deg=45.0)
    results = renderer.self_check(slot_ids)
    bad = {k: v.failure for k, v in results.items() if not v.ok}
    assert not bad, f"45° 斜视下失败: {list(bad)[:5]}"


def test_unknown_entity_rejected(registry):
    renderer = SlotLabelRenderer(registry)
    with pytest.raises(KeyError):
        renderer.render("Z-L9-C9")


def test_frame_hash_stable_and_content_bound(registry, slot_ids):
    """同一货位两次渲染必须同帧; 不同货位必须不同帧 (帧哈希绑定内容)。"""
    r = SlotLabelRenderer(registry)
    a1 = r.capture(slot_ids[0])
    a2 = r.capture(slot_ids[0])
    b = r.capture(slot_ids[1])
    assert a1.frame_sha256 == a2.frame_sha256, "同货位两次渲染帧哈希应一致"
    assert a1.frame_sha256 != b.frame_sha256, "不同货位帧哈希必须不同"