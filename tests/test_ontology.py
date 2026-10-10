"""W05 本体事务测试 — 文档 10.2/10.3/18.4 的可测子集。"""
import pytest

from services.world.ontology import (
    Binding, ConstraintViolation, GoodsLifecycle, GoodsUnit, Loc, LocKind,
    Pallet, PalletCargo, Slot, SlotOcc, World,
)


@pytest.fixture()
def world() -> World:
    w = World(run_id="RUN-T")
    w.register_slot(Slot("A-L3-C2", (0.415, 1.000, 0.400)))
    w.register_pallet(Pallet("PAL-001", 0.045, Loc(LocKind.STATION, "I01")))
    w.register_goods(GoodsUnit("G-001", "DEMO-A", 0.200, Loc(LocKind.FEED, "I-TRAY")))
    return w


def test_bind_and_store_atomic(world: World):
    world.bind_and_store("G-001", "PAL-001", Loc(LocKind.STATION, "I01"), "T1", 1.0)
    assert world.in_warehouse_count() == 1
    assert world.goods_location("G-001") == "PALLET:PAL-001"
    assert world.active_binding("G-001") == ("PAL-001", 1)
    loc, state, _ = world.pallet("PAL-001")
    assert state == PalletCargo.LOADED.value


def test_double_bind_rejected(world: World):
    world.bind_and_store("G-001", "PAL-001", Loc(LocKind.STATION, "I01"), "T1", 1.0)
    with pytest.raises(ConstraintViolation):
        world.bind_and_store("G-001", "PAL-001", Loc(LocKind.STATION, "I01"), "T2", 2.0)


def test_bind_to_loaded_pallet_rejected(world: World):
    w2 = World(run_id="RUN-T2")
    w2.register_pallet(Pallet("PAL-002", 0.045, Loc(LocKind.STATION, "I01"),
                              cargo_state=PalletCargo.LOADED))
    w2.register_goods(GoodsUnit("G-002", "DEMO-A", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    with pytest.raises(ConstraintViolation):
        w2.bind_and_store("G-002", "PAL-002", Loc(LocKind.STATION, "I01"), "T1", 1.0)


def test_unbind_grip_release_offsite_lifecycle(world: World):
    world.bind_and_store("G-001", "PAL-001", Loc(LocKind.STATION, "I01"), "T1", 1.0)
    # 出库: 先 OUTBOUND_PENDING (由 move/update 承担, 这里直接测卸货链)
    world._conn.execute("UPDATE goods SET lifecycle=? WHERE goods_id=?",
                        (GoodsLifecycle.OUTBOUND_PENDING.value, "G-001"))
    world._conn.commit()
    world.unbind_at_output("G-001", "PAL-001", "T2", 2.0)
    assert world.pallet("PAL-001")[1] == PalletCargo.EMPTY.value
    assert world.active_binding("G-001") is None
    # 设计 25.5「卸货接管」行标的是「计数不变」—— OUTBOUND_PENDING
    # **仍计在库**, 直到显式离场边界事件才 −1。
    assert world.in_warehouse_count() == 1, "卸货后仍计在库"
    # 离场
    world.exit_offsite("G-001", "T3", 3.0)
    assert world.goods_location("G-001") == "FEED:O-TRAY"
    assert world.in_warehouse_count() == 0, "离场后计数 −1"
    with pytest.raises(ConstraintViolation):
        world.exit_offsite("G-001", "T4", 4.0)  # 重复离场拒绝


def test_unknown_never_deleted(world: World):
    world.bind_and_store("G-001", "PAL-001", Loc(LocKind.STATION, "I01"), "T1", 1.0)
    world.move_pallet("PAL-001", Loc(LocKind.ROBOT, "F01"), "T2", 2.0)
    assert world.in_warehouse_count() == 1  # 运输中计数不变
    assert world.active_binding("G-001")[0] == "PAL-001"


def test_event_id_unique_and_replay(world: World):
    world.bind_and_store("G-001", "PAL-001", Loc(LocKind.STATION, "I01"), "T1", 1.0)
    rows = world._conn.execute("SELECT event_id, type FROM events ORDER BY event_id").fetchall()
    assert len(rows) == 1 and rows[0][1] == "CargoBindingConfirmed"


def test_command_idempotency(world: World):
    assert world.record_command("F01", "CMD-1", "ACCEPTED") is None
    again = world.record_command("F01", "CMD-1", "ACCEPTED")
    assert again == "ACCEPTED"  # 同 payload 重发返回原状态


def test_transport_keeps_binding(world: World):
    world.bind_and_store("G-001", "PAL-001", Loc(LocKind.STATION, "I01"), "T1", 1.0)
    world.move_pallet("PAL-001", Loc(LocKind.ROBOT, "F01"), "T2", 2.0)
    world.move_pallet("PAL-001", Loc(LocKind.SLOT, "A-L3-C2"), "T3", 3.0)
    assert world.goods_location("G-001") == "PALLET:PAL-001"
    assert world.in_warehouse_count() == 1
