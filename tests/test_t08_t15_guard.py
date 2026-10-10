"""T08 错误拦截 / T15 工位互锁 —— 身份核验与进入放行。

设计判据:
|T08|错产品 错托盘 错货位各 20 次|60 次均不放行, 库存不错误提交|
|T15|M01/M02活动、收拢未达、收拢旧帧、R01未退出各10次|
    |进入请求全部拒绝; 合法STOW_CLEAR与版本一致后原子放行|

「不放行」的判定不是"抛了异常", 而是**库存状态零变化** —— 拒绝后
World 必须与拒绝前逐字段一致。这才是 T08 的实质。
"""
from __future__ import annotations

import numpy as np
import pytest

from services.perception.scan import ScanFailure, ScanFailureReason, ScanResult
from services.scheduler.identity_guard import (IdentityGuard, RejectReason,
                                               Verification)
from services.scheduler.reservations import (PartialAcquisition,
                                             ReservationBoard,
                                             ReservationError)
from services.world.ontology import GoodsUnit, Loc, LocKind, Pallet, Slot, World

N = 20                      # 设计要求「各 20 次」


def ok_scan(entity: str, frame: str = "f0") -> ScanResult:
    return ScanResult(frame_id=frame, frame_sha256=f"sha-{frame}",
                      token=f"T-{entity}", entity_id=entity, failure=None,
                      px_per_module=8.0)


def bad_scan(reason=ScanFailureReason.CRC_MISMATCH, frame="fb") -> ScanResult:
    return ScanResult(frame_id=frame, frame_sha256=f"sha-{frame}", token=None,
                      entity_id=None, failure=ScanFailure(reason, "注入", f"sha-{frame}"),
                      px_per_module=None)


def fresh_world(run_id: str) -> World:
    w = World(run_id=run_id)
    w.register_slot(Slot("A-L1-C1", (0.415, 1.0, 0.4)))
    w.register_slot(Slot("A-L1-C2", (0.415, 1.05, 0.4)))
    w.register_pallet(Pallet("PAL-001", 0.045, Loc(LocKind.STATION, "I01")))
    w.register_pallet(Pallet("PAL-002", 0.045, Loc(LocKind.STATION, "I02")))
    w.register_goods(GoodsUnit("G-001", "A", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    return w


def snapshot(w: World) -> tuple:
    """库存快照 —— 拒绝后必须与拒绝前逐字段一致。"""
    rows = w._conn.execute(
        "SELECT goods_id, location, lifecycle FROM goods ORDER BY goods_id").fetchall()
    pal = w._conn.execute(
        "SELECT pallet_id, location, cargo_state FROM pallets "
        "ORDER BY pallet_id").fetchall()
    bind = w._conn.execute(
        "SELECT goods_id, pallet_id, binding_version, active FROM bindings "
        "ORDER BY goods_id, binding_version").fetchall()
    return (tuple(rows), tuple(pal), tuple(bind), w.in_warehouse_count())


# ---------------------------------------------------------------- T08
@pytest.mark.parametrize("which,expected,wrong", [
    ("错产品", "G-001", "G-999"),
    ("错托盘", "PAL-001", "PAL-999"),
    ("错货位", "A-L1-C1", "A-L1-C2"),
])
def test_t08_wrong_identity_never_released(which, expected, wrong):
    """错产品/错托盘/错货位各 20 次, 60 次全部拒绝且库存零变化。"""
    key = {"错产品": "scanned_goods", "错托盘": "scanned_pallet",
           "错货位": "scanned_slot"}[which]
    for i in range(N):
        w = fresh_world(f"T08-{which}-{i}")
        g = IdentityGuard()
        before = snapshot(w)
        kw = {"expected_goods": "G-001", "expected_pallet": "PAL-001",
              "expected_slot": "A-L1-C1",
              "scanned_goods": ok_scan("G-001", f"g{i}"),
              "scanned_pallet": ok_scan("PAL-001", f"p{i}"),
              "scanned_slot": ok_scan("A-L1-C1", f"s{i}")}
        kw[key] = ok_scan(wrong, f"x{i}")          # 注入错身份
        v = g.verify_handoff(**kw)
        assert not v, f"{which} 第{i+1}次被错误放行"
        assert snapshot(w) == before, f"{which} 第{i+1}次库存被改动"


def test_t08_correct_identity_passes():
    """身份正确时必须放行 —— 否则闸门形同虚设。"""
    g = IdentityGuard()
    v = g.verify_handoff(
        expected_goods="G-001", expected_pallet="PAL-001",
        expected_slot="A-L1-C1",
        scanned_goods=ok_scan("G-001", "g"), scanned_pallet=ok_scan("PAL-001", "p"),
        scanned_slot=ok_scan("A-L1-C1", "s"))
    assert v.ok, v.detail


def test_t08_scan_failure_blocks_release():
    """扫码失败(CRC/无符号等) 一律拒绝, 且不得当成功。"""
    for reason in ScanFailureReason:
        g = IdentityGuard()
        v = g.verify_handoff(
            expected_goods="G-001", expected_pallet="PAL-001",
            scanned_goods=bad_scan(reason, f"f-{reason.value}"),
            scanned_pallet=ok_scan("PAL-001", f"p-{reason.value}"))
        assert not v, f"{reason.value} 被错误放行"
        assert v.reason is RejectReason.NO_IDENTITY


def test_t08_duplicate_frame_rejected():
    """重复帧不得当两次独立观测 (设计 R08: 独立试验ID与原帧)。"""
    g = IdentityGuard()
    s = ok_scan("G-001", "same")
    assert g.verify_scan(s, "G-001").ok
    again = g.verify_scan(ok_scan("G-001", "same"), "G-001")
    assert not again and again.reason is RejectReason.STALE_FRAME
    g.reset_frames()
    assert g.verify_scan(ok_scan("G-001", "same"), "G-001").ok


def test_t08_missing_observation_blocks():
    """缺观测 (不是错观测) 也必须拒绝 —— 证据不足不等于通过。"""
    g = IdentityGuard()
    v = g.verify_handoff(expected_goods="G-001", expected_pallet="PAL-001",
                         scanned_goods=ok_scan("G-001", "g"))
    assert not v and v.reason is RejectReason.NO_IDENTITY


# ---------------------------------------------------------------- T15
def test_t15_entry_refused_while_arm_active():
    """M01 活动时 R01 请求进入 I01 → 拒绝 (T15)。"""
    b = ReservationBoard()
    b.acquire_all(["I01", "M01_WORKSPACE"], "M01", "TASK-1", 0.0)
    with pytest.raises(PartialAcquisition):
        b.acquire_all(["I01"], "R01", "TASK-2", 0.1)
    assert b.holder_of("I01") == "M01"


def test_t15_entry_refused_when_stow_not_reached():
    """收拢未达 (BLOCKED) → 不得放行车辆进入。"""
    b = ReservationBoard()
    b.acquire_all(["M01_WORKSPACE"], "M01", "TASK-1", 0.0)
    # M01 尚未收拢 → 不释放工位锁
    assert b.holder_of("M01_WORKSPACE") == "M01"
    with pytest.raises(PartialAcquisition):
        b.acquire_all(["M01_WORKSPACE", "I01"], "R01", "TASK-2", 0.1)


def test_t15_stale_stow_evidence_refused():
    """收拢旧帧 (证据过期) → UNKNOWN, 不得据此放行。"""
    from services.adapters.manipulator import ManipulatorAdapter
    from sim.manipulator_cell import ManipulatorCell

    c = ManipulatorCell(arm="M01")
    for _ in range(3000):
        c.step(c.substeps)
    a = ManipulatorAdapter("M01", c, capabilities=frozenset(
        ["arm_move_joints", "arm_move_tcp", "gripper_close", "gripper_open",
         "arm_stow", "stop"]))
    assert a.execute("stow", "s1") is not None
    assert a.stow_state(refresh=True).value == "STOW_CLEAR"
    # 冻结物理时钟 → 证据超出 0.1s 新鲜度窗口
    frozen = float(c.data.time)
    c.data.time = frozen + 0.5
    assert a.stow_state(refresh=False).value == "UNKNOWN", "旧证据不得判为 CLEAR"


def test_t15_vehicle_not_exited_refused():
    """R01 未退出源区 → 目标区不得放行 (设计 19.3: 带载退出完成前锁定)。"""
    b = ReservationBoard()
    # R01 占住交接站 + 其退出通道; 该通道未放行前 H01 不得进车
    b.register_exit_gate("EXIT_I01", ["I01", "H01", "O01"])
    b.acquire_all(["I01", "EXIT_I01"], "R01", "TASK-1", 0.0)
    with pytest.raises(PartialAcquisition) as ei:
        b.acquire_all(["H01"], "F01", "TASK-2", 0.1)
    assert "退出未完成" in str(ei.value)
    # R01 完成退出 → 释放退出锁, F01 仍需 H01 本身空闲
    b.release_all("R01")
    assert b.acquire_all(["H01"], "F01", "TASK-2", 0.2) == ["H01"]


def test_t15_atomic_release_only_after_fresh_stow():
    """合法 STOW_CLEAR + 版本一致 → 原子放行 (T15 正例)。"""
    b = ReservationBoard()
    b.acquire_all(["M01_WORKSPACE", "I01"], "M01", "TASK-1", 0.0)
    # M01 收拢完成 → 原子交接给 R01
    b.release_all("M01")
    got = b.acquire_all(["M01_WORKSPACE", "I01"], "R01", "TASK-2", 0.5)
    assert set(got) == {"M01_WORKSPACE", "I01"}
    assert b.holder_of("I01") == "R01"


def test_t15_partial_acquisition_holds_nothing():
    """部分取得必须**全不占** (设计 19.3: 不能部分占锁继续运动)。"""
    b = ReservationBoard()
    b.acquire_all(["H01"], "F01", "TASK-0", 0.0)
    with pytest.raises(PartialAcquisition):
        b.acquire_all(["I01", "H01"], "R01", "TASK-1", 0.1)
    assert b.holder_of("I01") is None, "失败的取得留下了半截锁"
    assert b.holder_of("H01") == "F01"


def test_t15_expired_lease_becomes_recovery_locked():
    """租约过期 → RECOVERY_LOCKED, 新进入被拒 (设计 19.3)。"""
    b = ReservationBoard(lease_s=1.0)
    b.acquire_all(["I01"], "R01", "TASK-1", 0.0)
    expired = b.tick(2.0)
    assert "I01" in expired
    with pytest.raises(PartialAcquisition):
        b.acquire_all(["I01"], "F01", "TASK-2", 2.1)
    # 观测确认空闲后才能解锁
    assert b.unlock_after_observation("I01", observed_free=True)
    assert b.acquire_all(["I01"], "F01", "TASK-2", 2.2) == ["I01"]