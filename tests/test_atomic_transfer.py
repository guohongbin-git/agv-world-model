"""B1 事务层可靠性验收 (GPT 评审 P0-1/P1-4/P1-7) — 全实测。

- 原子性: commit_transfer 中途注入 sqlite 故障 (代理在事务内第 N 次写入
  时断开) → 全表快照逐字节不变 (零残留);
- 幂等: 同 idempotency_key 重放返回首次结果, 变更/事件不重复应用;
- 乐观版本: 期望版本不符 → VersionConflict 拒绝且零残留;
- 证据: EvidenceVerdict 多帧门禁 (承托率/卸载/稳定窗/静止/位姿/租约)、
  过期拒绝、快照持久化且事件可经 evidence_id 追溯;
- 恢复: 物理已成功而提交失败 → RECONCILIATION_REQUIRED (任务挂起,
  禁止再启动) → 手动 reconcile 同键重放 → 重新 start 队列走到终态,
  与无故障 E2E 账本逐位一致;
- 身份: reset 重建全局唯一 run_id, 事件流/账本引用同步。
"""
from __future__ import annotations

import json
import re
import sqlite3
import types

import pytest

from services.world import evidence as ev
from services.world.evidence import (FRAME_INTERVAL_S, EvidenceFrame,
                                     EvidenceVerdict, build_verdict,
                                     collect_frames)
from services.world.ontology import (ConstraintViolation, GoodsLifecycle,
                                     GoodsUnit, Loc, LocKind, Pallet,
                                     PalletCargo, Slot, SlotOcc,
                                     VersionConflict, World)

TABLES = ("pallets", "goods", "slots", "bindings", "events", "command_log",
          "execution_evidence", "transfers")


def _snapshot(w: World) -> dict:
    return {t: w._conn.execute(f"SELECT * FROM {t}").fetchall() for t in TABLES}


class _CommitFaultConn:
    """sqlite 连接代理: 武装状态下第 ``fail_on_write`` 次 DML 写入注入故障。

    只在 armed=True 时计数 (测试直接武装, 或由 commit_transfer 包装器
    武装), 模拟「提交中途断电」—— 异常发生在事务内部, 必须整体回滚。
    """

    def __init__(self, real: sqlite3.Connection, fail_on_write: int):
        self._real = real
        self.fail_on_write = fail_on_write
        self.armed = False
        self.writes = 0

    def execute(self, sql, params=()):
        if self.armed and sql.lstrip().upper().startswith(
                ("INSERT", "UPDATE", "DELETE", "REPLACE")):
            self.writes += 1
            if self.writes == self.fail_on_write:
                raise sqlite3.OperationalError("INJECTED mid-commit fault")
        return self._real.execute(sql, params)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------- 证据构造
@pytest.fixture()
def world() -> World:
    w = World(run_id="RUN-AT")
    w.register_slot(Slot("A-L3-C1", (0.0, 0.0, 0.400), SlotOcc.OCCUPIED))
    w.register_pallet(Pallet("PAL-001", 0.022, Loc(LocKind.SLOT, "A-L3-C1"),
                             PalletCargo.LOADED))
    w.register_goods(GoodsUnit("G-001", "SKU-1", 0.023,
                               Loc(LocKind.PALLET, "PAL-001"),
                               GoodsLifecycle.IN_WAREHOUSE))
    return w


def _frames(n: int = 11, t0: float = 1.0, **over) -> tuple[EvidenceFrame, ...]:
    base: dict = dict(new_support=1.0, old_support=0.0, pallet_x=0.1,
                      pallet_y=0.0, pallet_z=0.4, speed=1e-4)
    base.update(over)
    return tuple(EvidenceFrame(t=t0 + FRAME_INTERVAL_S * i, **base)
                 for i in range(n))


def _verdict(w: World, frames=None, lease="aisle:F01") -> EvidenceVerdict:
    return build_verdict(verdict_id=f"{w.run_id}-EVD-0001", kind="PICK",
                         task_id="T1", pallet_id="PAL-001", vehicle_id="F01",
                         lease_id=lease, frames=frames or _frames())


def _pick_mutations() -> list[tuple[str, tuple]]:
    return [
        ("UPDATE pallets SET location=?, entity_version=entity_version+1 "
         "WHERE pallet_id=?", ("ROBOT:F01", "PAL-001")),
        ("UPDATE slots SET occupancy=? WHERE slot_id=?",
         (SlotOcc.EMPTY.value, "A-L3-C1"))]


def _pick_events(vid_id: str) -> list[tuple[str, dict]]:
    return [("PalletMoved", {"pallet_id": "PAL-001",
                             "to_location": "ROBOT:F01",
                             "evidence_id": vid_id})]


def _commit(w: World, v: EvidenceVerdict, key: str, expected=None,
            mutations=None, events=None, sim_time_s: float = 1.5):
    return w.commit_transfer(
        v.verdict_id, expected if expected is not None else {"pallets:PAL-001": 0},
        mutations if mutations is not None else _pick_mutations(),
        events if events is not None else _pick_events(v.verdict_id),
        key, task_id="T1", sim_time_s=sim_time_s, evidence_json=v.to_json())


# ================================================================ World 层
@pytest.mark.parametrize("fail_on_write", [1, 2, 3, 4, 5])
def test_mid_commit_fault_leaves_zero_residue(world, fail_on_write):
    """事务内第 N 次写入断开 (N=1..5 覆盖变更/事件/证据/登记) → 全表零残留。"""
    v = _verdict(world)
    before = _snapshot(world)
    proxy = _CommitFaultConn(world._conn, fail_on_write)
    world._conn = proxy
    proxy.armed = True
    with pytest.raises(sqlite3.OperationalError, match="INJECTED"):
        _commit(world, v, "K-ATOMIC")
    proxy.armed = False
    world._conn = proxy._real
    assert _snapshot(world) == before, "中途故障留下了局部持久化状态"
    assert not world.transfer_recorded("K-ATOMIC")
    # 故障后连接可用: 同键重试成功落账 (恢复语义的基础)
    out = _commit(world, v, "K-ATOMIC")
    assert out.committed and not out.replayed


def test_idempotent_replay_returns_first_result(world):
    """同 idempotency_key 重放: 返回首次结果, 变更与事件均不重复。"""
    v = _verdict(world)
    out1 = _commit(world, v, "K-IDEM")
    assert out1.committed and not out1.replayed
    assert world.pallet("PAL-001")[0] == "ROBOT:F01"
    n_events = world._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    # 重放携带**不同**的 mutations/events: 仍必须返回首次结果
    out2 = _commit(world, v, "K-IDEM",
                   mutations=[("UPDATE pallets SET location='X' "
                               "WHERE pallet_id='PAL-001'",)],
                   events=[("ShouldNeverAppear", {})])
    assert out2.committed and out2.replayed
    assert out2.event_ids == out1.event_ids
    assert world.pallet("PAL-001")[0] == "ROBOT:F01"      # 未应用新 mutations
    assert world._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] \
        == n_events                                        # 未写入新事件
    assert not world._conn.execute(
        "SELECT 1 FROM events WHERE type='ShouldNeverAppear'").fetchone()
    assert out2.new_versions == out1.new_versions


def test_version_conflict_rejected_zero_residue(world):
    """乐观版本检查: 期望版本不符 → VersionConflict, 且零残留。"""
    v = _verdict(world)
    before = _snapshot(world)
    with pytest.raises(VersionConflict):
        _commit(world, v, "K-VC", expected={"pallets:PAL-001": 7})
    assert _snapshot(world) == before
    assert not world.transfer_recorded("K-VC")


def test_version_bump_drives_subsequent_conflict(world):
    """提交递增 entity_version; 用旧期望再提交 (新键) 必须冲突。"""
    v = _verdict(world)
    out = _commit(world, v, "K-BUMP")
    assert out.new_versions["pallets:PAL-001"] == 1
    with pytest.raises(VersionConflict):
        _commit(world, v, "K-BUMP-2")            # 仍带 expected=0 (已过期)
    assert world.pallet("PAL-001")[2] == 1


def test_stale_evidence_rejected(world):
    """证据有效期: sim_time 超过 valid_until → 拒绝提交 (零残留)。"""
    v = _verdict(world, frames=_frames(t0=1.0))   # valid_until = 1.5
    before = _snapshot(world)
    with pytest.raises(ConstraintViolation, match="过期"):
        _commit(world, v, "K-STALE", sim_time_s=2.6)
    assert _snapshot(world) == before


def test_evidence_snapshot_persisted_and_traceable(world):
    """证据快照随事务持久化; 事件 payload 可反查原始多帧采样。"""
    v = _verdict(world)
    _commit(world, v, "K-TRACE", sim_time_s=1.5)
    stored = world.evidence(v.verdict_id)
    assert stored is not None
    assert len(stored["frames"]) == 11 and stored["kind"] == "PICK"
    assert stored["gates"]["min_new_support"] == ev.MIN_NEW_SUPPORT
    assert EvidenceVerdict.from_json(json.dumps(stored)) == v   # 往返一致
    row = world._conn.execute(
        "SELECT payload FROM events WHERE type='PalletMoved'").fetchone()
    eid = json.loads(row[0])["evidence_id"]
    assert eid == v.verdict_id
    assert world._conn.execute(
        "SELECT COUNT(*) FROM execution_evidence WHERE evidence_id=?",
        (eid,)).fetchone()[0] == 1
    # 幂等重放不复制证据快照
    _commit(world, v, "K-TRACE")
    assert world._conn.execute(
        "SELECT COUNT(*) FROM execution_evidence WHERE evidence_id=?",
        (eid,)).fetchone()[0] == 1


# ================================================================ 证据门禁
def _gate_case(name: str, frames, lease, expect_fail: str | None):
    return pytest.param(frames, lease, expect_fail, id=name)


@pytest.mark.parametrize("frames,lease,expect_fail", [
    _gate_case("pass", _frames(), "aisle:F01", None),
    _gate_case("low_support", _frames(new_support=0.70), "aisle:F01",
               "承托率"),
    _gate_case("old_support_unloaded", _frames(old_support=0.30), "aisle:F01",
               "旧支撑"),
    _gate_case("short_window", _frames(n=5), "aisle:F01", "稳定窗"),
    _gate_case("moving_pallet", _frames(speed=2e-2), "aisle:F01", "静止"),
    _gate_case("no_lease", _frames(), None, "租约"),
    _gate_case("pose_drift",
               _frames()[:5] + _frames(n=6, pallet_x=0.15)[1:], "aisle:F01",
               "位姿"),
])
def test_evidence_verdict_gates(frames, lease, expect_fail):
    v = build_verdict(verdict_id="V", kind="PICK", task_id="T",
                      pallet_id="P", vehicle_id="F01", lease_id=lease,
                      frames=frames)
    if expect_fail is None:
        assert v.passed and v.valid_until == pytest.approx(
            frames[-1].t + ev.VERDICT_TTL_S)
    else:
        assert not v.passed
        assert any(expect_fail in f for f in v.failures), v.failures


def test_collect_frames_covers_stable_window():
    clock = {"t": 0.0}

    def step():
        clock["t"] += FRAME_INTERVAL_S

    def probe():
        return EvidenceFrame(t=clock["t"], new_support=1.0, old_support=0.0,
                             pallet_x=0.0, pallet_y=0.0, pallet_z=0.0,
                             speed=0.0)

    frames = collect_frames(step=step, probe=probe)
    assert len(frames) == 1 + round(ev.STABLE_WINDOW_S / FRAME_INTERVAL_S)
    assert frames[-1].t - frames[0].t == pytest.approx(ev.STABLE_WINDOW_S)


# ================================================================ 服务层 E2E
@pytest.fixture(scope="module")
def svc():
    from twin.server import WorldService
    s = WorldService()
    s.gate.speed = 64.0        # 测试加速: 只解除墙钟配速, 物理步进不变
    return s


def test_reconciliation_required_and_recovery(svc):
    """物理成功 + 提交失败 → 待核对 → 手动 reconcile → 继续队列到终态。

    全链语义: 中途故障零残留 (回滚) / 待核对期间禁止再启动 / 同键重放
    落账 / 重新 start 后账本与无故障 E2E 逐位一致 (幂等键去重)。
    """
    svc.tasks = list(svc.HANDOFF_TASKS)
    key0 = f"{svc.task_id}:pick:PAL-001:A-L3-C1"

    real = svc.world._conn
    proxy = _CommitFaultConn(real, fail_on_write=1)   # 第一条 UPDATE 即断
    svc.world._conn = proxy
    orig_commit = World.commit_transfer

    def armed_commit(*a, **kw):
        proxy.armed = True
        try:
            return orig_commit(svc.world, *a, **kw)
        finally:
            proxy.armed = False

    svc.world.commit_transfer = armed_commit
    svc.mode = "running"
    try:
        svc.run_mission()
    finally:
        svc.world.__dict__.pop("commit_transfer", None)
        svc.world._conn = real

    # ---- 1) RECONCILIATION_REQUIRED 可达: 任务挂起, 有叙述有事件
    assert svc.mode == "reconciliation_required", svc.step_label
    assert list(svc.reconciliation) == [key0]
    assert any(m["text"].startswith("提交通道故障") for m in svc.chat)
    rows, _ = svc.events_since(0)
    assert any(e[1] == "ReconciliationRequired" for e in rows)

    # ---- 2) 零残留: 托盘仍在架上, 货位仍占用, 无转移事件/登记
    b = svc.board()
    pallets = {p[0]: (p[1], p[2]) for p in b["pallets"]}
    assert pallets["PAL-001"] == ("SLOT:A-L3-C1", "LOADED")
    assert dict(b["slots"])["A-L3-C1"] == "OCCUPIED"
    assert not any(e[1] == "PalletMoved" for e in rows)
    assert not svc.world.transfer_recorded(key0)

    # ---- 3) 待核对期间禁止再启动 (不允许依赖该状态的后续动作)
    assert svc.cmd("start")["mode"] == "reconciliation_required"

    # ---- 4) 手动 reconcile: 同键重放, 首次落账
    r = svc.reconcile("不存在的键")
    assert r["ok"] is False
    r = svc.reconcile(key0)
    assert r["ok"] is True and r["replayed"] is False
    assert svc.reconciliation == {}
    assert svc.mode == "idle"
    rows, _ = svc.events_since(0)
    chain = [json.loads(e[2])["to_location"] for e in rows
             if e[1] == "PalletMoved"]
    assert chain == ["ROBOT:F01"]                       # 恰一次, 无重复
    assert any(e[1] == "ReconciliationApplied" for e in rows)
    b = svc.board()
    pallets = {p[0]: (p[1], p[2]) for p in b["pallets"]}
    assert pallets["PAL-001"] == ("ROBOT:F01", "LOADED")
    assert dict(b["slots"])["A-L3-C1"] == "EMPTY"

    # ---- 5) 重新 start: 队列继续走到终态, 账本与无故障 E2E 一致
    svc.cmd("start")
    svc._thread.join(timeout=900)
    assert svc.mode == "done", svc.step_label
    rows, _ = svc.events_since(0)
    chain = [json.loads(e[2])["to_location"] for e in rows
             if e[1] == "PalletMoved"]
    assert chain == ["ROBOT:F01", "STATION:S1", "ROBOT:F02", "SLOT:H01"]
    assert sum(1 for e in rows if e[1] == "OutboundPending") == 1
    assert not any(e[1] == "CommitEvidenceMissing" for e in rows)
    b = svc.board()
    pallets = {p[0]: (p[1], p[2]) for p in b["pallets"]}
    goods = {g[0]: (g[2], g[3]) for g in b["goods"]}
    slots = dict(b["slots"])
    assert pallets["PAL-001"] == ("SLOT:H01", "LOADED")
    assert goods["G-001"] == ("PALLET:PAL-001", "OUTBOUND_PENDING")
    assert slots["H01"] == "OCCUPIED" and slots["A-L3-C1"] == "EMPTY"
    assert b["count"] == 3

    # ---- 6) 证据快照可追溯: 任一转移事件反查 11 帧原始采样
    row = next(e for e in rows if e[1] == "PalletMoved")
    eid = json.loads(row[2])["evidence_id"]
    snap = svc.world.evidence(eid)
    assert snap and len(snap["frames"]) == 11 and snap["kind"] == "PICK"
    assert snap["vehicle_id"] == "F01" and snap["lease_id"] == "aisle:F01"


def test_reset_rebuilds_global_unique_identity():
    """P1-7: reset 重建 run 身份 —— 全局唯一 uuid, 事件流/账本引用同步。"""
    from twin.server import WorldService
    s = WorldService()
    ids = [s.run_id]
    db_paths = [s.world.db_path]
    for _ in range(2):
        s.cmd("reset")
        ids.append(s.run_id)
        db_paths.append(s.world.db_path)
    pat = re.compile(r"^DEMO-\d{8}-\d{6}-[0-9a-f]{8}$")
    assert all(pat.match(i) for i in ids)
    assert len(set(ids)) == 3, "同秒多次 reset 必须产生互不相同的身份"
    assert len(set(db_paths)) == 3
    assert s.task_id == s.run_id + "-T01"
    assert s.world.run_id == s.run_id
    assert s.board()["run_id"] == s.run_id
    # 新 run 的事件引用当前身份
    s.world.emit(s.task_id, "Probe", {}, 0.0)
    eid = s.world._conn.execute("SELECT event_id FROM events").fetchone()[0]
    assert eid.startswith(s.run_id + "-EVT-")
    # 挂起提交状态随 reset 清空
    assert s.reconciliation == {}
