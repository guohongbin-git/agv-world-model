"""T12 完整演示场景 —— 同一冻结版本连续 20 轮闭环。

设计判据:
|T12 完整演示|同一冻结版本连续20轮|每轮 M01装托、R01/F01上架下架、M02卸货
     |及货物离场；错码支线正确拒绝，画面与库存一致|

以及 (设计 1065 行):
「连续20轮允许在每轮边界生成新的入料货物及空PAL、清理上一轮空PAL作为
显式**外部供料事件**; 不能在轮内复位或自动纠正物理失败, 每轮使用不同
goods_id/pallet_id 且记录边界事件, 不借此掩盖空托盘物流尚未实现。」

所以本模块严格区分两类事件:
* **轮内事件**: 装托/运输/上架/下架/卸货/离场 —— 任一失败即本轮失败,
  **不得**自动重试掩盖;
* **轮边界事件**: 供料 (新 goods + 新空 PAL) 与回收 (清理上一轮空 PAL) ——
  显式记录, 因为空托盘回流本版未实现。

正例 / 错码拒绝 / 受控故障**分别统计** (设计 621 行: 不能合并成一个
含义不明的成功率)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from services.scheduler.identity_guard import IdentityGuard, RejectReason
from services.perception.scan import ScanResult
from services.scheduler.replay import (EventKind, EventReplay,
                                     ReplayOutcome)
from services.scheduler.reservations import ReservationBoard
from services.scheduler.tasks import StepOutcome, TaskScheduler
from services.ui.run_recorder import RunRecorder
from services.world.ontology import (GoodsUnit, Loc, LocKind, Pallet,
                                     PalletCargo, Slot, World)

PALLET_EMPTY = PalletCargo.EMPTY.value

ROUNDS = 20
SLOTS = [f"A-L1-C{i}" for i in range(1, 4)]


def _run_id(i: int) -> str:
    return f"T12-R{i:02d}"


@dataclass
class BoundaryEvent:
    """轮边界外部供料/回收 (显式记录, 不掩盖空托盘物流未实现)。"""

    round_no: int
    sim_time_s: float
    kind: str            # "supply" / "recycle"
    detail: str

    def as_dict(self) -> dict:
        return {"round_no": self.round_no, "sim_time_s": self.sim_time_s,
                "kind": self.kind, "detail": self.detail}


@dataclass
class RoundOutcome:
    round_no: int
    ok: bool
    phase_reached: str = ""
    goods_id: str = ""
    pallet_id: str = ""
    reject_branch_ok: bool = False
    reject_reason: str = ""
    failures: list[str] = field(default_factory=list)
    boundary_before: list[str] = field(default_factory=list)
    boundary_after: list[str] = field(default_factory=list)
    in_warehouse_after: int = 0
    active_bindings_after: int = 0


class DemoScenario:
    """T12 场景驱动器。

    每轮: 边界供料 → 主流程 (入库→上架→下架→卸货→离场) → 边界回收。
    物理部分用**事件账本**推进 (M01/M02/R01/F01 的动作已由各自单元的
    T13/T06 覆盖), 本场景聚焦**跨设备事务链**的一致性 —— 库存、绑定、
    计数、错码拒绝, 以及「画面与库存一致」(由 RunRecorder 自检)。
    """

    def __init__(self, *, rounds: int = ROUNDS):
        self.rounds = rounds
        self.world = World(run_id="T12")
        for sid in SLOTS:
            self.world.register_slot(Slot(sid, (0.415, 1.0, 0.4)))
        self.board = ReservationBoard()
        self.board.register_exit_gate("EXIT_H01", ["H01", "I01", "O01"])
        self.recorder = RunRecorder(self.world)
        self.boundary_events: list[BoundaryEvent] = []
        self.outcomes: list[RoundOutcome] = []
        self.replay = EventReplay(self.world)
        self.guard = IdentityGuard()
        self._t = 0.0

    # ------------------------------------------------------------ 边界
    def _advance(self, dt: float = 0.5) -> None:
        self._t += dt

    def _supply(self, i: int) -> RoundOutcome:
        """轮边界供料: 新 goods (入料托台) + 新空 PAL (I01)。"""
        gid = f"G-{i:03d}"
        pid = f"PAL-{i:03d}"
        before = []
        if i > 1:
            prev = f"PAL-{i-1:03d}"
            prow = self.world.pallet(prev)
            if prow and prow[1] == PALLET_EMPTY:
                # LocKind 没有 OFFSITE —— 回收是**外部事件**, 落在
                # UNKNOWN 之外没有合适语义, 故记为 UNKNOWN 位置的显式
                # 标记 RECYCLE (设计: 空托盘回流本版未实现)。
                self.world._conn.execute(
                    "UPDATE pallets SET location=? WHERE pallet_id=?",
                    (str(Loc(LocKind.UNKNOWN, "RECYCLE")), prev))
                self.world._conn.commit()
                ev = BoundaryEvent(i, self._t, "recycle",
                                   f"{prev} 空托盘移出 (空托盘回流本版未实现)")
                self.boundary_events.append(ev)
                before.append(f"recycle:{prev}")
        self.world.register_goods(GoodsUnit(gid, "A", 0.2,
                                           Loc(LocKind.FEED, "I-TRAY")))
        self.world.register_pallet(Pallet(pid, 0.045, Loc(LocKind.STATION, "I01")))
        ev = BoundaryEvent(i, self._t, "supply",
                           f"供料 {gid}@入料托台 + 空托盘 {pid}@I01")
        self.boundary_events.append(ev)
        before.append(f"supply:{gid},{pid}")
        return RoundOutcome(round_no=i, ok=False, goods_id=gid, pallet_id=pid,
                            boundary_before=before)

    # ------------------------------------------------------------ 主流程
    def _main_flow(self, oc: RoundOutcome) -> RoundOutcome:
        gid, pid = oc.goods_id, oc.pallet_id
        r = self.replay
        oc.phase_reached = "FEED_PICK"
        self.recorder.note(f"入料接管 {gid}")
        if r.apply(EventKind.FEED_PICK, gid, self._next_seq()) is not ReplayOutcome.APPLIED:
            oc.failures.append("入料接管未生效")

        oc.phase_reached = "LOAD_PALLET"
        self.recorder.note(f"入库装托 {gid} → {pid}")
        if r.apply(EventKind.BIND_STORE, gid, self._next_seq(), pid) \
                is not ReplayOutcome.APPLIED:
            oc.failures.append("入库装托未提交")

        oc.phase_reached = "TRANSPORT"
        self.recorder.note(f"R01/F01 运输 {pid} → 货位")
        self.world.move_pallet(pid, Loc(LocKind.STATION, SLOTS[0]), "T12", self._t)
        # 设计 25.5: 运输只动 PAL 位置, 绑定货物有效位置仍是 PALLET
        if self.world.goods_location(gid) != f"PALLET:{pid}":
            oc.failures.append("运输后货物位置不再是 PALLET")

        oc.phase_reached = "ARRIVE_OUT"
        self.recorder.note("到出库站 (仍计在库)")
        if r.apply(EventKind.OUTBOUND, gid, self._next_seq()) \
                is not ReplayOutcome.APPLIED:
            oc.failures.append("到出库站未提交")
        if self.world.in_warehouse_count() < 1:
            oc.failures.append("到出库站后在库计数掉了")

        oc.phase_reached = "UNLOAD_PICK"
        self.recorder.note("M02 卸货接管")
        if r.apply(EventKind.UNBIND, gid, self._next_seq(), pid) is not ReplayOutcome.APPLIED:
            oc.failures.append("卸货接管未提交")
        if self.world.pallet(pid)[1] != PALLET_EMPTY:
            oc.failures.append("卸货后托盘未回到 EMPTY")

        oc.phase_reached = "PLACE_FEED"
        self.recorder.note("M02 出料放置")
        if r.apply(EventKind.PLACE_FEED, gid, self._next_seq()) is not ReplayOutcome.APPLIED:
            oc.failures.append("出料放置未提交")

        oc.phase_reached = "OFFSITE"
        self.recorder.note("货物离场 (计数 −1)")
        if r.apply(EventKind.EXIT, gid, self._next_seq()) is not ReplayOutcome.APPLIED:
            oc.failures.append("离场未提交")
        return oc

    # ------------------------------------------------------------ 错码支线
    def _reject_branch(self, oc: RoundOutcome) -> None:
        """错码支线: 预期结果是**拒绝**, 不要求继续搬运 (设计 621 行)。"""
        gid, pid = oc.goods_id, oc.pallet_id
        before = (self.world.goods_location(gid),
                  self.world.active_binding(gid),
                  self.world.in_warehouse_count())
        v = self.guard.verify_handoff(
            expected_goods=gid, expected_pallet=pid,
            expected_slot=SLOTS[0],
            scanned_goods=_scan(gid, f"r{oc.round_no}g"),
            scanned_pallet=_scan("PAL-OTHER", f"r{oc.round_no}p"),
            scanned_slot=_scan(SLOTS[0], f"r{oc.round_no}s"))
        after = (self.world.goods_location(gid),
                 self.world.active_binding(gid),
                 self.world.in_warehouse_count())
        oc.reject_branch_ok = (not v) and before == after
        oc.reject_reason = (v.reason.value if v.reason else "UNEXPECTED_PASS")
        self.recorder.note(f"错码支线: 拒绝={not v} 原因={oc.reject_reason}")
        if not oc.reject_branch_ok:
            oc.failures.append(f"错码支线未正确拒绝 ({oc.reject_reason})")

    # ------------------------------------------------------------ 主循环
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def run(self) -> list[RoundOutcome]:
        self._seq = 0
        for i in range(1, self.rounds + 1):
            self.recorder.start_round(i)
            oc = self._supply(i)
            self._advance()
            self.recorder.snapshot(self._t, "SUPPLIED")
            self._main_flow(oc)
            self._advance()
            self.recorder.snapshot(self._t, "OFFSITE")
            self._reject_branch(oc)
            oc.in_warehouse_after = self.world.in_warehouse_count()
            oc.active_bindings_after = int(self.world._conn.execute(
                "SELECT COUNT(*) FROM bindings WHERE active=1").fetchone()[0])
            oc.ok = (not oc.failures and oc.in_warehouse_after == 0
                     and oc.active_bindings_after == 0 and oc.reject_branch_ok)
            self.recorder.finish_round(ok=oc.ok,
                                       reject_branch_ok=oc.reject_branch_ok)
            self.outcomes.append(oc)
            self._advance()
            oc.boundary_after = [f"t={self._t:.1f}"]
        return self.outcomes

    # ------------------------------------------------------------ 报告
    def report(self) -> dict:
        cons = self.recorder.consistency_report()
        return {
            "rounds": self.rounds,
            "ok": sum(1 for o in self.outcomes if o.ok),
            "reject_branch_ok": sum(1 for o in self.outcomes if o.reject_branch_ok),
            "in_warehouse_final": self.world.in_warehouse_count(),
            "active_bindings_final": int(self.world._conn.execute(
                "SELECT COUNT(*) FROM bindings WHERE active=1").fetchone()[0]),
            "consistency": cons,
            "boundary_events": len(self.boundary_events),
            "failures": [f for o in self.outcomes for f in o.failures],
        }


def _scan(entity: str, frame: str) -> ScanResult:
    return ScanResult(frame_id=frame, frame_sha256=f"sha-{frame}",
                      token=f"T-{entity}", entity_id=entity, failure=None,
                      px_per_module=8.0)