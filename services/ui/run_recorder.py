"""运行记录与回放数据 —— T12「画面与库存一致」的基础。

设计要求:
|T12 完整演示|同一冻结版本连续20轮|每轮 M01装托、R01/F01上架下架、M02卸货
     |及货物离场；错码支线正确拒绝，画面与库存一致|

「画面与库存一致」意味着 UI 展示的状态必须**来自**库存账本, 而不是
另算一份。所以本模块只做两件事:
1. 从 `World` 读取权威状态, 生成逐帧快照;
2. 记录事件流, 供回放时重放 —— 且必须能**比对**「快照 vs 账本」,
   不一致即报警。

不做的事: 不缓存业务状态。缓存必然与账本漂移。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from services.world.ontology import World


@dataclass
class Frame:
    """某一仿真时刻的权威快照。"""

    sim_time_s: float
    round_no: int
    phase: str
    goods: dict[str, str]          # goods_id -> 位置
    lifecycles: dict[str, str]     # goods_id -> 生命周期
    pallets: dict[str, str]        # pallet_id -> 承载态
    in_warehouse: int
    active_bindings: int
    reservations: dict[str, str] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"sim_time_s": round(self.sim_time_s, 4), "round_no": self.round_no,
                "phase": self.phase, "goods": dict(self.goods),
                "lifecycles": dict(self.lifecycles),
                "pallets": dict(self.pallets),
                "in_warehouse": self.in_warehouse,
                "active_bindings": self.active_bindings,
                "reservations": dict(self.reservations),
                "anomalies": list(self.anomalies)}


@dataclass
class RoundResult:
    """一轮完整闭环的结果 (T12 每轮一条)。"""

    round_no: int
    ok: bool
    phase: str = ""
    frames: list[Frame] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    reject_branch_ok: bool = False        # 错码支线是否正确拒绝

    def as_dict(self) -> dict[str, Any]:
        return {"round_no": self.round_no, "ok": self.ok, "phase": self.phase,
                "frames": [f.as_dict() for f in self.frames],
                "events": list(self.events), "failures": list(self.failures),
                "reject_branch_ok": self.reject_branch_ok}


class RunRecorder:
    """逐帧记录 + 一致性自检。

    `consistency_report()` 是「画面与库存一致」的判定实现: 对每一帧,
    重算几个必须成立的恒等式, 而不是相信记录下来的数字。
    """

    def __init__(self, world: World):
        self.world = world
        self.frames: list[Frame] = []
        self.rounds: list[RoundResult] = []
        self._round: Optional[RoundResult] = None

    # ------------------------------------------------------------ 记录
    def start_round(self, round_no: int) -> RoundResult:
        r = RoundResult(round_no=round_no, ok=False)
        self.rounds.append(r)
        self._round = r
        return r

    def snapshot(self, sim_time_s: float, phase: str) -> Frame:
        w = self.world
        goods, cycles = {}, {}
        for row in w._conn.execute("SELECT goods_id, location, lifecycle FROM goods"):
            goods[row[0]] = row[1]
            cycles[row[0]] = row[2]
        pallets = {row[0]: row[1] for row in
                   w._conn.execute("SELECT pallet_id, cargo_state FROM pallets")}
        binds = w._conn.execute(
            "SELECT COUNT(*) FROM bindings WHERE active=1").fetchone()[0]
        f = Frame(sim_time_s=sim_time_s,
                  round_no=self._round.round_no if self._round else 0,
                  phase=phase, goods=goods, lifecycles=cycles,
                  pallets=pallets, in_warehouse=w.in_warehouse_count(),
                  active_bindings=binds)
        self.frames.append(f)
        if self._round is not None:
            self._round.frames.append(f)
        return f

    def note(self, message: str) -> None:
        if self._round is not None:
            self._round.events.append(message)

    def fail(self, message: str) -> None:
        if self._round is not None:
            self._round.failures.append(message)

    def finish_round(self, ok: bool, reject_branch_ok: bool = False) -> RoundResult:
        assert self._round is not None, "没有进行中的轮次"
        self._round.ok = ok
        self._round.reject_branch_ok = reject_branch_ok
        return self._round

    # ------------------------------------------------------------ 自检
    def consistency_report(self) -> dict[str, Any]:
        """逐帧校验恒等式 —— 「画面与库存一致」的可执行判据。"""
        anomalies: list[str] = []
        for f in self.frames:
            # 1. 在库计数必须等于 lifecycle 落在 IN_WAREHOUSE/OUTBOUND_PENDING 的件数
            expect = sum(1 for v in f.lifecycles.values()
                         if v in ("IN_WAREHOUSE", "OUTBOUND_PENDING"))
            if expect != f.in_warehouse:
                anomalies.append(
                    f"t={f.sim_time_s:.2f} 计数不符: 展示 {f.in_warehouse} "
                    f"vs 生命周期 {expect}")
            # 2. LOADED 托盘必须有活动绑定
            loaded = {k for k, v in f.pallets.items() if v == "LOADED"}
            bound = sum(1 for loc in f.goods.values()
                        if loc.startswith("PALLET:"))
            if len(loaded) != bound:
                anomalies.append(
                    f"t={f.sim_time_s:.2f} LOADED 托盘 {len(loaded)} "
                    f"vs 绑定货物 {bound}")
            # 3. IN_WAREHOUSE 货物的位置必须是托盘
            for gid, lc in f.lifecycles.items():
                if lc == "IN_WAREHOUSE" and not f.goods[gid].startswith("PALLET:"):
                    anomalies.append(
                        f"t={f.sim_time_s:.2f} {gid} 在库但位置 {f.goods[gid]}")
        return {"frames": len(self.frames), "anomalies": anomalies,
                "consistent": not anomalies}

    def export(self, path: str) -> str:
        payload = {"rounds": [r.as_dict() for r in self.rounds],
                   "frames": [f.as_dict() for f in self.frames],
                   "consistency": self.consistency_report()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path