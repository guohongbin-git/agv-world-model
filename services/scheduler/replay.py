"""W05 事件回放与故障注入 — T08 / T11 / T15 / T16。

设计判据:
|T08 错误拦截|错产品 错托盘 错货位各 20 次|60 次均不放行, 库存不错误提交|
|T11 库存回放|重复 乱序 缺失事件及快照恢复|重复幂等; 乱序可检测; 缺失不得伪造成功|
|T15 工位互斥|M01/M02活动、收拢未达、收拢旧帧、R01未退出各10次|
          |进入请求全部拒绝; 合法STOW_CLEAR与版本一致后原子放行|
|T16 装卸事务恢复|入料夹持后、装托松爪后、卸货夹持后、出料松爪后、离场提交中断各5次,
          |另注入重复/乱序事件|无重复抓放、计数或绑定; 允许证据不足进入UNKNOWN,
          |恢复正例应在证据恢复后自动完成|

核心不变量 (任何故障注入下都必须成立):
1. **不得重复抓放** —— 同一 command_id / 事务键只生效一次;
2. **计数不错** —— 在库计数只由生命周期迁移驱动, 不由事件重放驱动;
3. **绑定不错** —— 活动绑定至多一条, 版本单调;
4. **证据不足 → UNKNOWN**, 不得伪造成功;
5. **失败不提交** —— 事务原子, 失败后状态与失败前完全一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from services.world.ontology import (ConstraintViolation, Loc, LocKind,
                                     World)


class EventKind(str, Enum):
    BIND_STORE = "CargoBindingConfirmed"
    UNBIND = "CargoUnbound"
    OUTBOUND = "OutboundPending"
    EXIT = "GoodsExitConfirmed"
    FEED_PICK = "FeedPicked"
    PLACE_FEED = "PlacedAtOutputFeed"


class ReplayOutcome(str, Enum):
    APPLIED = "APPLIED"
    DUPLICATE = "DUPLICATE"       # 重复事件 → 幂等跳过
    OUT_OF_ORDER = "OUT_OF_ORDER"  # 乱序 → 可检测, 不静默应用
    BLOCKED = "BLOCKED"           # 前置状态不满足 → 拒绝
    UNKNOWN = "UNKNOWN"           # 证据不足


@dataclass
class ReplayReport:
    applied: int = 0
    duplicate: int = 0
    out_of_order: int = 0
    blocked: int = 0
    unknown: int = 0
    violations: list[str] = field(default_factory=list)

    def merge(self, other: "ReplayReport") -> "ReplayReport":
        return ReplayReport(
            self.applied + other.applied,
            self.duplicate + other.duplicate,
            self.out_of_order + other.out_of_order,
            self.blocked + other.blocked,
            self.unknown + other.unknown,
            self.violations + other.violations)

    @property
    def total(self) -> int:
        return (self.applied + self.duplicate + self.out_of_order
                + self.blocked + self.unknown)

    def as_dict(self) -> dict:
        return {"applied": self.applied, "duplicate": self.duplicate,
                "out_of_order": self.out_of_order, "blocked": self.blocked,
                "unknown": self.unknown, "total": self.total,
                "violations": list(self.violations)}


class EventReplay:
    """库存事件回放器 —— T11/T16 的判定核心。

    设计要求「重复幂等、乱序可检测、缺失不得伪造成功」。做法:
    * 每个事件带 **事务键** (goods+pallet+kind), 键已处理 → DUPLICATE;
    * 每个事件带 **前置条件**, 前置不满足 → BLOCKED (而不是抛异常或
      静默应用);
    * 前置状态**无法判定**(如 UNKNOWN 证据) → UNKNOWN, 留给后续恢复;
    * 乱序检测靠**因果序号**: 同 run 内单序号单调, 倒退即乱序。
    """

    def __init__(self, world: World):
        self.world = world
        self._seen: set[str] = set()
        self._last_seq = 0

    @staticmethod
    def key(goods_id: str, kind: EventKind, pallet_id: str = "") -> str:
        return f"{goods_id}|{kind.value}|{pallet_id}"

    def apply(self, kind: EventKind, goods_id: str, seq: int,
              pallet_id: str = "", task_id: str = "T", sim_time_s: float = 0.0,
              fault_at: Optional[str] = None) -> ReplayOutcome:
        """回放单个事件。

        fault_at: 注入点 — "before_commit"(提交前中断) / "after_commit"
        (提交后回执丢失)。前者必须**不产生任何状态变化**。
        """
        if seq < self._last_seq:
            return ReplayOutcome.OUT_OF_ORDER      # 乱序: 可检测, 不应用
        k = self.key(goods_id, kind, pallet_id)
        if k in self._seen:
            return ReplayOutcome.DUPLICATE         # 幂等
        w = self.world
        try:
            if kind is EventKind.BIND_STORE:
                pre = w.pallet(pallet_id)[1] == "EMPTY" and w.active_binding(goods_id) is None
            elif kind is EventKind.UNBIND:
                pre = w.active_binding(goods_id) is not None
            elif kind is EventKind.OUTBOUND:
                pre = w.lifecycle_of(goods_id) == "IN_WAREHOUSE"
            elif kind is EventKind.EXIT:
                pre = w.lifecycle_of(goods_id) == "OUTBOUND_PENDING"
            elif kind is EventKind.PLACE_FEED:
                pre = w.lifecycle_of(goods_id) == "OUTBOUND_PENDING"
            elif kind is EventKind.FEED_PICK:
                pre = w.lifecycle_of(goods_id) == "INBOUND"
            else:
                pre = False
        except (TypeError, IndexError):
            return ReplayOutcome.UNKNOWN            # 证据不足, 不伪造
        if not pre:
            return ReplayOutcome.BLOCKED

        if fault_at == "before_commit":
            return ReplayOutcome.UNKNOWN            # 中断: 不提交, 待恢复

        try:
            if kind is EventKind.BIND_STORE:
                w.bind_and_store(goods_id, pallet_id,
                                 Loc(LocKind.PALLET, pallet_id),
                                 task_id, sim_time_s)
            elif kind is EventKind.UNBIND:
                w.unbind_at_output(goods_id, pallet_id, task_id, sim_time_s)
            elif kind is EventKind.OUTBOUND:
                w.mark_outbound_pending(goods_id, task_id, sim_time_s)
            elif kind is EventKind.EXIT:
                w.exit_offsite(goods_id, task_id, sim_time_s)
            elif kind is EventKind.FEED_PICK:
                w.take_from_feed(goods_id, "M01", task_id, sim_time_s)
            elif kind is EventKind.PLACE_FEED:
                w.place_at_feed(goods_id, "O-TRAY", task_id, sim_time_s)
        except ConstraintViolation:
            return ReplayOutcome.BLOCKED

        if fault_at == "after_commit":
            # 回执丢失: 状态已变但调用方以为失败 → 重放时必须靠 key 去重
            self._seen.add(k)
            self._last_seq = seq
            return ReplayOutcome.APPLIED
        self._seen.add(k)
        self._last_seq = seq
        return ReplayOutcome.APPLIED