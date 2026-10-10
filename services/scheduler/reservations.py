"""W05 资源预约 — 设计文档 19.3/25.6。

- 原子取得全部所需资源, 不能部分占锁后无限等待
- 租约: 每 0.1s sim_time 续租, 期限 1s; 过期 → RECOVERY_LOCKED, 不代表物理空闲
- 源区带载退出完成前锁定; 目标区空叉退出完成前锁定
- 持有者是具体 robot_id, 任务相同也不共用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ResourceState(str, Enum):
    FREE = "FREE"
    HELD = "HELD"
    RECOVERY_LOCKED = "RECOVERY_LOCKED"   # 租约过期/观测不明, 停止新进入


class ReservationError(Exception):
    pass


class PartialAcquisition(ReservationError):
    """无法全部取得 → 不持有任何新增锁。"""


@dataclass
class Reservation:
    resource_id: str
    holder: str            # robot_id
    task_id: str
    expires_at_s: float


@dataclass
class ReservationBoard:
    """资源预约板。全部操作以 sim_time_s 为时钟。"""
    lease_s: float = 1.000
    renew_s: float = 0.100
    _held: dict[str, Reservation] = field(default_factory=dict)
    _locked: set[str] = field(default_factory=set)
    #: 退出通道 → 占用它会阻塞的区域 (设计 19.3: 「源区带载退出完成前锁定;
    #: 目标区空叉退出完成前锁定」)。资源按 ID 各自独立判断, 不显式声明
    #: 这层关联的话, 持有 I01 退出锁**不会**阻止别人申请 H01 ——
    #: 那正是「两车同时挤进交接站」要防的事。
    _exit_gates: dict[str, frozenset[str]] = field(default_factory=dict)

    def register_exit_gate(self, exit_resource: str, blocks: list[str]) -> None:
        """声明某退出通道被占用时必须保持空闲的区域。"""
        self._exit_gates[exit_resource] = frozenset(blocks)

    # ---------- 取得 ----------
    def acquire_all(self, resource_ids: list[str], holder: str, task_id: str,
                    sim_time_s: float) -> list[str]:
        """原子取得: 任一不可得则全部不占。返回取得的资源。"""
        want = list(dict.fromkeys(resource_ids))
        want_set = set(want)
        for exit_r, blocked in self._exit_gates.items():
            if not (want_set & blocked):
                continue
            if exit_r in self._locked:
                raise PartialAcquisition(f"{exit_r} RECOVERY_LOCKED")
            h = self._held.get(exit_r)
            if h is not None and h.holder != holder:
                raise PartialAcquisition(
                    f"{exit_r} 被 {h.holder} 占用 (退出未完成), "
                    f"{sorted(want_set & blocked)} 不得进入")
        for rid in want:
            if rid in self._locked:
                raise PartialAcquisition(f"{rid} RECOVERY_LOCKED")
            held = self._held.get(rid)
            if held is not None:
                if held.holder != holder:
                    raise PartialAcquisition(
                        f"{rid} 被 {held.holder} 持有 (task={held.task_id})")
                if held.task_id != task_id:
                    raise PartialAcquisition(
                        f"{rid} 持有者同车但任务不同: {held.task_id} != {task_id}")
        # 全部可取得
        exp = sim_time_s + self.lease_s
        acquired = []
        for rid in want:
            prev = self._held.get(rid)
            if prev is not None and prev.holder == holder and prev.task_id == task_id:
                prev.expires_at_s = exp       # 续租
            else:
                self._held[rid] = Reservation(rid, holder, task_id, exp)
            acquired.append(rid)
        return acquired

    # ---------- 释放 ----------
    def release(self, resource_id: str, holder: str) -> None:
        held = self._held.get(resource_id)
        if held is None or held.holder != holder:
            raise ReservationError(f"{resource_id} 不由 {holder} 持有")
        del self._held[resource_id]

    def release_all(self, holder: str) -> list[str]:
        mine = [r for r, h in self._held.items() if h.holder == holder]
        for r in mine:
            del self._held[r]
        return mine

    # ---------- 租约 ----------
    def renew(self, resource_ids: list[str], holder: str, sim_time_s: float) -> None:
        for rid in resource_ids:
            held = self._held.get(rid)
            if held is None or held.holder != holder:
                continue
            held.expires_at_s = sim_time_s + self.lease_s

    def tick(self, sim_time_s: float) -> list[str]:
        """过期检查: 过期 → RECOVERY_LOCKED (不删除, 需观测确认空闲才 unlock)。"""
        expired = [rid for rid, h in self._held.items() if sim_time_s > h.expires_at_s]
        for rid in expired:
            del self._held[rid]
            self._locked.add(rid)
        return expired

    def unlock_after_observation(self, resource_id: str, observed_free: bool) -> bool:
        """只有可靠观测证明空闲才能重新分配 (19.3)。"""
        if not observed_free:
            return False
        self._locked.discard(resource_id)
        return True

    # ---------- 查询 ----------
    def holder_of(self, resource_id: str) -> Optional[str]:
        h = self._held.get(resource_id)
        return h.holder if h else None

    def is_locked(self, resource_id: str) -> bool:
        return resource_id in self._locked

    def held_by(self, holder: str) -> list[str]:
        return sorted(r for r, h in self._held.items() if h.holder == holder)
