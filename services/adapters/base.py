"""W05 设备适配器 — 把物理仿真单元包装成统一的任务接口。

设计约束 (第 8 章接口 / 25.4):
* 设备只能通过**具名动作**被调度, 不暴露底层关节/actuator;
* 动作须有超时与失败语义 (FAILED / UNKNOWN 不推进下一步);
* 适配器负责把物理观测翻译成业务可读的状态 (STOW_CLEAR、载荷存在…);
* 规划与执行分离: 适配器只执行已给定的受限轨迹, 不自己决定去哪。

物理单元 (`sim/manipulator_cell.py`、`sim/forklift_cell.py`) 是"白模单设备",
本模块把两个单元 + 场景实例统一成 Agent 可调用的设备。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np


class ActionResult(str, Enum):
    OK = "OK"
    FAILED = "FAILED"           # 已确认失败, 可重试
    UNKNOWN = "UNKNOWN"         # 状态不明, 须人工/证据恢复 (设计 25.5)


class StowState(str, Enum):
    """收拢状态 (设计 25.6: STOW_CLEAR 是交接的前置证据)。"""

    STOW_CLEAR = "STOW_CLEAR"
    STOW_BLOCKED = "STOW_BLOCKED"     # 仍有臂体伸出 / 夹爪未收 / 速度高
    UNKNOWN = "UNKNOWN"


@dataclass
class ActionSpec:
    """具名动作的契约: 超时 + 稳定判据 + 观测函数。"""

    name: str
    timeout_s: float
    settle_s: float = 0.5


@dataclass
class DeviceAdapter:
    """设备适配器基类。

    子类实现 `_act_<name>` 执行具体动作, 并实现 `observe` 提供业务观测。
    基类统一处理: 超时计时、稳定窗口、失败语义、事件去重。
    """

    robot_id: str
    capabilities: frozenset[str]
    clock: Callable[[], float] = lambda: 0.0
    _done_events: set[str] = field(default_factory=set)

    # ---------- 能力 ----------
    def can(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise ValueError(f"{self.robot_id} 不具备能力 {capability}")

    # ---------- 事件去重 (设计 10.3: (run_id, robot_id, command_id) 唯一) ----------
    def already_done(self, command_id: str) -> bool:
        """同一命令重复到达时不得重复执行 (防重复抓放/记账)。"""
        return command_id in self._done_events

    def mark_done(self, command_id: str) -> None:
        self._done_events.add(command_id)

    # ---------- 子类实现 ----------
    def execute(self, action: str, command_id: str, **kwargs) -> ActionResult:
        raise NotImplementedError

    def observe(self) -> dict:
        raise NotImplementedError

    def stow_state(self) -> StowState:
        """收拢证据 (设计 25.6)。默认不满足, 须子类给出真实判据。"""
        return StowState.UNKNOWN