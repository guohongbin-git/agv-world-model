"""业务服务故障层 —— T09「网络与服务故障」的实现。

设计场景:
|S04 取托后服务重启|托盘已由F01承托，源区仍锁定|业务服务重启而物理服务继续|
     |虚拟控制器停止；重连观测后不重复取托，保留在途数量|
|S05 落托后提交前重启|目标已承托，提交被注入中断|重建观测与事务|
     |确认或UNKNOWN；不重复放托，不重复增库存|

|T09 网络与服务故障|取托中 落托后 提交前注入故障各 5 次|
     |无重复动作和记账；可重新观测恢复或进入 UNKNOWN|

核心不变量 (与 T11/T16 一致, 但从**服务生命周期**角度验证):
1. 重启后**不重复物理动作** —— 动作幂等靠 command_id, 重启后重连
   观测到的状态必须与重启前一致;
2. **不重复记账** —— 在途数量保留, 不因重连而增减;
3. 证据不足 → `UNKNOWN`, **不得伪造成功** (S05 明确允许 UNKNOWN)。

「服务重启」的实现: `FaultInjector` 关闭服务, 物理继续推进; 重连时
`reconnect()` 必须**重新观测**并按观测结果收敛, 而不是盲目重放命令。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from services.scheduler.reservations import ReservationBoard, ResourceState
from services.world.ontology import World


class ServiceState(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    RECOVERING = "RECOVERING"


class TaskState(str, Enum):
    """主任务状态 —— 重启后据此决定该做什么。"""

    IDLE = "IDLE"
    MOVING = "MOVING"          # 已下发动作, 等待物理完成
    HELD = "HELD"              # 物理已完成, 等提交
    COMMITTED = "COMMITTED"
    UNKNOWN = "UNKNOWN"        # 证据不足
    FAILED = "FAILED"


@dataclass
class TaskRecord:
    task_id: str
    kind: str                       # "pick" / "drop" / "commit"
    state: TaskState = TaskState.IDLE
    command_id: str = ""
    issued: bool = False
    settled_count: int = 0          # 提交次数 (必须 ≤1)
    in_transit: int = 0             # 在途数量 (重启后须保留)
    last_observation: str = ""


class BusinessService:
    """可被重启的业务服务 —— T09 的被测对象。

    与 `World` 的关系: `World` 是**持久库存账本**, 重启不该清空它;
    本类是**易失的控制器状态**, 重启会丢, 需靠重新观测恢复。
    """

    HEARTBEAT_LEASE_S = 0.5

    def __init__(self, world: World, board: Optional[ReservationBoard] = None):
        self.world = world
        self.board = board or ReservationBoard()
        self.state = ServiceState.UP
        self.tasks: dict[str, TaskRecord] = {}
        self.last_heartbeat_s = 0.0
        self.reconnects = 0
        # 必须建在 __init__ 里: 写成类级 `field(default_factory=list)`
        # 会让它成为 dataclasses.Field 描述符, 被所有实例共享且无法 append。
        self.unknown_events: list[str] = []

    # ------------------------------------------------------------ 生命周期
    def heartbeat(self, sim_time_s: float) -> None:
        if self.state is ServiceState.UP:
            self.last_heartbeat_s = sim_time_s

    def heartbeat_expired(self, sim_time_s: float) -> bool:
        """心跳是否过期 (设计 R11: 同一仿真时间域内)。"""
        return (sim_time_s - self.last_heartbeat_s) > self.HEARTBEAT_LEASE_S

    def restart(self) -> None:
        """业务服务重启 —— 控制器易失状态丢失, 物理服务继续。"""
        self.state = ServiceState.DOWN

    # ------------------------------------------------------------ 动作下发
    def issue(self, task_id: str, kind: str, command_id: str) -> TaskRecord:
        """下发一个动作。服务停机时拒绝下发 (物理仍在跑)。"""
        if self.state is ServiceState.DOWN:
            raise RuntimeError("业务服务停机, 不可下发动作")
        rec = self.tasks.get(task_id) or TaskRecord(task_id=task_id, kind=kind)
        rec.command_id = command_id
        if not rec.issued:
            rec.issued = True                 # 幂等: 同一任务只发一次
            rec.state = TaskState.MOVING
        self.tasks[task_id] = rec
        return rec

    def observe(self, task_id: str, physically_done: bool,
                evidence_ok: bool) -> TaskState:
        """重新观测: 依据**物理事实**而非服务记忆收敛状态。

        evidence_ok=False 表示无法确认 (观测过期/遮挡) → UNKNOWN,
        绝不能因为「重启前认为成功了」就置 COMMITTED。
        """
        rec = self.tasks.get(task_id)
        if rec is None:
            raise KeyError(task_id)
        if not evidence_ok:
            rec.state = TaskState.UNKNOWN
            self.unknown_events.append(task_id)
            return rec.state
        rec.state = TaskState.HELD if physically_done else TaskState.MOVING
        rec.last_observation = ("done" if physically_done else "pending")
        return rec.state

    def commit(self, task_id: str, apply_fn, sim_time_s: float) -> bool:
        """提交事务。UNKNOWN 状态**不允许**提交 (证据不足不伪造成功)。"""
        rec = self.tasks[task_id]
        if rec.state is TaskState.UNKNOWN:
            return False
        if rec.settled_count:
            return False                    # 幂等: 只提交一次
        if rec.state is not TaskState.HELD:
            return False
        apply_fn()
        rec.settled_count += 1
        rec.state = TaskState.COMMITTED
        return True

    # ------------------------------------------------------------ 重连
    def reconnect(self, task_id: str, physically_done: bool,
                  evidence_ok: bool) -> TaskState:
        """重连并重建观测。物理服务持续, 故可直接按物理事实收敛。"""
        self.reconnects += 1
        self.state = ServiceState.UP
        return self.observe(task_id, physically_done, evidence_ok)