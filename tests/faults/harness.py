"""故障注入框架 —— T09/T10 的判定基础。

设计要求 (S04/S05/S07/S09, T09/T10):
|T09 网络与服务故障|取托中 落托后 提交前注入故障各 5 次|
     |无重复动作和记账; 可重新观测恢复或进入 UNKNOWN|
|T10 模拟急停与失电|低位和第三层分别注入停止、驱动失电与制动失效|
     |模拟响应与停止距离有轨迹; 正常制动保持, 制动失效被检测并记为失败|

框架三条原则:
1. **注入点明确** —— 故障只能打在声明的边界上 (取托中/落托后/提交前/
   释放/急停), 不是任意时刻随机打断;
2. **不掩盖失败** —— 注入后必须记录响应与轨迹, 制动失效必须被**记为
   失败**, 不允许"优雅降级"成成功 (设计 S07: 无虚假成功);
3. **可复现** —— 固定 seed + 记录的注入参数, 同一失败可重放。

时钟是**单一仿真时间域** (设计 R11: 同一仿真时间域内心跳过期与停止轨迹),
所以注入时刻用 sim_time_s 表达, 不另起墙钟。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class FaultKind(str, Enum):
    # --- T09 网络与服务 ---
    SERVICE_RESTART = "service_restart"     # S04 业务服务重启 (物理继续)
    COMMIT_INTERRUPT = "commit_interrupt"   # S05 提交前中断
    HEARTBEAT_EXPIRE = "heartbeat_expire"   # 心跳过期 (R11)
    OBSERVATION_STALE = "observation_stale" # 定位/观测过期
    # --- T10 急停与失电 ---
    ESTOP = "estop"                         # 正常急停 → 制动保持
    DRIVE_POWER_LOSS = "drive_power_loss"   # 驱动失电
    BRAKE_FAIL = "brake_fail"               # 制动失效 (负例, 必须记为失败)


class InjectPoint(str, Enum):
    PICK_IN_PROGRESS = "pick_in_progress"   # 取托中
    DROP_DONE = "drop_done"                 # 落托后
    BEFORE_COMMIT = "before_commit"         # 提交前
    GRIP = "grip"                           # S09 夹持
    LIFT = "lift"                           # S09 抬离
    RELEASE = "release"                     # S09 释放
    AT_LOW_LEVEL = "at_low_level"           # T10 低位
    AT_THIRD_LEVEL = "at_third_level"       # T10 第三层


@dataclass
class FaultSpec:
    """一次注入的完整声明。"""

    kind: FaultKind
    point: InjectPoint
    sim_time_s: float = 0.0
    level: str = ""             # "low" / "third" (T10)
    seed: int = 0
    note: str = ""


@dataclass
class TrajectoryPoint:
    t_s: float
    x_m: float
    z_m: float
    v_mps: float
    loaded: bool


@dataclass
class FaultRun:
    """一次注入试验的完整记录 (可复现、可断言)。"""

    spec: FaultSpec
    detected: bool = False
    detection_reason: str = ""
    counted_as_failure: bool = False
    recovered: bool = False
    recovery_kind: str = ""            # "reobserved" / "unknown"
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    stop_distance_m: float = 0.0
    slip_m: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kind": self.spec.kind.value,
            "point": self.spec.point.value,
            "sim_time_s": self.spec.sim_time_s,
            "level": self.spec.level,
            "seed": self.spec.seed,
            "detected": self.detected,
            "detection_reason": self.detection_reason,
            "counted_as_failure": self.counted_as_failure,
            "recovered": self.recovered,
            "recovery_kind": self.recovery_kind,
            "stop_distance_m": self.stop_distance_m,
            "slip_m": self.slip_m,
            "notes": list(self.notes),
            "trajectory_points": len(self.trajectory),
        }


@dataclass
class FaultInjector:
    """按声明点触发故障, 并记录响应。

    injector 不直接改物理 —— 它只**告诉**被测对象"此处该失效了",
    由被测对象决定怎么响应。这样判定逻辑在被测侧, 可独立测试。
    """

    specs: list[FaultSpec] = field(default_factory=list)
    runs: list[FaultRun] = field(default_factory=list)

    def schedule(self, spec: FaultSpec) -> None:
        self.specs.append(spec)

    def due(self, sim_time_s: float, point: Optional[InjectPoint] = None
            ) -> list[FaultSpec]:
        """返回此刻应当触发的注入 (时间到 且 边界匹配)。"""
        out = []
        for s in self.specs:
            if any(r.spec is s for r in self.runs):
                continue
            if s.sim_time_s > sim_time_s:
                continue
            if point is not None and s.point != point:
                continue
            out.append(s)
        return out

    def fire(self, spec: FaultSpec) -> FaultRun:
        run = FaultRun(spec=spec)
        self.runs.append(run)
        return run

    def summary(self) -> dict:
        return {
            "total": len(self.runs),
            "detected": sum(1 for r in self.runs if r.detected),
            "counted_as_failure": sum(1 for r in self.runs if r.counted_as_failure),
            "recovered": sum(1 for r in self.runs if r.recovered),
        }