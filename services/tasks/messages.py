"""W05 消息契约 — 设计文档 18.2/18.3, schema 2.0。

所有入口必须校验 schema_version/run_id/单位/有限数值/实体引用。
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "2.0"

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class ClockDomain(str, Enum):
    SIMULATION = "simulation"


class Action(str, Enum):
    MOVE_TO = "move_to"
    ALIGN = "align"
    DOCK_LINEAR = "dock_linear"
    INSERT_FORKS = "insert_forks"
    LIFT_TO = "lift_to"
    LOWER_UNTIL_SUPPORTED = "lower_until_supported"
    WITHDRAW_FORKS = "withdraw_forks"
    RETREAT_LOADED = "retreat_loaded"
    PLATFORM_LIFT = "platform_lift"
    ARM_MOVE_JOINTS = "arm_move_joints"
    ARM_MOVE_TCP = "arm_move_tcp"
    GRIPPER_CLOSE = "gripper_close"
    GRIPPER_OPEN = "gripper_open"
    ARM_STOW = "arm_stow"
    STOP = "stop"


CAPABILITIES: dict[str, set[Action]] = {
    "R01": {Action.MOVE_TO, Action.ALIGN, Action.PLATFORM_LIFT, Action.STOP},
    "F01": {Action.MOVE_TO, Action.ALIGN, Action.DOCK_LINEAR, Action.INSERT_FORKS,
            Action.LIFT_TO, Action.LOWER_UNTIL_SUPPORTED, Action.WITHDRAW_FORKS,
            Action.RETREAT_LOADED, Action.STOP},
    # P3 Wave2: F02 与 F01 同规格 (同一 MJCF 模板), 能力集一致。
    "F02": {Action.MOVE_TO, Action.ALIGN, Action.DOCK_LINEAR, Action.INSERT_FORKS,
            Action.LIFT_TO, Action.LOWER_UNTIL_SUPPORTED, Action.WITHDRAW_FORKS,
            Action.RETREAT_LOADED, Action.STOP},
    "M01": {Action.ARM_MOVE_JOINTS, Action.ARM_MOVE_TCP, Action.GRIPPER_CLOSE,
            Action.GRIPPER_OPEN, Action.ARM_STOW, Action.STOP},
    "M02": {Action.ARM_MOVE_JOINTS, Action.ARM_MOVE_TCP, Action.GRIPPER_CLOSE,
            Action.GRIPPER_OPEN, Action.ARM_STOW, Action.STOP},
}


class Pose(BaseModel):
    frame_id: str = "world"
    position_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    orientation_xyzw: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    pose_type: Literal["front_axle_ground", "base_ground", "tcp"]


class MoveToParams(BaseModel):
    target: Pose
    speed_max_mps: Annotated[float, Field(gt=0, le=0.10)]


class DockLinearParams(BaseModel):
    signed_distance_m: float  # 带符号本体X位移
    load_mode: Literal["empty", "loaded"]
    speed_max_mps: Annotated[float, Field(gt=0, le=0.005)]


class InsertForksParams(BaseModel):
    depth_m: Annotated[float, Field(gt=0, le=0.065)]
    speed_max_mps: Annotated[float, Field(gt=0, le=0.01)]
    pallet_id: str


class LiftToParams(BaseModel):
    fork_top_z_m: FiniteFloat
    speed_max_mps: Annotated[float, Field(gt=0, le=0.010)]


class LowerUntilSupportedParams(BaseModel):
    support_z_m: FiniteFloat
    expected_object: Literal["pallet", "goods"]
    speed_max_mps: Annotated[float, Field(gt=0, le=0.010)]


class WithdrawForksParams(BaseModel):
    retreat_distance_m: Annotated[float, Field(gt=0)]
    speed_max_mps: Annotated[float, Field(gt=0, le=0.01)]


class RetreatLoadedParams(BaseModel):
    retreat_distance_m: Annotated[float, Field(gt=0)]
    speed_max_mps: Annotated[float, Field(gt=0, le=0.01)]


class PlatformLiftParams(BaseModel):
    platform_top_z_m: Literal[0.045, 0.065]
    speed_max_mps: Annotated[float, Field(gt=0, le=0.01)]


class ArmMoveJointsParams(BaseModel):
    joint_names: tuple[Literal["J1", "J2", "J3", "J4", "J5", "J6"], ...] = (
        "J1", "J2", "J3", "J4", "J5", "J6")
    joint_positions_rad: tuple[FiniteFloat, ...]
    joint_velocity_max_radps: Annotated[float, Field(gt=0)]
    joint_acceleration_max_radps2: Annotated[float, Field(gt=0)]

    @field_validator("joint_positions_rad")
    @classmethod
    def six_joints(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if len(v) != 6:
            raise ValueError("需 6 个关节值")
        return v


class TcpPose(Pose):
    pose_type: Literal["tcp"] = "tcp"
    tool_id: str


class ArmMoveTcpParams(BaseModel):
    tcp_pose: TcpPose
    path_kind: Literal["linear", "joint"]
    goods_id: Optional[str] = None
    planning_observation_ids: list[str] = Field(default_factory=list)
    tcp_speed_max_mps: Annotated[float, Field(gt=0, le=0.050)]
    tcp_acceleration_max_mps2: Annotated[float, Field(gt=0, le=0.200)]


class GripperCloseParams(BaseModel):
    goods_id: str
    target_gap_m: Annotated[float, Field(ge=0, le=0.075)]
    force_per_finger_max_n: Annotated[float, Field(gt=0, le=8.0)]


class GripperOpenParams(BaseModel):
    target_gap_m: Annotated[float, Field(ge=0, le=0.075)]
    support_id: str


class StopParams(BaseModel):
    reason: str
    level: Literal["controlled", "emergency"]


ActionParams = Union[
    MoveToParams, DockLinearParams, InsertForksParams, LiftToParams,
    LowerUntilSupportedParams, WithdrawForksParams, RetreatLoadedParams,
    PlatformLiftParams, ArmMoveJointsParams, ArmMoveTcpParams,
    GripperCloseParams, GripperOpenParams, StopParams,
]

_PARAMS_BY_ACTION: dict[Action, type[BaseModel]] = {
    Action.MOVE_TO: MoveToParams,
    Action.DOCK_LINEAR: DockLinearParams,
    Action.INSERT_FORKS: InsertForksParams,
    Action.LIFT_TO: LiftToParams,
    Action.LOWER_UNTIL_SUPPORTED: LowerUntilSupportedParams,
    Action.WITHDRAW_FORKS: WithdrawForksParams,
    Action.RETREAT_LOADED: RetreatLoadedParams,
    Action.PLATFORM_LIFT: PlatformLiftParams,
    Action.ARM_MOVE_JOINTS: ArmMoveJointsParams,
    Action.ARM_MOVE_TCP: ArmMoveTcpParams,
    Action.GRIPPER_CLOSE: GripperCloseParams,
    Action.GRIPPER_OPEN: GripperOpenParams,
    Action.STOP: StopParams,
}


class Command(BaseModel):
    """动作命令 (18.2)。停止可无预约; 其他动作必须持有预约。"""
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    clock_domain: ClockDomain = ClockDomain.SIMULATION
    command_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    robot_id: Literal["R01", "F01", "F02", "M01", "M02"]
    action: Action
    params: dict[str, Any]
    timeout_s: Annotated[float, Field(gt=0)]
    expected_state_version: int = Field(ge=0)
    required_reservations: list[str] = Field(default_factory=list)
    parent_command_id: Optional[str] = None

    @field_validator("required_reservations")
    @classmethod
    def stop_needs_no_reservation(cls, v: list[str], info) -> list[str]:
        if info.data.get("action") is not Action.STOP and not v:
            raise ValueError("非 stop 动作必须持有 required_reservations")
        return v

    def typed_params(self) -> BaseModel:
        model = _PARAMS_BY_ACTION[self.action]
        return model.model_validate(self.params)


class CapabilityMismatch(Exception):
    """CAPABILITY_MISMATCH: 机器人不支持该动作。"""
    pass


def check_capability(robot_id: str, action: Action) -> None:
    if action not in CAPABILITIES[robot_id]:
        raise CapabilityMismatch(f"{robot_id} 不支持 {action.value} (CAPABILITY_MISMATCH)")


class CommandResult(BaseModel):
    command_id: str
    robot_id: str
    status: Literal["ACCEPTED", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN"]
    error_code: Optional[str] = None
    last_safe_state: Optional[dict] = None
    result_evidence_ids: list[str] = Field(default_factory=list)
    state_version: int = 0


class Observation(BaseModel):
    """观测 (18.3): 新鲜度按 capture_sim_time_s 计算。"""
    observation_id: str
    source_id: str
    source_kind: Literal["camera", "virtual_encoder", "contact", "virtual_load", "pose_truth"]
    capture_sim_time_s: NonNegFloat
    available_sim_time_s: NonNegFloat
    frame_id: str = "world"
    sample_seq: int = Field(ge=0)
    asset_version: str
    calibration_version: str = "cal-0"
    measurement: dict[str, Any]
    quality_flags: list[str] = Field(default_factory=list)
    image_ref: Optional[str] = None
    image_sha256: Optional[str] = None
    command_id: Optional[str] = None

    @field_validator("available_sim_time_s")
    @classmethod
    def available_not_before_capture(cls, v: float, info) -> float:
        cap = info.data.get("capture_sim_time_s")
        if cap is not None and v < cap:
            raise ValueError("available 不能早于 capture")
        return v
