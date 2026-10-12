"""EvidenceVerdict — 版本化多帧物理证据对象 (GPT 评审 P1-4 / 设计文档 §4)。

设计 §4 H_valid = 位姿容差 ∧ 承托率≥95% ∧ 旧承载卸载≤5% ∧ 稳定 0.5s
∧ 有效租约。每次 commit_transfer 前构造一份 verdict:

- **多帧采样**: 稳定窗内每 0.05s 一帧 (承托率/旧支撑卸载比/位姿/速度/
  时间戳), 全帧过门才算「稳定」—— 单帧读数不再是充分证据;
- **版本化**: ``schema_version`` 随证据结构演进, 旧证据可被新代码解读;
- **序列化持久化**: ``to_json`` 随 commit_transfer 落 ``execution_evidence``
  表, 事件 payload 以 ``evidence_id`` 引用 —— 事后可从任一事件追溯到
  原始采样帧;
- **有效期**: ``valid_until`` = 末帧时刻 + ``VERDICT_TTL_S`` (0.5s 稳定窗),
  过期证据拒绝提交。

本模块保持纯 Python (无 mujoco 依赖): 物理采样由调用方注入 step/probe。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

SCHEMA_VERSION = 1

# ---------- 门禁常量 (设计 §4 H_valid, 待标定初值) ----------
MIN_NEW_SUPPORT = 0.95        # 新承载面承托率 ≥95%
MAX_OLD_SUPPORT = 0.05        # 旧承载卸载 ≤5%
STABLE_WINDOW_S = 0.5         # 稳定观测窗 0.5s
FRAME_INTERVAL_S = 0.05       # 采样帧间隔
VERDICT_TTL_S = 0.5           # verdict 有效期 (自末帧起算)
MAX_SPEED = 5e-3              # 稳定判据: 全域静止 (与 _ramp calm 一致)
POSE_TOL_M = 0.010            # 位姿容差: 稳定窗内位姿极差 ≤10mm

KIND_PICK = "PICK"            # 取托: 新支撑 = 叉臂, 旧支撑 = 横梁/台面
KIND_PLACE = "PLACE"          # 放托: 新支撑 = 横梁/台面, 旧支撑 = 叉臂

GATES = {"min_new_support": MIN_NEW_SUPPORT,
         "max_old_support": MAX_OLD_SUPPORT,
         "stable_window_s": STABLE_WINDOW_S,
         "max_speed": MAX_SPEED,
         "pose_tol_m": POSE_TOL_M,
         "ttl_s": VERDICT_TTL_S}


@dataclass(frozen=True)
class EvidenceFrame:
    """单帧物理观测 (纯数值, 可 JSON 序列化)。"""
    t: float                   # 仿真时间戳
    new_support: float         # 新承载面承托率 (叉臂或横梁/台面)
    old_support: float         # 旧承载面承载比 (卸载判据)
    pallet_x: float            # 托盘位姿
    pallet_y: float
    pallet_z: float
    speed: float               # 全域最大 |qvel| (静止判据)

    def to_dict(self) -> dict:
        return {"t": self.t, "new_support": self.new_support,
                "old_support": self.old_support,
                "pallet_x": self.pallet_x, "pallet_y": self.pallet_y,
                "pallet_z": self.pallet_z, "speed": self.speed}


@dataclass(frozen=True)
class EvidenceVerdict:
    """一次提交前的完整证据判定 (不可变; 失败原因显式枚举)。"""
    verdict_id: str
    schema_version: int
    kind: str                  # KIND_PICK / KIND_PLACE
    task_id: str
    pallet_id: str
    vehicle_id: str
    lease_id: str | None       # 有效租约 (巷道互斥持有令牌)
    frames: tuple[EvidenceFrame, ...]
    valid_until: float
    failures: tuple[str, ...] = ()

    # ---------- 判定 ----------
    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def sampled_at(self) -> float:
        return self.frames[-1].t if self.frames else float("nan")

    @property
    def stable_window_s(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].t - self.frames[0].t

    @property
    def support(self) -> float:
        """末帧新承载面承托率 (叙述/审计用)。"""
        return self.frames[-1].new_support if self.frames else float("nan")

    @property
    def old_support(self) -> float:
        return self.frames[-1].old_support if self.frames else float("nan")

    def summary(self) -> dict:
        return {"evidence_id": self.verdict_id, "kind": self.kind,
                "support": round(self.support, 4),
                "old_support": round(self.old_support, 4),
                "stable_window_s": round(self.stable_window_s, 3),
                "frames": len(self.frames)}

    # ---------- 序列化 ----------
    def to_json(self) -> str:
        return json.dumps({
            "verdict_id": self.verdict_id,
            "schema_version": self.schema_version,
            "kind": self.kind, "task_id": self.task_id,
            "pallet_id": self.pallet_id, "vehicle_id": self.vehicle_id,
            "lease_id": self.lease_id,
            "valid_until": self.valid_until,
            "failures": list(self.failures),
            "gates": dict(GATES),
            "frames": [f.to_dict() for f in self.frames],
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "EvidenceVerdict":
        d = json.loads(s)
        return cls(verdict_id=d["verdict_id"],
                   schema_version=d["schema_version"],
                   kind=d["kind"], task_id=d["task_id"],
                   pallet_id=d["pallet_id"], vehicle_id=d["vehicle_id"],
                   lease_id=d["lease_id"],
                   frames=tuple(EvidenceFrame(**f) for f in d["frames"]),
                   valid_until=d["valid_until"],
                   failures=tuple(d["failures"]))


def evaluate_gates(kind: str, frames: tuple[EvidenceFrame, ...],
                   lease_id: str | None) -> tuple[str, ...]:
    """设计 §4 H_valid 五门联合判定; 返回失败原因 (空 = 通过)。"""
    failures: list[str] = []
    if not frames:
        return ("无采样帧",)
    span = frames[-1].t - frames[0].t
    if span < STABLE_WINDOW_S - 1e-6:
        failures.append(f"稳定窗不足 ({span:.2f}s < {STABLE_WINDOW_S}s)")
    if any(f.new_support < MIN_NEW_SUPPORT for f in frames):
        lo = min(f.new_support for f in frames)
        failures.append(f"{kind} 承托率不足 (最低 {lo:.0%} < {MIN_NEW_SUPPORT:.0%})")
    if any(f.old_support > MAX_OLD_SUPPORT for f in frames):
        hi = max(f.old_support for f in frames)
        failures.append(f"旧支撑未卸载 (最高 {hi:.0%} > {MAX_OLD_SUPPORT:.0%})")
    if any(f.speed > MAX_SPEED for f in frames):
        hi = max(f.speed for f in frames)
        failures.append(f"未静止 (|v|max {hi:.1e} > {MAX_SPEED:.0e})")
    for i, key in enumerate(("pallet_x", "pallet_y", "pallet_z")):
        rng = max(getattr(f, key) for f in frames) - \
            min(getattr(f, key) for f in frames)
        if rng > POSE_TOL_M:
            failures.append(f"位姿超差 ({key} 极差 {rng * 1000:.1f}mm)")
    if not lease_id:
        failures.append("无有效租约")
    return tuple(failures)


def build_verdict(*, verdict_id: str, kind: str, task_id: str, pallet_id: str,
                  vehicle_id: str, lease_id: str | None,
                  frames: tuple[EvidenceFrame, ...]) -> EvidenceVerdict:
    """构造 + 判定 + 定有效期 (唯一入口, 保证 failures 与帧一致)。"""
    return EvidenceVerdict(
        verdict_id=verdict_id, schema_version=SCHEMA_VERSION, kind=kind,
        task_id=task_id, pallet_id=pallet_id, vehicle_id=vehicle_id,
        lease_id=lease_id, frames=tuple(frames),
        valid_until=(frames[-1].t + VERDICT_TTL_S) if frames else 0.0,
        failures=evaluate_gates(kind, tuple(frames), lease_id))


def collect_frames(*, step: Callable[[], None], probe: Callable[[], EvidenceFrame],
                   duration_s: float = STABLE_WINDOW_S,
                   interval_s: float = FRAME_INTERVAL_S) -> list[EvidenceFrame]:
    """连续采样 duration_s: 先取初帧, 之后每 interval_s 步进一帧。"""
    frames = [probe()]
    for _ in range(max(1, int(round(duration_s / interval_s)))):
        step()
        frames.append(probe())
    return frames
