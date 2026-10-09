"""W02 参数化白模 — 工程参数加载与派生参数。

单一参数源: ``configs/design.yaml``（SI 单位, 全部带单位后缀）。
本模块只做两件事:

1. 加载并按 ``SimpleNamespace`` 提供配置（内部一律米, 无任何 mm 换算）;
2. 集中声明 design.yaml 未覆盖的次要几何尺寸（``derived_parameters``）。
   依据 AGENTS.md 规则 3（未冻结数值皆为设计初值）, 这些派生值要么由已配置
   参数以公式推导, 要么显式记为初值, 全部随 manifest 落盘供审计;
   修订其中任何一项须记录理由并递增 design_version。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULT_CONFIG_RELPATH = "configs/design.yaml"


def load_params(config_path: str | Path | None = None) -> SimpleNamespace:
    """读取 design.yaml 并返回递归 SimpleNamespace（内部一律 SI 米）。"""
    path = Path(config_path) if config_path else _default_config_path()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "design_version" not in raw:
        raise ValueError(f"无效的设计参数文件: {path}")
    return _namespace(raw)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_RELPATH


def _namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_namespace(v) for v in obj]
    return obj


def config_sha256(config_path: str | Path | None = None) -> str:
    """输入参数哈希（manifest 必填, 设计文档 8.1）。"""
    path = Path(config_path) if config_path else _default_config_path()
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Derived:
    """design.yaml 未覆盖、由公式推导或记为初值的次要几何/质量参数。

    ``formula`` 字段说明来源; ``initial``=True 表示纯设计初值（非公式推导）。
    """

    values: dict
    notes: dict


def derive_parameters(p: SimpleNamespace) -> Derived:
    """全部派生参数的唯一定义点（cad/mjcf.py 与 cad/models.py 只读此处）。"""
    fr, fa, st, rk = p.forklift, p.flat_agv, p.station, p.rack
    ft, mp, gt = p.fork, p.manipulators, p.manipulators.gripper
    r = fr.wheel_radius_m

    forklift = {
        # 底盘: 轮半径→底盘下缘=1r(轮轴平面), 上缘=3r(轮径)。
        "chassis_z_range_m": [r, 3 * r],
        "chassis_x_range_m": list(fr.body_bounds_x_m),
        # 轮宽 = 0.6r（公式推导）。
        "wheel_width_m": 0.6 * r,
        # 门架导轨截面 = 0.8r, 贴前缘 x ∈ [body_x1, body_x1+0.8r], y=±0.3×车宽。
        "mast_rail_section_m": 0.8 * r,
        "mast_rail_x_range_m": [fr.body_bounds_x_m[1], fr.body_bounds_x_m[1] + 0.8 * r],
        "mast_rail_y_m": 0.3 * fr.body_width_m,
        "mast_rail_z_range_m": [3 * r, fr.mast_height_m],
        # 移动叉架挡板: 包住导轨的竖板, z 初值 10→90 mm。
        "carriage_plate_x_range_m": list(fr.body_bounds_x_m[1:2]) + [fr.body_bounds_x_m[1] + 0.8 * r],
        "carriage_plate_z_range_m": [0.010, 0.090],
        # 货叉: 从挡板前缘前伸 1 根叉长, 叉顶面 = transport_fork_top_z_m（举升零位）。
        "fork_x_range_m": [fr.body_bounds_x_m[1] + 0.8 * r,
                           fr.body_bounds_x_m[1] + 0.8 * r + ft.size_m[0]],
        "fork_top_z_m": fr.transport_fork_top_z_m,
        # 质量拆分（初值, 合计=配置值）: 整车固定质量含 4 轮与 2 导轨。
        "wheel_mass_kg": 0.050,
        "mast_rail_mass_kg": 0.050,
        "fork_mass_kg": 0.015,
        "chassis_mass_kg": round(fr.fixed_mass_kg - 4 * 0.050 - 2 * 0.050, 6),
        "carriage_plate_mass_kg": round(fr.carriage_mass_kg - 2 * 0.015, 6),
        # 行走电机转矩上限（初值, 配置未覆盖; 上限按 J1 转矩 3.0 N·m 的 1/6 记）。
        "wheel_drive_torque_max_nm": 0.500,
    }
    flat = {
        # 底盘: 下缘=轮轴平面 1r, 上缘=固定车体高度上限。
        "chassis_z_range_m": [fa.wheel_radius_m, fa.fixed_body_height_max_m],
        # 轮位: x=±(半长−2r), y=±(半宽−r)（轮外缘与车身边缘齐平, 公式推导）。
        "wheel_x_offset_m": fa.size_xy_m[0] / 2 - 2 * fa.wheel_radius_m,
        "wheel_y_offset_m": fa.size_xy_m[1] / 2 - fa.wheel_radius_m,
        "wheel_width_m": 0.010,
        # 顶升台厚 = 低位台面顶 − 固定车体高上限（公式推导）。
        "platform_thickness_m": fa.platform_z_m[0] - fa.fixed_body_height_max_m,
        # 质量初值（配置未覆盖; 顶升台与平台力 15 N 对应的机构估计）。
        "chassis_mass_kg": 0.800,
        "platform_mass_kg": 0.100,
        "wheel_mass_kg": 0.030,
        "wheel_drive_torque_max_nm": 0.300,
    }
    station = {
        # 支撑轨长 = 托盘长 + 2×20 mm 出入余量（初值）。
        "rail_length_m": p.pallet.body_size_m[0] + 0.040,
        # 轨 z: rail_base_z_m → support_z_m（公式推导, 轨面=55 mm）。
        "rail_z_range_m": [st.rail_base_z_m, st.support_z_m],
        # 支腿: 10 mm 方腿, 内侧面保证 entry_clear_width_min_m。
        "leg_size_m": 0.010,
        "leg_x_offset_m": (p.pallet.body_size_m[0] + 0.040) / 2 - 0.005,
        "leg_y_offset_m": st.entry_clear_width_min_m / 2 + 0.005,
        "leg_z_range_m": [0.0, st.rail_base_z_m],
    }
    rack = {
        "post_section_m": 0.020,   # 立柱截面初值
        "shelf_thickness_m": 0.008,  # 层板厚初值
        # 列距由配置货位列心差推导（0.130 m, 与整体 390 mm 自洽）。
        "column_pitch_m": rk.column_centers_group_a_x_m[1] - rk.column_centers_group_a_x_m[0],
        "post_z_range_m": [0.0, rk.overall_size_m[2]],
    }
    tray = {
        "plate_size_m": list(p.pallet.total_size_m[:2]),  # 台面按托盘总包络取
        "plate_thickness_m": 0.008,
        "leg_size_m": 0.010,
        # 台面顶 = 配置 feed_*_tray.top_z_m（55 mm, 与站体支撑面一致）。
        "top_z_m": p.regions.feed_i_tray.top_z_m,
    }
    arm = {
        # 连杆截面/关节壳初值（配置未覆盖的视觉与碰撞近似）。
        "j1_column_r_m": 0.030,
        "shoulder_r_m": 0.025,
        "upper_arm_section_m": 0.040,
        "forearm_section_m": 0.030,
        "wrist_r_m": [0.020, 0.015],
        # 质量拆分: upper_arm/forearm/wrist_gripper(0.20) + 关节壳初值,
        # J5+J6+tool+双指 = wrist_gripper 0.20 kg。
        "mass_split_kg": {
            "J1": 0.100, "J2": 0.100,
            "J3": mp.link_mass_kg.upper_arm, "J4": mp.link_mass_kg.forearm,
            "J5": 0.050, "J6": 0.050, "tool": 0.080, "finger_each": 0.010,
        },
    }
    table = {"thickness_m": 0.018}  # 台面板厚初值
    gripper = _gripper_geometry(p)

    values = {
        "table": table,
        "forklift": forklift,
        "flat_agv": flat,
        "station": station,
        "rack": rack,
        "feed_tray": tray,
        "manipulator": arm,
        "gripper": gripper.values,
    }
    notes = {
        "convention": "design.yaml 未覆盖的尺寸: formula=由配置参数推导; initial=设计初值(AGENTS.md 规则3)。",
        "gripper_axis_mapping": gripper.notes,
    }
    return Derived(values=values, notes=notes)


def _gripper_geometry(p: SimpleNamespace) -> Derived:
    """夹爪几何的工具系映射（工具 +X 沿腕→TCP, +Y 闭合方向, +Z=X×Y）。

    design.yaml 的 palm/finger/pad 尺寸按设计文档 25.3 的口诀顺序存储
    （掌 70×20×15 = Z宽×Y厚×X厚; 指 厚6×宽10×延45 = Y×Z×X）。
    映射依据: 掌体中心在 TCP 上方 45 mm 且掌底高于箱顶 12.5 mm
    ⇒ 掌沿工具 X 厚度只能取 15 mm。
    """
    gt = p.manipulators.gripper
    palm_z, palm_y, palm_x = gt.palm_size_m          # 0.070, 0.020, 0.015
    finger_ty, finger_tz, finger_lx = gt.finger_size_m  # 0.006, 0.010, 0.045
    pad_z, pad_x = gt.pad_size_m                     # 0.010, 0.015
    pad_thickness_m = 0.002  # 初值: 指垫嵌入指体内侧、与指内侧面齐平的垫厚
    values = {
        # 掌: 中心在 −palm_center_above_tcp_z_m(工具X), 厚X=palm_x, 厚Y=palm_y, 宽Z=palm_z。
        "palm_center_tool_m": [-gt.palm_center_above_tcp_z_m, 0.0, 0.0],
        "palm_size_tool_m": [palm_x, palm_y, palm_z],
        # 指: 长=finger_lx(沿X, 从掌底向 TCP 延伸), 厚=finger_ty(Y), 宽=finger_tz(Z);
        # 零间隙指心 y=±finger_ty/2（内侧面恰好贴合, gap=指内侧面距离）。
        "finger_size_tool_m": [finger_lx, finger_ty, finger_tz],
        "finger_center_x_tool_m": -gt.palm_center_above_tcp_z_m + palm_x / 2 + finger_lx / 2,
        "finger_center_y_tool_m": finger_ty / 2,
        # 指 x 范围（工具系）: 掌底 −0.0375 → +0.0075（越过 TCP 7.5 mm）。
        # 指垫: X 高 pad_x 居中于 TCP, Z 宽 pad_z, 厚 pad_thickness_m 嵌入指内侧齐平。
        "pad_size_tool_m": [pad_x, pad_thickness_m, pad_z],
        "pad_center_finger_m": [0.0, -(finger_ty - pad_thickness_m) / 2, 0.0],
    }
    notes = {
        "palm": "palm_size_m=[70,20,15]mm → 工具系(Z宽,Y厚,X厚); 掌底=TCP上方37.5mm=箱顶+12.5mm。",
        "finger": "finger_size_m=[6,10,45]mm → 工具系(Y厚,Z宽,X长); 指心±3mm@gap=0, 内侧面距=gap。",
        "pad": "pad_size_m=[10,15]mm → 工具系(Z宽,X高); 垫厚2mm嵌入指内侧齐平, 接触面即指内侧面。",
    }
    return Derived(values=values, notes=notes)
