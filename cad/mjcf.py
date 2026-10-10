"""W02 参数化白模 — MJCF 骨架生成。

与 build123d 模型共用同一份 :class:`cad.models.BodyNode` 运动树:
关节轴/限位/驱动上限逐项来自 configs/design.yaml, 碰撞凸包用 box/cylinder
基元近似（非凸托盘分解为多 box, 保留叉孔空隙）。

骨架口径
--------
* 每刚体 <inertial> 质量取配置/派生质量; 惯量为盒/圆柱解析近似的对角占位
  （并行轴对角项, 忽略惯性积 — W03 物理化时按整体验证复核）;
* 世界摆放均为 configs/design.yaml regions 的实例化: 站体 I01/H01/O01、
  入/出料托台、货架 A/B（列心均值, slot_center_y）、M01/M02（instances）、
  R01→park_p01、F01→park_p02（robots 顺序对应）;
* 骨架初始物: pallet 置于 I01 轨面（support_z）, goods 置于托盘支撑面
  （cargo_support_local_z）—— 场景初始化归属 W03+, 此处仅为骨架快照;
* 执行器: 仅配置给出上限的关节（六轴转矩、双指力、举升力、顶升力、
  行走轮转矩占位）。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Sequence

from .models import BodyNode, Prim
from .params import Derived, SimpleNamespace

_MU_TORSION = 0.005
_MU_ROLL = 0.0001

# 关节阻尼与转子惯量（按 design.yaml 各关节的量级选取）。这些不是设计给定值,
# 而是让"限转矩驱动 + 重力补偿"闭环收敛所需的最小物理量：无阻尼时机械臂
# 在补偿后仍以 ~1 rad/s 持续振荡（实测 M01/M02 的 J4）。
_JOINT_DAMPING = {"J1": 8.0, "J2": 12.0, "J3": 10.0,
                  "J4": 4.0, "J5": 3.0, "J6": 2.0}
_JOINT_ARMATURE = {"J1": 0.050, "J2": 0.060, "J3": 0.040,
                   "J4": 0.010, "J5": 0.008, "J6": 0.005}
_SLIDE_DAMPING = {"finger_l": 40.0, "finger_r": 40.0,
                  "forklift_lift": 200.0}
_SLIDE_ARMATURE = {"finger_l": 0.004, "finger_r": 0.004,
                   "forklift_lift": 0.020}

# 行走轮与顶升轮的关节阻尼 —— 必须**单独**给, 不能落到机械臂的默认值。
#
# 车轮稳态时 电机转矩 = 关节阻尼力矩, 故终端车速 = T/(damping·r):
#     damping=1.0 (原默认值) → 0.5/(1.0·0.025) = 12.5 mm/s
# 实测 10.6 mm/s 正是这个终端速度, 而设计要求 speed_max_mps=0.10
# (100 mm/s) —— 差 10 倍。1.0 这个值是为机械臂关节定的, 套到轮子上
# 相当于给车轮加了极大粘性阻力。
#
# 真实轮轴承阻尼很小(电机内部损耗已由转矩上限体现), 这里取 0.002,
# 使阻尼力矩在满速 4.0 rad/s 下仅 0.008 N·m (占 0.5 N·m 的 1.6%)。
_SPIN_DAMPING = 0.002
# 车轮 armature: 电机转子 + 轮体折算到轮轴的惯量。
#
# 不能取极小值: 实测 armature=2e-5 时车体在接触弹簧上上下共振
# (z 在 0.4~1.1mm 间周期起伏), 车速随之出现 1.8s 周期的大幅极限环
# (40mm/s 指令振幅达 51 mm/s)。取 0.01 后 z 抖动降到 0.40mm,
# 振幅降到 0.78 mm/s。
#
# 0.01 kg·m² 远小于轮体自身惯量 (m r²/2 = 0.05·0.025²/2 = 1.6e-5)… 但
# armature 是**附加**到关节上的惯量, 作用是给求解器一个非零的下界,
# 避免轮子质量相对车体过小导致的病态。实测量级有效。
_SPIN_ARMATURE = 0.01


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _geom_attrs(prim: Prim) -> dict:
    if prim.type == "box":
        return {"type": "box", "size": _fmt([s / 2 for s in prim.size_m])}
    if prim.type == "ellipsoid":
        # size 已是半轴 (车轮: 25 × 7.5 × 25 mm)
        return {"type": "ellipsoid", "size": _fmt(list(prim.size_m))}
    attrs = {"type": "cylinder", "size": _fmt([prim.radius_m, prim.height_m / 2])}
    if prim.axis == "y":
        attrs["euler"] = "90 0 0"
    return attrs


def _add_geom(parent: ET.Element, prim: Prim, prefix: str,
              default_friction: float, mask: str = "2 6") -> None:
    attrs = _geom_attrs(prim)
    mu = prim.friction if prim.friction is not None else default_friction
    attrs["friction"] = _fmt([mu, _MU_TORSION, _MU_ROLL])
    attrs["pos"] = _fmt(prim.center_m)
    attrs["contype"], attrs["conaffinity"] = mask.split()
    ET.SubElement(parent, "geom", name=f"{prefix}_{prim.name}", **attrs)


def _prim_inertia_diag(prim: Prim, mass: float) -> tuple[float, float, float]:
    """盒/圆柱/椭球对自身主轴的解析转动惯量（占位近似）。"""
    if prim.type == "box":
        dx, dy, dz = prim.size_m
        return (mass / 12 * (dy * dy + dz * dz),
                mass / 12 * (dx * dx + dz * dz),
                mass / 12 * (dx * dx + dy * dy))
    if prim.type == "ellipsoid":
        # 实心椭球: I_x = m(b²+c²)/5, y 轴为窄轴 (车轮)
        rx, ry, rz = prim.size_m
        return (mass * (ry * ry + rz * rz) / 5.0,
                mass * (rx * rx + rz * rz) / 5.0,
                mass * (rx * rx + ry * ry) / 5.0)
    r, h = prim.radius_m, prim.height_m
    axial = mass * r * r / 2
    trans = mass * (3 * r * r + h * h) / 12
    return (trans, axial, trans) if prim.axis == "y" else (trans, trans, axial)


def _add_inertial(parent: ET.Element, node: BodyNode) -> None:
    """质心=体积加权质心（com_x_m 覆盖 x）, 惯量=并行轴对角占位（忽略惯性积）。"""
    prims = node.prims
    total = node.mass_kg
    vols = [max(p.volume_m3(), 1e-12) for p in prims]
    vol_sum = sum(vols)
    com = [0.0, 0.0, 0.0]
    inertia = [0.0, 0.0, 0.0]
    for prim, vol in zip(prims, vols):
        m = total * vol / vol_sum
        c = prim.center_m
        for k in range(3):
            com[k] += m * c[k] / total
        ixx, iyy, izz = _prim_inertia_diag(prim, m)
        inertia[0] += ixx
        inertia[1] += iyy
        inertia[2] += izz
    # 并行轴定理（对角项）
    for prim, vol in zip(prims, vols):
        m = total * vol / vol_sum
        c = prim.center_m
        ixx, iyy, izz = _prim_inertia_diag(prim, m)
        dx, dy, dz = c[0] - com[0], c[1] - com[1], c[2] - com[2]
        inertia[0] += ixx + m * (dy * dy + dz * dz)
        inertia[1] += iyy + m * (dx * dx + dz * dz)
        inertia[2] += izz + m * (dx * dx + dy * dy)
    if node.com_x_m is not None:
        shift = node.com_x_m - com[0]
        com[0] = node.com_x_m
        inertia[1] += total * shift * shift
        inertia[2] += total * shift * shift
    ET.SubElement(parent, "inertial", pos=_fmt(com), mass=f"{total:.9g}",
                  diaginertia=_fmt(inertia))


def _add_site(parent: ET.Element, name: str, pos: Sequence[float]) -> None:
    ET.SubElement(parent, "site", name=name, pos=_fmt(pos),
                  size="0.004", rgba="1 0 0 0.4")


# 碰撞掩码分组（MuJoCo contype/conaffinity 位）:
#   bit0 = 车辆 (R01/F01 本体), bit1 = 货物/托盘/站台/货架等静态场景,
#   bit2 = 地面。
# 默认位 (2 6) = 货物/静态组互相碰撞, 且能被车辆和地面感知。
# 机械臂连杆用 (0 0) 禁自碰: 关节零位下基座圆柱与肩部圆柱按设计就是
# 套叠的 (实测初始穿透 40mm), 不禁自碰会在 t=0 直接把 QACC 打成 NaN。
# 限力与路径规划由执行层负责, 详见 sim/manipulator_cell.py 的同类处理。
_MASK_DEFAULT = "2 6"
_MASK_ARM_LINK = "0 0"
_MASK_FLOOR = "4 4"


def _add_body(parent: ET.Element, node: BodyNode, name: str, pos: Sequence[float],
              quat: Sequence[float] | None, prefix: str,
              default_friction: float, mask: str = _MASK_DEFAULT) -> ET.Element:
    attrs = {"name": name, "pos": _fmt(pos)}
    if quat is not None:
        attrs["quat"] = _fmt(quat)
    body = ET.SubElement(parent, "body", **attrs)
    joint = node.joint
    if joint is not None:
        if joint.kind == "free":
            ET.SubElement(body, "freejoint", name=f"{prefix}_root")
        else:
            # design.yaml 用 revolute/slide/hinge 语义名; MJCF 只认 hinge/slide/ball
            mjcf_type = {"revolute": "hinge", "hinge": "hinge",
                         "prismatic": "slide", "slide": "slide"}.get(joint.kind, joint.kind)
            jattrs = {"name": f"{prefix}_{joint.name}", "type": mjcf_type, "pos": "0 0 0",
                      "axis": _fmt(joint.axis)}
            if joint.range_rad_or_m is not None:
                jattrs["range"] = _fmt(joint.range_rad_or_m)
            # 关节阻尼/转动惯量: 设计 25.2 给出各关节速度与加速度上限, 意味着
            # 真实驱动带阻尼。缺失时 (实测 damping=0) 机械臂在重力补偿后仍会
            # 持续振荡 (J4 稳态 0.97 rad/s), 无法收敛到保持位。
            if mjcf_type == "hinge":
                # 车轮 (*_spin) 用独立的低阻尼值, 其余走机械臂表/默认值
                spin = joint.name.endswith("_spin")
                damping = _SPIN_DAMPING if spin else _JOINT_DAMPING.get(joint.name, 1.0)
                armature = _SPIN_ARMATURE if spin else _JOINT_ARMATURE.get(joint.name, 0.01)
                jattrs["damping"] = f"{damping:.4g}"
                jattrs["armature"] = f"{armature:.4g}"
            else:
                jattrs["damping"] = f"{_SLIDE_DAMPING.get(joint.name, 1.0):.4g}"
                jattrs["armature"] = f"{_SLIDE_ARMATURE.get(joint.name, 0.002):.4g}"
            ET.SubElement(body, "joint", **jattrs)
    if node.mass_kg is not None:
        _add_inertial(body, node)
    for prim in node.prims:
        _add_geom(body, prim, prefix, default_friction, mask)
    for site_name, site_pos in node.sites.items():
        _add_site(body, f"{prefix}_{site_name}", site_pos)
    for child in node.children:
        child_prefix = f"{prefix}_{child.name}"
        _add_body(body, child, child_prefix, child.pos_m, child.quat_wxyz,
                  child_prefix, default_friction, mask)
    return body


def _collect_instance_actuators(node: BodyNode, prefix: str) -> list[tuple[str, float]]:
    """实例化后的（MJCF 关节名, 驱动上限）— 与 _add_body 的命名规则一致。"""
    out: list[tuple[str, float]] = []
    if node.joint is not None and node.joint.actuated and node.joint.kind != "free":
        out.append((f"{prefix}_{node.joint.name}", node.joint.drive_max))
    for child in node.children:
        out += _collect_instance_actuators(child, f"{prefix}_{child.name}")
    return out


def build_mjcf(p: SimpleNamespace, derived: Derived,
               models: dict[str, BodyNode]) -> str:
    """生成 MJCF 骨架 XML 文本（derived 已折算进 models 树, 仅为签名一致）。"""
    reg = p.regions
    muj = ET.Element("mujoco", model=f"desktop_agv_demo_w02_d{p.design_version}")
    ET.SubElement(muj, "compiler", angle="radian", autolimits="true")
    ET.SubElement(muj, "option", timestep=f"{p.simulation.timestep_s:.9g}")
    default = ET.SubElement(muj, "default")
    ET.SubElement(default, "geom", rgba="0.75 0.75 0.78 1",
                  friction=_fmt([p.simulation.ground_friction_initial, _MU_TORSION, _MU_ROLL]))
    world = ET.SubElement(muj, "worldbody")

    ground_mu = p.simulation.ground_friction_initial
    box_mu = p.simulation.box_pallet_friction_initial
    instance_actuators: list[tuple[str, float]] = []

    # 托盘/货箱接触摩擦按 box_pallet_friction_initial 标注（不改树, 仅渲染时映射）。
    def _place(model_key: str, inst: str, pos: Sequence[float], quat=None,
               mask: str = _MASK_DEFAULT) -> None:
        mu = box_mu if model_key in ("pallet", "goods_box") else ground_mu
        _add_body(world, models[model_key], inst, pos, quat, inst, mu, mask)
        instance_actuators.extend(_collect_instance_actuators(models[model_key], inst))

    _place("table", "table", (0.0, 0.0, 0.0))
    _place("station", "station_i01", (reg.inbound_i01.x_m, reg.inbound_i01.y_m, 0.0))
    _place("station", "station_h01", (reg.handoff_h01.x_m, reg.handoff_h01.y_m, 0.0))
    _place("station", "station_o01", (reg.outbound_o01.x_m, reg.outbound_o01.y_m, 0.0))
    _place("feed_tray", "feed_i_tray", (reg.feed_i_tray.x_m, reg.feed_i_tray.y_m, 0.0))
    _place("feed_tray", "feed_o_tray", (reg.feed_o_tray.x_m, reg.feed_o_tray.y_m, 0.0))
    _place("rack_group", "rack_a",
           (sum(p.rack.column_centers_group_a_x_m) / len(p.rack.column_centers_group_a_x_m),
            p.rack.slot_center_y_m, 0.0))
    _place("rack_group", "rack_b",
           (sum(p.rack.column_centers_group_b_x_m) / len(p.rack.column_centers_group_b_x_m),
            p.rack.slot_center_y_m, 0.0))
    for inst in p.manipulators.instances:
        _place("manipulator", inst.robot_id, tuple(inst.base_position_m),
               mask=_MASK_ARM_LINK)
    _place("flat_agv", "R01", (reg.park_p01.x_m, reg.park_p01.y_m, 0.0),
           _yaw_quat(reg.park_p01.yaw_rad))
    _place("forklift", "F01", (reg.park_p02.x_m, reg.park_p02.y_m, 0.0),
           _yaw_quat(reg.park_p02.yaw_rad))
    _place("pallet", "pallet", (reg.inbound_i01.x_m, reg.inbound_i01.y_m,
                                p.station.support_z_m))
    _place("goods_box", "goods",
           (reg.inbound_i01.x_m, reg.inbound_i01.y_m,
            p.station.support_z_m + p.pallet.cargo_support_local_z_m))

    actuators = ET.SubElement(muj, "actuator")
    for joint_name, drive_max in instance_actuators:
        # MuJoCo 3.15 的 joint= 传动默认取"父链", 对嵌套运动树会把力矩
        # 施加到固定基座上 (实测整场景 t=0 即 DOF 4 NaN)。必须用
        # jointinparent= 才能作用在关节自身的 body 上。
        ET.SubElement(actuators, "motor", name=f"{joint_name}_drv",
                      jointinparent=joint_name, gear="1",
                      ctrlrange=_fmt([0.0, drive_max]),
                      forcerange=_fmt([-drive_max, drive_max]))

    ET.indent(muj, space="  ")
    return "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" + ET.tostring(muj, encoding="unicode")
