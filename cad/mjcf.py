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


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _geom_attrs(prim: Prim) -> dict:
    if prim.type == "box":
        return {"type": "box", "size": _fmt([s / 2 for s in prim.size_m])}
    attrs = {"type": "cylinder", "size": _fmt([prim.radius_m, prim.height_m / 2])}
    if prim.axis == "y":
        attrs["euler"] = "90 0 0"
    return attrs


def _add_geom(parent: ET.Element, prim: Prim, prefix: str,
              default_friction: float) -> None:
    attrs = _geom_attrs(prim)
    mu = prim.friction if prim.friction is not None else default_friction
    attrs["friction"] = _fmt([mu, _MU_TORSION, _MU_ROLL])
    attrs["pos"] = _fmt(prim.center_m)
    ET.SubElement(parent, "geom", name=f"{prefix}_{prim.name}", **attrs)


def _prim_inertia_diag(prim: Prim, mass: float) -> tuple[float, float, float]:
    """盒/圆柱对自身主轴的解析转动惯量（占位近似）。"""
    if prim.type == "box":
        dx, dy, dz = prim.size_m
        return (mass / 12 * (dy * dy + dz * dz),
                mass / 12 * (dx * dx + dz * dz),
                mass / 12 * (dx * dx + dy * dy))
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


def _add_body(parent: ET.Element, node: BodyNode, name: str, pos: Sequence[float],
              quat: Sequence[float] | None, prefix: str,
              default_friction: float) -> ET.Element:
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
            ET.SubElement(body, "joint", **jattrs)
    if node.mass_kg is not None:
        _add_inertial(body, node)
    for prim in node.prims:
        _add_geom(body, prim, prefix, default_friction)
    for site_name, site_pos in node.sites.items():
        _add_site(body, f"{prefix}_{site_name}", site_pos)
    for child in node.children:
        child_prefix = f"{prefix}_{child.name}"
        _add_body(body, child, child_prefix, child.pos_m, child.quat_wxyz,
                  child_prefix, default_friction)
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
    def _place(model_key: str, inst: str, pos: Sequence[float], quat=None) -> None:
        mu = box_mu if model_key in ("pallet", "goods_box") else ground_mu
        _add_body(world, models[model_key], inst, pos, quat, inst, mu)
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
        _place("manipulator", inst.robot_id, tuple(inst.base_position_m))
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
        ET.SubElement(actuators, "motor", name=f"{joint_name}_drv",
                      joint=joint_name, gear="1",
                      forcerange=_fmt([-drive_max, drive_max]))

    ET.indent(muj, space="  ")
    return "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" + ET.tostring(muj, encoding="unicode")
