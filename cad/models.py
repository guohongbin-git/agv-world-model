"""W02 参数化白模 — build123d 参数化模型定义。

单位约定
--------
design.yaml 与本模块内部一律 SI 米。mm 只出现在注释; 导出边界的唯一一次
单位转换发生在 cad/build.py: STEP 以米制模型写出毫米制几何(export unit=M),
STL 保持米(MuJoCo/Blender 直接可用)。

结构约定
--------
每个模型返回一棵 :class:`BodyNode` 树: 同一份节点同时给出

* ``shape``      — build123d 视觉/STEP 几何（托盘用布尔差保证叉孔真实贯通）;
* ``prims``      — box/cylinder 碰撞基元（MJCF 凸包近似与非凸分解的唯一来源）;
* ``joint``      — 关节（轴向量、限位、驱动上限与 design.yaml 完全一致）;
* ``sites``      — 关键 site（fork_tip / tcp / 货位中心 / 标签面）。

局部坐标系（MJCF body 帧与之一致, 见 cad/mjcf.py）
--------------------------------------------------
* pallet / goods_box: 底面中心为原点, +Z 上;
* forklift: 前轮轴中点为原点（body_bounds_x_m 的口径）, +X 前;
* flat_agv: 底盘几何中心, +X 前;
* station / rack_group / feed_tray: 落地中心, +Z 上;
* manipulator: 基座底面中心, 零位朝 +X 展开, base_frame +Z 上;
* table: 桌面几何中心, 顶面 z=0（世界即桌面上表面）。

design.yaml 未覆盖的次要尺寸一律来自 cad/params.derive_parameters（审计记录）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from build123d import Box, Compound, Cylinder, Location, Part, Pos, Rot, Sphere

from .params import Derived, SimpleNamespace

Vec3 = tuple[float, float, float]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Prim:
    """碰撞/视觉基元（模型局部系, 米）。"""

    name: str
    type: str                       # "box" | "cylinder" | "ellipsoid"
    center_m: Vec3
    size_m: Vec3 | None = None      # box 全尺寸 (dx,dy,dz)
    radius_m: float | None = None   # cylinder
    height_m: float | None = None   # cylinder 全高
    axis: str = "z"                 # cylinder 轴向: "z" | "y"
    friction: float | None = None   # 覆盖默认摩擦（如夹爪指垫）

    def volume_m3(self) -> float:
        if self.type == "box":
            return self.size_m[0] * self.size_m[1] * self.size_m[2]
        if self.type == "ellipsoid":
            rx, ry, rz = self.size_m
            return 4.0 / 3.0 * 3.141592653589793 * rx * ry * rz
        return 3.141592653589793 * self.radius_m**2 * self.height_m


@dataclass(frozen=True)
class JointSpec:
    """关节定义（轴向量/限位与 design.yaml 同源）。"""

    name: str
    kind: str                                    # "free" | "hinge" | "slide"
    axis: Vec3 = (0.0, 0.0, 1.0)
    range_rad_or_m: tuple[float, float] | None = None
    drive_max: float | None = None               # 转矩 N·m 或力 N（占位）
    actuated: bool = False


@dataclass
class BodyNode:
    """刚体节点: 平移先于自身关节旋转（设计文档 25.2）。"""

    name: str
    pos_m: Vec3 = (0.0, 0.0, 0.0)                # 相对父 body 帧
    quat_wxyz: tuple[float, float, float, float] | None = None
    mass_kg: float | None = None                 # None → 静体
    com_x_m: float | None = None                 # 配置的质心 x 覆盖
    joint: JointSpec | None = None
    prims: list[Prim] = field(default_factory=list)
    sites: dict[str, Vec3] = field(default_factory=dict)
    children: list["BodyNode"] = field(default_factory=list)
    shape: Any = None                            # build123d 视觉几何
    note: str = ""

    def iter_nodes(self):
        yield self
        for child in self.children:
            yield from child.iter_nodes()


# --------------------------------------------------------------------------- #
# 构建辅助
# --------------------------------------------------------------------------- #
def _box_prim(name: str, size: Sequence[float], center: Sequence[float], **kw) -> Prim:
    return Prim(name=name, type="box", center_m=tuple(center), size_m=tuple(size), **kw)


def _cyl_prim(name: str, radius: float, height: float, center: Sequence[float],
              axis: str = "z", **kw) -> Prim:
    return Prim(name=name, type="cylinder", center_m=tuple(center),
                radius_m=radius, height_m=height, axis=axis, **kw)


def _wheel_prim(name: str, radius: float, width: float,
                center: Sequence[float]) -> Prim:
    """车轮碰撞体 —— 真实圆柱。

    曾短暂换成 ellipsoid, 依据是「圆柱轮缘与地面相切时接触点落在铰轴上,
    会被运动学锁死」。该结论**错误**, 源于一个有缺陷的最小复现(电机漏设
    forcerange, ctrl 被钳到 0, 于是任何 geom 都报 0.0000 rad/s)。
    修正 harness 后实测: 真实圆柱轮在场景中能正常滚动。

    真实车轮本就是圆柱, 保持 cylinder。
    """
    return Prim(name=name, type="cylinder", center_m=tuple(center),
                radius_m=radius, height_m=width, axis="y")


def _shape_from_prims(prims: Sequence[Prim], label: str) -> Compound:
    solids = []
    for prim in prims:
        if prim.type == "box":
            solid = Pos(*prim.center_m) * Box(*prim.size_m)
        elif prim.type == "ellipsoid":
            # build123d 无椭球基元, 用**沿 y 压扁的球**近似。
            # STEP 只用于导出与目视, 碰撞用 MJCF 的 ellipsoid, 两者外形
            # 同为「扁圆盘」, 包络一致 (半径 r、宽 w)。
            rx, ry, rz = prim.size_m
            solid = Pos(*prim.center_m) * (Rot(90, 0, 0) * Sphere(rx)) \
                if abs(ry - rx) > 1e-9 else Pos(*prim.center_m) * Sphere(rx)
        else:
            cyl = Cylinder(prim.radius_m, prim.height_m)
            solid = (Rot(90, 0, 0) * cyl) if prim.axis == "y" else (Pos(*prim.center_m) * cyl)
            if prim.axis == "y":
                solid = Pos(*prim.center_m) * solid
        solid.label = prim.name
        solids.append(solid)
    comp = Compound(children=solids)
    comp.label = label
    return comp


def _bands(lo: float, hi: float, holes: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """[lo,hi] 去掉孔洞区间后的剩余条带（用于托盘纵向支撑条分解）。"""
    bands, cur = [], lo
    for h0, h1 in sorted(holes):
        if h0 > cur + 1e-12:
            bands.append((cur, h0))
        cur = max(cur, h1)
    if hi > cur + 1e-12:
        bands.append((cur, hi))
    return bands


def _rect_frame_prims(prefix: str, outer: Sequence[float], inner: Sequence[float],
                      z0: float, z1: float) -> list[Prim]:
    """矩形边框（外 outer=X/Y 全尺寸, 内 inner=净区尺寸）分解为 4 个不重叠 box。"""
    ox, oy = outer
    ix, iy = inner
    cz, dz = (z0 + z1) / 2, z1 - z0
    half = {"n": (0.0, (oy / 2 + iy / 2) / 2), "s": (0.0, -(oy / 2 + iy / 2) / 2),
            "e": ((ox / 2 + ix / 2) / 2, 0.0), "w": (-(ox / 2 + ix / 2) / 2, 0.0)}
    sizes = {"n": (ox, (oy - iy) / 2, dz), "s": (ox, (oy - iy) / 2, dz),
             "e": ((ox - ix) / 2, iy, dz), "w": ((ox - ix) / 2, iy, dz)}
    return [_box_prim(f"{prefix}_{k}", sizes[k], (v[0], v[1], cz)) for k, v in half.items()]


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# --------------------------------------------------------------------------- #
# 1. pallet — 托盘（布尔差保证叉孔真实贯通）
# --------------------------------------------------------------------------- #
def build_pallet(p: SimpleNamespace, d: Derived) -> BodyNode:
    """原点=底面中心。底板+顶板+3 纵向支撑条+顶面定位凸边; 2 个 X 向贯通叉孔。"""
    pal = p.pallet
    bx, by, bz = pal.body_size_m
    z0, z1 = pal.channel_z_m
    lh = pal.locator_height_m
    cw, ix, iy = pal.channel_width_m, *pal.locator_inner_size_m

    ribs = [_box_prim(f"pallet_rib_{i}", (bx, hi - lo, z1 - z0),
                      (0.0, (lo + hi) / 2, (z0 + z1) / 2))
            for i, (lo, hi) in enumerate(
                _bands(-by / 2, by / 2,
                       [(cy - cw / 2, cy + cw / 2) for cy in pal.channel_centers_y_m]))]
    prims = (
        [_box_prim("pallet_bottom_plate", (bx, by, z0), (0.0, 0.0, z0 / 2)),
         _box_prim("pallet_top_plate", (bx, by, bz - z1), (0.0, 0.0, (z1 + bz) / 2))]
        + ribs
        + _rect_frame_prims("pallet_locator", (bx, by), (ix, iy), bz, bz + lh)
    )

    # 视觉几何: 实体 − 叉孔(布尔差, X 向超出贯通) + 凸边框。
    body = Pos(0, 0, bz / 2) * Box(bx, by, bz)
    channels = [Pos(0, cy, (z0 + z1) / 2) * Box(1.5 * bx, cw, pal.channel_height_m)
                for cy in pal.channel_centers_y_m]
    outer = Pos(0, 0, bz + lh / 2) * Box(bx, by, lh)
    inner = Pos(0, 0, bz + lh / 2) * Box(ix, iy, lh)
    shape = body
    for chan in channels:
        shape = shape - chan
    shape = shape + (outer - inner)
    shape.label = "pallet"

    return BodyNode(
        name="pallet", mass_kg=pal.tare_mass_kg, joint=JointSpec(name="pallet_root", kind="free"),
        prims=prims, shape=shape,
        sites={"pallet_label": tuple(p.labels.pallet_label_center_m)},
        note="叉孔 y=±15, z 4→12, X 贯通; 凸边内净区 70×50, 总高 21mm。",
    )


# --------------------------------------------------------------------------- #
# 2. goods_box — 标准箱
# --------------------------------------------------------------------------- #
def build_goods_box(p: SimpleNamespace, d: Derived) -> BodyNode:
    """原点=底面中心。60×40×50 标准箱, +Y 面贴产品码。"""
    gx, gy, gz = p.goods.size_m
    prim = _box_prim("goods_body", (gx, gy, gz), (0.0, 0.0, gz / 2))
    return BodyNode(
        name="goods_box", mass_kg=p.goods.mass_kg,
        joint=JointSpec(name="goods_root", kind="free"), prims=[prim],
        shape=_shape_from_prims([prim], "goods_box"),
        sites={"goods_label": (0.0, gy / 2, gz / 2)},
        note="X 两侧供夹持, Y 侧贴 25mm 产品码。",
    )


# --------------------------------------------------------------------------- #
# 3. forklift — 叉车（前轮轴原点）
# --------------------------------------------------------------------------- #
def build_forklift(p: SimpleNamespace, d: Derived) -> BodyNode:
    """原点=前轮轴中点。底盘+门架双导轨(静)+4 轮(铰)+移动叉架(滑)+双货叉。"""
    fr, ft, dv = p.forklift, p.fork, d.values["forklift"]
    r = fr.wheel_radius_m
    x0, x1 = fr.body_bounds_x_m

    chassis_prims = [
        _box_prim("forklift_chassis", (x1 - x0, fr.body_width_m, dv["chassis_z_range_m"][1] - dv["chassis_z_range_m"][0]),
                  ((x0 + x1) / 2, 0.0, (dv["chassis_z_range_m"][0] + dv["chassis_z_range_m"][1]) / 2)),
        _box_prim("forklift_mast_rail_l",
                  (dv["mast_rail_section_m"], dv["mast_rail_section_m"],
                   dv["mast_rail_z_range_m"][1] - dv["mast_rail_z_range_m"][0]),
                  ((dv["mast_rail_x_range_m"][0] + dv["mast_rail_x_range_m"][1]) / 2,
                   -dv["mast_rail_y_m"],
                   (dv["mast_rail_z_range_m"][0] + dv["mast_rail_z_range_m"][1]) / 2)),
        _box_prim("forklift_mast_rail_r",
                  (dv["mast_rail_section_m"], dv["mast_rail_section_m"],
                   dv["mast_rail_z_range_m"][1] - dv["mast_rail_z_range_m"][0]),
                  ((dv["mast_rail_x_range_m"][0] + dv["mast_rail_x_range_m"][1]) / 2,
                   dv["mast_rail_y_m"],
                   (dv["mast_rail_z_range_m"][0] + dv["mast_rail_z_range_m"][1]) / 2)),
    ]
    chassis = BodyNode(
        name="forklift", mass_kg=fr.fixed_mass_kg, com_x_m=fr.fixed_com_x_m,
        joint=JointSpec(name="forklift_root", kind="free"),
        prims=chassis_prims, shape=_shape_from_prims(chassis_prims, "forklift_chassis"),
        note="固定车体质量含 4 轮与 2 导轨（拆分见派生参数）。",
    )

    def _wheel(name: str, x: float, y: float) -> BodyNode:
        prim = _wheel_prim(f"{name}", r, dv["wheel_width_m"], (0.0, 0.0, 0.0))
        return BodyNode(name=name, pos_m=(x, y, r), mass_kg=dv["wheel_mass_kg"],
                        joint=JointSpec(name=f"{name}_spin", kind="hinge",
                                        axis=(0.0, 1.0, 0.0),
                                        drive_max=dv["wheel_drive_torque_max_nm"], actuated=True),
                        prims=[prim], shape=_shape_from_prims([prim], f"{name}"))

    drive_y = [pt[1] for pt in fr.drive_contact_xy_m]
    rear_y = [pt[1] for pt in fr.rear_contact_xy_m]
    rear_x = fr.rear_contact_xy_m[0][0]
    for i, y in enumerate(drive_y):
        chassis.children.append(_wheel(f"wheel_drive_{i}", 0.0, y))
    for i, y in enumerate(rear_y):
        chassis.children.append(_wheel(f"wheel_rear_{i}", rear_x, y))

    plate = _box_prim("forklift_carriage_plate",
                      (dv["carriage_plate_x_range_m"][1] - dv["carriage_plate_x_range_m"][0],
                       fr.body_width_m,
                       dv["carriage_plate_z_range_m"][1] - dv["carriage_plate_z_range_m"][0]),
                      ((dv["carriage_plate_x_range_m"][0] + dv["carriage_plate_x_range_m"][1]) / 2,
                       0.0,
                       (dv["carriage_plate_z_range_m"][0] + dv["carriage_plate_z_range_m"][1]) / 2))
    fx = (dv["fork_x_range_m"][0] + dv["fork_x_range_m"][1]) / 2
    fz = dv["fork_top_z_m"] - ft.size_m[2] / 2
    forks = [_box_prim(f"forklift_fork_{s}", ft.size_m, (fx, cy, fz))
             for s, cy in zip(("l", "r"), ft.centers_y_m)]
    carriage_prims = [plate] + forks
    tip_x = dv["fork_x_range_m"][1]
    carriage = BodyNode(
        name="forklift_carriage", mass_kg=fr.carriage_mass_kg, com_x_m=fr.carriage_com_x_m,
        joint=JointSpec(name="forklift_lift", kind="slide", axis=(0.0, 0.0, 1.0),
                        range_rad_or_m=(0.0, fr.lift_travel_m), drive_max=fr.lift_force_max_n,
                        actuated=True),
        prims=carriage_prims, shape=_shape_from_prims(carriage_prims, "forklift_carriage"),
        sites={"fork_tip": (tip_x, 0.0, dv["fork_top_z_m"])},
        note="举升零位时叉顶面=transport_fork_top_z_m=65mm; fork_tip 在叉尖中心。",
    )
    chassis.children.append(carriage)
    return chassis


# --------------------------------------------------------------------------- #
# 4. flat_agv — 平板车
# --------------------------------------------------------------------------- #
def build_flat_agv(p: SimpleNamespace, d: Derived) -> BodyNode:
    """原点=底盘几何中心。160×110 底盘(≤35 高)+顶升台 70×40。"""
    fa, dv = p.flat_agv, d.values["flat_agv"]
    sx, sy = fa.size_xy_m
    zlo, zhi = dv["chassis_z_range_m"]
    chassis_prim = _box_prim("flat_agv_chassis", (sx, sy, zhi - zlo), (0.0, 0.0, (zlo + zhi) / 2))
    chassis = BodyNode(
        name="flat_agv", mass_kg=dv["chassis_mass_kg"],
        joint=JointSpec(name="flat_agv_root", kind="free"),
        prims=[chassis_prim], shape=_shape_from_prims([chassis_prim], "flat_agv_chassis"),
        note="固定车体高度 ≤35mm（顶升台除外）。",
    )
    for iy, sy_sign in ((0, -1.0), (1, 1.0)):
        for ix, sx_sign in ((0, -1.0), (1, 1.0)):
            prim = _wheel_prim(f"flat_agv_wheel_{iy}{ix}", fa.wheel_radius_m,
                               dv["wheel_width_m"], (0.0, 0.0, 0.0))
            chassis.children.append(BodyNode(
                name=f"flat_agv_wheel_{iy}{ix}",
                pos_m=(sx_sign * dv["wheel_x_offset_m"], sy_sign * dv["wheel_y_offset_m"],
                       fa.wheel_radius_m),
                mass_kg=dv["wheel_mass_kg"],
                joint=JointSpec(name=f"flat_agv_wheel_{iy}{ix}_spin", kind="hinge",
                                axis=(0.0, 1.0, 0.0),
                                drive_max=dv["wheel_drive_torque_max_nm"], actuated=True),
                prims=[prim], shape=_shape_from_prims([prim], f"flat_agv_wheel_{iy}{ix}")))
    pz = dv["platform_thickness_m"]
    plat_prim = _box_prim("flat_agv_platform", (*fa.platform_size_m, pz),
                          (0.0, 0.0, p.flat_agv.fixed_body_height_max_m + pz / 2))
    chassis.children.append(BodyNode(
        name="flat_agv_platform", mass_kg=dv["platform_mass_kg"],
        joint=JointSpec(name="flat_agv_lift", kind="slide", axis=(0.0, 0.0, 1.0),
                        range_rad_or_m=(0.0, fa.platform_travel_m),
                        drive_max=fa.platform_force_max_n, actuated=True),
        prims=[plat_prim], shape=_shape_from_prims([plat_prim], "flat_agv_platform"),
        note="低位台面顶=45mm, 高位=65mm。",
    ))
    return chassis


# --------------------------------------------------------------------------- #
# 5. station — 交接站（I01/H01/O01 复用）
# --------------------------------------------------------------------------- #
def build_station(p: SimpleNamespace, d: Derived) -> BodyNode:
    """原点=落地中心。两侧支撑轨(y=±27)+4 支腿(净宽≥140); 轨间为顶升窗口。"""
    st, dv, pal = p.station, d.values["station"], p.pallet
    rl, rz = dv["rail_length_m"], dv["rail_z_range_m"]
    prims: list[Prim] = []
    for s in ("l", "r"):
        y = st.rail_support_y_m[0] if s == "l" else st.rail_support_y_m[1]
        prims.append(_box_prim(f"station_rail_{s}",
                               (rl, st.rail_support_width_m, rz[1] - rz[0]),
                               (0.0, y, (rz[0] + rz[1]) / 2)))
    leg = dv["leg_size_m"]
    for sx, sx_sign in (("w", -1.0), ("e", 1.0)):
        for sy, sy_sign in (("s", -1.0), ("n", 1.0)):
            prims.append(_box_prim(f"station_leg_{sx}{sy}", (leg, leg, dv["leg_z_range_m"][1]),
                                   (sx_sign * dv["leg_x_offset_m"],
                                    sy_sign * dv["leg_y_offset_m"],
                                    dv["leg_z_range_m"][1] / 2)))
    node = BodyNode(
        name="station", prims=prims, shape=_shape_from_prims(prims, "station"),
        sites={"pallet_center": (0.0, 0.0, st.support_z_m)},
        note=f"轨面 {st.support_z_m*1000:.0f}mm; 托盘 {pal.body_size_m[0]*1000:.0f}×{pal.body_size_m[1]*1000:.0f} 承托于 y=±27mm; "
             f"中间顶升窗口=轨间净空, 支腿净宽≥{st.entry_clear_width_min_m*1000:.0f}mm。",
    )
    return node


# --------------------------------------------------------------------------- #
# 6. rack_group — 货架组（3 列 × 3 层）
# --------------------------------------------------------------------------- #
def build_rack_group(p: SimpleNamespace, d: Derived) -> BodyNode:
    """原点=落地中心。4 立柱+3 层板; 层板顶=支撑面 z=100/250/400; 9 货位 site。"""
    rk, dv = p.rack, d.values["rack"]
    ox, oy, oz = rk.overall_size_m
    sec, shelf_t = dv["post_section_m"], dv["shelf_thickness_m"]
    prims: list[Prim] = []
    for sx in ("w", "e"):
        for sy in ("s", "n"):
            prims.append(_box_prim(f"rack_post_{sx}{sy}", (sec, sec, oz),
                                   ((1.0 if sx == "e" else -1.0) * (ox - sec) / 2,
                                    (1.0 if sy == "n" else -1.0) * (oy - sec) / 2, oz / 2)))
    for li, sz in enumerate(rk.support_z_m):
        prims.append(_box_prim(f"rack_shelf_l{li}", (ox, oy, shelf_t), (0.0, 0.0, sz - shelf_t / 2)))
    sites = {f"slot_c{ci}_l{li}": (dv["column_pitch_m"] * (ci - (rk.columns - 1) / 2), 0.0, sz)
             for ci in range(rk.columns) for li, sz in enumerate(rk.support_z_m)}
    return BodyNode(
        name="rack_group", prims=prims, shape=_shape_from_prims(prims, "rack_group"),
        sites=sites,
        note=f"{rk.columns}列×{rk.levels}层; 层板顶=支撑面; slot site=货位中心。",
    )


# --------------------------------------------------------------------------- #
# 7. feed_tray — 入/出料托台
# --------------------------------------------------------------------------- #
def build_feed_tray(p: SimpleNamespace, d: Derived) -> BodyNode:
    """原点=落地中心。台面顶 z=55mm（=站体支撑面）。"""
    dv = d.values["feed_tray"]
    px, py = dv["plate_size_m"]
    pt, leg = dv["plate_thickness_m"], dv["leg_size_m"]
    top = dv["top_z_m"]
    prims = [_box_prim("feed_tray_plate", (px, py, pt), (0.0, 0.0, top - pt / 2))]
    for sx in ("w", "e"):
        for sy in ("s", "n"):
            prims.append(_box_prim(f"feed_tray_leg_{sx}{sy}", (leg, leg, top - pt),
                                   ((1.0 if sx == "e" else -1.0) * (px / 2 - leg / 2),
                                    (1.0 if sy == "n" else -1.0) * (py / 2 - leg / 2),
                                    (top - pt) / 2)))
    return BodyNode(
        name="feed_tray", prims=prims, shape=_shape_from_prims(prims, "feed_tray"),
        sites={"tray_center": (0.0, 0.0, top)},
        note=f"台面顶 {top*1000:.0f}mm; 箱件底面承托面。",
    )


# --------------------------------------------------------------------------- #
# 8. manipulator — 六轴链 + 平行夹爪
# --------------------------------------------------------------------------- #
def build_manipulator(p: SimpleNamespace, d: Derived) -> BodyNode:
    """原点=基座底面中心, 零位朝 +X。关节定义逐项取自 manipulators.joints。

    每段连杆几何放在**子关节 body** 内（平移先于旋转）, 零位装配与
    MJCF/IK 模型共用一份运动链。TCP=两指夹持中心, 工具 +X 沿腕→TCP。
    """
    mp, gt, dv = p.manipulators, p.manipulators.gripper, d.values
    arm_d, gr_d = dv["manipulator"], dv["gripper"]
    masses = arm_d["mass_split_kg"]

    base_prim = _cyl_prim("manipulator_base", mp.base_radius_m, mp.joints[0].offset_m[2],
                          (0.0, 0.0, mp.joints[0].offset_m[2] / 2))
    root = BodyNode(
        name="manipulator", prims=[base_prim],
        shape=_shape_from_prims([base_prim], "manipulator_base"),
        note="基座固定（静体）; 链: base→J1..J6→tool→双指。",
    )

    link_prims = {
        "J1": [_cyl_prim("manipulator_j1_column", arm_d["j1_column_r_m"], 0.060, (0.0, 0.0, 0.030))],
        "J2": [_cyl_prim("manipulator_shoulder", arm_d["shoulder_r_m"], 0.080, (0.0, 0.0, 0.0), axis="y")],
        "J3": [_box_prim("manipulator_upper_arm",
                         (mp.joints[2].offset_m[0], arm_d["upper_arm_section_m"], arm_d["upper_arm_section_m"]),
                         (-mp.joints[2].offset_m[0] / 2, 0.0, 0.0))],
        "J4": [_box_prim("manipulator_forearm",
                         (mp.joints[3].offset_m[0], arm_d["forearm_section_m"], arm_d["forearm_section_m"]),
                         (-mp.joints[3].offset_m[0] / 2, 0.0, 0.0))],
        "J5": [_cyl_prim("manipulator_wrist1", arm_d["wrist_r_m"][0], 0.050,
                         (-mp.joints[4].offset_m[0] / 2, 0.0, 0.0), axis="y")],
        "J6": [_cyl_prim("manipulator_wrist2", arm_d["wrist_r_m"][1], 0.030,
                         (-mp.joints[5].offset_m[0] / 2, 0.0, 0.0), axis="y")],
    }

    node = root
    for spec in mp.joints:
        child = BodyNode(
            name=spec.name, pos_m=tuple(spec.offset_m), mass_kg=masses[spec.name],
            joint=JointSpec(name=spec.name, kind=spec.kind, axis=tuple(spec.axis),
                            range_rad_or_m=(spec.lower_rad, spec.upper_rad),
                            drive_max=spec.torque_max_nm, actuated=True),
            prims=link_prims[spec.name],
            shape=_shape_from_prims(link_prims[spec.name], f"manipulator_{spec.name}"),
        )
        node.children.append(child)
        node = child

    # tool: 刚连于 J6, 原点即 TCP。掌体中心在 TCP 后 palm_center_above_tcp_z_m(工具 -X)。
    palm = _box_prim("manipulator_palm", gr_d["palm_size_tool_m"], gr_d["palm_center_tool_m"])
    tool = BodyNode(
        name="tool", pos_m=tuple(mp.tcp_offset_m), mass_kg=masses["tool"], prims=[palm],
        shape=_shape_from_prims([palm], "manipulator_palm"),
        sites={"tcp": (0.0, 0.0, 0.0)},
        note="掌底=TCP 上方 37.5mm=箱顶+12.5mm; tcp site=两指夹持中心。",
    )
    for s, sign in (("l", 1.0), ("r", -1.0)):
        finger = _box_prim(f"manipulator_finger_{s}", gr_d["finger_size_tool_m"],
                           (0.0, sign * gr_d["finger_center_y_tool_m"], 0.0))
        pad = _box_prim(f"manipulator_pad_{s}", gr_d["pad_size_tool_m"],
                        (gr_d["pad_center_finger_m"][0],
                         sign * gr_d["pad_center_finger_m"][1],
                         gr_d["pad_center_finger_m"][2]),
                        friction=mp.gripper.pad_friction_initial)
        finger_body = BodyNode(
            name=f"finger_{s}", pos_m=(gr_d["finger_center_x_tool_m"],
                                       sign * gr_d["finger_center_y_tool_m"], 0.0),
            mass_kg=masses["finger_each"],
            joint=JointSpec(name=f"gripper_finger_{s}", kind="slide",
                            axis=(0.0, sign, 0.0),
                            range_rad_or_m=tuple(mp.gripper.gap_range_m),
                            drive_max=mp.gripper.force_per_finger_max_n, actuated=True),
            prims=[finger, pad],
            shape=Compound(children=[*_shape_from_prims([finger], f"manipulator_finger_{s}").children,
                                     *_shape_from_prims([pad], f"manipulator_pad_{s}").children]),
            note=f"零位 gap=0 指内侧贴合; 滑动关节开距 {mp.gripper.gap_range_m[1]*1000:.0f}mm。",
        )
        tool.children.append(finger_body)
    node.children.append(tool)
    return root


# --------------------------------------------------------------------------- #
# 9. table — 桌面
# --------------------------------------------------------------------------- #
def build_table(p: SimpleNamespace, d: Derived) -> BodyNode:
    """桌面: 设计 3 节"桌面左下角为世界坐标原点", 边界 X 0→1800 / Y 0→1200 mm。

    因此桌面几何中心在 (length/2, width/2), 顶面 z=0 即桌面上表面。
    (此前把桌面建在原点, 与设计 §3 的边界定义及所有 region 坐标矛盾 ——
    实测 F01 停在 (1550,900)mm 落到桌面外, 车轮悬空自由落体。)
    """
    t = d.values["table"]["thickness_m"]
    center = (p.table.length_m / 2, p.table.width_m / 2)
    prim = _box_prim("table_top", (p.table.length_m, p.table.width_m, t),
                     (center[0], center[1], -t / 2))
    return BodyNode(
        name="table", prims=[prim], shape=_shape_from_prims([prim], "table"),
        note="静体; 原点在桌面左下角, 世界 z=0 = 桌面上表面。",
    )


BUILDERS = {
    "pallet": build_pallet,
    "goods_box": build_goods_box,
    "forklift": build_forklift,
    "flat_agv": build_flat_agv,
    "station": build_station,
    "rack_group": build_rack_group,
    "feed_tray": build_feed_tray,
    "manipulator": build_manipulator,
    "table": build_table,
}


def _compose_tree_shape(root: BodyNode) -> None:
    """多体模型: 根 shape 汇总全树几何（按各 body 帧位姿放置; STEP/STL 需要整树）。"""
    if not root.children:
        return
    placed = []

    def loc_of(node: BodyNode):
        base = Location(node.pos_m)
        q = node.quat_wxyz
        if q is None:
            return base
        yaw_deg = math.degrees(2.0 * math.atan2(q[3], q[0]))
        return base * Rot(0, 0, yaw_deg)

    def walk(node: BodyNode, base) -> None:
        here = base * loc_of(node)
        if node.shape is not None:
            placed.append(here * node.shape)
        for child in node.children:
            walk(child, here)

    walk(root, Location())
    comp = Compound(children=placed)
    comp.label = root.name
    root.shape = comp


def build_all_models(p: SimpleNamespace, derived: Derived | None = None) -> dict[str, BodyNode]:
    """构建全部 9 个参数化模型（W02 模型清单, 顺序固定）。"""
    derived = derived or derive_parameters(p)
    models = {name: builder(p, derived) for name, builder in BUILDERS.items()}
    for node in models.values():
        _compose_tree_shape(node)
    return models
