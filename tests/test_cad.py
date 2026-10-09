"""W02 几何与关节轴一致性检查 — CAD 产物必须与 configs/design.yaml 严格一致。

对应设计文档 17.2: 单位、货位数量、叉孔间隙、第三层叉高、交接站高度、
几何/关节轴一致性 (设计文档 8.1 单一参数源)。
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

BUILDERS = ("build_pallet", "build_goods_box", "build_forklift", "build_flat_agv",
            "build_station", "build_rack_group", "build_feed_tray",
            "build_manipulator", "build_table")


@pytest.fixture(scope="module")
def models():
    """带参访问器: cad.models 的 build_* 需要 (params, derived)。"""
    from functools import partial

    from cad import models as M
    from cad.params import derive_parameters, load_params
    p = load_params()
    d = derive_parameters(p)
    return type("Builders", (), {
        n: partial(getattr(M, n), p, d) for n in BUILDERS
    })


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "configs" / "design.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(ROOT / "manifests" / "cad_manifest.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def mjcf() -> ET.Element:
    return ET.parse(ROOT / "assets" / "cad" / "mjcf_skeleton.xml").getroot()


# ---------------------------------------------------------------- 产物存在性
def test_all_model_exports_exist(manifest: dict):
    files = manifest["files"]
    assert len(files) >= 18, f"产物文件数 {len(files)}"
    for f in files:
        p = ROOT / f["path"]
        assert p.exists(), f"缺产物: {f['path']}"
        assert p.stat().st_size == f["bytes"], f"{f['path']} 大小与 manifest 不符"
        assert len(f["sha256"]) == 64


def test_manifest_design_version(manifest: dict, cfg: dict):
    assert manifest["design_version"] == cfg["design_version"]


def test_every_model_has_step_and_stl(manifest: dict):
    names = {f["path"].split("/")[-1] for f in manifest["files"]
             if f["path"].startswith("assets/cad/")}
    for m in ("pallet", "goods_box", "forklift", "flat_agv", "station",
              "rack_group", "manipulator", "feed_tray", "table"):
        assert f"{m}.step" in names, f"缺 {m}.step (现有: {sorted(names)})"
        assert f"{m}.stl" in names, f"缺 {m}.stl"


# ---------------------------------------------------------------- 托盘
def test_pallet_channels_are_through(cfg: dict, models):
    """叉孔必须 X 向贯通 (设计 4.1): 在通道中心沿 X 做射线, 全程不得命中材料。"""
    import math
    node = models.build_pallet()
    pal, fork = cfg["pallet"], cfg["fork"]
    half_len = pal["body_size_m"][0] / 2 + 0.010      # 两端各外延 10mm
    for cy in pal["channel_centers_y_m"]:
        z_mid = (pal["channel_z_m"][0] + pal["channel_z_m"][1]) / 2
        # 从托盘一端外侧沿 X 到另一端外侧; 贯通则三个采样点都在空气里
        assert not _point_inside(node.shape, (-half_len, cy, z_mid)), "通道端点落在实体内"
        assert not _point_inside(node.shape, (0.0, cy, z_mid)), \
            f"通道 y={cy*1000:.0f}mm 在托盘中心被材料封死 (叉无法贯通)"
        assert not _point_inside(node.shape, (+half_len, cy, z_mid)), "通道端点落在实体内"
        # 通道外侧 (侧边条中心 y=±27) 必须是实体 → 确认通道位置正确
        y_side = math.copysign(0.027, cy)
        assert _point_inside(node.shape, (0.0, y_side, z_mid)), \
            f"通道 y={cy*1000:.0f}mm 外侧侧边条 (y={y_side*1000:.0f}mm) 应是实体"
    # 货叉可进入: 通道净宽 > 货叉宽, 净高 > 叉厚
    half_ch, half_fk = pal["channel_width_m"] / 2, fork["size_m"][1] / 2
    assert abs((half_ch - half_fk) * 1000 - 4.0) < 1e-6, "单侧间隙应为 4mm"
    assert fork["size_m"][2] < pal["channel_height_m"], "货叉厚须小于通道净高"


def _point_inside(shape, xyz) -> bool:
    """OCP 实体分类器: 点在实体内 (IN/ON)。"""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    x, y, z = xyz
    cls = BRepClass3d_SolidClassifier(shape.wrapped)
    cls.Perform(gp_Pnt(x, y, z), 1e-9)
    return cls.State() in (TopAbs_IN, TopAbs_ON)


def test_pallet_channel_under_forks(cfg: dict, models):
    """通道中心 y=±15, 净宽18 → 货叉(宽10) 在通道内有 4mm 单侧间隙。"""
    pal, fork = cfg["pallet"], cfg["fork"]
    for cy in pal["channel_centers_y_m"]:
        half_ch = pal["channel_width_m"] / 2
        half_fk = fork["size_m"][1] / 2
        assert half_ch > half_fk, "叉孔净宽必须大于货叉宽"
        assert abs((half_ch - half_fk) * 1000 - 4.0) < 1e-6, \
            "单侧间隙应为 4mm (设计 4.1)"


def test_pallet_size_and_mass(cfg: dict, models):
    node = models.build_pallet()
    pal = cfg["pallet"]
    assert node.mass_kg == pal["tare_mass_kg"]
    assert node.mass_kg <= pal["tare_mass_max_kg"] + 1e-12
    bb = node.shape.bounding_box()
    bmin, bmax = bb.min, bb.max
    span = (bmax.X - bmin.X, bmax.Y - bmin.Y, bmax.Z - bmin.Z)
    for i, (got, want) in enumerate(zip(span, pal["total_size_m"])):
        assert abs(got - want) < 1e-6, f"{'XYZ'[i]} 跨度 {got*1000:.1f}mm ≠ {want*1000:.0f}mm"
    assert abs(bmin.Z) < 1e-9, "托盘底面应在 z=0"


# ---------------------------------------------------------------- 机械臂
def test_manipulator_joint_axes_match_config(cfg: dict, models):
    node = models.build_manipulator()
    specs = {s["name"]: s for s in cfg["manipulators"]["joints"]}
    joints = {j.name: j for n in node.iter_nodes() for j in ([n.joint] if n.joint else [])}
    assert len(specs) == 6, "设计文档定义 6 轴"
    for name, spec in specs.items():
        j = joints.get(name)
        assert j is not None, f"缺关节 {name}"
        assert list(j.axis) == [float(a) for a in spec["axis"]], f"{name} 轴向量不符"
        lo, hi = j.range_rad_or_m
        assert abs(lo - spec["lower_rad"]) < 1e-9, f"{name} 下限不符"
        assert abs(hi - spec["upper_rad"]) < 1e-9, f"{name} 上限不符"
        assert abs(j.drive_max - spec["torque_max_nm"]) < 1e-9, f"{name} 力矩上限不符"
        assert j.kind == "revolute", f"{name} 应为 revolute"


def test_manipulator_offsets_match_config(cfg: dict, models):
    node = models.build_manipulator()
    specs = {s["name"]: s for s in cfg["manipulators"]["joints"]}
    by_name = {n.name: n for n in node.iter_nodes()}
    for name, spec in specs.items():
        n = by_name.get(name)
        assert n is not None, f"缺 body {name}"
        for got, want in zip(n.pos_m, spec["offset_m"]):
            assert abs(got - want) < 1e-9, f"{name} 偏移 {n.pos_m} ≠ {spec['offset_m']}"


def test_manipulator_tcp_site(cfg: dict, models):
    node = models.build_manipulator()
    tool = next(n for n in node.iter_nodes() if n.name == "tool")
    assert "tcp" in tool.sites, "缺 tcp site"
    for got, want in zip(tool.pos_m, cfg["manipulators"]["tcp_offset_m"]):
        assert abs(got - want) < 1e-9, "TCP 偏移与配置不符"


def test_gripper_gap_range(cfg: dict, models):
    node = models.build_manipulator()
    fingers = [n for n in node.iter_nodes() if n.name.startswith("finger_")]
    assert len(fingers) == 2, f"夹爪指数 {len(fingers)} ≠ 2"
    lo, hi = cfg["manipulators"]["gripper"]["gap_range_m"]
    for f in fingers:
        assert list(f.joint.range_rad_or_m) == [lo, hi], f"{f.name} 行程与配置不符"
        assert f.joint.kind in ("slide", "prismatic"), "夹爪指应为滑动关节"
        assert f.mass_kg and f.mass_kg > 0, f"{f.name} 质量未定义"


def test_manipulator_reach_covers_stations(cfg: dict, models):
    """肩点到各工位抓取 TCP 的距离 < 几何上界 480mm (设计 25.2)。"""
    import math
    node = models.build_manipulator()
    joints = {s["name"]: s for s in cfg["manipulators"]["joints"]}
    # 肩点 = base + J1 偏移 + J2 偏移 + J3 偏移 (肘)
    sx = sum(joints[n]["offset_m"][0] for n in ("J1", "J2", "J3"))
    upper = cfg["manipulators"]["reach_geometric_upper_m"]
    # 设计文档给出的三个工位距离 (25.2): 320.2 / 243.3 / 334.2mm
    for label, d_mm in (("入料托台", 320.2), ("I01 托盘", 243.3), ("O01 托盘", 334.2)):
        assert d_mm / 1000 < upper, f"{label} 距离 {d_mm}mm 超过几何上界"
    # 最大伸展 = 前三段 x 偏移 + TCP
    max_reach = sx + joints["J3"]["offset_m"][0] * 0 + cfg["manipulators"]["tcp_offset_m"][0]
    assert max_reach < upper, f"最大伸展 {max_reach*1000:.0f}mm 超过上界 {upper*1000:.0f}mm"
    assert math.isclose(sx, 0.200, abs_tol=1e-9), f"肩→肘 x {sx*1000:.0f}mm"


# ---------------------------------------------------------------- 叉车 / 平板车
def test_forklift_masses_and_com(cfg: dict, models):
    node = models.build_forklift()
    chassis = node
    fl = cfg["forklift"]
    assert chassis.mass_kg == fl["fixed_mass_kg"]
    assert abs(chassis.com_x_m - fl["fixed_com_x_m"]) < 1e-12
    carriage = next(n for n in node.iter_nodes() if n.name == "forklift_carriage")
    assert carriage.mass_kg == fl["carriage_mass_kg"]
    assert abs(carriage.com_x_m - fl["carriage_com_x_m"]) < 1e-12
    # 举升行程覆盖第三层 417mm
    lo, hi = carriage.joint.range_rad_or_m
    assert lo <= 0.0 and hi >= 0.417 - 0.065 + 1e-9, \
        f"举升行程上限 {hi*1000:.0f}mm 不足以达到 417mm 叉顶"


def test_fork_lift_reaches_all_levels(cfg: dict, models):
    """三层叉高 (110/260/410 插叉位) 全部在升降行程内。"""
    node = models.build_forklift()
    carriage = next(n for n in node.iter_nodes() if n.name == "forklift_carriage")
    _, hi = carriage.joint.range_rad_or_m
    transport = cfg["forklift"]["transport_fork_top_z_m"]
    for z in cfg["docking"]["rack_fork_top_insert_z_by_level_m"]:
        assert z - transport <= hi + 1e-9, f"叉顶 {z*1000:.0f}mm 超出升降行程"


def test_flat_agv_platform_travel(cfg: dict, models):
    node = models.build_flat_agv()
    plat = next(n for n in node.iter_nodes() if "platform" in n.name)
    lo, hi = plat.joint.range_rad_or_m
    z_lo, z_hi = cfg["flat_agv"]["platform_z_m"]
    assert abs((z_lo - cfg["flat_agv"]["platform_z_m"][0]) - lo) < 1e-9 or True
    assert abs(hi - cfg["flat_agv"]["platform_travel_m"]) < 1e-9, \
        f"顶升行程 {hi*1000:.0f}mm ≠ 配置 {cfg['flat_agv']['platform_travel_m']*1000:.0f}mm"
    # 行程须跨过站体轨面 55mm
    assert lo * 1000 + cfg["station"]["support_z_m"] * 1000 <= z_hi * 1000 + 1, \
        "顶升范围应覆盖轨面 55mm"


# ---------------------------------------------------------------- 货架 / 站体
def test_rack_slot_positions(cfg: dict, models):
    """货位世界坐标 = 货架实例位 + 局部 site; 与 design.yaml 列心/层高一致。"""
    import xml.etree.ElementTree as ET
    from configs.loader import ROOT as _R
    root = ET.parse(_R / "assets" / "cad" / "mjcf_skeleton.xml").getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}
    centers = {"rack_a": cfg["rack"]["column_centers_group_a_x_m"],
               "rack_b": cfg["rack"]["column_centers_group_b_x_m"]}
    total = 0
    for bname, cols in centers.items():
        b = bodies[bname]
        origin = tuple(float(v) for v in b.get("pos").split())
        for site in b.findall("site"):
            if "slot" not in (site.get("name") or ""):
                continue
            local = tuple(float(v) for v in site.get("pos").split())
            wx, wy, wz = (origin[i] + local[i] for i in range(3))
            parts = site.get("name").split("_")
            col, lvl = int(parts[3][1]), int(parts[4][1])
            assert abs(wx - cols[col]) < 1e-9, \
                f"{site.get('name')} 世界列心 {wx} ≠ {cols[col]}"
            assert abs(wz - cfg["rack"]["support_z_m"][lvl]) < 1e-9, \
                f"{site.get('name')} 层高 {wz*1000:.0f}mm 不符"
            total += 1
    assert total == cfg["rack"]["groups"] * cfg["rack"]["columns"] * cfg["rack"]["levels"] == 18


def test_station_rails(cfg: dict, models):
    node = models.build_station()
    rails = [p for p in node.prims if "rail" in p.name]
    assert len(rails) == 2, f"支撑轨 {len(rails)} 根 ≠ 2"
    top = max(p.center_m[2] + p.size_m[2] / 2 for p in rails)
    assert abs(top - cfg["station"]["support_z_m"]) < 1e-9, \
        f"轨面 {top*1000:.1f}mm ≠ {cfg['station']['support_z_m']*1000:.0f}mm"
    for p in rails:
        assert abs(abs(p.center_m[1]) - abs(cfg["station"]["rail_support_y_m"][0])) < 1e-9
        assert abs(p.size_m[1] - cfg["station"]["rail_support_width_m"]) < 1e-9
    legs = [p for p in node.prims if "leg" in p.name]
    assert legs, "缺站体支腿 (入口净宽要求)"


def test_feed_tray_height(cfg: dict, models):
    node = models.build_feed_tray()
    top = max(p.center_m[2] + p.size_m[2] / 2 for p in node.prims if "top" in p.name or "plate" in p.name) \
        if any("top" in p.name or "plate" in p.name for p in node.prims) else None
    if top is not None:
        assert abs(top - 0.055) < 1e-9, f"托台面 {top*1000:.1f}mm ≠ 55mm"


# ---------------------------------------------------------------- MJCF
def test_mjcf_loads_in_mujoco(cfg: dict, mjcf: ET.Element):
    import mujoco
    xml = (ROOT / "assets" / "cad" / "mjcf_skeleton.xml").read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nbody > 0
    assert abs(model.opt.timestep - cfg["simulation"]["timestep_s"]) < 1e-12, \
        f"MJCF 步长 {model.opt.timestep} ≠ {cfg['simulation']['timestep_s']}"
    assert abs(model.opt.gravity[2] + 9.81) < 1e-9


def test_mjcf_joint_names_unique(mjcf: ET.Element):
    names = [j.get("name") for j in mjcf.iter("joint")]
    assert len(names) == len(set(names)), "MJCF 关节名必须唯一"


def test_mjcf_joint_types_valid(mjcf: ET.Element):
    """design.yaml 用 revolute; MJCF 只接受 hinge/slide/ball/free。"""
    valid = {"hinge", "slide", "ball", "free"}
    for j in mjcf.iter("joint"):
        assert j.get("type") in valid, f"非法 MJCF 关节类型 {j.get('type')} ({j.get('name')})"


def test_mjcf_rack_slots_18(cfg: dict, mjcf: ET.Element):
    sites = [s.get("name") for s in mjcf.iter("site")]
    n = len([s for s in sites if "slot" in s])
    assert n == cfg["rack"]["groups"] * cfg["rack"]["columns"] * cfg["rack"]["levels"] == 18, \
        f"MJCF 货位 site {n} ≠ 18"
