"""投资人向照片级渲染 — 生产级**无人叉车 AGV** (激光导航) + 真实仓储货架。

与 render_photoreal.py 同一位姿真值 (assets/pbr/poses.csv, mj_step 逐帧),
同一碰撞包络 (configs/design.yaml); 区别是外观按真实工业设备重做:
  - 无人叉车: 无驾驶室/护顶架 → 低趴双色机身 (浅灰壳+深色底座), 前后
    激光安全扫描仪 (红色视窗), 顶置 LiDAR 圆柱, 琥珀警示灯 (自发光),
    急停蘑菇头+黄底面板, 状态 LED, 双级门架 (C 型钢+升降油缸+链条),
    调叉架+钩臂货叉 (blade 严格 = 物理叉孔包络), 后仰货架护栏
  - 货架: C 型钢立柱+底脚锚栓+横/斜撑片, 橙色横梁带挂耳, 三层视觉层
    (仅第三层有物理支撑面), 背网
  - 环境: 抛光混凝土地面+黄色安全标线, 背景双组货架 (景深虚化)

用法 (headless):
    blender -b -P scripts/render_agv.py -- --frame 225               # 测试帧
    blender -b -P scripts/render_agv.py -- --video out/agv-frames --step 8
资产: assets/pbr/{Concrete028,PaintedMetal001,MetalPlates006,Wood096}/, HDRI。
"""
from __future__ import annotations

import csv
import math
import os

import bpy

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
PBR = os.path.join(ROOT, "assets", "pbr")
POSES = f"{PBR}/poses.csv"

# ── 设计包络 (configs/design.yaml, m) ────────────────────────────────
SUPPORT_Z = 0.400                      # 第三层支撑面 (梁顶)
PAL_LX, PAL_LY, PAL_HZ = 0.080, 0.060, 0.021   # 托盘 80x60x21
CH_X0, CH_X1, CH_W, CH_Z = -0.195, 0.025, 0.150, 0.035   # 底盘包络
MAST_H = 0.540
FORK_ROOT, FORK_LEN, FORK_TOPZ = 0.040, 0.080, 0.065     # 叉根x/长/运输叉顶高
FORK_W, FORK_T, FORK_YS = 0.010, 0.0025, 0.015           # 叉宽/厚/半间距
CH_Y = 0.004, 0.012                  # 叉孔 z (设计 channel_z_m)
WHEEL_R = 0.025
BOX_LX, BOX_LY, BOX_LZ = 0.060, 0.040, 0.050   # 货物纸箱设计尺寸
STAGE_CX, STAGE_HX, STAGE_HY, STAGE_TOPZ = -0.340, 0.060, 0.050, 0.045


def sys_argv_after():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


import sys  # noqa: E402


def arg(flag, default=None):
    argv = sys_argv_after()
    return argv[argv.index(flag) + 1] if flag in argv else default


# ── 材质 helpers (与 render_photoreal 同一套约定) ────────────────────
def pbr_mat(name, subdir, *, tint=None, rough_mul=1.0,
            metallic=0.0, scale=1.0, wear=0.0, tint_mode="mix"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def img(fn):
        im = bpy.data.images.load(f"{PBR}/{subdir}/{subdir}_2K-JPG_{fn}.jpg")
        im.colorspace_settings.name = "sRGB" if fn == "Color" else "Non-Color"
        return im

    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    col = nt.nodes.new("ShaderNodeTexImage"); col.image = img("Color")
    rgh = nt.nodes.new("ShaderNodeTexImage"); rgh.image = img("Roughness")
    nrm = nt.nodes.new("ShaderNodeTexImage"); nrm.image = img("NormalGL")
    for x in (col, rgh, nrm):
        nt.links.new(mp.outputs["Vector"], x.inputs["Vector"])
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    base = col.outputs["Color"]
    # Blender 5.2 ShaderNodeMix(RGBA): A=6 B=7 Result=2 Factor=0。
    if tint is not None:
        mixn = nt.nodes.new("ShaderNodeMix"); mixn.data_type = "RGBA"
        if tint_mode == "mul":
            mixn.blend_type = "MULTIPLY"
            mixn.inputs[0].default_value = 1.0
        else:
            mixn.inputs[0].default_value = 0.10
        mixn.inputs[6].default_value = tint
        nt.links.new(base, mixn.inputs[7])
        base = mixn.outputs[2]
    if wear > 0:   # 边缘掉漆: Pointiness → ColorRamp → 裸金属灰
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.55
        ramp.color_ramp.elements[1].position = 0.70
        nt.links.new(geo.outputs["Pointiness"], ramp.inputs["Fac"])
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 0.4
        nt.links.new(ramp.outputs["Color"], mul.inputs[0])
        mixw = nt.nodes.new("ShaderNodeMix"); mixw.data_type = "RGBA"
        nt.links.new(mul.outputs[0], mixw.inputs[0])
        nt.links.new(base, mixw.inputs[6])
        mixw.inputs[7].default_value = (0.55, 0.54, 0.52, 1)
        base = mixw.outputs[2]
    nt.links.new(base, bsdf.inputs["Base Color"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = 0.05
    rr.inputs["To Max"].default_value = rough_mul
    nt.links.new(rgh.outputs["Color"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], bsdf.inputs["Roughness"])
    bsdf.inputs["Metallic"].default_value = metallic
    return m


def flat_mat(name, rgba, rough=0.5, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = rgba
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m


def emit_mat(name, rgba, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = rgba
    b.inputs["Emission Color"].default_value = rgba
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Roughness"].default_value = 0.3
    return m


def floor_mat():
    """抛光工业地坪: 大板混凝土 + 伸缩缝网格 + 光泽不均 + 缝内凹槽."""
    m = bpy.data.materials.new("floor_polished")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Metallic"].default_value = 0.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.32, 0.32, 0.32)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def tex(fn):
        im = bpy.data.images.load(
            f"{PBR}/Concrete028/Concrete028_2K-JPG_{fn}.jpg")
        im.colorspace_settings.name = "sRGB" if fn == "Color" else "Non-Color"
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = im
        return t

    col, nrm = tex("Color"), tex("NormalGL")
    nt.links.new(mp.outputs["Vector"], col.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], nrm.inputs["Vector"])
    # 伸缩缝: Brick 网格 (米制, 0.42m 板幅)
    br = nt.nodes.new("ShaderNodeTexBrick")
    for k, v in (("Scale", 1.0), ("Brick Width", 0.42),
                 ("Row Height", 0.42), ("Mortar Size", 0.0035)):
        br.inputs[k].default_value = v
    br.inputs["Color1"].default_value = (1, 1, 1, 1)
    br.inputs["Color2"].default_value = (1, 1, 1, 1)
    br.inputs["Mortar"].default_value = (0, 0, 0, 1)
    nt.links.new(tc.outputs["Object"], br.inputs["Vector"])
    mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
    nt.links.new(br.outputs["Fac"], mixc.inputs[0])
    nt.links.new(col.outputs["Color"], mixc.inputs[6])
    mixc.inputs[7].default_value = (0.030, 0.030, 0.033, 1)
    nt.links.new(mixc.outputs[2], b.inputs["Base Color"])
    # 光泽不均: 大尺度噪声 → rough 0.10..0.26, 缝道抬到 0.55
    noi = nt.nodes.new("ShaderNodeTexNoise")
    noi.inputs["Scale"].default_value = 1.6
    noi.inputs["Detail"].default_value = 5.0
    nt.links.new(tc.outputs["Object"], noi.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    for k, v in (("From Min", 0.35), ("From Max", 0.65),
                 ("To Min", 0.16), ("To Max", 0.32)):
        mr.inputs[k].default_value = v
    nt.links.new(noi.outputs["Fac"], mr.inputs["Value"])
    mixr = nt.nodes.new("ShaderNodeMix"); mixr.data_type = "FLOAT"
    nt.links.new(br.outputs["Fac"], mixr.inputs[0])
    nt.links.new(mr.outputs["Result"], mixr.inputs[2])
    mixr.inputs[3].default_value = 0.55
    nt.links.new(mixr.outputs[0], b.inputs["Roughness"])
    # 法线: 混凝土细纹 + 缝凹槽 bump
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = 0.55
    nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(br.outputs["Fac"], inv.inputs[1])
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Distance"].default_value = 0.0006
    bmp.inputs["Strength"].default_value = 0.6
    nt.links.new(inv.outputs[0], bmp.inputs["Height"])
    nt.links.new(nm.outputs["Normal"], bmp.inputs["Normal"])
    nt.links.new(bmp.outputs["Normal"], b.inputs["Normal"])
    return m


def paint_line_mat(name, rgba, faded):
    """磨损地坪漆: 噪声斑驳 + 边缘粗化, 像滚涂在混凝土上的标线漆."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noi = nt.nodes.new("ShaderNodeTexNoise")
    noi.inputs["Scale"].default_value = 55.0
    noi.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], noi.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.38
    ramp.color_ramp.elements[1].position = 0.62
    nt.links.new(noi.outputs["Fac"], ramp.inputs["Fac"])
    mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
    nt.links.new(ramp.outputs["Color"], mixc.inputs[0])
    mixc.inputs[6].default_value = rgba
    mixc.inputs[7].default_value = faded
    nt.links.new(mixc.outputs[2], b.inputs["Base Color"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    for k, v in (("From Min", 0.3), ("From Max", 0.7),
                 ("To Min", 0.42), ("To Max", 0.75)):
        mr.inputs[k].default_value = v
    nt.links.new(noi.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Distance"].default_value = 0.0004
    nt.links.new(noi.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], b.inputs["Normal"])
    return m


def box(name, size3, loc, mat, bevel=0.0012, parent=None, rot=None):
    bpy.ops.mesh.primitive_cube_add(size=2)
    o = bpy.context.object
    o.name = name
    o.scale = (size3[0] / 2, size3[1] / 2, size3[2] / 2)
    o.location = loc
    if rot:
        o.rotation_euler = rot
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        b = o.modifiers.new("bevel", "BEVEL")
        b.width, b.segments, b.limit_method = bevel, 2, "ANGLE"
    o.data.materials.append(mat)
    o["sem_label"] = name
    if parent:
        o.parent = parent
    return o


def cyl(name, r, depth, loc, mat, axis="Y", bevel=0.0, parent=None, verts=24):
    rot = {"Y": (math.pi / 2, 0, 0), "X": (0, math.pi / 2, 0),
           "Z": (0, 0, 0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth,
                                        location=loc, rotation=rot,
                                        vertices=verts)
    o = bpy.context.object
    o.name = name
    if bevel:
        b = o.modifiers.new("bevel", "BEVEL")
        b.width, b.segments = bevel, 3
    o.data.materials.append(mat)
    if parent:
        o.parent = parent
    return o


def sphere(name, r, loc, mat, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc,
                                         segments=24, ring_count=12)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    if parent:
        o.parent = parent
    return o


def empty(name, loc):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    bpy.context.collection.objects.link(e)
    return e


def load_poses():
    with open(POSES) as f:
        rd = csv.DictReader(f)
        return [{k: (float(v) if k != "seg" else int(float(v)))
                 for k, v in row.items()} for row in rd]


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = True
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 48
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720

    # ── 材质表 ──
    mat_conc = floor_mat()                 # 抛光地坪+伸缩缝+光泽不均
    mat_orange = pbr_mat("rack_beam", "PaintedMetal001",
                         tint=(0.60, 0.13, 0.01, 1), scale=30.0,
                         rough_mul=0.45, wear=1)
    mat_blue = pbr_mat("rack_upright", "PaintedMetal001",
                       tint=(0.03, 0.08, 0.26, 1), scale=25.0, rough_mul=0.5,
                       wear=1)
    mat_steel = pbr_mat("fork_steel", "MetalPlates006", scale=60.0,
                        metallic=1.0, rough_mul=0.35)
    mat_dark = pbr_mat("machinery_dark", "MetalPlates006",
                       tint=(0.05, 0.05, 0.06, 1), scale=50.0,
                       metallic=0.9, rough_mul=0.55)
    mat_wood = pbr_mat("pallet_wood", "Wood096", tint=(0.72, 0.48, 0.28, 1),
                       scale=30.0, rough_mul=0.8, tint_mode="mul")
    mat_rubber = flat_mat("tire", (0.018, 0.018, 0.02, 1), rough=0.85)
    mat_card = flat_mat("cardboard", (0.45, 0.30, 0.16, 1), rough=0.8)
    mat_label = flat_mat("label", (0.9, 0.9, 0.87, 1), rough=0.6)
    # AGV 机身
    mat_shell = flat_mat("agv_shell", (0.82, 0.83, 0.84, 1), rough=0.28)
    mat_base = flat_mat("agv_base", (0.035, 0.036, 0.04, 1), rough=0.5)
    mat_accent = flat_mat("agv_accent", (0.85, 0.32, 0.02, 1), rough=0.35)
    mat_chrome = flat_mat("chrome", (0.75, 0.76, 0.78, 1), rough=0.12,
                          metallic=1.0)
    mat_scanner = flat_mat("scanner_body", (0.02, 0.02, 0.023, 1), rough=0.4)
    mat_lens = emit_mat("scanner_lens", (1.0, 0.06, 0.04, 1), 4.0)
    mat_beacon = emit_mat("beacon_amber", (1.0, 0.42, 0.03, 1), 4.0)
    mat_led_g = emit_mat("led_green", (0.1, 1.0, 0.25, 1), 8.0)
    mat_ered = flat_mat("estop_red", (0.55, 0.02, 0.02, 1), rough=0.3)
    mat_yellow_plate = flat_mat("estop_plate", (0.85, 0.62, 0.02, 1),
                                rough=0.4)
    mat_line = paint_line_mat("floor_line", (0.78, 0.56, 0.02, 1),
                              (0.30, 0.24, 0.06, 1))
    mat_line_blk = paint_line_mat("floor_line_black", (0.020, 0.020, 0.022, 1),
                                  (0.075, 0.072, 0.070, 1))
    mat_mesh = flat_mat("wire_mesh", (0.28, 0.29, 0.31, 1), rough=0.5,
                        metallic=0.9)

    # ── 地面 + 安全标线 ──
    bpy.ops.mesh.primitive_plane_add(size=8)
    fl = bpy.context.object
    fl.name = "floor"
    fl.data.materials.append(mat_conc)
    fl["sem_label"] = "floor"
    # 标线 (磨损漆, 微离地防 z-fighting) + 巷道虚线 + 暂存区斑马警示带
    box("line_lane", (0.90, 0.014, 0.0012), (-0.10, -0.16, 0.0006),
        mat_line, bevel=0)
    for i in range(9):
        box(f"dash_{i}", (0.060, 0.014, 0.0012), (-0.52 + i * 0.105, 0.20,
            0.0006), mat_line, bevel=0)
    box("line_rack", (0.014, 0.46, 0.0012), (0.26, 0.0, 0.0006),
        mat_line, bevel=0)
    box("line_stage", (0.014, 0.16, 0.0012), (STAGE_CX - 0.09, 0.0, 0.0006),
        mat_line, bevel=0)
    for i in range(8):
        box(f"haz_{i}", (0.009, 0.030, 0.0012),
            (STAGE_CX - 0.080, -0.063 + i * 0.018, 0.0006),
            mat_line if i % 2 else mat_line_blk, bevel=0,
            rot=(0, 0, math.radians(45)))

    # ── 货架 (真实 selective rack): 立柱在巷道开口左右 (x=±0.048, y=±0.080),
    #    横梁沿深度 y 向布置在托盘两端下方 (x=±0.040), 托盘纵梁横跨双梁 ──
    def build_rack(tag, x0, y0, fy=0.080):
        grp = empty(f"rack_{tag}", (x0, y0, 0))
        H = SUPPORT_Z + 0.14
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = sx * 0.048, sy * fy
                # C 型钢柱: 腹板 (面向巷道) + 双翼缘朝 ±y
                box(f"upr_{tag}_{sx}_{sy}_web", (0.014, 0.003, H),
                    (x, y, H / 2), mat_blue, bevel=0.0006, parent=grp)
                for fx in (-0.0055, 0.0055):
                    box(f"upr_{tag}_{sx}_{sy}_f{fx:.3f}", (0.003, 0.010, H),
                        (x + fx, y, H / 2), mat_blue, bevel=0.0004,
                        parent=grp)
                # 底脚板 + 2 化学锚栓
                box(f"foot_{tag}_{sx}_{sy}", (0.026, 0.034, 0.0035),
                    (x, y, 0.0018), mat_dark, bevel=0.0006, parent=grp)
                for ay in (-0.011, 0.011):
                    cyl(f"anchor_{tag}_{sx}_{sy}_{ay}", 0.0016, 0.004,
                        (x, y + ay, 0.0055), mat_chrome, axis="Z",
                        parent=grp, verts=10)
        # 柱片横撑 (每侧前后柱间 x 向, 两层) + 侧向拉杆 (y 向, 底层)
        for sy in (-1, 1):
            for bz in (0.09, 0.24):
                box(f"brace_{tag}_{sy}_{bz}", (0.070, 0.004, 0.012),
                    (0, sy * fy, bz), mat_blue, bevel=0.0005, parent=grp)
        for sx in (-1, 1):
            box(f"tie_{tag}_{sx}", (0.004, 2 * fy - 0.010, 0.008),
                (sx * 0.048, 0, 0.055), mat_blue, bevel=0.0004, parent=grp)
        # 横梁: y 向双梁 x=±0.040 (顶面=层标高), 端部挂耳抱柱
        for lz in (0.160, 0.280, 0.400):
            for sx in (-1, 1):
                x = sx * 0.040
                box(f"beam_{tag}_{sx}_{lz}", (0.012, 2 * fy - 0.004, 0.007),
                    (x, 0, lz - 0.0035), mat_orange, bevel=0.0008,
                    parent=grp)
                for sy in (-1, 1):
                    box(f"hook_{tag}_{sx}_{sy}_{lz}",
                        (0.014, 0.004, 0.017), (x + sx * 0.003,
                        sy * (fy - 0.001), lz - 0.005), mat_orange,
                        bevel=0.0006, parent=grp)
            # 下层线网 (x 向细条搭在双梁上)
            if lz < SUPPORT_Z:
                for i in range(5):
                    box(f"deck_{tag}_{lz}_{i}", (0.078, 0.0018, 0.0018),
                        (0, -fy + 0.016 + i * (2 * fy - 0.032) / 4,
                         lz - 0.0005), mat_mesh, bevel=0, parent=grp)
        # 背网 (后排柱片)
        for i in range(5):
            box(f"back_{tag}_{i}", (0.078, 0.002, 0.010),
                (0, fy + 0.010, 0.08 + i * 0.09), mat_mesh, bevel=0,
                parent=grp)
        return grp

    rack = build_rack("main", 0.0, 0.0)

    # ── 暂存站: 钢台面 + 圆腿 + 下层板 ──
    box("stage_top", (STAGE_HX * 2, STAGE_HY * 2, 0.004),
        (STAGE_CX, 0, STAGE_TOPZ - 0.002), mat_steel, bevel=0.0008)
    for lx in (STAGE_CX - 0.052, STAGE_CX + 0.052):
        for ly in (-0.042, 0.042):
            cyl(f"stage_leg_{lx:.2f}_{ly:.0f}", 0.0035, 0.041,
                (lx, ly, 0.0205), mat_dark, axis="Z", parent=None, verts=14)
    box("stage_brace", (0.104, 0.005, 0.005), (STAGE_CX, 0, 0.012),
        mat_dark, bevel=0.0006)
    box("stage_shelf", (0.104, 0.084, 0.002), (STAGE_CX, 0, 0.016),
        mat_steel, bevel=0.0005)

    # ── 无人叉车 AGV (chassis empty 驱动) ──
    ch = empty("agv", (0, 0, 0))
    # 底座 (深色, 大圆角) x∈[-0.190,0.020]
    box("base", (0.208, 0.138, 0.030), (-0.085, 0, 0.027), mat_base,
        bevel=0.006, parent=ch)
    # 主壳体 (浅灰): 后高前低的楔形 = 两大圆角盒叠出
    box("shell_rear", (0.105, 0.128, 0.058), (-0.130, 0, 0.071), mat_shell,
        bevel=0.009, parent=ch)
    box("shell_front", (0.085, 0.122, 0.036), (-0.038, 0, 0.060), mat_shell,
        bevel=0.008, parent=ch)
    # 橙色腰线 + 前脸深色保险杠带
    box("stripe", (0.150, 0.129, 0.006), (-0.085, 0, 0.049), mat_accent,
        bevel=0.002, parent=ch)
    box("bumper_f", (0.010, 0.120, 0.026), (0.021, 0, 0.046), mat_base,
        bevel=0.005, parent=ch)
    box("bumper_r", (0.010, 0.120, 0.026), (-0.192, 0, 0.046), mat_base,
        bevel=0.005, parent=ch)
    # 前面板 (深灰竖面) + 状态 LED 条
    box("face", (0.006, 0.116, 0.030), (0.016, 0, 0.066), mat_base,
        bevel=0.003, parent=ch)
    box("led", (0.002, 0.060, 0.005), (0.0195, 0, 0.078), mat_led_g,
        bevel=0, parent=ch)
    # 激光安全扫描仪 (前低窗 + 后同位): 黑盒 + 红色发光视窗
    for sgn, sx in ((1, 0.0235), (-1, -0.1915)):
        box(f"scan_{sgn}", (0.008, 0.044, 0.016), (sx, 0, 0.058), mat_scanner,
            bevel=0.002, parent=ch)
        box(f"scan_lens_{sgn}", (0.002, 0.034, 0.007),
            (sx + sgn * 0.0045, 0, 0.058), mat_lens, bevel=0, parent=ch)
    # 顶置 LiDAR (SICK TiV5 风格: 圆柱+黑顶) 前左角
    cyl("lidar", 0.0155, 0.026, (-0.012, 0.043, 0.096), mat_scanner,
        axis="Z", parent=ch, verts=28)
    cyl("lidar_cap", 0.0125, 0.010, (-0.012, 0.043, 0.113), mat_base,
        axis="Z", parent=ch, verts=28)
    cyl("lidar_win", 0.0158, 0.010, (-0.012, 0.043, 0.100), mat_lens,
        axis="Z", parent=ch, verts=28)
    # 琥珀警示灯 (后顶): 底座 + 发光圆顶
    cyl("beacon_base", 0.011, 0.006, (-0.155, 0, 0.103), mat_base, axis="Z",
        parent=ch, verts=18)
    sphere("beacon", 0.0095, (-0.155, 0, 0.111), mat_beacon, parent=ch)
    # 急停: 黄方板 + 红蘑菇 (左侧后部)
    box("estop_plate", (0.016, 0.002, 0.016), (-0.135, -0.0645, 0.072),
        mat_yellow_plate, bevel=0.001, parent=ch)
    cyl("estop", 0.0055, 0.005, (-0.135, -0.0675, 0.072), mat_ered,
        axis="Y", parent=ch, verts=16)
    # 侧面散热格栅 (深色凹槽两条)
    for sgn in (-1, 1):
        box(f"vent_{sgn}", (0.055, 0.002, 0.012), (-0.115, sgn * 0.0645, 0.062),
            mat_base, bevel=0.001, parent=ch)
    # 顶盖检修缝
    box("hatch", (0.070, 0.085, 0.002), (-0.130, 0, 0.1005), mat_base,
        bevel=0.004, parent=ch)

    # 轮胎: 前驱动轮 (带轮毂罩) + 后从动轮
    for wx, wy, wr in ((0.0, -0.06, WHEEL_R), (0.0, 0.06, WHEEL_R),
                       (-0.16, -0.055, 0.018), (-0.16, 0.055, 0.018)):
        cyl(f"tire_{wx}_{wy}", wr, 0.017, (wx, wy, wr), mat_rubber,
            bevel=0.004, parent=ch)
        cyl(f"hub_{wx}_{wy}", wr * 0.55, 0.019, (wx, wy, wr), mat_base,
            bevel=0.002, parent=ch)
        cyl(f"hubcap_{wx}_{wy}", wr * 0.30, 0.021, (wx, wy, wr), mat_chrome,
            bevel=0.001, parent=ch)

    # 门架 (视觉件, 挂 chassis): 双 C 外轨 + 内轨 + 上下横梁 + 油缸 + 链条
    for wy in (-0.048, 0.048):
        box(f"mast_outer_{wy}", (0.016, 0.012, MAST_H), (0.030, wy, MAST_H / 2),
            mat_dark, bevel=0.0008, parent=ch)
        box(f"mast_inner_{wy}", (0.010, 0.010, MAST_H - 0.02),
            (0.045, wy, (MAST_H - 0.02) / 2), mat_steel, bevel=0.0006,
            parent=ch)
    box("mast_cross_b", (0.02, 0.108, 0.012), (0.032, 0, 0.012), mat_dark,
        bevel=0.0008, parent=ch)
    box("mast_cross_t", (0.02, 0.108, 0.010), (0.032, 0, MAST_H - 0.006),
        mat_dark, bevel=0.0008, parent=ch)
    # 升降油缸: 缸筒(暗) + 镀铬活塞杆, 门架中央
    cyl("cyl_barrel", 0.006, 0.24, (0.024, 0, 0.135), mat_base, axis="Z",
        parent=ch, verts=16)
    cyl("cyl_rod", 0.0032, 0.20, (0.024, 0, 0.34), mat_chrome, axis="Z",
        parent=ch, verts=12)
    # 链条 (两侧细带) + 顶部油管环
    for wy in (-0.038, 0.038):
        box(f"chain_{wy}", (0.003, 0.006, MAST_H - 0.03), (0.036, wy,
            (MAST_H - 0.03) / 2 + 0.01), mat_mesh, bevel=0, parent=ch)
    box("hose", (0.004, 0.06, 0.004), (0.020, 0, MAST_H - 0.012), mat_base,
        bevel=0.0015, parent=ch)
    # 门架顶警示灯 (小, 琥珀)
    sphere("beacon_mast", 0.006, (0.030, 0, MAST_H + 0.004), mat_beacon,
           parent=ch)

    # ── 货叉架 (carriage empty 驱动) ──
    cr = empty("carriage", (0, 0, 0))
    # 调叉器 (侧移箱) + 背架 (暗钢) + 橙色叉梁 + 钩臂货叉
    box("sideshift", (0.010, 0.070, 0.030), (0.020, 0, 0.040), mat_base,
        bevel=0.002, parent=cr)
    box("carriage_plate", (0.008, 0.088, 0.060), (0.028, 0, 0.040),
        mat_dark, bevel=0.001, parent=cr)
    box("fork_bar", (0.010, 0.088, 0.012), (0.036, 0, 0.058), mat_orange,
        bevel=0.0015, parent=cr)
    for wy in (-FORK_YS, FORK_YS):
        # 钩臂: 竖柄贴叉梁 (x 0.032..0.040), 上钩过叉梁顶
        box(f"fork_shank_{wy}", (0.008, FORK_W, 0.052),
            (0.036, wy, FORK_TOPZ - FORK_T / 2 - 0.024), mat_orange,
            bevel=0.0008, parent=cr)
        # blade 严格物理包络 x∈[0.040,0.120], 顶面 z=0.065
        box(f"fork_blade_{wy}", (FORK_LEN, FORK_W, FORK_T),
            (FORK_ROOT + FORK_LEN / 2, wy, FORK_TOPZ - FORK_T / 2),
            mat_orange, bevel=0.0004, parent=cr)
        # 叉尖下倾倒角 (进叉引导): 前端薄楔
        box(f"fork_tip_{wy}", (0.012, FORK_W, 0.0012),
            (FORK_ROOT + FORK_LEN - 0.006, wy, FORK_TOPZ - FORK_T - 0.0004),
            mat_orange, bevel=0, parent=cr)
    # 载货护栏 (backrest, 随叉架升)
    box("backrest_l", (0.005, 0.003, 0.10), (0.040, -0.042, 0.115), mat_steel,
        bevel=0.0008, parent=cr)
    box("backrest_r", (0.005, 0.003, 0.10), (0.040, 0.042, 0.115), mat_steel,
        bevel=0.0008, parent=cr)
    box("backrest_t", (0.005, 0.087, 0.004), (0.040, 0, 0.165), mat_steel,
        bevel=0.0008, parent=cr)

    # ── 托盘 (pallet empty 驱动): 面板+纵梁+底垫, 包络 80x60x21 ──
    pa = empty("pallet", (0, 0, 0))
    deck_t = 0.003
    n_deck = 5
    for i in range(n_deck):
        yy = -PAL_LY / 2 + PAL_LY / (n_deck - 1) * i
        box(f"pal_deck_{i}", (PAL_LX, PAL_LY / 5 - 0.003, deck_t),
            (0, yy, PAL_HZ - deck_t / 2), mat_wood, bevel=0.0002, parent=pa)
    for y in (-0.027, 0.0, 0.027):
        box(f"pal_stringer_{y}", (PAL_LX - 0.004, 0.010, 0.014),
            (0, y, 0.011), mat_wood, bevel=0.0004, parent=pa)
        box(f"pal_runner_{y}", (PAL_LX, 0.006, 0.004), (0, y, 0.002),
            mat_wood, bevel=0.0003, parent=pa)
        # 进叉端倒角 (前端下缘楔)
        box(f"pal_lead_{y}", (0.006, 0.006, 0.002),
            (PAL_LX / 2 - 0.002, y, 0.005), mat_wood, bevel=0, parent=pa)

    # ── 货物: 两层纸箱 (轻微歪斜+封箱带+面单) ──
    z0 = PAL_HZ
    box("carton_a", (0.074, 0.054, 0.030), (0, 0, z0 + 0.015), mat_card,
        bevel=0.0018, parent=pa)
    box("carton_b", (0.068, 0.048, 0.028), (0.002, -0.001, z0 + 0.044),
        mat_card, bevel=0.0018, parent=pa)
    box("tape_a", (0.075, 0.008, 0.0005), (0, 0, z0 + 0.0302), mat_label,
        bevel=0, parent=pa)
    box("tape_b", (0.069, 0.008, 0.0005), (0.002, -0.001, z0 + 0.0582),
        mat_label, bevel=0, parent=pa)
    box("label_a", (0.018, 0.0004, 0.012), (0.010, -0.0271, z0 + 0.016),
        mat_label, bevel=0, parent=pa)
    box("label_b", (0.016, 0.0004, 0.010), (0.012, -0.0241, z0 + 0.045),
        mat_label, bevel=0, parent=pa)
    # 条码 (面单上的深色细条)
    for i in range(6):
        box(f"bc_a_{i}", (0.0006 + (i % 3) * 0.0004, 0.0002, 0.007),
            (0.004 + i * 0.0016, -0.0274, z0 + 0.014), mat_base, bevel=0,
            parent=pa)

    # ── 背景货架 (虚化纵深, 无物理) ──
    for tag, by in (("bg1", 0.62), ("bg2", 1.15)):
        build_rack(tag, -0.15, by)
    build_rack("bg3", -0.75, -0.75)

    # ── 位姿驱动 (与 photoreal 完全一致) ──
    poses = load_poses()
    step = int(arg("--step", "3") or 3)
    frames = poses[::step]
    for ob, key in ((ch, "chassis"), (cr, "carriage"), (pa, "pallet")):
        ob.rotation_mode = "QUATERNION"
        for i, p in enumerate(frames):
            f = i + 1
            ob.location = (p[f"{key}_x"], p[f"{key}_y"], p[f"{key}_z"])
            ob.rotation_quaternion = (p[f"{key}_qw"], p[f"{key}_qx"],
                                      p[f"{key}_qy"], p[f"{key}_qz"])
            ob.keyframe_insert("location", frame=f)
            ob.keyframe_insert("rotation_quaternion", frame=f)

    # 油缸活塞杆: 底端固定 0.245, 顶端随叉架 (carriage_z + 0.07)
    rod = bpy.data.objects["cyl_rod"]
    for i, p in enumerate(frames):
        f = i + 1
        top = max(0.275, p["carriage_z"] + 0.070)
        rod.location = (0.024, 0, (0.245 + top) / 2)
        rod.scale = (1.0, 1.0, max(0.05, (top - 0.245) / 0.20))
        rod.keyframe_insert("location", frame=f)
        rod.keyframe_insert("scale", frame=f)

    # ── 相机: 更电影感 (长焦轻压缩 + f2.8) ──
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 45
    cam_data.clip_start = 0.005
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 2.8
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    focus = empty("focus", (0, 0, 0))
    look = [-0.18, 0.0, 0.22]
    for i, p in enumerate(frames):
        f = i + 1
        tgt = [(p["chassis_x"] + p["pallet_x"]) / 2 + 0.08, 0.0,
               max(0.22, min(0.42, (p["carriage_z"] + 0.066) * 0.7))]
        look = [lk + 0.15 * (t - lk) for lk, t in zip(look, tgt)]
        focus.location = look
        focus.keyframe_insert("location", frame=f)
        cam.location = (look[0] + 0.85, look[1] - 0.95, look[2] + 0.55)
        cam.keyframe_insert("location", frame=f)
    con = cam.constraints.new(type="TRACK_TO")
    con.target = focus
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    cam_data.dof.focus_object = focus

    # ── 布光: 仓库顶棚灯感 (主光+冷补+轮廓) ──
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(f"{PBR}/studio_small_09_2k.hdr")
    world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.20

    def area(name, loc, rot, size, energy, color=(1, 1, 1)):
        d = bpy.data.lights.new(name, "AREA")
        d.size, d.energy, d.color = size, energy, color
        o = bpy.data.objects.new(name, d)
        o.location, o.rotation_euler = loc, rot
        scene.collection.objects.link(o)

    area("key", (0.5, -0.6, 1.0), (math.radians(35), 0, math.radians(40)),
         0.8, 30, (1.0, 0.96, 0.90))
    area("fill", (-0.8, 0.5, 0.7), (math.radians(-45), 0, math.radians(-120)),
         0.6, 9, (0.85, 0.9, 1.0))
    area("rim", (-0.4, 0.9, 0.6), (math.radians(50), 0, math.radians(160)),
         0.5, 14, (0.95, 0.97, 1.0))

    scene.frame_start = 1
    scene.frame_end = len(frames)

    tf = arg("--frame")
    outdir = arg("--video")
    if tf:
        scene.frame_set(int(tf))
        os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
        scene.render.filepath = f"{os.path.join(ROOT, 'out')}/agv_frame_{tf}.png"
        bpy.ops.render.render(write_still=True)
        print("TEST FRAME ->", scene.render.filepath)
    elif outdir:
        os.makedirs(outdir, exist_ok=True)
        scene.render.filepath = f"{outdir}/f_"
        scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=True)
        print("VIDEO FRAMES ->", outdir)
    else:
        raise SystemExit("need --frame N or --video DIR")


main()
