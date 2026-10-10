"""照片级渲染 T04 取放闭环 — Blender 5.2 Cycles(Metal) + ambientCG PBR。

运动真值 = assets/pbr/poses.csv (MuJoCo mj_step 逐帧导出, 与测试同源);
本脚本只是"贴皮": 每台设备按**真实世界设备的等比缩小模型**建模 ——
木托盘(面板/横梁/进叉倒角)、C型钢货架+橙色step beam、平衡重叉车
(护顶架/座椅/方向盘/双C门架/L型货叉/轮胎)、纸箱货物 —— 但外形包络
严格等于 configs/design.yaml 的碰撞尺寸, 否则货叉与叉孔会穿帮。

用法 (headless):
    blender -b -P scripts/render_photoreal.py -- --frame 4500          # 测试帧
    blender -b -P scripts/render_photoreal.py -- --video out/pbr-frames --step 8
资产: assets/pbr/{Concrete028,PaintedMetal001,MetalPlates006,Wood096}/, studio HDRI。
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


def arg(flag: str, default: str | None = None) -> str | None:
    argv = sys_argv_after()
    return argv[argv.index(flag) + 1] if flag in argv else default


def sys_argv_after():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


import sys  # noqa: E402  (arg 需要, 放后面保持上方常量可读)

# ── 材质: ambientCG 三图 (Color/NormalGL/Roughness) + 平涂/乘法染色 ────
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
    # 8/9 是 ROTATION 端口, 接错不报错但颜色链路静默断 (实测踩过)。
    if tint is not None:
        # 工业喷漆用平涂 (PaintedMetal001 原贴图是琥珀黄, MULTIPLY 染不出蓝)
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


def box(name, size3, loc, mat, bevel=0.0012, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=2)
    o = bpy.context.object
    o.name = name
    o.scale = (size3[0] / 2, size3[1] / 2, size3[2] / 2)
    o.location = loc
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        b = o.modifiers.new("bevel", "BEVEL")
        b.width, b.segments, b.limit_method = bevel, 2, "ANGLE"
    o.data.materials.append(mat)
    o["sem_label"] = name
    if parent:
        o.parent = parent
    return o


def cyl(name, r, depth, loc, mat, axis="Y", bevel=0.0, parent=None):
    rot = {"Y": (math.pi / 2, 0, 0), "X": (0, math.pi / 2, 0),
           "Z": (0, 0, 0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth,
                                        location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    if bevel:
        b = o.modifiers.new("bevel", "BEVEL")
        b.width, b.segments = bevel, 3
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

    # ── 材质 ──
    mat_conc = pbr_mat("floor_concrete", "Concrete028", scale=8.0, rough_mul=0.95)
    mat_orange = pbr_mat("rack_beam", "PaintedMetal001",
                         tint=(0.60, 0.13, 0.01, 1), scale=30.0,
                         rough_mul=0.45, wear=1)
    mat_blue = pbr_mat("rack_upright", "PaintedMetal001",
                       tint=(0.03, 0.08, 0.26, 1), scale=25.0, rough_mul=0.5,
                       wear=1)
    mat_yellow = pbr_mat("vehicle_paint", "PaintedMetal001",
                         tint=(0.85, 0.55, 0.02, 1), scale=40.0, rough_mul=0.4)
    mat_steel = pbr_mat("fork_steel", "MetalPlates006", scale=60.0,
                        metallic=1.0, rough_mul=0.35)
    mat_dark = pbr_mat("machinery_dark", "MetalPlates006",
                       tint=(0.05, 0.05, 0.06, 1), scale=50.0,
                       metallic=0.9, rough_mul=0.55)
    mat_wood = pbr_mat("pallet_wood", "Wood096", tint=(0.72, 0.48, 0.28, 1),
                       scale=30.0, rough_mul=0.8, tint_mode="mul")
    mat_rubber = flat_mat("tire", (0.018, 0.018, 0.02, 1), rough=0.85)
    mat_hub = flat_mat("hub", (0.35, 0.35, 0.37, 1), rough=0.4, metallic=0.8)
    mat_plastic = flat_mat("black_plastic", (0.02, 0.02, 0.022, 1), rough=0.45)
    mat_card = flat_mat("cardboard", (0.45, 0.30, 0.16, 1), rough=0.8)
    mat_tape = flat_mat("tape", (0.55, 0.42, 0.25, 1), rough=0.35)
    mat_label = flat_mat("label", (0.9, 0.9, 0.87, 1), rough=0.6)

    # ── 地面 ──
    bpy.ops.mesh.primitive_plane_add(size=8)
    fl = bpy.context.object
    fl.name = "floor"
    fl.data.materials.append(mat_conc)
    fl["sem_label"] = "floor"

    # ── 货架: C型钢立柱(带孔板感) + 前/后 step beam + 中间托条 + 底板 ──
    H = SUPPORT_Z + 0.12
    for sx in (1, -1):
        for sy in (-1, 1):
            x, y = sx * 0.20, sy * 0.030
            box(f"upr_{sx}_{sy}", (0.012, 0.012, H), (x, y, H / 2),
                mat_blue, bevel=0.0008)
            # 立柱底脚板
            box(f"foot_{sx}_{sy}", (0.030, 0.030, 0.003), (x, y, 0.0015),
                mat_dark, bevel=0.0006)
        # 横向斜撑 (两层)
        box(f"brace_x_{sx}_lo", (0.004, 0.060, 0.004),
            (sx * 0.20, 0, 0.10), mat_blue, bevel=0.0006)
        box(f"brace_x_{sx}_hi", (0.004, 0.060, 0.004),
            (sx * 0.20, 0, SUPPORT_Z - 0.06), mat_blue, bevel=0.0006)
    # step beam: 前后两根 (托盘底条 y=±27 落在上面), 顶面 = SUPPORT_Z
    for y in (-0.027, 0.027):
        box(f"beam_{y:.3f}", (0.40, 0.014, 0.006), (0, y, SUPPORT_Z - 0.003),
            mat_orange, bevel=0.0008)
    box("beam_mid", (0.40, 0.010, 0.004), (0, 0, SUPPORT_Z - 0.002),
        mat_orange, bevel=0.0006)

    # ── 暂存站: 钢台面 + 4 腿 + 下横撑 ──
    box("stage_top", (STAGE_HX * 2, STAGE_HY * 2, 0.004),
        (STAGE_CX, 0, STAGE_TOPZ - 0.002), mat_steel)
    for lx in (STAGE_CX - 0.052, STAGE_CX + 0.052):
        for ly in (-0.042, 0.042):
            box(f"stage_leg_{lx:.2f}_{ly:.0f}", (0.006, 0.006, 0.041),
                (lx, ly, 0.0205), mat_dark, bevel=0.0006)
    box("stage_brace", (0.104, 0.005, 0.005), (STAGE_CX, 0, 0.012),
        mat_dark, bevel=0.0006)

    # ── 叉车 (chassis empty 驱动; 包络=设计值) ──
    ch = empty("forklift", (0, 0, 0))
    # 主体: 前段机罩 + 座舱段 + 后配重 (阶梯轮廓)
    box("body_hood", (0.10, 0.13, 0.055), (-0.030, 0, 0.0475), mat_yellow)
    box("body_mid", (0.09, 0.14, 0.075), (-0.115, 0, 0.0575), mat_yellow)
    box("cw", (0.045, 0.12, 0.085), (-0.1725, 0, 0.0625), mat_dark)
    # 护顶架: 4 立柱 + 顶框
    for gx in (-0.16, -0.02):
        for gy in (-0.062, 0.062):
            box(f"ohg_post_{gx}_{gy}", (0.006, 0.006, 0.185),
                (gx, gy, 0.095 + 0.055), mat_dark, bevel=0.0008)
    box("ohg_rail_f", (0.008, 0.132, 0.006), (-0.02, 0, 0.285), mat_dark, bevel=0.0008)
    box("ohg_rail_b", (0.008, 0.132, 0.006), (-0.16, 0, 0.285), mat_dark, bevel=0.0008)
    box("ohg_rail_l", (0.148, 0.008, 0.006), (-0.09, -0.062, 0.285), mat_dark, bevel=0.0008)
    box("ohg_rail_r", (0.148, 0.008, 0.006), (-0.09, 0.062, 0.285), mat_dark, bevel=0.0008)
    box("ohg_roof", (0.15, 0.135, 0.004), (-0.09, 0, 0.290), mat_dark, bevel=0.001)
    box("seat_pan", (0.045, 0.05, 0.012), (-0.125, 0, 0.098), mat_plastic, bevel=0.002)
    box("seat_back", (0.012, 0.05, 0.055), (-0.150, 0, 0.130), mat_plastic, bevel=0.002)
    cyl("wheel_st", 0.018, 0.004, (-0.045, 0, 0.145), mat_plastic, axis="X",
        parent=None)
    # 轮胎: 前驱动轮 r25, 后轮 r18 (设计 wheel_radius 0.025)
    for wx, wy, wr in ((0.0, -0.06, WHEEL_R), (0.0, 0.06, WHEEL_R),
                       (-0.16, -0.055, 0.018), (-0.16, 0.055, 0.018)):
        cyl(f"tire_{wx}_{wy}", wr, 0.016, (wx, wy, wr), mat_rubber, bevel=0.004)
        cyl(f"hub_{wx}_{wy}", wr * 0.45, 0.018, (wx, wy, wr), mat_hub)
    for wy in (-0.048, 0.048):
        box(f"mast_outer_{wy}", (0.016, 0.012, MAST_H), (0.030, wy, MAST_H / 2),
            mat_dark, bevel=0.0008)
        box(f"mast_inner_{wy}", (0.010, 0.010, MAST_H - 0.02), (0.045, wy, (MAST_H - 0.02) / 2),
            mat_steel, bevel=0.0006)
    box("mast_cross_b", (0.02, 0.108, 0.012), (0.032, 0, 0.012), mat_dark, bevel=0.0008)
    box("mast_cross_t", (0.02, 0.108, 0.010), (0.032, 0, MAST_H - 0.006), mat_dark, bevel=0.0008)
    # 所有叉车件挂到 chassis empty
    for o in bpy.data.objects:
        if o.type == "MESH" and (o.name.startswith(("body_", "cw", "ohg_", "seat_",
                                                    "wheel_st", "tire_", "hub_",
                                                    "mast_"))):
            o.parent = ch

    # 货叉架 (carriage empty 驱动): 挂板 + 叉梁 + 2 把 L 型货叉
    cr = empty("carriage", (0, 0, 0))
    # 挂板/叉梁贴门架内侧; 叉 blade 严格取物理包络 x 0.040..0.120
    box("carriage_plate", (0.008, 0.088, 0.055), (0.028, 0, 0.040),
        mat_steel, bevel=0.001, parent=cr)
    box("fork_bar", (0.012, 0.070, 0.010), (0.036, 0, 0.012),
        mat_steel, bevel=0.001, parent=cr)
    for wy in (-FORK_YS, FORK_YS):
        # L 型: 竖直柄 + 水平 blade (顶面 = carriage_z + 0.065 = 物理叉顶)
        box(f"fork_shank_{wy}", (0.008, FORK_W, 0.052),
            (0.044, wy, 0.036), mat_steel, bevel=0.0006, parent=cr)
        box(f"fork_blade_{wy}", (FORK_LEN, FORK_W, FORK_T),
            (FORK_ROOT + FORK_LEN / 2, wy, FORK_TOPZ - FORK_T / 2),
            mat_steel, bevel=0.0004, parent=cr)

    # ── 托盘 (pallet empty 驱动): 5 面板 + 3 纵梁 + 3 底垫, 包络 80x60x21 ──
    pa = empty("pallet", (0, 0, 0))
    deck_t = 0.003
    n_deck = 5
    for i in range(n_deck):
        yy = -PAL_LY / 2 + (i + 0.5) * PAL_LY / n_deck
        box(f"pal_deck_{i}", (PAL_LX, PAL_LY / n_deck - 0.003, deck_t),
            (0, yy, PAL_HZ - deck_t / 2), mat_wood, bevel=0.0003, parent=pa)
    # 纵梁 z 4..18: 位于 y=±27 与 0, 完全让开物理叉道 y=±8..22
    for y in (-0.027, 0.0, 0.027):
        box(f"pal_stringer_{y}", (PAL_LX - 0.004, 0.010, 0.014),
            (0, y, 0.011), mat_wood, bevel=0.0004, parent=pa)
        box(f"pal_runner_{y}", (PAL_LX, 0.006, 0.004), (0, y, 0.002),
            mat_wood, bevel=0.0003, parent=pa)
    # ── 货物: 两层纸箱 (设计 60x40x50 的等比堆码, 随托盘动) ──
    z0 = PAL_HZ
    box("carton_a", (0.074, 0.054, 0.030), (0, 0, z0 + 0.015), mat_card,
        bevel=0.0015, parent=pa)
    box("carton_b", (0.068, 0.048, 0.028), (0, 0, z0 + 0.044), mat_card,
        bevel=0.0015, parent=pa)
    box("tape_a", (0.075, 0.008, 0.0005), (0, 0, z0 + 0.0302), mat_tape,
        bevel=0, parent=pa)
    box("tape_b", (0.069, 0.008, 0.0005), (0, 0, z0 + 0.0582), mat_tape,
        bevel=0, parent=pa)
    box("label_a", (0.024, 0.0005, 0.016), (0.010, -0.0272, z0 + 0.014),
        mat_label, bevel=0, parent=pa)

    # ── 位姿动画 ──
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
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

    # ── 相机跟随 ──
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 40
    cam_data.clip_start = 0.005
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 4.0
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

    # ── 布光 ──
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(f"{PBR}/studio_small_09_2k.hdr")
    world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.22

    def area(name, loc, rot, size, energy, color=(1, 1, 1)):
        d = bpy.data.lights.new(name, "AREA")
        d.size, d.energy, d.color = size, energy, color
        o = bpy.data.objects.new(name, d)
        o.location, o.rotation_euler = loc, rot
        scene.collection.objects.link(o)

    area("key", (0.5, -0.6, 1.0), (math.radians(35), 0, math.radians(40)),
         0.8, 25, (1.0, 0.95, 0.88))
    area("fill", (-0.8, 0.5, 0.7), (math.radians(-45), 0, math.radians(-120)),
         0.6, 8, (0.85, 0.9, 1.0))

    scene.frame_start = 1
    scene.frame_end = len(frames)

    tf = arg("--frame")
    outdir = arg("--video")
    if tf:
        scene.frame_set(int(tf))
        os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
        scene.render.filepath = f"{os.path.join(ROOT, 'out')}/test_frame_{tf}.png"
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
