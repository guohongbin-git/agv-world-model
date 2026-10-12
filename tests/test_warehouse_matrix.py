"""M2 Wave1 — 128 架矩阵仓库物理层验收 (384 货位 / 3 层 / 混合车队)。

对应验收:
* 矩阵几何: 16 组 × 8 架 × 3 层 = 384 货位, 命名 R<r><c>-S<0..7>-L<0..2>,
  间距/通道净宽 (≥ 车辆包络 + 2×50mm) 程序化校验;
* 地面标志线: 程序化生成 (通道虚线/组轮廓/交接区/停车点), 改一个布局参数
  线全场联动 (col_aisle_x +0.20 → 虚线 x 平移 +0.20);
* AGV 潜伏顶升单往返: 交接台对中钻入 → 顶升载运 → 目标台降销放托,
  终态支撑成立 (≥85%), 穿透 < 5e-4;
* 混合交接链: 叉车第三层取架 → 放交接台 → AGV 接管 → 干线运输 → 放托;
* 4 车一个 MJCF 共存, 各自指令独立响应, 无名冲突;
* 旧 ForkliftCell 零破坏 (导入/构造回归)。
"""
from __future__ import annotations

import math

import mujoco
import pytest

from sim.controllers import Ramp
from sim.warehouse_cell import (FORKLIFT_ENVELOPE, SAFETY_MARGIN,
                                FleetController, MatrixLayout, WarehouseCell,
                                parse_slot, pallet_name_for)

PEN_LIMIT = 5e-4
CHAIN_SLOT = "R11-S0-L2"          # 第三层 (0.400) 取货 —— 顶层插叉位
CHAIN_STAND = "H02"
CHAIN_DOCK = "D_G12"


# ---------------------------------------------------------------- 布局/几何
def test_384_slots_geometry():
    L = MatrixLayout()
    slots = L.slot_ids()
    assert len(slots) == 16 * 8 * 3 == 384
    assert len(set(slots)) == 384
    # 命名契约: R<r><c>-S<0..7>-L<0..2>
    assert {s.split("-")[0] for s in slots} == {
        f"R{r}{c}" for r in range(1, 5) for c in range(1, 5)}
    for s in slots:
        assert 0 <= int(s[-1]) <= 2 and 0 <= int(s[5]) <= 7
    # 层高派生: 0.100/0.250/0.400 (同 forklift_cell levels)
    for l, z in enumerate((0.100, 0.250, 0.400)):
        assert L.slot_pose(f"R22-S3-L{l}")[3] == pytest.approx(z)
        assert L.slot_pose(f"R22-S6-L{l}")[3] == pytest.approx(z)
    # 前/后排 y 与列内架距
    x, yf, yaw, _ = L.slot_pose("R22-S0-L1")
    _, yb, _, _ = L.slot_pose("R22-S4-L1")
    cx, cy = L.group_center(2, 2)
    assert (x, yf, yb) == pytest.approx(
        (cx + (0 - 1.5) * 0.16, cy - 0.26, cy + 0.26))
    assert yaw == pytest.approx(math.pi / 2)   # 托盘通道轴 = 世界 ±y
    xs = [L.slot_pose(f"R22-S{k}-L1")[0] for k in range(4)]
    assert all(b - a == pytest.approx(0.16) for a, b in zip(xs, xs[1:]))


def test_aisle_clearance_envelope():
    """通道净宽 ≥ 车辆包络 + 2× 安全余量 (规格红线, 布局构造时强制)。"""
    L = MatrixLayout()
    need = max(FORKLIFT_ENVELOPE) + 2 * SAFETY_MARGIN
    assert min(L.p["col_aisle_x"], L.p["row_aisle_y"]) >= need
    assert min(L.col_aisle_xs()[i + 1] - L.col_aisle_xs()[i] - L.group_w
               for i in range(4)) >= need - 1e-9
    assert min(L.row_aisle_ys()[i + 1] - L.row_aisle_ys()[i] - L.group_h
               for i in range(4)) >= need - 1e-9
    # 通道不足必须拒绝构造 (红线程序化)
    with pytest.raises(ValueError, match="包络"):
        MatrixLayout(col_aisle_x=0.40)


def test_field_extent_about_5x4():
    L = MatrixLayout()
    # 场地约 5m × 4.5m (矩阵 3.94×3.88 + 东西边缘通道 + 南北泊位带)
    assert 4.6 <= L.field_w <= 5.4
    assert 4.2 <= L.field_h <= 5.0


# ---------------------------------------------------------------- 场景构建
@pytest.fixture(scope="module")
def cell():
    c = WarehouseCell(timestep_s=0.001, seed=7)
    c.warmup(0.5)
    return c


def test_matrix_scene_build(cell):
    m, L = cell.model, cell.layout
    # 128 架 × (3 层 × 2 梁 + 2 立柱) + 台 (交接+交付) × 6
    # + 车 (叉 9/台 + AGV 4/台) + 48 托盘 × 9 + 地面 + 标志线
    n_stands = (len(cell.layout.p["handoff_ids"])
                + cell.layout.p["group_rows"] * cell.layout.p["group_cols"])
    assert m.ngeom == (128 * 8 + n_stands * 6 + len(cell.forklifts) * 9
                       + len(cell.agvs) * 4 + 48 * 9 + 1
                       + len(cell.mark_names))
    # MJCF 编译零错已隐含; 每架每层双梁落位 (梁局部 x=-0.027, 架体世界位姿):
    for slot in ("R11-S0-L2", "R44-S7-L0"):
        x, y, _, z = L.slot_pose(slot)
        r, c, k, l = parse_slot(slot)
        gid = mujoco_name(m, f"rack_R{r}{c}_S{k}_beam_w_L{z:.3f}")
        bid = m.geom_bodyid[gid]
        assert m.body_pos[bid][0] == pytest.approx(x)
        assert m.geom_pos[gid][0] == pytest.approx(-0.027)
        assert m.body_pos[bid][1] == pytest.approx(y)
    # 播种 48 托 (384 的 12.5%, 规格带 10-15%), 跨层分布
    assert len(cell.pallets) == 48
    n_rack = sum(1 for n in cell.pallets if n.startswith("pallet_R"))
    assert n_rack == 47 and "pallet_H01" in cell.pallets
    levels = {n[-1] for n in cell.pallets if n.startswith("pallet_R")}
    assert levels == {"0", "1", "2"}
    # 播种托盘全部落稳 (支撑面 +1mm 沉降后 ≤2mm)
    for n in cell.pallets:
        z = cell.pallet_bottom_z(n)
        if n.startswith("pallet_R"):
            parts = n.split("_")             # pallet_R11_S0_L2
            expect = L.slot_pose(f"{parts[1]}-{parts[2]}-{parts[3]}")[3] + 0.001
            assert abs(z - expect) < 0.002, n
        else:
            assert abs(z - (L.p["stand_top_z"] + 0.001)) < 0.002, n
    # 初始全场无穿透
    assert cell.max_penetration() < PEN_LIMIT


def mujoco_name(m, name):
    import mujoco
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert i >= 0, name
    return i


# ---------------------------------------------------------------- 标志线
def test_ground_markings_procedural(cell):
    L = cell.layout
    marks = cell.mark_names
    # 四类齐全: 通道虚线 (黄) / 组轮廓 (白) / 交接区 (蓝) / 停车点
    assert sum(n.startswith("mark_aisle_col") for n in marks) >= 100
    assert sum(n.startswith("mark_aisle_row") for n in marks) >= 100
    for g in (f"mark_group_G{r}{c}_n" for r in range(1, 5)
              for c in range(1, 5)):
        assert g in marks
    assert f"mark_handoff_H01_n" in marks
    assert f"mark_dock_D_G11_n" in marks
    assert f"mark_park_F01_n" in marks
    # 纯视觉 (不参与接触) 且贴地
    m = cell.model
    for n in marks[:50]:
        g = mujoco_name(m, n)
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
        assert m.geom_pos[g][2] < 0.01


def test_markings_follow_layout_param():
    """改一个布局参数, 标志线跟着变 (程序化单源证据)。

    取 col_aisle_x 0.46→0.66: 矩阵对称加宽 —— 中列通道 (index 2) 恒在 0;
    通道/组心的世界位移由**派生布局**给出 (矩阵重新对中, 非线性平移),
    标志线/货架/交付台必须与派生值一致; 交接台 x 是显式布局参数, 不随动。
    """
    base = WarehouseCell(timestep_s=0.001, seed=7)
    shifted = WarehouseCell(timestep_s=0.001, seed=7,
                            layout=MatrixLayout(col_aisle_x=0.66))
    Lb, Ls = base.layout, shifted.layout
    mb, ms = base.model, shifted.model
    # 通道虚线跟随派生通道中心
    expect = Ls.col_aisle_xs()[3] - Lb.col_aisle_xs()[3]
    dash_b = mujoco_name(mb, "mark_aisle_col_3_d010")
    dash_s = mujoco_name(ms, "mark_aisle_col_3_d010")
    assert ms.geom_pos[dash_s][0] - mb.geom_pos[dash_b][0] == pytest.approx(expect)
    # 货架架体/交付台蓝框跟随派生组心 (c=3)
    expect = Ls.group_center(1, 3)[0] - Lb.group_center(1, 3)[0]
    beam_b = mb.body(mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_BODY,
                                       "rack_R13_S0")).pos[0]
    beam_s = ms.body(mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_BODY,
                                       "rack_R13_S0")).pos[0]
    assert beam_s - beam_b == pytest.approx(expect)
    hb = mujoco_name(mb, "mark_dock_D_G13_n")
    hs = mujoco_name(ms, "mark_dock_D_G13_n")
    assert ms.geom_pos[hs][0] - mb.geom_pos[hb][0] == pytest.approx(expect)
    # 交接台 x 为显式布局参数: 不随 col_aisle_x 平移
    fb = mujoco_name(mb, "mark_handoff_H04_n")
    fs = mujoco_name(ms, "mark_handoff_H04_n")
    assert ms.geom_pos[fs][0] - mb.geom_pos[fb][0] == pytest.approx(0.0)
    assert base.max_penetration() < PEN_LIMIT
    assert shifted.max_penetration() < PEN_LIMIT


# ---------------------------------------------------------------- AGV 往返
@pytest.fixture(scope="module")
def agv_result():
    """AGV 单往返: H01 预载托盘 → 对中钻入 → 顶升 → 运至 D_G11 → 降销放托。"""
    cell = WarehouseCell(timestep_s=0.001, seed=7)
    ctrl = FleetController(cell)
    cell.warmup(0.5)
    pn, agv = "pallet_H01", cell.agv("A01")
    carried = {}
    a1 = ctrl.run_sequence(cell.plan_agv_dock_lift("A01", "H01"),
                           settle_s=0.3)
    carried["z"] = cell.pallet_bottom_z(pn)
    carried["xy_off"] = (abs(cell.pallet_pose(pn)[0][0] - agv.xy()[0]),
                         abs(cell.pallet_pose(pn)[0][1] - agv.xy()[1]))
    carried["pin_ratio"] = cell.surface_load_ratio(pn, {agv.pin_gid})
    carried["agv_xy"] = agv.xy()
    a2 = ctrl.run_sequence(cell.plan_agv_transport("A01", "H01", "D_G11"),
                           settle_s=0.3)
    a3 = ctrl.run_sequence(
        cell.plan_agv_drop("A01", "D_G11",
                           corridor_x=cell.layout.stand_pose("H01")[0]),
        settle_s=0.5)
    pos = cell.pallet_pose(pn)[0]
    return dict(cell=cell, audits=(a1, a2, a3), carried=carried,
                final=dict(z=cell.pallet_bottom_z(pn),
                           x=pos[0], y=pos[1],
                           yaw=cell.pallet_yaw(pn),
                           ratio=cell.surface_load_ratio(
                               pn, cell.stand_gids["D_G11"]),
                           pin_contact=cell.pallet_touches(
                               pn, {agv.pin_gid})))


def test_agv_dock_alignment(agv_result):
    """对中钻入 ±10mm; 顶升后托盘由销盘承载 (≥85%) 且随车。"""
    c = agv_result["carried"]
    assert c["z"] > 0.118, f"托盘未离台: z={c['z']:.4f}"
    assert max(c["xy_off"]) <= 0.010, f"对中偏差 {c['xy_off']}"
    assert c["pin_ratio"] >= 0.85, f"销盘承载 {c['pin_ratio']:.2f}"
    assert agv_result["audits"][0]["violations"] == []


def test_agv_final_support(agv_result):
    """终态: 落在目标台条带上 (0.095±1.5mm), 对位 ±4mm, 支撑 ≥85%, 脱销。

    D_G11 航向 0° (开口沿 +x), AGV 载运中原地转 90° → 托盘终态 yaw≈0。
    """
    f = agv_result["final"]
    tx, ty, _ = agv_result["cell"].layout.stand_pose("D_G11")
    assert abs(f["z"] - 0.095) <= 0.0015, f"托盘底 {f['z'] * 1000:.2f}mm"
    assert abs(f["x"] - tx) <= 0.004 and abs(f["y"] - ty) <= 0.004
    assert f["yaw"] == pytest.approx(0.0, abs=0.035)
    assert f["ratio"] >= 0.85, f"条带支撑 {f['ratio']:.2f}"
    assert not f["pin_contact"], "销盘仍接触托盘"


def test_agv_roundtrip_no_penetration(agv_result):
    a1, a2, a3 = agv_result["audits"]
    assert all(a["violations"] == [] for a in (a1, a2, a3))
    assert max(a["max_pen"] for a in (a1, a2, a3)) < PEN_LIMIT


# ---------------------------------------------------------------- 混合交接链
@pytest.fixture(scope="module")
def chain_result():
    """全链: F01 第三层取架 → 放 H02 → A01 接管 → 运 D_G12 → 放托。"""
    cell = WarehouseCell(timestep_s=0.001, seed=7)
    ctrl = FleetController(cell)
    cell.warmup(0.5)
    segs = {}
    a_fk = ctrl.run_sequence(
        cell.plan_forklift_rack_to_stand("F01", CHAIN_SLOT, CHAIN_STAND),
        settle_s=0.3)
    segs["fk"] = a_fk
    pn = pallet_name_for(CHAIN_SLOT)
    fk_view = cell.forklifter("F01")
    stage = dict(
        z=cell.pallet_bottom_z(pn),
        x=cell.pallet_pose(pn)[0][0], y=cell.pallet_pose(pn)[0][1],
        ratio=cell.surface_load_ratio(pn, cell.stand_gids[CHAIN_STAND]),
        fork_contact=cell.pallet_touches(pn, fk_view.fork_gids),
        fk_xy=fk_view.xy(),
    )
    a1 = ctrl.run_sequence(cell.plan_agv_dock_lift("A01", CHAIN_STAND),
                           settle_s=0.3)
    agv = cell.agv("A01")
    carry = dict(z=cell.pallet_bottom_z(pn),
                 xy_off=(abs(cell.pallet_pose(pn)[0][0] - agv.xy()[0]),
                         abs(cell.pallet_pose(pn)[0][1] - agv.xy()[1])),
                 ratio=cell.surface_load_ratio(pn, {agv.pin_gid}))
    a2 = ctrl.run_sequence(
        cell.plan_agv_transport("A01", CHAIN_STAND, CHAIN_DOCK), settle_s=0.3)
    a3 = ctrl.run_sequence(
        cell.plan_agv_drop("A01", CHAIN_DOCK,
                           corridor_x=cell.layout.stand_pose(CHAIN_STAND)[0]),
        settle_s=0.5)
    pos = cell.pallet_pose(pn)[0]
    return dict(cell=cell, segs=segs, stage=stage, carry=carry,
                audits=(a_fk, a1, a2, a3),
                final=dict(z=cell.pallet_bottom_z(pn), x=pos[0], y=pos[1],
                           ratio=cell.surface_load_ratio(
                               pn, cell.stand_gids[CHAIN_DOCK]),
                           pin_contact=cell.pallet_touches(
                               pn, {cell.agv("A01").pin_gid})))


def test_forklift_third_level_pick(chain_result):
    """第三层 (0.400) 取托: 举升段目标 = 层高+17 (叉顶 0.417, 同 T04)。"""
    a_fk = chain_result["segs"]["fk"]
    assert a_fk["violations"] == []
    segs = a_fk["segments"]
    cell = chain_result["cell"]
    st = chain_result["stage"]
    lift_aid = cell.forklifter("F01").a["lift"]
    lift_targets = [s["target"] for s in segs if s["aid"] == lift_aid]
    assert max(lift_targets) == pytest.approx(0.400 + 0.017 - 0.065)
    # 放托: 台面条带支撑成立, 叉已脱离 (无叉-托接触)
    assert abs(st["z"] - 0.095) <= 0.0015, f"托盘底 {st['z'] * 1000:.2f}mm"
    hx, hy, _ = cell.layout.stand_pose(CHAIN_STAND)
    assert abs(st["x"] - hx) <= 0.004 and abs(st["y"] - hy) <= 0.004
    assert st["ratio"] >= 0.85, f"台面支撑 {st['ratio']:.2f}"
    assert not st["fork_contact"], "叉仍接触托盘"
    # 叉车已撤回停车位 (南泊位带西端), 距交接台 ≥0.3m 不挡 AGV 车道
    fx, fy = st["fk_xy"]
    assert math.hypot(fx - hx, fy - hy) >= 0.3


def test_chain_agv_leg(chain_result):
    c = chain_result["carry"]
    assert c["z"] > 0.118, f"AGV 未顶起: z={c['z']:.4f}"
    assert max(c["xy_off"]) <= 0.010, f"对中偏差 {c['xy_off']}"
    assert c["ratio"] >= 0.85, f"销盘承载 {c['ratio']:.2f}"


def test_chain_final_state(chain_result):
    f = chain_result["final"]
    cell = chain_result["cell"]
    tx, ty, _ = cell.layout.stand_pose(CHAIN_DOCK)
    assert abs(f["z"] - 0.095) <= 0.0015, f"托盘底 {f['z'] * 1000:.2f}mm"
    assert abs(f["x"] - tx) <= 0.004 and abs(f["y"] - ty) <= 0.004
    assert f["ratio"] >= 0.85, f"条带支撑 {f['ratio']:.2f}"
    assert not f["pin_contact"], "销盘仍接触托盘"
    a_fk, a1, a2, a3 = chain_result["audits"]
    assert all(a["violations"] == [] for a in (a_fk, a1, a2, a3))
    assert max(a["max_pen"] for a in (a_fk, a1, a2, a3)) < PEN_LIMIT


# ---------------------------------------------------------------- 8 车共存
def test_fleet_coexistence(chain_result):
    """同一 MJCF 内 8 车各自指令独立响应, 无名冲突 (编译已保证), 无串扰。

    W2 规格升级: 车队 4+4=8 (F01-F04 / A01-A04, 等级 L1-L3 异构)。
    """
    cell = chain_result["cell"]
    assert set(cell.forklifts) == {"F01", "F02", "F03", "F04"}
    assert set(cell.agvs) == {"A01", "A02", "A03", "A04"}
    assert len(cell.veh) == 8 and cell.model.nu == 32
    axes = {**{f"F0{i}": ("x", "y", "yaw", "lift") for i in range(1, 5)},
            **{f"A0{i}": ("x", "y", "yaw", "pin") for i in range(1, 5)}}
    ctrl = FleetController(cell)
    # 交接链收尾后先静置 2s: 排干各车位置伺服的收尾残速 (否则残速滑行
    # 会被误判为脉冲串扰)
    cell.step(2000)
    # 每车一根指令通道, 逐一脉冲; 其余全部关节零串扰
    pulses = [("F01", "x", +0.15), ("F02", "y", -0.15),
              ("F03", "x", +0.10), ("F04", "y", +0.10),
              ("A01", "yaw", +0.5), ("A02", "pin", +0.020),
              ("A03", "yaw", -0.5), ("A04", "pin", +0.015)]
    for vid, ax, delta in pulses:
        before = {v: [cell.veh[v].q(a) for a in axes[v]]
                  for v in cell.veh}
        view = cell.veh[vid]
        speed = 0.20 if ax in ("x", "y") else (0.5 if ax == "yaw" else 0.010)
        ctrl.run_sequence(
            [view.seg(ax, view.q(ax) + view.off[ax] + delta, speed)],
            settle_s=0.2)
        # 独立性判据: **其他车辆**的全部关节零位移 (自身上一脉冲的收尾残速
        # 属自身动力学, 不在此列)。线性阈值 0.5mm = 地面零距接触粘滑蠕变
        # 的实测噪声底 (接触 d=0 的摩擦微滑移需数秒释放); 任何真实的
        # 运动学/动力学耦合都是 mm 级, 0.5mm 足以甄别。
        for v in cell.veh:
            if v == vid:
                continue
            for i, a in enumerate(axes[v]):
                d = abs(cell.veh[v].q(a) - before[v][i])
                assert d < 5e-4, f"{vid}.{ax} 脉冲串扰到 {v}.{a} ({d:.2e})"
        assert abs(cell.veh[vid].q(ax) - before[vid][axes[vid].index(ax)]) \
            > 0.01, f"{vid}.{ax} 未响应"
    assert cell.max_penetration() < PEN_LIMIT


def test_ramp_limits_respected():
    """Ramp 梯形剖面: 巡航 ≤0.35 m/s, 加速度 ≤0.5 m/s² (规格)。"""
    r = Ramp(0.0, 1.0, 0.35, 0.30)
    vs = [(r.at(t + 0.01) - r.at(t)) / 0.01 for t in
          [i * 0.01 for i in range(int(r.duration / 0.01))]]
    assert max(vs) <= 0.35 + 0.02
    acc = max(abs(vs[i + 1] - vs[i]) / 0.01 for i in range(len(vs) - 1))
    assert acc <= 0.5, f"加速度 {acc:.2f} m/s²"


# ---------------------------------------------------------------- 旧世界回归
def test_old_forklift_cell_unaffected():
    """旧 ForkliftCell 构造/行为零破坏 (新模块不得产生导入副作用)。"""
    from sim.forklift_cell import ForkliftCell
    c = ForkliftCell(timestep_s=0.001)
    assert c.vehicles == ["F01"]
    assert c.k["insert_depth"] == 0.065
    plan = c.roundtrip_plan()
    assert len(plan) == 11
