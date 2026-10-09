"""W04 全场景 — 18 货位四设备世界的可测不变量。

覆盖本轮定位并修复的三处缺陷, 防止回归:
1. 桌面覆盖范围必须包含所有 station/rack/设备停车位 (设计 §3 左下角原点)
2. 执行器必须作用在关节自身 body (jointinparent) 且 ctrlrange 非零宽
3. 机械臂关节必须有阻尼, 否则重力补偿后仍持续振荡
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from scene import WarehouseScene


@pytest.fixture(scope="module")
def scene() -> WarehouseScene:
    return WarehouseScene()


def test_slots_are_18_two_groups_three_columns_three_levels(scene: WarehouseScene):
    """设计: 两组 × 三列 × 三层 = 18 货位, 每货位 1 个托盘。"""
    assert len(scene.slots) == 18
    groups = {s.group for s in scene.slots}
    assert groups == {"A", "B"}
    for g in ("A", "B"):
        cols = {s.column for s in scene.slots if s.group == g}
        levels = {s.level for s in scene.slots if s.group == g}
        assert len(cols) == 3, f"货架 {g} 列数 {len(cols)}"
        assert len(levels) == 3, f"货架 {g} 层数 {len(levels)}"
    assert len({s.slot_id for s in scene.slots}) == 18, "货位 id 必须唯一"


def test_slot_support_heights_match_design(scene: WarehouseScene):
    """三层支撑面 z=100/250/400mm (configs/design.yaml rack.support_z_m)。"""
    zs = sorted({round(s.support_z_m, 4) for s in scene.slots})
    assert zs == [0.1, 0.25, 0.4]


def test_table_covers_all_stations_and_devices(scene: WarehouseScene):
    """设计 §3: 桌面 X 0→1800 / Y 0→1200, 原点在左下角。

    此前桌面建在几何中心, 导致站体/货架 B/叉车停车位落到桌面外。
    """
    m = scene.model
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    table = None
    for g in range(m.ngeom):
        if name(mujoco.mjtObj.mjOBJ_GEOM, g).startswith("table"):
            table = g
    assert table is not None, "缺少桌面 geom"
    L = scene.params.table.length_m
    W = scene.params.table.width_m
    c = m.geom_pos[table]
    assert abs((c[0] - L / 2)) < 1e-6 and abs((c[1] - W / 2)) < 1e-6, \
        f"桌面中心应在 ({L/2:.3f},{W/2:.3f})（左下角为原点）, 实测 ({c[0]:.3f},{c[1]:.3f})"

    # 每个工位/货架/停车位都要落在桌面范围内
    reg = scene.params.regions
    pts = {
        "I01": (reg.inbound_i01.x_m, reg.inbound_i01.y_m),
        "H01": (reg.handoff_h01.x_m, reg.handoff_h01.y_m),
        "O01": (reg.outbound_o01.x_m, reg.outbound_o01.y_m),
        "R01": (reg.park_p01.x_m, reg.park_p01.y_m),
        "F01": (reg.park_p02.x_m, reg.park_p02.y_m),
    }
    for nm, (x, y) in pts.items():
        assert 0 <= x <= L and 0 <= y <= W, \
            f"{nm} ({x:.3f},{y:.3f}) 落在桌面 X[0,{L:.2f}] Y[0,{W:.2f}] 之外"
    for s in scene.slots:
        assert 0 <= s.x_m <= L and 0 <= s.y_m <= W, \
            f"货位 {s.slot_id} ({s.x_m:.3f},{s.y_m:.3f}) 落在桌面外"


def test_actuators_bind_to_own_joint(scene: WarehouseScene):
    """执行器必须 jointinparent (作用在关节自身 body), 否则力矩打到固定基座。

    MuJoCo 3.15 的 joint= 默认取父链 → 整场景 t=0 NaN。
    """
    m = scene.model
    assert m.nu > 0
    for a in range(m.nu):
        t = int(m.actuator_trntype[a])
        assert t == 1, f"执行器 {a} trntype={t} (应为 joint 传动)"
        cr = m.actuator_ctrlrange[a]
        assert cr[1] > cr[0], f"执行器 {a} ctrlrange 零宽 {cr}"


def test_arm_joints_have_damping(scene: WarehouseScene):
    """无阻尼时机械臂在重力补偿后仍以 ~1 rad/s 持续振荡 (实测 J4)。"""
    m = scene.model
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    checked = 0
    for j in range(m.njnt):
        jn = name(mujoco.mjtObj.mjOBJ_JOINT, j)
        if not (jn.startswith("M01_J1") or jn.startswith("M02_J1")):
            continue
        dof = int(m.jnt_dofadr[j])
        assert m.dof_damping[dof] > 0, f"{jn} 阻尼为 0"
        assert m.dof_armature[dof] > 0, f"{jn} 转子惯量为 0"
        checked += 1
    assert checked >= 12, f"只检查到 {checked} 个机械臂关节"


def test_scene_settles_and_holds_pose(scene: WarehouseScene):
    """重力补偿 + 阻尼后场景应收敛, 且初始物不被扰动。"""
    s = WarehouseScene()
    before = {n: s.body(n).copy() for n in ("pallet", "goods", "R01", "F01")}
    s.settle(5.0)
    assert s.max_speed() < 0.5, f"5s 后仍在运动 max|qvel|={s.max_speed():.4f}"
    s.settle(15.0)
    assert s.max_speed() < 0.1, f"20s 后未收敛 max|qvel|={s.max_speed():.4f}"
    for n, p0 in before.items():
        drift = float(np.linalg.norm(s.body(n) - p0))
        assert drift < 2e-3, f"{n} 静置漂移 {drift*1000:.2f}mm"


def test_slot_lookup_by_id_and_position(scene: WarehouseScene):
    s = scene.slots[0]
    assert scene.slot(s.slot_id) is s
    assert scene.slot_at(s.x_m, s.y_m) is s
    assert scene.slot_at(-5.0, -5.0) is None
    with pytest.raises(KeyError):
        scene.slot("NOPE")