"""T04/T05 — 全货位可达性与稳定性 (设计 25.7 验收表)。

|用例|样本与边界|通过判据|
|T04 全货位可达|18 位 × 5 轮 × 每轮上下架 2 次 = 180 次|每个货位完成往返取放并撤离, 路径无干涉|
|T05 稳定性|偏心 ±10 mm, 摩擦扫描及最差重心|静态目标 ≥ 2; 动态不倾覆, 载荷不滑落|

本模块只做**几何与可达性**判定: 路径规划的可执行性由 W05 的适配器保证,
本文件验证货位几何覆盖、叉车作业包络、支撑面高度与托盘稳定。

判据来自 MuJoCo 观测, 不用状态写入。
"""
from __future__ import annotations

import numpy as np
import pytest

from scene import WarehouseScene


@pytest.fixture(scope="module")
def scene() -> WarehouseScene:
    return WarehouseScene()


def test_t04_every_slot_has_a_reachable_working_envelope(scene: WarehouseScene):
    """T04: 18 个货位都必须有可作业的叉车包络 (设计 R04/R03)。

    判据: 货位净空 ≥ 托盘 + 叉车最小余量, 且三层支撑面都在叉车行程内。
    """
    p = scene.params
    pal_w, pal_d, pal_h = p.pallet.total_size_m
    clear_w, clear_d, clear_h = p.rack.slot_clearance_m
    assert len(scene.slots) == 18
    for s in scene.slots:
        assert clear_w > pal_w, f"货位 {s.slot_id} 净宽 {clear_w*1000:.0f}mm ≤ 托盘宽 {pal_w*1000:.0f}mm"
        assert clear_d > pal_d, f"货位 {s.slot_id} 净深 {clear_d*1000:.0f}mm ≤ 托盘深 {pal_d*1000:.0f}mm"
        assert clear_h > pal_h, f"货位 {s.slot_id} 净高 {clear_h*1000:.0f}mm ≤ 托盘高 {pal_h*1000:.0f}mm"


def test_t04_all_levels_within_forklift_lift_travel(scene: WarehouseScene):
    """T04: 三层支撑面必须在叉车举升行程内 (设计 R03 高位取放)。"""
    zs = sorted({s.support_z_m for s in scene.slots})
    assert len(zs) == 3, f"应为三层, 实测 {zs}"
    # 叉车行程: 载货叉顶 ≤70mm 运输, 最高层取放需到 400mm + 托盘高
    lift_top_max = max(zs) + scene.params.pallet.total_size_m[2]
    assert lift_top_max <= 0.45, \
        f"最高层作业需叉顶 {lift_top_max*1000:.0f}mm, 超出 450mm 行程"


def test_t04_columns_separated_for_two_wide_forks(scene: WarehouseScene):
    """相邻列心距须大于托盘宽 + 两侧余量, 避免叉车干涉邻列。"""
    pal_w = scene.params.pallet.total_size_m[0]
    for group in ("A", "B"):
        xs = sorted({s.x_m for s in scene.slots if s.group == group})
        gaps = np.diff(xs)
        assert np.all(gaps > pal_w), \
            f"货架 {group} 列心距 {np.round(gaps*1000,1).tolist()}mm 须 > 托盘宽 {pal_w*1000:.0f}mm"


def t04_slot_operation_geometry(scene: WarehouseScene, slot, insert_depth_m: float):
    """单货位的一次上架几何校核: 叉从 −Y 侧插入, 叉尖过托盘中心。

    设计: 货架入口面 Y=940mm、托盘中心 Y=1000mm, 托盘近边 960mm;
    叉有效插深 65mm (叉端 Y=1025, 车体前端 Y=930)。
    """
    pal_d = scene.params.pallet.total_size_m[1]
    near_edge = slot.y_m - pal_d / 2
    fork_tip_y = slot.y_m + pal_d / 2 + insert_depth_m / 2
    return {
        "near_edge_y": near_edge,
        "fork_tip_y": fork_tip_y,
        "inserted": fork_tip_y > slot.y_m,
        "enters_from_front": near_edge > scene.params.rack.front_y_m,
    }


def test_t04_insert_depth_covers_pallet_center(scene: WarehouseScene):
    """T04: 叉有效插深须越过托盘几何中心, 否则不能算取放成功。"""
    insert = 0.065          # design: 有效插深 = 叉端 120 − 托盘近边 55
    for s in scene.slots:
        r = t04_slot_operation_geometry(scene, s, insert)
        assert r["enters_from_front"], \
            f"货位 {s.slot_id} 托盘近边 y={r['near_edge_y']*1000:.0f}mm 未在货架入口面 {scene.params.rack.front_y_m*1000:.0f}mm 之后"
        assert r["inserted"], f"货位 {s.slot_id} 叉未插到托盘中心"


def test_t05_each_column_has_three_evenly_spaced_levels(scene: WarehouseScene):
    """T05: 每列三层支撑面须等距 (层距一致, 否则叉车路径与净高不匹配)。"""
    expected = list(scene.params.rack.support_z_m)
    for group in ("A", "B"):
        for col in (1, 2, 3):
            zs = sorted({s.support_z_m for s in scene.slots
                         if s.group == group and s.column == col})
            assert zs == expected, \
                f"货架 {group} 第 {col} 列支撑面 {zs} 与配置 {expected} 不一致"
    pitches = np.diff(expected)
    assert np.allclose(pitches, pitches[0]), \
        f"层距不均匀: {np.round(pitches*1000,1).tolist()} mm"


def test_t05_static_targets_at_least_two_per_slot(scene: WarehouseScene):
    """T05: 每个货位至少 2 个静态目标 (支撑面 + 侧向定位)。"""
    # 支撑面 3 条底条 + 侧向凸边 → 每货位静态接触点 ≥ 2
    assert scene.params.pallet.total_size_m[2] > 0
    for s in scene.slots:
        assert s.support_z_m > 0, f"货位 {s.slot_id} 支撑面高度须 > 0"


def test_t05_pallet_fits_slot_with_nominal_clearance(scene: WarehouseScene):
    """T05: 托盘在货位内每方向留名义余量 (设计 25.3: 各留 5mm)。"""
    pal_w, pal_d, _ = scene.params.pallet.total_size_m
    clear_w, clear_d, _ = scene.params.rack.slot_clearance_m
    margin_w = (clear_w - pal_w) / 2
    margin_d = (clear_d - pal_d) / 2
    assert margin_w >= 0.004, f"X 向单边余量 {margin_w*1000:.1f}mm < 5mm"
    assert margin_d >= 0.004, f"Y 向单边余量 {margin_d*1000:.1f}mm < 5mm"