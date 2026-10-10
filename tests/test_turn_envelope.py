"""转向包络回归测试。

锁住两个已复核的结论 (reports/w04_scene_deviation.md):
1. 固定车体最大角点回转半径 ≈ 209 mm, 扫掠圆直径在 430 mm 预留内。
2. P02 停车位原地旋转会越过 80 mm 桌边退让, 越界方向是 **+X**,
   需要沿 −X 驶出 ≥ 39 mm 才能原地旋转。

这些是几何判据, 不依赖仿真, 所以跑得快且确定。
"""
from __future__ import annotations

import math

import pytest
import yaml

from cad.models import build_all_models
from cad.params import derive_parameters, load_params


def _local_aabb(node, origin=(0.0, 0.0), out=None):
    """累积父级偏移, 返回子树所有 geom 在车体本地系的 (x0,x1,y0,y1)。"""
    if out is None:
        out = []
    ox, oy = origin
    for pr in node.prims:
        cx, cy = ox + pr.center_m[0], oy + pr.center_m[1]
        if pr.type == "box":
            sx, sy = pr.size_m[0], pr.size_m[1]
        elif pr.type == "cylinder":
            r = pr.radius_m
            if pr.axis == "y":
                sx, sy = 2 * r, pr.height_m      # 轴沿 y: x=直径, y=高
            else:
                sx, sy = 2 * r, 2 * r
        else:                                    # ellipsoid / sphere
            sx = 2 * (pr.radius_m or pr.size_m[0])
            sy = (pr.height_m or 2 * pr.size_m[1])
        out.append((node.name, pr.name, cx - sx / 2, cx + sx / 2,
                    cy - sy / 2, cy + sy / 2))
    for c in node.children:
        _local_aabb(c, (ox + c.pos_m[0], oy + c.pos_m[1]), out)
    return out


@pytest.fixture(scope="module")
def forklift_bounds():
    p = load_params()
    models = build_all_models(p, derive_parameters(p))
    boxes = _local_aabb(models["forklift"])
    fixed = [b for b in boxes if "carriage" not in b[0]]
    return dict(
        x0=min(b[2] for b in fixed), x1=max(b[3] for b in fixed),
        y0=min(b[4] for b in fixed), y1=max(b[5] for b in fixed),
    )


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open("configs/design.yaml"))


def test_turn_radius_matches_design_209mm(forklift_bounds):
    """固定车体最远角点半径 ≈ 209 mm (设计 155 行)。"""
    b = forklift_bounds
    r = max(math.hypot(x, y)
            for x in (b["x0"], b["x1"]) for y in (b["y0"], b["y1"]))
    assert abs(r - 0.209) < 0.002, f"回转半径 {r*1000:.1f} mm, 期望 ~209 mm"


def test_swept_circle_within_430mm_reserve(forklift_bounds):
    """扫掠外接圆直径不超过设计预留的 430 mm。"""
    b = forklift_bounds
    r = max(math.hypot(x, y)
            for x in (b["x0"], b["x1"]) for y in (b["y0"], b["y1"]))
    assert r * 2 <= 0.430, f"扫掠直径 {r*2000:.1f} mm 超出 430 mm 预留"


def _violations(x, y, r, setback=0.080, tx1=1.8, ty1=1.2):
    """返回该点原地旋转时越界的方向集合。"""
    out = set()
    if x + r > tx1 - setback:
        out.add("+X")
    if y + r > ty1 - setback:
        out.add("+Y")
    if x - r < setback:
        out.add("-X")
    if y - r < setback:
        out.add("-Y")
    return out


def test_p02_in_place_turn_violates_setback(cfg, forklift_bounds):
    """P02 停车位原地旋转必须越界 —— 设计 159 行的断言成立。"""
    b = forklift_bounds
    r = max(math.hypot(x, y)
            for x in (b["x0"], b["x1"]) for y in (b["y0"], b["y1"]))
    p02 = cfg["regions"]["park_p02"]
    assert _violations(p02["x_m"], p02["y_m"], r), \
        "P02 原地旋转不再越界, 设计 159 行的约束可能已过时"


def test_violation_direction_is_plus_x(cfg, forklift_bounds):
    """越界方向是 +X 而非 −Y —— 设计 159 行写的「先沿 −Y 驶出」方向有误。"""
    b = forklift_bounds
    r = max(math.hypot(x, y)
            for x in (b["x0"], b["x1"]) for y in (b["y0"], b["y1"]))
    p02 = cfg["regions"]["park_p02"]
    v = _violations(p02["x_m"], p02["y_m"], r)
    assert "+X" in v, "预期 +X 越界"
    assert "-Y" not in v, "-Y 未越界, 与设计描述不符"
    assert "+Y" not in v, "+Y 实测有余量(11mm), 不应越界"


def test_exiting_along_minus_y_does_not_help(cfg, forklift_bounds):
    """沿 −Y 驶出无法解决 +X 越界 —— 这是设计文档的方向错误。"""
    b = forklift_bounds
    r = max(math.hypot(x, y)
            for x in (b["x0"], b["x1"]) for y in (b["y0"], b["y1"]))
    p02 = cfg["regions"]["park_p02"]
    for dy in (0.10, 0.20, 0.30):
        assert "+X" in _violations(p02["x_m"], p02["y_m"] - dy, r), \
            f"沿 −Y 驶出 {dy*1000:.0f} mm 后 +X 仍应越界"


def test_exiting_along_minus_x_39mm_frees_turning(cfg, forklift_bounds):
    """沿 −X 驶出 ≥39 mm 后可原地旋转。"""
    b = forklift_bounds
    r = max(math.hypot(x, y)
            for x in (b["x0"], b["x1"]) for y in (b["y0"], b["y1"]))
    p02 = cfg["regions"]["park_p02"]
    assert _violations(p02["x_m"] - 0.039, p02["y_m"], r) == set(), \
        "沿 −X 驶出 39 mm 应可原地旋转"
    assert _violations(p02["x_m"] - 0.050, p02["y_m"], r) == set(), \
        "沿 −X 驶出 50 mm 应有余量"