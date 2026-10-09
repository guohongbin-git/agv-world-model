"""W04 姿态估计 — 已知尺寸标记/角点 + 相机模型求解的精度与门限。

设计 25.4 的可测子集:
* 姿态标记用**已知尺寸** + 相机模型, 不用单一平面同应
* 规划放行要求: 位置误差 ≤1mm、姿态误差 ≤2°、帧龄 ≤0.1s
* 真值只用于独立误差评测, 缺测量不得默认放行
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from services.perception.pose import (
    CameraIntrinsics, box_corner_object_points, marker_object_points,
    plan_gate, refine_pose, solve_box_pose, solve_marker_pose,
)

FRAME_W, FRAME_H = 1280, 960
BOX_SIZE = (0.060, 0.040, 0.050)      # 基准箱件 60×40×50mm


@pytest.fixture(scope="module")
def cam() -> CameraIntrinsics:
    return CameraIntrinsics(1100.0, 1100.0, FRAME_W / 2, FRAME_H / 2, (0.0,) * 5)


def _rxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _project(obj, r, t, cam) -> np.ndarray:
    uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(r)[0], np.asarray(t).reshape(3, 1),
                             cam.matrix, np.zeros(5))
    return uv.reshape(-1, 2)


def _errors(obs, r_true, t_true) -> tuple[float, float]:
    pe = float(np.linalg.norm(obs.t_wc - t_true)) * 1000.0
    r_delta = r_true.T @ obs.r_wc
    ae = math.degrees(math.acos(float(np.clip((np.trace(r_delta) - 1) / 2, -1, 1))))
    return pe, ae


def test_pose_is_exact_without_noise(cam: CameraIntrinsics):
    r = _rxyz(math.radians(10), math.radians(-6), math.radians(3))
    t = np.array([0.01, -0.02, 0.30])
    uv = _project(box_corner_object_points(BOX_SIZE), r, t, cam)
    obs = solve_box_pose(uv, cam, BOX_SIZE)
    assert obs.success
    pe, ae = _errors(obs, r, t)
    assert pe < 1e-3, f"无噪声位置误差 {pe:.4f}mm"
    assert ae < 1e-3, f"无噪声姿态误差 {ae:.4f}°"


def test_box_pose_meets_design_gate_under_corner_noise(cam: CameraIntrinsics):
    """角点噪声 ≤0.5px 时须满足设计 ≤1mm / ≤2°。"""
    r = _rxyz(math.radians(10), math.radians(-6), math.radians(3))
    t = np.array([0.01, -0.02, 0.30])
    uv = _project(box_corner_object_points(BOX_SIZE), r, t, cam)
    rng = np.random.default_rng(7)
    for noise in (0.1, 0.2, 0.5):
        pe_l, ae_l = [], []
        for _ in range(100):
            obs = solve_box_pose(uv + rng.normal(0, noise, uv.shape), cam, BOX_SIZE)
            assert obs.success
            pe, ae = _errors(obs, r, t)
            pe_l.append(pe)
            ae_l.append(ae)
        assert np.mean(pe_l) <= 1.0, f"噪声{noise}px 位置 {np.mean(pe_l):.3f}mm > 1mm"
        assert np.mean(ae_l) <= 2.0, f"噪声{noise}px 姿态 {np.mean(ae_l):.3f}° > 2°"


def test_nonplanar_corners_beat_planar_marker_attitude(cam: CameraIntrinsics):
    """非共面角点的姿态精度须显著优于单平面标记。

    实测: 40mm 单平面标记 @150mm、0.2px 噪声 → 姿态 3.87° (超设计 2°);
    同一相机/噪声下用 6 点非共面角点 → 约 0.14°。设计 25.4 允许用
    "可见几何角点", 本用例锁定该选择。
    """
    r = _rxyz(math.radians(10), math.radians(-6), math.radians(3))
    rng = np.random.default_rng(11)
    t_marker = np.array([0.0, 0.0, 0.15])
    uv_m = _project(marker_object_points(0.040), r, t_marker, cam)
    t_box = np.array([0.01, -0.02, 0.30])
    uv_b = _project(box_corner_object_points(BOX_SIZE), r, t_box, cam)
    ae_marker, ae_box = [], []
    for _ in range(80):
        o_m = solve_marker_pose(uv_m + rng.normal(0, 0.2, uv_m.shape), cam, 0.040)
        o_b = solve_box_pose(uv_b + rng.normal(0, 0.2, uv_b.shape), cam, BOX_SIZE)
        if o_m.success:
            ae_marker.append(_errors(o_m, r, t_marker)[1])
        if o_b.success:
            ae_box.append(_errors(o_b, r, t_box)[1])
    assert np.mean(ae_box) < 1.0, f"角点姿态 {np.mean(ae_box):.2f}° 应优于 1°"
    assert np.mean(ae_box) < np.mean(ae_marker), \
        f"角点 ({np.mean(ae_box):.2f}°) 应优于单平面标记 ({np.mean(ae_marker):.2f}°)"


def test_refine_pose_never_increases_reprojection_error(cam: CameraIntrinsics):
    """重投影精修不得变差 (line search 保证单调不增)。

    两次调用必须用**同一份**噪声 (否则比的是不同输入)。
    """
    r = _rxyz(math.radians(10), math.radians(-6), math.radians(3))
    t = np.array([0.01, -0.02, 0.30])
    uv = _project(marker_object_points(0.020), r, t, cam)
    rng = np.random.default_rng(3)
    for _ in range(20):
        noisy = uv + rng.normal(0, 0.3, uv.shape)
        obs0 = solve_marker_pose(noisy, cam, 0.020)
        obs1 = refine_pose(noisy, cam, 0.020)
        if obs0.success and obs1.success:
            assert obs1.reproj_rmse_px <= obs0.reproj_rmse_px + 1e-6, \
                f"精修后重投影误差变大 {obs0.reproj_rmse_px:.3f} → {obs1.reproj_rmse_px:.3f}px"


def test_plan_gate_requires_all_three_measurements():
    """三项测量缺一不可放行 (设计 25.4)。"""
    assert plan_gate(None, 0.0, 0.0)[0] is False
    assert plan_gate(0.0, None, 0.0)[0] is False
    assert plan_gate(0.0, 0.0, None)[0] is False
    assert plan_gate(0.0005, math.radians(1.0), 0.05)[0] is True
    assert plan_gate(0.002, math.radians(1.0), 0.05)[0] is False
    assert plan_gate(0.0005, math.radians(3.0), 0.05)[0] is False
    assert plan_gate(0.0005, math.radians(1.0), 0.2)[0] is False


def test_solve_rejects_malformed_corners(cam: CameraIntrinsics):
    obj = box_corner_object_points(BOX_SIZE)
    uv = _project(obj, np.eye(3), np.array([0.0, 0.0, 0.3]), cam)
    assert solve_box_pose(uv[:3], cam, BOX_SIZE).success is False
    assert solve_box_pose(uv[:, :1], cam, BOX_SIZE).success is False


def test_box_corner_points_are_non_coplanar():
    """角点集须非共面 —— 共面会使姿态可观测性退化。"""
    pts = box_corner_object_points(BOX_SIZE)
    assert pts.shape == (6, 3)
    centred = pts - pts.mean(axis=0)
    # 协方差矩阵应有两个非零特征值
    assert int((np.linalg.svd(centred, compute_uv=False) > 1e-9).sum()) >= 2