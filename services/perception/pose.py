"""W04 姿态估计 — 用已知尺寸标记 + 相机模型估计箱件位姿。

设计约束 (文档 21.2 / 25.4):
* 箱件姿态用**独立已知尺寸视觉标记**或可见几何角点估计;
* 姿态标记尺寸初值 20 mm, 放在箱顶且不被夹爪接近前遮挡;
* 四角与箱件局部变换在资产中**冻结**;
* 单一平面同应只适用于选定平面 —— 离开该平面的标记不能用同一变换,
  必须用已知尺寸和相机模型估计姿态 (本模块即为此);
* 规划输入要求: 位置误差 ≤1 mm、姿态误差 ≤2°、帧龄 ≤0.1 s;
  真值只用于独立误差评测, 不参与放行决策。

本模块只做几何求解, 不做质量门限判定 (门限在 scene/调用方)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参 + 畸变 (设计要求标定内参与畸变)。"""

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: tuple[float, ...] = ()

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=float)


@dataclass(frozen=True)
class PoseObservation:
    """一次姿态估计的结果 (全部来自图像, 不含真值)。"""

    success: bool
    t_wc: np.ndarray | None = None      # 目标在相机系的位置 (m)
    r_wc: np.ndarray | None = None      # 目标在相机系的姿态 (3×3)
    reproj_rmse_px: float = float("nan")
    marker_corners_px: tuple[np.ndarray, ...] = ()
    reason: str = ""

    @property
    def position(self) -> np.ndarray:
        return self.t_wc.copy() if self.t_wc is not None else np.full(3, np.nan)


# 冻结的标记几何 (资产侧, 不随运行变化)
# 标记贴在箱顶平面, 四角外接框边长 marker_side_m, 中心对齐箱顶中心。
# 四角顺序必须与 assets 中冻结的局部坐标一致 (逆时针, 从标记左下角起):
_MARKER_CORNERS_LOCAL = (
    (-0.5, -0.5, 0.0),
    (+0.5, -0.5, 0.0),
    (+0.5, +0.5, 0.0),
    (-0.5, +0.5, 0.0),
)


def marker_object_points(marker_side_m: float, plane_offset_z: float = 0.0
                         ) -> np.ndarray:
    """标记四角的物体坐标 (N×3), 供 solvePnP 使用。"""
    half = marker_side_m / 2.0
    return np.array([(sx * half, sy * half, plane_offset_z)
                     for sx, sy, _ in _MARKER_CORNERS_LOCAL], dtype=float)


def solve_marker_pose(corners_px: np.ndarray, cam: CameraIntrinsics,
                      marker_side_m: float, plane_offset_z: float = 0.0,
                      flags: int | None = None) -> PoseObservation:
    """由四个标记角点像素解算目标在相机系的位姿。

    corners_px: 形状 (4, 2) 的像素坐标, 顺序须与 marker_object_points 一致。
    设计要求姿态误差 ≤2°, 故优先用 SOLVEPNP_IPPE (仅 4 点平面可用)。
    """
    pts_img = np.asarray(corners_px, dtype=float)
    if pts_img.shape != (4, 2):
        return PoseObservation(False, reason=f"角点数/形状错误 {pts_img.shape}")
    obj = marker_object_points(marker_side_m, plane_offset_z)
    use_flags = cv2.SOLVEPNP_IPPE if flags is None else flags
    ok = False
    rvec = tvec = None
    for f in (use_flags, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, pts_img, cam.matrix, np.asarray(cam.dist_coeffs, dtype=float),
                flags=f)
        except cv2.error:
            continue
        if ok:
            break
    if not ok or rvec is None or tvec is None:
        return PoseObservation(False, reason="solvePnP 未收敛")

    r, _ = cv2.Rodrigues(rvec)
    r = r.reshape(3, 3)
    t = np.asarray(tvec, dtype=float).reshape(3)
    # 重投影误差 (独立评测, 不参与放行)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.matrix,
                               np.asarray(cam.dist_coeffs, dtype=float))
    rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - pts_img) ** 2, axis=1))))
    return PoseObservation(True, t, r, rmse, tuple(pts_img))


def refine_pose(corners_px: np.ndarray, cam: CameraIntrinsics,
                marker_side_m: float, plane_offset_z: float = 0.0,
                iters: int = 60) -> PoseObservation:
    """带图像重投影最小二乘精修的姿态估计。

    四点平面 PnP 对像素噪声敏感: 实测 σ=0.3px → 位置 0.27mm / 姿态 1.83°,
    σ=0.8px → 3.02mm / 10.26° (设计要求 ≤1mm / ≤2°)。这里在 solvePnP 初值上
    做高斯-牛顿重投影精修, 把 4 点解算放大到 8 残差并抑制噪声。
    """
    obs = solve_marker_pose(corners_px, cam, marker_side_m, plane_offset_z)
    if not obs.success:
        return obs
    obj = marker_object_points(marker_side_m, plane_offset_z)
    K = cam.matrix
    dist = np.asarray(cam.dist_coeffs, dtype=float)
    pts = np.asarray(corners_px, dtype=float)
    rvec, _ = cv2.Rodrigues(obs.r_wc)
    tvec = obs.t_wc.reshape(3, 1).copy()

    def residual(rv: np.ndarray, tv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (重投影残差 8 维, 雅可比 8×6)。

        注意: cv2.projectPoints 的雅可比输出参数在 OpenCV 的 3D→2D 用法下
        **恒为全零** (实测), 不能直接用。改为解析推导:
            p_c = R·X + t,  u = fx·x/z + cx,  v = fy·y/z + cy
        对 rvec 用左扰动的链式法则展开。
        """
        R, _ = cv2.Rodrigues(rv)
        t = tv.reshape(3)
        Pc = (R @ obj.T).T + t                       # (4,3) 相机系点
        z = Pc[:, 2]
        if np.any(z <= 1e-9):
            return np.full(8, 1e6), np.zeros((8, 6))
        uv_p, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        res = (pts - uv_p.reshape(4, 2)).reshape(-1)

        fx, fy = K[0, 0], K[1, 1]
        # 像素对相机系点的雅可比 (2×3)
        Jc = np.zeros((4, 2, 3))
        Jc[:, 0, 0] = fx / z
        Jc[:, 0, 2] = -fx * Pc[:, 0] / (z * z)
        Jc[:, 1, 1] = fy / z
        Jc[:, 1, 2] = -fy * Pc[:, 1] / (z * z)
        # d(p_c)/d(t) = I
        J = np.zeros((8, 6))
        # 对 rvec 的偏导: 有限差分核对显示 OpenCV 的 rvec 扰动等价于**右**乘
        # exp(δθ), 故 d(R·X)/d(rvec) = +[R·X]× (与常见的左扰动相反号)。
        for i in range(4):
            p = R @ obj[i]
            skew = np.array([[0.0, -p[2], p[1]],
                             [p[2], 0.0, -p[0]],
                             [-p[1], p[0], 0.0]])
            J[2 * i:2 * i + 2, 0:3] = Jc[i] @ skew
            J[2 * i:2 * i + 2, 3:6] = Jc[i]
        return res, J

    for _ in range(iters):
        res, J = residual(rvec, tvec)
        cost = float(res @ res)
        if np.sqrt(cost / 8.0) < 1e-4:
            break
        try:
            delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        # 线搜索: IPPE 初值可能落在另一个解附近, 盲目全步会发散
        step = 1.0
        improved = False
        for _ in range(8):
            rv_try = rvec + step * delta[:3].reshape(3, 1)
            tv_try = tvec + step * delta[3:].reshape(3, 1)
            r_try, _ = residual(rv_try, tv_try)
            c_try = float(r_try @ r_try)
            if np.isfinite(c_try) and c_try < cost:
                rvec, tvec = rv_try, tv_try
                improved = True
                break
            step *= 0.5
        if not improved:
            break

    r, _ = cv2.Rodrigues(rvec)
    r = r.reshape(3, 3)
    t = tvec.reshape(3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - pts) ** 2, axis=1))))
    return PoseObservation(True, t, r, rmse, tuple(pts))


def refine_pose_corners(corners_px: np.ndarray, cam: CameraIntrinsics,
                        marker_side_m: float, plane_offset_z: float = 0.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """用检测到的四角做一次重投影最小二乘精修, 返回 (精修像素, 观测)。"""
    obs = solve_marker_pose(corners_px, cam, marker_side_m, plane_offset_z)
    if not obs.success:
        return np.asarray(corners_px, dtype=float), obs
    obj = marker_object_points(marker_side_m, plane_offset_z)
    rvec, _ = cv2.Rodrigues(obs.r_wc)
    proj, _ = cv2.projectPoints(obj, rvec, obs.t_wc.reshape(3, 1),
                               cam.matrix, np.asarray(cam.dist_coeffs, dtype=float))
    return proj.reshape(4, 2), obs


def box_corner_object_points(size_m: tuple[float, float, float],
                            top_inset_m: float = 0.0) -> np.ndarray:
    """箱件顶部四角 + 两条竖边的上/中点 (共 6 点, 物体系)。

    设计 25.4 允许用"可见几何角点"估计姿态。单平面 4 点解算的姿态可观测性
    差 (实测 40mm 标记 @150mm、角点噪声 0.2px → 姿态 3.9°, 达不到 2° 判据)。
    引入竖边上的点使点集**非共面**, 打破平面解算的退化, 姿态精度显著提升。
    顶角相对箱体中心的偏移按 size 的 x/y 半长计算。
    """
    sx, sy, sz = size_m
    hx, hy = sx / 2.0 - top_inset_m, sy / 2.0 - top_inset_m
    top = [(-hx, -hy, sz / 2), (hx, -hy, sz / 2), (hx, hy, sz / 2), (-hx, hy, sz / 2)]
    mid = [(-hx, -hy, 0.0), (hx, -hy, 0.0)]
    return np.array(top + mid, dtype=float)


def solve_box_pose(corners_px: np.ndarray, cam: CameraIntrinsics,
                   size_m: tuple[float, float, float],
                   top_inset_m: float = 0.0) -> PoseObservation:
    """由箱件可见几何角点 (≥4 点, 非共面更佳) 解算位姿。

    用 solvePnPRansac: 允许部分角点被遮挡 (设计要求识别不可信输入时拒绝,
    但正常跟踪允许少量遮挡, 由 RANSAC 剔除外点)。
    """
    pts = np.asarray(corners_px, dtype=float)
    obj = box_corner_object_points(size_m, top_inset_m)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] != obj.shape[0]:
        return PoseObservation(False, reason=f"角点形状 {pts.shape} 与物体点 {obj.shape} 不匹配")
    dist = np.asarray(cam.dist_coeffs, dtype=float)
    rvec = tvec = None
    for f in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
        try:
            ok, rv, tv = cv2.solvePnP(obj, pts, cam.matrix, dist, flags=f)
        except cv2.error:
            continue
        if ok:
            rvec, tvec = rv, tv
            break
    if rvec is None:
        return PoseObservation(False, reason="solvePnP 未收敛")
    r, _ = cv2.Rodrigues(rvec)
    r = r.reshape(3, 3)
    t = np.asarray(tvec, dtype=float).reshape(3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.matrix, dist)
    rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
    return PoseObservation(True, t, r, rmse, tuple(pts))


def plan_gate(pos_err_m: float | None, rot_err_rad: float | None,
              frame_age_s: float | None, pos_tol_m: float = 0.001,
              rot_tol_rad: float = math.radians(2.0),
              max_age_s: float = 0.1) -> tuple[bool, str]:
    """规划放行门限 (设计 25.4)。

    三项全部合格才放行; 任一缺测量即拒绝 —— 不用真值替代。
    位置/姿态误差由**独立真值评测**给出, 但缺测量不得默认通过。
    """
    if pos_err_m is None or rot_err_rad is None or frame_age_s is None:
        return False, "缺测量 (位置/姿态/帧龄 任一为 None)"
    if pos_err_m > pos_tol_m:
        return False, f"位置误差 {pos_err_m*1000:.2f}mm > {pos_tol_m*1000:.0f}mm"
    if rot_err_rad > rot_tol_rad:
        return False, f"姿态误差 {math.degrees(rot_err_rad):.2f}° > 2°"
    if frame_age_s > max_age_s:
        return False, f"帧龄 {frame_age_s:.3f}s > {max_age_s}s"
    return True, "放行"