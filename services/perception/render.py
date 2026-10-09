"""MuJoCo 相机渲染桥：把标签 QR 纹理贴到几何表面，真实渲染出感知输入帧。

对应设计文档 8.1/21.2：感知只吃 MuJoCo 渲染图像；纹理以图像方式贴到
geom material，相机有具名安装位姿与可检查视场。本模块只做"贴图+拍摄"，
不提供任何绕过图像的扫码捷径。

投影模型：
  - 水平视场在标签距离处覆盖 fov_width_m（design.yaml 初值 0.100 m）；
  - 相机可绕标签竖直轴旋转 view_deg（斜视 45° 预算核对用）；
  - px/module（正对）= width_px × label_side_m / fov_width_m / 29。
"""

from __future__ import annotations

import math

import cv2
import mujoco
import numpy as np

from services.perception.labels import MODULES_PER_SIDE

FOV_WIDTH_M = 0.100  # design.yaml: local_fov_width_at_label_distance_max_m
LABEL_SIDE_M = 0.015  # design.yaml: pallet_label_side_m
FRAME_WIDTH_PX = 1280  # design.yaml: local_frame_width_px
FRAME_HEIGHT_PX = 960


def frontal_px_per_module(
    width_px: int = FRAME_WIDTH_PX,
    label_side_m: float = LABEL_SIDE_M,
    fov_width_m: float = FOV_WIDTH_M,
) -> float:
    """正对拍摄的理论每模块像素（含静区 29 模块）。"""
    return width_px * label_side_m / fov_width_m / MODULES_PER_SIDE


def _camera_axes(view_deg: float) -> tuple[str, tuple[float, float, float]]:
    """绕世界 Z 轴旋转 view_deg 后的相机 xyaxes 与指向标签的视轴方向。

    相机局部 -Z 为视轴；初始正对时位于标签 -Y 侧向 +Y 看。
    """
    a = math.radians(view_deg)
    view_dir = (math.sin(a), math.cos(a), 0.0)  # 相机 → 标签
    x_axis = (math.cos(a), -math.sin(a), 0.0)  # 像面水平方向
    z_axis = tuple(-c for c in view_dir)
    y_axis = (
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    )
    xyaxes = " ".join(f"{v:.9f}" for v in (*x_axis, *y_axis))
    return xyaxes, view_dir


def render_label_frame(
    label_png_bytes: bytes,
    *,
    view_deg: float = 0.0,
    distance_m: float = 0.150,
    fov_width_m: float = FOV_WIDTH_M,
    label_side_m: float = LABEL_SIDE_M,
    width_px: int = FRAME_WIDTH_PX,
    height_px: int = FRAME_HEIGHT_PX,
) -> np.ndarray:
    """渲染贴有 QR 纹理的标签平面，返回 BGR uint8 帧。

    标签为 15 mm 薄板（各面整幅贴图），相机在给定距离与斜视角拍摄。
    """
    half = label_side_m / 2.0
    half_thickness = label_side_m / 30.0  # 0.5 mm 薄板
    xyaxes, view_dir = _camera_axes(view_deg)
    cam_pos = tuple(-distance_m * c for c in view_dir)
    fovy_deg = math.degrees(
        2.0 * math.atan((fov_width_m / 2.0 / distance_m) * height_px / width_px)
    )
    xml = f"""
<mujoco model="label_scan">
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{width_px}" offheight="{height_px}"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="label_tex" type="2d" file="label.png"/>
    <material name="label_mat" texture="label_tex" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <geom name="label_plate" type="box" size="{half} {half_thickness} {half}"
          pos="0 0 0" material="label_mat"/>
    <light directional="true" dir="{'%.6f %.6f %.6f' % view_dir}" diffuse="1 1 1"
           specular="0 0 0" ambient="0.2 0.2 0.2"/>
    <camera name="scan_cam" pos="{'%.9f %.9f %.9f' % cam_pos}" xyaxes="{xyaxes}"
            fovy="{fovy_deg:.6f}"/>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml, assets={"label.png": label_png_bytes})
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "scan_cam")
    renderer = mujoco.Renderer(model, height=height_px, width=width_px)
    try:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render()
    finally:
        renderer.close()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
