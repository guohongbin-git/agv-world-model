"""把 T04 三层取放闭环渲染成 MP4 (1280x720@30, 中文 HUD, 相机跟随)。

用法:
    PYTHONPATH=. ./.venv/bin/python scripts/render_roundtrip_mp4.py

输出 deliverables/roundtrip.mp4。物理与测试完全同源 (roundtrip_plan +
CellController 的控制周期/整定), 只是离线逐帧渲染, 不引入任何视觉专用状态。
"""
from __future__ import annotations

import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.forklift_cell import ForkliftCell
from sim.controllers import CellController

W, H, FPS = 1280, 720, 30
OUT = "deliverables/roundtrip.mp4"
SEG_NAMES = [
    "升至插叉高度", "接近货架前沿", "插叉 65mm", "举升离架",
    "直线退出站体", "降至运输高度", "转运至暂存站", "放托",
    "微降脱叉", "退叉撤离", "叉架复位",
]

_FONT_CANDS = ["/System/Library/Fonts/STHeiti Light.ttc",
               "/System/Library/Fonts/PingFang.ttc",
               "/System/Library/Fonts/Hiragino Sans GB.ttc",
               "/Library/Fonts/Arial Unicode.ttf"]


def _load_font(size: int):
    last = None
    for path in _FONT_CANDS:
        try:
            return ImageFont.truetype(path, size)
        except OSError as e:
            last = e
    raise RuntimeError(f"无可用中文字体: {last}")


def main() -> int:
    font = _load_font(24)
    small = _load_font(20)
    cell = ForkliftCell(timestep_s=0.0005)
    c = CellController(cell)
    d = cell.data
    d.ctrl[cell.a_x] = cell.k["start_axle_x"] - cell.q0_x
    d.ctrl[cell.a_lift] = 0.0
    for _ in range(500):
        mujoco.mj_step(cell.model, d)

    r = mujoco.Renderer(cell.model, H, W)
    cam = mujoco.MjvCamera()
    look = np.array([-0.15, 0.0, 0.25])
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT],
        stdin=subprocess.PIPE)

    next_frame = d.time + 1.0 / FPS

    def emit(seg_i: int) -> None:
        nonlocal next_frame
        tgt = np.array([(cell.axle_x + cell.pallet_pose[0][0]) / 2 + 0.05,
                        0.0, max(0.18, cell.fork_top_z * 0.55)])
        look[:] += 0.06 * (tgt - look)
        cam.lookat[:] = look
        cam.distance, cam.azimuth, cam.elevation = 0.82, 118.0, -13.0
        mujoco.mj_forward(cell.model, d)
        r.update_scene(d, camera=cam)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([0, H - 44, W, H], fill=(8, 10, 14, 178))
        name = SEG_NAMES[seg_i] if seg_i < len(SEG_NAMES) else ""
        dr.text((14, H - 36), f"T04 三层取放闭环 · 段{seg_i + 1}/11 {name}",
                font=font, fill=(255, 235, 180, 255))
        st = (f"t={d.time:5.1f}s  叉顶 {cell.fork_top_z * 1000:5.1f}mm  "
              f"托盘底 {cell.pallet_bottom_z * 1000:5.1f}mm  "
              f"托盘x {cell.pallet_pose[0][0] * 1000:+6.1f}mm  "
              f"叉承托 {cell.support_load_ratio('fork'):.2f}")
        dr.text((W - 640, H - 34), st, font=small, fill=(210, 220, 235, 255))
        proc.stdin.write(img.tobytes())
        next_frame += 1.0 / FPS

    emit(0)
    for i, (idx, spec) in enumerate(cell.roundtrip_plan()):
        ramp = spec() if callable(spec) else spec
        q0 = cell.q0_x if idx == cell.a_x else 0.0
        t_seg = d.time
        for _ in range(int(ramp.duration * c.CTRL_HZ) + 1):
            d.ctrl[idx] = ramp.at(d.time - t_seg) - q0
            for _ in range(c.substeps):
                mujoco.mj_step(cell.model, d)
                if d.time >= next_frame:
                    emit(i)
        for _ in range(int(0.4 * c.CTRL_HZ * c.substeps)):
            mujoco.mj_step(cell.model, d)
            if d.time >= next_frame:
                emit(i)
    for _ in range(FPS * 2):                     # 结尾定格 2s
        mujoco.mj_step(cell.model, d)
        emit(len(SEG_NAMES))
    r.close()
    proc.stdin.close()
    proc.wait()
    print(f"已写 {OUT} (帧≈{int(d.time * FPS)})")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
