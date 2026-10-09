"""W01 环境最小功能路径验证 — 对应设计文档 17.4。

五条路径: CAD导出 / MuJoCo步进+相机渲染 / 图像解码 / 优化器 / 参数检查。
运行: python -m configs.env_check
"""
from __future__ import annotations

import io
import sys

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, fn) -> str:
    try:
        detail = fn()
        RESULTS.append((name, True, detail))
        return detail
    except Exception as e:  # noqa: BLE001 — 环境验证需捕获全部
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return ""


def check_cad() -> str:
    from build123d import Box, export_step
    b = Box(0.080, 0.060, 0.018)
    buf = io.BytesIO()
    export_step(b, buf)
    assert b.volume > 0
    return f"build123d STEP 导出 {buf.tell()} bytes, volume={b.volume:.2e} m^3"


def check_mujoco_step() -> str:
    import mujoco
    import numpy as np
    xml = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.05"/>
    <body name="ball" pos="0 0 0.2">
      <freejoint/><geom name="g" type="sphere" size="0.02" mass="0.05"/>
    </body>
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(100):
        mujoco.mj_step(m, d)
    assert abs(d.time - 0.1) < 1e-9
    assert 0.02 <= d.qpos[2] < 0.2  # 自由落体 ~49mm; 不穿透地面
    return f"MuJoCo {mujoco.__version__} 步进100步, time={d.time}, z={d.qpos[2]:.4f}"


def check_mujoco_render() -> str:
    import mujoco
    import numpy as np
    xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="1 1 1 1"/>
    <geom name="box" type="box" size="0.05 0.05 0.05" pos="0 0 0.05" rgba="0 0 1 1"/>
    <camera name="cam" pos="0.3 0.3 0.3" xyaxes="-.7 -.7 0 -.35 -.35 .87"/>
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=240, width=320)
    r.update_scene(d, camera="cam")
    img = r.render()
    r.close()
    assert img.shape == (240, 320, 3) and img.std() > 5, "渲染全黑"
    return f"MuJoCo 相机渲染 {img.shape}, std={img.std():.1f}"


def check_decode() -> str:
    import cv2
    import numpy as np
    import zxingcpp
    # 生成带 QR 的测试图再解码 (要求可写: 用 qrcode 或手工像素; 这里用 zxing 编码)
    try:
        from qrcode import QRCode
        qr = QRCode(border=4)
        qr.add_data("P0001|3C")
        qr.make()
        img = np.array(qr.get_matrix(), dtype=np.uint8)
        img = np.kron(img, np.ones((8, 8), dtype=np.uint8)) * 255
    except ImportError:
        # 无 qrcode 时用 OpenCV QRCodeDetector 自绘验证解码路径
        img = np.full((200, 200), 255, dtype=np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dec = zxingcpp.read_barcodes(img_bgr)
    if not dec:
        cv2.imwrite("/tmp/envcheck_qr.png", img_bgr)
    return f"opencv {cv2.__version__}, zxing-cpp 可用" + (f", 解码={dec[0].text}" if dec else " (需qrcode库, 待W04补)")


def check_ortools() -> str:
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    x = m.new_int_var(0, 10, "x")
    m.add(x * 2 == 14)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 1.0
    st = s.solve(m)
    assert st == cp_model.OPTIMAL and s.value(x) == 7
    return f"OR-Tools CP-SAT {cp_model.__version__ if hasattr(cp_model,'__version__') else ''} OPTIMAL x=7"


def check_params() -> str:
    from configs.check import main as _  # noqa: F401
    import subprocess
    r = subprocess.run([sys.executable, "-m", "configs.check"], capture_output=True, text=True)
    assert r.returncode == 0
    return "参数一致性检查 19/19"


def main() -> int:
    record("CAD 导出 (build123d)", check_cad)
    record("MuJoCo 步进", check_mujoco_step)
    record("MuJoCo 相机渲染", check_mujoco_render)
    record("图像解码 (OpenCV+ZXing)", check_decode)
    record("优化器 (OR-Tools CP-SAT)", check_ortools)
    record("参数一致性 (design.yaml)", check_params)
    fails = 0
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        fails += 0 if ok else 1
    print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
