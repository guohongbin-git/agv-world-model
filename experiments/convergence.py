#!/usr/bin/env python3
"""R1 物理收敛实验 (论文 II 第 3 章)。

问题: 行为判据 (65mm 插叉 / 5mm/s 到位 / 托盘支撑比 ≈ 1) 是否依赖数值参数
(时间步)? 若结论随 dt 漂移, 上层行为验证即无效。

方法: 生产同款执行路径 —— ForkliftCell.roundtrip_plan() 的 11 段抓放计划 +
CellController.run_sequence —— 扫描 dt ∈ {0.002, 0.001, 0.0005, 0.00025},
以最细 dt 为参考解, 度量终态:
  - 托盘底面 z 误差 (mm)          —— 放托精度 (判据: 台面 45mm)
  - 托盘 roll/pitch (mrad)        —— 接触稳定性
  - 叉-托盘支撑载荷比             —— 承托成立 (判据 ≈ 1)
  - 最大接触穿透 (mm)             —— 刚体约束求解质量
  - 墙钟秒                        —— 实时性成本

判定: 选型 dt 须使终态 z 误差 < 1mm 且姿态差 < 2mrad (远小于 5mm 行为判据)。
"""
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.controllers import CellController  # noqa: E402
from sim.forklift_cell import ForkliftCell  # noqa: E402

DTS = [0.002, 0.001, 0.0005, 0.00025]


def run(dt: float) -> dict:
    c = ForkliftCell(timestep_s=dt, levels=[0.100])
    ctrl = CellController(c)
    t0 = time.perf_counter()
    # 动态段探针: 包装 Ramp.at, 捕捉带载转运 (车辆运动) 中的最大倾角
    max_roll_dyn = max_pitch_dyn = 0.0
    peak_f = 0.0
    f6 = np.zeros(6)

    def probe():
        nonlocal max_roll_dyn, max_pitch_dyn, peak_f
        q = c.data.xquat[c.pallet_bid]
        r = abs(float(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]),
                                 1 - 2 * (q[1] ** 2 + q[2] ** 2))))
        p = abs(float(np.arcsin(np.clip(
            2 * (q[0] * q[2] - q[3] * q[1]), -1, 1))))
        max_roll_dyn = max(max_roll_dyn, r)
        max_pitch_dyn = max(max_pitch_dyn, p)
        for i in range(c.data.ncon):
            con = c.data.contact[i]
            gid = {con.geom1, con.geom2}
            if gid & c.fork_gids and gid & set(c.pallet_geoms):
                mujoco.mj_contactForce(c.model, c.data, i, f6)
                peak_f = max(peak_f, float(abs(f6[0])))

    def wrap(spec):
        """Ramp.at 挂探针 (CTRL_HZ 频率, 足以捕捉动态峰值)。"""
        def build():
            rmp = spec() if callable(spec) else spec
            orig_at = rmp.at
            def at(t):
                probe()
                return orig_at(t)
            rmp.at = at
            return rmp
        return build

    ctrl.run_sequence([(idx, wrap(sp)) for idx, sp in
                       c.roundtrip_plan()], settle_s=0.5)
    for _ in range(int(0.5 / dt)):                # 终态沉降
        mujoco.mj_step(c.model, c.data)
    wall = time.perf_counter() - t0

    pos, quat = c.pallet_pose
    roll = float(np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]),
                            1 - 2 * (quat[1] ** 2 + quat[2] ** 2)))
    pitch = float(np.arcsin(np.clip(
        2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1)))
    return {"dt": dt,
            "pal_z_mm": float(c.pallet_bottom_z) * 1e3,
            "roll_mrad": roll * 1e3, "pitch_mrad": pitch * 1e3,
            "support_ratio": round(float(c.support_load_ratio("beam")), 3),
            "pen_mm": float(c.max_penetration()) * 1e3,
            "dyn_pitch_mrad": max_pitch_dyn * 1e3,
            "peak_fork_n": round(peak_f, 2),
            "wall_s": round(wall, 1)}


def main() -> None:
    rows = [run(dt) for dt in DTS]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))

    ref = rows[-1]
    print("\n=== 收敛 (vs dt=%.5f) ===" % ref["dt"])
    ok = True
    for r in rows[:-1]:
        dz = abs(r["pal_z_mm"] - ref["pal_z_mm"])
        drot = max(abs(r["roll_mrad"] - ref["roll_mrad"]),
                   abs(r["pitch_mrad"] - ref["pitch_mrad"]))
        verdict = "PASS" if dz < 1.0 and drot < 2.0 else "FAIL"
        ok &= verdict == "PASS"
        print(f"dt={r['dt']:.5f}: 托盘z差 {dz:.2f}mm, 姿态差 {drot:.2f}mrad, "
              f"穿透 {r['pen_mm']:.3f}mm, 墙钟 {r['wall_s']}s [{verdict}]")
    print(f"参考终态: 托盘底 z={ref['pal_z_mm']:.2f}mm (判据 45mm), "
          f"roll {ref['roll_mrad']:.2f} / pitch {ref['pitch_mrad']:.2f} mrad, "
          f"支撑比 {ref['support_ratio']}, 穿透 {ref['pen_mm']:.3f}mm")
    print("结论:", "dt ≤ %.5f 收敛, 行为判据与数值参数无关" % rows[-2]["dt"]
          if ok else "存在 dt 依赖 —— 需收紧选型")


if __name__ == "__main__":
    main()
