"""T04 三层取放实时演示 — MuJoCo 交互窗口。

用法:
    PYTHONPATH=. ./.venv/bin/python scripts/watch_roundtrip.py [倍速]

窗口操作: 左键拖=轨道, 滚轮=缩放, 右键拖=平移, 双击=对焦, ESC=跟随切换。
每段完成打印叉顶/托盘位姿, 跑完保持窗口打开直到用户关闭。
"""
from __future__ import annotations

import sys
import time

import mujoco.viewer

from sim.forklift_cell import ForkliftCell
from sim.controllers import CellController


def main() -> int:
    speed = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    cell = ForkliftCell(timestep_s=0.0005)
    c = CellController(cell)
    d = cell.data
    d.ctrl[cell.a_x] = cell.k["start_axle_x"] - cell.q0_x
    d.ctrl[cell.a_lift] = 0.0
    for _ in range(500):                       # 预沉降
        mujoco.mj_step(cell.model, d)

    plan = cell.roundtrip_plan()
    wall0 = time.monotonic()
    sim0 = d.time
    with mujoco.viewer.launch_passive(cell.model, d) as v:
        for i, (idx, spec) in enumerate(plan, 1):
            ramp = spec() if callable(spec) else spec
            q0 = cell.q0_x if idx == cell.a_x else 0.0
            t_seg = d.time
            n = int(ramp.duration * c.CTRL_HZ) + 1
            for _ in range(n):
                d.ctrl[idx] = ramp.at(d.time - t_seg) - q0
                for _ in range(c.substeps):
                    mujoco.mj_step(cell.model, d)
                v.sync()
                lag = (d.time - sim0) / speed - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
            # 段间整定
            for _ in range(int(0.4 * c.CTRL_HZ * c.substeps)):
                mujoco.mj_step(cell.model, d)
            v.sync()
            print(f"段{i:>2}/{len(plan)}: 叉顶={cell.fork_top_z*1000:6.1f}mm "
                  f"托盘底={cell.pallet_bottom_z*1000:6.1f}mm "
                  f"托盘x={cell.pallet_pose[0][0]*1000:+7.1f}mm "
                  f"叉承托={cell.support_load_ratio('fork'):.2f}", flush=True)
        print("闭环完成, 窗口保持 (关闭即退出)", flush=True)
        while v.is_running():
            mujoco.mj_step(cell.model, d)
            v.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
