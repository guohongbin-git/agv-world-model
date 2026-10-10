"""把 T04 取放闭环的**物理位姿**导出成 CSV, 供 Blender 照片级渲染驱动。

物理与 tests 完全同源 (roundtrip_plan + CellController 控制周期), 这里只
逐帧采样刚体世界位姿, 不做任何渲染 —— 运动是真值, Blender 端只是贴皮。

用法:
    PYTHONPATH=. ./.venv/bin/python scripts/export_poses.py

输出 /tmp/pbr/poses.csv:
    frame,time,seg,chassis_*(x3),chassis_q(x4),carriage_*(x3),pallet_*(x3),pallet_q(x4)
四元数为 MuJoCo wxyz 约定, Blender 端转换。
"""
from __future__ import annotations

import csv

import mujoco

from sim.forklift_cell import ForkliftCell
from sim.controllers import CellController

FPS = 30
OUT = "/tmp/pbr/poses.csv"
BODIES = ("chassis", "carriage", "pallet")


def main() -> int:
    cell = ForkliftCell(timestep_s=0.0005)
    c = CellController(cell)
    d = cell.data
    d.ctrl[cell.a_x] = cell.k["start_axle_x"] - cell.q0_x
    d.ctrl[cell.a_lift] = 0.0
    for _ in range(500):
        mujoco.mj_step(cell.model, d)

    rows: list[list[float]] = []
    next_frame = d.time + 1.0 / FPS

    def sample(seg: int) -> None:
        nonlocal next_frame
        mujoco.mj_forward(cell.model, d)
        row = [len(rows), round(d.time, 4), seg]
        for b in BODIES:
            bid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_BODY, b)
            row += list(d.xpos[bid]) + list(d.xquat[bid])
        rows.append(row)
        next_frame += 1.0 / FPS

    sample(0)
    for i, (idx, spec) in enumerate(cell.roundtrip_plan()):
        ramp = spec() if callable(spec) else spec
        q0 = cell.q0_x if idx == cell.a_x else 0.0
        t_seg = d.time
        for _ in range(int(ramp.duration * c.CTRL_HZ) + 1):
            d.ctrl[idx] = ramp.at(d.time - t_seg) - q0
            for _ in range(c.substeps):
                mujoco.mj_step(cell.model, d)
                if d.time >= next_frame:
                    sample(i + 1)
        for _ in range(int(0.4 * c.CTRL_HZ * c.substeps)):
            mujoco.mj_step(cell.model, d)
            if d.time >= next_frame:
                sample(i + 1)
    # 结尾定格 2s
    for _ in range(FPS * 2):
        mujoco.mj_step(cell.model, d)
        if d.time >= next_frame:
            sample(12)

    cols = (["frame", "time", "seg"]
            + [f"{b}_{p}" for b in BODIES for p in ("x", "y", "z", "qw", "qx", "qy", "qz")])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"已写 {OUT}: {len(rows)} 帧, 末态 t={d.time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
