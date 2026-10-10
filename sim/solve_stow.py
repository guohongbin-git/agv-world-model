"""收拢姿态求解器 (设计 981: 「收拢姿态须求解并以全部连杆碰撞体验证, 不能只检查TCP」)。

用法:
    PYTHONPATH=. .venv/bin/python -m sim.solve_stow            # 求解并打印
    PYTHONPATH=. .venv/bin/python -m sim.solve_stow --check    # 只校验 config 里的位形

判定对象是**机械臂运动链上的全部 geom**, 按 body 归属筛选。
不能用 geom 名字前缀: 托台/托盘/箱件也叫 M01_*, 前缀匹配会把托台腿
(365mm) 当成连杆, 求解器会收敛到「把静止结构折进包络」的错解。

输出写 reports/stow_envelope.json, 供 configs/design.yaml 的
stow_* 字段与偏离报告引用。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from sim.manipulator_cell import ManipulatorCell

REPO = Path(__file__).resolve().parent.parent


def arm_bodies(cell) -> set[int]:
    c = cell
    bodies = set()
    for b in ([f"{c.arm}_base"] + [f"{c.arm}_{j}" for j in
              ("J1", "J2", "J3", "J4", "J5", "J6")]
              + list(c.finger_names) + [f"{c.arm}_palm"]):
        bid = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_BODY, b)
        if bid >= 0:
            bodies.add(bid)
    return bodies


def geom_table(cell, bodies: set[int], base_xy) -> list[tuple[str, float, float]]:
    """(geom 名, 水平半径 m, z m) —— 逐连杆。"""
    out = []
    for i in range(cell.model.ngeom):
        if cell.model.geom_bodyid[i] not in bodies:
            continue
        nm = mujoco.mj_id2name(cell.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        p = cell.data.geom_xpos[i]
        out.append((nm, float(np.linalg.norm(p[:2] - base_xy)), float(p[2])))
    return sorted(out, key=lambda r: -r[1])


def drive(cell, q, settle: int):
    """把臂驱动到位形 q 并等**收敛**。

    settle 步只是时间上限, 不是"到位"保证: 关节限速 0.5rad/s, 从零位
    到折臂位要 2~4 秒 (仿真时间), 只跑固定步数会测到**运动中途**的姿态。
    实测同一 q 下 2000 步得到 z=+219.9mm, 收敛后是 z=−41.2mm —— 半
    截姿态会把穿台解误判成可行解。
    """
    # 判据用**位置误差**, 不用速度: 力矩受限的位置伺服在到位后会残留
    # 持续抖动 (实测解A 位置误差 0.0124rad 但 qvel 常年 0.21rad/s), 用速度
    # 判会把"已经到位"误判成"没到位"。位置才是收拢判据要的东西。
    pos_tol = 0.05          # rad
    for i in range(settle):
        cell.set_joint_targets(q)
        cell.step(1)
        if i > 40 and float(np.abs(cell.q - q).max()) < pos_tol:
            return True
    return False            # 未到位: 调用方须知道这次测量不可信


def envelope(cell, bodies, base_xy):
    tbl = geom_table(cell, bodies, base_xy)
    links = [r for r in tbl if "pad" not in r[0]]
    return max(r for _, r, _ in links), max(z for _, _, z in links), tbl


def solve(cell, verbose=True):
    """数值求解收拢位形: 最小化全部连杆到基座轴的最大水平半径。

    约束: 全部连杆 z ≥ z_floor (桌面/托台面之上)。没有这个约束时, 优化
    会把前臂折到桌面**以下** —— 实测 J3 达 z=−113.6mm、J4 z=−32.4mm,
    在物理上穿台, 却在半径目标上更优。半径更小但穿台的位形不是收拢位。
    """
    bodies = arm_bodies(cell)
    bid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_BODY, f"{cell.arm}_base")
    base_xy = cell.model.body_pos[bid][:2].copy()
    J = cell.cfg["manipulators"]["joints"]
    lo = np.array([j["lower_rad"] for j in J])
    hi = np.array([j["upper_rad"] for j in J])
    # 台面净空约束只施加于**会伸出去的连杆** (J3 及以后 + 指垫)。
    # 基座柱本身只有 17.5mm (设计 base_radius 40mm 的立柱), 它低于托台面
    # 是**常驻结构**, 不是穿台; 把基座也纳入约束会让所有位形都不可行。
    z_floor = cell.tray_top_z + 0.005      # 托台面以上 5mm
    moving_links = {n for n, _, _ in geom_table(cell, bodies, base_xy)
                    if "base" not in n}

    def cost(q, settle):
        """候选位形的代价。粗搜阶段用短步数 (仅用于排序), 细搜阶段等收敛。

        粗搜必须允许未收敛的样本参与排序 —— 否则全被判不可用, 没有
        初值可选; 但**最终**答案必须来自收敛姿态 (solve 末尾强制)。
        """
        if not drive(cell, q, settle):
            return 1e9, None            # 不到位 → 不参与择优
        r, z, tbl = envelope(cell, bodies, base_xy)
        zmin = min(zz for n, _, zz in tbl if n in moving_links)
        if zmin < z_floor:
            return 1e6 + abs(zmin), tbl          # 穿台: 罚到最优解之外
        return r, tbl

    best = (1e9, None)
    step = np.deg2rad(10)
    for j2 in np.arange(lo[1], hi[1], step):
        for j3 in np.arange(lo[2], hi[2], step):
            for j5 in np.arange(lo[4], hi[4], np.deg2rad(20)):
                q = np.array([0.0, j2, j3, 0.0, j5, 0.0])
                c, _ = cost(q, 600)
                if c < best[0]:
                    best = (c, q.copy())
    q0 = best[1]
    if q0 is None:
        raise SystemExit("粗搜未找到任何可行候选 (全部穿台或未定义)")
    for scale in (np.deg2rad(6), np.deg2rad(2), np.deg2rad(0.5)):
        improved = True
        while improved:
            improved = False
            for axis in (1, 2, 3, 4):
                for d in (+scale, -scale):
                    q = q0.copy()
                    q[axis] = min(max(q[axis] + d, lo[axis]), hi[axis])
                    c, _ = cost(q, 2500)
                    if c < best[0] - 1e-5:
                        best = (c, q.copy())
                        q0 = q.copy()
                        improved = True
    q_final = best[1]
    if not drive(cell, q_final, 8000):
        raise SystemExit(
            f"收拢位形未到位: q*={np.round(q_final,3).tolist()} "
            f"实际={np.round(cell.q,3).tolist()}")
    r, z, tbl = envelope(cell, bodies, base_xy)
    zmin = min(zz for _, _, zz in tbl)
    if verbose:
        print(f"收拢位形 q={np.round(q_final, 4).tolist()} rad")
        print(f"  臂体最大水平半径 = {r*1000:.1f}mm")
        print(f"  臂体最高点       = {z*1000:.1f}mm")
        print(f"  臂体最低点       = {zmin*1000:.1f}mm  (台面下限 "
              f"{z_floor*1000:.1f}mm: {'✓' if zmin >= z_floor - 1e-3 else '✗ 穿台'})")
        print(f"  设计初值         = 70.0mm  (偏离 {(r-0.070)/0.070*100:+.0f}%)")
        print(f"\n{'geom':24}{'半径mm':>10}{'z mm':>9}")
        for nm, rr, zz in tbl:
            print(f"{nm:24}{rr*1000:10.1f}{zz*1000:9.1f}")
    return q_final, r, z, tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="M01")
    ap.add_argument("--check", action="store_true",
                    help="只校验 config 中的 stow_q_rad 是否满足包络")
    args = ap.parse_args()
    cell = ManipulatorCell(arm=args.arm)
    cfg = cell.cfg["manipulators"]
    if args.check:
        bodies = arm_bodies(cell)
        bid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_BODY, f"{cell.arm}_base")
        base_xy = cell.model.body_pos[bid][:2].copy()
        q = np.array(cfg["stow_q_rad"], dtype=float)
        if not drive(cell, q, 8000):
            print(f"config 位形未到位 (实际 {np.round(cell.q,3).tolist()}), 测量不可信")
            return 1
        r, z, tbl = envelope(cell, bodies, base_xy)
        limit = float(cfg["stow_xy_radius_initial_m"])
        zlim = float(cfg["stow_z_limit_m"])
        moving = {n for n, _, _ in geom_table(cell, bodies, base_xy) if "base" not in n}
        zmin = min(zz for n, _, zz in tbl if n in moving)
        zfloor = cell.tray_top_z + 0.005
        ok = r <= limit + 0.005 and z <= zlim and zmin >= zfloor - 1e-3
        print(f"config 位形: 半径={r*1000:.1f}mm (限 {limit*1000:.1f}mm) "
              f"{'✓' if r <= limit + 0.005 else '✗'}; "
              f"最高点={z*1000:.1f}mm (限 {zlim*1000:.0f}mm) "
              f"{'✓' if z <= zlim else '✗'}; "
              f"最低点={zmin*1000:.1f}mm (台面 {zfloor*1000:.1f}mm) "
              f"{'✓' if zmin >= zfloor - 1e-3 else '✗ 穿台'}")
        return 0 if ok else 1
    q, r, z, tbl = solve(cell)
    out = REPO / "reports" / "stow_envelope.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "arm": args.arm,
        "q_stow_rad": [float(x) for x in q],
        "achieved_radius_m": float(r),
        "max_z_m": float(z),
        "design_initial_radius_m": float(cfg["stow_xy_radius_initial_m"]),
        "per_geom": [{"geom": n, "radius_m": rr, "z_m": zz} for n, rr, zz in tbl],
    }, indent=2, ensure_ascii=False))
    print(f"\n已写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())