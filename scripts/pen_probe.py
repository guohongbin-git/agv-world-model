"""zero_collision 失败定位: 跑 fleet8 fixture, 追踪最大穿透接触对。

只读包装 mj_step 记录 (时间, 穿透, geom 对, 两车体) —— 不改物理。
用法: PYTHONPATH=. ./.venv/bin/python scripts/pen_probe.py
"""
from __future__ import annotations

import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from sim.mission import MissionRunner                       # noqa: E402
from sim.pallet_support import ledger_from_cell             # noqa: E402
from sim.warehouse_cell import WarehouseCell                  # noqa: E402

worst = []

orig_step = mujoco.mj_step


def step(m, d):
    orig_step(m, d)
    if d.ncon:
        pen = d.contact.dist.min()
        if pen < -1e-4:
            worst.append((round(float(d.time), 2), float(pen)))


def main():
    cell = WarehouseCell(timestep_s=0.001, seed=7)
    cell.warmup(0.5)
    led = ledger_from_cell(cell)
    slots = sorted(k for k, v in led.loc.items()
                   if v.startswith("slot:") and "-S" in v)
    occupied = {v for v in led.loc.values() if v.startswith("stand:")}
    free_nodes = sorted(f"stand:{s}" for s in cell.layout.p["handoff_ids"])
    free_stands = sorted(n.split(":", 1)[1] for n in free_nodes
                         if n not in occupied)[:4]
    picks, used = [], set()
    for name in slots:
        if len(picks) == 4:
            break
        slot = led.loc[name][len("slot:"):]
        if slot in used:
            continue
        used.add(slot)
        picks.append({"kind": "pick", "slot": slot, "level": int(name[-1]),
                      "stand": free_stands[len(picks)]})
    stands = [t["stand"] for t in picks]
    docks = ("D_G11", "D_G22", "D_G33", "D_G44")
    tasks = picks + [{"kind": "haul", "stand": stands[i], "dock": docks[i],
                      "after": [i]} for i in range(4)]

    # 每 200 步检查一次最大穿透的接触对 (避免每步开销)
    deep = []
    cnt = [0]

    def step2(m, d):
        orig_step(m, d)
        cnt[0] += 1
        if d.ncon and cnt[0] % 200 == 0:
            i = int(np.argmin(d.contact.dist))
            if d.contact.dist[i] < -5e-5:
                c = d.contact[i]
                n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                       int(c.geom1)) or "?"
                n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                       int(c.geom2)) or "?"
                deep.append((round(float(d.time), 2), float(c.dist), n1, n2))

    mujoco.mj_step = step2
    mr = MissionRunner(cell)
    res = mr.run(tasks, max_s=2400)
    mujoco.mj_step = orig_step
    print("op_errors:", res.get("op_errors"))
    # 汇总最深的 12 条 (按穿透排序), 并打印时间聚簇
    seen = {}
    for t, d_, a, b in deep:
        key = tuple(sorted((a, b)))
        if key not in seen or d_ < seen[key][1]:
            seen[key] = (t, d_, a, b)
    print("\n最深穿透 (按 geom 对):")
    for k, (t, d_, a, b) in sorted(seen.items(), key=lambda kv: kv[1][1])[:12]:
        print(f"  t={t:7.2f} pen={-d_*1e3:7.2f}mm  {a} ↔ {b}")
    n = len(deep)
    print(f"\n总采样 {n} 条穿透记录")


if __name__ == "__main__":
    main()