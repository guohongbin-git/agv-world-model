"""车-车碰撞对因: fleet8 场景下对齐 (交通事件流 × 两车轨迹 × 接触)。

定位 pen_probe 报的 A02_base↔F04_chassis (t≈467, 11mm):
- 交通事件 (claim_ok/claim_wait/arrival/release) 带仿真时间
- 涉事车辆 xy/yaw 每 0.1s
- 穿透 >0.1mm 的接触对带时间

用法: PYTHONPATH=. ./.venv/bin/python scripts/pen_align.py
"""
from __future__ import annotations

import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from sim.mission import MissionRunner                        # noqa: E402
from sim.pallet_support import ledger_from_cell              # noqa: E402
from sim.warehouse_cell import WarehouseCell                   # noqa: E402

orig_step = mujoco.mj_step


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

    mr = MissionRunner(cell)
    tm = mr.traffic
    trace = []
    pen = []
    cnt = [0]

    def step(m, d):
        orig_step(m, d)
        cnt[0] += 1
        if cnt[0] % 100 == 0:                     # 0.1s
            row = [round(float(d.time), 2)]
            for v in ("F03", "F04", "A02"):
                p = cell.veh[v].xy()
                row += [round(p[0], 3), round(p[1], 3)]
            trace.append(row)
        if d.ncon and cnt[0] % 100 == 0:
            for i in range(d.ncon):
                if d.contact.dist[i] < -1e-4:
                    c = d.contact[i]
                    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                           int(c.geom1)) or "?"
                    n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                           int(c.geom2)) or "?"
                    if ("A02" in n1 + n2) or ("F04" in n1 + n2) or \
                            ("F03" in n1 + n2):
                        pen.append((round(float(d.time), 2),
                                    round(float(c.dist) * 1e3, 2),
                                    n1, n2))

    mujoco.mj_step = step
    res = mr.run(tasks, max_s=2400)
    mujoco.mj_step = orig_step

    print("op_errors:", res.get("op_errors"))
    print("\n=== 穿透记录 (涉 F03/F04/A02, >0.1mm) ===")
    for t, d_, a, b in pen[:40]:
        print(f"  t={t:7.2f} {d_:7.2f}mm  {a} ↔ {b}")

    print("\n=== 交通事件 (t=430..500) ===")
    for t, vid, kind, info in tm.events:
        if 430 <= t <= 500:
            print(f"  t={t:7.2f} {vid:4s} {kind:11s} {info}")

    print("\n=== 轨迹 (t=460..475, F03/F04/A02) ===")
    for row in trace:
        if 460 <= row[0] <= 475:
            print(f"  t={row[0]:7.2f} F03=({row[1]},{row[2]}) "
                  f"F04=({row[3]},{row[4]}) A02=({row[5]},{row[6]})")


if __name__ == "__main__":
    main()