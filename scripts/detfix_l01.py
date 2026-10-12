"""detfix 定向 L0/L1 低位层 pick+putaway 验证 (Main 指定的 press 失败点位)。

出库失败列 (pick): R34-S6-L0, R41-S2-L1, R44-S2-L1
补货失败列 (putaway): R12-S6-L0, R13-S0-L0, R31-S4-L1, R31-S5-L0
构造: 3 个低位 pick → 交接台, 4 个 putaway (3 用刚取托 + 1 台上预载)。
断言: op_errors 空 / 托盘落位 ±5mm / 全车回 park / 无发散。
用法: ./.venv/bin/python scripts/detfix_l01.py
"""
from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, ".")
from sim.mission import MissionRunner                     # noqa: E402
from sim.pallet_support import ledger_from_cell           # noqa: E402
from sim.warehouse_cell import WarehouseCell, pallet_name_for  # noqa: E402

PICK_SLOTS = [("R34-S6", 0), ("R41-S2", 1), ("R44-S2", 1)]
PUT_SLOTS = [("R12-S6", 0), ("R13-S0", 0), ("R31-S4", 1), ("R31-S5", 0)]


def main():
    seed_slots = [f"{s}-L{l}" for s, l in PICK_SLOTS]
    cell = WarehouseCell(timestep_s=0.001, seed=7,
                         seed_slots=seed_slots,
                         preload_stands=("H01", "H05"))
    cell.warmup(0.5)
    led = ledger_from_cell(cell)
    tasks = []
    for i, (s, l) in enumerate(PICK_SLOTS):
        tasks.append({"kind": "pick", "slot": s, "level": l,
                      "stand": f"H0{i + 2}"})
    for i, (s, l) in enumerate(PUT_SLOTS):
        t = {"kind": "putaway", "slot": s, "level": l,
             "stand": f"H0{i + 2}"}
        if i < 3:
            t["after"] = [i]               # 用刚 pick 出的托
        tasks.append(t)                    # 第 4 个用 H05 预载托
    print("tasks:", json.dumps(tasks))
    mr = MissionRunner(cell)
    res = mr.run(tasks, max_s=2400)
    print("op_errors:", res["op_errors"])
    print("max_pen:", res["audit"]["max_pen"],
          "violations:", res["audit"]["violations"][:5])
    ok = not res["op_errors"]
    # 终位校验
    for (s, l), i in zip(PUT_SLOTS, range(4)):
        pn = (pallet_name_for(f"{PICK_SLOTS[i][0]}-L{PICK_SLOTS[i][1]}")
              if i < 3 else "pallet_H05")
        sx, sy, _, sz = cell.layout.slot_pose(f"{s}-L{l}")
        pos = cell.pallet_pose(pn)[0]
        err = math.hypot(pos[0] - sx, pos[1] - sy)
        print(f"put {s}-L{l} ← {pn}: dxy={err:.4f} dz={pos[2] - sz:+.4f}")
        ok = ok and err <= 0.005 and abs(pos[2] - sz) <= 0.002
    for vid in cell.veh:
        px, py, _ = cell.layout.p["park"][vid]
        vx, vy = cell.veh[vid].xy()
        e = math.hypot(vx - px, vy - py)
        ok = ok and e < 0.02
        if e >= 0.02:
            print(f"park fail {vid}: {e:.3f}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
