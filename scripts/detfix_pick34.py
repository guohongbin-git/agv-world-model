"""detfix 插桩复现 v2: 直跑 MissionRunner (test_8task_determinism 同任务, seed=7)。

只读包装 mujoco.mj_step (不改物理/控制, 不破确定性):
- mjWARN_BADQACC 计数增量 + qacc 有限性 → 发散定位 (DOF/体)。
- 1kHz 环形缓冲 (15s): 两目标托盘 p/v/a、涉托接触对 (任意对手)、全场
  最大接触力及其 geom 对、F02/F04 位姿。触发器落盘后继续:
  * 托盘 z<0.055 (脱叉坠落) / 离叉车 0.3m
  * 发散 (停止)
- 5Hz 全场快照: 8 车位姿 + 活动段。
用法: ./.venv/bin/python scripts/detfix_repro.py out.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque

import mujoco
import numpy as np

sys.path.insert(0, ".")
from sim.mission import MissionRunner                     # noqa: E402
from sim.pallet_support import ledger_from_cell           # noqa: E402
from sim.warehouse_cell import WarehouseCell              # noqa: E402

RING_S = 90.0
OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/detfix/analysis.json"


PICK_SLOTS = [("R34-S6", 0), ("R41-S2", 1), ("R44-S2", 1)]
PUT_SLOTS = [("R12-S6", 0), ("R13-S0", 0), ("R31-S4", 1), ("R31-S5", 0)]


def build_tasks(cell, led):
    return [{"kind": "pick", "slot": "R34-S6", "level": 0,
             "stand": "H02"}]


class BlowUp(Exception):
    pass


class Recorder:
    def __init__(self, cell: WarehouseCell, mr: MissionRunner):
        self.cell, self.mr = cell, mr
        m = cell.model
        self.warn = mujoco.mjtWarning.mjWARN_BADQACC
        self.prev_warn = 0
        self.blow = None
        self.triggers: list[dict] = []
        self.jobs_log: list[dict] = []
        self.snapshots: list[dict] = []          # 5Hz 全场
        self.ring: deque = deque(maxlen=int(RING_S * 1000))
        self._last_job_t = -1.0
        self._last_snap_t = -1.0
        self.gname = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                      for g in range(m.ngeom)}
        self.pallets = ("pallet_R34_S6_L0", "pallet_H05")
        self.qadr, self.vadr = {}, {}
        for pn in self.pallets:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"free_{pn}")
            self.qadr[pn] = int(m.jnt_qposadr[j])
            self.vadr[pn] = int(m.jnt_dofadr[j])
        self.pg = {pn: set(cell.pallets[pn].gids) for pn in self.pallets}
        self.fbuf = np.zeros(6)
        self.t0_wall = time.time()
        self._last_t = cell.data.time

    def after_step(self, m, d):
        w = d.warning[self.warn].number
        if self.blow is None and (w > self.prev_warn
                                  or not np.isfinite(d.qacc).all()):
            dof = int(d.warning[self.warn].lastinfo)
            body = int(m.dof_bodyid[dof]) if 0 <= dof < m.nv else -1
            self.blow = {"t": float(self._last_t), "dof": dof,
                         "body": mujoco.mj_id2name(
                             m, mujoco.mjtObj.mjOBJ_BODY, body)
                         if body >= 0 else "?"}
            self._fire("blow", self.blow["t"])
            raise BlowUp(json.dumps(self.blow))
        self.prev_warn = w
        self._last_t = float(d.time)
        self._n = getattr(self, "_n", 0) + 1
        if self._n % 5 == 0:                 # 200Hz 采样 (事件演化 ~50ms 级)
            self._ring(m, d)
        self._check_triggers(d)
        if d.time - self._last_snap_t >= 0.2:
            self._last_snap_t = float(d.time)
            self._snap(d)
        if d.time - self._last_job_t >= 0.1:
            self._last_job_t = float(d.time)
            self._log_jobs(d)

    # 1kHz 环形: 托盘态 + 涉托接触 + 全场最大力对
    def _ring(self, m, d):
        t = float(d.time)
        row = {"t": round(t, 4)}
        for pn in self.pallets:
            qa, va = self.qadr[pn], self.vadr[pn]
            row[pn] = ([round(float(x), 5) for x in d.qpos[qa:qa + 3]],
                       [round(float(x), 2) for x in d.qvel[va:va + 6]])
        fmax, fpair, cs = 0.0, None, []
        pg = self.pg
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            mujoco.mj_contactForce(m, d, i, self.fbuf)
            fn = float(np.linalg.norm(self.fbuf[:3]))
            if fn > fmax:
                fmax, fpair = fn, (self.gname[g1], self.gname[g2])
            if fn > 0.4 and any(g in pgs for pgs in pg.values()
                                for g in (g1, g2)):
                cs.append([self.gname[g1], self.gname[g2],
                           round(fn, 2), round(float(c.dist) * 1000, 3)])
        row["fmax"] = round(fmax, 2)
        row["fpair"] = fpair
        row["cs"] = cs
        f2 = self.cell.forklifts.get("F02")
        if f2 is not None:
            row["F02"] = [round(x, 4) for x in (*f2.xy(), f2.q("lift"))]
        self.ring.append(row)

    def _check_triggers(self, d):
        for pn, st in (("pallet_R34_S6_L0", self.trig_s7),
                       ("pallet_H05", self.trig_s6)):
            if st.get("done"):
                continue
            z = d.qpos[self.qadr[pn] + 2]
            if z < 0.045:                       # 坠地 (运输高度 0.052 以深)
                self._fire(pn + "_fall", float(d.time))
                st["done"] = True

    trig_s7: dict = {}
    trig_s6: dict = {}

    def _fire(self, tag: str, t: float):
        rows = list(self.ring)
        tail = [r for r in rows if r["t"] >= t - 6.0]
        self.triggers.append({"tag": tag, "t": round(t, 4),
                              "ring_tail": tail})
        print(f"\n=== TRIGGER {tag} at t={t:.3f} "
              f"(wall {time.time() - self.t0_wall:.0f}s) ===", flush=True)
        for r in tail[-40::4]:
            print(f"  t={r['t']:.3f} " + " ".join(
                f"{pn[7:]}=({r[pn][0][0]:+.3f},{r[pn][0][1]:+.3f},"
                f"{r[pn][0][2]:.4f})" for pn in self.pallets)
                + f" fmax={r['fmax']:.3g} fpair={r['fpair']}")
            for c in r["cs"]:
                print(f"      {c[0]} <-> {c[1]} F={c[2]}N pen={c[3]}mm")

    def _snap(self, d):
        veh = {}
        for vid, view in self.cell.veh.items():
            aux = "lift" if vid.startswith("F") else "pin"
            veh[vid] = [round(float(x), 4)
                        for x in (*view.xy(), view.q("yaw"), view.q(aux))]
        self.snapshots.append({"t": round(float(d.time), 2), "veh": veh})

    def _log_jobs(self, d):
        snap = {}
        for vid, job in self.mr.jobs.items():
            if job is None:
                continue
            cur = self.mr.runner.cur[vid]
            ramp = None
            if cur is not None:
                ramp = {"aid": int(cur[0]),
                        "target": round(float(cur[1].target), 4),
                        "te": round(float(d.time - cur[4]), 3)}
            snap[vid] = {"kind": job.task["kind"],
                         "arg": job.task.get("slot") or job.task.get("stand")
                         or job.task.get("dock"),
                         "phase": job.phase, "mode": job.mode, "ramp": ramp}
        self.jobs_log.append({"t": round(float(d.time), 2), "jobs": snap})


def main():
    cell = WarehouseCell(timestep_s=0.001, seed=7,
                         seed_slots=[f"{s}-L{l}" for s, l in PICK_SLOTS],
                         preload_stands=("H01", "H05"))
    cell.warmup(0.5)
    led = ledger_from_cell(cell)
    tasks = build_tasks(cell, led)
    print("tasks:", json.dumps(tasks))
    mr = MissionRunner(cell)
    rec = Recorder(cell, mr)
    orig_step = mujoco.mj_step
    nstep = [0]

    def step(m, d):
        orig_step(m, d)
        nstep[0] += 1
        rec.after_step(m, d)
        if nstep[0] % 200000 == 0:
            print(f"  [wall {time.time() - rec.t0_wall:6.0f}s] "
                  f"sim t={d.time:8.2f}s", flush=True)

    mujoco.mj_step = step
    blow = None
    try:
        res = mr.run(tasks, max_s=2400)
        print("mission done; op_errors:", res["op_errors"])
        print("audit max_pen:", res["audit"]["max_pen"],
              "violations:", res["audit"]["violations"][:5])
    except BlowUp as e:
        blow = json.loads(str(e))
        print("BlowUp:", blow)
    finally:
        mujoco.mj_step = orig_step
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    slim = {"blow": blow, "tasks": tasks, "triggers": rec.triggers,
            "jobs_log": rec.jobs_log, "snapshots": rec.snapshots}
    with open(OUT, "w") as f:
        json.dump(slim, f)
    print(f"saved {OUT} wall={time.time() - rec.t0_wall:.0f}s")


if __name__ == "__main__":
    main()
