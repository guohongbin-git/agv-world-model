#!/usr/bin/env python
"""W3b — 30 分钟连续压测: demand 驱动 8 车矩阵仓库 (:8400 服务 headless).

用法:
    ./.venv/bin/python scripts/press.py                  # 30 min, 全速仿真
    ./.venv/bin/python scripts/press.py --duration 300   # 5 min 冒烟压测

驱动: MatrixService (与 :8400 同一代码路径, 不起 HTTP);
demand 引擎按需求曲线持续出单 (出库 pick+haul / 补货 fetch+putaway),
指标每 60s 快照一次 (吞吐/利用率/锁等待/内存 RSS/事件深度), 收尾
drain (停单 → 在飞任务清空) 后判据:

  [mem]       RSS 首尾增长 < 10%
  [deadlock]  全程无 >180s 无进展; drain 后 pending+active == 0; mode ≠ failed
  [collision] max_pen < 5e-4 m 且穿透违规 == 0
  [ops]       物理任务失败 == 0 (op_errors 为空, W2a 相位机健康)

报告: reports/press_matrix_<时间戳>.json (含全部快照序列) + 控制台摘要。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from twin.matrix_server import MatrixService  # noqa: E402

SNAPSHOT_S = 60.0


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip()) / 1024.0

FLEET_VIDS = ("F01", "F02", "F03", "F04", "A01", "A02", "A03", "A04")
UTIL_WIN_N = 5                      # 与服务端 UTIL_WIN 对齐


def _util_verdict(snaps: list) -> dict:
    """各车滑动窗利用率均值 ≥60% (跳过前 5 个快照热身)。"""
    warm = [s for s in snaps[5:]
            if s.get("util_window", {}).get("n_samples", 0) >= UTIL_WIN_N]
    if not warm:
        return {"ok": False, "per_vehicle_mean": {}, "fleet_mean": 0.0,
                "criterion": "各车滑动窗利用率均值 ≥ 60% (无有效样本)"}
    per = {v: round(sum(s["util_window"]["per_vehicle"].get(v, 0.0)
                        for s in warm) / len(warm), 3) for v in FLEET_VIDS}
    fleet = round(sum(s["util_window"]["fleet"] for s in warm) / len(warm), 3)
    return {"ok": all(v >= 0.60 for v in per.values()),
            "per_vehicle_mean": per, "fleet_mean": fleet,
            "criterion": "各车滑动窗利用率均值 ≥ 60% (跳过前 5 个快照热身)"}



def snap(svc: MatrixService, wall_s: float, sim0: float) -> dict:
    m = svc.metrics()
    sim = float(svc.cell.data.time)
    return {
        "wall_s": round(wall_s, 1),
        "sim_s": round(sim, 1),
        "sim_per_wall": round((sim - sim0) / max(wall_s, 1e-9), 3),
        "rss_mb": round(rss_mb(), 1),
        "mode": svc.mode,
        "throughput_tasks_h": m["throughput_tasks_h"],
        "orders_done": m["orders_done"],
        "tasks_done": m["tasks_done"],
        "tasks_failed": m["tasks_failed"],
        "chains_open": m["chains_open"],
        "queue_depth": m["queue_depth"],
        "jobs_active": m["jobs_active"],
        "utilization": m["utilization"],
        "lock_waits": m["lock_waits"],
        "lock_wait_s": m["lock_wait_s"],
        "collisions": m["collisions"],
        "max_pen_m": m["max_pen_m"],
        "decisions_total": m["decisions_total"],
        "feed_depth": m["feed_depth"],
        "decision_depth": m["decision_depth"],
        "op_errors": m["op_errors"],
        "stall_s": m["stall_s"],
        "wall_progress_s": m["wall_progress_s"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=1800.0,
                    help="压测墙钟秒数 (默认 1800 = 30 min)")
    ap.add_argument("--multiplier", type=float, default=1.0,
                    help="需求曲线倍率")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--drain-s", type=float, default=480.0,
                    help="收尾停单 drain 窗口 (秒)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    svc = MatrixService(seed=a.seed, multiplier=a.multiplier, speed=0)
    t0 = time.monotonic()
    sim0 = float(svc.cell.data.time)
    svc.start()
    print(f"[press] run={svc.run_id} duration={a.duration:.0f}s "
          f"multiplier={a.multiplier} — 每 {SNAPSHOT_S:.0f}s 快照", flush=True)

    snaps: list[dict] = []
    drain_t0 = None
    while True:
        time.sleep(SNAPSHOT_S)
        wall = time.monotonic() - t0
        s = snap(svc, wall, sim0)
        snaps.append(s)
        util = " ".join(f"{v}:{u:.0%}" for v, u in s["utilization"].items())
        print(f"[press {wall/60:5.1f}m] sim={s['sim_s']:8.1f}s "
              f"({s['sim_per_wall']:.2f}x) rss={s['rss_mb']:7.1f}MB "
              f"thr={s['throughput_tasks_h']:5.1f}/h ord={s['orders_done']:3d} "
              f"fail={s['tasks_failed']:2d} open={s['chains_open']:2d} "
              f"wait={s['lock_waits']:4d} pen={s['max_pen_m']*1000:5.2f}mm "
              f"dec={s['decisions_total']:4d} | {util}", flush=True)
        remaining = a.duration - wall
        if remaining <= 0:
            break
        if remaining <= a.drain_s and svc._demand_on:
            svc._demand_on = False
            drain_t0 = wall
            print("[press] 收尾 drain: 停止出单, 等待在飞任务清空", flush=True)

    # drain 等待 (最长再加 drain_s)
    if drain_t0 is not None:
        deadline = time.monotonic() - t0 + a.drain_s + SNAPSHOT_S
        while time.monotonic() - t0 < deadline:
            m = svc.metrics()
            if m["chains_open"] == 0 and m["queue_depth"] == 0 \
                    and m["jobs_active"] == 0:
                break
            time.sleep(5.0)
        wall = time.monotonic() - t0
        snaps.append(snap(svc, wall, sim0))
        print(f"[press {wall/60:5.1f}m] drain 完成: "
              f"open={snaps[-1]['chains_open']} queue={snaps[-1]['queue_depth']}",
              flush=True)
    svc.stop()

    # ---- 判据 ----
    first, last = snaps[0], snaps[-1]
    mem_growth = (last["rss_mb"] - first["rss_mb"]) / max(first["rss_mb"], 1e-9)
    m = svc.metrics()
    # 零死锁口径: 仅当系统有任务在身时才计入停滞 (空场等待需求不算)
    with_work = [s for s in snaps
                 if s["chains_open"] or s["queue_depth"] or s["jobs_active"]]
    max_no_progress = max((s["wall_progress_s"] for s in with_work), default=0.0)
    verdict = {
        "mem": {
            "ok": mem_growth < 0.10,
            "rss_first_mb": first["rss_mb"], "rss_last_mb": last["rss_mb"],
            "growth_pct": round(mem_growth * 100, 2),
            "criterion": "RSS 首尾增长 < 10%"},
        "deadlock": {
            "ok": (svc.mode != "failed"
                   and max_no_progress < 180.0
                   and (m["chains_open"] == 0 or
                        (m["jobs_active"] > 0 and m["wall_progress_s"] < 120))),
            "mode": svc.mode,
            "max_no_progress_with_work_s": round(max_no_progress, 1),
            "chains_open_end": m["chains_open"],
            "queue_depth_end": m["queue_depth"],
            "jobs_active_end": m["jobs_active"],
            "criterion": "任务全部完成或活跃; 有任务时无 >180s 停滞; mode≠failed"},
        "utilization": _util_verdict(snaps),
        "collision": {
            "ok": m["max_pen_m"] < 5e-4 and m["collisions"] == 0,
            "max_pen_m": m["max_pen_m"],
            "violations": m["collisions"],
            "criterion": "max_pen < 5e-4 m 且穿透违规 == 0"},
        "ops": {
            "ok": m["tasks_failed"] == 0 and m["op_errors"] == 0,
            "tasks_failed": m["tasks_failed"],
            "op_errors": m["op_errors"],
            "criterion": "物理任务零失败 (op_errors 空)"},
    }
    report = {
        "run_id": svc.run_id,
        "argv": vars(a),
        "snapshots": snaps,
        "final_metrics": m,
        "verdict": verdict,
        "all_green": all(v["ok"] for v in verdict.values()),
        "utilization_final": m["utilization"],
        "orders_done": m["orders_done"],
        "tasks_done": m["tasks_done"],
        "throughput_tasks_h_final": m["throughput_tasks_h"],
    }
    out = a.out or (REPO / "reports" /
                    f"press_matrix_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print("\n===== 压测判据 =====")
    for k, v in verdict.items():
        mark = "✅ PASS" if v["ok"] else "❌ FAIL"
        extra = {kk: vv for kk, vv in v.items() if kk not in ("ok", "criterion")}
        print(f"[{k:9s}] {mark}  {v['criterion']}  {extra}")
    print(f"\n订单完成 {m['orders_done']} | 任务腿 {m['tasks_done']} | "
          f"决策事件 {m['decisions_total']} | 锁等待 {m['lock_waits']} 次 "
          f"{m['lock_wait_s']}s | 报告 → {out}")
    return 0 if report["all_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
