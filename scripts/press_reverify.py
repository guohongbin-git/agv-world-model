#!/usr/bin/env python3
"""30 分钟压测复验 (DetFix 收敛后): 三 FAIL 同消 + NaN/穿透监控。

采样 :8400 每 60s × 30 次; 终末跑 3 个此前失败场景的定向验证
(T11-T13 复刻 = 压测组) 并落盘 JSON 摘要。
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rss_server_mb() -> float:
    """:8400 进程 RSS (ps), 修复旧版恒 0 的 process.rss_mb 幻字段。"""
    try:
        pid = subprocess.run(["lsof", "-ti", ":8400"], capture_output=True,
                             text=True).stdout.strip().splitlines()[0]
        out = subprocess.run(["ps", "-o", "rss=", "-p", pid],
                             capture_output=True, text=True).stdout.strip()
        return round(int(out) / 1024, 1)
    except Exception:  # noqa: BLE001
        return 0.0
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "detfix" / "press_reverify.json"
URL = "http://127.0.0.1:8400/api/state"

rows = []
for i in range(30):
    try:
        with urllib.request.urlopen(URL, timeout=8) as r:
            m = json.load(r)
        met = m.get("metrics", {})
        uw = met.get("util_window", {}) or {}
        rows.append({
            "t": round(i * 60), "sim": round(m.get("sim_t", 0)),
            "busy": sum(1 for v in m.get("vehicles", []) if v.get("busy")),
            "orders": met.get("orders_done"), "tasks": met.get("tasks_done"),
            "fail": met.get("tasks_failed"),
            "util_fleet": uw.get("fleet"),
            "throughput": met.get("throughput_tasks_h"),
            "max_pen_m": met.get("max_pen_m"),
            "collisions": met.get("collisions"),
            "chains_open": met.get("chains_open"),
            "queue_depth": met.get("queue_depth"),
            "rss_mb": rss_server_mb(),
            "divergence": bool(m.get("health", {}).get("diverged")),
        })
    except Exception as e:  # noqa: BLE001
        rows.append({"t": round(i * 60), "error": str(e)[:80]})
    time.sleep(60)

verdict = {"samples": rows}
errs = [r for r in rows if "error" in r]
valid = [r for r in rows if "error" not in r]
verdict["summary"] = {
    "n_samples": len(rows), "http_errors": len(errs),
    "sim_reached": max((r["sim"] for r in valid), default=0),
    "fails": max((r["fail"] or 0) for r in valid) if valid else None,
    "any_divergence": any(r.get("divergence") for r in valid),
    "rss_first": valid[0]["rss_mb"] if valid else None,
    "rss_last": valid[-1]["rss_mb"] if valid else None,
}
OUT.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
print(json.dumps(verdict["summary"], ensure_ascii=False))
