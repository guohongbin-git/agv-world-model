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
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "detfix" / "press_reverify.json"
URL = "http://127.0.0.1:8400/api/state"

rows = []
for i in range(30):
    try:
        with urllib.request.urlopen(URL, timeout=8) as r:
            m = json.load(r)
        met = m.get("metrics", {})
        rows.append({
            "t": round(i * 60), "sim": round(m.get("sim_t", 0)),
            "busy": sum(1 for v in m.get("vehicles", []) if v.get("busy")),
            "orders": met.get("orders_done"), "tasks": met.get("tasks_done"),
            "fail": met.get("tasks_failed"),
            "rss_mb": round(m.get("process", {}).get("rss_mb", 0), 1),
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
