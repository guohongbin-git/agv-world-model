#!/usr/bin/env python3
"""消融 harness: Decision Memory 有/无对照 (论文 II H2 证据脚本)。

前提: server 已在 :8300 运行, decision_memory.jsonl 已有同情境记录
(先跑一遍避障 scenario 生成)。

方法: 同一避障场景 (注入锥桶@-0.25 → 用户 15s 后移除) 各跑一轮:
  A 无记忆: 暂移 memory 文件 -> 受阻时无历史提示
  B 有记忆: 记录在位 -> 受阻时叙述含历史经验
指标: 受阻->恢复 墙钟、重试次数、叙述中是否含提示、任务结局。
预测: 机械指标两组相同 (恢复都靠感知证据); 差异在信息层 —— 提示
让用户在受阻当下即知道"该做什么、上次多快", 压缩在环决策时间
(该时间本身属 H2 用户研究范畴, 本脚本固化可复现的机制证据)。
"""
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8300"
RUNS = Path(__file__).resolve().parents[1] / "twin" / "runs"
MEM = RUNS / "decision_memory.jsonl"


def api(path: str, payload: dict | None = None):
    if payload is None:
        with urllib.request.urlopen(BASE + path, timeout=30) as r:
            return json.load(r)
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def wait_mode(target: str, timeout_s: float) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if api("/api/state")["mode"] == target:
            return time.monotonic() - t0
        time.sleep(0.5)
    return -1.0


def scenario(mem_backup: Path | None) -> dict:
    api("/api/chat", {"text": "复位"})
    time.sleep(2)
    api("/api/chat", {"text": "快 8 倍"})
    api("/api/chat", {"text": "把 A-L3-C1 那托下架到 H01"})
    assert wait_mode("running", 30) > 0
    time.sleep(28)                                 # AGV 进入接近段
    api("/api/chat", {"text": "前方放个锥桶挡住"})
    t_block = time.monotonic()
    assert wait_mode("blocked", 60) > 0
    chat_at_block = [c["text"] for c in api("/api/state").get("chat", [])]
    has_hint = any("历史经验" in c for c in chat_at_block[-3:])
    time.sleep(15)                                 # 模拟用户响应延迟
    api("/api/chat", {"text": "锥桶我移走了"})
    done = wait_mode("done", 240)
    return {"blocked_to_done_s": round(time.monotonic() - t_block, 1)
            if done > 0 else None,
            "memory_hint": has_hint,
            "completed": done > 0}


def main() -> None:
    bak = RUNS / "decision_memory.jsonl.bak"
    have_mem = MEM.exists()
    if have_mem:
        shutil.move(MEM, bak)

    print("== 条件 A: 无记忆 ==")
    a = scenario(bak)
    print(json.dumps(a, ensure_ascii=False))

    if bak.exists():
        shutil.move(bak, MEM)
    if not MEM.exists():                           # 无记录则先造一条
        print("(无记忆记录, 跳过条件 B —— 先跑一遍避障 scenario)")
        return
    print("== 条件 B: 有记忆 ==")
    b = scenario(None)
    print(json.dumps(b, ensure_ascii=False))

    print("\n=== 对照 ===")
    print(f"提示呈现: A={a['memory_hint']}  B={b['memory_hint']}")
    print(f"恢复完成: A={a['completed']}  B={b['completed']}  "
          "(机械路径相同 —— 差异在信息层, 见模块 docstring)")


if __name__ == "__main__":
    sys.exit(main())
