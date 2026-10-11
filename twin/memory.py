#!/usr/bin/env python3
"""Decision Memory (论文 II): 情境 → 处置 → 结局 的持久化记忆。

设计:
- 情境签名 (signature): 受阻原因 + 受阻步骤 + 障碍位置桶 (0.05m)。
  同签名 = 同一"情境类型", 记忆可跨 run 复用。
- 记录 (record): 处置动作序列 + 结局 (恢复耗时/重试次数/是否成功)。
- 复用: 受阻时按签名查询 —— 命中则把历史处置作为"经验提示"交给
  叙述层与 agent, 缩短人工决策回路; 未命中则如实报告无经验。

反模式 (明确不做):
- 不记忆"上帝视角状态"——只记感知可见的情境与验证过的处置;
- 不自动重放处置——记忆只提供提示, 恢复仍须感知证据 (H1 红线)。
"""
import json
import time
from pathlib import Path


class DecisionMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.records[r["signature"]] = r

    @staticmethod
    def signature(reason: str, step_label: str, obstacle_x: float | None,
                  bucket_m: float = 0.05) -> str:
        x = "" if obstacle_x is None else f"@{round(obstacle_x / bucket_m) * bucket_m:.2f}"
        return f"{reason}|{step_label}|{x}"

    def lookup(self, sig: str) -> dict | None:
        return self.records.get(sig)

    def record(self, sig: str, actions: list[str], outcome: str,
               resume_s: float | None, retries: int) -> None:
        rec = {"signature": sig, "actions": actions, "outcome": outcome,
               "resume_s": resume_s, "retries": retries,
               "ts": time.time()}
        self.records[sig] = rec
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def hint(self, sig: str) -> str | None:
        """复用提示: 有历史则给处置建议, 无则 None。"""
        rec = self.lookup(sig)
        if not rec or rec["outcome"] != "recovered":
            return None
        return ("历史经验 (Decision Memory): 同类情境 ["
                + sig + "] 上次处置为 " + " → ".join(rec["actions"])
                + f", 耗时 {rec['resume_s']:.0f}s / 重试 {rec['retries']} 次"
                "后恢复。建议采取相同处置。")
