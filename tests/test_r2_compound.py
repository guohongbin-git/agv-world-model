#!/usr/bin/env python3
"""R2 复合压力场景: 双车交接 × LiDAR 避障 × 巷道互斥 × Decision Memory。

复合点 (多系统拼接处最易碎):
1. F01 让位行驶 (持有 AisleLock) 中注入锥桶 → P4a 避障触发 blocked;
   此时锁**必须保持被 F01 持有** (受阻 ≠ 放弃巷道, 否则 F02 会闯入
   与抛锚车同巷道);
2. F02 的接力任务在队列中等待锁 —— 不得因 F01 受阻而误启动;
3. 移除锥桶 → 重感知恢复 → 让位完成释放锁 → F02 接力 → 账本链完整;
4. 恢复成功写 Decision Memory (P4b)。

预测: blocked 期间 mode=blocked、锁持有者不变; 移除后任务序列无旁路
地走完; 全程零碰撞。
"""
import json
import time

import pytest

from twin.server import PacedForkliftAdapter, WorldService

PEN_LIMIT = 5e-4


@pytest.fixture(scope="module")
def pen_guard():
    orig = PacedForkliftAdapter._tick
    max_pen = [0.0]

    def _tick(self):
        orig(self)
        if self._n % 200 == 0:
            max_pen[0] = max(max_pen[0], self.cell.max_penetration())

    PacedForkliftAdapter._tick = _tick
    yield max_pen
    PacedForkliftAdapter._tick = orig


@pytest.fixture(scope="module")
def svc(pen_guard):
    s = WorldService()
    s.gate.speed = 64.0
    return s


def _wait(pred, timeout_s: float, poll: float = 0.2) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if pred():
            return True
        time.sleep(poll)
    return False


def test_handoff_blocked_by_cone_lock_integrity_and_recovery(svc, pen_guard):
    pen_guard[0] = 0.0
    svc.tasks = list(svc.HANDOFF_TASKS)
    svc.mode = "running"

    import threading
    th = threading.Thread(target=svc.run_mission, daemon=True)
    th.start()

    # 0) 等 F01 取货完成进入让位前移段 (axle 越过交接台), 再在其路径
    #    前方注入锥桶 —— 太早注入会在取货段 (不行车) 无触发
    def letting_out() -> bool:
        return (svc.mode == "running"
                and svc.cell.vehicle("F01").axle_x > -0.35)
    assert _wait(letting_out, 180), f"F01 未进入让位段: {svc.step_label}"
    cone_x = svc.cell.vehicle("F01").axle_x + 0.15
    svc.place_cone(cone_x)

    # 1) LiDAR 检测 → 避障停车
    assert _wait(lambda: svc.mode == "blocked", 120), \
        f"未进入 blocked: {svc.step_label}"
    assert svc.blocked_reason == "cone"
    events, _ = svc.events_since(0)
    assert any(e[1] == "ObstacleDetected" for e in events)

    # 2) 锁完整性: F01 受阻但巷道锁仍被持有 (F02 不得闯入)
    lock_info = svc.aisle.info() if hasattr(svc.aisle, "info") else None
    holder = getattr(svc.aisle, "owner", None) or lock_info
    assert holder, f"blocked 期间锁被释放: {holder!r}"
    f02_axle = svc.cell.vehicle("F02").axle_x
    assert f02_axle < -0.9, f"F02 越位进入作业区: {f02_axle}"

    # 3) 用户移除锥桶 → 重感知恢复 → 全链完成
    assert _wait(lambda: svc.mode == "blocked", 5)      # 仍在等
    svc.remove_cones()
    assert _wait(lambda: svc.mode == "done", 240), \
        f"恢复后未完成: {svc.step_label}"

    # 4) 账本链完整 (交接 + 过户 + 出库待命)
    rows, _ = svc.events_since(0)
    chain = [json.loads(e[2]).get("to_location")
             for e in rows if e[1] == "PalletMoved"]
    assert chain == ["ROBOT:F01", "STATION:S1", "ROBOT:F02", "SLOT:H01"]
    b = svc.board()
    assert dict(b["slots"])["H01"] == "OCCUPIED"
    assert {g[0]: g[3] for g in b["goods"]}["G-001"] == "OUTBOUND_PENDING"

    # 5) Decision Memory 落盘 (blocked→recovered)
    mem = svc.memory.records
    assert any(r["outcome"] == "recovered" and r["retries"] >= 1
               for r in mem.values()), f"记忆未记录: {mem}"

    # 6) 零碰撞
    assert pen_guard[0] < PEN_LIMIT, f"max_pen={pen_guard[0]:.2e} 超限"
