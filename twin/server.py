"""T04 世界模型实时服务 — agent 唯一入口, 物理/事务/事件三路流式推送.

    PYTHONPATH=. ./.venv/bin/python twin/server.py     # http://127.0.0.1:8300

架构 (docs/agentic_twin_design.md §1/§4, 对齐设计文档 §8.1 数据流):

    用户 ⇄ /api/chat ⇄ WarehouseAgent (twin/agent.py, 唯一指令入口)
                          │ task.start / pause / speed / query…
                          ▼
    ForkliftAdapter 判据层 (≤5mm/s 插叉、退叉门禁、承托观测)
                          │ 物理步进全部汇于 _tick → PacedForkliftAdapter 配速+采样
                          ▼
    ForkliftCell (MuJoCo 2kHz)          World (SQLite 事务 + Event, 唯一业务写入)
        └── 60Hz 位姿快照 → /ws ──→ three.js 投影 + 库存板 + 事件流

演示任务 (单 F01 一轮下架, 经真实判据链):
    A-L3-C1 货架第三层托盘 → 暂存站 H01;
    离梁后提交 PAL-001→ROBOT:F01; 放托退叉验真后提交 →STATION:H01 + 出库待命。
"""
from __future__ import annotations

import asyncio
import json
import numpy as np
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from sim.forklift_cell import ForkliftCell
from services.adapters.base import ActionResult
from services.adapters.forklift import ForkliftAdapter
from services.world.ontology import (GoodsLifecycle, GoodsUnit, Loc, LocKind,
                                     Pallet, PalletCargo, Slot, SlotOcc,
                                     World)

WEB = Path(__file__).parent / "web"
RUNS = Path(__file__).parent / "runs"
PUB_HZ = 60
FORK_CAPS = frozenset({"drive", "fork_lift", "fork_insert",
                       "fork_withdraw", "stop"})
BODIES = ("chassis", "carriage", "pallet")

RUN_ID = time.strftime("DEMO-%Y%m%d-%H%M%S")
TASK_ID = RUN_ID + "-T01"


class Gate:
    """配速/暂停/采样共享状态 (mission 线程写, ws 线程读)。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.paused = False
        self.speed = 1.0
        self.t0 = time.monotonic()
        self.sim0 = 0.0
        self.snap: dict = {"t": 0.0, "next_t": 0.0}


class PacedForkliftAdapter(ForkliftAdapter):
    """物理步进唯一入口 hook: 暂停门 + 墙钟配速 (速度倍率) + 60Hz 位姿采样。"""

    def __init__(self, robot_id, cell, gate: Gate, **kw):
        super().__init__(robot_id=robot_id, cell=cell, **kw)
        self.gate = gate
        self._n = 0

    def _tick(self) -> None:
        mujoco.mj_step(self.cell.model, self.cell.data)
        self._n += 1
        if self._n % 100 != 0:         # 每 0.05s 仿真检查一次暂停/采样
            return
        g = self.gate
        while True:
            with g.lock:
                paused = g.paused
            if not paused:
                break
            time.sleep(0.05)
        d = self.cell.data
        if d.time >= g.snap.get("next_t", 0.0):
            mujoco.mj_forward(self.cell.model, d)
            snap = {"t": round(d.time, 3), "next_t": d.time + 1.0 / PUB_HZ}
            for b in BODIES:
                bid = mujoco.mj_name2id(self.cell.model,
                                        mujoco.mjtObj.mjOBJ_BODY, b)
                snap[b] = [round(v, 5) for v in d.xpos[bid]] + \
                          [round(v, 5) for v in d.xquat[bid]]
            with g.lock:
                snap["next_t"] = max(snap["next_t"], g.snap.get("next_t", 0))
                g.snap = snap
                speed = g.speed
            # 墙钟配速: sim 时间以 speed 倍推进
            target = g.t0 + (d.time - g.sim0) / max(0.25, speed)
            lead = target - time.monotonic()
            if lead > 0:
                time.sleep(min(lead, 0.05))


class WorldService:
    """世界模型 + 物理 + 任务编排; 业务写入只经 World 事务。"""

    def __init__(self) -> None:
        self.gate = Gate()
        self.mode = "idle"          # idle|running|paused|done|failed
        self.step = -1
        self.step_label = "待命"
        self.chat: deque = deque(maxlen=200)
        self.chat_seq = 0
        self.db_lock = threading.RLock()   # 串行化 World 跨线程访问
        self._board_cache: tuple | None = None
        self.build()

    # ---------- 构建/复位 ----------
    def build(self) -> None:
        cell = ForkliftCell(timestep_s=0.0005)
        d = cell.data
        d.ctrl[cell.a_x] = cell.k["start_axle_x"] - cell.q0_x
        d.ctrl[cell.a_lift] = 0.0
        for _ in range(1000):
            mujoco.mj_step(cell.model, d)
        self.cell = cell
        self.adapter = PacedForkliftAdapter("F01", cell, self.gate,
                                            capabilities=FORK_CAPS)
        RUNS.mkdir(exist_ok=True)
        self.world = World(run_id=RUN_ID,
                           db_path=str(RUNS / f"{RUN_ID}.sqlite"))
        self._seed_ledger()
        with self.gate.lock:
            self.gate.sim0 = float(d.time)
            self.gate.t0 = time.monotonic()
            self.gate.paused = False
            self.gate.snap = {"t": round(float(d.time), 3), "next_t": 0.0}
        self.mode = "idle"
        self.step = -1
        self.step_label = "待命"

    def _seed_ledger(self) -> None:
        """注册本体事实 (与物理初态一致): 货位/托盘/货物/绑定。"""
        w = self.world
        w.register_slot(Slot("A-L3-C1", (0.0, 0.0, 0.400), SlotOcc.OCCUPIED))
        w.register_slot(Slot("H01", (-0.340, 0.0, 0.045), SlotOcc.EMPTY))
        w.register_pallet(Pallet("PAL-001", 0.022,
                                 Loc(LocKind.SLOT, "A-L3-C1"),
                                 PalletCargo.LOADED))
        w.register_goods(GoodsUnit("G-001", "COKE-500ML×24", 0.023,
                                   Loc(LocKind.PALLET, "PAL-001"),
                                   GoodsLifecycle.IN_WAREHOUSE))
        w._conn.execute("INSERT INTO bindings VALUES(?,?,?,1)",
                        ("G-001", "PAL-001", 1))
        w._conn.commit()

    # ---------- 任务步骤 (判据层动作序列 + 事务提交点) ----------
    def _steps(self) -> list:
        k = self.cell.k
        x_ins = -0.040 + k["insert_depth"] - (k["fork_root_x"] + k["fork_len"])
        x_edge = -0.040 - k["approach_gap"] - k["fork_len"] - k["fork_root_x"]
        x_clear = -0.200 - k["fork_root_x"] - k["fork_len"] - 0.010
        off = 0.094                       # 托盘中心相对轴偏置 (实测值)
        x_stage = -0.340 - off
        x_out = x_stage - k["retreat_loaded"] - 0.060
        return [
            ("lift", dict(target_z=k["insert_z"], speed_mps=0.010),
             "升至插叉高度 (支撑面+10)", None),
            ("move", dict(target_axle_x=x_edge, speed_mps=0.020),
             "接近货架前沿", None),
            ("move", dict(target_axle_x=x_ins, speed_mps=0.005),
             "插叉 65mm (≤5mm/s 判据)", None),
            ("lift", dict(target_z=k["lift_z"], speed_mps=0.002),
             "举升离梁 (支撑面+17)", self._commit_on_robot),
            ("move", dict(target_axle_x=x_clear, speed_mps=0.020),
             "带载直线退出 (叉尖越过横梁)", None),
            ("lift", dict(target_z=k["transport_z"], speed_mps=0.010),
             "降至运输高度 65mm", None),
            ("move", dict(target_axle_x=x_stage, speed_mps=0.020),
             "低位转运至暂存站上方", None),
            ("lift", dict(target_z=0.057, speed_mps=0.010),
             "举至放托位 (台面+12)", None),
            ("lift", dict(target_z=0.0565, speed_mps=0.002),
             "微降脱离叉孔上壁", None),
            ("move", dict(target_axle_x=x_out, speed_mps=0.020),
             "空叉退出 (越过近边+10mm)", self._commit_on_stage),
        ]

    # 提交点: 只在物理证据成立时写 World
    def _commit_on_robot(self) -> bool:
        c = self.cell
        if not c.pallet_fork_contact() or c.pallet_bottom_z < 0.402:
            return False
        self.world.move_pallet("PAL-001", Loc(LocKind.ROBOT, "F01"),
                               TASK_ID, float(c.data.time))
        self.world._conn.execute(
            "UPDATE slots SET occupancy='EMPTY' WHERE slot_id='A-L3-C1'")
        self.world._conn.commit()
        self.narrate("物理证据确认: PAL-001 已离梁、由 F01 叉臂承托。"
                     "事务提交 PAL-001→ROBOT:F01, 货位 A-L3-C1 释放。"
                     "在库计数不变 (绑定 G-001 有效位置仍是 PALLET)。")
        return True

    def _commit_on_stage(self) -> bool:
        c = self.cell
        if c.pallet_fork_contact() or not (0.042 <= c.pallet_bottom_z <= 0.048):
            return False
        t = float(c.data.time)
        self.world.move_pallet("PAL-001", Loc(LocKind.STATION, "H01"),
                               TASK_ID, t)
        self.world.mark_outbound_pending("G-001", TASK_ID, t)
        self.world._conn.execute(
            "UPDATE slots SET occupancy='OCCUPIED' WHERE slot_id='H01'")
        self.world._conn.commit()
        self.narrate("放托验真: 叉臂脱离、托盘底面 45mm 落在 H01 台面。"
                     "事务提交 PAL-001→STATION:H01, G-001→出库待命 "
                     "(仍计在库, 只有离场才减)。任务完成。")
        return True

    # ---------- mission 线程 ----------
    def run_mission(self) -> None:
        d = self.cell.data
        try:
            self._run_steps(d)
        except Exception as e:
            self.mode = "failed"
            self.step_label = f"异常: {type(e).__name__}"
            self.narrate(f"mission 线程异常: {e!r} — 未提交的事务保持原状。")

    def _ramp(self, idx: int, qadr: int, q_target: float,
              speed: float, calm: bool = False) -> bool:
        """梯形速度剖面驱动到 q_target (R04 限速语义), 复用 adapter._tick
        (配速/采样/暂停门都在里面)。到位判据 = 位置误差 + 可选全域静止。"""
        from sim.controllers import Ramp
        d = self.cell.data
        q0 = float(d.qpos[qadr])
        ramp = Ramp(q0, q_target, speed)
        t0 = d.time
        tick = self.adapter._tick
        n = int((ramp.duration + 12.0) * 2000)      # 预算: 行程 + 12s 整定
        for _ in range(n):
            d.ctrl[idx] = ramp.at(min(d.time - t0, ramp.duration))
            tick()
            if abs(float(d.qpos[qadr]) - q_target) <= 1e-3:
                if not calm or float(np.abs(d.qvel).max()) < 5e-3:
                    return True
        return abs(float(d.qpos[qadr]) - q_target) <= 2e-3

    def _steps(self) -> list:
        k = self.cell.k
        x_ins = -0.040 + k["insert_depth"] - (k["fork_root_x"] + k["fork_len"])
        x_edge = -0.040 - k["approach_gap"] - k["fork_len"] - k["fork_root_x"]
        x_clear = -0.200 - k["fork_root_x"] - k["fork_len"] - 0.010
        off = 0.094                       # 托盘中心相对轴偏置 (实测值)
        x_stage = -0.340 - off
        x_out = x_stage - k["retreat_loaded"] - 0.060
        ax, lf = self.cell.a_x, self.cell.a_lift
        qx, ql = self.cell.qadr_x, self.cell.qadr_lift
        q0x, tz = self.cell.q0_x, k["transport_z"]

        def mv(x, v, calm=False):
            return lambda: self._ramp(ax, qx, x - q0x, v, calm)

        def up(z, v, calm=False):
            return lambda: self._ramp(lf, ql, z - tz, v, calm)
        return [
            (up(k["insert_z"], 0.010), "升至插叉高度 (支撑面+10)", None),
            (mv(x_edge, 0.020), "接近货架前沿", None),
            (mv(x_ins, 0.005), "插叉 65mm (≤5mm/s 判据)", None),
            (up(k["lift_z"], 0.002), "举升离梁 (支撑面+17)",
             self._commit_on_robot),
            (mv(x_clear, 0.020, True), "带载直线退出 (叉尖越过横梁)", None),
            (up(tz, 0.010, True), "降至运输高度 65mm", None),
            (mv(x_stage, 0.020, True), "低位转运至暂存站上方", None),
            (up(0.057, 0.010, True), "举至放托位 (台面+12)", None),
            (up(0.0565, 0.002, True), "微降脱离叉孔上壁", None),
            (mv(x_out, 0.020), "空叉退出 (越过近边+10mm)",
             self._commit_on_stage),
        ]

    def _run_steps(self, d) -> None:
        for i, (fn, label, commit) in enumerate(self._steps()):
            with self.gate.lock:
                self.step = i
                self.step_label = label
            self.narrate(f"[{i + 1}/10] {label} — 开始 (判据层限速执行)")
            if not fn():
                with self.db_lock:
                    self.world.emit(TASK_ID, "StepFailed",
                                    {"step": i, "label": label},
                                    float(d.time))
                self.mode = "failed"
                self.step_label = f"失败: {label}"
                self.narrate(f"步骤 [{label}] FAILED — 不提交事务, "
                             "状态保持, 等待人工/agent 决策。")
                return
            if commit is not None:
                with self.db_lock:
                    ok = commit()
                if not ok:
                    with self.db_lock:
                        self.world.emit(TASK_ID, "CommitEvidenceMissing",
                                        {"step": i, "label": label},
                                        float(d.time))
                    self.mode = "failed"
                    self.step_label = f"证据不足: {label}"
                    self.narrate(f"[{label}] 物理执行完成但承托/脱离证据不满足"
                                 " — 拒绝提交 (agent 可改计划, 不能改事实)。")
                    return
            with self.db_lock:
                self.world.emit(TASK_ID, "StepCompleted",
                                {"step": i, "label": label}, float(d.time))
        self.step = 10
        self.step_label = "任务完成"
        self.mode = "done"

    # ---------- 命令 (只接受来自 agent 层的调用) ----------
    def cmd(self, op: str, value=None) -> dict:
        if op == "start":
            if self.mode in ("idle", "done", "failed"):
                self.mode = "running"
                with self.gate.lock:
                    self.gate.paused = False
                    self.gate.t0 = time.monotonic()
                    self.gate.sim0 = float(self.cell.data.time)
                self._thread = threading.Thread(target=self.run_mission,
                                                daemon=True)
                self._thread.start()
                self.narrate("任务已受理: 将 A-L3-C1 的 PAL-001 下架至 H01。"
                             "执行走判据层 (插叉≤5mm/s、退叉门禁、承托验真),"
                             "库存只在证据成立后提交。")
        elif op == "pause":
            with self.gate.lock:
                self.gate.paused = True
            self.narrate("已暂停: 物理冻结在当前状态, 目标位保持。")
        elif op == "resume":
            with self.gate.lock:
                self.gate.paused = False
                sim_now = float(self.cell.data.time) - self.gate.sim0
                self.gate.t0 = time.monotonic() - sim_now / self.gate.speed
            self.narrate("已恢复执行。")
        elif op == "speed":
            v = max(0.25, min(8.0, float(value or 1)))
            with self.gate.lock:
                sim_now = float(self.cell.data.time) - self.gate.sim0
                self.gate.speed = v
                self.gate.t0 = time.monotonic() - sim_now / v
            self.narrate(f"播放倍率 = {v:g}× (物理仿真不变, 只改墙钟配速)。")
        elif op == "reset":
            self.mode = "idle"
            self.build()
            self.narrate("已复位: 新 run、账本重播种、与物理初态一致。")
        return {"mode": self.mode, "step": self.step,
                "label": self.step_label, "speed": self.gate.speed}

    # ---------- 输出 ----------
    def narrate(self, text: str) -> None:
        self.chat_seq += 1
        self.chat.append({"seq": self.chat_seq, "role": "agent",
                          "text": text,
                          "t": round(float(self.cell.data.time), 1)})

    def board(self) -> dict:
        now = time.monotonic()
        if self._board_cache and now - self._board_cache[0] < 1.0:
            return self._board_cache[1]
        with self.db_lock:
            c = self.world._conn
            out = {
                "run_id": RUN_ID,
                "count": self.world.in_warehouse_count(),
                "slots": [list(r) for r in c.execute(
                    "SELECT slot_id, occupancy FROM slots")],
                "pallets": [list(r) for r in c.execute(
                    "SELECT pallet_id, location, cargo_state FROM pallets")],
                "goods": [list(r) for r in c.execute(
                    "SELECT goods_id, sku, location, lifecycle FROM goods")],
            }
        self._board_cache = (now, out)
        return out

    def events_since(self, seq: int):
        with self.db_lock:
            rows = self.world._conn.execute(
                "SELECT event_id, type, payload, sim_time_s FROM events "
                "ORDER BY event_id").fetchall()
        out = [{"eid": e[0], "type": e[1],
                "payload": json.loads(e[2]), "t": round(e[3], 1)}
               for e in rows[seq:]]
        return rows, out

    def state(self, ev_cursor: int, chat_cursor: int) -> dict:
        with self.gate.lock:
            snap = dict(self.gate.snap)
            speed = self.gate.speed
            paused = self.gate.paused
        rows, events = self.events_since(ev_cursor)
        chats = [m for m in self.chat if m["seq"] > chat_cursor]
        return {
            "poses": snap, "sim_t": snap.get("t", 0),
            "mode": self.mode, "paused": paused, "speed": speed,
            "step": self.step, "step_label": self.step_label,
            "events": events, "n_events": len(rows),
            "chat": chats, "board": self.board(),
        }


SVC = WorldService()
app = FastAPI(title="T04 World Model")


@app.middleware("http")
async def _nocache(request, call_next):
    resp = await call_next(request)
    if request.url.path.endswith((".html", "/")):
        resp.headers["cache-control"] = "no-store"
    return resp


@app.post("/api/chat")
async def chat(body: dict):
    from twin.agent import handle
    text = str(body.get("text", ""))
    return handle(text, SVC)


@app.get("/api/state")
async def state():
    return SVC.state(0, 0)


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    ev_cur, chat_cur = 0, SVC.chat_seq
    try:
        while True:
            st = SVC.state(ev_cur, chat_cur)
            ev_cur = st["n_events"]
            if st["chat"]:
                chat_cur = st["chat"][-1]["seq"]
            await ws.send_text(json.dumps(st, ensure_ascii=False))
            await asyncio.sleep(1.0 / PUB_HZ)
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8300, log_level="warning")
