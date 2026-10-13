#!/usr/bin/env python3
"""Paper I 消融 harness — B0-B3 机制开关 × 故障矩阵确定性子集 (7 指标 + 审计 schema).

设计来源: docs/knowledge/paper1_fault_matrix.md「消融设计」节 (GPT 评审定稿):

| 组 | 物理证据门控 | 原子事务 | 失败对账 |
| B0 直接完成回报 | 无 | 无 | 无 |
| B1 仅物理门控   | 有 | 无 (旧式逐次 commit) | 无 |
| B2 门控+事务    | 有 | 有 (commit_transfer) | 无 (RECONCILIATION 后不恢复, 挂死) |
| B3 完整方案     | 有 | 有 | 有 (recover_open_commands / reconcile) |
| B4 工业强基线   | 无 (PLC 状态字) | 有 (op_id 幂等) | 有 (journal 对账+重试≤2, 不读真值) |

场景子集 (确定性): S01/S05/S06, D02/D03/D04/D06, H01/H03, C01。
指标 (每场景×组×seed): Unsafe Commit Rate / False Rejection Rate /
Atomicity Violation Rate / Recovery Success Rate / Recovery Time /
Task Completion Rate / Ledger–Physical Divergence Duration。

实现原则 (不改生产代码, 只在实验进程内组合):
1. 物理层 = FakeCell 运动学孪生 (truth = 物理真值; observed = 传感器口径,
   故障注入即篡改 observed)。经 WorldService(resume_cell=...) 注入, 并把
   twin.server / twin.perception 的模块级 mujoco 换成 fake —— 因此
   PREPARE / EvidenceVerdict 五门 / commit_transfer 原子事务 /
   recover_open_commands 重启对账 / reconcile 全部走**生产代码**。
2. B0/B1 的「旧式逐次 commit」= harness 侧 legacy 写入器 (每条 DML 独立
   commit, 无版本检查/无幂等前置检查), 复用生产 SQL 语义。
3. B2 = 生产 commit_transfer (不带 command_id, 无 PREPARED 机制)。
4. I4 证据有效性 (身份/时效) 属「门控」档: B1/B2 在 harness 提交前检查,
   B3 以 VERIFIED 阶段绑定检查实现 (生产 `_sample_verdict` 天然按命令
   托盘构造证据, 绑定由构造保证; S05/S06 注入经 `_sample_verdict` 实例
   补丁送达后由该检查拒绝)。
5. 重派发 (re-dispatch) 政策: 仅当系统产生**安全失败信号** (证据不足拒
   绝 / StalePlanRejected / 恢复取消) 后才允许 agent 重新派发 —— 崩溃
   后无任何记录可依据 = 挂死 (B0-B2 的 D02/D03/D04 结局)。
6. 重启 = 关闭旧连接后 WorldService(resume_db, resume_cell) —— 生产
   恢复路径原样执行 (物理世界不回滚, FakeCell 保留 truth)。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sqlite3
import sys
import time
import types
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
RUNS = ROOT / "twin" / "runs"

# 生产模块 (真实 import: 模块级 SVC 构建一次真实 cell, 一次性成本, 此后不用)
import twin.server as srv                      # noqa: E402
import twin.perception as tper                 # noqa: E402
from services.world.evidence import (          # noqa: E402
    EvidenceFrame, EvidenceVerdict, build_verdict, collect_frames)
from services.world.ontology import (          # noqa: E402
    GoodsLifecycle, GoodsUnit, Loc, LocKind, Pallet, PalletCargo, Slot,
    SlotOcc, TransferStage, World, WorldError)

TABLES = ("pallets", "goods", "slots", "bindings", "events", "command_log",
          "execution_evidence", "transfers")
TIMESTEP = 0.0005
PALLET_HALF_H = 0.011
GROUPS = ("B0", "B1", "B2", "B3", "B4")


# ================================================================ fake mujoco
class _MjtObj:
    mjOBJ_BODY = 1
    mjOBJ_GEOM = 2
    mjOBJ_JOINT = 3


class _FakeMujoco:
    """转发到 FakeCell 的最小 mujoco 面 (mj_step 经 model.cell 派发)。"""

    def mj_step(self, model, data):
        model.cell._tick()

    def mj_forward(self, model, data):
        return 0

    def mj_name2id(self, m, objtype, name):
        return zlib.crc32(name.encode("utf-8")) % 1000

    def mj_id2name(self, m, objtype, jid):
        return ""

    def mj_ray(self, m, d, *a, **k):
        return -1.0

    mjtObj = _MjtObj


FAKE_MUJOCO = _FakeMujoco()
srv.mujoco = FAKE_MUJOCO
tper.mujoco = FAKE_MUJOCO


# ================================================================ FakeCell
class _FakeModel:
    def __init__(self, cell):
        self.cell = cell
        self.opt = types.SimpleNamespace(timestep=TIMESTEP)


class _FakeData:
    def __init__(self):
        self.time = 0.0
        self.qpos = np.zeros(8)
        self.qvel = np.zeros(8)
        self.ctrl = np.zeros(8)
        self.xpos = np.zeros((1024, 3))
        self.xquat = np.zeros((1024, 4))
        self.mocap_pos = np.zeros((4, 3))


POSE = {
    "SLOT:A-L3-C1": (0.000, 0.0, 0.411),    # 底面 0.400 (L3 梁)
    "ROBOT:F01": (-0.526, 0.0, 0.076),      # 底面 0.065 (运输高度)
    "ROBOT:F02": (-0.726, 0.0, 0.076),
    "SLOT:H01": (-0.330, 0.0, 0.056),       # 底面 0.045 (出库台)
    "FLOOR": (-0.250, 0.0, 0.032),          # 滑落后落地
}


class FakeCell:
    """运动学物理孪生: truth = 真值 (evaluator 专读), observed = 传感器口径。

    生产代码 (WorldService/_sample_verdict/_observe_transfer_state) 只读
    observed 侧接口; 故障注入 = 篡改 observed 覆盖项。claim = 当前物理状态
    支持的逻辑归属 (None = 过渡态, 不计账实差异)。
    """

    def __init__(self):
        self.model = _FakeModel(self)
        self.data = _FakeData()
        self.vehicles = ["F01", "F02"]
        self.cone_mocapid = []
        self.pallet_bid = 999
        self.k = {"transport_z": 0.065}
        self.truth = {"claim": "SLOT:A-L3-C1", "xyz": list(POSE["SLOT:A-L3-C1"]),
                      "fork": {"F01": 0.03, "F02": 0.0}, "beam": 0.97,
                      "speed": 0.0}
        self.observed: dict = {}
        self._fall = None

    # ---- tick (fake mujoco.mj_step 驱动; 采样窗口内推进脚本时间线) ----
    def _tick(self):
        self.data.time += TIMESTEP
        if self._fall:
            f, t = self._fall, self.data.time
            if f["t0"] <= t <= f["end"]:
                a = (t - f["t0"]) / (f["end"] - f["t0"])
                x = f["x0"] + (f["x1"] - f["x0"]) * a
                z = f["z0"] + (f["z1"] - f["z0"]) * a
                self.truth["xyz"] = [x, 0.0, z]
                self.truth["beam"] = f["beam0"] * (1 - a)
                self.truth["fork"] = {v: s * (1 - a)
                                      for v, s in self.truth["fork"].items()}
                self.truth["speed"] = abs(f["z1"] - f["z0"]) / (
                    f["end"] - f["t0"])
            elif t > f["end"]:
                self.truth["xyz"] = list(POSE["FLOOR"])
                self.truth["beam"] = 0.0
                self.truth["fork"] = {v: 0.0 for v in self.vehicles}
                self.truth["speed"] = 0.0
                self.truth["claim"] = "FLOOR"
                self._fall = None

    def advance(self, duration: float):
        for _ in range(max(1, int(round(duration / TIMESTEP)))):
            self._tick()

    def animate_to(self, *, xyz=None, fork=None, beam=None, claim=None,
                   duration: float):
        """线性动画到目标 (确定性); 结束置静止。"""
        s_xyz = list(self.truth["xyz"])
        s_fork = dict(self.truth["fork"])
        s_beam = self.truth["beam"]
        n = max(1, int(round(duration / TIMESTEP)))
        p0 = np.array(s_xyz)
        p1 = np.array(xyz if xyz else s_xyz)
        for i in range(1, n + 1):
            self.data.time += TIMESTEP
            a = i / n
            if xyz:
                self.truth["xyz"] = [float(v) for v in p0 + (p1 - p0) * a]
            if fork:
                self.truth["fork"] = {
                    v: float(s_fork.get(v, 0.0)
                             + (fork.get(v, s_fork.get(v, 0.0))
                                - s_fork.get(v, 0.0)) * a)
                    for v in self.vehicles}
            if beam is not None:
                self.truth["beam"] = float(s_beam + (beam - s_beam) * a)
            self.truth["speed"] = float(
                np.linalg.norm(p1 - p0) / duration) if i < n else 0.0
        if xyz:
            self.truth["xyz"] = [float(v) for v in p1]
        if claim is not None:
            self.truth["claim"] = claim
        self.truth["speed"] = 0.0

    def arm_fall(self, delay=0.15, fall_dur=0.12):
        self._fall = {"t0": self.data.time + delay,
                      "end": self.data.time + delay + fall_dur,
                      "z0": self.truth["xyz"][2], "z1": POSE["FLOOR"][2],
                      "x0": self.truth["xyz"][0], "x1": POSE["FLOOR"][0],
                      "beam0": self.truth["beam"]}

    # ---- 生产侧只读接口 (observed 口径) ----
    @property
    def pallet_pose(self):
        o = self.observed.get("pose")
        p = o if o is not None else self.truth["xyz"]
        return np.array(p, float), np.array([1.0, 0.0, 0.0, 0.0])

    @property
    def pallet_bottom_z(self):
        o = self.observed.get("bottom_z")
        return float(o if o is not None
                     else self.truth["xyz"][2] - PALLET_HALF_H)

    def pallet_fork_contact(self, vid=None):
        o = self.observed.get(("contact", vid))
        if o is not None:
            return bool(o)
        return self.truth["fork"].get(vid, 0.0) >= 0.05

    def support_load_ratio(self, surface="beam", vid=None):
        o = self.observed.get(("support", surface, vid))
        if o is not None:
            return float(o)
        if surface == "fork":
            return self.truth["fork"].get(vid, 0.0)
        return self.truth["beam"]

    @property
    def axle_x(self):
        return float(self.data.qpos[0])

    def vehicle(self, vid):
        return types.SimpleNamespace(
            a_x=0, a_lift=1, qadr_x=0, qadr_lift=1, q0_x=0.0,
            start_axle_x=-0.620, axle_x=self.axle_x, vid=vid)

    # ---- evaluator 侧 (真值) ----
    def near(self, loc_key: str) -> bool:
        x, _, z = self.truth["xyz"]
        # POSE 存中心高度; zz 换算成支撑面 (托盘底面) 高度
        zx, _, zz = POSE[loc_key][0], POSE[loc_key][1], \
            POSE[loc_key][2] - PALLET_HALF_H
        return abs(x - zx) <= 0.012 and abs((z - PALLET_HALF_H) - zz) <= 0.006

    def truth_valid_pick(self, vid: str) -> bool:
        return (self.truth["fork"].get(vid, 0.0) >= 0.95
                and self.truth["beam"] <= 0.05 and self.truth["speed"] <= 5e-3)

    def truth_valid_place(self, vid: str) -> bool:
        return (self.truth["beam"] >= 0.95
                and self.truth["fork"].get(vid, 0.0) <= 0.05
                and self.near("SLOT:H01") and self.truth["speed"] <= 5e-3)

    def evaluator_snapshot(self) -> dict:
        return {"t": round(self.data.time, 4), "claim": self.truth["claim"],
                "fork": {k: round(v, 3) for k, v in self.truth["fork"].items()},
                "beam": round(self.truth["beam"], 3),
                "pallet_xyz": [round(v, 4) for v in self.truth["xyz"]],
                "observed_overrides": sorted(
                    str(k) for k in self.observed.keys())}


# ================================================================ harness 基础
class InjectedCrash(RuntimeError):
    pass


class AckLost(RuntimeError):
    """COMMIT 已落账, 调用方未收到 ACK 即死亡 (D06)。"""


class _FaultConn:
    """sqlite 代理: 武装状态下第 N 次 DML 写入注入故障 (事务内断电)。"""

    def __init__(self, real, fail_on_write):
        self._real = real
        self.fail_on_write = fail_on_write
        self.armed = False
        self.writes = 0

    def execute(self, sql, params=()):
        if self.armed and sql.lstrip().upper().startswith(
                ("INSERT", "UPDATE", "DELETE", "REPLACE")):
            self.writes += 1
            if self.writes == self.fail_on_write:
                raise sqlite3.OperationalError("INJECTED mid-commit fault")
        return self._real.execute(sql, params)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


def state_hash(w: World) -> str:
    h = hashlib.sha256()
    for t in TABLES:
        for row in w._conn.execute(f"SELECT * FROM {t}").fetchall():
            h.update(repr(row).encode())
    return h.hexdigest()[:16]


def seed_world(w: World) -> None:
    """与生产 _seed_ledger 相同的初始事实 (3 货位 + H01 + 3 托盘 + 3 货物)。"""
    w.register_slot(Slot("A-L3-C1", (0.0, 0.0, 0.400), SlotOcc.OCCUPIED))
    w.register_slot(Slot("A-L2-C1", (0.0, 0.0, 0.250), SlotOcc.OCCUPIED))
    w.register_slot(Slot("A-L1-C1", (0.0, 0.0, 0.100), SlotOcc.OCCUPIED))
    w.register_slot(Slot("H01", (-0.330, 0.0, 0.045), SlotOcc.EMPTY))
    for pid, slot in (("PAL-001", "A-L3-C1"), ("PAL-002", "A-L1-C1"),
                      ("PAL-003", "A-L2-C1")):
        w.register_pallet(Pallet(pid, 0.022, Loc(LocKind.SLOT, slot),
                                 PalletCargo.LOADED))
    for gid, pid, sku in (("G-001", "PAL-001", "SKU-1"),
                          ("G-002", "PAL-002", "SKU-2"),
                          ("G-003", "PAL-003", "SKU-3")):
        w.register_goods(GoodsUnit(gid, sku, 0.023,
                                   Loc(LocKind.PALLET, pid),
                                   GoodsLifecycle.IN_WAREHOUSE))
        w._conn.execute("INSERT INTO bindings VALUES(?,?,?,1)", (gid, pid, 1))
    w._conn.commit()


def ledger_pallet_loc(w: World) -> str:
    return w._conn.execute(
        "SELECT location FROM pallets WHERE pallet_id='PAL-001'"
    ).fetchone()[0]


def i4_fails(v: EvidenceVerdict, pid: str, t_now: float) -> list[str]:
    """I4 证据有效性 (身份正确 / 未失效) —— 门控档检查。"""
    out = []
    if v.pallet_id != pid:
        out.append(f"证据身份与命令不符 (I4): {v.pallet_id} ≠ {pid}")
    if t_now > v.valid_until + 1e-9:
        out.append(f"证据过期 (I4): now {t_now:.3f}s > valid_until "
                   f"{v.valid_until:.3f}s")
    return out


def naive_report_ok(ctx, phase: str, vid: str) -> bool:
    """B0「直接完成回报」判据 = 生产 _commit_* 前置单帧信号 (observed)。"""
    c = ctx.cell
    if phase == "PICK":
        return bool(c.pallet_fork_contact(vid)) and \
            c.pallet_bottom_z >= 0.400 + 0.003
    z = 0.045
    return (not c.pallet_fork_contact(vid)) and \
        (z - 0.002 <= c.pallet_bottom_z <= z + 0.005)


def sample_verdict_harness(ctx, kind: str, pid: str, vid: str
                           ) -> EvidenceVerdict:
    """B1/B2 的证据采样 = 生产 _sample_verdict 同构 (collect+build)。"""
    cell = ctx.cell
    per_frame = max(1, int(round(0.05 / TIMESTEP)))
    new_surf = "fork" if kind == "PICK" else "beam"
    old_surf = "beam" if kind == "PICK" else "fork"
    prev: dict = {"pos": None, "t": None}

    def probe():
        pos, _ = cell.pallet_pose
        t = float(cell.data.time)
        if prev["pos"] is None:
            speed = 0.0
        else:
            dt = max(t - prev["t"], 1e-9)
            speed = float(np.linalg.norm(
                np.asarray(pos) - prev["pos"]) / dt)
        prev.update(pos=np.asarray(pos), t=t)
        return EvidenceFrame(
            t=t, new_support=float(cell.support_load_ratio(new_surf, vid)),
            old_support=float(cell.support_load_ratio(old_surf, vid)),
            pallet_x=float(pos[0]), pallet_y=float(pos[1]),
            pallet_z=float(pos[2]), speed=speed)

    def step():
        for _ in range(per_frame):
            cell._tick()

    frames = collect_frames(step=step, probe=probe)
    return build_verdict(
        verdict_id=ctx.next_evidence_id(), kind=kind, task_id=ctx.task_ref,
        pallet_id=pid, vehicle_id=vid, lease_id=f"aisle:{vid}",
        frames=tuple(frames))


# ================================================================ 运行上下文
class Ctx:
    def __init__(self, group: str, fault_id: str, seed: int):
        self.group, self.fault_id, self.seed = group, fault_id, seed
        self.rng = random.Random(
            zlib.crc32(f"{seed}|{fault_id}|{group}".encode()) & 0xFFFFFFFF)
        self.run_id = f"ABL-{fault_id}-{group}-S{seed}"
        self.cell = FakeCell()
        self.db_path: str | None = None
        self.svc = None
        self._w: World | None = None
        self._ev_seq = itertools.count(1)
        self._retry_seq = itertools.count(1)
        self.audit: list[dict] = []
        self.cps: list[tuple] = []          # (t, ledger_loc, claim, note)
        self.incidents: list[dict] = []
        self.commits = 0
        self.unsafe = 0
        self.valid_ops = 0
        self.false_rej = 0
        self.atomic_faults = 0
        self.atomic_viol = 0
        self.safe_failure = False
        self._task_override = None
        # B4 工业强基线: 幂等表 (op_id → 状态) + 模拟 PLC 状态字 + journal
        self.b4_ops: dict[str, str] = {}
        self.plc = _PlcStub()
        self._journal: "_Journal | None" = None

    # ---------- world / service ----------
    @property
    def task_ref(self) -> str:
        return self._task_override or (
            self.svc.task_id if self.svc else self.run_id)

    @property
    def world(self) -> World:
        return self.svc.world if self.svc else self._w

    def next_evidence_id(self) -> str:
        return f"{self.run_id}-EVD-{next(self._ev_seq):04d}"

    def ensure_world(self):
        if self.group == "B3":
            svc = srv.WorldService(resume_cell=self.cell)
            svc.gate.speed = 64.0
            self.svc = svc
            self.db_path = svc.world.db_path
            svc._seed_ledger()
        else:
            self.db_path = str(RUNS / f"{self.run_id}-{time.time_ns()}.sqlite")
            self._w = World(run_id=self.run_id, db_path=self.db_path)
            seed_world(self._w)
        self.checkpoint("seeded")

    def restart_world(self):
        """B0/B1/B2 的「重启」= 重开账本 (无恢复机制可调用)。"""
        self._w._conn.close()
        self._w = World(run_id=self.run_id, db_path=self.db_path)

    def restart_service(self):
        """B3 的重启 = 生产 resume 路径 (build 内自动 recover)。"""
        t_restart = self.cell.data.time
        self.svc.world._conn.close()
        svc2 = srv.WorldService(resume_db=self.db_path, resume_cell=self.cell)
        svc2.gate.speed = 64.0
        self.svc = svc2
        return t_restart, list(svc2.last_recovery)

    @property
    def journal(self) -> "_Journal":
        """B4 前置写日志 (JSONL 落盘, 路径含 time_ns 防跨 run 撞名)。"""
        if self._journal is None:
            self._journal = _Journal(
                RUNS / f"{self.run_id}-{time.time_ns()}.b4.journal.jsonl")
        return self._journal

    def restart_b4(self):
        """B4 的重启 = 重开账本 + journal 对账 (设备字/进程分离存续:
        WMS 崩溃不拖垮设备, 状态字在 ctx 上保留)。"""
        self._w._conn.close()
        self._w = World(run_id=self.run_id, db_path=self.db_path)
        return float(self.cell.data.time), _b4_reconcile(self)

    # ---------- 观测 / 记账 ----------
    def checkpoint(self, note=""):
        self.cps.append((self.cell.data.time, ledger_pallet_loc(self.world),
                         self.cell.truth["claim"], note))

    def divergence_totals(self) -> tuple[float, float]:
        """(故障性分歧, 提交管线滞后) 仿真秒。

        - 故障性分歧: 非正常管线滞后造成的账实不一致 —— 不安全提交、
          崩溃后未恢复、拒绝后托盘离位等, 直到被恢复/合法提交闭合或
          持续到终局 (截尾);
        - 管线滞后: 物理完成 → 合法提交 之间的固有窗口 (B3 的 0.5s
          稳定采样窗是其安全机制的代价, 不计故障性分歧)。
        区分依据: 区间终点是**本转移的合法提交** (attempt:*:committed /
        acked_lost) → 归为管线滞后, 其余计入故障性分歧。
        """
        fault, lag = 0.0, 0.0
        for (t1, l1, c1, _), (t2, l2, c2, n2) in zip(self.cps, self.cps[1:]):
            if c1 is None or l1 == c1:
                continue
            dt = t2 - t1
            if n2.startswith("attempt:") and n2.endswith(
                    (":committed", ":acked_lost")):
                lag += dt
            else:
                fault += dt
        return round(fault, 4), round(lag, 4)

    def cleanup(self):
        try:
            if self.svc:
                self.svc.world._conn.close()
            elif self._w:
                self._w._conn.close()
        except sqlite3.Error:
            pass
        if self.db_path:
            for suf in ("", "-wal", "-shm"):
                p = Path(str(self.db_path) + suf)
                if p.exists():
                    p.unlink()
        if self._journal is not None and self._journal.path.exists():
            self._journal.path.unlink()


# ================================================================ legacy 写入器 (B0/B1)
def _write_steps(ctx, phase, vid, verdict):
    """旧式逐次 commit 步骤 (每步独立提交; B0 无证据步)。"""
    t = float(ctx.cell.data.time)
    eid = verdict.verdict_id if verdict else ""
    if phase == "PICK":
        to = f"ROBOT:{vid}"
        steps = [
            ("pallet_loc",
             ("UPDATE pallets SET location=?, entity_version=entity_version+1 "
              "WHERE pallet_id=?", (to, "PAL-001"))),
            ("slot_free", ("UPDATE slots SET occupancy=? WHERE slot_id=?",
                           (SlotOcc.EMPTY.value, "A-L3-C1"))),
            ("event", ("emit", "PalletMoved",
                       {"pallet_id": "PAL-001", "to_location": to,
                        "evidence_id": eid})),
        ]
        if verdict is not None and ctx.group != "B0":
            steps.append(("evidence", ("evidence", verdict)))
        steps.append(("transfer", ("transfer",
                                   f"{ctx.task_ref}:pick:PAL-001:A-L3-C1",
                                   eid, t)))
    else:
        to = "SLOT:H01"
        ev = [{"pallet_id": "PAL-001", "to_location": to,
               "evidence_id": eid},
              {"goods_id": "G-001", "evidence_id": eid}]
        steps = [
            ("pallet_loc",
             ("UPDATE pallets SET location=?, entity_version=entity_version+1 "
              "WHERE pallet_id=?", (to, "PAL-001"))),
            ("goods_lifecycle",
             ("UPDATE goods SET lifecycle=?, entity_version=entity_version+1 "
              "WHERE goods_id=?",
              (GoodsLifecycle.OUTBOUND_PENDING.value, "G-001"))),
            ("slot_occ", ("UPDATE slots SET occupancy=? WHERE slot_id=?",
                          (SlotOcc.OCCUPIED.value, "H01"))),
            ("event", ("emit2", [("PalletMoved", ev[0]),
                                 ("OutboundPending", ev[1])])),
        ]
        if verdict is not None and ctx.group != "B0":
            steps.append(("evidence", ("evidence", verdict)))
        steps.append(("transfer", ("transfer",
                                   f"{ctx.task_ref}:place:PAL-001:H01",
                                   eid, t)))
    return steps


def _run_legacy(ctx, steps, crash_after=None):
    w = ctx.world
    for i, (name, op) in enumerate(steps, 1):
        if crash_after is not None and i == crash_after + 1:
            raise InjectedCrash(f"legacy_write:{name}")
        kind = op[0]
        if kind == "emit":
            w.emit(ctx.task_ref, op[1], op[2], float(ctx.cell.data.time))
        elif kind == "emit2":
            for type_, payload in op[1]:
                w.emit(ctx.task_ref, type_, payload,
                       float(ctx.cell.data.time))
        elif kind == "evidence":
            v = op[1]
            d = json.loads(v.to_json())
            w._conn.execute(
                "INSERT INTO execution_evidence VALUES(?,?,?,?,?,?,?,?,?)",
                (v.verdict_id, v.schema_version, v.kind, v.pallet_id,
                 v.vehicle_id, v.task_id, v.to_json(),
                 float(d["valid_until"]), float(ctx.cell.data.time)))
            w._conn.commit()
        elif kind == "transfer":
            _, key, eid, t = op
            w._conn.execute(
                "INSERT INTO transfers VALUES(?,?,?,?)",
                (key, eid, json.dumps({"event_ids": [],
                                       "new_versions": {}}), t))
            w._conn.commit()
        else:
            w._conn.execute(op[0], op[1])
            w._conn.commit()


# ================================================================ B2 原子提交
def _b2_mutations_events(ctx, phase, vid, verdict):
    eid = verdict.verdict_id
    if phase == "PICK":
        to = f"ROBOT:{vid}"
        mut = [("UPDATE pallets SET location=?, entity_version=entity_version+1"
                " WHERE pallet_id=?", (to, "PAL-001")),
               ("UPDATE slots SET occupancy=? WHERE slot_id=?",
                (SlotOcc.EMPTY.value, "A-L3-C1"))]
        evs = [("PalletMoved", {"pallet_id": "PAL-001", "to_location": to,
                                "evidence_id": eid})]
    else:
        mut = [("UPDATE pallets SET location=?, entity_version=entity_version+1"
                " WHERE pallet_id=?", ("SLOT:H01", "PAL-001")),
               ("UPDATE goods SET lifecycle=?, entity_version=entity_version+1"
                " WHERE goods_id=?",
                (GoodsLifecycle.OUTBOUND_PENDING.value, "G-001")),
               ("UPDATE slots SET occupancy=? WHERE slot_id=?",
                (SlotOcc.OCCUPIED.value, "H01"))]
        evs = [("PalletMoved", {"pallet_id": "PAL-001",
                                "to_location": "SLOT:H01",
                                "evidence_id": eid}),
               ("OutboundPending", {"goods_id": "G-001",
                                    "evidence_id": eid})]
    return mut, evs


def _b2_expected(ctx, phase):
    w = ctx.world
    exp = {"pallets:PAL-001": w.entity_version_of("pallets:PAL-001")}
    if phase == "PLACE":
        exp["goods:G-001"] = w.entity_version_of("goods:G-001")
    return exp


# ================================================================ B4 工业强基线
B4_MAX_RETRIES = 2              # 失败腿自动重试上限 (首试 + 2)
B4_BACKOFF_S = 0.1              # 重试间仿真退避 (线性)


class _PlcStub:
    """B4 模拟 PLC 状态字 —— 设备侧唯一完成来源。

    「到点位」由设备自身动作程序置位 (任务状态机/设备字), 与 observed
    传感器通道和物理真值严格分道: B4 不读 cell.observed, 不读 cell.truth。
    设备字不感知负载/接触 —— H 类机械故障 (半托/滑落) 对状态字不可见,
    这是工业基线与 B3 证据门控的本质差异 (消融设计意图)。
    """
    ACK_POLL_MAX = 3            # 有限重询次数
    ACK_POLL_BACKOFF_S = 0.05   # 重询间仿真退避

    def __init__(self):
        self.words: dict[str, str] = {}     # op_id -> BUSY | DONE

    def issue(self, op_id: str) -> None:
        self.words[op_id] = "BUSY"

    def finish(self, op_id: str) -> None:
        """设备内: 动作程序到点 → 状态字置位 (模拟伺服 in-position)。"""
        if self.words.get(op_id) == "BUSY":
            self.words[op_id] = "DONE"

    def ack(self, op_id: str) -> bool:
        return self.words.get(op_id) == "DONE"

    def poll(self, op_id: str, cell) -> bool:
        """有限重询: 至多 ACK_POLL_MAX 次, 间隔退避推进仿真时钟。"""
        for i in range(self.ACK_POLL_MAX):
            if self.ack(op_id):
                return True
            cell.advance(self.ACK_POLL_BACKOFF_S * (i + 1))
        return self.ack(op_id)


class _Journal:
    """B4 前置写日志 (write-ahead): 每次状态变更先落一条 JSONL 再执行;
    崩溃/重启后折叠出每 op 最后状态, 依工业可得信息 (journal + 设备字 +
    超时) 对账 —— 不读真值/传感器。"""

    def __init__(self, path):
        self.path = Path(path)
        self.entries: list[dict] = []
        if self.path.exists():          # 进程级重启: 从文件恢复
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))

    def append(self, kind: str, *, t: float, op: str, **fields) -> None:
        e = {"kind": kind, "t": round(float(t), 4), "op": op, **fields}
        self.entries.append(e)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


class _NoVerdict:
    """B4 无证据门控: 复用 B2 变更集构造时以空 verdict_id 顶替。"""
    verdict_id = ""


def _b4_mutations_events(ctx, phase, vid):
    """B4 账本变更集 = 生产 SQL 语义, evidence_id 为空 (无证据档)。"""
    return _b2_mutations_events(ctx, phase, vid, _NoVerdict())


def _b4_leg(ctx, phase, vid, *, key, expected=None, crash=None, actuate=None):
    """B4 单腿协议: 幂等去重 → 预检 → 下发 → 设备字 ACK (有限重询) → 原子提交。

    - op_id = 转移幂等键: 跨重试/重启稳定;
    - 重复回报 → 生产 commit_transfer 幂等重放 (零重复事件/记账);
    - ACK 未置位 → 退避重试 ≤ B4_MAX_RETRIES → chain_failed (账本零写入);
    - 预检版本冲突为确定性失败 (盲重试会吞掉 C01 并发错误) → 立即失败;
    - crash 注入: ("pre_exec",) 下发后执行前 | ("pre_commit",) ACK 后提交前
      | ("faultconn", n) 提交事务内 | ("ack",) 提交后回执丢失;
    - actuate: None=该腿标准动作程序 | callable=场景注入动作 | False=设备
      无动作程序 (超时路径, 仅测试/对账语义)。
    返回 (status, reason, verdict_id, incident)。
    """
    w, cell, plc = ctx.world, ctx.cell, ctx.plc
    mut, evs = _b4_mutations_events(ctx, phase, vid)
    exp = expected if expected is not None else _b2_expected(ctx, phase)
    spec = {"mutations": [[sql, list(p)] for sql, p in mut],
            "events": [[ty, p] for ty, p in evs],
            "expected": dict(exp)}

    # 1) 幂等表: op 已提交 → 重复回报去重 (生产幂等重放, 零写入)
    if ctx.b4_ops.get(key) == "COMMITTED" or w.transfer_recorded(key):
        r = w.commit_transfer("", exp, mut, evs, key, task_id=ctx.task_ref,
                              sim_time_s=float(cell.data.time))
        ctx.b4_ops[key] = "COMMITTED"
        return ("committed",
                f"重复回报去重 (op_id 幂等重放 replayed={r.replayed})",
                "", None)

    # 2) 预检 (乐观版本, 下发动作前; 与生产提交内检查同语义)
    for entity, ver in exp.items():
        cur = w.entity_version_of(entity)
        if cur != ver:
            ctx.b4_ops[key] = "FAILED"
            ctx.safe_failure = True
            ctx.journal.append("FAILED", t=float(cell.data.time), op=key,
                               reason=f"预检版本冲突: {entity} v{ver}≠v{cur}")
            return ("chain_failed",
                    f"计划版本过期 (预检): {entity} 期望 v{ver}, 实际 v{cur}"
                    " → chain_failed (重派发需重规划)", "", None)

    if actuate is None:                 # 默认: 该腿标准动作程序
        actuate = (_physical_place_ok if phase == "PLACE"
                   else _physical_pick_ok)
    incident = None
    last_reason = "设备字未置位 (ACK 超时, 有限重询穷尽)"
    for attempt in range(1, B4_MAX_RETRIES + 2):
        ctx.journal.append("ISSUED", t=float(cell.data.time), op=key,
                           attempt=attempt, spec=spec)
        plc.issue(key)                  # 下发: 状态字 BUSY (未执行)
        if crash == ("pre_exec",):
            raise InjectedCrash("pre_exec")
        if actuate:                     # 设备执行动作程序, 到点即置位
            actuate(ctx, vid)
            plc.finish(key)
            ctx.journal.append("EXECUTED", t=float(cell.data.time), op=key,
                               attempt=attempt)
        if not plc.poll(key, cell):
            ctx.cell.advance(B4_BACKOFF_S * attempt)     # 退避后再询
            continue
        if crash == ("pre_commit",):
            raise InjectedCrash("pre_commit")
        proxy = None
        real = w._conn
        if crash and crash[0] == "faultconn" and attempt == 1:
            proxy = _FaultConn(real, crash[1])
            w._conn = proxy
            proxy.armed = True
        try:
            ctx.journal.append("COMMIT_BEGIN", t=float(cell.data.time),
                               op=key, attempt=attempt)
            w.commit_transfer("", exp, mut, evs, key, task_id=ctx.task_ref,
                              sim_time_s=float(cell.data.time))
        except sqlite3.OperationalError:
            last_reason = "提交事务中断 (原子回滚, 零残留)"
            ctx.journal.append("ERROR", t=float(cell.data.time), op=key,
                               attempt=attempt, reason=last_reason)
            if attempt > B4_MAX_RETRIES:
                break
            ctx.cell.advance(B4_BACKOFF_S * attempt)     # 退避后幂等重试
            continue
        except WorldError as e:     # 确定性拒绝 (提交内版本冲突等)
            ctx.b4_ops[key] = "FAILED"
            ctx.safe_failure = True
            ctx.journal.append("FAILED", t=float(cell.data.time), op=key,
                               reason=f"生产拒绝: {e}")
            return ("chain_failed", f"生产拒绝: {e} → chain_failed", "", None)
        finally:
            if proxy is not None:
                proxy.armed = False
                w._conn = real
        ctx.journal.append("COMMITTED", t=float(cell.data.time), op=key,
                           attempt=attempt)
        ctx.b4_ops[key] = "COMMITTED"
        if crash and crash[0] == "faultconn" and incident is None:
            # 事务内故障已由幂等重试闭合 —— 记 incident 供场景侧闭合
            incident = _mk_incident(ctx, key, ctx.fault_id, "mid_commit")
        if crash == ("ack",):
            raise AckLost(key)
        return ("committed", "设备字 ACK → 原子提交 (工业基线, 无证据门控)",
                "", incident)
    ctx.b4_ops[key] = "FAILED"
    ctx.safe_failure = True
    ctx.journal.append("FAILED", t=float(cell.data.time), op=key,
                       reason=last_reason)
    return ("chain_failed",
            f"{last_reason}; 重试 {B4_MAX_RETRIES} 次仍失败 → chain_failed",
            "", None)


def _b4_reconcile(ctx) -> list[dict]:
    """B4 重启对账: 只用 journal + 设备字 + 超时策略 (工业可得信息)。

    - EXECUTED/ACK 已置位且无终态 → 幂等重放提交 (committed_by_replay /
      already_committed);
    - 仅 ISSUED (设备未执行) → 有限重询 → 超时 → 标 failed 并回滚该腿
      (无 COMMITTED 即无账本写入; COMMIT_BEGIN 后的归属由 transfers 表
      幂等登记裁决, sqlite 原子性保证零残留/零重复);
    - 已有终态 (COMMITTED/FAILED) → 无事可做。
    不读 cell.truth / cell.observed —— 与 B3 的真值重观测对账严格分道。
    """
    w, plc = ctx.world, ctx.plc
    state: dict[str, dict] = {}
    for e in ctx.journal.entries:
        st = state.setdefault(e["op"], {"issued": False, "executed": False,
                                        "terminal": None, "spec": None})
        k = e["kind"]
        if k == "ISSUED":
            st["issued"] = True
            st["spec"] = e.get("spec") or st["spec"]
        elif k == "EXECUTED":
            st["executed"] = True
        elif k in ("COMMITTED", "FAILED"):
            st["terminal"] = k

    def _replay(op: str, st: dict) -> tuple[str, bool]:
        t0 = float(ctx.cell.data.time)
        if w.transfer_recorded(op):     # 提交已落账 (journal 缺终态行)
            ctx.b4_ops[op] = "COMMITTED"
            ctx.journal.append("COMMITTED", t=t0, op=op, attempt="reconcile")
            return "already_committed", True
        spec = st["spec"]
        try:
            w.commit_transfer("", spec["expected"], spec["mutations"],
                              spec["events"], op, task_id=ctx.task_ref,
                              sim_time_s=t0)
        except WorldError:
            ctx.b4_ops[op] = "FAILED"
            ctx.journal.append("FAILED", t=t0, op=op,
                               reason="reconcile_replay_rejected")
            return "replay_rejected_failed", False
        ctx.b4_ops[op] = "COMMITTED"
        ctx.journal.append("COMMITTED", t=t0, op=op, attempt="reconcile")
        return "committed_by_replay", True

    rep: list[dict] = []
    for op, st in state.items():
        if st["terminal"] is not None or not st["issued"]:
            continue
        t0 = float(ctx.cell.data.time)
        if st["executed"] or plc.ack(op):
            decision, recovered = _replay(op, st)
        else:
            for i in range(plc.ACK_POLL_MAX):      # 超时策略: 有限重询
                if plc.ack(op):
                    break
                ctx.cell.advance(plc.ACK_POLL_BACKOFF_S * (i + 1))
            if plc.ack(op) and st["spec"]:
                decision, recovered = _replay(op, st)
            else:
                ctx.b4_ops[op] = "FAILED"
                ctx.safe_failure = True
                ctx.journal.append("FAILED", t=float(ctx.cell.data.time),
                                   op=op, reason="reconcile_timeout")
                # 已达成一致决定并回滚该腿 → 一致性恢复 (任务链失败)
                decision, recovered = "timeout_failed_rolled_back", True
        rep.append({"command_id": op, "decision": decision,
                    "recovered": recovered,
                    "recovery_time_s":
                        round(float(ctx.cell.data.time) - t0, 4)})
    return rep


# ================================================================ 提交尝试 (组开关核心)
def commit_attempt(ctx, phase, vid="F01", *, forged=None, crash=None,
                   expected=None, reject_reason="", injection_point="",
                   actuate=None):
    """一次取/放提交尝试, 按组走不同机制组合。

    crash: None | ("entry",) | ("legacy_after", n) | ("faultconn", n)
           | ("ack",) | ("pre_exec",) | ("pre_commit",)     (后两个为 B4)
    actuate: 仅 B4 —— 腿动作程序 (None=标准动作, False=设备不执行)。
    返回 outcome dict; 审计记录按矩阵 V0.1 schema 落账。
    """
    w = ctx.world
    key = (f"{ctx.task_ref}:pick:PAL-001:A-L3-C1" if phase == "PICK"
           else f"{ctx.task_ref}:place:PAL-001:H01")
    pre_hash = state_hash(w)
    truth_snap = ctx.cell.evaluator_snapshot()
    verdict_id = ""
    status, reason = "rejected", reject_reason or "unknown"
    incident = None
    atomicity_fault = crash is not None and crash[0] in ("legacy_after",
                                                         "faultconn", "ack")
    if atomicity_fault:
        ctx.atomic_faults += 1

    try:
        if ctx.group == "B0":
            if crash == ("entry",):
                raise InjectedCrash("entry")
            report_ok = (forged.passed if forged is not None
                         else naive_report_ok(ctx, phase, vid))
            if not report_ok:
                status, reason = "rejected", "完成回报不成立 (report)"
            else:
                steps = _write_steps(ctx, phase, vid, forged)
                ca = crash[1] if crash and crash[0] == "legacy_after" else None
                _run_legacy(ctx, steps, ca)
                if crash == ("ack",):
                    raise AckLost(key)
                status, reason = "committed", "直接完成回报 (无验证)"
                verdict_id = forged.verdict_id if forged else ""

        elif ctx.group == "B1":
            if crash == ("entry",):
                raise InjectedCrash("entry")
            v = forged if forged is not None else \
                sample_verdict_harness(ctx, phase, "PAL-001", vid)
            verdict_id = v.verdict_id
            fails = list(v.failures) + i4_fails(v, "PAL-001",
                                                float(ctx.cell.data.time))
            if fails:
                status, reason = "rejected", "；".join(fails)
            else:
                steps = _write_steps(ctx, phase, vid, v)
                ca = crash[1] if crash and crash[0] == "legacy_after" else None
                _run_legacy(ctx, steps, ca)
                if crash == ("ack",):
                    raise AckLost(key)
                status, reason = "committed", "门控通过+逐次写入"

        elif ctx.group == "B2":
            if crash == ("entry",):
                raise InjectedCrash("entry")
            v = forged if forged is not None else \
                sample_verdict_harness(ctx, phase, "PAL-001", vid)
            verdict_id = v.verdict_id
            ident = [f for f in i4_fails(v, "PAL-001",
                                         float(ctx.cell.data.time))
                     if "身份" in f]
            if list(v.failures) or ident:
                status, reason = "rejected", "；".join(list(v.failures) + ident)
            else:                      # TTL 留给生产 commit_transfer
                mut, evs = _b2_mutations_events(ctx, phase, vid, v)
                exp = expected or _b2_expected(ctx, phase)
                real = w._conn
                proxy = None
                if crash and crash[0] == "faultconn":
                    proxy = _FaultConn(real, crash[1])
                    w._conn = proxy
                    proxy.armed = True
                try:
                    w.commit_transfer(
                        v.verdict_id, exp, mut, evs, key,
                        task_id=ctx.task_ref,
                        sim_time_s=float(ctx.cell.data.time),
                        evidence_json=v.to_json(), command_id="")
                    if crash == ("ack",):
                        raise AckLost(key)
                    status, reason = "committed", "门控+原子事务"
                except WorldError as e:
                    status, reason = "rejected", f"生产拒绝: {e}"
                except sqlite3.OperationalError:
                    status, reason = "crashed", "事务内注入故障 (已回滚)"
                    incident = _mk_incident(
                        ctx, key, "D-faultconn", "mid_commit")
                finally:
                    if proxy is not None:
                        proxy.armed = False
                        w._conn = real

        elif ctx.group == "B4":
            # B4 工业强基线: 不读证据/传感器/真值 —— 完成判据 = 模拟 PLC
            # 状态字 ACK; forged 注入对本组不可见 (协议分道)。
            if crash == ("entry",):
                raise InjectedCrash("entry")
            status, reason, verdict_id, incident = _b4_leg(
                ctx, phase, vid, key=key, expected=expected, crash=crash,
                actuate=actuate)

        else:  # B3 —— 生产路径
            svc = ctx.svc
            if crash == ("entry",):
                fname = "_commit_pick" if phase == "PICK" else "_commit_place"

                def boom(*a, **kw):
                    raise SystemExit(3)
                setattr(svc, fname, boom)
                try:
                    svc.aisle.acquire(vid)
                    try:
                        if phase == "PICK":
                            svc._commit_pick("L3", "PAL-001", vid)
                        else:
                            svc._commit_place("H01", "PAL-001", "G-001", vid)
                    finally:
                        svc.aisle.release(vid)
                except SystemExit:
                    status, reason = "crashed", "物理完成、BEGIN 前进程退出"
                    incident = _mk_incident(ctx, key, "D03", "before_begin")
                finally:
                    svc.__dict__.pop(fname, None)
            else:
                real = svc.world._conn
                proxy = None
                orig_commit = srv.World.commit_transfer
                if crash and crash[0] == "faultconn":
                    proxy = _FaultConn(real, crash[1])
                    svc.world._conn = proxy

                    def armed(*a, **kw):
                        proxy.armed = True
                        try:
                            return orig_commit(svc.world, *a, **kw)
                        finally:
                            proxy.armed = False
                    svc.world.commit_transfer = armed
                try:
                    svc.aisle.acquire(vid)
                    try:
                        if phase == "PICK":
                            ok = svc._commit_pick("L3", "PAL-001", vid)
                        else:
                            ok = svc._commit_place("H01", "PAL-001", "G-001",
                                                   vid)
                    finally:
                        svc.aisle.release(vid)
                    if ok:
                        if crash == ("ack",):
                            raise AckLost(key)
                        status = "committed"
                        reason = "PREPARED→VERIFIED→原子提交"
                        verdict_id = _last_evidence_id(ctx, key)
                    else:
                        status = "rejected"
                        # 拒绝原因取生产叙述 (证据门禁失败原因显式枚举)
                        narr = next((m["text"] for m in reversed(svc.chat)
                                     if "证据" in m["text"] or "提交被拒" in
                                     m["text"]), "")
                        reason = reject_reason or \
                            (narr[:120] or "证据门禁/前置拒绝 (生产路径)")
                        with svc.db_lock:
                            svc.world.emit(svc.task_id,
                                           "CommitEvidenceMissing",
                                           {"phase": phase},
                                           float(ctx.cell.data.time))
                        ctx.safe_failure = True
                except srv.ReconciliationRequired as e:
                    status = "reconciliation_required"
                    reason = f"提交通道故障: {e}"
                    incident = _mk_incident(ctx, key, "D04", "mid_commit")
                finally:
                    if proxy is not None:
                        svc.world.__dict__.pop("commit_transfer", None)
                        svc.world._conn = real

    except InjectedCrash as e:
        status, reason = "crashed", f"注入崩溃: {e}"
        if incident is None:
            incident = _mk_incident(ctx, key, ctx.fault_id,
                                    injection_point or "mid_write")
    except AckLost:
        status, reason = "acked_lost", "COMMIT 已落账, ACK 丢失 (调用方死亡)"
        incident = _mk_incident(ctx, key, "D06", "ack_boundary")
        incident["commit_landed"] = True

    landed = status in ("committed", "acked_lost")   # acked_lost: 已落账
    if landed:
        ctx.commits += 1
    post_hash = state_hash(w)
    post_truth = ctx.cell.evaluator_snapshot()

    # ---- evaluator: 真值判定 ----
    valid = bool(ctx.cell.truth_valid_pick(vid) if phase == "PICK"
                 else ctx.cell.truth_valid_place(vid))
    if valid:
        ctx.valid_ops += 1
    unsafe = False
    if landed:
        unsafe = not valid or post_truth["claim"] is None
        if unsafe:
            ctx.unsafe += 1
    elif status == "rejected" and valid:
        ctx.false_rej += 1

    # ---- I3 原子性残留检查 (仅对注入故障的提交尝试) ----
    inv = {"I1": True, "I2": True, "I3": True, "I4": True, "I5": True,
           "I6": None}
    if atomicity_fault or status == "crashed":
        if _residue_check(ctx, phase, vid):
            ctx.atomic_viol += 1
            inv["I3"] = False
    if landed:
        inv["I2"] = not unsafe
        inv["I4"] = _evidence_valid_in_db(ctx, verdict_id, key)
        to_loc = f"ROBOT:{vid}" if phase == "PICK" else "SLOT:H01"
        n_dup = _moved_events_to(ctx, to_loc)
        inv["I5"] = n_dup <= 1
        if n_dup > 1:
            ctx.atomic_viol += 1

    rec = {
        "run_id": ctx.run_id, "seed": ctx.seed, "command_id": key,
        "fault_id": ctx.fault_id,
        "injection_point": injection_point or crash_label(crash),
        "pre_state_hash": pre_hash, "evidence_id": verdict_id,
        "commit_outcome": status, "reason": reason,
        "post_state_hash": post_hash,
        "invariant_results": inv,
        "recovery_result": None,
        "evaluator": {"truth": truth_snap, "truth_valid": valid,
                      "unsafe": unsafe, "observed_after": post_truth},
        "group": ctx.group, "scenario": ctx.fault_id,
    }
    ctx.audit.append(rec)
    if incident is not None:
        incident["audit_index"] = len(ctx.audit) - 1
    ctx.checkpoint(f"attempt:{phase}:{status}")
    return {"status": status, "reason": reason, "key": key,
            "incident": incident}


def crash_label(crash):
    if crash is None:
        return ""
    return {"entry": "before_begin", "legacy_after": "mid_write",
            "faultconn": "mid_commit", "ack": "ack_boundary",
            "pre_exec": "after_issue", "pre_commit": "after_ack"}[crash[0]]


def _mk_incident(ctx, key, fault_id, point) -> dict:
    inc = {"fault_id": fault_id, "command_id": key, "injection_point": point,
           "t_crash": round(ctx.cell.data.time, 4), "recovered": False,
           "decision": "", "recovery_time_s": None}
    ctx.incidents.append(inc)
    return inc


def _last_evidence_id(ctx, key) -> str:
    row = ctx.world._conn.execute(
        "SELECT payload FROM events WHERE type='PalletMoved' ORDER BY "
        "event_id DESC").fetchone()
    if row:
        return json.loads(row[0]).get("evidence_id", "")
    return ""


def _pallet_moved_count(ctx) -> int:
    return ctx.world._conn.execute(
        "SELECT COUNT(*) FROM events WHERE type='PalletMoved'"
    ).fetchone()[0]


def _moved_events_to(ctx, to_loc: str) -> int:
    """同一目标位置的 PalletMoved 事件数 (I5 重复检测, 按 to_location)。"""
    n = 0
    for (p,) in ctx.world._conn.execute(
            "SELECT payload FROM events WHERE type='PalletMoved'"):
        if json.loads(p).get("to_location") == to_loc:
            n += 1
    return n


def _residue_check(ctx, phase, vid) -> bool:
    """I3: 提交尝试后残留必须 all-or-nothing (状态/事件/登记/货位一致)。"""
    w = ctx.world
    loc = ledger_pallet_loc(w)
    moved = (loc == f"ROBOT:{vid}") if phase == "PICK" else \
        (loc == "SLOT:H01")
    n_ev = _pallet_moved_count(ctx)
    has_key = w._conn.execute(
        "SELECT COUNT(*) FROM transfers WHERE idempotency_key LIKE ?",
        (f"%:{'pick' if phase == 'PICK' else 'place'}:PAL-001%",)
    ).fetchone()[0]
    slot = dict(w._conn.execute(
        "SELECT slot_id, occupancy FROM slots").fetchall())
    if phase == "PICK":
        consistent = (not moved) or (
            moved and n_ev >= 1 and has_key >= 1
            and slot["A-L3-C1"] == "EMPTY")
    else:
        goods = w._conn.execute(
            "SELECT lifecycle FROM goods WHERE goods_id='G-001'"
        ).fetchone()[0]
        consistent = (not moved) or (
            moved and n_ev >= 1 and has_key >= 1
            and slot["H01"] == "OCCUPIED"
            and goods == "OUTBOUND_PENDING")
    return not consistent


def _evidence_valid_in_db(ctx, verdict_id, key) -> bool:
    if not verdict_id:
        return False           # B0 结构性无证据 (I4 违反)
    row = ctx.world._conn.execute(
        "SELECT payload, sim_time_s FROM execution_evidence WHERE "
        "evidence_id=?", (verdict_id,)).fetchone()
    if row is None:
        return False
    d = json.loads(row[0])
    return d.get("pallet_id") == "PAL-001" and \
        row[1] <= d["valid_until"] + 1e-9


# ================================================================ 场景脚本
def _physical_pick_ok(ctx, vid="F01"):
    """取托完成态 = 举升离梁 +17mm (生产 commit 点: 仍在货位上方)。

    生产 _task_steps 在「举升离梁 (+17)」即触发 pick 提交 —— 提交时刻
    托盘仍在槽位上方 (bottom ≈ 槽位 +17mm), 之后才带载退出降运输高度;
    _commit_pick 前置 (bottom_z ≥ 槽位+3mm) 与此对齐。
    """
    j = ctx.rng.uniform(0.9, 1.1)
    ctx.cell.animate_to(xyz=(0.000, 0.0, 0.428), fork={vid: 0.97},
                        beam=0.03, claim=f"ROBOT:{vid}", duration=0.8 * j)
    ctx.cell.advance(0.25)
    ctx.checkpoint("pick_settled")


def _physical_pick_partial(ctx, vid="F01"):
    ctx.cell.animate_to(xyz=(-0.100, 0.0, 0.428), fork={vid: 0.80},
                        beam=0.20, claim=None, duration=0.8)
    ctx.cell.advance(0.25)
    ctx.checkpoint("pick_partial")


def _physical_place_ok(ctx, vid="F01"):
    j = ctx.rng.uniform(0.9, 1.1)
    ctx.cell.animate_to(xyz=POSE["SLOT:H01"], beam=0.97, fork={vid: 0.03},
                        claim="SLOT:H01", duration=0.8 * j)
    ctx.cell.advance(0.25)
    ctx.checkpoint("place_settled")


def _retry_clean_pick_place(ctx, vid="F01"):
    """安全失败后的重派发: 清障 → 重取 → 重放 → 终态。"""
    _physical_pick_ok(ctx, vid)
    out = commit_attempt(ctx, "PICK", vid, injection_point="retry")
    if out["status"] != "committed":
        return out
    _physical_place_ok(ctx, vid)
    return commit_attempt(ctx, "PLACE", vid, injection_point="retry")


def sc_S01(ctx):
    """货叉没接触托盘但高度信号显示已抬起 (接触/高度信号被篡改, 支撑诚实)。"""
    if ctx.group == "B3":
        _b3_prepare(ctx)
    ctx.cell.observed[("contact", "F01")] = True
    ctx.cell.observed["bottom_z"] = 0.4035
    ctx.cell.advance(0.6)
    out = commit_attempt(ctx, "PICK", injection_point="evidence_gate")
    ctx.cell.observed.pop(("contact", "F01"), None)
    ctx.cell.observed.pop("bottom_z", None)
    if out["status"] == "rejected":
        _retry_clean_pick_place(ctx)
    else:  # B0 已错误提交 → 放托回报必然不成立
        ctx.cell.advance(0.5)
        commit_attempt(ctx, "PLACE", injection_point="post_fault")


def sc_S05(ctx):
    """重放旧 EvidenceVerdict 超过有效窗 (期间托盘被外部放回货架)。"""
    if ctx.group == "B3":
        _b3_prepare(ctx)
    # 采证时刻: 合法举起
    ctx.cell.animate_to(xyz=POSE["ROBOT:F01"], fork={"F01": 0.97}, beam=0.03,
                        claim="ROBOT:F01", duration=0.8)
    v_stale = sample_verdict_harness(ctx, "PICK", "PAL-001", "F01")
    if ctx.group == "B3":
        orig = _b3_patch_sampler_i4(ctx, v_stale)
        # 完成信号 (接触/高度) 同样来自被劫持管线 → 生产前置放行,
        # 拒绝必须来自证据时效检查 (I4) 而非碰巧的前置失败
        ctx.cell.observed[("contact", "F01")] = True
        ctx.cell.observed["bottom_z"] = 0.420
    # 外部放回 + 超窗
    ctx.cell.animate_to(xyz=POSE["SLOT:A-L3-C1"], fork={"F01": 0.03},
                        beam=0.97, claim="SLOT:A-L3-C1", duration=0.5)
    ctx.cell.advance(1.0 + ctx.rng.uniform(0.0, 0.4))
    out = commit_attempt(ctx, "PICK", forged=v_stale,
                         injection_point="evidence_gate")
    if ctx.group == "B3":
        ctx.svc._sample_verdict = orig
        ctx.cell.observed.pop(("contact", "F01"), None)
        ctx.cell.observed.pop("bottom_z", None)
    if out["status"] == "rejected":
        _retry_clean_pick_place(ctx)
    else:
        ctx.cell.advance(0.5)
        commit_attempt(ctx, "PLACE", injection_point="post_fault")


def sc_S06(ctx):
    """用托盘 A (PAL-002) 的证据提交托盘 B (PAL-001) 的转移。"""
    if ctx.group == "B3":
        _b3_prepare(ctx)
        v_wrong = _synthetic_wrong_pallet_verdict(ctx)
        orig = _b3_patch_sampler_i4(ctx, v_wrong)
        # 冒名证据自带的完成信号让生产前置放行 → 拒绝必须来自
        # 证据-命令身份绑定检查 (I4)
        ctx.cell.observed[("contact", "F01")] = True
        ctx.cell.observed["bottom_z"] = 0.420
    else:
        v_wrong = _synthetic_wrong_pallet_verdict(ctx)
    out = commit_attempt(ctx, "PICK", forged=v_wrong,
                         injection_point="evidence_gate")
    if ctx.group == "B3":
        ctx.svc._sample_verdict = orig
        ctx.cell.observed.pop(("contact", "F01"), None)
        ctx.cell.observed.pop("bottom_z", None)
    if out["status"] == "rejected":
        _retry_clean_pick_place(ctx)
    else:
        ctx.cell.advance(0.5)
        commit_attempt(ctx, "PLACE", injection_point="post_fault")


def _b3_patch_sampler_i4(ctx, forged: EvidenceVerdict):
    """S05/S06: 证据管线被劫持, 送达过期/冒名 verdict。

    在 VERIFIED 阶段施加 I4 绑定检查 (身份/时效) —— 生产 `_sample_verdict`
    按命令托盘**构造**证据, 绑定由构造保证; 被劫持管线下该检查需显式
    化。不符即以门禁失败拒绝, 生产 `_commit_pick` 叙述并返回 False。
    返回原方法用于恢复。
    """
    svc = ctx.svc
    orig = svc._sample_verdict

    def patched(kind, pid, vid):
        v = forged
        extra = i4_fails(v, pid, float(ctx.cell.data.time))
        if extra:
            v = EvidenceVerdict(
                verdict_id=v.verdict_id, schema_version=v.schema_version,
                kind=v.kind, task_id=v.task_id, pallet_id=v.pallet_id,
                vehicle_id=v.vehicle_id, lease_id=v.lease_id,
                frames=v.frames, valid_until=v.valid_until,
                failures=tuple(v.failures) + tuple(extra))
        return v

    svc._sample_verdict = patched
    return orig


def _synthetic_wrong_pallet_verdict(ctx) -> EvidenceVerdict:
    frames = tuple(EvidenceFrame(
        t=ctx.cell.data.time + 0.05 * i, new_support=0.99, old_support=0.01,
        pallet_x=-0.526, pallet_y=0.0, pallet_z=0.076, speed=1e-4)
        for i in range(11))
    return build_verdict(verdict_id=ctx.next_evidence_id(), kind="PICK",
                         task_id=ctx.task_ref, pallet_id="PAL-002",
                         vehicle_id="F01", lease_id="aisle:F01",
                         frames=frames)


def sc_H01(ctx):
    """托盘抬升但叉臂承托比不足阈值 (真值=观测, 机械性故障)。"""
    if ctx.group == "B3":
        _b3_prepare(ctx)
    if ctx.group == "B4":
        # 半托机械态: 设备字只反映「到点」, 不承载支撑比 → 基线照常提交
        # (不安全提交) → 随后完整举升补完链条。B4 无稳定窗/支撑证据。
        out = commit_attempt(ctx, "PICK", injection_point="evidence_gate",
                             actuate=_physical_pick_partial)
        ctx.cell.animate_to(xyz=POSE["ROBOT:F01"], fork={"F01": 0.97},
                            beam=0.03, claim="ROBOT:F01", duration=0.4)
        ctx.cell.advance(0.2)
        commit_attempt(ctx, "PLACE", injection_point="post_fault")
        return
    _physical_pick_partial(ctx)
    out = commit_attempt(ctx, "PICK", injection_point="evidence_gate")
    if out["status"] == "rejected":
        _retry_clean_pick_place(ctx)
    else:  # B0 提前过户 → 物理随后完成 → 继续放托
        ctx.cell.animate_to(xyz=POSE["ROBOT:F01"], fork={"F01": 0.97},
                            beam=0.03, claim="ROBOT:F01", duration=0.4)
        ctx.cell.advance(0.2)
        _physical_place_ok(ctx)
        commit_attempt(ctx, "PLACE", injection_point="post_fault")


def sc_H03(ctx):
    """放托后托盘滑落, 瞬态接触货架 (瞬态期提交必错, 稳定窗捕获坠落)。"""
    if ctx.group == "B3":
        _b3_prepare(ctx)
    if ctx.group == "B4":
        out = commit_attempt(ctx, "PICK", injection_point="prefault")
        assert out["status"] == "committed", out
        # 滑落: 瞬态接触 H01 即「到点」→ 设备字置位 → B4 无稳定窗照常提交
        def _faulted_place(c, vid):
            c.cell.animate_to(xyz=(-0.330, 0.0, 0.057), beam=0.90,
                              fork={"F01": 0.03}, claim=None, duration=0.3)
            c.cell.arm_fall(delay=0.15, fall_dur=0.12)
        commit_attempt(ctx, "PLACE", injection_point="evidence_gate",
                       actuate=_faulted_place)
        ctx.cell.advance(0.5)               # 坠落完成 (B4 无法捕获)
        return
    _physical_pick_ok(ctx)
    out = commit_attempt(ctx, "PICK", injection_point="prefault")
    assert out["status"] == "committed", out
    # 滑落: 瞬态接触 H01 (0.3s 内) 后坠落落地
    ctx.cell.animate_to(xyz=(-0.330, 0.0, 0.057), beam=0.90,
                        fork={"F01": 0.03}, claim=None, duration=0.3)
    ctx.cell.arm_fall(delay=0.15, fall_dur=0.12)
    out = commit_attempt(ctx, "PLACE", injection_point="evidence_gate")
    if out["status"] == "rejected":
        ctx.cell.advance(0.4)                      # 坠落完成
        # 人工复位: 拾回叉臂 → 重新放托
        ctx.cell.animate_to(xyz=POSE["ROBOT:F01"], fork={"F01": 0.97},
                            beam=0.03, claim="ROBOT:F01", duration=0.5)
        _physical_place_ok(ctx)
        commit_attempt(ctx, "PLACE", injection_point="retry")
    else:
        # B0: 瞬态期已提交 → 坠落照常发生 → 账实分歧持续到终局
        ctx.cell.advance(0.5)


def _hang_restart(ctx, incident):
    """B0-B2 崩溃后: 重启但无任何在途记录 → 挂死。"""
    ctx.cell.advance(ctx.rng.uniform(1.5, 2.5))
    ctx.restart_world()
    ctx.checkpoint("restart_hung")
    incident["decision"] = "hung_no_record"
    _close_incident(ctx, incident, recovered=False)


def _close_incident(ctx, incident, recovered, decision="", t_close=None):
    incident["recovered"] = recovered
    incident["decision"] = decision or incident["decision"]
    if recovered and t_close is not None:
        incident["recovery_time_s"] = round(
            t_close - incident.get("t_restart", incident["t_crash"]), 4)
    if incident.get("audit_index") is not None:
        ctx.audit[incident["audit_index"]]["recovery_result"] = {
            k: v for k, v in incident.items()
            if k in ("recovered", "decision", "recovery_time_s")}


def _b3_recovery_ok(ctx, rep) -> bool:
    decisions = {r.get("decision") for r in rep}
    return bool(rep) and decisions <= {"cancelled_not_executed",
                                       "committed_new_evidence",
                                       "already_committed"} and \
        ctx.svc.world.open_prepared_commands() == []


def sc_D02(ctx):
    """PREPARED 成功后物理动作前进程退出。"""
    if ctx.group == "B4":
        # B4 对应注入: op 下发 (journal ISSUED) 后、设备执行前进程退出。
        out = commit_attempt(ctx, "PICK", crash=("pre_exec",),
                             injection_point="after_issue")
        inc = out["incident"]
        ctx.cell.advance(ctx.rng.uniform(1.5, 2.5))
        ctx.checkpoint("crash")
        t_restart, rep = ctx.restart_b4()
        inc["t_restart"] = t_restart
        ok = bool(rep) and all(r["recovered"] for r in rep)
        _close_incident(ctx, inc, recovered=ok,
                        decision=f"journal_reconcile:"
                                 f"{sorted({r['decision'] for r in rep})}",
                        t_close=ctx.cell.data.time)
        ctx.checkpoint("recovered")
        return                            # 腿已超时 failed → 链失败
    if ctx.group == "B3":
        svc = ctx.svc
        orig = srv.WorldService._prepare_task

        def crash_after_prepare(*a, **kw):    # 实例属性: 不绑定 self
            orig(svc, *a, **kw)
            raise SystemExit(2)
        svc._prepare_task = crash_after_prepare
        inc = _mk_incident(ctx, "", "D02", "after_prepare")
        try:
            _b3_prepare(ctx)
        except SystemExit:
            pass
        finally:
            svc.__dict__.pop("_prepare_task", None)
        ctx.cell.advance(ctx.rng.uniform(1.5, 2.5))
        ctx.checkpoint("crash")
        t_restart, rep = ctx.restart_service()
        inc["t_restart"] = t_restart
        ok = _b3_recovery_ok(ctx, rep)
        _close_incident(ctx, inc, recovered=ok,
                        decision=f"restart_recover:"
                                 f"{sorted({r.get('decision') for r in rep})}",
                        t_close=ctx.cell.data.time)
        ctx.checkpoint("recovered")
        if ok:
            svc = ctx.svc
            svc.task_id = svc.run_id + "-T01R1"     # 重派发新命令
            _b3_prepare(ctx)                        # 新命令先 PREPARED
            _physical_pick_ok(ctx)
            assert commit_attempt(ctx, "PICK",
                                  injection_point="retry")["status"] == \
                "committed"
            _physical_place_ok(ctx)
            commit_attempt(ctx, "PLACE", injection_point="retry")
    else:
        inc = _mk_incident(ctx, "", "D02", "before_physical")
        _hang_restart(ctx, inc)


def sc_D03(ctx):
    """物理动作完成、BEGIN 前进程退出。"""
    if ctx.group == "B4":
        # B4 对应注入: 设备已执行 (EXECUTED+ACK) 但提交未落账时崩溃。
        out = commit_attempt(ctx, "PICK", crash=("pre_commit",),
                             injection_point="before_begin")
        inc = out["incident"]
        ctx.cell.advance(ctx.rng.uniform(1.5, 2.5))
        ctx.checkpoint("crash")
        t_restart, rep = ctx.restart_b4()
        inc["t_restart"] = t_restart
        ok = bool(rep) and all(r["recovered"] for r in rep) and \
            ctx.world.transfer_recorded(out["key"])
        _close_incident(ctx, inc, recovered=ok,
                        decision=f"journal_reconcile:"
                                 f"{sorted({r['decision'] for r in rep})}",
                        t_close=ctx.cell.data.time)
        ctx.checkpoint("recovered")
        if ok:
            commit_attempt(ctx, "PLACE", injection_point="post_recovery")
        return
    if ctx.group == "B3":
        _b3_prepare(ctx)
    _physical_pick_ok(ctx)
    out = commit_attempt(ctx, "PICK", crash=("entry",),
                         injection_point="before_begin")
    inc = out["incident"]
    if ctx.group == "B3":
        ctx.cell.advance(ctx.rng.uniform(1.5, 2.5))
        ctx.checkpoint("crash")
        t_restart, rep = ctx.restart_service()
        inc["t_restart"] = t_restart
        ok = _b3_recovery_ok(ctx, rep)
        _close_incident(ctx, inc, recovered=ok,
                        decision=f"restart_recover:"
                                 f"{sorted({r.get('decision') for r in rep})}",
                        t_close=ctx.cell.data.time)
        ctx.checkpoint("recovered")
        if ok:
            _b3_prepare_place_only(ctx)
            _physical_place_ok(ctx)
            commit_attempt(ctx, "PLACE", injection_point="post_recovery")
    else:
        _hang_restart(ctx, inc)


def sc_D04(ctx):
    """事务中托盘位置更新后异常 (B0/B1: 第 n 步后崩溃; B2/B3: 事务内断开)。"""
    if ctx.group == "B4":
        # B4: 提交事务内故障 → 原子回滚零残留 → 腿内幂等重试 (≤2) 闭合
        out = commit_attempt(ctx, "PICK", crash=("faultconn", 1),
                             injection_point="mid_commit")
        inc = out["incident"]
        if inc is None:                    # 重试闭合路径: 手动补记 incident
            inc = _mk_incident(ctx, out["key"], "D04", "mid_commit")
        ok = out["status"] == "committed"
        _close_incident(ctx, inc, recovered=ok,
                        decision="atomic_rollback+idempotent_retry:"
                                 f"{out['status']}",
                        t_close=ctx.cell.data.time)
        ctx.checkpoint("recovered")
        if ok:
            commit_attempt(ctx, "PLACE", injection_point="post_recovery")
        return
    if ctx.group == "B3":
        _b3_prepare(ctx)
    _physical_pick_ok(ctx)
    if ctx.group in ("B0", "B1"):
        n = ctx.rng.choice([1, 2, 3])
        out = commit_attempt(ctx, "PICK", crash=("legacy_after", n),
                             injection_point="mid_write")
        _hang_restart(ctx, out["incident"])
    elif ctx.group == "B2":
        n = ctx.rng.choice([1, 2, 3, 4])
        out = commit_attempt(ctx, "PICK", crash=("faultconn", n),
                             injection_point="mid_commit")
        # 原子回滚 → 零残留 → 但无恢复机制: 挂死, 分歧持续
        _hang_restart(ctx, out["incident"])
    else:
        n = ctx.rng.choice([1, 2, 3, 4])
        out = commit_attempt(ctx, "PICK", crash=("faultconn", n),
                             injection_point="mid_commit")
        inc = out["incident"]
        svc = ctx.svc
        ctx.cell.advance(ctx.rng.uniform(0.5, 1.0))     # 人工核对延迟
        r = svc.reconcile(out["key"])
        ok = bool(r.get("ok"))
        _close_incident(ctx, inc, recovered=ok,
                        decision=f"manual_reconcile:{'ok' if ok else 'fail'}",
                        t_close=ctx.cell.data.time)
        ctx.checkpoint("reconciled")
        if ok:
            _physical_place_ok(ctx)
            commit_attempt(ctx, "PLACE", injection_point="post_recovery")


def sc_D06(ctx):
    """COMMIT 成功但调用方未收到 ACK。"""
    if ctx.group == "B4":
        # B4: 提交已落账 (journal COMMITTED + transfers 幂等登记) 而回执
        # 丢失 → 重启对账判 already_committed, op_id 去重保证零重复事件。
        out = commit_attempt(ctx, "PICK", crash=("ack",),
                             injection_point="ack_boundary")
        inc = out["incident"]
        ctx.cell.advance(ctx.rng.uniform(1.5, 2.5))
        t_restart, rep = ctx.restart_b4()
        inc["t_restart"] = t_restart
        ok = (ctx.world.transfer_recorded(out["key"])
              and _pallet_moved_count(ctx) == 1)
        _close_incident(ctx, inc, recovered=ok,
                        decision="closed_in_journal (op_id 幂等去重)",
                        t_close=ctx.cell.data.time)
        ctx.checkpoint("recovered")
        if ok:
            commit_attempt(ctx, "PLACE", injection_point="post_recovery")
        return
    if ctx.group == "B3":
        _b3_prepare(ctx)
    _physical_pick_ok(ctx)
    out = commit_attempt(ctx, "PICK", crash=("ack",),
                         injection_point="ack_boundary")
    inc = out["incident"]
    ctx.cell.advance(ctx.rng.uniform(1.5, 2.5))
    if ctx.group == "B3":
        # pick 阶段已在提交事务内闭合 (重启扫描对其无事可做); place 命令
        # 未执行 → 恢复安全取消; 零重复 PalletMoved 即 D06 期望达成。
        t_restart, rep = ctx.restart_service()
        inc["t_restart"] = t_restart
        ok = _b3_recovery_ok(ctx, rep) and _pallet_moved_count(ctx) == 1
        _close_incident(ctx, inc, recovered=ok,
                        decision="closed_in_transaction + place_safe_cancel",
                        t_close=ctx.cell.data.time)
        ctx.checkpoint("recovered")
        if ok:
            _b3_prepare_place_only(ctx)
            _physical_place_ok(ctx)
            commit_attempt(ctx, "PLACE", injection_point="post_recovery")
    elif ctx.group == "B2":
        # 调用方超时重试 → 生产幂等重放返回首次结果
        ctx.restart_world()
        before_ev = _pallet_moved_count(ctx)
        v = sample_verdict_harness(ctx, "PICK", "PAL-001", "F01")
        mut, evs = _b2_mutations_events(ctx, "PICK", "F01", v)
        try:
            r = ctx.world.commit_transfer(
                v.verdict_id,
                {"pallets:PAL-001":
                 ctx.world.entity_version_of("pallets:PAL-001")},
                mut, evs, out["key"], task_id=ctx.task_ref,
                sim_time_s=float(ctx.cell.data.time),
                evidence_json=v.to_json(), command_id="")
            zero_dup = _pallet_moved_count(ctx) == before_ev
            ok = bool(r.replayed) and zero_dup
            _close_incident(ctx, inc, recovered=ok,
                            decision=f"idempotent_retry:replayed={r.replayed}",
                            t_close=ctx.cell.data.time)
        except (WorldError, sqlite3.Error) as e:
            _close_incident(ctx, inc, recovered=False,
                            decision=f"retry_failed:{e!r}")
        ctx.checkpoint("retry")
        if inc["recovered"]:
            _physical_place_ok(ctx)
            commit_attempt(ctx, "PLACE", injection_point="post_recovery")
    else:
        # B0/B1: 重试 → 无幂等检查 → 重复写入 + 撞键
        ctx.restart_world()
        try:
            v = None if ctx.group == "B0" else \
                sample_verdict_harness(ctx, "PICK", "PAL-001", "F01")
            _run_legacy(ctx, _write_steps(ctx, "PICK", "F01", v), None)
            ok = True   # 不该到这里: transfers 撞键
        except sqlite3.IntegrityError as e:
            ok = False
            ctx.atomic_viol += 1
            rec = ctx.audit[inc["audit_index"]]
            rec["invariant_results"]["I5"] = False
            _close_incident(ctx, inc, recovered=False,
                            decision=f"duplicate_commit "
                                     f"(PalletMoved×{_pallet_moved_count(ctx)}, {e})")
        ctx.checkpoint("retry")


def sc_C01(ctx):
    """两机器人持同一托盘旧版本任务并发提交 (第二车的完成信号被污染)。"""
    first = ctx.rng.choice(["F01", "F02"])
    second = "F02" if first == "F01" else "F01"
    plan_v0 = {"pallets:PAL-001": 0, "goods:G-001": 0}
    if ctx.group == "B3":
        _b3_prepare(ctx, vid=first)          # T01: first 车 L3→H01
        t02 = f"{ctx.svc.run_id}-T02"
        c2 = f"{t02}:pick:PAL-001:A-L3-C1"
        with ctx.svc.db_lock:
            ctx.svc.world.prepare_transfer(
                c2, task_id=t02,
                intent={"vehicle": second, "pallet_id": "PAL-001",
                        "goods_id": "G-001", "src": "L3", "dst": "H01",
                        "phase": "PICK"},
                resource_lease=f"aisle:{second}",
                expected_versions=plan_v0,
                sim_time_s=float(ctx.cell.data.time))
    if ctx.group == "B4":
        # 第一车正常腿; 第二车持旧计划 → 下发前预检版本冲突 (确定性失败,
        # 盲重试会吞掉并发错误) → chain_failed, 第二车不动作。
        out1 = commit_attempt(ctx, "PICK", first, expected=plan_v0,
                              injection_point="first_committer")
        assert out1["status"] == "committed", out1
        ctx._task_override = f"{ctx.run_id}-T02"
        out2 = commit_attempt(ctx, "PICK", second, expected=plan_v0,
                              injection_point="concurrent_commit")
        ctx._task_override = None
        assert out2["status"] == "chain_failed", out2
        # 落托由腿协议内动作程序执行 (单次 place_settled, 与 B0-B2 同形)
        commit_attempt(ctx, "PLACE", first, injection_point="first_committer")
        return
    _physical_pick_ok(ctx, first)
    out1 = commit_attempt(ctx, "PICK", first, expected=plan_v0,
                          injection_point="first_committer")
    assert out1["status"] == "committed", out1
    # 第二车: 旧版本任务 + 完成信号污染 (C01 并发提交竞争)
    ctx.cell.observed[("contact", second)] = True
    ctx.cell.observed[("support", "fork", second)] = 0.97
    ctx.cell.observed["bottom_z"] = 0.420     # 虚报「已举升过梁」
    if ctx.group == "B3":
        c2 = f"{ctx.svc.run_id}-T02:pick:PAL-001:A-L3-C1"
        try:
            with ctx.svc.db_lock:
                ctx.svc.world.revalidate_prepared(c2)
            out2 = {"status": "rejected", "reason": "unreachable"}
        except WorldError as e:
            out2 = {"status": "rejected",
                    "reason": f"旧计划拒绝 (C02 重验): {e}"}
            with ctx.svc.db_lock:
                ctx.svc.world.emit(ctx.svc.task_id, "StalePlanRejected",
                                   {"command_id": c2, "reason": str(e)},
                                   float(ctx.cell.data.time))
                ctx.svc.world.cancel_prepared(
                    c2, reason="C01 并发提交被版本检查拒绝",
                    sim_time_s=float(ctx.cell.data.time))
            ctx.safe_failure = True
            ctx.audit.append({
                "run_id": ctx.run_id, "seed": ctx.seed, "command_id": c2,
                "fault_id": "C01", "injection_point": "concurrent_commit",
                "pre_state_hash": state_hash(ctx.world),
                "evidence_id": "", "commit_outcome": "rejected",
                "reason": out2["reason"],
                "post_state_hash": state_hash(ctx.world),
                "invariant_results": {"I1": True, "I2": True, "I3": True,
                                      "I4": True, "I5": True, "I6": None},
                "recovery_result": None,
                "evaluator": {"truth": ctx.cell.evaluator_snapshot(),
                              "truth_valid": False, "unsafe": False,
                              "observed_after":
                                  ctx.cell.evaluator_snapshot()},
                "group": ctx.group, "scenario": "C01"})
    else:
        ctx._task_override = f"{ctx.run_id}-T02"
        out2 = commit_attempt(ctx, "PICK", second, expected=plan_v0,
                              injection_point="concurrent_commit")
        ctx._task_override = None
    ctx.cell.observed.pop(("contact", second), None)
    ctx.cell.observed.pop(("support", "fork", second), None)
    ctx.cell.observed.pop("bottom_z", None)
    # 第一车完成链条 (B0/B1 的错误提交将被合法放托覆盖, 但 Unsafe 已记)
    _physical_place_ok(ctx, first)
    commit_attempt(ctx, "PLACE", first, injection_point="first_committer")


# ---------- B3 生产路径辅助 ----------
def _b3_prepare(ctx, vid="F01"):
    svc = ctx.svc
    t = float(ctx.cell.data.time)
    with svc.db_lock:
        pick_cmd, place_cmd = svc._prepare_task(
            vid, "PAL-001", "G-001", "L3", "H01", t)
        for c in (pick_cmd, place_cmd):
            svc.world.revalidate_prepared(c)
            svc.world.advance_stage(c, TransferStage.EXECUTING,
                                    sim_time_s=t, note="ablation physical")
    return pick_cmd, place_cmd


def _b3_prepare_place_only(ctx, vid="F01"):
    """恢复取消原命令后, 以新命令重派发放托 (agent 重规划)。"""
    svc = ctx.svc
    svc.task_id = f"{svc.run_id}-T01R{next(ctx._retry_seq)}"
    key = f"{svc.task_id}:place:PAL-001:H01"
    t = float(ctx.cell.data.time)
    with svc.db_lock:
        svc.world.prepare_transfer(
            key, task_id=svc.task_id,
            intent={"vehicle": vid, "pallet_id": "PAL-001",
                    "goods_id": "G-001", "src": f"ROBOT:{vid}",
                    "dst": "H01", "phase": "PLACE"},
            resource_lease=f"aisle:{vid}",
            expected_versions={"pallets:PAL-001":
                               svc.world.entity_version_of("pallets:PAL-001"),
                               "goods:G-001":
                               svc.world.entity_version_of("goods:G-001")},
            sim_time_s=t)
        svc.world.advance_stage(key, TransferStage.EXECUTING, sim_time_s=t,
                                note="replan place after recovery")


SCRIPTS = {
    "S01": sc_S01, "S05": sc_S05, "S06": sc_S06,
    "D02": sc_D02, "D03": sc_D03, "D04": sc_D04, "D06": sc_D06,
    "H01": sc_H01, "H03": sc_H03, "C01": sc_C01,
}

SCENARIO_META = {
    "S01": ("A 感知与证据故障", "取托核对时刻: 接触/高度信号虚报已抬起"),
    "S05": ("A 感知与证据故障", "重放旧 EvidenceVerdict 超过有效窗"),
    "S06": ("A 感知与证据故障", "用托盘 A 的证据提交托盘 B 的转移"),
    "D02": ("B SQLite 事务与持久化故障", "PREPARED 后物理前进程退出"),
    "D03": ("B SQLite 事务与持久化故障", "物理完成、BEGIN 前进程退出"),
    "D04": ("B SQLite 事务与持久化故障", "事务中托盘位置更新后异常"),
    "D06": ("B SQLite 事务与持久化故障", "COMMIT 成功但 ACK 未送达"),
    "H01": ("C 部分交接与物理异常", "抬升但叉臂承托比不足阈值"),
    "H03": ("C 部分交接与物理异常", "托盘滑落后瞬态接触货架"),
    "C01": ("D 多车并发与恢复", "两机器人同托盘旧版本任务并发提交"),
}


# ================================================================ 运行与指标
def run_one(group: str, fault_id: str, seed: int) -> dict:
    ctx = Ctx(group, fault_id, seed)
    try:
        ctx.ensure_world()
        SCRIPTS[fault_id](ctx)
        ctx.checkpoint("final")
        w = ctx.world
        pal = w.pallet("PAL-001")
        slots = dict(w._conn.execute(
            "SELECT slot_id, occupancy FROM slots").fetchall())
        goods = w._conn.execute(
            "SELECT lifecycle FROM goods WHERE goods_id='G-001'").fetchone()
        completed = bool(
            pal and pal[0] == "SLOT:H01" and pal[1] == "LOADED"
            and slots.get("H01") == "OCCUPIED"
            and slots.get("A-L3-C1") == "EMPTY"
            and goods and goods[0] == "OUTBOUND_PENDING"
            and ctx.cell.truth["claim"] == "SLOT:H01")
        div_fault, div_lag = ctx.divergence_totals()
        metrics = {
            "unsafe_commit_rate":
                round(ctx.unsafe / ctx.commits, 4) if ctx.commits else None,
            "commits": ctx.commits, "unsafe": ctx.unsafe,
            "false_rejection_rate":
                round(ctx.false_rej / ctx.valid_ops, 4)
                if ctx.valid_ops else None,
            "valid_ops": ctx.valid_ops, "false_rejections": ctx.false_rej,
            "atomicity_violation_rate":
                round(ctx.atomic_viol / ctx.atomic_faults, 4)
                if ctx.atomic_faults else None,
            "atomic_fault_attempts": ctx.atomic_faults,
            "atomicity_violations": ctx.atomic_viol,
            "recovery_success_rate":
                round(sum(1 for i in ctx.incidents if i["recovered"])
                      / len(ctx.incidents), 4) if ctx.incidents else None,
            "incidents": len(ctx.incidents),
            "recovery_time_s": _mean([i["recovery_time_s"]
                                      for i in ctx.incidents
                                      if i["recovered"] is not None]),
            "task_completion": 1 if completed else 0,
            "divergence_duration_s": div_fault,
            "commit_lag_divergence_s": div_lag,
            "n_pallet_moved_events": _pallet_moved_count(ctx),
        }
        return {"run_id": ctx.run_id, "group": group,
                "scenario": fault_id, "seed": seed, "metrics": metrics,
                "incidents": ctx.incidents, "audit": ctx.audit,
                "timeline_checkpoints": [
                    {"t": round(t, 3), "ledger": l, "truth": c, "note": n}
                    for t, l, c, n in ctx.cps]}
    finally:
        ctx.cleanup()


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


METRIC_KEYS = ["unsafe_commit_rate", "false_rejection_rate",
               "atomicity_violation_rate", "recovery_success_rate",
               "recovery_time_s", "task_completion",
               "divergence_duration_s"]


def aggregate(runs: list[dict]) -> dict:
    agg: dict = {"by_group": {}, "by_group_scenario": {}}
    for g in sorted({r["group"] for r in runs}):
        rs = [r for r in runs if r["group"] == g]
        agg["by_group"][g] = {
            k: _mean([r["metrics"][k] for r in rs]) for k in METRIC_KEYS}
        agg["by_group"][g]["n_runs"] = len(rs)
    for (g, s) in sorted({(r["group"], r["scenario"]) for r in runs}):
        rs = [r for r in runs if r["group"] == g and r["scenario"] == s]
        agg["by_group_scenario"][f"{g}|{s}"] = {
            k: _mean([r["metrics"][k] for r in rs]) for k in METRIC_KEYS}
    return agg


def hypothesis_check(agg: dict) -> dict:
    bg = agg["by_group"]

    def m(g, k):
        v = bg.get(g, {}).get(k)
        return v if v is not None else 0.0

    return {
        "B0_unsafe_gt_0": {
            "pass": m("B0", "unsafe_commit_rate") > 0,
            "detail": f"B0 unsafe={m('B0', 'unsafe_commit_rate')}"},
        "B3_unsafe_zero": {
            "pass": bg.get("B3", {}).get("unsafe_commit_rate") == 0,
            "detail": f"B3 unsafe={bg.get('B3', {}).get('unsafe_commit_rate')}"},
        "B3_atomicity_zero": {
            "pass": bg.get("B3", {}).get("atomicity_violation_rate") == 0,
            "detail": f"B3 atomicity="
                      f"{bg.get('B3', {}).get('atomicity_violation_rate')}"},
        "B3_recovery_ge_B2": {
            "pass": m("B3", "recovery_success_rate")
            >= m("B2", "recovery_success_rate"),
            "detail": f"B3={m('B3', 'recovery_success_rate')} "
                      f"B2={m('B2', 'recovery_success_rate')}"},
        "B3_completion_not_collapsed": {
            "pass": m("B3", "task_completion") >= 0.9
            and m("B3", "task_completion") >= m("B2", "task_completion"),
            "detail": f"B3={m('B3', 'task_completion')} "
                      f"B2={m('B2', 'task_completion')} "
                      f"B0={m('B0', 'task_completion')}"},
        "B3_divergence_lt_B0": {
            "pass": m("B3", "divergence_duration_s")
            < m("B0", "divergence_duration_s"),
            "detail": f"B3={m('B3', 'divergence_duration_s')}s "
                      f"B0={m('B0', 'divergence_duration_s')}s"},
    }


MODELING_NOTES = [
    "物理层 = FakeCell 运动学孪生 (truth/observed 分离), 经 "
    "WorldService(resume_cell=) 与 fake mujoco 注入; PREPARED/EvidenceVerdict"
    "/commit_transfer/recover_open_commands/reconcile 均为生产代码路径。",
    "B0/B1 旧式逐次 commit 为 harness 侧 legacy 写入器 (每条 DML 独立提交, "
    "无版本/幂等前置检查), SQL 语义与生产 mutation 一致。",
    "I4 (证据身份/时效) 属门控档: B1/B2 harness 提交前检查; B3 以 VERIFIED "
    "阶段绑定检查实现 (生产构造即绑定, 注入经 _sample_verdict 实例补丁送达)。",
    "重派发仅在系统安全失败信号 (拒绝事件/StalePlanRejected/恢复取消) 后允许; "
    "崩溃无记录 = 挂死 —— B0-B2 的 D02/D03/D04 与 B0/B1 的 D06 结局。",
    "Divergence = 账本托盘位置 ≠ 物理逻辑归属 (claim; 过渡态不计) 的仿真时长;"
    "拆分为故障性 (divergence_duration_s, 主指标) 与管线滞后"
    "(commit_lag_divergence_s, 物理完成→合法提交的固有窗口)。",
    "本 3-seed 快验只验方向 (B0 Unsafe>0 / B3 零 Unsafe 且完成率不塌); "
    "30-seed 主实验待 DetFix 回归后。",
    "B4 工业强基线: 完成判据 = 模拟 PLC 状态字 (伺服到点位, 任务状态机侧), "
    "不读 observed 传感器与物理真值; op_id 幂等去重 + journal 前置日志 + "
    "重启对账 (超时→failed+回滚) + 失败腿重试≤2 → S/D 类故障下账实一致 "
    "(vs B0 挂死/重复); 但 H 类机械故障 (半托/滑落) 对设备字不可见 → "
    "不安全提交仍发生 —— 证据门控 (B1-B3) 的增量价值所在。",
]


def demo_contrast(fault_id: str, seed: int) -> int:
    """故障对照演示: 同一故障下 B0 (普通系统) vs B3 (证据系统) 并排叙事。

    用途: 汇报/视频的实证脚本 —— 证明"产品价值差异"不是话术:
    普通系统在故障下报出**错误完成态**, 证据系统拒绝、隔离、对账后闭合。
    """
    if fault_id not in SCRIPTS:
        print(f"未知场景 {fault_id}; 可用: {', '.join(SCRIPTS)}")
        return 2
    cat, desc = SCENARIO_META[fault_id]
    print("=" * 72)
    print(f"故障对照演示 · {fault_id} [{cat}] · seed={seed}")
    print(f"场景: {desc}")
    print("=" * 72)
    verdicts = {}
    for g, title in (("B0", "B0 普通系统 (无门控/无原子事务/无对账)"),
                     ("B3", "B3 证据系统 (PREPARED→证据→原子提交→对账)")):
        r = run_one(g, fault_id, seed)
        m = r["metrics"]
        print(f"\n--- {title} ---")
        for cp in r["timeline_checkpoints"]:
            same = "一致" if cp["ledger"] == cp["truth"] else "**分歧**"
            extra = f"  [{cp['note']}]" if cp.get("note") else ""
            print(f"  t={cp['t']:>6.2f}s  账本={cp['ledger'] or '—':<18}"
                  f"真值={cp['truth'] or '—':<18}{same}{extra}")
        for inc in r["incidents"]:
            print(f"  · 故障事件 {inc.get('fault_id', '')}: "
                  f"{'已恢复' if inc.get('recovered') else '未恢复'}"
                  + (f" (恢复 {inc['recovery_time_s']:.2f}s 仿真秒)"
                     if inc.get("recovered") else ""))
        unsafe = m["unsafe"] or 0
        print(f"  终局: 任务完成={'是' if m['task_completion'] else '否'} | "
              f"不安全提交 {unsafe}/{m['commits']} | 原子性违规 "
              f"{m['atomicity_violations']}/{m['atomic_fault_attempts']} | "
              f"恢复 {m['recovery_success_rate']} | 账实分歧 "
              f"{m['divergence_duration_s']}s")
        # 严格判定: 完成必须建立在"账实一致"上
        final = r["timeline_checkpoints"][-1] if r["timeline_checkpoints"] else None
        consistent = bool(final and final["ledger"] == final["truth"])
        verdicts[g] = {"completed": bool(m["task_completion"]),
                       "unsafe": unsafe, "consistent": consistent}
        print(f"  裁决: {'安全闭合' if (consistent and not unsafe) else '存在错误完成/未闭合'}")
    print("\n=== 对照结论 ===")
    b0, b3 = verdicts["B0"], verdicts["B3"]
    print(f"  B0: 完成={b0['completed']} 不安全提交={b0['unsafe']} "
          f"账实一致={b0['consistent']}")
    print(f"  B3: 完成={b3['completed']} 不安全提交={b3['unsafe']} "
          f"账实一致={b3['consistent']}")
    wrong_done = b0["completed"] and not b0["consistent"]
    hung = (not b0["completed"]) and not b0["consistent"]
    if wrong_done or b0["unsafe"] > 0:
        print("  差异: 同一故障下 B0 提交出**未经物理证实的状态/错误完成**, "
              "B3 拒绝→隔离→对账后闭合 —— 证据系统的可验证价值。")
    elif hung and b3["completed"] and b3["consistent"]:
        print("  差异: B0 停在**账实分歧且未闭合** (无恢复机制), B3 重观测"
              "对账后闭合 —— 可恢复性的可验证价值。")
    else:
        print("  该场景下 B0/B3 表现接近 (换 S01/D02/D03/H01 类场景对比度更高)。")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", default=",".join(SCRIPTS.keys()))
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--seed", type=int, default=0,
                    help="基础种子 (seed = base..base+n-1)")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--demo", default="", help="故障对照演示: 场景 id "
                    "(如 D03) → B0 vs B3 并排时间线 (视频/汇报脚本)")
    args = ap.parse_args(argv)

    if args.demo:
        return demo_contrast(args.demo, args.seed)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    bad = [s for s in scenarios if s not in SCRIPTS] + \
          [g for g in groups if g not in GROUPS]
    if bad:
        ap.error(f"未知场景/组: {bad}")
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    runs: list[dict] = []
    t0 = time.time()
    for g in groups:
        for s in scenarios:
            for sd in seeds:
                r = run_one(g, s, sd)
                runs.append(r)
                mm = r["metrics"]
                print(f"[{g}|{s}|seed{sd}] unsafe={mm['unsafe']}/"
                      f"{mm['commits']} atomic_viol={mm['atomicity_violations']}"
                      f"/{mm['atomic_fault_attempts']} recovery="
                      f"{mm['recovery_success_rate']} completion="
                      f"{mm['task_completion']} divergence="
                      f"{mm['divergence_duration_s']}s", flush=True)
    agg = aggregate(runs)
    hyp = hypothesis_check(agg)
    report = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "harness": "experiments/fault_ablation.py",
            "matrix": "docs/knowledge/paper1_fault_matrix.md V0.1 (消融节)",
            "groups": {
                "B0": "直接完成回报: 无门控/无原子事务/无对账 (legacy 写入器)",
                "B1": "仅物理门控: EvidenceVerdict 五门+I4; 逐次 commit",
                "B2": "门控+事务: 生产 commit_transfer (无 PREPARED/恢复)",
                "B3": "完整: PREPARED→VERIFIED→原子提交→重启对账/reconcile",
                "B4": "工业强基线: 设备字 ACK+op_id 幂等+journal 前置日志+"
                      "重启对账(超时→failed+回滚)+失败腿重试≤2; "
                      "不读传感器/真值, 无证据门控",
            },
            "scenario_meta": {k: {"category": v[0], "fault": v[1]}
                              for k, v in SCENARIO_META.items()},
            "seeds": seeds, "scenarios": scenarios,
            "n_runs": len(runs), "wall_s": round(time.time() - t0, 1),
            "modeling_notes": MODELING_NOTES,
            "metrics_def": {
                "unsafe_commit_rate": "不安全提交/总提交 (evaluator 真值判定)",
                "false_rejection_rate": "真值有效却被拒/真值有效机会",
                "atomicity_violation_rate":
                    "I3/I5 违规数/注入故障的提交尝试数",
                "recovery_success_rate": "恢复成功/崩溃·对账事件数",
                "recovery_time_s": "重启(或核对)→闭合的仿真秒 (均值)",
                "task_completion": "终态账实一致且任务链完成",
                "divergence_duration_s":
                    "故障性账实分歧时长: 不安全提交/崩溃未恢复/拒绝后离位"
                    "造成的账本≠物理归属, 到闭合或终局 (截尾); 不含物理完成"
                    "→合法提交的管线滞后窗 (见 commit_lag_divergence_s)",
                "commit_lag_divergence_s":
                    "提交管线滞后窗 (物理完成→合法提交, 含 0.5s 稳定采样)",
            },
        },
        "aggregate": agg,
        "hypothesis_check": hyp,
        "runs": runs,
    }
    out = args.out or str(
        REPORTS / f"fault_ablation_{time.strftime('%Y%m%d-%H%M%S')}.json")
    REPORTS.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print(f"\n=== 按组聚合 ({len(runs)} runs, {time.time() - t0:.1f}s) ===")
    hdr = f"{'group':<4} {'unsafe':>7} {'falserej':>8} {'atomic':>7} " \
          f"{'recover':>7} {'recov_s':>7} {'compl':>6} {'diverg_s':>8}"
    print(hdr)
    for g in groups:
        a = agg["by_group"][g]
        print(f"{g:<4} {str(a['unsafe_commit_rate']):>7} "
              f"{str(a['false_rejection_rate']):>8} "
              f"{str(a['atomicity_violation_rate']):>7} "
              f"{str(a['recovery_success_rate']):>7} "
              f"{str(a['recovery_time_s']):>7} "
              f"{str(a['task_completion']):>6} "
              f"{str(a['divergence_duration_s']):>8}")
    print("\n=== 假设检验 ===")
    for k, v in hyp.items():
        print(f"[{'PASS' if v['pass'] else 'FAIL'}] {k}: {v['detail']}")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
