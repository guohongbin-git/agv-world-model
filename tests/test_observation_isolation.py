#!/usr/bin/env python3
"""P0-3 观测隔离验收: 观测/真值分离 (GPT 评审 B2 波)。

本文件是 **evaluator** —— 边界外唯一可以读真值 (mocap_pos) 的地方,
专门用来证明边界内「有噪声但无泄漏」:

(a) 静态扫描: 决策侧代码 (observation/agent/memory 全文 + server 决策
    函数 _ramp/_run_steps/run_mission/cmd) 零真值属性引用, 且正向接线
    ObservationChannel;
(b) 噪声标定: front_obstacle / cone_estimate 相对真值的偏差 ~ 0 均值、
    ≤3σ 方差 (σ=3mm 传感器噪声; 无泄漏 ⇒ 方差>0 但有界);
(c) clear_confirmed: 连续 N 次扫描无障碍才 True, 障碍重现打断连胜;
(d) 复合回归: 注入 → _ramp 避障 blocked → 移除 → 连胜恢复 (server 级,
    复刻 R2 让位段几何: 降叉 20mm → 前移穿台下净空)。
    全任务级 (账本/锁/记忆) 复合见 tests/test_r2_compound.py。
"""
from __future__ import annotations

import ast
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim.forklift_cell import ForkliftCell
from twin.observation import ObservationChannel

ROOT = Path(__file__).resolve().parents[1]
SIGMA = 0.003                     # twin/perception.Lidar 默认距离噪声
# 真值特征 (AST 属性名口径, 免疫 docstring/注释里的文字提及):
# mocap_pos/mocap_quat/cone_mocapid = mocap 真值读取; cones = 注入真值账本。
FORBIDDEN_ATTRS = frozenset(
    {"mocap_pos", "mocap_quat", "cone_mocapid", "cones"})


# ---------------------------------------------------------------------------
# (a) 决策侧零真值引用 (静态扫描)
# ---------------------------------------------------------------------------
def _attrs(node) -> set:
    """子树内被访问的属性名 (Attribute.attr) —— 只看代码, 不看文字。"""
    return {n.attr for n in ast.walk(node)
            if isinstance(n, ast.Attribute)}


def test_decision_modules_no_truth_reference():
    """观测通道 + agent + 记忆全文不得出现任何真值读取。"""
    for rel in ("twin/observation.py", "twin/agent.py", "twin/memory.py"):
        tree = ast.parse((ROOT / rel).read_text())
        bad = _attrs(tree) & FORBIDDEN_ATTRS
        assert not bad, f"{rel} 引用真值属性: {sorted(bad)}"


def test_server_decision_funcs_no_truth_reference():
    """server 决策函数 (导航/受阻/恢复/指令分发) 不得读真值, 且必须
    正向接线观测通道 —— 防止静默退回上帝视角。"""
    src = (ROOT / "twin/server.py").read_text()
    names = {"_ramp", "_run_steps", "run_mission", "cmd"}
    funcs, segs = {}, {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            funcs.setdefault(node.name, node)
            seg = ast.get_source_segment(src, node)
            assert seg is not None
            segs.setdefault(node.name, seg)
    assert set(funcs) == names, f"缺失决策函数: {names - set(funcs)}"
    for name, node in funcs.items():
        bad = _attrs(node) & FORBIDDEN_ATTRS
        assert not bad, f"决策函数 {name} 引用真值属性: {sorted(bad)}"
    # 正向接线 (mocap 真值只允许出现在 place_cone/remove_cones 环境操作)
    assert "front_obstacle" in segs["_ramp"]
    assert "cone_estimate" in segs["_run_steps"]
    assert "clear_confirmed" in segs["_run_steps"]


# ---------------------------------------------------------------------------
# 场景: F01@-0.56 前方 -0.45 / 后方 -1.30 (F02 车尾之外, 免其底盘
# 遮挡弧采样) 各一锥 —— evaluator 侧布置
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cell():
    return ForkliftCell(levels=[0.100, 0.250, 0.400],
                        cones=[(-0.45, 0.0), (-1.30, 0.0)],
                        vehicles=["F01", "F02"])


def _cone_x(cell_, i: int = 0) -> float:
    """evaluator 读真值 (边界外)。"""
    return float(cell_.data.mocap_pos[cell_.cone_mocapid[i]][0])


# ---------------------------------------------------------------------------
# (b) 噪声标定: 估计 ~ N(0, ≤σ) —— 有噪声 (方差>0) 但无泄漏 (均值≈0)
# ---------------------------------------------------------------------------
def test_front_obstacle_noise_calibration(cell):
    ch = ObservationChannel(cell)
    ax = cell.vehicle("F01").axle_x
    truth = _cone_x(cell) - ax - 0.025            # 前锥表面距离 (真值)
    ests = [ch.front_obstacle("F01").near_dist_est for _ in range(150)]
    e = np.array(ests) - truth
    assert abs(e.mean()) < 0.002, f"均值偏移 {e.mean():.5f} (泄漏特征)"
    assert 0.0002 < e.std() < 3 * SIGMA, f"std={e.std():.5f} 超界"
    assert len(set(np.round(ests, 6))) > 100, "估计无方差 —— 真值复读"


def test_front_obstacle_backward_direction(cell):
    """倒车方向 (watch=-1): 扇形旋 180°, 后方锥桶同样可测。"""
    ch = ObservationChannel(cell)
    ax = cell.vehicle("F01").axle_x
    truth = ax - _cone_x(cell, 1) - 0.025          # 后锥表面距离
    ests = [ch.front_obstacle("F01", direction=-1.0).near_dist_est
            for _ in range(100)]
    e = np.array(ests) - truth
    assert abs(e.mean()) < 0.002, f"倒车估计均值偏移 {e.mean():.5f}"
    assert 0.0002 < e.std() < 3 * SIGMA
    # 前向看见的是前锥 (方向语义: 后锥不在前向走廊)
    fwd = ch.front_obstacle("F01")
    assert fwd.near_dist_est < 0.5


def test_cone_estimate_noise_calibration(cell):
    ch = ObservationChannel(cell)
    truth = _cone_x(cell)
    xs, ys = [], []
    for _ in range(120):
        ch.front_obstacle("F01")                   # 扫描 (噪声独立)
        est, ts = ch.cone_estimate("F01")
        assert est is not None and ts >= 0.0
        xs.append(est[0])
        ys.append(est[1])
    e = np.array(xs) - truth
    assert abs(e.mean()) < 0.002, f"中心估计偏移 {e.mean():.5f}"
    assert 0.0002 < e.std() < 3 * SIGMA, f"std={e.std():.5f}"
    assert abs(np.mean(ys)) < 0.01, "横向估计应 ≈0 (锥在巷道中线)"


def test_cone_estimate_none_when_clear(cell):
    """无扫描/无命中 → (None, ts)。"""
    ch = ObservationChannel(cell)
    est, ts = ch.cone_estimate("F01")
    assert est is None and ts >= 0.0


# ---------------------------------------------------------------------------
# (c) 恢复确认: 连续 N 次扫描无障碍
# ---------------------------------------------------------------------------
def test_clear_confirmed_streak(cell):
    ch = ObservationChannel(cell, clear_n=3)
    d = cell.data

    def _env_remove():
        for mid in cell.cone_mocapid:              # evaluator 侧环境操作
            if mid >= 0:
                d.mocap_pos[mid] = [0.0, 5.0, 0.045]
        mujoco.mj_forward(cell.model, d)

    # 障碍在场: 反复确认均为 False
    for _ in range(5):
        assert not ch.clear_confirmed("F01")

    _env_remove()
    assert not ch.clear_confirmed("F01")           # 第 1 次
    assert not ch.clear_confirmed("F01")           # 第 2 次
    assert ch.clear_confirmed("F01")               # 第 3 次 → True
    hist = ch.clear_history("F01")
    assert len(hist) == 3 and all(c for _, _, c in hist)
    assert hist[0][1] < hist[1][1] < hist[2][1]    # 扫描序号严格递增
    assert hist[0][0] <= hist[2][0]                # 时间戳非降 (带时间戳)

    # 障碍重现 → 连胜打断; 再移除须重新数满
    d.mocap_pos[cell.cone_mocapid[0]] = [-0.45, 0.0, 0.045]
    mujoco.mj_forward(cell.model, d)
    assert not ch.clear_confirmed("F01")
    _env_remove()
    assert not ch.clear_confirmed("F01")
    assert not ch.clear_confirmed("F01")
    assert ch.clear_confirmed("F01")

    # reset_clear: 旧连胜不得复用
    ch.reset_clear("F01")
    assert ch.clear_history("F01") == []
    assert not ch.clear_confirmed("F01")


# ---------------------------------------------------------------------------
# (d) 复合回归: 注入 → 避障 blocked → 移除 → 连胜恢复 (server 级)
# ---------------------------------------------------------------------------
def test_compound_inject_blocked_remove_recover():
    from twin.server import WorldService
    svc = WorldService()
    svc.gate.speed = 64.0                          # 解除墙钟配速
    ad = svc.adapters["F01"]
    v = ad.view
    tz = svc.cell.k["transport_z"]
    # 复刻 R2 让位段: 降叉至叉顶 20mm (穿台下净空的前置段)
    assert svc._ramp(ad, v.a_lift, v.qadr_lift, 0.020 - tz, 0.010)
    assert abs(v.fork_top_z - 0.020) < 2e-3

    cone_x = -0.20            # R2 注入带 (axle>-0.35 时 +0.15) 的下界
    svc.place_cone(cone_x)
    assert svc.blocked_reason is None

    # 前移让位 (真实 _ramp + 物理步进) → 观测通道检测 → blocked
    ok = svc._ramp(ad, v.a_x, v.qadr_x, 0.020 - v.q0_x, 0.020, watch=+1.0)
    assert not ok, "避障未触发"
    assert svc.blocked_reason == "cone"
    assert svc._blocked_dir == 1.0
    # 停车点与锥面保持安全距离 (0.12 阈值附近, 未压上)
    stop_gap = cone_x - v.axle_x - 0.025
    assert 0.0 < stop_gap < 0.20, f"停车间隙异常: {stop_gap:.3f}"

    # 受阻时刻记忆签名用的是观测估计 (与真值差在噪声量级内)
    est_xy, _ = svc.obs.cone_estimate("F01")
    assert est_xy is not None
    assert abs(est_xy[0] - cone_x) < 0.01, "锥桶估计偏离真值超噪声量级"

    # 移除 → 恢复确认需连续 N 次扫描 (不能一次放行)
    svc.remove_cones()
    assert svc.blocked_reason == "cone"            # 移锥不直接解除
    assert not svc.obs.clear_confirmed("F01")
    assert not svc.obs.clear_confirmed("F01")
    assert svc.obs.clear_confirmed("F01")

    # 恢复后继续让位段至待命位 (axle=+0.020); blocked_reason 的清除
    # 发生在 _run_steps 恢复循环 (全任务级见 test_r2_compound)
    ok = svc._ramp(ad, v.a_x, v.qadr_x, 0.020 - v.q0_x, 0.020, watch=+1.0)
    assert ok
    assert abs(v.axle_x - 0.020) < 2e-3
