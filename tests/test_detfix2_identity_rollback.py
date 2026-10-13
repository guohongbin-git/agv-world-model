"""detfix 二轮定向用例: 托盘身份重绑 + 失败腿台账回滚 (live 5f82cc59 取证)。

背景 (30 分钟压测 live run MX-20261013-004740-5f82cc59):
* t=312 ORD-000006 (R23-S1) / t=557 ORD-000013 (R31-S2) chain_failed
  「无托盘可取」—— 补货开单即计划占位 (demand._place), 同 SKU 出库单
  先于 putaway 落位发射, pick 门控按名查账必空;
* 补货回填货位的是交付台旧托 (名字带原货位, 如 pallet_R23_S4_L1 坐
  slot:R32-S1), 后续按 slot 命名推断的出库 pick 永久查无此托;
* 物理失败腿的盲记账把幻影位置写进台账 → stand/dock 被死账锁死。

修复语义 (物理即真值):
* pick 门控: 货位空但有在途补货链 → 等落位; 有异名占位者 → 重绑链主角
  与作业对象 (mission 按 task["pallet"] 闭环作业);
* _finish_op: 物理校验失败即回滚本腿台账搬动 (失败不认账)。
"""
from __future__ import annotations

import math

import pytest

from sim.mission import Job, MissionRunner
from sim.pallet_support import ledger_from_cell
from sim.warehouse_cell import WarehouseCell, pallet_name_for
from twin.matrix_server import Chain, MatrixService, Spec


def _free_base(cell, loc: dict) -> str:
    """账本无托的货位基名 (level 无关, 如 'R31-S2')。"""
    occ = {w[len("slot:"):] for w in loc.values() if w.startswith("slot:")}
    bases = sorted({s.rsplit("-L", 1)[0] for s in cell.layout.slot_ids()})
    return next(b for b in bases if b not in occ)


@pytest.fixture(scope="module")
def rig():
    """未启动线程的 MatrixService (物理/账本/门控真实构件)。"""
    return MatrixService(seed=7, multiplier=1.0)


# ---------------------------------------------------------------- 门控
def test_gate_waits_while_replenish_inbound(rig):
    """幻影货位: 在途补货链未死 → pick 等落位不判死; 链死了才 fail。"""
    svc = rig
    mr = svc.mr
    loc = svc._ledger_loc()
    base = _free_base(svc.cell, loc)
    c_rep = Chain(task_id="REP-T", kind="REPLENISH", desc="t")
    c_rep.specs = [
        Spec(uid="RF", task_id="REP-T", kind="fetch", vid="A01",
             task={"kind": "fetch", "dock": "", "stand": "H02"}),
        Spec(uid="RP", task_id="REP-T", kind="putaway", vid="F01",
             task={"kind": "putaway", "stand": "H02", "slot": base,
                   "level": 2}, deps=["RF"]),
    ]
    c_out = Chain(task_id="ORD-T", kind="OUTBOUND", desc="t",
                  pallet=pallet_name_for(f"{base}-L2"))
    s_pick = Spec(uid="OP", task_id="ORD-T", kind="pick", vid="F02",
                  task={"kind": "pick", "slot": base, "level": 2,
                        "stand": "H03"})
    c_out.specs = [s_pick]
    svc.chains.extend([c_rep, c_out])
    try:
        assert not mr.jobs[s_pick.vid]
        # 补货链 pending → 货位计划占位但物理未落 → 等待 (不判死)
        assert svc._gate(s_pick, c_out, svc._ledger_loc()) is False
        assert c_out.state == "pending"
        assert not any("无托盘可取" in f["detail"] for f in svc.feeds)
        # 补货链死亡 → 无在途供给 → 才判死 (原语义保留)
        c_rep.state = "failed"
        assert svc._gate(s_pick, c_out, svc._ledger_loc()) is False
        assert c_out.state == "failed"
        assert any("无托盘可取" in f["detail"] for f in svc.feeds)
    finally:
        svc.chains.remove(c_rep)
        svc.chains.remove(c_out)


def test_gate_rebinds_foreign_occupant(rig):
    """异名占位者: 补货回填的旧托坐镇货位 → 重绑链主角+作业对象。"""
    svc = rig
    mr = svc.mr
    loc = svc._ledger_loc()
    base = _free_base(svc.cell, loc)
    occupant = next(n for n, w in sorted(loc.items())
                    if w.startswith("slot:"))
    saved = loc[occupant]
    mr.ledger.loc[occupant] = f"slot:{base}"
    c_out = Chain(task_id="ORD-T2", kind="OUTBOUND", desc="t",
                  pallet=pallet_name_for(f"{base}-L2"))
    s_pick = Spec(uid="OP2", task_id="ORD-T2", kind="pick", vid="F02",
                  task={"kind": "pick", "slot": base, "level": 2,
                        "stand": "H03"})
    c_out.specs = [s_pick]
    svc.chains.append(c_out)
    try:
        assert svc._gate(s_pick, c_out, svc._ledger_loc()) is True
        assert c_out.pallet == occupant
        assert s_pick.task["pallet"] == occupant
        assert c_out.reserved.get("stand") == "H03"
    finally:
        svc.chains.remove(c_out)
        mr.ledger.loc[occupant] = saved


# ---------------------------------------------------------------- 台账
def test_finish_op_rolls_back_ledger_on_physical_failure(rig):
    """物理失败腿不认账: 校验失败 → 本腿台账搬动回滚 (防幻影死账)。"""
    svc = rig
    mr = svc.mr
    pn = "pallet_R11_S0_L2"                     # 固定交接链测试位
    before = mr.ledger.loc[pn]
    assert before.startswith("slot:")
    mr.jobs["F01"] = Job(vid="F01", task={"kind": "pick",
                                          "slot": "R11-S0", "level": 2,
                                          "stand": "H02"})
    n_err = len(mr.op_errors)
    try:
        mr._finish_op("F01", final=True)        # 托盘实际仍在货位 → 校验必败
        assert len(mr.op_errors) == n_err + 1
        assert "物理校验失败" in mr.op_errors[-1]
        assert mr.ledger.loc[pn] == before      # 回滚, 账本不留幻影
    finally:
        del mr.op_errors[n_err:]
        mr.jobs["F01"] = None


# ---------------------------------------------------------------- 物理 E2E
def test_pick_explicit_pallet_foreign_occupant_e2e():
    """异名占位全链: putaway 回填 → 后续出库按 task["pallet"] 取托成功。

    旧代码按 slot+level 命名推断 pallet_R11_S4_L2 (不存在):
    ledger.move 断言崩 / 叉车对空位插叉。重绑后物理闭环零 op_errors。"""
    cell = WarehouseCell(timestep_s=0.001, seed=7,
                         seed_slots=["R11-S0-L2"],
                         preload_stands=("H01",))
    cell.warmup(0.5)
    assert ledger_from_cell(cell).loc["pallet_R11_S0_L2"] == "slot:R11-S0"
    mr = MissionRunner(cell)
    res = mr.run([
        {"kind": "pick", "slot": "R11-S0", "level": 2, "stand": "H02"},
        {"kind": "putaway", "stand": "H02", "slot": "R11-S4", "level": 2,
         "after": [0]},
    ], max_s=2400)
    assert res["op_errors"] == []
    assert res["ledger"]["pallet_R11_S0_L2"] == "slot:R11-S4"
    ex, ey, _, ez = cell.layout.slot_pose("R11-S4-L2")
    pos = cell.pallet_pose("pallet_R11_S0_L2")[0]
    assert math.hypot(pos[0] - ex, pos[1] - ey) < 0.02
    assert abs(pos[2] - ez) < 0.02              # 异名占位物理成立
    res2 = mr.run([
        {"kind": "pick", "slot": "R11-S4", "level": 2, "stand": "H03",
         "pallet": "pallet_R11_S0_L2"},
    ], max_s=2400)
    assert res2["op_errors"] == []
    assert res2["ledger"]["pallet_R11_S0_L2"] == "stand:H03"


# ---------------------------------------------------------------- dock 越台
def _edge(mr, a, b):
    for n, e in mr.graph.adj[a]:
        if n == b:
            return e
    raise AssertionError(f"无邻接边 {a}->{b}")


def test_forklift_transit_lifts_over_committed_dock_pallet():
    """detfix2: 交付台坐在行通道线上 —— 叉车 transit 穿越「有托 dock」
    必须高位越台 (guard 原只覆盖 stand)。旧码 transit 叉底 0.102 <
    台上托盘顶板 0.115 → 铲飞已提交托盘 (live 5f82cc59 t≈484: F02 列
    通道穿越 D_G14, ORD-000002 已提交托盘被铲离台位斜楔 0.055,
    REP-000004 fetch 物理校验失败 dz=-0.040 → 整链失败)。"""
    import mujoco

    cell = WarehouseCell(timestep_s=0.001, seed=7,
                         seed_slots=["R11-S0-L2"],
                         preload_stands=("D_G14",))
    cell.warmup(0.5)
    assert ledger_from_cell(cell).loc["pallet_D_G14"] == "dock:D_G14"
    ex, ey = cell.layout.stand_pose("D_G14")[:2]
    ez = 0.095
    pos = cell.pallet_pose("pallet_D_G14")[0]
    assert math.hypot(pos[0] - ex, pos[1] - ey) < 0.02
    assert abs(pos[2] - ez) < 0.02              # 预载托盘确实坐在交付台

    mr = MissionRunner(cell)
    vid = "F02"
    view = cell.veh[vid]
    d = cell.data
    # 把 F02 摆到台位东侧两节点 (关节坐标 = 世界坐标 − park 偏置):
    # sm:2:4:3 (1.89) → sm:2:4:2 (1.73) → dock:D_G14 (1.65) → sm:2:4:1
    d.qpos[view.qadr["x"]] = 1.89 - view.off["x"]
    d.qpos[view.qadr["y"]] = -1.165 - view.off["y"]
    d.qpos[view.qadr["yaw"]] = math.pi
    mujoco.mj_forward(cell.model, d)

    job = Job(vid=vid, task={"kind": "pick", "slot": "R11-S0",
                             "level": 2, "stand": "H02"})
    job.path = ["sm:2:4:3", "sm:2:4:2", "dock:D_G14", "sm:2:4:1",
                "sm:2:4:0"]
    job.edges = [_edge(mr, "sm:2:4:3", "sm:2:4:2"),
                 _edge(mr, "sm:2:4:2", "dock:D_G14"),
                 _edge(mr, "dock:D_G14", "sm:2:4:1"),
                 _edge(mr, "sm:2:4:1", "sm:2:4:0")]
    mr.quad_of[vid] = 2                          # 面西

    def lift_legs():
        return [s for s in mr.runner.queues[vid]
                if s[0] == view.a["lift"]]

    # 边 0 (不触台, 下边触台): 起步升叉 —— 叉尖 1.89-0.12=1.77 尚在
    # 托盘东缘 1.69 之外, 原地升叉不接托 (detfix2 提前一节点语义)
    mr._submit_edge(vid, job, 0)
    assert len(lift_legs()) == 1 and job.lift_armed
    while mr.runner.busy(vid):
        mr.runner.run_until_any_idle(120.0)
    # 边 1 (进台) 与边 2 (离台, 仍触 dock 节点): 保持高叉, 不重复升
    mr._submit_edge(vid, job, 1)
    assert lift_legs() == []
    while mr.runner.busy(vid):
        mr.runner.run_until_any_idle(120.0)
    mr._submit_edge(vid, job, 2)
    assert lift_legs() == []
    while mr.runner.busy(vid):
        mr.runner.run_until_any_idle(120.0)
    # 边 3 (离开台位区): 起步回降
    mr._submit_edge(vid, job, 3)
    assert len(lift_legs()) == 1 and not job.lift_armed
    while mr.runner.busy(vid):
        mr.runner.run_until_any_idle(120.0)
    pos = cell.pallet_pose("pallet_D_G14")[0]
    assert math.hypot(pos[0] - ex, pos[1] - ey) < 0.02
    assert abs(pos[2] - ez) < 0.02               # 已提交托盘原位未动
