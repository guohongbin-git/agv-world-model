"""T01-T03 S00 单货位台架物理闭环 (第三层, 白模)。

对应验收:
- T01 插叉配合: 插入全程无 >0.5mm 穿透; 接触真实发生
- T02 全行程升降: 稳定后误差 ≤1mm, 无漂移
- T03 第三层取放: 抬离5mm 带载退出 放置 承托≥90% 空叉退出托盘留在目标
"""
from __future__ import annotations

import pytest

from sim.controllers import CellController, Ramp
from sim.forklift_cell import ForkliftCell

X_START = -0.560     # 叉尖 -0.44, 完全在支撑板(-0.20)、暂存台(-0.40..-0.28)与其立柱(-0.398)之外
X_TIP_AT_EDGE = -0.160
X_INSERTED = -0.095  # 轴心在托盘中心后 95mm


@pytest.fixture(scope="module")
def result() -> dict:
    cell = ForkliftCell(timestep_s=0.0005)  # 文档 0.5ms 复验步长
    ctrl = CellController(cell)
    m, d = cell.model, cell.data
    # 预沉降 0.5s: 托盘稳定落在支撑面; 控制目标先置于初始位, 防止底盘被默认ctrl拉走
    d.ctrl[cell.a_x] = X_START - cell.q0_x
    d.ctrl[cell.a_lift] = 0.0
    for _ in range(500):
        mujoco_step(cell)
    A_X, A_L = cell.a_x, cell.a_lift
    k = cell.k
    X_SAFE = X_INSERTED - k["retreat_loaded"]        # 带载退出后的安全轴 (托盘中心 -0.215)
    X_STAGE = -0.340 - k["axle_behind_center"]        # 托盘中心对准暂存台 (-0.340)
    ramps = [
        # --- 从第三层取出 ---
        (A_L, Ramp(0.0, k["insert_z"] - k["transport_z"], 0.010)),            # 1 远端升到 410
        (A_X, Ramp(X_START, X_TIP_AT_EDGE, 0.005)),                            # 2 接近前沿
        (A_X, Ramp(X_TIP_AT_EDGE, X_INSERTED, 0.005)),                         # 3 插叉 65mm
        (A_L, Ramp(k["insert_z"] - k["transport_z"],
                   k["lift_z"] - k["transport_z"], 0.002)),                    # 4 抬离→417 (离面5mm)
        (A_X, Ramp(X_INSERTED, X_SAFE, 0.005)),                                 # 5 带载退出 120mm
        (A_L, Ramp(k["lift_z"] - k["transport_z"], 0.0, 0.010)),               # 6 降至运输高度(叉顶65)
        (A_X, Ramp(X_SAFE, X_STAGE, 0.005)),                                    # 7 低位转运至暂存台
        (A_L, Ramp(0.0, k["insert_z"] - k["transport_z"], 0.010)),             # 8 举升到插叉位
        (A_X, Ramp(X_STAGE, X_INSERTED, 0.005)),                                # 9 带载回插
        (A_L, Ramp(k["insert_z"] - k["transport_z"], 0.0, 0.010)),             # 10 放托: 叉降至410, 托盘落暂存台
        (A_X, Ramp(X_INSERTED, X_INSERTED - k["withdraw_empty"], 0.005)),      # 11 空叉退出 105mm
        (A_L, Ramp(k["insert_z"] - k["transport_z"], 0.0, 0.010)),             # 12 降至运输
    ]
    audit = ctrl.run_sequence(ramps, settle_s=0.5)
    audit["final"] = {
        "axle_x": cell.axle_x,
        "fork_top_z": cell.fork_top_z,
        "pallet_bottom_z": cell.pallet_bottom_z,
        "pallet_x": float(cell.pallet_pose[0][0]),
        "support_ratio": cell.support_load_ratio(),
        "pallet_fork_contact": cell.pallet_fork_contact(),
    }
    return audit


def mujoco_step(cell: ForkliftCell) -> None:
    import mujoco
    mujoco.mj_step(cell.model, cell.data)


def test_no_excess_penetration(result: dict):
    assert result["violations"] == [], result["violations"]
    assert result["max_pen"] < 5e-4, f"最大穿透 {result['max_pen']*1000:.3f}mm"


def test_fork_contact_during_insert(result: dict):
    # 插叉抬离段: 托盘曾被叉举起 → 底面高于支撑面
    seg = result["segments"]
    lift_states = [s for s in seg if s["fork_top_z"] > 0.412]
    assert lift_states, "从未达到抬离高度"
    assert any(s["pallet_bottom_z"] > 0.401 for s in lift_states), "托盘未被抬离"


def test_lift_accuracy_within_1mm(result: dict):
    # 全程最高点段: 叉顶 417±1mm (settle 后)
    at_top = [s for s in result["segments"] if abs(s["fork_top_z"] - 0.417) < 0.002]
    assert at_top, "未到达 417mm"
    assert abs(at_top[0]["fork_top_z"] - 0.417) <= 0.001


def test_pallet_carried_then_placed(result: dict):
    f = result["final"]
    # 放置判据: 暂存台面 45mm (平板顶升台面), 托盘底误差 ≤0.5mm
    assert abs(f["pallet_bottom_z"] - 0.045) <= 0.0005, f"托盘底面 {f['pallet_bottom_z']*1000:.2f}mm"
    # 支撑承托 ≥90%
    assert f["support_ratio"] >= 0.90, f"支撑承托 {f['support_ratio']*100:.0f}%"
    # 空叉已退出: 无叉-托接触
    assert not f["pallet_fork_contact"], "空叉仍接触托盘"
    # 托盘留在暂存台: |x + 0.300| ≤2mm
    assert abs(f["pallet_x"] + 0.340) <= 0.002, f"托盘位置 {f['pallet_x']*1000:.2f}mm"


def test_step_budget(result: dict):
    """用时在理论下限 + 每段稳定余量内 (不设绝对时间目标)。"""
    # 实际段: 6段升降@10mm/s(345+345+352+345=1387mm) + 2段啮合/放托@2mm/s + 平移
    lift_fast = (0.345 + 0.345 + 0.352 + 0.345) / 0.010
    lift_slow = (0.007 + 0.007) / 0.002
    moves = (0.180 + 0.055 + 0.065 + 0.245 + 0.180 + 0.055 + 0.065 + 0.105) / 0.005
    settle = 13 * 0.5
    t_theory = lift_fast + lift_slow + moves + settle
    assert result["segments"][-1]["t_end"] < t_theory * 1.3, \
        f"用时 {result['segments'][-1]['t_end']:.0f}s 远超理论 {t_theory:.0f}s"
