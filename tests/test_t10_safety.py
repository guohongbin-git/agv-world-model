"""T10 模拟急停与失电 —— 低位与第三层分别注入三种故障。

设计判据:
|T10 模拟急停与失电|低位和第三层分别注入停止、驱动失电与制动失效|
     |模拟响应与停止距离有轨迹; 正常制动保持, 制动失效被检测并记为失败;
       实体试验留待 H0/H1|

关键断言 (不是"没抛异常"):
1. **三种模式物理上必须可区分** —— 若急停/失电/制动失效测出来一样,
   判据就无从谈起: 急停主动制动距离最短, 失电靠阻尼滑行更远, 制动失效
   最远。任两模式距离相同即失败。
2. **正常制动保持** —— 急停后载荷不滑不下, 判为安全。
3. **制动失效必须记为失败** —— 无论位移多小 (设计 S07 无虚假成功),
   且资源被冻结。
4. **有轨迹** —— 每个用例都记录 (t, x, z, v) 采样点 (设计 R11)。
5. **高度影响** —— 第三层载荷在高位, 失电时下沉风险大于低位。
"""
from __future__ import annotations

import mujoco
import pytest

from sim.forklift_cell import ForkliftCell
from sim.safety import StopMode, VehicleSafety

LOW_Z = 0.135


def build(level: str) -> ForkliftCell:
    """构造载货车辆: third = 第三层 (叉顶 0.417), low = 低位 (叉顶 0.135)。"""
    c = ForkliftCell(seed=1)
    for _ in range(2000):
        mujoco.mj_step(c.model, c.data)
    # 插叉位取 ForkliftCell.roundtrip_plan 的 x_ins (叉尖越过托盘近边 65mm)。
    # 此前写死 -0.0596 (叉根进到托盘中心) 使叉顶死顶货架横梁:
    # ctrl 打满 +5N 而 qpos 锁在 319mm, 叉顶卡 384mm 纹丝不动。
    c.data.ctrl[c.a_x] = c.roundtrip_plan()[2][1]().target - c.q0_x
    for _ in range(40000):
        mujoco.mj_step(c.model, c.data)
    # a_lift 的 ctrl 是**关节坐标 qpos**, 不是叉顶世界高度:
    #   fork_top_z = qpos + transport_z (实测稳态 sag 0.28mm)
    # 载货必须升到 lift_z (叉顶 417): 叉顶要到通道顶 412 才抬得动托盘。
    # 插叉高度 410 只把叉送进叉孔, 托盘仍坐在横梁上 —— 承托判据恒 0,
    # 失电也无载荷可沉 (T06/T10 同根因)。
    c.data.ctrl[c.a_lift] = c.k["lift_z"] - c.k["transport_z"]
    for _ in range(60000):
        mujoco.mj_step(c.model, c.data)
    if level == "low":
        # 载货不能在货架投影内下降: 横梁 (x±200, z395..400) 会接住托盘
        # (实测低位托盘坐梁、叉空滑脱, "载货"名存实亡)。先退到横梁外
        # (托盘 x∈[-380,-300] 全在梁外), 再降叉才是真·低位载货。
        c.data.ctrl[c.a_x] = -0.340 - c.q0_x
        for _ in range(40000):
            mujoco.mj_step(c.model, c.data)
        c.data.ctrl[c.a_lift] = LOW_Z - c.k["transport_z"]
        for _ in range(40000):
            mujoco.mj_step(c.model, c.data)
        # 载货在低位: 托盘随叉下降, 叉顶应低于第三层
        assert c.fork_top_z < 0.30, f"低位叉顶应 <0.30, 实测 {c.fork_top_z:.3f}"
        # 托盘必须真在叉上, 不许悄悄坐回横梁
        assert c.support_load_ratio("fork") > 0.90, \
            f"低位托盘未载于叉: 叉承托 {c.support_load_ratio('fork'):.2f}"
    else:
        assert c.fork_top_z > 0.40, f"第三层叉顶应 >0.40, 实测 {c.fork_top_z:.3f}"
    return c


@pytest.mark.parametrize("level", ["low", "third"])
def test_t10_three_modes_are_distinguishable(level):
    """三种模式必须给出不同的停止距离, 否则判据无意义。"""
    recs = {}
    for mode in (StopMode.ESTOP, StopMode.DRIVE_POWER_LOSS, StopMode.BRAKE_FAIL):
        cell = build(level)
        vs = VehicleSafety(cell)
        recs[mode] = vs.apply(mode, level, duration_s=1.0, initial_v_mps=0.2)
    d_estop = recs[StopMode.ESTOP].stop_distance_m
    d_loss = recs[StopMode.DRIVE_POWER_LOSS].stop_distance_m
    d_fail = recs[StopMode.BRAKE_FAIL].stop_distance_m
    assert d_estop < d_loss < d_fail, (
        f"{level}: 停止距离未分级 estop={d_estop*1000:.2f} "
        f"power_loss={d_loss*1000:.2f} brake_fail={d_fail*1000:.2f} mm")


@pytest.mark.parametrize("level", ["low", "third"])
def test_t10_estop_holds_and_is_safe(level):
    """正常急停: 主动制动保持, 载荷无滑移, 判为安全。"""
    cell = build(level)
    vs = VehicleSafety(cell)
    rec = vs.apply(StopMode.ESTOP, level, duration_s=1.0, initial_v_mps=0.2)
    assert rec.held, f"{level} 急停后仍有残余速度"
    assert rec.safe and not rec.failure, f"{level} 急停应判安全: {rec.reason}"
    assert rec.stop_distance_m < 2e-3, (
        f"{level} 急停距离 {rec.stop_distance_m*1000:.2f}mm 过大")
    assert rec.slip_m < 1e-3, f"{level} 载荷滑移 {rec.slip_m*1000:.2f}mm"
    assert not vs.resources_frozen, "急停不应冻结资源"


@pytest.mark.parametrize("level", ["low", "third"])
def test_t10_brake_failure_always_counted_as_failure(level):
    """制动失效: **无论位移多小都记为失败**, 并冻结资源 (设计 S07)。"""
    cell = build(level)
    vs = VehicleSafety(cell)
    rec = vs.apply(StopMode.BRAKE_FAIL, level, duration_s=1.0, initial_v_mps=0.2)
    assert rec.failure is True, "制动失效必须记为失败"
    assert rec.safe is False, "制动失效不得判为安全 (无虚假成功)"
    assert vs.resources_frozen, "制动失效必须冻结业务资源"
    assert "制动失效" in rec.reason
    # 停止距离应大于同高度下正常急停
    cell2 = build(level)
    ok = VehicleSafety(cell2).apply(StopMode.ESTOP, level, duration_s=1.0,
                                    initial_v_mps=0.2)
    assert rec.stop_distance_m > ok.stop_distance_m, (
        f"{level} 制动失效距离未大于急停距离")


def test_t10_third_level_power_loss_risks_load_drop():
    """第三层失电: 载荷在高位, 下沉风险必须被检出并记为失败。"""
    cell = build("third")
    vs = VehicleSafety(cell)
    rec = vs.apply(StopMode.DRIVE_POWER_LOSS, "third", duration_s=1.0,
                   initial_v_mps=0.2)
    assert rec.drop_m > 2e-3, (
        f"第三层失电载荷下沉 {rec.drop_m*1000:.2f}mm 未被检出")
    assert rec.failure is True, "第三层失电造成载荷下沉必须记为失败"
    assert vs.resources_frozen


def test_t10_trajectory_recorded_for_every_case():
    """每个用例都必须有停止轨迹 (设计 R11)。"""
    for level in ("low", "third"):
        for mode in StopMode:
            cell = build(level)
            rec = VehicleSafety(cell).apply(mode, level, duration_s=1.0,
                                           initial_v_mps=0.2)
            assert len(rec.traj) >= 10, (
                f"{level}/{mode.value} 轨迹点仅 {len(rec.traj)} 个")
            assert all(len(p) == 4 for p in rec.traj), "轨迹点应为 (t,x,z,v)"
            d = rec.as_dict()
            for k in ("mode", "level", "stop_distance_m", "drop_m", "slip_m",
                      "held", "safe", "failure"):
                assert k in d, f"报告缺字段 {k}"


def test_t10_thaw_restores_drive_and_brakes():
    """试验复位必须还原传动与阻尼 (不靠写死常量)。"""
    cell = build("third")
    vs = VehicleSafety(cell)
    gear_before = cell.model.actuator_gear.copy()
    damp_before = cell.model.dof_damping.copy()
    vs.apply(StopMode.BRAKE_FAIL, "third", duration_s=0.5, initial_v_mps=0.2)
    assert vs.resources_frozen
    vs.thaw()
    assert not vs.resources_frozen
    assert (cell.model.actuator_gear == gear_before).all(), "传动未还原"
    assert (cell.model.dof_damping == damp_before).all(), "阻尼未还原"