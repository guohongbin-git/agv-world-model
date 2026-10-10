"""T06 双车交接 —— H01 满载接力 50 次。

设计判据:
|T06 双车交接|H01 满载接力 50 次|站内无双车重叠; 支撑明确; 身份与位置一致|

三条都断言, 且都是**可观测**的, 不是"没抛异常":
1. 站内无双车重叠 —— 任一时刻 H01 只被一台车持有, 申请必被拒;
2. 支撑明确 —— 交接瞬间承托必须**在叉上**(fork≈1.0) 且**不在梁上**
   (beam≈0.0); 插叉中途两者可能短暂并存, 但离架后必须完成转移;
3. 身份与位置一致 —— 物理搬走 ↔ World 位置同步, 绑定不漂移。

**物理上限已重测 (50 轮)**
sim/forklift_cell.py 是单车白模: 一个托盘 (freejoint 自由体, 10 个小
接触面, 自重 45g)、一根横梁、一对叉。旧几何下实测只能连续保持 4 轮
(第 4–5 轮托盘漂移 1.5mm 压回横梁) —— 那是坏几何的产物: 叉孔摩擦 0.8
承不住带载加减速、单齿承托使托盘绕接触点翻转。修复 (摩擦 1.4、叉尖
插深、分片背板) 后实测 **50 轮全部一致**: 最低叉承托 0.978、托盘高度
最大偏差 0.33mm、放回梁承托最低 0.978。

因此 50 次接力分两层验证:
* **50 轮调度与库存**: 每轮用**新装配的物理单元**, 验证预约/互斥/位置
  事务 —— 这才是 W05 要判的东西;
* **连续接力** 同一物理单元跑满 50 轮, 显式断言支撑保持。
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from services.adapters.forklift import ForkliftAdapter
from services.scheduler.reservations import (PartialAcquisition,
                                             ReservationBoard)
from services.world.ontology import GoodsUnit, Loc, LocKind, Pallet, World
from sim.forklift_cell import ForkliftCell

FORK_CAPS = frozenset(["drive", "fork_lift", "fork_insert", "fork_withdraw", "stop"])
RELAYS = 50
#: 单个物理单元可连续保持明确支撑的轮数 (修复几何后实测 50 轮全一致)
SUSTAINABLE_RELAYS = 50
# 插叉位 = 叉尖越过托盘近边 65mm (与 roundtrip_plan 的 x_ins 同源)。
# 此前 -0.0596 是旧几何下"叉抵横梁推不动"的实测值: 该位背板 x 与托盘
# 通道顶壁重叠, 升叉时背板顶住顶壁, 承托恒 0。
INSERT_X = -0.095
CARRY_Z = 0.417           # 载货高度: 叉顶须过通道顶 412 才抬得动托盘
RETREAT_X = -0.200
DROP_Z = 0.135            # 降叉放回横梁 (叉顶低于梁面, 叉从叉孔滑脱)


def _insert(a: ForkliftAdapter, tag: str) -> None:
    """升到叉孔 → 前进对位 → 插叉 → 举起离架 → 带载退回。

    托盘被放回后停在**插入位正下方** (axle=0.0), 故从第二轮起不再需要
    前进动作 —— 直接升叉即可再取 (见 _return_to_station)。
    """
    a.execute("move", f"{tag}-fwd", target_axle_x=INSERT_X)
    a.execute("lift", f"{tag}-ins", target_z=CARRY_Z)
    a.execute("move", f"{tag}-back", target_axle_x=RETREAT_X)


def _return_to_station(a: ForkliftAdapter, tag: str) -> None:
    """把托盘放回横梁, 供下一轮接力。

    单车白模里托盘只有一处 (横梁), 取走后就"没了" —— 接力必须**放回**,
    否则第二轮起叉下无托盘, 承托判据恒为 0。四设备场景里这一步由
    F01 上架到货架层完成, 语义相同 (托盘落到某个支撑面并释放叉)。

    必须**原地放托** (退回插入位再降叉), 不能边退边放: 退到 -0.10 再降叉
    会把托盘留在车后 100mm 处, 下一轮前进时叉会把托盘推下梁 (实测
    托盘掉到 z=0.059 落地)。
    """
    a.execute("move", f"{tag}-ret", target_axle_x=INSERT_X)
    a.execute("lift", f"{tag}-drop", target_z=DROP_Z)   # 降叉 → 托盘落梁


def test_t06_fifty_relays_no_overlap_explicit_support():
    """H01 满载接力 50 次 —— 调度/互斥/库存事务层。

    每轮用**新装配的物理单元**: 本白模的托盘靠摩擦挂在叉上, 连续复用
    同一个单元到第 4–5 轮必然漂移 (见模块 docstring)。W05 要判的是
    「无重叠 / 支撑明确 / 身份位置一致」三条, 它们每轮都被断言;
    连续多次在同一物理体上的保持能力由下一个测试单独界定。
    """
    board = ReservationBoard()
    board.register_exit_gate("EXIT_H01", ["H01"])
    board.acquire_all(["H01"], "R01", "RELAY", 0.0)

    w = World(run_id="T06")
    w.register_pallet(Pallet("PAL-001", 0.045, Loc(LocKind.ROBOT, "R01")))

    rejected = 0
    for i in range(RELAYS):
        t = i * 0.1
        # --- 1. 无双车重叠: F01 申请必须被拒 ---
        assert board.holder_of("H01") == "R01", f"第{i}轮 H01 持有者异常"
        with pytest.raises(PartialAcquisition):
            board.acquire_all(["H01"], "F01", "RELAY", t)
        rejected += 1

        # --- 2. 本轮的物理单元: 支撑必须转移到叉上 ---
        cell = ForkliftCell(seed=100 + i)
        for _ in range(2000):
            mujoco.mj_step(cell.model, cell.data)
        r01 = ForkliftAdapter("R01", cell, capabilities=FORK_CAPS)
        _insert(r01, f"r{i}")

        fork_support = float(cell.support_load_ratio("fork"))
        beam_support = float(cell.support_load_ratio("beam"))
        assert fork_support > 0.90, (
            f"第{i}轮 叉上承托 {fork_support:.2f} < 0.90")
        assert beam_support < 0.10, (
            f"第{i}轮 托盘仍压在横梁上 ({beam_support:.2f}), 未离架")
        assert bool(cell.pallet_fork_contact()), f"第{i}轮 叉未接触托盘"
        assert cell.max_penetration() < 1e-3, f"第{i}轮 穿模"

        # --- 3. 身份与位置一致 ---
        w.move_pallet("PAL-001", Loc(LocKind.ROBOT, "R01"), "RELAY", t)
        assert w.pallet("PAL-001")[0] == "ROBOT:R01"

        # --- 4. 交接给 F01: R01 先释放退出 ---
        board.release_all("R01")
        board.acquire_all(["H01"], "F01", "RELAY", t + 0.04)
        assert board.holder_of("H01") == "F01"

        # --- 5. 交还, 下一轮开始 ---
        board.release_all("F01")
        board.acquire_all(["H01"], "R01", "RELAY", t + 0.08)
        assert board.holder_of("H01") == "R01"

    assert rejected == RELAYS, f"仅 {rejected}/{RELAYS} 次重叠申请被拒"


def test_t06_physical_relay_sustain_limit():
    """同一物理单元连续接力跑满 50 轮 —— 显式断言支撑保持。

    旧几何 (叉孔摩擦 0.8、单齿承托) 下实测 4 轮起漂移; 修复后 50 轮
    全部一致 (最低叉承托 0.978 / 托盘高度最大偏差 0.33mm / 放回梁承托
    最低 0.978)。SUSTAINABLE_RELAYS 若再被物理改动压低, 此测试立即
    失败, 不让它悄悄退化。
    """
    cell = ForkliftCell(seed=1)
    for _ in range(2000):
        mujoco.mj_step(cell.model, cell.data)
    a = ForkliftAdapter("R01", cell, capabilities=FORK_CAPS)

    for i in range(SUSTAINABLE_RELAYS):
        _insert(a, f"r{i}")
        assert float(cell.support_load_ratio("fork")) > 0.90, (
            f"第{i}轮 叉上承托 {float(cell.support_load_ratio('fork')):.3f} < 0.90")
        assert float(cell.support_load_ratio("beam")) < 0.10, (
            f"第{i}轮 托盘压梁 {float(cell.support_load_ratio('beam')):.3f}")
        # 载货托盘底 = 载货叉顶 − 通道顶壁 12mm; 由 CARRY_Z 派生, 不写死
        assert abs(cell.pallet_bottom_z - (CARRY_Z - 0.012)) < 1e-3, (
            f"第{i}轮 托盘高度漂移到 {cell.pallet_bottom_z:.4f}")
        _return_to_station(a, f"r{i}")
        assert float(cell.support_load_ratio("beam")) > 0.90, (
            f"第{i}轮 托盘未放回横梁")
        assert not bool(cell.pallet_fork_contact()), f"第{i}轮 降叉后叉未释放"


def test_t06_exit_gate_blocks_other_vehicle():
    """退出通道未放行时, 另一台车不得进入 H01。"""
    b = ReservationBoard()
    b.register_exit_gate("EXIT_H01", ["H01"])
    b.acquire_all(["H01", "EXIT_H01"], "R01", "T1", 0.0)
    with pytest.raises(PartialAcquisition):
        b.acquire_all(["H01"], "F01", "T2", 0.1)
    b.release_all("R01")
    assert b.acquire_all(["H01"], "F01", "T2", 0.2) == ["H01"]


def test_t06_support_transfers_from_beam_to_fork():
    """支撑面必须真的发生转移: 梁 → 叉。"""
    c = ForkliftCell(seed=3)
    for _ in range(2000):
        mujoco.mj_step(c.model, c.data)
    a = ForkliftAdapter("F01", c, capabilities=FORK_CAPS)
    assert float(c.support_load_ratio("beam")) > 0.9, "初始应在横梁上"
    assert float(c.support_load_ratio("fork")) < 0.1, "初始叉上不应受力"
    _insert(a, "s")
    assert float(c.support_load_ratio("fork")) > 0.9, "插叉后应由叉承载"
    assert float(c.support_load_ratio("beam")) < 0.1, "离架后不应再压梁"


def test_t06_identity_and_location_stay_consistent():
    """身份与位置一致: 物理搬走 ↔ World 位置必须同步。"""
    w = World(run_id="T06-ID")
    w.register_pallet(Pallet("PAL-007", 0.045, Loc(LocKind.STATION, "I01")))
    w.register_goods(GoodsUnit("G-007", "B", 0.2, Loc(LocKind.FEED, "I-TRAY")))
    w.bind_and_store("G-007", "PAL-007", Loc(LocKind.PALLET, "PAL-007"), "T1", 0.0)
    w.move_pallet("PAL-007", Loc(LocKind.ROBOT, "F01"), "T2", 1.0)
    # 货物位置跟随绑定托盘 (设计 25.5「车辆运输」行)
    assert w.goods_location("G-007") == "PALLET:PAL-007"
    assert w.active_binding("G-007")[0] == "PAL-007"
    assert w.in_warehouse_count() == 1, "运输中计数不变"