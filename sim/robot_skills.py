"""robot_skills —— 预验证技能库 (Voyager 范式, 供 agent 层调用)。

设计原则 (对齐 docs/knowledge/related_work.md §4):
- 技能 = **可执行代码 + 前置条件 + 后置断言 + 证据快照**; agent 只做
  "选技能 + 传参数 + 读结果 + 异常时换技能", 不生成物理动作序列。
- 技能本身**不新造物理能力** —— 每个技能展开为 MissionRunner 的既有
  任务字典 (已由 fleet 测试验证的相位机), 因此 agent 不可能组合出物理
  上不成立的序列 (H1 零伪造原则的技能层延伸)。
- 后置断言 + 返回的 evidence 快照 = 学习闭环的素材 (Voyager 的
  environment-feedback refinement: 失败技能 → 参数/策略修正)。

与调度层的关系: Dispatcher (CP-SAT) 产出**计划**; 技能库是**执行入口**;
执行前的世界状态重验在 matrix_server._gate / mission 相位机内 (C02),
技能层的前置条件与之互补 (技能层做"可调用性"检查, 执行层做"发射时"检查)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from sim.warehouse_cell import pallet_name_for


@dataclass
class Skill:
    name: str
    doc: str
    params: dict[str, str]
    build: Callable[[dict], list[dict]]
    pre: Callable[[Any, dict], str | None] | None = None
    post: Callable[[Any, dict], dict] | None = None


@dataclass
class SkillResult:
    skill: str
    ok: bool
    detail: str
    tasks: list[dict] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


def _loc(mr, key: str) -> str | None:
    return mr.ledger.loc.get(key)


def _pallet_at(mr, loc: str) -> str | None:
    for pn, l in mr.ledger.loc.items():
        if l == loc:
            return pn
    return None


# ---------------------------------------------------------------- 前置条件
def _pre_pick(mr, p: dict) -> str | None:
    slot, level = p["slot"], int(p.get("level", 0))
    pn = pallet_name_for(f"{slot}-L{level}")
    if _loc(mr, pn) != f"slot:{slot}":
        return f"货位 {slot} L{level} 无托盘 ({pn})"
    if _pallet_at(mr, f"stand:{p['stand']}"):
        return f"交接台 {p['stand']} 已被占"
    return None


def _pre_haul(mr, p: dict) -> str | None:
    if not _pallet_at(mr, f"stand:{p['stand']}"):
        return f"交接台 {p['stand']} 无托盘可运"
    if _pallet_at(mr, f"dock:{p['dock']}"):
        return f"交付台 {p['dock']} 已被占"
    return None


def _pre_putaway(mr, p: dict) -> str | None:
    if _pallet_at(mr, f"stand:{p['stand']}") is None:
        return f"交接台 {p['stand']} 无托盘可上架"
    if _pallet_at(mr, f"slot:{p['slot']}"):
        return f"目标货位 {p['slot']} 非空"
    return None


def _pre_fetch(mr, p: dict) -> str | None:
    if _pallet_at(mr, f"stand:{p['stand']}"):
        return f"交接台 {p['stand']} 已被占"
    return None


# ---------------------------------------------------------------- 后置断言
def _post_loc(expect_fn):
    """通用后置断言: 技能终点台账核对 + 该托盘物理位姿证据快照。"""
    def f(mr, p: dict) -> dict:
        expect = expect_fn(p)
        pn = _pallet_at(mr, expect)
        ev = {"pallet": pn, "loc": expect if pn else None, "expect": expect}
        if pn:
            pos = mr.cell.pallet_pose(pn)[0]
            ev["pos"] = [round(float(v), 4) for v in pos]
        else:
            ev["mismatch"] = f"台账终点无托盘 (期望 {expect})"
        return ev
    return f


# ---------------------------------------------------------------- 技能表
def _b_pick(p: dict) -> list[dict]:
    return [{"kind": "pick", "slot": p["slot"], "level": int(p.get("level", 0)),
             "stand": p["stand"]}]


def _b_haul(p: dict) -> list[dict]:
    return [{"kind": "haul", "stand": p["stand"], "dock": p["dock"]}]


def _b_outbound(p: dict) -> list[dict]:
    return [{"kind": "pick", "slot": p["slot"], "level": int(p.get("level", 0)),
             "stand": p["stand"]},
            {"kind": "haul", "stand": p["stand"], "dock": p["dock"],
             "after": [0]}]


def _b_replenish(p: dict) -> list[dict]:
    return [{"kind": "fetch", "stand": p["stand"], "dock": p.get("dock", "")},
            {"kind": "putaway", "stand": p["stand"], "slot": p["slot"],
             "level": int(p.get("level", 0)), "after": [0]}]


SKILLS: dict[str, Skill] = {s.name: s for s in [
    Skill("fork_pick_rack",
          "叉车从货架货位取托盘并放到交接台",
          {"slot": "货位 (如 R11-S0)", "level": "层 0..2", "stand": "交接台 (如 H02)"},
          _b_pick, _pre_pick,
          _post_loc(lambda p: f"stand:{p['stand']}")),
    Skill("agv_transport",
          "AGV 从交接台顶升接管托盘并干运至交付台",
          {"stand": "交接台", "dock": "交付台 (如 D_G11)"},
          _b_haul, _pre_haul,
          _post_loc(lambda p: f"dock:{p['dock']}")),
    Skill("outbound_order",
          "出库: 叉车取货 → 交接台 → AGV 干运至交付台 (两车协同)",
          {"slot": "货位", "level": "层", "stand": "交接台", "dock": "交付台"},
          _b_outbound, None,
          _post_loc(lambda p: f"dock:{p['dock']}")),
    Skill("replenish_order",
          "补货: AGV 从交付台取托 → 交接台 → 叉车上架到货位",
          {"slot": "目标货位", "level": "层", "stand": "交接台", "dock": "来源交付台"},
          _b_replenish,
          lambda mr, p: (_pre_fetch(mr, p) or _pre_putaway(mr, p)),
          _post_loc(lambda p: f"slot:{p['slot']}")),
]}


def catalog() -> list[dict]:
    """技能清单 (供 LLM 提示/工具描述)。"""
    return [{"name": s.name, "doc": s.doc,
             "params": s.params} for s in SKILLS.values()]


def invoke(mr, name: str, params: dict, max_s: float = 2400.0) -> SkillResult:
    """调用技能: 前置检查 → 展开任务 → MissionRunner 执行 → 后置断言。"""
    sk = SKILLS.get(name)
    if sk is None:
        return SkillResult(name, False, f"未知技能 ({name}); 可用: "
                           + ", ".join(SKILLS))
    if sk.pre:
        why = sk.pre(mr, params)
        if why:
            return SkillResult(name, False, f"前置条件不满足: {why}")
    tasks = sk.build(params)
    before = len(mr.op_errors)
    res = mr.run(tasks, max_s=max_s)
    new_errs = mr.op_errors[before:]
    ev = {"op_errors": list(new_errs), "tasks": tasks}
    if sk.post and not new_errs:
        ev.update(sk.post(mr, params))
    ok = not new_errs
    # 学习素材: 失败时给出可复用的诊断字段
    if not ok:
        ev["diagnosis"] = ("物理校验失败 — 检查对位/载重/巷道占用; "
                           "矩阵 V0.1 门控层应拒绝该提交")
    return SkillResult(name, ok,
                       "完成" if ok else f"失败: {'; '.join(new_errs[:2])}",
                       tasks, ev)