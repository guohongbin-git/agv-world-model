"""生成 W07 的 v2.0.0 验收报告 (T01–T16)。

用法:
    PYTHONPATH=. .venv/bin/python reports/make_v2_report.py

设计要求 (第 14 章末):
「T03 的 100 次零失败是首版筛选目标, 不足以证明 99% 可靠性达到给定
置信度。每轮成功必须满足动作和库存两方面; 失败分为几何、动力学、视觉、
通信、业务五类, **报告条件和观测, 不删除失败样本**。」

因此报告必须区分三档, 不能混成一个"通过率":
  PASS      判据已实现且实测通过, 且已按设计数量执行
  PARTIAL   判据已实现且通过, 但**执行次数少于设计要求** (如实标注缺口)
  FAIL      有实测失败样本
本项目当前应为 PASS + PARTIAL, 不应有 FAIL。
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from reports.acceptance_cases import CASES, CaseResult, collect  # noqa: E402

OUT_MD = REPO / "reports" / "v2.0.0.md"
OUT_JSON = REPO / "reports" / "v2.0.0.json"


def classify(res: CaseResult) -> str:
    if not res.ran:
        return "FAIL"
    if res.failed:
        return "FAIL"
    return "PASS" if CASES[res.case]["quantified"] else "PARTIAL"


def main() -> int:
    results = collect()
    rows = []
    for case in sorted(CASES):
        res = results[case]
        meta = CASES[case]
        rows.append((case, meta["title"], classify(res), res, meta))

    n_pass = sum(1 for _, _, s, _, _ in rows if s == "PASS")
    n_part = sum(1 for _, _, s, _, _ in rows if s == "PARTIAL")
    n_fail = sum(1 for _, _, s, _, _ in rows if s == "FAIL")
    total_passed = sum(r.passed for _, _, _, r, _ in rows)
    total_failed = sum(r.failed for _, _, _, r, _ in rows)

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    a = lines.append
    a("# 桌面仓储 AGV 演示 v2.0.0 验收报告")
    a("")
    a(f"生成时间: {stamp}")
    a("")
    a("## 1. 总体状态")
    a("")
    a(f"- 用例总数: **{len(rows)}** (T01–T16)")
    a(f"- PASS (判据已按设计数量执行且通过): **{n_pass}**")
    a(f"- PARTIAL (判据已通过但执行次数少于设计要求): **{n_part}**")
    a(f"- FAIL (有实测失败样本): **{n_fail}**")
    a(f"- 累计测试断言: **{total_passed} 通过 / {total_failed} 失败**")
    a("")
    if n_fail:
        a("> **存在失败样本, 按设计要求不得删除, 详见第 3 节。**")
    else:
        a("> 无失败样本。PARTIAL 项的次数缺口见第 4 节, 已如实列出。")
    a("")
    a("## 2. 逐用例结果")
    a("")
    a("|用例|名称|状态|断言(过/总)|判定标准|")
    a("|---|---|---|---|---|")
    for case, title, status, res, meta in rows:
        a(f"|{case}|{title}|**{status}**|{res.passed}/{res.total}|{meta['criterion']}|")
    a("")
    a("## 3. 失败样本")
    a("")
    any_fail = False
    for case, title, status, res, _ in rows:
        if res.failures:
            any_fail = True
            a(f"### {case} {title}")
            for f in res.failures:
                a(f"- `{f}`")
            a("")
    if not any_fail:
        a("无。本轮全部测试文件均无失败样本。")
        a("")
    a("## 4. 未达设计执行次数的缺口 (PARTIAL)")
    a("")
    a("设计对以下用例规定了执行次数, 当前实现覆盖了判据但未按次数重复:")
    a("")
    for case, title, status, res, meta in rows:
        if status == "PARTIAL":
            a(f"- **{case} {title}** — {meta['criterion']}")
            a(f"  - 现状: {res.passed} 项断言通过; 缺口: {meta.get('note', '')}")
    a("")
    a("## 5. 失败分类口径")
    a("")
    a("设计要求把失败分为几何、动力学、视觉、通信、业务五类, 且"
      "「每轮成功必须同时满足动作和库存两方面」。本轮无失败样本, "
      "故分类表为空; 后续出现失败时按此口径登记, 不合并统计。")
    a("")
    a("## 6. 关键物理偏差记录")
    a("")
    a("### 6.1 收拢包络 (设计 981)")
    a("")
    a("设计给的「基座XY半径70mm」被实测为**可达**, 不是不可达:")
    a("`sim/solve_stow.py` 数值求解得 `q=[0, -1.3528, -0.2962, 0, -0.3056, 0]`,")
    a("臂体水平最大半径 **42.3mm** (设计 70mm 内), 最高点 526.0mm (限 600mm),")
    a("活动连杆最低点 70.0mm ≥ 托台面 55mm (不穿台)。")
    a("")
    a("追查过程中发现 `<position>` 执行器漏写 `ctrlrange`, MuJoCo 默认 `[0,0]`,")
    a("每次 `mj_step` 把 ctrl 夹到 0 —— 关节几乎不动。**在此之前所有抓取、")
    a("收拢与轨迹结果都来自一台欠驱动的臂**, 修复后收拢包络才降到 42.3mm。")
    a("")
    a("### 6.2 T06 单车白模接力上限")
    a("")
    a("`sim/forklift_cell.py` 是单车白模 (一个 freejoint 托盘 + 一根横梁 + 一对叉)。")
    a("连续复用同一单元实测: r0–r3 完全一致 (托盘 4100mm、叉承托 1.000、梁 0.000),")
    a("r4 托盘下沉 1.5mm 并压上横梁 (接触穿透 1.2µm), r5 起不可逆地留在梁上。")
    a("")
    a("故 T06 分两层验证: 调度/库存层跑满 50 轮 (每轮新装配物理单元),")
    a("物理保持上限写成显式常量 `SUSTAINABLE_RELAYS = 4`。")
    a("**未放宽任何阈值** (0.90 / 0.10 / 1µm 保持原值)。详见 `t06_relay_limit.md`。")
    a("")
    a("## 7. 证据文件")
    a("")
    a("|文件|内容|")
    a("|---|---|")
    a("|`reports/environment.md`|环境与依赖锁定|")
    a("|`reports/stow_envelope.json`|收拢位形求解结果与逐连杆包络|")
    a("|`reports/t06_relay_limit.md`|T06 偏离说明与排除过程|")
    a("|`reports/t12_run20.json`|T12 20 轮逐帧快照与一致性校验|")
    a("|`reports/v2.0.0.json`|本报告的机器可读版本|")
    a("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "generated_at": stamp,
        "summary": {"total": len(rows), "pass": n_pass, "partial": n_part,
                    "fail": n_fail, "assertions_passed": total_passed,
                    "assertions_failed": total_failed},
        "cases": [{"case": c, "title": t, "status": s,
                   "criterion": m["criterion"], "note": m.get("note", ""),
                   **r.as_dict()}
                  for c, t, s, r, m in rows],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"PASS={n_pass} PARTIAL={n_part} FAIL={n_fail} "
          f"断言 {total_passed}/{total_passed + total_failed}")
    print(f"已写 {OUT_MD}")
    print(f"已写 {OUT_JSON}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())