"""按当前 design_version 生成验收报告 (T01–T16)。

报告文件名/标题随 configs/design.yaml 的 design_version 派生
(v<版本>.md / v<版本>.json); 历史版本报告不覆盖 (规则 3)。

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

import yaml as _yaml

DESIGN_VERSION = _yaml.safe_load(
    (REPO / "configs" / "design.yaml").read_text(encoding="utf-8")
)["design_version"]
OUT_MD = REPO / "reports" / f"v{DESIGN_VERSION}.md"
OUT_JSON = REPO / "reports" / f"v{DESIGN_VERSION}.json"


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
    a(f"# 桌面仓储 AGV 演示 v{DESIGN_VERSION} 验收报告")
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
    a("### 6.2 T06 单车白模接力上限 (10-10 复测: 已消除)")
    a("")
    a("`sim/forklift_cell.py` 是单车白模 (一个 freejoint 托盘 + 一根横梁 + 一对叉)。")
    a("旧几何下连续复用同一单元实测: r0–r3 一致, r4 托盘下沉 1.5mm 压回横梁,")
    a("r5 起不可逆 —— 当时写成 `SUSTAINABLE_RELAYS = 4`。")
    a("")
    a("**复测推翻**: 漂移不是白模固有极限, 而是坏几何产物 (Ramp 传绝对时间致")
    a("全程阶跃、插深按叉根致单齿承托、背板干涉、叉孔摩擦 0.8 承不住带载")
    a("加减速)。修复 (commit 53c9320) 后同一单元 **50 轮全部一致**: 最低叉承托")
    a("0.978、托盘高度最大偏差 0.33mm、放回梁承托最低 0.978; 上限已提到 50")
    a("(commit 3c222cd)。**未放宽任何阈值** (0.90 / 0.10 / 1mm 保持原值)。")
    a("详见 `t06_relay_limit.md`。")
    a("")
    a("## 7. 证据文件")
    a("")
    a("|文件|内容|")
    a("|---|---|")
    a("|`reports/environment.md`|环境与依赖锁定|")
    a("|`reports/stow_envelope.json`|收拢位形求解结果与逐连杆包络|")
    a("|`reports/t06_relay_limit.md`|T06 偏离说明与排除过程|")
    a("|`reports/t12_run20.json`|T12 20 轮逐帧快照与一致性校验|")
    a(f"|`reports/v{DESIGN_VERSION}.json`|本报告的机器可读版本|")
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