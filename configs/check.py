"""W01 参数一致性检查 — design.yaml 内在约束核验。

对应设计文档 17.2: 单位转换、货位数量、叉孔间隙、第三层叉高与交接站高度。
运行: python -m configs.check
"""
from __future__ import annotations

import sys

import yaml

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, ok, detail))


def main(path: str = "configs/design.yaml") -> int:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pal, fork, rack = cfg["pallet"], cfg["fork"], cfg["rack"]
    dock, st = cfg["docking"], cfg["station"]

    # 1 单位: 全部键带单位后缀
    def walk(d, prefix=""):
        bad = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if prefix == "units":
                continue
            if isinstance(v, dict):
                bad += walk(v, key)
            elif any(s in k for s in ("size", "width", "height", "depth", "length", "radius",
                                      "offset", "center", "z_m", "x_m", "y_m", "travel", "clearance")) \
                    and not k.endswith(("_m", "_m2", "_mps", "_mps2", "_mm", "_px", "_rad", "_radps", "_radps2")):
                bad.append(key)
        return bad
    unit_bad = walk(cfg)
    check("单位后缀", not unit_bad, f"缺失后缀: {unit_bad}" if unit_bad else "所有长度/速度字段带单位后缀")

    # 2 货位数量 18
    slots = rack["groups"] * rack["columns"] * rack["levels"]
    check("货位数量", slots == 18, f"{rack['groups']}组×{rack['columns']}列×{rack['levels']}层={slots}")

    # 3 货位坐标完整
    na = len(rack["column_centers_group_a_x_m"])
    nb = len(rack["column_centers_group_b_x_m"])
    lv = len(rack["support_z_m"])
    check("货位坐标", na == rack["columns"] and nb == rack["columns"] and lv == rack["levels"],
          f"A列{na} B列{nb} 层{lv}")

    # 4 叉孔配合: channel_width > fork_width
    cw, fw = pal["channel_width_m"], fork["size_m"][1]
    check("叉孔宽度配合", cw > fw, f"叉孔{cw*1000:.0f}mm > 货叉{fw*1000:.1f}mm, 单侧间隙{(cw-fw)/2*1000:.1f}mm")

    # 5 叉孔高度配合: 货叉厚 < 叉孔净高
    fh, ch = fork["size_m"][2], pal["channel_height_m"]
    check("叉孔高度配合", fh < ch, f"叉厚{fh*1000:.1f}mm < 净高{ch*1000:.0f}mm")

    # 6 叉孔贯通: 开孔底 4mm ≥ 底板厚, 托盘总高含凸边
    check("叉孔底面高度", pal["channel_z_m"][0] > 0, f"开孔底 {pal['channel_z_m'][0]*1000:.0f}mm")
    check("凸边总高", abs(pal["total_size_m"][2] - pal["body_size_m"][2] - pal["locator_height_m"]) < 1e-9,
          f"{pal['total_size_m'][2]*1000:.0f}mm = {pal['body_size_m'][2]*1000:.0f}+{pal['locator_height_m']*1000:.0f}mm")

    # 7 第三层插叉高度: 支撑面+10mm; 抬离=+17mm
    top = rack["support_z_m"][-1]
    check("第三层插叉高度", abs(dock["rack_fork_top_insert_z_by_level_m"][-1] - (top + 0.010)) < 1e-9,
          f"410mm = 400+10")
    check("第三层抬离高度", abs(dock["rack_fork_top_lift_z_by_level_m"][-1] - (top + 0.017)) < 1e-9,
          f"417mm = 400+17 (抬离5mm+2mm间隙)")

    # 8 叉程覆盖: 最大叉高 ≥ 417mm
    check("货叉行程", fork["top_z_range_m"][1] >= dock["rack_fork_top_lift_z_by_level_m"][-1],
          f"max {fork['top_z_range_m'][1]*1000:.0f}mm ≥ 417mm")

    # 9 交接站: 轨面55mm, 叉顶面65插/72抬
    check("交接站轨面", abs(st["support_z_m"] - 0.055) < 1e-9, f"{st['support_z_m']*1000:.0f}mm")
    check("H01插叉高度", abs(dock["handoff_fork_top_insert_z_m"] - (st["support_z_m"] + 0.010)) < 1e-9,
          f"65mm = 55+10")
    check("H01抬离高度", abs(dock["handoff_fork_top_lift_z_m"] - (st["support_z_m"] + 0.017)) < 1e-9,
          f"72mm = 55+17")

    # 10 层高与货架净高: 层间距150, 托盘18+货物50+抬升 < 150
    need = pal["total_size_m"][2] + cfg["goods"]["size_m"][2]
    check("层高净空", rack["level_pitch_m"] > need, f"层距{rack['level_pitch_m']*1000:.0f} > 托盘+货物{need*1000:.0f}mm")

    # 11 货位净空容纳托盘+箱
    w, d = rack["slot_clearance_m"][0], rack["slot_clearance_m"][1]
    pw, pd = pal["total_size_m"][0], pal["total_size_m"][1]
    check("货位净空", w > pw and d > pd, f"宽{w*1000:.0f}>{pw*1000:.0f} 深{d*1000:.0f}>{pd*1000:.0f}mm")

    # 12 总载荷 ≤ 机器人能力
    total = pal["cargo_mass_max_kg"] + pal["tare_mass_max_kg"]
    check("载荷核算", abs(total - 0.250) < 1e-9, f"200g货物+50g托盘={total*1000:.0f}g")

    # 13 平板车顶升: 45→65mm 覆盖轨面55±抬离
    plat = cfg["flat_agv"]["platform_z_m"]
    check("顶升行程", plat[0] < st["support_z_m"] < plat[1], f"{plat[0]*1000:.0f}→{plat[1]*1000:.0f}mm 跨轨面55mm")

    # 14 叉孔横向中心与货叉中心一致
    check("叉-孔对中", pal["channel_centers_y_m"] == fork["centers_y_m"], "±15mm")

    # 15 速度限值与文档 R05 一致
    sp = cfg["speed_limits"]
    check("速度限值", sp["normal_max_mps"] == 0.10 and sp["loaded_turn_max_mps"] == 0.05
          and sp["insert_max_mps"] == 0.01 and sp["high_approach_max_mps"] == 0.005, "0.10/0.05/0.01/0.005 m/s")

    fails = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print(f"\n{len(CHECKS) - len(fails)}/{len(CHECKS)} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
