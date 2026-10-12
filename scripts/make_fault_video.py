"""故障对照演示视频生成 (W4b)。

素材 = 真实消融实验输出 (experiments/fault_ablation.py --demo):
同一故障 (S01 虚报抬起) 下 B0 普通系统 vs B3 证据系统的并排时间线。
产出 = deliverables/fault_contrast_demo.mp4 (1920x1080, 中文旁白)。

用法: ./.venv/bin/python scripts/make_fault_video.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
TMP = Path("/tmp/fault_video")
W, H = 1920, 1080
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
BG = (13, 17, 23)
FG = (230, 237, 243)
DIM = (139, 148, 158)
ACC = (88, 166, 255)
BAD = (248, 81, 73)
GOOD = (63, 185, 80)
YEL = (210, 153, 34)

BEATS = [
    ("title", "AGV World Model",
     ["故障对照演示: 同一故障, 两种系统的结局",
      "", "Evidence-Grounded Transactional Execution (Paper I)",
      "desktop-agv-demo · 2026-10-12"],
     "这套系统要回答一个问题：当物理世界和数据库出现分歧时，机器人的执行结果还可信吗？"
     "我们用同一个故障，跑两种系统，给答案。"),
    ("scene", "故障注入 · S01",
     ["场景: 取托核对时刻",
      "", "传感器虚报「已抬起」——高度信号说托盘已被叉起,",
      "但接触证据表明: 叉根本没有碰到托盘。",
      "", "物理真值: 托盘仍在货位 A-L3-C1"],
     "注入 S01 故障：取托核对时刻，高度信号虚报已经抬起，但接触力证据表明，"
     "货叉根本没有碰到托盘。物理真值是：托盘仍在货位。"),
    ("b0", "B0 普通系统 (无门控 / 无原子事务 / 无对账)",
     ["  t=0.60s  账本 = ROBOT:F01        真值 = SLOT:A-L3-C1   分歧",
      "  t=1.10s  账本 = ROBOT:F01        真值 = SLOT:A-L3-C1   分歧",
      "",
      "  → 提交了从未发生的取货: 账本认为「车上有托」",
      "  → 物理上, 托盘还躺在货位里。",
      "",
      "  不安全提交 1/1 · 账实分歧 0.5s · 无恢复机制"],
     "普通系统在零点六秒提交了这次取货。账本记录：托盘在车上。"
     "而物理真值：托盘还在货位里。系统开始相信自己搬了货。"),
    ("b0v", "B0 的终局",
     ["账本状态: 托盘 ROBOT:F01  (系统认为在车上)",
      "物理真值: 托盘 SLOT:A-L3-C1  (实际在货位)",
      "",
      "→ 错误完成态: 后续所有决策都建立在虚假事实上",
      "→ 没有证据门控的系统, 无法察觉自己错了"],
     "这就是错误完成态。之后每一张订单、每一次调度，都建立在这个虚假事实上。"
     "没有证据门控的系统，根本不知道自己错了。"),
    ("b3", "B3 证据系统 (PREPARED → 证据 → 原子提交 → 对账)",
     ["  t=1.10s  PICK:rejected    ← 五门证据不成立, 提交被拒",
      "  t=2.11s  重试真取 → 承托率 98% ∧ 稳定窗 0.5s ∧ 位姿达标",
      "  t=2.61s  PICK:committed   ← 证据成立, 原子提交",
      "  t=4.16s  PLACE:committed  ← 放托闭合",
      "",
      "  终局: 账本 = SLOT:H01   真值 = SLOT:H01   一致"],
     "证据系统在同样的故障下，拒绝了这次提交：五门联合判据不成立。"
     "它重试、真的叉起了托盘，拿到百分之九十八承托率和半秒稳定窗，才原子提交。"
     "最终账本与物理真值完全一致。"),
    ("b3v", "B3 的裁决",
     ["任务完成: 是         不安全提交: 0 / 2",
      "原子性违规: 0 / 0    故障恢复: 1.00",
      "",
      "→ 拒绝无法证实的事实, 但不停摆: 重试后真取、正常完成",
      "→ 恢复靠重新观测, 不靠一句口令、不靠超时猜测"],
     "它拒绝无法证实的事实，但没有停摆：重试、真取、正常完成。"
     "恢复靠的是重新观测，不是超时猜测，也不是人为指令。"),
    ("ablation", "消融全景 (120 runs · 4 组 × 10 场景 × 3 seed)",
     ["不安全提交率      B0 61.9%  →  B1 4.8%  →  B2 0  →  B3 0",
      "原子性违规        B0 100%   →  B1 100%  →  B2 0  →  B3 0",
      "故障恢复率        B2 25%    →  B3 100%  (0.32s)",
      "任务完成率        B0 20%    →  B3 100%",
      "",
      "  三档机制各自挡住哪类故障: 门控 → 原子事务 → 恢复对账"],
     "把三个机制逐个拆开做消融：只有门控，不安全提交降到百分之四点八；"
     "加上原子事务，降为零；再加上恢复对账，完成率从百分之二十升到百分之百。"
     "安全指标全零的同时，系统没有变成什么都不做的摆设。"),
    ("end", "结论",
     ["数据库可以回滚, 物理世界不能回滚。",
      "",
      "证据系统的价值不是让机器人动起来,",
      "而是让每一次执行都成为企业可消费的可信状态:",
      "  可验证 · 可追溯 · 可恢复。",
      "",
      "github.com/guohongbin-git/agv-world-model"],
     "数据库可以回滚，物理世界不能回滚。证据系统的价值不是让机器人动起来，"
     "而是让每一次执行，都成为可验证、可追溯、可恢复的可信状态。"),
]


def render(kind: str, title: str, lines: list[str], path: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT, 58)
    f_sub = ImageFont.truetype(FONT, 40)
    f_body = ImageFont.truetype(FONT, 34)
    col = {"b0": BAD, "b0v": BAD, "b3": GOOD, "b3v": GOOD,
           "ablation": YEL, "scene": ACC, "end": ACC}.get(kind, FG)
    d.rectangle([0, 0, W, 8], fill=col)
    d.text((100, 90), title, font=f_title, fill=col)
    y = 220
    for ln in lines:
        if ln.strip():
            c = FG
            if "分歧" in ln or "未发生" in ln or "拒绝" in ln and kind == "b3":
                c = BAD if "分歧" in ln else GOOD
            if "一致" in ln and "不一致" not in ln:
                c = GOOD
            if ln.strip().startswith("→") or ln.strip().startswith("·"):
                c = col
            d.text((100, y), ln, font=f_body, fill=c)
        y += 56
    d.text((100, H - 70), "desktop-agv-demo · MuJoCo 物理证据门控 · Paper I",
           font=f_sub, fill=DIM)
    img.save(path)


def tts(text: str, path: Path) -> float:
    """macOS say (Tingting 中文) → aac (edge-tts 端点 403, 改离线合成)。"""
    aiff = path.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Tingting", "-r", "190",
                    "-o", str(aiff), text], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(aiff),
                    "-c:a", "aac", "-b:a", "160k", str(path)], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def main() -> int:
    TMP.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    beats = []
    for i, (kind, title, lines, narr) in enumerate(BEATS):
        png = TMP / f"f_{i:02d}.png"
        render(kind, title, lines, png)
        mp3 = TMP / f"a_{i:02d}.m4a"
        dur = tts(narr, mp3)
        beats.append((png, mp3, dur))
        print(f"[{i}] {title[:40]}  audio {dur:.1f}s")
    # 每 beat → 视频段 (静帧 hold + 旁白)
    segs = []
    for i, (png, mp3, dur) in enumerate(beats):
        seg = TMP / f"s_{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-loop", "1", "-i", str(png),
                        "-i", str(mp3),
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-r", "30",
                        "-c:a", "aac", "-b:a", "160k",
                        "-t", f"{dur + 0.7:.2f}",
                        "-af", "apad=pad_dur=0.7",
                        str(seg)], check=True)
        segs.append(seg)
    lst = TMP / "list.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    final = OUT / "fault_contrast_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat",
                    "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(final)], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", str(final)],
                         capture_output=True, text=True).stdout.strip()
    print(f"\n成片: {final} ({float(dur):.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())