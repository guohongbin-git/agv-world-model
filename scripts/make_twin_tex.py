"""生成 Web 孪生 (glTF) 用的预染色 PBR 贴图.

glTF 只能序列化 直连 Image→Principled 的贴图链路; render_agv.pbr_mat 的
mix/mul 染色节点导不出去 → 这里把同样的染色语义**烘焙进像素** (线性域,
与 Blender 节点一致):
  mix 模式: out = tint*0.90 + tex*0.10   (平涂喷漆, 留 10% 纹理呼吸)
  mul 模式: out = tint * tex * 2.2       (木色)
外加: 地坪 = 混凝土平铺 + 0.42m 伸缩缝网格 (对应 floor_mat 的 Brick 语义)。

用法: PYTHONPATH=. ./.venv/bin/python scripts/make_twin_tex.py
输出: assets/pbr/twin/{name}.jpg + {name}_nrm.jpg (1K)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
PBR = os.path.join(ROOT, "assets", "pbr")
OUT = os.path.join(PBR, "twin")
SIZE = 1024

SPEC = [
    # out_name, subdir, mode, linear_tint(0-1)
    ("rack_beam", "PaintedMetal001", "mix", (0.60, 0.13, 0.01)),
    ("rack_upright", "PaintedMetal001", "mix", (0.03, 0.08, 0.26)),
    ("machinery_dark", "MetalPlates006", "mix", (0.05, 0.05, 0.06)),
    ("fork_steel", "MetalPlates006", "raw", None),
    ("pallet_wood", "Wood096", "mul", (0.72, 0.48, 0.28)),
]

_LUT_S2L = None
_LUT_L2S = None


def s2l(a):
    g = a / 255.0
    lin = np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)
    return lin


def l2s(a):
    v = np.clip(a, 0.0, 1.0)
    g = np.where(v <= 0.0031308, v * 12.92,
                 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return np.round(g * 255).astype(np.uint8)


def load(sub, fn):
    im = Image.open(f"{PBR}/{sub}/{sub}_2K-JPG_{fn}.jpg").convert("RGB")
    return np.asarray(im.resize((SIZE, SIZE), Image.LANCZOS))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, subdir, mode, tint in SPEC:
        a = s2l(load(subdir, "Color").astype(np.float64))
        if mode == "mix":
            out = np.array(tint)[None, None, :] * 0.90 + a * 0.10
        elif mode == "mul":
            out = a * (np.array(tint)[None, None, :] * 2.2)
        else:
            out = a
        Image.fromarray(l2s(out)).save(f"{OUT}/{name}.jpg", quality=88)
        Image.fromarray(load(subdir, "NormalGL")).save(
            f"{OUT}/{name}_nrm.jpg", quality=85)
        print("wrote", name)
    # 地坪: 平铺 4x4 格 + 缝
    f = s2l(load("Concrete028", "Color").astype(np.float64))
    f = np.rot90(f, k=1)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    seam = ((xx % (SIZE // 4) < 6) | (yy % (SIZE // 4) < 6))
    f *= 0.85
    f[seam] = [0.030, 0.030, 0.033]
    Image.fromarray(l2s(f)).save(f"{OUT}/floor.jpg", quality=88)
    Image.fromarray(load("Concrete028", "NormalGL")).save(
        f"{OUT}/floor_nrm.jpg", quality=85)
    print("wrote floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
