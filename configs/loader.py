"""design.yaml 加载器 — 全项目唯一入口。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load() -> dict:
    """读取 configs/design.yaml (缓存按 mtime 失效)。"""
    p = ROOT / "configs" / "design.yaml"
    return _load(str(p), p.stat().st_mtime)
