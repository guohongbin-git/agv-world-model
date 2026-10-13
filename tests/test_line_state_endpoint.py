"""/api/line/state 两态测试 (蓝图 §15 产线 tab; 文件契约数据源:
dc-twin 发布器写 line_state.json, matrix_server 只读透传)。

不依赖 TestClient/httpx: 端点为纯读代理, 直接调协程 + 断言路由注册。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from twin import matrix_server

LINE_STATE = "/api/line/state"


def _payload() -> dict:
    return {"available": True, "run_id": "LINE-T",
            "written_at": "2026-10-13T00:00:00+00:00",
            "core_version": "2.5.0",
            "state": {"sim_t": 12.5,
                      "station": {"ST-PRESS": {"state": "pressing"}},
                      "production": {"physical_done": 2,
                                     "ledger_committed": 2,
                                     "mismatches": []}}}


def _get() -> dict:
    return asyncio.run(matrix_server.line_state())


def test_route_registered():
    assert any(getattr(r, "path", None) == LINE_STATE
               for r in matrix_server.app.routes)


def test_line_state_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LINE_STATE_PATH", str(tmp_path / "nope.json"))
    assert _get() == {"available": False}        # 面板显示产线离线


def test_line_state_reads_published_file(tmp_path, monkeypatch):
    p = tmp_path / "line_state.json"
    p.write_text(json.dumps(_payload(), ensure_ascii=False),
                 encoding="utf-8")
    monkeypatch.setenv("LINE_STATE_PATH", str(p))
    assert _get() == _payload()                  # 透传, 不改写契约


def test_line_state_corrupt_file(tmp_path, monkeypatch):
    p = tmp_path / "line_state.json"
    p.write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("LINE_STATE_PATH", str(p))
    assert _get() == {"available": False}        # 坏文件按离线降级, 不 5xx
