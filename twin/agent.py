"""WarehouseAgent — 世界模型的唯一指令入口.

对话 → 意图 → 服务命令 (task/pause/speed/query)。P1 内核为确定性规则
(演示零依赖); 设 TWIN_LLM_BASE (OpenAI 兼容端点, 如本机 oMLX) 后升级为
LLM 工具调用循环, 工具集与设计文档 §2 的六个对外工具对齐。

铁律: agent 只能调用工具, 不能触碰库存写入 —— 业务事实只经 World 事务,
而 World 事务只在物理证据成立时被服务层提交。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

LLM_BASE = os.environ.get("TWIN_LLM_BASE", "")     # e.g. http://127.0.0.1:1234/v1
LLM_MODEL = os.environ.get("TWIN_LLM_MODEL", "")

SYSTEM = (
    "你是仓储调度 Agent, 管理一个 1:15 桌面仓库数字空间"
    " (F01/F02 两台无人叉车, 同一巷道互斥作业)。"
    "你只能通过工具影响世界: 下发任务、暂停、调速、查询库存与事件。"
    "你不能修改库存事实 —— 事实只由物理执行证据经世界模型事务产生。"
    "回答用简体中文, 简短专业。"
)

TOOLS = [
    {"type": "function", "function": {
        "name": "task_start", "description": "下发一轮下架任务: A-L3-C1 → H01",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "task_handoff",
        "description": "下发双车交接: F01 取货放交接台 S1, F02 等巷道空闲后"
                       "接力送 H01 ( pallet 所有权 ROBOT:F01→STATION:S1→"
                       "ROBOT:F02→SLOT:H01 )",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "task_pause", "description": "暂停当前任务",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "task_resume", "description": "恢复已暂停任务",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "task_reset", "description": "复位: 新 run, 账本与物理归初态",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "set_speed", "description": "播放倍率 0.25~8 (物理不变)",
        "parameters": {"type": "object",
                       "properties": {"value": {"type": "number"}},
                       "required": ["value"]}}},
    {"type": "function", "function": {
        "name": "inventory_query", "description": "查询库存板: 货位/托盘/货物/计数",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "inject_cone", "description": "向巷道注入锥桶障碍 (x 位置)",
        "parameters": {"type": "object",
                       "properties": {"value": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "clear_cone", "description": "移除场景中的锥桶",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "events_query", "description": "查询最近世界模型事件",
        "parameters": {"type": "object",
                       "properties": {"limit": {"type": "integer"}}}}},
]


def _exec(cmd: str, svc, value=None) -> str:
    """工具落地 (与规则内核共用, 幂等)。"""
    if cmd == "task_start":
        r = svc.cmd("start")
        return json.dumps(r, ensure_ascii=False)
    if cmd == "task_handoff":
        return json.dumps(svc.cmd("dispatch", list(svc.HANDOFF_TASKS)),
                          ensure_ascii=False)
    if cmd == "task_pause":
        return json.dumps(svc.cmd("pause"), ensure_ascii=False)
    if cmd == "task_resume":
        return json.dumps(svc.cmd("resume"), ensure_ascii=False)
    if cmd == "task_reset":
        return json.dumps(svc.cmd("reset"), ensure_ascii=False)
    if cmd == "set_speed":
        return json.dumps(svc.cmd("speed", value), ensure_ascii=False)
    if cmd == "inventory_query":
        return json.dumps(svc.board(), ensure_ascii=False)
    if cmd == "task_dispatch":
        return json.dumps(svc.cmd("dispatch", value), ensure_ascii=False)
    if cmd == "inject_cone":
        return json.dumps(svc.cmd("inject_cone", value), ensure_ascii=False)
    if cmd == "clear_cone":
        return json.dumps(svc.cmd("clear_cone"), ensure_ascii=False)
    if cmd == "events_query":
        rows, ev = svc.events_since(max(0, len(rows0(svc)) - int(value or 5)))
        return json.dumps(ev, ensure_ascii=False)
    return json.dumps({"error": "UNKNOWN_TOOL"})


def rows0(svc):
    r, _ = svc.events_since(0)
    return r


# ---------- 规则内核 (P1 默认, 离线确定性) ----------
def rules(text: str, svc) -> tuple[str, list]:
    t = text.strip()
    tools: list = []
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×倍]?", t)
    if any(w in t for w in ("暂停", "停一下", "pause")):
        tools.append(("task_pause", None))
        reply = "已暂停。物理冻结在当前状态, 目标位保持; 说\u201c继续\u201d恢复。"
    elif any(w in t for w in ("继续", "恢复", "resume")):
        tools.append(("task_resume", None))
        reply = "已恢复执行。"
    elif any(w in t for w in ("复位", "重置", "reset")):
        tools.append(("task_reset", None))
        reply = ("已复位: 新 run_id, 库存账本重新播种, 与物理初态一致。"
                 "历史 run 的 SQLite 保留在 twin/runs/ 可审计。")
    elif any(w in t for w in ("进度", "到哪", "在做什么", "怎么样")):
        st = svc.state(0, 0)
        reply = (f"任务状态 {svc.mode}, 第 {st['step']}/{st['total'] or '?'} 步: "
                 f"{svc.step_label}; 仿真时间 {st['sim_t']:.1f}s。")
    elif any(w in t for w in ("接力", "交接", "双车", "F02")):
        if svc.mode == "running":
            reply = ("已有任务在执行中 (双车互斥, 交接须等当前队列完成); "
                     "可先\u201c暂停\u201d或\u201c复位\u201d。")
        else:
            tools.append(("task_handoff", None))
            reply = ("已下发双车交接: ①F01 从 A-L3-C1 取 PAL-001 放交接台 S1"
                     " (放托验真后提交 STATION:S1, 随即前移让位清出巷道) "
                     "②F02 等巷道空闲后接力, S1 取托送出库位 H01"
                     " (提交 SLOT:H01 + 出库待命)。交接是单一托盘的所有权"
                     "转移链 ROBOT:F01 → STATION:S1 → ROBOT:F02 → SLOT:H01,"
                     " 巷道作业区全程互斥。")
    elif any(w in t for w in ("放回", "上架", "归位", "回架")):
        tools.append(("task_dispatch", [("PAL-001", "G-001", "STAGE", "L3")]))
        reply = ("已下发上架任务: PAL-001 从 H01 放回 A-L3-C1。"
                 "回架走真实对位: ≤5mm/s 插入 → 缓降落梁 → 脱离叉孔上壁 → "
                 "证据验真后事务提交 (生命周期回到 IN_WAREHOUSE)。")
    elif any(w in t for w in ("完整", "两轮", "整理", "闭环", "跑一遍")):
        tools.append(("task_dispatch", None))
        reply = ("已下发完整闭环队列: ①PAL-001 下架 L3→H01 ②放回 H01→L3。"
                 "两任务串行, 每步判据限速, 事务只在证据成立时提交。")
    elif any(w in t for w in ("取", "下架", "搬", "放到", "开始", "启动",
                              "执行", "跑一轮", "出库")):
        if svc.mode == "running":
            reply = "已有任务在执行中; 可先\u201c暂停\u201d或\u201c复位\u201d。"
        else:
            tools.append(("task_dispatch",
                          [("PAL-001", "G-001", "L3", "STAGE")]))
            reply = ("任务已下发: PAL-001 下架 A-L3-C1 → H01。执行走判据层,"
                     " 库存只在物理证据成立后提交 —— 看右侧事件流。")
    elif any(w in t for w in ("移走", "移除", "拿走", "清走", "搬走")):
        tools.append(("clear_cone", None))
        reply = ("锥桶已移出场景。F01 会先用 LiDAR 连续确认通道恢复, "
                 "再重新规划并继续任务 —— 恢复需要感知证据, 不靠一句口令。")
    elif any(w in t for w in ("锥桶", "障碍", "挡住", "注入")):
        m2 = re.search(r"(-?\d+\.?\d*)", t)
        x = float(m2.group(1)) if m2 else -0.25
        tools.append(("inject_cone", x))
        reply = (f"已在巷道 x={x:+.2f}m 注入锥桶。这是环境变化 —— "
                 "如果 F01 正在执行任务, 它的 LiDAR 会自行检测并安全停车, "
                 "我随后报告处置结果。")
    elif any(w in t for w in ("库存", "多少货", "盘点", "几托", "状态")):
        tools.append(("inventory_query", None))
        b = svc.board()
        pal = ", ".join(f"{p[0]}@{p[1]}({p[2]})" for p in b["pallets"])
        gd = ", ".join(f"{g[0]}·{g[1]}@{g[2]}[{g[3]}]" for g in b["goods"])
        reply = (f"当前在库 {b['count']} 件。托盘: {pal}。货物: {gd}。"
                 f"货位: " + ", ".join(f"{s[0]}={s[1]}" for s in b["slots"])
                 + "。以上均来自世界模型事务查询, 非画面硬编码。")
    elif any(w in t for w in ("事件", "日志", "审计", "发生过")):
        tools.append(("events_query", 6))
        _, ev = svc.events_since(max(0, len(rows0(svc)) - 6))
        reply = "最近事件: " + ("; ".join(
            f"[{e['t']}s] {e['type']} {e['payload'].get('label', '')}"
            f"{e['payload'].get('to_location', '')}" for e in ev) or "无") \
            + "。完整事件账本在 SQLite, 可回放。"
    elif any(w in t for w in ("你好", "介绍", "能做什么", "help", "帮助")):
        reply = ("我能: ①\u201c把 A-L3-C1 那托下架到 H01\u201d下发单车任务 "
                 "②\u201cF02 接力送 H01\u201d下发双车交接 "
                 "③\u201c暂停/继续/复位\u201d控制执行 ④\u201c快 4 倍\u201d调播放倍率 "
                 "⑤\u201c库存/进度/事件\u201d查询世界模型。所有指令走同一条"
                 "判据→物理→事务链, 没有旁路。")
    else:
        reply = ("没听懂。试试: \u201c把 A-L3-C1 那托下架到 H01\u201d / "
                 "\u201cF02 接力送 H01\u201d / \u201c暂停\u201d / \u201c快 4 倍\u201d / "
                 "\u201c库存\u201d / \u201c事件\u201d。")
    for cmd, v in tools:
        _exec(cmd, svc, v)
    return reply, [{"tool": c, "arg": v} for c, v in tools]


# ---------- LLM 工具循环 (设 TWIN_LLM_BASE 后启用) ----------
def _llm(text: str, svc) -> tuple[str, list] | None:
    if not LLM_BASE:
        return None
    try:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": text}]
        body = json.dumps({"model": LLM_MODEL or "default",
                           "messages": msgs, "tools": TOOLS,
                           "temperature": 0.2}).encode()
        used: list = []
        for _ in range(4):
            req = urllib.request.Request(
                LLM_BASE.rstrip("/") + "/chat/completions", data=body,
                headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                j = json.loads(r.read())
            msg = j["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content", ""), used
            msgs.append(msg)
            for c in calls:
                fn = c["function"]["name"]
                arg = json.loads(c["function"].get("arguments") or "{}")
                v = arg.get("value") or arg.get("limit")
                obs = _exec(fn, svc, v)
                used.append({"tool": fn, "arg": v})
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": obs})
        return "（本轮工具调用达到上限）", used
    except Exception:
        return None          # LLM 不可达 → 回落规则内核


def handle(text: str, svc) -> dict:
    out = _llm(text, svc)
    if out is None:
        out = rules(text, svc)
    reply, tools = out
    return {"reply": reply, "tools": tools, "engine": "llm" if LLM_BASE else "rules"}
