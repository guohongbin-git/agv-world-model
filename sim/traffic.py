"""M2 Wave2 — 巷道/交叉口锁矩阵 (openTCS Block 语义的网格特化)。

资源模型 (与 sim/navigation.NavGraph 单源对应):
- **lane 锁**: 每条列通道 / 行通道一线一锁 (col:i / row:j) + 槽位接近
  stub 锁 (stub:...) + 停车 stub 锁 (park:vid)。stub 边声明 crosses →
  claim 时连同被穿越的行通道一起独占 (车体横跨巷道走廊的物理事实)。
- **节点锁**: 交叉点 / 交接台 / 交付台 / 槽位嘴 / 接近点 / 停车点各一把
  —— 即"交叉口锁"。

防死锁定理 (本实现的核心保证):
  路由资源 = 整条剩余路径的 {lane(含 crosses)} ∪ {节点} 集合, **原子
  all-or-nothing** 申请 ("按路径顺序申请, 拿不齐即等待")。反证: 若等待
  图成环, 环上某车 B 已持有节点 c 并等待更前方资源, 但 B 进入 c 的前提
  是其整条路由 (含更前方全部资源) 当时全部空闲且被原子取得 —— 矛盾。
  故零死锁。
  释放采用"到达下一节点才释放上一节点"滞后策略 (retain-until-arrival);
  stub 锚点 (内巷道槽位嘴 sm) 保持到车体回到线上 —— 覆盖 stub 作业期间
  车体仍横跨巷道走廊的包络; 路由终点释放全部 lane 与历史节点 (仅留当前
  所在节点 —— 车停在那里, 物理占用)。

事件流 (环形缓冲, 24h 连续运行内存有界):
  claim_ok / claim_wait (含阻塞资源@持有者) / arrival / release / release_all
  —— W3 决策面板直接消费。

学术参照: openTCS Block (区域互斥) 的网格特化; Space-Time A* 的软化版
占用的数据源即本类 occupancy() 快照 (sim/navigation.a_star)。
"""
from __future__ import annotations

from collections import defaultdict, deque


class TrafficManager:
    """锁矩阵运行时: 原子整路由申请 / 滞后释放 / 等待事件 / 占用快照。"""

    def __init__(self, graph, event_cap: int = 4096):
        self.graph = graph
        self.t = 0.0
        self.lane_holder: dict[str, str] = {}
        self.node_holder: dict[str, str] = {}
        self.held: dict[str, set] = defaultdict(set)     # vid → 资源 id
        self.events: deque = deque(maxlen=event_cap)     # (t, vid, kind, detail)
        self.last_blocked: dict[str, dict] = {}          # vid → 确定阻塞资源
        self.stats: dict[str, dict] = defaultdict(
            lambda: dict(waits=0, wait_s=0.0, claims=0))
        self._wait_since: dict[str, float] = {}

    # ------------------------------------------------------------ 查询
    def occupancy(self, exclude: str | None = None) -> tuple[set, set]:
        """他车持有的 (lanes, nodes) —— 供 A* 动态锁惩罚。"""
        lanes = {r for r, v in self.lane_holder.items() if v != exclude}
        nodes = {r for r, v in self.node_holder.items() if v != exclude}
        return lanes, nodes

    def holder_of(self, resource: str) -> str | None:
        return self.lane_holder.get(resource) or self.node_holder.get(resource)

    def events_view(self, n: int | None = None) -> list:
        return list(self.events) if n is None else list(self.events)[-n:]

    def prune_exclusions(self, vid: str):
        """规划前剪枝: 已无人持有的硬排除资源恢复可用。

        (若只在 claim_route 内剪枝, plan None 的分支永远轮不到剪枝 →
        陈旧排除可使等待者永久无路 → 假死锁。)
        """
        acc = self.last_blocked.get(vid)
        if acc:
            acc["lanes"] = [r for r in acc["lanes"] if r in self.lane_holder]
            acc["nodes"] = [r for r in acc["nodes"] if r in self.node_holder]

    # ------------------------------------------------------------ 申请
    def claim_route(self, vid: str, edges, nodes: list[str],
                    extra_lanes: tuple = (), extra_nodes: tuple = ()) -> bool:
        """原子申请整条路由 (lane 含 crosses + 节点 + 作业外延资源)。

        拿不齐 → 一个不取, 记 claim_wait 事件 (阻塞资源@持有者), 返回
        False; 上层按节拍重试, 重试前以最新占用重跑 A* → 自动绕行。
        """
        lanes = {e.lane for e in edges}
        extra_nodes = set(extra_nodes)
        for e in edges:
            lanes |= set(e.crosses)
            extra_nodes |= set(e.nodes)
        lanes |= set(extra_lanes)
        lane_req, node_req = sorted(lanes), sorted(set(nodes) | extra_nodes)
        # 过期排除剪枝: 上次失败硬排除的资源若已无人持有, 恢复为可用
        # (否则阻塞者离场后等待者仍永久绕行 → 假死锁)
        acc = self.last_blocked.get(vid)
        if acc:
            acc["lanes"] = [r for r in acc["lanes"] if r in self.lane_holder]
            acc["nodes"] = [r for r in acc["nodes"] if r in self.node_holder]
        blk_lanes = [r for r in lane_req
                     if self.lane_holder.get(r) not in (None, vid)]
        blk_nodes = [r for r in node_req
                     if self.node_holder.get(r) not in (None, vid)]
        self.stats[vid]["claims"] += 1
        if blk_lanes or blk_nodes:
            self.stats[vid]["waits"] += 1
            if vid not in self._wait_since:
                self._wait_since[vid] = self.t
            acc = self.last_blocked.setdefault(
                vid, {"lanes": [], "nodes": []})
            for r in blk_lanes:
                if r not in acc["lanes"]:
                    acc["lanes"].append(r)
            for r in blk_nodes:
                if r not in acc["nodes"]:
                    acc["nodes"].append(r)
            self.events.append((self.t, vid, "claim_wait", sorted(
                f"{r}@{self.lane_holder.get(r) or self.node_holder.get(r)}"
                for r in blk_lanes + blk_nodes)[:4]))
            return False
        self.last_blocked.pop(vid, None)
        for r in lane_req:
            self.lane_holder[r] = vid
        for r in node_req:
            self.node_holder[r] = vid
        self.held[vid] |= set(lane_req) | set(node_req)
        if vid in self._wait_since:
            self.stats[vid]["wait_s"] += self.t - self._wait_since[vid]
            del self._wait_since[vid]
        self.events.append((self.t, vid, "claim_ok", node_req[:6]))
        return True

    # ------------------------------------------------------------ 推进/释放
    def notify_progress(self, vid: str, path: list[str], edges, idx: int):
        """车辆沿路由到达 path[idx]: 滞后释放上一节点与改换的 lane。"""
        cur = path[idx]
        prev = path[idx - 1] if idx > 0 else None
        nxt = path[idx + 1] if idx + 1 < len(path) else None
        if prev is not None:
            anchor = self.graph.anchor_node(prev)
            deep = self.graph.stub_internal(cur) or (
                nxt is not None and self.graph.stub_internal(nxt))
            if not (anchor and deep):
                self._release(vid, prev, is_node=True)
        if idx >= len(edges):                    # 路由终点: 清 lane 与历史节点
            for r in sorted(self.held[vid]):
                if r in self.lane_holder and self.lane_holder[r] == vid:
                    self._release(vid, r, is_node=False)
                elif r in self.node_holder and self.node_holder[r] == vid \
                        and r != cur:
                    self._release(vid, r, is_node=True)
        else:
            pe = edges[idx - 1] if idx > 0 else None
            ne = edges[idx]
            if pe is not None and pe.lane != ne.lane \
                    and pe.lane not in ne.crosses:
                self._release(vid, pe.lane, is_node=False)
        self.events.append((self.t, vid, "arrival", cur))

    def release_resources(self, vid: str, resources):
        """作业外延资源显式释放 (如 place 后的 stand 节点/col lane)。"""
        for r in resources:
            self._release(vid, r, is_node=r in self.node_holder)
        self.events.append((self.t, vid, "release",
                            sorted(resources)[:4]))

    def release_all(self, vid: str):
        """复位路径: 释放该车全部资源 (恢复流程用)。"""
        for r in sorted(self.held[vid]):
            if self.lane_holder.get(r) == vid:
                del self.lane_holder[r]
            elif self.node_holder.get(r) == vid:
                del self.node_holder[r]
        self.held[vid].clear()
        self.last_blocked.pop(vid, None)
        if vid in self._wait_since:
            self.stats[vid]["wait_s"] += self.t - self._wait_since[vid]
            del self._wait_since[vid]
        self.events.append((self.t, vid, "release_all", ""))

    def _release(self, vid: str, res: str, is_node: bool):
        table = self.node_holder if is_node else self.lane_holder
        if table.get(res) == vid:
            del table[res]
        self.held[vid].discard(res)

    # ------------------------------------------------------------ 复位
    def reset(self):
        self.lane_holder.clear()
        self.node_holder.clear()
        self.held.clear()
        self._wait_since.clear()
