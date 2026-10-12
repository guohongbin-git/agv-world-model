#!/usr/bin/env python3
"""Observation 接口 (GPT 评审 P0-3 观测隔离): 决策侧唯一感知入口。

边界 (观测/真值分离):
- 决策侧 (server 避障/恢复确认, agent, Decision Memory) 只消费本模块:
  * front_obstacle(vid)  —— 行车方向走廊内最近动态障碍距离 (扫描估计)
  * cone_estimate(vid)   —— 最近一次扫描的锥桶中心估计 (记忆签名用)
  * clear_confirmed(vid) —— 连续 N 次扫描 (带时间戳/序号) 走廊无障碍
- 真值 (mocap_pos 注入坐标) 只存在于环境操作 (place_cone/remove_cones,
  模拟现场人员) 与 evaluator (tests, 边界外专读真值来证明「有噪声但
  无泄漏」)。本模块不读取任何真值 —— 一切数字来自 LiDAR 扫描
  (mj_ray 物理光线 + σ=3mm 距离噪声 + 2% 丢束, twin/perception.py)。

估计器 (与旧解析真值版同构, 但输入换成扫描):
锥桶是已知半径 r=25mm 的注册动态路标。扫描命中其可见表面 (含噪),
按命中点聚类、逐命中定半径反演 (al + sqrt(r²-la²)) 得中心估计;
行车距离 = 中心纵向距离 - r —— 对应旧 front_distance 的 dx-0.025,
但 dx 来自带噪扫描而不是 data.mocap_pos。

扫描面高度 z=0.05m: 穿过全场唯一的水平净空带 (暂存台面 45mm 之上,
最低货物箱底 57mm 之下) —— 0.09m 会被车载货物箱 (77~207mm) 完全
遮挡带载前行方向的视线, 0.05m 对锥桶 (高 90mm) 全高可见。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import mujoco

# 已知路标几何 (传感器模型知识, 非真值): 锥桶圆柱 r25 / 行车走廊半宽
# (锥半径 0.025 + 车半宽裕量 0.045, 与旧解析版口径一致)
CONE_RADIUS = 0.025
CORRIDOR_HALF_WIDTH = 0.070
SCAN_Z = 0.05


@dataclass(frozen=True)
class Obs:
    """一次行车方向观测: 最近障碍距离估计 + 时间戳 + 置信度。"""
    near_dist_est: float     # 走廊内最近锥面距离 (含噪); 无障碍 = max_range
    timestamp: float         # 扫描时刻 (仿真时钟)
    confidence: float        # [0,1]: 走廊内锥桶命中点数 / 5 (截 1.0);
                             # 无障碍时 = 1.0 (丢束残余风险由 clear_confirmed
                             # 的连续 N 次门控覆盖)


class ObservationChannel:
    """每车一条观测通道; 对外只给估计, 不给真值。"""

    def __init__(self, cell, sigma: float = 0.003, drop: float = 0.02,
                 clear_n: int = 3):
        from twin.perception import Lidar
        self.cell = cell
        self.clear_n = int(clear_n)
        self._lidars = {vid: Lidar(cell, z=SCAN_Z, sigma=sigma, drop=drop,
                                   seed=11 + i)
                        for i, vid in enumerate(cell.vehicles)}
        # vid -> 最近一次扫描记录 (scan 结果 + 原点 + 时刻)
        self._last_scan: dict[str, dict] = {}
        # vid -> 最近 clear_n 次恢复确认扫描 [(ts, seq, clear), ...]
        self._streak: dict[str, deque] = {}

    # ------------------------------------------------------------------
    def _scan(self, vid: str, direction: float = +1.0,
              forward: bool = False) -> tuple[dict, float, float]:
        """执行一次该车的扫描; 返回 (scan, origin_x, ts)。

        forward=True 时先 mj_forward —— 物理停走 (blocked 等待) 期间
        mocap 写入 (锥桶移除) 不会自己刷新几何缓存, 需显式前向。
        """
        lid = self._lidars.get(vid)
        if lid is None:
            raise KeyError(f"未注册车辆: {vid} (已注册 {list(self._lidars)})")
        ax = self.cell.vehicle(vid).axle_x
        if forward:
            mujoco.mj_forward(self.cell.model, self.cell.data)
        scan = lid.scan(axle_x=ax, direction=direction)
        ts = float(self.cell.data.time)
        self._last_scan[vid] = {"scan": scan, "origin_x": ax, "ts": ts}
        return scan, ax, ts

    @staticmethod
    def _clusters(points: list[tuple[float, float]], origin_x: float
                  ) -> list[tuple[float, float]]:
        """锥桶命中点 (含噪表面点) 单链聚类 → 各簇中心估计。

        逐命中精确几何 (定半径反演): 命中 p 在半径 r 的圆上, 沿方位
        投影 al 与横向 la 满足 center_al = al + sqrt(r²-la²) —— 对任意
        距离无偏 (连续弧的弦平均只在近距离近似成立: 远处仅 2 束命中
        时 πr/4 修正会偏 ~4mm)。簇均值给出方位 û, 中心 = 原点 + D̄·û。
        """
        clusters: list[list[tuple[float, float]]] = []
        for p in points:
            for c in clusters:
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.08
                       for q in c):
                    c.append(p)
                    break
            else:
                clusters.append([p])
        ests = []
        for c in clusters:
            mx = sum(p[0] for p in c) / len(c)
            my = sum(p[1] for p in c) / len(c)
            ux, uy = mx - origin_x, my - 0.0
            n = math.hypot(ux, uy) or 1.0
            ux, uy = ux / n, uy / n                    # 方位单位向量
            d_sum = 0.0
            for px, py in c:
                al = (px - origin_x) * ux + py * uy    # 沿方位投影
                la = -(px - origin_x) * uy + py * ux   # 横向 (带符号)
                d_sum += al + math.sqrt(
                    max(CONE_RADIUS ** 2 - la * la, 0.0))
            d = d_sum / len(c)
            ests.append((origin_x + ux * d, uy * d))
        return ests

    @staticmethod
    def _corridor(ests: list[tuple[float, float]], origin_x: float,
                  direction: float, max_range: float
                  ) -> list[tuple[float, float]]:
        """行车方向走廊内的簇: 0 < 纵向距离 < 量程 且 |横向| ≤ 半宽。"""
        return [e for e in ests
                if 0.0 < (e[0] - origin_x) * direction < max_range
                and abs(e[1]) <= CORRIDOR_HALF_WIDTH]

    def _estimate(self, vid: str, scan: dict, ax: float, ts: float,
                  direction: float) -> Obs:
        """从一次扫描估计行车方向最近锥桶距离。"""
        lid = self._lidars[vid]
        ests = self._clusters(scan["cones"], ax)
        corr = self._corridor(ests, ax, direction, lid.max_range)
        if not corr:
            return Obs(lid.max_range, ts, 1.0)
        near = min((e[0] - ax) * direction - CONE_RADIUS for e in corr)
        return Obs(max(0.01, near), ts, min(1.0, len(corr) / 5.0))

    def front_obstacle(self, vid: str, direction: float = +1.0) -> Obs:
        """行车方向最近动态障碍 (注册锥桶) 距离估计 —— 避障判据唯一入口。"""
        scan, ax, ts = self._scan(vid, direction=direction)
        return self._estimate(vid, scan, ax, ts, direction)

    def cone_estimate(self, vid: str) -> \
            tuple[tuple[float, float] | None, float]:
        """最近一次扫描的锥桶中心估计 (含噪) —— 记忆签名/叙述只用这个。

        返回 ((x_est, y_est), ts); 无锥桶命中则 (None, ts)。
        取走廊内最近簇 (与避障口径一致); 走廊外有锥桶时取最近簇。
        """
        rec = self._last_scan.get(vid)
        ts = float(self.cell.data.time)
        if not rec or not rec["scan"]["cones"]:
            return None, ts
        ax = rec["origin_x"]
        ests = self._clusters(rec["scan"]["cones"], ax)
        if not ests:
            return None, ts
        lid = self._lidars[vid]
        corr = self._corridor(ests, ax, +1.0, lid.max_range)
        pool = corr or ests
        best = min(pool, key=lambda e: abs(e[0] - ax))
        return (round(best[0], 4), round(best[1], 4)), rec["ts"]

    # ------------------------------------------------------------------
    def clear_confirmed(self, vid: str, direction: float = +1.0) -> bool:
        """恢复确认: 连续 clear_n 次扫描 (带时间戳/序号) 走廊无障碍。

        每次调用执行一次当前扫描; 物理停走 (blocked 等待) 期间 mocap
        写入 (锥桶移除) 不会自己刷新几何缓存, 故先 mj_forward。看到
        障碍即打断连胜。仿真冻结时时间戳不前进, 连续性由扫描序号
        (lidar.scan_count) 保证 —— 不会用旧扫描凑数。
        """
        scan, ax, ts = self._scan(vid, direction=direction, forward=True)
        obs = self._estimate(vid, scan, ax, ts, direction)
        lid = self._lidars[vid]
        clear = obs.near_dist_est >= lid.max_range - 1e-9
        hist = self._streak.setdefault(vid, deque(maxlen=self.clear_n))
        hist.append((obs.timestamp, lid.scan_count, clear))
        return len(hist) == self.clear_n and all(c for _, _, c in hist)

    def reset_clear(self, vid: str) -> None:
        """清空恢复确认连胜 (进入 blocked 时调用, 防旧扫描凑数)。"""
        self._streak.pop(vid, None)

    def clear_history(self, vid: str) -> list[tuple[float, int, bool]]:
        """恢复确认扫描历史 (只读, 供测试/UI 叙述)。"""
        return list(self._streak.get(vid, ()))
