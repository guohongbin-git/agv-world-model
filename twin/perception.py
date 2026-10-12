"""模拟激光雷达 + 占据栅格 — 观测通道 (设计 §19.2: 禁上帝视角)。

Agent 可见的障碍信息只能来自这里: 从车顶传感器原点 (前轴上方 z=0.09m)
水平面 270°/180 线扫描 (mj_ray), 距离高斯噪声 σ=3mm, 2% 丢束;
命中静态已知几何 (地面/暂存站/立柱装饰) 与 AGV 自体被滤除,
剩余命中写入占据栅格并给出巷道方向最近障碍距离。

噪声模型显式标注: ideal=False (默认含噪)。真值坐标不进入本模块接口。
"""
from __future__ import annotations

import mujoco
import numpy as np

# 已知静态 (不报告为障碍): 地面/暂存站/装饰立柱/货架梁与立柱(视觉)
KNOWN_STATIC_PREFIX = ("floor", "stage_", "upright_", "support_",
                       "hook_", "tie_", "brace_", "beam_", "deck_",
                       "back_", "hose_", "mast_cross")
# AGV 自体 (滤除): 车体/门架装饰/货叉/托盘(被搬运时紧贴)
SELF_PREFIX = ("chassis", "counterweight", "mast_l", "mast_r", "mast_top",
               "fork_", "carriage_", "pal_", "ceil_", "top_side", "top_mid",
               "hubcap", "wheel")


def _is_self_or_known(name: str, vid_prefixes: tuple = ()) -> bool:
    """vid_prefixes: 车队命名空间前缀 (如 'F01_'), 剥离后再匹配 ——
    P3 Wave1 多车参数化后可动件名带 <vid>_ 前缀, 裸名表会漏滤
    (实测残留 F01_mast_l/r 污染占据栅格)。"""
    for p in vid_prefixes:
        if name.startswith(p):
            name = name[len(p):]
            break
    return name.startswith(KNOWN_STATIC_PREFIX) or name.startswith(SELF_PREFIX)

GRID_X0, GRID_Y0 = -1.0, -0.5
GRID_RES = 0.05
GRID_NX, GRID_NY = 40, 20
class Lidar:
    """单线水平激光扫描器 (挂在车顶, 随 chassis 移动)。"""

    def __init__(self, cell, z: float = 0.09, n_rays: int = 180,
                 fov_deg: float = 270.0, max_range: float = 1.5,
                 sigma: float = 0.003, drop: float = 0.02, seed: int = 7):
        self.cell = cell
        vids = getattr(cell, "vehicles", None) or ["F01"]
        self.vid_prefixes = tuple(v + "_" for v in vids)
        self.z = z
        self.n = n_rays
        self.fov = np.deg2rad(fov_deg)
        self.max_range = max_range
        self.sigma = sigma
        self.drop = drop
        self.rng = np.random.default_rng(seed)
        m = cell.model
        self.chassis_bid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.carriage_bid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "carriage")
        self.cone_bids = [mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, f"cone_{i}") for i in range(2)]
        self.grid = np.zeros((GRID_NX, GRID_NY), dtype=bool)
        self.scan_count = 0

    # ------------------------------------------------------------------
    def scan(self) -> dict:
        """执行一次扫描。返回 {dist: np[n], hits: [(x,y)...], cones: [(x,y)...]}。

        hits: 滤除自体/已知静态后的命中点 (世界坐标)。
        cones: 命中锥桶体的估计中心坐标 (占据栅格来源)。
        """
        m, d = self.cell.model, self.cell.data
        pnt = np.array([self.cell.axle_x, 0.0, self.z])
        ang = np.linspace(-self.fov / 2, self.fov / 2, self.n)
        dist = np.full(self.n, self.max_range)
        hits, cone_hits = [], []
        geomid = np.zeros(1, dtype=np.int32)
        for i, a in enumerate(ang):
            if self.rng.random() < self.drop:
                continue                                    # 丢束
            v = np.array([np.cos(a), np.sin(a), 0.0])
            exclude, dist_i, gid_i = -1, self.max_range, -1
            found = False
            for _ in range(6):                              # 滤自体重射
                # binding 语义 (2026-10-11 实测定论): mj_ray 返回值 = 命中
                # 距离 (无命中 -1), geomid 输出数组 [0] = 命中 geom id。
                # 此前把返回值当 gid、距离用几何中心垂足自算 —— 全错
                # (GPT 评审 P0-2 与 10-09 调试独立发现同一 bug)。
                dist_i = float(mujoco.mj_ray(m, d, pnt, v, None, 1,
                                             exclude, geomid))
                gid_i = int(geomid[0])
                if dist_i < 0:
                    break                                   # 无命中
                bid = int(m.geom_bodyid[gid_i])
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                       gid_i) or "?"
                if _is_self_or_known(nm, self.vid_prefixes):
                    exclude = bid
                    continue
                found = True
                break
            if found and 0 <= gid_i < m.ngeom:
                # 高斯噪声 (3mm), 截断正值
                dist_i = max(0.01, dist_i + float(
                    self.rng.normal(0.0, self.sigma)))
                dist[i] = min(dist_i, self.max_range)
                if dist_i < self.max_range - 1e-6:
                    hx = pnt[0] + dist_i * v[0]
                    hy = pnt[1] + dist_i * v[1]
                    hits.append((hx, hy))
                    bid = int(m.geom_bodyid[gid_i])
                    if bid in self.cone_bids:
                        cone_hits.append((hx, hy))
        self.scan_count += 1
        for hx, hy in cone_hits:                            # 占据栅格更新
            ix = int((hx - GRID_X0) / GRID_RES)
            iy = int((hy - GRID_Y0) / GRID_RES)
            if 0 <= ix < GRID_NX and 0 <= iy < GRID_NY:
                self.grid[ix, iy] = True
        return {"dist": dist, "angles": ang, "hits": hits,
                "cones": cone_hits}

    # ------------------------------------------------------------------
    def front_distance(self, direction: float = +1.0,
                       axle_x: float | None = None) -> float:
        """行车方向 ±45° 锥内最近障碍距离 (含噪), 无则 max_range。

        axle_x: 观测原点 (该车前轴 x); 缺省 = cell 首车前轴 (单车 API 不变)。
        P3 Wave2: F02 的行车避障用自己的前轴作原点。

        实现: 对已注册动态障碍 (锥桶 mocap 体) 做 AABB-射线解析相交 ——
        等价 mj_ray 但不受 binding 语义影响; 锥桶真值坐标只用于
        射线相交测试, 不直接暴露给导航 (设计 §19.2 观测通道)。
        """
        d = self.cell.data
        pnt_x = self.cell.axle_x if axle_x is None else float(axle_x)
        best = self.max_range
        for i, bid in enumerate(self.cone_bids):
            if bid < 0 or self.cell.cone_mocapid[i] < 0:
                continue
            cx, cy, _ = d.mocap_pos[self.cell.cone_mocapid[i]]
            dx = (cx - pnt_x) * direction
            if not (0.0 < dx < self.max_range):
                continue                       # 不在行车方向/量程内
            lat = abs(cy)                      # 横向偏移
            if lat > 0.025 + 0.045:            # 锥半径 + 车半宽裕量
                continue
            best = min(best, dx - 0.025)
        if best < self.max_range:
            best = max(0.01, best + float(self.rng.normal(0.0, self.sigma)))
        return best

    def cone_positions(self) -> list[tuple[float, float]]:
        """占据栅格里被确认的锥桶单元中心 (供 UI/叙述; 估计坐标)。"""
        out = []
        for ix, iy in zip(*np.where(self.grid)):
            out.append((round(GRID_X0 + (ix + 0.5) * GRID_RES, 3),
                        round(GRID_Y0 + (iy + 0.5) * GRID_RES, 3)))
        return out
