"""M2 Wave1 — 128 架矩阵仓库物理层 (新场景, 不改动 forklift_cell.py)。

布局 (LAYOUT 单一来源, 地面标志线/货架/交接台全部程序化生成):
- 4×4 货架组矩阵, 每组 8 架 = 2 排(前/后) × 4 列 × 每架 3 层
  → **128 架 / 384 货位**; 组号 G<r><c> (r 行 c 列, 1..4),
  货位 R<r><c>-S<0..7>-L<0..2> (S0-3 前排朝南巷道, S4-7 后排朝组内巷道;
  层高 0.100/0.250/0.400, 同 forklift_cell levels 机制)。
- 组间列通道 (x 向 0.46) / 行通道 (y 向 0.44) ≥ 叉车包络 0.33 + 2×50mm 余量。
- 南缘泊位带: 交接台 H01..H04 (航向 +90°, 交接轴沿世界 +y);
  交付台 D_G<r><c> 每组一台、位于目标组巷道口 (r=1..3 在组间行通道,
  r=4 在北泊位带; 航向 0°, 开口沿世界 ±x)。
- 地面标志线: 行车通道中心黄虚线 / 货架组轮廓白线 / 交接台区蓝框 /
  停车点白框 —— 全部由布局参数推导, 改一个参数全场联动。

车辆 (同一 MJCF 多车, 可动部分 <vid>_ 前缀):
- 叉车 F01/F02: 底盘升级为 x+y 滑动 + yaw 铰 (原地任意转向) + 门架举升 + 双叉;
  叉/门架几何沿用 forklift_cell 的设计值 (fork_root 0.040, fork_len 0.080,
  transport_z 0.065, 插深 0.065), 运动学不变故插叉物理与 T01-T04 同源;
  三层插叉位 = 层高 + 0.010/+0.017 (T04 已验证的层高派生关系)。
- AGV A01/A02: 潜伏顶升 —— 圆盘底盘 r=0.12 h=0.06 + 顶销盘 (行程 0.030)。
  顶销半径 0.018: 托盘实际 80×60mm, 销盘必须钻入托盘底部投影内, r=0.08 的
  名义值物理不可行, 按托盘真实尺寸缩小 (语义不变: 顶销圆盘承接凸台)。

托盘: 沿用 forklift_cell 白模 + 底面中央凸台 (40×40×30mm) = AGV 顶销承接面;
  货架每架每层两根横梁 (无中梁), 给凸台让出净空 (层间距 0.150 > 凸台 0.030);
  台面条带中开 60×48mm 孔, 凸台垂入孔内, 销盘从孔中顶升。
  总质量 0.048kg ≤ tare_mass_max 0.050。

交接物理链: 叉车取架 → 放交接台 (托盘落台面条带) → AGV 对中钻入 → 顶销升
  (托盘离台 27mm) → 原地转 90° (通道轴对准交付台开口) → 干线运输 →
  目标组巷道口对中钻入 → 降销放托 → 退场。
  全程位置伺服 + Ramp 梯形限速 (巡航 0.30-0.35, 加速度 ≤0.5 m/s²)。
  叉车回程穿台区须先升叉至 0.100 (叉顶): 过凸台顶 3.8mm/台内顶板下 7mm,
  否则空叉滑轨背板 (z 0.060-0.076) 会撞上台面托盘凸台。

时钟: 纯仿真; 控制 100Hz, 物理 1ms (同 forklift_cell)。
"""
from __future__ import annotations

import math
import re

import mujoco
import numpy as np

from configs.loader import load
from sim.controllers import Ramp

# ---------------------------------------------------------------- 布局单源
LAYOUT = {
    "group_rows": 4, "group_cols": 4,       # 4×4 = 16 组
    "bays_per_row": 4, "rack_rows": 2,      # 每组 8 架 = 2 排 × 4 列
    "levels_z": [0.100, 0.250, 0.400],      # 每架 3 层 → 128 架/384 货位
    "bay_pitch_x": 0.16,                    # 架间距 (架宽 0.14 + 缘距)
    "rack_depth": 0.12,                     # 架深 (横梁 y 向长 0.11)
    "inner_aisle_y": 0.40,                  # 组内两排架之间巷道
    "col_aisle_x": 0.46,                    # 组间列通道
    "row_aisle_y": 0.44,                    # 组间行通道
    "edge_aisle_x": 0.50,                   # 场地东西边缘通道
    "berth_depth_y": 0.32,                  # 南北泊位带深度
    "stand_top_z": 0.095,                   # 交接台条带顶面 (托盘落位面)
    "seed_pallets": 48,                     # 播种托盘数 (384 的 ~12.5%)
    "handoff_ids": ["H01", "H02", "H03", "H04", "H05"],
    # 交接台 x: 五条列通道中心 (西/东边缘 + 三条组间), 保证叉车/AGV 沿通道直线交接
    "handoff_x": [-2.22, -1.10, 0.0, 1.10, 2.22],
    # 停车点 (与地面白框联动): F01 在横移带最西端 (西端死区, 无过境交通;
    # W1 链式计划的回程腿 y=-2.42 语义不变); F02 在南带外排东端 (y=-2.85
    # 行上 x∈(-2.22,-1.10) 是 W1 plan_agv_dock_lift(A01,H02) 的盲走走廊,
    # 不可停车 —— 东端 2.60 为死区); F03/F04 居中间隙; AGV 对齐列通道。
    # 各车静态间距 ≥0.33 (包络不重叠)。
    "park": {"F01": (-2.60, -2.42, 0.0), "F02": (2.60, -2.85, 0.0),
             "F03": (0.50, -2.85, 0.0), "F04": (1.60, -2.85, 0.0),
             "A01": (-2.22, -2.85, 0.0), "A02": (0.0, -2.85, 0.0),
             "A03": (1.10, -2.85, 0.0), "A04": (2.22, -2.85, 0.0)},
    # 异构等级: L1 轻 (快/举升快/载重小) → L3 重 (慢/载重大)
    "vehicle_class": {"F01": "L1", "F02": "L2", "F03": "L3", "F04": "L3",
                      "A01": "L1", "A02": "L1", "A03": "L2", "A04": "L3"},
}

# 车辆等级参数 (W2 规格): 载重上限 = 托盘+货 (kg); 行车速度/加速度 (m/s, m/s²);
# 举升/顶升速度 (m/s); 原地转向 (rad/s, rad/s²); 横移 (strafe, m/s, 仅导航微调用)。
# 精度段速度 (插叉/对中/落托) 仍取 K 常数, 不随等级缩放。
VEHICLE_CLASSES = {
    "L1": dict(load_max_kg=0.08, cruise=0.35, accel=0.50, lift=0.015,
               yaw_speed=0.6, yaw_accel=1.0, strafe=0.12),
    "L2": dict(load_max_kg=0.15, cruise=0.25, accel=0.35, lift=0.010,
               yaw_speed=0.45, yaw_accel=0.8, strafe=0.10),
    "L3": dict(load_max_kg=0.30, cruise=0.15, accel=0.25, lift=0.007,
               yaw_speed=0.30, yaw_accel=0.5, strafe=0.08),
}

# 车辆包络 (通道净宽校核用): 叉车全长 0.33 (配重后缘→叉尖) / 宽 0.15; AGV 直径 0.24
FORKLIFT_ENVELOPE = (0.33, 0.15)
AGV_ENVELOPE = (0.24, 0.24)
SAFETY_MARGIN = 0.05          # 单侧安全余量 (规格: 包络 + 2× 余量)

_SLOT_RE = re.compile(r"R([1-9])([1-9])-S([0-7])(?:-L([0-2]))?$")


# ---------------------------------------------------------------- 节点注册表
#: 场景节点源 (L3 场景载荷经此注册进布局/调度索引, 不硬编码进主路径):
#: fn(MatrixLayout) -> {node_id: (x, y, z)}。node_id 全局唯一 —— 重名在
#: extra_nodes() 求值时抛 ValueError (内核节点名同样不可占用)。
_NODE_SOURCES: list = []


def register_node_source(fn):
    """注册场景节点源; 返回 fn 本身 (可作装饰器)。

    契约: fn 纯函数 (同 layout 同结果, 不持引用); 语义上是「节点类」
    的一段派生规则 —— 如 dc-twin 的 tk:* 卡车位 (D_G 延伸段)。
    """
    _NODE_SOURCES.append(fn)
    return fn


def parse_slot(slot_id: str) -> tuple[int, int, int, int]:
    """'R23-S5-L1' → (r=2, c=3, k=5, level=1); 缺省层 = L0。"""
    m = _SLOT_RE.fullmatch(slot_id)
    if not m:
        raise ValueError(f"货位 id 非法: {slot_id!r} (形如 R23-S5-L1)")
    return int(m[1]), int(m[2]), int(m[3]), int(m[4] or 0)


class MatrixLayout:
    """布局派生量计算 —— 世界坐标的唯一权威 (XML 与标志线全部经此生成)。"""

    def __init__(self, **overrides):
        p = dict(LAYOUT)
        p.update(overrides)
        self.p = p
        self.group_w = p["bays_per_row"] * p["bay_pitch_x"]
        self.group_h = 2 * p["rack_depth"] + p["inner_aisle_y"]
        self.matrix_w = p["group_cols"] * self.group_w + \
            (p["group_cols"] - 1) * p["col_aisle_x"]
        self.matrix_h = p["group_rows"] * self.group_h + \
            (p["group_rows"] - 1) * p["row_aisle_y"]
        self.handoff_y = -(self.matrix_h / 2 + p["berth_depth_y"] / 2)
        self.dock_y = +(self.matrix_h / 2 + p["berth_depth_y"] / 2)
        self.field_w = self.matrix_w + 2 * p["edge_aisle_x"]
        self.field_h = self.matrix_h + 2 * p["berth_depth_y"]
        # 通道净宽校核 (规格红线: 通道 ≥ 车辆包络 + 2× 安全余量)
        need_f = max(FORKLIFT_ENVELOPE) + 2 * SAFETY_MARGIN      # 0.43
        need_a = max(AGV_ENVELOPE) + 2 * SAFETY_MARGIN           # 0.34
        for key, need in (("col_aisle_x", need_f), ("row_aisle_y", need_f),
                          ("inner_aisle_y", need_a)):
            if p[key] + 1e-9 < need:
                raise ValueError(
                    f"通道 {key}={p[key]} 不足车辆包络+2×余量 ({need:.2f})")
        if len(p["handoff_x"]) != len(p["handoff_ids"]):
            raise ValueError("handoff_x 与 handoff_ids 数量不一致")

    # ---- 组/货位 ----
    def group_center(self, r: int, c: int) -> tuple[float, float]:
        """组 G<r><c> 中心 (r=1 在南)。"""
        x0 = -self.matrix_w / 2 + self.group_w / 2
        y0 = -self.matrix_h / 2 + self.group_h / 2
        return (x0 + (c - 1) * (self.group_w + self.p["col_aisle_x"]),
                y0 + (r - 1) * (self.group_h + self.p["row_aisle_y"]))

    def slot_ids(self) -> list[str]:
        out = []
        for r in range(1, self.p["group_rows"] + 1):
            for c in range(1, self.p["group_cols"] + 1):
                for k in range(8):
                    for l in range(len(self.p["levels_z"])):
                        out.append(f"R{r}{c}-S{k}-L{l}")
        return out

    def slot_pose(self, slot_id: str) -> tuple[float, float, float, float]:
        """货位 (架中心 x, 架中心 y, 托盘 yaw, 层支撑面 z)。

        托盘通道轴 = 世界 ±y (yaw +90°); z = 该层横梁顶面。
        """
        r, c, k, l = parse_slot(slot_id)
        cx, cy = self.group_center(r, c)
        row_off = self.group_h / 2 - self.p["rack_depth"] / 2      # 0.26
        y = cy - row_off if k < 4 else cy + row_off                # 前/后排
        x = cx + (k % 4 - 1.5) * self.p["bay_pitch_x"]
        return x, y, math.pi / 2, self.p["levels_z"][l]

    # ---- 通道 ----
    def col_aisle_xs(self) -> list[float]:
        """列通道中心 x, 西→东 (含两条边缘通道)。"""
        x0 = -self.matrix_w / 2
        inner = [x0 + (i + 1) * (self.group_w + self.p["col_aisle_x"])
                 - self.p["col_aisle_x"] / 2
                 for i in range(self.p["group_cols"] - 1)]
        edge = self.matrix_w / 2 + self.p["edge_aisle_x"] / 2
        return sorted([-edge] + inner + [edge])

    def row_aisle_ys(self) -> list[float]:
        """行通道中心 y, 南→北 (含两条边缘通道)。"""
        y0 = -self.matrix_h / 2
        inner = [y0 + (i + 1) * (self.group_h + self.p["row_aisle_y"])
                 - self.p["row_aisle_y"] / 2
                 for i in range(self.p["group_rows"] - 1)]
        edge = self.matrix_h / 2 + self.p["row_aisle_y"] / 2
        return sorted([-edge] + inner + [edge])

    # ---- 交接台 ----
    def handoff_poses(self) -> dict[str, tuple[float, float, float]]:
        return {sid: (x, self.handoff_y, math.pi / 2)
                for sid, x in zip(self.p["handoff_ids"], self.p["handoff_x"])}

    def dock_poses(self) -> dict[str, tuple[float, float, float]]:
        """交付台 D_G<r><c>: 目标组的巷道口, 每组独占一台 (航向 0°)。

        r=1..3: 该组北侧行通道内, 但**南移 0.085m** —— 叉车车体几何中心
        在原点后方 0.085 (底盘盒局部 x=-0.085), 沿通道中心线行驶时车体
        恰好居中穿越台位墙隙 (±22mm); AGV 圆盘对称于原点, 沿台位中心线
        (即 dock_y) 行驶穿越 (±12mm, 同 H 台)。两车各走各的既定车道,
        互不冲突。r=4: 北泊位带 (y=+2.10, 仅 AGV 通行, 不偏移)。
        台位开口沿世界 ±x —— AGV 在 H 台接管 (通道轴 +y) 后原地转 90°,
        沿行通道东行对中钻入 (yaw 执行器由此获得真实用途)。
        """
        aisles = self.row_aisle_ys()          # [-2.16,-1.08,0,1.08,2.16]
        out = {}
        for r in range(1, self.p["group_rows"] + 1):
            for c in range(1, self.p["group_cols"] + 1):
                cx, _ = self.group_center(r, c)
                if r == self.p["group_rows"]:
                    y = self.dock_y
                else:
                    y = aisles[r] - 0.085     # 叉车车体中心补偿
                out[f"D_G{r}{c}"] = (cx, y, 0.0)
        return out

    def stand_pose(self, sid: str) -> tuple[float, float, float]:
        if sid in self.p["handoff_ids"]:
            return self.handoff_poses()[sid]
        if sid in self.dock_poses():
            return self.dock_poses()[sid]
        raise KeyError(f"未知交接台 {sid!r}; 可用: 交接台 {self.p['handoff_ids']}"
                       f" / 交付台 D_G<r><c>")

    def extra_nodes(self) -> dict[str, tuple[float, float, float]]:
        """合并全部注册节点源的命名节点 (布局可扩展点; 内核节点名保留)。

        重名 (源间互撞或撞内核 H/D/货位名) 即抛 ValueError —— 注册表
        求值时统一拦截, 主路径消费方无需感知来源数量。
        """
        out: dict[str, tuple[float, float, float]] = {}
        reserved = {**self.handoff_poses(), **self.dock_poses()}
        for src in _NODE_SOURCES:
            for name, p in dict(src(self)).items():
                if name in out or name in reserved or _SLOT_RE.fullmatch(name):
                    raise ValueError(f"注册节点重名/占用内核名: {name!r}")
                out[name] = (float(p[0]), float(p[1]), float(p[2]))
        return out


# ---------------------------------------------------------------- 车辆视图
class _VehView:
    """单车轻量视图: 执行器地址 + 关节地址 + 世界→关节换算偏移。"""

    __slots__ = ("cell", "vid", "a", "off", "qadr")

    def __init__(self, cell: "WarehouseCell", vid: str,
                 a: dict[str, int], off: dict[str, float],
                 qadr: dict[str, int]):
        self.cell, self.vid, self.a, self.off, self.qadr = \
            cell, vid, a, off, qadr

    def q(self, axis: str) -> float:
        return float(self.cell.data.qpos[self.qadr[axis]])

    def seg(self, axis: str, target: float, speed: float,
            accel: float = 0.30):
        """一段轨迹规格 (执行器id, 起点延迟构造的 Ramp 工厂, 关节偏移)。"""
        cell, aid, off = self.cell, self.a[axis], self.off[axis]

        def build():
            return Ramp(self.q(axis) + off, target, speed, accel)
        return (aid, build, off)

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.vid} "
                f"x={self.q('x') + self.off['x']:+.3f} "
                f"y={self.q('y') + self.off['y']:+.3f}>")


class ForkliftView(_VehView):
    """叉车: x/y/yaw/lift 四通道。fork_top = 0.065 + lift_q (同 forklift_cell)。"""

    __slots__ = ("fork_gids",)

    def __init__(self, *args, fork_gids: set[int]):
        super().__init__(*args)
        self.fork_gids = fork_gids

    @property
    def fork_top_z(self) -> float:
        return 0.065 + self.q("lift")

    def xy(self) -> tuple[float, float]:
        return self.q("x") + self.off["x"], self.q("y") + self.off["y"]


class AgvView(_VehView):
    """AGV: x/y/yaw/pin 四通道 (圆盘轴对称, yaw 仅朝向演示)。"""

    __slots__ = ("pin_gid",)

    def __init__(self, *args, pin_gid: int):
        super().__init__(*args)
        self.pin_gid = pin_gid

    def xy(self) -> tuple[float, float]:
        return self.q("x") + self.off["x"], self.q("y") + self.off["y"]


class PalletView:
    __slots__ = ("name", "bid", "gids", "mass")

    def __init__(self, name: str, bid: int, gids: list[int], mass: float):
        self.name, self.bid, self.gids, self.mass = name, bid, gids, mass


# ---------------------------------------------------------------- 计划常数
# 与 forklift_cell 同源的插叉几何 (fork/pallet 来自 configs/design.yaml)
K = dict(
    transport_z=0.065, fork_root_x=0.040, fork_len=0.080,
    insert_depth=0.065, approach_gap=0.055, pallet_half_x=0.040,
    pallet_center_off=0.094,          # 托盘中心相对前轴 (实测值, 同 T01)
    transit_fork_top=0.1045,          # 空叉回穿台区高度: 叉底 0.102 > 台上托盘
                                      # 轨顶 0.099 (轨条带 0.095-0.099), 叉顶
                                      # 0.1045 < 叉孔顶板底 0.107 (2.5mm 窗)
    # 速度剖面
    fk_cruise=0.30, fk_approach=0.05, fk_insert=0.010,
    lift_slow=0.002, lift=0.010, place=0.005,
    yaw_speed=0.6, yaw_accel=1.0,
    agv_cruise=0.35, agv_dock=0.030, agv_pin=0.008, pin_travel=0.030,
)


def pallet_name_for(slot_id: str) -> str:
    r, c, k, l = parse_slot(slot_id)
    return f"pallet_R{r}{c}_S{k}_L{l}"


class WarehouseCell:
    """128 架 × 3 层矩阵仓库: 384 货位 + 交接台 + 混合车队 + 地面标志线。"""

    def __init__(self, timestep_s: float = 0.001, seed: int = 0,
                 layout: MatrixLayout | None = None,
                 forklifts: tuple[str, ...] = ("F01", "F02", "F03", "F04"),
                 agvs: tuple[str, ...] = ("A01", "A02", "A03", "A04"),
                 seed_slots: list[str] | None = None,
                 preload_stands: tuple[str, ...] = ("H01",)):
        self.layout = layout or MatrixLayout()
        self.vehicles = [str(v) for v in (*forklifts, *agvs)]
        for v in self.vehicles:
            if not re.fullmatch(r"[A-Za-z0-9_]+", v):
                raise ValueError(f"车辆 id 只允许 [A-Za-z0-9_]: {v!r}")
        if len(set(self.vehicles)) != len(self.vehicles):
            raise ValueError(f"车辆 id 重复: {self.vehicles}")
        self.forklift_ids = [str(v) for v in forklifts]
        self.agv_ids = [str(v) for v in agvs]
        self.cfg = load()
        self.seed = seed
        self._seed_slots = seed_slots
        self._preload_stands = tuple(preload_stands)
        self.model = mujoco.MjModel.from_xml_string(self._xml(timestep_s))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._cache_ids()

    # ---------- id 缓存 ----------
    def _cache_ids(self):
        m, T = self.model, mujoco.mjtObj

        def _id(kind, name):
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise KeyError(f"模型中找不到 {name!r}")
            return i

        self.forklifts: dict[str, ForkliftView] = {}
        for vid in self.forklift_ids:
            p = f"{vid}_"
            a = {k: _id(T.mjOBJ_ACTUATOR, f"{p}drive_{k}")
                 for k in ("x", "y", "yaw")}
            a["lift"] = _id(T.mjOBJ_ACTUATOR, f"{p}lift")
            qadr = {"x": m.jnt_qposadr[_id(T.mjOBJ_JOINT, f"{p}chassis_x")],
                    "y": m.jnt_qposadr[_id(T.mjOBJ_JOINT, f"{p}chassis_y")],
                    "yaw": m.jnt_qposadr[_id(T.mjOBJ_JOINT, f"{p}chassis_yaw")],
                    "lift": m.jnt_qposadr[_id(T.mjOBJ_JOINT, f"{p}lift")]}
            px, py, _ = self.layout.p["park"][vid]
            off = {"x": px, "y": py, "yaw": 0.0, "lift": 0.0}
            fg = {_id(T.mjOBJ_GEOM, f"{p}fork_l"),
                  _id(T.mjOBJ_GEOM, f"{p}fork_r")}
            self.forklifts[vid] = ForkliftView(
                self, vid, a, off, qadr, fork_gids=fg)
        self.agvs: dict[str, AgvView] = {}
        for vid in self.agv_ids:
            p = f"{vid}_"
            a = {k: _id(T.mjOBJ_ACTUATOR, f"{p}drive_{k}")
                 for k in ("x", "y", "yaw")}
            a["pin"] = _id(T.mjOBJ_ACTUATOR, f"{p}pin")
            qadr = {"x": m.jnt_qposadr[_id(T.mjOBJ_JOINT, f"{p}chassis_x")],
                    "y": m.jnt_qposadr[_id(T.mjOBJ_JOINT, f"{p}chassis_y")],
                    "yaw": m.jnt_qposadr[_id(T.mjOBJ_JOINT, f"{p}chassis_yaw")],
                    "pin": m.jnt_qposadr[_id(T.mjOBJ_JOINT, f"{p}pin_lift")]}
            px, py, _ = self.layout.p["park"][vid]
            off = {"x": px, "y": py, "yaw": 0.0, "pin": 0.0}
            self.agvs[vid] = AgvView(self, vid, a, off, qadr,
                                     pin_gid=_id(T.mjOBJ_GEOM,
                                                 f"{p}pin_plate"))
        self.veh = {**self.forklifts, **self.agvs}
        # 托盘
        self.pallets: dict[str, PalletView] = {}
        for name in self._pallet_names():
            bid = _id(T.mjOBJ_BODY, name)
            gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
            self.pallets[name] = PalletView(
                name, bid, gids, float(m.body_mass[bid]))
        # 交接台条带 (托盘落位支撑面)
        self.stand_gids = {
            sid: {_id(T.mjOBJ_GEOM, f"stand_{sid}_strip_{s}{y}")
                  for s in ("p", "n") for y in ("p", "n")}
            for sid in (*self.layout.p["handoff_ids"],
                        *self.layout.dock_poses())}
        # 地面标志线 geom 名清单 (渲染可见性 + 参数联动校验用)
        self.mark_names = sorted(
            n for n in (mujoco.mj_id2name(m, T.mjOBJ_GEOM, g)
                        for g in range(m.ngeom))
            if n and n.startswith("mark_"))
        self.floor_gid = _id(T.mjOBJ_GEOM, "floor")

    def _pallet_names(self) -> list[str]:
        """播种清单: 交货位固定 R11-S0-L2 (交接链测试) + 随机分散 + 台上预载。"""
        L = self.layout
        n_total = L.p["seed_pallets"]
        if self._seed_slots is not None:
            slots = [s for s in self._seed_slots if s != "R11-S0-L2"]
        else:
            rng = np.random.default_rng(self.seed)
            all_slots = [s for s in L.slot_ids() if s != "R11-S0-L2"]
            n_rack = n_total - len(self._preload_stands) - 1
            slots = [all_slots[i] for i in
                     sorted(rng.choice(len(all_slots), size=n_rack,
                                       replace=False))]
        slots.append("R11-S0-L2")
        names = [pallet_name_for(s) for s in slots]
        names += [f"pallet_{sid}" for sid in self._preload_stands]
        return names

    # ---------- XML 生成 ----------
    def _pallet_xml(self, name: str, x: float, y: float, z: float,
                    yaw: float) -> str:
        """托盘自由体: forklift_cell 白模 + 底面中央凸台 (顶销承接面)。"""
        q = f"{math.cos(yaw / 2):.7f} 0 0 {math.sin(yaw / 2):.7f}"
        return f"""
    <body name="{name}" pos="{x:.6f} {y:.6f} {z + 0.001:.6f}" quat="{q}">
      <freejoint name="free_{name}"/>
      <geom name="{name}_bottom_l" type="box" pos="0 -0.027 0.002" size="0.040 0.003 0.002" mass="0.006" rgba="0.55 0.40 0.22 1"/>
      <geom name="{name}_bottom_r" type="box" pos="0 0.027 0.002" size="0.040 0.003 0.002" mass="0.006" rgba="0.55 0.40 0.22 1"/>
      <geom name="{name}_bottom_c" type="box" pos="0 0 0.002" size="0.040 0.006 0.002" mass="0.010" rgba="0.55 0.40 0.22 1"/>
      <geom name="{name}_ceil_l" type="box" pos="0 -0.015 0.0135" size="0.040 0.007 0.0015" mass="0.008" friction="1.4 0.005 0.0001" rgba="0.72 0.55 0.32 1"/>
      <geom name="{name}_ceil_r" type="box" pos="0 0.015 0.0135" size="0.040 0.007 0.0015" mass="0.008" friction="1.4 0.005 0.0001" rgba="0.72 0.55 0.32 1"/>
      <geom name="{name}_top_side_l" type="box" pos="0 -0.027 0.0175" size="0.040 0.003 0.0025" mass="0.002" rgba="0.78 0.62 0.38 1"/>
      <geom name="{name}_top_side_r" type="box" pos="0 0.027 0.0175" size="0.040 0.003 0.0025" mass="0.002" rgba="0.78 0.62 0.38 1"/>
      <geom name="{name}_top_mid" type="box" pos="0 0 0.017" size="0.040 0.008 0.001" mass="0.002" rgba="0.78 0.62 0.38 1"/>
      <!-- AGV 顶销承接凸台: 垂入台面条带中孔 / 货架双梁之间 (层间距 0.15 > 0.03) -->
      <geom name="{name}_boss" type="box" pos="0 0 -0.015" size="0.020 0.020 0.015" mass="0.004" rgba="0.45 0.32 0.16 1"/>
    </body>"""

    def _rack_bay_xml(self, gid: str, x: float, y: float) -> str:
        """单架 3 层: 每层两根横梁 (沿 y, 托盘条落在梁上) + 通高视觉立柱。
        无中梁给凸台让位; 层高取 levels_z (同 forklift_cell levels 机制)。"""
        ups = []
        for z in self.layout.p["levels_z"]:
            ups.append(
                f'      <geom name="rack_{gid}_beam_w_L{z:.3f}" type="box" pos="-0.027 0 {z - 0.0025:.4f}" size="0.003 0.055 0.0025" rgba="0.80 0.42 0.10 1"/>')
            ups.append(
                f'      <geom name="rack_{gid}_beam_e_L{z:.3f}" type="box" pos="0.027 0 {z - 0.0025:.4f}" size="0.003 0.055 0.0025" rgba="0.80 0.42 0.10 1"/>')
        return f"""
    <body name="rack_{gid}" pos="{x:.6f} {y:.6f} 0">
{chr(10).join(ups)}
      <geom name="rack_{gid}_up_w" type="box" pos="-0.072 0 0.225" size="0.004 0.058 0.225" mass="0" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
      <geom name="rack_{gid}_up_e" type="box" pos="0.072 0 0.225" size="0.004 0.058 0.225" mass="0" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
    </body>"""

    def _stand_xml(self, sid: str, x: float, y: float,
                   heading: float, top: float) -> str:
        """交接/交付台: 4 条带 (托盘条支撑面, 中开孔) + 两侧导向墙。
        局部 +x = 交接轴 (车行方向); 墙在局部 y=±0.140, 叉车宽 0.15 /
        AGV 直径 0.24 均可穿行, 台下净空 0.091 > AGV 高 0.062。
        H 台 (南带) 墙实体碰撞; D 台 (行通道线上, 邻内圈槽位走廊) 墙为
        视觉导向 (contype=0): 行通道线 x 轴已被槽位走廊 (±0.075) 与列
        走廊饱和, 实体墙使内圈槽位 (组心 ±0.08) 不可达 —— 防撞职能由
        锁矩阵承担 (M2W2: dock 节点互斥 + 内圈 stub 追加邻 dock 节点)。
        """
        q = f"{math.cos(heading / 2):.7f} 0 0 {math.sin(heading / 2):.7f}"
        g = f"stand_{sid}"
        solid = "" if sid.startswith("H") else ' contype="0" conaffinity="0"'
        # 条带垫块 x 内移 (±[0.022,0.044], 座面 18mm > W1 的 10mm): 让出
        # 组心 ±0.08 内圈槽位的叉车 carriage 抬升走廊 (板达面位 ±0.022,
        # 原 ±0.090 会顶死垫块) —— pin 孔 y 向 48mm 不变, x 向 44mm。
        return f"""
    <body name="{g}" pos="{x:.6f} {y:.6f} 0" quat="{q}">
      <geom name="{g}_strip_pp" type="box" pos="0.033 0.047 {top - 0.002:.4f}" size="0.011 0.023 0.002" rgba="0.32 0.46 0.52 1"/>
      <geom name="{g}_strip_pn" type="box" pos="0.033 -0.047 {top - 0.002:.4f}" size="0.011 0.023 0.002" rgba="0.32 0.46 0.52 1"/>
      <geom name="{g}_strip_np" type="box" pos="-0.033 0.047 {top - 0.002:.4f}" size="0.011 0.023 0.002" rgba="0.32 0.46 0.52 1"/>
      <geom name="{g}_strip_nn" type="box" pos="-0.033 -0.047 {top - 0.002:.4f}" size="0.011 0.023 0.002" rgba="0.32 0.46 0.52 1"/>
      <geom name="{g}_wall_p" type="box" pos="0 0.140 {top / 2:.4f}" size="0.090 0.008 {top / 2:.4f}" rgba="0.22 0.32 0.38 1"{solid}/>
      <geom name="{g}_wall_n" type="box" pos="0 -0.140 {top / 2:.4f}" size="0.090 0.008 {top / 2:.4f}" rgba="0.22 0.32 0.38 1"{solid}/>
    </body>"""

    def _forklift_xml(self, vid: str) -> str:
        """叉车可动块: x+y 滑动 + yaw 铰 (2-DOF 平面底盘) + 举升滑轨 + 双叉。
        叉/门架几何沿用 forklift_cell 设计值 (运动学不变, 插叉物理同源)。"""
        p = f"{vid}_"
        fx = K["fork_root_x"] + K["fork_len"] / 2
        px, py, _ = self.layout.p["park"][vid]
        return f"""
    <body name="{p}chassis" pos="{px:.6f} {py:.6f} 0">
      <joint name="{p}chassis_x" type="slide" axis="1 0 0" range="-5.2 5.2" damping="20"/>
      <joint name="{p}chassis_y" type="slide" axis="0 1 0" range="-5.2 5.2" damping="20"/>
      <!-- 原地任意转向: 不设 range (与旧台架 ±0.2 的差别即 2-DOF 升级点) -->
      <joint name="{p}chassis_yaw" type="hinge" axis="0 0 1" damping="5"/>
      <geom name="{p}chassis_body" type="box" pos="-0.085 0 0.0175" size="0.110 0.075 0.0175" mass="1.50" rgba="0.90 0.66 0.05 1"/>
      <geom name="{p}counterweight" type="box" pos="-0.20 0 0.0175" size="0.010 0.060 0.0175" mass="0.0" rgba="0.18 0.18 0.20 1"/>
      <geom name="{p}mast_l" type="box" pos="0.005 -0.045 0.25" size="0.006 0.006 0.25" mass="0" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
      <geom name="{p}mast_r" type="box" pos="0.005 0.045 0.25" size="0.006 0.006 0.25" mass="0" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
      <geom name="{p}mast_top" type="box" pos="0.005 0 0.497" size="0.008 0.051 0.006" mass="0" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
      <body name="{p}carriage" pos="0 0 0">
        <joint name="{p}lift" type="slide" axis="0 0 1" range="-0.050 0.385" damping="60"/>
        <geom name="{p}carriage_plate_l" type="box" pos="0.020 -0.015 0.068" size="0.010 0.007 0.008" mass="0.125" rgba="0.35 0.36 0.40 1"/>
        <geom name="{p}carriage_plate_r" type="box" pos="0.020 0.015 0.068" size="0.010 0.007 0.008" mass="0.125" rgba="0.35 0.36 0.40 1"/>
        <geom name="{p}fork_l" type="box" pos="{fx:.4f} -0.015 {K['transport_z'] - 0.00125:.5f}" size="{K['fork_len'] / 2:.4f} 0.005 0.00125" mass="0.02" friction="1.4 0.02 0.0002" rgba="0.15 0.15 0.17 1"/>
        <geom name="{p}fork_r" type="box" pos="{fx:.4f} 0.015 {K['transport_z'] - 0.00125:.5f}" size="{K['fork_len'] / 2:.4f} 0.005 0.00125" mass="0.02" friction="1.4 0.02 0.0002" rgba="0.15 0.15 0.17 1"/>
      </body>
    </body>"""

    def _agv_xml(self, vid: str) -> str:
        """潜伏顶升 AGV: 圆盘底盘 (r0.12 h0.06) + 顶销盘 (行程 0.030)。
        底盘底面 z=0.002 悬浮 (平面滑动导轨运动学, 无地面接触拖拽)。"""
        p = f"{vid}_"
        px, py, _ = self.layout.p["park"][vid]
        return f"""
    <body name="{p}chassis" pos="{px:.6f} {py:.6f} 0">
      <joint name="{p}chassis_x" type="slide" axis="1 0 0" range="-5.2 5.2" damping="20"/>
      <joint name="{p}chassis_y" type="slide" axis="0 1 0" range="-5.2 5.2" damping="20"/>
      <joint name="{p}chassis_yaw" type="hinge" axis="0 0 1" damping="5"/>
      <geom name="{p}base" type="cylinder" pos="0 0 0.032" size="0.120 0.030" mass="2.0" friction="0.1 0.005 0.0001" rgba="0.15 0.35 0.60 1"/>
      <geom name="{p}stripe" type="box" pos="0.095 0 0.032" size="0.020 0.004 0.0305" mass="0" rgba="0.90 0.75 0.10 1" contype="0" conaffinity="0"/>
      <body name="{p}pin" pos="0 0 0.060">
        <joint name="{p}pin_lift" type="slide" axis="0 0 1" range="0 {K['pin_travel']}" damping="30"/>
        <!-- 销盘 r=0.018: 托盘 80×60mm, 销盘必须钻入其投影内 (名义 0.08 物理不可行) -->
        <geom name="{p}pin_plate" type="cylinder" pos="0 0 0" size="0.018 0.002" mass="0.05" friction="0.8 0.005 0.0001" rgba="0.75 0.76 0.80 1"/>
        <geom name="{p}pin_stem" type="cylinder" pos="0 0 -0.016" size="0.006 0.014" mass="0" rgba="0.45 0.46 0.50 1" contype="0" conaffinity="0"/>
      </body>
    </body>"""

    def _actuators_xml(self) -> str:
        out = []
        for vid in self.forklift_ids:
            p = f"{vid}_"
            out += [
                f'    <position name="{p}drive_x" joint="{p}chassis_x" kp="800" forcerange="-20 20"/>',
                f'    <position name="{p}drive_y" joint="{p}chassis_y" kp="800" forcerange="-20 20"/>',
                f'    <position name="{p}drive_yaw" joint="{p}chassis_yaw" kp="150" forcerange="-20 20"/>',
                f'    <position name="{p}lift" joint="{p}lift" kp="10000" forcerange="-5 5"/>',
            ]
        for vid in self.agv_ids:
            p = f"{vid}_"
            out += [
                f'    <position name="{p}drive_x" joint="{p}chassis_x" kp="500" forcerange="-30 30"/>',
                f'    <position name="{p}drive_y" joint="{p}chassis_y" kp="500" forcerange="-30 30"/>',
                f'    <position name="{p}drive_yaw" joint="{p}chassis_yaw" kp="60" forcerange="-8 8"/>',
                f'    <position name="{p}pin" joint="{p}pin_lift" kp="2000" forcerange="-10 10"/>',
            ]
        return "\n".join(out)

    # ---- 地面标志线 (全部程序化, 布局参数联动) ----
    YEL, WHT, BLU = "0.85 0.68 0.05 1", "0.92 0.92 0.95 1", "0.15 0.35 0.85 1"

    def _dash_line(self, name: str, fixed: float, lo: float, hi: float,
                   along_y: bool, rgba: str) -> str:
        """中心虚线: 段长 0.08 间隔 0.06, z≈0 (contype=0 纯视觉地贴)。"""
        pitch, half = 0.14, 0.040
        out = []
        j = 0
        c = lo + half
        while c <= hi:
            pos = (fixed, c, 0.0004) if along_y else (c, fixed, 0.0004)
            size = "0.005 0.040 0.0004" if along_y else "0.040 0.005 0.0004"
            out.append(f'    <geom name="{name}_d{j:03d}" type="box" '
                       f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
                       f'size="{size}" rgba="{rgba}" contype="0" conaffinity="0"/>')
            j += 1
            c = lo + half + j * pitch
        return "\n".join(out)

    def _rect_outline(self, name: str, cx: float, cy: float,
                      hx: float, hy: float, rgba: str) -> str:
        """矩形轮廓线 (4 条细带, z≈0 纯视觉)。"""
        t = 0.004
        return "\n".join(
            f'    <geom name="{name}_{side}" type="box" pos="{x:.6f} {y:.6f} 0.0004" '
            f'size="{sx} {sy} 0.0004" rgba="{rgba}" contype="0" conaffinity="0"/>'
            for side, (x, y, sx, sy) in {
                "n": (cx, cy + hy, hx + t, t), "s": (cx, cy - hy, hx + t, t),
                "e": (cx + hx, cy, t, hy + t), "w": (cx - hx, cy, t, hy + t),
            }.items())

    def _markings_xml(self) -> str:
        L = self.layout
        out = ["    <!-- 地面标志线: 布局参数程序化生成 (改参数全场联动) -->"]
        yspan = L.matrix_h / 2 + L.p["berth_depth_y"] + 0.10
        xspan = L.matrix_w / 2 + L.p["edge_aisle_x"] + 0.05
        for i, ax in enumerate(L.col_aisle_xs()):       # 列通道中心黄虚线 (沿 y)
            out.append(self._dash_line(f"mark_aisle_col_{i}", ax,
                                       -yspan, yspan, True, self.YEL))
        for i, ay in enumerate(L.row_aisle_ys()):       # 行通道中心黄虚线 (沿 x)
            out.append(self._dash_line(f"mark_aisle_row_{i}", ay,
                                       -xspan, xspan, False, self.YEL))
        for r in range(1, L.p["group_rows"] + 1):       # 货架组轮廓白线
            for c in range(1, L.p["group_cols"] + 1):
                cx, cy = L.group_center(r, c)
                out.append(self._rect_outline(
                    f"mark_group_G{r}{c}", cx, cy,
                    L.group_w / 2, L.group_h / 2, self.WHT))
        for sid, (sx, sy, _) in {**L.handoff_poses(),
                                 **L.dock_poses()}.items():   # 交接台区蓝框
            zone = "handoff" if sid.startswith("H") else "dock"
            out.append(self._rect_outline(
                f"mark_{zone}_{sid}", sx, sy, 0.17, 0.22, self.BLU))
        for vid, (px, py, _) in L.p["park"].items():    # 停车点白框
            out.append(self._rect_outline(
                f"mark_park_{vid}", px, py, 0.18, 0.14, self.WHT))
        return "\n".join(out)

    def _xml(self, dt: float) -> str:
        L = self.layout
        top = L.p["stand_top_z"]
        # --- 货架 (16 组 × 8 架 × 3 层) ---
        racks = []
        for r in range(1, L.p["group_rows"] + 1):
            for c in range(1, L.p["group_cols"] + 1):
                for k in range(8):
                    x, y, _, _ = L.slot_pose(f"R{r}{c}-S{k}-L0")
                    racks.append(self._rack_bay_xml(f"R{r}{c}_S{k}", x, y))
        # --- 播种托盘 (货架位 + 交接台预载) ---
        pallets = []
        for name in self._pallet_names():
            if name.startswith("pallet_R"):
                # pallet_R11_S0_L2 -> R11-S0-L2 (S/L 前的下划线即分隔)
                parts = name.split("_")          # ['pallet','R11','S0','L2']
                slot = f"{parts[1]}-{parts[2]}-{parts[3]}"
                x, y, yaw, z = L.slot_pose(slot)
                pallets.append(self._pallet_xml(name, x, y, z, yaw))
            else:                                 # pallet_H01 → 交接台面上
                sid = name[len("pallet_"):]
                x, y, hd = L.stand_pose(sid)
                pallets.append(self._pallet_xml(name, x, y, top, hd))
        # --- 交接台 ---
        stands = [self._stand_xml(sid, x, y, hd, top)
                  for sid, (x, y, hd) in {**L.handoff_poses(),
                                          **L.dock_poses()}.items()]
        return f"""
<mujoco model="M2W1_matrix_warehouse">
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 -9.81" iterations="200"
          tolerance="1e-10" noslip_iterations="20" noslip_tolerance="1e-8"/>
  <size nconmax="600" njmax="400"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.45 0.45 0.48" specular="0.1 0.1 0.1"/>
    <map fogstart="3.0" fogend="9.0"/>
    <rgba fog="0.06 0.07 0.09 1"/>
  </visual>
  <asset>
    <texture type="2d" name="floorgrid" builtin="checker"
             rgb1="0.30 0.32 0.36" rgb2="0.20 0.22 0.26" width="64" height="64"/>
    <material name="floormat" texture="floorgrid" texrepeat="60 60" reflectance="0.12"/>
  </asset>
  <default>
    <geom rgba="0.7 0.7 0.75 1" condim="3" friction="0.5 0.005 0.0001"
          solref="0.0002 1" solimp="0.999 0.9999 0.00001"/>
    <joint damping="0.1" armature="0.001"/>
  </default>
  <worldbody>
    <light pos="0 -3 3.5" dir="0 0.5 -1" diffuse="0.8 0.78 0.74" castshadow="true"/>
    <light pos="4 2 3.0" dir="-0.6 -0.3 -1" diffuse="0.35 0.37 0.42" castshadow="true"/>
    <light pos="-3 3 3.0" dir="0.5 -0.4 -1" diffuse="0.30 0.30 0.34" castshadow="false"/>
    <geom name="floor" type="plane" size="3.2 3.2 0.05" friction="0.5" material="floormat"/>
{self._markings_xml()}
{"".join(racks)}
{"".join(stands)}
{"".join(pallets)}
{"".join(self._forklift_xml(v) for v in self.forklift_ids)}
{"".join(self._agv_xml(v) for v in self.agv_ids)}
  </worldbody>
  <actuator>
{self._actuators_xml()}
  </actuator>
</mujoco>
"""

    # ---------- 状态 ----------
    def pallet_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        p = self.pallets[name]
        return (self.data.xpos[p.bid].copy(), self.data.xquat[p.bid].copy())

    def pallet_bottom_z(self, name: str) -> float:
        return float(self.pallet_pose(name)[0][2])

    def pallet_yaw(self, name: str) -> float:
        q = self.pallet_pose(name)[1]
        return 2.0 * math.atan2(float(q[2]), float(q[0]))

    def contacts(self) -> list[tuple[int, int, float]]:
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            out.append((c.geom1, c.geom2, float(c.dist)))
        return out

    def max_penetration(self) -> float:
        return max((-d for _, _, d in self.contacts()), default=0.0)

    def surface_load_ratio(self, pallet: str, gids: set[int]) -> float:
        """指定支撑面承担的竖向力 / 托盘重 (放置/载运判据, 同 forklift_cell)。"""
        pg = set(self.pallets[pallet].gids)
        total, f1 = 0.0, np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in gids and g2 in pg:
                sign = -1.0
            elif g2 in gids and g1 in pg:
                sign = 1.0
            else:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, f1)
            frame = c.frame.reshape(3, 3)
            total += max(0.0, sign * (frame @ f1[:3])[2])
        w = self.pallets[pallet].mass * 9.81
        return total / w if w else 0.0

    def pallet_touches(self, pallet: str, gids: set[int]) -> bool:
        pg = set(self.pallets[pallet].gids)
        return any((g1 in gids and g2 in pg) or (g2 in gids and g1 in pg)
                   for g1, g2, _ in self.contacts())

    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def warmup(self, seconds: float = 0.5):
        """初始沉降 (托盘落稳梁/条带); 控制目标已在关节零位, 无伺服窜动。"""
        self.step(int(seconds / self.model.opt.timestep))

    def forklifter(self, vid: str) -> ForkliftView:
        return self.forklifts[vid]

    def agv(self, vid: str) -> AgvView:
        return self.agvs[vid]

    def vclass(self, vid: str) -> dict:
        """车辆等级参数 (速度/加速度/举升/载重上限), 见 VEHICLE_CLASSES。"""
        return VEHICLE_CLASSES[self.layout.p["vehicle_class"][vid]]

    # ---------- 计划 (交接物理链) ----------
    def plan_forklift_rack_to_stand(self, vid: str, slot_id: str,
                                    stand_id: str):
        """叉车: 货位(任意层)取托 → 交接台放托 → 撤回行通道停车位。

        几何与 T01/T04 同源: 叉尖越过托盘近边 0.065 → 前轴距托盘中心 0.094;
        插叉/抬离高度 = 层支撑面 +0.010/+0.017 (levels 派生, 任意层通用)。
        """
        v, L = self.forklifter(vid), self.layout
        sx, sy, _, sz = L.slot_pose(slot_id)
        hx, hy, _ = L.stand_pose(stand_id)
        top = L.p["stand_top_z"]
        ins_z, lift_z = sz + 0.010, sz + 0.017
        place_z, rel_z = top + 0.012, top + 0.012 - 0.0005
        # y 向运动链 (heading +y): 托盘近边 = sy-0.040
        edge = sy - K["pallet_half_x"]
        app = edge - K["approach_gap"] - K["fork_len"] - K["fork_root_x"]
        x_ins = edge + K["insert_depth"] - K["fork_len"] - K["fork_root_x"]
        app2 = hy - K["pallet_half_x"] - K["approach_gap"] \
            - K["fork_len"] - K["fork_root_x"]
        place_axle = hy - K["pallet_center_off"]
        clear_y = hy - 0.32          # 交接台墙带以南 (墙 y ≥ hy-0.19)
        aisle_y = L.p["park"][vid][1]
        s = v.seg

        def slift(z_abs: float, speed: float):
            """绝对叉顶高度 → lift 关节量 (fork_top = 0.065 + q, 同 T01-T04)。"""
            return s("lift", z_abs - K["transport_z"], speed)

        # 路线要点 (与 T01 同构: 托盘必须位于车前, 从巷道侧接近):
        # - 车体**不得横穿货位线**: 滑轨背板 (z 0.405-0.421 @插叉高度) 会扫过
        #   L2 托盘顶板 (0.4175-0.420) 把托盘撞落 —— 先沿列通道南下到南带
        #   (纵向穿 H 台墙隙: 车宽 0.15 < 墙隙 0.264), 再南带西行, 从**巷道
        #   侧 (南)** 北上 approach, 托盘始终在车前。
        # - 台位墙隙仅容车体横向 ±57mm, 墙带 (y hy±0.19) 内禁止横向行驶。
        # - 交接台自南侧 approach (app2), 高位越台后原地降托。
        return [
            slift(ins_z, K["lift"]),                 # 1 升至插入高度 (层+10)
            s("yaw", math.pi / 2, K["yaw_speed"],
              K["yaw_accel"]),                       # 2 原地转向, 面向 +y
            s("y", clear_y, K["fk_cruise"]),         # 3 沿列通道南下穿墙隙
            s("x", sx, K["fk_cruise"]),              # 4 南带西行对准货位 x
            s("y", app, K["fk_cruise"]),             # 5 北上 approach (托盘在前)
            s("y", x_ins, K["fk_insert"]),           # 6 插叉 65mm
            slift(lift_z, K["lift_slow"]),           # 7 举升, 托盘离梁
            s("y", app, K["fk_approach"]),           # 8 带载南退离架
            slift(K["transport_z"], K["lift"]),      # 9 降至运输位
            s("y", clear_y, K["fk_cruise"]),         # 10 南下清空墙带
            s("x", hx, K["fk_cruise"]),              # 11 南带东行至交接台走廊
            # 12 高位越台 (在台区以南完成): 托盘轨顶须越过台面条带底 (0.091)
            # —— 若已伸入台下再举升, 顶轨条会顶死条带底面 (实测卡在 0.071);
            # 高位轨底 0.125 > 台面/墙顶 0.095。
            slift(place_z + 0.030, K["place"]),
            s("y", app2, K["fk_cruise"]),            # 13 高位北上穿隙至 approach
            s("y", place_axle, K["fk_insert"]),      # 14 高位进给: 中心→台中心
            slift(place_z, K["place"]),              # 15 降托落台 (凸台入孔)
            slift(rel_z, K["lift_slow"]),            # 16 微降脱叉
            s("y", clear_y, K["fk_approach"]),       # 17 空叉退出台位
            # 18 回程穿台区高度: 叉底 0.102 须越过台面托盘轨顶 (0.099)、
            # 叉顶 0.1045 须低于叉孔顶板底 (0.107) —— 原取 0.100 时叉体
            # 骑在轨条带内 (0.095-0.099), 北穿时中轨条把托盘拖走 (实测)。
            slift(K["transit_fork_top"], K["lift"]),
            s("y", aisle_y, K["fk_cruise"]),         # 19 回停车位 y (泊位带南缘)
            s("x", L.p["park"][vid][0], K["fk_cruise"]),  # 20 西回停车位 x
            slift(K["transport_z"], K["lift"]),      # 21 叉复位
        ]

    def plan_agv_dock_lift(self, vid: str, stand_id: str):
        """AGV: 对中钻入台底 (±10mm) → 顶销升 → 托盘离台。"""
        v, L = self.agv(vid), self.layout
        fx, fy, _ = L.stand_pose(stand_id)
        s = v.seg
        return [
            s("x", fx, K["agv_cruise"]),             # 台位对中 (泊位带外侧)
            s("y", fy - 0.25, K["agv_cruise"]),      # 接近
            s("y", fy, K["agv_dock"]),               # 对中钻入 (慢速终段)
            s("pin", K["pin_travel"], K["agv_pin"]),  # 顶销升, 托盘离台
        ]

    def plan_agv_transport(self, vid: str, from_stand: str, to_stand: str):
        """AGV 载运: 退场 → 原地转 90° (托盘通道轴 → 世界 +x) → 沿列通道
        北上 → 目标行通道东行 → 交付台对中钻入 (台位开口沿 +x)。"""
        v, L = self.agv(vid), self.layout
        fx, fy, _ = L.stand_pose(from_stand)
        tx, ty, _ = L.stand_pose(to_stand)
        band_s = fy - L.p["berth_depth_y"] + 0.03
        s = v.seg
        return [
            s("y", band_s, K["agv_cruise"]),         # 载运退出台位
            # 原地转 -90°: 托盘通道轴 +y → +x (对准交付台开口); 慢速防滑
            s("yaw", -math.pi / 2, 0.4, 0.5),
            s("y", ty, K["agv_cruise"]),             # 沿列通道北上至目标行
            s("x", tx - 0.25, K["agv_cruise"]),      # 东行至台位 approach
            s("x", tx, K["agv_dock"]),               # 对中钻入 (heading +x)
        ]

    def plan_agv_drop(self, vid: str, to_stand: str, corridor_x: float):
        """AGV: 降销放托 (托盘落台面条带) → 退场 → 经列通道回泊位带。"""
        v, L = self.agv(vid), self.layout
        tx, ty, _ = L.stand_pose(to_stand)
        fy = L.handoff_y
        band_s = fy - L.p["berth_depth_y"] + 0.03
        s = v.seg
        return [
            s("pin", 0.0, 0.005),                    # 降销, 托盘落位
            s("x", tx - 0.25, K["agv_cruise"]),      # 退出台位
            s("x", corridor_x, K["agv_cruise"]),     # 西回列通道
            s("y", band_s, K["agv_cruise"]),         # 南回泊位带
        ]


class FleetController:
    """多车序列执行器: 100Hz 位置闭环 + Ramp 梯形限速 + 穿透审计。

    与 CellController 同构, 但段规格为 (执行器id, Ramp工厂, 关节偏移)
    三元组 —— 支持任意车辆任意通道, 世界坐标目标经偏移换算为关节量。
    """

    CTRL_HZ = 100

    def __init__(self, cell: WarehouseCell):
        self.cell = cell
        self.substeps = max(1, round(0.01 / cell.model.opt.timestep))

    def run_sequence(self, segments, settle_s: float = 0.3,
                     max_pen_m: float = 5e-4, audit_every: int = 10) -> dict:
        m, d = self.cell.model, self.cell.data
        audit: dict = {"segments": [], "max_pen": 0.0, "violations": []}
        step_n = 0
        for aid, build, off in segments:
            ramp = build()
            n_ctrl = int(ramp.duration * self.CTRL_HZ) + 1
            t_seg0 = d.time
            for _ in range(n_ctrl):
                d.ctrl[aid] = ramp.at(d.time - t_seg0) - off
                for _ in range(self.substeps):
                    mujoco.mj_step(m, d)
                    step_n += 1
                    if step_n % audit_every == 0:
                        pen = self.cell.max_penetration()
                        audit["max_pen"] = max(audit["max_pen"], pen)
                        if pen < -max_pen_m:
                            audit["violations"].append(
                                f"穿透 {pen * 1000:.2f}mm @ t={d.time:.3f}s")
                            return audit
            d.ctrl[aid] = ramp.target - off
            for _ in range(int(settle_s * self.CTRL_HZ)):
                for _ in range(self.substeps):
                    mujoco.mj_step(m, d)
                    step_n += 1
                    if step_n % audit_every == 0:
                        pen = self.cell.max_penetration()
                        audit["max_pen"] = max(audit["max_pen"], pen)
            audit["segments"].append({"t_end": d.time, "aid": aid,
                                      "target": ramp.target - off})
        return audit
