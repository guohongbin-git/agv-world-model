"""W03 S00 单货位台架 — 白模 MJCF (T01/T02/T03 物理闭环, CAD 碰撞体就位前)。

几何链 (设计文档 4.1/20.1/20.2, 全部来自 configs/design.yaml):
- 货叉顶面运输位 65mm, 第三层插叉位 410mm, 抬离位 417mm
- 托盘: 叉孔 y=±15±9mm, z 4→12mm, X 向贯通; 近边=中心-40mm (车从 -X 接近)
- 插满: 叉尖越过近边 65mm → 前轮轴在托盘中心 -95mm
- 支撑面 400mm; 抬离后托盘底面 = 417-12 = 405mm (离面 5mm)

时钟: 纯仿真; 控制 100Hz, 物理 1ms。
"""
from __future__ import annotations

import re

import mujoco
import numpy as np

from configs.loader import load  # 参数单一来源


class VehicleView:
    """多车参数化后的**单车轻量视图** (per-车量只读句柄)。

    ForkliftCell 的 XML 中每辆车的可动部分以 ``<vid>_`` 前缀命名, 顶层
    ``a_x``/``qadr_x``/``axle_x`` 等属性固定指向**首车**以保持单车 API
    兼容; 其它车辆经 ``cell.vehicle(vid)`` 取本视图。共享量 (model/data/
    pallet/rack/接触审计) 留在 cell 顶层, 不在此重复暴露。
    """

    __slots__ = ("cell", "vid", "j_x", "j_lift", "qadr_x", "qadr_lift",
                 "a_x", "a_lift", "q0_x", "fork_geoms")

    def __init__(self, cell: "ForkliftCell", vid: str,
                 j_x: int, j_lift: int, qadr_x: int, qadr_lift: int,
                 a_x: int, a_lift: int, q0_x: float,
                 fork_geoms: tuple[int, int]):
        self.cell = cell
        self.vid = vid
        self.j_x = j_x
        self.j_lift = j_lift
        self.qadr_x = qadr_x
        self.qadr_lift = qadr_lift
        self.a_x = a_x
        self.a_lift = a_lift
        self.q0_x = q0_x
        self.fork_geoms = fork_geoms

    @property
    def fork_gids(self) -> set[int]:
        return set(self.fork_geoms)

    @property
    def start_axle_x(self) -> float:
        """本车初始前轴 x (chassis body_pos[0]); 编队错位见 _start_axle_x。"""
        return self.q0_x

    @property
    def axle_x(self) -> float:
        return self.q0_x + float(self.cell.data.qpos[self.qadr_x])

    @property
    def fork_top_z(self) -> float:
        return 0.065 + float(self.cell.data.qpos[self.qadr_lift])

    def fork_tip_x(self) -> float:
        return self.axle_x + self.cell.k["fork_root_x"] + self.cell.k["fork_len"]

    def __repr__(self) -> str:
        return (f"<VehicleView {self.vid} axle_x={self.axle_x:+.3f} "
                f"fork_top_z={self.fork_top_z:.3f}>")


class ForkliftCell:
    """S00: N 辆叉车 + 单托盘 + 400mm 支撑面。坐标: 托盘中心=原点, 车沿 -X 侧接近。

    多车参数化 (P3 双车交接前置): ``vehicles=["F01","F02"]`` 在同一巷道
    生成 N 辆同规格车, 每辆车的可动部分 (chassis body 树 + drive_x/lift
    执行器) 在 XML 中以 ``<vid>_`` 前缀命名; 静态环境 (地面/货架/暂存台/
    托盘) 只有一份共享。后车沿 -X 依次后撤停放, 见 ``_start_axle_x``。
    ``vehicles`` 缺省为 ``["F01"]``, 与参数化前的单车行为逐位一致;
    顶层 ``a_x``/``qadr_x`` 等属性固定指向**首车**以保持单车 API 兼容,
    其它车辆经 ``cell.vehicle(vid)`` 取 :class:`VehicleView` 视图。
    """

    def __init__(self, timestep_s: float = 0.001, seed: int = 0,
                 support_z: float | None = None,
                 levels: list[float] | None = None,
                 cones: list[tuple[float, float]] | None = None,
                 vehicles: list[str] | None = None):
        """levels: 多层货架扩展 (P2 多货物), 如 [0.100, 0.250, 0.400]。
        给定则每层各建一组支撑梁 + 一个自由托盘 (pallet / pallet_b / pallet_c),
        support_z 参数被 levels 覆盖; 不给则保持 S00 单层基准 (T01-T04 测试)。
        cones: 巷道锥桶 [(x,y),...], 静态圆柱 r25 h90 (P4 避障/感知)。
        锥桶实现为 **mocap 体** (cone_0/cone_1): 运行时可经 data.mocap_pos
        注入/移除 (移除 = 移出场景 y=5), 无需重建模型。
        vehicles: 编队车辆 id (F01/F02/...), id 仅允许 [A-Za-z0-9_] 且不得
        重复 (id 直接进 MuJoCo 元素名); None = ["F01"] 单车基准。"""
        # --- 编队参数 (须在建模前定, XML 按车辆数生成) ---
        self.vehicles = ["F01"] if vehicles is None else [str(v) for v in vehicles]
        if not self.vehicles:
            raise ValueError("vehicles 不能为空列表")
        _bad = [v for v in self.vehicles if not re.fullmatch(r"[A-Za-z0-9_]+", v)]
        if _bad:
            raise ValueError(f"车辆 id 只允许 [A-Za-z0-9_] (进 MuJoCo 名): {_bad}")
        if len(set(self.vehicles)) != len(self.vehicles):
            raise ValueError(f"车辆 id 重复: {self.vehicles}")
        # 支撑面高度可参数化 (T04 三层货位 100/250/400mm)。留空则用设计默认
        # 最高层 —— T01-T03 的单台架基准即此高度。
        self.levels = [float(z) for z in levels] if levels else None
        self.cone_xy = [tuple(c) for c in (cones or [])][:2]
        self.cfg = load()
        self.support_z = float(self.cfg["rack"]["support_z_m"][-1]
                               if support_z is None else support_z)
        if self.levels:
            self.support_z = self.levels[-1]
        self.model = mujoco.MjModel.from_xml_string(self._xml(timestep_s))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.rng = np.random.default_rng(seed)
        # 关键 id —— 每车一份 (movable 部分以 <vid>_ 前缀命名, 见 _vehicle_xml)
        m, d = self.model, self.data
        self._veh: dict[str, VehicleView] = {
            vid: self._vehicle_ids(vid) for vid in self.vehicles}
        # 单车 API 兼容: 顶层属性 = 首车 (缺省即唯一车 F01) 的 id。
        _v0 = self._veh[self.vehicles[0]]
        self.j_x = _v0.j_x
        self.j_lift = _v0.j_lift
        self.qadr_x = _v0.qadr_x
        self.qadr_lift = _v0.qadr_lift
        self.a_x = _v0.a_x
        self.a_lift = _v0.a_lift
        # chassis_x 的 q 是相对车体初始位姿的位移; 初始 x=该车的 start_axle_x
        self.q0_x = _v0.q0_x
        self.fork_geoms = list(_v0.fork_geoms)
        self.fork_gids = set(_v0.fork_geoms)
        self.pallet_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pallet")
        # 多层扩展 (levels): 附加托盘 id 表 (基准 pallet 之外)
        self.pallet_bids = {"pallet": self.pallet_bid}
        for _sfx in ("_b", "_c", "_d"):
            _b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                   f"pallet{_sfx}")
            if _b >= 0:
                self.pallet_bids[f"pallet{_sfx}"] = _b
        # 锥桶 mocap 地址 (运行时经 data.mocap_pos 注入/移除)
        self.cone_mocapid = []
        for i in range(2):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"cone_{i}")
            self.cone_mocapid.append(m.body_mocapid[b] if b >= 0 else -1)
        for i, (cx, cy) in enumerate(self.cone_xy):
            if self.cone_mocapid[i] >= 0:
                d.mocap_pos[self.cone_mocapid[i]] = [cx, cy, 0.045]
        self.pallet_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                             for g in ("pal_bottom_l", "pal_bottom_r", "pal_bottom_c",
                                       "ceil_l", "ceil_r", "top_side_l", "top_side_r",
                                       "top_mid")]
        self.support_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                             for g in ("support_l", "support_c", "support_r", "stage_top")]
        self.floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._ctrl_target = np.zeros(2)  # [x, lift] 位置目标

    # ---------- 多车编队 ----------
    def vehicle(self, vid: str) -> VehicleView:
        """按车辆 id 取单车视图 (per-车量: a_x/axle_x/fork_top_z/...)。"""
        try:
            return self._veh[vid]
        except KeyError:
            raise KeyError(f"未知车辆 {vid!r}; 已编入: {self.vehicles}") from None

    def _start_axle_x(self, i: int) -> float:
        """车 i 的初始前轴 x —— P3 编队同巷道错位停放。

        车体包络相对前轴: 最后端 = 配重后缘 axle−0.21m, 最前端 = 叉尖
        axle+0.12m (fork_root 0.040 + fork_len 0.080), 全长 0.33m。
        纵队取 k["vehicle_spacing"]=0.40m → 相邻两车初始净空 ≥70mm,
        初始位姿零重叠 (双车 smoke 实测)。F01 (i=0) **精确取**
        k["start_axle_x"] (抬升安全轴), 单车初始位姿与参数化前逐位一致;
        车 i 后撤 i×0.40m。后车 chassis_x 关节位移上限按货架插叉位反推
        放宽 (P3 Wave2, 见 _vehicle_xml 的 q_hi 注释)。
        """
        return self.k["start_axle_x"] - i * self.k["vehicle_spacing"]

    def _vehicle_ids(self, vid: str) -> VehicleView:
        """解析单车 `<vid>_` 前缀的全部关键 id (找不到即抛错, 不静默 −1)。"""
        m = self.model
        T = mujoco.mjtObj

        def _id(kind, name):
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise KeyError(f"模型中找不到 {name!r} (车辆 {vid}) —— "
                               f"XML 模板与 id 缓存不同步")
            return i

        j_x = _id(T.mjOBJ_JOINT, f"{vid}_chassis_x")
        j_lift = _id(T.mjOBJ_JOINT, f"{vid}_lift")
        return VehicleView(
            cell=self, vid=vid,
            j_x=j_x, j_lift=j_lift,
            qadr_x=m.jnt_qposadr[j_x],      # qpos 按地址, 非关节id
            qadr_lift=m.jnt_qposadr[j_lift],
            a_x=_id(T.mjOBJ_ACTUATOR, f"{vid}_drive_x"),
            a_lift=_id(T.mjOBJ_ACTUATOR, f"{vid}_lift"),
            q0_x=float(m.body_pos[_id(T.mjOBJ_BODY, f"{vid}_chassis")][0]),
            fork_geoms=(_id(T.mjOBJ_GEOM, f"{vid}_fork_l"),
                        _id(T.mjOBJ_GEOM, f"{vid}_fork_r")),
        )

    # ---------- 几何常量 (由 design.yaml 派生, mm→m) ----------
    #
    # insert_z / lift_z 必须随 support_z 派生: 插叉高度由**支撑面**决定
    # (叉顶须压在托盘通道顶壁下沿), 不同层高的插入高度不同。原先两者
    # 硬编码 0.410/0.417 只对第三层(400mm)成立。
    @property
    def k(self):
        c = self.cfg
        sz = self.support_z
        k = {
            "transport_z": 0.065,
            # 设计 176 行: 「插叉时货叉顶面设置为支撑面上方 10 mm, 先保留 2 mm
            # 竖向间隙; 举升至上方 17 mm, 接触叉孔顶部后使托盘底面抬离
            # 支撑面 5 mm。第三层因此需要货叉顶面 z = 417 mm」。
            # 叉孔窗口 = 托盘局部 z 4~12mm; 叉厚 2.5mm。
            # 插到 +10 时叉占 [7.5,10], 完全在窗口内, 距顶壁还有 2mm ——
            # 撬升行程充足。此前误取「支撑面 + 通道顶壁 − 0.5」= +11.5,
            # 距顶壁仅 0.5mm, 叉一开始就顶在通道顶壁上, 承托力恒为
            # 0.57× 托盘重, 抬多高都不变。
            "insert_z": sz + 0.010,
            "lift_z": sz + 0.017,           # 支撑面 +17 = 417mm @第三层
            "support_z": sz,
            "fork_root_x": 0.040,
            "fork_len": 0.080,
            "insert_depth": 0.065,
            "approach_gap": 0.055,
            "axle_behind_center": 0.095,
            "retreat_loaded": 0.120,
            "withdraw_empty": 0.105,
            "lift_travel": 0.385,        # 叉顶 65→450
            "channel_top_local": 0.012,  # 通道顶壁底面(托盘局部), 仅作净空校核
            "start_axle_x": -0.560,      # 抬升安全轴: 全程无干涉
            # 多车同巷道纵队间距: 车体包络相对前轴 [−0.21, +0.12] 全长 0.33m
            # (配重后缘→叉尖), 0.40m 间距使相邻两车初始净空 ≥70mm, 零重叠。
            # 见 _start_axle_x。
            "vehicle_spacing": 0.40,
        }
        # 货架插叉位的前轴 x —— 「叉尖越过托盘近边 65mm」的作业点 (托盘中
        # 心 = 原点, 托盘 -X 边 = -0.040): axle = -0.040 + insert_depth
        # − (fork_root_x + fork_len) = -0.095。非首车 chassis_x 行程上限
        # 由此推导 (_vehicle_xml), 行车计划的 x_ins 亦取此单源。
        k["rack_insert_axle_x"] = (-0.040 + k["insert_depth"]
                                   - (k["fork_root_x"] + k["fork_len"]))
        return k

    # ---------- MJCF ----------
    def _pal_block(self, name: str, z: float, sfx: str) -> str:
        """标准托盘自由体 XML (底条×3 + 通道顶壁×2 + 顶板×3), geom 名加后缀。"""
        return f"""
    <body name="{name}" pos="0 0 {z}">
      <freejoint name="pal{sfx}"/>
      <geom name="pal_bottom_l{sfx}" type="box" pos="0 -0.027 0.002" size="0.040 0.003 0.002" mass="0.006" rgba="0.55 0.40 0.22 1"/>
      <geom name="pal_bottom_r{sfx}" type="box" pos="0 0.027 0.002" size="0.040 0.003 0.002" mass="0.006" rgba="0.55 0.40 0.22 1"/>
      <geom name="pal_bottom_c{sfx}" type="box" pos="0 0 0.002" size="0.040 0.006 0.002" mass="0.010" rgba="0.55 0.40 0.22 1"/>
      <geom name="ceil_l{sfx}" type="box" pos="0 -0.015 0.0135" size="0.040 0.007 0.0015"
            mass="0.008" friction="1.4 0.005 0.0001" rgba="0.72 0.55 0.32 1"/>
      <geom name="ceil_r{sfx}" type="box" pos="0 0.015 0.0135" size="0.040 0.007 0.0015"
            mass="0.008" friction="1.4 0.005 0.0001" rgba="0.72 0.55 0.32 1"/>
      <geom name="top_side_l{sfx}" type="box" pos="0 -0.027 0.0175" size="0.040 0.003 0.0025" mass="0.002" rgba="0.78 0.62 0.38 1"/>
      <geom name="top_side_r{sfx}" type="box" pos="0 0.027 0.0175" size="0.040 0.003 0.0025" mass="0.002" rgba="0.78 0.62 0.38 1"/>
      <geom name="top_mid{sfx}" type="box" pos="0 0 0.017" size="0.040 0.008 0.001" mass="0.002" rgba="0.78 0.62 0.38 1"/>
    </body>"""

    def _vehicle_xml(self, vid: str, axle0: float) -> str:
        """单车可动部分 XML (chassis body 树), 全部 body/geom/joint 名加
        ``<vid>_`` 前缀 —— MuJoCo 名字全局唯一, 多车同名即编译失败。

        静态环境 (地面/货架/暂存台/托盘) 不在本块, N 车共享一份。
        与单车的唯一差异是 chassis 初始 pos (=axle0, 见 _start_axle_x);
        关节/执行器/几何参数与单车完全一致。
        """
        k = self.k
        p = f"{vid}_"
        # chassis_x 行程上限 (P3 Wave2): 首车保持 +0.65 —— 单车工作包络,
        # max() 取等, XML 与参数化前逐位一致。后车初始位每车后撤
        # k["vehicle_spacing"] (见 _start_axle_x), 要到达**同一点**作业
        # (货架插叉位 axle = k["rack_insert_axle_x"] = -0.095), 关节位移
        # 上限按 q_hi = x_ins − axle0 + 5mm 余量推导:
        #   i=1: -0.095 − (-0.960) = +0.865 → range 上限 0.870
        # (交接台放托 axle≈-0.444/-0.424 所需 +0.536/+0.516 均被覆盖)。
        # 只放宽限位, 不改任何动力学参数 —— 单车物理行为逐位不变。
        # 下限 -0.8 保持: 后撤需求 = 编队净空 + 退距, 远小于 0.8m。
        q_hi = max(0.65, k["rack_insert_axle_x"] - axle0 + 0.005)
        return f"""
    <!-- 车辆 {vid}: 平面底盘(滑动x+铰z) + 升降滑轨; 初始前轴 x={axle0:.3f} -->
    <body name="{p}chassis" pos="{axle0} 0 0">
      <!-- 行程上限须覆盖满插入深度: 插叉深 65mm 指**叉尖**越过托盘近边,
           axle 到 -95mm。首车 qpos = -95-(-560) = +465mm, 原上限 +500mm;
           后车初始更靠后, 上限按插叉位反推 (见上方 q_hi 注释)。 -->
      <joint name="{p}chassis_x" type="slide" axis="1 0 0" range="-0.8 {q_hi:.3f}" damping="20"/>
      <joint name="{p}chassis_yaw" type="hinge" axis="0 0 1" range="-0.2 0.2" damping="5"/>
      <geom name="{p}chassis_body" type="box" pos="-0.085 0 0.0175"
            size="0.110 0.075 0.0175" mass="1.50" rgba="0.90 0.66 0.05 1"/>
      <geom name="{p}counterweight" type="box" pos="-0.20 0 0.0175"
            size="0.010 0.060 0.0175" mass="0.0" rgba="0.18 0.18 0.20 1"/>
      <!-- 装饰门架 (contype=0 纯视觉): 双导轨 + 顶横梁, 叉车轮廓 -->
      <geom name="{p}mast_l" type="box" pos="0.005 -0.045 0.25" size="0.006 0.006 0.25"
            mass="0" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
      <geom name="{p}mast_r" type="box" pos="0.005 0.045 0.25" size="0.006 0.006 0.25"
            mass="0" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
      <geom name="{p}mast_top" type="box" pos="0.005 0 0.497" size="0.008 0.051 0.006"
            mass="0" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
      <body name="{p}carriage" pos="0 0 0">
        <joint name="{p}lift" type="slide" axis="0 0 1" range="-0.050 {k['lift_travel']}" damping="60"/>
        <!-- 背板 (托盘后立面挡板), 分两片置于 y∈[8,22]/[-22,-8] —— 与叉孔
           通道顶壁 ceil_l/r 同 y, 即三条支撑横梁 (y=0±6, ±27±3) 之间的空隙:
           - 整片 y±60 的背板升到第三层必撞横梁下沿 (实测 support_*↔背板
             12 个接触, ctrl 打满 +5N 而 qpos 锁死 319mm, 叉顶卡 384mm);
           - 背板还须**高于叉顶**: 放托时叉顶降到台面 45+12=57mm, 若背板
             在叉下方会先压上暂存台面 (实测承 2.07N 整叉架重量, 叉顶死锁)。
           两片与 ceil_l/r 接触即可阻止托盘滑向车 (刚性连接)。 -->
        <geom name="{p}carriage_plate_l" type="box" pos="0.020 -0.015 0.068"
              size="0.010 0.007 0.008" mass="0.125" rgba="0.35 0.36 0.40 1"/>
        <geom name="{p}carriage_plate_r" type="box" pos="0.020 0.015 0.068"
              size="0.010 0.007 0.008" mass="0.125" rgba="0.35 0.36 0.40 1"/>
        <geom name="{p}fork_l" type="box" pos="{k['fork_root_x'] + k['fork_len']/2} -0.015 {k['transport_z'] - 0.00125}"
              size="{k['fork_len']/2} 0.005 0.00125" mass="0.02" friction="1.4 0.02 0.0002" rgba="0.15 0.15 0.17 1"/>
        <geom name="{p}fork_r" type="box" pos="{k['fork_root_x'] + k['fork_len']/2} 0.015 {k['transport_z'] - 0.00125}"
              size="{k['fork_len']/2} 0.005 0.00125" mass="0.02" friction="1.4 0.02 0.0002" rgba="0.15 0.15 0.17 1"/>
      </body>
    </body>"""

    def _vehicle_actuators_xml(self, vid: str) -> str:
        """单车两个位置执行器 (行车/升降), 同样按 `<vid>_` 前缀命名。"""
        p = f"{vid}_"
        return f"""
    <position name="{p}drive_x" joint="{p}chassis_x" kp="800" forcerange="-20 20"/>
    <position name="{p}lift" joint="{p}lift" kp="10000" forcerange="-5 5"/>"""

    def _xml(self, dt: float) -> str:
        k = self.k
        lv = self.levels or []
        extra_sup, extra_pal = "", ""
        for i, z in enumerate(lv[:-1]):        # 最高层 = 基准 pallet
            sfx = ("_b", "_c", "_d")[i]
            extra_sup += (
                f'    <geom name="support_l{sfx}" type="box" pos="0 -0.027 '
                f'{z - 0.0025}" size="0.046 0.003 0.0025" '
                f'rgba="0.80 0.42 0.10 1"/>\n'
                f'    <geom name="support_c{sfx}" type="box" pos="0 0 '
                f'{z - 0.0025}" size="0.046 0.006 0.0025" '
                f'rgba="0.80 0.42 0.10 1"/>\n'
                f'    <geom name="support_r{sfx}" type="box" pos="0 0.027 '
                f'{z - 0.0025}" size="0.046 0.003 0.0025" '
                f'rgba="0.80 0.42 0.10 1"/>\n')
            extra_pal += self._pal_block(f"pallet{sfx}", z + 0.001, sfx)
        nc = 200 + 150 * max(0, len(lv) - 1) + 120 * (len(self.vehicles) - 1)
        cones_xml = ""
        for i in range(2):
            cx, cy = self.cone_xy[i] if i < len(self.cone_xy) else (0.0, 5.0)
            cones_xml += (f'    <body name="cone_{i}" mocap="true" '
                          f'pos="{cx} {cy} 0.045">\n'
                          f'      <geom name="cone_{i}" type="cylinder" '
                          f'size="0.025 0.045" rgba="0.95 0.45 0.05 1" '
                          f'friction="0.8 0.005 0.0001"/>\n'
                          f'    </body>\n')
        # 每辆车一份可动块 + 执行器; 静态环境 (上方全部) 只有一份共享
        vehicles_xml = "".join(
            self._vehicle_xml(vid, self._start_axle_x(i))
            for i, vid in enumerate(self.vehicles))
        actuators_xml = "".join(self._vehicle_actuators_xml(vid)
                                for vid in self.vehicles)
        return f"""
<mujoco model="S00_bench">
  <!-- MuJoCo 3.15 起 <compiler angle> 默认为 degree; 本项目统一 SI 弧度 -->
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 -9.81" iterations="200"
          tolerance="1e-10" noslip_iterations="30" noslip_tolerance="1e-8"/>
  <size nconmax="{nc}" njmax="300"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>   <!-- 离屏帧缓冲 (纯视觉) -->
    <headlight diffuse="0.45 0.45 0.48" specular="0.1 0.1 0.1"/>
    <map fogstart="1.2" fogend="3.5"/>
    <rgba fog="0.06 0.07 0.09 1"/>
  </visual>
  <asset>
    <!-- 纯视觉: 地面网格纹理 (物理不变) -->
    <texture type="2d" name="floorgrid" builtin="checker"
             rgb1="0.30 0.32 0.36" rgb2="0.20 0.22 0.26" width="64" height="64"/>
    <material name="floormat" texture="floorgrid" texrepeat="40 40" reflectance="0.12"/>
  </asset>
  <default>
    <geom rgba="0.7 0.7 0.75 1" condim="3" friction="0.5 0.005 0.0001"
          solref="0.0002 1" solimp="0.999 0.9999 0.00001"/>
    <joint damping="0.1" armature="0.001"/>
  </default>
  <worldbody>
    <!-- 三灯布光: 主光投影 + 冷色补光 + 顶部环境光 -->
    <light pos="0.6 -0.7 1.2" dir="-0.3 0.3 -1" diffuse="0.8 0.78 0.74" castshadow="true"/>
    <light pos="-0.9 0.7 1.0" dir="0.3 -0.2 -1" diffuse="0.35 0.37 0.42" castshadow="true"/>
    <light pos="0 0 1.6" dir="0 0 -1" diffuse="0.28 0.28 0.30" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 0.05" friction="0.5" material="floormat"/>
    <!-- 装饰立柱: contype/conaffinity=0, 不参与接触求解 (规则 1: 纯视觉) -->
    <geom name="upright_fl" type="box" pos="0.20 -0.033 {(k['support_z'] + 0.10)/2}"
          size="0.004 0.004 {(k['support_z'] + 0.10)/2}" rgba="0.25 0.30 0.45 1"
          contype="0" conaffinity="0"/>
    <geom name="upright_fr" type="box" pos="0.20 0.033 {(k['support_z'] + 0.10)/2}"
          size="0.004 0.004 {(k['support_z'] + 0.10)/2}" rgba="0.25 0.30 0.45 1"
          contype="0" conaffinity="0"/>
    <geom name="upright_bl" type="box" pos="-0.20 -0.033 {(k['support_z'] + 0.10)/2}"
          size="0.004 0.004 {(k['support_z'] + 0.10)/2}" rgba="0.25 0.30 0.45 1"
          contype="0" conaffinity="0"/>
    <geom name="upright_br" type="box" pos="-0.20 0.033 {(k['support_z'] + 0.10)/2}"
          size="0.004 0.004 {(k['support_z'] + 0.10)/2}" rgba="0.25 0.30 0.45 1"
          contype="0" conaffinity="0"/>
    <!-- 支撑面: 三条横梁 (y=±27,0), 与托盘底条同位; 高度随 support_z 参数化。
     x 向只在 bay 宽度内 (±0.046): 真实 selective rack 的梁不横跨巷道,
     AGV 载货抬升路径 (巷道上空) 必须无梁 (P2 实测 400mm 全跨时 T2
     载货抬升撞 L1 梁, 10.9N 死锁 → 复现并修正)。 -->
    <geom name="support_l" type="box" pos="0 -0.027 {k['support_z'] - 0.0025}" size="0.046 0.003 0.0025" rgba="0.80 0.42 0.10 1"/>
    <geom name="support_c" type="box" pos="0 0 {k['support_z'] - 0.0025}" size="0.046 0.006 0.0025" rgba="0.80 0.42 0.10 1"/>
    <geom name="support_r" type="box" pos="0 0.027 {k['support_z'] - 0.0025}" size="0.046 0.003 0.0025" rgba="0.80 0.42 0.10 1"/>
{extra_sup}
    <!-- 暂存站 (T03 第二货位): 台面顶 z=45mm; 台下净空 z=0..44mm 供底盘(宽±75mm)通过,
         立柱 y=±85mm 在底盘宽度之外; 叉(±20mm)从台下中心贯通 -->
    <geom name="stage_top" type="box" pos="-0.340 0 0.0445" size="0.060 0.050 0.0005" rgba="0.32 0.46 0.52 1"/>
    <geom name="stage_leg_l" type="box" pos="-0.395 -0.085 0.022" size="0.003 0.003 0.022" rgba="0.22 0.32 0.38 1"/>
    <geom name="stage_leg_r" type="box" pos="-0.395 0.085 0.022" size="0.003 0.003 0.022" rgba="0.22 0.32 0.38 1"/>
    <!-- 托盘 自由体, 原点=底面(底条局部z 0..4) -->
    <body name="pallet" pos="0 0 {k['support_z'] + 0.001}">
      <freejoint name="pal"/>
      <geom name="pal_bottom_l" type="box" pos="0 -0.027 0.002" size="0.040 0.003 0.002" mass="0.006" rgba="0.55 0.40 0.22 1"/>
      <geom name="pal_bottom_r" type="box" pos="0 0.027 0.002" size="0.040 0.003 0.002" mass="0.006" rgba="0.55 0.40 0.22 1"/>
      <geom name="pal_bottom_c" type="box" pos="0 0 0.002" size="0.040 0.006 0.002" mass="0.010" rgba="0.55 0.40 0.22 1"/>
      <!-- 通道顶壁: 独立条带 (y=±15±7, z 12..15mm)。叉顶面推的是通道顶壁而非整块顶板,
           独立几何才能让接触求解生成多点分布接触 (整块顶板底面只会在叉尖成线)。
           y 半宽 7mm (原 5mm) 是为了与叉 (y 半宽 5mm) 有 4mm 横向重叠余量:
           插入 65mm 时叉与托盘 x 向仅重叠 15mm, 接触区窄; 插入后实测
           fork_l 零接触、fork_r 两接触, 单齿承托使托盘绕单点翻转必掉。
           加宽后两齿都能建立接触。 -->
      <!-- 叉孔摩擦 1.4: T10 契约要求 0.2m/s 急停 <2mm (减速度 ≥10 m/s²),
           载荷不滑移须 μg > 减速度, 即 μ ≥ 1.02。原 0.5/0.8 下 μg=7.8,
           两条款数学上不可同时满足。 -->
      <geom name="ceil_l" type="box" pos="0 -0.015 0.0135" size="0.040 0.007 0.0015"
            mass="0.008" friction="1.4 0.005 0.0001" rgba="0.72 0.55 0.32 1"/>
      <geom name="ceil_r" type="box" pos="0 0.015 0.0135" size="0.040 0.007 0.0015"
            mass="0.008" friction="1.4 0.005 0.0001" rgba="0.72 0.55 0.32 1"/>
      <!-- 顶板其余部分: 通道两侧边带 + 中桥, 避开叉的插入面 -->
      <geom name="top_side_l" type="box" pos="0 -0.027 0.0175" size="0.040 0.003 0.0025" mass="0.002" rgba="0.78 0.62 0.38 1"/>
      <geom name="top_side_r" type="box" pos="0 0.027 0.0175" size="0.040 0.003 0.0025" mass="0.002" rgba="0.78 0.62 0.38 1"/>
      <geom name="top_mid" type="box" pos="0 0 0.017" size="0.040 0.008 0.001" mass="0.002" rgba="0.78 0.62 0.38 1"/>
      <!-- 叉从 -X 侧贯通进入; 设计 180 行「真实叉孔贯通结构无背面挡块」。
           原先有 pal_back_l/r 两块 condim=1 挡块, 压在货架横梁上,
           使 support_load_ratio('beam') 长期读到非零值而误判托盘仍压梁。 -->
    </body>
{extra_pal}
{cones_xml}
{vehicles_xml}
  </worldbody>
  <actuator>{actuators_xml}
  </actuator>
</mujoco>
"""

    # ---------- 状态 ----------
    @property
    def axle_x(self) -> float:
        return self.q0_x + float(self.data.qpos[self.qadr_x])

    @property
    def fork_top_z(self) -> float:
        return 0.065 + float(self.data.qpos[self.qadr_lift])

    @property
    def pallet_pose(self) -> tuple[np.ndarray, np.ndarray]:
        bid = self.pallet_bid
        return (self.data.xpos[bid].copy(), self.data.xquat[bid].copy())

    @property
    def pallet_bottom_z(self) -> float:
        # 托盘体原点 = 底条底面 (局部 z=0 平面)
        pos, _ = self.pallet_pose
        return float(pos[2])

    def fork_tip_x(self) -> float:
        return self.axle_x + self.k["fork_root_x"] + self.k["fork_len"]

    # ---------- 接触审计 ----------
    def contacts(self) -> list[tuple[int, int, float]]:
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            out.append((c.geom1, c.geom2, float(c.dist)))
        return out

    def max_penetration(self) -> float:
        return max((-d for _, _, d in self.contacts()), default=0.0)

    def pallet_fork_contact(self, vid: str | None = None) -> bool:
        """托盘与指定车辆货叉的接触 (vid 缺省 = 首车, 单车 API 不变)。"""
        fg = self.vehicle(vid or self.vehicles[0]).fork_gids
        pg = set(self.pallet_geoms)
        return any((g1 in fg and g2 in pg) or (g2 in fg and g1 in pg)
                   for g1, g2, _ in self.contacts())

    def support_load_ratio(self, surface: str = "beam",
                           vid: str | None = None) -> float:
        """支撑面承担的竖向力 / 托盘总重 (放置判据)。

        mj_contactForce 返回的是"施加于 geom1 的力"; 支撑梁 geom1 被托盘向下压,
        故取 -fz (向上为正)。

        surface="beam": 货架横梁 —— 托盘**放**在梁上时的判据;
        surface="fork": 叉顶 —— 托盘**被叉举着**时的判据 (vid 指明哪辆车,
        缺省 = 首车)。
        二者是不同事件: 插叉后托盘可能同时压着梁又压着叉, 只读一个会把
        「叉上载着但还没离梁」误判成「叉空了」。
        """
        gids = (self.support_gids if surface == "beam"
                else self.vehicle(vid or self.vehicles[0]).fork_gids)
        total = 0.0
        f1 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 not in gids and g2 not in gids:
                continue
            if g1 in gids and g2 in self.pallet_geoms:
                sign = -1.0      # 支撑梁被压 → 力向下, 取反得托盘受到的向上托力
            elif g2 in gids and g1 in self.pallet_geoms:
                sign = 1.0
            else:
                continue        # 非托盘对象不计入
            mujoco.mj_contactForce(self.model, self.data, i, f1)
            frame = c.frame.reshape(3, 3)
            f_world = frame @ f1[:3]
            total += max(0.0, sign * f_world[2])
        weight = 0.045 * 9.81
        return total / weight if weight else 0.0

    # ---------- 物理往返 (取放 + 撤离) ----------
    def roundtrip_plan(self) -> list[tuple[int, "object"]]:
        """一次完整取放的执行计划: (执行器idx, 轨迹)。

        从架上取托盘 → 带载退出 → 降至运输位 → 转运至暂存站 → 放托盘 →
        空叉退出。共 12 段, 与设计 848/863 行一致 (前进 ≤5 mm/s)。
        计划按 `self.k` 动态生成, 故任意 support_z 层高都适用。

        返回 (执行器idx, 构造Ramp的可调用对象) 列表 —— 每段的 Ramp 在
        **该段开始时**才构造, 起点取当时的实际位置。若一次性构造, 12 段
        会共用同一个起点值, 轨迹全错。

        求值交给 CellController.run_sequence 的 callables 形式。
        """
        k = self.k
        # 几何由**物理定义 + 实测**确定, 不引用 axle_behind_center 之类的
        # 名义常数 (该常数与本台架实际几何不符)。
        #
        # 叉从托盘 -X 侧贯通插入, **承重接触面是叉根**(叉根进入通道后
        # 托盘底条压在叉根到叉尖之间)。实测 (support_z=250mm):
        #   axle=-15.0mm 时 叉根=+25.0mm 叉尖=+105.0mm 托盘中心=+1.2mm
        #   → 托盘中心 ≈ 叉根 ≈ axle + fork_root_x  (偏置 40mm, 非 80mm)
        #   → 插入深度 = 叉尖 − 托盘 +X 边 = 105−40 = 65mm, 与设计一致
        # 我先前按「托盘重心在叉尖」取 80mm 偏置, 差了 40mm, 使暂存站
        # 对位整体偏了一个托盘半长。
        half_pal = 0.040
        pal_entry_x = -half_pal                       # 通道入口 (托盘 -X 边)
        # 托盘中心相对 axle 的偏置。**实测 94mm** (叉尖插深 65mm 时):
        # 段7 末 axle=−403.0, 托盘中心 x=−309.4 → 偏置 93.6mm。
        # 此前三次取错: 按几何推 80mm; 在「插深按叉根」的旧几何下测 24mm;
        # 改按叉尖算后又沿用 63mm (那是插深尚未生效时的值)。
        # 判据是「托盘中心对准暂存站 −340mm」, 该值直接决定对位精度,
        # 只能实测, 不能推算。
        off = 0.094
        # 设计 155 行的「插叉深 65mm」指**叉尖**越过托盘近边的深度, 不是叉根
        # 位置。此前按叉根算 (x_ins = 通道入口 + 65 − fork_root_x = −15mm),
        # 使叉与托盘的 x 重叠只剩 15mm、承托面全落在托盘后部 (x 25~40),
        # 而重心在 x=0 → 托盘绕承托点前倾 4.17°, 左叉零接触、离梁即掉。
        # 按叉尖算: axle = 通道入口 + 65 − (叉根+叉长) = −95mm,
        # 与原 T01 的 X_INSERTED=−0.095 一致。
        x_ins = pal_entry_x + k["insert_depth"] - (k["fork_root_x"] + k["fork_len"])
        x_edge = pal_entry_x - k["approach_gap"] - k["fork_len"] - k["fork_root_x"]
        # 转运: 让托盘中心对准暂存站 (托盘中心 = axle + off)
        x_stage = -0.340 - off
        # 脱离货架: 判据是**叉尖**完全越过货架横梁 (x ∈ [−200, +200]),
        # 不是托盘中心。叉尖 = axle + fork_root_x + fork_len = axle + 120mm,
        # 故 axle ≤ −320mm 才算退出。
        # 先前用「托盘中心 ≤ −240」算出 axle=−280, 此时叉尖仍在 −160mm,
        # 叉顶 (支撑面+17) 会撞上横梁, 托盘失去承托当坠 —— 实测三层分别
        # 掉到地面 / 台面, 400mm 层还横向飞出 538mm。
        # 设计 202 行: 「取托后先直线退出站体, 再把叉顶面降至运输初值」。
        x_clear = -0.200 - k["fork_root_x"] - k["fork_len"] - 0.010
        # 放托位: 托盘底面须落在暂存台面顶面 45mm。叉顶托在通道顶壁底面(托盘
        # 局部 12mm), 故叉顶 = 45 + 12 = 57mm。
        # (先前取 45+10=55 会让叉顶低于托盘底 2mm, 托盘被台面顶住、
        #  叉降到 88mm 就卡死 —— 叉顶降不到目标。)
        place_z = 0.045 + 0.012
        # 脱叉确认: 再降 0.5mm 使叉与叉孔上壁脱离 (设计 299 行
        # 「继续降到 h_s+10 mm, 令双叉与叉孔上壁脱离接触」)
        release_z = place_z - 0.0005

        def lift(target_z: float, speed: float = 0.010):
            """叉架 → 绝对叉顶高度 target_z。

            注意空间: CellController 对 lift 通道直接写 `ramp.at(t)`,
            **不减 transport_z 基址**, 故 Ramp 必须在**关节 qpos 相对量**
            空间构造 (T01-T03 的既有做法)。本方法收绝对高度, 内部换算。
            起点同样取该段开始时的实际 qpos, 保证 12 段首尾相接。
            """
            def build():
                from .controllers import Ramp
                q = float(self.data.qpos[self.qadr_lift])
                return Ramp(q, target_z - self.k["transport_z"], speed)
            return (self.a_lift, build)

        def move(target_x: float, speed: float = 0.005):
            """底盘 → 绝对轴位。起点取该段开始时的实际轴位。"""
            def build():
                from .controllers import Ramp
                return Ramp(self.axle_x, target_x, speed)
            return (self.a_x, build)

        return [
            lift(k["insert_z"]),                             # 1 升至插入高度 (支撑面+10)
            move(x_edge),                                    # 2 接近前沿
            move(x_ins),                                     # 3 插叉 65mm
            lift(k["lift_z"], 0.002),                        # 4 举升, 托盘离面
            move(x_clear),                                   # 5 带载直线退出, 脱离货架横梁区
            lift(k["transport_z"], 0.010),                   # 6 叉顶降至运输高度 65mm
            move(x_stage),                                   # 7 低位转运至暂存站
            lift(place_z),                                   # 8 举到放托位
            lift(release_z, 0.002),                          # 9 微降, 令叉与叉孔上壁脱离
            # 10 空叉退出: 车须**继续后退**才能抽出叉。叉在托盘通道内时
            # 前进会重新插进去 —— 实测段10 若 move(+180) 托盘被拖走,
            # 叉承托回到 0.98。退 120mm(retreat_loaded) 再加 60mm 余量。
            move(x_stage - k["retreat_loaded"] - 0.060),
            lift(k["transport_z"]),                          # 11 叉复位
        ]
