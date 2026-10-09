"""W03 S00 单货位台架 — 白模 MJCF (T01/T02/T03 物理闭环, CAD 碰撞体就位前)。

几何链 (设计文档 4.1/20.1/20.2, 全部来自 configs/design.yaml):
- 货叉顶面运输位 65mm, 第三层插叉位 410mm, 抬离位 417mm
- 托盘: 叉孔 y=±15±9mm, z 4→12mm, X 向贯通; 近边=中心-40mm (车从 -X 接近)
- 插满: 叉尖越过近边 65mm → 前轮轴在托盘中心 -95mm
- 支撑面 400mm; 抬离后托盘底面 = 417-12 = 405mm (离面 5mm)

时钟: 纯仿真; 控制 100Hz, 物理 1ms。
"""
from __future__ import annotations

import mujoco
import numpy as np

from configs.loader import load  # 参数单一来源


class ForkliftCell:
    """S00: 单叉车 + 单托盘 + 400mm 支撑面。坐标: 托盘中心=原点, 车沿 -X 侧接近。"""

    def __init__(self, timestep_s: float = 0.001, seed: int = 0):
        self.cfg = load()
        self.model = mujoco.MjModel.from_xml_string(self._xml(timestep_s))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.rng = np.random.default_rng(seed)
        # 关键 id
        m, d = self.model, self.data
        self.j_x = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "chassis_x")
        self.j_lift = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "lift")
        self.qadr_x = m.jnt_qposadr[self.j_x]     # qpos 按地址, 非关节id
        self.qadr_lift = m.jnt_qposadr[self.j_lift]
        self.a_x = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_x")
        self.a_lift = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
        # chassis_x 的 q 是相对车体初始位姿的位移; 初始 x=-0.215
        self.q0_x = float(self.model.body_pos[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")][0])
        self.pallet_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pallet")
        self.fork_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"fork_{s}")
                           for s in ("l", "r")]
        self.pallet_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                             for g in ("pal_bottom_l", "pal_bottom_r", "pal_bottom_c",
                                       "ceil_l", "ceil_r", "top_side_l", "top_side_r",
                                       "top_mid", "pal_back_l", "pal_back_r")]
        self.support_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                             for g in ("support_l", "support_c", "support_r", "stage_top")]
        self.floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._ctrl_target = np.zeros(2)  # [x, lift] 位置目标

    # ---------- 几何常量 (由 design.yaml 派生, mm→m) ----------
    @property
    def k(self):
        c = self.cfg
        return {
            "transport_z": 0.065,
            "insert_z": 0.410,
            "lift_z": 0.417,
            "support_z": 0.400,
            "fork_root_x": 0.040,
            "fork_len": 0.080,
            "insert_depth": 0.065,
            "approach_gap": 0.055,
            "axle_behind_center": 0.095,
            "retreat_loaded": 0.120,
            "withdraw_empty": 0.105,
            "lift_travel": 0.385,        # 叉顶 65→450
            "channel_top_local": 0.012,  # 叉孔顶(托盘局部)
            "start_axle_x": -0.560,      # 抬升安全轴: 全程无干涉
        }

    # ---------- MJCF ----------
    def _xml(self, dt: float) -> str:
        k = self.k
        # 托盘: 三条底条(z0-4) + 顶板(z12-18) + 背块; 总 45g
        return f"""
<mujoco model="S00_bench">
  <option timestep="{dt}" gravity="0 0 -9.81" iterations="200"
          tolerance="1e-10" noslip_iterations="30" noslip_tolerance="1e-8"/>
  <size nconmax="200" njmax="300"/>
  <default>
    <geom rgba="0.7 0.7 0.75 1" condim="3" friction="0.5 0.005 0.0001"
          solref="0.0002 1" solimp="0.999 0.9999 0.00001"/>
    <joint damping="0.1" armature="0.001"/>
  </default>
  <worldbody>
    <light pos="0.5 -0.5 1.0" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.05" friction="0.5"/>
    <!-- 支撑面 400mm: 三条横梁 (y=±27,0), 与托盘底条同位 -->
    <geom name="support_l" type="box" pos="0 -0.027 0.3975" size="0.20 0.003 0.0025" rgba="0.5 0.6 0.7 1"/>
    <geom name="support_c" type="box" pos="0 0 0.3975" size="0.20 0.006 0.0025" rgba="0.5 0.6 0.7 1"/>
    <geom name="support_r" type="box" pos="0 0.027 0.3975" size="0.20 0.003 0.0025" rgba="0.5 0.6 0.7 1"/>
    <!-- 暂存站 (T03 第二货位): 台面顶 z=45mm; 台下净空 z=0..44mm 供底盘通过,
         叉(±15mm)从台下中心穿过; 立柱在台面四角 y=±48mm, 高于底盘宽度的叉通道两侧 -->
    <geom name="stage_top" type="box" pos="-0.340 0 0.0445" size="0.060 0.050 0.0005" rgba="0.4 0.5 0.4 1"/>
    <geom name="stage_leg_l" type="box" pos="-0.395 -0.048 0.022" size="0.003 0.003 0.022" rgba="0.3 0.4 0.3 1"/>
    <geom name="stage_leg_r" type="box" pos="-0.395 0.048 0.022" size="0.003 0.003 0.022" rgba="0.3 0.4 0.3 1"/>
    <!-- 托盘 自由体, 原点=底面(底条局部z 0..4) -->
    <body name="pallet" pos="0 0 0.401">
      <freejoint name="pal"/>
      <geom name="pal_bottom_l" type="box" pos="0 -0.027 0.002" size="0.040 0.003 0.002" mass="0.006"/>
      <geom name="pal_bottom_r" type="box" pos="0 0.027 0.002" size="0.040 0.003 0.002" mass="0.006"/>
      <geom name="pal_bottom_c" type="box" pos="0 0 0.002" size="0.040 0.006 0.002" mass="0.010"/>
      <!-- 通道顶壁: 独立条带 (y=±15±5, z 12..15mm)。叉顶面推的是通道顶壁而非整块顶板,
           独立几何才能让接触求解生成多点分布接触 (整块顶板底面只会在叉尖成线)。 -->
      <geom name="ceil_l" type="box" pos="0 -0.015 0.0135" size="0.040 0.005 0.0015" mass="0.008"/>
      <geom name="ceil_r" type="box" pos="0 0.015 0.0135" size="0.040 0.005 0.0015" mass="0.008"/>
      <!-- 顶板其余部分: 通道两侧边带 + 中桥, 避开叉的插入面 -->
      <geom name="top_side_l" type="box" pos="0 -0.027 0.015" size="0.040 0.003 0.003" mass="0.002"/>
      <geom name="top_side_r" type="box" pos="0 0.027 0.015" size="0.040 0.003 0.003" mass="0.002"/>
      <geom name="top_mid" type="box" pos="0 0 0.017" size="0.040 0.010 0.001" mass="0.002"/>
      <!-- 叉从 -X 侧贯通进入; 真实叉孔贯通结构无背面挡块 -->
      <geom name="pal_back_l" type="box" pos="-0.0375 -0.027 0.002" size="0.0025 0.003 0.002" mass="0.0005" condim="1"/>
      <geom name="pal_back_r" type="box" pos="-0.0375 0.027 0.002" size="0.0025 0.003 0.002" mass="0.0005" condim="1"/>
    </body>
    <!-- 叉车: 平面底盘(滑动x+铰z) + 升降滑轨; T05 稳定性另用自由体变体 -->
    <body name="chassis" pos="{k['start_axle_x']} 0 0">
      <joint name="chassis_x" type="slide" axis="1 0 0" range="-0.8 0.5" damping="20"/>
      <joint name="chassis_yaw" type="hinge" axis="0 0 1" range="-0.2 0.2" damping="5"/>
      <geom name="chassis_body" type="box" pos="-0.085 0 0.0175"
            size="0.110 0.075 0.0175" mass="1.50"/>
      <geom name="counterweight" type="box" pos="-0.20 0 0.0175"
            size="0.010 0.060 0.0175" mass="0.0"/>
      <body name="carriage" pos="0 0 0">
        <joint name="lift" type="slide" axis="0 0 1" range="0 {k['lift_travel']}" damping="60"/>
        <geom name="carriage_plate" type="box" pos="0.020 0 0.030"
              size="0.010 0.060 0.008" mass="0.25"/>
        <geom name="fork_l" type="box" pos="{k['fork_root_x'] + k['fork_len']/2} -0.015 {k['transport_z'] - 0.00125}"
              size="{k['fork_len']/2} 0.005 0.00125" mass="0.02" friction="0.8 0.02 0.0002"/>
        <geom name="fork_r" type="box" pos="{k['fork_root_x'] + k['fork_len']/2} 0.015 {k['transport_z'] - 0.00125}"
              size="{k['fork_len']/2} 0.005 0.00125" mass="0.02" friction="0.8 0.02 0.0002"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="drive_x" joint="chassis_x" kp="800" forcerange="-20 20"/>
    <position name="lift" joint="lift" kp="10000" forcerange="-5 5"/>
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

    def pallet_fork_contact(self) -> bool:
        fg, pg = set(self.fork_geoms), set(self.pallet_geoms)
        return any((g1 in fg and g2 in pg) or (g2 in fg and g1 in pg)
                   for g1, g2, _ in self.contacts())

    def support_load_ratio(self) -> float:
        """支撑面承担的竖向力 / 托盘总重 (放置判据)。

        mj_contactForce 返回的是"施加于 geom1 的力"; 支撑梁 geom1 被托盘向下压,
        故取 -fz (向上为正)。"""
        total = 0.0
        f1 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 not in self.support_gids and g2 not in self.support_gids:
                continue
            if g1 in self.support_gids and g2 in self.pallet_geoms:
                sign = -1.0      # 支撑梁被压 → 力向下, 取反得托盘受到的向上托力
            elif g2 in self.support_gids and g1 in self.pallet_geoms:
                sign = 1.0
            else:
                continue        # 非托盘对象不计入
            mujoco.mj_contactForce(self.model, self.data, i, f1)
            frame = c.frame.reshape(3, 3)
            f_world = frame @ f1[:3]
            total += max(0.0, sign * f_world[2])
        weight = 0.045 * 9.81
        return total / weight if weight else 0.0
