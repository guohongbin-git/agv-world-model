"""W08 机械臂单元 — 单臂 (M01) + 标准箱件 + 入料托台/空托盘。

与 S00 同样的纪律:
* 只用 actuator (position/motor) 驱动, 不写 qpos, 不用 weld, 不用运动学挂载;
* 箱件是自由刚体, 夹持靠指垫法向接触 + 摩擦;
* 成功判据来自 MuJoCo 观测 (接触力、位置误差、承托)。

设计 25.3:
* 箱件 60×40×50mm, 200g; 抓取 TCP z = 箱底 + 25mm (指垫中心)
* 抬离 ≥50mm, 位置漂移 ≤1mm/角度 ≤1° 持续 0.5s → GRASP_CONFIRMED
* 落托后目标承托 ≥90%, 开爪撤离 50mm 后箱件仍在目标
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from configs.loader import load


@dataclass
class ArmState:
    qpos: np.ndarray
    tcp_pos: np.ndarray
    tcp_mat: np.ndarray
    jacp: np.ndarray
    jacr: np.ndarray


class ManipulatorCell:
    """S08 机械臂单元: 单臂六轴 + 夹爪 + 箱件 + 托台 + 空托盘。"""

    CTRL_HZ = 100

    def __init__(self, timestep_s: float = 0.0005, arm: str = "M01", seed: int = 0):
        self.cfg = load()
        self.arm = arm
        self.model = mujoco.MjModel.from_xml_string(self._xml(timestep_s))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        m = self.model
        self.j_names = [f"{arm}_{j}" for j in ("J1", "J2", "J3", "J4", "J5", "J6")]
        self.jidx, self.qadr, self.dadr = [], [], []
        for jn in self.j_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self._p(jn)}_jn")
            self.jidx.append(jid)
            self.qadr.append(m.jnt_qposadr[jid])
            self.dadr.append(m.jnt_dofadr[jid])
        self.finger_names = [f"{arm}_finger_l", f"{arm}_finger_r"]
        self.f_jidx, self.f_qadr = [], []
        for fn in self.finger_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self._p(fn)}_jn")
            self.f_jidx.append(jid)
            self.f_qadr.append(m.jnt_qposadr[jid])
        self.act_j = [self._act(f"{jn}_drv") for jn in self.j_names]
        self.act_f = [self._act(f"{fn}_drv") for fn in self.finger_names]
        self.tcp_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{self._p(arm)}_tcp")
        self.box_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_goods")
        self.pad_gids = [self._geom(f"{arm}_pad_l"), self._geom(f"{arm}_pad_r")]
        self.tray_top_z = 0.055
        # 派生几何 (来自 configs)
        g = self.cfg["goods"]["size_m"]
        self.box_size = tuple(g)
        self.box_mass = self.cfg["goods"]["mass_kg"]
        self.tcp_speed_max = self.cfg["manipulators"]["tcp_speed_max_mps"]
        self.contact_speed_max = self.cfg["manipulators"]["contact_approach_speed_max_mps"]
        self.grip = self.cfg["manipulators"]["gripper"]
        self.stow_r = self.cfg["manipulators"]["stow_xy_radius_initial_m"]

    # ------------------------------------------------------------ helpers
    def _p(self, name: str) -> str:
        """CAD 生成的实例前缀: M01_J1_J2_... 形式。"""
        if name.startswith(self.arm):
            return name
        return f"{self.arm}_{name}"

    def _find_prefixed(self, kind, suffix: str) -> int:
        m = self.model
        for i in range(m.njnt if kind == "joint" else 0):
            pass
        return -1

    def _act(self, name: str) -> int:
        m = self.model
        for a in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if n and n.endswith(name):
                return a
        raise KeyError(name)

    def _geom(self, name: str) -> int:
        m = self.model
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            raise KeyError(name)
        return g

    # ------------------------------------------------------------ state
    @property
    def q(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.qadr])

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_sid].copy()

    @property
    def tcp_mat(self) -> np.ndarray:
        return self.data.site_xmat[self.tcp_sid].reshape(3, 3).copy()

    @property
    def gap(self) -> float:
        return float(self.data.qpos[self.f_qadr[0]] + self.data.qpos[self.f_qadr[1]])

    @property
    def box_pos(self) -> np.ndarray:
        return self.data.xpos[self.box_bid].copy()

    @property
    def box_bottom_z(self) -> float:
        return float(self.data.xpos[self.box_bid][2] - self.box_size[2] / 2)

    def jacobian(self, q: np.ndarray | None = None, d=None) -> tuple[np.ndarray, np.ndarray]:
        """TCP 雅可比 (6 列)。用中心差分数值雅可比: 确定性且不依赖 mj_jacSite 的
        约束状态 (设计 25.4 要求雅可比来自模型, 数值微分满足同一目的)。"""
        q0 = self.q if q is None else np.asarray(q, dtype=float)
        h = 1e-5
        dp = np.zeros((3, 6))
        dr = np.zeros((3, 6))
        for i in range(6):
            qa, qb = q0.copy(), q0.copy()
            qa[i] += h
            qb[i] -= h
            sa, sb = self.fk(qa), self.fk(qb)
            dp[:, i] = (sa.tcp_pos - sb.tcp_pos) / (2 * h)
            dr[:, i] = _rotvec(sa.tcp_mat @ sb.tcp_mat.T) / (2 * h)
        return dp, dr

    # ------------------------------------------------------------ IK
    def fk(self, q: np.ndarray) -> ArmState:
        """在独立规划状态上做 FK (不写活动物理状态, 设计 25.4)。"""
        d2 = mujoco.MjData(self.model)
        d2.qpos[:] = self.data.qpos
        for a, val in zip(self.qadr, q):
            d2.qpos[a] = val
        mujoco.mj_kinematics(self.model, d2)
        return ArmState(
            qpos=np.array(q), tcp_pos=d2.site_xpos[self.tcp_sid].copy(),
            tcp_mat=d2.site_xmat[self.tcp_sid].reshape(3, 3).copy(),
            jacp=np.zeros((3, 6)), jacr=np.zeros((3, 6)),
        )

    def ik(self, target_pos: np.ndarray, target_mat: np.ndarray | None = None,
           q_init: np.ndarray | None = None, seeds: int = 8,
           max_iter: int = 200) -> tuple[np.ndarray, bool, dict]:
        """阻尼最小二乘 IK (位置尺度 50mm, 旋转尺度 1rad, 8 个确定性种子)。

        判据: 位置残差 ≤1mm, 旋转 ≤2°, 关节距硬限位 ≥0.05rad。
        """
        ik_cfg = self.cfg["manipulators"]["ik"]
        pos_scale = ik_cfg["pos_scale_m"]
        rot_scale = ik_cfg["rot_scale_rad"]
        lo = np.array([self.model.jnt_range[j][0] for j in self.jidx])
        hi = np.array([self.model.jnt_range[j][1] for j in self.jidx])
        margin = ik_cfg["joint_limit_margin_rad"]
        rng = np.random.default_rng(0)
        best_q, best_cost, best_info = None, float("inf"), {}
        starts = []
        if q_init is not None:
            starts.append(np.array(q_init))
        while len(starts) < seeds:
            starts.append(lo + rng.random(6) * (hi - lo))
        for q0 in starts[:seeds]:
            q = np.clip(np.array(q0), lo + margin, hi - margin)
            for _ in range(max_iter):
                st = self.fk(q)
                Jp, Jr = self.jacobian(q)
                dp = target_pos - st.tcp_pos
                err_p = np.linalg.norm(dp)
                ang = 0.0
                v_err = np.zeros(3)
                if target_mat is not None:
                    R_err = target_mat @ st.tcp_mat.T
                    cosang = max(-1.0, min(1.0, (np.trace(R_err) - 1) / 2))
                    ang = math.acos(cosang)
                    v_err = _rotvec(R_err)
                    if err_p <= ik_cfg["pos_tol_m"] and ang <= ik_cfg["rot_tol_rad"]:
                        return q, True, {"pos_err_m": err_p, "rot_err_rad": ang}
                cost = (err_p / pos_scale) ** 2 + (ang / rot_scale) ** 2
                if cost < best_cost:
                    best_cost, best_q = cost, q.copy()
                J = np.vstack([Jp / pos_scale, Jr / rot_scale]) if target_mat is not None \
                    else Jp / pos_scale
                e = np.concatenate([dp / pos_scale, v_err / rot_scale]) if target_mat is not None \
                    else dp / pos_scale
                lam = max(1e-4, 1e-3 / max(np.linalg.norm(J), 1e-6))
                dq = np.linalg.solve(J.T @ J + lam ** 2 * np.eye(6), J.T @ e)
                n_dq = np.linalg.norm(dq)
                if n_dq > 0.2:
                    dq *= 0.2 / n_dq
                q = np.clip(q + dq, lo + margin, hi - margin)
        st = self.fk(best_q)
        err_p = float(np.linalg.norm(target_pos - st.tcp_pos))
        ang = 0.0
        if target_mat is not None:
            R = target_mat @ st.tcp_mat.T
            ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
        info = {"pos_err_m": err_p, "rot_err_rad": ang}
        ok = err_p <= ik_cfg["pos_tol_m"] and (target_mat is None or ang <= ik_cfg["rot_tol_rad"])
        return best_q, ok, info

    # ------------------------------------------------------------ 控制
    def set_joint_targets(self, q_target: np.ndarray, fingers: float | None = None) -> None:
        """纯位置伺服 (设计 25.4: 限转矩驱动 + 关节反馈)。

        不用重力前馈: ctrl = kp·Δq, 其中 kp 必须足够大以便在重力负载下
        建立静力 (稳态误差 = |重力矩| / kp, 要求 <0.01rad → kp ≳ 200)。
        力矩上限由 forcerange 给定 (设计 25.2 的转矩限值)。
        """
        kp = 2000.0
        for i, a in enumerate(self.act_j):
            err = q_target[i] - self.data.qpos[self.qadr[i]]
            self.data.ctrl[a] = kp * err
        if fingers is not None:
            for a in self.act_f:
                self.data.ctrl[a] = fingers / 2
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def move_joints(self, q_target: np.ndarray, max_vel: float = 0.5,
                    max_acc: float = 1.0, settle_s: float = 0.5) -> dict:
        """关节空间受限轨迹 (梯形速度剖面) — 不写 qpos, 只给 position 执行器。"""
        q0 = self.q
        n = int(max(1, np.max(np.abs(q_target - q0)) / max_vel / (1.0 / self.CTRL_HZ)))
        traj = []
        v = np.zeros(6)
        cur = q0.copy()
        for i in range(n):
            t = i / self.CTRL_HZ
            v_next = np.clip((q_target - cur) * self.CTRL_HZ, -max_vel, max_vel)
            v_next = np.clip(v_next - v, -max_acc * 0.01, max_acc * 0.01)
            cur = cur + v_next * 0.01
            v = v_next
            traj.append(cur.copy())
            self.set_joint_targets(cur, self.gap)
            self.step(self.substeps)
            if np.max(np.abs(q_target - cur)) < 1e-4:
                break
        self.set_joint_targets(q_target, self.gap)
        for _ in range(int(settle_s * self.CTRL_HZ)):
            self.step(self.substeps)
        return {"q": self.q, "tcp": self.tcp_pos}

    @property
    def substeps(self) -> int:
        return max(1, round(0.01 / self.model.opt.timestep))

    # ------------------------------------------------------------ 观测
    def finger_contacts(self) -> tuple[bool, bool, float, float]:
        """两指是否分别接触箱件 + 各自法向力 (N)。"""
        forces = []
        contact = []
        f = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.box_bid not in (c.geom1 // 1000, c.geom2 // 1000):
                pass
            g1_body = self.model.geom_bodyid[c.geom1]
            g2_body = self.model.geom_bodyid[c.geom2]
            if self.box_bid in (g1_body, g2_body) and (c.geom1 in self.pad_gids or c.geom2 in self.pad_gids):
                mujoco.mj_contactForce(self.model, self.data, i, f)
                forces.append(float(np.linalg.norm(f[:3])))
                contact.append(c.geom1 in self.pad_gids)
        left = any(fl and not fr for fl, fr in zip(contact, contact)) or \
            (len(forces) >= 1 and contact and contact[0])
        right = len(forces) >= 2
        return (bool(left), bool(right),
                float(forces[0]) if forces else 0.0,
                float(forces[1]) if len(forces) > 1 else 0.0)

    def box_supported_ratio(self) -> float:
        """箱件底面承托的竖向力 / 箱重 (落托判据 ≥0.90)。"""
        total = 0.0
        f = np.zeros(6)
        tray_gids = {self._geom(f"{self.arm}_tray_top")}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if not ({c.geom1, c.geom2} & tray_gids):
                continue
            if self.model.geom_bodyid[c.geom1] == self.box_bid or \
               self.model.geom_bodyid[c.geom2] == self.box_bid:
                mujoco.mj_contactForce(self.model, self.data, i, f)
                frame = c.frame.reshape(3, 3)
                fz = (frame @ f[:3])[2]
                if self.model.geom_bodyid[c.geom1] == self.box_bid:
                    total += max(0.0, fz)
                else:
                    total += max(0.0, -fz)
        return total / (self.box_mass * 9.81) if self.box_mass else 0.0

    def relative_drift(self, dt: float = 0.5) -> tuple[float, float]:
        """箱件相对 TCP 的位置/角度漂移 (GRASP_CONFIRMED 判据 ≤1mm/1°)。"""
        q = self.q
        st = self.fk(q)
        R = st.tcp_mat
        rel0 = R.T @ (self.box_pos - st.tcp_pos)
        p0 = rel0.copy()
        n = int(dt / (self.model.opt.timestep * self.substeps))
        drifts = []
        angs = []
        for _ in range(n):
            self.step(self.substeps)
            st = self.fk(self.q)
            rel = st.tcp_mat.T @ (self.box_pos - st.tcp_pos)
            drifts.append(float(np.linalg.norm(rel - p0)))
            cosang = max(-1.0, min(1.0, float(np.dot(rel, p0) / (np.linalg.norm(rel) * np.linalg.norm(p0) + 1e-12))))
            angs.append(math.degrees(math.acos(cosang)))
        return max(drifts) if drifts else 0.0, max(angs) if angs else 0.0

    def max_penetration(self) -> float:
        return max((-float(self.data.contact[i].dist) for i in range(self.data.ncon)),
                   default=0.0)

    # ------------------------------------------------------------ XML
    def _xml(self, dt: float) -> str:
        mp = self.cfg["manipulators"]
        g = self.cfg["goods"]["size_m"]
        gm = self.cfg["goods"]["mass_kg"]
        gr = self.cfg["manipulators"]["gripper"]
        j = mp["joints"]
        a = self.arm
        base = self.cfg["regions"][f"m01_base_m" if a == "M01" else "m02_base_m"]
        feed = self.cfg["regions"]["feed_i_tray" if a == "M01" else "feed_o_tray"]
        box_z = feed["top_z_m"] + g[2] / 2

        def jxml(spec, i, prefix):
            off = spec["offset_m"]
            return f"""
    <body name="{prefix}_{spec['name']}" pos="{off[0]} {off[1]} {off[2]}">
      <joint name="{prefix}_{spec['name']}_jn" type="hinge" axis="{spec['axis'][0]} {spec['axis'][1]} {spec['axis'][2]}"
             range="{spec['lower_rad']} {spec['upper_rad']}" damping="2" armature="0.01"/>
      <geom name="{prefix}_{spec['name']}_link" type="capsule" size="0.018"
            fromto="0 0 0 {off[0]} {off[1]} {off[2]}" mass="{0.10}"/>
      <inertial pos="{off[0]/2} {off[1]/2} {off[2]/2}" mass="{0.10}"
                diaginertia="0.0002 0.0002 0.0001"/>"""

        tcp = mp["tcp_offset_m"]
        palm = gr["palm_size_m"]
        pad_x, pad_z = gr["pad_size_m"][0], gr["pad_size_m"][1]      # 10 × 15mm
        fng = gr["finger_size_m"]                                    # 6 × 10 × 45mm
        # 指垫中心在 TCP 下方 25mm (抓取 TCP z = 箱底 + 25mm, 设计 25.3)
        pad_center_z = -0.025
        fi = "        "          # 手指位于 tool 内 (8 空格)
        fingers = ""
        for sign, side in ((1, "l"), (-1, "r")):
            fingers += f"""
{fi}<body name="{a}_finger_{side}" pos="0 {sign * 0.0375} 0">
{fi}  <joint name="{a}_finger_{side}_jn" type="slide" axis="0 {sign} 0"
{fi}         range="0 {gr['gap_range_m'][1]}" damping="1" armature="0.001"/>
{fi}  <geom name="{a}_finger_{side}_link" type="box" size="{fng[0]/2} {fng[1]/2} {fng[2]/2}"
{fi}        pos="0 0 {-fng[2]/2 + 0.0075}" mass="0.02"/>
{fi}  <geom name="{a}_pad_{side}" type="box" size="{pad_x/2} 0.0015 {pad_z/2}"
{fi}        pos="0 {sign * 0.0015} {pad_center_z}" mass="0.005" friction="0.6 0.01 0.0002"/>
{fi}</body>"""

        def make_tool_block(indent: str) -> str:
            return f"""
{indent}  <body name="{a}_tool" pos="{tcp[0]} 0 0">
{indent}    <geom name="{a}_palm" type="box" size="{palm[0]/2} {palm[1]/2} {palm[2]/2}"
{indent}          pos="0 0 {gr['palm_center_above_tcp_z_m']}" mass="0.08"/>
{indent}    <inertial pos="0 0 {gr['palm_center_above_tcp_z_m']}" mass="0.08"
{indent}              diaginertia="1e-4 1e-4 1e-4"/>
{indent}    <site name="{a}_tcp" pos="0 0 0" size="0.005" rgba="0 1 0 0.6"/>
{fingers}
{indent}  </body>"""

        # 执行器: MuJoCo 3.15 的 joint= 传动默认取"父链", 必须用 jointinparent
        # 才把力矩施加在关节自身 body 上 (否则施在固定的 base 上被 weld 锁死)
        # 转矩上限: 设计 25.2 初值 ×2; J3 肘关节需承担前臂+腕+夹爪+箱件
        # (合计 ~0.5kg × 0.25m ≈ 1.2 N·m 静态, 长力臂动态更高), 故再给 1.5× 余量。
        kp_gain = 2000.0
        trq_scale = {"J1": 2.0, "J2": 2.0, "J3": 3.0, "J4": 2.5, "J5": 2.5, "J6": 3.0}
        actuators = "".join(
            f'\n    <position name="{a}_{s["name"]}_drv" jointinparent="{a}_{s["name"]}_jn" '
            f'kp="{kp_gain}" forcerange="-{trq_scale[s["name"]] * s["torque_max_nm"]:.1f} '
            f'{trq_scale[s["name"]] * s["torque_max_nm"]:.1f}"/>'
            for s in j) + "".join(
            f'\n    <position name="{a}_finger_{sd}_drv" jointinparent="{a}_finger_{sd}_jn" '
            f'kp="400" forcerange="-8 8"/>' for sd in ("l", "r"))

        # 关节链: 递归生成, 保证开闭配对; tool 挂在最后一个关节 (J6) 内
        def emit_chain(i: int, indent: str) -> str:
            if i >= len(j):
                return ""
            spec = j[i]
            pfx = f"{a}_{spec['name']}"
            off = spec["offset_m"]
            # 连杆 geom 只覆盖本关节到下一关节之间的一小段 (12mm), 避免相邻连杆自碰;
            # 质量/惯量由显式 <inertial> 给出, 与 geom 无关
            half_x = 0.012
            z_half = 0.013
            mass = 0.10 if i == 0 else 0.08
            head = "" if i > 0 else f"""
    <body name="{a}_base" pos="{base[0]} {base[1]} {base[2]}">
      <geom name="{a}_base_geom" type="cylinder" size="0.04 0.0175" pos="0 0 0.0175" mass="0.3"/>
      <site name="{a}_tcp_home" pos="0 0 0" size="0.004" rgba="1 0 0 0.3"/>"""
            inner = emit_chain(i + 1, indent + "  ")
            if i == len(j) - 1:
                inner = make_tool_block(indent) + inner
            return head + f"""
{indent}<body name="{pfx}" pos="{off[0]} {off[1]} {off[2]}">
{indent}  <joint name="{pfx}_jn" type="hinge" axis="{spec['axis'][0]} {spec['axis'][1]} {spec['axis'][2]}"
{indent}         range="{spec['lower_rad']} {spec['upper_rad']}" damping="0.5" armature="0.002"/>
{indent}  <geom name="{pfx}_link" type="box" size="{half_x} {z_half} {z_half}"
{indent}        pos="{half_x} 0 0" mass="{mass}" contype="0" conaffinity="0"/>
{indent}  <inertial pos="{off[0]/2 if off[0] else 0.1} 0 0" mass="{mass}"
{indent}          diaginertia="1e-4 1e-4 1e-4"/>
{inner}{indent}</body>"""

        chain_open = emit_chain(0, "    ") + "\n    </body>"   # base body 闭合

        return f"""
<mujoco model="{a}_cell">
  <!-- MuJoCo 3.15 起 <compiler angle> 默认为 degree; 本项目统一 SI 弧度 -->
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 -9.81" iterations="200"
          tolerance="1e-10" noslip_iterations="30"/>
  <size nconmax="300" njmax="400"/>
  <default>
    <geom condim="3" friction="0.5 0.005 0.0001" solref="0.0002 1" solimp="0.999 0.9999 0.00001"/>
    <joint damping="1"/>
  </default>
  <worldbody>
    <light pos="{base[0]} {base[1]} 0.9" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="2 2 0.05" friction="0.5"/>
    <!-- 入料托台 (面 z=55mm): 扁平托盘贴桌面, 支腿在 y 远端;
         机械臂从 -Y 侧接近时不与支腿干涉 (支腿仅占台后缘) -->
    <geom name="{a}_tray_leg" type="box" pos="{feed['x_m']} {feed['y_m'] - 0.045} 0.0265"
          size="0.045 0.004 0.0265" rgba="0.4 0.45 0.5 1"/>
    <geom name="{a}_tray_top" type="box" pos="{feed['x_m']} {feed['y_m']} {feed['top_z_m'] - 0.0025}"
          size="0.05 0.04 0.0025" rgba="0.55 0.6 0.65 1"/>
    <!-- 箱件 (自由刚体 200g, 设计 25.3) -->
    <body name="{a}_goods" pos="{feed['x_m']} {feed['y_m']} {box_z}">
      <freejoint name="{a}_goods_fj"/>
      <geom name="{a}_goods_geom" type="box" size="{g[0]/2} {g[1]/2} {g[2]/2}"
            mass="{gm}" friction="0.4 0.01 0.0002"/>
    </body>
    <!-- 六轴链 + 掌体 + 双指 (链尾挂 tool) -->
{chain_open}
  </worldbody>
  <actuator>{actuators}
  </actuator>
</mujoco>
"""


def _rotvec(R_err: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 旋转向量 (轴角)。"""
    cos = (np.trace(R_err) - 1) / 2
    cos = max(-1.0, min(1.0, cos))
    ang = math.acos(cos)
    if ang < 1e-9:
        return np.zeros(3)
    ax = np.array([R_err[2, 1] - R_err[1, 2],
                   R_err[0, 2] - R_err[2, 0],
                   R_err[1, 0] - R_err[0, 1]]) / (2 * math.sin(ang))
    return ax * ang
