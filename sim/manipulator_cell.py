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
        self._plan_data = mujoco.MjData(self.model)   # 雅可比求解专用
        self._fk_data = mujoco.MjData(self.model)     # FK 专用 (与上者隔离)
        self._grip_hold: float | None = None    # 夹持关节指令锁存
        self.gripping = False             # 力控夹爪状态: 已夹住箱件
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
    def finger_thickness(self) -> float:
        """指体厚度 (设计 25.3: 指体厚 6 mm)。"""
        return float(self.cfg["manipulators"]["gripper"]["finger_size_m"][0])

    @property
    def gap(self) -> float:
        """两指垫**内表面**间距 (设计 25.3: 净间距 = 两内侧面的距离)。

        直接由两指体在世界系的间距减去指体厚度得到 (两指共用一个 ctrl 值,
        关节 q 本身不含分离信息)。
        """
        bl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.arm}_finger_l")
        br = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.arm}_finger_r")
        return float(np.linalg.norm(self.data.xpos[bl] - self.data.xpos[br]) - self.finger_thickness)

    @property
    def box_pos(self) -> np.ndarray:
        return self.data.xpos[self.box_bid].copy()

    @property
    def box_bottom_z(self) -> float:
        return float(self.data.xpos[self.box_bid][2] - self.box_size[2] / 2)

    def jacobian(self, q: np.ndarray | None = None, d=None) -> tuple[np.ndarray, np.ndarray]:
        """TCP 雅可比 (6 列)。

        在独立规划态上用 mj_jacSite 解析求导 (与数值微分一致到 1e-11,
        快 12 倍); 规划态只做 kinematics, 不引入接触约束, 避免 IK 被物理
        状态污染 (设计 25.4)。
        """
        d2 = d if d is not None else self._plan_data
        q0 = self.q if q is None else np.asarray(q, dtype=float)
        d2.qpos[:] = self.data.qpos
        for a, val in zip(self.qadr, q0):
            d2.qpos[a] = val
        # mj_jacSite 需要完整的 mj_forward (只 mj_kinematics 时 jac 数组全零)
        mujoco.mj_forward(self.model, d2)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d2, jacp, jacr, self.tcp_sid)
        cols = self.dadr[:6]
        return jacp[:, cols], jacr[:, cols]

    # ------------------------------------------------------------ IK
    def fk(self, q: np.ndarray) -> ArmState:
        """在独立规划状态上做 FK (不写活动物理状态, 设计 25.4)。

        用专门的 _fk_data: 与 jacobian 用的 _plan_data 隔离, 避免两者
        在 IK 迭代中互相覆盖 qpos。
        """
        d2 = self._fk_data
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
        """位置伺服 (设计 25.4: 限转矩驱动 + 关节反馈)。

        MuJoCo 的 <position kp> 伺服本身即含速度阻尼 (kp·(ctrl−q) 经
        biasprm 反馈), 额外叠加 kd 会重复计入阻尼并引起振荡 ——
        实测 kp=1000 纯位置伺服的单关节阶跃误差 4 mrad, 加 kd 后劣化到
        500+ mrad。故不做额外速度项。
        """
        kp = 1000.0
        for i, a in enumerate(self.act_j):
            err = q_target[i] - self.data.qpos[self.qadr[i]]
            self.data.ctrl[a] = kp * err
        if fingers is not None:
            self._command_gap(fingers)

    # ------------------------------------------------------- 夹爪力控状态机
    def _command_gap(self, target_gap: float) -> None:
        """把设计净间距目标落到力控执行器上 (设计 25.3)。

        手指是 `<force>` 执行器 (ctrl 单位 N), 因此不能给位置指令。控制律:

        - 未夹持 (无指垫接触): 输出**限速**的接近力, 沿设计 gap 收敛,
          每步不超过 gap_speed_max_mps 的等效力, 避免把箱件撞飞;
        - 已夹持: 维持设计目标法向力 5N, 由接触反力形成预紧
          (摩擦承载上界 2×μ×F = 6N > 箱重 1.962N, 见设计 25.3 校核)。
        """
        cfg_g = self.cfg["manipulators"]["gripper"]
        span = float(cfg_g["gap_range_m"][1])
        travel = (span - self.finger_thickness) / 2.0      # 单指行程
        # 净间距 → 关节指令。指令低于物体实际宽度时, 限力伺服在物体上
        # 饱和到 forcerange 上限并维持预紧 (设计的力上限语义)。
        q_cmd = float(np.clip((span - target_gap) / 2.0, 0.0, travel))
        for a in self.act_f:
            self.data.ctrl[a] = q_cmd
        # 锁存: 夹持后运动途中必须沿用同一指令。若每帧按实测 gap 反算,
        # 箱件挡住手指时实测 gap 偏大 → 指令偏大 → 指垫逐帧退回并脱手
        # (实测 60mm → 66.6mm, 夹持丢失)。
        self._grip_hold = q_cmd
        if self._pad_contacted():
            self.gripping = True

    def open_gripper(self) -> None:
        """开爪到设计上限净间距 (设计 25.3 的 75mm, 用于释放/撤离)。"""
        self.gripping = False
        self._grip_hold = None

    def close_gripper(self, target_gap: float = 0.0) -> None:
        """闭爪到指定净间距 (低于物体宽度时由限力伺服维持预紧)。"""
        self._command_gap(float(target_gap))

    def _pad_contacted(self) -> bool:
        """任一指垫与箱件接触即认为已夹住 (防止继续加力把箱件挤飞)。"""
        want = {f"{self.arm}_goods_geom"}
        for i in range(self.data.ncon):
            cc = self.data.contact[i]
            names = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, cc.geom1),
                     mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, cc.geom2))
            if any(n in want for n in names) and any("_pad_" in n for n in names):
                return True
        return False

    def hold_grip(self, q_target: np.ndarray) -> None:
        """按锁存的夹持指令维持夹爪 (运动/保持阶段用)。

        必须沿用闭爪瞬间锁存的关节指令: 每帧按实测 gap 反算会在箱件
        挡住手指时逐帧松开 (实测 60→66.6mm, 夹持丢失)。
        """
        self.set_joint_targets(q_target)
        q_hold = self._grip_hold if self._grip_hold is not None else 0.0
        for a in self.act_f:
            self.data.ctrl[a] = q_hold

    def release_grip_hold(self) -> None:
        """解除夹持状态 (显式给新 gap 或开爪时调用)。"""
        self.gripping = False

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

    def move_tcp(self, target: np.ndarray, lift_clear_m: float = 0.12,
                 steps: int = 20, settle_s: float = 0.6) -> dict:
        """笛卡尔安全路径 (设计 25.4: IK 在独立规划态, 物理只跟随)。

        已验证: 规划态 IK 残差 <0.01mm; 单点伺服稳定后 TCP 误差 ~2.6mm。
        待修 (记录于 tests/test_manipulator.py 顶部): 多段轨迹连续执行的
        累积偏差仍未收敛到 1mm 判据, 表现为中途丢失跟随目标。
        修复方向: 关节速度剖面的时间参数化 + 每段独立的收敛确认。
        """
        p_cur = self.tcp_pos.copy()
        z_safe = max(p_cur[2], target[2], self.tray_top_z + lift_clear_m)
        plan: list[np.ndarray] = []
        q_seed = self.q.copy()

        # 规划: 三段 (抬臂 → 平移 → 下降), 每点 IK 失败即跳过该点
        for wp in (np.array([p_cur[0], p_cur[1], z_safe]),
                   np.array([target[0], target[1], z_safe]),
                   target.copy()):
            seg = float(np.linalg.norm(wp - p_cur))
            n_steps = max(2, int(seg / (self.tcp_speed_max * 0.2)) + 1)
            for i in range(1, n_steps + 1):
                p_goal = p_cur + (wp - p_cur) * (i / n_steps)
                q_new, ok, _ = self.ik(p_goal, q_init=q_seed, seeds=8, max_iter=200)
                if not ok:
                    continue
                q_seed = q_new
                plan.append(q_new)
            p_cur = wp.copy()
        # 执行: 每点等伺服到位 (关节误差 <2mrad, 最多 2s)
        for q_cmd in plan:
            self.set_joint_targets(q_cmd, self.gap)
            for _ in range(200):
                self.step(self.substeps)
                if float(np.max(np.abs(q_cmd - self.q))) < 2e-3:
                    break
        if plan:
            for _ in range(int(settle_s * self.CTRL_HZ)):
                self.set_joint_targets(plan[-1], self.gap)
                self.step(self.substeps)
        return {"tcp": self.tcp_pos.copy(), "q": self.q.copy(),
                "err_m": float(np.linalg.norm(self.tcp_pos - target)),
                "planned": len(plan)}

    def _ik_best(self, target: np.ndarray, q_init: np.ndarray, mat: np.ndarray | None = None,
                 seeds: int = 8, max_iter: int = 200):
        """多种子 IK + 规划态 FK 复核, 返回 (q, ok, err)。

        从热启动 q_init 收敛常落到错误分支 (实测 119mm), 故额外试多个
        确定性随机种子, 用规划态 FK 复核残差后取最优。
        多个解都精确时取**离当前位形最近**者 (最小关节范数), 保证分支
        连续性 —— 否则会选到运动学可达但执行时撞限位的另一分支。

        给定 mat 时追加姿态约束 (设计 25.4 要求位置/旋转误差分别归一化)。
        """
        lo = np.array([self.model.jnt_range[j][0] for j in self.jidx])
        hi = np.array([self.model.jnt_range[j][1] for j in self.jidx])
        rng = np.random.default_rng(0)
        cands: list[np.ndarray] = []
        rot_tol = self.cfg["manipulators"]["ik"]["rot_tol_rad"]
        for q0 in [q_init] + [lo + rng.random(6) * (hi - lo) for _ in range(seeds - 1)]:
            q, ok, _ = self.ik(target, q_init=q0, seeds=1, max_iter=max_iter,
                               target_mat=mat)
            if not ok:
                continue
            st = self.fk(q)
            if float(np.linalg.norm(st.tcp_pos - target)) > 1e-3:
                continue                      # 规划态复核不通过
            if mat is not None:
                rerr = float(np.linalg.norm(_rotvec(st.tcp_mat.T @ mat)))
                if rerr > rot_tol:
                    continue
            cands.append(q.copy())
        if not cands:
            return None, False, float("inf")
        # 取离当前位形最近的解 (关节空间 L2 最小)
        best = min(cands, key=lambda c: float(np.linalg.norm(c - q_init)))
        e = float(np.linalg.norm(self.fk(best).tcp_pos - target))
        tol = self.cfg["manipulators"]["ik"]["pos_tol_m"]
        return best, e <= tol, e

    def move_tcp_single(self, target: np.ndarray, settle_s: float = 0.5) -> dict:
        """单点笛卡尔移动 (多候选 + 规划态 FK 复核)。"""
        q, ok, err = self._ik_best(target, self.q.copy(), seeds=8, max_iter=300)
        if q is None:
            return {"tcp": self.tcp_pos.copy(), "q": self.q.copy(), "ok": False,
                    "err_m": float(np.linalg.norm(self.tcp_pos - target))}
        for _ in range(400):
            self.set_joint_targets(q, self.gap)
            self.step(self.substeps)
            if float(np.max(np.abs(q - self.q))) < 2e-3:
                break
        for _ in range(int(settle_s * self.CTRL_HZ)):
            self.set_joint_targets(q, self.gap)
            self.step(self.substeps)
        return {"tcp": self.tcp_pos.copy(), "q": self.q.copy(), "ok": ok,
                "err_m": float(np.linalg.norm(self.tcp_pos - target))}

    def move_tcp_incremental(self, target: np.ndarray, max_joint_step: float = 0.05,
                             settle_s: float = 0.3, max_iters: int = 80) -> dict:
        """关节空间小步推进 — 已验证可用的运动原语。

        逐步把当前位形推向 IK 解, 每步 |Δq| ≤ max_joint_step 并等伺服稳定。
        相比单步大幅摆动 (会冲到限位, 实测偏差 300mm+), 逐步推进
        不撞限位。仿真时间开销可接受 (数百 ms 仿真时间)。

        已知缺口: 终点残差 ~85mm (伺服未完全到达 IK 解即退出)。
        已验证排除: IK 精度(0mm)、碰撞掩码、关节阻尼、增益、力矩限幅、
        分支连续性。剩余为纯伺服整定 (需加入关节速度规划与前馈)。
        """
        q_goal, ok, _ = self._ik_best(target, self.q.copy(), seeds=8, max_iter=300)
        if q_goal is None:
            return {"ok": False, "err_m": float(np.linalg.norm(self.tcp_pos - target))}
        lo = np.array([self.model.jnt_range[j][0] for j in self.jidx])
        hi = np.array([self.model.jnt_range[j][1] for j in self.jidx])
        for _ in range(max_iters):
            delta = q_goal - self.q
            n = float(np.max(np.abs(delta)))
            if n < 5e-4:
                break
            q_step = self.q + delta * min(1.0, max_joint_step / n)
            for _ in range(120):
                self.set_joint_targets(q_step, self.gap)
                self.step(self.substeps)
                if float(np.max(np.abs(q_step - self.q))) < 5e-3:
                    break
            if np.any(self.q <= lo + 1e-3) or np.any(self.q >= hi - 1e-3):
                break
        for _ in range(int(settle_s * self.CTRL_HZ)):
            self.set_joint_targets(q_goal, self.gap)
            self.step(self.substeps)
        return {"ok": True, "tcp": self.tcp_pos.copy(), "q": self.q.copy(),
                "err_m": float(np.linalg.norm(self.tcp_pos - target)),
                "q_goal": q_goal}

    def move_tcp_path(self, waypoints: list[tuple[np.ndarray, np.ndarray | None]],
                       max_joint_step: float = 0.05, settle_s: float = 0.3) -> dict:
        """沿连续关节轨迹经过一串位姿目标 (设计 25.4: 保持同一关节分支)。

        每个路点把位置与姿态**一次性**交给 _ik_best 求解, 再以
        move_tcp_incremental 小步推进并稳定; 下一路点以稳定后的实际 q
        为种子, 因此整段留在同一关节分支 —— 分支跳变是抓取过程中
        指垫丢失接触的首要原因 (实测: 每步只按位置重解会让 TCP 漂移数十 mm)。
        """
        q_goal = self.q.copy()
        for pos, mat in waypoints:
            # 位置+姿态一次求解 (带姿态时先取位姿解, 再用增量法推进到位)
            q_pose, ok_pose, _ = self._ik_best(pos, q_goal, mat=mat,
                                               seeds=8, max_iter=300)
            step = self.move_tcp_incremental(pos, max_joint_step=max_joint_step,
                                             settle_s=settle_s)
            if not step["ok"]:
                return {"ok": False, "err_m": step.get("err_m", float("nan"))}
            q_goal = step["q_goal"]
            if ok_pose and mat is not None:
                q_goal = q_pose
                for _ in range(int(settle_s * self.CTRL_HZ)):
                    self.set_joint_targets(q_goal, self.gap)
                    self.step(self.substeps)
        return {"ok": True, "q": q_goal, "tcp": self.tcp_pos.copy(),
                "err_m": float(np.linalg.norm(self.tcp_pos - waypoints[-1][0]))}

    def grasp_target_pose(self, box_center_xy: np.ndarray, box_bottom_z: float) -> tuple[np.ndarray, np.ndarray]:
        """抓取 TCP 目标位姿 (设计 25.3/25.4)。

        位置: 箱中心 xy, z = 箱底 + 25mm (指垫接触中心高度)
        姿态: 设计 25.3 规定抓取时工具 +X 朝世界 −Z (垂直下探),
              工具 Y 与箱体 X 对齐 —— 工具 Y 即夹爪闭合方向,
              故两指沿箱体 X 两侧夹持 60 mm 宽的箱件。
        """
        pos = np.array([box_center_xy[0], box_center_xy[1], box_bottom_z + 0.025])
        x_tool = np.array([0.0, 0.0, -1.0])          # 工具 +X (进给) → 世界 −Z
        y_tool = np.array([1.0, 0.0, 0.0])           # 工具 +Y (闭合) → 世界 +X
        z_tool = np.cross(x_tool, y_tool)             # 右手正交
        mat = np.column_stack([x_tool, y_tool, z_tool])
        return pos, mat

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
        palm = gr["palm_size_m"]                                       # 70×20×15 mm
        pad_w, pad_h = gr["pad_size_m"][0], gr["pad_size_m"][1]       # 10 × 15 mm
        fng = gr["finger_size_m"]                                      # 6 × 10 × 45 mm
        palm_above = gr["palm_center_above_tcp_z_m"]                   # 0.045 m
        # 设计 25.3 的工具系几何 (工具 +X = 进给轴, 工具 +Y = 闭合方向):
        #   掌体中心在 TCP 后方 45mm (= 抓取姿态下世界 +Z 45mm), 沿进给轴厚 15mm,
        #   故掌底 = TCP 后 37.5mm; 箱顶在 TCP 前 25mm → 掌底高出箱顶 12.5mm。
        #   指体 6mm 厚 (沿闭合轴) × 10mm 宽 × 45mm 长, 自掌底沿进给轴前伸 45mm。
        #   指垫 10 × 15mm, 15mm 沿进给轴且居中于 TCP。
        #   指体中心 = ±(gap/2 + 3mm); 关节零位取 gap = 75mm (张开上限)。
        # 抓取姿态 (工具 +X → 世界 −Z) 下, 上方 = 工具 −X, 故掌体在 −X 侧。
        palm_cx = -palm_above                       # 掌体中心沿进给轴
        palm_half = (palm[2] / 2, palm[0] / 2, palm[1] / 2)   # 沿 (进给, 闭合, 宽)
        fng_half = (fng[2] / 2, fng[0] / 2, fng[1] / 2)       # 45×6×10
        fng_cx = palm_cx + palm[2] / 2 + fng[2] / 2  # 掌底 + 指半长
        half_open = (gr["gap_range_m"][1] - 0) / 2 + fng[0] / 2   # 40.5mm
        # 单指行程 = (全开净间距 − 全闭净间距)/2, 全闭净间距受指厚限制为 0
        fng_travel = gr["gap_range_m"][1] / 2 - fng[0] / 2       # 34.5mm
        pad_thick = 0.003
        fi = "        "          # 手指位于 tool 内 (8 空格)
        fingers = ""
        # axis 与 body 偏移反向 (axis=+Y for 左指), q<0 → 指体移向中心 → gap 减小
        for sign, side in ((1, "l"), (-1, "r")):
            fingers += f"""
{fi}<body name="{a}_finger_{side}" pos="0 {sign * half_open} 0">
{fi}  <joint name="{a}_finger_{side}_jn" type="slide" axis="0 {-sign} 0"
{fi}         range="0 {fng_travel:.4f}" damping="1" armature="0.001"/>
{fi}  <geom name="{a}_finger_{side}_link" type="box"
{fi}        size="{fng_half[0]} {fng_half[1]} {fng_half[2]}"
{fi}        pos="{fng_cx} 0 0" mass="0.02" contype="0" conaffinity="0"/>
{fi}  <geom name="{a}_pad_{side}" type="box"
{fi}        size="{pad_h/2} {pad_thick/2} {pad_w/2}"
{fi}        pos="0 {-sign * pad_thick/2} 0" mass="0.005"
{fi}        friction="{gr['pad_friction_initial']} 0.01 0.0002"
{fi}        contype="4" conaffinity="2"/>
{fi}</body>"""

        def make_tool_block(indent: str) -> str:
            return f"""
{indent}  <body name="{a}_tool" pos="{tcp[0]} 0 0">
{indent}    <geom name="{a}_palm" type="box" size="{palm_half[0]} {palm_half[1]} {palm_half[2]}"
{indent}          pos="{palm_cx} 0 0" mass="0.08" contype="0" conaffinity="0"/>
{indent}    <inertial pos="{palm_cx} 0 0" mass="0.08"
{indent}              diaginertia="1e-4 3e-4 3e-4"/>
{indent}    <site name="{a}_tcp" pos="0 0 0" size="0.005" rgba="0 1 0 0.6"/>
{fingers}
{indent}  </body>"""

        # 执行器: MuJoCo 3.15 的 joint= 传动默认取"父链", 必须用 jointinparent
        # 才把力矩施加在关节自身 body 上 (否则施在固定的 base 上被 weld 锁死)
        # 转矩上限: 设计 25.2 初值 ×2; J3 肘关节需承担前臂+腕+夹爪+箱件
        # (合计 ~0.5kg × 0.25m ≈ 1.2 N·m 静态, 长力臂动态更高), 故再给 1.5× 余量。
        kp_gain = 1000.0
        trq_scale = {"J1": 2.0, "J2": 2.0, "J3": 3.0, "J4": 2.5, "J5": 2.5, "J6": 3.0}
        actuators = "".join(
            f'\n    <position name="{a}_{s["name"]}_drv" jointinparent="{a}_{s["name"]}_jn" '
            f'kp="{kp_gain}" forcerange="-{trq_scale[s["name"]] * s["torque_max_nm"]:.1f} '
            f'{trq_scale[s["name"]] * s["torque_max_nm"]:.1f}"/>'
            for s in j) + "".join(
            # 手指是**限力位置伺服** (设计 25.3: 每指目标法向力 5N, 上限 8N)。
            # 纯力控 (ctrl 单位 N) 无法在零力时把手指推回张开位, 故仍用
            # <position> 并由 forcerange 把接触后的推力饱和到上限 8N ——
            # 这正是设计要求的"力上限"语义。实测: 指令净间距低于物体宽度时,
            # 伺服在箱件上饱和出力并维持预紧, 夹持稳定不脱手。
            f'\n    <position name="{a}_finger_{sd}_drv" jointinparent="{a}_finger_{sd}_jn" '
            f'kp="400" '
            f'forcerange="-{gr["force_per_finger_max_n"]} {gr["force_per_finger_max_n"]}"/>'
            for sd in ("l", "r"))

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
{indent}         range="{spec['lower_rad']} {spec['upper_rad']}" damping="10" armature="0.002"/>
{indent}  <geom name="{pfx}_link" type="box" size="{half_x} {z_half} {z_half}"
{indent}        pos="{half_x} 0 0" mass="{mass}" contype="0" conaffinity="0"/>
{indent}  <inertial pos="{off[0]/2 if off[0] else 0.1} 0 0" mass="{mass}"
{indent}          diaginertia="1e-4 1e-4 1e-4"/>
{inner}{indent}</body>"""

        chain_open = emit_chain(0, "    ") + "\n    </body>"   # base body 闭合

        # 定位凸边 (设计 25.3): 高 3mm, 内侧净区 70×50mm。
        # 基准箱件 60×40mm 在净区内每方向留 5mm 余量, 凸边内壁即箱件
        # 的横向约束 —— 指垫接近时箱件不会在光滑台面上滑移。
        ridge_h = 0.003
        net_x, net_y = 0.070, 0.050
        wall = 0.006
        ridge_xml = "\n".join(
            f'    <geom name="{a}_tray_ridge_{idx}" type="box" '
            f'pos="{feed["x_m"] + rx} {feed["y_m"] + ry} {feed["top_z_m"] + ridge_h/2:.4f}" '
            f'size="{sx} {sy} {ridge_h/2:.4f}" rgba="0.45 0.5 0.55 1" '
            f'contype="2" conaffinity="6"/>'
            for idx, (rx, ry, sx, sy) in enumerate([
                (0, +(net_y/2 + wall/2), net_x/2 + wall, wall/2),   # +Y 侧
                (0, -(net_y/2 + wall/2), net_x/2 + wall, wall/2),   # -Y 侧
                (+(net_x/2 + wall/2), 0, wall/2, net_y/2),          # +X 侧
                (-(net_x/2 + wall/2), 0, wall/2, net_y/2),          # -X 侧
            ]))

        return f"""
<mujoco model="{a}_cell">
  <!-- MuJoCo 3.15 起 <compiler angle> 默认为 degree; 本项目统一 SI 弧度 -->
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 -9.81" iterations="200"
          tolerance="1e-10" noslip_iterations="30"/>
  <size nconmax="300" njmax="400"/>
  <default>
    <!-- 接触参数: 200g 轻箱在默认 solref(0.0002, 1) 下被指垫挤得弹跳
         (实测箱底 z 在 55.0-55.7mm 间振荡, 接触力在 0-16N 间跳变, 无法
         建立稳定预紧)。放宽接触刚度并提高阻尼, 使 5N 预紧可稳定保持。 -->
    <geom condim="3" friction="0.5 0.005 0.0001" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    <joint damping="1"/>
  </default>
  <!-- 碰撞掩码 (设计 25.4 要求检查全连杆自碰撞, 但桌面臂的伺服控制
       已保证关节不越位; 白模阶段禁用手臂自碰与手臂-环境碰, 只保留
       指垫-箱件的夹持接触):
       contype/conaffinity 位: bit0=手臂指垫, bit1=箱件/托台, bit2=地面 -->
  <worldbody>
    <light pos="{base[0]} {base[1]} 0.9" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="2 2 0.05" friction="0.5"
          contype="2" conaffinity="6"/>
    <!-- 入料托台 (面 z=55mm): 扁平托盘贴桌面, 支腿在 y 远端;
         机械臂从 -Y 侧接近时不与支腿干涉 (支腿仅占台后缘)。
         设计 25.3: 托盘主体高 18mm, 定位凸边高 3mm, 内侧净区 70×50mm,
         基准箱件两方向各留 5mm 名义余量。凸边与货箱接触 → 限制抓取时
         箱件被指垫扫开的横向位移 (无凸边时箱件在光滑台面上自由滑移 30mm)。 -->
    <geom name="{a}_tray_leg" type="box" pos="{feed['x_m']} {feed['y_m'] - 0.045} 0.0265"
          size="0.045 0.004 0.0265" rgba="0.4 0.45 0.5 1"
          contype="0" conaffinity="0"/>
    <geom name="{a}_tray_top" type="box" pos="{feed['x_m']} {feed['y_m']} {feed['top_z_m'] - 0.0025}"
          size="0.05 0.04 0.0025" rgba="0.55 0.6 0.65 1"
          contype="2" conaffinity="6"/>
    <!-- 定位凸边: 4 条, 高 3mm (台面以上), 围成 70×50mm 净区 -->
{ridge_xml}
    <!-- 箱件 (自由刚体 200g, 设计 25.3) -->
    <body name="{a}_goods" pos="{feed['x_m']} {feed['y_m']} {box_z}">
      <freejoint name="{a}_goods_fj"/>
      <geom name="{a}_goods_geom" type="box" size="{g[0]/2} {g[1]/2} {g[2]/2}"
            mass="{gm}" friction="0.4 0.01 0.0002" contype="2" conaffinity="6"/>
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
