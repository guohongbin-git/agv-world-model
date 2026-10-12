# AGENTS.md — desktop-agv-demo

桌面 AGV 与机器人仓储 Demo（三类四设备虚拟仿真）。规格基线：`docs/桌面AGV与机器人仓储Demo_阶段一仿真详细设计_v2.0.0.md`，参数单一来源：`configs/design.yaml`。

## 硬规则

1. **物理即真值**：任何"成功"必须来自 MuJoCo 观测（接触、位姿、受力）。禁止动画绑定、weld、写 qpos、隐藏刚性约束、从对象属性直读扫码结果。
2. **单位后缀**：所有配置字段带 `_m/_kg/_s/_rad/_n` 后缀；控制器接口一律 SI（米、弧度），mm 只存在于图纸与注释。
3. **不冻结**：无"已冻结"标识的数值都是设计初值，修订须记录理由；不覆盖历史报告，新 run 用新 `run_id`。
4. **版本联动**：改几何/接口须同步 CAD、MJCF、BOM、测试，递增 `design_version`（补丁=小修，次版本=兼容调整，主版本=破坏性）。
5. **时钟**：仿真内只用 `sim_time_s`（control_clock_domain=simulation）；墙钟只用于审计，禁止混算。
6. Python 3.12 venv（`.venv/`；原设计基线 3.10.14 经负责人 2026-10-09 批准升级，原因见 reports/environment.md）；测试 `./.venv/bin/python -m pytest tests/ -x -q`；参数检查 `./.venv/bin/python -m configs.check`。
7. 每次取放/库存逻辑修改：跑受影响的验收用例子集；不删失败样本。

## 架构红线（M2 起，改代码前先读）

1. **单一指令入口**：agent（LLM/规则）只能提交意图；物理执行走判据层，禁止 agent 直写状态。
2. **物理证据门控**：`EvidenceVerdict` 五门联合判据（承托率≥95% ∧ 旧支撑卸载≤5% ∧ 0.5s 稳定窗 ∧ 静止 ∧ 位姿≤10mm），11 帧采样、版本化、TTL。
3. **原子事务**：`World.commit_transfer` 单事务（乐观版本→变更→事件→证据→幂等）；物理成功而提交失败 → `RECONCILIATION_REQUIRED` + 现场重观测恢复。`PREPARED` 必须在不可逆物理影响前持久化（数据库可回滚、物理不可回滚）。
4. **观测隔离**：决策侧（agent/调度/记忆）零真值引用（测试 AST 强制）；真值只流向 evaluator。感知走 `twin/observation.py`。
5. **需求真实性**：SKU 目录与需求曲线由真实数据标定（Kaggle 零售 456 万行 → 40 SKU + 周内/月度因子），不凭空捏造需求。
6. **调度/执行分离**：CP-SAT 只产出计划；执行前按台账重验（C02，`matrix_server._gate`）；调度器不得直接写库存或占用。
7. **技能层**（`sim/robot_skills.py`，Voyager 范式）：agent 只选技能+传参；技能展开为已验证的相位机任务序列，agent 不可能组合出物理不成立的序列。

## 目录

- `configs/` design.yaml（参数单一来源）+ 参数/环境检查
- `cad/` build123d 参数化模型 → STEP/网格/碰撞体
- `sim/` MJCF 与仿真层：`warehouse_cell.py`（128 架矩阵+8 车）、`demand.py`（需求引擎）、`dispatch.py`（CP-SAT 调度）、`navigation.py`（A*+运动）、`traffic.py`（锁矩阵）、`mission.py`（相位机+台账）、`robot_skills.py`（技能库）、`forklift_cell.py`（M1 双车场景，保留）
- `twin/` 数字孪生服务：`matrix_server.py`（:8400 矩阵服务/news）、`server.py`（:8300 M1 双车）、`observation.py`、`memory.py`、`agent.py`、`web/matrix.html`
- `services/` world（本体+事务+证据）/ perception / tasks / scheduler / adapters
- `experiments/` 论文实验：`fault_ablation.py`（B0-B3 消融）、`convergence.py`、`solver_sweep.py`
- `scripts/` 压测与取证工具（`press.py`、`pen_probe.py`、`pen_align.py`、`detfix_*.py`）
- `data/demand/` 真实零售数据标定的 SKU 目录与需求曲线
- `docs/knowledge/` 相关工作、Paper I 故障矩阵、发散取证
- `tests/` T01-T16 + M1/M2 用例（pytest）
- `reports/` 环境、验收证据（按 run_id / 版本归档）
- `deliverables/` 影片与最终报告

## 当前状态（2026-10-13）

**M2 矩阵仓库主线**：128 架（4×4 组×8 架×3 层=384 货位）+ 8 车异构车队（F01-F04 叉车 / A01-A04 潜伏顶升 AGV，L1/L2/L3 承重分级、能力强速度慢）+ 需求驱动 24h continuous。
- 已落地：CP-SAT 调度（能力约束/时间窗/回退贪心）+ 网格 A* + 巷道/交叉口锁矩阵 + 决策面板（`:8400`）+ 利用率反馈控制 + 技能库 + 原子事务/观测隔离/恢复（B1-B3）+ 24h 自恢复看门狗
- 验证：`pytest tests/` 300+ 用例；Paper I 消融 120 runs（B0-B3）；30 分钟压测脚本 `scripts/press.py`
- **物理收敛已闭**（DetFix，2026-10-13）：t=558s 六层发散链根断——三修复（落架释放高度 z+0.010 消 1.5mm 楔 / row:j 锁持有至退场 / 载运路由 hard-block 有托台位），全库 320 passed + fleet 15/15（含 determinism 双 run 逐位一致）+ L0/L1 press 点位 7/7 PASS。取证：`docs/knowledge/detfix_fault_chain.md` + `reports/detfix/run1-14`。30min 压测复验进行中
- design_version 2.3.0（`reports/v2.3.0.md`）；旧 W01-W07 表述已废弃
