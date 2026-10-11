# 论文实验协议 v0.1（预注册草案）

> 来源：GPT 方法学审稿报告（会话 6acb00a1，2026-10-11）+ 本系统可行性映射。
> 状态：**评测前冻结前可讨论**；冻结后不得改动主假设与终点定义。
> 暂定标题：*Evidence-Grounded Embodied Agents: Transactional World Models and
> Continual Decision Learning for Warehouse Robotics*

## 0. 贡献定位（采纳审稿裁决）

不写"Voyager+Reflexion+MuJoCo+SQLite 组合"。理论贡献定义为：

> **Evidence-Grounded Decision Learning under Physical and Transactional
> Constraints** —— 物理约束与业务事务不变量共同作用下，语言 Agent 如何通过
> 可验证的执行证据形成可靠决策，并在后续任务中实现可测量的适应与迁移。

拆成两篇：
- **Paper I**（H1+H2）：证据门控的事务执行可靠性 → arXiv → RAS / T-ASE / workshop
- **Paper II**（H3）：持续决策学习（Memory×Curriculum 因子 + 负迁移 + 污染）→ CoRL 系

## 1. 三个可证伪假设

| 假设 | 主张 | 自变量 | 主终点 | 预注册门槛（候选） |
|---|---|---|---|---|
| H1 证据约束 | 状态报告被延迟/重复/伪造时，物理证据门禁显著降低错误业务提交 | 有/无 Evidence Gate | 假提交率 FCR↓ | 降 ≥80%，合法误拒 ≤2% |
| H2 世界模型决策 | 同感知同运动控制下，业务世界模型+约束工具 Agent 比纯 LLM 代理有效完成率更高 | 决策架构 | VCR | ≥ +10pp vs B1 |
| H3 持续学习 | 证据验证过的 Decision Memory + 自课程改善未见布局表现 | 记忆/课程 | held-out VCR、AULC | 显著优于 M0C0 |

⚠️ H1 审稿警告（必须回应）：门禁若被编程为拒绝无证据提交，"零假提交"是代码性质
不是发现。H1 必须测不确定物理条件下的正确性：传感器误报、接触假阳性、部分交接、
事件重放、数据库故障、证据过期（= R2 故障注入）。

## 2. 实验矩阵

### 2.1 基线（共享同一执行内核，公平性 G2）

| 组 | 决策层 | 世界模型 | 证据门禁 | Memory |
|---|---|---|---|---|
| B0 | 强规则调度（优先级+EDD+预约+确定性恢复） | 有 | 有 | 无 |
| B1 | 纯 LLM 工具代理 | 仅非权威任务描述 | 有（独立安全层） | 无 |
| B2 | LLM + 完整世界模型 | 有 | 有 | 无 |
| B3 | B2 + 自课程 + Decision Memory | 有 | 有 | 有 |

B1 必须允许有限状态查询（不许削成稻草人）；H2 不宣称必赢 B0，
宣称**复杂度/异常上升时鲁棒性差距扩大**（并报告 LLM 成本）。

### 2.2 条件变量（4 维，分层抽样非全因子）

任务复杂度 2/5/10 段 × 动态扰动 无/单锥桶/多障碍 × 观测质量 理想/标称/延迟丢包 ×
布局泛化 已见/新拓扑/新交接点。五类情景：标准运营/动态避障/双车协作/对抗异常/跨布局迁移。
每类预置可行:不可行比例，独立验证器剔除不可达任务出分母。

### 2.3 样本预算

- 主实验：**4 组 × 240 配对场景 = 960 episode**（5 类 × 48 场景，族内配对）
- 240 场景 = **30 布局/任务族 × 8 扰动**，按族聚类估计不确定性
- 5 个独立种子流：layout / task / disturbance / sensor / agent；冻结派生方式
  （禁止共享全局流——分支执行步数不得改变后续场景）
- 学习实验（H3 独立）：10 个学习 run × 空记忆起步 × 200 episode，每 20 ep
  在固定 60 held-out 场景评估；测试禁写回；另留完全冻结终测集
- 本系统可行性：1 episode ≈ 277s sim ≈ 11s 墙钟（2kHz 物理 ~25× 实时），
  960 episode 主实验 ≈ **3 CPU-小时**；H3 2000+600 episode ≈ 8 CPU-小时 ✅

### 2.4 学习曲线报告

横轴=训练 episode 数（含失败），纵轴=held-out VCR，逐 run 均值 + 95% CI
（**run 层聚类，不得把 run 内 episode 当独立重复**）；报 AULC；
配对场景差值 + 聚类 bootstrap；二元结局用布局×任务随机效应分层 logistic；
超时用生存分析截尾；Holm 多重检验校正；报效应量+CI+原始分布，不只 p 值；
每组报 token/调用数/成本/墙钟。

## 3. 指标体系（三层，不合成总分；安全与账实是硬约束）

**运营**：VCR（主）· 吞吐（失败/重试计入时间分母）· 米/托 · 周期 P50/P95 ·
利用率 · 异常恢复时间 · 账实一致率（TRANSFER_PENDING 有窗口上限）· 事务滞后 · 成本
**安全/事实完整性（P0）**：Invalid Commit · False Accept · False Reject · Collision ·
穿透深度 P95 · Lease Violation · Unsupported Action · Evidence Coverage · Recovery Success
**学习**：Held-out VCR · AULC · Lesson Reuse · **Beneficial Reuse（配对反事实重放：
同一可序列化快照分叉两个完整 rollout，不是比文字）** · Repeated Error ·
Negative Transfer · Layout Transfer Gain · Memory Efficiency

术语纪律：不更新 LLM 参数 → 写 **memory-augmented continual adaptation**，不写"终身学习"。

## 4. 消融（E0–E6 之上）

A1 去 Memory · A2 有 Memory 禁证据过滤（污染路径）· A3 去自课程 · A4 只有课程无 Memory
（**A1–A4 = Memory×Curriculum 2×2 因子，同预算分离主效应与交互**）·
A5 门禁→信任 LLM 自报（独立事务副本，不关共同安全内核）· A6 去事务原子性多步直写 ·
A7 去资源预约 · A8 去业务世界模型只留物理摘要 · A9 理想真值感知（=性能上界对照）·
A10 换 LLM 型号/温度。
两个 token 匹配对照：无记忆组读等长任务摘要 + 等长无关 lesson 负对照。
每组报告实际启用组件清单（防"消融没消到"）。

## 5. 三项加强实验

- **R1 物理忠实度**：插叉/放托/交接三工况 × 时间步 1/0.5/0.25ms 收敛研究 +
  摩擦/质量/阻尼扰动稳健性；2kHz 只是数值收敛参考不是真值。
  （桌面实物接驳台 20–30 次重复交接对照 = 外部有效性王牌，列入 P5）
- **R2 证据故障注入**（≥6 种）：伪造/重复/延迟/丢失/传感器误判/提交后断连；
  重点"物理已完成但 SQLite 提交失败"→ 不得谎称未发生、不得带未决状态继续搬运
- **R3 Memory 投毒与分布漂移**：伪装 lesson 注入 → 系统能否隔离纠正；
  训练布局最优策略在测试布局反转（接驳点搬家）→ 证明学到条件化经验而非模板

## 6. 交付门禁

G0 协议冻结 · G1 物理收敛报告 · G2 基线公平对照表 · G3 960 episode 全量日志 ·
G4 学习 run 隔离+Memory 快照（无泄漏审计）· G5 消融+故障证据 ·
G6 统计报告（含负面结果）· G7 开放复现（场景生成器+种子+评测器）

Episode 标准日志字段：run_id / episode_id / layout_family_id / seed_bundle /
system_variant / model_version / prompt_hash / memory_snapshot_hash /
physics_model_hash / sensor_config_hash / task_graph / event_log /
physical_evidence / inventory_movements / trajectory_log / tool_call_log /
final_metrics / failure_reason。**物理真值日志与 Agent 观测日志物理分离**
（本系统双通道设计天然满足，直接作为方法学卖点）。

## 7. 与本仓库的映射（现状 → 缺口）

| 协议要求 | 已就绪 | 缺口 |
|---|---|---|
| 证据门禁+事务 | World/Event/CommitEvidenceMissing ✅ | 证据过期/部分交接注入 |
| 判据执行内核 | ForkliftAdapter+Ramp+210 测试 ✅ | — |
| 多货物 | 三层 cell ✅（P2 进行中） | 任务 DAG/B0 规则调度器 |
| 双车+交接 | WarehouseScene 存在（未接线） | P3 |
| 锥桶+LiDAR 感知 | 设计定稿（mj_multiRay 参数表） | P4 实现 |
| Decision Memory | 概念+表设计 | twin/learning.py |
| 自课程生成器 | — | P4 |
| 场景生成器+种子五流 | — | paper/harness（G7 要求） |
| 评测器+统计 | — | paper/harness |

## 8. 时间线建议

1. **P2 完成**（多货物+DAG+B0 规则基线）→ 支撑 H2 主实验
2. **P4**（锥桶+感知+重规划）→ 支撑 H1 故障注入 + R2
3. **learning harness**（run 管理+种子五流+held-out 集）→ H3
4. Paper I 初稿（H1+H2 实验完成后）；R1 物理收敛实验可并行先行（纯仿真，最便宜）
