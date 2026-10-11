# Agentic 世界模型演示：目标设计（与 ChatGPT 联合评审 v1.0）

> 来源：GPT 架构设计方案 v1.0（会话 6acb00a1）+ 本仓库现状修订。
> 文献基础见 `agentic_world_model_literature.md`。
> 核心裁决：**保留 MuJoCo + SQLite World + FastAPI/WS + three.js 路线，
> 增加感知建图、时空调度、任务编排、Saga 事务协调四能力。**

## 0. 一句话产品定义

投资人看到的不是"两台小车在 3D 里跑"，而是一个仓储 Agent：
**理解业务目标 → 自主分解任务 → 协调机器人 → 发现环境变化 → 修正计划 →
验证物理执行 → 提交库存事实 → 主动解释结果。**
铁律：**Agent 可以改变计划，但不能改变未经物理验证的事实。**

## 1. 四权分立（权力边界）

| 层 | 可以决定 | 不能做 |
|---|---|---|
| LLM Agent | 任务目标、调用调度工具、提出改派、解释异常 | 直接改库存、覆盖安全停车 |
| 调度器 | 车辆分配、任务顺序、接力点、时空预约 | 绕过车辆能力约束 |
| 安全与运动控制 | 感知、制动、轨迹跟踪、验证交接 | 擅自认定业务任务完成 |
| 世界模型 (World) | 校验并提交事实、记录状态与证据 | 根据语言描述伪造物理事实 |

LLM 只隔几十毫秒做业务决策，不碰电机；安全回路不经 LLM。

**物理能力冻结**：F01（叉车）可叉取/升降/运输；R01（平板车）只能运输已装载到
台面的货物；接力必须经物理可实现的交接（H01 缓冲台或 F01 直接放 R01 台面）。
"锥桶挡路→改派 R01"只能是**运输段**的改派，不是非法替代叉取动作。

## 2. Agent 工具集（对外 6 个，强类型 JSON Schema）

| 工具 | 参数 | 返回 |
|---|---|---|
| task.submit | intent, sku, qty, source, destination, priority, idempotency_key | task_id, dag, assignments, status |
| task.reassign | task_id, leg_id, from_robot, to_robot, reason, expected_version | accepted, new_assignment, lease_changes, reason_code |
| perception.refresh | robot_id, scan_mode, max_age_ms | scan_id, map_version, occupied_cells, confidence, observed_at |
| navigation.replan | robot_id, goal, map_version, reservation_version, policy | path_id, waypoints, eta, clearance, status |
| robot.pause | robot_id, mode, reason, expected_version | stopped, stop_pose, state, held_leases |
| inventory.query | sku, location, include_in_transit, consistency | on_hand, reserved, available, in_transit, version |

内部确定性接口（不暴露给 LLM）：traffic.reserve / handoff.verify /
execution.confirm / inventory.commit_transfer / safety.stop / task.recover。
**绝不提供"任意库存写入"工具**——库存变化只能由满足条件的执行事件触发。

## 3. 感知闭环：禁止上帝视角（双通道隔离）

| 真值通道（仅仿真/评测） | 观测通道（agent 可用） |
|---|---|
| 锥桶 mjData 真实坐标 | mj_multiRay 激光距离（270°/180 线/10Hz/σ=3mm/丢束 2%） |
| 真实碰撞接触 | 感知障碍概率（2-3 帧持续确认） |
| 全部车辆精确位姿 | 带误差定位估计 |
| 全局无障碍几何 | 静态地图 + 已观测动态障碍 |

地图三层：静态层（货架/墙）+ 动态层（锥桶）+ 预约层（车辆时空占用），
融合 + 车辆 footprint 膨胀 → 分车型代价地图（Nav2 costmap 思想，不引入 ROS）。
评测碰撞统计走真值通道，与业务链路物理隔离。

## 4. 事务与回滚：短事务 + Saga

SQLite 事务保证数据原子性，但**不能回滚已发生的物理运动**。
货物状态机：STORED/INBOUND → RESERVED → ON_F01/ON_R01 → AT_HANDOFF → STORED/OUTBOUND。
交接**分两次提交**（F01→H01 一次；R01 顶升验真再一次）；第二次失败货物仍属 H01。

失败处理矩阵（节选）：
| 失败点 | 数据 | 物理 |
|---|---|---|
| 租约失败未执行 | 事务回滚 | 车辆不启动 |
| 托盘在叉臂上 | 不改归属 | 安全放货/等待/请求人工 |
| 托盘已放缓冲台 | AT_HANDOFF 不可假回滚 | 重排 R01 取走 |
| 提交失败但物理已交接 | 保留事实进"待核对" | R01 禁运，幂等重试 |
| 车辆失联租约过期 | 资源隔离（≠空闲） | 停止再分配，观测确认后释放 |

**复用现有资产**：World 六表（pallets/goods/slots/bindings/events/command_log）+
ReservationBoard + 幂等 command_log 已实现大部分；新增表仅
`task_legs`（可改派任务段）、`perception_observations`、`execution_evidence`、`outbox`。
H_valid = 位姿容差 ∧ 承托率≥95% ∧ 旧承载卸载≤5% ∧ 稳定 0.5s ∧ 有效租约（沿用
≤5mm/s 插叉判据与 support_load_ratio 观测，为 R01 台面另标阈值）。

## 5. 投资人演示脚本（6 轮 6 哇点）

初始场景：**8 托盘、3 层货架、2 台车、1 交接站、2 条不同宽度通道**。

1. "仓库里还有多少货？两台机器人在干嘛？" → 对话即驾驶舱（库存板从 SQLite 查询，非硬编码）
2. "把入料口两托上到 A 区二层，两台车同时干活" → 自主分解 + 异构协作（R01 接驳、F01 上架、错峰占用）
3. "通道突然被挡住会怎样？"（现场放锥桶）→ **非脚本化感知 + 自主改派**；三同步证据：
   左=F01 实时停车，中=点云命中锥桶栅格变占据，右=旧路径失效新路径亮起任务版本更新
4. "把另一条通道也封上" → **知道自己做不到**：PATH_INFEASIBLE，冻结任务、库存不乱、请求人工
5. "锥桶移走了" → **扫描确认→地图清除→重规划→取预约→恢复**（不是用户说了就动）
6. "刚才干了什么？有没有丢货？" → 可审计复盘：事件时间线 + 账实核对 + 事务 ID

## 6. 消融验收（交付门禁，非演示附件）

原则：证明**移除某能力后系统以可预测方式失败**。
E0 完整基线 / E1 关动态感知（安全层仍停车）/ E2 隔离环境关制动（碰撞率↑）/
E3 关重规划（发现障碍但停滞）/ E4 关改派（task starvation）/ E5 关时空预约（会车冲突）/
E6 关事务门禁（伪造完成→账实不一致，正式环境必拒）。

防"表演式智能"：≥100 固定种子随机场景，三类覆盖（改派可行/原车可绕/全封死）。
**上帝视角泄漏审计**：反事实实验——同观测流下改变未观测锥桶位置，规划输出必须不变；
静态依赖检查 + 运行时数据源标记为补充。
量化门槛（待标定初值）：感知→制动 P95≤300ms、障碍召回≥95%、无碰撞完成率≥99%、
不可行正确拒绝 100%、库存守恒错误 0、非法能力调用 0、重复事件重复计数 0。

## 7. 实施路线（对齐现有代码）

- **P1 世界模型服务化**（当前孪生升级）：twin/server.py 重写为
  World+ForkliftAdapter+paced `_tick`（速度倍率 1×/2×/4×/8×）；agent = oMLX
  (127.0.0.1:1234, OpenAI 兼容) 工具调用 + 离线规则兜底；UI = 聊天 + 库存板 + 事件流。
  单 F01、单托、现有 cell。
- **P2 多货物多货位**：cell 扩展 3 托 3 位；task.submit → DAG 分解（legs）；库存板真实计数。
- **P3 双车 + 交接**：WarehouseScene（18 货位 4 设备）接入或扩展 cell 至 R01；
  F01→H01→R01 三段交接证据链；task.reassign 走运输段。
- **P4 锥桶 + LiDAR + 重规划**：mj_multiRay 扫描器、占据栅格、A*（25mm 网格 +
  包络膨胀，§19.2 口径）、PATH_INFEASIBLE、§6 消融测试入 CI。
