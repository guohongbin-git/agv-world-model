# 相关工作与成熟框架调研 (M2 调度层, 2026-10-11)

## 1. MAPF (多智能体路径规划) —— W2a 导航+锁的学术坐标
我们自研的"网格 A* + 巷道/交叉口锁矩阵"在学术上是 **time-windowed MAPF 的
离散资源简化**。对照:
- **pymapf** (PyPI): CBS / PIBT / LaCAM / LNS 求解器集合 —— 可作为**对拍
  oracle**: 同一网格+任务, 比较 PIBT/LaCAM 的完成时间 vs 我们的锁矩阵,
  量化锁粒度设计的次优性 (论文实验素材)。
- **GavinPHR/Multi-Agent-Path-Finding**: CBS + Space-Time A* 干净实现 —
  Space-Time A* 把锁占用当时空障碍, 是我们 lock_penalty A* 的严格版;
  若 8 车并发出现锁等待瓶颈, 升级路径明确。
- **PIBT / LaCAM**: 8 车实时性场景的工业级替代 (优先继承+旋转), 若锁矩阵
  死锁率不达标时的 Plan B。
- **open-rmf/mapf** (Apache): 工业级但绑定 ROS 生态, 不引入。

## 2. 车队管理语义 —— openTCS (Fraunhofer IML, 工业标准)
不引 Java 框架, 但**调度层术语与状态机对齐 openTCS**:
- `TransportOrder`: DISPATCHABLE → BEING_PROCESSED → FINISHED/FAILED
  —— 对齐我们的 PENDING → ASSIGNED → COMMITTED (物理证据后提交, 比
  openTCS 更严格, 是我们的差异点)
- `Block`: 静态区域互斥 —— **就是我们的巷道/交叉口锁矩阵**, 工业等价物
- 点/边路由 + 车辆管理: 与 LAYOUT 派生网格同构
论文/文档引用: openTCS 是 AGV 调度的事实工业开源, 我们的"锁+有序申请"
在其 Block 语义内有直接对应。

## 3. 两阶段调度 —— 与我们 A*+CP-SAT 组合同构
- **mmmorgandata/agv-warehouse-scheduling**: A*/JPS 路径层 + 分配层的
  two-stage 管线 —— 与我们 (W2a 路径) + (W2b CP-SAT) 的分层完全同构,
  可引用其评估方法 (makespan/等待时间/利用率)。
- OR-Tools CP-SAT: 设计文档 §19.4 既定路线 (能力约束/时间窗/100ms tick/
  超时回退), 无需更换。

## 4. LLM agent 技能库 —— Voyager (TMLR 2024) 范式
- 技能 = 预验证可执行代码 (函数+断言+示例), LLM 只做选择+参数+异常处置
- environment-feedback refinement = "agent 学习仓库管理"的学习机制落点
- W3 `sim/robot_skills.py` 按此设计 (已定)

## 5. 结论: 引入 vs 自研
| 模块 | 决策 | 理由 |
|---|---|---|
| MAPF 求解器 | 自研锁矩阵为主, pymapf 对拍 | MuJoCo 深耦合, 网格世界不匹配外框架; 对拍保正确性 |
| 车队调度语义 | 对齐 openTCS 术语 | 工业标准, 术语即论文语言 |
| 优化器 | OR-Tools CP-SAT (既定) | 设计文档 §19.4 |
| 技能库 | 自研薄层, Voyager 范式 | 领域闭环, 重框架无收益 |
| 仿真平台 | 自研 MuJoCo cell (已建成) | 物理证据链是核心差异点, 通用平台无事务层 |
