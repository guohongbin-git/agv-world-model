# DC 数字孪生场景设计 — M2 矩阵仓库 × PepsiCo/Omniverse 风格配送中心

状态: **v0.3 设计稿, 待签字** (架构定位: 开源复刻) · 2026-10-13 · 参照: NVIDIA GTC D4109
(PepsiCo DC × Omniverse + Metropolis) + NVIDIA 官方文档/专利调研

## 1. 视频要素 → 我们的映射

| 视频要素 | 我们的对应物 | 差异化 |
|---|---|---|
| 照片级 DC 数字孪生 (Omniverse) | MuJoCo 1kHz 物理仿真 + 既有 Blender 照片级渲染管线 | 他们是运动学演示; 我们是**物理可证**数字孪生 (接触/承托/稳定性可判) |
| 叉车/托盘全仓流 | 128 架 384 货位 + 8 车异构 + CP-SAT 调度 (已建) | — |
| Dock/月台作业 (卡车装卸) | **缺** → 本设计新增 dock yard | — |
| Metropolis 视觉 AI (摄像头感知/标签识别) | **缺** → 本设计新增"虚拟摄像头"传感器模型 | 他们的感知喂分析看板; 我们的感知喂**证据门控** (感知不可信→拒单) |
| What-if 吞吐优化 (假期峰值/布局) | 30-seed 场景扫描管线 (fault_ablation 同源) | 我们加"故障下吞吐"维度 |
| 数字孪生↔实体同步愿景 | 证据事务层: 每次物理变更有证据才 COMMITTED | **他们的演示完全没有这一层** (GPT 尽调确认的空白) |

## 2. 场景拓扑 (在现有 M2 上扩展)

```
        ┌── 入库月台 I1-I3 ──┐                    ┌── 出库月台 O1-O3 ──┐
卡车 →  │ 车位(装卸位, 静体) │ → stand 交接台 →   │ 128 架存储矩阵     │ → stand → │ 车位 │ → 卡车
        └───────────────────┘                    └───────────────────┘
             ↑ demand.engine 入库到达时刻表              出库按订单流 (已有)
```

- **P0 月台与卡车**: 南侧扩 6 车位 (3 入 3 出), 卡车=带货箱的静体 agent
  (到达/停靠/离场时刻表由 DemandEngine 扩展); 入库=卸托入存, 出库=拣选
  集托装车; 装车完成→卡车离场 (账本出库闭合, 证据链延伸到"托盘上车")
- **P1 虚拟摄像头 ("Metropolis 对应物")**: 每个月台+关键巷道设 ROI 摄像头,
  产生检测事件流 (托盘出现/离开/标签码), **带噪声模型** (漏检 p_miss、
  误检 p_fp、延迟 t_lag、遮挡) —— 检测事件是感知融合层唯一输入;
  **门控证据只允许来自融合层, 禁止读真值** (呼应 GPT 真值隔离要求,
  这是与 PepsiCo 演示的本质区别: 我们的 AI 感知错→系统拒单对账,
  他们的感知只是看板)
- **P2 What-if 优化器**: 场景矩阵扫描 (车队 6/8/10 × 月台 4/6 × 到达
  倍率 1/2/3) × 30 seed → 吞吐/利用率/账实一致率曲面 + 峰值(假期)曲线;
  复用 press/ablation 基建
- **P3 照片级演示片**: 物理真值 CSV → Blender 照片级渲染 (既有管线),
  产 PepsiCo 风格 DC 全景动画 + 证据门控特写 (对应故障对照片手法)

## 3. 与现有资产的复用率

| 组件 | 复用 | 新增 |
|---|---|---|
| 物理/调度/锁矩阵/需求引擎 | ~90% | truck agent + 月台几何 (warehouse_cell) |
| 证据门控/事务/恢复 | ~90% | "托盘上车"作为新证据事件类型 |
| 面板 :8400 | ~70% | 月台甘特/摄像头事件流/KPI 曲线 |
| 30-seed 扫描 | ~80% | 场景矩阵驱动器 |
| 照片级渲染 | ~90% | DC 全景机位 + 月台场景件 |

## 4. 分期与验收

| 期 | 内容 | 验收 |
|---|---|---|
| P0 (1-2 天) | 月台几何 + truck agent + 入出库闭环 + 账本出库事件 | pytest 全绿 + 30min continuous 一辆卡车完成装车离场且账实一致 |
| P1 (2-3 天) | 虚拟摄像头 + 感知融合层 + 门控接融合层 (真值隔离报告) | 注入漏检/误检时错误提交率不升; 真值隔离审计 PASS |
| P2 (1-2 天) | What-if 扫描器 + KPI 曲面 | 3×3×3 场景 × 30 seed 报告出表 |
| P3 (1 天) | 照片级演示片 (60-90s) | 成片 + 证据门控特写 |

## 5. 风险

- 卡车 agent 引入新物理体 (装卸碰撞) → 静体化保守起步
- 摄像头噪声模型失真 → 参数对齐文献 + 敏感性扫描
- 范围蔓延: P3 前不做行人/ conveyor 输送线 (视频里有但非核心)


## 6. NVIDIA 官方文档/论文/专利调研 (v0.2 增补)

### 6.1 对标架构: Warehouse Blueprint (docs.nvidia.com/vss 3.0)

PepsiCo 演示背后的正式产品。分层: Foundational Microservices
(Warehouse 2D/3D Blueprint + VIOS 微服务) → Blueprints → AI Agents
(如 Safety Inspector) → Agentic Application; 配 Simulation/SDG 支线。
核心用例: 资产追踪/安全接近监测/产线过程优化; 数据出口=实时告警+
分析 API+WMS/ERP 集成 (Kafka 事件流)。

**映射与差异**:
- 我们的"虚拟摄像头→感知融合层"对标其 2D Blueprint 检测事件流
  (他们: detections→analytics; 我们: detections→**证据门控**);
  事件总线形态对齐其 Kafka 出口 (我们的 feeds/decision ring)。
- 他们**没有任何"物理执行证据→账本提交裁决"层**: Warehouse Blueprint
  止于感知/告警/分析; Mega Blueprint (KION) 是 WMS 给机器人派任务 +
  数字孪生预演, 事务边界仍是消息 ACK。→ 我们的证据事务层在两层
  Blueprint 之间是空白 (与 GPT 尽调结论互证)。
- SDG 对应: 我们的故障注入管线 = 面向证据协议测试的 SDG
  (他们用 Replicator 生成感知训练数据)。

### 6.2 专利地形 (US)

- US12084280 (仓库物流流控制)、US12314060 (数字孪生+ML 价值链规划)、
  **US12585282B2 (用数字孪生+人类反馈训练库存管理机器人)**、
  US11767167 (料箱检测) —— 数字孪生仓储方向已被大厂占位。
- 启示: 通用"数字孪生仓储/视觉检测"不可专利; 可主张点在**执行证据
  协议**: 五门判据的时序窗、传感器仲裁与账本原子提交的耦合、故障下
  恢复对账协议 (方法专利方向, 需 FTO 检索)。呼应 GPT 尽调"护城河在
  可审计执行证据契约"。

### 6.3 设计修订 (受文档影响)

1. P1 感知融合层事件 schema 对齐 2D Blueprint 检测事件 (id/type/
   confidence/timestamp/source), 便于未来跨系统演示与论文对照。
2. 增加一个 AI-Agent 样例对标 (Safety Inspector → 我们的"证据审计
   agent": 消费事件流, 对账实分歧发告警) —— 放 P2, 不阻塞主线。
3. 术语: 对外叙事用 "Evidence-Grounded Warehouse Digital Twin",
   避免与 NVIDIA Warehouse Blueprint 重名。


## 7. v0.3 架构定位: 用本仓库开源栈复刻 Warehouse Blueprint 范式

**原则: 零 NVIDIA 依赖** —— 纯 Python 开源栈 (mujoco/numpy/ortools/
PIL), pip 可装, 全链路可复现。目标是"开源版 Warehouse Blueprint":
同构分层架构, 换掉专有底座, 并补上他们缺失的证据事务层。

### 7.1 组件对标 (他们的 → 我们的)

| Warehouse Blueprint 组件 | 专有底座 | 开源复刻 (本仓库) |
|---|---|---|
| 物理仿真 (Isaac Sim/Omniverse) | USD/Omniverse Kit | **MuJoCo 1kHz** warehouse_cell |
| 感知 (Metropolis/DeepStream 视觉) | DNN 摄像头管线 | **虚拟摄像头传感器模型** (检测事件+噪声) |
| 2D Blueprint 检测事件流 | Kafka 出口 | feeds/decision ring + 同 schema 事件 |
| 3D 数字孪生渲染 | Omniverse Renderer | **Blender 照片级管线** (真值 CSV 驱动, 既有) |
| AI Agents (Safety Inspector 等) | NIM 微服务 | Python agent (证据审计/告警), 消费事件 ring |
| WMS/ERP 集成 | 企业连接器 | 账本 SQLite + JSON API (:8400) |
| Simulation/SDG | Replicator | **故障注入管线** (面向证据协议的 SDG) |
| (缺失) | — | **证据门控事务执行层** (我们的独占层) |

### 7.2 分层架构 (开源版)

```
L4  Agentic 应用     : 证据审计 agent / what-if 优化 agent      (P2)
L3  Blueprints       : DC 场景 blueprint (月台+存储+车队)        (P0)
L2  感知微服务       : 虚拟摄像头 → 检测事件流 (噪声模型)        (P1)
L1  基础微服务       : MuJoCo 物理 + 锁矩阵调度 + 账本事务       (已有)
L1.5 (独占) 证据事务 : PREPARED→证据→原子提交→恢复对账          (已有)
```

### 7.3 叙事

"开源、物理可证的 Warehouse Blueprint 同构实现 —— 补上了专有方案
缺失的执行证据层": 这同时回应 GPT 尽调 (可复现性=学术可信度) 与
开源社区价值 (无 USD/Omniverse 门槛)。
