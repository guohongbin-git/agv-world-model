# AGV World Model：物理证据门控的 Agentic 仓储世界模型

**English**: An agentic world model for warehouse management in which an LLM/rule agent
is the *only* command entry, physical execution runs in a contact-faithful MuJoCo engine,
and inventory facts commit to a transactional ledger **only when sensor evidence
(support ratio, contact, withdrawal clearance) proves them**.

> 核心命题：*具身智能的深层价值不是替代体力，而是把物理执行变成持续可观测、可验证、
> 可学习的过程。* Agent 可以改变计划，但不能改变未经物理验证的事实。
> 全文论证见 [docs/agentic_world_model_literature.md](docs/agentic_world_model_literature.md)。

## 架构

```
自然语言 ⇄ WarehouseAgent (工具调用 + 规则兜底)
   │ task.submit / pause / speed / query      ← 唯一指令入口
   ▼
判据层 ForkliftAdapter (插叉≤5mm/s、退叉高度门禁、承托率观测)
   ▼
MuJoCo 2kHz 接触物理 (ForkliftCell: 车/叉/托盘/货架/暂存站)
   ▼
World (SQLite 事务: 托盘位置·货位占用·绑定·计数 + Event 账本)
   ▼
展示层 three.js/WebSocket 实时投影 + 库存板 + 事件流 (只读)
```

- **单一工程参数源**：全部几何/速度/判据来自 `configs/design.yaml`（版本化，含 manifest）
- **事务边界**：库存写入只发生在物理证据成立处（`CommitEvidenceMissing` 会拒绝提交）
- **禁上帝视角**：规划器读感知估计，MuJoCo 真值仅用于独立验收（设计文档 §19.2）

## 在线演示（本地）

```bash
PYTHONPATH=. python twin/server.py     # http://127.0.0.1:8300
```

页面内对 Agent 说：`把 A-L3-C1 那托下架到 H01` · `快 4 倍` · `库存` · `事件`。
理论页 `/theory.html`；照片级影片在 `deliverables/`。

## 目录

| 路径 | 内容 |
|---|---|
| `sim/` | MuJoCo 物理单元（叉车 cell、机械臂 cell、安全、控制器） |
| `services/` | 世界模型本体/事务、判据适配器、任务调度、预约、感知、T12 场景 |
| `twin/` | 实时世界模型服务（FastAPI+WS）、WarehouseAgent、three.js 前端 |
| `scene/` | 全场景装配（18 货位 4 设备） |
| `cad/` | design.yaml → STEP/MJCF/网格 参数化生成 |
| `tests/` | 210 项回归（物理判据、事务恒等式、安全、场景） |
| `docs/` | 详细设计 v2.0.0、文献裁决、Agentic 目标设计、验收报告 |
| `deliverables/` | 三段同源位姿驱动的视频（白模 / 照片级 / 投资人版） |

## 进行中

- **P2** 多货物多货位 + 任务 DAG（物理多层 cell 已落地，210 测试全绿）
- **P3** 双车（F01+R01）+ 交接事务链
- **P4** 锥桶 + `mj_multiRay` 激光雷达 + 占据栅格重规划 + E0–E6 消融
- **Learning loop** 自课程 + Decision Memory + 成长曲线

## 论文

实验协议（预注册草案，与 GPT 方法学审稿联合制定）：[paper/EXPERIMENT_DESIGN.md](paper/EXPERIMENT_DESIGN.md)。
暂定标题 *Evidence-Grounded Embodied Agents*；贡献定义为物理与事务双重约束下、
基于可验证执行证据的决策学习；路线 = 两篇（H1+H2 事务可靠性 / H3 持续决策学习）。
