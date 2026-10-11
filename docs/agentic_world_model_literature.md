# Agentic 世界模型：文献基础与论点裁决

> 回答核心问题：**"具身智能的深层价值不是替代体力，而是解决信息问题——人类执行的偏差收集不完整，
> 导致 plan 无法充分优化迭代；机器人第一次让 计划→执行→偏差→学习 闭环真正闭合。"**
> 结论：论点成立，处于四条文献脉络的交汇处；需两处精确化（见 §5）。

## 1. 理论源头：控制论（偏差必须可测量才可能被优化）

- Åström & Murray, *Feedback Systems*, 2008/2021：性能与鲁棒性的前提是反馈信号存在。
  人类执行 = **开环执行器**：偏差发生但无传感器，信息物理性丢失。
  机器人 = **可测量执行器**：每个关节、每次接触、每帧力都可记录。
- Wiener, *Cybernetics*, 1948：控制即反馈；没有测量就没有控制。

## 2. CPS / 数字孪生闭环（plan-vs-actual gap 是核心问题）

- Tao et al., *Digital twin driven cyber-physical production system*, 2019（五维模型）；
  Siemens, *Closed-loop manufacturing / Leveraging the closed-loop digital twin*：
  数字孪生的定义性能力 = 计划↔执行双向同步，执行偏差回流模型，而非"3D 可视化"。
- *Digital Twinning and Optimization of Manufacturing Process Flows* (ASME JMSE 2023)：
  "工厂变成数据富集环境，海量感知数据回流模型构建"。
- LLM×DT 前沿（2025）：*Leveraging LLM Agents and Digital Twins for Fault Handling*
  (arXiv 2505.02076)；*IIoT-enabled digital twin with multi-agent LLM* (Robotics&CIM 2025)；
  *Self-evolving digital twins: LLM-powered multi-agent approach* (2026)——
  孪生+LLM agent 已是正式研究方向，本项目与之同代际。

## 3. 机器人即测量系统（国家研究计划级别背书）

- NIST GCR 24-054, *Research Opportunities for Advancing Measurement Science for
  Manufacturing Robotics*, 2024：把机器人执行当测量系统研究——可追溯、可标定的执行数据。
- NISTIR 8090, *Measuring and Representing the Performance of Manufacturing Assembly Robots*。

## 4. 数据飞轮（论点的最热当代表述）

- **Robot-Powered Data Flywheels** (Grannen, Pan, … Sadigh, Stanford, arXiv 2511.19647, 2025-11)：
  "transform robots from FM consumers into data generators"。实证：Scanford 图书机器人部署
  2 周，其执行数据微调 VLM：书目识别 32.0%→71.8%，跨域多语 OCR 同步提升，另省 18.7 人时。
  ——**执行数据反哺优化的定量证明**。
- **Towards a Data Flywheel for Embodied Intelligence in Logistics** (Yu, Chen, … Zhang,
  北航, arXiv 2606.05960, 2026-06)：物流场景；"把日常运营转化为可复用数据资产，用世界模型
  合成长尾监督（WM-DAgger），部署反馈回流策略改进"——与本论题几乎同文。
- *Pathway to Generalist Robots: Scaling Law, Data Flywheel, Humanlike Embodiment*
  (UT Austin RPL, 2024)：数据飞轮 = 通用机器人三支柱之一。
- 工业先例：Kiva/Amazon——货架每次移动成为事务事件，库存记录从"事后盘点"变"持续验证"；
  没有准确库存事实，slotting/补货算法无从迭代（HBS/MIT TOM 案例）。

## 5. 两处精确化（让论点更强）

1. **机器人并非零偏差**：sim-to-real gap 就是机器人执行偏差。差别不在有无偏差，而在偏差
   **可测量、可归因、可学习**。人类偏差是信息黑洞；机器人偏差是数据资产。
2. **信息优势属于"感知+记录+可查询"的具身系统**，不属于泛自动化：盲目执行程序的工业机器人
   同样丢信息。工程推论：世界模型（事务+事件日志）+ 感知通道隔离（禁上帝视角）是论点成立的
   必要条件——本仓库的消融判据（关感知必撞锥桶、账实恒等式）即该论点的可执行检验。

## 6. 投资人一句话

> 人自动化干活，顺带自动化了"知道干得怎么样"。机器人每动一下都产生可审计的物理事实，
> 计划第一次能持续自我优化——这才是 agent 敢管仓库的原因。

## 7. 世界模型概念校准（避免与生成式视频混淆）

- Ha & Schmidhuber, *World Models*, arXiv 1803.10122, 2018：世界模型 = 环境的紧凑可预测内部表征。
- Hafner et al., *DreamerV3*, Nature 2023：在想象（世界模型）中学习控制。
- LeCun, *A Path Towards Autonomous Machine Intelligence* (JEPA), 2022：预测世界状态支撑规划。
- 综述：*Understanding World or Predicting Future?* (arXiv 2411.14499, ACM CSUR 2025)——
  世界模型两派："理解世界" vs "预测未来"。
- 本项目取**经典控制论意义**：世界模型 = 可查询、可提交、可回放的世界状态机
  （本体+事务+事件日志+物理内核），与生成式视频世界模型（Sora/Genie）正交；
  后者可作为长期合成数据源（WM-DAgger 思路）接入。

## 8. LLM Agent × 机器人执行（agentic 层的文献）

- SayCan (Ahn et al., 2022)：LLM+可供性接地；Code as Policies (Liang et al., 2022)：LLM 写策略代码；
- SayPlan (Rana et al., 2023)：3D 场景图作世界模型供 LLM 规划——与本项目"本体查询"同构；
- Voyager (Wang et al., 2023)：LLM agent 在模拟器中终身学习+技能库；
- RAP: Reasoning with Language Model is Planning with World Model (Hao et al., 2023)：
  LLM 推理显式绑定世界模型搜索——本设计"agent 查询/提交、内核执行"的理论依据；
- WALL-E (Zhang et al., 2024, arXiv 2410.07484)：LLM agent + 学到的精确世界模型 + MPC，
  证明世界模型精度决定 agent 规划质量；
- LSMART (arXiv 2602.15721)：AGV 车队 FMS+MAPF 终身多智能体测试床（评测方法参照）。

## 9. GPT 独立检索的补充结论（会话 6acb00a1）

**裁决一致**：论点成立；创新点表述需升级——不说"机器人第一次闭合反馈回路"（控制论 70 年前
就做了），说"**将物理执行证据系统性地纳入企业级决策学习回路**"。

**第三条支持主线（我方检索遗漏，GPT 补上）**：
- Process Mining（van der Aalst / Adriansyah et al. 2012, *Replaying History on Process Models*）：
  事件日志×流程模型对齐 → 识别执行偏差/合规/瓶颈。传统系统只存计划与结果，
  执行细粒度事件第一次让"计划究竟怎样被执行"可重建。

**五处必须精确化（与本报告 §5 互补）**：
1. 偏差非人类独有 → "所有物理执行都存在不确定性"
2. 人工记录非全无 → "人工填报存在观测延迟、选择性记录、精度限制"
3. 机器人数据非天然完整 → "适当感知+记录的机器人**显著提高**可观测性"（遮挡/漂移/丢包同在）
4. **记录偏差 ≠ 归因**（Pearl 2009 因果推断）：晚到 20 秒可证明"晚了"，不自动证明"因锥桶"
5. 数据多 ≠ 计划好：需质量治理与可验证性

**警示文献**：Bainbridge 1983, *Ironies of Automation*——自动化把人留在更难的异常处置位；
理论必须保留人作为知识来源、异常解释者与最终授权者。

**三层产品模型（Execution Intelligence 叙事）**：
- L1 执行可观测性（传统控制可做）→ L2 决策可追溯性（业务系统可做）→
  **L3 跨任务决策学习（差异化所在）**：每次策略改进须能回答 旧策略/观察偏差/修改内容/
  验证测试/改善幅度/新风险 六问。
- 组件映射：MuJoCo=物理实验环境；传感器=执行观测；SQLite World=事实关联；
  DAG/调度器=计划约束；**Decision Memory=决策-依据-结果版本史**（新增概念）；
  Agent=意图与授权；优化引擎=假设检验与策略更新。
- 学习链条：Physical Execution → Execution Evidence → Decision Memory → Policy Improvement。
- 最有价值资产不是第 100 次成功，而是"第 37 次为什么失败、改派是否真比等待好"。
