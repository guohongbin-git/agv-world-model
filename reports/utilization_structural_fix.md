# 利用率结构修复报告 (CP-SAT 均衡 + 闲置车预置 + 背压检查)

日期: 2026-10-12 · 执行: UtilBoost · 代码: sim/dispatch.py, twin/matrix_server.py, twin/web/matrix.html

## 1. 交付的三项改动

### 1.1 CP-SAT 负载均衡项 (sim/dispatch.py)
- 目标由 `1000·makespan + Σstart` 改为 `1000·makespan + Σstart + 2·max_v(load_v)`,
  `load_v = 批前累计偏置 + 本批指派给 v 的腿时长和 (tick)`。
- 批前偏置经新构造参数 `Dispatcher(load_offset_provider=...)` 注入 (跨批记忆);
  matrix_server 传入 **真实作业 busy** (总 busy − 预置空驶, 见 _real_busy_s),
  预置不与真任务抢"低负载"资格。
- **权重 = 2 的依据**: makespan 权重 1000 ⇒ 1 tick makespan ≈ 500 tick 峰值负载,
  而单批内换车的峰值摆幅典型 ≪500 tick —— 均衡只在 makespan/Σstart 平局带内
  起作用, 不撼动主目标; 实测 (6 轻货批, A03/A04 偏置 30000 tick):
  无偏置时 A01:2/A02:2/A03:1/A04:1 → 有偏置时 **A01:3/A02:3/A03:0/A04:0**
  (轻货 haul 全部落到 L1 快车)。
- 既有 16 个 test_dispatch 用例全绿 (含 CP-SAT 严格优于贪心、24 任务 <1s、
  确定性同种子同解 —— 均衡项与该场景的更优拆分方向一致, 未破坏)。

### 1.2 闲置车预置 / 巡游 (twin/matrix_server.py)
- 触发: 每 60s 滑动窗采样后 + 独立 12s 重试节拍 (`REPOS_TICK_S`):
  窗口利用率 <40% (`REPOS_UTIL_TH`) 且无在飞任务 → 预置移动, 同车冷却 60s,
  同时在途 ≤4 辆。
- 目标: 最近的高占用 stand 列 staging 节点 `x:i:2` (纯 transit 交点, 永非
  作业目标); 驻位车在 pk↔热点列 / 热点列之间**巡游交替** (多目标逐一尝试
  claim, 避开被真任务独占的热点列通道)。
- 语义: 预置 = 普通 nav 任务 (不占订单链, 无账本变更, 不走 commit_transfer);
  计入 busy (被派发的生产性空驶) 但**单列** `_repos_busy_s`。
- 抢占: 真任务 gate 前 `_repos_yield` 标记让位; **安全驻位状态机** —— 只允许
  驻在 `x:i:2..5` 或自身 pk (stand/dock/sf/sm/x:i:0/他人 pk 是真任务目标节点,
  驻其上 = 永久卡死该任务, 实测复现过 A02 驻 x:2:0 卡死 F02 standstrip);
  不安全节点则改向最近安全节点再驻。锁矩阵看门狗: 等待者的被占资源在预置
  路由/驻位车内 → 强制让位 (驻位车回泊)。
- 决策事件 `type="Reposition"` (solver_status=PLANNING) 进面板; feed 事件
  reposition / reposition_staged / reposition_preempt; 面板已加对应渲染分支。
- 样例事件 (seed7 实测):
  `{"type":"Reposition","vid":"A02","ts_s":66.18,"from":"pk:A02","to":"x:2:2",
    "reason":"滑动窗利用率 0% < 40% 且无在飞任务 → 预置到热点列 staging 节点 (真任务即刻抢占)",
    "solver_status":"PLANNING"}`
  feed: `{"t":76.84,"vid":"A02","kind":"reposition_staged","detail":"已驻位热点列 @ x:2:2"}`

### 1.3 背压检查 (lever 3, 只报告不改语义)
- **chains_open 全程顶格 24/24 = MAX_CHAINS**; 出库 stand 持有 4/4 =
  MAX_OUT_STANDS 大部分时间顶格 —— 并行度受这两个语义上限约束 (未放宽)。
- 发现 `_pump_demand` 背压交互: 队列一旦越过 BACKLOG_CAP=24
  (generate_until 整批入队可跳过阈值), 投喂与生成**同时**暂停 → 在飞链耗尽后
  系统断供空转 (实测队列冻结 25, open 24→18, 叉车闲置)。
- 实验过"只停生成、不停投喂"的修复: 更糟 (fleet 崩到 25%) —— 投喂恢复后
  补货单洪泛 (库存被出库开单扣减 → 阈值触发 REPLENISH 洪水 → fetch 等
  dock 载托 → dock 又等出库 haul → 环形饥饿, 队列 154, tasks=0)。
  **已回退该实验**, 原语义保留; 正确修法需要 demand 层决策 (如补货单
  限速/阈值), 不属于本任务三杠杆范围。

## 2. 实测对比 (seed7, multiplier 初始 8×, speed=0 headless)

### 调整前 (HEAD 基线, 15min 同口径对照, /tmp/baseline_head.json)
- 最终窗口 fleet **19.8%** (峰值 50% 后崩落); 累计 F01:41/F02:40/F03:41/F04:64,
  A01:9/A02:22/A03:23/A04:38; 932 sim-s 仅 11 腿/4 单, 多数车后期闲置。
- 用户实时证据 (live :8400, sim 1606s): fleet 49.9%, F01:62/F02:99/F03:74/F04:100,
  **A01:0/A02:0**/A03:38/A04:26。

### 调整后 (15min, reports/util_accept_run.json 为均衡+预置+回退背压实验前版本)
- 窗口 fleet 轨迹: 32→47→53→54→**60 (sim 400-460 峰)**→56→54→50 (末期);
  最终窗口 per-vehicle: F01:20/F02:74/F03:37/F04:82, A01:34/A02:42/A03:16/A04:99;
  **AGV 均值 47.6%** (>30% ✓), 叉车峰值 82% (<100% ✓), pen 0.14mm /
  穿透违规 0 / 物理失败 0; 预置 30 次移动。
- 结构性结论: A01/A02 从 0% 提升到 34%/42% (均衡+预置生效); 但 fleet 窗口
  末期回落 —— pick 腿物理时长 ~250-300s/腿 (DetFix 可靠性修复后的现实) +
  stand 4/5 上限 + 补货洪泛相互作用, 真实作业供给顶在 ~50%。

## 3. 验收判据状态
- [✓] test_dispatch.py 16/16 全绿 (含 CP-SAT vs 贪心)
- [✓] AGV 滑动窗 >30% (均值 47.6%), 叉车 <100% (峰值 82%)
- [✗] fleet 滑动窗 ≥60% **持续不满足**: 峰值达 60% (sim 400-460), 15min 末 50%
- 零碰撞/零物理失败/零停滞; 预置抢占与看门狗让位均有事件证据

## 4. 遗留 (需主控决策)
1. pick 腿物理时长 (~250-300s) 是 fleet 上限的根因之一 —— DetFix 领域。
2. 补货洪泛 × 投喂暂停的背压交互需要 demand 层方案 (补货限速或阈值上调);
   当前"冻结"行为反而是稳定器, 不宜单独解除。
3. MAX_OUT_STANDS=4/5 与 MAX_CHAINS=24 是语义上限, 按红线未动。
