# 真值隔离审计 (GPT 攻击三复核) — 2026-10-13

背景: GPT 尽调提出最深的证伪点——"若证据判据读仿真真值, 实验退化为
知道真值的系统更准"。scout 初版审计误用 /tmp/agv_head 过期副本
(误报 fault_ablation.py 不存在); 本版为规范仓库逐行复核结论。

## 裁决: 架构级隔离 PASS, 外部效度缺口一项

### 隔离机制 (experiments/fault_ablation.py, 已实现)

| 通道 | 内容 | 可读者 |
|---|---|---|
| `FakeCell.truth` | 运动学真值 (xyz/fork/beam/speed/claim) | **evaluator 专读** (completed 判定/divergence) |
| `FakeCell.observed` | 传感器口径 (pose/bottom_z/contact/support) | 生产侧唯一接口 (WorldService→五门判据) |
| 故障注入 | **篡改 observed 覆盖项** (S01 虚报抬起等) | 注入清单落日志 `observed_overrides`, 判据路径不可见 |

- twin/server.py `_sample_verdict` (622-665) 读 `c.pallet_pose` /
  `c.support_load_ratio` —— 在 harness 内 `c` = FakeCell, 生产侧接口
  返回 **observed 口径** (218-238 行), 非真值。
- 决策侧另有 AST 强制隔离 (tests/test_observation_isolation.py:
  观测通道 = mj_ray LiDAR + σ=3mm + 2% 丢束, 零真值引用)。
- 组间公平: B0-B3 共享同一执行内核与 observed 通道, 差异仅在决策层
  (EXPERIMENT_DESIGN.md §4), 由构造保证同分布。

### 残留缺口 (诚实披露)

1. **传感器默认理想**: observed==truth (无噪声/延迟/漂移), S01 类虚报
   靠注入显式篡改。真实传感器会自然漂移——这是外部效度的剩余缺口。
   缓解: (a) 论文明确 scope "ideal-sensor regime + 注入式故障";
   (b) 加噪声包装器 (audit 建议 #1), 或由 DC 孪生场景 P1 虚拟摄像头
   噪声模型承接。
2. **恢复对账 API**: B3 重新观测走同一 observed 口径 ✓ (非真值直读)。
3. twin/server.py 在真实 MuJoCo 模式下 `c.pallet_pose` 是真值直读
   (理想传感器) —— 与 1 同源, 属 sensor-model 议题非泄漏。

### 对 GPT 攻击三的回应措辞 (Paper 用)

"判据输入严格取自传感器口径通道 (observed); 仿真真值仅供独立评分器;
故障注入通过篡改传感器通道实现且对判据路径不可见。当前处于理想传感器
regime, 噪声/延迟鲁棒性列为后续工作 (A9)。"
