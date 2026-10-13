# 硬编码普查 (2026-10-13, 只报未修)

来源: superpowers 文档审查波 · 硬编码切片 (scout 只读)。
结论: **configs/design.yaml 的「单一参数源」实际只覆盖 M1 场景**
(pallet/fork/rack/station/precision/docking 等); M2 矩阵主线的全部
运行控制参数以模块级常量存在, 无单一来源、无 design_version 联动通道。

## A. 运行控制参数 → 应入 design.yaml [tuning]

| 常量 | 位置 | 风险 | 级别 |
|---|---|---|---|
| DEMAND_MULT_MAX=3.0 | matrix_server:1107 | 防雪崩核心, 报告引用 3× | **high** |
| util<0.60→×1.3 / >0.85→×0.85 | matrix_server:1125-28 | README/AGENTS 公开宣称行为 | **high** |
| saturated = 0.8×MAX_CHAINS | matrix_server:1124 | v2.4.0 饱和语义 | med |
| MAX_CHAINS=24 / BACKLOG_CAP=24 | matrix_server:53-56 | 队列/链上限 | med |
| UTIL_SAMPLE_S/UTIL_WIN | matrix_server:58 | 改了会变利用率读数口径 | med |
| REPOS_UTIL_TH=0.40 等 5 项 | matrix_server:61-65 | 预置行为 | low-med |
| 批次阈值 <3 单/60s | matrix_server:649 | 派单节奏 | low |
| 吞吐拖尾窗 600s | matrix_server:1167 | tasks/h 指标口径 | low |
| CP-SAT 预算/权重/作业时长档 | sim/dispatch.py | 调度质量 | med |
| A* 罚项 _STRAFE/_TURN/_LOCK/_NODE, 0.040 sf 偏移, 载货减速 0.6/0.5 | sim/navigation.py | 路径行为 | med |

## B. 双真值 (sync 风险, 必须收敛)

1. `sim/dispatch.py:45 VEHICLE_LEVELS` (L1/L2/L3 载重 0.08/0.15/0.…)
   ↔ `sim/warehouse_cell.py VEHICLE_CLASSES` —— 载重/速度档位两处声明
2. `FLEET_STUB` 停车位 ↔ `LAYOUT.park` —— 泊位两处声明

另: `sim/demand.py _SEED_PALLETS=48` ↔ `LAYOUT.seed_pallets` 双声明。

## C. 布局派生几何写死 (sim/mission.py, 应由 LAYOUT 派生)

越台 guard 0.125/0.102/0.115、托板顶 0.1115、释放高度 z+0.010
(DetFix 一轮, 有实测注释)、对中 ±0.010/2mm 等。

## D. 已修

- matrix_server MAX_CHAINS=16 死常量 (后 24 覆盖) 已删除 (本次)。
- design.yaml 2.3.0→2.4.0 联动已修 (本次, DocAuditDesign 发现)。

## 处置建议

参数化属行为等值重构, 涉及 sim/twin 多文件 + 测试口径
(UTIL 窗口径改动会变利用率读数), 单独立项一个波次执行:
先 HIGH 三项 (需求钳制组) 入 yaml → 双真值收敛 → 布局派生几何。
执行窗口: 压测 #3 判定后 (避免中途改变被测行为)。
