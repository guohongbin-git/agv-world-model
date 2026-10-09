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

## 目录

- `configs/` design.yaml + 参数/环境检查
- `cad/` build123d 参数化模型 → STEP/网格/碰撞体
- `sim/` MJCF、控制器、VirtualAdapter
- `scene/` 全场景、标签纹理、MuJoCo 相机
- `services/` perception / world / tasks / scheduler / adapters
- `tests/` T01-T16 用例（pytest）
- `reports/` 环境、验收证据（按 run_id 归档）
- `deliverables/` 影片与最终报告

## 当前状态

W01 环境验证进行中；W02-W07 未开始。完成判据见设计文档第 17.1 节。
