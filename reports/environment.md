# W01 环境报告

日期: 2026-10-09 · 平台: macOS 27.0 (Build 26A428), Apple Silicon (arm64)

## 基线偏差记录（经负责人批准）

| 项 | 设计文档 | 实际 | 原因与证据 |
|---|---|---|---|
| Python | 3.10.14 | **3.12**（/opt/homebrew/bin/python3.12，负责人 2026-10-09 批准全项目升级） | scipy 1.15.3 cp310 macosx_arm64 轮子以旧 SDK 构建，`scipy/sparse/linalg/_propack/_spropack.cpython-310-darwin.so` 在 macOS 27 dyld 加载失败（`section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero`）。py3.10 的 scipy 上限即 1.15.3，无新版轮子。build123d 0.12.0 导入链依赖 scipy.sparse。实测 py3.12 + scipy 1.18.1 同平台加载正常。 |
| 3.10 补丁版本 | 3.10.14 | （已随升级弃用）3.10.19 | 本机 .local 仅 3.10.19；未采用 |

文档 9.2 预案的"单独提升 CAD 环境版本"未采用——3.12 全栈单环境更简单，mujoco/ortools/zxing-cpp/opencv 均支持 3.12。

## 锁定依赖（.venv, python 3.12）

以 `pip freeze > configs/requirements.lock` 为准（W01 完成时生成）。核心：
- mujoco（含相机渲染验证 std>0）
- build123d 0.12.0 + cadquery-ocp-novtk（cp312 wheel，61MB，导入+STEP 导出通过）
- opencv-python + zxing-cpp（QR 生成→解码往返通过: `P0001|3C`）
- ortools（CP-SAT 求解 OPTIMAL 通过）
- pydantic/fastapi/uvicorn/pytest/pyyaml/jsonschema/scipy/qrcode

## 最小功能路径验证（configs/env_check.py）

| 路径 | 结果 |
|---|---|
| CAD 导出 build123d | PASS（STEP 导出） |
| MuJoCo 步进 | PASS（自由落体轨迹正确） |
| MuJoCo 相机渲染 | PASS（240×320 渲染 std=10.5） |
| 图像解码 OpenCV+ZXing | PASS（生成→解码往返） |
| OR-Tools CP-SAT | PASS（OPTIMAL x=7） |
| 参数一致性 19/19 | PASS |

## 其他发现

- 无头渲染验证：MuJoCo Renderer 在本机可用（无需 GUI），W04 相机识读路线可行。
- 参数检查 19 项全过（configs/check.py）：18 货位、叉孔-货叉配合（单侧 4mm 间隙）、第三层 410/417mm 叉高、交接站 55/65/72mm 高度链、速度限值等。
