# PR 2 复用勘查报告

**生成日期**：2026-11-14 · Agent B
**基线**：`main @ e63f77b`
**目的**：为 PR 2（固定相机批量观测 worker）定位现有实现、复用点、最小改造方案。

---

## 6 条限制现状摘要

| # | 限制 | 现状 |
|---|---|---|
| 1 | 只接受无畸变/零 skew/`opencv_c2w`/`fx=fy` | **需小改** — `calibration_to_usd_camera`（`depth_observation.py` L44-82）和 `pinhole_intrinsics`（`cameras.py` L14-22）各加一个开关参数 |
| 2 | 标定绑定某个 PLY 和 `source_to_world` | **需小改** — `rendered_camera_calibration`（`camera_calibration.py` L18-19）将必需参数改为可选；`depth_observation.main()`（L372-375）的绑定校验拆出供批量 worker 复用 |
| 3 | 无光源时自动补光 | **需小改** — `depth_observation.py` L235-239 的 fallback DomeLight 和 `scene_preparation.py` L252-253 的无条件 DomeLight 各加显式策略参数 |
| 4 | 要求 RGB+深度同时成功 + 拒绝均匀图像 | **需小改** — `depth_observation.py` L257-271 拆出独立校验函数，通过 `capture_mode` 参数控制 |
| 5 | 输出含"repaired scene"文案 | **需小改** — `depth_observation.py` L306 和 L336 的硬编码文案参数化 |
| 6 | `metersPerUnit=1` 只建立数值单位关系 | **需小改** — `normalize_camera_z`（`depth_observation.py` L118-130）的单位白名单参数化；回执中的 `metric_scale_verified: False` **已满足**正确声明 |

---

## 复用清单统计

- **直接复用**：35 项
  - geometry 原语 7 项
  - execution/tasks 8 项
  - execution/processes 5 项
  - execution/workers 8 项
  - execution/resources 3 项
  - workflows/contracts 4 项
- **需抽取内部函数**：7 项
  - `depth_observation` 4 处
  - `camera_calibration` 1 处
  - `scene_preparation` 2 处
- **需新增**：6 项
  - `benchmark_observation.py` 5 个函数/入口
  - 编排 stage 1 个

---

## 关键发现

1. `depth_observation.py` 的核心验证逻辑（`calibration_to_usd_camera` L31-105 和 `normalize_camera_z` L108-161）已经是纯 CPU 可测试的公开函数，只需参数化而不需要重写。

2. `_capture()`（L168-342）是 Isaac-only 函数，包含场景打开、相机创建、光源检测、Replicator 捕获、哈希校验的完整流程。批量观测需要拆出单次视图捕获逻辑（`_capture_single_view`）和相机创建逻辑（`_create_observation_camera`）。

3. 巡览截图被误当参考相机渲染的最大风险点是 `scene_video.py` L228 和 `depth_observation.py` 共享相同的标定文件格式但语义不同。缓解方式是校验 `calibration_kind` 字段（L58 vs L300）。

4. 执行基础设施（`run_task_graph` + `execute_task` + `ResourcePool` + `device_lease` + `execute_container` + `StageLedger`）完整覆盖了 PR 2 所需的调度、GPU 租约、容器、取消、缓存、回执功能，**不需要新增任何第二套 scheduler**。

---

## 与 PR 1 合同的对接点

`core/benchmark_contract.py` 已提供：
- `validate_render_request` —— PR 2 worker 的输入合同
- `validate_view_set` —— 视图集合冻结
- `RENDER_MODES = {"rgb", "rgbd"}` —— 与 PR 2 的 RGB-only vs RGB-D 区分对齐
- `lighting.policy ∈ {"frozen", "scene_default"}` —— 与 PR 2 的显式光照冻结对齐

## 与 PR 5 脚手架的对接点

`tests/fixtures/fake_render_worker.py` 已定义 receipt 顶层键：
```
{synthetic_fixture, banner, worker, scene_id, seed, want_depth,
 render_settings, views: [{view_id, width, height, rgb_file, rgb_sha256, ...}],
 completed, expected, duration_seconds}
```
PR 2 真 worker 沿用同一布局，仅改 `synthetic_fixture=False` 并附真实证据。
