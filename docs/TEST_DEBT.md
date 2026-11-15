# SimReady 测试债务清单（TEST_DEBT）

**版本**：v1.0 · 2026-11-14
**基线**：`main @ e63f77b`（v0.10.0）
**配套文档**：`docs/DEVELOPMENT_PLAN_20261114.md`
**本文档定位**：单独跟踪每层测试缺口与清算进度；不承载代码交付计划。

---

## 0. 状态图例

- ✅ 覆盖良好（无需本轮清算）
- ⚠️ 间接覆盖或部分覆盖（本轮清算）
- ❌ 几乎无直接测试（本轮清算）
- 🔵 缓补（下轮，需真实权重/数据/GPU）
- ☑ 已清算（PR 合入后勾选）

---

## 1. 必补（本轮，CPU 合成可测）

| # | 层 | 模块 | 行数 | 状态 | 拟新增测试文件 | 清算 PR | 完成 |
|---|---|---|---|---|---|---|---|
| 1 | core | `artifacts / evidence / validation / types / reconstruction_contract` | ~700 | ⚠️ | `tests/core/test_evidence.py` `test_validation.py` `test_reconstruction_contract.py` | PR 1 | ☐ |
| 2 | geometry | `cameras / depth_projection / polygons / transforms_validation` | 177 | ❌ | `tests/geometry/test_depth_projection.py` `test_polygons.py` `test_transforms_validation.py` | PR 1 | ☐ |
| 3 | CLI 层 | 6 个 `cli.py` | 299 | ❌ | `tests/architecture/test_cli_contracts.py` | PR 1 | ☐ |
| 4 | evaluation/inspection | `comparison / expectations / physical_checks / physical_receipt / scene_package` | 1031 | ❌ | `tests/evaluation/inspection/test_comparison.py` `test_expectations.py` `test_physical_checks.py` `test_physical_receipt.py` `test_scene_package.py` | PR 3 | ☐ |
| 5 | evaluation | `protocol / observations` | ~400 | ⚠️ | `tests/evaluation/test_protocol.py` `test_observations.py` | PR 3 | ☐ |
| 6 | assets/appearance | `gaussian_asset.py` | 218 | ❌ | `tests/assets/test_gaussian_asset.py` | PR 3 | ☐ |
| 7 | assets/association | `semantic_projection.py` | 504 | ⚠️ | `tests/assets/test_semantic_projection.py` | PR 3 | ☐ |
| 8 | assets/compilation | `collision_geometry / scene_preparation / spec_validation / usd_layers` | 776 | ❌ | `tests/assets/test_compilation.py` | PR 3 | ☐ |
| 9 | assets/physics | `category_priors / physical_authoring / physical_spec` | 540 | ⚠️ | 补入 `tests/assets/test_physical_spec.py`（与 simulation 侧呼应） | PR 2/3 | ☐ |
| 10 | execution/processes | `cancellation / local / remote_polling / supervision` | 424 | ⚠️ | `tests/execution/test_processes_edges.py` | PR 4 | ☐ |
| 11 | execution/resources | `gpu_capability / resource_pool` | 296 | ❌ | `tests/execution/test_resources.py` | PR 4 | ☐ |
| 12 | workflows/stages | `semantic_stage / video_execution` | ~300 | ⚠️ | `tests/workflows/test_stages_edges.py` | PR 4 | ☐ |
| 13 | workflows/planning | `pipeline_spec / video_contracts` | ~400 | ⚠️ | `tests/workflows/test_planning_schema.py` | PR 4 | ☐ |
| 14 | simulation | `camera_calibration / depth_observation / scene_preparation / gaussian_asset / physical_spec` 边界 | ~1500 | ⚠️ | `tests/simulation/test_camera_calibration_edges.py` `test_depth_observation_edges.py` `test_scene_preparation_edges.py` `test_gaussian_asset.py` `test_physical_spec.py` | PR 2 | ☐ |

### 1.1 各测试文件关键测试点

#### PR 1 承载

- **`tests/core/test_evidence.py`**：路径逃逸拒绝；`ensure_unchanged` 哈希篡改检测；循环引用；缺失引用
- **`tests/core/test_validation.py`**：非有限数、类型混淆、边界值、嵌套结构
- **`tests/core/test_reconstruction_contract.py`**：未知字段、重复 ID、fit/eval 重叠、哈希错配
- **`tests/geometry/test_depth_projection.py`**：反投影精度、退化深度、坐标系一致性
- **`tests/geometry/test_polygons.py`**：退化多边形、自交、空集
- **`tests/geometry/test_transforms_validation.py`**：变换逆一致性、非刚体拒绝
- **`tests/architecture/test_cli_contracts.py`**：6 个 CLI 的退出码、stderr 语义、参数解析、未知 flag 拒绝

#### PR 2 承载

- **`tests/simulation/test_camera_calibration_edges.py`**：畸变/skew/fx≠fy/非 opencv_c2w 全部拒绝；标定与 PLY 解耦后哈希变化检测
- **`tests/simulation/test_depth_observation_edges.py`**：RGB-only vs RGB-D；空白背景合法；近乎均匀图合法化；camera-z 与有效像素规则
- **`tests/simulation/test_scene_preparation_edges.py`**：缺光源策略显式拒绝；光照政策冻结后变化检测
- **`tests/simulation/test_gaussian_asset.py`**：PLY 加载失败、坐标系转换、USD 导出
- **`tests/simulation/test_physical_spec.py`**：`metersPerUnit=1` 不冒充实测尺度的文案检查

#### PR 3 承载

- **`tests/evaluation/test_protocol.py`**：schema 版本兼容、未知 protocol 拒绝、阈值与渲染配置分离
- **`tests/evaluation/test_observations.py`**：观测注册、视图 ID 唯一性、缺失视图
- **`tests/evaluation/inspection/test_comparison.py`**：正反例
- **`tests/evaluation/inspection/test_expectations.py`**：正反例
- **`tests/evaluation/inspection/test_physical_checks.py`**：正反例
- **`tests/evaluation/inspection/test_physical_receipt.py`**：正反例
- **`tests/evaluation/inspection/test_scene_package.py`**：正反例
- **`tests/assets/test_gaussian_asset.py`**：PLY 加载、坐标系、USD 导出
- **`tests/assets/test_semantic_projection.py`**：mask→3D 反投影精度、遮挡剔除、空 mask
- **`tests/assets/test_compilation.py`**：USD 层合成、碰撞几何生成失败、spec 验证正反例

#### PR 4 承载

- **`tests/execution/test_processes_edges.py`**：cancellation 边界、local 子进程清理、remote_polling 断线重连、supervision 超时
- **`tests/execution/test_resources.py`**：gpu_capability 检测、resource_pool 租约泄漏、并发租约
- **`tests/workflows/test_stages_edges.py`**：semantic_stage / video_execution 的取消传播、超时、恢复
- **`tests/workflows/test_planning_schema.py`**：pipeline_spec schema 演化、未知版本拒绝

---

## 2. 缓补（下轮，需真实权重/数据/GPU）

| # | 子项 | 拟新增测试文件 | 外部依赖 | 当前状态 | 清算 PR | 完成 |
|---|---|---|---|---|---|---|
| 15 | Grounding DINO + SAM 真实推理 | `tests/ingestion/test_grounded_real.py` | HF 权重 `IDEA-Research/grounding-dino-tiny @ a2bb814d…` + `facebook/sam-vit-base`；torch>=2.6 + transformers>=4.51 | **权重未缓存**，代码已就位 | PR 6a | ☐ |
| 16 | NeRF2Physics 真实特征回归 | `tests/ingestion/test_nerf2physics_real.py` | 上游 repo `ajzhai/NeRF2Physics` 钉 40 位 commit；真实 CLIP 特征缓存；真实密度 GT | **特征未生成** | PR 6b | ☐ |
| 17 | NeRF2Physics 真实质量指标 | `tests/evaluation/test_nerf2physics_real.py` | 上游参考实现输出 + 跨 scene GT | 指标公式已测，跨 scene 真实评测缺 | PR 6c | ☐ |
| 18 | Isaac Kit 真实渲染 | `tests/simulation/test_isaac_kit_render.py` | 云端 Isaac Sim 容器（用户决策：走云端，不用本机 `isaac-sim:6.1.0`）；固定两视图相机；参考像素 | 云端环境待建 | PR 6d | ☐ |
| 19 | 端到端真实链路 | `tests/workflows/test_real_pipeline_e2e.py` | 上述全部 + 真实视频/重建产物 | 缺 | PR 6e | ☐ |

### 2.1 opt-in 机制

- 统一 pytest marker：`@pytest.mark.real_model` / `@pytest.mark.gpu`
- 默认 `-m "not real_model and not gpu"` 跳过
- CI 通过 `--run-real-model` / `--run-gpu` 显式启用
- fake worker 结果**不得**计为真实渲染通过；GPU 项没跑就标"未验证"

### 2.2 已确认资源

- **云端 Isaac Sim**：PR 5 真实 GPU 回归走云端（用户决策 2026-11-14），本机容器不用
- 本机 GPU：NVIDIA GB10，驱动 580.159.03
- 模型缓存 `~/.cache/simready_models/`：目前仅 `dinov2_small`，grounding-dino / sam-vit-base **未下载**
- 现有 fixtures：`tests/fixtures/{docker_cli, pipeline, scene_candidate, scene_preparation, synthetic_task, task_execution, video_worker}.py`

### 2.3 PR 6 排期

**等 PR 1–5 合入后再启动**（用户决策 2026-11-14）。本轮不启动权重下载、不生成 CLIP 特征、不建云端 Isaac 环境。

---

## 3. 清算进度追踪

### 3.1 按 PR

| PR | 承载测试文件数 | 状态 |
|---|---|---|
| PR 1 | 7（core×3 + geometry×3 + CLI×1） | ☐ |
| PR 2 | 5（simulation 边界×5） | ☐ |
| PR 3 | 8（evaluation×2 + inspection×5 + assets×3 — `test_physical_spec.py` 与 PR 2 呼应） | ☐ |
| PR 4 | 4（execution×2 + workflows×2） | ☐ |
| PR 5 | 0（仅 CI/wheel 骨架，无新测试债务清算） | ☐ |
| PR 6（下轮） | 5（真实模型推理×5） | ☐ |

### 3.2 按层完成度

| 层 | 清算前 | 清算后目标 |
|---|---|---|
| core | ⚠️ | ✅ |
| geometry | ❌ | ✅ |
| CLI | ❌ | ✅ |
| evaluation/inspection | ❌ | ✅ |
| evaluation（其他） | ⚠️ | ✅ |
| assets/appearance | ❌ | ✅ |
| assets/association | ⚠️ | ✅ |
| assets/compilation | ❌ | ✅ |
| assets/physics | ⚠️ | ✅ |
| execution/processes | ⚠️ | ✅ |
| execution/resources | ❌ | ✅ |
| workflows/stages | ⚠️ | ✅ |
| workflows/planning | ⚠️ | ✅ |
| simulation 边界 | ⚠️ | ✅ |
| 真实模型推理（4 项） | 🔵 | 下轮 ✅ |

---

## 4. 清算纪律

1. 每个新增测试必须能证出一个故障或一项指标正确性；不允许只复制实现逻辑凑覆盖率。
2. 每个 PR 合入前：本领域测试 + `tests/architecture/test_boundaries.py` + merge-gate 全绿。
3. 测试文件合入后回本文件勾"完成"，并更新 3.1 / 3.2 进度。
4. 缓补项默认 skip；任何把 fake worker 当真实渲染通过的行为视为违反红线。
5. 真实模型推理的权重/特征/GT 准备情况，在 PR 6 启动前回本文件 2.2 节更新。
