# SimReady Infra 开发计划（含测试债务清算）

**版本**：v1.0 · 2026-11-14
**基线**：`main @ e63f77b`（v0.10.0），工作区 `/home/nvidia/algotithm/simready_infra_pr`
**上游交接**：`/home/nvidia/Desktop/SimReady_infra开发交接_20261114.md`（PR #20 附带，纯 docs）
**本文档定位**：在交接文档 PR 1–5 基础上，**补全每层测试债务清单**，并把"真实模型推理"显式单独立项。不改变交接文档的范围红线。

---

## 0. 全局红线（承接交接文档）

1. 不重做已完成的模块拆分、嵌套取消、GPU 租约。
2. 不把 `aloho-hackson` 整支合回 main。
3. 不伪造真实运行结论："已实现且实测" / "仅 CPU 或合成验证" / "未实现缺外部输入" 三栏分列。
4. fake worker 结果不得计为真实渲染通过；GPU 项没跑就标"未验证"。
5. 测试不为了凑覆盖率而复制实现逻辑；每个新增测试必须能证出一个故障或一项指标正确性。

---

## 1. 每层测试缺口全景（实测盘点，非估计）

测试状态图例：✅ 覆盖良好 · ⚠️ 间接覆盖或部分覆盖 · ❌ 几乎无直接测试

| # | 层 | 模块 | 行数 | 状态 | 主要缺口 |
|---|---|---|---|---|---|
| 1 | core | `artifacts / evidence / validation / types / reconstruction_contract` | ~700 | ⚠️ | 仅 `test_json_io.py`；路径逃逸、`ensure_unchanged` 哈希篡改、循环引用、边界值未直接测 |
| 2 | ingestion/segmentation | `grounded.py / semantic.py` | 602 + 616 | ⚠️ | NMS/别名/参数拒绝已测；**真实 Grounding DINO + SAM 推理零测试**；mask 后处理、空检测、CUDA OOM、权重下载失败未测 |
| 3 | ingestion/reconstruction | `aholo_client / aholo_reconstruction / providers / stage` | ~1500 | ✅ | 覆盖较好，仅补边界 |
| 4 | ingestion | `nerf2physics.py / video_frames.py` | 339 + ? | ⚠️ | 合成 fixture 已测；**真实 CLIP 特征/真实质量预测零测试**；损坏视频、fps 漂移、时间戳错乱未测 |
| 5 | simulation | `camera_calibration / depth_observation / depth_projection / gaussian_asset / scene_preparation / category_priors / physical_authoring / physical_spec / …`（14 模块） | ~3000 | ⚠️ | `depth_observation/scene_video/physical_probe` 有测；其余边界弱；**真实 Isaac Kit 渲染零测试** |
| 6 | evaluation | `scoring / verification / protocol / observations / asset_graph` | ~1100 | ⚠️ | `verification/candidate_scoring/asset_graph` 有测；`protocol.py` schema 演化、`observations.py` 注册未专测 |
| 7 | evaluation/inspection | `comparison / expectations / physical_assets / physical_checks / physical_receipt / scene_package` | 1031 | ❌ | 几乎全靠 `test_physical_assets.py` 间接覆盖；其余 5 个模块零直接测试 |
| 8 | assets/appearance | `gaussian_asset.py` | 218 | ❌ | **零直接测试**：PLY 加载、坐标系转换、USD 导出 |
| 9 | assets/association | `semantic_projection.py` | 504 | ⚠️ | 间接覆盖；mask→3D 反投影精度、遮挡剔除未直接测 |
| 10 | assets/compilation | `collision_geometry / scene_preparation / spec_validation / usd_layers` | 776 | ❌ | **零直接测试**：USD 层合成、碰撞几何、spec 验证、场景准备 |
| 11 | assets/physics | `category_priors / physical_authoring / physical_candidate / physical_spec` | 687 | ⚠️ | candidate 打包已测；其余 3 个弱 |
| 12 | execution/processes | `cancellation / local / remote_polling / supervision` | 424 | ⚠️ | 间接覆盖；cancellation 边界、local 清理、remote_polling 断线、supervision 超时直接测试不足 |
| 13 | execution/resources | `gpu_capability / resource_pool` | 296 | ❌ | **零直接测试**：GPU 能力检测、资源池租约泄漏 |
| 14 | execution/tasks | `cache / environment / graph / schema / worker` | ~700 | ✅ | 覆盖较好 |
| 15 | execution/workers | `device_lease / device_recovery / isaac_container / isaac_environment / isaac_launcher / profiling / worker_groups` | ~1200 | ✅ | device_lease/isaac_lifecycle 有测 |
| 16 | workflows/stages | `physical_estimation / semantic_stage / video_execution` | ~500 | ⚠️ | `physical_estimation` 已测；其余 2 个边界弱 |
| 17 | workflows/pipelines | `baseline_matrix / entry / pipeline_config / semantic_assets / video_pipeline` | ~1500 | ✅ | 覆盖较好 |
| 18 | workflows/planning | `pipeline_spec / video_contracts` | ~400 | ⚠️ | 间接覆盖；schema 演化弱 |
| 19 | workflows/contracts | `stage_ledger / validation` | ~600 | ✅ | stage_ledger 覆盖好 |
| 20 | geometry | `cameras / depth_projection / polygons / transforms_validation` | 177 | ❌ | 仅 `test_cameras.py`；其余 3 个零直接测试 |
| 21 | CLI 层 | 6 个 `cli.py` | 299 | ❌ | **零直接测试**：参数解析、退出码、stdout/stderr 语义 |

**总计**：21 层中 ❌ 6 层、⚠️ 11 层、✅ 4 层。

---

## 2. 还缺什么（按"必补 / 缓补"分两档）

### 2.1 必补（本轮，CPU 合成可测）

全部为基础设施层的单元/契约测试，不需要真实权重、真实数据、GPU。

| 缺口层 | 测试文件（拟新增） | 关键测试点 |
|---|---|---|
| core | `tests/core/test_evidence.py` | 路径逃逸拒绝；`ensure_unchanged` 哈希篡改检测；循环引用；缺失引用 |
| core | `tests/core/test_validation.py` | 非有限数、类型混淆、边界值、嵌套结构 |
| core | `tests/core/test_reconstruction_contract.py` | 未知字段、重复 ID、fit/eval 重叠、哈希错配 |
| geometry | `tests/geometry/test_depth_projection.py` `test_polygons.py` `test_transforms_validation.py` | 反投影精度、退化多边形、变换逆一致性 |
| CLI | `tests/architecture/test_cli_contracts.py` | 6 个 CLI 的退出码、stderr 语义、参数解析、未知 flag 拒绝 |
| evaluation/inspection | `test_comparison.py / test_expectations.py / test_physical_checks.py / test_physical_receipt.py / test_scene_package.py` | 各模块正反例、故障注入 |
| evaluation | `tests/evaluation/test_protocol.py / test_observations.py` | schema 版本兼容、未知 protocol 拒绝、视图注册唯一性、缺失视图 |
| assets/appearance | `tests/assets/test_gaussian_asset.py` | PLY 加载失败、坐标系转换、USD 导出 |
| assets/association | `tests/assets/test_semantic_projection.py` | mask→3D 反投影、遮挡剔除、空 mask |
| assets/compilation | `tests/assets/test_compilation.py` | USD 层合成、碰撞几何、spec_validation 正反例 |
| execution/processes | `tests/execution/test_processes_edges.py` | cancellation 边界、local 子进程清理、remote_polling 断线重连、supervision 超时 |
| execution/resources | `tests/execution/test_resources.py` | gpu_capability 检测、resource_pool 租约泄漏、并发租约 |
| workflows/stages | `tests/workflows/test_stages_edges.py` | semantic_stage / video_execution 的取消传播、超时、恢复 |
| workflows/planning | `tests/workflows/test_planning_schema.py` | pipeline_spec schema 演化、未知版本拒绝 |
| simulation 边界 | `tests/simulation/test_camera_calibration_edges.py / test_depth_observation_edges.py / test_scene_preparation_edges.py / test_gaussian_asset.py / test_physical_spec.py` | 畸变/skew/fx≠fy 拒绝、RGB-only vs RGB-D、空白背景合法、光照政策冻结、`metersPerUnit=1` 不冒充实测尺度 |

### 2.2 缓补（下轮，需真实权重/数据/GPU）

全部为真实模型推理与真实渲染的端到端测试，默认 skip，需显式 opt-in。

| 子项 | 测试文件（拟新增） | 外部依赖 | 当前状态 |
|---|---|---|---|
| Grounding DINO + SAM 真实推理 | `tests/ingestion/test_grounded_real.py` | HF 权重 `IDEA-Research/grounding-dino-tiny @ a2bb814d…` + `facebook/sam-vit-base`；torch>=2.6 + transformers>=4.51；GPU 或 CPU（慢） | **权重未缓存**，代码已就位，测试默认 skip |
| NeRF2Physics 真实特征回归 | `tests/ingestion/test_nerf2physics_real.py` | 上游 repo `ajzhai/NeRF2Physics`（需钉 40 位 commit）；真实 CLIP 特征缓存；真实密度 GT | **特征未生成**，公式测试已就位，真实对照缺 |
| NeRF2Physics 真实质量指标 | `tests/evaluation/test_nerf2physics_real.py` | 上游参考实现输出 + 跨 scene GT | 指标公式已测，跨 scene 真实评测缺 |
| Isaac Kit 真实渲染 | `tests/simulation/test_isaac_kit_render.py` | Isaac Sim 容器 `nvcr.io/nvidia/isaac-sim:6.1.0`（**已在本机运行 3 天，healthy**）；固定两视图相机；参考像素 | **容器已可用**，像素对照测试缺 |
| 端到端真实链路 | `tests/workflows/test_real_pipeline_e2e.py` | 上述全部 + 真实视频/重建产物 | 缺 |

**已确认资源**（截至 2026-11-14）：
- **云端 Isaac Sim**：PR 5 真实 GPU 回归走云端（用户决策 2026-11-14），本机容器 `isaac-sim:6.1.0` 不用。
- 本机 GPU：NVIDIA GB10（驱动 580.159.03）。
- 模型缓存目录 `~/.cache/simready_models/` 已有 `dinov2_small`，**尚无 grounding-dino / sam-vit-base**。
- 现有 fixtures：`tests/fixtures/{docker_cli, pipeline, scene_candidate, scene_preparation, synthetic_task, task_execution, video_worker}.py`。

**缓补项的 opt-in 机制**：统一用 pytest marker（`@pytest.mark.real_model` / `@pytest.mark.gpu`），默认 `-m "not real_model and not gpu"` 跳过，CI 通过 `--run-real-model` / `--run-gpu` 显式启用。

**测试债务单独跟踪**：所有清算项与进度见 `docs/TEST_DEBT.md`，本文不再承载细粒度清单。

---

## 3. 开发计划（5 PR + 测试债务嵌入）

### 总策略

- 测试补充**嵌入对应 PR**，不另起炉灶，避免测试债再积一轮。
- 真实模型推理**单独立 PR 6**，标记 blocked-by PR 1–5 + 外部输入。
- 每个 PR 都必须跑：本领域测试 + `tests/architecture/test_boundaries.py` + merge-gate。

### PR 1：冻结 benchmark 输入合同 + core/geometry/CLI 测试加固

**代码交付**（承接交接文档）：
- 新增 `core/benchmark_contract.py`：版本化 schema + 纯校验器（dataset / view set / render request / evaluation request）。
- 正反例测试。

**测试债务交付**：
- `tests/core/test_evidence.py` `test_validation.py` `test_reconstruction_contract.py`
- `tests/geometry/test_depth_projection.py` `test_polygons.py` `test_transforms_validation.py`
- `tests/architecture/test_cli_contracts.py`（6 个 CLI）

**验收**：交接文档 PR 1 验收矩阵 + 上述测试全绿 + `test_boundaries.py` 通过。

---

### PR 2：固定相机批量观测 worker + simulation 边界测试

**代码交付**：
- 新增 `simulation/benchmark_observation.py`。
- 复用 `depth_observation.py`，解决畸变/skew/fx≠fy 拒绝、标定解耦、光照冻结、RGB-only vs RGB-D、`metersPerUnit` 尺度声明。

**测试债务交付**：
- `tests/simulation/test_camera_calibration_edges.py`
- `tests/simulation/test_depth_observation_edges.py`
- `tests/simulation/test_scene_preparation_edges.py`
- `tests/simulation/test_gaussian_asset.py`
- `tests/simulation/test_physical_spec.py`

**验收**：至少两组不同相机的真实 Kit 捕获 + 像素投影对照；崩溃/超时/取消不发布完整回执。

---

### PR 3：candidate 组装与独立评分 + evaluation/assets 测试加固

**代码交付**：
- 新增 `workflows/stages/benchmark_evaluation.py`。
- candidate builder 从已核查 render receipt 构建；区分执行状态与质量状态。

**测试债务交付**：
- `tests/evaluation/test_protocol.py` `test_observations.py`
- `tests/evaluation/inspection/test_comparison.py` `test_expectations.py` `test_physical_checks.py` `test_physical_receipt.py` `test_scene_package.py`
- `tests/assets/test_gaussian_asset.py` `test_semantic_projection.py` `test_compilation.py`

**验收**：FAIL/INCOMPLETE 不被包装成执行成功；缓存命中也重新核查；篡改任一图像/receipt/protocol 不能复用旧成绩。

---

### PR 4：矩阵自动评测与多场景汇总 + execution/workflows 测试加固

**代码交付**：
- `baseline_matrix.py` 增加自动 render/evaluate 选项；新增 `workflows/pipelines/benchmark_suite.py`（薄编排）。
- 外部 candidate / 自动 candidate / 重建中间产物三模式。

**测试债务交付**：
- `tests/execution/test_processes_edges.py`
- `tests/execution/test_resources.py`
- `tests/workflows/test_stages_edges.py`
- `tests/workflows/test_planning_schema.py`

**验收**：2 场景 × 2 成员注入 1 失败仍保留 4 行；中途取消停止下游；分母=计划样本。

---

### PR 5：CI、安装包、真实 GPU 回归

**代码交付**：
- CPU CI 增加合同、fake worker 故障注入、candidate 构造、独立评分测试。
- 扩展 wheel 仓库外 smoke。
- 真实 GPU 回归（**云端 Isaac Sim**，用户决策 2026-11-14）：固定两视图捕获 + 取消/超时清理；本机容器不用。

**验收**：wheel 装仓库外可跑；GPU 项没跑就标"未验证"；云端 Isaac 环境的镜像 ID、设备能力、日志、输出哈希全部入证据目录。

---

### PR 6（缓补，下轮）：真实模型推理接入

**前置**：PR 1–5 合入 + 外部输入就绪。**排期决策（2026-11-14）**：等 PR 1–5 合入后再启动，本轮不下载权重、不生成特征、不建云端 Isaac 环境。

| 子项 | 交付 | 当前缺口 |
|---|---|---|
| 6a Grounding DINO + SAM | `test_grounded_real.py` + 权重缓存 + 参考 mask IoU > 阈值 | 权重未下载 |
| 6b NeRF2Physics 特征回归 | `test_nerf2physics_real.py` + 真实 CLIP 特征 + 与上游对齐 | 特征未生成、上游 commit 未钉 |
| 6c NeRF2Physics 质量指标 | `test_nerf2physics_real.py`（evaluation）+ 跨 scene GT | 跨 scene GT 缺 |
| 6d Isaac Kit 渲染 | `test_isaac_kit_render.py` + 两视图像素对照 | 容器已就绪，测试缺 |
| 6e 端到端 | `test_real_pipeline_e2e.py` | 全部上游就绪后才能跑 |

**opt-in 机制**：pytest marker `real_model` / `gpu`，默认 skip。

---

## 4. 并行执行安排

### 依赖图

```
PR 1（合同 + core/geometry/CLI 测试）
  ├─→ PR 2（渲染 worker + simulation 测试）
  │     └─→ PR 3（candidate + evaluation/assets 测试）
  │           └─→ PR 4（矩阵汇总 + execution/workflows 测试）
  │                 └─→ PR 5（CI + wheel + GPU 回归）
  │                       └─→ PR 6（真实模型推理，下轮）
  └─→ PR 5 脚手架部分（fake worker、fixtures、wheel smoke 骨架）可并行
```

### Agent 分工（建议）

| Agent | 任务 | 与谁并行 | 何时启动 |
|---|---|---|---|
| **A** | PR 1 全部（合同 schema + core/geometry/CLI 测试） | B、C | 立即 |
| **B** | PR 2 设计/复用勘查（只读，产出复用清单 + 现状差距报告） | A、C | 立即 |
| **C** | PR 5 脚手架（CPU CI fake worker + fixtures 目录 + wheel smoke 骨架） | A、B | 立即 |
| **D** | PR 2 本体（等 A 出合同字段后） | — | A 完成后 |
| **E** | PR 3 本体（等 D 出 render receipt 后） | — | D 完成后 |
| **F** | PR 4 本体（等 E 出 candidate builder 后） | — | E 完成后 |
| **G** | PR 5 本体（等 F 完成后） | — | F 完成后 |
| **H** | PR 6（下轮，等外部输入） | — | PR 1–5 合入后 |

**说明**：
- A/B/C 可立即并行（B 是只读勘查，C 不依赖具体字段）。
- D/E/F/G 严格串行（每个都依赖上游产物身份）。
- H 独立排期。

### 工作分支策略（用户决策 2026-11-14）

- 从 `main @ e63f77b` 切 5 个 feature 分支：`feat/benchmark-contract` `feat/benchmark-observation` `feat/benchmark-evaluation` `feat/benchmark-suite` `feat/ci-wheel-gpu`。
- **不在** `docs/developer-guide-baselines-benchmarks`（PR #20 源分支）上接着写。
- **不动** `aloho-hackson` 分支。
- 每个 PR 合入后，下一个分支 rebase 到最新 main。

---

## 5. 交付自检三栏（每个 PR 完成后必填）

| 栏 | 内容 |
|---|---|
| 已实现且实测 | 列出真实跑过的代码 + 测试 + 命令 |
| 仅 CPU / 合成验证 | 列出只在合成夹具上验证的部分 |
| 未实现 / 缺外部输入 | 列出缺失的真实模型、数据、镜像、GPU 项，以及解锁条件 |

---

## 6. 已确认的 4 项决策（2026-11-14）

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 分支基线 | ✅ 从干净 main `e63f77b` 切 5 个 feature 分支；不在 PR #20 源分支上接着写 |
| 2 | GPU 回归落点 | ✅ 走**云端 Isaac Sim**；本机 `isaac-sim:6.1.0` 容器不用 |
| 3 | PR 6 排期 | ✅ 等 PR 1–5 合入后再启动；本轮不下载权重、不生成特征、不建云端 Isaac 环境 |
| 4 | 测试债务跟踪 | ✅ 单独起 `docs/TEST_DEBT.md`，本文档不再承载细粒度清单 |

**下一步**：按 Agent 分工立即并行派 A（PR 1）/ B（PR 2 只读勘查）/ C（PR 5 脚手架）；A 完成后串行 D→E→F→G。
