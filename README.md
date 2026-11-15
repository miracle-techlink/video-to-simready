# Video to SimReady

核心 video → reconstruction → USD pipeline，以及隔离的 Isaac Sim 容器部署。

[![CI](https://github.com/miracle-techlink/video-to-simready/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/miracle-techlink/video-to-simready/actions/workflows/test.yml) · [开发者指南：架构、baseline 与 benchmark 接入](docs/README.md)

| 分支 | 内容 |
|---|---|
| `main` | 核心流水线、baseline 接入、资产与物性、独立验收、执行设施、Isaac Sim 部署及必要测试 |
| [`aloho-hackson`](https://github.com/miracle-techlink/video-to-simready/tree/aloho-hackson) | 分离前的完整版本：Aholo 黑客松、专用机器人任务、Agent 修复／经验、研究代码、视频物料与历史报告 |

main 保留 `artifact`、`command`、`aholo` 重建后端。Aholo 云端模型由服务方提供，本仓库实现客户端与下游处理；其他算法通过冻结输入、环境身份和统一 bundle 接入。

## 开发进度（每日更新）

- 日期口径：Asia/Shanghai（北京时间）；提交/合并时间由 GitHub 记录换算。
- 最近核对：2026-11-15 15:14 CST，远端 main `b25ec48ccb59bbe585c7f7fbf7664371c7dabf71`。
- 记录起点为公开仓库创建日（2026-11-10 15:38 CST），非任何云服务投产日期。无记录的日期仅表示“未找到可核验记录”，不推断当天无人工作。

| 日期 | 新增 / 合入 | 已验证功能与范围（限制） |
|---|---|---|
| 11-10 | 仓库创建；首个提交 [`d83d6e2`](https://github.com/miracle-techlink/video-to-simready/commit/d83d6e2)（带入的 PowerGrid/Aholo 黑客松历史内容） | 仅建立基线；初始历史内容中的功能不作为当日新验证声明 |
| 11-11 | —（未找到可核验记录） | — |
| 11-12 | [#1](https://github.com/miracle-techlink/video-to-simready/pull/1) 资源感知执行器（GPU UUID 租约/调度、哈希缓存、CPU fixture）；[#2](https://github.com/miracle-techlink/video-to-simready/pull/2) bounded agent/记忆/技能；[#3](https://github.com/miracle-techlink/video-to-simready/pull/3) simready 模块化与统一 CLI；[#4](https://github.com/miracle-techlink/video-to-simready/pull/4) 变电站场景 physics→独立审计→渲染回放 | #1 的 79 项 CPU 测试及合成调度/缓存演示，不代表真实 GPU 验证；#4 完成 20/20 检查与 714 帧 1080p 回放，复用已冻结 Aholo 场景/锚点，不是新视频重建或通用导航验证（[当日性能档案](https://github.com/miracle-techlink/video-to-simready/blob/ff5edab/docs/profiling.md)、[审计报告](https://github.com/miracle-techlink/video-to-simready/blob/ff5edab/docs/audits/2026-11-12-substation-profile.md)，文件现仅存在于该历史提交） |
| 11-13 | [#5](https://github.com/miracle-techlink/video-to-simready/pull/5) 场景工作流/机器人接口/Isaac 部署打包；[#6](https://github.com/miracle-techlink/video-to-simready/pull/6) 一条命令场景交付；[#7](https://github.com/miracle-techlink/video-to-simready/pull/7) 可恢复 new-video 流水线；[#8](https://github.com/miracle-techlink/video-to-simready/pull/8)–[#12](https://github.com/miracle-techlink/video-to-simready/pull/12) 文档（白皮书/README/Aholo API/物性计划）；[#13](https://github.com/miracle-techlink/video-to-simready/pull/13) 多类别语义资产与可审计物体物理 | #7 为 427 项测试 + 1 跳过（该 PR 时点数量）：旧视频 + 已声明的缓存 PLY 导出/恢复 + 12 帧 GPU 巡览，非新云重建、非外观质量/尺度/任务通过；#13 真实 DINO+SAM 图像运行 + RGBD、9 项合成 PhysX 响应通过——无语义 GT 与真实物性准确性结论 |
| 11-14 | [#14](https://github.com/miracle-techlink/video-to-simready/pull/14) 模块化流水线/阶段恢复/baseline 矩阵/merge-gate/main 绑定 SHA 安装包；[#15](https://github.com/miracle-techlink/video-to-simready/pull/15) 仓库清理；[#16](https://github.com/miracle-techlink/video-to-simready/pull/16) main 核心/专用演示分离；[#17](https://github.com/miracle-techlink/video-to-simready/pull/17)–[#19](https://github.com/miracle-techlink/video-to-simready/pull/19) 工程指南/示例整合/runtime 子层 | CI 侧 merge-gate 与 main 安装包绑定 SHA 交付上线（CPU 范围）；仓库外服务器完成真实门店视频 12 关键帧全新重建→USD→Isaac 巡览（`d65696a`，111,365 高斯点，21:12 CST `checked_at`）——尺度未验证、物性未评估、任务未验证、未修复的诊断性巡览。摘要见 [docs/progress/2026-11-14-acceptance.json](docs/progress/2026-11-14-acceptance.json)（历史验收记录，非当日 main 整链通过；其中 v6 静态修复预览完成日期未单独核实，另见今日“复核历史记录”） |
| 11-15 | 详见下节 | 详见下节 |

### 2026-11-15（今日，截至 15:14 CST）

**已合入 main**（HEAD [`b25ec48`](https://github.com/miracle-techlink/video-to-simready/commit/b25ec48ccb59bbe585c7f7fbf7664371c7dabf71)）：

- [#21](https://github.com/miracle-techlink/video-to-simready/pull/21) benchmark 基础设施（14:52 合入，`1aa5233`）：benchmark 输入合同、批量观测 worker 代码、candidate 独立评分、矩阵/suite 编排、CPU 合同测试、真实模型 opt-in 测试骨架。main CI（[run 43687418431](https://github.com/miracle-techlink/video-to-simready/actions/runs/43687418431)）：Python 3.10/3.12 CPU 回归、contract、package、merge-gate 全部通过——**CPU/打包范围**，不含真实模型/GPU。
- [#20](https://github.com/miracle-techlink/video-to-simready/pull/20) 外部开发者指南（15:02 合入）。
- [#22](https://github.com/miracle-techlink/video-to-simready/pull/22) CI 证据与验收指南（15:10 合入，`b25ec48`）：CI 失败证据保留（JUnit+artifact）、CI scope 报告（real_model/gpu 显式 NOT_RUN）、wheel smoke 安装加固、[手动验收指南](docs/ACCEPTANCE_RUNBOOK.md)。main push CI（[run 43688764667](https://github.com/miracle-techlink/video-to-simready/actions/runs/43688764667)）通过；GPU 的 Actions 自动通道**仍未接入**（仓库无 secrets/runner，本 PR 不改变这一点）。

**已验证记录（绑定运行版本）**：

- PR #21 合入前候选 [`bcd89d8`](https://github.com/miracle-techlink/video-to-simready/commit/bcd89d88579dfd7e38a4d7bc2dcfd4e1466e15fc)（代码现已随 PR #21 进入 main）：其上的独立 CPU 验收为全量 1274 通过/19 跳过/0 失败，真实 CLI suite 编排与防篡改/取消反例通过。计数属于当时运行版本，不代表最新 main 各平台计数。
- 历史基线 `8870f88` 上的真实 DINO+SAM 在门店视频 3 帧经生产入口运行成功（30 个检测实例、工件自洽核验）——执行正确性范围，无人工参考 mask，**检测/分割准确率未验证**。

**已提交待验 / 未合入**：

- benchmark GPU 入口真实复验发现启动/相对路径/prim 名/退出码问题、auto 输入包缺相机证据；修复补丁（`c4bed58` 等）已提交在 fix 分支但**未经正式复验，不算 GPU 通过**。历史基线 `8870f88` 上的 `depth_observation` 双视角成功仅为独立诊断，既不证明新入口、也不适用新补丁。

**开发中（无通过声明）**：真实图片 CLIP 新特征编码（`feat/real-image-feature-encoding`）与真实物性测量 GT 仍缺失（NeRF2Physics 公式与缓存特征回归已测）；`test/real-data-segmentation-acceptance`：COCO val2017 真实小样本评估与复核中；地瓜分支 `feat/digua-hackathon-drone-substation`：合成变电站 + PhysX 四旋翼 PID/预设航点录制实现中，无成片、无训练与自主导航结论。

**复核历史记录**：11-14 摘要 JSON 中的 v6 静态修复预览（`REPAIRED_PREVIEW_REVIEWED`）经今日复核确认记录存在；其 delivery/summary/quality-review JSON 均无完成/审核绝对时间字段，完成日期未单独核实（`date_status: not_recorded`），不计为 11-14 当日完成项。

### 下一步计划 / To-do

- [ ] **P0** 固定同一最终 SHA 打通 production 真实 GPU 多视角→candidate→独立评分→suite 全链。完成标准：进程退出码、输入/输出 hash、失败注入与恢复证据齐备并归档；必要修复合入 main。
- [x] **P0** 合入 #22（CI 证据/手动验收指南）并保持 main 检查通过。完成标准：PR 合并且 main CI 绿；GPU Actions 自动通道仍明确未接入。（已达成：15:10 合入 `b25ec48`，main push CI [run 43688764667](https://github.com/miracle-techlink/video-to-simready/actions/runs/43688764667) 通过；GPU 自动通道未接入状态未变）
- [ ] **P1** 真实图片 CLIP 新特征编码与缓存回放明确区分，并按真实输入/模型/协议给出真实 GT 分割评估报告。完成标准：报告含数据/模型 hash 与逐项指标，不以合成特征冒充。
- [ ] **P1** 地瓜黑客松首版：合成变电站四旋翼物理闭环悬停/航点跟踪 + 真实轨迹/碰撞检查 + 20–30s 视频。完成标准：物理闭环指标与录制成片；预设航点/脚本相机不写成已训练自主导航。
- [ ] **P1/P2** 以带尺度/位姿/几何参考的真实小数据集（TUM/ScanNet++ 等按许可）验收重建与碰撞空间；未知区域保持未知。完成标准：独立参考下的误差报告；静态导航不以家具精确质量为先决条件。
- [ ] **P2** 机器人/无人机感知定位→规划避障→控制反馈→巡检覆盖闭环，再做导航/飞行训练与独立任务验收。
- [ ] **P2** 受控 GPU CI：明确资源/凭据/固定模型镜像与数据/失败证据；服务器自动部署与回滚为后续项，不伪称现有 CD 已包含。

### 维护规则

- 合并影响功能、验收或计划的 PR 时，同步更新当日条目（本表由人工维护，未部署自动定时任务）。
- 一条“已验证”必须绑定实际版本（commit/CI run/产物）；不同版本的验证不得拼成一次全链验收。
- 没有记录的日子写“未找到可核验记录”，不虚构功能或结果。

## CPU 快速开始

需要 Python 3.10+ 与系统 FFmpeg；无需 GPU、账号或私有数据。

```bash
git clone https://github.com/miracle-techlink/video-to-simready.git
cd video-to-simready
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[physical-assets]'
python examples/pipeline/create_fixture.py --out /tmp/simready-inputs
simready run --pipeline /tmp/simready-inputs/pipeline.json --out /tmp/simready-run
simready run --pipeline /tmp/simready-inputs/pipeline.json --out /tmp/simready-run --resume
```

此示例使用合成输入，验证 USD 导出、独立参数读回和恢复。真实视频与重建复用见[视频示例](examples/pipeline/README.md#真实视频)，多算法对比见[baseline 矩阵](examples/reconstruction/README.md#比较后端)。

## Isaac Sim worker

```bash
docker build -t simready-isaac-core:6.1.0 -f deploy/isaac/Dockerfile deploy/isaac
python scripts/launch_isaac.py --assets examples/pipeline --out runs/isaac-check \
  --doctor --launch-smoke --accept-eula
```

执行前准备 Docker、NVIDIA Container Toolkit、适用的 NVIDIA 许可与 GPU 环境。源码与输入只读挂载；镜像不包含场景、模型权重或密钥。环境身份和运行约束见[容器与运行](docs/README.md#容器与运行)。

## 开发边界

模块分工、代码构造、baseline／benchmark 接入及测试流程统一见[开发者指南](docs/README.md)，包含可运行的后端接入与独立评分示例。公开示例验证接口与恢复，完整几何、真实物性与模型精度需各自评测。唯一命令入口为 `simready`；专用场景、Agent 及旧 CLI 别名使用完整分支。
