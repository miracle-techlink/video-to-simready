# 开发进度详细记录（归档快照）

> 本文档是 **2026-11-15 19:30 CST 的归档快照**，不是实时状态。最新动态见 [README 的 Latest News](../README.md#latest-news)。
> 维护说明：有实质变更的当天更新 README 动态或本文档；无进展无需凑条目。一条“已验证”必须绑定实际版本（commit/CI run/产物）；不同版本的验证不得拼成一次全链验收；没有记录的日子写“未找到可核验记录”，不虚构功能或结果。

## 开发进度（每日更新）

- 日期口径：Asia/Shanghai（北京时间）；提交/合并时间由 GitHub 记录换算。
- 最近核对：2026-11-15 19:30 CST；本次 PR 基于远端 main `99c69e0b1a9c9b073254db3c600b82b88e0af22c`，本段另记录之后完成的服务器演示。
- 记录起点为公开仓库创建日（2026-11-10 15:38 CST），非任何云服务投产日期。无记录的日期仅表示“未找到可核验记录”，不推断当天无人工作。

| 日期 | 新增 / 合入 | 已验证功能与范围（限制） |
|---|---|---|
| 11-10 | 仓库创建；首个提交 [`d83d6e2`](https://github.com/miracle-techlink/video-to-simready/commit/d83d6e2)（带入的 PowerGrid/Aholo 黑客松历史内容） | 仅建立基线；初始历史内容中的功能不作为当日新验证声明 |
| 11-11 | —（未找到可核验记录） | — |
| 11-12 | [#1](https://github.com/miracle-techlink/video-to-simready/pull/1) 资源感知执行器（GPU UUID 租约/调度、哈希缓存、CPU fixture）；[#2](https://github.com/miracle-techlink/video-to-simready/pull/2) bounded agent/记忆/技能；[#3](https://github.com/miracle-techlink/video-to-simready/pull/3) simready 模块化与统一 CLI；[#4](https://github.com/miracle-techlink/video-to-simready/pull/4) 变电站场景 physics→独立审计→渲染回放 | #1 的 79 项 CPU 测试及合成调度/缓存演示，不代表真实 GPU 验证；#4 完成 20/20 检查与 714 帧 1080p 回放，复用已冻结 Aholo 场景/锚点，不是新视频重建或通用导航验证（[当日性能档案](https://github.com/miracle-techlink/video-to-simready/blob/ff5edab/docs/profiling.md)、[审计报告](https://github.com/miracle-techlink/video-to-simready/blob/ff5edab/docs/audits/2026-11-12-substation-profile.md)，文件现仅存在于该历史提交） |
| 11-13 | [#5](https://github.com/miracle-techlink/video-to-simready/pull/5) 场景工作流/机器人接口/Isaac 部署打包；[#6](https://github.com/miracle-techlink/video-to-simready/pull/6) 一条命令场景交付；[#7](https://github.com/miracle-techlink/video-to-simready/pull/7) 可恢复 new-video 流水线；[#8](https://github.com/miracle-techlink/video-to-simready/pull/8)–[#12](https://github.com/miracle-techlink/video-to-simready/pull/12) 文档（白皮书/README/Aholo API/物性计划）；[#13](https://github.com/miracle-techlink/video-to-simready/pull/13) 多类别语义资产与可审计物体物理 | #7 为 427 项测试 + 1 跳过（该 PR 时点数量）：旧视频 + 已声明的缓存 PLY 导出/恢复 + 12 帧 GPU 巡览，非新云重建、非外观质量/尺度/任务通过；#13 真实 DINO+SAM 图像运行 + RGBD、9 项合成 PhysX 响应通过——无语义 GT 与真实物性准确性结论 |
| 11-14 | [#14](https://github.com/miracle-techlink/video-to-simready/pull/14) 模块化流水线/阶段恢复/baseline 矩阵/merge-gate/main 绑定 SHA 安装包；[#15](https://github.com/miracle-techlink/video-to-simready/pull/15) 仓库清理；[#16](https://github.com/miracle-techlink/video-to-simready/pull/16) main 核心/专用演示分离；[#17](https://github.com/miracle-techlink/video-to-simready/pull/17)–[#19](https://github.com/miracle-techlink/video-to-simready/pull/19) 工程指南/示例整合/runtime 子层 | CI 侧 merge-gate 与 main 安装包绑定 SHA 交付上线（CPU 范围）；仓库外服务器完成真实门店视频 12 关键帧全新重建→USD→Isaac 巡览（`d65696a`，111,365 高斯点，21:12 CST `checked_at`）——尺度未验证、物性未评估、任务未验证、未修复的诊断性巡览。摘要见 [docs/progress/2026-11-14-acceptance.json](progress/2026-11-14-acceptance.json)（历史验收记录，非当日 main 整链通过；其中 v6 静态修复预览完成日期未单独核实，另见今日“复核历史记录”） |
| 11-15 | 详见下节 | 详见下节 |

### 2026-11-15（今日，截至 19:30 CST）

**已合入 main**（HEAD [`99c69e0`](https://github.com/miracle-techlink/video-to-simready/commit/99c69e0b1a9c9b073254db3c600b82b88e0af22c)）：

- [#21](https://github.com/miracle-techlink/video-to-simready/pull/21) benchmark 基础设施（14:52 合入，`1aa5233`）：benchmark 输入合同、批量观测 worker 代码、candidate 独立评分、矩阵/suite 编排、CPU 合同测试、真实模型 opt-in 测试骨架。main CI（[run 43687418431](https://github.com/miracle-techlink/video-to-simready/actions/runs/43687418431)）：Python 3.10/3.12 CPU 回归、contract、package、merge-gate 全部通过——**CPU/打包范围**，不含真实模型/GPU。
- [#20](https://github.com/miracle-techlink/video-to-simready/pull/20) 外部开发者指南（15:02 合入）。
- [#22](https://github.com/miracle-techlink/video-to-simready/pull/22) CI 证据与验收指南（15:10 合入，`b25ec48`）：CI 失败证据保留（JUnit+artifact）、CI scope 报告（real_model/gpu 显式 NOT_RUN）、wheel smoke 安装加固、[手动验收指南](ACCEPTANCE_RUNBOOK.md)。main push CI（[run 43688764667](https://github.com/miracle-techlink/video-to-simready/actions/runs/43688764667)）通过；GPU 的 Actions 自动通道**仍未接入**（仓库无 secrets/runner，本 PR 不改变这一点）。
- [#24](https://github.com/miracle-techlink/video-to-simready/pull/24) 真实图像 2D 区域 CLIP 编码（17:49 合入，`1917d8c`）：增加输入、权重、实现/环境与缓存内容的证据绑定；产物保持 2D 语义，不声明三维点对齐或物理验证。
- [#26](https://github.com/miracle-techlink/video-to-simready/pull/26) COCO 小样本真实分割评测（17:54 合入，`3d27b60`）：增加冻结协议、GT 转换、预测 mask/run 证据校验与可复用 CPU 评测工具。已记录小样本指标是问题发现，不构成质量达标声明。
- [#27](https://github.com/miracle-techlink/video-to-simready/pull/27) Isaac 实测部署指南（18:09 合入，`99c69e0`）：更新本地安装和运行说明。

**已验证记录（绑定运行版本）**：

- PR #21 合入前候选 [`bcd89d8`](https://github.com/miracle-techlink/video-to-simready/commit/bcd89d88579dfd7e38a4d7bc2dcfd4e1466e15fc)（代码现已随 PR #21 进入 main）：其上的独立 CPU 验收为全量 1274 通过/19 跳过/0 失败，真实 CLI suite 编排与防篡改/取消反例通过。计数属于当时运行版本，不代表最新 main 各平台计数。
- 历史基线 `8870f88` 上的真实 DINO+SAM 在门店视频 3 帧经生产入口运行成功（30 个检测实例、工件自洽核验）——执行正确性范围，无人工参考 mask，**检测/分割准确率未验证**。
- 工厂静态三图（`d539dfc` 上的记录）：仅确认资产可见性；不代表生产 GPU 全链、冷缓存或 Docker 6.1 验收。
- Factory + Iris 飞行演示（服务器运行 `factory-full-record-20261115-191500`）：57.27 秒、每路 1718 帧，预设航点加反馈跟踪；轨迹 RMSE 0.362 m、最大误差 1.039 m。视频、双视角抽查图、轨迹 CSV、哈希与验收范围记录位于 `/root/gpufree-data/liuyue/drone-tests/factory-full-record-20261115-191500/`。这是 Isaac Sim 开放 Factory USD 场景，不是实地扫描；未训练自主导航，接触计数不能证明全场景/桨叶无碰撞，资产纹理/材质警告也已记录。

**已提交待验 / 未合入**：

- [#25](https://github.com/miracle-techlink/video-to-simready/pull/25) 显式 native backend 生产入口；尚未合入。`d539dfc` 上服务器 1355 通过/21 跳过，native 6.2 warm 生产 worker 双视角 + 独立评分已通过；GPU 入口启动/相对路径/prim 名/退出码与 auto 输入包相机证据等修复也在该版本落地。尚未完成的是新 native 配置贯通后同 SHA 的正式 auto suite + private 冷缓存复验；Docker 6.1 仍因 outer UID-map EPERM blocked。warm 通过不等于整链验收。历史基线 `8870f88` 的 `depth_observation` 双视角成功仅为独立诊断。

**开发中（无通过声明）**：三维点特征对齐、真实物性测量 GT、较大真实数据集分割评测与质量改进仍待完成。地瓜分支已有 Factory + Iris 预设航点双视角仿真演示；自主定位/规划/避障/导航训练与独立安全验收仍未完成。

**复核历史记录**：11-14 摘要 JSON 中的 v6 静态修复预览（`REPAIRED_PREVIEW_REVIEWED`）经今日复核确认记录存在；其 delivery/summary/quality-review JSON 均无完成/审核绝对时间字段，完成日期未单独核实（`date_status: not_recorded`），不计为 11-14 当日完成项。

### 下一步计划 / To-do

- [ ] **P0** 固定同一最终 SHA 打通 production 真实 GPU 多视角→candidate→独立评分→suite 全链。完成标准：进程退出码、输入/输出 hash、失败注入与恢复证据齐备并归档；必要修复合入 main。
- [x] **P0** 合入 #22（CI 证据/手动验收指南）并保持 main 检查通过。完成标准：PR 合并且 main CI 绿；GPU Actions 自动通道仍明确未接入。（已达成：15:10 合入 `b25ec48`，main push CI [run 43688764667](https://github.com/miracle-techlink/video-to-simready/actions/runs/43688764667) 通过；GPU 自动通道未接入状态未变）
- [ ] **P1** 将已上线的真实图像 2D 区域特征与三维点特征对齐。完成标准：独立核验的跨视角/几何绑定；不以 2D 缓存特征冒充 3D 对齐。
- [ ] **P1** 在更大真实数据集完成分割评测与质量改进。完成标准：冻结数据/模型/协议、逐项指标和误检漏检分析；小样本工具或单次指标不写成质量达标。
- [x] **P1** 地瓜黑客松：工业工厂场景与 Iris 四旋翼模型的预设航点跟踪演示。已完成双视角录制与轨迹归档；不代表自主导航、真实工厂扫描或完整碰撞验收。
- [ ] **P1/P2** 以带尺度/位姿/几何参考的真实小数据集（TUM/ScanNet++ 等按许可）验收重建与碰撞空间；未知区域保持未知。完成标准：独立参考下的误差报告；静态导航不以家具精确质量为先决条件。
- [ ] **P2** 机器人/无人机感知定位→规划避障→控制反馈→巡检覆盖闭环，再做导航/飞行训练与独立任务验收。
- [ ] **P2** 受控 GPU CI：明确资源/凭据/固定模型镜像与数据/失败证据；服务器自动部署与回滚为后续项，不伪称现有 CD 已包含。
- [ ] **持续** 按“模块归属”把通用能力（含地瓜等活动分支产出的通用部分）经 review+CI 回归 main；活动分支保留专用场景/航线/演示配置并从 main 同步。完成标准：通用模块无长期滞留活动分支的情况，每次回 main 都有针对性测试与（涉真实的）执行复验。

### 维护规则

- 合并影响功能、验收或计划的 PR 时，同步更新当日条目（本表由人工维护，未部署自动定时任务）。
- 一条“已验证”必须绑定实际版本（commit/CI run/产物）；不同版本的验证不得拼成一次全链验收。
- 没有记录的日子写“未找到可核验记录”，不虚构功能或结果。

### 模块归属（主线目标）

- **通用能力最终进 main**：重建/输入合同/资产打包、GPU worker 与执行路径修复、图像特征提取与缓存身份、通用评估指标/数据验收工具/可复用测试、通用渲染/录制/机器人接口、CI/CD 与工程/进度文档。feature/fix 分支只是临时开发与 PR 入口，不把通用模块规划为只交付活动分支。
- **活动分支只保留活动内容**（如地瓜黑客松分支）：变电站专用场景布局与资产引用、专用航点/任务/演示参数、活动脚本与录屏索引。活动分支从 main 同步通用实现，不做复制式各自维护，也不整体合并未验收功能。
