# 真实模型 / GPU 验收运行手册(手动受控入口)

本手册定义 CI 之外的真实权重与 GPU 验收入口。CI(portable-ci)只覆盖
CPU 合同、契约子集与 wheel 包;`real_model` / `gpu` marker 默认跳过,
**CI 绿灯不构成模型/GPU/导航验收**(见 `scripts/ci_scope_report.py` 输出)。

## 1. 状态口径(强制)

| 状态 | 含义 | 何时可用 |
|---|---|---|
| PASS | 真实生产入口在冻结输入上执行成功且断言通过 | 仅当有逐条证据 |
| BLOCKED | 所需真实权重/数据/GPU/凭据缺失 | 必须写明缺什么 |
| NOT_RUN | 本次未执行(含 opt-in marker 未启用) | 诚实记录 |
| (禁止) | 全部 skip / 零执行 / 合成替代 / fake worker 计为 PASS | 永不可 |

指定必须的 case 缺失时,报告整体不得标 PASS;不得静默放行。

## 2. 冻结验收上下文(每次运行前)

1. **git SHA**:验收 worktree 检出到目标 commit,记录
   `git rev-parse HEAD` 与 `git rev-parse HEAD^{tree}`;运行期间不得改码。
2. **模型权重 hash**:对照 `docs/PR6_ASSETS.md` 逐文件 sha256(权重在
   `~/.cache/simready_models/`)。任一不匹配 → 记录实际 hash 并 BLOCKED,
   不得临时替换权重后继续计为同一验收。
3. **数据 hash**:真实视频/帧/场景 USD 的 sha256 与抽帧索引(抽帧命令
   一并记录,如 `ffmpeg -vf select=eq(n\,300)`)。
4. **环境版本**:Python、torch/transformers、Isaac Sim 版本与启动器
   (`isaac-python.sh` 的 env 路径)、GPU 型号。容器路径需记录镜像 tag
   及其 digest,不得以宿主机原生 RC 版本冒充容器 GA tag 验收。
5. 所有以上写入运行目录 `context.json`,产物引用其 hash。

## 3. 真实模型验收(CPU,本地)

```bash
# 生产入口真实推理(非测试 fixture)
python -m simready.ingestion.segmentation.grounded <image> <out> \
    --queries table,chair --device cpu --local-files-only
# marker 化测试入口(与生产同权重、同入口)
python -m pytest -m real_model --run-real-model -q
```

- 参考标准:无人工参考 mask → IoU/精度只能 NOT_EVALUATED,仅认证
  "真实推理执行成功 + 产物自洽"。
- NeRF2Physics:真实图像→CLIP 特征的生产路径缺失时为 BLOCKED(上游
  适配器职责),缓存特征回归必须标注"缓存来源"。

## 4. 真实 GPU 验收(手动,服务器)

1. **协调窗口**:先读 `.simready-collaboration/gpu-coordination.json`,
   与现有调度员确认后使用远端
   `/root/gpufree-data/.simready/gpu-coordination.lock`;CI 开发不得
   抢占其他项目(含地瓜无人机)窗口。
2. **代码快照**:把冻结 SHA 的源码 rsync 到服务器专用目录(如
   `/root/gpufree-data/.simready/acceptance-<date>/src-<sha>`),记录
   `find -type f | sort | xargs sha256sum` 的清单 hash。
3. **worker 调用**:调用**真实 production worker**(当前为
   `simready.simulation.depth_observation`;新 benchmark worker 的
   argv/import 修复归 `fix/benchmark-gpu-worker-launch` 分支,修复
   合入并实测前,其结果按缺陷记录,不得以 depth_observation 绕行
   冒充新入口通过)。
4. **回执校验**:逐视角核对 RGB 非空(唯一色数)、depth 有效像素比、
   相机 pose 变化、输入场景/plan 的 hash 绑定、取消(SIGTERM→exit
   143,无假 receipt)与重跑恢复。回执规范沿用
   `simready-gpu-acceptance` 路的证据格式。
5. **产物归档**:回执、日志、截图 hash 后归档到验收目录并绑定 SHA。

## 5. GitHub Actions 边界(现状,不得虚报)

机器人策略接入、Isaac Sim GPU 回放与实时 Astra demo 的单独步骤见
[XPolicyLab 模型接入与 Astra Isaac Sim 验收](ASTRA_ISAAC_ACCEPTANCE.md)。

- 仓库 Actions secrets 为空,self-hosted runner 为空:**"GPU 自动 CI
  已接好"不成立**。组织/环境级资源未核实,不得引用。
- 不注册共享服务器 runner,不上传 SSH 凭据,不对 fork/外部 PR 自动
  执行 GPU 代码,不引入付费服务。
- 如未来增加 `workflow_dispatch` GPU 入口:无 runner/凭据时必须在
  job 输出与命名上显式 `NOT_RUN`,禁止留下表面绿色的空 GPU job。

## 6. 发布链路(与本手册分离)

PR → CI(static / cpu 3.10+3.12 / contract / package)→ merge-gate
(fail-closed)→ 合并 → main 重跑 CI → main-package-SHA+SHA256SUMS。
真实模型/GPU 验收状态**单列**,不并入 CI 绿灯;服务器部署与回滚、
自动 PyPI 发布本轮均未启用。
