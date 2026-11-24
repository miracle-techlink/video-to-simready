# XPolicyLab 模型接入与 Astra Isaac Sim 验收

`XPolicyLabProvider` 接收本地模型实例。每个 episode 先调用
`reset()`，每步按 `update_obs(obs)`、`get_action()` 的顺序执行。
由调用方提供观测组装器和动作解码器，将模型自身的观测与动作格式转换到
`PolicyAdapterLoop` 的单步接口。观测可包含解码后的 RGB 数组、指令和实测状态。
模型依赖由宿主提供，仓库的 CPU 测试无需安装 XPolicyLab 或模型权重。

本例采用 [XPolicyLab 单模型接口](https://github.com/XPolicyLab/XPolicyLab/blob/4a9e3afd73b86a48c5db076f5cf9dbfbef1a180e/model_template.py)
和 [Astra Direct EEF](https://github.com/XPolicyLab/XPolicyLab/tree/4a9e3afd73b86a48c5db076f5cf9dbfbef1a180e/policy/GPT_6_Astra_Direct_EEF)
的 `move_eef` 参数结构。官方 Astra 策略在环境客户端调用模型 API。
这里用现有 Messages 网关提供实时推理，并保留绝对世界坐标、米单位以及
`targets` / `note` 字段。升降台 demo 只接收 `left_z`，将其转换为
单个移动关节的位移。它覆盖模型接口、动作映射、Isaac 物理与验收链路。
官方 ARX X5 双臂、末端旋转、夹爪、碰撞规划和抓取能力仍需对应的场景与验收。

## 代码质量和 CPU 回归

GitHub Actions 对 `main` 和 `feat/robot-adapters` 的 PR 执行 Ruff、
架构边界、编译、Python 3.10/3.12 CPU 回归、契约子集和 wheel 验证。
CPU 回归使用合成动作、模拟推理端和 CPU 测试对象，覆盖模型生命周期、RGB
输入、动作解析、越界和过期动作拒绝，以及虚假成功拒绝。安装 OpenUSD 后会
额外检查生成场景中的关节、限位和驱动。

```bash
python -m pip install -e '.[dev,test,physical-assets]'
ruff check src/simready tests examples scripts deploy
ruff format --check src/simready tests examples scripts deploy
python scripts/check_architecture.py
python -m compileall -q src/simready examples scripts deploy
python -m pytest -q --junitxml=runs/cpu.xml
```

## GPU 与 demo 验收

在具备 NVIDIA RTX 驱动和 Isaac Sim 6.2 原生 launcher 的环境中，检出 PR
的目标提交。使用独立输出目录，保留模拟器日志与证据。
共享 GPU 调度和运行目录约定见 [验收运行手册](ACCEPTANCE_RUNBOOK.md)。

```bash
export SIMREADY_ISAAC_PYTHON=/path/to/isaac-sim/python.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p runs
RUN_DIR=$(mktemp -d "$PWD/runs/astra-XXXXXX")

python -m pytest tests/robotics/test_xpolicylab_astra_demo.py -m gpu --run-gpu -q \
  --basetemp="$RUN_DIR/pytest" --junitxml="$RUN_DIR/gpu.xml"

# 用本地动作文件回放。可替换为同一任务的实际 Astra move_eef 提案。
"$SIMREADY_ISAAC_PYTHON" examples/robot_adapters/isaac_astra_demo.py \
  --action examples/robot_adapters/configs/astra-lift-action.json \
  --out "$RUN_DIR/replay"

# 配置好本机 Astra Messages 网关后，使用当前渲染的 RGB 请求一次真实推理。
"$SIMREADY_ISAAC_PYTHON" examples/robot_adapters/isaac_astra_demo.py \
  --live-astra --out "$RUN_DIR/live"
```

实时模式从进程环境读取 `ASTRA_MESSAGES_BASE_URL`、`ASTRA_MESSAGES_API_KEY`、
`ASTRA_MESSAGES_MODEL` 和 `ASTRA_MESSAGES_EXPECTED_MODEL`。
预配置的本机网关 URL 须包含 `/v1/messages`。服务端若返回其他模型，
实时运行会失败。详见 [Astra provider](NOTES_ASTRA_POLICY_PROVIDER.md)。
JSON fixture 是合成动作，回放成功只证明该场景中的模拟执行与验收。

### 判定与归档

- 升降台从约 `0.10 m` 移到标记世界高度 `0.13 m`。读取独立的刚体位置，
  要求上移至少 `0.015 m`、目标高度误差和关节跟踪误差都不超过
  `0.004 m`，且物理步和反馈时间向前推进。
- `verdict.json` 记录源码提交、工作树状态、GPU 驱动、Kit 版本、场景与
  动作输入哈希、动作提案、实测状态、回执和误差。归档
  `scene.usda`、`before.png`、`after.png` 与对应的 verdict。
  验收结果应来自干净的目标提交。
- CPU 检查、GPU 回放和实时 Astra 分别记录状态。缺少运行环境或实时配置
  时返回 `BLOCKED`（退出码 2）。动作被拒绝、物理结果超差或采集出错时
  判 `FAIL`，返回非零。以 `PASS` 回执和对应证据为准。
