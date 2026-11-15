# Isaac Sim 本地安装指南（6.x 时代）

**版本**：v1.0 · 2026-11-15
**适用**：Isaac Sim 6.1 / 6.2（GA 2026-10-14 发布）
**信息源**：云端回归机（原生 6.2.0-rc.14，L40）、GB10 本机（6.1.0 ARM64 容器）、社区踩坑记录（CUDA 11.8 + 4.5 时代的坑已按 6.x 复核标注）

---

## 0. 三种安装方式怎么选

| 方式 | 适用 | 优点 | 缺点 |
|---|---|---|---|
| **容器**（推荐） | 有 docker 的生产/回归机 | 可复现、**镜像 ID 可记录**（PR5 GPU 回归的硬要求）、不污染宿主 | 首次拉取 ~20GB |
| **standalone 压缩包** | 无 docker 的服务器（如 GPUFree 云端机） | 解压即用、root 可控 | 升级要手动、无镜像 ID 概念 |
| **pip 安装** | 快速验证、纯 CPU 逻辑 | 秒级安装 | **无渲染**，GPU 功能不可用 |

> 判断依据：需要渲染/GPU → 容器或 standalone；只跑 USD 合同/解析 → pip 即可。

---

## 1. 容器方式（x86_64 生产机）

```bash
# 前置：NVIDIA 驱动 ≥ 580，nvidia-container-toolkit 最新版
docker pull nvcr.io/nvidia/isaac-sim:6.2.0

# 无头渲染验证
docker run --gpus all -e ACCEPT_EULA=Y --rm \
  nvcr.io/nvidia/isaac-sim:6.2.0 \
  /isaac-sim/python.sh -m simready.simulation.environment_check

# 本仓库镜像（默认 6.2.0，与云端 6.2.0-rc.14 同线）
docker build -t simready-isaac deploy/isaac/

# ARM64 主机（GB10 等）必须显式降版本覆盖：
docker build --build-arg ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:6.1.0 \
  -t simready-isaac deploy/isaac/
```

**GB10 / ARM64 特别警告**（实测）：裸机跑 Isaac 会**卡死整机**，必须容器；GB10 用 `6.1.0` ARM64 镜像（6.2.0 ARM64 镜像可用性需拉取时确认）。

**本仓库 Dockerfile 的依赖面**（与云端原生环境对齐，2026-11-15 实测云端：Python 3.12.3 / numpy 2.3.1 / scipy 1.17.0 / pillow 12.3.0，无 torch）：

```
numpy>=2,<3  scipy>=1.10  Pillow>=9.1  opencv-python-headless>=4.8  usd-exchange==3.0.1
```

## 2. standalone 方式（无 docker 的服务器）

```bash
mkdir -p /root/isaacsim
# 从 NVIDIA 官网下载 isaac-sim-standalone@6.2.0...linux-x86_64.zip
unzip "isaac-sim-standalone@6.2.0*.zip" -d /root/isaacsim
cd /root/isaacsim && ./post_install.sh
alias omni_python='/root/isaacsim/python.sh'
```

GPUFree 云端机（`/root/isaacsim`）就是这个布局。注意 6.x 的 `python.sh` 裸跑**不带 kit 环境**：`import isaacsim.core`、`from pxr import Usd` 都不可用，必须走 kit app 启动（`--enable omni.kit.app`）或本仓库的 SimulationApp 引导模式。

## 3. Assets 配置（4.5 时代的坑在 6.x 仍适用）

```bash
mkdir ~/isaacsim_assets && unzip isaac-sim-assets-*.zip -d ~/isaacsim_assets
```

修改 `apps/isaacsim.exp.base.kit`：

```ini
[settings]
persistent.isaac.asset_root.default = "/home/<user>/isaacsim_assets/Assets/Isaac/6.2"
exts."isaacsim.asset.browser".folders = [
    "/home/<user>/isaacsim_assets/Assets/Isaac/6.2/Isaac/Robots",
    ...
]
```

**关键一步（不可省）**：改完必须再开一次并显式传 asset_root，否则 curobo/机器人样例报 `NoneType has no attribute GetPath`：

```bash
./isaac-sim.sh --/persistent/isaac/asset_root/default="/home/<user>/isaacsim_assets/Assets/Isaac/6.2"
```

## 4. 新栈集成（2026-11-15 L40 实测回写）

### 4.1 物理求解器：Newton（Steel 的载体）

**先纠正一个常见误传**：截至 6.2.0-rc.14，安装内**没有独立的 Steel 求解器**（grep 到的 "steel" 全是颜色常量和名为 Steel 的 Material 假阳性）。实际情况：Steel 是 Skild AI 2026-11-09 才开源进 **Newton main** 的求解器，本 rc 内置的是 **Newton 1.6.1**；本镜像无 Steel，如需使用需另行核对上游 Newton 的支持版本线（本次未验证）。

**确切启用键名（实测验证）**：

- USD 侧：物理场景 prim 上的 token 属性 **`newton:solver`**，取值映射 `{mujoco→MjcSceneAPI, xpbd→NewtonXpbdSceneAPI, vbd→NewtonVbdSceneAPI}`，配合 `NewtonSceneAPI`
- Python 侧：`isaacsim.physics.newton.switch_newton_solver("xpbd", scene_prim)`
- settings 键：`exts."isaacsim.physics.newton".auto_switch_on_startup` / `.capture_graph_physics_step`
- PhysX 侧不变：`physxScene:solverType`（TGS/PGS），**没有** steel 键

**基准（L40，1000 方块落地同场景，2026-11-15 实测）**：

| 后端 | steps/s | env-steps/s |
|---|---|---|
| PhysX（GPU/TGS） | 731.9 | 0.73M |
| **Newton-XPBD（CUDA graph）** | **5092.5** | **5.09M（≈6.96×）** |

（GB10 历史数据 1.13M env-steps/s 量级一致；批量并行任务收益最大。）

**基准限制（原证据口径，必须随数字一起引用）**：单次运行、单 seed 的吞吐 smoke；未比较 XPBD 与 TGS 的物理保真差异；不可泛化为质量或性能保证。

### 4.2 OpenUSD / usd-exchange（L40 实测）

- **裸 `python.sh` 不带 pxr**；`usd-exchange==3.0.1` wheel 自带完整 pxr 运行时（USD 0.26.8），裸跑 import 即可
- kit 内 pxr 是 0.25.11，usd-exchange 的 pxr **不会遮挡** kit 内版本
- ⚠️ **不要把 usd-exchange 装进基础 kit 的 site-packages**（会给共享解释器埋雷）；CPU pxr 工作走隔离 venv
- 本仓库 `deploy/isaac/check_usd.py` 原样跑通：`gaussian_schema_authoring=PASS`（`UsdVol.ParticleField3DGaussianSplat` 可用）；实测读取编译产物 scene.usda 成功（16 prims）

### 4.3 CuRobo（实测版本真相：没有 2.0）

**上游不存在 CuRobo 2.x**。2026-11-14 的 main（`71ce172`）= tag **v0.9.0 之后 21 个提交**（版本计数勘误见服务器证据目录 `newstack-20261115/errata.json`），包已更名 `nvidia-curobo`，`warp-lang` 成核心依赖，`[isaacsim]` extra 已删除（改用 `[cu12-torch]`）。

```bash
# 独立 venv（不要污染 Isaac python）
python3 -m venv venv-curobo && source venv-curobo/bin/activate
pip install torch==2.9.1 --index-url <镜像>    # curobo main 需 torch>=2.7（dlopen libtorch_cuda_linalg.so）
git clone https://github.com/NVlabs/curobo.git
cd curobo && git checkout 71ce1726925adfdcc7d0d39dabec5f2d6bb71acd   # 固定到实测提交，避免安装漂移
pip install -e ./curobo[cu12-torch] --no-build-isolation
# JIT 需把 venv 的 nvidia lib 目录 + torch/lib 加进 LD_LIBRARY_PATH
```

**实测冒烟（L40，sm_89）**：官方 Franka pose-to-pose 规划成功（11 waypoints，轨迹 0.23s）；冷跑 117s（warp JIT 编译）/ 热跑 60s。

**4.5 时代踩过的坑，6.x 复核状态**：

| 坑 | 状态 |
|---|---|
| `opencv-python` 装最新版拉爆 numpy → scipy/numba 冲突 | 仍会发生，装 `opencv-python-headless` 固定版本 |
| GLX 渲染（VNC 场景）`GLXBadFBConfig` | 仍需要 `MESA_GL_VERSION_OVERRIDE=4.6` + `__GLX_VENDOR_LIBRARY_NAME=nvidia` |
| nvblox 编译 `libunwind` 链接错（`undefined reference _Ux86_64_*`） | `apt remove libunwind-dev` 后重建 glog/nvblox（robby8 已验证） |
| nvblox_torch `-lglog` 找不到 | 仍需手改 CMakeLists 显式 `GLOG_DIR`（见社区帖） |
| `DeformableMaterialView` import 错（NVIDIA 自己的 bug） | 4.5 特有；6.x 未见复现，遇到再查 |
| `nvidia-srl-*` 系列包版本漂移 | 别手动 pip 装，让 isaacsim 自带版本保持 `0.10.0/0.14.0/0.19.1/0.7.2` 这一档 |
| **交互式 SSH 环境变量让 kit 启动段错误**（async_engine 阶段 `_asyncio` 初始化崩溃） | **新坑（L40 实测 4 次验证）**：用 `env -i` 干净环境包装器启动；同一毒源二号表现是非干净环境下 torch/kit 触发 `_PyErr_SetLocaleString` 符号错（系统 libpython 遮蔽） |
| pytorch.org 当次下载较慢 | 改用可达镜像源（当次实测显著更快；未记录量化速率） |

## 5. 安装后验证清单

```bash
# 1) 本仓库环境自检（容器/standalone 均可）
omni_python -m simready.simulation.environment_check

# 2) 渲染冻结检测：交付前必须做首尾帧 diff
#    （IsaacLab Camera/预制场景文件都可能静默冻结渲染；无论加载预制场景还是运行时组装，都需首尾帧/相机变化验证）

# 3) GPU 回归（本仓库 PR5 验收口径）
#    - 一次固定两视图捕获 + 像素投影对照
#    - 一次取消/超时清理（进程和容器确实退出后才释放租约）
#    - 记录：源码 commit、镜像 ID 或 standalone 版本、设备能力、日志、输出哈希
```

## 6. 版本兼容速查（2026-11 实测）

| 环境 | Isaac | Python | numpy | torch | 渲染 |
|---|---|---|---|---|---|
| GPUFree 云端（L40，standalone） | 6.2.0-rc.14（含 Newton 1.6.1） | 3.12.3 | 2.3.1 | 隔离 venv 2.9.1+cu128 | ✅（kit 引导，需干净环境包装器） |
| GB10 本机（容器） | 6.1.0 ARM64 | 3.12 | 2.x | 无 | ✅（kitonly 16.6 FPS 实测） |
| Dockerfile 默认 | 6.2.0 GA tag（**未实测**，rc.14 证据不引作 GA 验收） | 3.12 | ≥2,<3 | 无 | **未验证**（rc.14 渲染证据不顶替 GA） |

**新栈实测版本**（2026-11-15）：Newton 1.6.1 内置（本镜像无 Steel；上游支持版本线未验证）；curobo 实测固定于 `71ce172`（tag v0.9.0 之后 21 个提交；`nvidia-curobo`，无 2.x）；usd-exchange 3.0.1 自带 pxr 0.26.8。

---

**相关文档**：`deploy/isaac/Dockerfile`（本仓库镜像）· `docs/DEVELOPMENT_PLAN_20261114.md`（PR5 GPU 回归口径）
