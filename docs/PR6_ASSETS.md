# PR 6 外部资源清单

生成日期：2026-11-15。所有下载均通过 `https://huggingface.co` 直连完成（系统已有代理，`hf-mirror` 反而不通）。

## 权重

| 模型 | 路径 | 大小 | 来源 commit | 状态 |
|---|---|---|---|---|
| grounding-dino-tiny | `~/.cache/simready_models/grounding-dino-tiny/` | 658 MB | `a2bb814dd30d776dcf7e30523b00659f4f141c71`（与 `grounded.py` 的 `DETECTOR_REVISION` 一致） | 已下载并校验 |
| sam-vit-base | `~/.cache/simready_models/sam-vit-base/` | 358 MB | `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`（与 `grounded.py` 的 `MASK_MODEL_REVISION` 一致） | 已下载并校验 |

### grounding-dino-tiny 文件 sha256

| 文件 | sha256 |
|---|---|
| config.json | `eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a` |
| model.safetensors | `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` |
| preprocessor_config.json | `8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60` |
| processor_config.json | `f36668ddf22403a332f978057d527cf285b01468bc3431b04094a7bafa6aba59` |
| special_tokens_map.json | `b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3` |
| tokenizer_config.json | `d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56` |
| vocab.txt | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

config 校验：`architectures = ["GroundingDinoForObjectDetection"]`，`model_type = "grounding-dino"`；safetensors 魔数头有效。

### sam-vit-base 文件 sha256

| 文件 | sha256 |
|---|---|
| config.json | `5ebd0d8643b486f3a716bf17c2a15531eb818b2b96ff0c6c5dcc88fa015161af` |
| model.safetensors | `892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579` |
| preprocessor_config.json | `225545a743c654e3c495ec6f545a0eaba57c8ba3fbbd8483b3cb1c0fc58db517` |
| processor_config.json | `f36668ddf22403a332f978057d527cf285b01468bc3431b04094a7bafa6aba59` |

config 校验：`architectures = ["SamModel"]`，`model_type = "sam"`。

## 上游仓库

- NeRF2Physics: <https://github.com/ajzhai/NeRF2Physics>
  - pinned commit: `7311210f62b120f6896b3112b231c26003ce70e3`（main @ 2025-11-16 "friction and hardness dataset link"，克隆时 main 最新）
  - LICENSE: MIT（Copyright (c) 2024 ajzhai），sha256 `32dcb4ed82e531e3e88d07f5bf10aa98f5f977dc1e0e9e2726e306179e206e73`
  - 探针克隆位置：`/tmp/nerf2physics_probe`（临时；数据集未下载）
- 关键源文件清单（供 `nerf2physics.py` 的 `upstream.source_files` 引用，均为上述 commit 下的 blob sha256）：

| 文件 | sha256 | 作用 |
|---|---|---|
| feature_fusion.py | `594f05613386e8cf08e726397f279aeb7d02e319ed77fe85f160d7472ebd79f8` | CLIP patch 特征融合（可见性聚合的出处） |
| predict_property.py | `e8ede6aaddf71cb56e8330c9190303b28b2508d1724eb42f33ba63d67d81baeb` | 属性回归 / 官方预测 |
| material_proposal.py | `c934ced01db20bf24c8a1c09f10123a977d4820fa47f1332cf3d04d42ac7716c` | 材质候选与区间 |
| captioning.py | `ae2c3ce8b602df83743cb72a9b501ca8db422c85ae0825f274960c2c72c319e0` |  caption LLM 提示 |
| utils.py | `bae0d235051898f6642ac8f91d5ab841e8ee11c4732777fed1953102f492ec62` | 公共工具 |
| LICENSE | `32dcb4ed82e531e3e88d07f5bf10aa98f5f977dc1e0e9e2726e306179e206e73` | MIT |

## CLIP 特征缓存

- 状态：已生成（样例）
- 路径：`~/.cache/simready_models/clip_cache/`
  - `sample_image_features.npy` (1×512 float32, L2-normed) sha256 `9865e804d702c65b23f9b8fdef88a2b4bd518d89a0d901b03e7fe878347b1e0f`
  - `sample_text_features.npy` (4×512 float32, L2-normed) sha256 `df2199900aaea177066fba389ecbc3ac9bd35892299d2e7d94b9e1aee5617408`
  - `clip_cache_meta.json` sha256 `25ad14bc4947c438f251ac5bee25b75e491b8a18eaeeb95321135a1ad93a5e17`
- 模型：open_clip `ViT-B-32` / `laion2b_s34b_b79k`（open_clip_torch 4.2.2，`pip install --user` 安装，未改 pyproject）
- 内容：一张纯色测试图 + 4 条材质短语（cabinet/door/chair/floor）的特征向量，供冒烟测试用；真实材质特征库留待 PR 6 本体生成

## 下载命令（可复现）

```bash
# grounding-dino-tiny（钉死 revision = DETECTOR_REVISION）
mkdir -p ~/.cache/simready_models/grounding-dino-tiny && cd ~/.cache/simready_models/grounding-dino-tiny
for f in config.json model.safetensors processor_config.json tokenizer_config.json \
         vocab.txt special_tokens_map.json preprocessor_config.json; do
  curl -sL --retry 3 -o "$f" \
    "https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/a2bb814dd30d776dcf7e30523b00659f4f141c71/$f"
done

# sam-vit-base（钉死 revision = MASK_MODEL_REVISION）
mkdir -p ~/.cache/simready_models/sam-vit-base && cd ~/.cache/simready_models/sam-vit-base
for f in config.json model.safetensors processor_config.json preprocessor_config.json; do
  curl -sL --retry 3 -o "$f" \
    "https://huggingface.co/facebook/sam-vit-base/resolve/70c1a07f894ebb5b307fd9eaaee97b9dfc16068f/$f"
done

# NeRF2Physics（钉 commit，不含数据集）
git clone https://github.com/ajzhai/NeRF2Physics /tmp/nerf2physics_probe
git -C /tmp/nerf2physics_probe checkout 7311210f62b120f6896b3112b231c26003ce70e3

# CLIP 样例缓存（需 torch + open_clip_torch）
pip install --user open_clip_torch   # 需走官方 PyPI：-i https://pypi.org/simple
python3 <生成脚本见 clip_cache_meta.json 对应流程>
```

## 解锁条件

PR 6 测试激活需要：

- [x] Grounding DINO 权重齐全（658 MB，sha256 已记录）
- [x] SAM 权重齐全（358 MB，sha256 已记录）
- [x] NeRF2Physics commit 已钉（`7311210f...`，MIT LICENSE sha256 已记录）
- [x] CLIP 特征缓存样例就绪（真实库生成脚本属 PR 6 本体）
- [ ] `transformers` 未安装 —— grounded.py 运行时需要（当前环境只有 torch 2.14.0+cpu）
- [ ] GPU 或 CPU 推理路径确认 —— 当前环境 torch 为 CPU 版，Grounding DINO + SAM CPU 推理可行但慢，测试需设超时预期
