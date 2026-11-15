# 语义、物性与参数估计示例

三种输入共用核心资产编译及独立读回；合成夹具用于接口验证，不提供真实重建或实物精度成绩。先安装 `.[physical-assets]`，分层与采用规则见[工程指南](../../docs/README.md#规范流水线)。

## 显式物性

生成柜体、桌、椅、地面和按钮的示例声明，随后编译并独立重开 USD：

```bash
python examples/assets/physical_fixture.py --output runs/physical-input
python -m simready.assets.compilation.scene_preparation \
  --ply runs/physical-input/source.ply \
  --spec runs/physical-input/scene_spec.json --output runs/physical-scene
simready evaluation physical-assets --scene-directory runs/physical-scene \
  --scene-spec runs/physical-input/scene_spec.json --out runs/physical-audit.json
```

`PASS` 表示对象材质、碰撞、刚体模式、质量／惯量及关节等与声明一致，不能用导出的 catalog 自证参数真实。

在已准备的 [Isaac 容器](../../README.md#isaac-sim-worker)中运行合成动力学探针，直接复用统一 launcher：

```bash
python scripts/launch_isaac.py --assets examples/assets --out runs/physical-response \
  --accept-eula --worker -m simready.simulation.physical_probe --out /output/probe
```

本机 SDK 使用 `--backend local --python /path/to/isaac/python`，worker 的 `--out` 改为宿主实际路径；worker argv 必须放最后。协议比较摩擦滑块和相同外力下不同质量刚体，输出 `probe/trajectories.csv` 与 `probe/probe.json`；统一 `launch.json` 的 `seconds` 保存启动到清理的耗时。每次使用新输出目录。

## 语义关联

```bash
python examples/assets/semantic_fixture.py --out runs/semantic-input
simready workflows semantic-assets --ply runs/semantic-input/source.ply \
  --scene-spec runs/semantic-input/scene_spec.json \
  --segmentation-manifest runs/semantic-input/segmentation/manifest.json \
  --calibration runs/semantic-input/calibration.json --out runs/semantic-result
```

夹具显式生成四类 mask 及相机，预期四个静态代理和独立 USD 审计通过；不是分割模型成绩。真实图像先安装 `.[physical-assets,segmentation]`，改用：

```bash
simready workflows semantic-assets --image /data/scene/image.png \
  --calibration /data/scene/calibration.json --ply /data/scene/splats.ply \
  --scene-spec /data/scene/scene_spec.json --out runs/semantic-real
```

默认 Grounding DINO + SAM、CPU 推理；匹配 CUDA 环境可用 `--device cuda:0`，独立环境用 `--segmentation-python`，已有离线权重用 `--local-files-only`。首次模型下载与权限由运行环境提供。

图像、PLY、坐标和标定必须相互对应。巡览会输出逐帧 calibration；采集修改后场景的 RGB 与 z-depth 可用：

```bash
simready simulation depth-observation --scene /data/scene/repaired.usda \
  --scene-spec /data/scene/scene_spec.json --ply /data/scene/splats.ply \
  --camera-calibration /data/scene/calibration.json --out runs/observation
```

使用返回的新图像及标定重新分割；真实视频相机须先配准，不能套用渲染相机。第三方 mask 可按[导入格式](../../src/simready/ingestion/segmentation/semantic.py)提供，并明确标记导入。

输出保留 segmentation、projection、scene、physical_audit.json 和 workflow.json。`REVIEW_REQUIRED` 保留候选并返回 2；`--priors-file` 替换类别先验时只自动采用静态接触参数，动态质量、惯量及尺度须明确声明。遮挡几何、关节和任务另行验收。

## NeRF2Physics

对缓存特征做 CPU cosine/softmax 回归，或导入官方预测；不在本次运行 NeRF、CLIP 或材料语言模型：

```bash
python examples/assets/nerf2physics_fixture.py --out runs/nerf2physics-input
python -m simready.ingestion.nerf2physics --request runs/nerf2physics-input/request.json \
  --out runs/nerf2physics-estimation
```

夹具模式为 `synthetic_fixture`；真实输入使用 `feature_regression` 或 `official_prediction_replay`。请求相对路径基于 request 文件；真实输入须绑定上游 40 位 commit、本地源码与许可证哈希、模型／权重身份、对象关联及尺度证据，不能按名字猜测对象一致。

NPZ（禁用 pickle）包含 `patch_features[V,N,D]`、布尔 `visible[V,N]`、`text_features[M,D]` 和有序 `material_names[M]`。材料区间单位为 `kg/m^3` 或无量纲 `1`；质量积分另需 `source_points[N,3]`、`query_points[Q,3]`、`voxel_volume_m3` 及体积证据。当前只实现 carving 最近邻体素积分，没有厚度估计。

独立评分读取官方 `{scene_id:[low_kg,high_kg]}` 预测和 `{product_id:measured_kg}` 真值：

```bash
python -m simready.evaluation.nerf2physics --predictions preds_mass.json \
  --ground-truth filtered_product_weights.json --scene-ids heldout-scene-ids.json \
  --out mass-evaluation.json
```

缺失／非法预测保留在覆盖率分母；ADE、ALDE、APE、MnRE 对有效预测计算，不设置物理验收阈值，材料区间不是统计置信区间。相对织物的摩擦不能赋给任意接触对，Shore 硬度不转为弹簧参数。

## 视觉特征编码（真实图像）

对真实图像的显式 2D 像素区域运行 CLIP 图像／文本 encoder（F-P1 生产入口）；先安装 `.[visual-features]` 并下载一次权重（上游 ViT-B-16 / `datacomp_xl_s13b_b90k`，LAION MIT，约 600MB），此后离线加载：

```bash
python examples/assets/visual_features_real.py --image /data/frame_800.png \
  --regions /data/regions.json --weights /path/to/open_clip_pytorch_model.bin \
  --prompts wood metal plastic --prompt-prefix "a photo of " --prompt-suffix " material" \
  --out runs/visual-features/frame_800 --cache-dir runs/visual-features/cache
# 或直接: simready ingestion encode-features --request request.json --out runs/vf [--cache-dir DIR]
```

`regions.json` 为 `[{region_id, bbox_xyxy:[x0,y0,x1,y1]}]`（解码后栅格像素坐标）；带 `mask`（0/255 全图二值）时必须显式给 `background: {mode: zero|white}`。输出 `region-features.npz`（`image_features[R,D]` 未归一化、`region_ids[R]`、可选 `text_features[M,D]` L2 归一化与 `prompts[M]`）及 `encoding-receipt.json`。

receipt 全链可追踪：源图与权重 sha256、逐区域 bbox／掩码、实际预处理管线、最终文本、seed、设备与线程、耗时与版本；`new_visual_inference_performed=true` 仅当本次真实执行 encoder，缓存命中（key 覆盖模型＋权重＋图像＋区域＋预处理＋prompt＋seed）记 `cache_hit=true` 且不冒充推理。缓存回放有三重完整性约束：cache-meta 记录 `features.npz` 的 sha256，回放前强校验，不匹配即视为 miss 重新编码并在 receipt 标 `cache_event`；回放要求写入时的实现身份（visual_features.py 代码摘要＋torch/open_clip 版本）与当前一致，升级实现/环境后旧条目自动失效重编码；缓存条目只在全部冻结输入复验通过后写入，运行中途失败不残留条目。回放 receipt 继承首次编码的时间、内容哈希、环境与预处理管线。这些是普通 image-crop 区域特征，不是上游 `get_patch_features` 的 3D 点投影 patch：没有 depth／相机／点云就没有可见性语义，`R,D` 也不得广播成 feature_regression 的 `V,N,D`；进入物性回归仍需真实几何对齐管线（多视角 patch＋可见性），当前如实标记为剩余依赖。`feature_regression` 请求可绑定 encoding receipt（`visual_inference` 块），仅当 receipt 记录真实编码且 payload 特征与其输出字节一致时 `new_visual_inference_performed` 才为 true。

依据 [NeRF2Physics 官方代码](https://github.com/ajzhai/NeRF2Physics)、[回归](https://github.com/ajzhai/NeRF2Physics/blob/master/predict_property.py)、[特征](https://github.com/ajzhai/NeRF2Physics/blob/master/feature_fusion.py)、[评分](https://github.com/ajzhai/NeRF2Physics/blob/master/evaluation.py)及 [MIT 许可](https://github.com/ajzhai/NeRF2Physics/blob/master/LICENSE)；权重来自 [LAION CLIP-ViT-B-16-DataComp.XL-s13B-b90K](https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K)（MIT）。几何对齐（V,N,D）的真实视觉推断与公开数据评分尚未完成；数据许可独立核对。
