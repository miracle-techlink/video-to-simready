"""Evidence-bound real-image CLIP feature encoding (2D region contract).

This is the production entry that actually runs a CLIP image/text encoder on
real images (audit gap F-P1). It encodes explicit 2D pixel regions of one
source image; it is deliberately NOT the upstream ``get_patch_features``
3D-patch-projection path: without depth maps, camera poses and a point cloud
there is no visibility or occlusion to claim, and the resulting ``R,D``
region features are not the ``V,N,D`` patch layout consumed by
:mod:`simready.ingestion.nerf2physics` — they must never be broadcast into
one. Third-party weights stay outside the runtime package; torch/open_clip
are imported lazily inside the encoder factory only, and no network access
happens outside that factory.
"""

import argparse
import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from simready.core.artifacts import digest, read_json, write_json
from simready.core.evidence import ensure_unchanged, reference, resolve_ref
from simready.core.types import PathLike
from simready.core.validation import require

CACHE_LAYOUT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1

# Upstream NeRF2Physics pinned reference (feature_fusion.py constants):
# ViT-B-16 / datacomp_xl_s13b_b90k, 224 input, 512 output dims.
UPSTREAM_REPOSITORY = "https://github.com/ajzhai/NeRF2Physics"
UPSTREAM_COMMIT = "7311210f62b120f6896b3112b231c26003ce70e3"
DEFAULT_BACKBONE = "ViT-B-16"
DEFAULT_PRETRAINED = "datacomp_xl_s13b_b90k"
CLIP_INPUT_SIZE = 224
CLIP_OUTPUT_SIZE = 512
PREPROCESS_MODES = ("open_clip_default", "squash_224")
BACKGROUND_MODES = ("zero", "white")
HEX64 = re.compile(r"[a-f0-9]{64}")


def _atomic_write_npz(path: Path, arrays: dict) -> None:
    """Write an NPZ through a unique temp file so readers never see halves."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            import numpy as np

            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _validated_request(request, resolve):
    """Validate the request document, freezing every referenced file."""
    require(
        type(request.get("schema_version")) is int
        and request["schema_version"] == REQUEST_SCHEMA_VERSION,
        "Encoding request requires schema_version=1",
    )
    image_ref = request.get("image")
    require(
        isinstance(image_ref, dict)
        and isinstance(image_ref.get("path"), str)
        and isinstance(image_ref.get("sha256"), str)
        and HEX64.fullmatch(image_ref["sha256"]),
        "An explicit source image reference with sha256 is required",
    )
    resolve(image_ref)
    regions = request.get("regions")
    require(
        isinstance(regions, list) and regions,
        "At least one explicit pixel region is required",
    )
    seen = set()
    for region in regions:
        require(isinstance(region, dict), "Each region must be an object")
        region_id = region.get("region_id")
        require(
            isinstance(region_id, str) and region_id and region_id not in seen,
            "Region ids must be nonempty and unique",
        )
        seen.add(region_id)
        bbox = region.get("bbox_xyxy")
        require(
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(type(v) is int for v in bbox),
            f"Region {region_id}: bbox_xyxy needs four integer pixel coordinates",
        )
        if "mask" in region:
            mask_ref = region["mask"]
            require(
                isinstance(mask_ref, dict) and isinstance(mask_ref.get("path"), str),
                f"Region {region_id}: mask must be a file reference",
            )
            require(
                isinstance(region.get("background"), dict)
                and region["background"].get("mode") in BACKGROUND_MODES,
                f"Region {region_id}: masked regions need an explicit background "
                "mode (zero|white); the compositing choice must not be implicit",
            )
    model = request.get("model")
    require(
        isinstance(model, dict)
        and isinstance(model.get("backbone"), str)
        and model["backbone"]
        and isinstance(model.get("pretrained"), str)
        and model["pretrained"],
        "CLIP model identity (backbone, pretrained) is required",
    )
    weights_ref = model.get("weights")
    require(
        isinstance(weights_ref, dict)
        and isinstance(weights_ref.get("path"), str)
        and isinstance(weights_ref.get("sha256"), str)
        and HEX64.fullmatch(weights_ref["sha256"]),
        "A local weights file reference with sha256 is required (offline)",
    )
    if "expected_feature_dim" in model:
        require(
            type(model["expected_feature_dim"]) is int
            and model["expected_feature_dim"] > 0,
            "expected_feature_dim must be a positive integer",
        )
    preprocess = request.get("preprocess", {"mode": PREPROCESS_MODES[0]})
    require(
        isinstance(preprocess, dict)
        and preprocess.get("mode") in PREPROCESS_MODES
        and type(preprocess.get("input_size", CLIP_INPUT_SIZE)) is int
        and preprocess["input_size"] == CLIP_INPUT_SIZE,
        f"preprocess.mode must be one of {PREPROCESS_MODES} at input_size 224",
    )
    texts = request.get("texts")
    if texts is not None:
        require(isinstance(texts, dict), "texts must be an object")
        prompts = texts.get("prompts")
        require(
            isinstance(prompts, list)
            and prompts
            and all(isinstance(p, str) and p for p in prompts)
            and len(set(prompts)) == len(prompts),
            "texts.prompts must be nonempty unique strings",
        )
        require(
            isinstance(texts.get("prefix", ""), str)
            and isinstance(texts.get("suffix", ""), str),
            "texts.prefix/suffix must be strings",
        )
    require(
        type(request.get("seed", 0)) is int and request["seed"] >= 0,
        "seed must be a nonnegative integer",
    )
    device = request.get("device", "cpu")
    require(
        isinstance(device, str) and (device == "cpu" or device.startswith("cuda")),
        "device must be cpu or cuda[:N]",
    )
    threads = request.get("cpu_threads", 2)
    require(
        type(threads) is int and 1 <= threads <= 16,
        "cpu_threads must be an integer in [1, 16]",
    )
    return request


def _cache_core(request, weights_sha256, region_metas):
    texts = request.get("texts")
    return {
        "layout": CACHE_LAYOUT_VERSION,
        "model": {
            "backbone": request["model"]["backbone"],
            "pretrained": request["model"]["pretrained"],
            "weights_sha256": weights_sha256,
        },
        "image_sha256": request["image"]["sha256"],
        "regions": region_metas,
        "preprocess": {
            "mode": request.get("preprocess", {"mode": PREPROCESS_MODES[0]})["mode"],
            "input_size": CLIP_INPUT_SIZE,
        },
        "texts": None
        if texts is None
        else {
            "prompts": texts["prompts"],
            "prefix": texts.get("prefix", ""),
            "suffix": texts.get("suffix", ""),
        },
        "seed": request.get("seed", 0),
        "device": request.get("device", "cpu"),
        "cpu_threads": request.get("cpu_threads", 2),
    }


def feature_cache_key(core) -> str:
    """Stable identity of one encoding: model+weights+image+regions+prompts+setup."""
    import hashlib

    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _create_open_clip_encoder(model_spec, weights_path, device, cpu_threads):
    """Load the pinned open_clip model offline; torch is imported only here."""
    import time

    try:
        import open_clip
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Install simready-pipeline[visual-features] for CLIP encoding"
        ) from error
    started = time.perf_counter()
    torch.manual_seed(model_spec["seed"])
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no CUDA device is available")
        target = torch.device(device)
    else:
        torch.set_num_threads(cpu_threads)
        target = torch.device("cpu")
    backbone = model_spec["backbone"]
    # ``pretrained=<local file path>`` loads the state dict offline; the hub
    # is never contacted.
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=str(weights_path)
    )
    model = model.eval().to(target)
    tokenizer = open_clip.get_tokenizer(backbone)
    resize = next(t for t in preprocess.transforms if hasattr(t, "size"))
    normalize = next(t for t in preprocess.transforms if hasattr(t, "mean"))
    pipeline_report = {
        "resize_size": getattr(resize, "size", None),
        "resize_interpolation": str(getattr(resize, "interpolation", None).name),
        "normalize_mean": list(normalize.mean),
        "normalize_std": list(normalize.std),
    }
    transform = preprocess
    if model_spec.get("preprocess_mode") == "squash_224":
        from torchvision import transforms as vision_transforms

        transform = vision_transforms.Compose(
            [
                vision_transforms.Resize(
                    (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
                    interpolation=vision_transforms.InterpolationMode.BICUBIC,
                ),
                vision_transforms.ToTensor(),
                vision_transforms.Normalize(normalize.mean, normalize.std),
            ]
        )
        pipeline_report["deviation"] = (
            "squash_224 stretches the region to a square; this departs from the "
            "upstream preprocess_fn (shorter-side resize + center crop)"
        )

    class _Encoder:
        feature_dim = int(
            getattr(getattr(model, "visual", None), "output_dim", None)
            or CLIP_OUTPUT_SIZE
        )

        def encode_images(self, images):
            import numpy as np

            batch = torch.stack([transform(image) for image in images]).to(target)
            with torch.no_grad():
                features = model.encode_image(batch).float().cpu().numpy()
            return np.asarray(features, dtype=np.float32)

        def encode_texts(self, prompts):
            import numpy as np

            tokens = tokenizer(list(prompts)).to(target)
            with torch.no_grad():
                features = model.encode_text(tokens).float().cpu().numpy()
            features = np.asarray(features, dtype=np.float32)
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            if (norms <= 0).any():
                raise ValueError("Zero text feature norm")
            return features / norms

    return _Encoder(), {
        "load_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "open_clip_version": getattr(open_clip, "__version__", "unknown"),
        "preprocess_pipeline": pipeline_report,
    }


def _region_crops(request, decoded, resolve, region_metas):
    """Validated (region_id, crop) pairs with per-region provenance metadata."""
    import numpy as np
    from PIL import Image

    width, height = decoded.size
    crops = []
    mode = request.get("preprocess", {"mode": PREPROCESS_MODES[0]})["mode"]
    for region in request["regions"]:
        x0, y0, x1, y1 = region["bbox_xyxy"]
        require(
            0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height,
            f"Region {region['region_id']}: bbox {region['bbox_xyxy']} exceeds "
            f"decoded raster {width}x{height}",
        )
        crop = decoded.crop((x0, y0, x1, y1))
        meta = {
            "region_id": region["region_id"],
            "bbox_xyxy": [x0, y0, x1, y1],
            "width_px": x1 - x0,
            "height_px": y1 - y0,
            "mask_sha256": None,
            "background": None,
        }
        if "mask" in region:
            mask_path = resolve(region["mask"])
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L"))
            require(
                mask.shape == (height, width)
                and bool(np.isin(np.unique(mask), (0, 255)).all()),
                f"Region {region['region_id']}: mask must be binary 0/255 at "
                f"{width}x{height}",
            )
            background = region["background"]["mode"]
            pixels = np.asarray(crop.convert("RGB")).astype(np.uint8).copy()
            region_mask = mask[y0:y1, x0:x1] > 0
            pixels[~region_mask] = 255 if background == "white" else 0
            crop = Image.fromarray(pixels)
            meta.update(
                mask_sha256=digest(mask_path),
                background=background,
                masked_pixels_inside_bbox=int(region_mask.sum()),
            )
        # open_clip_default resizes the shorter side to 224 then center-crops;
        # record which fraction of the resized region survives that crop.
        if mode == "open_clip_default":
            scale = CLIP_INPUT_SIZE / min(crop.size)
            scaled = (crop.size[0] * scale, crop.size[1] * scale)
            keep = min(CLIP_INPUT_SIZE, scaled[0]) * min(CLIP_INPUT_SIZE, scaled[1])
            meta["center_crop_fraction_encoded"] = round(
                float(keep / (scaled[0] * scaled[1])), 6
            )
        else:
            meta["center_crop_fraction_encoded"] = 1.0
        crops.append((region["region_id"], crop))
        region_metas.append(meta)
    return crops


def _validate_features(features, expected_rows, feature_dim, label):
    import numpy as np

    array = np.asarray(features, dtype=np.float32)
    require(
        array.shape == (expected_rows, feature_dim),
        f"{label} must be {expected_rows}x{feature_dim}, got {array.shape}",
    )
    require(bool(np.isfinite(array).all()), f"{label} must be finite")
    require(
        bool((np.linalg.norm(array, axis=1) > 0).all()),
        f"{label} must have nonzero norms",
    )
    return array


def _validate_text_features(text_features, prompts, feature_dim):
    import numpy as np

    array = _validate_features(
        text_features, len(prompts), feature_dim, "Text features"
    )
    norms = np.linalg.norm(array, axis=1)
    require(
        bool(((norms > 0.99) & (norms < 1.01)).all()),
        "Text features must be L2-normalized (upstream get_text_features)",
    )
    return array


def _final_texts(texts):
    return [
        texts.get("prefix", "") + prompt + texts.get("suffix", "")
        for prompt in texts["prompts"]
    ]


def encode_regions(
    request_path: PathLike,
    output_dir: PathLike,
    *,
    cache_dir: PathLike | None = None,
    encoder_factory: Callable[..., tuple] = _create_open_clip_encoder,
) -> Path:
    """Encode explicit 2D regions of one real image; returns the receipt path.

    The receipt records whether the encoder actually ran this time
    (``new_visual_inference_performed``) or a cache entry was replayed; reading
    cached features never sets that flag. The injectable ``encoder_factory``
    exists for contract tests only — production always uses open_clip.
    """
    import time

    request_path = Path(request_path).resolve()
    output = Path(output_dir).resolve()
    require(not output.exists(), "Encoding output already exists")
    frozen = [reference(request_path)]

    def resolve(ref):
        path = resolve_ref(ref, request_path.parent)
        resolved = {"path": str(path), "sha256": ref["sha256"]}
        if resolved not in frozen:
            frozen.append(resolved)
        return path

    request = _validated_request(read_json(request_path), resolve)
    weights_path = resolve(request["model"]["weights"])
    image_path = resolve(request["image"])
    started = time.perf_counter()

    from PIL import Image

    decoded = Image.open(image_path).convert("RGB")
    width, height = decoded.size
    region_metas = []
    crops = _region_crops(request, decoded, resolve, region_metas)
    weights_sha256 = digest(weights_path)
    core = _cache_core(request, weights_sha256, region_metas)
    cache_key = feature_cache_key(core)
    texts = request.get("texts")
    prompts = _final_texts(texts) if texts is not None else None

    cache_root = Path(cache_dir).resolve() if cache_dir else None
    cache_entry = cache_root / cache_key if cache_root else None
    encoder_executed = True
    pipeline_report = None
    load_report = None
    cache_meta = None
    if cache_entry is not None and (cache_entry / "features.npz").is_file():
        import numpy as np

        cache_npz = cache_entry / "features.npz"
        cache_meta = read_json(cache_entry / "cache-meta.json")
        require(cache_meta.get("key") == cache_key, "Cache metadata key mismatch")
        cached_dim = int(cache_meta["feature_dim"])
        expected_dim = request["model"].get("expected_feature_dim")
        require(
            expected_dim is None or expected_dim == cached_dim,
            f"Cache feature dim {cached_dim} differs from expected {expected_dim}",
        )
        with np.load(cache_npz, allow_pickle=False) as cached:
            require(
                cached["region_ids"].tolist() == [r["region_id"] for r in region_metas],
                "Cache entry region ids differ from the request",
            )
            features = _validate_features(
                cached["image_features"],
                len(crops),
                cached_dim,
                "Cached image features",
            )
            text_features = None
            if texts is not None:
                require(
                    prompts is not None and cached["prompts"].tolist() == prompts,
                    "Cache entry prompts differ from the request",
                )
                text_features = _validate_text_features(
                    cached["text_features"], prompts, cached_dim
                )
        arrays = {
            "region_ids": np.array([r["region_id"] for r in region_metas]),
            "image_features": features,
        }
        if text_features is not None:
            arrays["text_features"] = text_features
            arrays["prompts"] = np.array(prompts)
        encoder_executed = False

    if encoder_executed:
        import numpy as np

        model_spec = {
            "backbone": request["model"]["backbone"],
            "pretrained": request["model"]["pretrained"],
            "seed": request.get("seed", 0),
            "preprocess_mode": request.get("preprocess", {"mode": PREPROCESS_MODES[0]})[
                "mode"
            ],
        }
        encoder, load_report = encoder_factory(
            model_spec,
            weights_path,
            request.get("device", "cpu"),
            request.get("cpu_threads", 2),
        )
        feature_dim = (
            request["model"].get("expected_feature_dim") or encoder.feature_dim
        )
        require(
            type(feature_dim) is int and feature_dim > 0,
            "Encoder did not report a usable feature dimension",
        )
        features = _validate_features(
            encoder.encode_images([crop for _, crop in crops]),
            len(crops),
            feature_dim,
            "Image features",
        )
        arrays = {
            "region_ids": np.array([r["region_id"] for r in region_metas]),
            "image_features": features,
        }
        if texts is not None:
            text_features = _validate_text_features(
                encoder.encode_texts(prompts), prompts, feature_dim
            )
            arrays["text_features"] = text_features
            arrays["prompts"] = np.array(prompts)
        pipeline_report = load_report["preprocess_pipeline"]
        if cache_entry is not None:
            cache_entry.mkdir(parents=True, exist_ok=False)
            _atomic_write_npz(cache_entry / "features.npz", arrays)
            write_json(
                cache_entry / "cache-meta.json",
                {
                    "key": cache_key,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "layout": CACHE_LAYOUT_VERSION,
                    "backbone": request["model"]["backbone"],
                    "pretrained": request["model"]["pretrained"],
                    "weights_sha256": weights_sha256,
                    "feature_dim": int(features.shape[1]),
                },
            )

    require(
        not any(Path(ref["path"]).is_relative_to(output) for ref in frozen),
        "Output cannot contain encoding inputs",
    )
    ensure_unchanged(frozen)
    output.mkdir(parents=True)
    features_path = output / "region-features.npz"
    _atomic_write_npz(features_path, arrays)
    backbone_match = (
        request["model"]["backbone"] == DEFAULT_BACKBONE
        and request["model"]["pretrained"] == DEFAULT_PRETRAINED
    )
    runtime = {
        "cache_hit": not encoder_executed and cache_meta is not None,
        "total_seconds": round(time.perf_counter() - started, 4),
    }
    if encoder_executed:
        runtime.update(
            encoder_load_seconds=round(load_report["load_seconds"], 4),
            torch_version=load_report["torch_version"],
            open_clip_version=load_report["open_clip_version"],
        )
    else:
        runtime.update(
            cached_from=str(cache_entry / "features.npz"),
            first_encoded_at=cache_meta.get("created_at"),
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_type": "visual_feature_receipt",
        "inference_kind": "2d_region_clip_encoding",
        "source_type": "model_inference"
        if encoder_executed
        else "cached_feature_replay",
        "new_visual_inference_performed": encoder_executed,
        "encoder_executed_this_run": encoder_executed,
        "cache_key": cache_key,
        "cache_hit": runtime["cache_hit"],
        "observed_at": datetime.now().astimezone().isoformat(),
        "image": {
            "path": str(image_path),
            "sha256": request["image"]["sha256"],
            "decoded_width": width,
            "decoded_height": height,
            "pixel_coordinates": "encoded_raster_no_exif_rotation",
        },
        "regions": region_metas,
        "model": {
            "backbone": request["model"]["backbone"],
            "pretrained": request["model"]["pretrained"],
            "weights": {"path": str(weights_path), "sha256": weights_sha256},
            "weights_bytes": weights_path.stat().st_size,
            "expected_feature_dim": request["model"].get("expected_feature_dim"),
            "feature_dim": int(features.shape[1]),
            "upstream_backbone_match": backbone_match,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
        },
        "preprocess": {
            **request.get("preprocess", {"mode": PREPROCESS_MODES[0]}),
            "input_size": CLIP_INPUT_SIZE,
            **({"pipeline": pipeline_report} if pipeline_report else {}),
        },
        "texts": None
        if texts is None
        else {
            "prompts": texts["prompts"],
            "prefix": texts.get("prefix", ""),
            "suffix": texts.get("suffix", ""),
            "final_texts": prompts,
        },
        "seed": request.get("seed", 0),
        "device": request.get("device", "cpu"),
        "cpu_threads": request.get("cpu_threads", 2),
        "runtime": runtime,
        "geometry_alignment": {
            "aligned_to_3d_points": False,
            "depth_used": False,
            "cameras_used": False,
            "pointcloud_used": False,
            "note": (
                "2D pixel regions only; upstream get_patch_features visibility "
                "semantics are not claimed and cannot be reconstructed without "
                "depth, camera poses and a 3D point cloud."
            ),
        },
        "outputs": {"features_npz": reference(features_path)},
        "evidence": frozen,
        "limitations": [
            "2D region encoding: no 3D visibility, occlusion or depth semantics.",
            "Region features are R,D; they are not a V,N,D patch-feature layout "
            "and must not be broadcast or tiled into one for feature_regression.",
            "Model outputs are not ground truth; no physical-property or "
            "material-accuracy claim is made.",
        ],
    }
    receipt_path = output / "encoding-receipt.json"
    write_json(receipt_path, receipt)
    return receipt_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional feature cache; without it every run encodes fresh",
    )
    args = parser.parse_args(argv)
    receipt_path = encode_regions(args.request, args.out, cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "features": str(Path(receipt_path).parent / "region-features.npz"),
                "receipt_sha256": digest(receipt_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
