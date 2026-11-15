"""Build a real-image 2D-region CLIP encoding request from explicit regions.

This example wires the production encoder entry
(``simready.ingestion.visual_features``) for real images. It creates an
evidence-bound request (source image + sha256, explicit integer pixel
regions, pinned local weights + sha256, explicit preprocessing, optional
material text prompts and seed), then runs the encoder offline. Region crops
are ordinary 2D image crops; they are NOT upstream 3D point-projected patch
features — no depth, camera or point-cloud alignment is claimed or possible
here, and the resulting ``R,D`` features must not be broadcast into the
``V,N,D`` layout consumed by feature_regression.

Typical use (weights downloaded once, offline afterwards):

    python examples/assets/visual_features_real.py \
      --image /data/frames/frame_800.png \
      --regions /data/regions.json \
      --weights ~/.cache/simready_models/clip/ViT-B-16-datacomp_xl/open_clip_pytorch_model.bin \
      --prompts wood metal plastic --prompt-prefix "a photo of " --prompt-suffix " material" \
      --out runs/visual-features/frame_800 --cache-dir runs/visual-features/cache

``regions.json`` is a list of ``{"region_id": ..., "bbox_xyxy": [x0, y0, x1, y1]}``
objects; each bbox is in decoded-raster pixel coordinates of the source image.
Optional per-region masks (binary 0/255, full image size) may be added under
``"mask": {"path": ...}`` together with an explicit ``"background"`` mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simready.core.artifacts import digest, read_json, write_json


def build_request(
    image: Path,
    regions_path: Path,
    weights: Path,
    *,
    prompts: list[str],
    prefix: str,
    suffix: str,
    seed: int,
    preprocess_mode: str,
    cpu_threads: int,
) -> dict:
    regions = read_json(regions_path)
    if not isinstance(regions, list) or not regions:
        raise SystemExit(f"regions file must be a nonempty list: {regions_path}")
    request = {
        "schema_version": 1,
        "image": {"path": str(image.resolve()), "sha256": digest(image)},
        "regions": regions,
        "model": {
            "backbone": "ViT-B-16",
            "pretrained": "datacomp_xl_s13b_b90k",
            "weights": {"path": str(weights.resolve()), "sha256": digest(weights)},
            "expected_feature_dim": 512,
        },
        "preprocess": {"mode": preprocess_mode, "input_size": 224},
        "seed": seed,
        "device": "cpu",
        "cpu_threads": cpu_threads,
    }
    if prompts:
        request["texts"] = {"prompts": prompts, "prefix": prefix, "suffix": suffix}
    return request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Real source image")
    parser.add_argument(
        "--regions",
        type=Path,
        required=True,
        help="JSON list of {region_id, bbox_xyxy[, mask, background]}",
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Local open_clip checkpoint"
    )
    parser.add_argument(
        "--prompts", nargs="*", default=[], help="Material text prompts to encode"
    )
    parser.add_argument("--prompt-prefix", default="", help="Upstream-style prefix")
    parser.add_argument("--prompt-suffix", default="", help="Upstream-style suffix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preprocess-mode",
        choices=("open_clip_default", "squash_224"),
        default="open_clip_default",
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--request-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    request = build_request(
        args.image,
        args.regions,
        args.weights,
        prompts=args.prompts,
        prefix=args.prompt_prefix,
        suffix=args.prompt_suffix,
        seed=args.seed,
        preprocess_mode=args.preprocess_mode,
        cpu_threads=args.cpu_threads,
    )
    request_path = args.request_out or (
        args.out.parent / f"{args.out.name}-request.json"
    )
    request_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(request_path, request)

    from simready.ingestion.visual_features import encode_regions

    receipt = encode_regions(request_path, args.out, cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "features": str(Path(receipt).parent / "region-features.npz"),
                "request": str(request_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
