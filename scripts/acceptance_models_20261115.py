#!/usr/bin/env python
"""Lane-B acceptance reproducibility script (2026-11-15, baseline 8870f88).

Reproduces the artifact-level checks recorded in
.simready-collaboration/acceptance-20261115/models/acceptance-matrix.json:

1. mask/label_map/manifest self-consistency for every grounded.py output dir;
2. exact-phrase unknown-fallback for multi-word detector labels;
3. NeRF2Physics mass-metric positive/negative cases (synthetic, formulas only);
4. independent reimplementation cross-check of regress_features.

This is acceptance tooling, not production code. It makes no accuracy claims:
without human reference masks and measured physical ground truth, only
execution success, internal consistency, and math fidelity are checkable.
Real-frame inference itself is driven by the production module CLI:
python -m simready.ingestion.segmentation.grounded --image I --output-dir O
    --categories cabinet,counter,chair,table,door,shelf --cpu-threads 4 --local-files-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from simready.evaluation.nerf2physics import evaluate_mass_predictions
from simready.ingestion.nerf2physics import regress_features
from simready.ingestion.segmentation.grounded import select_detections

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}  {detail}")
    if not condition:
        FAILURES.append(name)


def segmentation_consistency(output_dir: Path) -> None:
    manifest = json.loads((output_dir / "manifest.json").read_text())
    label_map = np.array(Image.open(output_dir / "label_map.png"))
    total = 0
    for inst in manifest["instances"]:
        mask = np.array(Image.open(output_dir / inst["mask"]["path"])) > 127
        check(
            f"{output_dir.name}:{inst['asset_id']} mask/label_map agree",
            bool((label_map[mask] == inst["raw_label_id"]).all()),
        )
        check(
            f"{output_dir.name}:{inst['asset_id']} area matches manifest",
            int(mask.sum()) == inst["area_px"],
        )
        total += int(mask.sum())
    unlabeled = getattr(Image.open(output_dir / "label_map.png"), "info", {}).get(
        "unlabeled", 65535
    )
    check(
        f"{output_dir.name} derived masks mutually exclusive",
        int((label_map != 65535).sum()) == total,
    )
    raw = len(list((output_dir / "raw_masks").glob("*.png")))
    expected = len(manifest["instances"]) + len(manifest["rejected_candidates"])
    check(
        f"{output_dir.name} raw masks preserved for every candidate",
        raw == expected,
        f"{raw}/{expected}",
    )
    _ = unlabeled


def label_fallback() -> None:
    queries = {"chair": "chair", "table": "table"}
    rows = select_detections(
        [[10, 10, 100, 100]],
        [0.9],
        ["chair table"],
        queries,
        width=720,
        height=1280,
    )
    check(
        "multi-word label falls back to unknown",
        rows and rows[0]["category"] == "unknown",
    )


def metric_cases(tmp: Path) -> None:
    # evaluate_mass_predictions refuses to overwrite; write into a fresh subdir
    # each run and keep inputs identical to the recorded evidence.
    run = tmp / "metric_repro"
    if run.exists():
        for stale in run.glob("*.json"):
            stale.unlink()
    else:
        run.mkdir(parents=True)
    (run / "gt.json").write_text(json.dumps({"obj": 2.0}))
    (run / "perfect.json").write_text(json.dumps({"obj_s0": [2.0, 2.0]}))
    good = evaluate_mass_predictions(
        run / "perfect.json", run / "gt.json", ["obj_s0"], run / "good.json"
    )
    check(
        "perfect prediction scores zero error and passes thresholds",
        good["metrics"]["ADE_kg"] == 0
        and good["metrics"]["MnRE"] == 1.0
        and good["physical_validation_passed"],
    )
    (run / "bad.json").write_text(json.dumps({"obj_s0": [4.0, 6.0]}))
    bad = evaluate_mass_predictions(
        run / "bad.json",
        run / "gt.json",
        ["obj_s0", "obj_s1"],
        run / "bad_out.json",
    )
    check(
        "missing scene keeps coverage honest and fails thresholds",
        bad["status"] == "INCOMPLETE"
        and bad["prediction_coverage"] == 0.5
        and not bad["physical_validation_passed"],
    )


def regression_math(features_npz: Path, details_json: Path) -> None:
    arrays = np.load(features_npz)
    details = json.loads(Path(details_json).read_text())
    patch = arrays["patch_features"].astype(float)
    visible = arrays["visible"]
    texts = arrays["text_features"].astype(float)
    ranges = np.array(
        [[300.0, 800.0], [500.0, 900.0], [50.0, 350.0], [2000.0, 2600.0]]
    )
    counts = visible.sum(0)
    valid = counts > 0
    agg = (patch * visible[..., None]).sum(0)[valid] / counts[valid, None]
    agg /= np.linalg.norm(agg, axis=1, keepdims=True)
    unit = texts / np.linalg.norm(texts, axis=1, keepdims=True)
    logits = agg @ unit.T / 0.1
    weights = np.exp(logits - logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    intervals = weights @ ranges
    check(
        "independent math reproduces adapter point intervals",
        np.allclose(intervals, details["point_intervals"], atol=1e-6),
    )
    check(
        "independent math reproduces adapter mean weights",
        np.allclose(weights.mean(0), details["mean_material_weights"], atol=1e-6),
    )
    _ = regress_features  # imported to pin the module under test


def main(argv: list[str]) -> int:
    evidence = Path(
        argv[1] if len(argv) > 1 else ".simready-collaboration/acceptance-20261115/models"
    )
    for frame in ("frame_200", "frame_800", "frame_1500"):
        directory = evidence / "dino_sam" / frame
        if directory.is_dir():
            segmentation_consistency(directory)
    label_fallback()
    metric_cases(evidence / "n2p")
    features = evidence / "n2p" / "real_reg" / "features.npz"
    details = evidence / "n2p" / "real_reg" / "out9" / "regression-details.json"
    if features.is_file() and details.is_file():
        regression_math(features, details)
    else:
        print("SKIP regression math (lane evidence files absent)")
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nALL CHECKS PASSED")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
