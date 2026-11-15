"""Real-data segmentation acceptance against independent human GT (COCO val2017).

Lane simready-data-validation, 2026-11-15. Fills the "ran the model but never
measured accuracy" gap from acceptance lane B (F-M4: no human reference masks
existed, so IoU/precision/recall could not be computed).

Subcommands
  fetch     Apply the frozen selection rule, range-extract the official COCO
            annotation member, download the 5 selected val2017 images, build
            one-way GT masks + input manifest. Refuses to overwrite frozen GT.
  run       Execute the production entry ``python -m
            simready.ingestion.segmentation.grounded`` per image (subprocess,
            CPU, pinned cached models, argv recorded verbatim).
  evaluate  Frozen greedy class-restricted matching vs GT: TP/FN/FP census,
            precision/recall/F1, matched IoU stats, overlays. Read-only on GT
            and prediction artifacts.
  report    Assemble machine-readable summary + markdown report.

The evaluation protocol is frozen in
examples/acceptance/real_segmentation/PROTOCOL.md (commit a5b80dd, dated
before any prediction existed). No quality threshold is defined; measured
scores are findings, not pass/fail. Scores are COCO-val in-distribution
measurements of the perception front-end only — never Starbucks/substation
scene quality, never product accuracy.

Pure evaluation functions (mask_iou, box_iou, greedy_match, census,
ap_sweep, GT rasterizers, selection rule) are importable without torch so
unit tests can run them on synthetic fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
ART_DEFAULT = Path(
    "/home/nvidia/algotithm/.simready-collaboration/parallel-followup-20261115/"
    "simready-data-validation/artifacts"
)
PROTOCOL_REF = "examples/acceptance/real_segmentation/PROTOCOL.md @ a5b80dd"

ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_ZIP_ETAG = "f4bbac642086de4f52a3fdda2de5fa2c"
ANN_MEMBER = "annotations/instances_val2017.json"
ANN_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
IMG_URL_FMT = "http://images.cocodataset.org/val2017/{name}"

VOCAB_COCO_NAMES = ("chair", "couch", "bed", "dining table")
CATEGORY_MAP = {"chair": "chair", "couch": "couch", "bed": "bed", "dining table": "dining_table"}
QUERY_CATEGORIES = ("bed", "chair", "couch", "dining_table")
MIN_VOCAB_INSTANCES = 3
MIN_GT_AREA_FOR_SELECTION = 1000.0
PIPELINE_MIN_AREA_PX = 256
TAU_PRIMARY = 0.50
TAU_SECONDARY = (0.30, 0.75)
BOX_TAU = 0.50
NUM_IMAGES = 5

DISCLAIMERS = [
    "GT is official human COCO val2017 annotation; model predictions never touch GT files.",
    "Scores are in-distribution (COCO domain) measurements of the perception front-end; "
    "not Starbucks/substation scene quality, not product accuracy, not extrapolable.",
    "No pre-approved quality threshold exists; numbers are findings, not pass/fail.",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# fetch: annotation member extraction, image download, GT construction
# --------------------------------------------------------------------------


class _HttpRangeFile:
    """Read-only byte-range file over the official annotation archive."""

    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self._pos = 0
        self.fetched = 0

    def seek(self, offset: int, whence: int = 0) -> None:
        if whence == 0:
            self._pos = offset
        elif whence == 2:
            self._pos = self.size - offset
        else:
            self._pos += offset

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self._pos
        if n == 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if not data:
            raise RuntimeError("short range read")
        self.fetched += len(data)
        self._pos += len(data)
        return data


def _extract_annotation_member(dest: Path) -> dict[str, Any]:
    """Fetch only instances_val2017.json from the zip via HTTP ranges."""
    if dest.exists() and _sha256(dest) == ANN_SHA256:
        return {"status": "cached", "sha256": ANN_SHA256, "bytes_fetched": 0}
    with urllib.request.urlopen(urllib.request.Request(ANN_ZIP_URL, method="HEAD"), timeout=60) as r:
        size = int(r["Content-Length"])
        etag = (r.get("ETag") or "").strip('"')
    if etag and etag != ANN_ZIP_ETAG:
        raise RuntimeError(
            f"annotation archive etag changed ({etag}); frozen protocol assumes the 2018 file"
        )
    rf = _HttpRangeFile(ANN_ZIP_URL, size)
    rf.seek(max(0, size - 65536))
    tail = rf.read()
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("zip EOCD not found")
    (_, _, _, _, _, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[eocd : eocd + 22])
    rf.seek(cd_off)
    cd = rf.read(cd_size)
    pos, member = 0, None
    while pos < len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        (_sig, _v, _n, _fl, method, _mt, _md, _crc, csize, usize, nlen, elen,
         _clen, _disk, _ia, _ea, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
        name = cd[pos + 46 : pos + 46 + nlen].decode()
        if name == ANN_MEMBER:
            member = (lho, csize, usize, method)
        pos += 46 + nlen + elen
    if member is None:
        raise RuntimeError(f"{ANN_MEMBER} not present in archive")
    lho, csize, usize, method = member
    rf.seek(lho)
    local = rf.read(512)
    nlen, elen = struct.unpack("<HH", local[26:30])
    if local[:4] != b"PK\x03\x04" or local[30 : 30 + nlen].decode() != ANN_MEMBER:
        raise RuntimeError("zip local header mismatch")
    rf.seek(lho + 30 + nlen + elen)
    raw = rf.read(csize)
    obj = zlib.decompress(raw, -15) if method == 8 else raw
    if len(obj) != usize:
        raise RuntimeError("annotation member size mismatch")
    dest.write_bytes(obj)
    got = _sha256(dest)
    if got != ANN_SHA256:
        raise RuntimeError(f"annotation sha256 mismatch: {got}")
    return {"status": "downloaded", "sha256": got, "bytes_fetched": rf.fetched}


def _select_images(coco: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Frozen deterministic rule: ascending image_id, first NUM_IMAGES that
    have >= MIN_VOCAB_INSTANCES non-crowd vocab annotations of area >= 1000."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    vocab = set(VOCAB_COCO_NAMES)
    counts: dict[int, int] = defaultdict(int)
    for ann in coco["annotations"]:
        if cats[ann["category_id"]] in vocab and not ann.get("iscrowd"):
            if float(ann["area"]) >= MIN_GT_AREA_FOR_SELECTION:
                counts[ann["image_id"]] += 1
    qualified = sorted(iid for iid, n in counts.items() if n >= MIN_VOCAB_INSTANCES)
    images = {im["id"]: im for im in coco["images"]}
    return [images[iid] for iid in qualified[:NUM_IMAGES]], len(qualified)


def _polygon_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterize COCO polygons with pixel-center sampling (matplotlib Path).

    Per-polygon containment, union across polygons — mirrors pycocotools
    frPyObjects+merge semantics closely enough that mask_px tracks the
    official ``area`` (checked by the caller's sanity bound).
    """
    from matplotlib.path import Path as MplPath

    yy, xx = np.mgrid[0:height, 0:width]
    centers = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    inside = np.zeros(height * width, dtype=bool)
    for poly in polygons:
        if len(poly) < 6:
            continue
        path = MplPath(np.array(poly, dtype=float).reshape(-1, 2))
        inside |= path.contains_points(centers)
    return inside.reshape(height, width)


def _rle_mask(counts: Any, height: int, width: int) -> np.ndarray | None:
    """Decode COCO uncompressed (list) RLE. Column-major per COCO convention.
    Compressed (bytes) RLE is refused — no pycocotools dependency is allowed."""
    if not isinstance(counts, list):
        return None
    flat = np.zeros(height * width, dtype=bool)
    idx, value = 0, False
    for run in counts:
        if value:
            flat[idx : idx + run] = True
        idx += run
        value = not value
    return flat.reshape((height, width), order="F")


def _build_gt_mask(ann: dict[str, Any], height: int, width: int) -> tuple[np.ndarray | None, str]:
    seg = ann.get("segmentation")
    if isinstance(seg, dict):
        if ann.get("iscrowd"):
            return None, "crowd_excluded"
        mask = _rle_mask(seg.get("counts"), height, width)
        return (mask, "rle_decoded") if mask is not None else (None, "conversion_failed")
    if not seg:
        return None, "conversion_failed"
    return _polygon_mask(seg, height, width), "polygon_rasterized"


def _rasterization_ok(mask: np.ndarray, area: float) -> bool:
    px = int(mask.sum())
    return 0.80 * area <= px <= 1.10 * area + 8


def cmd_fetch(args: argparse.Namespace) -> int:
    art = Path(args.artifacts).resolve()
    ds = art / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {"started_at": _now()}

    ann_path = ds / "instances_val2017.json"
    log["annotation_extraction"] = _extract_annotation_member(ann_path)
    coco = json.loads(ann_path.read_text())
    images, qualified = _select_images(coco)
    log["selection"] = {
        "rule": (
            "ascending image_id, first 5 images with >=3 non-crowd vocab "
            "annotations of area>=1000px (frozen in PROTOCOL.md)"
        ),
        "qualified_count": qualified,
        "image_ids": [im["id"] for im in images],
    }
    licenses = {l["id"]: l for l in coco["licenses"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}

    gt_dir = ds / "gt"
    img_dir = ds / "images"
    gt_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "frozen_at": _now(),
        "protocol": PROTOCOL_REF,
        "annotation_file": {
            "name": ANN_MEMBER,
            "sha256": ANN_SHA256,
            "source_url": ANN_ZIP_URL,
            "etag": ANN_ZIP_ETAG,
        },
        "selection_rule": log["selection"]["rule"],
        "images": [],
        "gt": [],
    }
    for im in images:
        name = im["file_name"]
        jpg = img_dir / name
        if jpg.exists():
            status = "cached"
        else:
            with urllib.request.urlopen(IMG_URL_FMT.format(name=name), timeout=120) as r:
                jpg.write_bytes(r.read())
            status = "downloaded"
        width, height = im["width"], im["height"]
        lic = licenses[im["license"]]
        entry = {
            "image_id": im["id"],
            "file_name": name,
            "path": str(jpg),
            "sha256": _sha256(jpg),
            "official_url": IMG_URL_FMT.format(name=name),
            "coco_url": im.get("coco_url"),
            "flickr_url": im.get("flickr_url"),
            "width": width,
            "height": height,
            "license_id": im["license"],
            "license_name": lic.get("name"),
            "license_url": lic.get("url"),
            "fetch_status": status,
        }
        anns = [
            a
            for a in coco["annotations"]
            if a["image_id"] == im["id"] and cats[a["category_id"]] in CATEGORY_MAP
        ]
        crowd = below = ok = 0
        for ann in anns:
            coco_name = cats[ann["category_id"]]
            record = {
                "ann_id": ann["id"],
                "image_id": im["id"],
                "coco_category": coco_name,
                "category": CATEGORY_MAP[coco_name],
                "coco_area_px": float(ann["area"]),
                "bbox_xywh": [float(v) for v in ann["bbox"]],
                "gt_provenance": "human_coco_val2017_official",
            }
            if ann.get("iscrowd"):
                crowd += 1
                record["status"] = "crowd_excluded"
            elif float(ann["area"]) < PIPELINE_MIN_AREA_PX:
                below += 1
                record["status"] = "below_pipeline_floor"
            else:
                mask, how = _build_gt_mask(ann, height, width)
                if mask is None:
                    record["status"] = how
                elif not _rasterization_ok(mask, float(ann["area"])):
                    raise RuntimeError(
                        f"GT rasterization sanity failed ann={ann['id']}: "
                        f"{int(mask.sum())} px vs coco area {ann['area']}"
                    )
                else:
                    rel = f"gt/{im['id']}_{ann['id']}.png"
                    out = ds / rel
                    if out.exists():
                        raise RuntimeError(f"GT frozen, refusing overwrite: {out}")
                    Image.fromarray(mask.astype(np.uint8) * 255).save(out)
                    record.update(
                        status="ok",
                        conversion=how,
                        mask_path=str(out),
                        mask_sha256=_sha256(out),
                        mask_px=int(mask.sum()),
                    )
                    ok += 1
            manifest["gt"].append(record)
        entry.update(
            gt_crowd_excluded=crowd,
            gt_below_pipeline_floor=below,
            gt_matched_count=ok,
        )
        manifest["images"].append(entry)
    manifest["gt_total_matched"] = sum(1 for r in manifest["gt"] if r["status"] == "ok")
    out_path = ds / "input_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    log.update(finished_at=_now(), input_manifest=str(out_path),
               gt_total_matched=manifest["gt_total_matched"])
    (art / "logs").mkdir(parents=True, exist_ok=True)
    (art / "logs" / "fetch.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({"manifest": str(out_path), "images": len(images),
                      "gt_matched": manifest["gt_total_matched"]}, indent=2))
    return 0


# --------------------------------------------------------------------------
# run: production entry per image (subprocess, argv verbatim)
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    art = Path(args.artifacts).resolve()
    manifest = json.loads((art / "dataset" / "input_manifest.json").read_text())
    run_root = art / "runs"
    env_cfg = {
        "OMP_NUM_THREADS": str(args.cpu_threads),
        "MKL_NUM_THREADS": str(args.cpu_threads),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
        "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    env = {**os.environ, **env_cfg}
    runs = []
    for entry in manifest["images"]:
        out_dir = run_root / str(entry["image_id"])
        if out_dir.exists():
            raise RuntimeError(f"run output exists, refusing overwrite: {out_dir}")
        argv = [
            sys.executable, "-m", "simready.ingestion.segmentation.grounded",
            "--image", entry["path"],
            "--output-dir", str(out_dir),
            "--categories", ",".join(QUERY_CATEGORIES),
            "--device", "cpu",
            "--cpu-threads", str(args.cpu_threads),
            "--local-files-only",
        ]
        t0 = time.time()
        proc = subprocess.run(argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        wall = time.time() - t0
        log_path = art / "logs" / f"run_{entry['image_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(argv)}\ncwd={REPO_ROOT}\nenv={json.dumps(env_cfg, indent=2)}\n"
            f"exit={proc.returncode} wall_s={wall:.1f}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        runs.append({
            "image_id": entry["image_id"],
            "image_sha256": entry["sha256"],
            "argv": argv,
            "exit_code": proc.returncode,
            "wall_s": round(wall, 1),
            "log": str(log_path),
        })
        if proc.returncode != 0:
            raise RuntimeError(
                f"production entry failed for image {entry['image_id']}; log: {log_path}"
            )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_index.json").write_text(json.dumps(
        {"generated_at": _now(), "env": env_cfg, "runs": runs}, indent=2))
    print(json.dumps({"runs": len(runs), "root": str(run_root)}, indent=2))
    return 0


# --------------------------------------------------------------------------
# evaluate: frozen matching + metrics + overlays (pure functions first)
# --------------------------------------------------------------------------


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / union


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    iou_fn: Callable[[dict[str, Any], dict[str, Any]], float],
    *,
    tau: float,
) -> dict[str, Any]:
    """Frozen rule: class-restricted greedy on (-IoU, gt_index, pred_index).

    Cross-class pairs never form; unmatched GT are FN, unmatched predictions
    (including category 'unknown') are FP. Empty prediction list => all FN.
    """
    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["category"] != p["category"]:
                continue
            pairs.append((iou_fn(g, p), gi, pi))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    taken_g: set[int] = set()
    taken_p: set[int] = set()
    matches = []
    for iou, gi, pi in pairs:
        if gi in taken_g or pi in taken_p:
            continue
        taken_g.add(gi)
        taken_p.add(pi)
        matches.append({"gt": gi, "pred": pi, "iou": iou, "tp": iou >= tau})
    return {
        "tau": tau,
        "matches": matches,
        "tp": [m for m in matches if m["tp"]],
        "fn": [i for i in range(len(gt)) if i not in taken_g],
        "fp": [i for i in range(len(pred)) if i not in taken_p],
    }


def census(gt: list[dict[str, Any]], pred: list[dict[str, Any]],
           result: dict[str, Any]) -> dict[str, Any]:
    """TP/FP/FN counts + PRF1 + matched IoU stats + fault census.

    F1 = 2TP/(2TP+FP+FN); precision None when no predictions exist (vacuous,
    never silently skipped); class errors tallied when the best cross-class
    IoU >= tau (diagnostic only — they are already FP+FN in the counts).
    """
    ious = [m["iou"] for m in result["tp"]]
    ntp, nfp, nfn = len(result["tp"]), len(result["fp"]), len(result["fn"])
    fn_rows = [
        {"category": gt[i]["category"], "area_px": gt[i]["coco_area_px"]}
        for i in result["fn"]
    ]
    fp_rows = [
        {
            "category": pred[i]["category"],
            "raw_label": pred[i].get("raw_label"),
            "score": pred[i].get("score"),
            "area_px": int(pred[i]["mask"].sum()) if pred[i].get("mask") is not None else None,
        }
        for i in result["fp"]
    ]
    class_errors = []
    for gi in result["fn"]:
        g = gt[gi]
        best, best_p = 0.0, None
        for pi in result["fp"]:
            p = pred[pi]
            if p["category"] == g["category"] or p.get("mask") is None:
                continue
            v = mask_iou(g["mask"], p["mask"])
            if v > best:
                best, best_p = v, pi
        if best >= result["tau"] and best_p is not None:
            class_errors.append({
                "gt_category": g["category"],
                "gt_area_px": g["coco_area_px"],
                "pred_category": pred[best_p]["category"],
                "pred_raw_label": pred[best_p].get("raw_label"),
                "iou": round(best, 4),
            })
    return {
        "tp": ntp,
        "fp": nfp,
        "fn": nfn,
        "precision": ntp / (ntp + nfp) if (ntp + nfp) else None,
        "recall": ntp / (ntp + nfn) if (ntp + nfn) else None,
        "f1": (2 * ntp / (2 * ntp + nfp + nfn)) if (2 * ntp + nfp + nfn) else None,
        "matched_iou": {
            "mean": float(np.mean(ious)) if ious else None,
            "median": float(np.median(ious)) if ious else None,
            "min": min(ious) if ious else None,
            "max": max(ious) if ious else None,
            "values": [round(v, 4) for v in ious],
        },
        "fn_gt": fn_rows,
        "fp_pred": fp_rows,
        "class_error_pairs": class_errors,
    }


def ap_sweep(gt: list[dict[str, Any]], pred: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    """Score-ranked PR sweep + all-point AP with precision envelope.

    Descriptor only — flagged small_sample_ap_unstable. Deterministic tie
    handling: stable sort on -score, pred list order preserved.
    """
    ranked = sorted(range(len(pred)), key=lambda i: (-(pred[i].get("score") or 0.0), i))
    points = []
    for k in range(1, len(ranked) + 1):
        subset = [pred[i] for i in ranked[:k]]
        res = greedy_match(gt, subset, lambda g, p: mask_iou(g["mask"], p["mask"]), tau=tau)
        ntp, nfp = len(res["tp"]), len(res["fp"])
        points.append({
            "k": k,
            "score_cutoff": subset[-1].get("score"),
            "precision": ntp / (ntp + nfp) if (ntp + nfp) else None,
            "recall": ntp / len(gt) if gt else None,
        })
    ap = None
    if points:
        rec = [p["recall"] or 0.0 for p in points]
        pre = [p["precision"] if p["precision"] is not None else 0.0 for p in points]
        env = pre[:]
        for i in range(len(pre) - 2, -1, -1):
            env[i] = max(env[i], env[i + 1])
        ap, prev_r = 0.0, 0.0
        for r, p in zip(rec, env):
            ap += (r - prev_r) * p
            prev_r = r
    return {
        "tau": tau,
        "points": points,
        "all_point_ap": ap,
        "small_sample_ap_unstable": True,
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a bool mask via 4-neighbor shifts (no scipy)."""
    m = mask
    padded = np.pad(m, 1, constant_values=False)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return m & ~interior


def _overlay(image_path: Path, gt: list[dict[str, Any]],
             pred: list[dict[str, Any]], result: dict[str, Any],
             out_png: Path) -> None:
    """GT contours green; TP blue; FP red; FN GT filled yellow. Review aid."""
    base = Image.open(image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    def paint(mask: np.ndarray, fill: tuple[int, int, int, int]) -> None:
        if int(mask.sum()) == 0:
            return
        color = Image.new("RGBA", base.size, fill)
        alpha = Image.fromarray(mask.astype(np.uint8) * 255)
        nonlocal layer
        layer = Image.alpha_composite(
            layer, Image.composite(color, Image.new("RGBA", base.size, (0, 0, 0, 0)), alpha)
        )

    for gi in result["fn"]:
        paint(gt[gi]["mask"], (255, 255, 0, 90))
    for m in result["tp"]:
        paint(pred[m["pred"]]["mask"], (0, 128, 255, 60))
    for pi in result["fp"]:
        paint(pred[pi]["mask"], (255, 0, 0, 45))
    composite = Image.alpha_composite(base, layer).convert("RGB")
    draw = ImageDraw.Draw(composite, "RGBA")

    def contour(mask: np.ndarray, color: tuple[int, int, int, int]) -> None:
        ys, xs = np.nonzero(_boundary(mask))
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((x, y), fill=color)

    for gi in result["fn"]:
        contour(gt[gi]["mask"], (200, 200, 0, 255))
    for m in result["tp"]:
        contour(pred[m["pred"]]["mask"], (0, 90, 255, 255))
    for pi in result["fp"]:
        contour(pred[pi]["mask"], (255, 0, 0, 255))
    for g in gt:
        contour(g["mask"], (0, 255, 0, 255))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_png)


def _load_gt(manifest: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in manifest["gt"]:
        if rec["status"] != "ok":
            continue
        path = Path(rec["mask_path"])
        if _sha256(path) != rec["mask_sha256"]:
            raise RuntimeError(f"GT mask hash mismatch ann={rec['ann_id']} — GT must never change")
        by_image[rec["image_id"]].append({**rec, "mask": np.asarray(Image.open(path).convert("1"))})
    return by_image


def _load_predictions(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    m = json.loads((run_dir / "manifest.json").read_text())
    derived = []
    for inst in m["instances"]:
        arr = np.asarray(Image.open(run_dir / inst["mask"]).convert("1"))
        derived.append({**inst, "mask": arr})
    raw, seen = [], set()
    for cand in [*m["instances"], *m.get("rejected_candidates", [])]:
        if cand["asset_id"] in seen:
            continue
        seen.add(cand["asset_id"])
        arr = np.asarray(Image.open(run_dir / cand["raw_mask"]["path"]).convert("1"))
        raw.append({**cand, "mask": arr})
    return derived, raw, m


def _evaluate_set(label: str, gt_map: dict[int, list[dict[str, Any]]],
                  pred_map: dict[int, list[dict[str, Any]]],
                  image_entries: list[dict[str, Any]], art: Path) -> dict[str, Any]:
    tau_set = (TAU_PRIMARY, *TAU_SECONDARY)
    per_image: dict[str, dict[str, Any]] = {}
    pooled_gt: list[dict[str, Any]] = []
    pooled_pred: list[dict[str, Any]] = []
    micro_counts = {f"tau_{t}": {"tp": 0, "fp": 0, "fn": 0} for t in tau_set}
    for entry in image_entries:
        iid = entry["image_id"]
        gt = gt_map[iid]
        pred = pred_map[iid]
        pooled_gt.extend(gt)
        pooled_pred.extend(pred)
        per_image[str(iid)] = {}
        for tau in tau_set:
            res = greedy_match(gt, pred, lambda g, p: mask_iou(g["mask"], p["mask"]), tau=tau)
            cen = census(gt, pred, res)
            per_image[str(iid)][f"tau_{tau}"] = {
                "counts": {k: cen[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                "matched_iou": cen["matched_iou"] if tau == TAU_PRIMARY else cen["matched_iou"]["values"],
                "fn_gt": cen["fn_gt"] if tau == TAU_PRIMARY else None,
                "fp_pred": cen["fp_pred"] if tau == TAU_PRIMARY else None,
                "class_error_pairs": cen["class_error_pairs"] if tau == TAU_PRIMARY else None,
            }
            micro_counts[f"tau_{tau}"]["tp"] += cen["tp"]
            micro_counts[f"tau_{tau}"]["fp"] += cen["fp"]
            micro_counts[f"tau_{tau}"]["fn"] += cen["fn"]
            if tau == TAU_PRIMARY:
                res_primary = res
        out_png = art / "overlays" / label / f"{iid}.png"
        _overlay(Path(entry["path"]), gt, pred, res_primary, out_png)
        per_image[str(iid)]["overlay"] = str(out_png)
        per_image[str(iid)]["overlay_sha256"] = _sha256(out_png)
    micro = {}
    for key, c in micro_counts.items():
        micro[key] = {
            **c,
            "precision": c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None,
            "recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None,
            "f1": (2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"])) if (2 * c["tp"] + c["fp"] + c["fn"]) else None,
        }
    per_class: dict[str, dict[str, Any]] = {}
    ap: dict[str, Any] = {}
    for cat in QUERY_CATEGORIES:
        gt_c = [g for g in pooled_gt if g["category"] == cat]
        pred_c = [p for p in pooled_pred if p["category"] == cat]
        per_class[cat] = {}
        for tau in tau_set:
            res = greedy_match(gt_c, pred_c, lambda g, p: mask_iou(g["mask"], p["mask"]), tau=tau)
            per_class[cat][f"tau_{tau}"] = census(gt_c, pred_c, res)
        ap[cat] = ap_sweep(gt_c, pred_c, TAU_PRIMARY)
    unknowns = [p for p in pooled_pred if p["category"] == "unknown"]
    return {
        "per_image": per_image,
        "per_class": per_class,
        "micro": micro,
        "unknown_category_predictions": {
            "count": len(unknowns),
            "rows": [{"raw_label": u.get("raw_label"), "score": u.get("score")} for u in unknowns],
        },
        "ap_sweep": ap,
        "ap_note": "descriptor only; 24 matched GT instances make AP unstable",
    }


def _box_level(gt_map, pred_map, image_entries) -> dict[str, Any]:
    counts = {"tp": 0, "fp": 0, "fn": 0}

    def gt_box(g):
        x, y, w, h = g["bbox_xywh"]
        return [x, y, x + w, y + h]

    def pred_box(p):
        return p["detection_bbox_xyxy"]

    for entry in image_entries:
        iid = entry["image_id"]
        gt = gt_map[iid]
        pred = pred_map[iid]
        res = greedy_match(gt, pred, lambda g, p: box_iou(gt_box(g), pred_box(p)), tau=BOX_TAU)
        counts["tp"] += len(res["tp"])
        counts["fp"] += len(res["fp"])
        counts["fn"] += len(res["fn"])
    counts["precision"] = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else None
    counts["recall"] = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else None
    counts["f1"] = (2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"])) if (2 * counts["tp"] + counts["fp"] + counts["fn"]) else None
    return {"tau_box": BOX_TAU, "micro": counts}


def cmd_evaluate(args: argparse.Namespace) -> int:
    art = Path(args.artifacts).resolve()
    manifest = json.loads((art / "dataset" / "input_manifest.json").read_text())
    run_index = json.loads((art / "runs" / "run_index.json").read_text())
    gt_map = _load_gt(manifest)
    entries = manifest["images"]
    derived_map, raw_map, prod_manifests = {}, {}, {}
    for entry in entries:
        run_dir = art / "runs" / str(entry["image_id"])
        d, r, pm = _load_predictions(run_dir)
        derived_map[entry["image_id"]] = d
        raw_map[entry["image_id"]] = r
        prod_manifests[str(entry["image_id"])] = {
            "worker_sha256": pm["provenance"]["worker_sha256"],
            "models": pm["model"]["components"],
            "parameters": pm["parameters"],
            "instance_count": len(pm["instances"]),
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _now(),
        "protocol": PROTOCOL_REF,
        "execution_classification": {
            "predictions": "real_inference_new (production DINO+SAM, CPU)",
            "gt": "gt_official_coco (human annotation, untouched)",
        },
        "gt_totals": {
            "matched": sum(1 for r in manifest["gt"] if r["status"] == "ok"),
            "crowd_excluded": sum(1 for r in manifest["gt"] if r["status"] == "crowd_excluded"),
            "below_pipeline_floor": sum(1 for r in manifest["gt"] if r["status"] == "below_pipeline_floor"),
            "conversion_failed": sum(1 for r in manifest["gt"] if r["status"] == "conversion_failed"),
        },
        "prediction_sets": {
            "derived_PRIMARY": _evaluate_set("derived", gt_map, derived_map, entries, art),
            "raw_SECONDARY": _evaluate_set("raw", gt_map, raw_map, entries, art),
        },
        "box_level_secondary": _box_level(gt_map, derived_map, entries),
        "runs": {str(r["image_id"]): {"wall_s": r["wall_s"], "argv": r["argv"], "exit_code": r["exit_code"], "log": r["log"]} for r in run_index["runs"]},
        "production_manifests": prod_manifests,
        "disclaimers": DISCLAIMERS,
        "quality_threshold": "none established; measured values are findings",
    }
    out = art / "metrics.json"
    if out.exists():
        raise RuntimeError(f"metrics exist, refusing overwrite: {out}")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"metrics": str(out)}, indent=2))
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def cmd_report(args: argparse.Namespace) -> int:
    art = Path(args.artifacts).resolve()
    metrics = json.loads((art / "metrics.json").read_text())
    m = metrics["prediction_sets"]["derived_PRIMARY"]["micro"]["tau_0.5"]
    r5 = metrics["prediction_sets"]["raw_SECONDARY"]["micro"]["tau_0.5"]
    lines = [
        "# Real-data segmentation acceptance — measured accuracy (lane simready-data-validation)",
        "",
        f"Generated {metrics['generated_at']}. Protocol {metrics['protocol']}.",
        f"GT: {metrics['gt_totals']['matched']} matched human COCO val2017 instances "
        f"({metrics['gt_totals']['below_pipeline_floor']} below pipeline floor excluded, "
        f"{metrics['gt_totals']['crowd_excluded']} crowd excluded).",
        "",
        "## Primary result (derived production masks, tau_mask=0.50)",
        "",
        f"- micro P/R/F1 = {_fmt(m['precision'])} / {_fmt(m['recall'])} / {_fmt(m['f1'])} "
        f"(TP {m['tp']} / FP {m['fp']} / FN {m['fn']})",
        "",
        "## Secondary views",
        "",
        f"- raw masks (pre overlap-resolution): P/R/F1 = {_fmt(r5['precision'])} / {_fmt(r5['recall'])} / {_fmt(r5['f1'])} (TP {r5['tp']} / FP {r5['fp']} / FN {r5['fn']})",
    ]
    for tau, key in ((0.3, "tau_0.3"), (0.75, "tau_0.75")):
        mm = metrics["prediction_sets"]["derived_PRIMARY"]["micro"][key]
        lines.append(
            f"- derived @tau {tau}: P/R/F1 = {_fmt(mm['precision'])} / {_fmt(mm['recall'])} / {_fmt(mm['f1'])} "
            f"(TP {mm['tp']} / FP {mm['fp']} / FN {mm['fn']})"
        )
    b = metrics["box_level_secondary"]["micro"]
    lines.append(
        f"- box-level @tau 0.50: P/R/F1 = {_fmt(b['precision'])} / {_fmt(b['recall'])} / {_fmt(b['f1'])} "
        f"(TP {b['tp']} / FP {b['fp']} / FN {b['fn']})"
    )
    lines += ["", "## Findings, not pass/fail", ""]
    lines += [f"- {d}" for d in metrics["disclaimers"]]
    lines += ["", f"Full machine-readable metrics: {art / 'metrics.json'}", ""]
    out = art / "REPORT.md"
    out.write_text("\n".join(lines))
    print(json.dumps({"report": str(out)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=str(ART_DEFAULT))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    p_run = sub.add_parser("run")
    p_run.add_argument("--cpu-threads", type=int, default=2)
    sub.add_parser("evaluate")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    return {"fetch": cmd_fetch, "run": cmd_run, "evaluate": cmd_evaluate, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
