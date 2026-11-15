"""Real-data segmentation acceptance evaluation — pure-function unit tests.

Covers the frozen matching/metric protocol from
examples/acceptance/real_segmentation/PROTOCOL.md against synthetic fixtures
only: perfect match, empty predictions, misaligned masks, class errors,
unknown-category predictions, greedy tie-break determinism, COCO GT
rasterization (polygon + RLE), the deterministic dataset selection rule,
and the PR sweep. No network, no model weights, no real images.

The real-data run (production DINO+SAM vs human COCO GT) is executed by
scripts/acceptance_real_segmentation.py outside pytest and recorded in the
lane artifacts; these tests verify the evaluation code itself behaves
correctly on cases with known answers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "acceptance_real_segmentation.py"

_spec = importlib.util.spec_from_file_location("acceptance_real_segmentation", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("acceptance_real_segmentation", mod)
_spec.loader.exec_module(mod)


def box_mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def gt_row(cat, mask, area=None):
    return {
        "category": cat,
        "mask": mask,
        "coco_area_px": float(area if area is not None else int(mask.sum())),
        "bbox_xywh": [0.0, 0.0, 1.0, 1.0],
    }


def pred_row(cat, mask, score=0.5, raw_label=None):
    return {
        "category": cat,
        "mask": mask,
        "score": score,
        "raw_label": raw_label or cat,
        "detection_bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
    }


MASK_IOU = lambda g, p: mod.mask_iou(g["mask"], p["mask"])  # noqa: E731


class TestCannedCases:
    """Protocol-mandated demonstration cases with known answers."""

    def test_perfect_match_full_credit(self):
        gt = [gt_row("chair", box_mask(20, 20, 2, 2, 10, 10))]
        pred = [pred_row("chair", box_mask(20, 20, 2, 2, 10, 10))]
        res = mod.greedy_match(gt, pred, MASK_IOU, tau=0.5)
        cen = mod.census(gt, pred, res)
        assert (cen["tp"], cen["fp"], cen["fn"]) == (1, 0, 0)
        assert cen["precision"] == cen["recall"] == cen["f1"] == 1.0
        assert cen["matched_iou"]["values"] == [1.0]

    def test_empty_predictions_all_missed_not_skipped(self):
        gt = [
            gt_row("chair", box_mask(20, 20, 1, 1, 8, 8)),
            gt_row("bed", box_mask(20, 20, 10, 10, 18, 18)),
        ]
        res = mod.greedy_match(gt, [], MASK_IOU, tau=0.5)
        cen = mod.census(gt, [], res)
        assert (cen["tp"], cen["fp"], cen["fn"]) == (0, 0, 2)
        assert cen["recall"] == 0.0
        assert cen["precision"] is None  # vacuous, never a fake 1.0
        assert cen["f1"] == 0.0
        assert {row["category"] for row in cen["fn_gt"]} == {"chair", "bed"}

    def test_misaligned_below_tau_is_fn_plus_fp(self):
        gt = [gt_row("chair", box_mask(20, 20, 2, 2, 10, 10))]  # 64 px
        off = [gt_row("chair", box_mask(20, 20, 6, 6, 14, 14))]  # 64 px, 16 px overlap
        pred = [pred_row("chair", off[0]["mask"])]
        iou = mod.mask_iou(gt[0]["mask"], pred[0]["mask"])
        assert abs(iou - 16 / 112) < 1e-9
        res = mod.greedy_match(gt, pred, MASK_IOU, tau=0.5)
        cen = mod.census(gt, pred, res)
        assert (cen["tp"], cen["fp"], cen["fn"]) == (0, 1, 1)
        # same geometry matches at the permissive secondary tau
        res30 = mod.greedy_match(gt, pred, MASK_IOU, tau=0.1)
        assert len(res30["tp"]) == 1

    def test_wrong_class_never_matches_and_is_tallied(self):
        m = box_mask(20, 20, 2, 2, 10, 10)
        gt = [gt_row("chair", m)]
        pred = [pred_row("couch", m, score=0.9)]
        res = mod.greedy_match(gt, pred, MASK_IOU, tau=0.5)
        cen = mod.census(gt, pred, res)
        assert (cen["tp"], cen["fp"], cen["fn"]) == (0, 1, 1)
        assert cen["class_error_pairs"] == [
            {
                "gt_category": "chair",
                "gt_area_px": 64.0,
                "pred_category": "couch",
                "pred_raw_label": "couch",
                "iou": 1.0,
            }
        ]

    def test_unknown_category_prediction_is_pure_fp(self):
        gt = [gt_row("chair", box_mask(20, 20, 2, 2, 10, 10))]
        pred = [
            pred_row(
                "unknown", box_mask(20, 20, 2, 2, 10, 10), raw_label="counter table"
            )
        ]
        res = mod.greedy_match(gt, pred, MASK_IOU, tau=0.5)
        cen = mod.census(gt, pred, res)
        assert (cen["tp"], cen["fp"], cen["fn"]) == (0, 1, 1)
        assert cen["fp_pred"][0]["raw_label"] == "counter table"

    def test_one_to_one_greedy_no_double_counting(self):
        g1 = gt_row("chair", box_mask(30, 30, 0, 0, 10, 10))
        g2 = gt_row("chair", box_mask(30, 30, 12, 0, 22, 10))
        p = pred_row("chair", box_mask(30, 30, 4, 0, 14, 10))  # overlaps both
        gt = [g1, g2]
        res = mod.greedy_match(gt, [p], MASK_IOU, tau=0.3)
        assert len(res["tp"]) == 1 and len(res["fn"]) == 1 and len(res["fp"]) == 0

    def test_greedy_tie_break_deterministic(self):
        m = box_mask(20, 20, 5, 5, 15, 15)
        gt = [gt_row("chair", m)]
        p1 = pred_row("chair", box_mask(20, 20, 0, 5, 10, 15), score=0.4)
        p2 = pred_row("chair", box_mask(20, 20, 10, 5, 20, 15), score=0.4)
        for _ in range(3):
            res = mod.greedy_match(gt, [p1, p2], MASK_IOU, tau=0.1)
            assert res["tp"][0]["pred"] == 0  # equal IoU+score: lower pred index


class TestGeometry:
    def test_mask_iou_bounds_and_empty(self):
        a = box_mask(10, 10, 0, 0, 5, 5)
        assert mod.mask_iou(a, a) == 1.0
        assert mod.mask_iou(a, np.zeros_like(a)) == 0.0
        z = np.zeros((4, 4), dtype=bool)
        assert mod.mask_iou(z, z) == 0.0  # empty union defined 0

    def test_box_iou_known_values(self):
        a = [0, 0, 10, 10]
        assert mod.box_iou(a, a) == 1.0
        assert abs(mod.box_iou(a, [5, 0, 15, 10]) - 50 / 150) < 1e-12
        assert mod.box_iou(a, [20, 20, 30, 30]) == 0.0


class TestGroundTruthConversion:
    # The rasterizer is matplotlib-based (lazy import, matplotlib.path) and
    # is pinned in the pyproject test extra so these conversion tests run in
    # CI. The implementation is deliberately the exact one that produced the
    # frozen GT masks — swapping it after scores existed would change GT
    # bytes, which the protocol forbids.

    def test_polygon_rasterization_matches_area(self):
        # axis-aligned rectangle polygon; pixel-center sampling
        m = mod._polygon_mask([[2.0, 2.0, 10.0, 2.0, 10.0, 8.0, 2.0, 8.0]], 12, 12)
        assert m.sum() == 8 * 6
        assert m[2:8, 2:10].all() and not m[0:2, :].any()

    def test_polygon_union_across_parts(self):
        p1 = [1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0]
        p2 = [7.0, 1.0, 9.0, 1.0, 9.0, 5.0, 7.0, 5.0]
        m = mod._polygon_mask([p1, p2], 8, 12)
        assert m.sum() == 16 + 8
        assert m[1:5, 1:5].all() and m[1:5, 7:9].all()

    def test_uncompressed_rle_column_major(self):
        # h=3,w=4, counts start with a background run: [3,3,3,3] =>
        # col0 off, col1 on, col2 off, col3 on (column-major traversal)
        counts = [3, 3, 3, 3]
        m = mod._rle_mask(counts, 3, 4)
        assert m.dtype == np.bool_
        assert m[:, 1].all() and m[:, 3].all()
        assert not m[:, 0].any() and not m[:, 2].any()

    def test_compressed_rle_refused_honestly(self):
        assert mod._rle_mask(b"\x00\x01", 4, 4) is None
        ann = {"segmentation": {"counts": b"\x00"}, "iscrowd": 0}
        mask, how = mod._build_gt_mask(ann, 4, 4)
        assert mask is None and how == "conversion_failed"

    def test_crowd_rle_excluded(self):
        ann = {"segmentation": {"counts": [4, 4]}, "iscrowd": 1}
        mask, how = mod._build_gt_mask(ann, 4, 4)
        assert mask is None and how == "crowd_excluded"

    def test_rasterization_sanity_bounds(self):
        m = box_mask(10, 10, 0, 0, 5, 5)  # 25 px
        assert mod._rasterization_ok(m, 25.0)
        assert mod._rasterization_ok(m, 26.0)
        assert not mod._rasterization_ok(m, 40.0)  # raster >> declared area
        assert not mod._rasterization_ok(m, 15.0)  # raster << declared area


def _mini_coco():
    def img(iid):
        return {
            "id": iid,
            "file_name": f"{iid:012d}.jpg",
            "width": 64,
            "height": 48,
            "license": 4,
            "coco_url": "",
            "flickr_url": "",
        }

    cats = [
        {"id": 1, "name": "chair"},
        {"id": 2, "name": "couch"},
        {"id": 3, "name": "person"},
    ]
    anns = []
    for iid in (11, 22, 33, 44, 55, 66, 77):  # 7 qualifying candidates
        for _ in range(3):
            anns.append(
                {
                    "id": len(anns),
                    "image_id": iid,
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": 2000.0,
                    "bbox": [0, 0, 10, 10],
                    "segmentation": [[1, 1, 30, 1, 30, 20, 1, 20]],
                }
            )
    # decoys: too few instances / too small / crowd / out of vocab
    anns.append(
        {
            "id": 99,
            "image_id": 5,
            "category_id": 1,
            "iscrowd": 0,
            "area": 2000.0,
            "bbox": [0, 0, 10, 10],
            "segmentation": [[1, 1, 30, 1, 30, 20, 1, 20]],
        }
    )
    anns.append(
        {
            "id": 100,
            "image_id": 66,
            "category_id": 2,
            "iscrowd": 1,
            "area": 9000.0,
            "bbox": [0, 0, 10, 10],
            "segmentation": {"counts": [4, 4]},
        }
    )
    anns.append(
        {
            "id": 101,
            "image_id": 22,
            "category_id": 3,
            "iscrowd": 0,
            "area": 5000.0,
            "bbox": [0, 0, 10, 10],
            "segmentation": [[1, 1, 30, 1, 30, 20, 1, 20]],
        }
    )
    return {
        "images": [img(i) for i in (5, 11, 22, 33, 44, 55, 66, 77)],
        "categories": cats,
        "annotations": anns,
        "licenses": [{"id": 4, "name": "BY", "url": "u"}],
    }


class TestSelectionRule:
    def test_first_five_ascending_no_cherry_picking(self):
        images, qualified = mod._select_images(_mini_coco())
        assert qualified == 7
        assert [im["id"] for im in images] == [11, 22, 33, 44, 55]

    def test_out_of_vocab_and_small_and_crowd_ignored(self):
        images, _ = mod._select_images(_mini_coco())
        assert 5 not in [im["id"] for im in images]  # 1 instance only


class TestPRSweep:
    def test_perfect_predictions_ap_one(self):
        gt = [
            gt_row("chair", box_mask(20, 20, 0, 0, 8, 8)),
            gt_row("chair", box_mask(20, 20, 10, 10, 18, 18)),
        ]
        pred = [
            pred_row("chair", box_mask(20, 20, 0, 0, 8, 8), score=0.9),
            pred_row("chair", box_mask(20, 20, 10, 10, 18, 18), score=0.8),
        ]
        out = mod.ap_sweep(gt, pred, 0.5)
        assert out["all_point_ap"] == 1.0
        assert out["points"][-1]["recall"] == 1.0

    def test_ap_zero_when_nothing_matches(self):
        gt = [gt_row("chair", box_mask(20, 20, 0, 0, 8, 8))]
        pred = [pred_row("chair", box_mask(20, 20, 12, 12, 19, 19), score=0.9)]
        out = mod.ap_sweep(gt, pred, 0.5)
        assert out["all_point_ap"] == 0.0

    def test_empty_predictions_sweep_empty_not_crash(self):
        out = mod.ap_sweep([gt_row("chair", box_mask(8, 8, 0, 0, 4, 4))], [], 0.5)
        assert out["points"] == [] and out["all_point_ap"] is None


class TestOverlaySmoke:
    def test_overlay_written_and_classified(self, tmp_path):
        gt = [
            gt_row("chair", box_mask(24, 24, 2, 2, 10, 10)),
            gt_row("bed", box_mask(24, 24, 14, 14, 20, 20)),
        ]
        pred = [pred_row("chair", box_mask(24, 24, 2, 2, 10, 10))]
        src = tmp_path / "src.png"
        Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(src)
        res = mod.greedy_match(gt, pred, MASK_IOU, tau=0.5)
        out = tmp_path / "ov.png"
        mod._overlay(src, gt, pred, res, out)
        assert out.exists() and out.stat().st_size > 0


class TestF1Formula:
    def test_census_counts_consistent(self):
        gt = [gt_row("chair", box_mask(20, 20, 0, 0, 10, 10)) for _ in range(3)]
        pred = [
            pred_row("chair", box_mask(20, 20, 0, 0, 10, 10)),
            pred_row("chair", box_mask(20, 20, 0, 0, 4, 4)),
        ]  # partial dup -> FP
        res = mod.greedy_match(gt, pred, MASK_IOU, tau=0.5)
        cen = mod.census(gt, pred, res)
        assert cen["tp"] == 1 and cen["fp"] == 1 and cen["fn"] == 2
        assert abs(cen["f1"] - (2 * 1) / (2 * 1 + 1 + 2)) < 1e-12


def test_protocol_constants_frozen():
    assert mod.TAU_PRIMARY == 0.50 and mod.TAU_SECONDARY == (0.30, 0.75)
    assert mod.PIPELINE_MIN_AREA_PX == 256
    assert mod.CATEGORY_MAP["dining table"] == "dining_table"
    assert set(mod.QUERY_CATEGORIES) == {"bed", "chair", "couch", "dining_table"}
