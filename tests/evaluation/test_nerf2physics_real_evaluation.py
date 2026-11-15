"""Real cross-scene NeRF2Physics mass evaluation tests (PR 6c).

Skipped unless ``--run-real-model`` is passed AND the cross-scene evaluation
bundle exists. No downloads.

Unlock conditions:
  1. Evaluation bundle staged under ``tests/fixtures/data/pr6/evaluation/``:
       scene_ids           JSON array of scene IDs (>= 3 scenes)
       ground_truth        JSON object mapping product_id -> kg
       predictions_pass    JSON object mapping scene_id -> [low, high] kg
                           that satisfies every physical-validation threshold
       predictions_fail    same shape, with at least one scene violating a
                           threshold (used to test threshold semantics)
       reference_metrics   JSON object {"pass": {...}, "fail": {...}} with
                           ADE_kg/ALDE/APE/MnRE/proposal_range_coverage from
                           the independent reference computation
  2. Thresholds below are the single source of truth for
     ``physical_validation_passed`` semantics; the evaluation module is
     expected to adopt them. Until it does, the semantic tests fail and that
     failure is the signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simready.core.artifacts import read_json
from simready.evaluation.nerf2physics import evaluate_mass_predictions
from tests.paths import REPOSITORY_ROOT

pytestmark = pytest.mark.real_model

# ---------------------------------------------------------------------------
# Parameterization.
# ---------------------------------------------------------------------------

EVAL_DIR = REPOSITORY_ROOT / "tests" / "fixtures" / "data" / "pr6" / "evaluation"

REQUIRED_EVAL_FILES = (
    "scene_ids",
    "ground_truth",
    "predictions_pass",
    "predictions_fail",
    "reference_metrics",
)

MIN_SCENE_COUNT = 3
METRIC_TOLERANCE = 0.05  # 5% relative tolerance vs reference metrics
METRIC_KEYS = ("ADE_kg", "ALDE", "APE", "MnRE", "proposal_range_coverage")

# Physical validation thresholds. A scored report "passes" only when every
# bound holds. These are evaluation-policy constants, not model behaviour.
THRESHOLD_ADE_KG = 0.5
THRESHOLD_APE = 0.5
THRESHOLD_MNRE_MIN = 0.5
THRESHOLD_COVERAGE_MIN = 0.5


def _expected_physical_validation(metrics: dict) -> bool:
    """Single source of truth for what physical_validation_passed should mean."""
    if any(metrics[key] is None for key in METRIC_KEYS):
        return False
    return (
        metrics["ADE_kg"] <= THRESHOLD_ADE_KG
        and metrics["APE"] <= THRESHOLD_APE
        and metrics["MnRE"] >= THRESHOLD_MNRE_MIN
        and metrics["proposal_range_coverage"] >= THRESHOLD_COVERAGE_MIN
    )


# ---------------------------------------------------------------------------
# Asset loading.
# ---------------------------------------------------------------------------


def _load_eval_bundle() -> dict[str, Path]:
    files = {}
    missing = []
    for key in REQUIRED_EVAL_FILES:
        path = EVAL_DIR / f"{key}.json"
        if not path.exists():
            missing.append(str(path))
        files[key] = path
    if missing:
        pytest.skip("evaluation bundle not staged: " + ", ".join(missing))
    return files


@pytest.fixture(scope="module")
def eval_bundle():
    return _load_eval_bundle()


def _run_evaluation(eval_bundle, prediction_key: str, output_path: Path):
    files = eval_bundle
    scene_ids = read_json(files["scene_ids"])
    assert isinstance(scene_ids, list) and len(scene_ids) >= MIN_SCENE_COUNT, (
        f"cross-scene evaluation needs >= {MIN_SCENE_COUNT} scenes, "
        f"got {scene_ids!r}"
    )
    return evaluate_mass_predictions(
        files[prediction_key],
        files["ground_truth"],
        scene_ids,
        output_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scored_report_has_full_coverage(eval_bundle, tmp_path):
    report = _run_evaluation(eval_bundle, "predictions_pass", tmp_path / "pass.json")
    assert report["status"] == "SCORED", f"errors: {report['errors']}"
    assert report["prediction_coverage"] == 1.0
    assert report["scored_count"] == report["expected_count"]
    assert len(report["samples"]) >= MIN_SCENE_COUNT


@pytest.mark.parametrize("case", ["pass", "fail"])
def test_metrics_match_reference(eval_bundle, tmp_path, case):
    report = _run_evaluation(
        eval_bundle, f"predictions_{case}", tmp_path / f"{case}.json"
    )
    reference = read_json(eval_bundle["reference_metrics"])[case]
    assert report["status"] == "SCORED"
    for key in METRIC_KEYS:
        got = report["metrics"][key]
        want = reference[key]
        assert got is not None, f"{key} unexpectedly None for {case} case"
        tolerance = abs(want) * METRIC_TOLERANCE
        assert abs(got - want) <= tolerance, (
            f"{case}.{key}: {got} vs reference {want} (tolerance {tolerance})"
        )


def test_physical_validation_passes_on_in_threshold_predictions(eval_bundle, tmp_path):
    """Metrics within thresholds must flip physical_validation_passed to True.

    The current evaluation module hard-codes ``physical_validation_passed``
    to ``False``. This test fails until the module adopts the threshold
    semantics defined in this file; the failure is the intended signal.
    """
    report = _run_evaluation(eval_bundle, "predictions_pass", tmp_path / "pass.json")
    expected = _expected_physical_validation(report["metrics"])
    assert expected is True, (
        f"fixture contract broken: pass-case metrics {report['metrics']} do not "
        f"satisfy the thresholds; fix tests/fixtures/data/pr6/evaluation/"
        "predictions_pass.json"
    )
    assert report["physical_validation_passed"] is True, (
        "metrics satisfy every threshold "
        f"(ADE<={THRESHOLD_ADE_KG}, APE<={THRESHOLD_APE}, "
        f"MnRE>={THRESHOLD_MNRE_MIN}, coverage>={THRESHOLD_COVERAGE_MIN}) "
        f"yet physical_validation_passed is {report['physical_validation_passed']}; "
        "the evaluation module must adopt the threshold semantics"
    )


def test_physical_validation_fails_on_out_of_threshold_predictions(eval_bundle, tmp_path):
    report = _run_evaluation(eval_bundle, "predictions_fail", tmp_path / "fail.json")
    expected = _expected_physical_validation(report["metrics"])
    assert expected is False, (
        f"fixture contract broken: fail-case metrics {report['metrics']} still "
        f"satisfy the thresholds; fix tests/fixtures/data/pr6/evaluation/"
        "predictions_fail.json"
    )
    assert report["physical_validation_passed"] is False, (
        "metrics violate at least one threshold yet "
        f"physical_validation_passed is {report['physical_validation_passed']}"
    )


def test_threshold_semantics_flip_with_metrics():
    """The pass/fail predicate itself must respond to each threshold."""
    good = {
        "ADE_kg": THRESHOLD_ADE_KG / 2,
        "ALDE": 0.1,
        "APE": THRESHOLD_APE / 2,
        "MnRE": (THRESHOLD_MNRE_MIN + 1) / 2,
        "proposal_range_coverage": (THRESHOLD_COVERAGE_MIN + 1) / 2,
    }
    assert _expected_physical_validation(good) is True
    for key, bad_value in (
        ("ADE_kg", THRESHOLD_ADE_KG * 2),
        ("APE", THRESHOLD_APE * 2),
        ("MnRE", THRESHOLD_MNRE_MIN / 2),
        ("proposal_range_coverage", THRESHOLD_COVERAGE_MIN / 2),
    ):
        mutated = {**good, key: bad_value}
        assert _expected_physical_validation(mutated) is False, (
            f"predicate insensitive to {key}"
        )
    none_metrics = {**good, "ADE_kg": None}
    assert _expected_physical_validation(none_metrics) is False
