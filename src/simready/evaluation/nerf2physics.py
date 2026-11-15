"""Independent mass scoring against explicit measured evaluation coverage.

ADE/ALDE/APE/MnRE follow the official NeRF2Physics evaluation definitions.
Missing and invalid predictions remain in the coverage denominator. No metric
threshold promotes predictions to measured physical parameters or simulation truth.
"""

import argparse
import math
from pathlib import Path

from simready.core.artifacts import read_json, write_json
from simready.core.evidence import ensure_unchanged, reference
from simready.core.types import JsonObject, PathLike
from simready.core.validation import number, require

# Physical-validation thresholds (evaluation-policy constants, not model behaviour).
# A scored report "passes" only when every bound holds. Coverage is judged by
# ``proposal_range_coverage`` (truth falls inside predicted interval), not by
# ``prediction_coverage`` (rows scored / scenes declared); the latter is kept
# as a report field only.
PHYSICAL_VALIDATION_THRESHOLDS = {
    "ADE_kg_max": 0.5,
    "APE_max": 0.5,
    "MnRE_min": 0.5,
    "proposal_range_coverage_min": 0.5,
}


def _physical_validation_passed(metrics: dict) -> bool:
    """Threshold semantics for physical_validation_passed.

    Returns True only when every metric is present and within its bound.
    """
    if any(
        metrics.get(key) is None
        for key in ("ADE_kg", "APE", "MnRE", "proposal_range_coverage")
    ):
        return False
    t = PHYSICAL_VALIDATION_THRESHOLDS
    return (
        metrics["ADE_kg"] <= t["ADE_kg_max"]
        and metrics["APE"] <= t["APE_max"]
        and metrics["MnRE"] >= t["MnRE_min"]
        and metrics["proposal_range_coverage"] >= t["proposal_range_coverage_min"]
    )


def evaluate_mass_predictions(
    predictions_path: PathLike,
    ground_truth_path: PathLike,
    scene_ids: list[str],
    output_path: PathLike,
) -> JsonObject:
    """Score official ``scene_id -> [kg, kg]`` predictions; coverage is caller-declared."""
    require(
        isinstance(scene_ids, list)
        and scene_ids
        and all(isinstance(item, str) and item for item in scene_ids)
        and len(set(scene_ids)) == len(scene_ids),
        "Declare unique evaluation scene IDs",
    )
    frozen = [reference(predictions_path), reference(ground_truth_path)]
    predictions, truths = read_json(predictions_path), read_json(ground_truth_path)
    require(
        isinstance(predictions, dict) and isinstance(truths, dict),
        "Predictions and ground truths must be objects",
    )
    rows, errors, absolute, logarithmic, relative, ratios, covered = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for scene in scene_ids:
        product_id = scene.split("_")[0]
        try:
            truth = number(truths[product_id], "ground truth kg", 1e-15)
            interval = predictions[scene]
            require(
                isinstance(interval, list) and len(interval) == 2,
                "Prediction needs [low, high] kg",
            )
            low, high = (number(value, "predicted kg", 1e-15) for value in interval)
            require(low <= high, "Prediction interval is reversed")
            midpoint = (low + high) / 2
        except (KeyError, TypeError, ValueError) as error:
            errors.append({"scene_id": scene, "reason": str(error)})
            continue
        error = abs(midpoint - truth)
        absolute.append(error)
        logarithmic.append(abs(math.log(midpoint) - math.log(truth)))
        relative.append(error / truth)
        ratios.append(min(midpoint / truth, truth / midpoint))
        covered.append(low <= truth <= high)
        rows.append(
            {
                "scene_id": scene,
                "ground_truth_kg": truth,
                "interval_kg": interval,
                "midpoint_kg": midpoint,
                "absolute_error_kg": error,
            }
        )

    def mean(values):
        return sum(values) / len(values) if values else None

    metrics = {
        "ADE_kg": mean(absolute),
        "ALDE": mean(logarithmic),
        "APE": mean(relative),
        "MnRE": mean(ratios),
        "proposal_range_coverage": mean(covered),
    }
    prediction_coverage = len(rows) / len(scene_ids)
    report = {
        "schema_version": 1,
        "status": "SCORED" if not errors else "INCOMPLETE",
        "method": "NeRF2Physics official mass metrics (independent implementation)",
        "expected_scenes": scene_ids,
        "expected_count": len(scene_ids),
        "scored_count": len(rows),
        "prediction_coverage": prediction_coverage,
        "metrics": metrics,
        "errors": errors,
        "samples": rows,
        "extra_prediction_ids": sorted(set(predictions) - set(scene_ids)),
        "evidence": frozen,
        "physical_validation_passed": _physical_validation_passed(metrics),
        "limitations": [
            "Metrics use only valid predictions; coverage includes every declared scene.",
            "Proposal ranges are not statistical confidence intervals.",
            "Input provenance and measured ground truth must be established by the caller.",
        ],
    }
    output = Path(output_path).resolve()
    require(not output.exists(), "Evaluation output already exists")
    require(
        output not in {Path(ref["path"]) for ref in frozen},
        "Evaluation cannot overwrite inputs",
    )
    ensure_unchanged(frozen)
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--scene-ids",
        type=Path,
        required=True,
        help="JSON array defining the complete intended evaluation set",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_mass_predictions(
        args.predictions, args.ground_truth, read_json(args.scene_ids), args.out
    )
    return 0 if report["status"] == "SCORED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
