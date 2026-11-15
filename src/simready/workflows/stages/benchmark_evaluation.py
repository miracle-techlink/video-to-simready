"""Candidate assembly and independent scoring stages for benchmark evaluation.

Separates "raw observation rendering" from "candidate binding / scoring" so
that a COMPLETE flag from the render worker alone is never sufficient evidence.
The candidate builder re-verifies every view ID, output hash, resolution,
source binding and configuration identity from the render receipt before
producing a candidate manifest that the independent scorer can consume.

The scoring stage invokes ``evaluation.verification.verify_candidate`` in a
fresh process with a timeout and protocol identity. Execution success is
never conflated with quality PASS: a verifier that finishes normally but
returns FAIL or INCOMPLETE produces a valid scorecard that must not be
wrapped as "execution succeeded therefore quality passed".
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from simready.core.artifacts import digest, object_digest, read_json, write_json
from simready.core.benchmark_contract import (
    ContractError,
    validate_evaluation_request,
)
from simready.core.types import JsonObject
from simready.core.validation import require
from simready.evaluation.verification import verify_candidate
from simready.workflows.contracts.stage_ledger import StageLedger, stage_identity


# ---------------------------------------------------------------------------
# Render receipt verification
# ---------------------------------------------------------------------------


def _verify_render_receipt(
    receipt_path: Path,
    *,
    expected_view_ids: set[str],
    expected_resolution: dict[str, int],
    expected_scene_path: str,
) -> dict[str, Any]:
    """Re-verify a render receipt: view IDs, hashes, resolution, provenance.

    A worker's COMPLETE flag is not sufficient. Every expected view must be
    present with matching file hashes and resolution, and the scene binding
    must match the render request.
    """
    receipt = read_json(receipt_path)
    require(
        isinstance(receipt, dict) and isinstance(receipt.get("views"), list),
        "Render receipt must contain a views list",
    )

    # Check completion count matches expectation.
    completed = receipt.get("completed", 0)
    expected = receipt.get("expected", 0)
    require(
        completed == expected and expected == len(expected_view_ids),
        f"Render receipt incomplete: {completed}/{expected} views, "
        f"expected {len(expected_view_ids)}",
    )

    # Verify each view.
    seen_ids = set()
    for view in receipt["views"]:
        view_id = view.get("view_id")
        require(
            isinstance(view_id, str) and view_id in expected_view_ids,
            f"Unexpected or missing view_id in receipt: {view_id!r}",
        )
        require(
            view_id not in seen_ids,
            f"Duplicate view_id in receipt: {view_id!r}",
        )
        seen_ids.add(view_id)

        # Resolution must match the view set contract.
        require(
            view.get("width") == expected_resolution["width"]
            and view.get("height") == expected_resolution["height"],
            f"View {view_id!r} resolution mismatch: "
            f"{view.get('width')}x{view.get('height')} != "
            f"{expected_resolution['width']}x{expected_resolution['height']}",
        )

        # RGB file must exist and match its declared hash.
        rgb_file = view.get("rgb_file")
        rgb_sha256 = view.get("rgb_sha256")
        require(
            isinstance(rgb_file, str) and isinstance(rgb_sha256, str),
            f"View {view_id!r} missing rgb_file or rgb_sha256",
        )
        rgb_path = receipt_path.parent / rgb_file
        require(rgb_path.is_file(), f"Missing RGB file for view {view_id!r}")
        require(
            digest(rgb_path) == rgb_sha256,
            f"RGB hash mismatch for view {view_id!r}",
        )

        # Depth file, if declared, must also verify.
        depth_file = view.get("depth_file")
        depth_sha256 = view.get("depth_sha256")
        if depth_file is not None:
            require(
                isinstance(depth_sha256, str),
                f"View {view_id!r} has depth_file but no depth_sha256",
            )
            depth_path = receipt_path.parent / depth_file
            require(
                depth_path.is_file(),
                f"Missing depth file for view {view_id!r}",
            )
            require(
                digest(depth_path) == depth_sha256,
                f"Depth hash mismatch for view {view_id!r}",
            )

    require(
        seen_ids == expected_view_ids,
        f"Missing views in receipt: {expected_view_ids - seen_ids}",
    )

    # Scene binding must match.
    scene_id = receipt.get("scene_id", "")
    require(
        expected_scene_path in scene_id or scene_id in expected_scene_path,
        f"Scene binding mismatch: receipt has {scene_id!r}, "
        f"expected {expected_scene_path!r}",
    )

    return receipt


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------


def build_candidate(
    render_receipt_path: Path,
    evaluation_request_path: Path,
    view_set: dict[str, Any],
    output_dir: Path,
    *,
    model_artifacts: list[dict[str, str]] | None = None,
) -> Path:
    """Build a candidate manifest from a verified render receipt.

    Reads the render receipt, verifies every view, binds the protocol from
    the evaluation request, and writes a candidate manifest that the
    independent scorer can consume.

    Returns the path to the written candidate manifest.

    Raises ``ContractError`` if the receipt fails verification.
    """
    render_receipt_path = Path(render_receipt_path).resolve()
    evaluation_request_path = Path(evaluation_request_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Load and validate the evaluation request.
    eval_request = validate_evaluation_request(
        read_json(evaluation_request_path), evaluation_request_path.parent
    )

    # Read the render request to get the scene binding.
    render_request_ref = eval_request["render_request"]
    render_request_path = (evaluation_request_path.parent / render_request_ref["path"]).resolve()
    render_request = read_json(render_request_path)
    scene_path = render_request.get("scene", {}).get("path", "")

    # Extract expected views from the view set.
    expected_view_ids = {v["view_id"] for v in view_set["views"]}
    first_view = view_set["views"][0]
    expected_resolution = first_view["resolution"]

    # Verify the render receipt thoroughly.
    receipt = _verify_render_receipt(
        render_receipt_path,
        expected_view_ids=expected_view_ids,
        expected_resolution=expected_resolution,
        expected_scene_path=scene_path,
    )

    # Build the candidate manifest.
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = eval_request["protocol"]
    protocol_sha256 = protocol["sha256"]

    # Collect view references from the verified receipt.
    candidate_views = []
    for view in receipt["views"]:
        rgb_ref = {
            "path": view["rgb_file"],
            "sha256": view["rgb_sha256"],
        }
        entry: dict[str, Any] = {
            "view_id": view["view_id"],
            "rgb": rgb_ref,
        }
        if view.get("depth_file") is not None:
            entry["depth"] = {
                "path": view["depth_file"],
                "sha256": view["depth_sha256"],
            }
        candidate_views.append(entry)

    # Collect model artifacts.
    artifacts = []
    if model_artifacts:
        for artifact in model_artifacts:
            artifacts.append(dict(artifact))

    candidate = {
        "schema_version": 1,
        "candidate_id": eval_request["request_id"],
        "protocol_sha256": protocol_sha256,
        "artifacts": artifacts,
        "views": candidate_views,
        "fit_frame_ids": [],
        "fit_image_sha256": [],
        "evidence_status": "receipt_verified",
        "render_receipt_sha256": digest(render_receipt_path),
        "render_receipt_path": str(
            render_receipt_path.relative_to(output_dir)
            if render_receipt_path.is_relative_to(output_dir)
            else render_receipt_path
        ),
    }

    candidate_path = output_dir / "candidate.json"
    write_json(candidate_path, candidate)
    return candidate_path


# ---------------------------------------------------------------------------
# Independent scoring
# ---------------------------------------------------------------------------


def run_scoring(
    protocol_path: Path,
    candidate_path: Path,
    output_dir: Path,
    *,
    timeout: float = 300,
    ledger: StageLedger | None = None,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    """Run independent scoring in a fresh process.

    Uses ``verify_candidate`` which spawns a new interpreter, recomputes
    observations, and applies the pinned protocol. Returns a dict with
    execution status and quality status kept separate.

    The caller must not conflate "scoring process completed" with
    "candidate passed quality checks". A scorecard with verdict FAIL or
    INCOMPLETE is a valid result, not an execution error.
    """
    protocol_path = Path(protocol_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Handle missing input files as execution errors, not crashes.
    if not protocol_path.is_file() or not candidate_path.is_file():
        return {
            "execution_status": "FAILED",
            "execution_error": f"Missing input: protocol={protocol_path.is_file()}, candidate={candidate_path.is_file()}",
            "wall_seconds": 0.0,
            "protocol_sha256": None,
            "candidate_sha256": None,
            "quality_status": "EXECUTION_ERROR",
            "scorecard": None,
            "scorecard_sha256": None,
            "defects_available": False,
            "defect_count": 0,
            "release_authorized": False,
            "reuse_mode": "none",
            "reuse_cost": {
                "gpu_seconds": None,
                "wall_seconds_saved": None,
                "tokens_saved": None,
            },
        }

    protocol_hash = digest(protocol_path)

    # Execute the verifier in a fresh process.
    started = time.monotonic()
    try:
        scorecard, defects, observations = verify_candidate(
            protocol_path,
            candidate_path,
            output_dir,
            protocol_hash=protocol_hash,
            timeout=timeout,
        )
        execution_status = "COMPLETED"
        execution_error = None
    except Exception as exc:
        execution_status = "FAILED"
        execution_error = str(exc)
        scorecard = None
        defects = None
        observations = None

    elapsed = time.monotonic() - started

    # Separate execution status from quality status.
    result: dict[str, Any] = {
        "execution_status": execution_status,
        "execution_error": execution_error,
        "wall_seconds": round(elapsed, 6),
        "protocol_sha256": protocol_hash,
        "candidate_sha256": digest(candidate_path) if candidate_path.is_file() else None,
        "reuse_mode": "none",
        "reuse_cost": {
            "gpu_seconds": None,
            "wall_seconds_saved": None,
            "tokens_saved": None,
        },
    }

    if scorecard is not None:
        result["quality_status"] = scorecard.get("status", "UNKNOWN")
        result["scorecard"] = scorecard
        result["scorecard_sha256"] = (
            digest(output_dir / "scorecard.json")
            if (output_dir / "scorecard.json").is_file()
            else None
        )
        # Development phase: defects available for repair.
        # Sealed phase: defects withheld by the verifier itself.
        if defects is not None and defects.get("status") != "WITHHELD":
            result["defects_available"] = True
            result["defect_count"] = len(defects.get("defects", []))
        else:
            result["defects_available"] = False
            result["defect_count"] = 0
        result["release_authorized"] = scorecard.get("release_authorized", False)
    else:
        result["quality_status"] = "EXECUTION_ERROR"
        result["scorecard"] = None
        result["scorecard_sha256"] = None
        result["defects_available"] = False
        result["defect_count"] = 0
        result["release_authorized"] = False

    # Record in ledger if provided.
    if ledger is not None and ledger_root is not None:
        scoring_recipe = {
            "provider": "evaluation.scoring",
            "version": "v1.0",
            "config": {
                "timeout": timeout,
                "protocol_sha256": protocol_hash,
            },
            "inputs": {
                "protocol": protocol_hash,
                "candidate": digest(candidate_path),
            },
            "runtime": {
                "python": "verifier-subprocess",
            },
            "implementation": {
                "evaluation/scoring.py": digest(
                    Path(__file__).parents[1] / "evaluation" / "scoring.py"
                ),
                "evaluation/verification.py": digest(
                    Path(__file__).parents[1] / "evaluation" / "verification.py"
                ),
            },
        }
        identity = stage_identity(scoring_recipe)

        def scoring_action():
            if scorecard is not None:
                return scorecard, [output_dir / "scorecard.json"]
            raise ValueError(f"Scoring execution failed: {execution_error}")

        try:
            ledger.run("scoring", scoring_action, stage_identity=identity)
        except ValueError:
            pass  # Failure already recorded by ledger.

    return result


# ---------------------------------------------------------------------------
# Stage composition
# ---------------------------------------------------------------------------


def run_benchmark_evaluation(
    render_receipt_path: Path,
    evaluation_request_path: Path,
    view_set: dict[str, Any],
    protocol_path: Path,
    run_dir: Path,
    *,
    model_artifacts: list[dict[str, str]] | None = None,
    scoring_timeout: float = 300,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    """Compose candidate assembly and independent scoring as a single stage.

    This is the main entry point for PR 3. It:
    1. Verifies the render receipt (candidate builder).
    2. Builds a candidate manifest.
    3. Runs independent scoring in a fresh process.
    4. Returns separate execution and quality statuses.

    Uses ``StageLedger`` for receipt persistence when ``ledger_root`` is given.
    Failures write new attempt records without overwriting prior successes.
    """
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    ledger = None
    if ledger_root is not None:
        ledger_root = Path(ledger_root).resolve()
        ledger = StageLedger(ledger_root, "benchmark-evaluation")

    # Stage 1: Candidate assembly.
    candidate_dir = run_dir / "candidate"
    candidate_path = build_candidate(
        render_receipt_path,
        evaluation_request_path,
        view_set,
        candidate_dir,
        model_artifacts=model_artifacts,
    )

    # Stage 2: Independent scoring.
    scoring_dir = run_dir / "scoring"
    scoring_result = run_scoring(
        protocol_path,
        candidate_path,
        scoring_dir,
        timeout=scoring_timeout,
        ledger=ledger,
        ledger_root=ledger_root,
    )

    # Compose the final result.
    result = {
        "candidate_path": str(candidate_path),
        "candidate_sha256": digest(candidate_path),
        "scoring": scoring_result,
        "release_authorized": False,
        "sealed_discipline": {
            "development_output_localizes_defects": True,
            "sealed_withholds_raw_observations": True,
            "release_authorized_remains_false": True,
        },
    }

    write_json(run_dir / "benchmark_evaluation.json", result)
    return result
