"""Multi-scene benchmark orchestration: thin wrapper over per-scene matrices.

Each scene is executed as an independent ``baseline_matrix`` run; this module
only sequences them, records per scene×member rows, and aggregates. It does
not implement its own renderer, candidate builder, scorer or scheduler.

Aggregation rules:
- The denominator for failure rates is the total number of planned
  scene×member samples, never the number of successful runs.
- Quality metrics are reported per scene first; cross-scene aggregation is
  the unweighted mean of per-scene pass rates so that a scene with many
  frames does not dominate.
- Missing quality (NOT_EVALUATED / EXECUTION_ERROR) is never filled with
  zero and never silently dropped; it is counted separately.
- No cross-protocol or cross-split ranking is produced, and no backend is
  promoted automatically.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from simready.core.artifacts import object_digest, read_json, write_json
from simready.core.evidence import ensure_unchanged, reference, resolve_ref
from simready.core.validation import number
from simready.execution.processes.local import ProcessCancelledError
from simready.workflows.contracts.stage_ledger import failure_status
from simready.workflows.pipelines.baseline_matrix import execute_matrix, plan_matrix
from simready.workflows.planning.pipeline_spec import fields

_SUITE_SCHEMA_VERSION = 1


def _validate_suite(config, output):
    """Validate a suite manifest; return (scenes, frozen_refs)."""
    raw = read_json(config)
    fields(
        raw,
        {"schema_version", "kind", "scenes", "evaluation", "suite_id"},
        "suite",
    )
    if (
        type(raw.get("schema_version")) is not int
        or raw["schema_version"] != _SUITE_SCHEMA_VERSION
    ):
        raise ValueError(f"Expected suite schema_version={_SUITE_SCHEMA_VERSION}")
    if raw.get("kind") != "benchmark_suite":
        raise ValueError("Suite kind must be benchmark_suite")
    scenes = raw.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Suite requires a nonempty scenes list")
    seen = set()
    frozen = [reference(config)]
    validated = []
    for row in scenes:
        fields(row, {"scene_id", "matrix"}, "scene")
        sid = row.get("scene_id")
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError("Suite scene IDs must be unique nonempty strings")
        seen.add(sid)
        matrix_path = resolve_ref(row.get("matrix"), config.parent)
        matrix_ref = reference(matrix_path)
        frozen.append(matrix_ref)
        validated.append({"scene_id": sid, "matrix": matrix_ref})
    timeout = number(raw.get("evaluation", {}).get("timeout_s", 300), "suite timeout")
    if timeout <= 0:
        raise ValueError("Suite timeout must be positive")
    return validated, frozen, raw.get("suite_id", "unnamed-suite")


def plan_suite(
    config,
    output,
    *,
    until="reconstruction",
    accept_eula=False,
    submit_reconstruction=False,
):
    """Freeze the suite manifest and every referenced scene matrix."""
    config, output = Path(config).resolve(), Path(output).absolute()
    if output.is_symlink():
        raise ValueError("Suite output must not be a symlink")
    output = output.resolve()
    scenes, frozen, suite_id = _validate_suite(config, output)
    if any(Path(ref["path"]).is_relative_to(output) for ref in frozen):
        raise ValueError("Suite output must not contain frozen inputs")
    plan = {
        "schema_version": 1,
        "suite_id": suite_id,
        "out": str(output),
        "until": until,
        "execution_authorization": {
            "accept_eula": bool(accept_eula),
            "submit_reconstruction": bool(submit_reconstruction),
        },
        "scenes": scenes,
        "frozen_references": frozen,
    }
    plan["identity"] = object_digest(plan)
    return plan


def _run_scene(scene, plan, suite_root):
    """Execute one scene matrix; return the raw matrix report plus scene context.

    Cancellation and interrupts propagate immediately; other exceptions are
    captured as a scene-level failure so that remaining scenes still run.
    """
    matrix_path = Path(scene["matrix"]["path"])
    scene_out = suite_root / "scenes" / scene["scene_id"]
    matrix_plan = plan_matrix(
        matrix_path,
        scene_out,
        until=plan["until"],
        **plan["execution_authorization"],
    )
    started = time.monotonic()
    try:
        report = execute_matrix(matrix_plan)
    except (ProcessCancelledError, KeyboardInterrupt):
        raise
    except BaseException as error:
        report = {
            "status": failure_status(error),
            "error": f"{type(error).__name__}: {error}",
            "members": [],
        }
    elapsed = time.monotonic() - started
    return {
        "scene_id": scene["scene_id"],
        "matrix_identity": matrix_plan["identity"],
        "planned_members": len(matrix_plan["members"]),
        "report": report,
        "wall_seconds": round(elapsed, 6),
    }


def _aggregate(scene_results, plan_scenes):
    """Per-scene and cross-scene aggregation with fixed denominators.

    ``plan_scenes`` is the suite plan's scene list; scenes that never started
    contribute their planned member count to the denominator but zero to all
    execution counters.
    """
    per_scene = []
    total_planned = total_executed = total_failed = total_cancelled = 0
    total_timeout = total_not_run = 0
    total_quality_measured = total_quality_pass = total_quality_fail = 0
    total_quality_incomplete = total_quality_execution_error = 0
    total_quality_not_evaluated = 0
    all_rows = []

    for sr in scene_results:
        report = sr["report"]
        members = report.get("members", [])
        planned = len(members)
        executed = sum(1 for m in members if m.get("status") not in {"NOT_RUN"})
        failed = sum(1 for m in members if m.get("status") == "FAILED")
        cancelled = sum(1 for m in members if m.get("status") == "CANCELLED")
        timeout = sum(1 for m in members if m.get("status") == "TIMEOUT")
        not_run = sum(1 for m in members if m.get("status") == "NOT_RUN")

        q_measured = q_pass = q_fail = q_incomplete = q_error = q_not_eval = 0
        rows = []
        for m in members:
            quality = m.get("quality", {})
            q_status = quality.get("status", "NOT_EVALUATED")
            if q_status == "MEASURED":
                q_measured += 1
                v_status = quality.get("verifier_status", "UNKNOWN")
                if v_status == "PASS":
                    q_pass += 1
                elif v_status == "FAIL":
                    q_fail += 1
                elif v_status == "INCOMPLETE":
                    q_incomplete += 1
                else:
                    q_error += 1
            elif q_status == "EXECUTION_ERROR":
                q_error += 1
            else:
                q_not_eval += 1
            rows.append(
                {
                    "member_id": m.get("id"),
                    "execution_status": m.get("status", "NOT_RUN"),
                    "quality_status": q_status,
                    "verifier_status": quality.get("verifier_status"),
                    "candidate_mode": m.get("candidate_mode", "external"),
                    "wall_seconds": m.get("wall_seconds"),
                    "error": m.get("error"),
                    "missing_quality": q_status != "MEASURED",
                    "candidate_path": (m.get("artifacts") or {}).get("candidate"),
                    "scorecard_path": (m.get("artifacts") or {}).get("scorecard"),
                    "receipt_path": (m.get("artifacts") or {}).get("render_receipt"),
                }
            )

        total_planned += planned
        total_executed += executed
        total_failed += failed
        total_cancelled += cancelled
        total_timeout += timeout
        total_not_run += not_run
        total_quality_measured += q_measured
        total_quality_pass += q_pass
        total_quality_fail += q_fail
        total_quality_incomplete += q_incomplete
        total_quality_execution_error += q_error
        total_quality_not_evaluated += q_not_eval
        all_rows.extend(rows)

        per_scene.append(
            {
                "scene_id": sr["scene_id"],
                "planned": planned,
                "executed": executed,
                "failed": failed,
                "cancelled": cancelled,
                "timeout": timeout,
                "not_run": not_run,
                "quality_measured": q_measured,
                "quality_pass": q_pass,
                "quality_fail": q_fail,
                "quality_incomplete": q_incomplete,
                "quality_execution_error": q_error,
                "quality_not_evaluated": q_not_eval,
                "pass_rate": q_pass / planned if planned else None,
                "failure_rate": failed / planned if planned else None,
                "wall_seconds": sr["wall_seconds"],
            }
        )

    # Scenes that never started still contribute their planned members.
    # We read the frozen matrix file to learn the member count without executing.
    started_ids = {sr["scene_id"] for sr in scene_results}
    for scene in plan_scenes:
        scene_id = scene["scene_id"]
        if scene_id in started_ids:
            continue
        matrix_path = Path(scene["matrix"]["path"])
        matrix_raw = read_json(matrix_path)
        planned_count = len(matrix_raw.get("members", []))
        total_planned += planned_count
        per_scene.append(
            {
                "scene_id": scene_id,
                "planned": planned_count,
                "executed": 0,
                "failed": 0,
                "cancelled": 0,
                "timeout": 0,
                "not_run": planned_count,
                "quality_measured": 0,
                "quality_pass": 0,
                "quality_fail": 0,
                "quality_incomplete": 0,
                "quality_execution_error": 0,
                "quality_not_evaluated": 0,
                "pass_rate": None,
                "failure_rate": None,
                "wall_seconds": None,
            }
        )

    # Cross-scene: unweighted mean of per-scene rates; denominators stay explicit.
    scene_count = len(per_scene)
    mean_pass_rate = (
        sum(s["pass_rate"] or 0 for s in per_scene) / scene_count
        if scene_count
        else None
    )
    mean_failure_rate = (
        sum(s["failure_rate"] or 0 for s in per_scene) / scene_count
        if scene_count
        else None
    )

    return {
        "per_scene": per_scene,
        "cross_scene": {
            "scene_count": scene_count,
            "total_planned": total_planned,
            "total_executed": total_executed,
            "total_failed": total_failed,
            "total_cancelled": total_cancelled,
            "total_timeout": total_timeout,
            "total_not_run": total_not_run,
            "total_quality_measured": total_quality_measured,
            "total_quality_pass": total_quality_pass,
            "total_quality_fail": total_quality_fail,
            "total_quality_incomplete": total_quality_incomplete,
            "total_quality_execution_error": total_quality_execution_error,
            "total_quality_not_evaluated": total_quality_not_evaluated,
            "mean_pass_rate": mean_pass_rate,
            "mean_failure_rate": mean_failure_rate,
        },
        "rows": all_rows,
    }


def execute_suite(plan):
    """Run all scenes in order, then aggregate. Cancel propagates; partial results persist."""
    root = Path(plan["out"])
    if root.exists() or root.is_symlink():
        raise FileExistsError("Choose a fresh suite output directory")
    expected = dict(plan)
    identity = expected.pop("identity")
    if object_digest(expected) != identity:
        raise ValueError("Suite manifest identity mismatch")
    ensure_unchanged(plan["frozen_references"])
    root.mkdir(parents=True)
    write_json(root / "suite-plan.json", plan)

    report = {
        "schema_version": 1,
        "suite_identity": identity,
        "status": "RUNNING",
        "suite_id": plan["suite_id"],
        "until": plan["until"],
        "scenes": [],
        "aggregation": None,
        "ranking": None,
        "default_provider_promotion": None,
    }
    write_json(root / "suite-result.json", report)

    scene_results = []
    try:
        for scene in plan["scenes"]:
            result = _run_scene(scene, plan, root)
            scene_results.append(result)
            report["scenes"].append(
                {
                    "scene_id": result["scene_id"],
                    "matrix_identity": result["matrix_identity"],
                    "status": result["report"].get("status", "UNKNOWN"),
                    "wall_seconds": result["wall_seconds"],
                }
            )
            write_json(root / "suite-result.json", report)
        report["aggregation"] = _aggregate(scene_results, plan["scenes"])
        report["status"] = "COMPLETE"
    except BaseException as error:
        report["status"] = failure_status(error)
        report["aggregation"] = _aggregate(scene_results, plan["scenes"])
        raise
    finally:
        write_json(root / "suite-result.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--until", choices=["reconstruction"], default="reconstruction")
    scope.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--submit-reconstruction", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_suite(
        args.suite,
        args.out,
        until=None if args.full else args.until,
        accept_eula=args.accept_eula,
        submit_reconstruction=args.submit_reconstruction,
    )
    result = plan if args.dry_run else execute_suite(plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if args.dry_run or result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
