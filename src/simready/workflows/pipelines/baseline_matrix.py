"""Sequential, frozen pipeline comparisons; no scheduler or automatic ranking.

Input declarations and raw-file checks establish identity, not independent proof
of training isolation or render provenance. Synthetic fixtures test interfaces.

Candidate modes (set per member via ``candidate_source``):

- ``external``: candidate manifest must exist at planning time; its content is
  frozen and re-verified. This is the pre-PR4 behaviour.
- ``auto``: the render contract, view set and evaluation protocol are frozen at
  planning time; at execution time, after the reconstruction stage completes,
  the benchmark renderer produces a fresh receipt, a candidate is built from
  that receipt, and independent scoring is run. The candidate identity is
  bound to actual artifacts, not to a planned file that did not yet exist.
- ``reconstruction_only``: ``--until reconstruction`` semantics; no render or
  scoring is attempted. Quality is reported as ``NOT_EVALUATED`` with an
  explicit reason.

The matrix never ranks members or promotes a backend. Aggregation across
scenes is delegated to ``benchmark_suite.py``; this module remains
single-scene.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import time
from pathlib import Path

from simready.core.artifacts import digest, object_digest, read_json, write_json
from simready.core.evidence import ensure_unchanged, reference, resolve_ref
from simready.core.reconstruction_contract import verify_bundle
from simready.core.validation import number
from simready.evaluation.protocol import validate_candidate, validate_protocol
from simready.evaluation.verification import verify_candidate
from simready.execution.processes.local import ProcessCancelledError
from simready.workflows.contracts.stage_ledger import failure_status
from simready.workflows.pipelines.video_pipeline import execute_video, make_plan
from simready.workflows.planning.pipeline_spec import fields, load_pipeline
from simready.workflows.stages.benchmark_evaluation import (
    build_candidate,
    run_scoring,
)

# ---------------------------------------------------------------------------
# Helpers (unchanged from pre-PR4)
# ---------------------------------------------------------------------------


def _references(value, base):
    """Freeze nested file references using the containing document's directory."""
    found = []
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            found.append(reference(resolve_ref(value, base)))
        else:
            for item in value.values():
                found.extend(_references(item, base))
    elif isinstance(value, list):
        for item in value:
            found.extend(_references(item, base))
    return found


def _content(value):
    """Path relocation does not change a referenced file's content identity."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            return {"sha256": value["sha256"]}
        return {key: _content(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_content(item) for item in value]
    return value


def comparison_status(members, mode, camera_evidence):
    reasons, signatures = [], []
    for member in members:
        plan = member["plan"]
        config = plan["pipeline"]["reconstruction"]["config"]
        frames, coordinates = config["frames"], config["coordinates"]
        algorithm = config["algorithm"]
        if not algorithm.get("version") or not algorithm.get("implementation_sha256"):
            reasons.append(f"{member['id']}: algorithm identity incomplete")
        if frames["status"] != "declared" or not all(
            frames[key] for key in ("training", "heldout")
        ):
            reasons.append(f"{member['id']}: training/heldout declaration incomplete")
        if mode == "fixed_observations" and (
            coordinates["scale_status"] != "verified"
            or coordinates["frame"] == "unknown"
            or coordinates["unit"] not in {"m", "cm", "mm"}
        ):
            reasons.append(f"{member['id']}: camera/metric scale evidence incomplete")
        stages = {
            name: {key: value for key, value in recipe.items() if key != "dependencies"}
            for name, recipe in plan["stage_contracts"]["recipes"].items()
            if name != "reconstruction"
        }
        signatures.append(
            object_digest(
                _content(
                    {"stages": stages, "frames": frames, "coordinates": coordinates}
                )
            )
        )
    if mode == "fixed_observations":
        if camera_evidence is None:
            reasons.append("Shared camera evidence missing")
        if len(set(signatures)) != 1:
            reasons.append(
                "Non-reconstruction configuration or frame/scale declarations differ"
            )
    return {
        "status": "COMPARABILITY_UNCONFIRMED" if reasons else "DECLARED_INPUTS_MATCH",
        "mode": mode,
        "reasons": reasons,
        "independent_training_isolation_verified": False,
        "render_provenance_independently_verified": False,
        "scope": "Content identities and caller declarations only; backends may access the complete video",
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

_CANDIDATE_MODES = {"external", "auto"}
_RENDER_CONTRACT_FIELDS = {
    "scene",
    "view_set",
    "renderer",
    "image_settings",
    "lighting",
    "seed",
    "budget",
}


def _freeze_render_contract(raw, config):
    """Freeze the auto-candidate render contract: scene, views, renderer identity.

    Only files that already exist are frozen; the reconstruction binding is
    deliberately absent because in auto mode the artifact is produced during
    execution and is bound only afterwards (handover PR 4).
    """
    from simready.core.benchmark_contract import validate_view_set

    contract = fields(raw, _RENDER_CONTRACT_FIELDS, "render_contract")
    scene_path = resolve_ref(contract["scene"], config.parent)
    scene_ref = reference(scene_path)
    view_set_path = resolve_ref(contract["view_set"], config.parent)
    view_set_ref = reference(view_set_path)
    view_set = validate_view_set(read_json(view_set_path), view_set_path.parent)
    frozen = [scene_ref, view_set_ref, *_references(view_set, view_set_path.parent)]
    declaration = {
        key: copy.deepcopy(contract[key])
        for key in ("renderer", "image_settings", "lighting", "seed", "budget")
    }
    declaration["scene"] = {"path": scene_ref["path"], "sha256": scene_ref["sha256"]}
    declaration["view_set"] = {
        "path": view_set_ref["path"],
        "sha256": view_set_ref["sha256"],
    }
    return {"declaration": declaration, "frozen": frozen}


def _freeze_candidate_source(raw, config, protocol, protocol_ref):
    """Validate and freeze a member's ``candidate_source`` declaration.

    Returns ``(mode, candidate_ref, frozen_extra)``. ``candidate_ref`` is
    only non-None for ``external`` mode.
    """
    if raw is None:
        # No candidate source declared → external with no candidate (legacy).
        return "external", None, []
    fields(raw, {"mode", "reference"}, "candidate_source")
    mode = raw.get("mode", "external")
    if mode not in _CANDIDATE_MODES:
        raise ValueError(f"Unknown candidate_source mode: {mode!r}")
    if mode == "external":
        if raw.get("reference") is None:
            return "external", None, []
        if protocol is None:
            raise ValueError(
                "External candidate evidence requires a shared evaluation protocol"
            )
        candidate_path = resolve_ref(raw["reference"], config.parent)
        candidate_ref = reference(candidate_path)
        candidate = read_json(candidate_path)
        validate_candidate(candidate, protocol_ref["sha256"], protocol)
        extra = [candidate_ref, *_references(candidate, candidate_path.parent)]
        return "external", candidate_ref, extra
    # auto mode: no pre-existing reference allowed.
    if raw.get("reference") is not None:
        raise ValueError("Auto candidate source must not bind a pre-existing reference")
    if protocol is None:
        raise ValueError("Auto candidate requires a shared evaluation protocol")
    return "auto", None, []


def plan_matrix(
    config,
    output,
    *,
    until="reconstruction",
    accept_eula=False,
    submit_reconstruction=False,
):
    config, output = Path(config).resolve(), Path(output).absolute()
    if output.is_symlink():
        raise ValueError("Matrix output must not be a symlink")
    output = output.resolve()
    raw = read_json(config)
    fields(
        raw,
        {"schema_version", "mode", "purpose", "video", "members", "evaluation"},
        "matrix",
    )
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Expected matrix schema_version=1")
    mode = raw.get("mode")
    if mode not in {"fixed_observations", "end_to_end"}:
        raise ValueError("Matrix mode must be fixed_observations or end_to_end")
    purpose = raw.get("purpose", "interface_validation")
    if purpose not in {"interface_validation", "quality_comparison"}:
        raise ValueError("Unknown matrix purpose")
    if until not in {None, "reconstruction"}:
        raise ValueError("Matrix supports until=reconstruction or an explicit full run")
    video = reference(resolve_ref(raw.get("video"), config.parent))
    evaluation = fields(
        raw.get("evaluation", {}),
        {"protocol", "camera_evidence", "timeout_s", "render_contract"},
        "evaluation",
    )
    timeout = evaluation.get("timeout_s", 300)
    if number(timeout, "evaluation timeout") <= 0:
        raise ValueError("Evaluation timeout must be positive")
    frozen = [reference(config), video]
    protocol_ref = camera_ref = protocol = None
    render_contract = None
    if evaluation.get("protocol") is not None:
        protocol_path = resolve_ref(evaluation["protocol"], config.parent)
        protocol_ref = reference(protocol_path)
        protocol = validate_protocol(read_json(protocol_path))
        frozen += [protocol_ref, *_references(protocol, protocol_path.parent)]
    if evaluation.get("camera_evidence") is not None:
        camera_ref = reference(
            resolve_ref(evaluation["camera_evidence"], config.parent)
        )
        frozen.append(camera_ref)
    if evaluation.get("render_contract") is not None:
        render_contract = _freeze_render_contract(evaluation["render_contract"], config)
        frozen += render_contract["frozen"]
    rows = raw.get("members")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Matrix requires at least two members")
    members, names = [], set()
    for row in rows:
        fields(row, {"id", "pipeline", "candidate", "candidate_source"}, "member")
        name = row.get("id")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name)
            or name in names
        ):
            raise ValueError("Member IDs must be unique safe directory names")
        names.add(name)
        path = resolve_ref(row.get("pipeline"), config.parent)
        pipeline_ref = reference(path)
        args = load_pipeline(
            path,
            output / "members" / name,
            until=until,
            accept_eula=accept_eula,
            submit_reconstruction=submit_reconstruction,
        )
        plan = make_plan(args)
        if plan["source_video_sha256"] != video["sha256"]:
            raise ValueError("Every matrix member must use the same frozen video")

        # Resolve candidate source: explicit per-member "candidate_source" wins;
        # otherwise the legacy "candidate" key implies external mode.
        if "candidate_source" in row:
            cand_mode, candidate_ref, extra = _freeze_candidate_source(
                row["candidate_source"], config, protocol, protocol_ref
            )
        elif row.get("candidate") is not None:
            cand_mode = "external"
            if protocol is None:
                raise ValueError(
                    "Candidate evidence requires a shared evaluation protocol"
                )
            candidate_path = resolve_ref(row["candidate"], config.parent)
            candidate_ref = reference(candidate_path)
            candidate = read_json(candidate_path)
            validate_candidate(candidate, protocol_ref["sha256"], protocol)
            extra = [candidate_ref, *_references(candidate, candidate_path.parent)]
        else:
            cand_mode, candidate_ref, extra = "external", None, []
        if cand_mode == "auto" and render_contract is None:
            raise ValueError(
                "Auto candidate source requires a frozen evaluation.render_contract"
            )
        frozen += extra

        frozen += [pipeline_ref, *_references(plan, path.parent)]
        members.append(
            {
                "id": name,
                "pipeline": pipeline_ref,
                "candidate": candidate_ref,
                "candidate_mode": cand_mode,
                "plan": plan,
            }
        )
    frozen = sorted(
        {(ref["path"], ref["sha256"]): ref for ref in frozen}.values(),
        key=lambda ref: ref["path"],
    )
    if any(Path(ref["path"]).is_relative_to(output) for ref in frozen):
        raise ValueError("Matrix output must not contain frozen inputs")
    manifest = {
        "schema_version": 1,
        "mode": mode,
        "purpose": purpose,
        "video": video,
        "out": str(output),
        "until": until,
        "execution_authorization": {
            "accept_eula": bool(accept_eula),
            "submit_reconstruction": bool(submit_reconstruction),
        },
        "members": members,
        "frozen_references": frozen,
        "evaluation": {
            "protocol": protocol_ref,
            "camera_evidence": camera_ref,
            "timeout_s": timeout,
            "protocol_is_caller_supplied": True,
            "render_contract": (
                render_contract["declaration"] if render_contract else None
            ),
        },
        "comparison": comparison_status(members, mode, camera_ref),
        "ranking": None,
        "default_provider_promotion": None,
    }
    manifest["identity"] = object_digest(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Quality evaluation — external candidate (pre-PR4 path)
# ---------------------------------------------------------------------------


def _quality(member, manifest, directory, reconstruction):
    evaluation = manifest["evaluation"]
    unknown = {
        "status": "NOT_EVALUATED",
        "reason": "Real heldout render/reference evidence not supplied",
    }
    if (
        manifest["purpose"] != "quality_comparison"
        or not member["candidate"]
        or not evaluation["protocol"]
    ):
        return unknown
    protocol_path = Path(evaluation["protocol"]["path"])
    protocol = read_json(protocol_path)
    if any(view["reference_kind"] != "real_capture" for view in protocol["views"]):
        return {
            **unknown,
            "reason": "Synthetic or estimated references do not establish model quality",
        }
    bundle = verify_bundle(
        reconstruction["bundle_path"], source_video_sha256=manifest["video"]["sha256"]
    )
    if "synthetic" in bundle["algorithm"]["name"].lower():
        return {
            **unknown,
            "reason": "Synthetic interface fixture is not a model quality baseline",
        }
    frames = bundle["frames"]
    if (
        frames["status"] != "declared"
        or not frames["training"]
        or not frames["heldout"]
    ):
        return {
            **unknown,
            "reason": "Training and heldout frame declarations are required",
        }
    candidate_path = Path(member["candidate"]["path"])
    candidate = read_json(candidate_path)
    heldout = {(item["id"], item["sha256"]) for item in frames["heldout"]}
    if (
        not {(view["frame_id"], view["rgb"]["sha256"]) for view in protocol["views"]}
        <= heldout
    ):
        raise ValueError(
            "Evaluation views are not declared heldout reconstruction frames"
        )
    for document in (protocol, candidate):
        if set(document.get("fit_frame_ids", [])) != {
            item["id"] for item in frames["training"]
        } or set(document.get("fit_image_sha256", [])) != {
            item["sha256"] for item in frames["training"]
        }:
            raise ValueError(
                "Evaluation fit declarations differ from reconstruction training frames"
            )
    if reconstruction["ply_sha256"] not in {
        item["sha256"] for item in candidate["artifacts"]
    }:
        raise ValueError(
            "Candidate evidence is not bound to this reconstruction artifact"
        )
    camera = evaluation["camera_evidence"]
    binding = candidate.get("comparison_evidence", {})
    if not camera or binding != {
        "source_video_sha256": manifest["video"]["sha256"],
        "reconstruction_sha256": reconstruction["ply_sha256"],
        "camera_sha256": camera["sha256"],
    }:
        raise ValueError(
            "Candidate needs explicit source/reconstruction/camera comparison_evidence"
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    scorecard, feedback, _ = verify_candidate(
        protocol_path,
        candidate_path,
        directory,
        protocol_hash=evaluation["protocol"]["sha256"],
        timeout=evaluation["timeout_s"],
    )
    return {
        "status": "MEASURED",
        "scorecard": reference(directory / "scorecard.json"),
        "verifier_status": scorecard["status"],
        "feedback_status": feedback.get("status", "DEVELOPMENT"),
        "independently_checked": "Raw evidence hashes and metrics recomputed by existing verifier",
        "caller_declared": "Training access, camera correspondence and render provenance",
        "release_authorized": False,
    }


# ---------------------------------------------------------------------------
# Quality evaluation — auto candidate (render + build + score)
# ---------------------------------------------------------------------------


def render_auto_observations(contract, view_set_path, reconstruction, directory):
    """Materialize a render request bound to the actual reconstruction and render.

    The planning-time contract froze the renderer identity, scene, view set
    and scoring inputs; the reconstruction binding is added here, after the
    artifact exists. Everything is written into ``directory`` as a
    self-contained, contract-valid bundle (scene/view set/reconstruction
    copies plus ``render_request.json``), so no frozen input is rewritten and
    no future file is read. Returns the render request and receipt paths.
    """
    from simready.core.benchmark_contract import validate_render_request
    from simready.simulation.benchmark_observation import (
        plan_benchmark_observation,
        run_benchmark_observation,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scene_src = Path(contract["scene"]["path"])
    ply_src = Path(reconstruction["ply_path"])
    shutil.copy2(scene_src, directory / scene_src.name)
    shutil.copy2(view_set_path, directory / "view_set.json")
    shutil.copy2(ply_src, directory / ply_src.name)
    request = {
        "schema_version": 1,
        "kind": "render_request",
        "request_id": f"auto-{directory.name}",
        "scene": {"path": scene_src.name, "sha256": digest(scene_src)},
        "reconstruction": {"path": ply_src.name, "sha256": digest(ply_src)},
        "view_set": {"path": "view_set.json", "sha256": digest(view_set_path)},
        "renderer": copy.deepcopy(contract["renderer"]),
        "image_settings": copy.deepcopy(contract["image_settings"]),
        "lighting": copy.deepcopy(contract["lighting"]),
        "seed": contract["seed"],
        "budget": copy.deepcopy(contract["budget"]),
    }
    request_path = directory / "render_request.json"
    write_json(request_path, request)
    validate_render_request(read_json(request_path), request_path.parent)
    plan = plan_benchmark_observation(
        request_path, directory / "view_set.json", directory / "render"
    )
    run_benchmark_observation(plan)
    return request_path, directory / "render" / "receipt.json"


def _quality_auto(member, manifest, directory, reconstruction):
    """Render, assemble the candidate from the verified receipt, then score.

    Planning froze the render contract (renderer identity, scene, view set),
    the scoring protocol and the timeout. Here the reconstruction has already
    completed and been hash-checked by the stage ledger; the render request is
    materialized with the actual reconstruction binding, the renderer runs,
    ``build_candidate`` re-verifies every view/hash/scene binding from the
    receipt, and only then is the candidate frozen and scored in a fresh
    verifier process. Execution and quality statuses stay separate.
    """
    evaluation = manifest["evaluation"]
    unknown = {
        "status": "NOT_EVALUATED",
        "reason": "Auto candidate evaluation requires quality_comparison purpose",
    }
    contract = evaluation.get("render_contract")
    if manifest["purpose"] != "quality_comparison" or not evaluation["protocol"]:
        return unknown
    if not contract:
        return {
            **unknown,
            "reason": "Auto candidate requires a frozen render contract",
        }
    ensure_unchanged(manifest["frozen_references"])

    directory.mkdir(parents=True, exist_ok=True)
    request_path, receipt_path = render_auto_observations(
        contract,
        Path(contract["view_set"]["path"]),
        reconstruction,
        directory / "render-input",
    )
    if not receipt_path.is_file():
        raise ValueError("Benchmark renderer produced no receipt")

    view_set = read_json(Path(contract["view_set"]["path"]))
    evaluation_request_path = _write_evaluation_request(
        directory, request_path, evaluation["protocol"], member["id"]
    )
    candidate_path = build_candidate(
        receipt_path,
        evaluation_request_path,
        view_set,
        directory / "candidate",
        model_artifacts=[
            {
                "artifact_id": "reconstruction",
                "path": str(Path(reconstruction["ply_path"])),
                "sha256": reconstruction["ply_sha256"],
            }
        ],
    )

    scoring_result = run_scoring(
        Path(evaluation["protocol"]["path"]),
        candidate_path,
        directory / "scoring",
        timeout=evaluation["timeout_s"],
    )
    scorecard_path = directory / "scoring" / "scorecard.json"
    return {
        "status": "MEASURED",
        "scorecard": reference(scorecard_path) if scorecard_path.is_file() else None,
        "verifier_status": scoring_result.get("quality_status", "UNKNOWN"),
        "feedback_status": "DEVELOPMENT"
        if scoring_result.get("defects_available")
        else "WITHHELD",
        "independently_checked": "Render receipt verified, candidate assembled, verifier recomputed metrics",
        "caller_declared": "Training access, camera correspondence and render provenance",
        "release_authorized": False,
    }


def _write_evaluation_request(directory, render_request_path, protocol_ref, request_id):
    """Write an evaluation_request manifest referencing the materialized request."""
    from simready.core.benchmark_contract import validate_evaluation_request

    directory.mkdir(parents=True, exist_ok=True)
    er = {
        "schema_version": 1,
        "kind": "evaluation_request",
        "request_id": f"auto-{request_id}",
        "protocol": {
            "name": "auto",
            "version": "1.0",
            "sha256": protocol_ref["sha256"],
        },
        "candidate_source": {"mode": "auto"},
        "scorer": {
            "name": "simready-scorer",
            "version": "1.0",
            "code_sha256": digest(Path(__file__)),
        },
        "deadline": "2026-12-31",
        "render_request": {
            "path": str(
                Path(render_request_path).relative_to(Path(directory).resolve())
            ),
            "sha256": digest(render_request_path),
        },
    }
    path = directory / "evaluation_request.json"
    write_json(path, er)
    # Validate so a contract violation here fails before any candidate exists.
    validate_evaluation_request(read_json(path), path.parent)
    return path


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _artifact_refs(member, quality, evaluation_dir):
    """Collect existing candidate/render-receipt/scorecard references for a row."""
    refs = {
        "candidate": member["candidate"]["path"] if member["candidate"] else None,
        "render_receipt": None,
        "scorecard": None,
    }
    if member["candidate_mode"] == "auto":
        candidate = evaluation_dir / "candidate" / "candidate.json"
        if candidate.is_file():
            refs["candidate"] = str(candidate)
        render_receipt = evaluation_dir / "render-input" / "render" / "receipt.json"
        if render_receipt.is_file():
            refs["render_receipt"] = str(render_receipt)
    scorecard = quality.get("scorecard") if isinstance(quality, dict) else None
    if isinstance(scorecard, dict):
        refs["scorecard"] = scorecard.get("path")
    return refs


def execute_matrix(manifest):
    """Run frozen members in order, preserving failed and interrupted rows."""
    root = Path(manifest["out"])
    if root.exists() or root.is_symlink():
        raise FileExistsError("Choose a fresh matrix output directory")
    expected = dict(manifest)
    identity = expected.pop("identity")
    if object_digest(expected) != identity:
        raise ValueError("Matrix manifest identity mismatch")
    ensure_unchanged(manifest["frozen_references"])
    root.mkdir(parents=True)
    write_json(root / "matrix-plan.json", manifest)
    report = {
        "schema_version": 1,
        "matrix_identity": identity,
        "status": "RUNNING",
        "mode": manifest["mode"],
        "purpose": manifest["purpose"],
        "until": manifest["until"],
        "comparison": copy.deepcopy(manifest["comparison"]),
        "ranking": None,
        "default_provider_promotion": None,
        "members": [
            {
                "id": member["id"],
                "status": "NOT_RUN",
                "quality": {"status": "NOT_EVALUATED"},
                "candidate_mode": member["candidate_mode"],
            }
            for member in manifest["members"]
        ],
    }
    write_json(root / "matrix-result.json", report)
    try:
        for member, row in zip(manifest["members"], report["members"], strict=True):
            started = time.monotonic()
            try:
                ensure_unchanged(manifest["frozen_references"])
                args = load_pipeline(
                    Path(member["pipeline"]["path"]),
                    Path(member["plan"]["out"]),
                    until=manifest["until"],
                    **manifest["execution_authorization"],
                )
                if make_plan(args) != member["plan"]:
                    raise ValueError(
                        "Frozen pipeline implementation or configuration changed"
                    )
                row["status"] = "RUNNING"
                write_json(root / "matrix-result.json", report)
                state, code = execute_video(args, member["plan"])
                row.update(
                    status=state["status"],
                    exit_code=code,
                    completed_pipeline=code == 0 and state["status"] == "COMPLETE",
                )
                if state["status"] == "CANCELLED":
                    raise ProcessCancelledError(0)
                if state["status"] == "INTERRUPTED":
                    raise KeyboardInterrupt()
                if code:
                    row["error"] = {
                        "type": "PipelineFailure",
                        "message": state.get("error", state["status"]),
                    }
                ensure_unchanged(manifest["frozen_references"])
                if make_plan(args) != member["plan"]:
                    raise ValueError(
                        "Pipeline inputs or implementation changed during execution"
                    )
                if code == 0:
                    receipt = read_json(
                        Path(member["plan"]["out"]) / "stages/reconstruction.json"
                    )
                    row["reconstruction"] = receipt["result"]
                    evaluation_dir = root / "evaluation" / member["id"]
                    if member["candidate_mode"] == "auto":
                        row["quality"] = _quality_auto(
                            member,
                            manifest,
                            evaluation_dir,
                            receipt["result"],
                        )
                    else:
                        row["quality"] = _quality(
                            member,
                            manifest,
                            evaluation_dir,
                            receipt["result"],
                        )
                    # Record artifact references only for files that exist;
                    # a missing artifact is never fabricated.
                    row["artifacts"] = _artifact_refs(
                        member, row["quality"], evaluation_dir
                    )
                    ensure_unchanged(manifest["frozen_references"])
            except BaseException as error:
                row.update(
                    status=failure_status(error),
                    error={"type": type(error).__name__, "message": str(error)},
                )
                row["quality"] = {
                    "status": "NOT_EVALUATED",
                    "reason": "Member execution or evidence validation failed",
                }
                if not isinstance(error, Exception) or row["status"] in {
                    "CANCELLED",
                    "INTERRUPTED",
                }:
                    raise
            finally:
                row["wall_seconds"] = round(time.monotonic() - started, 6)
                write_json(root / "matrix-result.json", report)
        report["status"] = (
            "COMPLETE"
            if all(
                row.get("exit_code") == 0 and "error" not in row
                for row in report["members"]
            )
            else "COMPLETED_WITH_FAILURES"
        )
    except BaseException as error:
        report["status"] = failure_status(error)
        raise
    finally:
        report["counts"] = {
            "planned": len(report["members"]),
            "not_run": sum(row["status"] == "NOT_RUN" for row in report["members"]),
            "quality_measured": sum(
                row["quality"]["status"] == "MEASURED" for row in report["members"]
            ),
        }
        write_json(root / "matrix-result.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--until", choices=["reconstruction"], default="reconstruction")
    scope.add_argument(
        "--full",
        action="store_true",
        help="Explicitly run the complete configured pipelines",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--submit-reconstruction", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_matrix(
        args.matrix,
        args.out,
        until=None if args.full else args.until,
        accept_eula=args.accept_eula,
        submit_reconstruction=args.submit_reconstruction,
    )
    result = plan if args.dry_run else execute_matrix(plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if args.dry_run or result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
