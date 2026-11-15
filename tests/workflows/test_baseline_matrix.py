import copy
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from simready.core.artifacts import digest, read_json, write_json
from simready.core.evidence import reference
from simready.execution.processes.local import ProcessCancelledError
from simready.workflows.pipelines import baseline_matrix as matrix
from tests.paths import REPOSITORY_ROOT


@pytest.fixture
def prepared(tmp_path):
    prepare = runpy.run_path(
        str(REPOSITORY_ROOT / "examples/reconstruction/prepare_fixture.py")
    )["prepare"]
    path = prepare(tmp_path / "inputs")
    return path, tmp_path / "run"


def test_dry_run_is_read_only_and_freezes_two_distinct_backend_modes(
    prepared, monkeypatch
):
    path, out = prepared
    monkeypatch.setattr(
        matrix, "execute_video", lambda *a: pytest.fail("Planning executed")
    )
    plan = matrix.plan_matrix(path, out)
    assert not out.exists()
    assert plan == matrix.plan_matrix(path, out)
    assert plan["until"] == "reconstruction"
    assert plan["comparison"]["status"] == "COMPARABILITY_UNCONFIRMED"
    assert (
        len(
            {
                row["plan"]["stage_contracts"]["identities"]["reconstruction"]
                for row in plan["members"]
            }
        )
        == 2
    )
    assert matrix.main(["--matrix", str(path), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_real_cpu_matrix_runs_command_then_import_without_quality_claim(prepared):
    path, out = prepared
    report = matrix.execute_matrix(matrix.plan_matrix(path, out))
    assert report["status"] == "COMPLETE"
    assert report["counts"] == {"planned": 2, "not_run": 0, "quality_measured": 0}
    assert [row["reconstruction"]["execution_mode"] for row in report["members"]] == [
        "fresh_command",
        "import",
    ]
    assert all(
        row["status"] == "STAGE_READY" and not row["completed_pipeline"]
        for row in report["members"]
    )
    assert all(row["quality"]["status"] == "NOT_EVALUATED" for row in report["members"])
    assert report["ranking"] is report["default_provider_promotion"] is None
    assert read_json(out / "matrix-result.json") == report
    with pytest.raises(FileExistsError):
        matrix.execute_matrix(matrix.plan_matrix(path, out))


@pytest.mark.parametrize("mutation", ["video", "duplicate", "escape", "unknown"])
def test_invalid_matrix_contracts_fail_before_output(prepared, mutation):
    path, out = prepared
    raw = read_json(path)
    if mutation == "video":
        other = path.parent / "other.mp4"
        other.write_bytes(b"other")
        raw["video"] = reference(other)
    elif mutation == "duplicate":
        raw["members"][1]["id"] = raw["members"][0]["id"]
    elif mutation == "escape":
        raw["members"][0]["id"] = "../escape"
    else:
        raw["unexpected"] = True
    write_json(path, raw)
    with pytest.raises(ValueError):
        matrix.plan_matrix(path, out)
    assert not out.exists()


def test_frozen_input_changes_rejected_before_matrix_directory(prepared):
    path, out = prepared
    plan = matrix.plan_matrix(path, out)
    Path(plan["video"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        matrix.execute_matrix(plan)
    assert not out.exists()


def test_member_failure_is_retained_and_next_member_runs(prepared, monkeypatch):
    path, out = prepared
    calls = []

    def fail(args, plan):
        calls.append(plan["out"])
        return {"status": "FAILED", "error": "deliberate fixture failure"}, 2

    monkeypatch.setattr(matrix, "execute_video", fail)
    report = matrix.execute_matrix(matrix.plan_matrix(path, out))
    assert len(calls) == 2
    assert report["status"] == "COMPLETED_WITH_FAILURES"
    assert report["counts"]["planned"] == 2
    assert all(
        row["error"]["message"] == "deliberate fixture failure"
        for row in report["members"]
    )


@pytest.mark.parametrize("raised", [False, True])
def test_cancellation_preserves_not_run_rows_and_propagates(
    prepared, monkeypatch, raised
):
    path, out = prepared
    calls = []

    def cancel(args, plan):
        calls.append(plan["out"])
        if raised:
            raise ProcessCancelledError(0)
        return {"status": "CANCELLED"}, 2

    monkeypatch.setattr(matrix, "execute_video", cancel)
    with pytest.raises(ProcessCancelledError):
        matrix.execute_matrix(matrix.plan_matrix(path, out))
    report = read_json(out / "matrix-result.json")
    assert len(calls) == 1
    assert report["status"] == "CANCELLED"
    assert [row["status"] for row in report["members"]] == ["CANCELLED", "NOT_RUN"]


def test_declared_match_is_not_independent_training_isolation(prepared):
    path, out = prepared
    members = matrix.plan_matrix(path, out)["members"]
    evidence = reference(path)
    for member in members:
        config = member["plan"]["pipeline"]["reconstruction"]["config"]
        config["algorithm"] = {
            "name": "declared",
            "version": "1",
            "implementation_sha256": "1" * 64,
        }
        config["frames"] = {
            "status": "declared",
            "training": [{"id": "train", "sha256": "2" * 64}],
            "heldout": [{"id": "test", "sha256": "3" * 64}],
            "evidence": evidence,
        }
        config["coordinates"] = {
            "frame": "z_up",
            "unit": "m",
            "scale_status": "verified",
            "evidence": evidence,
        }
    result = matrix.comparison_status(members, "fixed_observations", evidence)
    assert result["status"] == "DECLARED_INPUTS_MATCH"
    assert not result["independent_training_isolation_verified"]
    changed = copy.deepcopy(members)
    changed[1]["plan"]["stage_contracts"]["recipes"]["observations"]["config"][
        "fps"
    ] = 7
    assert (
        matrix.comparison_status(changed, "fixed_observations", evidence)["status"]
        == "COMPARABILITY_UNCONFIRMED"
    )


def test_unknown_quality_does_not_invoke_verifier(prepared, monkeypatch):
    path, out = prepared
    plan = matrix.plan_matrix(path, out)
    monkeypatch.setattr(
        matrix, "verify_candidate", lambda *a, **k: pytest.fail("No quality evidence")
    )
    result = matrix._quality(plan["members"][0], plan, out / "quality", {})
    assert result["status"] == "NOT_EVALUATED"


def test_real_reference_quality_binding_and_sealed_summary(tmp_path, monkeypatch):
    from tests.fixtures.scene_candidate import make_scene_candidate

    p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
    protocol["phase"] = "sealed_release"
    protocol["views"][0].update(split="sealed_release", reference_kind="real_capture")
    protocol["fit_frame_ids"] = ["training"]
    protocol["fit_image_sha256"] = ["1" * 64]
    write_json(p, protocol)
    candidate.update(
        protocol_sha256=digest(p),
        fit_frame_ids=["training"],
        fit_image_sha256=["1" * 64],
    )
    model_hash = digest(tmp_path / "model.bin")
    camera = tmp_path / "camera.json"
    write_json(camera, {"camera": "caller declaration"})
    candidate["comparison_evidence"] = {
        "source_video_sha256": "2" * 64,
        "reconstruction_sha256": model_hash,
        "camera_sha256": digest(camera),
    }
    write_json(c, candidate)
    bundle = {
        "algorithm": {"name": "operator_model"},
        "frames": {
            "status": "declared",
            "training": [{"id": "training", "sha256": "1" * 64}],
            "heldout": [
                {"id": "test-frame", "sha256": protocol["views"][0]["rgb"]["sha256"]}
            ],
        },
    }
    monkeypatch.setattr(matrix, "verify_bundle", lambda *a, **k: bundle)
    member = {"candidate": reference(c)}
    manifest = {
        "purpose": "quality_comparison",
        "video": {"sha256": "2" * 64},
        "evaluation": {
            "protocol": reference(p),
            "camera_evidence": reference(camera),
            "timeout_s": 30,
        },
    }
    reconstruction = {
        "bundle_path": "unused-in-controlled-test",
        "ply_sha256": model_hash,
    }
    result = matrix._quality(
        member, manifest, tmp_path / "evaluation/one", reconstruction
    )
    assert result["status"] == "MEASURED"
    assert result["feedback_status"] == "WITHHELD"
    assert result["release_authorized"] is False
    reconstruction["ply_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="not bound"):
        matrix._quality(member, manifest, tmp_path / "evaluation/two", reconstruction)


# ---------------------------------------------------------------------------
# Auto candidate mode (PR 4)
# ---------------------------------------------------------------------------


def _auto_member_fixture(inputs):
    """Build scene/view-set/protocol files plus a frozen render contract.

    Files are written next to the matrix config because render-contract
    references resolve against the config directory. The renderer and scorer
    are replaced in tests; the fixture only proves the planning/execution
    semantics of the auto candidate path.
    """
    tmp_path = inputs
    rng = np.random.default_rng(3)
    scene = tmp_path / "scene.usd"
    scene.write_text("#usda 1.0\n")
    cam = tmp_path / "cam.json"
    write_json(cam, {"camera": "fixture"})
    views = []
    for vid in ("view-a", "view-b"):
        image = tmp_path / f"{vid}.png"
        Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(image)
        views.append(
            {
                "view_id": vid,
                "resolution": {"width": 16, "height": 16},
                "intrinsics": [[16, 0, 8], [0, 16, 8], [0, 0, 1]],
                "camera_to_world": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "distortion": {"model": "none", "coefficients": []},
                "preprocessing": "none",
                "camera_evidence": {"path": cam.name, "sha256": digest(cam)},
            }
        )
    view_set = tmp_path / "view_set.json"
    write_json(
        view_set,
        {
            "schema_version": 1,
            "kind": "view_set",
            "view_set_id": "vs-fixture",
            "coordinate_frame": "world",
            "unit": "m",
            "views": views,
        },
    )
    reference_views = []
    for vid, frame in (("view-a", "frame-a"), ("view-b", "frame-b")):
        reference_views.append(
            {
                "view_id": vid,
                "asset_id": "fixture-asset",
                "frame_id": frame,
                "split": "development",
                "reference_kind": "synthetic_gt",
                "rgb": {
                    "path": f"{vid}.png",
                    "sha256": digest(tmp_path / f"{vid}.png"),
                },
                "regions": [{"asset_id": "fixture-asset", "bbox_xyxy": [0, 0, 8, 8]}],
            }
        )

    def check(name, track, metric, limit):
        return dict(
            check_id=name,
            track=track,
            metric=metric,
            op=">=",
            limit=limit,
            required=True,
            severity="critical",
            calibration="structural",
            defect_kind="appearance_mismatch",
        )

    protocol = tmp_path / "protocol.json"
    write_json(
        protocol,
        {
            "schema_version": 1,
            "benchmark_id": "auto-fixture",
            "profile": "observer_calibration",
            "phase": "development",
            "fit_frame_ids": [],
            "fit_image_sha256": [],
            "views": reference_views,
            "checks": [
                check("coverage", "coverage", "coverage.observation_fraction", 1),
                check("visual", "appearance", "appearance.psnr_db.mean", 22),
            ],
        },
    )
    return {
        "scene": scene,
        "view_set": view_set,
        "protocol": protocol,
        "render_contract": {
            "scene": {"path": scene.name, "sha256": digest(scene)},
            "view_set": {"path": view_set.name, "sha256": digest(view_set)},
            "renderer": {
                "name": "fixture-renderer",
                "version": "1",
                "code_sha256": "a" * 64,
                "environment": {"image": "fixture", "python": "3.12"},
            },
            "image_settings": {"width": 16, "height": 16, "mode": "rgb"},
            "lighting": {"policy": "frozen", "exposure": 0.0, "seed": 0},
            "seed": 7,
            "budget": {"max_wall_seconds": 60},
        },
    }


def _auto_config(path, fixture, *, with_contract=True):
    """Rewrite the prepared matrix config: auto member + quality purpose.

    ``fixture`` must have been built in ``path.parent``.
    """
    doc = read_json(path)
    doc["members"][0]["candidate_source"] = {"mode": "auto"}
    doc["purpose"] = "quality_comparison"
    doc.setdefault("evaluation", {})
    doc["evaluation"]["protocol"] = {
        "path": str(fixture["protocol"]),
        "sha256": digest(fixture["protocol"]),
    }
    doc["evaluation"]["timeout_s"] = 30
    if with_contract:
        doc["evaluation"]["render_contract"] = fixture["render_contract"]
    write_json(path, doc)
    return doc


def test_auto_mode_reference_rejected_at_planning(prepared, tmp_path):
    path, out = prepared
    fixture = _auto_member_fixture(path.parent)
    doc = _auto_config(path, fixture)
    doc["members"][0]["candidate_source"] = {
        "mode": "auto",
        "reference": {"path": "candidate.json", "sha256": "0" * 64},
    }
    write_json(path, doc)
    with pytest.raises(ValueError, match="must not bind a pre-existing reference"):
        matrix.plan_matrix(path, out)


def test_auto_mode_plans_without_preexisting_candidate(prepared, tmp_path):
    path, out = prepared
    fixture = _auto_member_fixture(path.parent)
    _auto_config(path, fixture)
    plan = matrix.plan_matrix(path, out)
    assert plan["members"][0]["candidate_mode"] == "auto"
    # No candidate may be bound at planning time in auto mode.
    assert plan["members"][0]["candidate"] is None
    # The render contract is frozen with the renderer identity.
    contract = plan["evaluation"]["render_contract"]
    assert contract["renderer"]["name"] == "fixture-renderer"
    assert Path(contract["scene"]["path"]).is_file()
    assert Path(contract["view_set"]["path"]).is_file()
    # The legacy member keeps external mode.
    assert plan["members"][1]["candidate_mode"] == "external"


def test_auto_mode_requires_render_contract(prepared, tmp_path):
    path, out = prepared
    fixture = _auto_member_fixture(path.parent)
    _auto_config(path, fixture, with_contract=False)
    with pytest.raises(
        ValueError, match="requires a frozen evaluation.render_contract"
    ):
        matrix.plan_matrix(path, out)


def _fake_renderer(scene_id):
    """Return a renderer stub writing a receipt bound to the given scene."""

    def render(contract, view_set_path, reconstruction, directory):
        from simready.core.benchmark_contract import validate_render_request

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        scene_src = Path(contract["scene"]["path"])
        ply_src = Path(reconstruction["ply_path"])
        shutil.copy2(scene_src, directory / scene_src.name)
        shutil.copy2(ply_src, directory / ply_src.name)
        shutil.copy2(view_set_path, directory / "view_set.json")
        request = {
            "schema_version": 1,
            "kind": "render_request",
            "request_id": f"auto-{directory.name}",
            "scene": {"path": scene_src.name, "sha256": digest(scene_src)},
            "reconstruction": {"path": ply_src.name, "sha256": digest(ply_src)},
            "view_set": {"path": "view_set.json", "sha256": digest(view_set_path)},
            "renderer": dict(contract["renderer"]),
            "image_settings": dict(contract["image_settings"]),
            "lighting": dict(contract["lighting"]),
            "seed": contract["seed"],
            "budget": dict(contract["budget"]),
        }
        request_path = directory / "render_request.json"
        write_json(request_path, request)
        validate_render_request(read_json(request_path), request_path.parent)
        view_set = read_json(view_set_path)
        render_dir = directory / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        views = []
        for view in view_set["views"]:
            rgb = render_dir / f"{view['view_id']}.rgb.png"
            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(rgb)
            views.append(
                {
                    "view_id": view["view_id"],
                    "width": 16,
                    "height": 16,
                    "rgb_file": rgb.name,
                    "rgb_sha256": digest(rgb),
                    "depth_file": None,
                    "depth_sha256": None,
                }
            )
        write_json(
            render_dir / "receipt.json",
            {
                "synthetic_fixture": True,
                "worker": "test-fake-renderer",
                "scene_id": scene_id or scene_src.name,
                "seed": contract["seed"],
                "want_depth": False,
                "render_settings": dict(contract["image_settings"]),
                "views": views,
                "completed": len(views),
                "expected": len(views),
                "duration_seconds": 0.0,
            },
        )
        return request_path, render_dir / "receipt.json"

    return render


def test_auto_mode_binds_candidate_after_render(prepared, tmp_path, monkeypatch):
    path, out = prepared
    fixture = _auto_member_fixture(path.parent)
    _auto_config(path, fixture)
    plan = matrix.plan_matrix(path, out)

    monkeypatch.setattr(matrix, "render_auto_observations", _fake_renderer(None))
    scoring_calls = []

    def fake_scoring(protocol, candidate, directory, timeout=300, **kwargs):
        scoring_calls.append((protocol, candidate, directory))
        scoring_dir = Path(directory)
        scoring_dir.mkdir(parents=True, exist_ok=True)
        write_json(scoring_dir / "scorecard.json", {"status": "PASS", "checks": []})
        return {
            "execution_status": "COMPLETED",
            "quality_status": "PASS",
            "defects_available": True,
            "scorecard": {"status": "PASS"},
        }

    monkeypatch.setattr(matrix, "run_scoring", fake_scoring)
    report = matrix.execute_matrix(plan)
    assert report["status"] == "COMPLETE", report
    row = report["members"][0]
    assert row["candidate_mode"] == "auto"
    assert row["quality"]["status"] == "MEASURED"
    assert row["quality"]["verifier_status"] == "PASS"
    assert row["quality"]["release_authorized"] is False
    assert len(scoring_calls) == 1
    # The candidate exists only because the render produced verifiable output.
    candidate_dir = Path(out) / "evaluation" / row["id"] / "candidate"
    assert (candidate_dir / "candidate.json").is_file()
    candidate = read_json(candidate_dir / "candidate.json")
    assert candidate["evidence_status"] == "receipt_verified"
    assert {v["view_id"] for v in candidate["views"]} == {"view-a", "view-b"}
    # Artifact references point at the real products.
    assert Path(row["artifacts"]["candidate"]).is_file()
    assert Path(row["artifacts"]["render_receipt"]).is_file()
    assert Path(row["artifacts"]["scorecard"]).is_file()
    # The member without candidate_source stays external and NOT_EVALUATED.
    assert report["members"][1]["candidate_mode"] == "external"
    assert report["members"][1]["quality"]["status"] == "NOT_EVALUATED"


def test_auto_mode_scene_mismatch_fails_closed(prepared, tmp_path, monkeypatch):
    path, out = prepared
    fixture = _auto_member_fixture(path.parent)
    _auto_config(path, fixture)
    plan = matrix.plan_matrix(path, out)
    # The fake renderer claims a different scene than the frozen contract.
    monkeypatch.setattr(
        matrix, "render_auto_observations", _fake_renderer("other-scene.usd")
    )
    monkeypatch.setattr(
        matrix,
        "run_scoring",
        lambda *a, **k: pytest.fail("Scoring must not run on unbound evidence"),
    )
    report = matrix.execute_matrix(plan)
    row = report["members"][0]
    assert row["status"] == "FAILED"
    assert "scene" in row["error"]["message"].lower()
    assert row["quality"]["status"] == "NOT_EVALUATED"
    assert report["status"] == "COMPLETED_WITH_FAILURES"
