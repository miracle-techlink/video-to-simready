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


# ---------------------------------------------------------------------------
# render_auto_observations bundle completeness (F-GPU-5)
# ---------------------------------------------------------------------------


def _bundle_view_set(root, *, evidence_dir="manifests"):
    """View set whose camera evidence lives in a subdirectory, like real runs."""
    evidence = root / evidence_dir
    evidence.mkdir(parents=True, exist_ok=True)
    views = []
    for vid in ("view-a", "view-b"):
        cam = evidence / f"cam-{vid}.json"
        write_json(cam, {"camera": vid})
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
                "camera_evidence": {
                    "path": f"{evidence_dir}/cam-{vid}.json",
                    "sha256": digest(cam),
                },
            }
        )
    view_set = root / "view_set.json"
    write_json(
        view_set,
        {
            "schema_version": 1,
            "kind": "view_set",
            "view_set_id": "vs-bundle",
            "coordinate_frame": "world",
            "unit": "m",
            "views": views,
        },
    )
    return view_set


def _bundle_contract(scene):
    return {
        "scene": {"path": str(scene), "sha256": digest(scene)},
        "renderer": {
            "name": "isaac-sim",
            "version": "6.1.0",
            "code_sha256": "a" * 64,
            "environment": {"image": "fixture", "python": "3.12"},
        },
        "image_settings": {"width": 16, "height": 16, "mode": "rgb"},
        "lighting": {"policy": "scene_default", "exposure": 0.0},
        "seed": 7,
        "budget": {"max_wall_seconds": 60},
    }


def _fake_gpu_execution(render_dir):
    """Replace only the GPU execution boundary; planning stays production."""

    def _run(plan):
        render = Path(plan["output_dir"])
        render.mkdir(parents=True, exist_ok=True)
        views = []
        for view in plan["views"]:
            image = render / f"{view['view_id']}.rgb.png"
            Image.new("RGB", (16, 16), color=(4, 5, 6)).save(image)
            views.append(
                {
                    "view_id": view["view_id"],
                    "width": 16,
                    "height": 16,
                    "rgb_file": image.name,
                    "rgb_sha256": digest(image),
                    "depth_file": None,
                    "depth_sha256": None,
                }
            )
        write_json(
            render / "receipt.json",
            {
                "synthetic_fixture": False,
                "worker": "gpu-boundary-stub",
                "scene_id": "bundle-test",
                "seed": 7,
                "want_depth": False,
                "render_settings": {"width": 16, "height": 16, "mode": "rgb"},
                "views": views,
                "completed": len(views),
                "expected": len(views),
                "duration_seconds": 0.0,
            },
        )
        return {"returncode": 0}

    return _run


class TestRenderAutoBundle:
    """The auto-render bundle must satisfy the frozen contracts it carries."""

    def test_real_view_set_passes_production_planning(self, tmp_path, monkeypatch):
        """Evidence in subdirs must be bundled so plan_benchmark_observation
        succeeds on real contract inputs (the live F-GPU-5 failure)."""
        scene = tmp_path / "scene.usda"
        scene.write_text("#usda 1.0\n")
        ply = tmp_path / "recon.ply"
        ply.write_text("ply\n")
        view_set = _bundle_view_set(tmp_path)
        reconstruction = {"ply_path": str(ply), "ply_sha256": digest(ply)}
        monkeypatch.setattr(
            "simready.simulation.benchmark_observation.run_benchmark_observation",
            _fake_gpu_execution(tmp_path / "render"),
        )
        request_path, receipt_path = matrix.render_auto_observations(
            _bundle_contract(scene),
            view_set,
            reconstruction,
            tmp_path / "render-input",
        )
        bundle = tmp_path / "render-input"
        # Production planning (validate_render_request + validate_view_set +
        # plan_benchmark_observation) ran against the bundle and passed.
        assert request_path.is_file()
        assert receipt_path.is_file()
        for vid in ("view-a", "view-b"):
            bundled = bundle / "manifests" / f"cam-{vid}.json"
            assert bundled.is_file()
            assert digest(bundled) == digest(tmp_path / "manifests" / f"cam-{vid}.json")
        # The frozen view set bytes are carried unchanged.
        assert digest(bundle / "view_set.json") == digest(view_set)
        request = read_json(request_path)
        assert request["view_set"]["sha256"] == digest(view_set)

    def test_tampered_evidence_rejected_by_planning(self, tmp_path, monkeypatch):
        """Evidence hash drift must fail validation, not be waved through."""
        scene = tmp_path / "scene.usda"
        scene.write_text("#usda 1.0\n")
        ply = tmp_path / "recon.ply"
        ply.write_text("ply\n")
        view_set = _bundle_view_set(tmp_path)
        (tmp_path / "manifests" / "cam-view-a.json").write_text(
            '{"camera": "tampered"}'
        )
        reconstruction = {"ply_path": str(ply), "ply_sha256": digest(ply)}
        monkeypatch.setattr(
            "simready.simulation.benchmark_observation.run_benchmark_observation",
            _fake_gpu_execution(tmp_path / "render"),
        )
        from simready.core.benchmark_contract import ContractError

        with pytest.raises(ContractError, match="camera evidence"):
            matrix.render_auto_observations(
                _bundle_contract(scene),
                view_set,
                reconstruction,
                tmp_path / "render-input",
            )

    def test_missing_evidence_rejected(self, tmp_path, monkeypatch):
        """A view set whose evidence file is absent cannot be bundled."""
        scene = tmp_path / "scene.usda"
        scene.write_text("#usda 1.0\n")
        ply = tmp_path / "recon.ply"
        ply.write_text("ply\n")
        view_set = _bundle_view_set(tmp_path)
        (tmp_path / "manifests" / "cam-view-b.json").unlink()
        reconstruction = {"ply_path": str(ply), "ply_sha256": digest(ply)}
        monkeypatch.setattr(
            "simready.simulation.benchmark_observation.run_benchmark_observation",
            _fake_gpu_execution(tmp_path / "render"),
        )
        with pytest.raises(FileNotFoundError):
            matrix.render_auto_observations(
                _bundle_contract(scene),
                view_set,
                reconstruction,
                tmp_path / "render-input",
            )

    def test_scene_usd_dependencies_carried_and_resolve(self, tmp_path, monkeypatch):
        """A sublayer reference must still resolve inside the bundle."""
        pytest.importorskip("pxr")
        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        sublayer = scene_dir / "extra_layer.usda"
        sublayer.write_text('#usda 1.0\n\ndef Scope "Extra"\n{\n}\n')
        scene = scene_dir / "scene.usda"
        scene.write_text(
            "#usda 1.0\n"
            "(\n    subLayers = [@./extra_layer.usda@]\n"
            ')\n\ndef Scope "Root"\n{\n}\n'
        )
        ply = tmp_path / "recon.ply"
        ply.write_text("ply\n")
        view_set = _bundle_view_set(tmp_path)
        reconstruction = {"ply_path": str(ply), "ply_sha256": digest(ply)}
        monkeypatch.setattr(
            "simready.simulation.benchmark_observation.run_benchmark_observation",
            _fake_gpu_execution(tmp_path / "render"),
        )
        matrix.render_auto_observations(
            _bundle_contract(scene),
            view_set,
            reconstruction,
            tmp_path / "render-input",
        )
        bundle = tmp_path / "render-input"
        assert (bundle / "scene.usda").is_file()
        assert (bundle / "extra_layer.usda").is_file()
        from pxr import UsdUtils

        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(bundle / "scene.usda"))
        assert not unresolved

    def test_unresolved_scene_reference_rejected_before_bundling(
        self, tmp_path, monkeypatch
    ):
        """A scene with dangling references must not enter a valid-looking bundle."""
        pytest.importorskip("pxr")
        scene = tmp_path / "scene.usda"
        scene.write_text(
            "#usda 1.0\n"
            "(\n    subLayers = [@./missing_layer.usda@]\n"
            ')\n\ndef Scope "Root"\n{\n}\n'
        )
        ply = tmp_path / "recon.ply"
        ply.write_text("ply\n")
        view_set = _bundle_view_set(tmp_path)
        reconstruction = {"ply_path": str(ply), "ply_sha256": digest(ply)}
        monkeypatch.setattr(
            "simready.simulation.benchmark_observation.run_benchmark_observation",
            _fake_gpu_execution(tmp_path / "render"),
        )
        with pytest.raises(ValueError, match="unresolved"):
            matrix.render_auto_observations(
                _bundle_contract(scene),
                view_set,
                reconstruction,
                tmp_path / "render-input",
            )

    def test_bundle_name_collision_is_fatal(self, tmp_path, monkeypatch):
        """Two different files claiming one bundle path must not overwrite."""
        pytest.importorskip("pxr")
        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        (scene_dir / "dep").mkdir()
        # Scene USD dependency at bundle-relative path dep/extra.usda.
        (scene_dir / "dep" / "extra.usda").write_text(
            '#usda 1.0\n\ndef Scope "Extra"\n{\n}\n'
        )
        scene = scene_dir / "scene.usda"
        scene.write_text(
            "#usda 1.0\n"
            "(\n    subLayers = [@./dep/extra.usda@]\n"
            ')\n\ndef Scope "Root"\n{\n}\n'
        )
        ply = tmp_path / "recon.ply"
        ply.write_text("ply\n")
        view_set = _bundle_view_set(tmp_path)
        # Camera evidence claims the SAME bundle-relative path (dep/extra.usda)
        # as the scene dependency but with different bytes: fatal, never a
        # silent overwrite of either frozen input.
        conflicting = tmp_path / "dep" / "extra.usda"
        conflicting.parent.mkdir(exist_ok=True)
        conflicting.write_text('{"camera": "different-bytes-than-scene-dep"}')
        manifest = read_json(view_set)
        manifest["views"] = manifest["views"][:1]
        manifest["views"][0]["camera_evidence"] = {
            "path": "dep/extra.usda",
            "sha256": digest(conflicting),
        }
        write_json(view_set, manifest)
        reconstruction = {"ply_path": str(ply), "ply_sha256": digest(ply)}
        monkeypatch.setattr(
            "simready.simulation.benchmark_observation.run_benchmark_observation",
            _fake_gpu_execution(tmp_path / "render"),
        )
        with pytest.raises(ValueError, match="collision"):
            matrix.render_auto_observations(
                _bundle_contract(scene),
                view_set,
                reconstruction,
                tmp_path / "render-input",
            )

    def test_scene_texture_asset_carried_and_hash_preserved(
        self, tmp_path, monkeypatch
    ):
        """A relative texture asset (non-layer) must travel with the bundle."""
        pytest.importorskip("pxr")
        scene_dir = tmp_path / "scene"
        (scene_dir / "textures").mkdir(parents=True)
        texture = scene_dir / "textures" / "albedo.png"
        texture.write_bytes(b"\x89PNG-fake-albedo")
        scene = scene_dir / "scene.usda"
        scene.write_text(
            "#usda 1.0\n"
            "\n"
            'def Scope "Root"\n'
            "{\n"
            '    def Mesh "M"\n'
            "    {\n"
            "        asset inputs:texture = @./textures/albedo.png@\n"
            "    }\n"
            "}\n"
        )
        ply = tmp_path / "recon.ply"
        ply.write_text("ply\n")
        view_set = _bundle_view_set(tmp_path)
        reconstruction = {"ply_path": str(ply), "ply_sha256": digest(ply)}
        monkeypatch.setattr(
            "simready.simulation.benchmark_observation.run_benchmark_observation",
            _fake_gpu_execution(tmp_path / "render"),
        )
        matrix.render_auto_observations(
            _bundle_contract(scene),
            view_set,
            reconstruction,
            tmp_path / "render-input",
        )
        bundle = tmp_path / "render-input"
        bundled_texture = bundle / "textures" / "albedo.png"
        assert bundled_texture.is_file()
        assert digest(bundled_texture) == digest(texture)
        from pxr import UsdUtils

        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(bundle / "scene.usda"))
        assert not unresolved

    def test_out_of_tree_reference_rejected_despite_file_existing(
        self, tmp_path, monkeypatch
    ):
        """An absolute/escaping reference must be rejected even though the
        file still exists at its original location: reachability there never
        makes the bundle self-contained."""
        pytest.importorskip("pxr")
        (tmp_path / "shared").mkdir()
        (tmp_path / "shared" / "tex.png").write_bytes(b"\x89PNG-shared")
        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        scene = scene_dir / "scene.usda"
        scene.write_text(
            "#usda 1.0\n"
            "\n"
            'def Scope "Root"\n'
            "{\n"
            '    def Mesh "M"\n'
            "    {\n"
            "        asset inputs:texture = @../shared/tex.png@\n"
            "    }\n"
            "}\n"
        )
        ply = tmp_path / "recon.ply"
        ply.write_text("ply\n")
        view_set = _bundle_view_set(tmp_path)
        reconstruction = {"ply_path": str(ply), "ply_sha256": digest(ply)}
        monkeypatch.setattr(
            "simready.simulation.benchmark_observation.run_benchmark_observation",
            _fake_gpu_execution(tmp_path / "render"),
        )
        with pytest.raises(ValueError, match="outside its own directory"):
            matrix.render_auto_observations(
                _bundle_contract(scene),
                view_set,
                reconstruction,
                tmp_path / "render-input",
            )
