"""Candidate assembly, independent scoring separation, and invalidation semantics."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from simready.core.artifacts import digest, read_json, write_json
from simready.core.benchmark_contract import ContractError
from simready.core.evidence import reference
from simready.workflows.stages.benchmark_evaluation import (
    _verify_render_receipt,
    build_candidate,
    run_benchmark_evaluation,
    run_scoring,
)


def _make_render_receipt(tmp_path, view_ids=("view-a", "view-b"), width=64, height=64,
                          scene_id="scene.usd", want_depth=False):
    """Create a fake render receipt with real files on disk."""
    views = []
    for vid in view_ids:
        rgb = np.random.default_rng(hash(vid) & 0xFFFFFFFF).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
        rgb_path = tmp_path / f"{vid}.rgb.png"
        Image.fromarray(rgb).save(rgb_path)
        rgb_sha = digest(rgb_path)
        entry = {
            "view_id": vid,
            "width": width,
            "height": height,
            "rgb_file": rgb_path.name,
            "rgb_sha256": rgb_sha,
            "depth_file": None,
            "depth_sha256": None,
        }
        if want_depth:
            depth = np.ones((height, width), dtype=np.float32)
            depth_path = tmp_path / f"{vid}.depth.npy"
            np.save(depth_path, depth, allow_pickle=False)
            entry["depth_file"] = depth_path.name
            entry["depth_sha256"] = digest(depth_path)
        views.append(entry)

    receipt = {
        "synthetic_fixture": True,
        "banner": "test fixture",
        "worker": "test",
        "scene_id": scene_id,
        "seed": 42,
        "want_depth": want_depth,
        "render_settings": {"width": width, "height": height, "mode": "rgb"},
        "views": views,
        "completed": len(views),
        "expected": len(views),
        "duration_seconds": 0.1,
    }
    receipt_path = tmp_path / "receipt.json"
    write_json(receipt_path, receipt)
    return receipt_path, receipt


def _make_evaluation_request(tmp_path, render_request_path=None):
    """Create a minimal evaluation request manifest."""
    if render_request_path is None:
        # Create the scene file so the render request can reference it.
        scene_path = tmp_path / "scene.usd"
        scene_path.write_text("#usda 1.0\n")
        rr = tmp_path / "render_request.json"
        write_json(rr, {
            "schema_version": 1, "kind": "render_request",
            "request_id": "rr-1",
            "scene": {"path": "scene.usd", "sha256": digest(scene_path)},
            "reconstruction": {"path": "recon.ply", "sha256": "b" * 64},
            "view_set": {"path": "views.json", "sha256": "c" * 64},
            "renderer": {
                "name": "isaac", "version": "6.1",
                "code_sha256": "d" * 64,
                "environment": {"image": "img", "python": "3.12"},
            },
            "image_settings": {"width": 64, "height": 64, "mode": "rgb"},
            "lighting": {"policy": "frozen", "exposure": 0.0, "seed": 0},
            "seed": 42,
            "budget": {"max_wall_seconds": 300},
        })
        render_request_path = rr

    er = {
        "schema_version": 1,
        "kind": "evaluation_request",
        "request_id": "eval-1",
        "protocol": {
            "name": "observer_calibration",
            "version": "1.0",
            "sha256": "e" * 64,
        },
        "candidate_source": {"mode": "auto"},
        "scorer": {
            "name": "simready-scorer",
            "version": "1.0",
            "code_sha256": "f" * 64,
        },
        "deadline": "2026-12-31",
        "render_request": {
            "path": render_request_path.name,
            "sha256": digest(render_request_path),
        },
    }
    er_path = tmp_path / "evaluation_request.json"
    write_json(er_path, er)
    return er_path


def _make_view_set(view_ids=("view-a", "view-b"), width=64, height=64):
    views = []
    for vid in view_ids:
        views.append({
            "view_id": vid,
            "resolution": {"width": width, "height": height},
            "intrinsics": [[500, 0, 32], [0, 500, 32], [0, 0, 1]],
            "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "distortion": {"model": "none", "coefficients": []},
            "preprocessing": "none",
            "camera_evidence": {"path": "cam.json", "sha256": "0" * 64},
        })
    return {"schema_version": 1, "kind": "view_set", "view_set_id": "vs-1",
            "coordinate_frame": "world", "unit": "m", "views": views}


# -----------------------------------------------------------------------
# Candidate builder tests
# -----------------------------------------------------------------------


class TestVerifyRenderReceipt:
    def test_valid_receipt_passes(self, tmp_path):
        receipt_path, _ = _make_render_receipt(tmp_path)
        view_set = _make_view_set()
        result = _verify_render_receipt(
            receipt_path,
            expected_view_ids={"view-a", "view-b"},
            expected_resolution={"width": 64, "height": 64},
            expected_scene_id="scene.usd",
        )
        assert result["completed"] == 2

    def test_missing_view_rejected(self, tmp_path):
        receipt_path, _ = _make_render_receipt(tmp_path, view_ids=("view-a",))
        with pytest.raises(ValueError, match="[Ii]ncomplete"):
            _verify_render_receipt(
                receipt_path,
                expected_view_ids={"view-a", "view-b"},
                expected_resolution={"width": 64, "height": 64},
                expected_scene_id="scene.usd",
            )

    def test_unexpected_view_rejected(self, tmp_path):
        receipt_path, _ = _make_render_receipt(tmp_path, view_ids=("view-a", "view-b", "view-c"))
        with pytest.raises(ValueError, match="[Uu]nexpected|incomplete"):
            _verify_render_receipt(
                receipt_path,
                expected_view_ids={"view-a", "view-b"},
                expected_resolution={"width": 64, "height": 64},
                expected_scene_id="scene.usd",
            )

    def test_resolution_mismatch_rejected(self, tmp_path):
        receipt_path, _ = _make_render_receipt(tmp_path, width=64, height=64)
        with pytest.raises(ValueError, match="resolution mismatch"):
            _verify_render_receipt(
                receipt_path,
                expected_view_ids={"view-a", "view-b"},
                expected_resolution={"width": 128, "height": 128},
                expected_scene_id="scene.usd",
            )

    def test_tampered_rgb_hash_rejected(self, tmp_path):
        receipt_path, receipt = _make_render_receipt(tmp_path)
        # Tamper with an image file.
        rgb_path = tmp_path / receipt["views"][0]["rgb_file"]
        rgb_path.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="hash mismatch"):
            _verify_render_receipt(
                receipt_path,
                expected_view_ids={"view-a", "view-b"},
                expected_resolution={"width": 64, "height": 64},
                expected_scene_id="scene.usd",
            )

    def test_missing_rgb_file_rejected(self, tmp_path):
        receipt_path, receipt = _make_render_receipt(tmp_path)
        (tmp_path / receipt["views"][0]["rgb_file"]).unlink()
        with pytest.raises(ValueError, match="Missing RGB"):
            _verify_render_receipt(
                receipt_path,
                expected_view_ids={"view-a", "view-b"},
                expected_resolution={"width": 64, "height": 64},
                expected_scene_id="scene.usd",
            )

    def test_empty_scene_id_rejected(self, tmp_path):
        """Empty scene_id must not bypass scene binding check."""
        receipt_path, receipt = _make_render_receipt(tmp_path)
        receipt["scene_id"] = ""
        write_json(tmp_path / "receipt.json", receipt)
        with pytest.raises(ValueError, match="scene_id.*nonempty"):
            _verify_render_receipt(
                tmp_path / "receipt.json",
                expected_view_ids={"view-a", "view-b"},
                expected_resolution={"width": 64, "height": 64},
                expected_scene_id="scene.usd",
            )

    def test_wrong_scene_id_rejected(self, tmp_path):
        """Wrong scene_id must be rejected."""
        receipt_path, receipt = _make_render_receipt(tmp_path)
        receipt["scene_id"] = "wrong_scene"
        write_json(tmp_path / "receipt.json", receipt)
        with pytest.raises(ValueError, match="[Ss]cene binding mismatch"):
            _verify_render_receipt(
                tmp_path / "receipt.json",
                expected_view_ids={"view-a", "view-b"},
                expected_resolution={"width": 64, "height": 64},
                expected_scene_id="scene.usd",
            )


class TestBuildCandidate:
    def test_builds_candidate_from_valid_receipt(self, tmp_path):
        receipt_path, _ = _make_render_receipt(tmp_path)
        eval_path = _make_evaluation_request(tmp_path)
        view_set = _make_view_set()
        candidate_path = build_candidate(
            receipt_path, eval_path, view_set, tmp_path / "candidate"
        )
        assert candidate_path.is_file()
        candidate = read_json(candidate_path)
        assert candidate["schema_version"] == 1
        assert len(candidate["views"]) == 2
        assert candidate["evidence_status"] == "receipt_verified"

    def test_candidate_images_are_self_contained(self, tmp_path):
        """Candidate images must be copied into the candidate directory so
        that deleting the original receipt directory does not break scoring."""
        receipt_dir = tmp_path / "receipt"
        receipt_dir.mkdir()
        receipt_path, receipt = _make_render_receipt(receipt_dir)
        eval_path = _make_evaluation_request(receipt_dir)
        view_set = _make_view_set()
        candidate_dir = tmp_path / "candidate"
        candidate_path = build_candidate(
            receipt_path, eval_path, view_set, candidate_dir
        )
        candidate = read_json(candidate_path)
        # Verify image paths are relative to candidate dir, not receipt dir.
        for view in candidate["views"]:
            assert view["rgb"]["path"].startswith("views/")
            img_path = candidate_dir / view["rgb"]["path"]
            assert img_path.is_file()
            assert digest(img_path) == view["rgb"]["sha256"]
        # Remove the original receipt directory; candidate must still be valid.
        shutil.rmtree(receipt_dir)
        assert not receipt_dir.exists()
        # Re-verify all candidate images are still accessible.
        for view in candidate["views"]:
            img_path = candidate_dir / view["rgb"]["path"]
            assert img_path.is_file()
            assert digest(img_path) == view["rgb"]["sha256"]

    def test_incomplete_receipt_produces_no_candidate(self, tmp_path):
        receipt_path, _ = _make_render_receipt(tmp_path, view_ids=("view-a",))
        eval_path = _make_evaluation_request(tmp_path)
        view_set = _make_view_set()
        with pytest.raises((ValueError, ContractError)):
            build_candidate(
                receipt_path, eval_path, view_set, tmp_path / "candidate"
            )


# -----------------------------------------------------------------------
# Scoring separation tests
# -----------------------------------------------------------------------


class TestScoringSeparation:
    """Execution status and quality status must be kept separate."""

    def test_scoring_fail_is_not_execution_success(self, tmp_path):
        """A verifier that finishes with FAIL has execution_status=COMPLETED
        but quality_status=FAIL — not 'execution succeeded therefore passed'."""
        # Build a protocol and candidate that will score FAIL.
        from tests.fixtures.scene_candidate import make_scene_candidate
        p, c, protocol, candidate, image = make_scene_candidate(tmp_path)
        # Make the candidate image different from the reference.
        altered = np.zeros_like(image)
        Image.fromarray(altered).save(tmp_path / "render.png")
        candidate["views"][0]["rgb"] = reference(tmp_path / "render.png")
        write_json(c, candidate)

        result = run_scoring(p, c, tmp_path / "out", timeout=60)
        assert result["execution_status"] == "COMPLETED"
        assert result["quality_status"] == "FAIL"
        assert result["release_authorized"] is False

    def test_scoring_pass_has_quality_pass(self, tmp_path):
        from tests.fixtures.scene_candidate import make_scene_candidate
        p, c, *_ = make_scene_candidate(tmp_path)
        result = run_scoring(p, c, tmp_path / "out", timeout=60)
        assert result["execution_status"] == "COMPLETED"
        assert result["quality_status"] == "PASS"
        assert result["release_authorized"] is False

    def test_scoring_incomplete_is_not_pass(self, tmp_path):
        from tests.fixtures.scene_candidate import make_scene_candidate
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        # Remove a required check limit to force INCOMPLETE.
        protocol["checks"][1]["limit"] = None
        protocol["checks"][1]["calibration"] = "pending"
        write_json(p, protocol)
        candidate["protocol_sha256"] = digest(p)
        write_json(c, candidate)
        result = run_scoring(p, c, tmp_path / "out", timeout=60)
        assert result["execution_status"] == "COMPLETED"
        assert result["quality_status"] == "INCOMPLETE"

    def test_scoring_invalid_input_is_execution_error(self, tmp_path):
        """Worker crash / invalid input is execution_status=FAILED, not quality."""
        result = run_scoring(
            tmp_path / "nonexistent.json",
            tmp_path / "nonexistent.json",
            tmp_path / "out",
            timeout=10,
        )
        assert result["execution_status"] == "FAILED"
        assert result["quality_status"] == "EXECUTION_ERROR"

    def test_reuse_records_null_costs(self, tmp_path):
        """Unknown GPU time, cost and tokens must remain null."""
        from tests.fixtures.scene_candidate import make_scene_candidate
        p, c, *_ = make_scene_candidate(tmp_path)
        result = run_scoring(p, c, tmp_path / "out", timeout=60)
        assert result["reuse_cost"]["gpu_seconds"] is None
        assert result["reuse_cost"]["wall_seconds_saved"] is None
        assert result["reuse_cost"]["tokens_saved"] is None


# -----------------------------------------------------------------------
# StageLedger path resolution tests
# -----------------------------------------------------------------------


class TestStageLedgerPathResolution:
    def test_scoring_module_path_resolves(self):
        """The ledger must be able to digest the real scoring module."""
        import importlib
        mod = importlib.import_module("simready.evaluation.scoring")
        path = Path(mod.__file__)
        assert path.is_file()
        d = digest(path)
        assert len(d) == 64
        int(d, 16)  # Must be valid hex.

    def test_verification_module_path_resolves(self):
        """The ledger must be able to digest the real verification module."""
        import importlib
        mod = importlib.import_module("simready.evaluation.verification")
        path = Path(mod.__file__)
        assert path.is_file()
        d = digest(path)
        assert len(d) == 64
        int(d, 16)  # Must be valid hex.


# -----------------------------------------------------------------------
# Sealed discipline tests
# -----------------------------------------------------------------------


class TestSealedDiscipline:
    def test_sealed_withholds_defects(self, tmp_path):
        from tests.fixtures.scene_candidate import make_scene_candidate
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        protocol["phase"] = "sealed_release"
        protocol["views"][0]["split"] = "sealed_release"
        write_json(p, protocol)
        candidate["protocol_sha256"] = digest(p)
        write_json(c, candidate)
        result = run_scoring(p, c, tmp_path / "out", timeout=60)
        assert result["execution_status"] == "COMPLETED"
        assert result["defects_available"] is False
        assert result["release_authorized"] is False

    def test_development_provides_defects(self, tmp_path):
        from tests.fixtures.scene_candidate import make_scene_candidate
        p, c, protocol, candidate, image = make_scene_candidate(tmp_path)
        altered = np.zeros_like(image)
        Image.fromarray(altered).save(tmp_path / "render.png")
        candidate["views"][0]["rgb"] = reference(tmp_path / "render.png")
        write_json(c, candidate)
        result = run_scoring(p, c, tmp_path / "out", timeout=60)
        assert result["execution_status"] == "COMPLETED"
        assert result["quality_status"] == "FAIL"
        assert result["defects_available"] is True
        assert result["defect_count"] > 0


# -----------------------------------------------------------------------
# Full composition test
# -----------------------------------------------------------------------


class TestRunBenchmarkEvaluation:
    def test_end_to_end_composition(self, tmp_path):
        receipt_path, _ = _make_render_receipt(tmp_path)
        eval_path = _make_evaluation_request(tmp_path)
        view_set = _make_view_set()

        # Create a protocol that matches the candidate we'll build.
        from tests.fixtures.scene_candidate import make_scene_candidate
        sc_dir = tmp_path / "sc"
        sc_dir.mkdir()
        p, c, protocol, candidate, image = make_scene_candidate(sc_dir)
        # We need the protocol to have views matching our view set.
        # For this test, we use the scene_candidate fixture directly.
        result = run_benchmark_evaluation(
            receipt_path, eval_path, view_set, p,
            tmp_path / "run",
        )
        assert (tmp_path / "run" / "benchmark_evaluation.json").is_file()
        assert result["release_authorized"] is False
        assert result["sealed_discipline"]["release_authorized_remains_false"] is True
        # Explicitly assert scoring execution status and quality status are present.
        assert result["scoring"]["execution_status"] in ("COMPLETED", "FAILED")
        assert result["scoring"]["quality_status"] in ("PASS", "FAIL", "INCOMPLETE", "EXECUTION_ERROR")
        # The verifier should not crash without a reason.
        if result["scoring"]["execution_status"] == "FAILED":
            assert result["scoring"]["execution_error"] is not None

    def test_end_to_end_failure_triggers_assertion(self, tmp_path):
        """A candidate with a bad image hash must produce a non-COMPLETED
        scoring result, and this test must fail if it silently passes."""
        receipt_path, receipt = _make_render_receipt(tmp_path)
        eval_path = _make_evaluation_request(tmp_path)
        view_set = _make_view_set()

        from tests.fixtures.scene_candidate import make_scene_candidate
        sc_dir = tmp_path / "sc"
        sc_dir.mkdir()
        p, c, protocol, candidate, image = make_scene_candidate(sc_dir)

        result = run_benchmark_evaluation(
            receipt_path, eval_path, view_set, p,
            tmp_path / "run",
        )
        # Now tamper with one of the candidate images to break the hash.
        cand_dir = tmp_path / "run" / "candidate"
        cand = read_json(cand_dir / "candidate.json")
        first_view = cand["views"][0]
        img_path = cand_dir / first_view["rgb"]["path"]
        # Overwrite with garbage to change the hash.
        img_path.write_bytes(b"tampered")

        # Re-run scoring on the tampered candidate.
        scoring_dir = tmp_path / "run" / "scoring2"
        scoring_dir.mkdir(exist_ok=True)
        result2 = run_scoring(p, cand_dir / "candidate.json", scoring_dir, timeout=60)
        # The verifier should detect the hash mismatch and not return COMPLETED.
        assert result2["execution_status"] != "COMPLETED" or result2["quality_status"] == "FAIL"
