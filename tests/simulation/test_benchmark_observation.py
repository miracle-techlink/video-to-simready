"""Benchmark observation worker: contract validation, receipt layout, failure semantics."""

import json
from pathlib import Path

import numpy as np
import pytest

from simready.core.artifacts import digest, write_json
from simready.core.benchmark_contract import (
    ContractError,
    validate_render_request,
    validate_view_set,
)


def _view_set_manifest():
    return {
        "kind": "view_set",
        "schema_version": 1,
        "view_set_id": "vs-1",
        "coordinate_frame": "world",
        "unit": "m",
        "views": [
            {
                "view_id": "front",
                "resolution": {"width": 64, "height": 64},
                "intrinsics": [[50, 0, 31.5], [0, 50, 31.5], [0, 0, 1]],
                "camera_to_world": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 5],
                    [0, 0, 0, 1],
                ],
                "distortion": {"model": "none", "coefficients": []},
                "preprocessing": "none",
                "camera_evidence": {"path": "cam_front.json", "sha256": "a" * 64},
            },
            {
                "view_id": "side",
                "resolution": {"width": 64, "height": 64},
                "intrinsics": [[50, 0, 31.5], [0, 50, 31.5], [0, 0, 1]],
                "camera_to_world": [
                    [0, -1, 0, 5],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "distortion": {"model": "none", "coefficients": []},
                "preprocessing": "none",
                "camera_evidence": {"path": "cam_side.json", "sha256": "b" * 64},
            },
        ],
    }


def _write_view_set(tmp_path, manifest=None):
    if manifest is None:
        manifest = _view_set_manifest()
    for view in manifest["views"]:
        ev_path = tmp_path / view["camera_evidence"]["path"]
        ev_path.write_text(json.dumps({"camera": view["view_id"]}))
        view["camera_evidence"]["sha256"] = digest(ev_path)
    path = tmp_path / "view_set.json"
    write_json(path, manifest)
    return path


def _write_render_request(tmp_path, **overrides):
    """Write a minimal valid render request manifest."""
    scene_file = tmp_path / "scene.usda"
    scene_file.write_text("#usda 1.0\n")
    view_set_file = _write_view_set(tmp_path)
    reconstruction_file = tmp_path / "reconstruction.json"
    reconstruction_file.write_text(json.dumps({"schema_version": 1}))

    manifest = {
        "kind": "render_request",
        "schema_version": 1,
        "request_id": "req-1",
        "scene": {"path": "scene.usda", "sha256": digest(scene_file)},
        "reconstruction": {"path": "reconstruction.json", "sha256": digest(reconstruction_file)},
        "view_set": {"path": "view_set.json", "sha256": digest(view_set_file)},
        "renderer": {
            "name": "isaac-sim",
            "version": "6.1.0",
            "code_sha256": "a" * 64,
            "environment": {
                "image": "nvcr.io/nvidia/isaac-sim:6.1.0",
                "python": "3.10",
            },
        },
        "image_settings": {"width": 64, "height": 64, "mode": "rgb"},
        "lighting": {"policy": "scene_default", "exposure": 0.0},
        "seed": 42,
        "budget": {"max_wall_seconds": 3600.0},
    }
    manifest.update(overrides)
    path = tmp_path / "render_request.json"
    write_json(path, manifest)
    return path


class TestPlanValidation:
    """plan_benchmark_observation validates the full input chain."""

    def test_valid_plan_accepted(self, tmp_path):
        from simready.simulation.benchmark_observation import plan_benchmark_observation

        req_path = _write_render_request(tmp_path)
        vs_path = tmp_path / "view_set.json"
        plan = plan_benchmark_observation(req_path, vs_path, tmp_path / "out")
        assert len(plan["views"]) == 2
        assert plan["render_request"]["request_id"] == "req-1"

    def test_invalid_render_request_rejected(self, tmp_path):
        from simready.simulation.benchmark_observation import plan_benchmark_observation

        req_path = _write_render_request(tmp_path, seed=-1)
        vs_path = tmp_path / "view_set.json"
        with pytest.raises(ContractError, match="seed"):
            plan_benchmark_observation(req_path, vs_path, tmp_path / "out")

    def test_invalid_view_set_rejected(self, tmp_path):
        from simready.simulation.benchmark_observation import plan_benchmark_observation

        req_path = _write_render_request(tmp_path)
        # Overwrite view_set with duplicate view IDs
        bad_manifest = _view_set_manifest()
        bad_manifest["views"][1]["view_id"] = "front"
        vs_path = tmp_path / "view_set.json"
        for view in bad_manifest["views"]:
            ev_path = tmp_path / view["camera_evidence"]["path"]
            ev_path.write_text(json.dumps({"camera": view["view_id"]}))
            view["camera_evidence"]["sha256"] = digest(ev_path)
        write_json(vs_path, bad_manifest)
        # Update the render_request to match the new view_set hash
        req = json.loads(req_path.read_text())
        req["view_set"]["sha256"] = digest(vs_path)
        write_json(req_path, req)
        with pytest.raises(ContractError, match="Duplicate view"):
            plan_benchmark_observation(req_path, vs_path, tmp_path / "out")

    def test_nonsquare_pixel_rejected(self, tmp_path):
        """Views with fx != fy must be rejected for benchmark use."""
        from simready.simulation.benchmark_observation import plan_benchmark_observation

        bad_manifest = _view_set_manifest()
        bad_manifest["views"][0]["intrinsics"][1][1] = 45  # fy != fx
        for view in bad_manifest["views"]:
            ev_path = tmp_path / view["camera_evidence"]["path"]
            ev_path.write_text(json.dumps({"camera": view["view_id"]}))
            view["camera_evidence"]["sha256"] = digest(ev_path)
        vs_path = tmp_path / "view_set.json"
        write_json(vs_path, bad_manifest)

        # Build render_request manually to reference the modified view_set
        scene_file = tmp_path / "scene.usda"
        scene_file.write_text("#usda 1.0\n")
        reconstruction_file = tmp_path / "reconstruction.json"
        reconstruction_file.write_text(json.dumps({"schema_version": 1}))
        req_manifest = {
            "kind": "render_request",
            "schema_version": 1,
            "request_id": "req-1",
            "scene": {"path": "scene.usda", "sha256": digest(scene_file)},
            "reconstruction": {"path": "reconstruction.json", "sha256": digest(reconstruction_file)},
            "view_set": {"path": "view_set.json", "sha256": digest(vs_path)},
            "renderer": {
                "name": "isaac-sim",
                "version": "6.1.0",
                "code_sha256": "a" * 64,
                "environment": {
                    "image": "nvcr.io/nvidia/isaac-sim:6.1.0",
                    "python": "3.10",
                },
            },
            "image_settings": {"width": 64, "height": 64, "mode": "rgb"},
            "lighting": {"policy": "scene_default", "exposure": 0.0},
            "seed": 42,
            "budget": {"max_wall_seconds": 3600.0},
        }
        req_path = tmp_path / "render_request.json"
        write_json(req_path, req_manifest)
        with pytest.raises(ValueError, match="square pixels"):
            plan_benchmark_observation(req_path, vs_path, tmp_path / "out")


class TestReceiptLayout:
    """Receipt must match the fake_render_worker layout for interchangeability."""

    def test_receipt_top_level_keys(self, tmp_path):
        from simready.simulation.benchmark_observation import _build_receipt

        receipt = _build_receipt(
            render_request={
                "seed": 42,
                "image_settings": {"width": 64, "height": 64, "mode": "rgb"},
                "scene": {"path": "scene.usda"},
            },
            view_set={"views": [{"view_id": "v0"}, {"view_id": "v1"}]},
            views_result=[
                {
                    "view_id": "v0",
                    "width": 64,
                    "height": 64,
                    "rgb_file": "v0.rgb.png",
                    "rgb_sha256": "a" * 64,
                    "depth_file": None,
                    "depth_sha256": None,
                },
                {
                    "view_id": "v1",
                    "width": 64,
                    "height": 64,
                    "rgb_file": "v1.rgb.png",
                    "rgb_sha256": "b" * 64,
                    "depth_file": None,
                    "depth_sha256": None,
                },
            ],
            started=0.0,
        )
        # Must match fake_render_worker receipt keys
        expected_keys = {
            "synthetic_fixture", "banner", "worker", "scene_id", "seed",
            "want_depth", "render_settings", "views", "completed", "expected",
            "duration_seconds",
        }
        assert expected_keys <= set(receipt.keys())
        assert receipt["synthetic_fixture"] is False
        assert receipt["completed"] == receipt["expected"] == 2

    def test_receipt_views_have_required_fields(self, tmp_path):
        from simready.simulation.benchmark_observation import _build_receipt

        receipt = _build_receipt(
            render_request={
                "seed": 0,
                "image_settings": {"width": 32, "height": 32, "mode": "rgbd"},
                "scene": {"path": "s.usda"},
            },
            view_set={"views": [{"view_id": "v0"}]},
            views_result=[
                {
                    "view_id": "v0",
                    "width": 32,
                    "height": 32,
                    "rgb_file": "v0.rgb.png",
                    "rgb_sha256": "c" * 64,
                    "depth_file": "v0.depth.npy",
                    "depth_sha256": "d" * 64,
                },
            ],
            started=0.0,
        )
        view = receipt["views"][0]
        assert view["view_id"] == "v0"
        assert view["rgb_sha256"] == "c" * 64
        assert view["depth_sha256"] == "d" * 64


class TestFailureSemantics:
    """Crash, timeout, cancel must not publish complete success receipts."""

    def test_rgb_mode_flag_matches_render_mode(self):
        """want_depth in receipt matches render_request mode."""
        from simready.simulation.benchmark_observation import _build_receipt

        for mode, expected in [("rgb", False), ("rgbd", True)]:
            receipt = _build_receipt(
                render_request={
                    "seed": 0,
                    "image_settings": {"width": 32, "height": 32, "mode": mode},
                    "scene": {"path": "s.usda"},
                },
                view_set={"views": []},
                views_result=[],
                started=0.0,
            )
            assert receipt["want_depth"] is expected

    def test_incomplete_views_reflected_in_receipt(self):
        """Missing views are recorded, not silently padded."""
        from simready.simulation.benchmark_observation import _build_receipt

        receipt = _build_receipt(
            render_request={
                "seed": 0,
                "image_settings": {"width": 32, "height": 32, "mode": "rgb"},
                "scene": {"path": "s.usda"},
            },
            view_set={"views": [{"view_id": "v0"}, {"view_id": "v1"}, {"view_id": "v2"}]},
            views_result=[
                {
                    "view_id": "v0",
                    "width": 32,
                    "height": 32,
                    "rgb_file": "v0.rgb.png",
                    "rgb_sha256": "a" * 64,
                    "depth_file": None,
                    "depth_sha256": None,
                },
            ],
            started=0.0,
        )
        assert receipt["completed"] == 1
        assert receipt["expected"] == 3
