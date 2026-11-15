"""Observation computation: metric correctness, view registration, missing views."""

import numpy as np
import pytest
from PIL import Image

from simready.core.artifacts import digest, write_json
from simready.core.evidence import reference
from simready.evaluation.observations import (
    depth_metrics,
    image_metrics,
    instance_metrics,
    observe,
)
from tests.fixtures.scene_candidate import make_scene_candidate


class TestImageMetrics:
    def test_identical_images(self):
        img = np.random.default_rng(42).random((32, 32, 3))
        result = image_metrics(img, img)
        assert result["mse"] == 0.0
        assert result["psnr_db"] > 100  # Very high PSNR for identical images
        assert abs(result["ssim"] - 1.0) < 1e-6

    def test_completely_different_images(self):
        a = np.zeros((32, 32, 3))
        b = np.ones((32, 32, 3))
        result = image_metrics(a, b)
        assert result["mse"] == 1.0
        assert result["psnr_db"] == 0.0

    def test_shape_mismatch_rejected(self):
        a = np.zeros((32, 32, 3))
        b = np.zeros((64, 64, 3))
        with pytest.raises(ValueError, match="[Mm]ismatch"):
            image_metrics(a, b)

    def test_too_small_rejected(self):
        a = np.zeros((5, 5, 3))
        with pytest.raises(ValueError, match="too small"):
            image_metrics(a, a)

    def test_nonfinite_rejected(self):
        a = np.full((32, 32, 3), np.nan)
        with pytest.raises(ValueError, match="[Nn]onfinite"):
            image_metrics(a, a)


class TestDepthMetrics:
    def test_perfect_depth(self):
        depth = np.ones((16, 16))
        result = depth_metrics(depth, depth)
        assert result["valid_fraction"] == 1.0
        assert result["mae_m"] == 0.0

    def test_partial_coverage(self):
        expected = np.ones((16, 16))
        actual = np.ones((16, 16))
        actual[:8] = 0  # Top half invalid
        result = depth_metrics(expected, actual)
        assert result["valid_fraction"] == 0.5
        assert result["mae_m"] == 0.0

    def test_no_valid_reference_rejected(self):
        with pytest.raises(ValueError, match="[Nn]o valid"):
            depth_metrics(np.zeros((16, 16)), np.ones((16, 16)))

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="[Ss]hape"):
            depth_metrics(np.ones((8, 8)), np.ones((16, 16)))


class TestInstanceMetrics:
    def test_perfect_match(self):
        ref = [
            {"instance_id": "a", "center_px": [10, 10], "category": "button"},
            {"instance_id": "b", "center_px": [20, 20], "category": "lamp"},
        ]
        result = instance_metrics(ref, ref)
        assert result["missing_count"] == 0
        assert result["extra_count"] == 0
        assert result["wrong_category_count"] == 0

    def test_missing_instance(self):
        ref = [
            {"instance_id": "a", "center_px": [10, 10], "category": "button"},
            {"instance_id": "b", "center_px": [20, 20], "category": "lamp"},
        ]
        pred = [ref[0]]
        result = instance_metrics(ref, pred)
        assert result["missing_count"] == 1
        assert "b" in result["missing_ids"]

    def test_extra_instance(self):
        ref = [{"instance_id": "a", "center_px": [10, 10], "category": "button"}]
        pred = ref + [{"instance_id": "x", "center_px": [30, 30], "category": "button"}]
        result = instance_metrics(ref, pred)
        assert result["extra_count"] == 1

    def test_wrong_category(self):
        ref = [{"instance_id": "a", "center_px": [10, 10], "category": "button"}]
        pred = [{"instance_id": "a", "center_px": [10, 10], "category": "lamp"}]
        result = instance_metrics(ref, pred)
        assert result["wrong_category_count"] == 1

    def test_critical_missing_counted(self):
        ref = [
            {"instance_id": "a", "center_px": [10, 10], "category": "button", "critical": True},
        ]
        result = instance_metrics(ref, [])
        assert result["missing_critical_count"] == 1

    def test_empty_reference_rejected(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            instance_metrics([], [])

    def test_duplicate_ids_rejected(self):
        ref = [
            {"instance_id": "a", "center_px": [10, 10], "category": "button"},
            {"instance_id": "a", "center_px": [20, 20], "category": "button"},
        ]
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            instance_metrics(ref, ref)


class TestObserve:
    def test_missing_view_recorded_as_issue(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        candidate["views"] = []  # Remove all views
        write_json(c, candidate)
        result = observe(
            protocol, candidate, p.parent, c.parent
        )
        assert any(i["code"] == "MISSING_VIEW" for i in result["issues"])
        assert result["metrics"]["coverage.observation_fraction"] == 0.0

    def test_view_id_uniqueness_in_protocol(self, tmp_path):
        """Protocol validation ensures view IDs are unique; observe trusts this."""
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        # Duplicate view_id would be caught by validate_protocol.
        # Here we verify observe works correctly with the valid fixture.
        result = observe(protocol, candidate, p.parent, c.parent)
        assert result["metrics"]["coverage.observation_fraction"] > 0

    def test_full_frame_and_region_metrics(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        result = observe(protocol, candidate, p.parent, c.parent)
        # Should have full-frame + region results.
        full_frames = [r for r in result["regions"] if r["full_frame"]]
        regions = [r for r in result["regions"] if not r["full_frame"]]
        assert len(full_frames) == 1  # One view
        assert len(regions) == 1  # One ROI

    def test_appearance_metrics_computed(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        result = observe(protocol, candidate, p.parent, c.parent)
        assert "appearance.psnr_db.mean" in result["metrics"]
        assert "appearance.ssim.mean" in result["metrics"]

    def test_input_files_tracked(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        result = observe(protocol, candidate, p.parent, c.parent)
        paths = [f["path"] for f in result["input_files"]]
        assert any("source.png" in p for p in paths)
        assert any("render.png" in p for p in paths)
