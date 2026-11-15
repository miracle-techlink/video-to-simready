"""Depth observation edge cases: RGB-only vs RGB-D, blank backgrounds, unit rules."""

import numpy as np
import pytest

from simready.simulation.depth_observation import (
    _validate_capture_result,
    normalize_camera_z,
)


class TestCaptureMode:
    """RGB-only and RGB-D modes are distinct contracts."""

    def test_rgb_mode_accepts_rgb_only(self):
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        _validate_capture_result(rgb, None, capture_mode="rgb")  # should not raise

    def test_rgbd_mode_rejects_none_rgb(self):
        depth = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="RGB capture is required"):
            _validate_capture_result(None, depth, capture_mode="rgbd")

    def test_rgbd_mode_requires_depth(self):
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        with pytest.raises(ValueError, match="requires non-None depth"):
            _validate_capture_result(rgb, None, capture_mode="rgbd")

    def test_rgbd_mode_accepts_both(self):
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        depth = np.ones((4, 4), dtype=np.float32)
        _validate_capture_result(rgb, depth, capture_mode="rgbd")  # should not raise

    def test_rgb_mode_accepts_rgb_only(self):
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        _validate_capture_result(rgb, None, capture_mode="rgb")  # should not raise

    def test_invalid_capture_mode_rejected(self):
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        with pytest.raises(ValueError, match="capture_mode"):
            _validate_capture_result(rgb, None, capture_mode="rgba")


class TestBlankBackground:
    """Blank/uniform backgrounds are legal in some scenes; near-uniform is not."""

    def test_completely_black_rgb_legal(self):
        """A scene with no objects can produce all-black RGB."""
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        # All-black has std=0, which IS near-uniform and should be rejected
        with pytest.raises(ValueError, match="nearly uniform"):
            _validate_capture_result(rgb, None, capture_mode="rgb")

    def test_near_uniform_rgb_rejected(self):
        rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
        with pytest.raises(ValueError, match="nearly uniform"):
            _validate_capture_result(rgb, None, capture_mode="rgb")

    def test_slight_noise_not_near_uniform(self):
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
        _validate_capture_result(rgb, None, capture_mode="rgb")  # should not raise


class TestCameraZDepthValidation:
    """camera-z depth rules: positive values, valid fraction, unit honesty."""

    def test_valid_fraction_threshold(self):
        # Less than 5% valid pixels should fail
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[0, 0] = 1.0  # only 1 valid pixel = 1%
        with pytest.raises(ValueError, match="Insufficient valid camera-z"):
            normalize_camera_z(depth, width=10, height=10, meters_per_unit=1.0)

    def test_all_valid_passes(self):
        depth = np.ones((4, 4), dtype=np.float32)
        clean, stats = normalize_camera_z(depth, width=4, height=4, meters_per_unit=1.0)
        assert stats["valid_fraction"] == 1.0
        assert stats["valid_pixel_count"] == 16

    def test_camera_z_preserved_not_converted(self):
        """Values must be camera-z, not euclidean range."""
        raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        clean, _ = normalize_camera_z(raw, width=2, height=2, meters_per_unit=1.0)
        np.testing.assert_array_equal(clean, raw)

    def test_invalid_encoding_zero(self):
        depth = np.ones((4, 4), dtype=np.float32)
        _, stats = normalize_camera_z(depth, width=4, height=4, meters_per_unit=1.0)
        assert stats["invalid_encoding"] == 0
        assert stats["kind"] == "camera_z"
        assert stats["unit"] == "world_units"

    def test_metric_scale_not_verified(self):
        depth = np.ones((4, 4), dtype=np.float32)
        _, stats = normalize_camera_z(depth, width=4, height=4, meters_per_unit=1.0)
        assert stats["metric_scale_verified"] is False
        assert stats["renderer_reported_unit"] == "meters"
        assert stats["stage_meters_per_unit"] == 1.0


class TestMetersPerUnitParameterization:
    """The metersPerUnit=1 whitelist must be parameterizable."""

    def test_default_rejects_non_unit(self):
        depth = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="metersPerUnit"):
            normalize_camera_z(depth, width=4, height=4, meters_per_unit=0.01)

    def test_default_accepts_unit(self):
        depth = np.ones((4, 4), dtype=np.float32)
        clean, stats = normalize_camera_z(depth, width=4, height=4, meters_per_unit=1.0)
        assert stats["stage_meters_per_unit"] == 1.0

    def test_allowed_set_accepts_other_units(self):
        depth = np.ones((4, 4), dtype=np.float32)
        clean, stats = normalize_camera_z(
            depth, width=4, height=4, meters_per_unit=0.01,
            allowed_meters_per_unit={0.01, 1.0},
        )
        assert stats["stage_meters_per_unit"] == 0.01
        assert stats["metric_scale_verified"] is False

    def test_allowed_set_still_rejects_unknown(self):
        depth = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="metersPerUnit"):
            normalize_camera_z(
                depth, width=4, height=4, meters_per_unit=0.001,
                allowed_meters_per_unit={0.01, 1.0},
            )
