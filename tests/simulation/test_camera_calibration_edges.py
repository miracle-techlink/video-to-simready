"""Camera calibration edge cases: distortion, skew, non-square pixels, decoupling."""

import copy

import numpy as np
import pytest

from simready.geometry.cameras import look_at_c2w
from simready.simulation.camera_calibration import rendered_camera_calibration
from simready.simulation.depth_observation import calibration_to_usd_camera


def _camera():
    return {
        "schema_version": 1,
        "camera_convention": "opencv_c2w",
        "distortion": "none",
        "image_size": [640, 360],
        "intrinsics": [[400, 0, 319.5], [0, 400, 179.5], [0, 0, 1]],
        "camera_to_world": look_at_c2w([2, 3, 4], [0, 0, 0]).tolist(),
    }


class TestDistortionRejection:
    """Distorted cameras must be explicitly rejected by default."""

    def test_distorted_camera_rejected_by_default(self):
        calibration = _camera()
        calibration["distortion"] = "opencv"
        with pytest.raises(ValueError, match="undistorted"):
            calibration_to_usd_camera(calibration)

    def test_distorted_camera_rejected_when_allow_distortion_false(self):
        calibration = _camera()
        calibration["distortion"] = "opencv"
        with pytest.raises(ValueError, match="undistorted"):
            calibration_to_usd_camera(calibration, allow_distortion=False)

    def test_distorted_camera_accepted_when_allow_distortion_true(self):
        calibration = _camera()
        calibration["distortion"] = "opencv"
        result = calibration_to_usd_camera(calibration, allow_distortion=True)
        assert result["width"] == 640
        assert result["height"] == 360


class TestSkewRejection:
    """Skewed intrinsics must be explicitly rejected by default."""

    def test_skewed_camera_rejected_by_default(self):
        calibration = _camera()
        calibration["intrinsics"][0][1] = 2.0
        with pytest.raises(ValueError, match="zero-skew"):
            calibration_to_usd_camera(calibration)

    def test_skewed_camera_rejected_when_allow_skew_false(self):
        calibration = _camera()
        calibration["intrinsics"][0][1] = 2.0
        with pytest.raises(ValueError, match="zero-skew"):
            calibration_to_usd_camera(calibration, allow_skew=False)

    def test_skewed_camera_accepted_when_allow_skew_true(self):
        calibration = _camera()
        calibration["intrinsics"][0][1] = 2.0
        result = calibration_to_usd_camera(calibration, allow_skew=True)
        assert result["width"] == 640


class TestNonSquarePixelRejection:
    """fx != fy must be explicitly rejected by default."""

    def test_nonsquare_rejected_by_default(self):
        calibration = _camera()
        calibration["intrinsics"][1][1] = 390
        with pytest.raises(ValueError, match="square pixels"):
            calibration_to_usd_camera(calibration)

    def test_nonsquare_rejected_when_allow_nonsquare_false(self):
        calibration = _camera()
        calibration["intrinsics"][1][1] = 390
        with pytest.raises(ValueError, match="square pixels"):
            calibration_to_usd_camera(calibration, allow_nonsquare_pixels=False)

    def test_nonsquare_accepted_when_allow_nonsquare_true(self):
        calibration = _camera()
        calibration["intrinsics"][1][1] = 390
        result = calibration_to_usd_camera(calibration, allow_nonsquare_pixels=True)
        assert result["width"] == 640
        # Aperture differs for non-square pixels
        assert result["horizontal_aperture_mm"] != result["vertical_aperture_mm"]


class TestNonOpencvC2wRejection:
    """Non-opencv_c2w camera conventions must be rejected."""

    def test_unreal_convention_rejected(self):
        calibration = _camera()
        calibration["camera_convention"] = "unreal_c2w"
        with pytest.raises(ValueError, match="opencv_c2w"):
            calibration_to_usd_camera(calibration)

    def test_opengl_convention_rejected(self):
        calibration = _camera()
        calibration["camera_convention"] = "opengl_c2w"
        with pytest.raises(ValueError, match="opencv_c2w"):
            calibration_to_usd_camera(calibration)


class TestCalibrationPlyDecoupling:
    """Shared benchmark cameras must not be bound to a specific PLY."""

    def test_shared_calibration_omits_ply_binding(self, tmp_path):
        image = tmp_path / "image.png"
        # Write a minimal valid PNG (8x8 RGB)
        from PIL import Image

        Image.new("RGB", (8, 8), (128, 128, 128)).save(image)
        result = rendered_camera_calibration(
            image,
            eye=[5, 0, 0],
            target=[0, 0, 0],
            focal_length_mm=21.0,
            horizontal_aperture_mm=36.0,
            evidence="shared benchmark camera",
        )
        assert result["calibration_kind"] == "shared_render_camera"
        assert "ply_sha256" not in result
        assert "source_to_world_sha256" not in result
        assert result["real_camera_calibrated"] is False

    def test_shared_calibration_hash_differs_from_bound(self, tmp_path):
        from PIL import Image

        image = tmp_path / "image.png"
        Image.new("RGB", (8, 8), (128, 128, 128)).save(image)
        shared = rendered_camera_calibration(
            image,
            eye=[5, 0, 0],
            target=[0, 0, 0],
            focal_length_mm=21.0,
            horizontal_aperture_mm=36.0,
            evidence="shared benchmark camera",
        )
        bound = rendered_camera_calibration(
            image,
            ply_sha256="a" * 64,
            source_to_world=np.eye(4).tolist(),
            eye=[5, 0, 0],
            target=[0, 0, 0],
            focal_length_mm=21.0,
            horizontal_aperture_mm=36.0,
            evidence="bound to specific member",
        )
        # Different binding should produce different dicts
        assert shared != bound
        assert shared["calibration_kind"] != bound["calibration_kind"]

    def test_ply_and_transform_must_be_provided_together(self, tmp_path):
        from PIL import Image

        image = tmp_path / "image.png"
        Image.new("RGB", (8, 8), (128, 128, 128)).save(image)
        with pytest.raises(ValueError, match="together"):
            rendered_camera_calibration(
                image,
                ply_sha256="a" * 64,
                eye=[0, 0, 5],
                target=[0, 0, 0],
                focal_length_mm=21.0,
                horizontal_aperture_mm=36.0,
                evidence="incomplete binding",
            )

    def test_transform_without_ply_rejected(self, tmp_path):
        from PIL import Image

        image = tmp_path / "image.png"
        Image.new("RGB", (8, 8), (128, 128, 128)).save(image)
        with pytest.raises(ValueError, match="together"):
            rendered_camera_calibration(
                image,
                source_to_world=np.eye(4).tolist(),
                eye=[0, 0, 5],
                target=[0, 0, 0],
                focal_length_mm=21.0,
                horizontal_aperture_mm=36.0,
                evidence="incomplete binding",
            )

    def test_bound_calibration_preserves_ply_hash(self, tmp_path):
        from PIL import Image

        image = tmp_path / "image.png"
        Image.new("RGB", (8, 8), (128, 128, 128)).save(image)
        result = rendered_camera_calibration(
            image,
            ply_sha256="b" * 64,
            source_to_world=np.eye(4).tolist(),
            eye=[5, 0, 0],
            target=[0, 0, 0],
            focal_length_mm=21.0,
            horizontal_aperture_mm=36.0,
            evidence="bound to member A",
        )
        assert result["ply_sha256"] == "b" * 64
        assert "source_to_world_sha256" in result
