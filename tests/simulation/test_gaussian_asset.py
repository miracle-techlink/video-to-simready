"""Gaussian asset edge cases: PLY loading, coordinate conversion, USD export."""

import numpy as np
import pytest

from simready.assets.appearance.gaussian_asset import (
    decode_vertices,
    read_ply,
    write_appearance,
)
from tests.fixtures.scene_preparation import write_ply


class TestPlyLoadingFailures:
    """Malformed PLY inputs must be rejected before allocation."""

    def test_not_a_ply_file(self, tmp_path):
        path = tmp_path / "fake.ply"
        path.write_bytes(b"not a ply file")
        with pytest.raises(ValueError, match="Not a PLY"):
            read_ply(path)

    def test_ascii_ply_rejected(self, tmp_path):
        path = tmp_path / "ascii.ply"
        path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n"
        )
        with pytest.raises(ValueError, match="binary_little_endian"):
            read_ply(path)

    def test_oversized_header_rejected(self, tmp_path):
        path = tmp_path / "huge.ply"
        path.write_bytes(b"ply\n" + b"comment padding\n" * 5000 + b"end_header\n")
        with pytest.raises(ValueError, match="header exceeds"):
            read_ply(path)

    def test_empty_vertex_count_rejected(self, tmp_path):
        path = tmp_path / "empty.ply"
        path.write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
            b"property float x\nend_header\n"
        )
        with pytest.raises(ValueError, match="empty"):
            read_ply(path)


class TestCoordinateConversion:
    """PLY to USD coordinate conversion must preserve data integrity."""

    def test_gaussian_decoding(self, tmp_path):
        """3DGS PLY decodes to correct scale/rotation/opacity."""
        path = write_ply(tmp_path / "test.ply", [[0, 0, 0], [1, 2, 3]])
        data = decode_vertices(read_ply(path))
        assert data["kind"] == "gaussian_splats"
        assert data["points"].shape == (2, 3)
        assert data["rotations"].shape == (2, 4)
        assert data["scales"].shape == (2, 3)
        assert data["opacities"].shape == (2,)
        assert data["sh"].shape[0] == 2 * 4  # 2 points x 4 SH coefficients (degree 1)

    def test_rgb_point_decoding(self, tmp_path):
        """RGB point cloud decodes to uint8 colors without Gaussian params."""
        path = write_ply(tmp_path / "rgb.ply", [[0, 0, 0]], gaussian=False)
        data = decode_vertices(read_ply(path))
        assert data["kind"] == "rgb_points"
        assert "scales" not in data
        assert data["colors"].shape == (1, 3)

    def test_zero_gaussian_quaternion_rejected(self, tmp_path):
        properties = read_ply(write_ply(tmp_path / "bad.ply", [[0, 0, 0]]))
        properties["rot_0"][0] = 0
        properties["rot_1"][0] = 0
        properties["rot_2"][0] = 0
        properties["rot_3"][0] = 0
        with pytest.raises(ValueError, match="Zero-length"):
            decode_vertices(properties)

    def test_nan_position_rejected(self, tmp_path):
        properties = read_ply(write_ply(tmp_path / "bad.ply", [[0, 0, 0]]))
        properties["x"][0] = np.nan
        with pytest.raises(ValueError):
            decode_vertices(properties)


class TestUsdExport:
    """USD appearance export must be portable and self-contained."""

    def test_gaussian_usd_export(self, tmp_path):
        """3DGS PLY exports to USD with correct splat count."""
        pytest.importorskip("pxr")
        from pxr import Usd, UsdVol

        path = write_ply(tmp_path / "test.ply", [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        data = decode_vertices(read_ply(path))
        output = tmp_path / "appearance.usdc"
        result = write_appearance(data, output, np.eye(4).tolist())
        assert result["kind"] == "gaussian_splats"
        assert result["count"] == 3
        stage = Usd.Stage.Open(str(output))
        splats = UsdVol.ParticleField3DGaussianSplat(
            stage.GetPrimAtPath("/Appearance/Source/Splats")
        )
        assert len(splats.GetPositionsAttr().Get()) == 3

    def test_rgb_usd_export(self, tmp_path):
        """RGB point cloud exports to USD Points."""
        pytest.importorskip("pxr")
        from pxr import Usd, UsdGeom

        path = write_ply(tmp_path / "rgb.ply", [[0, 0, 0]], gaussian=False)
        data = decode_vertices(read_ply(path))
        output = tmp_path / "appearance.usdc"
        result = write_appearance(data, output, np.eye(4).tolist())
        assert result["kind"] == "rgb_points"
        stage = Usd.Stage.Open(str(output))
        points = UsdGeom.Points(stage.GetPrimAtPath("/Appearance/Source/Points"))
        assert len(points.GetPointsAttr().Get()) == 1

    def test_existing_output_rejected(self, tmp_path):
        """USD export must not overwrite existing files."""
        pytest.importorskip("pxr")

        path = write_ply(tmp_path / "test.ply", [[0, 0, 0]])
        data = decode_vertices(read_ply(path))
        output = tmp_path / "existing.usdc"
        output.write_bytes(b"existing")
        with pytest.raises(FileExistsError):
            write_appearance(data, output, np.eye(4).tolist())

    def test_non_sim3_transform_rejected(self, tmp_path):
        """USD export must validate the coordinate transform."""
        pytest.importorskip("pxr")

        path = write_ply(tmp_path / "test.ply", [[0, 0, 0]])
        data = decode_vertices(read_ply(path))
        output = tmp_path / "test.usdc"
        with pytest.raises(ValueError, match="source_to_world"):
            write_appearance(data, output, np.diag([1, 2, 1, 1]).tolist())
