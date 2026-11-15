"""Semantic projection: mask->3D backprojection, occlusion culling, empty masks."""

import numpy as np
import pytest
from PIL import Image

from simready.assets.association.semantic_projection import (
    build_semantic_spec,
    visible_point_pixels,
)
from simready.core.artifacts import digest, object_digest, write_json


class TestVisiblePointPixels:
    """Test the z-buffer visibility culling used by mask->3D projection."""

    def _camera_and_intrinsics(self):
        c2w = np.eye(4)
        intrinsics = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
        return c2w, intrinsics

    def test_single_point_projects_to_center(self):
        c2w, intrinsics = self._camera_and_intrinsics()
        points = np.array([[0, 0, 5]], dtype=float)  # 5m in front
        indices, px, py = visible_point_pixels(
            points, c2w, intrinsics, (100, 100)
        )
        assert len(indices) == 1
        assert abs(px[0] - 50) <= 1
        assert abs(py[0] - 50) <= 1

    def test_behind_camera_culled(self):
        c2w, intrinsics = self._camera_and_intrinsics()
        points = np.array([[0, 0, -5]], dtype=float)  # Behind camera
        indices, px, py = visible_point_pixels(
            points, c2w, intrinsics, (100, 100)
        )
        assert len(indices) == 0

    def test_occluded_point_culled(self):
        """A closer point at the same pixel occludes a farther point."""
        c2w, intrinsics = self._camera_and_intrinsics()
        points = np.array([
            [0, 0, 5],   # Near
            [0.01, 0.01, 10],  # Far, projects to same pixel
        ], dtype=float)
        indices, px, py = visible_point_pixels(
            points, c2w, intrinsics, (100, 100),
            relative_depth_tolerance=0.0,
        )
        assert len(indices) == 1
        assert indices[0] == 0  # Only the near point survives

    def test_out_of_frame_culled(self):
        c2w, intrinsics = self._camera_and_intrinsics()
        points = np.array([[100, 0, 5]], dtype=float)  # Way to the right
        indices, px, py = visible_point_pixels(
            points, c2w, intrinsics, (100, 100)
        )
        assert len(indices) == 0

    def test_invalid_intrinsics_rejected(self):
        c2w = np.eye(4)
        bad_intrinsics = np.array([[0, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
        with pytest.raises(ValueError, match="pinhole"):
            visible_point_pixels(
                np.array([[0, 0, 5]]), c2w, bad_intrinsics, (100, 100)
            )

    def test_skewed_intrinsics_rejected(self):
        c2w = np.eye(4)
        skewed = np.array([[100, 5, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
        with pytest.raises(ValueError, match="pinhole"):
            visible_point_pixels(
                np.array([[0, 0, 5]]), c2w, skewed, (100, 100)
            )

    def test_empty_points(self):
        c2w, intrinsics = self._camera_and_intrinsics()
        indices, px, py = visible_point_pixels(
            np.empty((0, 3)), c2w, intrinsics, (100, 100)
        )
        assert len(indices) == 0

    def test_pixel_limit_enforced(self):
        c2w, intrinsics = self._camera_and_intrinsics()
        with pytest.raises(ValueError, match="megapixel"):
            visible_point_pixels(
                np.array([[0, 0, 5]]), c2w, intrinsics, (5000, 5000)
            )


class TestBuildSemanticSpec:
    """Test the full build_semantic_spec pipeline with synthetic fixtures."""

    def _make_fixtures(self, tmp_path, with_depth=False):
        """Create PLY, segmentation manifest, and calibration for testing."""
        from simready.assets.appearance.gaussian_asset import (
            decode_vertices,
            read_ply,
        )

        # Create a simple PLY with a few points.
        rng = np.random.default_rng(42)
        n = 50
        points = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        colors = rng.integers(0, 256, (n, 3), dtype=np.uint8)

        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        for name in ("x", "y", "z"):
            header += f"property float {name}\n"
        for name in ("red", "green", "blue"):
            header += f"property uchar {name}\n"
        header += "end_header\n"

        ply_path = tmp_path / "scene.ply"
        with open(ply_path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(n):
                for v in points[i]:
                    f.write(np.array([v], dtype="<f4").tobytes())
                for v in colors[i]:
                    f.write(np.array([v], dtype="u1").tobytes())

        # Source image.
        img = np.random.default_rng(7).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "source.png"
        Image.fromarray(img).save(img_path)

        # Mask for one object.
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        mask_path = tmp_path / "mask_obj1.png"
        Image.fromarray(mask).save(mask_path)

        manifest = {
            "schema_version": 1,
            "source_image": {
                "path": "source.png",
                "sha256": digest(img_path),
                "width": 100,
                "height": 100,
            },
            "instances": [
                {
                    "asset_id": "obj1",
                    "category": "chair",
                    "mask": {
                        "path": "mask_obj1.png",
                        "sha256": digest(mask_path),
                    },
                    "area_px": 400,
                    "score": 0.9,
                }
            ],
        }
        manifest_path = tmp_path / "segmentation.json"
        write_json(manifest_path, manifest)

        # Calibration.
        c2w = np.eye(4)
        c2w[2, 3] = 3  # Camera at z=3 looking at origin.
        intrinsics = [[50, 0, 50], [0, 50, 50], [0, 0, 1]]
        calibration = {
            "schema_version": 1,
            "camera_convention": "opencv_c2w",
            "distortion": "none",
            "image_size": [100, 100],
            "intrinsics": intrinsics,
            "camera_to_world": c2w.tolist(),
            "image_sha256": digest(img_path),
            "ply_sha256": digest(ply_path),
            "source_to_world_sha256": object_digest(np.eye(4).tolist()),
            "evidence": "test calibration",
        }

        if with_depth:
            depth = np.full((100, 100), 3.0, dtype=np.float32)
            depth_path = tmp_path / "depth.npy"
            np.save(depth_path, depth, allow_pickle=False)
            calibration["depth"] = {
                "kind": "camera_z",
                "unit": "world_units",
                "path": "depth.npy",
                "sha256": digest(depth_path),
            }

        cal_path = tmp_path / "calibration.json"
        write_json(cal_path, calibration)

        return ply_path, manifest_path, cal_path

    def test_empty_mask_produces_no_assets(self, tmp_path):
        """A mask with zero area is rejected by the manifest validator."""
        ply, manifest_path, cal_path = self._make_fixtures(tmp_path)
        # Overwrite mask with empty one.
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask_path = tmp_path / "mask_obj1.png"
        Image.fromarray(mask).save(mask_path)
        manifest = {
            "schema_version": 1,
            "source_image": {
                "path": "source.png",
                "sha256": digest(tmp_path / "source.png"),
                "width": 100,
                "height": 100,
            },
            "instances": [
                {
                    "asset_id": "obj1",
                    "category": "chair",
                    "mask": {"path": "mask_obj1.png", "sha256": digest(mask_path)},
                    "area_px": 0,
                    "score": 0.9,
                }
            ],
        }
        write_json(manifest_path, manifest)
        with pytest.raises(ValueError, match="area"):
            build_semantic_spec(ply, manifest_path, cal_path, tmp_path / "out")

    def test_low_score_skipped(self, tmp_path):
        ply, manifest_path, cal_path = self._make_fixtures(tmp_path)
        manifest = read_json(manifest_path)
        manifest["instances"][0]["score"] = 0.1
        write_json(manifest_path, manifest)
        result = build_semantic_spec(
            ply, manifest_path, cal_path, tmp_path / "out",
            min_score=0.25,
        )
        assert len(result["accepted"]) == 0
        assert len(result["skipped"]) == 1
        assert result["skipped"][0]["reason"] == "low_segmentation_score"

    def test_unknown_category_skipped(self, tmp_path):
        ply, manifest_path, cal_path = self._make_fixtures(tmp_path)
        manifest = read_json(manifest_path)
        manifest["instances"][0]["category"] = "unknown_thing"
        write_json(manifest_path, manifest)
        result = build_semantic_spec(
            ply, manifest_path, cal_path, tmp_path / "out",
        )
        assert len(result["accepted"]) == 0
        assert any(
            s["reason"] == "category_has_no_physical_prior"
            for s in result["skipped"]
        )

    def test_output_must_not_exist(self, tmp_path):
        ply, manifest_path, cal_path = self._make_fixtures(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(FileExistsError):
            build_semantic_spec(ply, manifest_path, cal_path, out)

    def test_calibration_hash_mismatch_rejected(self, tmp_path):
        ply, manifest_path, cal_path = self._make_fixtures(tmp_path)
        cal = read_json(cal_path)
        cal["ply_sha256"] = "0" * 64
        write_json(cal_path, cal)
        with pytest.raises(ValueError, match="hash mismatch"):
            build_semantic_spec(ply, manifest_path, cal_path, tmp_path / "out")


def read_json(path):
    from simready.core.artifacts import read_json as _read
    return _read(path)
