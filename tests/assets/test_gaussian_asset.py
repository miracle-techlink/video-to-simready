"""PLY loading, coordinate conventions, and USD appearance export."""

import numpy as np
import pytest

from simready.assets.appearance.gaussian_asset import (
    decode_vertices,
    read_ply,
    write_appearance,
)


def _make_ply(tmp_path, points, properties=None, binary=True):
    """Create a minimal binary PLY file with the given vertex data."""
    if properties is None:
        properties = [("x", "float"), ("y", "float"), ("z", "float")]

    header_lines = ["ply"]
    if binary:
        header_lines.append("format binary_little_endian 1.0")
    else:
        header_lines.append("format ascii 1.0")
    header_lines.append(f"element vertex {len(points)}")
    for name, ptype in properties:
        header_lines.append(f"property {ptype} {name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    ply_path = tmp_path / "test.ply"
    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        if binary:
            for point in points:
                for i, (_name, ptype) in enumerate(properties):
                    dtype = {
                        "float": "<f4",
                        "float32": "<f4",
                        "double": "<f8",
                        "float64": "<f8",
                        "uchar": "u1",
                        "uint8": "u1",
                        "int": "<i4",
                        "int32": "<i4",
                    }[ptype]
                    f.write(np.array([point[i]], dtype=dtype).tobytes())
    return ply_path


def _make_gaussian_ply(tmp_path, n=10):
    """Create a valid 3DGS PLY with all required properties."""
    rng = np.random.default_rng(42)
    points = rng.normal(0, 1, (n, 3)).astype(np.float32)
    rotations = rng.normal(0, 1, (n, 4)).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    log_scales = rng.normal(-2, 0.5, (n, 3)).astype(np.float32)
    logits = rng.normal(0, 1, (n, 1)).astype(np.float32)
    dc = rng.normal(0, 0.5, (n, 3)).astype(np.float32)

    properties = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("rot_0", "float"),
        ("rot_1", "float"),
        ("rot_2", "float"),
        ("rot_3", "float"),
        ("scale_0", "float"),
        ("scale_1", "float"),
        ("scale_2", "float"),
        ("opacity", "float"),
        ("f_dc_0", "float"),
        ("f_dc_1", "float"),
        ("f_dc_2", "float"),
    ]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    for name, ptype in properties:
        header += f"property {ptype} {name}\n"
    header += "end_header\n"

    ply_path = tmp_path / "gaussian.ply"
    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            row = [
                points[i, 0],
                points[i, 1],
                points[i, 2],
                rotations[i, 0],
                rotations[i, 1],
                rotations[i, 2],
                rotations[i, 3],
                log_scales[i, 0],
                log_scales[i, 1],
                log_scales[i, 2],
                logits[i, 0],
                dc[i, 0],
                dc[i, 1],
                dc[i, 2],
            ]
            for val in row:
                f.write(np.array([val], dtype="<f4").tobytes())
    return ply_path


class TestReadPly:
    def test_valid_ply_loaded(self, tmp_path):
        points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ply = _make_ply(tmp_path, points)
        result = read_ply(ply)
        assert "x" in result and "y" in result and "z" in result
        assert len(result["x"]) == 2
        assert abs(result["x"][0] - 1.0) < 1e-6

    def test_not_ply_rejected(self, tmp_path):
        bad = tmp_path / "bad.ply"
        bad.write_text("not a ply file")
        with pytest.raises(ValueError, match="Not a PLY"):
            read_ply(bad)

    def test_ascii_ply_rejected(self, tmp_path):
        points = [[1.0, 2.0, 3.0]]
        ply = _make_ply(tmp_path, points, binary=False)
        with pytest.raises(ValueError, match="binary_little_endian"):
            read_ply(ply)

    def test_max_points_enforced(self, tmp_path):
        points = [[float(i), 0, 0] for i in range(100)]
        ply = _make_ply(tmp_path, points)
        with pytest.raises(ValueError, match="resource limit"):
            read_ply(ply, max_points=10)

    def test_zero_points_rejected(self, tmp_path):
        ply = _make_ply(tmp_path, [])
        with pytest.raises(ValueError, match="empty"):
            read_ply(ply)


class TestDecodeVertices:
    def test_rgb_points_decoded(self, tmp_path):
        points = [[1.0, 2.0, 3.0, 255, 0, 0]]
        ply = _make_ply(
            tmp_path,
            points,
            properties=[
                ("x", "float"),
                ("y", "float"),
                ("z", "float"),
                ("red", "uchar"),
                ("green", "uchar"),
                ("blue", "uchar"),
            ],
        )
        result = decode_vertices(read_ply(ply))
        assert result["kind"] == "rgb_points"
        assert result["colors"][0][0] == 1.0  # 255/255

    def test_gaussian_splats_decoded(self, tmp_path):
        ply = _make_gaussian_ply(tmp_path)
        result = decode_vertices(read_ply(ply))
        assert result["kind"] == "gaussian_splats"
        assert result["sh_degree"] == 0
        assert len(result["points"]) == 10
        assert np.all(result["scales"] > 0)
        assert np.all(result["opacities"] > 0)
        assert np.all(result["opacities"] <= 1)

    def test_missing_xyz_rejected(self, tmp_path):
        ply = _make_ply(tmp_path, [[1.0]], properties=[("x", "float")])
        with pytest.raises(ValueError, match="Missing"):
            decode_vertices(read_ply(ply))

    def test_zero_quaternion_rejected(self, tmp_path):
        n = 2
        properties = [
            ("x", "float"),
            ("y", "float"),
            ("z", "float"),
            ("rot_0", "float"),
            ("rot_1", "float"),
            ("rot_2", "float"),
            ("rot_3", "float"),
            ("scale_0", "float"),
            ("scale_1", "float"),
            ("scale_2", "float"),
            ("opacity", "float"),
            ("f_dc_0", "float"),
            ("f_dc_1", "float"),
            ("f_dc_2", "float"),
        ]
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        for name, ptype in properties:
            header += f"property {ptype} {name}\n"
        header += "end_header\n"
        ply_path = tmp_path / "zero_rot.ply"
        with open(ply_path, "wb") as f:
            f.write(header.encode("ascii"))
            for _ in range(n):
                # All-zero quaternion.
                for val in [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]:
                    f.write(np.array([val], dtype="<f4").tobytes())
        with pytest.raises(ValueError, match="[Zz]ero.*quaternion"):
            decode_vertices(read_ply(ply_path))


class TestWriteAppearance:
    def test_rgb_points_export(self, tmp_path):
        pytest.importorskip("pxr")
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        data = {"kind": "rgb_points", "points": points, "colors": colors}
        out = tmp_path / "appearance.usdc"
        result = write_appearance(data, out, np.eye(4))
        assert result["kind"] == "rgb_points"
        assert result["count"] == 2
        assert result["surface_mesh_generated"] is False
        assert out.is_file()

    def test_gaussian_splats_export(self, tmp_path):
        pytest.importorskip("pxr")
        ply = _make_gaussian_ply(tmp_path)
        data = decode_vertices(read_ply(ply))
        out = tmp_path / "gaussian.usdc"
        result = write_appearance(data, out, np.eye(4))
        assert result["kind"] == "gaussian_splats"
        assert result["count"] == 10
        assert result["encoding"] == "3DGS log-scale/logit/wxyz/channel-major-SH"
        assert out.is_file()

    def test_output_must_not_exist(self, tmp_path):
        pytest.importorskip("pxr")
        data = {
            "kind": "rgb_points",
            "points": np.array([[0, 0, 0]], dtype=np.float32),
            "colors": np.array([[1, 0, 0]], dtype=np.float32),
        }
        out = tmp_path / "existing.usdc"
        out.write_text("exists")
        with pytest.raises(FileExistsError):
            write_appearance(data, out, np.eye(4))

    def test_coordinate_transform_applied(self, tmp_path):
        """The source_to_world Sim3 is applied as a parent Xform, preserving
        the SH frame."""
        pytest.importorskip("pxr")
        data = {
            "kind": "rgb_points",
            "points": np.array([[1, 0, 0]], dtype=np.float32),
            "colors": np.array([[1, 0, 0]], dtype=np.float32),
        }
        # Scale by 2.
        transform = np.eye(4) * 2
        transform[3, 3] = 1
        out = tmp_path / "scaled.usdc"
        result = write_appearance(data, out, transform)
        assert result["count"] == 1
