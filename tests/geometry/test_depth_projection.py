"""Direct coverage for depth backprojection geometry."""

import numpy as np
import pytest

from simready.geometry.depth_projection import backproject_depth


def _intrinsics(fx=100.0, fy=100.0, cx=32.0, cy=24.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def test_backproject_identity_pose_returns_camera_frame_points():
    depth = np.full((48, 64), 2.0, dtype=np.float64)
    world, linear, x, y, count = backproject_depth(
        depth, np.eye(4), _intrinsics()
    )
    assert count == 48 * 64
    # Center pixel projects to (0, 0, depth)
    center = (y == 24) & (x == 32)
    np.testing.assert_allclose(world[center][0], [0.0, 0.0, 2.0], atol=1e-9)
    # A pixel one step right of center has x = depth / fx
    right = (y == 24) & (x == 33)
    np.testing.assert_allclose(world[right][0], [2.0 / 100.0, 0.0, 2.0], atol=1e-9)
    assert (linear == y * 64 + x).all()


def test_backproject_applies_camera_to_world_rotation():
    depth = np.full((4, 4), 1.0, dtype=np.float64)
    c2w = np.eye(4)
    # Rotate 90 degrees about Y: camera +Z becomes world +X
    c2w[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    world, _, x, y, _ = backproject_depth(
        depth, c2w, _intrinsics(cx=1.5, cy=1.5)
    )
    # camera +Z (third column of rotation) maps to world +X
    np.testing.assert_allclose(world[:, 0], 1.0, atol=1e-9)
    # camera +X (first column) maps to world -Z
    np.testing.assert_allclose(world[:, 2], -(x - 1.5) / 100.0, atol=1e-9)


def test_backproject_roundtrip_with_projection():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 3.0, size=(32, 32))
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, -2.0, 0.5]
    intrinsics = _intrinsics(fx=80.0, fy=90.0, cx=16.0, cy=16.0)
    world, _, x, y, _ = backproject_depth(depth, c2w, intrinsics)
    # world = R @ local + t  =>  local = R.T @ (world - t)
    rotation = c2w[:3, :3]
    camera = (world - c2w[:3, 3]) @ rotation
    projected_x = intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]
    projected_y = intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]
    np.testing.assert_allclose(projected_x, x, atol=1e-9)
    np.testing.assert_allclose(projected_y, y, atol=1e-9)
    np.testing.assert_allclose(camera[:, 2], depth[y, x], atol=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_backproject_excludes_degenerate_depth(bad):
    depth = np.full((4, 4), 1.0, dtype=np.float64)
    depth[0, 0] = bad
    _, linear, x, y, count = backproject_depth(depth, np.eye(4), _intrinsics(cx=1.5, cy=1.5))
    assert count == 15
    assert not (linear == 0).any()


def test_backproject_rejects_non_2d_and_non_float():
    with pytest.raises(ValueError, match="floating-point raster"):
        backproject_depth(np.zeros((4, 4), dtype=np.int32), np.eye(4), _intrinsics())
    with pytest.raises(ValueError, match="floating-point raster"):
        backproject_depth(np.zeros((2, 2, 2)), np.eye(4), _intrinsics())


def test_backproject_rejects_non_rigid_pose():
    pose = np.eye(4)
    pose[:3, :3] *= 2.0
    with pytest.raises(ValueError, match="not a rigid pose"):
        backproject_depth(np.ones((2, 2)), pose, _intrinsics())


def test_backproject_rejects_oversized_or_invalid_max_points():
    depth = np.ones((4, 4))
    with pytest.raises(ValueError, match="max_points"):
        backproject_depth(depth, np.eye(4), _intrinsics(), max_points=0)
    with pytest.raises(ValueError, match="max_points"):
        backproject_depth(depth, np.eye(4), _intrinsics(), max_points=1.5)


def test_backproject_deterministic_subsampling():
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.5, 2.0, size=(32, 32))
    world_a, linear_a, _, _, _ = backproject_depth(
        depth, np.eye(4), _intrinsics(), max_points=16
    )
    world_b, linear_b, _, _, _ = backproject_depth(
        depth, np.eye(4), _intrinsics(), max_points=16
    )
    np.testing.assert_array_equal(linear_a, linear_b)
    np.testing.assert_allclose(world_a, world_b, atol=0)
    assert len(linear_a) == 16
