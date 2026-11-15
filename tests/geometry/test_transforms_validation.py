"""Direct coverage for finite array and Sim3 transform validation."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simready.geometry.cameras import rigid_inverse
from simready.geometry.transforms_validation import finite_array, validate_sim3


def test_finite_array_returns_float64_and_validates_shape():
    result = finite_array([[1, 2], [3, 4]], "m", (2, 2))
    assert result.dtype == np.float64
    with pytest.raises(ValueError, match="expected shape"):
        finite_array([1, 2], "m", (2, 2))


@pytest.mark.parametrize("value", [[np.nan], [np.inf], [None]])
def test_finite_array_rejects_nonfinite_and_nonnumeric(value):
    with pytest.raises(ValueError, match="Invalid finite"):
        finite_array(value, "m")


def test_finite_array_rejects_nonnumeric_strings():
    with pytest.raises((ValueError, TypeError)):
        finite_array(["a"], "m")


def test_validate_sim3_accepts_identity_and_uniform_scale():
    np.testing.assert_allclose(validate_sim3(np.eye(4)), np.eye(4), atol=0)
    scaled = np.eye(4)
    scaled[:3, :3] *= 2.5
    np.testing.assert_allclose(validate_sim3(scaled), scaled, atol=0)


def test_validate_sim3_accepts_rotation_translation_scale():
    rng = np.random.default_rng(0)
    rotation = Rotation.random(random_state=rng).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation * 3.0
    transform[:3, 3] = [1.0, -2.0, 0.5]
    np.testing.assert_allclose(validate_sim3(transform), transform, atol=0)


def test_validate_sim3_rejects_reflection():
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive uniform scale and rotation"):
        validate_sim3(reflection)


def test_validate_sim3_rejects_nonuniform_scale():
    transform = np.diag([1.0, 2.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="positive uniform scale and rotation"):
        validate_sim3(transform)


def test_validate_sim3_rejects_shear():
    transform = np.eye(4)
    transform[0, 1] = 0.1
    with pytest.raises(ValueError, match="positive uniform scale and rotation"):
        validate_sim3(transform)


def test_validate_sim3_rejects_nonaffine_last_row():
    transform = np.eye(4)
    transform[3] = [0.0, 0.0, 0.1, 1.0]
    with pytest.raises(ValueError, match="must be affine"):
        validate_sim3(transform)


@pytest.mark.parametrize("scale", [0.0, 1e-9, 1e9])
def test_validate_sim3_rejects_extreme_scale(scale):
    transform = np.eye(4)
    transform[:3, :3] *= scale
    with pytest.raises(ValueError, match="positive uniform scale and rotation"):
        validate_sim3(transform)


def test_rigid_inverse_roundtrip():
    rng = np.random.default_rng(2)
    rotation = Rotation.random(random_state=rng).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = rigid_inverse(c2w)
    np.testing.assert_allclose(w2c @ c2w, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(c2w @ w2c, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(w2c[:3, 3], -rotation.T @ c2w[:3, 3], atol=1e-12)


def test_rigid_inverse_rejects_scaled_and_reflected():
    scaled = np.eye(4)
    scaled[:3, :3] *= 2.0
    with pytest.raises(ValueError, match="not a rigid pose"):
        rigid_inverse(scaled)
    reflected = np.eye(4)
    reflected[0, 0] = -1.0
    with pytest.raises(ValueError, match="not a rigid pose"):
        rigid_inverse(reflected)


def test_rigid_inverse_rejects_nan_and_wrong_shape():
    with pytest.raises(ValueError, match="Finite rigid"):
        rigid_inverse(np.full((4, 4), np.nan))
    with pytest.raises(ValueError, match="Finite rigid"):
        rigid_inverse(np.eye(3))
