"""CPU tests for the quadcopter controller math (no Isaac required)."""

from __future__ import annotations

import numpy as np
import pytest

from digua_hackathon.controller import SPEC, PositionAttitudeController


def test_rotation_matrix_is_orthonormal():
    quat = np.array([0.9, 0.1, 0.2, -0.3])
    rot = PositionAttitudeController.rotation_matrix(quat)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)


def test_desired_rotation_aligns_thrust_and_yaw():
    ctrl = PositionAttitudeController()
    rot = ctrl.desired_rotation(np.array([0.0, 0.0, 1.0]), yaw_rad=0.7)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.allclose(rot[:, 2], [0.0, 0.0, 1.0], atol=1e-9)
    heading = rot[:, 0]
    assert np.isclose(np.arctan2(heading[1], heading[0]), 0.7, atol=1e-9)
    assert np.isclose(heading[2], 0.0, atol=1e-6)


def test_mixer_roundtrip_unsaturated():
    ctrl = PositionAttitudeController()
    force = SPEC.mass_kg * 9.81
    torque = np.array([0.02, -0.03, 0.01])
    thrusts, saturated = ctrl.mixer(force, torque)
    assert not saturated
    assert np.all(thrusts >= 0.0)
    achieved_force, achieved_torque = ctrl.wrench_from_thrusts(thrusts)
    assert np.isclose(achieved_force, force, atol=1e-9)
    assert np.allclose(achieved_torque, torque, atol=1e-9)


def test_mixer_clamps_to_rotor_limits():
    ctrl = PositionAttitudeController()
    thrusts, saturated = ctrl.mixer(5 * SPEC.rotor_max_thrust_n, np.zeros(3))
    assert saturated
    assert np.allclose(thrusts, SPEC.rotor_max_thrust_n)
    achieved, _ = ctrl.wrench_from_thrusts(thrusts)
    assert achieved <= 4 * SPEC.rotor_max_thrust_n + 1e-9


def test_wrench_matches_rotor_geometry():
    """wrench_from_thrusts must equal sum(r_i x f_i) from rotor_positions()."""
    from digua_hackathon.controller import rotor_positions

    ctrl = PositionAttitudeController()
    thrusts = np.array([3.0, 1.5, 2.0, 4.0])
    force, torque = ctrl.wrench_from_thrusts(thrusts)
    assert np.isclose(force, thrusts.sum())
    expected = np.zeros(3)
    for r, f in zip(rotor_positions(), thrusts, strict=True):
        expected += np.cross(r, np.array([0.0, 0.0, f]))
    # x/y torque must come purely from rotor geometry
    assert np.allclose(torque[:2], expected[:2], atol=1e-12)
    # z torque follows the declared reaction pattern (not a geometric term)
    assert np.isclose(torque[2], thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])


def test_hover_thrust_within_envelope():
    hover = SPEC.mass_kg * 9.81
    per_rotor = hover / 4
    assert per_rotor < SPEC.rotor_max_thrust_n, "hover must not saturate rotors"
    assert hover * 2.0 < 4 * SPEC.rotor_max_thrust_n, "TWR must exceed 2 for control"


def test_attitude_error_zero_for_matching_rotation():
    ctrl = PositionAttitudeController()
    quat = np.array([0.95, 0.1, -0.2, 0.2])
    quat /= np.linalg.norm(quat)
    rot = ctrl.rotation_matrix(quat)
    err = ctrl.attitude_error_vee(rot @ rot.T - rot @ rot.T)
    assert np.allclose(err, 0.0, atol=1e-12)


def test_step_returns_finite_commands_near_hover():
    ctrl = PositionAttitudeController()
    state = dict(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_body=np.zeros(3),
        ref_position=np.array([0.0, 0.0, 2.0]),
        ref_velocity=np.zeros(3),
        ref_yaw=0.0,
    )
    cmd = ctrl.step(**state)
    thrusts = np.asarray(cmd["thrusts_n"])
    assert np.isfinite(thrusts).all()
    assert abs(cmd["total_thrust_n"] - SPEC.mass_kg * 9.81) < 0.2
    assert not cmd["saturated"]
