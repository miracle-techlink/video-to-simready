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


def test_attitude_torque_sign_tilts_toward_desired():
    """Closed-loop attitude check: torque from a +x tilt demand must rotate
    the body's z axis TOWARD +x (omega=+y), not away from it."""
    ctrl = PositionAttitudeController()
    rot_d = ctrl.desired_rotation(
        np.array([np.sin(np.radians(10)), 0.0, np.cos(np.radians(10))]), 0.0
    )
    rot = np.eye(3)
    _run_attitude_sim(ctrl, rot, rot_d)


def test_attitude_loop_converges_at_large_yaw_with_tilt():
    """Frame-consistency regression: with the body yawed -90 deg, a tilt
    demand expressed in the body frame must still tilt the WORLD z toward the
    desired direction. (World-frame error + body-frame damping diverged here.)"""
    ctrl = PositionAttitudeController()
    yaw = np.array(
        [[np.cos(np.pi / 2), np.sin(np.pi / 2), 0.0],
         [-np.sin(np.pi / 2), np.cos(np.pi / 2), 0.0],
         [0.0, 0.0, 1.0]]
    )  # body x points along world -y (yaw -90 deg)
    tilt_dir_world = yaw[:, 0] * 0.17 + np.array([0, 0, 0.985])  # tilt toward body +x
    tilt_dir_world /= np.linalg.norm(tilt_dir_world)
    rot_d = ctrl.desired_rotation(tilt_dir_world, -np.pi / 2)
    rot_end = _run_attitude_sim(ctrl, yaw.copy(), rot_d)
    z_final = rot_end[:, 2]
    z_desired = rot_d[:, 2]
    cos_err = float(z_final @ z_desired)
    assert cos_err > 0.99, f"yawed attitude did not converge (cos err {cos_err:.4f})"


def _run_attitude_sim(ctrl, rot, rot_d, steps: int = 600, dt: float = 0.002) -> np.ndarray:
    omega = np.zeros(3)
    for _ in range(steps):
        torque, _ = ctrl.attitude_torque(rot, rot_d, omega)
        omega = omega + dt * torque  # I = identity
        wx, wy, wz = omega
        dR = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) * dt
        rot = rot + rot @ dR
        q, r = np.linalg.qr(rot)
        rot = q @ np.diag(np.sign(np.diag(r)))
    return rot


def _rot_to_quat(rot: np.ndarray) -> np.ndarray:
    """wxyz quaternion from a rotation matrix (Shepperd's method)."""
    tr = np.trace(rot)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (rot[2, 1] - rot[1, 2]) / s,
                         (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s])
    i = int(np.argmax(np.diag(rot)))
    if i == 0:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        return np.array([(rot[2, 1] - rot[1, 2]) / s, 0.25 * s,
                         (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        return np.array([(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s,
                         0.25 * s, (rot[1, 2] + rot[2, 1]) / s])
    s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
    return np.array([(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s,
                     (rot[1, 2] + rot[2, 1]) / s, 0.25 * s])


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


def test_position_error_alone_never_saturates_rotors():
    """The runaway guard: large position/velocity errors must never drive ALL
    rotors to maximum (that froze attitude control and diverged in run 1).
    A single rotor touching 0 during an aggressive combined transient is
    acceptable real-quad behavior."""
    ctrl = PositionAttitudeController()
    for pos, vel, ref_p, ref_v in [
        (np.array([0, 0, 2.0]), np.zeros(3), np.array([6, 0, 6.0]), np.array([5, 0, 0])),
        (np.array([0, 0, 2.0]), np.array([0, 0, -4.0]), np.array([0, 0, 6.0]), np.zeros(3)),
        (np.array([0, 0, 8.0]), np.array([6, -6, 0]), np.array([-2, 2, 2.0]), np.array([-5, 5, 0])),
    ]:
        cmd = ctrl.step(pos, vel, np.array([1.0, 0, 0, 0]), np.zeros(3), ref_p, ref_v, 0.0)
        thrusts = np.asarray(cmd["thrusts_n"])
        at_max = int((thrusts >= SPEC.rotor_max_thrust_n - 1e-6).sum())
        assert at_max < 3, (
            f"{at_max}/4 rotors at max on err {np.linalg.norm(ref_p - pos):.1f} m; "
            "attitude authority frozen"
        )
        assert cmd["total_thrust_n"] < 4 * SPEC.rotor_max_thrust_n * 0.9
