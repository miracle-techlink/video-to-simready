"""Quadcopter stable controller: position PD -> attitude PD -> rotor mixer.

This is a hand-tuned geometric PD controller (Lee et al. style), NOT a
trained policy. It consumes simulated state feedback (pose + velocities from
PhysX) and emits per-rotor thrust commands that the demo applies as forces
on the simulated rigid body. Waypoints come from the pre-planned route in
``layout.py``; there is no perception, mapping, or obstacle avoidance here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81  # m/s^2, matches the Isaac physics scene


@dataclass(frozen=True)
class DroneSpec:
    """Quadcopter body parameters shared by USD authoring and the controller."""

    mass_kg: float = 2.0
    arm_half_xy: float = 0.24  # rotor plane offset |x|=|y|=arm_half_xy
    rotor_max_thrust_n: float = 12.0
    body_size: tuple[float, float, float] = (0.30, 0.30, 0.12)
    envelope: tuple[float, float, float] = (0.62, 0.62, 0.22)  # collision box
    prop_radius: float = 0.115
    # Diagonal inertia about the rotor plane (box + arms lumped), kg*m^2.
    inertia_diag: tuple[float, float, float] = (0.030, 0.030, 0.055)
    name: str = "synthetic_quad_x0.68"


SPEC = DroneSpec()


def rotor_positions(spec: DroneSpec = SPEC) -> np.ndarray:
    """Rotor hub positions in the body frame, X configuration (4, 3)."""
    l = spec.arm_half_xy
    return np.array(
        [[l, l, 0.0], [-l, l, 0.0], [-l, -l, 0.0], [l, -l, 0.0]]
    )


class PositionAttitudeController:
    """Cascaded PD tracking of a reference position/velocity/yaw trajectory."""

    def __init__(
        self,
        spec: DroneSpec = SPEC,
        kp_pos: float = 4.0,
        kd_pos: float = 2.4,
        kp_att: float = 9.0,
        kd_att: float = 3.2,
        max_tilt_deg: float = 30.0,
    ) -> None:
        self.spec = spec
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.max_tilt = float(np.radians(max_tilt_deg))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        """Body-to-world rotation from a (w, x, y, z) unit quaternion."""
        w, x, y, z = np.asarray(quat_wxyz, dtype=float) / np.linalg.norm(quat_wxyz)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def desired_rotation(thrust_dir: np.ndarray, yaw_rad: float) -> np.ndarray:
        """Body-to-world rotation whose +z is thrust_dir and heading is yaw."""
        z_b = thrust_dir / np.linalg.norm(thrust_dir)
        x_c = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        y_b = np.cross(z_b, x_c)
        if np.linalg.norm(y_b) < 1e-6:  # thrust parallel to heading: pick y
            y_b = np.array([0.0, 0.0, 1.0]) if abs(z_b[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y_b = y_b / np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        return np.column_stack([x_b, y_b, z_b])

    @staticmethod
    def attitude_error_vee(rot_err: np.ndarray) -> np.ndarray:
        """Vee map of a skew rotation error matrix (small-angle usable)."""
        return 0.5 * np.array(
            [rot_err[2, 1] - rot_err[1, 2], rot_err[0, 2] - rot_err[2, 0], rot_err[1, 0] - rot_err[0, 1]]
        )

    # -- main step ---------------------------------------------------------

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        quat_wxyz: np.ndarray,
        angular_velocity_body: np.ndarray,
        ref_position: np.ndarray,
        ref_velocity: np.ndarray,
        ref_yaw: float,
    ) -> dict[str, object]:
        """One 240 Hz control step from simulated state to rotor thrusts."""
        acc = (
            self.kp_pos * (ref_position - position)
            + self.kd_pos * (ref_velocity - velocity)
        )
        acc[2] += GRAVITY
        acc[:2] = np.clip(acc[:2], -6.0, 6.0)
        thrust_vec = self.spec.mass_kg * acc
        total = float(np.linalg.norm(thrust_vec))
        if total < 1e-3:
            thrust_vec = np.array([0.0, 0.0, self.spec.mass_kg * GRAVITY])
            total = float(self.spec.mass_kg * GRAVITY)
        z_des = thrust_vec / total
        # clamp tilt: blend desired z toward vertical when over-tilted
        tilt = float(np.arccos(np.clip(z_des[2], -1.0, 1.0)))
        if tilt > self.max_tilt:
            blend = self.max_tilt / tilt
            z_des = (1 - blend) * np.array([0.0, 0.0, 1.0]) + blend * z_des
            z_des /= np.linalg.norm(z_des)

        rot = self.rotation_matrix(quat_wxyz)
        rot_d = self.desired_rotation(z_des, ref_yaw)
        torque, e_rot = self.attitude_torque(rot, rot_d, angular_velocity_body)

        thrusts, saturated = self.mixer(total, torque)
        wrench = self.wrench_from_thrusts(thrusts)
        return {
            "thrusts_n": thrusts.tolist(),
            "total_thrust_n": float(wrench[0]),
            "torque_body": wrench[1].tolist(),
            "attitude_error": e_rot.tolist(),
            "position_error_m": float(np.linalg.norm(ref_position - position)),
            "saturated": bool(saturated),
        }

    def attitude_torque(
        self, rot: np.ndarray, rot_d: np.ndarray, angular_velocity_body: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """PD attitude torque toward rot_d; returns (torque, attitude_error).

        Sign verified by unit test: for Rd=Ry(+theta) from identity this must
        yield +y torque (omega=+y tilts body z toward +x, matching Rd).
        """
        e_rot = self.attitude_error_vee(rot_d @ rot.T - rot @ rot_d.T)
        return self.kp_att * e_rot - self.kd_att * angular_velocity_body, e_rot

    def mixer(
        self, total_thrust: float, torque: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Map total thrust + body torque to four rotor thrusts (X config).

        Rotor order: 0(+x+y), 1(-x+y), 2(-x-y), 3(+x-y); all thrust +z.
        Torque authority comes from rotor moment arms; rotor reaction torques
        about their spin axes are modeled as direct yaw authority, documented
        as a simplification (their magnitude is negligible at demo speeds).
        """
        l = self.spec.arm_half_xy
        # inv(A) for the geometry-derived A in wrench_from_thrusts():
        # rotor order 0(+x+y), 1(-x+y), 2(-x-y), 3(+x-y), all thrust +z.
        inv = np.array(
            [
                [0.25, 0.25 / l, -0.25 / l, 0.25],
                [0.25, 0.25 / l, 0.25 / l, -0.25],
                [0.25, -0.25 / l, 0.25 / l, 0.25],
                [0.25, -0.25 / l, -0.25 / l, -0.25],
            ]
        )
        thrusts = inv @ np.array([total_thrust, torque[0], torque[1], torque[2]])
        clipped = np.clip(thrusts, 0.0, self.spec.rotor_max_thrust_n)
        saturated = bool(not np.allclose(clipped, thrusts, atol=1e-6))
        return clipped, saturated

    def wrench_from_thrusts(
        self, thrusts: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Net force and body torque actually produced by rotor thrusts.

        Torque rows are r_i x f_i over the geometry in rotor_positions():
        tx = l*(f0+f1-f2-f3), ty = l*(-f0+f1+f2-f3); tz is the assigned
        rotor reaction-torque pattern (f0-f1+f2-f3).
        """
        l = self.spec.arm_half_xy
        a = np.array(
            [
                [1, 1, 1, 1],
                [l, l, -l, -l],
                [-l, l, l, -l],
                [1, -1, 1, -1],
            ]
        )
        wrench = a @ thrusts
        return float(wrench[0]), wrench[1:]
