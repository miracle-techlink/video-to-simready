"""Real 3DR Iris quadcopter spec, extracted from the pinned Pegasus asset.

Values below are read from the pinned iris.usd (commit e13dc65, BSD-3-Clause,
PX4 Development Team / Lorenz Meier / Thomas Gubler — notice in
iris-license.rst, kept beside the asset). This is a real-airframe model:
meshes, rotors on Z revolute joints, ccw/cw prop pairs, authored mass/inertia.

Extracted facts (iris.usda audit 2026-11-15):
  /vehicle root            : RigidBodyAPI + MassAPI, mass 1.5 kg,
                             diagonal inertia (0.029125, 0.029125, 0.055225)
  rotor0..3                : rigid bodies on PhysicsRevoluteJoint axis Z
                             attached to /vehicle/body/body at the positions
                             in ROTOR_HUBS below; prop meshes iris_prop_ccw /
                             iris_prop_cw mark spin directions
  materials                : embedded Look prims referencing NVIDIA MDL
                             (Carpet_Gray / Linen_Blue) — resolved or locally
                             re-bound at import, never replaced by primitives
"""

from __future__ import annotations

import numpy as np

from .controller import DroneSpec

# rotor hub local centers in the BODY frame — from the rotor Xform translates
# (astra-integration-audit table; the joint localPos0 values live in the
# rotated body0 joint frame and must NOT be used directly for mixing)
ROTOR_HUBS = np.array(
    [
        [0.137985, -0.206716, 0.023],   # rotor0 (mesh iris_prop_ccw)
        [-0.125112, 0.218754, 0.023],   # rotor1 (mesh iris_prop_ccw, +90 deg frame)
        [0.138, 0.202576, 0.023],       # rotor2 (mesh iris_prop_cw)
        [-0.124154, -0.222346, 0.023],  # rotor3 (mesh iris_prop_cw)
    ]
)

IRIS_SPEC = DroneSpec(
    mass_kg=1.5,
    # effective arm half-projection used by the mixer: mean |x|,|y| of hubs
    arm_half_xy=0.1715,
    # 3DR Iris class hover ~1.2-1.4 kg-force total; keep TWR ~2 with margin
    rotor_max_thrust_n=7.5,
    body_size=(0.35, 0.35, 0.13),
    envelope=(0.55, 0.55, 0.25),
    prop_radius=0.095,
    inertia_diag=(0.029125, 0.029125, 0.055225),
    # total flight mass incl. rotors is queried at runtime (rotor masses are
    # density-derived, body 1.5 kg is authored); spec value is the fallback
    name="3dr_iris_pegasus_e13dc65",
)

# AddReference maps the asset defaultPrim (/vehicle) ONTO the referencing
# prim: composed paths live directly under /World/Iris (no /vehicle level)
IRIS_PRIM_PATH = "/World/Iris"
IRIS_BODY_LINK = "/World/Iris/body"  # authored rigid body (audit)
IRIS_ROTOR_PRIMS = [f"/World/Iris/rotor{i}" for i in range(4)]


def rotor_hubs() -> np.ndarray:
    """Rotor hub positions (4,3) in the body frame, from the pinned asset."""
    return ROTOR_HUBS.copy()


# declared spin signs for mixing AND the visual spin loop (single source):
# diagonal pairs share direction; absolute truth is validated in smoke by
# yaw-tracking behavior (flip both signs together if yaw diverges)
REACTION_PATTERN = np.array([1.0, 1.0, -1.0, -1.0])


def mixer_from_hubs(hubs: np.ndarray) -> np.ndarray:
    """Geometry-derived mixer A: wrench = A @ thrusts for hubs (4,3).

    Rows: total thrust, tau_x, tau_y (from r_i x f_i over the REAL hub
    positions), tau_z (declared reaction pattern — alternating spin signs on
    the diagonal pairs, sign convention validated by smoke yaw behavior).
    """
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    row_x = [float(np.cross(hubs[i], up) @ ex) for i in range(4)]
    row_y = [float(np.cross(hubs[i], up) @ ey) for i in range(4)]
    return np.vstack([np.ones(4), row_x, row_y, REACTION_PATTERN])
