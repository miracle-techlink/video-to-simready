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

# rotor hubs in the body frame, straight from the joint localPos0 values
ROTOR_HUBS = np.array(
    [
        [-0.20673409, -0.1375972, 0.023],   # rotor0 (prop ccw)
        [0.21869345, 0.12500195, 0.023],    # rotor1 (prop ccw, hub frame rotated 90 deg)
        [0.2032209, -0.13830696, 0.023],    # rotor2 (prop cw)
        [-0.219, 0.126, 0.023],             # rotor3 (prop cw), joint3 localPos
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
    name="3dr_iris_pegasus_e13dc65",
)

IRIS_PRIM_PATH = "/World/Iris/vehicle"
IRIS_BODY_LINK = "/World/Iris/vehicle/body/body"
IRIS_ROTOR_PRIMS = [f"/World/Iris/vehicle/rotor{i}" for i in range(4)]


def rotor_hubs() -> np.ndarray:
    """Rotor hub positions (4,3) in the body frame, from the pinned asset."""
    return ROTOR_HUBS.copy()


def mixer_from_hubs(hubs: np.ndarray) -> np.ndarray:
    """Geometry-derived mixer A: wrench = A @ thrusts for hubs (4,3).

    Rows: total thrust, tau_x, tau_y (from r_i x f_i over the REAL hub
    positions), tau_z (declared reaction pattern ccw,cw,cw,ccw alternating
    on the X quad: rotor0 ccw, rotor1 ccw-hub-rotated, rotor2 cw, rotor3 cw
    → reaction torques -,+,-,+ by spin sign).
    """
    torque_rows = []
    for axis in range(3):
        row = []
        for i in range(4):
            arm = np.zeros(3)
            arm[axis] = 1.0
            row.append(float(np.cross(hubs[i], np.array([0.0, 0.0, 1.0])) @ arm))
        torque_rows.append(row)
    reaction = [-1.0, 1.0, -1.0, 1.0]  # by prop ccw/cw assignment
    return np.vstack([np.ones(4), np.array(torque_rows), reaction])
