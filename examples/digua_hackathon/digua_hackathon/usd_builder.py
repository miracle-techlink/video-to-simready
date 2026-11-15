"""USD authoring for the synthetic substation scene and quadcopter body.

Requires ``pxr`` (present in the Isaac env without launching a SimulationApp,
so scene generation itself is a CPU-only step). Meshes are simple analytic
shapes (boxes/cylinders) with flat displayColor — deliberately lightweight so
the demo renders at interactive rates on one L40.

Authoring facts recorded in the stage metadata: synthetic origin, generator
version, metersPerUnit=1, Z-up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .controller import SPEC, rotor_positions
from .layout import (
    Box,
    CORRIDOR_COLOR,
    CORRIDOR_WIDTH,
    Cylinder,
    FENCE_RECT,
    GROUND_SIZE,
    GRAVEL_COLOR,
    substation_layout,
)

GENERATOR_VERSION = "digua-hackathon-drone v1"


def _set_display_color(prim, color) -> None:
    """Author primvars:displayColor directly (DisplayColorAPI removed in USD 0.26)."""
    from pxr import Gf, Sdf

    attr = prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, True
    )
    attr.Set([Gf.Vec3f(*color)])



def _add_box(stage, parent_path: str, name: str, prim: Box) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    path = f"{parent_path}/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3f(*prim.center))
    cube.AddScaleOp().Set(Gf.Vec3f(*prim.size))
    _set_display_color(cube.GetPrim(), prim.color)
    if prim.collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _add_cylinder(stage, parent_path: str, name: str, prim: Cylinder) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = f"{parent_path}/{name}"
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(prim.radius)
    cyl.CreateHeightAttr(prim.height)
    # pxr cylinders are Y-up by default; scene is Z-up
    cyl.AddRotateXOp().Set(90.0)
    cyl.AddTranslateOp().Set(Gf.Vec3f(*prim.center))
    _set_display_color(cyl.GetPrim(), prim.color)
    if prim.collision:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())


def build_substation(stage) -> dict[str, object]:
    """Author the synthetic substation into an OPEN stage (runtime-assembly
    path: referenced/composed stage files freeze physics-driven rendering on
    this Isaac build; session-authored prims animate correctly)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    world = UsdGeom.Scope.Define(stage, "/Substation")

    # ground: gravel plane + darker inspection corridor strip
    ground = UsdGeom.Cube.Define(stage, "/Substation/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(*GROUND_SIZE, 0.1))
    _set_display_color(ground.GetPrim(), GRAVEL_COLOR)
    # static collider: CollisionAPI only (RigidBodyAPI would make it dynamic)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    corridor = UsdGeom.Cube.Define(stage, "/Substation/Corridor")
    corridor.CreateSizeAttr(1.0)
    corridor.AddTranslateOp().Set(Gf.Vec3f(0.0, 9.0, 0.011))
    corridor.AddScaleOp().Set(Gf.Vec3f(GROUND_SIZE[0] - 8.0, CORRIDOR_WIDTH, 0.02))
    _set_display_color(corridor.GetPrim(), CORRIDOR_COLOR)

    prims = substation_layout()
    counters: dict[str, int] = {}
    for prim in prims:
        kind = prim.kind
        counters[kind] = counters.get(kind, 0) + 1
        if isinstance(prim, Box):
            _add_box(stage, "/Substation", f"{kind}_{counters[kind]:03d}", prim)
        else:
            _add_cylinder(stage, "/Substation", f"{kind}_{counters[kind]:03d}", prim)

    # sky dome + sun for pleasant path-traced exteriors
    dome = UsdLux.DomeLight.Define(stage, "/Substation/SkyDome")
    dome.CreateIntensityAttr(900.0)
    sun = UsdLux.DistantLight.Define(stage, "/Substation/Sun")
    sun.CreateIntensityAttr(550.0)
    sun.AddRotateXOp().Set(-38.0)
    sun.AddRotateZOp().Set(28.0)

    # physics scene with fixed-step dt (also used by the drone demo run)
    scene = UsdPhysics.Scene.Define(stage, "/Substation/PhysicsScene")
    scene.CreateGravityMagnitudeAttr(9.81)

    xs, ys = [], []
    for p in prims:
        xs.append(p.center[0])
        ys.append(p.center[1])
    return {
        "path": None,
        "generator": GENERATOR_VERSION,
        "origin": "synthetic_code_generated_runtime_assembly",
        "prim_kinds": counters,
        "layout_extent_centered_xy": [
            float(min(xs)),
            float(min(ys)),
            float(max(xs)),
            float(max(ys)),
        ],
        "fence_rect": list(FENCE_RECT),
    }


def build_drone(stage, parent_path: str = "/World/Drone", start_position=None) -> dict[str, object]:
    """Author the quadcopter rigid body into an OPEN stage at parent_path."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, parent_path)
    if start_position is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*start_position))
    body = UsdGeom.Cube.Define(stage, f"{parent_path}/Body")
    body.CreateSizeAttr(1.0)
    body.AddScaleOp().Set(Gf.Vec3f(*SPEC.body_size))
    _set_display_color(body.GetPrim(), (0.85, 0.30, 0.08))
    battery = UsdGeom.Cube.Define(stage, f"{parent_path}/Battery")
    battery.CreateSizeAttr(1.0)
    battery.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.045))
    battery.AddScaleOp().Set(Gf.Vec3f(0.16, 0.16, 0.05))
    _set_display_color(battery.GetPrim(), (0.1, 0.1, 0.1))
    # camera pod under the nose (declared sensor mount, view rendered by child cam)
    pod = UsdGeom.Cube.Define(stage, f"{parent_path}/CameraPod")
    pod.CreateSizeAttr(1.0)
    pod.AddTranslateOp().Set(Gf.Vec3f(0.21, 0.0, -0.055))
    pod.AddScaleOp().Set(Gf.Vec3f(0.08, 0.08, 0.07))
    _set_display_color(pod.GetPrim(), (0.05, 0.05, 0.06))
    # FPV camera on the pod mount: looks along body +x with 20 deg down-tilt
    # (USD cameras look down local -z; RotateY(-70deg) yields forward+down)
    fpv = UsdGeom.Camera.Define(stage, f"{parent_path}/FPVCamera")
    fpv.CreateFocalLengthAttr(2.79)
    fpv.CreateHorizontalApertureAttr(3.896)
    fpv.AddRotateYOp().Set(-70.0)
    fpv.AddTranslateOp().Set(Gf.Vec3f(0.21, 0.0, -0.055))

    for i, (rx, ry, _) in enumerate(rotor_positions()):
        arm = UsdGeom.Cube.Define(stage, f"{parent_path}/Arm_{i}")
        mid = np.array([rx, ry, 0.0]) / 2
        arm_len = float(np.linalg.norm([rx, ry]))
        arm.CreateSizeAttr(1.0)
        arm.AddTranslateOp().Set(Gf.Vec3f(*mid))
        arm.AddRotateZOp().Set(float(np.degrees(np.arctan2(ry, rx))))
        arm.AddScaleOp().Set(Gf.Vec3f(arm_len, 0.035, 0.025))
        _set_display_color(arm.GetPrim(), (0.2, 0.2, 0.22))
        hub = UsdGeom.Cylinder.Define(stage, f"{parent_path}/Motor_{i}")
        hub.CreateRadiusAttr(0.035)
        hub.CreateHeightAttr(0.05)
        hub.AddRotateXOp().Set(90.0)
        hub.AddTranslateOp().Set(Gf.Vec3f(rx, ry, 0.035))
        _set_display_color(hub.GetPrim(), (0.12, 0.12, 0.13))
        disk = UsdGeom.Cylinder.Define(stage, f"{parent_path}/RotorDisk_{i}")
        disk.CreateRadiusAttr(SPEC.prop_radius)
        disk.CreateHeightAttr(0.008)
        # op order rotateX->translate->rotateZ: the last op is the local spin
        # the demo animates cosmetically at runtime
        disk.AddRotateXOp().Set(90.0)
        disk.AddTranslateOp().Set(Gf.Vec3f(rx, ry, 0.075))
        disk.AddRotateZOp().Set(0.0)
        _set_display_color(disk.GetPrim(), (0.09, 0.09, 0.10))

    # physics: rigid body root with envelope collision box
    rb = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(rb)
    UsdPhysics.CollisionAPI.Apply(rb)
    mass_api = UsdPhysics.MassAPI.Apply(rb)
    mass_api.CreateMassAttr(SPEC.mass_kg)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*SPEC.inertia_diag))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    collision_box = UsdGeom.Cube.Define(stage, f"{parent_path}/CollisionEnvelope")
    collision_box.CreateSizeAttr(1.0)
    collision_box.AddScaleOp().Set(Gf.Vec3f(*SPEC.envelope))
    UsdPhysics.CollisionAPI.Apply(collision_box.GetPrim())

    return {
        "path": None,
        "name": SPEC.name,
        "mass_kg": SPEC.mass_kg,
        "arm_half_xy_m": SPEC.arm_half_xy,
        "rotor_max_thrust_n": SPEC.rotor_max_thrust_n,
        "envelope_m": list(SPEC.envelope),
        "inertia_diag_kgm2": list(SPEC.inertia_diag),
        "generator": GENERATOR_VERSION,
        "origin": "synthetic_code_generated",
    }


def write_substation_usd(path: Path) -> dict[str, object]:
    """Provenance asset: build_substation into a fresh standalone layer."""
    from pxr import Usd

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)
    meta = build_substation(stage)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Substation"))
    stage.GetRootLayer().Save()
    meta["path"] = str(path)
    return meta


def write_drone_usd(path: Path) -> dict[str, object]:
    """Provenance asset: build_drone into a fresh standalone layer."""
    from pxr import Usd

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)
    meta = build_drone(stage, "/Quadcopter")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Quadcopter"))
    stage.GetRootLayer().Save()
    meta["path"] = str(path)
    return meta
