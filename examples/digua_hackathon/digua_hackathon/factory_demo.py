"""Factory + real Iris drone demo: assemble, fly, record (Digua redo, 2026-11-15).

Assets (pinned, licenses kept beside them — see factory-drone-redo handoff):
  * Scene: NVIDIA USD Explorer Sample "automotive factory" (CAD-authored
    assembly-plant sample; NOT a scan of a named real plant; Omniverse terms,
    local use, no redistribution). Centimeter stage (metersPerUnit=0.01)
    referenced under a 0.01 scale in a METER wrapper stage.
  * Drone: Pegasus/PX4 3DR Iris, commit e13dc65, BSD-3-Clause (PX4 Dev Team /
    Meier / Gubler). Real meshes, rotors on Z revolute joints, authored
    mass 1.5 kg / inertia. External NVIDIA Carpet_Gray/Linen_Blue material
    textures are re-pointed to locally cached copies (never primitives).

Honesty: waypoint tracking with a hand-written PD controller on the REAL
airframe dynamics (mass/inertia/rotor joints from the asset); rotor thrust
is a simplified wrench model applied to the body link; rotor spin uses small
alternating joint torques (visual + physically-consistent reaction torques).
NOT a trained policy, NOT autonomous perception/avoidance. The factory mesh
has no collision primitives; flight safety comes from clearance-planned
corridors, and contacts are logged (ground plane collides).

Usage (server Isaac env):
  python -m digua_hackathon.factory_demo --mode preview --out runs/factory
  python -m digua_hackathon.factory_demo --mode smoke   --out runs/factory
  python -m digua_hackathon.factory_demo --mode record  --out runs/factory
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .controller import PositionAttitudeController
from .iris_spec import (IRIS_BODY_LINK, IRIS_SPEC, REACTION_PATTERN,
                      mixer_from_hubs, rotor_hubs)

ASSETS_ROOT = Path("/root/gpufree-data/.simready/assets/factory-iris")
FACTORY_USD = ASSETS_ROOT / "extracted/Usd_Explorer/Samples/Examples/2023_2/Factory/Factory.usd"
IRIS_USD = ASSETS_ROOT / "iris/iris.usd"
MATERIALS = ASSETS_ROOT / "materials"

PHYSICS_DT = 1.0 / 240.0
RENDER_FPS = 30
ROTOR_SPIN_TORQUE = 0.004  # N*m, alternating; visual spin + real reaction

# Factory world bounds (meters, from parse-report.json cm bounds / 100):
#   x [-17.0, 19.4], y [-30.6, 30.5], z [-0.3, 12.8]
# Draft corridor: center aisle at x~0, z>=4 keeps above assembly equipment
# pending the helper's rendered layout confirmation; margins >=5 m to
# walls/roof everywhere.
FACTORY_ROUTE = [
    {"pos": [-6.0, 0.0, 2.0], "hold": 2.0, "label": "takeoff_check"},
    {"pos": [-6.0, 0.0, 3.8], "hold": 0.0, "label": "climb"},
    {"pos": [-6.0, -12.0, 4.2], "hold": 0.0, "label": "approach_aisle"},
    {"pos": [-6.0, -20.0, 4.2], "hold": 1.0, "label": "station_south_line"},
    {"pos": [0.0, -8.0, 4.2], "hold": 1.0, "label": "station_mid_west"},
    {"pos": [0.0, 8.0, 4.2], "hold": 1.0, "label": "station_mid_east"},
    {"pos": [-6.0, 18.0, 4.2], "hold": 1.0, "label": "station_north_line"},
    {"pos": [-6.0, 24.0, 6.0], "hold": 0.0, "label": "climb_overview"},
    {"pos": [-6.0, -18.0, 7.5], "hold": 0.0, "label": "high_return"},
    {"pos": [-6.0, 0.0, 2.6], "hold": 2.0, "label": "final_hover"},
]
ROUTE_SPEED = 3.0  # m/s (indoor factory: conservative cruise
ROUTE_ACCEL = 2.5  # m/s^2

MODE_SETTINGS = {
    "preview": {"width": 1280, "height": 720, "duration_s": 0.0, "rt_subframes": 4,
                "encode": False, "physics": False},
    "smoke": {"width": 640, "height": 360, "duration_s": 12.0, "rt_subframes": 1,
              "encode": False, "physics": True},
    "record": {"width": 1280, "height": 720, "duration_s": 33.0, "rt_subframes": 2,
               "encode": True, "physics": True},
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Stage assembly (runtime; meter wrapper; unit + material handling explicit)
# ---------------------------------------------------------------------------


def _repoint_iris_materials(stage, iris_root: str, materials_dir: Path) -> int:
    """Repoint external NVIDIA material asset paths to local cached files."""
    from pxr import Sdf

    local_by_name: dict[str, str] = {}
    for sub in materials_dir.iterdir() if materials_dir.exists() else []:
        if sub.is_dir():
            for f in sub.iterdir():
                local_by_name[f.name] = str(f)
    if not local_by_name:
        return 0
    count = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(iris_root):
            continue
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            val = attr.Get()
            if val is None or not getattr(val, "path", ""):
                continue
            name = Path(str(val.path)).name
            if name in local_by_name:
                attr.Set(Sdf.AssetPath(local_by_name[name]))
                count += 1
    return count




def _zero_saved_velocities(stage, root: str) -> int:
    """Zero physics:linearVelocity / physics:angularVelocity under root."""
    from pxr import UsdPhysics

    count = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        for name in ("physics:velocity", "physics:angularVelocity"):
            attr = prim.GetAttribute(name)
            if attr and attr.Get() not in (None, (0.0, 0.0, 0.0)):
                attr.Set((0.0, 0.0, 0.0))
                count += 1
    return count


def build_factory_stage(stage, factory_usd: Path, iris_usd: Path, start: np.ndarray) -> dict:
    """Assemble factory (0.01-scaled reference) + Iris (1.0) in a meter stage."""
    from pxr import Gf, UsdGeom, UsdPhysics

    # author local ops BEFORE adding references: the referenced roots carry
    # their own xformOpOrder, and AddScaleOp/Translate afterwards collides
    # with the composed op names. Local (stronger) order overrides cleanly.
    factory = UsdGeom.Xform.Define(stage, "/World/Factory")
    factory.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))  # cm layer -> meters
    factory.GetPrim().GetReferences().AddReference(str(factory_usd))

    iris = UsdGeom.Xform.Define(stage, "/World/Iris")
    iris.AddTranslateOp().Set(Gf.Vec3d(*start))
    iris.GetPrim().GetReferences().AddReference(str(iris_usd))

    # physics-only ground plane at z=0 (invisible; the factory floor mesh has
    # no collision of its own)
    ground = UsdGeom.Cube.Define(stage, "/World/GroundCollider")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.1))
    ground.AddScaleOp().Set(Gf.Vec3f(80.0, 80.0, 0.2))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdGeom.Imageable(ground.GetPrim()).MakeInvisible()

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityMagnitudeAttr(9.81)

    mats = _repoint_iris_materials(stage, "/World/Iris", MATERIALS)
    return {"factory_reference": str(factory_usd), "iris_reference": str(iris_usd),
            "factory_scale": 0.01, "material_paths_repointed": mats}


# ---------------------------------------------------------------------------
# Smooth reference (same generator discipline as the substation demo)
# ---------------------------------------------------------------------------


class RouteTracker:
    def __init__(self, route: list[dict], speed: float, accel: float) -> None:
        self.route = [np.array(r["pos"], dtype=float) for r in route]
        self.holds = [float(r.get("hold", 0.0)) for r in route]
        self.labels = [r.get("label", "") for r in route]
        self.speed = speed
        self.accel = accel
        self._samples = self._build()

    def _bang(self, t: float) -> np.ndarray:
        left = self.holds[0]
        if t < left:
            return self.route[0]
        for a, b, hb in zip(self.route, self.route[1:], self.holds[1:]):
            leg = float(np.linalg.norm(b - a))
            travel = leg / self.speed
            if t < left + travel:
                d = (b - a) / leg
                return a + d * (t - left) * self.speed
            left += travel
            if t < left + hb:
                return b
            left += hb
        return self.route[-1]

    def _build(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        total = self.holds[0]
        for a, b, hb in zip(self.route, self.route[1:], self.holds[1:]):
            total += np.linalg.norm(b - a) / self.speed + hb
        span = total + 4.0
        n = int(span / PHYSICS_DT) + 1
        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        labels = []
        p = self.route[0].copy()
        v_prev = np.zeros(3)
        pos[0] = p
        labels.append(self.labels[0])
        for i in range(1, n):
            t = i * PHYSICS_DT
            target = self._bang(t)
            v = (target - p) * 5.0
            sp = np.linalg.norm(v)
            if sp > self.speed:
                v *= self.speed / sp
            dv = np.linalg.norm(v - v_prev)
            if dv > self.accel * PHYSICS_DT:
                v = v_prev + (v - v_prev) * (self.accel * PHYSICS_DT / dv)
            p = p + v * PHYSICS_DT
            pos[i] = p
            vel[i] = v
            v_prev = v
            labels.append(self._label_at(t))
        return pos, vel, labels

    def _label_at(self, t: float) -> str:
        left = self.holds[0]
        if t < left:
            return self.labels[0]
        for i, (a, b, hb) in enumerate(zip(self.route, self.route[1:], self.holds[1:]), start=1):
            left += np.linalg.norm(b - a) / self.speed
            if t < left + hb:
                return self.labels[i]
            left += hb
        return self.labels[-1]

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, str]:
        pos, vel, labels = self._samples
        idx = min(max(int(t / PHYSICS_DT), 0), len(pos) - 1)
        return pos[idx], vel[idx], labels[idx]

    @property
    def duration(self) -> float:
        return (len(self._samples[0]) - 1) * PHYSICS_DT


# ---------------------------------------------------------------------------
# Isaac run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    from isaacsim import SimulationApp

    settings = MODE_SETTINGS[args.mode]
    app = SimulationApp(
        {"headless": True, "limit_cpu_threads": 4, "fast_shutdown": True},
        experience=os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.base.kit"),
    )
    exit_code = 0
    try:
        _simulate(args, settings, app)
    except BaseException as error:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        (args.out / f"failure_{args.mode}.json").write_text(
            json.dumps({"status": "FAILED", "error": str(error), "at": now_iso()}, indent=2)
        )
        exit_code = 1
    finally:
        app.close(exit_code=exit_code)
    return exit_code


def _simulate(args: argparse.Namespace, settings: dict, app) -> None:
    import carb.settings
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep  # noqa: E402

    for key, value in {
        "/rtx/rendermode": "RealTime",
        "/rtx/post/aa/op": 4,
        "/rtx/post/dlss/execMode": 2,
        "/rtx/dldenoiser/enabled": True,
    }.items():
        carb.settings.get_settings().set(key, value)

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    route = FACTORY_ROUTE
    start = np.array(route[0]["pos"], dtype=float)
    assembly = build_factory_stage(stage, Path(args.factory_usd), Path(args.iris_usd), start)

    rep.orchestrator.set_capture_on_play(False)
    width, height = settings["width"], settings["height"]

    # report composed world bounds (unit-conversion evidence)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], False)
    factory_box = bb.ComputeWorldBound(stage.GetPrimAtPath("/World/Factory"))
    iris_box = bb.ComputeWorldBound(stage.GetPrimAtPath("/World/Iris"))
    frange = factory_box.ComputeAlignedRange()
    irange = iris_box.ComputeAlignedRange()
    fmin, fmax = frange.GetMin(), frange.GetMax()
    imin, imax = irange.GetMin(), irange.GetMax()
    bounds = {
        "factory_world_bounds_m": {"min": list(fmin), "max": list(fmax)},
        "iris_world_bounds_m_at_start": {"min": list(imin), "max": list(imax),
                                          "note": "Iris span ~0.55 m expected"},
        "assembly": assembly,
    }
    (args.out / "bounds_report.json").write_text(json.dumps(bounds, indent=2))
    print(f"[{now_iso()}] bounds: factory {list(fmin)}..{list(fmax)} iris {list(imin)}..{list(imax)}",
          flush=True)

    # cameras
    chase = UsdGeom.Camera.Define(stage, "/World/Cameras/Chase")
    chase.CreateFocalLengthAttr(18.0)
    chase.CreateHorizontalApertureAttr(20.955)
    chase.CreateVerticalApertureAttr(20.955 * height / width)
    chase.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
    chase_translate = UsdGeom.Xformable(chase.GetPrim()).AddTransformOp()
    fpv = UsdGeom.Camera.Define(stage, "/World/Cameras/FPV")
    fpv.CreateFocalLengthAttr(2.79)
    fpv.CreateHorizontalApertureAttr(3.896)
    fpv.CreateVerticalApertureAttr(3.896 * height / width)
    fpv.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
    fpv_translate = UsdGeom.Xformable(fpv.GetPrim()).AddTransformOp()

    product_chase = rep.create.render_product(str(chase.GetPath()), (width, height))
    product_fpv = rep.create.render_product(str(fpv.GetPath()), (width, height))

    def make_writer(out_dir: Path, product):
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(out_dir))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        writer.attach([product])
        return writer

    frames_chase = args.out / f"frames_{args.mode}_chase"
    frames_fpv = args.out / f"frames_{args.mode}_fpv"
    writer_chase = make_writer(frames_chase, product_chase)
    writer_fpv = make_writer(frames_fpv, product_fpv)

    def look_at(eye, target, up=None):
        up_vec = Gf.Vec3d(*(up if up is not None else (0, 0, 1)))
        return (Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), up_vec).GetInverse())

    if args.mode == "preview":
        # static multi-shot: overview x2, assembly line, drone close-up
        shots = [
            ("overview_far", np.array([30.0, -45.0, 16.0]), np.array([0, 0, 4.0])),
            ("overview_high", np.array([-24.0, 0.0, 14.0]), np.array([2, 0, 3.0])),
            ("line_level", np.array([0.0, -20.0, 5.0]), np.array([0, 10, 2.5])),
            ("drone_closeup", start + np.array([-1.6, -1.2, 0.7]), start),
        ]
        for i, (name, eye, tgt) in enumerate(shots):
            chase_translate.Set(look_at(eye, tgt))
            carb.settings.get_settings().set("/rtx-transient/post/dlss/forceParamReset", True)
            rep.orchestrator.step(rt_subframes=settings["rt_subframes"],
                                  delta_time=0.0, pause_timeline=False)
        rep.orchestrator.wait_until_complete()
        writer_chase.detach()
        writer_fpv.detach()
        _collect_preview(args.out, frames_chase, shots)
        return

    # --- physics flight (smoke / record) --------------------------------
    from isaacsim.core.experimental.prims import RigidPrim

    for check_path in [IRIS_BODY_LINK] + [f"/World/Iris/rotor{i}" for i in range(4)]:
        if not stage.GetPrimAtPath(check_path).IsValid():
            raise RuntimeError(f"composed Iris prim missing: {check_path}")
    rotor_paths = [f"/World/Iris/rotor{i}" for i in range(4)]
    # contact view: filter against the ground collider; body-only monitoring
    # (rotor contacts not tracked — stated in the run record)
    body = RigidPrim(
        IRIS_BODY_LINK,
        max_contact_count=64,
        contact_filter_paths=["/World/GroundCollider"],
    )
    rotors = [RigidPrim(pth, max_contact_count=0) for pth in rotor_paths]
    # zero the asset's saved nonzero velocities at USD level (authoritative;
    # angularVelocity in USD is deg/s) before physics initialization
    zeroed = _zero_saved_velocities(stage, "/World/Iris")
    print(f"[{now_iso()}] velocities zeroed on {zeroed} rigid bodies", flush=True)
    SimulationManager.enable_fabric(False)
    SimulationManager.set_physics_dt(PHYSICS_DT)
    SimulationManager.initialize_physics()

    from pxr import PhysxSchema

    body_prim = stage.GetPrimAtPath(IRIS_BODY_LINK)
    if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI) is False:
        PhysxSchema.PhysxContactReportAPI.Apply(body_prim)

    # runtime mass: body + rotors from the tensor API (rotor masses are
    # density-derived; the authored 1.5 kg covers the body only)
    import dataclasses

    runtime_mass = None
    try:
        masses = [float(np.asarray(v.get_masses().numpy(), dtype=float)[0])
                  for v in [body, *rotors]]
        runtime_mass = float(sum(masses))
        print(f"[{now_iso()}] runtime masses body+rotors: {masses} sum={runtime_mass}",
              flush=True)
    except Exception as error:
        print(f"[{now_iso()}] mass query FAILED ({error}); refusing to fly on a "
              "guessed mass", flush=True)
        raise
    if not 0.5 < runtime_mass < 5.0:
        raise RuntimeError(f"implausible runtime mass {runtime_mass} kg")
    flight_spec = dataclasses.replace(IRIS_SPEC, mass_kg=runtime_mass)
    controller = PositionAttitudeController(
        spec=flight_spec, mixer_matrix=mixer_from_hubs(rotor_hubs()),
        max_tilt_deg=25.0,
    )
    tracker = RouteTracker(route, ROUTE_SPEED, ROUTE_ACCEL)

    duration = settings["duration_s"] or min(36.0, tracker.duration + 2.0)
    total_frames = int(round(duration * RENDER_FPS))
    ticks_per_frame = round((1.0 / RENDER_FPS) / PHYSICS_DT)

    # warmup with writers detached
    writer_chase.detach()
    writer_fpv.detach()
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
    rep.orchestrator.wait_until_complete()
    writer_chase.attach([product_chase])
    writer_fpv.attach([product_fpv])

    rows: list[dict] = []
    contact_view_ok = True
    contacts_log: list[dict] = []
    abort = None
    last_yaw = 0.0
    spin_sign = REACTION_PATTERN  # same pattern as the mixer yaw row
    started = time.perf_counter()

    def state():
        pos, quat = body.get_world_poses()
        lin, ang = body.get_velocities()
        q = np.asarray(quat.numpy(), dtype=float)[0]
        r = PositionAttitudeController.rotation_matrix(q)
        return (np.asarray(pos.numpy(), dtype=float)[0], q,
                np.asarray(lin.numpy(), dtype=float)[0],
                np.asarray(ang.numpy(), dtype=float)[0], r)

    tick = 0
    for frame in range(total_frames):
        for _ in range(ticks_per_frame):
            sim_t = tick * PHYSICS_DT  # one global clock: ref, state, stamp
            p, q, v, w_world, rot = state()
            w_body = rot.T @ w_world
            ref_p, ref_v, label = tracker.at(sim_t)
            if np.linalg.norm(ref_v) > 0.05:
                dyaw = (np.arctan2(ref_v[1], ref_v[0]) - last_yaw + np.pi) % (2 * np.pi) - np.pi
                last_yaw += float(np.clip(dyaw, -controller.YAW_RATE_LIMIT_RADS * PHYSICS_DT,
                                          controller.YAW_RATE_LIMIT_RADS * PHYSICS_DT))
            cmd = controller.step(p, v, q, w_body, ref_p, ref_v, last_yaw)
            force = rot @ np.array([0.0, 0.0, cmd["total_thrust_n"]])
            torque = rot @ np.asarray(cmd["torque_body"], dtype=float)
            body.apply_forces_and_torques_at_pos(forces=force.reshape(1, 3),
                                                 torques=torque.reshape(1, 3))
            # rotor visual spin: speed-loop torque about each rotor's spin
            # axis measured in the BODY frame (rotors share the body z axis)
            for r_i, spin in zip(rotors, spin_sign):
                _, r_ang = r_i.get_velocities()
                w_r_world = np.asarray(r_ang.numpy(), dtype=float)[0]
                w_r_body = rot.T @ w_r_world
                tau_spin = ROTOR_SPIN_TORQUE * spin * (30.0 - spin * w_r_body[2]) / 30.0
                r_i.apply_forces_and_torques_at_pos(
                    torques=(rot @ np.array([0.0, 0.0, tau_spin])).reshape(1, 3))
            SimulationManager.step()
            tick += 1
            if contact_view_ok:
                try:
                    cf = body.get_net_contact_forces(dt=PHYSICS_DT)
                    contact_n = float(np.linalg.norm(np.asarray(cf.numpy(), dtype=float)[0]))
                except Exception as error:
                    contact_view_ok = False
                    contact_n = None
                    print(f"[{now_iso()}] contact view FAILED mid-run: {error}",
                          flush=True)
            else:
                contact_n = None
            if contact_n is not None and contact_n > 0.5 and (
                not contacts_log or sim_t - contacts_log[-1]["t"] > 0.5
            ):
                contacts_log.append({"t": round(sim_t, 3), "force_n": round(contact_n, 2)})
            rows.append({
                "t": round(sim_t, 4), "label": label,
                "ref_x": ref_p[0], "ref_y": ref_p[1], "ref_z": ref_p[2],
                "pos_x": p[0], "pos_y": p[1], "pos_z": p[2],
                "thrust_n": round(cmd["total_thrust_n"], 3),
                "tracking_error_m": round(cmd["position_error_m"], 4),
                "contact_n": (round(contact_n, 2) if contact_n is not None else None),
                "saturated": int(cmd["saturated"]),
            })
            if not np.isfinite(p).all() or cmd["position_error_m"] > 5.0 or p[2] < 0.05:
                abort = (f"aborted t={sim_t:.2f}s err={cmd['position_error_m']:.2f} "
                         f"z={p[2]:.2f}")
                break
        if abort:
            break

        p, _, _, _, rot = state()
        heading = np.array([np.cos(last_yaw), np.sin(last_yaw), 0.0])
        eye = p - heading * 3.2 + np.array([0.3, -0.9, 1.2])
        chase_translate.Set(look_at(eye, p + heading * 0.8))
        eye_f = p + rot @ np.array([0.16, 0.0, -0.04])
        fwd_f = rot @ np.array([0.9397, 0.0, -0.3420])
        fpv_translate.Set(look_at(eye_f, eye_f + fwd_f, rot @ np.array([0, 0, 1.0])))

        carb.settings.get_settings().set("/rtx-transient/post/dlss/forceParamReset", True)
        rep.orchestrator.step(rt_subframes=settings["rt_subframes"],
                              delta_time=0.0, pause_timeline=False)
        if frame % 30 == 0:
            print(f"[{now_iso()}] frame {frame}/{total_frames} t={tick * PHYSICS_DT:.1f}s", flush=True)

    writer_chase.detach()
    writer_fpv.detach()
    rep.orchestrator.wait_until_complete()
    wall = time.perf_counter() - started
    _finalize(args, settings, rows, contacts_log, abort, wall, total_frames,
              width, height, tracker.duration, bounds,
              runtime_mass=runtime_mass, contact_view_ok=contact_view_ok)


def _collect_preview(out: Path, frames_dir: Path, shots) -> None:
    images = sorted(frames_dir.glob("rgb_*.png"))
    names = [s[0] for s in shots]
    index = []
    for i, (img, name) in enumerate(zip(images, names, strict=False)):
        target = out / f"preview_{i}_{name}.png"
        img.rename(target)
        index.append({"shot": name, "path": str(target)})
    (out / "preview_index.json").write_text(
        json.dumps({"generated_at": now_iso(), "shots": index}, indent=2))
    print(f"[{now_iso()}] preview written: {out}", flush=True)


def _finalize(args, settings, rows, contacts_log, abort, wall, total_frames,
              width, height, mission_s, bounds, *, runtime_mass=None,
              contact_view_ok=True) -> None:
    csv_path = args.out / f"trajectory_{args.mode}.csv"
    if rows:
        fields = list(rows[0].keys())
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text("t,label\n")  # no ticks logged
    errors = [r["tracking_error_m"] for r in rows]
    videos = []
    frame_counts = {}
    for cam in ("chase", "fpv"):
        frames_dir = args.out / f"frames_{args.mode}_{cam}"
        images = sorted(frames_dir.glob("rgb_*.png"))
        ordered = []
        for i, img in enumerate(images):
            tgt = frames_dir / f"capture_{i:06d}.png"
            if img != tgt:
                img.rename(tgt)
            ordered.append(tgt)
        frame_counts[cam] = len(ordered)
        if settings["encode"] and not abort and ordered:
            video = args.out / f"factory_iris_{args.mode}_{cam}.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(RENDER_FPS),
                 "-i", str(frames_dir / "capture_%06d.png"), "-frames:v", str(len(ordered)),
                 "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "20",
                 "-movflags", "+faststart", str(video)], check=True)
            probe = json.loads(subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,nb_frames", "-of", "json",
                 str(video)], text=True))["streams"][0]
            videos.append({"camera": cam, "path": str(video),
                           "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                           "frames": int(probe["nb_frames"])})
    record = {
        "schema_version": 1,
        "status": "ABORTED" if abort else "COMPLETE",
        "abort_reason": abort,
        "mode": args.mode,
        "generated_at": now_iso(),
        "scene": "NVIDIA USD Explorer automotive factory sample (CAD-authored, "
                 "Omniverse terms, local use, not a named real plant scan)",
        "drone": "3DR Iris (Pegasus commit e13dc65, BSD-3) real meshes/mass/inertia",
        "control": "handwritten cascaded PD on the real Iris body dynamics "
                   "(runtime-measured mass); lumped-thrust flight with visual "
                   "rotor-spin assist — NOT a full motor/aero model, NOT trained, "
                   "NOT autonomous",
        "physics_notes": "factory meshes carry no collision primitives; safety by "
                         "clearance-planned corridor; ground collider + PhysX contact "
                         "logging active; rotor spin via small alternating torques",
        "bounds": bounds,
        "runtime_mass_kg": runtime_mass,
        "mission_s": round(mission_s, 2),
        "ticks": len(rows),
        "tracking_rmse_m": (float(np.sqrt(np.mean(np.square(errors))))
                            if rows else None),
        "tracking_max_m": (float(np.max(errors)) if rows else None),
        "contact_events": contacts_log,
        "contact_status": ("measured_body_vs_ground" if contact_view_ok
                           else "unavailable_after_failure"),
        "collided": (bool(contacts_log) if contact_view_ok else None),
        "contact_monitoring_scope": "body link vs ground collider only; rotor "
                                    "contacts not tracked",
        "video": videos,
        "frames_rendered": frame_counts,
        "trajectory_csv": str(csv_path),
        "wall_s": round(wall, 1),
    }
    (args.out / f"run_{args.mode}.json").write_text(json.dumps(record, indent=2))
    print(f"[{now_iso()}] run {record['status']} -> {args.out / f'run_{args.mode}.json'}",
          flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["preview", "smoke", "record"], default="preview")
    parser.add_argument("--factory-usd", default=str(FACTORY_USD))
    parser.add_argument("--iris-usd", default=str(IRIS_USD))
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if not Path(args.factory_usd).exists():
        raise SystemExit(f"factory USD missing: {args.factory_usd}")
    if not Path(args.iris_usd).exists():
        raise SystemExit(f"iris USD missing: {args.iris_usd}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
