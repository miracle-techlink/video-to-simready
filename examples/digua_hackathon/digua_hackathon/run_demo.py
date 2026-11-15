"""Digua hackathon substation drone demo: simulate, control, record.

What this demo IS:
  * a SYNTHETIC code-generated substation scene (no real site, no real2sim);
  * a SYNTHETIC quadcopter rigid body flown by a hand-written cascaded PD
    controller using SIMULATED state feedback (PhysX pose/velocities);
  * waypoint tracking along a pre-planned collision-free inspection route;
  * two cameras (world chase cam + drone-mounted FPV) recorded in one physics
    run at 720p, RealTime rendering, encoded to mp4 with ffmpeg.

What this demo is NOT: autonomous perception, obstacle avoidance from
sensors, or a trained policy. Collision records come from PhysX contact
forces; clearance numbers are geometric bookkeeping against the known
authored layout.

Usage (server, Isaac env):
  python -m digua_hackathon.run_demo --generate-assets --out runs/demo
  python -m digua_hackathon.run_demo --mode smoke  --out runs/demo
  python -m digua_hackathon.run_demo --mode record --out runs/demo

Asset generation is CPU-only (pxr) and runs before any SimulationApp.
The smoke mode is a short low-res run meant to validate the physics/control/
capture chain before paying for the full recording.
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

from .controller import SPEC, PositionAttitudeController
from .layout import (
    MISSION_NAME,
    MISSION_WAYPOINTS,
    clearance_to_obstacles,
    mission_duration_s,
    obstacles_for_clearance,
    reference_trajectory,
)

PHYSICS_DT = 1.0 / 240.0
RENDER_FPS = 30
ENVELOPE_RADIUS = float(np.linalg.norm(np.array(SPEC.envelope)) / 2.0)  # ~0.34 m

MODE_SETTINGS = {
    "smoke": {"width": 640, "height": 360, "duration_s": 6.0, "rt_subframes": 1,
              "encode": False},
    "record": {"width": 1280, "height": 720, "duration_s": None, "rt_subframes": 2,
               "encode": True},  # None = mission duration + settle margin
}
RECORD_MAX_S = 34.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Asset generation (CPU-only, no SimulationApp, pxr import inside builder)
# ---------------------------------------------------------------------------


def generate_assets(out_dir: Path) -> dict[str, object]:
    from .usd_builder import write_drone_usd, write_substation_usd

    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    scene_meta = write_substation_usd(assets / "substation_synthetic.usda")
    drone_meta = write_drone_usd(assets / "quadcopter_synthetic.usda")
    start = MISSION_WAYPOINTS[0].position
    stage_meta = _compose_stage(assets / "demo_stage.usda", start)
    manifest = {
        "generated_at": now_iso(),
        "generator": "digua_hackathon.usd_builder",
        "scene": scene_meta,
        "drone": drone_meta,
        "stage": stage_meta,
        "claims": {
            "scene_origin": "synthetic_code_generated_not_real_reconstruction",
            "drone_origin": "synthetic_code_generated",
        },
    }
    (assets / "assets_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _compose_stage(stage_path: Path, start_position: tuple[float, float, float]) -> dict:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(stage_path))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    scene_ref = UsdGeom.Xform.Define(stage, "/World/Substation")
    scene_ref.GetPrim().GetReferences().AddReference(
        str(stage_path.parent / "substation_synthetic.usda")
    )
    drone_holder = UsdGeom.Xform.Define(stage, "/World/Drone")
    drone_holder.GetPrim().GetReferences().AddReference(
        str(stage_path.parent / "quadcopter_synthetic.usda")
    )
    drone_holder.AddTranslateOp().Set(Gf.Vec3d(*start_position))
    stage.GetRootLayer().Save()
    return {
        "path": str(stage_path),
        "drone_prim_path": "/World/Drone",
        "drone_start_position_m": list(start_position),
    }


# ---------------------------------------------------------------------------
# Isaac run (SimulationApp first; every omni import stays below this line)
# ---------------------------------------------------------------------------


def run_isaac(args: argparse.Namespace) -> dict[str, object]:
    from isaacsim import SimulationApp

    settings = dict(MODE_SETTINGS[args.mode])
    if settings["duration_s"] is None:
        settings["duration_s"] = min(RECORD_MAX_S, mission_duration_s() + 1.0)
    app = SimulationApp(
        {"headless": True, "limit_cpu_threads": 4, "fast_shutdown": True},
        experience=os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.base.kit"),
    )
    run_record: dict[str, object] | None = None
    exit_code = 1
    try:
        run_record = _simulate(args, settings, app)
        exit_code = 0
    except BaseException as error:  # noqa: BLE001 - evidence-preserving catch
        import traceback

        traceback.print_exc()
        (args.out / f"failure_{args.mode}.json").write_text(
            json.dumps({"status": "FAILED", "error": str(error), "at": now_iso()}, indent=2)
        )
    finally:
        app.close(exit_code=exit_code)
    if run_record is None:
        raise SystemExit(exit_code)
    return run_record


def _simulate(args: argparse.Namespace, settings: dict, app: object) -> dict[str, object]:
    import carb.settings
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, Usd, UsdGeom

    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep  # noqa: E402

    # RealTime raster path with DLSS: the first-delivery bar is "clearly
    # observable 720p", not path tracing.
    for key, value in {
        "/rtx/rendermode": "RealTime",
        "/rtx/post/aa/op": 4,
        "/rtx/post/dlss/execMode": 2,
        "/rtx/dldenoiser/enabled": True,
    }.items():
        carb.settings.get_settings().set(key, value)

    stage_path = str((args.out / "assets" / "demo_stage.usda").resolve())
    if not omni.usd.get_context().open_stage(stage_path):
        raise RuntimeError(f"could not open stage {stage_path}")
    stage = omni.usd.get_context().get_stage()
    drone_prim = stage.GetPrimAtPath(args.drone_path)
    if not drone_prim or not drone_prim.IsA(UsdGeom.Xform):
        raise RuntimeError(f"drone prim missing: {args.drone_path}")

    rep.orchestrator.set_capture_on_play(False)

    # --- cameras -------------------------------------------------------
    width, height = settings["width"], settings["height"]
    chase = UsdGeom.Camera.Define(stage, "/World/Cameras/Chase")
    chase.CreateFocalLengthAttr(24.0)
    chase.CreateHorizontalApertureAttr(20.955)
    chase.CreateVerticalApertureAttr(20.955 * height / width)
    chase.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
    chase_translate = UsdGeom.Xformable(chase.GetPrim()).AddTransformOp()

    fpv_prim = stage.GetPrimAtPath(f"{args.drone_path}/FPVCamera")
    if not fpv_prim or not fpv_prim.IsA(UsdGeom.Camera):
        raise RuntimeError("FPV camera missing from drone asset")
    fpv = UsdGeom.Camera(fpv_prim)
    fpv.CreateVerticalApertureAttr(3.896 * height / width)
    fpv.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))

    product_chase = rep.create.render_product(str(chase.GetPath()), (width, height))
    product_fpv = rep.create.render_product(str(fpv_prim.GetPath()), (width, height))

    frames_chase_dir = args.out / f"frames_{args.mode}_chase"
    frames_fpv_dir = args.out / f"frames_{args.mode}_fpv"

    def make_writer(out_dir: Path, product) -> object:
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(out_dir))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        writer.attach([product])
        return writer

    writer_chase = make_writer(frames_chase_dir, product_chase)
    writer_fpv = make_writer(frames_fpv_dir, product_fpv)

    # --- physics + controller -------------------------------------------
    from isaacsim.core.experimental.prims import RigidPrim

    drone = RigidPrim(args.drone_path)  # wrap before physics initialization
    SimulationManager.set_physics_dt(PHYSICS_DT)
    SimulationManager.initialize_physics()
    controller = PositionAttitudeController()
    obstacles = obstacles_for_clearance()

    duration_s = float(settings["duration_s"])
    ticks_per_frame = round((1.0 / RENDER_FPS) / PHYSICS_DT)
    total_frames = int(round(duration_s * RENDER_FPS))

    # warmup renders (shader compile) with writers detached
    writer_chase.detach()
    writer_fpv.detach()
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
    rep.orchestrator.wait_until_complete()
    writer_chase.attach([product_chase])
    writer_fpv.attach([product_fpv])

    log_rows: list[dict] = []
    contact_events: list[dict] = []
    started = time.perf_counter()
    abort_reason = None
    rotor_spin = 0.0
    last_yaw = 0.0
    overview_frames = int(4.0 * RENDER_FPS)  # first 4 s: scene overview orbit
    scene_center = np.array([0.0, 0.0, 2.0])
    spin_ops = []
    for i in range(4):
        disk = stage.GetPrimAtPath(f"{args.drone_path}/RotorDisk_{i}")
        op = next(
            (o for o in UsdGeom.Xformable(disk).GetOrderedXformOps()
             if "rotateZ" in o.GetOpName()),
            None,
        )
        spin_ops.append(op)

    def drone_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions, quats = drone.get_world_poses()
        lin, ang = drone.get_velocities()
        return (
            np.asarray(positions.numpy(), dtype=float)[0],
            np.asarray(quats.numpy(), dtype=float)[0],  # wxyz
            np.asarray(lin.numpy(), dtype=float)[0],
            np.asarray(ang.numpy(), dtype=float)[0],  # world frame
            PositionAttitudeController.rotation_matrix(
                np.asarray(quats.numpy(), dtype=float)[0]
            ),
        )

    for frame in range(total_frames):
        t = frame / RENDER_FPS
        for _ in range(ticks_per_frame):
            p, q, v, w_world, rot = drone_state()
            w_body = rot.T @ w_world
            ref_p, ref_v, label = reference_trajectory(t)
            if np.linalg.norm(ref_v) > 0.05:
                last_yaw = float(np.arctan2(ref_v[1], ref_v[0]))
            cmd = controller.step(p, v, q, w_body, ref_p, ref_v, last_yaw)
            force_world = rot @ np.array([0.0, 0.0, cmd["total_thrust_n"]])
            torque_world = rot @ np.asarray(cmd["torque_body"], dtype=float)
            drone.apply_forces_and_torques_at_pos(
                forces=force_world.reshape(1, 3),
                torques=torque_world.reshape(1, 3),
            )
            SimulationManager.step()
            sim_t = t + PHYSICS_DT
            contacts = drone.get_net_contact_forces()
            contact_n = float(np.linalg.norm(np.asarray(contacts.numpy(), dtype=float)[0]))
            clearance, nearest = clearance_to_obstacles(p, obstacles, ENVELOPE_RADIUS)
            if contact_n > 0.5 and (not contact_events or sim_t - contact_events[-1]["t"] > 0.5):
                contact_events.append(
                    {"t": round(sim_t, 4), "force_n": round(contact_n, 2),
                     "nearest_obstacle": nearest}
                )
            log_rows.append(
                {
                    "t": round(sim_t, 4), "label": label,
                    "ref_x": ref_p[0], "ref_y": ref_p[1], "ref_z": ref_p[2],
                    "pos_x": p[0], "pos_y": p[1], "pos_z": p[2],
                    "vel": round(float(np.linalg.norm(v)), 4),
                    "thrust_n": round(cmd["total_thrust_n"], 3),
                    "tracking_error_m": round(cmd["position_error_m"], 4),
                    "clearance_m": round(clearance, 3),
                    "nearest_obstacle": nearest,
                    "contact_n": round(contact_n, 3),
                    "saturated": int(cmd["saturated"]),
                }
            )
            if not np.isfinite(p).all() or cmd["position_error_m"] > 6.0 or p[2] < 0.05:
                abort_reason = (
                    f"aborted at t={sim_t:.2f}s: err={cmd['position_error_m']:.2f} "
                    f"z={p[2]:.2f} finite={bool(np.isfinite(p).all())}"
                )
                break
        if abort_reason:
            break

        # chase camera: overview orbit first, then follow the drone
        p, _, _, _, _ = drone_state()
        if frame < overview_frames:
            theta = 2.0 * np.pi * frame / max(1, overview_frames)
            eye = scene_center + np.array(
                [27.0 * np.cos(theta + 0.7), 27.0 * np.sin(theta + 0.7), 14.0]
            )
            look = scene_center
        else:
            heading = np.array([np.cos(last_yaw), np.sin(last_yaw), 0.0])
            eye = p - heading * 5.0 + np.array([0.0, 0.0, 1.9])
            look = p + heading * 1.2
        chase_translate.Set(_look_at(eye, look))

        # cosmetic rotor spin (visual only; thrust is force application)
        rotor_spin = (rotor_spin + 40.0 / RENDER_FPS) % 360.0
        for i, op in enumerate(spin_ops):
            if op is not None:
                op.Set(rotor_spin if i % 2 == 0 else -rotor_spin)

        carb.settings.get_settings().set("/rtx-transient/post/dlss/forceParamReset", True)
        rep.orchestrator.step(
            rt_subframes=settings["rt_subframes"], delta_time=0.0, pause_timeline=False
        )
        if frame % 30 == 0:
            print(f"[{now_iso()}] frame {frame}/{total_frames} t={t:.1f}s", flush=True)

    writer_chase.detach()
    writer_fpv.detach()
    rep.orchestrator.wait_until_complete()
    wall_s = time.perf_counter() - started

    return _finalize(args, settings, log_rows, contact_events, abort_reason,
                     wall_s, total_frames, width, height)


def _look_at(eye: np.ndarray, target: np.ndarray):
    from pxr import Gf

    return (
        Gf.Matrix4d()
        .SetLookAt(
            Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)
        )
        .GetInverse()
    )


def _contiguous_frames(frames_dir: Path) -> list[Path]:
    """Rename writer outputs to contiguous capture_%06d.png (encoder-friendly)."""
    images = sorted(frames_dir.glob("rgb_*.png"))
    ordered = []
    for index, path in enumerate(images):
        target = frames_dir / f"capture_{index:06d}.png"
        if path != target:
            path.rename(target)
        ordered.append(target)
    return ordered


def _finalize(
    args, settings, log_rows, contact_events, abort_reason, wall_s,
    total_frames, width, height,
) -> dict[str, object]:
    csv_path = args.out / f"trajectory_{args.mode}.csv"
    fields = list(log_rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_rows)
    errors = [row["tracking_error_m"] for row in log_rows]
    clearances = [row["clearance_m"] for row in log_rows]
    videos = []
    frame_counts = {}
    for cam in ("chase", "fpv"):
        frames_dir = args.out / f"frames_{args.mode}_{cam}"
        ordered = _contiguous_frames(frames_dir)
        frame_counts[cam] = len(ordered)
        if settings["encode"] and not abort_reason and ordered:
            video = args.out / f"digua_substation_drone_{args.mode}_{cam}.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-framerate", str(RENDER_FPS),
                 "-i", str(frames_dir / "capture_%06d.png"),
                 "-frames:v", str(len(ordered)),
                 "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
                 "-crf", "20", "-movflags", "+faststart", str(video)],
                check=True,
            )
            probe = json.loads(subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,nb_frames",
                 "-of", "json", str(video)], text=True))["streams"][0]
            videos.append({
                "camera": cam, "path": str(video),
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "frames": int(probe["nb_frames"]),
                "width": int(probe["width"]), "height": int(probe["height"]),
            })
    record = {
        "schema_version": 1,
        "status": "ABORTED" if abort_reason else "COMPLETE",
        "abort_reason": abort_reason,
        "mode": args.mode,
        "generated_at": now_iso(),
        "mission": MISSION_NAME,
        "waypoints": [w.position for w in MISSION_WAYPOINTS],
        "mission_duration_s": round(mission_duration_s(), 2),
        "physics_dt": PHYSICS_DT,
        "render_fps": RENDER_FPS,
        "controller": "cascaded_position_attitude_pd_handwritten",
        "state_feedback": "simulated_physx_pose_and_velocities",
        "claims": {
            "scene": "synthetic_substation_no_real2sim",
            "control": "scripted_waypoint_tracking_not_autonomous_not_trained",
            "clearance": "geometric_vs_known_layout_not_sensor_based",
        },
        "ticks_logged": len(log_rows),
        "frames_rendered": frame_counts,
        "tracking_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "tracking_max_m": float(np.max(errors)),
        "min_clearance_m": float(np.min(clearances)),
        "contact_events": contact_events,
        "collided": bool(contact_events),
        "video": videos,
        "trajectory_csv": str(csv_path),
        "wall_time_s": round(wall_s, 2),
    }
    (args.out / f"run_{args.mode}.json").write_text(json.dumps(record, indent=2))
    print(f"[{now_iso()}] run {record['status']}: {args.out / f'run_{args.mode}.json'}",
          flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "record"], default="smoke")
    parser.add_argument("--generate-assets", action="store_true")
    parser.add_argument("--drone-path", default="/World/Drone")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if args.generate_assets:
        manifest = generate_assets(args.out)
        print(json.dumps(manifest["stage"], indent=2))
        return 0
    if not (args.out / "assets" / "demo_stage.usda").exists():
        raise SystemExit("assets missing; run with --generate-assets first")
    run_isaac(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
