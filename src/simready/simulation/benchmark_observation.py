"""Batch fixed-camera observation worker consuming frozen render requests.

Orchestrates multi-view RGB/RGB-D capture via the existing Isaac launcher.
Only accepts frozen render requests validated by core/benchmark_contract.py.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from simready.core.artifacts import digest, object_digest, read_json, write_json
from simready.core.benchmark_contract import (
    ContractError,
    validate_render_request,
    validate_view_set,
)
from simready.geometry.cameras import rigid_inverse
from simready.simulation.depth_observation import (
    _capture_single_view,
    _create_observation_camera,
    _validate_capture_result,
    calibration_to_usd_camera,
    normalize_camera_z,
)


def plan_benchmark_observation(
    render_request_path: Path,
    view_set_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Plan a batch observation from frozen manifests; return execution plan.

    Reads and validates the render request and view set. Computes per-view
    camera settings. Returns a plan dict with per-view camera parameters
    ready for worker execution.
    """
    render_request = validate_render_request(
        read_json(render_request_path), render_request_path.parent
    )
    view_set = validate_view_set(read_json(view_set_path), view_set_path.parent)

    for view in view_set["views"]:
        intrinsics = np.asarray(view["intrinsics"], dtype=float)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        if not np.isclose(fx, fy, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"View {view['view_id']!r}: benchmark requires square pixels (fx=fy); "
                "undistort and resample first, then re-freeze the view set"
            )

    return {
        "render_request": render_request,
        "view_set": view_set,
        "output_dir": str(output_dir),
        "views": view_set["views"],
    }


def run_benchmark_observation(
    plan: dict[str, Any],
    *,
    backend: str = "docker",
    repo: Path | None = None,
    assets: Path | None = None,
    image: str = "simready-isaac-core:6.1.0",
    gpu: str = "0",
    timeout_s: float = 7200,
    cleanup_timeout_s: float = 15,
) -> dict[str, Any]:
    """Execute a planned batch observation via the Isaac launcher.

    Uses launch_worker with a task graph; each view is a separate task
    sharing the same GPU lease. Returns the aggregated receipt.
    """
    from simready.execution.workers.isaac_launcher import launch_worker

    render_request = plan["render_request"]
    view_set = plan["view_set"]
    output_dir = Path(plan["output_dir"])
    mode = render_request["image_settings"]["mode"]
    want_depth = mode == "rgbd"

    output_dir.mkdir(parents=True, exist_ok=True)

    plan_file = output_dir / "observation_plan.json"
    write_json(
        plan_file,
        {
            "render_request": render_request,
            "view_set": view_set,
            "mode": mode,
            "want_depth": want_depth,
        },
    )

    worker = [
        "-m",
        "simready.simulation.benchmark_observation",
        "--plan",
        str(plan_file),
        "--out",
        str(output_dir),
    ]

    return launch_worker(
        worker,
        repo=repo or Path.cwd(),
        assets=assets or Path.cwd(),
        output=output_dir,
        backend=backend,
        image=image,
        gpu=gpu,
        timeout_s=timeout_s,
        cleanup_timeout_s=cleanup_timeout_s,
    )


def _build_receipt(
    render_request: dict,
    view_set: dict,
    views_result: list[dict],
    started: float,
) -> dict:
    """Build the receipt matching the fake_render_worker layout."""
    per_view = []
    for v in views_result:
        per_view.append(
            {
                "view_id": v["view_id"],
                "width": v["width"],
                "height": v["height"],
                "rgb_file": v.get("rgb_file"),
                "rgb_sha256": v.get("rgb_sha256"),
                "depth_file": v.get("depth_file"),
                "depth_sha256": v.get("depth_sha256"),
            }
        )

    return {
        "synthetic_fixture": False,
        "banner": "Benchmark observation worker. Real rendering evidence.",
        "worker": "simready.simulation.benchmark_observation",
        "scene_id": render_request.get("scene", {}).get("path", "unknown"),
        "seed": render_request["seed"],
        "want_depth": render_request["image_settings"]["mode"] == "rgbd",
        "render_settings": render_request["image_settings"],
        "views": per_view,
        "completed": len(per_view),
        "expected": len(view_set["views"]),
        "duration_seconds": round(time.monotonic() - started, 6),
    }


def _worker_main(argv=None) -> int:
    """Isaac worker entry point: reads plan, captures views, writes receipt."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subframes", type=int, default=8)
    parser.add_argument("--warmup-updates", type=int, default=12)
    parser.add_argument("--capture-attempts", type=int, default=8)
    args = parser.parse_args(argv)

    plan_data = read_json(args.plan)
    render_request = plan_data["render_request"]
    view_set = plan_data["view_set"]
    want_depth = plan_data["want_depth"]

    validate_render_request(render_request, args.plan.parent)
    validate_view_set(view_set, args.plan.parent)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    views_result = []
    app = None
    try:
        import os

        import carb
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux, UsdUtils

        from isaacsim import SimulationApp

        app = SimulationApp(
            {"headless": True, "limit_cpu_threads": 4, "fast_shutdown": True},
            experience=os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.base.kit"),
        )

        import omni.replicator.core as rep

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "omni.replicator.core", True
        )

        settings = carb.settings.get_settings()
        for key, value in {
            "/app/player/playSimulations": False,
            "/rtx/rendermode": "RealTimePathTracing",
            "/rtx/post/aa/op": 4,
            "/rtx/post/dlss/execMode": 2,
            "/rtx/dldenoiser/enabled": True,
        }.items():
            settings.set(key, value)

        scene_path = Path(render_request["scene"]["path"])
        if not omni.usd.get_context().open_stage(str(scene_path)):
            raise ValueError("Isaac could not open the source scene")
        omni.timeline.get_timeline_interface().pause()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage_units = float(UsdGeom.GetStageMetersPerUnit(stage))

        lighting = render_request["lighting"]
        if lighting["policy"] == "frozen":
            existing_lights = [
                p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)
            ]
            if not existing_lights:
                light = UsdLux.DomeLight.Define(stage, "/SimReadyBenchmark/Light")
                light.CreateIntensityAttr(300.0)
                light.CreateExposureAttr(float(lighting.get("exposure", 0)))

        for view in view_set["views"]:
            view_id = view["view_id"]
            intrinsics = np.asarray(view["intrinsics"], dtype=float)
            c2w = np.asarray(view["camera_to_world"], dtype=float)

            calibration = {
                "schema_version": 1,
                "camera_convention": "opencv_c2w",
                "distortion": "none",
                "image_size": [
                    view["resolution"]["width"],
                    view["resolution"]["height"],
                ],
                "intrinsics": view["intrinsics"],
                "camera_to_world": view["camera_to_world"],
            }

            camera_settings = calibration_to_usd_camera(calibration)

            camera_path = _create_observation_camera(
                stage, camera_settings, scope_path=f"/SimReadyBenchmark/{view_id}"
            )

            rgb, depth, statistics, captured_attempt = _capture_single_view(
                rep,
                camera_path,
                camera_settings["width"],
                camera_settings["height"],
                want_depth,
                app,
                args.warmup_updates,
                args.subframes,
                args.capture_attempts,
                stage_units,
                capture_mode="rgbd" if want_depth else "rgb",
            )

            _validate_capture_result(
                rgb, depth, capture_mode="rgbd" if want_depth else "rgb"
            )

            rgb_path = out_dir / f"{view_id}.rgb.png"
            Image.fromarray(rgb[:, :, :3]).save(rgb_path)
            result = {
                "view_id": view_id,
                "width": camera_settings["width"],
                "height": camera_settings["height"],
                "rgb_file": rgb_path.name,
                "rgb_sha256": digest(rgb_path),
                "depth_file": None,
                "depth_sha256": None,
            }
            if want_depth and depth is not None:
                depth_path = out_dir / f"{view_id}.depth.npy"
                np.save(depth_path, depth, allow_pickle=False)
                result["depth_file"] = depth_path.name
                result["depth_sha256"] = digest(depth_path)

            views_result.append(result)

        receipt = _build_receipt(render_request, view_set, views_result, started)
        write_json(out_dir / "receipt.json", receipt)
        print(f"Benchmark observation complete: {out_dir / 'receipt.json'}", flush=True)
        return 0
    except BaseException as error:
        import traceback

        traceback.print_exc()
        failure = {
            "synthetic_fixture": False,
            "status": "FAILED",
            "error": str(error),
            "elapsed_s": time.monotonic() - started,
        }
        write_json(out_dir / "failure.json", failure)
        return 1
    finally:
        if app is not None:
            app.close(exit_code=0 if not views_result else 1)


def main(argv=None) -> int:
    """CLI entry point for batch observation."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    plan_p = sub.add_parser("plan", help="Plan a batch observation")
    plan_p.add_argument("--render-request", type=Path, required=True)
    plan_p.add_argument("--view-set", type=Path, required=True)
    plan_p.add_argument("--output", type=Path, required=True)

    run_p = sub.add_parser("run", help="Execute a planned batch observation")
    run_p.add_argument("--render-request", type=Path, required=True)
    run_p.add_argument("--view-set", type=Path, required=True)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--backend", choices=["local", "docker"], default="docker")
    run_p.add_argument("--gpu", default="0")
    run_p.add_argument("--timeout-s", type=float, default=7200)

    worker_p = sub.add_parser("worker", help="Isaac worker entry (internal)")
    worker_p.add_argument("--plan", type=Path, required=True)
    worker_p.add_argument("--out", type=Path, required=True)
    worker_p.add_argument("--subframes", type=int, default=8)
    worker_p.add_argument("--warmup-updates", type=int, default=12)
    worker_p.add_argument("--capture-attempts", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = plan_benchmark_observation(
            args.render_request, args.view_set, args.output
        )
        write_json(args.output / "plan.json", plan)
        print(f"Plan written: {args.output / 'plan.json'}")
        return 0
    elif args.command == "run":
        plan = plan_benchmark_observation(
            args.render_request, args.view_set, args.output
        )
        return run_benchmark_observation(
            plan,
            backend=args.backend,
            gpu=args.gpu,
            timeout_s=args.timeout_s,
        )
    elif args.command == "worker":
        return _worker_main(argv)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
