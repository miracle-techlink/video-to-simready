#!/usr/bin/env python3
"""Server-side, file-tool Astra-in-the-loop canary for the reBot B601-RS.

This runs only a simulated Isaac articulation. Astra (or another host agent)
reads observation.json + observation.png, writes command.json, and receives a
fresh state/image after each bounded target. There is no real hardware path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset", type=Path, required=True, help="official ReBot_Arm_RS.usda"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).parent / "profiles/rebot-b601-rs.json",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument(
        "--max-joint-step", type=float, default=0.08, help="rad per agent action"
    )
    parser.add_argument("--ticks-per-action", type=int, default=15)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    command_path = args.run_dir / "command.json"
    observation_path = args.run_dir / "observation.json"
    image_path = args.run_dir / "observation.png"
    events_path = args.run_dir / "events.jsonl"

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "disable_viewport_updates": False,
            "renderer": "RaytracedLighting",
            "limit_cpu_threads": 16,
        }
    )
    try:
        import numpy as np
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux

        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "src"))
        from simready.robotics import (
            IsaacSimArticulationAdapter,
            JointCommand,
            load_robot_profile,
            validate_joint_command,
        )

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.set_extension_enabled_immediate("omni.replicator.core", True)
        profile = load_robot_profile(args.profile)
        world = World(
            stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0
        )
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        robot_prim = "/World/ReBot"
        add_reference_to_stage(str(args.asset.resolve()), robot_prim)
        # Follow Seeed's Isaac receiver: wrap the referenced robot prim; its USD
        # contains the articulation-root API and fixed base relationship.
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(900)
        cam = UsdGeom.Camera.Define(stage, "/World/ObserverCamera")
        cam.GetFocalLengthAttr().Set(24.0)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))
        camera_xf = UsdGeom.Xformable(cam.GetPrim())
        camera_xf.AddTransformOp().Set(
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(0.62, -0.78, 0.58),
                Gf.Vec3d(0.0, 0.0, 0.20),
                Gf.Vec3d(0.0, 0.0, 1.0),
            )
            .GetInverse()
        )
        robot = SingleArticulation(prim_path=robot_prim, name="rebot_b601_rs")
        world.scene.add(robot)
        world.reset()
        adapter = IsaacSimArticulationAdapter(profile, robot, world)
        adapter.connect()
        timeline = omni.timeline.get_timeline_interface()
        for _ in range(10):
            world.step(render=True)
        print(
            json.dumps(
                {
                    "event": "ARTICULATION_CONNECTED",
                    "dof_names": list(robot.dof_names),
                    "timeline_playing": bool(timeline.is_playing()),
                }
            ),
            flush=True,
        )
        for _ in range(10):
            world.step(render=True)
        names = list(profile.groups["arm"])
        if any(name not in robot.dof_names for name in profile.joint_map()):
            raise RuntimeError("USD articulation DOF names do not match the profile")

        render_product = rep.create.render_product(str(cam.GetPath()), (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([render_product])

        def publish_observation(sequence: int, last_event: dict | None = None) -> dict:
            try:
                state = adapter.get_state()
            except BaseException as error:
                atomic_json(
                    args.run_dir / "observation-error.json",
                    {"error": repr(error), "traceback": traceback.format_exc()},
                )
                raise
            rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
            rep.orchestrator.wait_until_complete()
            pixels = np.asarray(rgb.get_data())
            if pixels.ndim != 3 or pixels.shape[0] != 480 or pixels.shape[1] != 640:
                raise RuntimeError(f"invalid RGB observation shape {pixels.shape}")
            Image.fromarray(pixels[..., :3].astype(np.uint8)).save(image_path)
            payload = {
                "schema_version": 1,
                "sequence": sequence,
                "timestamp_monotonic_ns": state.received_monotonic_ns,
                "source": state.source,
                "robot_id": profile.robot_id,
                "task": "Inspect the simulated reBot arm; choose small safe joint targets and reassess after each action.",
                "positions_rad_or_m": state.positions,
                "velocities_rad_s_or_m_s": state.velocities,
                "rgb_path": str(image_path),
                "last_action_feedback": last_event,
                "motion_is_simulation_only": True,
                "task_success": None,
            }
            atomic_json(observation_path, payload)
            return payload

        last_command_mtime = 0
        state = publish_observation(0)
        print(
            json.dumps(
                {
                    "event": "READY",
                    "observation": str(observation_path),
                    "image": str(image_path),
                }
            ),
            flush=True,
        )
        applied = 0
        while app.is_running() and applied < args.max_actions:
            if not command_path.exists():
                time.sleep(0.05)
                continue
            mtime = command_path.stat().st_mtime_ns
            if mtime <= last_command_mtime:
                time.sleep(0.05)
                continue
            raw = json.loads(command_path.read_text(encoding="utf-8"))
            last_command_mtime = mtime
            group = raw.get("group")
            positions = raw.get("positions")
            sequence = raw.get("sequence")
            if (
                group != "arm"
                or not isinstance(positions, dict)
                or not isinstance(sequence, int)
            ):
                raise ValueError(
                    "command schema requires integer sequence, group='arm', and positions map"
                )
            command = JointCommand(
                group=group,
                positions=positions,
                issued_monotonic_ns=time.monotonic_ns(),
                valid_for_s=10.0,
                sequence=sequence,
            )
            validate_joint_command(
                profile, command, now_monotonic_ns=time.monotonic_ns()
            )
            # Replicator may leave the physics timeline paused after publishing
            # an RGB frame. Re-arm physics and the articulation tensor view
            # before reading feedback or applying the next policy action.
            if not timeline.is_playing():
                timeline.play()
            world.step(render=True)
            # initialize() is idempotent on this Isaac 6.2 legacy wrapper and
            # recreates its tensor view after a paused timeline.
            robot.initialize()
            current = adapter.get_state().positions
            deltas = {joint: abs(positions[joint] - current[joint]) for joint in names}
            if max(deltas.values(), default=0) > args.max_joint_step:
                raise ValueError(
                    f"action exceeds per-joint delta cap {args.max_joint_step} rad: {deltas}"
                )
            receipt = adapter.send_joint_position(command)
            for _ in range(max(0, args.ticks_per_action - 1)):
                # Keep stepping through World so the legacy articulation's
                # physics view stays initialized. Bare Kit app updates do not
                # guarantee a physics tick and can invalidate joint feedback.
                world.step(render=True)
            state = adapter.get_state()
            event = {
                "sequence": receipt.sequence,
                "accepted": receipt.accepted,
                "target": dict(positions),
                "measured": dict(state.positions),
                "max_abs_target_error_rad": max(
                    abs(positions[joint] - state.positions[joint]) for joint in names
                ),
            }
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            archived = args.run_dir / f"command-{sequence:03d}.json"
            command_path.replace(archived)
            publish_observation(sequence, event)
            applied += 1
            print(json.dumps({"event": "ACTION_APPLIED", **event}), flush=True)

        summary = {
            "status": "SIM_CONTROL_LOOP_COMPLETE",
            "robot_id": profile.robot_id,
            "actions_applied": applied,
            "hardware_connected": False,
            "task_success": None,
            "observation": str(observation_path),
            "events": str(events_path),
        }
        atomic_json(args.run_dir / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        rgb.detach()
        return 0
    except BaseException as error:
        atomic_json(
            args.run_dir / "exception.json",
            {"error": repr(error), "traceback": traceback.format_exc()},
        )
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
