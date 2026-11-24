"""Run one local policy action in a self-contained Isaac Sim lift scene."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from simready.core.artifacts import digest, loads_json, object_digest, write_json
from simready.robotics.astra_eef import AstraMoveEef, AstraMoveEefModel
from simready.robotics.astra_policy import AstraPolicyConfig, AstraPolicyProvider
from simready.robotics.cartesian_actions import (
    CartesianEefObservation,
    CartesianEefTarget,
    CartesianSafetyLimits,
    GripperTarget,
    TimedActionSegment,
    validate_cartesian_action_segment,
)
from simready.robotics.contracts import JointSpec, RobotProfile
from simready.robotics.policy_loop import (
    PolicyAdapterLoop,
    PolicyLoopConfig,
    VerificationResult,
)
from simready.robotics.qwen_policy import PolicyDecision, ToolCall
from simready.robotics.xpolicylab import XPolicyLabProvider

ROOT = Path(__file__).resolve().parents[2]
BASE_Z = 0.10
GOAL_Z = 0.13
TOLERANCE_M = 0.004
MIN_MOTION_M = 0.015
JOINT = "lift"
ROBOT = "/World/Lift"
CARRIAGE = f"{ROBOT}/Carriage"
CAMERA = "/World/Camera"
INSTRUCTION = (
    "Lift the orange carriage to the green marker. The target height is "
    "world left_z = 0.13 metres. Propose one move_eef action for left_z."
)
PROFILE = RobotProfile(
    robot_id="simready-isaac-lift-demo",
    display_name="Isaac single-axis lift",
    joints=(JointSpec(JOINT, "lift", "m", -0.04, 0.07),),
    groups={"lift": (JOINT,)},
    state_transport="isaac_sim",
    simulation_command_transport="isaac_sim",
)
LIMITS = CartesianSafetyLimits(
    "world", (-0.001, -0.001, 0.06), (0.001, 0.001, 0.17), 0, 1, 0.04, 0.01, 1
)
ACTION = AstraMoveEef(("left_z",))


class ReplayModel:
    """Read one offline action; report replay separately from live inference."""

    def __init__(self, data: Any) -> None:
        self._data = data
        self._observed = False

    def reset(self) -> None:
        self._observed = False

    def update_obs(self, obs: dict[str, Any]) -> None:
        self._observed = bool(obs)

    def get_action(self) -> Any:
        if not self._observed:
            raise ValueError("replay requires a fresh observation")
        self._observed = False
        return self._data


def build_observation(instruction, observation, _images):
    scene = observation["caller_observation"]
    return {
        "instruction": instruction,
        "vision": {"head": {"color": scene["rgb"]}},
        "state": {
            "left_z": scene["measured_z"],
            "joints_m": observation["robot_state"]["positions"],
        },
        "extra_info": {"detailed_instruction": "Move the lift to the marker height."},
    }


def decode_action(action, observation, tools):
    scene = observation["caller_observation"]
    pose = CartesianEefObservation(
        target_at(scene["measured_z"]), scene["measured_monotonic_ns"]
    )
    targets = ACTION.parse(action)
    sense_time = time.monotonic_ns()
    segment = TimedActionSegment(
        target_at(targets["left_z"]), GripperTarget(1.0), sense_time, 1.0, 0.5, 0
    )
    validate_cartesian_action_segment(
        LIMITS,
        segment,
        current_observation=pose,
        max_observation_age_s=65,
        now_monotonic_ns=sense_time,
        last_sequence=None,
    )
    scene["evidence"]["proposed_world_targets_m"] = targets
    name = tools[0]["function"]["name"]
    return PolicyDecision((ToolCall(name, {JOINT: targets["left_z"] - BASE_Z}),))


def target_at(z):
    return CartesianEefTarget("world", (0.0, 0.0, z), (0.0, 0.0, 0.0, 1.0))


class LiftVerifier:
    """Read the carriage transform from physics and compare to the fixed goal."""

    def __init__(self, height, step_index, before_z, before_step, evidence):
        self._height = height
        self._step_index = step_index
        self._before_z = before_z
        self._before_step = before_step
        self._evidence = evidence

    def verify(self, *, command, receipt, state_before, state_after, timeout_s):
        del timeout_s  # The in-process physics reads do not wait for another step.
        z = self._height()
        step = self._step_index()
        if not math.isfinite(z):
            return VerificationResult(False, "nonfinite carriage height")
        error_m = abs(z - GOAL_Z)
        motion_m = z - self._before_z
        tracking_error_m = abs(state_after.positions[JOINT] - command.positions[JOINT])
        feedback_advanced = (
            step > self._before_step
            and state_after.received_monotonic_ns > state_before.received_monotonic_ns
            and state_before.received_monotonic_ns
            <= receipt.accepted_monotonic_ns
            <= state_after.received_monotonic_ns
        )
        self._evidence.update(
            final_height_m=z,
            goal_error_m=error_m,
            vertical_motion_m=motion_m,
            joint_tracking_error_m=tracking_error_m,
            physics_steps=[self._before_step, step],
            feedback_advanced=feedback_advanced,
        )
        accepted = (
            feedback_advanced
            and error_m <= TOLERANCE_M
            and tracking_error_m <= TOLERANCE_M
            and motion_m >= MIN_MOTION_M
        )
        return VerificationResult(accepted, "measured lift position and joint tracking")


def author_scene(stage):
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

    def cube(path, xyz, scale, color):
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1)
        shape.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        shape.AddScaleOp().Set(Gf.Vec3d(*scale))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return shape

    def rigid_body(path, z):
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(1.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.001, 0.001, 0.001))

    root = UsdGeom.Xform.Define(stage, ROBOT)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    base = f"{ROBOT}/Base"
    rigid_body(base, 0)
    rigid_body(CARRIAGE, BASE_Z)
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT}/FixedBase")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(base)])
    joint = UsdPhysics.PrismaticJoint.Define(stage, f"{ROBOT}/{JOINT}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(CARRIAGE)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, BASE_Z))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-0.04)
    joint.CreateUpperLimitAttr(0.07)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(400)
    drive.CreateDampingAttr(40)
    drive.CreateMaxForceAttr(100)
    cube(f"{base}/Pedestal", (0, 0, 0), (0.09, 0.09, 0.04), (0.22, 0.26, 0.33))
    cube(f"{CARRIAGE}/Tool", (0, 0, 0), (0.07, 0.07, 0.035), (1.0, 0.35, 0.04))
    cube("/World/Marker", (0.10, 0, GOAL_Z), (0.045, 0.06, 0.008), (0.15, 0.85, 0.30))
    cube("/World/Table", (0, 0, -0.035), (0.8, 0.8, 0.02), (0.3, 0.35, 0.45))
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(800)
    camera = UsdGeom.Camera.Define(stage, CAMERA)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 20))
    camera.AddTransformOp().Set(
        Gf.Matrix4d()
        .SetLookAt(
            Gf.Vec3d(0.37, -0.61, 0.30), Gf.Vec3d(0.03, 0, 0.10), Gf.Vec3d(0, 0, 1)
        )
        .GetInverse()
    )


def run_isaac(model, out, report):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        import numpy as np
        import omni.kit.app
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from PIL import Image

        from simready.robotics.isaac_sim import IsaacSimArticulationAdapter

        kit = omni.kit.app.get_app()
        kit.get_extension_manager().set_extension_enabled_immediate(
            "omni.replicator.core", True
        )
        import omni.replicator.core as rep

        report["kit_build"] = kit.get_build_version()
        report["phase"] = "scene_initialization"
        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
        stage = omni.usd.get_context().get_stage()
        author_scene(stage)
        world.get_physics_context().set_gravity(0)
        scene_file = out / "scene.usda"
        if not stage.GetRootLayer().Export(str(scene_file)):
            raise RuntimeError("could not export the synthetic scene")
        report["scene_sha256"] = digest(scene_file)
        robot = world.scene.add(SingleArticulation(prim_path=ROBOT, name="demo_lift"))
        carriage = world.scene.add(SingleRigidPrim(prim_path=CARRIAGE, name="carriage"))
        world.reset()
        adapter = IsaacSimArticulationAdapter(
            PROFILE, robot, world, physics_steps_per_command=60
        )
        adapter.connect()

        def height():
            return float(carriage.get_world_pose()[0][2])

        for _ in range(12):
            world.step(render=True)
        rep.orchestrator.set_capture_on_play(False)
        product = rep.create.render_product(CAMERA, (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])

        def capture(name):
            rep.orchestrator.step(rt_subframes=8, delta_time=0, pause_timeline=False)
            rep.orchestrator.wait_until_complete()
            frame = np.asarray(rgb.get_data())
            if (
                frame.shape not in {(480, 640, 3), (480, 640, 4)}
                or frame.dtype != np.uint8
            ):
                raise RuntimeError("invalid renderer RGB frame")
            pixels = frame[..., :3].copy()
            if not np.any(pixels != pixels[0, 0]):
                raise RuntimeError("renderer returned a uniform RGB frame")
            image_path = out / name
            Image.fromarray(pixels).save(image_path)
            report[name] = digest(image_path)
            return pixels

        try:
            report["phase"] = "observation"
            pixels = capture("before.png")
            before_step = world.current_time_step_index
            before_z = height()
            measured_ns = time.monotonic_ns()
            if not math.isfinite(before_z) or abs(before_z - BASE_Z) > TOLERANCE_M:
                raise RuntimeError("initial carriage height is outside tolerance")
            report.update(
                initial_height_m=before_z, observation_monotonic_ns=measured_ns
            )
            verifier = LiftVerifier(
                height,
                lambda: world.current_time_step_index,
                before_z,
                before_step,
                report,
            )
            policy = XPolicyLabProvider(model, build_observation, decode_action)
            policy.reset()
            loop = PolicyAdapterLoop(
                policy,
                adapter,
                verifier,
                PolicyLoopConfig(
                    "lift", "set_lift", max_joint_step=0.04, state_timeout_s=65
                ),
            )
            report["phase"] = "policy_and_physics"
            result = loop.run_once(
                instructions=INSTRUCTION,
                observation={
                    "rgb": pixels,
                    "measured_z": before_z,
                    "measured_monotonic_ns": measured_ns,
                    "evidence": report,
                },
            )
            result_data = asdict(result)
            if result.status == "post_state_invalid":
                result_data["state_after"] = None
            report["policy_result"] = result_data
            report["phase"] = "result_capture"
            capture("after.png")
            if (
                result.status == "verified"
                and report["before.png"] != report["after.png"]
            ):
                report.update(status="PASS", phase="complete")
        finally:
            rgb.detach([product])
            product.destroy()
            adapter.close()
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--action", type=Path, help="local move_eef JSON replay")
    mode.add_argument("--live-astra", action="store_true")
    parser.add_argument(
        "--out", type=Path, required=True, help="new evidence directory"
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "mode": "live_astra" if args.live_astra else "local_action_replay",
        "status": "FAIL",
        "phase": "preflight",
        "goal_height_m": GOAL_Z,
        "tolerance_m": TOLERANCE_M,
        "minimum_motion_m": MIN_MOTION_M,
        "profile_sha256": object_digest(asdict(PROFILE)),
    }
    try:

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(ROOT), *args], text=True, timeout=15
            ).strip()

        report["source_commit"] = git("rev-parse", "HEAD")
        report["tracked_tree_dirty"] = bool(git("status", "--porcelain", "-uno"))
        if args.action:
            action_text = args.action.read_bytes()
            report["action_sha256"] = sha256(action_text).hexdigest()
            action = loads_json(action_text.decode("utf-8"))
            ACTION.parse(action)
            model = ReplayModel(action)
        else:
            config = AstraPolicyConfig()
            required = (
                config.api_key_env,
                config.base_url_env,
                config.model_env,
                config.expected_model_env,
            )
            missing = [
                name for name in required if not os.environ.get(name, "").strip()
            ]
            if missing:
                report.update(
                    status="BLOCKED",
                    reason="missing environment: " + ", ".join(missing),
                )
                return 2
            report["expected_model"] = os.environ[config.expected_model_env]
            model = AstraMoveEefModel(AstraPolicyProvider(config), ACTION)
        try:
            report["nvidia_gpu_inventory"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).splitlines()
        except (OSError, subprocess.SubprocessError):
            report.update(
                status="BLOCKED", reason="NVIDIA GPU/driver unavailable via nvidia-smi"
            )
            return 2
        if (
            not report["nvidia_gpu_inventory"]
            or importlib.util.find_spec("isaacsim") is None
        ):
            report.update(
                status="BLOCKED",
                reason="NVIDIA GPU or Isaac Sim Python runtime unavailable",
            )
            return 2
        run_isaac(model, out, report)
    except Exception as error:
        report["status"] = "FAIL"
        report["exception_type"] = type(error).__name__
        if report["phase"] == "preflight":
            report["reason"] = (
                "could not prepare execution context or parse local action"
            )
        raise
    finally:
        write_json(out / "verdict.json", report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
