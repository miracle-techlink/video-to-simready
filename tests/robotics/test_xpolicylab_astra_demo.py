from __future__ import annotations

import os
import subprocess
import time

import numpy as np
import pytest

from examples.robot_adapters import isaac_astra_demo as demo
from simready.core.artifacts import digest, read_json
from simready.robotics.astra_eef import AstraMoveEefModel
from simready.robotics.contracts import CommandReceipt, RobotState
from simready.robotics.policy_loop import PolicyAdapterLoop, PolicyLoopConfig
from simready.robotics.qwen_policy import PolicyDecision, ToolCall
from simready.robotics.xpolicylab import XPolicyLabProvider

FIXTURE = demo.ROOT / "examples/robot_adapters/configs/astra-lift-action.json"


class CpuLift:
    """Deterministic fixture exercising the policy boundary and verifier."""

    execution_mode = "simulation"
    profile = demo.PROFILE

    def __init__(self):
        self.position = 0.0
        self.step = 0

    def get_state(self, *, timeout_s):
        return RobotState({demo.JOINT: self.position}, time.perf_counter_ns())

    def send_joint_position(self, command):
        self.position = command.positions[demo.JOINT]
        self.step += 1
        return CommandReceipt(command.sequence, True, time.perf_counter_ns())


class FakeAstra:
    def decide(self, *, instructions, observation, tools, image_data_urls):
        assert instructions == demo.INSTRUCTION
        assert observation["state"]["left_z"] == demo.BASE_Z
        assert tools[0]["function"]["name"] == "move_eef"
        assert image_data_urls[0].startswith("data:image/png;base64,")
        action = read_json(FIXTURE)
        return PolicyDecision((ToolCall(action["tool"], action["arguments"]),))


def lift_loop(model, *, height=None, step=None, observation_age_s=0):
    lift = CpuLift()
    evidence = {}
    verifier = demo.LiftVerifier(
        height or (lambda: demo.BASE_Z + lift.position),
        step or (lambda: lift.step),
        demo.BASE_Z,
        0,
        evidence,
    )
    provider = XPolicyLabProvider(model, demo.build_observation, demo.decode_action)
    loop = PolicyAdapterLoop(
        provider,
        lift,
        verifier,
        PolicyLoopConfig("lift", "set_lift", state_timeout_s=65),
        now_monotonic_ns=time.perf_counter_ns,
    )

    def run():
        return loop.run_once(
            instructions=demo.INSTRUCTION,
            observation={
                "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "measured_z": demo.BASE_Z,
                "measured_monotonic_ns": time.monotonic_ns()
                - int(observation_age_s * 1e9),
                "evidence": evidence,
            },
        )

    return lift, provider, run, evidence


@pytest.mark.parametrize("live_shape", [False, True])
def test_model_lifecycle_through_policy_loop(live_shape):
    model = (
        AstraMoveEefModel(FakeAstra(), demo.ACTION)
        if live_shape
        else demo.ReplayModel(read_json(FIXTURE))
    )
    lift, provider, run, evidence = lift_loop(model)
    assert run().status == "policy_error"
    assert lift.step == 0
    provider.reset()
    result = run()
    assert result.status == "verified", result
    assert evidence["goal_error_m"] == pytest.approx(0)
    assert evidence["vertical_motion_m"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    "targets",
    [
        {},
        {"left_z": float("nan")},
        {"left_z": True},
        {"left_z": 10**400},
        {"left_z": "0.13"},
        {"right_z": 0.13},
        {"left_yaw_deg": 0},
        {"left_gripper": 0.5},
        {"left_z": 0.16},
        {"left_z": 0.4},
    ],
)
def test_invalid_or_out_of_bounds_action_cannot_dispatch(targets):
    action = {"tool": "move_eef", "arguments": {"targets": targets, "note": "test"}}
    lift, provider, run, _ = lift_loop(demo.ReplayModel(action))
    provider.reset()
    assert run().status == "policy_error"
    assert run().detail == "reset the local model before running an episode"
    assert lift.step == 0


def test_stale_cartesian_observation_cannot_dispatch():
    lift, provider, run, _ = lift_loop(
        demo.ReplayModel(read_json(FIXTURE)), observation_age_s=66
    )
    provider.reset()
    assert "stale" in run().detail
    assert lift.step == 0


@pytest.mark.parametrize(
    ("height", "step"), [(demo.BASE_Z, 1), (demo.GOAL_Z, 0), (float("nan"), 1)]
)
def test_independent_verification_rejects_false_success(height, step):
    _, provider, run, _ = lift_loop(
        demo.ReplayModel(read_json(FIXTURE)), height=lambda: height, step=lambda: step
    )
    provider.reset()
    assert run().status == "verification_failed"


def test_authored_scene_has_fixed_base_and_expected_joint_contract():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    demo.author_scene(stage)
    joint = UsdPhysics.PrismaticJoint.Get(stage, f"{demo.ROBOT}/{demo.JOINT}")
    assert joint.GetAxisAttr().Get() == "Z"
    spec = demo.PROFILE.joint_map()[demo.JOINT]
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(spec.lower)
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(spec.upper)
    assert (
        UsdPhysics.DriveAPI.Get(joint.GetPrim(), "linear").GetStiffnessAttr().Get() > 0
    )
    assert stage.GetPrimAtPath(demo.ROBOT).HasAPI(UsdPhysics.ArticulationRootAPI)
    assert UsdPhysics.FixedJoint.Get(stage, f"{demo.ROBOT}/FixedBase")
    position = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(demo.CARRIAGE)
    )
    assert position.ExtractTranslation()[2] == pytest.approx(demo.BASE_Z)


@pytest.mark.gpu
def test_real_isaac_demo_replay(tmp_path):
    """Use the native Isaac Python launcher on the designated GPU host."""
    launcher = os.environ.get("SIMREADY_ISAAC_PYTHON")
    assert launcher, "SIMREADY_ISAAC_PYTHON is required when --run-gpu is set"
    out = tmp_path / "replay"
    paths = [str(demo.ROOT / "src")]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    completed = subprocess.run(
        [launcher, demo.__file__, "--action", str(FIXTURE), "--out", str(out)],
        cwd=demo.ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(paths)},
        check=False,
        timeout=600,
    )
    report = read_json(out / "verdict.json")
    assert completed.returncode == 0 and report["status"] == "PASS", report
    assert report["mode"] == "local_action_replay"
    assert report["scene_sha256"] == digest(out / "scene.usda")
    assert report["action_sha256"] == digest(FIXTURE)
    assert report["goal_error_m"] <= demo.TOLERANCE_M
    assert report["feedback_advanced"] is True
    for name in ("before.png", "after.png"):
        assert report[name] == digest(out / name)
