"""Deterministic CPU-only module-integration test for the PR #31 gap.

Exercises the real public modules together: ``QwenPolicyProvider.decide``
parses a typed tool call through a fake HTTP transport, test-local host-loop
glue translates the tool-call arguments into a ``JointCommand`` (there is
deliberately no production bridge; see docs/ROBOT_ADAPTERS.md and
examples/robot_adapters/isaac_rebot_agent_loop.py), the public
``validate_joint_command`` gate accepts or rejects it, and
``IsaacRos2ControlAdapter`` executes it through an injected fake ROS 2
transport. Observed goals, receipts, and post-action states are asserted
independently of the policy response.

Synthetic evidence marking: the Qwen HTTP seam and the ROS 2 transport are
fakes. No network call, real model inference, Isaac runtime, GPU, or robot
command occurs. Passing these tests is NOT a substitute for the PR #31 live
gates in docs/ROBOT_ADAPTER_ROADMAP_AND_ACCEPTANCE.md: Qwen live inference,
Isaac ROS 2 live action, and the end-to-end closed-loop canary. This test must
never be cited as a live PASS.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from simready.robotics.contracts import JointCommand, validate_joint_command
from simready.robotics.profiles import load_robot_profile
from simready.robotics.qwen_policy import QwenPolicyProvider
from simready.robotics.ros2_control_adapter import (
    GoalOutcome,
    IsaacRos2ControlAdapter,
    RclpyJointState,
    Ros2AdapterConfig,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_robot_profile(
    ROOT / "examples/robot_adapters/profiles/rebot-b601-rs.json"
)
ARM = list(PROFILE.groups["arm"])
ALL_JOINTS = list(PROFILE.joint_map())
TEST_KEY = "test-only-credential"  # never a real secret
TARGETS = {
    "joint1": 0.05,
    "joint2": 0.02,
    "joint3": 0.03,
    "joint4": 0.01,
    "joint5": 0.0,
    "joint6": 0.02,
}
TOOL_SCHEMA = (
    {
        "type": "function",
        "function": {
            "name": "set_arm_joint_targets",
            "description": "Set bounded radian targets for the six arm joints.",
            "parameters": {
                "type": "object",
                "properties": {name: {"type": "number"} for name in ARM},
                "required": list(ARM),
            },
        },
    },
)


# -- fakes -----------------------------------------------------------------


class FakeQwenTransport:
    """Callable HTTP seam: records requests, returns one canned response."""

    def __init__(self, response):
        self.response = response
        self.seen: list[dict] = []

    def __call__(self, url, headers, body, timeout_s):
        self.seen.append(
            {"url": url, "headers": dict(headers), "body": body, "timeout_s": timeout_s}
        )
        return self.response


class FakeClock:
    def __init__(self, now_ns: int = 10_000_000_000):
        self._now_ns = now_ns

    def now_ns(self) -> int:
        return self._now_ns

    def advance(self, seconds: float) -> None:
        self._now_ns += int(seconds * 1e9)


class FakeGoalHandle:
    def __init__(self, accepted, joint_names, positions):
        self.accepted = accepted
        self.joint_names = joint_names
        self.positions = positions

    @property
    def detail(self) -> str:
        return ""


class FakeTrajectoryClient:
    """Records goals and simulates the Isaac ros2_control plant.

    On a successful FollowJointTrajectory result the fake moves the injected
    joint-state source to the goal positions so post-action state can be
    observed independently of the policy response.
    """

    def __init__(self, source, *, accepted: bool = True, outcome=None):
        self._source = source
        self.server_ready = True
        self.accepted = accepted
        self.outcome = outcome or GoalOutcome(status="succeeded", error_code=0)
        self.sent: list[dict] = []

    def server_is_ready(self) -> bool:
        return self.server_ready

    def send_goal(self, joint_names, positions, duration_s, tolerance_s):
        self.sent.append(
            {
                "joint_names": list(joint_names),
                "positions": list(positions),
                "duration_s": duration_s,
                "tolerance_s": tolerance_s,
            }
        )
        return FakeGoalHandle(self.accepted, list(joint_names), list(positions))

    def wait_for_result(self, goal, timeout_s):
        if self.outcome.status == "succeeded":
            _publish_positions(
                self._source, dict(zip(goal.joint_names, goal.positions, strict=True))
            )
        return self.outcome


class FakeJointStateSource:
    def __init__(self, message: RclpyJointState | None = None):
        self.message = message
        self.spin_calls = 0

    def latest_state(self):
        return self.message

    def spin_once(self, timeout_s):
        self.spin_calls += 1


@dataclass
class FakeTransport:
    trajectory_client: FakeTrajectoryClient
    joint_state_source: FakeJointStateSource
    clock: FakeClock

    def close(self):
        pass


# -- helpers ---------------------------------------------------------------


def joint_state_message(positions_by_name: dict[str, float] | None = None):
    values = {name: 0.0 for name in ALL_JOINTS}
    values.update(positions_by_name or {})
    return RclpyJointState(
        name=list(ALL_JOINTS),
        position=[values[name] for name in ALL_JOINTS],
        velocity=[],
        effort=[],
    )


def _publish_positions(source, positions_by_name) -> None:
    merged = {name: 0.0 for name in ALL_JOINTS}
    current = source.message
    if current is not None:
        merged.update(dict(zip(current.name, current.position, strict=True)))
    merged.update(positions_by_name)
    source.message = joint_state_message(merged)


def make_adapter():
    clock = FakeClock()
    source = FakeJointStateSource(joint_state_message())
    client = FakeTrajectoryClient(source)
    transport = FakeTransport(
        trajectory_client=client, joint_state_source=source, clock=clock
    )
    adapter = IsaacRos2ControlAdapter(
        PROFILE, Ros2AdapterConfig(), transport=transport, clock=clock
    )
    return adapter, transport, clock


def tool_call_to_joint_command(arguments, clock, *, sequence: int = 1):
    """Test-local host-loop glue: typed tool-call arguments -> JointCommand."""
    return JointCommand(
        group="arm",
        positions=dict(arguments),
        issued_monotonic_ns=clock.now_ns(),
        valid_for_s=2.0,
        sequence=sequence,
    )


def _tool_call_response(arguments: dict, *, call_id: str = "call-1") -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "set_arm_joint_targets",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6},
    }


def _decide(qwen_env, arguments: dict):
    transport = FakeQwenTransport(_tool_call_response(arguments))
    decision = QwenPolicyProvider(transport=transport).decide(
        instructions="Choose small safe arm joint targets.",
        observation={"joint_positions": {name: 0.0 for name in ALL_JOINTS}},
        tools=TOOL_SCHEMA,
    )
    return decision, transport


@pytest.fixture()
def qwen_env(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.test/v1/chat/completions")
    monkeypatch.setenv("QWEN_MODEL", "qwen-3.8-test")
    monkeypatch.setenv("QWEN_API_KEY", TEST_KEY)


# -- happy path ------------------------------------------------------------


def test_tool_call_result_flows_through_validation_into_adapter(qwen_env):
    adapter, transport, clock = make_adapter()
    adapter.connect()
    observation = adapter.get_state()

    qwen = FakeQwenTransport(_tool_call_response(TARGETS))
    decision = QwenPolicyProvider(transport=qwen).decide(
        instructions="Choose small safe arm joint targets.",
        observation={"joint_positions": dict(observation.positions)},
        tools=TOOL_SCHEMA,
    )

    # The provider returned one typed, inert tool call; nothing was executed.
    assert decision.refusal is False
    assert decision.finish_reason == "tool_calls"
    (call,) = decision.calls
    assert call.name == "set_arm_joint_targets"
    assert call.arguments == TARGETS
    assert call.call_id == "call-1"
    assert call.index == 0
    assert len(qwen.seen) == 1
    request = qwen.seen[0]
    assert request["url"] == "https://qwen.test/v1/chat/completions"
    assert request["headers"]["Authorization"] == f"Bearer {TEST_KEY}"
    assert request["body"]["model"] == "qwen-3.8-test"
    assert request["body"]["tools"] == list(TOOL_SCHEMA)

    # Host-loop translation, then the public validation gate, then execution.
    command = tool_call_to_joint_command(call.arguments, clock)
    validate_joint_command(PROFILE, command, now_monotonic_ns=clock.now_ns())
    receipt = adapter.send_joint_position(command)

    # Independent assertions on the receipt and the observed adapter state.
    assert receipt.accepted is True
    assert receipt.sequence == 1
    assert "succeeded" in receipt.detail
    assert "measured=" in receipt.detail
    (sent,) = transport.trajectory_client.sent
    assert sent["joint_names"] == ARM
    assert sent["positions"] == [TARGETS[name] for name in ARM]
    assert sent["duration_s"] == pytest.approx(2.0)
    assert sent["tolerance_s"] == pytest.approx(adapter.config.goal_tolerance_s)
    observed = adapter.get_state()
    assert dict(observed.positions) == {
        **{name: 0.0 for name in ALL_JOINTS},
        **TARGETS,
    }
    assert observed.source == "isaac_ros2_control"
    assert "rclpy" not in sys.modules
    assert "isaacsim" not in sys.modules


# -- negative cases --------------------------------------------------------


def test_out_of_bounds_tool_call_is_rejected_before_any_goal(qwen_env):
    adapter, transport, clock = make_adapter()
    adapter.connect()
    # joint2 lower bound is 0.0; -0.04 is out of limits but under the 0.08
    # delta cap, so the contract-limit check (not the delta guard) rejects it.
    arguments = {**TARGETS, "joint2": -0.04}
    decision, _ = _decide(qwen_env, arguments)
    (call,) = decision.calls
    assert call.arguments == arguments

    command = tool_call_to_joint_command(call.arguments, clock)
    with pytest.raises(ValueError, match="outside"):
        validate_joint_command(PROFILE, command, now_monotonic_ns=clock.now_ns())
    with pytest.raises(ValueError, match="outside"):
        adapter.send_joint_position(command)
    assert transport.trajectory_client.sent == []

    # One bad action does not kill the session: a later valid command runs.
    good = tool_call_to_joint_command(TARGETS, clock)
    receipt = adapter.send_joint_position(good)
    assert receipt.accepted is True
    assert dict(adapter.get_state().positions) == {
        **{name: 0.0 for name in ALL_JOINTS},
        **TARGETS,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {name: value for name, value in TARGETS.items() if name != "joint6"},
        {**TARGETS, "elbow": 0.1},
    ],
    ids=["missing-joint", "unknown-joint"],
)
def test_malformed_tool_call_is_rejected_before_any_goal(qwen_env, arguments):
    adapter, transport, clock = make_adapter()
    adapter.connect()
    decision, _ = _decide(qwen_env, arguments)
    (call,) = decision.calls

    command = tool_call_to_joint_command(call.arguments, clock)
    with pytest.raises(ValueError, match="command joints differ from group"):
        validate_joint_command(PROFILE, command, now_monotonic_ns=clock.now_ns())
    with pytest.raises(ValueError, match="command joints differ from group"):
        adapter.send_joint_position(command)
    assert transport.trajectory_client.sent == []
    # State is unchanged: no goal was ever executed.
    assert dict(adapter.get_state().positions) == {name: 0.0 for name in ALL_JOINTS}
