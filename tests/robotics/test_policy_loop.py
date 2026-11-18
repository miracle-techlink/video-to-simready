"""CPU-only fail-closed tests for the simulation policy-to-adapter loop.

The policy, ROS 2 transport, and verifier below are fakes. No Qwen endpoint,
Isaac runtime, network, GPU, or robot is contacted. These tests are synthetic
contract evidence, NOT a substitute for Qwen-live, Isaac-ROS2-live, or
end-to-end-live acceptance gates in docs/ROBOT_ADAPTER_ROADMAP_AND_ACCEPTANCE.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from simready.robotics.contracts import CommandReceipt, RobotProfile, RobotState
from simready.robotics.policy_loop import (
    PolicyAdapterLoop,
    PolicyLoopConfig,
    VerificationResult,
)
from simready.robotics.profiles import load_robot_profile
from simready.robotics.qwen_policy import (
    PolicyDecision,
    QwenApiError,
    QwenPolicyProvider,
    QwenTimeoutError,
    ToolCall,
)
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
TARGETS = {
    "joint1": 0.05,
    "joint2": 0.02,
    "joint3": 0.03,
    "joint4": 0.01,
    "joint5": 0.0,
    "joint6": 0.02,
}
CONFIG = PolicyLoopConfig(group="arm", tool_name="set_arm_joint_targets")
CONFIG_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": CONFIG.tool_name,
            "parameters": {
                "type": "object",
                "properties": {name: {"type": "number"} for name in ARM},
                "required": ARM,
            },
        },
    },
)
TEST_KEY = "test-only-credential"


class FakeQwenTransport:
    def __init__(self, response):
        self.response = response
        self.seen: list[dict] = []

    def __call__(self, url, headers, body, timeout_s):
        self.seen.append({"url": url, "headers": dict(headers), "body": body})
        return self.response


class FakeClock:
    def __init__(self, now_ns: int = 10_000_000_000):
        self._now_ns = now_ns

    def now_ns(self) -> int:
        return self._now_ns

    def advance(self, seconds: float) -> None:
        self._now_ns += int(seconds * 1e9)


@dataclass
class FakeGoalHandle:
    accepted: bool = True

    @property
    def detail(self) -> str:
        return ""


class FakeJointStateSource:
    def __init__(self, message: RclpyJointState):
        self.message = message

    def latest_state(self):
        return self.message

    def spin_once(self, timeout_s):
        pass


class FakeTrajectoryClient:
    def __init__(self, source: FakeJointStateSource):
        self.source = source
        self.accepted = True
        self.outcome = GoalOutcome(status="succeeded", error_code=0)
        self.sent: list[dict] = []

    def server_is_ready(self) -> bool:
        return True

    def send_goal(self, joint_names, positions, duration_s, tolerance_s):
        self.sent.append(
            {
                "joint_names": list(joint_names),
                "positions": list(positions),
                "duration_s": duration_s,
                "tolerance_s": tolerance_s,
            }
        )
        return FakeGoalHandle(accepted=self.accepted)

    def wait_for_result(self, goal, timeout_s):
        if self.outcome.succeeded:
            current = dict(
                zip(
                    self.source.message.name,
                    self.source.message.position,
                    strict=True,
                )
            )
            current.update(
                dict(
                    zip(
                        self.sent[-1]["joint_names"],
                        self.sent[-1]["positions"],
                        strict=True,
                    )
                )
            )
            self.source.message = joint_state_message(current)
        return self.outcome


@dataclass
class FakeTransport:
    trajectory_client: FakeTrajectoryClient
    joint_state_source: FakeJointStateSource
    clock: FakeClock

    def close(self):
        pass


def joint_state_message(positions: dict[str, float] | None = None) -> RclpyJointState:
    values = {name: 0.0 for name in ALL_JOINTS}
    values.update(positions or {})
    return RclpyJointState(
        name=list(ALL_JOINTS),
        position=[values[name] for name in ALL_JOINTS],
        velocity=[],
        effort=[],
    )


def make_adapter():
    clock = FakeClock()
    source = FakeJointStateSource(joint_state_message())
    client = FakeTrajectoryClient(source)
    transport = FakeTransport(client, source, clock)
    adapter = IsaacRos2ControlAdapter(
        PROFILE, Ros2AdapterConfig(), transport=transport, clock=clock
    )
    adapter.connect()
    return adapter, transport, clock


def tool_call_response(arguments: dict) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": CONFIG.tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                },
            }
        ]
    }


class FakePolicy:
    def __init__(self, decision=None, error: Exception | None = None):
        self.decision = decision
        self.error = error
        self.seen: list[dict] = []

    def decide(self, *, instructions, observation, tools, image_data_urls=()):
        self.seen.append(
            {
                "observation": dict(observation),
                "tools": tools,
                "image_data_urls": tuple(image_data_urls),
            }
        )
        if self.error is not None:
            raise self.error
        return self.decision


class ClockAdvancingPolicy(FakePolicy):
    """Simulates inference latency after the adapter has supplied state."""

    def __init__(self, decision, clock: FakeClock, delay_s: float):
        super().__init__(decision)
        self._clock = clock
        self._delay_s = delay_s

    def decide(self, *, instructions, observation, tools, image_data_urls=()):
        decision = super().decide(
            instructions=instructions,
            observation=observation,
            tools=tools,
            image_data_urls=image_data_urls,
        )
        self._clock.advance(self._delay_s)
        return decision


class FakeVerifier:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or VerificationResult(accepted=True, detail="tracked")
        self.error = error
        self.seen: list[dict] = []

    def verify(self, *, command, receipt, state_before, state_after, timeout_s):
        self.seen.append(
            {
                "command": command,
                "receipt": receipt,
                "before": state_before,
                "after": state_after,
                "timeout_s": timeout_s,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


def loop_for(policy, adapter, verifier, clock):
    return PolicyAdapterLoop(
        policy, adapter, verifier, CONFIG, now_monotonic_ns=clock.now_ns
    )


@pytest.fixture()
def qwen_env(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.test/v1/chat/completions")
    monkeypatch.setenv("QWEN_MODEL", "qwen-3.8-test")
    monkeypatch.setenv("QWEN_API_KEY", TEST_KEY)


# -- construction ----------------------------------------------------------


def test_configuration_and_adapter_scope_fail_closed():
    with pytest.raises(ValueError, match="positive number"):
        PolicyLoopConfig(group="arm", tool_name="move", state_timeout_s=0)
    with pytest.raises(ValueError, match="max_joint_step"):
        PolicyLoopConfig(group="arm", tool_name="move", max_joint_step=0)

    class HardwareAdapter:
        # A frozen dataclass profile cannot be flagged hardware-verified without
        # mutating the shared one, so build a locally flagged profile.
        execution_mode = "simulation"

        profile = RobotProfile(
            robot_id="hw-test-robot",
            display_name="Hardware Test Robot",
            joints=PROFILE.joints,
            groups=dict(PROFILE.groups),
            state_transport=PROFILE.state_transport,
            simulation_command_transport=PROFILE.simulation_command_transport,
            hardware_command_transport="hw",
            hardware_verified=True,
        )

        def get_state(self, *, timeout_s):
            raise AssertionError("never constructed with a hardware adapter")

        def send_joint_position(self, command):
            raise AssertionError("never constructed with a hardware adapter")

    with pytest.raises(ValueError, match="simulation-compatible"):
        PolicyAdapterLoop(FakePolicy(), HardwareAdapter(), FakeVerifier(), CONFIG)


def test_adapter_without_simulation_execution_mode_is_rejected():
    # The reBot profile ships an uncommissioned hardware command transport with
    # hardware_verified=False. A generic injected adapter without an explicit
    # simulation execution mode must be rejected even though the profile is not
    # hardware-verified: the commissioning flag alone proves nothing physical.
    class UnmarkedAdapter:
        profile = PROFILE

        def get_state(self, *, timeout_s):
            raise AssertionError("must not run without simulation capability")

        def send_joint_position(self, command):
            raise AssertionError("must not run without simulation capability")

    assert PROFILE.hardware_command_transport is not None
    assert PROFILE.hardware_verified is False
    with pytest.raises(ValueError, match="execution_mode='simulation'"):
        PolicyAdapterLoop(FakePolicy(), UnmarkedAdapter(), FakeVerifier(), CONFIG)


def test_simulation_adapters_declare_execution_mode():
    # Both production simulation adapters expose the marker the loop requires.
    from simready.robotics.isaac_sim import IsaacSimArticulationAdapter

    ros2_adapter, _, _ = make_adapter()
    assert ros2_adapter.execution_mode == "simulation"
    marker = getattr(IsaacSimArticulationAdapter, "execution_mode", None)
    assert isinstance(marker, property) and marker.fget.__name__ == "execution_mode"


def test_oversized_within_profile_delta_sends_no_goal():
    # joint1 target 1.4 rad is inside the profile limits but far beyond the
    # default 0.08 rad per-step cap measured against state_before.
    arguments = {**TARGETS, "joint1": 1.4}
    decision = PolicyDecision(
        calls=(ToolCall(name=CONFIG.tool_name, arguments=arguments),)
    )
    adapter, transport, clock = make_adapter()
    result = loop_for(FakePolicy(decision), adapter, FakeVerifier(), clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == "step_limit_exceeded"
    assert "step cap" in result.detail
    assert result.dispatched is False
    assert transport.trajectory_client.sent == []
    # A later step within the cap is unaffected.
    ok = loop_for(
        FakePolicy(valid_decision()), adapter, FakeVerifier(), clock
    ).run_once(instructions="go", observation={})
    assert ok.status == "verified"


def test_boundary_delta_at_step_cap_is_allowed():
    arguments = {**TARGETS, "joint1": CONFIG.max_joint_step}
    decision = PolicyDecision(
        calls=(ToolCall(name=CONFIG.tool_name, arguments=arguments),)
    )
    adapter, _, clock = make_adapter()
    result = loop_for(FakePolicy(decision), adapter, FakeVerifier(), clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == "verified"


def test_image_data_urls_forwarded_to_multimodal_blocks_without_logging_bytes(qwen_env):
    # End-to-end vision path: images passed to run_once(image_data_urls=...) are
    # forwarded to the provider and become OpenAI image_url content blocks in
    # the request; the raw data URL never appears in the result.
    adapter, _, clock = make_adapter()
    image = "data:image/png;base64,YWJjZGU="
    http = FakeQwenTransport(tool_call_response(TARGETS))
    loop = loop_for(QwenPolicyProvider(transport=http), adapter, FakeVerifier(), clock)
    result = loop.run_once(
        instructions="choose safe targets",
        observation={"task": "x"},
        image_data_urls=[image],
    )
    assert result.status == "verified"
    (request,) = http.seen
    content = request["body"]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1] == {"type": "image_url", "image_url": {"url": image}}
    assert image not in result.detail
    assert image not in str(result)


def test_stale_state_after_inference_latency_sends_no_goal():
    # Qwen latency ages state_before past state_timeout_s between the pre-
    # inference read and dispatch; the revalidation must fail closed.
    adapter, transport, clock = make_adapter()
    policy = ClockAdvancingPolicy(valid_decision(), clock, CONFIG.state_timeout_s + 0.1)
    result = loop_for(policy, adapter, FakeVerifier(), clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == "state_invalid"
    assert "no longer fresh" in result.detail
    assert result.dispatched is False
    assert transport.trajectory_client.sent == []


def test_fresh_state_after_short_inference_still_dispatches():
    adapter, transport, clock = make_adapter()
    policy = ClockAdvancingPolicy(valid_decision(), clock, 0.1)
    result = loop_for(policy, adapter, FakeVerifier(), clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == "verified"
    assert len(transport.trajectory_client.sent) == 1


def test_multiple_tool_calls_send_no_command():
    decision = PolicyDecision(
        calls=(
            ToolCall(name=CONFIG.tool_name, arguments=TARGETS),
            ToolCall(name=CONFIG.tool_name, arguments=TARGETS, index=1),
        )
    )
    adapter, transport, clock = make_adapter()
    result = loop_for(FakePolicy(decision), adapter, FakeVerifier(), clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == "no_single_tool_call"
    assert "2" in result.detail
    assert result.dispatched is False
    assert transport.trajectory_client.sent == []


def test_tool_schema_is_exactly_the_commandable_arm_group():
    adapter, _, clock = make_adapter()
    loop = loop_for(FakePolicy(), adapter, FakeVerifier(), clock)
    (schema,) = loop.tool_schema
    parameters = schema["function"]["parameters"]
    assert schema["function"]["name"] == CONFIG.tool_name
    assert set(parameters["properties"]) == set(ARM)
    assert set(parameters["required"]) == set(ARM)
    assert parameters["additionalProperties"] is False


# -- valid flow ------------------------------------------------------------


def test_qwen_tool_call_validates_dispatches_and_independently_verifies(qwen_env):
    adapter, transport, clock = make_adapter()
    verifier = FakeVerifier()
    http = FakeQwenTransport(tool_call_response(TARGETS))
    loop = loop_for(QwenPolicyProvider(transport=http), adapter, verifier, clock)

    result = loop.run_once(
        instructions="choose safe targets", observation={"task": "x"}
    )

    assert result.status == "verified"
    assert result.dispatched is True
    assert result.command.sequence == 0
    assert dict(result.command.positions) == TARGETS
    assert result.receipt.accepted is True
    (sent,) = transport.trajectory_client.sent
    assert sent["joint_names"] == ARM
    assert sent["positions"] == [TARGETS[name] for name in ARM]
    assert dict(result.state_after.positions) == {
        **{name: 0.0 for name in ALL_JOINTS},
        **TARGETS,
    }
    (check,) = verifier.seen
    assert dict(check["before"].positions) == {name: 0.0 for name in ALL_JOINTS}
    assert dict(check["after"].positions) == dict(result.state_after.positions)
    assert check["timeout_s"] == CONFIG.verification_timeout_s
    assert len(http.seen) == 1
    assert '"positions"' in http.seen[0]["body"]["messages"][-1]["content"]


def test_accepted_steps_increment_sequence(qwen_env):
    adapter, transport, clock = make_adapter()
    http = FakeQwenTransport(tool_call_response(TARGETS))
    loop = loop_for(QwenPolicyProvider(transport=http), adapter, FakeVerifier(), clock)
    first = loop.run_once(instructions="one", observation={})
    second = loop.run_once(instructions="two", observation={})
    assert (first.command.sequence, second.command.sequence) == (0, 1)
    assert len(transport.trajectory_client.sent) == 2


# -- fail closed before dispatch ------------------------------------------


@pytest.mark.parametrize(
    ("decision", "status"),
    [
        (PolicyDecision(refusal=True), "policy_refusal"),
        (PolicyDecision(calls=(), refusal=False), "no_single_tool_call"),
        (
            PolicyDecision(calls=(ToolCall(name="fire_lasers", arguments={}),)),
            "unknown_tool",
        ),
    ],
)
def test_refusal_empty_or_unknown_tool_sends_no_command(decision, status):
    adapter, transport, clock = make_adapter()
    result = loop_for(FakePolicy(decision), adapter, FakeVerifier(), clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == status
    assert result.dispatched is False
    assert transport.trajectory_client.sent == []
    assert result.receipt is None


@pytest.mark.parametrize(
    "arguments",
    [
        {name: value for name, value in TARGETS.items() if name != "joint6"},
        {**TARGETS, "joint6": "0.02"},
        {**TARGETS, "joint6": True},
        {**TARGETS, "joint2": -0.04},
    ],
    ids=["missing", "string", "bool", "out-of-limits"],
)
def test_malformed_or_invalid_arguments_send_no_command(arguments):
    adapter, transport, clock = make_adapter()
    decision = PolicyDecision(
        calls=(ToolCall(name=CONFIG.tool_name, arguments=arguments),)
    )
    result = loop_for(FakePolicy(decision), adapter, FakeVerifier(), clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == "validation_failed"
    assert result.command is not None
    assert result.dispatched is False
    assert transport.trajectory_client.sent == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (QwenTimeoutError("timeout"), "policy_timeout"),
        (TimeoutError("timeout"), "policy_timeout"),
        (QwenApiError(502, "bad gateway"), "policy_error"),
    ],
)
def test_policy_timeout_or_error_sends_no_command(error, status):
    adapter, transport, clock = make_adapter()
    result = loop_for(FakePolicy(error=error), adapter, FakeVerifier(), clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == status
    assert result.dispatched is False
    assert transport.trajectory_client.sent == []


def test_initial_state_timeout_sends_no_command():
    class TimeoutAdapter:
        execution_mode = "simulation"
        profile = PROFILE

        def get_state(self, *, timeout_s):
            raise TimeoutError("no feedback")

        def send_joint_position(self, command):
            raise AssertionError("must not dispatch")

    result = PolicyAdapterLoop(
        FakePolicy(), TimeoutAdapter(), FakeVerifier(), CONFIG
    ).run_once(instructions="go", observation={})
    assert result.status == "state_timeout"
    assert result.dispatched is False


def test_invalid_pre_command_state_sends_no_command():
    class InvalidStateAdapter:
        execution_mode = "simulation"
        profile = PROFILE

        def get_state(self, *, timeout_s):
            return RobotState(
                positions={name: 0.0 for name in ARM},
                received_monotonic_ns=10_000_000_000,
            )

        def send_joint_position(self, command):
            raise AssertionError("must not dispatch")

    result = PolicyAdapterLoop(
        FakePolicy(), InvalidStateAdapter(), FakeVerifier(), CONFIG
    ).run_once(instructions="go", observation={})
    assert result.status == "state_invalid"
    assert result.dispatched is False


@pytest.mark.parametrize("attribute", ["positions", "received_monotonic_ns"])
def test_malformed_state_type_errors_fail_closed_not_crash(attribute):
    # A transport that returns a malformed contract object (None fields,
    # non-int timestamp) must produce a state_invalid result, not an unhandled
    # TypeError escaping run_once's fail-closed contract.
    class MalformedStateAdapter:
        execution_mode = "simulation"
        profile = PROFILE

        def get_state(self, *, timeout_s):
            return RobotState(
                positions={name: 0.0 for name in ALL_JOINTS},
                received_monotonic_ns=(None if attribute == "received_monotonic_ns" else 10_000_000_000),
            )

        def send_joint_position(self, command):
            raise AssertionError("must not dispatch")

    result = PolicyAdapterLoop(
        FakePolicy(valid_decision()), MalformedStateAdapter(), FakeVerifier(), CONFIG
    ).run_once(instructions="go", observation={})
    if attribute == "received_monotonic_ns":
        assert result.status == "state_invalid"
        assert "cannot convert" in result.detail or "None" in result.detail
        assert result.dispatched is False
    else:
        assert result.status in {"state_invalid", "validation_failed"}
        assert result.dispatched is False


def test_adapter_exception_after_dispatch_fails_closed_without_success_claim():
    # send_joint_position raising must surface as adapter_error with no receipt
    # and never re-report the loop's last verified status.
    class ExplodingAdapter:
        execution_mode = "simulation"
        profile = PROFILE

        def get_state(self, *, timeout_s):
            return RobotState(
                positions={name: 0.0 for name in ALL_JOINTS},
                received_monotonic_ns=10_000_000_000,
            )

        def send_joint_position(self, command):
            raise RuntimeError("transport exploded")

    clock = FakeClock()
    loop = PolicyAdapterLoop(
        FakePolicy(valid_decision()),
        ExplodingAdapter(),
        FakeVerifier(),
        CONFIG,
        now_monotonic_ns=clock.now_ns,
    )
    result = loop.run_once(instructions="go", observation={})
    assert result.status == "adapter_error"
    assert result.dispatched is False
    assert "transport exploded" in result.detail


def test_adapter_timeout_after_dispatch_reports_timeout():
    class TimingOutAdapter:
        execution_mode = "simulation"
        profile = PROFILE

        def get_state(self, *, timeout_s):
            return RobotState(
                positions={name: 0.0 for name in ALL_JOINTS},
                received_monotonic_ns=10_000_000_000,
            )

        def send_joint_position(self, command):
            raise TimeoutError("goal watchdog expired")

    result = PolicyAdapterLoop(
        FakePolicy(valid_decision()),
        TimingOutAdapter(),
        FakeVerifier(),
        CONFIG,
        now_monotonic_ns=FakeClock().now_ns,
    ).run_once(instructions="go", observation={})
    assert result.status == "adapter_timeout"
    assert result.dispatched is False
    assert "watchdog" in result.detail


def test_sequence_wedge_on_adpater_exception_does_not_lock_sequence():
    # A valid command whose dispatch raises still consumes its sequence slot:
    # the next run's command uses sequence+1, not a reused lower value that
    # the adapter's monotonic check would then reject.
    class FirstExplodingAdapter:
        execution_mode = "simulation"
        profile = PROFILE
        calls = 0

        def get_state(self, *, timeout_s):
            return RobotState(
                positions={name: 0.0 for name in ALL_JOINTS},
                received_monotonic_ns=10_000_000_000,
            )

        def send_joint_position(self, command):
            FirstExplodingAdapter.calls += 1
            if FirstExplodingAdapter.calls == 1:
                raise RuntimeError("first dispatch explodes")
            return CommandReceipt(
                sequence=command.sequence,
                accepted=True,
                accepted_monotonic_ns=10_000_000_000,
            )

    clock = FakeClock()
    loop = PolicyAdapterLoop(
        FakePolicy(valid_decision()),
        FirstExplodingAdapter(),
        FakeVerifier(),
        CONFIG,
        now_monotonic_ns=clock.now_ns,
    )
    first = loop.run_once(instructions="go", observation={})
    assert first.status == "adapter_error"
    second = loop.run_once(instructions="go", observation={})
    assert second.status == "verified"
    assert (first.command.sequence, second.command.sequence) == (0, 1)


# -- failures after dispatch -----------------------------------------------


def valid_decision():
    return PolicyDecision(calls=(ToolCall(name=CONFIG.tool_name, arguments=TARGETS),))


def test_adapter_rejection_does_not_run_verifier():
    adapter, transport, clock = make_adapter()
    transport.trajectory_client.accepted = False
    verifier = FakeVerifier()
    result = loop_for(FakePolicy(valid_decision()), adapter, verifier, clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == "adapter_rejected"
    assert result.dispatched is True
    assert result.receipt.accepted is False
    assert verifier.seen == []


def test_verifier_rejection_is_not_a_verified_result():
    adapter, _, clock = make_adapter()
    verifier = FakeVerifier(VerificationResult(accepted=False, detail="tracking error"))
    result = loop_for(FakePolicy(valid_decision()), adapter, verifier, clock).run_once(
        instructions="go", observation={}
    )
    assert result.status == "verification_failed"
    assert result.dispatched is True
    assert result.verification.accepted is False


def test_verifier_timeout_is_reported():
    adapter, _, clock = make_adapter()
    result = loop_for(
        FakePolicy(valid_decision()),
        adapter,
        FakeVerifier(error=TimeoutError("verifier timed out")),
        clock,
    ).run_once(instructions="go", observation={})
    assert result.status == "verification_timeout"
    assert result.dispatched is True
    assert result.state_after is not None
    assert result.verification is None
