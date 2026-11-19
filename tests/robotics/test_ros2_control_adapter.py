"""CPU tests for the optional Isaac ros2_control adapter (no rclpy needed).

All transports are fakes implementing the protocols injected into
``IsaacRos2ControlAdapter``; the rclpy transport is only exercised for its
missing-dependency failure path.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from simready.robotics.contracts import JointCommand
from simready.robotics.profiles import load_robot_profile
from simready.robotics.ros2_control_adapter import (
    GoalOutcome,
    IsaacRos2ControlAdapter,
    MonotonicClock,
    RclpyJointState,
    RclpyTransport,
    Ros2AdapterConfig,
    Ros2TransportError,
    _RclpyTrajectoryClient,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_robot_profile(
    ROOT / "examples/robot_adapters/profiles/rebot-b601-rs.json"
)
ARM = list(PROFILE.groups["arm"])


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
    detail: str = ""


class FakeTrajectoryClient:
    def __init__(self, outcome: GoalOutcome | None = None, accepted: bool = True):
        self.server_ready = True
        self.outcome = outcome or GoalOutcome(status="succeeded", error_code=0)
        self.accepted = accepted
        self.sent: list[dict] = []
        self.cancelled: list[FakeGoalHandle] = []
        self.cancel_succeeds = True
        self.send_error: Exception | None = None
        self.wait_error: Exception | None = None

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
        if self.send_error is not None:
            raise self.send_error
        return FakeGoalHandle(accepted=self.accepted)

    def wait_for_result(self, goal, timeout_s):
        if self.wait_error is not None:
            raise self.wait_error
        return self.outcome

    def cancel_goal(self, goal, timeout_s):
        self.cancelled.append(goal)
        return self.cancel_succeeds


class FakeJointStateSource:
    def __init__(self, message: RclpyJointState | None = None):
        self.message = message
        self.spin_calls = 0

    def latest_state(self):
        return self.message

    def spin_once(self, timeout_s):
        self.spin_calls += 1

    def publish(self, message: RclpyJointState) -> None:
        self.message = message


@dataclass
class FakeTransport:
    trajectory_client: FakeTrajectoryClient
    joint_state_source: FakeJointStateSource
    clock: FakeClock
    closed: bool = field(default=False)

    def close(self):
        self.closed = True


def joint_state_message(
    positions_by_name: dict[str, float] | None = None,
    *,
    order: list[str] | None = None,
    stamp_ns: int | None = None,
    received_monotonic_ns: int | None = None,
    velocities: dict[str, float] | None = None,
) -> RclpyJointState:
    names = order or list(PROFILE.joint_map())
    all_names = {name: 0.0 for name in PROFILE.joint_map()}
    all_names.update(positions_by_name or {})
    positions_by_name = all_names
    return RclpyJointState(
        name=list(names),
        position=[positions_by_name[name] for name in names],
        velocity=(
            [velocities[name] for name in names] if velocities is not None else []
        ),
        effort=[],
        stamp_ns=stamp_ns,
        received_monotonic_ns=received_monotonic_ns,
    )


def make_adapter(
    *,
    outcome: GoalOutcome | None = None,
    accepted: bool = True,
    state: RclpyJointState | None = None,
    clock: FakeClock | None = None,
    config: Ros2AdapterConfig | None = None,
) -> tuple[IsaacRos2ControlAdapter, FakeTransport, FakeClock]:
    clock = clock or FakeClock()
    transport = FakeTransport(
        trajectory_client=FakeTrajectoryClient(outcome=outcome, accepted=accepted),
        joint_state_source=FakeJointStateSource(state),
        clock=clock,
    )
    adapter = IsaacRos2ControlAdapter(
        PROFILE, config or Ros2AdapterConfig(), transport=transport, clock=clock
    )
    return adapter, transport, clock


def command(
    positions: dict[str, float],
    *,
    sequence: int = 1,
    clock: FakeClock | None = None,
    valid_for_s: float = 2.0,
) -> JointCommand:
    now = clock.now_ns() if clock else 10_000_000_000
    return JointCommand(
        group="arm",
        positions=positions,
        issued_monotonic_ns=now,
        valid_for_s=valid_for_s,
        sequence=sequence,
    )


# -- module hygiene -------------------------------------------------------


def test_module_import_does_not_load_rclpy():
    # This test module imported the adapter above, so these absence checks prove
    # its module import did not require the host's optional ROS packages.
    assert "rclpy" not in sys.modules
    assert "control_msgs" not in sys.modules
    assert "trajectory_msgs" not in sys.modules


def test_adapter_satisfies_robot_adapter_protocol():
    from simready.robotics.contracts import RobotAdapter

    adapter, _, _ = make_adapter()
    assert isinstance(adapter, RobotAdapter)


# -- config ---------------------------------------------------------------


def test_config_resolves_follow_joint_trajectory_action_name():
    flat = Ros2AdapterConfig()
    assert flat.follow_joint_trajectory_action == (
        "/rebot_arm_rs_isaac_trajectory_controller/follow_joint_trajectory"
    )
    namespaced = Ros2AdapterConfig(namespace="/rebot", controller_name="arm_controller")
    assert namespaced.follow_joint_trajectory_action == (
        "/rebot/arm_controller/follow_joint_trajectory"
    )


def test_config_rejects_bad_values():
    with pytest.raises(ValueError, match="goal_timeout_s must be positive"):
        Ros2AdapterConfig(goal_timeout_s=0)
    with pytest.raises(ValueError, match="controller_name must not start"):
        Ros2AdapterConfig(controller_name="/ctrl")
    with pytest.raises(ValueError, match="namespace must contain"):
        Ros2AdapterConfig(namespace="/")


@pytest.mark.parametrize(
    "invalid", [True, float("nan"), float("inf"), float("-inf"), "1"]
)
def test_config_rejects_nonfinite_nonreal_and_bool_positive_values(invalid):
    with pytest.raises(ValueError):
        Ros2AdapterConfig(goal_timeout_s=invalid)


# -- connect --------------------------------------------------------------


def test_connect_waits_then_succeeds_once_joint_states_cover_profile():
    class WaitingSource(FakeJointStateSource):
        def __init__(self):
            super().__init__(None)
            self.spins = 0

        def spin_once(self, timeout_s):
            self.spins += 1
            if self.spins >= 2:
                self.message = joint_state_message()

    clock = FakeClock()
    waiting = WaitingSource()
    transport = FakeTransport(
        trajectory_client=FakeTrajectoryClient(),
        joint_state_source=waiting,
        clock=clock,
    )
    adapter = IsaacRos2ControlAdapter(PROFILE, transport=transport, clock=clock)
    adapter.connect()
    assert waiting.spins >= 2


def test_connect_times_out_with_action_server_and_missing_joints_named():
    transport = FakeTransport(
        trajectory_client=FakeTrajectoryClient(),
        joint_state_source=FakeJointStateSource(joint_state_message()),
        clock=FakeClock(),
    )
    transport.trajectory_client.server_ready = False
    adapter = IsaacRos2ControlAdapter(
        PROFILE,
        Ros2AdapterConfig(connect_timeout_s=0.2, spin_period_s=0.01),
        transport=transport,
    )
    with pytest.raises(Ros2TransportError, match="action server"):
        adapter.connect()


def test_connect_requires_joint_state_coverage_of_profile_joints():
    partial = RclpyJointState(
        name=[ARM[0]],
        position=[0.0],
        velocity=[],
        effort=[],
    )
    transport = FakeTransport(
        trajectory_client=FakeTrajectoryClient(),
        joint_state_source=FakeJointStateSource(partial),
        clock=FakeClock(),
    )
    adapter = IsaacRos2ControlAdapter(
        PROFILE,
        Ros2AdapterConfig(connect_timeout_s=0.1, spin_period_s=0.01),
        transport=transport,
    )
    with pytest.raises(Ros2TransportError, match="complete /joint_states"):
        adapter.connect()


# -- get_state ------------------------------------------------------------


def test_get_state_normalizes_joint_order_and_stamps():
    clock = FakeClock(now_ns=20_000_000_000)
    reversed_order = list(reversed(PROFILE.joint_map()))
    positions = {
        name: 0.1 * (index + 1) for index, name in enumerate(PROFILE.joint_map())
    }
    message = RclpyJointState(
        name=reversed_order,
        position=[positions[name] for name in reversed_order],
        velocity=[0.0] * len(reversed_order),
        effort=[],
        stamp_ns=123_450_000_000,
    )
    adapter, _, _ = make_adapter(state=message, clock=clock)
    adapter.connect()
    state = adapter.get_state()
    assert set(state.positions) == set(PROFILE.joint_map())
    assert state.positions == positions
    assert state.source_timestamp_ns == 123_450_000_000
    assert state.source == "isaac_ros2_control"
    assert set(state.velocities) == set(PROFILE.joint_map())
    assert state.efforts == {}


def test_get_state_rejects_missing_joint_and_mismatched_vectors():
    all_names = list(PROFILE.joint_map())
    message = RclpyJointState(
        name=all_names[:-1], position=[0.0] * (len(all_names) - 1)
    )
    full = joint_state_message()
    source = FakeJointStateSource(full)
    adapter = IsaacRos2ControlAdapter(
        PROFILE,
        Ros2AdapterConfig(),
        transport=FakeTransport(
            trajectory_client=FakeTrajectoryClient(),
            joint_state_source=source,
            clock=FakeClock(),
        ),
        clock=FakeClock(),
    )
    adapter.connect()
    source.message = message
    with pytest.raises(Ros2TransportError, match="missing profile joints"):
        adapter.get_state()

    bad = RclpyJointState(name=all_names, position=[0.0] * (len(all_names) - 1))
    source2 = FakeJointStateSource(full)
    adapter2 = IsaacRos2ControlAdapter(
        PROFILE,
        Ros2AdapterConfig(),
        transport=FakeTransport(
            trajectory_client=FakeTrajectoryClient(),
            joint_state_source=source2,
            clock=FakeClock(),
        ),
        clock=FakeClock(),
    )
    adapter2.connect()
    source2.message = bad
    with pytest.raises(Ros2TransportError, match="position length"):
        adapter2.get_state()


def test_get_state_rejects_stale_receipt():
    clock = FakeClock(now_ns=20_000_000_000)
    stale = joint_state_message(received_monotonic_ns=10_000_000_000)
    adapter, _, _ = make_adapter(state=stale, clock=clock)
    adapter.connect()
    with pytest.raises(Ros2TransportError, match="stale"):
        adapter.get_state()


def test_rclpy_joint_state_callback_stamps_receive_monotonic_time():
    transport = RclpyTransport.__new__(RclpyTransport)
    message = SimpleNamespace(
        name=ARM,
        position=[0.0] * len(ARM),
        velocity=[],
        effort=[],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=34)),
    )
    before_ns = time.monotonic_ns()
    transport._on_joint_state(message)
    after_ns = time.monotonic_ns()

    assert before_ns <= transport._latest.received_monotonic_ns <= after_ns
    assert transport._latest.stamp_ns == 12_000_000_034


def test_get_state_raises_before_connect_and_without_state():
    adapter, _, _ = make_adapter()
    with pytest.raises(RuntimeError, match="connect\\(\\) must be called"):
        adapter.get_state()
    valid_state = joint_state_message()
    empty_source = FakeJointStateSource(valid_state)
    empty = FakeTransport(
        trajectory_client=FakeTrajectoryClient(),
        joint_state_source=empty_source,
        clock=FakeClock(),
    )
    empty_adapter = IsaacRos2ControlAdapter(
        PROFILE, Ros2AdapterConfig(), transport=empty, clock=FakeClock()
    )
    empty_adapter.connect()
    empty_source.message = None
    with pytest.raises(Ros2TransportError, match="no JointState"):
        empty_adapter.get_state()


# -- send_joint_position --------------------------------------------------


def test_send_goal_uses_group_order_duration_and_tolerance():
    clock = FakeClock()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    adapter, transport, _ = make_adapter(state=joint_state_message(), clock=clock)
    adapter.connect()
    receipt = adapter.send_joint_position(command(target, clock=clock))
    assert receipt.accepted
    sent = transport.trajectory_client.sent[0]
    assert sent["joint_names"] == ARM
    assert sent["positions"] == [target[name] for name in ARM]
    assert sent["tolerance_s"] == pytest.approx(adapter.config.goal_tolerance_s)


def test_send_caps_goal_duration_at_configured_max():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(state=joint_state_message(), clock=clock)
    adapter.connect()
    big = command({name: 0.0 for name in ARM}, clock=clock, valid_for_s=60.0)
    adapter.send_joint_position(big)
    assert transport.trajectory_client.sent[0]["duration_s"] == pytest.approx(
        adapter.config.max_goal_duration_s
    )


@pytest.mark.parametrize(
    "outcome",
    [
        GoalOutcome(status="succeeded", error_code=None),
        GoalOutcome(status="succeeded", error_code=1),
    ],
)
def test_send_rejects_succeeded_status_without_clean_trajectory_result(outcome):
    clock = FakeClock()
    adapter, _, _ = make_adapter(
        state=joint_state_message(), clock=clock, outcome=outcome
    )
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    receipt = adapter.send_joint_position(command(target, clock=clock))
    assert receipt.accepted is False
    assert f"error_code={outcome.error_code}" in receipt.detail


def test_send_accepts_only_on_successful_result_and_updates_sequence():
    clock = FakeClock()
    succeeded = GoalOutcome(status="succeeded", error_code=0, detail="")
    adapter, transport, _ = make_adapter(
        state=joint_state_message(), clock=clock, outcome=succeeded
    )
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    receipt = adapter.send_joint_position(command(target, clock=clock))
    assert receipt.accepted
    assert "succeeded" in receipt.detail
    assert "measured=" in receipt.detail


@pytest.mark.parametrize(
    "outcome",
    [
        GoalOutcome(status="aborted", error_code=-1, detail="physx"),
        GoalOutcome(status="preempted", error_code=-2, detail="replaced"),
        GoalOutcome(status="timeout", detail="slow"),
    ],
)
def test_send_returns_rejected_receipt_on_bad_goal_outcomes(outcome):
    clock = FakeClock()
    adapter, transport, _ = make_adapter(
        state=joint_state_message(), clock=clock, outcome=outcome
    )
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    receipt = adapter.send_joint_position(command(target, clock=clock))
    assert receipt.accepted is False
    if outcome.status == "timeout":
        assert "timed out" in receipt.detail
    else:
        assert outcome.status in receipt.detail
        assert f"error_code={outcome.error_code}" in receipt.detail
    # An accepted goal consumes its sequence even if its terminal result is
    # unsuccessful: use the next sequence for a host-loop retry. A timed-out
    # goal must first be canceled, and cancellation failure blocks new motion.
    if outcome.status == "timeout":
        assert len(transport.trajectory_client.cancelled) == 1
        assert "goal canceled" in receipt.detail
    retry = adapter.send_joint_position(command(target, clock=clock, sequence=2))
    assert retry.accepted is False
    succeeded = GoalOutcome(status="succeeded", error_code=0)
    adapter, _, _ = make_adapter(
        state=joint_state_message(), clock=clock, outcome=succeeded
    )
    adapter.connect()
    accepted = adapter.send_joint_position(command(target, clock=clock))
    assert accepted.accepted
    with pytest.raises(ValueError, match="sequence must increase"):
        adapter.send_joint_position(command(target, clock=clock, sequence=1))


def test_send_rejected_goal_returns_unaccepted_receipt_and_survives():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(
        state=joint_state_message(), clock=clock, accepted=False
    )
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    receipt = adapter.send_joint_position(command(target, clock=clock))
    assert receipt.accepted is False
    assert "rejected" in receipt.detail
    # A rejected goal must not consume the sequence: a later good command works.
    good = GoalOutcome(status="succeeded", error_code=0)
    transport.trajectory_client.accepted = True
    transport.trajectory_client.outcome = good
    receipt2 = adapter.send_joint_position(command(target, clock=clock, sequence=1))
    assert receipt2.accepted


def test_send_enforces_per_joint_delta_cap():
    clock = FakeClock()
    adapter, _, _ = make_adapter(state=joint_state_message(), clock=clock)
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = adapter.config.max_joint_delta + 0.01
    with pytest.raises(ValueError, match="delta cap"):
        adapter.send_joint_position(command(target, clock=clock))


def test_send_enforces_contract_validation_fail_closed():
    clock = FakeClock()
    adapter, _, _ = make_adapter(state=joint_state_message(), clock=clock)
    adapter.connect()
    out_of_limit = {name: 0.0 for name in ARM}
    # joint2 lower bound is 0.0; -0.04 stays under the 0.08 delta cap so the
    # contract-limit check (not the local delta guard) is what must reject it.
    out_of_limit[ARM[1]] = -0.04
    with pytest.raises(ValueError, match="outside"):
        adapter.send_joint_position(command(out_of_limit, clock=clock))
    expired = command(
        {name: 0.0 for name in ARM},
        clock=FakeClock(now_ns=20_000_000_000),
    )
    with pytest.raises(ValueError, match="expired"):
        adapter.send_joint_position(expired)
    stale_sequence = command({name: 0.0 for name in ARM}, clock=clock, sequence=-1)
    with pytest.raises(ValueError, match="sequence must be nonnegative"):
        adapter.send_joint_position(stale_sequence)
    # A valid accepted command consumes its sequence...
    good = command({name: 0.0 for name in ARM}, clock=clock, sequence=2)
    receipt = adapter.send_joint_position(good)
    assert receipt.accepted
    # ...and reusing that sequence afterwards is rejected.
    with pytest.raises(ValueError, match="sequence must increase"):
        adapter.send_joint_position(
            command({name: 0.0 for name in ARM}, clock=clock, sequence=2)
        )


def test_timeout_cancels_before_returning_rejection():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(
        state=joint_state_message(),
        clock=clock,
        outcome=GoalOutcome(status="timeout", detail="controller stalled"),
    )
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05

    receipt = adapter.send_joint_position(command(target, clock=clock))

    assert receipt.accepted is False
    assert len(transport.trajectory_client.cancelled) == 1
    assert "goal canceled" in receipt.detail
    # Once cancellation reaches a terminal result, the next monotonically newer
    # command is allowed and does not continue the old trajectory.
    transport.trajectory_client.outcome = GoalOutcome(status="succeeded", error_code=0)
    assert adapter.send_joint_position(
        command(target, clock=clock, sequence=2)
    ).accepted


def test_failed_timeout_cancellation_blocks_new_commands_until_stop():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(
        state=joint_state_message(),
        clock=clock,
        outcome=GoalOutcome(status="timeout"),
    )
    transport.trajectory_client.cancel_succeeds = False
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05

    receipt = adapter.send_joint_position(command(target, clock=clock))

    assert receipt.accepted is False
    assert "unresolved" in receipt.detail
    with pytest.raises(Ros2TransportError, match="previous goal is still unresolved"):
        adapter.send_joint_position(command(target, clock=clock, sequence=2))
    with pytest.raises(Ros2TransportError, match="could not confirm cancellation"):
        adapter.stop()
    assert len(transport.trajectory_client.cancelled) == 2
    assert len(transport.trajectory_client.sent) == 1
    adapter.close()
    with pytest.raises(RuntimeError, match="connect"):
        adapter.send_joint_position(command(target, clock=clock, sequence=2))


# -- stop/close -----------------------------------------------------------


def test_stop_holds_measured_positions():
    clock = FakeClock()
    positions = {name: 0.02 * (index + 1) for index, name in enumerate(ARM)}
    adapter, transport, _ = make_adapter(
        state=joint_state_message(positions), clock=clock
    )
    adapter.connect()
    # stop() needs a commanded group; send one arm command first.
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    adapter.send_joint_position(command(target, clock=clock))
    adapter.stop()
    sent = transport.trajectory_client.sent[-1]
    measured = adapter.get_state().positions
    assert sent["joint_names"] == ARM
    assert sent["positions"] == [measured[name] for name in ARM]


def test_stop_holds_only_the_commanded_group_not_other_joints():
    # Counterexample: with all joints in the state, stop() must not target the
    # gripper joints the controller never received; only the commanded group.
    clock = FakeClock()
    adapter, transport, _ = make_adapter(state=joint_state_message(), clock=clock)
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    adapter.send_joint_position(command(target, clock=clock))
    adapter.stop()
    hold = transport.trajectory_client.sent[-1]
    assert hold["joint_names"] == ARM
    assert "gripper_joint1" not in hold["joint_names"]


def test_stop_without_commanded_group_fails_closed_without_state_only_hold():
    adapter, transport, _ = make_adapter(state=joint_state_message())
    adapter.connect()
    with pytest.raises(Ros2TransportError, match="no commanded joint group"):
        adapter.stop()
    assert transport.trajectory_client.sent == []


def test_stop_without_state_raises():
    state = joint_state_message()
    source = FakeJointStateSource(state)
    adapter = IsaacRos2ControlAdapter(
        PROFILE,
        Ros2AdapterConfig(),
        transport=FakeTransport(
            trajectory_client=FakeTrajectoryClient(),
            joint_state_source=source,
            clock=FakeClock(),
        ),
        clock=FakeClock(),
    )
    adapter.connect()
    source.message = None
    with pytest.raises(Ros2TransportError, match="fresh JointState"):
        adapter.stop()


def test_stop_cancels_an_unresolved_goal_before_holding():
    clock = FakeClock()

    # A client whose wait_for_result never resolves leaves the goal active.
    class NeverResolvingClient(FakeTrajectoryClient):
        def wait_for_result(self, goal, timeout_s):
            # The original command times out; after its cancellation is
            # confirmed, the newly sent hold can complete normally.
            if len(self.sent) == 1:
                return GoalOutcome(status="timeout")
            return GoalOutcome(status="succeeded", error_code=0)

    source = FakeJointStateSource(joint_state_message())
    transport = FakeTransport(
        trajectory_client=NeverResolvingClient(),
        joint_state_source=source,
        clock=clock,
    )
    adapter = IsaacRos2ControlAdapter(PROFILE, transport=transport, clock=clock)
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    transport.trajectory_client.cancel_succeeds = False
    receipt = adapter.send_joint_position(command(target, clock=clock))
    assert receipt.accepted is False
    # Cancellation stays unconfirmed, so stop() must refuse to send a hold
    # trajectory that would compete with the still-running goal.
    with pytest.raises(Ros2TransportError, match="could not confirm cancellation"):
        adapter.stop()
    assert len(transport.trajectory_client.sent) == 1

    transport.trajectory_client.cancel_succeeds = True
    adapter.stop()

    # One failed timeout cancellation, one refused stop cancellation, then one
    # confirmed cancellation immediately before the hold trajectory.
    assert len(transport.trajectory_client.cancelled) == 3
    # The hold trajectory is the last goal sent, targeting the original arm
    # command group rather than all state joints.
    hold = transport.trajectory_client.sent[-1]
    assert hold["joint_names"] == ARM
    assert hold["positions"] == [adapter.get_state().positions[name] for name in ARM]


def test_get_state_stamps_the_receipt_time_of_the_message():
    # RobotState.received_monotonic_ns must carry when the transport received
    # the JointState, not when get_state() read it: the loop's max_age check
    # measures feedback age, not read latency.
    clock = FakeClock(now_ns=25_000_100_000)
    message = joint_state_message(received_monotonic_ns=25_000_000_000)
    adapter, _, _ = make_adapter(state=message, clock=clock)
    adapter.connect()
    state = adapter.get_state()
    assert state.received_monotonic_ns == 25_000_000_000


def test_close_is_idempotent_and_does_not_close_injected_transport():
    adapter, transport, _ = make_adapter(state=joint_state_message())
    adapter.connect()
    adapter.close()
    adapter.close()
    assert transport.closed is False
    with pytest.raises(RuntimeError, match="connect"):
        adapter.get_state()


def test_stop_hold_rejection_permanently_blocks_motion():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(
        state=joint_state_message(), clock=clock, accepted=False
    )
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    receipt = adapter.send_joint_position(command(target, clock=clock))
    assert receipt.accepted is False
    assert "goal rejected" in receipt.detail
    transport.trajectory_client.accepted = True
    adapter.send_joint_position(command(target, clock=clock, sequence=2))
    transport.trajectory_client.accepted = False
    with pytest.raises(Ros2TransportError, match="hold goal rejected"):
        adapter.stop()
    transport.trajectory_client.accepted = True
    # The failed hold permanently blocks motion: a retried stop and any new
    # send are refused.
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.stop()
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.send_joint_position(command(target, clock=clock, sequence=3))


def test_stop_hold_timeout_permanently_blocks_motion():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(
        state=joint_state_message(),
        clock=clock,
        outcome=GoalOutcome(status="timeout"),
    )
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    receipt = adapter.send_joint_position(command(target, clock=clock))
    assert receipt.accepted is False
    # Timeout on the original goal leaves it canceled, so stop() proceeds to
    # the hold, which times out (client still returns timeout outcomes).
    with pytest.raises(Ros2TransportError, match="hold goal timed out"):
        adapter.stop()
    transport.trajectory_client.outcome = GoalOutcome(status="succeeded", error_code=0)
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.stop()
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.send_joint_position(command(target, clock=clock, sequence=2))


def test_stop_hold_wait_exception_permanently_blocks_motion():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(state=joint_state_message(), clock=clock)
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    adapter.send_joint_position(command(target, clock=clock))
    transport.trajectory_client.wait_error = RuntimeError("controller disconnected")
    with pytest.raises(Ros2TransportError, match="hold goal result wait failed"):
        adapter.stop()
    transport.trajectory_client.wait_error = None
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.send_joint_position(command(target, clock=clock, sequence=2))


def test_stop_hold_failure_result_permanently_blocks_motion():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(
        state=joint_state_message(),
        clock=clock,
        outcome=GoalOutcome(status="aborted", error_code=-1),
    )
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    adapter.send_joint_position(command(target, clock=clock))
    with pytest.raises(Ros2TransportError, match="hold goal aborted"):
        adapter.stop()
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.send_joint_position(command(target, clock=clock, sequence=2))


def test_send_goal_exception_latches_permanent_dispatch_block():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(state=joint_state_message(), clock=clock)
    adapter.connect()
    target = {name: 0.0 for name in ARM}
    target[ARM[0]] = 0.05
    transport.trajectory_client.send_error = TimeoutError("no goal response")
    with pytest.raises(Ros2TransportError, match="dispatch outcome is unknown"):
        adapter.send_joint_position(command(target, clock=clock))
    transport.trajectory_client.send_error = None
    # Counterexample: a second send must not dispatch at all.
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.send_joint_position(command(target, clock=clock, sequence=2))
    assert len(transport.trajectory_client.sent) == 1
    # And stop() must not send a hold either.
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.stop()
    assert len(transport.trajectory_client.sent) == 1


def test_uncertain_dispatch_latch_survives_close_and_reconnect():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(state=joint_state_message(), clock=clock)
    adapter.connect()
    transport.trajectory_client.send_error = TimeoutError("no goal response")
    with pytest.raises(Ros2TransportError, match="dispatch outcome is unknown"):
        adapter.send_joint_position(command({name: 0.0 for name in ARM}, clock=clock))
    adapter.close()
    adapter.connect()
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.send_joint_position(
            command({name: 0.0 for name in ARM}, clock=clock, sequence=2)
        )


def test_close_with_unresolved_active_goal_latches_permanent_block():
    clock = FakeClock()
    adapter, transport, _ = make_adapter(
        state=joint_state_message(),
        clock=clock,
        outcome=GoalOutcome(status="timeout"),
    )
    transport.trajectory_client.cancel_succeeds = False
    adapter.connect()
    receipt = adapter.send_joint_position(
        command({name: 0.0 for name in ARM}, clock=clock)
    )
    assert receipt.accepted is False
    # The goal stays unresolved; close() must latch it before disconnecting.
    adapter.close()
    adapter.connect()
    with pytest.raises(Ros2TransportError, match="motion is blocked"):
        adapter.send_joint_position(
            command({name: 0.0 for name in ARM}, clock=clock, sequence=2)
        )


# -- rclpy transport (failure path only) ----------------------------------


def _rclpy_client():
    node = SimpleNamespace()
    client = _RclpyTrajectoryClient(
        rclpy=SimpleNamespace(), node=node, action_client=None
    )
    return client


class FakeFuture:
    """Deferred result future matching the rclpy ``done()/result()`` shape."""

    def __init__(self):
        self._done = False
        self._result: Any = None
        self._error: Exception | None = None

    def done(self) -> bool:
        return self._done

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def resolve(self, value: Any) -> None:
        self._result = value
        self._done = True

    def fail(self, error: Exception) -> None:
        self._error = error
        self._done = True


class CancelGoalHandle:
    """Goal handle whose cancel/result futures are fully controllable."""

    def __init__(self):
        self.cancel_calls = 0
        self.result_calls = 0

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self._cancel_future

    def get_result_async(self):
        self.result_calls += 1
        return self._result_future


def test_cancel_goal_requires_terminal_status_and_clean_cancel_response():
    # Cancel confirmation must be earned: a done cancel response plus a result
    # future whose status is non-terminal (CANCELING=8) is NOT confirmation.
    client = _rclpy_client()
    handle = CancelGoalHandle()
    handle._cancel_future = FakeFuture()
    handle._result_future = FakeFuture()

    def spin():
        for future, value in (
            (handle._cancel_future, SimpleNamespace(return_code=0)),
            (handle._result_future, SimpleNamespace(status=8)),
        ):
            if future.done():
                continue
            already = getattr(future, "_resolved_once", False)
            if already:
                future.resolve(value)
            else:
                future._resolved_once = True

    client._spin = spin
    assert client.cancel_goal(_WrapHandle(handle), 1.0) is False
    assert handle.cancel_calls == 1
    assert handle.result_calls == 1


class _WrapHandle:
    def __init__(self, handle):
        self._handle = handle


@pytest.mark.parametrize(
    "scenario",
    [
        "cancel_goal_async_raises",
        "cancel_future_never_done",
        "cancel_result_raises",
        "cancel_return_code_rejected",
        "cancel_return_code_unknown",
        "result_future_never_done",
        "result_result_raises",
        "get_result_async_raises",
        "result_status_executing",
        "result_status_unknown",
    ],
)
def test_cancel_goal_fail_closed_paths(scenario):
    client = _rclpy_client()
    handle = CancelGoalHandle()
    handle._cancel_future = FakeFuture()
    handle._result_future = FakeFuture()
    client._spin = lambda: None  # no progress; deadlines are wall-clock 1.0s

    if scenario == "cancel_goal_async_raises":
        handle.cancel_goal_async = lambda: (_ for _ in ()).throw(
            RuntimeError("cancel unavailable")
        )
    elif scenario == "cancel_future_never_done":
        pass
    elif scenario == "cancel_result_raises":
        handle._cancel_future.fail(RuntimeError("rclpy down"))
    elif scenario == "cancel_return_code_rejected":
        handle._cancel_future.resolve(SimpleNamespace(return_code=1))
    elif scenario == "cancel_return_code_unknown":
        handle._cancel_future.resolve(SimpleNamespace(return_code=None))
    elif scenario == "result_future_never_done":
        handle._cancel_future.resolve(SimpleNamespace(return_code=0))
    elif scenario == "get_result_async_raises":
        handle._cancel_future.resolve(SimpleNamespace(return_code=0))
        handle.get_result_async = lambda: (_ for _ in ()).throw(
            RuntimeError("no result channel")
        )
    elif scenario == "result_result_raises":
        handle._cancel_future.resolve(SimpleNamespace(return_code=0))
        handle._result_future.fail(RuntimeError("result channel broke"))
    elif scenario == "result_status_executing":
        handle._cancel_future.resolve(SimpleNamespace(return_code=0))
        handle._result_future.resolve(SimpleNamespace(status=2))
    elif scenario == "result_status_unknown":
        handle._cancel_future.resolve(SimpleNamespace(return_code=0))
        handle._result_future.resolve(SimpleNamespace(status=0))

    assert client.cancel_goal(_WrapHandle(handle), 0.05) is False


@pytest.mark.parametrize("terminal_status", [4, 5, 6])
def test_cancel_goal_confirms_terminal_status(terminal_status):
    client = _rclpy_client()
    handle = CancelGoalHandle()
    handle._cancel_future = FakeFuture()
    handle._result_future = FakeFuture()
    handle._cancel_future.resolve(SimpleNamespace(return_code=0))
    handle._result_future.resolve(SimpleNamespace(status=terminal_status))
    client._spin = lambda: None
    assert client.cancel_goal(_WrapHandle(handle), 0.05) is True


@pytest.mark.parametrize("terminated_code", [0, 3])
def test_cancel_goal_accepts_goal_terminated_return_code(terminated_code):
    client = _rclpy_client()
    handle = CancelGoalHandle()
    handle._cancel_future = FakeFuture()
    handle._result_future = FakeFuture()
    handle._cancel_future.resolve(SimpleNamespace(return_code=terminated_code))
    handle._result_future.resolve(SimpleNamespace(status=5))
    client._spin = lambda: None
    assert client.cancel_goal(_WrapHandle(handle), 0.05) is True


def test_cancel_goal_rejects_return_code_2():
    client = _rclpy_client()
    handle = CancelGoalHandle()
    handle._cancel_future = FakeFuture()
    handle._result_future = FakeFuture()
    handle._cancel_future.resolve(SimpleNamespace(return_code=2))
    handle._result_future.resolve(SimpleNamespace(status=5))
    client._spin = lambda: None
    assert client.cancel_goal(_WrapHandle(handle), 0.05) is False


def test_rclpy_transport_without_rclpy_raises_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_ros_import(name, *args, **kwargs):
        if name.split(".")[0] in {"rclpy"}:
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ros_import)
    with pytest.raises(Ros2TransportError, match="ROS 2 runtime not available"):
        RclpyTransport(PROFILE, Ros2AdapterConfig())


def test_default_transport_is_created_lazily_on_connect():
    adapter = IsaacRos2ControlAdapter(PROFILE)
    # No transport is constructed (and no ROS import attempted) until connect().
    assert adapter._transport is None
    assert adapter._owned_transport is True


def test_monotonic_clock_matches_time_module():
    import time

    before = time.monotonic_ns()
    value = MonotonicClock().now_ns()
    after = time.monotonic_ns()
    assert before <= value <= after
