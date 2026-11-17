"""CPU tests for the optional Isaac ros2_control adapter (no rclpy needed).

All transports are fakes implementing the protocols injected into
``IsaacRos2ControlAdapter``; the rclpy transport is only exercised for its
missing-dependency failure path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

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
        return FakeGoalHandle(accepted=self.accepted)

    def wait_for_result(self, goal, timeout_s):
        return self.outcome


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
    adapter, _, _ = make_adapter(
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
    # unsuccessful: use the next sequence for a host-loop retry.
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


# -- stop/close -----------------------------------------------------------


def test_stop_holds_measured_positions():
    clock = FakeClock()
    positions = {name: 0.02 * (index + 1) for index, name in enumerate(ARM)}
    adapter, transport, _ = make_adapter(
        state=joint_state_message(positions), clock=clock
    )
    adapter.connect()
    adapter.stop()
    sent = transport.trajectory_client.sent[0]
    measured = adapter.get_state().positions
    assert sent["joint_names"] == list(PROFILE.joint_map())
    assert sent["positions"] == [measured[name] for name in PROFILE.joint_map()]


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


def test_close_is_idempotent_and_does_not_close_injected_transport():
    adapter, transport, _ = make_adapter(state=joint_state_message())
    adapter.connect()
    adapter.close()
    adapter.close()
    assert transport.closed is False
    with pytest.raises(RuntimeError, match="connect"):
        adapter.get_state()


# -- rclpy transport (failure path only) ----------------------------------


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
