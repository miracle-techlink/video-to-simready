"""Optional ROS 2 adapter for Isaac Sim ``ros2_control`` (Module C).

Targets the Isaac Sim 6.x ``isaacsim.ros2.control`` route: the simulator hosts
an in-process ros2_control ``controller_manager`` for a USD articulation, and
this adapter drives one declared joint group through that controller's
``control_msgs/action/FollowJointTrajectory`` action server while normalizing
``sensor_msgs/msg/JointState`` feedback to the canonical contracts.

Neither ``rclpy`` nor any ROS message package is imported at module import
time. All ROS interaction goes through two transport protocols injected into
:class:`IsaacRos2ControlAdapter`; the default :class:`RclpyTransport`
implementation resolves them lazily inside ``connect()`` so the module (and the
CPU test suite) works without a ROS runtime. See ``docs/ISAAC_ROS2_ADAPTER.md``
for the Isaac-side Controller Manager, joint-name, clock, and YAML setup.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol

from simready.robotics.contracts import (
    CommandReceipt,
    JointCommand,
    RobotProfile,
    RobotState,
    validate_joint_command,
)

STATE_SOURCE = "isaac_ros2_control"


class Ros2TransportError(RuntimeError):
    """A ROS 2 transport dependency or interaction failed."""


@dataclass(frozen=True)
class Ros2AdapterConfig:
    """Names and timing limits for one ros2_control controller manager.

    ``controller_name`` is the ros2_control controller (not the manager): the
    FollowJointTrajectory action server lives at
    ``/<namespace>/<controller_name>/follow_joint_trajectory``.
    """

    namespace: str = ""
    controller_name: str = "rebot_arm_rs_isaac_trajectory_controller"
    joint_states_topic: str = "/joint_states"
    state_max_age_s: float = 0.5
    connect_timeout_s: float = 10.0
    server_wait_timeout_s: float = 10.0
    goal_timeout_s: float = 30.0
    goal_tolerance_s: float = 0.05
    max_joint_delta: float = 0.08
    max_goal_duration_s: float = 5.0
    spin_period_s: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            "state_max_age_s",
            "connect_timeout_s",
            "server_wait_timeout_s",
            "goal_timeout_s",
            "goal_tolerance_s",
            "max_joint_delta",
            "max_goal_duration_s",
            "spin_period_s",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"{name} must be a real number, got {type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value}")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.controller_name.startswith("/"):
            raise ValueError("controller_name must not start with '/'")
        if self.namespace and not self.namespace.strip("/"):
            raise ValueError("namespace must contain a nonempty path segment")

    @property
    def follow_joint_trajectory_action(self) -> str:
        ns = self.namespace.strip("/")
        name = f"{self.controller_name}/follow_joint_trajectory"
        return f"/{ns}/{name}" if ns else f"/{name}"


@dataclass(frozen=True)
class GoalOutcome:
    """Normalized FollowJointTrajectory result for one sent goal.

    ``status`` is one of ``succeeded``, ``aborted``, ``preempted``,
    ``rejected``, ``timeout``. ``error_code`` mirrors
    ``control_msgs/FollowJointTrajectory.Result`` when the server reported one.

    ``succeeded`` requires both the terminal status ``succeeded`` and
    ``error_code == 0``: FollowJointTrajectory results can carry a nonzero
    error_code even alongside an OK-looking status, and a missing error_code
    means the server never confirmed a clean result.
    """

    status: str
    error_code: int | None = None
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded" and self.error_code == 0


class GoalHandleLike(Protocol):
    """The accepted-goal handle returned by ``TrajectoryClientLike.send_goal``."""

    accepted: bool

    @property
    def detail(self) -> str: ...


class TrajectoryClientLike(Protocol):
    """Monitored FollowJointTrajectory client side."""

    def server_is_ready(self) -> bool: ...

    def send_goal(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        duration_s: float,
        tolerance_s: float,
    ) -> GoalHandleLike: ...

    def wait_for_result(
        self, goal: GoalHandleLike, timeout_s: float
    ) -> GoalOutcome: ...

    def cancel_goal(self, goal: GoalHandleLike, timeout_s: float) -> bool: ...


class JointStateLike(Protocol):
    """The ``sensor_msgs/msg/JointState`` shape the adapter consumes.

    Message arrays are parallel to ``name``. ``received_monotonic_ns`` is the
    local receipt time the cache recorded (defaults to "now" for plain
    objects, i.e. not stale); ``stamp_ns`` carries the ROS header stamp when
    available.
    """

    name: Sequence[str]
    position: Sequence[float]
    velocity: Sequence[float]
    effort: Sequence[float]


class JointStateSourceLike(Protocol):
    """Latest cached JointState; populated by the transport's subscription."""

    def latest_state(self) -> JointStateLike | None: ...

    def spin_once(self, timeout_s: float) -> None: ...


class ClockLike(Protocol):
    """Local receipt clock; monotonic nanoseconds like ``time.monotonic_ns``."""

    def now_ns(self) -> int: ...


class Transport(Protocol):
    """The ROS 2 runtime surface this adapter needs, injected for tests."""

    @property
    def trajectory_client(self) -> TrajectoryClientLike: ...

    @property
    def joint_state_source(self) -> JointStateSourceLike: ...

    @property
    def clock(self) -> ClockLike: ...

    def close(self) -> None: ...


class MonotonicClock:
    """Default ``ClockLike`` implementation over ``time.monotonic_ns``."""

    def now_ns(self) -> int:
        return time.monotonic_ns()


def _stamp_ns(message: Any) -> int | None:
    cached_stamp = getattr(message, "stamp_ns", None)
    if cached_stamp is not None:
        return int(cached_stamp)
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    seconds = getattr(stamp, "sec", 0)
    nanoseconds = getattr(stamp, "nanosec", 0)
    if seconds is None or nanoseconds is None:
        return None
    return int(seconds) * 1_000_000_000 + int(nanoseconds)


def _field_map(message: JointStateLike, values: Sequence[float]) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(message.name)}


class IsaacRos2ControlAdapter:
    """``RobotAdapter`` over an Isaac-hosted ros2_control controller manager.

    The adapter owns no Isaac/ROS lifecycle: the caller brings up the
    simulation, the in-process ``controller_manager`` (Isaac
    ``Ros2ControlManager.setup``), the ``/clock`` publisher, and controller
    activation, then injects a ``Transport``. With ``transport=None`` a
    :class:`RclpyTransport` is created on ``connect()`` and requires a real
    ROS 2 runtime.
    """

    def __init__(
        self,
        profile: RobotProfile,
        config: Ros2AdapterConfig | None = None,
        *,
        transport: Transport | None = None,
        clock: ClockLike | None = None,
    ):
        self.profile = profile
        self._config = config or Ros2AdapterConfig()
        self._transport = transport
        self._clock = clock
        self._owned_transport = transport is None
        self._connected = False
        self._last_sequence = -1
        self._last_positions: Mapping[str, float] | None = None
        self._active_goal: GoalHandleLike | None = None
        self._commanded_group: str | None = None
        # A send_goal failure can occur after the server accepted the request but
        # before this process received a handle. Without a handle it cannot
        # cancel or observe the trajectory, so only an external controller stop
        # followed by a new adapter instance may resume motion.
        self._dispatch_uncertain = False
        self._motion_blocked_reason: str | None = None

    @property
    def config(self) -> Ros2AdapterConfig:
        return self._config

    @property
    def execution_mode(self) -> str:
        """This adapter drives Isaac-hosted ros2_control simulation only."""
        return "simulation"

    def connect(self) -> None:
        if self._connected:
            return
        transport = self._transport
        if transport is None:
            transport = RclpyTransport(self.profile, self._config)
            self._transport = transport
        clock = self._clock or transport.clock
        self._clock = clock
        deadline = time.monotonic() + self._config.connect_timeout_s
        while time.monotonic() < deadline:
            if (
                transport.trajectory_client.server_is_ready()
                and not self._state_coverage(
                    transport.joint_state_source.latest_state()
                )
            ):
                self._connected = True
                return
            transport.joint_state_source.spin_once(self._config.spin_period_s)
        raise Ros2TransportError(
            "connect timed out waiting for the ros2_control action server "
            f"{self._config.follow_joint_trajectory_action!r} and a complete "
            f"/joint_states sample for joints {sorted(self.profile.joint_map())}"
        )

    def get_state(self, *, timeout_s: float = 0.0) -> RobotState:
        self._require_connected()
        source = self._transport.joint_state_source
        message = source.latest_state()
        deadline = self._clock.now_ns() + int(max(0.0, timeout_s) * 1e9)
        while message is None and self._clock.now_ns() < deadline:
            source.spin_once(self._config.spin_period_s)
            message = source.latest_state()
        if message is None:
            raise Ros2TransportError(
                f"no JointState received on {self._config.joint_states_topic}"
            )
        missing = self._state_coverage(message)
        if missing:
            raise Ros2TransportError(
                f"JointState on {self._config.joint_states_topic} is missing "
                f"profile joints: {sorted(missing)}"
            )
        self._validate_joint_state_vectors(message)
        received_ns = getattr(message, "received_monotonic_ns", None)
        if received_ns is None:
            received_ns = self._clock.now_ns()
        age_ns = self._clock.now_ns() - int(received_ns)
        if age_ns < 0 or age_ns > self._config.state_max_age_s * 1e9:
            raise Ros2TransportError(
                f"JointState receipt is stale or from the future: age_ns={age_ns}"
            )
        positions = _field_map(message, message.position)
        velocities = _field_map(message, message.velocity) if message.velocity else {}
        efforts = _field_map(message, message.effort) if message.effort else {}
        state = RobotState(
            positions={name: positions[name] for name in self.profile.joint_map()},
            received_monotonic_ns=int(received_ns),
            source_timestamp_ns=_stamp_ns(message),
            velocities=(
                {name: velocities[name] for name in self.profile.joint_map()}
                if velocities
                else {}
            ),
            efforts=(
                {name: efforts[name] for name in self.profile.joint_map()}
                if efforts
                else {}
            ),
            source=STATE_SOURCE,
        )
        self._last_positions = state.positions
        return state

    def send_joint_position(self, command: JointCommand) -> CommandReceipt:
        self._require_connected()
        now = self._clock.now_ns()
        validate_joint_command(self.profile, command, now_monotonic_ns=now)
        if command.sequence <= self._last_sequence:
            raise ValueError("command sequence must increase monotonically")
        self._require_motion_unblocked()
        if self._active_goal is not None:
            # A previous goal never reached a terminal result (caller ignored a
            # receipt, or the host loop crashed between send and report). Fail
            # closed instead of issuing an untracked goal whose outcome could
            # silently interleave with the last trajectory.
            raise Ros2TransportError(
                "a previous goal is still unresolved; call stop() to cancel and "
                "re-sync before sending the next command"
            )
        positions = {name: float(command.positions[name]) for name in command.positions}
        self._check_joint_delta(positions)
        duration = min(command.valid_for_s, self._config.max_goal_duration_s)
        client = self._transport.trajectory_client
        try:
            goal = client.send_goal(
                list(self.profile.groups[command.group]),
                [positions[name] for name in self.profile.groups[command.group]],
                duration,
                self._config.goal_tolerance_s,
            )
        except Exception as error:
            self._block_uncertain_dispatch(error)
            raise Ros2TransportError(
                "goal dispatch outcome is unknown; no further motion or hold may "
                "be sent by this adapter. Stop the external controller and create "
                "a new adapter instance before resuming"
            ) from error
        if not goal.accepted:
            return self._receipt(
                command, accepted=False, detail=f"goal rejected: {goal.detail}"
            )
        self._last_sequence = command.sequence
        self._active_goal = goal
        self._commanded_group = command.group
        try:
            outcome = client.wait_for_result(goal, self._config.goal_timeout_s)
        except Exception as error:
            cancelled = self._cancel_silent(goal)
            if cancelled:
                self._active_goal = None
            return self._receipt(
                command,
                accepted=False,
                detail=(
                    f"goal result wait failed: {error}; "
                    f"{'goal canceled' if cancelled else 'goal remains unresolved; call stop() or externally stop the controller'}"
                ),
            )
        if outcome.status == "timeout":
            # Cancel before reading feedback so a timeout cannot leave the old
            # trajectory running while the caller receives a safe failure.
            cancelled = self._cancel_silent(goal)
            if cancelled:
                self._active_goal = None
            measured = self._safe_state()
            tracking = (
                f"; measured={dict(measured.positions)}" if measured is not None else ""
            )
            return self._receipt(
                command,
                accepted=False,
                detail=(
                    f"goal timed out after {self._config.goal_timeout_s}s without a result; "
                    f"{'goal canceled' if cancelled else 'goal remains unresolved; call stop() or close()'}"
                    f"{tracking}"
                ),
            )
        self._active_goal = None
        measured = self._safe_state()
        tracking = ""
        if measured is not None:
            tracking = f"; measured={dict(measured.positions)}"
        if not outcome.succeeded:
            return self._receipt(
                command,
                accepted=False,
                detail=f"goal {outcome.status} error_code={outcome.error_code} "
                f"{outcome.detail}{tracking}".strip(),
            )
        return self._receipt(
            command,
            accepted=True,
            detail=f"FollowJointTrajectory succeeded (error_code={outcome.error_code})"
            f"{tracking}",
        )

    def stop(self) -> None:
        """Cancel any unresolved goal, then hold at the last measured positions.

        This is a command-stream hold: it cancels an in-flight FollowJointTrajectory
        goal (bounded by ``goal_timeout_s`` and verified against a terminal
        result), then re-targets the freshest measured positions of the joint
        group this controller actually received commands for, with one short
        trajectory. The hold goal is itself tracked: a timeout, exception, or
        rejected hold permanently blocks this adapter from further motion and
        raises ``Ros2TransportError``. If the adapter never sent a command, no
        hold can be derived and stop() raises instead of inventing one. It is
        not an E-stop, controller deactivation, collision response, or hardware
        safety mechanism; Isaac's playback-tied controller manager remains the
        final authority on motion.
        """
        self._require_connected()
        self._require_motion_unblocked()
        client = self._transport.trajectory_client
        if self._active_goal is not None:
            goal = self._active_goal
            if not self._cancel_silent(goal):
                # Do not send a competing hold trajectory while the preceding
                # goal might still be driving the controller. Keep the adapter
                # blocked (_active_goal stays set); a retried stop() may
                # attempt the cancellation again.
                raise Ros2TransportError(
                    "stop() could not confirm cancellation of the active goal; "
                    "no hold command was sent"
                )
            self._active_goal = None
        measured = self._safe_state()
        if measured is None:
            raise Ros2TransportError(
                "stop() needs a fresh JointState; no state received yet"
            )
        if self._commanded_group is None:
            # A state-only hold can invent motion for joints that were never
            # commanded (e.g. a gripper) and is therefore not a safe stop.
            raise Ros2TransportError(
                "stop() has no commanded joint group to hold; no command has "
                "reached this controller through this adapter"
            )
        names = list(self.profile.groups[self._commanded_group])
        hold_target = {name: measured.positions[name] for name in names}
        try:
            hold_goal = client.send_goal(
                names,
                [hold_target[name] for name in names],
                self._config.spin_period_s,
                self._config.goal_tolerance_s,
            )
        except Exception as error:
            self._block_uncertain_dispatch(error)
            raise Ros2TransportError(
                "hold dispatch outcome is unknown; no further motion may be "
                "sent by this adapter. Stop the external controller and create "
                "a new adapter instance before resuming"
            ) from error
        if not hold_goal.accepted:
            self._block_motion(f"hold goal rejected: {hold_goal.detail}")
            raise Ros2TransportError(f"hold goal rejected: {hold_goal.detail}")
        self._active_goal = hold_goal
        try:
            outcome = client.wait_for_result(hold_goal, self._config.goal_timeout_s)
        except Exception as error:
            cancelled = self._cancel_silent(hold_goal)
            if cancelled:
                self._active_goal = None
            self._block_motion(f"hold goal result wait failed: {error}")
            raise Ros2TransportError(
                f"hold goal result wait failed: {error}; "
                f"{'hold canceled' if cancelled else 'hold remains unresolved'}"
            ) from error
        if outcome.status == "timeout":
            cancelled = self._cancel_silent(hold_goal)
            if cancelled:
                self._active_goal = None
            self._block_motion(
                f"hold goal timed out after {self._config.goal_timeout_s}s"
            )
            raise Ros2TransportError(
                f"hold goal timed out after {self._config.goal_timeout_s}s; "
                f"{'hold canceled' if cancelled else 'hold remains unresolved'}"
            )
        self._active_goal = None
        if not outcome.succeeded:
            self._block_motion(
                f"hold goal {outcome.status} error_code={outcome.error_code}"
            )
            raise Ros2TransportError(
                f"hold goal {outcome.status} error_code={outcome.error_code} "
                f"{outcome.detail}".strip()
            )

    def close(self) -> None:
        """Release this adapter's ROS resources only; never a motion stop.

        Dropping the transport connection does not stop, cancel, or confirm any
        trajectory: an unresolved goal, an unconfirmed cancellation, or an
        unknown dispatch remains the external controller's responsibility to
        stop. Before disconnecting, any unresolved active goal and any
        uncertain dispatch are latched into a permanent block that survives
        close()/connect() cycles on this instance, so a reconnect cannot send
        new motion on top of an untracked trajectory. After close() this
        instance rejects all motion; resuming motion requires stopping the
        controller externally and constructing a new adapter.
        """
        if self._active_goal is not None:
            self._block_motion("close() called with an unresolved active goal")
        self._connected = False
        self._active_goal = None
        if self._owned_transport and self._transport is not None:
            self._transport.close()
            self._transport = None

    # -- helpers -----------------------------------------------------------

    def _state_coverage(self, message: JointStateLike | None) -> set[str]:
        if message is None:
            return set(self.profile.joint_map())
        return set(self.profile.joint_map()) - set(message.name)

    def _validate_joint_state_vectors(self, message: JointStateLike) -> None:
        names_count = len(message.name)
        fields = (
            ("position", message.position),
            ("velocity", message.velocity),
            ("effort", message.effort),
        )
        for name, values in fields:
            if values and len(values) != names_count:
                raise Ros2TransportError(
                    f"JointState {name} length {len(values)} does not match "
                    f"name length {names_count}"
                )
        if len(message.position) != names_count:
            raise Ros2TransportError(
                f"JointState position length {len(message.position)} does not "
                f"match name length {names_count}"
            )

    def _check_joint_delta(self, positions: Mapping[str, float]) -> None:
        reference = self._last_positions or self._safe_state_snapshot()
        if not reference:
            return
        deltas = {
            name: abs(positions[name] - reference[name])
            for name in positions
            if name in reference
        }
        worst = max(deltas.values(), default=0.0)
        if worst > self._config.max_joint_delta:
            raise ValueError(
                f"command exceeds per-joint delta cap "
                f"{self._config.max_joint_delta}: {deltas}"
            )

    def _safe_state(self) -> RobotState | None:
        try:
            return self.get_state()
        except Ros2TransportError:
            return None

    def _block_uncertain_dispatch(self, error: Exception) -> None:
        """Latch that a dispatched goal may exist without a known handle."""
        self._dispatch_uncertain = True
        self._block_motion(f"goal dispatch outcome is unknown: {error}")

    def _block_motion(self, reason: str) -> None:
        """Latch an unsafe motion state until this adapter is discarded.

        Neither close() nor reconnect clears the latch: a closed connection
        cannot confirm a trajectory ended, so pretending otherwise would invent
        a terminal state for unknown motion. Resumption always goes through an
        externally stopped controller and a new adapter instance.
        """
        self._motion_blocked_reason = reason

    def _require_motion_unblocked(self) -> None:
        if self._motion_blocked_reason is None:
            return
        if self._dispatch_uncertain:
            raise Ros2TransportError(
                "motion is blocked because a prior goal dispatch outcome is "
                "unknown; stop the external controller and create a new adapter "
                "instance before resuming"
            )
        raise Ros2TransportError(
            f"motion is blocked: {self._motion_blocked_reason}; stop the external "
            "controller and create a new adapter instance before resuming"
        )

    def _cancel_silent(self, goal: GoalHandleLike) -> bool:
        """Best-effort bounded cancellation that never masks a timeout report."""
        try:
            return bool(
                self._transport.trajectory_client.cancel_goal(
                    goal, self._config.goal_timeout_s
                )
            )
        except Exception:
            return False

    def _safe_state_snapshot(self) -> Mapping[str, float]:
        state = self._safe_state()
        return state.positions if state is not None else {}

    def _receipt(
        self, command: JointCommand, *, accepted: bool, detail: str
    ) -> CommandReceipt:
        return CommandReceipt(
            sequence=command.sequence,
            accepted=accepted,
            accepted_monotonic_ns=self._clock.now_ns(),
            detail=detail,
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "connect() must be called before using the ROS 2 adapter"
            )


@dataclass
class RclpyJointState:
    """Plain JointState mirror for tests and non-rclpy callers.

    ``received_monotonic_ns`` lets a cache stamp the moment the message
    arrived; tests omit it to mean "fresh right now".
    """

    name: list[str] = field(default_factory=list)
    position: list[float] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)
    stamp_ns: int | None = None
    received_monotonic_ns: int | None = None


class RclpyTransport:
    """Default ``Transport`` over ``rclpy``; created lazily, never imported early.

    Subscribes to ``config.joint_states_topic`` with a sensor-data QoS and
    exposes one action client for the controller's FollowJointTrajectory
    server. Requires the ROS 2 runtime that Isaac's in-process controller
    manager provides (see ``docs/ISAAC_ROS2_ADAPTER.md``).
    """

    def __init__(self, profile: RobotProfile, config: Ros2AdapterConfig):
        self._profile = profile
        self._config = config
        try:
            import rclpy
            from rclpy.action import ActionClient
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
        except ImportError as error:
            raise Ros2TransportError(
                "ROS 2 runtime not available: install/source a ROS 2 Jazzy "
                "environment (Isaac Sim bundles rclpy and control_msgs via "
                "isaacsim.ros2.core) before creating RclpyTransport"
            ) from error
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
            self._initialized_rclpy = True
        else:
            self._initialized_rclpy = False
        self._node = Node("simready_ros2_control_adapter")
        self._latest: RclpyJointState | None = None
        self._action_client = ActionClient(
            self._node,
            self._follow_joint_trajectory_type(),
            config.follow_joint_trajectory_action,
        )
        self._subscription = self._node.create_subscription(
            self._joint_state_type(),
            config.joint_states_topic,
            self._on_joint_state,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

    # -- Transport protocol surface ----------------------------------------

    @property
    def trajectory_client(self) -> TrajectoryClientLike:
        return self._trajectory_client

    @property
    def joint_state_source(self) -> JointStateSourceLike:
        return self._joint_state_source

    @property
    def clock(self) -> ClockLike:
        return MonotonicClock()

    def close(self) -> None:
        self._node.destroy_subscription(self._subscription)
        self._action_client.destroy()
        self._node.destroy_node()
        if self._initialized_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()

    # -- internals -----------------------------------------------------------

    def _on_joint_state(self, message: Any) -> None:
        self._latest = RclpyJointState(
            name=list(message.name),
            position=[float(value) for value in message.position],
            velocity=[float(value) for value in message.velocity],
            effort=[float(value) for value in message.effort],
            stamp_ns=_stamp_ns(message),
            received_monotonic_ns=time.monotonic_ns(),
        )

    def _joint_state_type(self) -> Any:
        from sensor_msgs.msg import JointState

        return JointState

    def _follow_joint_trajectory_type(self) -> Any:
        from control_msgs.action import FollowJointTrajectory

        return FollowJointTrajectory

    @property
    def _trajectory_client(self) -> _RclpyTrajectoryClient:
        return _RclpyTrajectoryClient(self._rclpy, self._node, self._action_client)

    @property
    def _joint_state_source(self) -> _RclpyJointStateSource:
        return _RclpyJointStateSource(self)


class _RclpyJointStateSource:
    def __init__(self, transport: RclpyTransport):
        self._transport = transport

    def latest_state(self) -> JointStateLike | None:
        return self._transport._latest

    def spin_once(self, timeout_s: float) -> None:
        self._transport._rclpy.spin_once(self._transport._node, timeout_sec=timeout_s)


class _RclpyTrajectoryClient:
    def __init__(self, rclpy: Any, node: Any, action_client: Any):
        self._rclpy = rclpy
        self._node = node
        self._action_client = action_client

    def server_is_ready(self) -> bool:
        return bool(self._action_client.server_is_ready())

    def send_goal(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        duration_s: float,
        tolerance_s: float,
    ) -> GoalHandleLike:
        goal = self._build_goal(joint_names, positions, duration_s, tolerance_s)
        future = self._action_client.send_goal_async(goal)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            self._spin()
        if not future.done():
            raise Ros2TransportError(
                "goal response did not arrive within 10s of send_goal_async"
            )
        return _RclpyGoalHandle(future.result())

    def wait_for_result(self, goal: GoalHandleLike, timeout_s: float) -> GoalOutcome:
        handle = goal._handle
        future = handle.get_result_async()
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            self._spin()
        if not future.done():
            return GoalOutcome(status="timeout", detail="no result before deadline")
        response = future.result()
        status_int = int(getattr(response, "status", 0))
        result = getattr(response, "result", None)
        error_code = getattr(result, "error_code", None)
        detail = str(getattr(result, "error_string", "") or "")
        return GoalOutcome(
            status=_status_name(status_int),
            error_code=int(error_code) if error_code is not None else None,
            detail=detail,
        )

    def cancel_goal(self, goal: GoalHandleLike, timeout_s: float) -> bool:
        """Request cancellation and verify the goal reached a terminal state.

        True only when the cancel request itself got a favorable answer
        (ERROR_NONE, or ERROR_GOAL_TERMINATED when the goal was already
        terminal) and the goal's result reports a terminal status
        (SUCCEEDED/CANCELED/ABORTED, goal status codes 4/5/6). Exceptions,
        timeouts, rejected cancel requests, and non-terminal statuses
        (e.g. CANCELING) return False so the adapter stays fail-closed.
        """
        handle = goal._handle
        try:
            cancel_future = handle.cancel_goal_async()
        except Exception:
            return False
        deadline = time.monotonic() + timeout_s
        while not cancel_future.done() and time.monotonic() < deadline:
            self._spin()
        if not cancel_future.done():
            return False
        try:
            cancel_response = cancel_future.result()
        except Exception:
            return False
        return_code = getattr(cancel_response, "return_code", None)
        # 0 = ERROR_NONE; 3 = ERROR_GOAL_TERMINATED (already terminal, the
        # result check below still applies). 1/2 mean the server refused or
        # never knew the goal: cancellation is not confirmed. A malformed
        # response cannot establish that safety property either.
        try:
            if int(return_code) not in (0, 3):
                return False
        except (TypeError, ValueError):
            return False
        # The cancel response only acknowledges the request; the goal must also
        # reach a terminal state (CANCELED/ABORTED/SUCCEEDED) before returning.
        try:
            result_future = handle.get_result_async()
        except Exception:
            return False
        deadline = time.monotonic() + timeout_s
        while not result_future.done() and time.monotonic() < deadline:
            self._spin()
        if not result_future.done():
            return False
        try:
            result_response = result_future.result()
        except Exception:
            return False
        try:
            status = int(getattr(result_response, "status", 0))
        except (TypeError, ValueError):
            return False
        return status in (4, 5, 6)

    def _build_goal(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        duration_s: float,
        tolerance_s: float,
    ) -> Any:
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        goal = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        seconds = int(duration_s)
        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = int((duration_s - seconds) * 1e9)
        trajectory.points = [point]
        goal.trajectory = trajectory
        goal.goal_time_tolerance.sec = int(tolerance_s)
        goal.goal_time_tolerance.nanosec = int((tolerance_s - int(tolerance_s)) * 1e9)
        return goal

    def _spin(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.01)


@dataclass
class _RclpyGoalHandle:
    """Adapts an rclpy ClientGoalHandle into ``GoalHandleLike``."""

    _handle: Any

    @property
    def accepted(self) -> bool:
        return bool(self._handle.accepted)

    @property
    def detail(self) -> str:
        return ""


_STATUS_NAMES = {
    0: "unknown",
    1: "accepted",
    2: "executing",
    3: "canceled",
    4: "succeeded",
    5: "canceled",
    6: "aborted",
}


def _status_name(status: int) -> str:
    if status == 4:
        return "succeeded"
    if status == 6:
        return "aborted"
    if status in (3, 5):
        return "preempted"
    if status == 1:
        return "executing"
    return "unknown"
