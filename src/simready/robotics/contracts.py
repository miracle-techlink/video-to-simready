"""Small, dependency-free boundary contracts for robot I/O.

All revolute-joint values use radians and all prismatic-joint values use metres.
Transport-specific code (ROS 2, vendor SDKs, Isaac Sim) belongs behind
``RobotAdapter`` and must normalize data before it crosses this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class JointSpec:
    name: str
    group: str
    unit: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.group:
            raise ValueError("joint name and group must be nonempty")
        if self.unit not in {"rad", "m"}:
            raise ValueError(f"unsupported canonical joint unit: {self.unit!r}")
        if (self.lower is None) != (self.upper is None):
            raise ValueError(f"joint {self.name!r} must define both limits or neither")
        if self.lower is not None and self.upper is not None:
            if not math.isfinite(self.lower) or not math.isfinite(self.upper):
                raise ValueError(f"joint {self.name!r} limits must be finite")
            if self.lower >= self.upper:
                raise ValueError(f"joint {self.name!r} lower limit must be below upper")


@dataclass(frozen=True)
class RobotProfile:
    robot_id: str
    display_name: str
    joints: tuple[JointSpec, ...]
    groups: Mapping[str, tuple[str, ...]]
    state_transport: str
    simulation_command_transport: str
    hardware_command_transport: str | None = None
    hardware_verified: bool = False
    commandable_groups: tuple[str, ...] | None = None
    asset_reference: str | None = None
    source_reference: str | None = None

    def __post_init__(self) -> None:
        if not self.robot_id or not self.display_name:
            raise ValueError("robot_id and display_name must be nonempty")
        if not self.joints:
            raise ValueError("robot profile must declare at least one joint")
        names = [joint.name for joint in self.joints]
        if len(names) != len(set(names)):
            raise ValueError("robot profile has duplicate joint names")
        declared = set(names)
        grouped: set[str] = set()
        for group, members in self.groups.items():
            if not group or not members:
                raise ValueError(
                    "joint groups must have a name and at least one member"
                )
            unknown = set(members) - declared
            if unknown:
                raise ValueError(
                    f"group {group!r} names unknown joints: {sorted(unknown)}"
                )
            if grouped.intersection(members):
                raise ValueError("a joint may belong to only one command group")
            mismatched = [
                name for name in members if self.joint_map()[name].group != group
            ]
            if mismatched:
                raise ValueError(
                    f"group {group!r} disagrees with joint declarations: {sorted(mismatched)}"
                )
            grouped.update(members)
        if grouped != declared:
            raise ValueError(
                f"every joint must belong to one group; missing={sorted(declared - grouped)}"
            )
        if self.commandable_groups is not None:
            unknown_commandable = set(self.commandable_groups) - set(self.groups)
            if unknown_commandable:
                raise ValueError(
                    "commandable_groups names unknown groups: "
                    f"{sorted(unknown_commandable)}"
                )
            if len(self.commandable_groups) != len(set(self.commandable_groups)):
                raise ValueError("commandable_groups must not contain duplicates")
        if self.hardware_verified and not self.hardware_command_transport:
            raise ValueError(
                "verified hardware profile must declare a command transport"
            )

    def group_is_commandable(self, group: str) -> bool:
        """Whether this profile permits commands for ``group`` at any backend."""
        return self.commandable_groups is None or group in self.commandable_groups

    def joint_map(self) -> dict[str, JointSpec]:
        return {joint.name: joint for joint in self.joints}


@dataclass(frozen=True)
class RobotState:
    """Normalized feedback. ``received_monotonic_ns`` is local receipt time."""

    positions: Mapping[str, float]
    received_monotonic_ns: int
    source_timestamp_ns: int | None = None
    velocities: Mapping[str, float] = field(default_factory=dict)
    efforts: Mapping[str, float] = field(default_factory=dict)
    source: str = "unknown"


@dataclass(frozen=True)
class JointLimitDiagnostic:
    """A measured limit overshoot accepted within an explicit state tolerance."""

    joint_name: str
    measured_value: float
    nearest_limit: float
    overshoot: float
    direction: str
    unit: str


@dataclass(frozen=True)
class JointCommand:
    """A bounded position target for exactly one declared joint group."""

    group: str
    positions: Mapping[str, float]
    issued_monotonic_ns: int
    valid_for_s: float
    sequence: int


@dataclass(frozen=True)
class CommandReceipt:
    sequence: int
    accepted: bool
    accepted_monotonic_ns: int
    detail: str = ""


@runtime_checkable
class RobotAdapter(Protocol):
    """Transport adapter implemented separately for sim and each hardware model."""

    @property
    def profile(self) -> RobotProfile: ...

    def connect(self) -> None: ...

    def get_state(self, *, timeout_s: float) -> RobotState: ...

    def send_joint_position(self, command: JointCommand) -> CommandReceipt: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def validate_robot_state(
    profile: RobotProfile,
    state: RobotState,
    *,
    now_monotonic_ns: int,
    max_age_s: float,
    measured_limit_tolerance: float = 0.0,
) -> tuple[JointLimitDiagnostic, ...]:
    """Validate measured feedback and report tolerated limit overshoots.

    Measured joint positions are never clamped: values just outside a
    declared limit but within ``measured_limit_tolerance`` are accepted and
    reported as :class:`JointLimitDiagnostic` entries, while larger
    overshoots are rejected. Requested commands stay strictly bounded; see
    :func:`validate_joint_command`.
    """
    if max_age_s <= 0:
        raise ValueError("max_age_s must be positive")
    if (
        not isinstance(measured_limit_tolerance, (int, float))
        or isinstance(measured_limit_tolerance, bool)
        or not math.isfinite(measured_limit_tolerance)
        or measured_limit_tolerance < 0
    ):
        raise ValueError("measured_limit_tolerance must be a nonnegative finite number")
    age_ns = now_monotonic_ns - state.received_monotonic_ns
    if age_ns < 0 or age_ns > max_age_s * 1_000_000_000:
        raise ValueError(f"robot state is stale or from the future: age_ns={age_ns}")
    expected = set(profile.joint_map())
    received = set(state.positions)
    if received != expected:
        raise ValueError(
            f"joint state names differ from profile: missing={sorted(expected - received)}, "
            f"unknown={sorted(received - expected)}"
        )
    diagnostics = _validate_measured_values(
        profile, state.positions, measured_limit_tolerance
    )
    for field_name, values in (
        ("velocities", state.velocities),
        ("efforts", state.efforts),
    ):
        if values:
            if set(values) != expected:
                raise ValueError(f"{field_name} joint names differ from profile")
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{field_name} contains a non-finite value")
    return diagnostics


def _validate_measured_values(
    profile: RobotProfile,
    values: Mapping[str, float],
    tolerance: float,
) -> tuple[JointLimitDiagnostic, ...]:
    joint_map = profile.joint_map()
    diagnostics: list[JointLimitDiagnostic] = []
    for name, value in values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"joint {name!r} value must be finite")
        if not math.isfinite(value):
            raise ValueError(f"joint {name!r} value must be finite")
        joint = joint_map[name]
        if joint.lower is None or joint.upper is None:
            continue
        if joint.lower <= value <= joint.upper:
            continue
        if value < joint.lower:
            nearest_limit = joint.lower
            direction = "below_lower"
        else:
            nearest_limit = joint.upper
            direction = "above_upper"
        overshoot = abs(value - nearest_limit)
        if overshoot > tolerance:
            raise ValueError(
                f"joint {name!r} measured value {value} outside [{joint.lower}, {joint.upper}] "
                f"{joint.unit} by more than tolerance {tolerance}"
            )
        diagnostics.append(
            JointLimitDiagnostic(
                joint_name=name,
                measured_value=value,
                nearest_limit=nearest_limit,
                overshoot=overshoot,
                direction=direction,
                unit=joint.unit,
            )
        )
    return tuple(diagnostics)


def validate_joint_command(
    profile: RobotProfile,
    command: JointCommand,
    *,
    now_monotonic_ns: int,
    hardware: bool = False,
) -> None:
    """Reject expired, malformed, out-of-limit, or uncommissioned commands."""
    if hardware and not profile.hardware_verified:
        raise ValueError(f"hardware commands are disabled for {profile.robot_id}")
    if not profile.group_is_commandable(command.group):
        raise ValueError(
            f"joint group {command.group!r} is not commandable for {profile.robot_id}"
        )
    if not math.isfinite(command.valid_for_s) or command.valid_for_s <= 0:
        raise ValueError("command valid_for_s must be a positive finite duration")
    if command.sequence < 0:
        raise ValueError("command sequence must be nonnegative")
    age_ns = now_monotonic_ns - command.issued_monotonic_ns
    if age_ns < 0 or age_ns > command.valid_for_s * 1_000_000_000:
        raise ValueError(f"joint command expired or from the future: age_ns={age_ns}")
    if command.group not in profile.groups:
        raise ValueError(f"unknown joint group: {command.group!r}")
    expected = set(profile.groups[command.group])
    received = set(command.positions)
    if received != expected:
        raise ValueError(
            f"command joints differ from group {command.group!r}: "
            f"missing={sorted(expected - received)}, unknown={sorted(received - expected)}"
        )
    _validate_values(profile, command.positions)


def _validate_values(profile: RobotProfile, values: Mapping[str, float]) -> None:
    joint_map = profile.joint_map()
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"joint {name!r} value must be finite")
        joint = joint_map[name]
        if joint.lower is not None and joint.upper is not None:
            if value < joint.lower or value > joint.upper:
                raise ValueError(
                    f"joint {name!r} value {value} outside [{joint.lower}, {joint.upper}] {joint.unit}"
                )
