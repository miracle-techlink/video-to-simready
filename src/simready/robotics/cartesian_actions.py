"""Pure validation contracts for bounded Cartesian action segments.

This module validates action proposals only. It does not provide inverse
kinematics, collision checking, trajectory generation, or robot execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_QUATERNION_NORM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CartesianEefTarget:
    """An EEF pose expressed in metres and a normalized ``wxyz`` quaternion."""

    frame: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    position_unit: str = "m"
    quaternion_order: str = "wxyz"


@dataclass(frozen=True)
class GripperTarget:
    """A parallel-jaw gripper aperture expressed in metres."""

    opening_m: float
    unit: str = "m"


@dataclass(frozen=True)
class TimedActionSegment:
    """A short-lived Cartesian target proposal with explicit replay metadata."""

    target: CartesianEefTarget
    gripper: GripperTarget
    issued_monotonic_ns: int
    valid_for_s: float
    duration_s: float
    sequence: int


@dataclass(frozen=True)
class CartesianSafetyLimits:
    """Static Cartesian workspace, delta, gripper, and horizon limits."""

    workspace_frame: str
    workspace_lower_m: tuple[float, float, float]
    workspace_upper_m: tuple[float, float, float]
    gripper_opening_lower_m: float
    gripper_opening_upper_m: float
    max_translation_step_m: float
    max_rotation_step_rad: float
    max_horizon_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_frame, str) or not self.workspace_frame:
            raise ValueError("workspace_frame must be a nonempty string")
        lower = _vector(self.workspace_lower_m, "workspace_lower_m", 3)
        upper = _vector(self.workspace_upper_m, "workspace_upper_m", 3)
        if any(
            lower_value >= upper_value
            for lower_value, upper_value in zip(lower, upper, strict=True)
        ):
            raise ValueError("workspace lower bounds must be below upper bounds")
        lower_gripper = _finite_number(
            self.gripper_opening_lower_m, "gripper_opening_lower_m"
        )
        upper_gripper = _finite_number(
            self.gripper_opening_upper_m, "gripper_opening_upper_m"
        )
        if lower_gripper >= upper_gripper:
            raise ValueError("gripper opening lower bound must be below upper bound")
        _positive_finite(self.max_translation_step_m, "max_translation_step_m")
        rotation = _positive_finite(self.max_rotation_step_rad, "max_rotation_step_rad")
        if rotation > math.pi:
            raise ValueError("max_rotation_step_rad must not exceed pi")
        _positive_finite(self.max_horizon_s, "max_horizon_s")


def validate_cartesian_action_segment(
    limits: CartesianSafetyLimits,
    segment: TimedActionSegment,
    *,
    current_eef: CartesianEefTarget,
    now_monotonic_ns: int,
    last_sequence: int | None,
) -> None:
    """Reject malformed, stale, replayed, or out-of-bounds action segments."""
    if not isinstance(limits, CartesianSafetyLimits):
        raise ValueError("limits must be CartesianSafetyLimits")
    if not isinstance(segment, TimedActionSegment):
        raise ValueError("segment must be TimedActionSegment")
    if not isinstance(now_monotonic_ns, int) or isinstance(now_monotonic_ns, bool):
        raise ValueError("now_monotonic_ns must be an integer")
    if not isinstance(segment.issued_monotonic_ns, int) or isinstance(
        segment.issued_monotonic_ns, bool
    ):
        raise ValueError("segment issued_monotonic_ns must be an integer")
    if not isinstance(segment.sequence, int) or isinstance(segment.sequence, bool):
        raise ValueError("segment sequence must be an integer")
    if segment.sequence < 0:
        raise ValueError("segment sequence must be nonnegative")
    if last_sequence is not None:
        if not isinstance(last_sequence, int) or isinstance(last_sequence, bool):
            raise ValueError("last_sequence must be an integer or None")
        if segment.sequence <= last_sequence:
            raise ValueError("segment sequence must increase monotonically")

    valid_for_s = _positive_finite(segment.valid_for_s, "segment valid_for_s")
    duration_s = _positive_finite(segment.duration_s, "segment duration_s")
    if valid_for_s > limits.max_horizon_s:
        raise ValueError("segment valid_for_s exceeds max_horizon_s")
    if duration_s > limits.max_horizon_s:
        raise ValueError("segment duration_s exceeds max_horizon_s")
    if duration_s > valid_for_s:
        raise ValueError("segment duration_s must not exceed valid_for_s")
    age_ns = now_monotonic_ns - segment.issued_monotonic_ns
    if age_ns < 0 or age_ns > valid_for_s * 1_000_000_000:
        raise ValueError(
            f"Cartesian action segment expired or from the future: age_ns={age_ns}"
        )

    target_position, target_orientation = _validate_eef_target(
        segment.target, limits.workspace_frame, "segment target"
    )
    current_position, current_orientation = _validate_eef_target(
        current_eef, limits.workspace_frame, "current_eef"
    )
    gripper_opening = _validate_gripper_target(segment.gripper)

    for axis, value, lower, upper in zip(
        "xyz",
        target_position,
        limits.workspace_lower_m,
        limits.workspace_upper_m,
        strict=True,
    ):
        if value < lower or value > upper:
            raise ValueError(
                f"segment target {axis} position {value} outside workspace [{lower}, {upper}] m"
            )
    if (
        gripper_opening < limits.gripper_opening_lower_m
        or gripper_opening > limits.gripper_opening_upper_m
    ):
        raise ValueError(
            "segment gripper opening outside "
            f"[{limits.gripper_opening_lower_m}, {limits.gripper_opening_upper_m}] m"
        )

    translation_delta_m = math.dist(current_position, target_position)
    if translation_delta_m > limits.max_translation_step_m:
        raise ValueError(
            "segment translation delta "
            f"{translation_delta_m} m exceeds {limits.max_translation_step_m} m"
        )
    rotation_delta_rad = _quaternion_distance(current_orientation, target_orientation)
    if rotation_delta_rad > limits.max_rotation_step_rad:
        raise ValueError(
            "segment rotation delta "
            f"{rotation_delta_rad} rad exceeds {limits.max_rotation_step_rad} rad"
        )


def _validate_eef_target(
    target: object,
    expected_frame: str,
    label: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if not isinstance(target, CartesianEefTarget):
        raise ValueError(f"{label} must be CartesianEefTarget")
    if not isinstance(target.frame, str) or not target.frame:
        raise ValueError(f"{label} frame must be a nonempty string")
    if target.frame != expected_frame:
        raise ValueError(
            f"{label} frame {target.frame!r} does not match {expected_frame!r}"
        )
    if target.position_unit != "m":
        raise ValueError(f"{label} position_unit must be 'm'")
    if target.quaternion_order != "wxyz":
        raise ValueError(f"{label} quaternion_order must be 'wxyz'")
    position = _vector(target.position_m, f"{label} position_m", 3)
    orientation = _vector(target.orientation_wxyz, f"{label} orientation_wxyz", 4)
    norm = math.sqrt(sum(value * value for value in orientation))
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        raise ValueError(f"{label} orientation_wxyz must be normalized")
    return position, orientation


def _validate_gripper_target(target: object) -> float:
    if not isinstance(target, GripperTarget):
        raise ValueError("segment gripper must be GripperTarget")
    if target.unit != "m":
        raise ValueError("segment gripper unit must be 'm'")
    return _finite_number(target.opening_m, "segment gripper opening_m")


def _quaternion_distance(
    current: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    dot_product = abs(sum(a * b for a, b in zip(current, target, strict=True)))
    return 2.0 * math.acos(min(1.0, max(0.0, dot_product)))


def _vector(value: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise ValueError(f"{name} must be a tuple of {length} finite values")
    return tuple(_finite_number(item, name) for item in value)


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_finite(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number
