import math

import pytest

from simready.robotics.cartesian_actions import (
    CartesianEefObservation,
    CartesianEefTarget,
    CartesianSafetyLimits,
    GripperTarget,
    TimedActionSegment,
    validate_cartesian_action_segment,
)

NOW = 10_000_000_000
IDENTITY_XYZW = (1.0, 0.0, 0.0, 0.0)
MAX_OBSERVATION_AGE_S = 0.2

LIMITS = CartesianSafetyLimits(
    workspace_frame="robot_base",
    workspace_lower_m=(-0.5, -0.5, 0.0),
    workspace_upper_m=(0.5, 0.5, 0.5),
    gripper_opening_lower_fraction=0.0,
    gripper_opening_upper_fraction=0.35,
    max_translation_step_m=0.08,
    max_rotation_step_rad=0.2,
    max_horizon_s=1.0,
)
CURRENT = CartesianEefTarget(
    frame="robot_base",
    position_m=(0.0, 0.0, 0.25),
    orientation_xyzw=IDENTITY_XYZW,
)
CURRENT_OBSERVATION = CartesianEefObservation(
    target=CURRENT,
    received_monotonic_ns=NOW,
)


def make_segment(
    *,
    sequence: int = 1,
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.25),
    orientation_xyzw: tuple[float, float, float, float] = IDENTITY_XYZW,
    frame: str = "robot_base",
    opening_fraction: float = 0.02,
    issued_monotonic_ns: int = NOW,
    valid_for_s: float = 0.5,
    duration_s: float = 0.2,
) -> TimedActionSegment:
    return TimedActionSegment(
        target=CartesianEefTarget(
            frame=frame,
            position_m=position_m,
            orientation_xyzw=orientation_xyzw,
        ),
        gripper=GripperTarget(opening_fraction=opening_fraction),
        issued_monotonic_ns=issued_monotonic_ns,
        valid_for_s=valid_for_s,
        duration_s=duration_s,
        sequence=sequence,
    )


def make_observation(
    *,
    received_monotonic_ns: int = NOW,
    target: CartesianEefTarget = CURRENT,
) -> CartesianEefObservation:
    return CartesianEefObservation(
        target=target,
        received_monotonic_ns=received_monotonic_ns,
    )


def validate(
    segment: TimedActionSegment,
    *,
    limits: CartesianSafetyLimits = LIMITS,
    current_observation: CartesianEefObservation = CURRENT_OBSERVATION,
    max_observation_age_s: float = MAX_OBSERVATION_AGE_S,
    now_monotonic_ns: int = NOW,
    last_sequence: int | None = None,
) -> None:
    validate_cartesian_action_segment(
        limits,
        segment,
        current_observation=current_observation,
        max_observation_age_s=max_observation_age_s,
        now_monotonic_ns=now_monotonic_ns,
        last_sequence=last_sequence,
    )


def test_valid_segment_on_inclusive_boundary_is_accepted():
    segment = make_segment(
        position_m=(0.08, 0.0, 0.25),
        opening_fraction=LIMITS.gripper_opening_upper_fraction,
        sequence=7,
    )
    validate(segment, last_sequence=6)


def test_malformed_segment_type_is_rejected():
    with pytest.raises(ValueError, match="segment must be TimedActionSegment"):
        validate(object())  # type: ignore[arg-type]


def test_malformed_limits_type_is_rejected():
    with pytest.raises(ValueError, match="limits must be CartesianSafetyLimits"):
        validate(make_segment(), limits=object())  # type: ignore[arg-type]


def test_boolean_now_timestamp_is_rejected():
    with pytest.raises(ValueError, match="now_monotonic_ns must be an integer"):
        validate(make_segment(), now_monotonic_ns=True)  # type: ignore[arg-type]


def test_boolean_sequence_is_rejected():
    segment = make_segment(sequence=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence must be an integer"):
        validate(segment)


@pytest.mark.parametrize("issued_monotonic_ns", [True, 0.5, None])
def test_malformed_issued_timestamp_is_rejected(issued_monotonic_ns):
    segment = make_segment(issued_monotonic_ns=issued_monotonic_ns)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="issued_monotonic_ns must be an integer"):
        validate(segment)


def test_negative_sequence_is_rejected():
    with pytest.raises(ValueError, match="sequence must be nonnegative"):
        validate(make_segment(sequence=-1))


def test_duplicate_sequence_is_rejected():
    with pytest.raises(ValueError, match="must increase monotonically"):
        validate(make_segment(sequence=5), last_sequence=5)


def test_out_of_order_sequence_is_rejected():
    with pytest.raises(ValueError, match="must increase monotonically"):
        validate(make_segment(sequence=4), last_sequence=5)


def test_malformed_last_sequence_is_rejected():
    with pytest.raises(ValueError, match="last_sequence must be an integer or None"):
        validate(make_segment(), last_sequence=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    ["valid_for_s", "duration_s"],
)
def test_nonpositive_or_nonfinite_timing_is_rejected(field_name):
    for bad_value in [0.0, -0.1, math.inf, math.nan, True]:
        segment = make_segment(**{field_name: bad_value})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=f"{field_name} must be"):
            validate(segment)


def test_validity_above_max_horizon_is_rejected():
    with pytest.raises(ValueError, match="valid_for_s exceeds max_horizon_s"):
        validate(make_segment(valid_for_s=LIMITS.max_horizon_s + 0.1))


def test_duration_above_max_horizon_is_rejected():
    with pytest.raises(ValueError, match="duration_s exceeds max_horizon_s"):
        validate(make_segment(duration_s=LIMITS.max_horizon_s + 0.1))


def test_duration_above_validity_is_rejected():
    with pytest.raises(ValueError, match="duration_s must not exceed valid_for_s"):
        validate(make_segment(valid_for_s=0.3, duration_s=0.4))


def test_expired_segment_is_rejected():
    segment = make_segment(issued_monotonic_ns=NOW - 1_000_000_000, valid_for_s=0.5)
    with pytest.raises(ValueError, match="expired or from the future"):
        validate(segment)


def test_future_segment_is_rejected():
    segment = make_segment(issued_monotonic_ns=NOW + 1)
    with pytest.raises(ValueError, match="expired or from the future"):
        validate(segment)


def test_malformed_target_type_is_rejected():
    segment = make_segment()
    object.__setattr__(segment, "target", object())
    with pytest.raises(ValueError, match="segment target must be CartesianEefTarget"):
        validate(segment)


def test_nonstring_frame_is_rejected():
    segment = make_segment(frame=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frame must be a nonempty string"):
        validate(segment)


def test_wrong_frame_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        validate(make_segment(frame="world"))


def test_wrong_position_unit_is_rejected():
    segment = make_segment()
    object.__setattr__(segment.target, "position_unit", "mm")
    with pytest.raises(ValueError, match="position_unit must be 'm'"):
        validate(segment)


def test_wrong_quaternion_order_is_rejected():
    segment = make_segment()
    object.__setattr__(segment.target, "quaternion_order", "wxyz")
    with pytest.raises(ValueError, match="quaternion_order must be 'xyzw'"):
        validate(segment)


@pytest.mark.parametrize(
    ("position_m", "name"),
    [
        ((0.0, 0.0), "position_m must be a tuple of 3"),
        ((0.0, 0.0, 0.25, 0.0), "position_m must be a tuple of 3"),
        ([0.0, 0.0, 0.25], "position_m must be a tuple of 3"),
        ((0.0, 0.0, math.nan), "position_m must be a finite number"),
    ],
)
def test_malformed_target_position_is_rejected(position_m, name):
    with pytest.raises(ValueError, match=name):
        validate(make_segment(position_m=position_m))


@pytest.mark.parametrize(
    ("orientation_xyzw", "name"),
    [
        ((1.0, 0.0, 0.0), "orientation_xyzw must be a tuple of 4"),
        ([1.0, 0.0, 0.0, 0.0], "orientation_xyzw must be a tuple of 4"),
        ((1.0, 0.0, 0.0, math.nan), "must be a finite number"),
        ((0.5, 0.0, 0.0, 0.0), "must be normalized"),
    ],
)
def test_malformed_target_orientation_is_rejected(orientation_xyzw, name):
    with pytest.raises(ValueError, match=name):
        validate(make_segment(orientation_xyzw=orientation_xyzw))


def test_negated_quaternion_is_the_same_orientation():
    segment = make_segment(
        orientation_xyzw=(-1.0, 0.0, 0.0, 0.0),
        position_m=CURRENT.position_m,
    )
    validate(segment)


def test_malformed_current_observation_is_rejected():
    with pytest.raises(
        ValueError, match="current_observation must be CartesianEefObservation"
    ):
        validate(make_segment(), current_observation=object())  # type: ignore[arg-type]


def test_current_observation_wrong_frame_is_rejected():
    current = CartesianEefTarget(
        frame="world",
        position_m=CURRENT.position_m,
        orientation_xyzw=IDENTITY_XYZW,
    )
    with pytest.raises(ValueError, match="current_observation frame"):
        validate(make_segment(), current_observation=make_observation(target=current))


def test_current_observation_denormalized_quaternion_is_rejected():
    current = CartesianEefTarget(
        frame="robot_base",
        position_m=CURRENT.position_m,
        orientation_xyzw=(0.5, 0.0, 0.0, 0.0),
    )
    with pytest.raises(
        ValueError, match="current_observation orientation_xyzw must be normalized"
    ):
        validate(make_segment(), current_observation=make_observation(target=current))


@pytest.mark.parametrize("received_monotonic_ns", [True, 0.5, None])
def test_malformed_observation_timestamp_is_rejected(received_monotonic_ns):
    observation = make_observation(received_monotonic_ns=received_monotonic_ns)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="received_monotonic_ns must be an integer"):
        validate(make_segment(), current_observation=observation)


@pytest.mark.parametrize("max_observation_age_s", [0.0, -0.1, math.inf, math.nan])
def test_malformed_max_observation_age_is_rejected(max_observation_age_s):
    with pytest.raises(ValueError, match="max_observation_age_s must be"):
        validate(make_segment(), max_observation_age_s=max_observation_age_s)  # type: ignore[arg-type]


def test_stale_current_observation_is_rejected():
    stale_ns = NOW - int((MAX_OBSERVATION_AGE_S + 0.1) * 1_000_000_000)
    observation = make_observation(received_monotonic_ns=stale_ns)
    with pytest.raises(ValueError, match="observation is stale or from the future"):
        validate(make_segment(), current_observation=observation)


def test_future_current_observation_is_rejected():
    observation = make_observation(received_monotonic_ns=NOW + 1)
    with pytest.raises(ValueError, match="observation is stale or from the future"):
        validate(make_segment(), current_observation=observation)


def test_current_observation_at_max_age_boundary_is_accepted():
    boundary_ns = NOW - int(MAX_OBSERVATION_AGE_S * 1_000_000_000)
    observation = make_observation(received_monotonic_ns=boundary_ns)
    validate(make_segment(), current_observation=observation)


def test_malformed_gripper_target_is_rejected():
    segment = make_segment()
    object.__setattr__(segment, "gripper", object())
    with pytest.raises(ValueError, match="gripper must be GripperTarget"):
        validate(segment)


def test_gripper_target_has_no_unit_field_anymore():
    segment = make_segment()
    assert not hasattr(segment.gripper, "unit")
    assert 0.0 <= segment.gripper.opening_fraction <= 1.0


@pytest.mark.parametrize(
    "opening_fraction",
    [math.nan, math.inf, True, None, "0.02"],
)
def test_malformed_gripper_opening_is_rejected(opening_fraction):
    segment = make_segment(opening_fraction=opening_fraction)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="opening_fraction must be a finite number"):
        validate(segment)


@pytest.mark.parametrize(
    ("position_m", "name"),
    [
        ((0.0, 0.0, 0.51), "z position"),
        ((0.0, 0.0, -0.01), "z position"),
        ((0.51, 0.0, 0.25), "x position"),
    ],
)
def test_target_outside_workspace_is_rejected(position_m, name):
    with pytest.raises(ValueError, match=name):
        validate(make_segment(position_m=position_m))


@pytest.mark.parametrize(
    "opening_fraction",
    [
        -0.001,  # below the canonical [0, 1] domain
        1.001,  # above the canonical [0, 1] domain
        0.4,  # inside [0,1] but outside the configured [0, 0.35] task subrange
    ],
)
def test_gripper_opening_outside_bounds_is_rejected(opening_fraction):
    with pytest.raises(
        ValueError, match="opening_fraction must lie within|gripper opening outside"
    ):
        validate(make_segment(opening_fraction=opening_fraction))


def test_translation_delta_above_step_limit_is_rejected():
    with pytest.raises(ValueError, match="translation delta"):
        validate(
            make_segment(position_m=(0.081, 0.0, 0.25)),
        )


def test_rotation_delta_above_step_limit_is_rejected():
    half = math.sqrt(0.5)
    quarter_turn = (half, 0.0, 0.0, half)
    with pytest.raises(ValueError, match="rotation delta"):
        validate(make_segment(orientation_xyzw=quarter_turn))


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"workspace_upper_m": (-0.5, 0.5, 0.5)}, "workspace"),
        ({"gripper_opening_upper_fraction": -0.1}, "must lie within"),
        ({"max_translation_step_m": 0.0}, "max_translation_step_m"),
        ({"max_rotation_step_rad": math.pi + 0.1}, "must not exceed pi"),
        ({"max_horizon_s": math.inf}, "max_horizon_s"),
        ({"workspace_frame": ""}, "workspace_frame"),
    ],
)
def test_invalid_limits_configuration_is_rejected(kwargs, name):
    with pytest.raises(ValueError, match=name):
        CartesianSafetyLimits(
            **{
                "workspace_frame": "robot_base",
                "workspace_lower_m": (-0.5, -0.5, 0.0),
                "workspace_upper_m": (0.5, 0.5, 0.5),
                "gripper_opening_lower_fraction": 0.0,
                "gripper_opening_upper_fraction": 1.0,
                "max_translation_step_m": 0.08,
                "max_rotation_step_rad": 0.2,
                "max_horizon_s": 1.0,
                **kwargs,
            }
        )


def test_validator_returns_none_and_sends_nothing():
    assert validate(make_segment()) is None
