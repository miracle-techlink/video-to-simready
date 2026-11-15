from pathlib import Path

import pytest

from simready.robotics import (
    JointCommand,
    RobotState,
    load_robot_profile,
    validate_joint_command,
    validate_robot_state,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "examples/robot_adapters/profiles"


@pytest.mark.parametrize(
    "filename",
    ["rebot-b601-rs.json", "agibot-genie-g2-crsb-sim.json"],
)
def test_pilot_profiles_load_and_normalized_contract_accepts_state(filename):
    profile = load_robot_profile(PROFILES / filename)
    now = 10_000_000_000
    state = RobotState(
        positions={joint.name: 0.0 for joint in profile.joints},
        received_monotonic_ns=now,
        source="profile-canary",
    )
    validate_robot_state(profile, state, now_monotonic_ns=now, max_age_s=0.1)


def test_rebot_arm_command_is_group_scoped_bounded_and_hardware_fail_closed():
    profile = load_robot_profile(PROFILES / "rebot-b601-rs.json")
    now = 10_000_000_000
    command = JointCommand(
        group="arm",
        positions={joint: 0.0 for joint in profile.groups["arm"]},
        issued_monotonic_ns=now,
        valid_for_s=0.1,
        sequence=1,
    )
    validate_joint_command(profile, command, now_monotonic_ns=now)
    with pytest.raises(ValueError, match="hardware commands are disabled"):
        validate_joint_command(profile, command, now_monotonic_ns=now, hardware=True)


def test_genie_sim_profile_does_not_claim_hardware_command():
    profile = load_robot_profile(PROFILES / "agibot-genie-g2-crsb-sim.json")
    assert profile.hardware_command_transport is None
    assert profile.hardware_verified is False


def test_profile_rejects_joint_group_disagreement():
    from simready.robotics import JointSpec, RobotProfile

    with pytest.raises(ValueError, match="disagrees"):
        RobotProfile(
            robot_id="bad",
            display_name="bad",
            joints=(JointSpec("j1", "arm", "rad"),),
            groups={"gripper": ("j1",)},
            state_transport="test",
            simulation_command_transport="test",
        )
