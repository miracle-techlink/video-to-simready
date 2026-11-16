"""Module D config tests: reBot B601-RS ROS 2 profile and adapter config.

These are CPU-only configuration/contract checks. They load JSON artifacts
and exercise the shared validation contracts; they never import ROS, launch
a ROS graph, or touch physical hardware. Passing them is not evidence of a
hardware or ROS runtime PASS — see docs/REBOT_ROS2_RUNBOOK.md.
"""

import json
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
CONFIGS = ROOT / "examples/robot_adapters/configs"
OFFICIAL_PROFILE = PROFILES / "rebot-b601-rs-official-ros2.json"
OFFICIAL_CONFIG = CONFIGS / "rebot-b601-rs-official-ros2.json"
COMMUNITY_PROFILE = PROFILES / "rebot-b601-rs.json"

# Joint order and limits traced from the official Seeed driver sources at
# commit 87780c6a5c2fb43b55ddebda5eac375ceb9c5a40 (ReBot_Arm_RS.urdf,
# rebotarm_hardware.yaml). Evidence recorded in docs/REBOT_ROS2_RUNBOOK.md.
OFFICIAL_ARM_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
OFFICIAL_ARM_LIMITS = {
    "joint1": (-2.8, 2.8),
    "joint2": (0.0, 3.14),
    "joint3": (0.0, 3.14),
    "joint4": (-1.57, 1.57),
    "joint5": (-1.57, 1.57),
    "joint6": (-3.14, 3.14),
}
# At the pinned official-driver commit, hardware_manager.py returns
# ["gripper_joint1", "gripper_joint2"] for the RS model and
# ros_publishers.py appends both to JointState.name with metre-valued
# positions ratio * (0.05, 0.0715). The profile must accept that full
# 8-joint sample, not reject its own driver's state.
OFFICIAL_GRIPPER_JOINTS = ["gripper_joint1", "gripper_joint2"]
OFFICIAL_GRIPPER_LIMITS = {
    "gripper_joint1": (0.0, 0.05),
    "gripper_joint2": (0.0, 0.0715),
}
OFFICIAL_STATE_ORDER = OFFICIAL_ARM_ORDER + OFFICIAL_GRIPPER_JOINTS
# Community ros2_control driver URDF macro limits that differ from the
# checked-in values before revision 2 of rebot-b601-rs.json.
COMMUNITY_CORRECTED_LIMITS = {
    "joint2": (0.0, 3.63),
    "joint3": (0.0, 4.23),
    "joint4": (-1.88, 1.60),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", [OFFICIAL_PROFILE, OFFICIAL_CONFIG])
def test_official_artifacts_exist_and_are_valid_json(path):
    document = load_json(path)
    assert document["schema_version"] == 1


def test_official_profile_loads_and_matches_driver_joint_order_and_limits():
    profile = load_robot_profile(OFFICIAL_PROFILE)
    assert profile.joint_map().keys() == set(OFFICIAL_STATE_ORDER)
    assert profile.groups["arm"] == tuple(OFFICIAL_ARM_ORDER)
    assert profile.groups["gripper"] == tuple(OFFICIAL_GRIPPER_JOINTS)
    for name, (lower, upper) in OFFICIAL_ARM_LIMITS.items():
        joint = profile.joint_map()[name]
        assert joint.unit == "rad"
        assert joint.lower == pytest.approx(lower)
        assert joint.upper == pytest.approx(upper)
    for name, (lower, upper) in OFFICIAL_GRIPPER_LIMITS.items():
        joint = profile.joint_map()[name]
        assert joint.unit == "m"
        assert joint.lower == pytest.approx(lower)
        assert joint.upper == pytest.approx(upper)


def test_official_profile_uses_documented_default_joint_state_topic():
    profile = load_robot_profile(OFFICIAL_PROFILE)
    assert (
        profile.state_transport
        == "ros2:sensor_msgs/msg/JointState:/rebotarm/joint_states"
    )


def test_official_profile_is_fail_closed():
    profile = load_robot_profile(OFFICIAL_PROFILE)
    assert profile.hardware_verified is False
    assert profile.hardware_command_transport is None
    assert profile.simulation_command_transport is None


def test_official_profile_contract_accepts_arm_state_and_rejects_hardware():
    profile = load_robot_profile(OFFICIAL_PROFILE)
    now = 10_000_000_000
    state = RobotState(
        positions={joint: 0.0 for joint in OFFICIAL_STATE_ORDER},
        received_monotonic_ns=now,
        source="config-test",
    )
    validate_robot_state(profile, state, now_monotonic_ns=now, max_age_s=0.1)
    command = JointCommand(
        group="arm",
        positions={joint: 0.0 for joint in OFFICIAL_ARM_ORDER},
        issued_monotonic_ns=now,
        valid_for_s=0.1,
        sequence=1,
    )
    validate_joint_command(profile, command, now_monotonic_ns=now)
    with pytest.raises(ValueError, match="hardware commands are disabled"):
        validate_joint_command(profile, command, now_monotonic_ns=now, hardware=True)


def test_official_profile_accepts_full_eight_joint_driver_state_sample():
    """The real /rebotarm/joint_states message carries six arm joints plus
    the two appended RS gripper finger joints (hardware_manager.py +
    ros_publishers.py at the pinned commit). A six-joint-only profile would
    reject its own driver's sample as unknown gripper joints; this profile
    must validate the full sample."""
    profile = load_robot_profile(OFFICIAL_PROFILE)
    now = 10_000_000_000
    # Fingers at rest inside their stroke: gripper_joint1 in [0, 0.05],
    # gripper_joint2 in [0, 0.0715] (RS limits constant in ros_publishers.py).
    positions = {joint: 0.0 for joint in OFFICIAL_ARM_ORDER}
    positions["gripper_joint1"] = 0.025
    positions["gripper_joint2"] = 0.035
    state = RobotState(
        positions=positions,
        received_monotonic_ns=now,
        source="official-joint-state-sample",
    )
    validate_robot_state(profile, state, now_monotonic_ns=now, max_age_s=0.1)


def test_official_profile_rejects_gripper_finger_out_of_stroke():
    profile = load_robot_profile(OFFICIAL_PROFILE)
    now = 10_000_000_000
    positions = {joint: 0.0 for joint in OFFICIAL_STATE_ORDER}
    positions["gripper_joint2"] = 0.08
    state = RobotState(
        positions=positions,
        received_monotonic_ns=now,
        source="config-test",
    )
    with pytest.raises(ValueError, match="gripper_joint2' value 0.08 outside"):
        validate_robot_state(profile, state, now_monotonic_ns=now, max_age_s=0.1)


def test_official_profile_full_state_velocities_and_efforts_validate():
    profile = load_robot_profile(OFFICIAL_PROFILE)
    now = 10_000_000_000
    positions = {joint: 0.0 for joint in OFFICIAL_STATE_ORDER}
    velocities = {joint: 0.0 for joint in OFFICIAL_STATE_ORDER}
    # The publisher duplicates the gripper motor torque into both finger
    # effort slots; the contract only requires finite values with matching
    # names, so a duplicated finite torque passes.
    efforts = {joint: 0.0 for joint in OFFICIAL_STATE_ORDER}
    state = RobotState(
        positions=positions,
        received_monotonic_ns=now,
        velocities=velocities,
        efforts=efforts,
        source="official-joint-state-sample",
    )
    validate_robot_state(profile, state, now_monotonic_ns=now, max_age_s=0.1)


def test_official_profile_rejects_out_of_limit_command():
    profile = load_robot_profile(OFFICIAL_PROFILE)
    now = 10_000_000_000
    positions = {joint: 0.0 for joint in OFFICIAL_ARM_ORDER}
    positions["joint2"] = -0.1  # joint2 lower bound is 0.0
    command = JointCommand(
        group="arm",
        positions=positions,
        issued_monotonic_ns=now,
        valid_for_s=0.1,
        sequence=2,
    )
    with pytest.raises(ValueError, match="joint 'joint2' value -0.1 outside"):
        validate_joint_command(profile, command, now_monotonic_ns=now)


def test_official_config_matches_profile_and_declares_no_executable_endpoints():
    config = load_json(OFFICIAL_CONFIG)
    profile = load_robot_profile(OFFICIAL_PROFILE)
    assert config["kind"] == "ros2_adapter_config"
    assert config["robot_id"] == profile.robot_id
    assert config["state"]["topic"] == "/rebotarm/joint_states"
    assert config["state"]["type"] == "sensor_msgs/msg/JointState"
    assert config["state"]["joint_order"] == OFFICIAL_STATE_ORDER
    assert config["state"]["arm_joint_order"] == OFFICIAL_ARM_ORDER
    assert config["state"]["position_units"]["arm"] == "rad"
    assert config["state"]["position_units"]["gripper_joint1"] == "m"
    assert config["state"]["position_units"]["gripper_joint2"] == "m"
    assert config["state"]["gripper_finger_limits_m"]["gripper_joint1"] == [0.0, 0.05]
    assert config["state"]["gripper_finger_limits_m"]["gripper_joint2"] == [0.0, 0.0715]
    assert (
        config["gripper"]["source_documented_state_finger_joints"]
        == OFFICIAL_GRIPPER_JOINTS
    )
    # Fail-closed: no executable command surface, only evidence fields.
    assert config["arm_command"]["action"] is None
    assert config["arm_command"]["type"] is None
    assert config["arm_command"]["source_documented_joint_order_required"] is True
    assert config["gripper"]["action"] is None
    assert config["gripper"]["type"] is None
    # The source-documented gripper is a radian motor axis, not metres.
    assert config["gripper"]["source_documented_position_unit"] == "rad"
    assert config["gripper"]["source_documented_closed_position"] == 0.0
    assert config["gripper"]["source_documented_open_position"] == 5.0
    assert config["stop"]["command"] is None
    assert config["stop"]["safety_rated_emergency_stop"] is None
    hardware = config["hardware"]
    assert hardware["verified"] is False
    assert hardware["ros_graph_observed"] is False
    assert hardware["physical_robot_observed"] is False
    assert hardware["hardware_commands_sent"] is False
    # Evidence fields must be strings, never dials an adapter could dial.
    assert isinstance(config["arm_command"]["source_documented_action"], str)
    assert isinstance(config["stop"]["source_documented_disable_service"], str)


def test_official_config_driver_source_is_pinned():
    config = load_json(OFFICIAL_CONFIG)
    source = config["driver_source"]
    assert source["repository"].startswith("https://github.com/Seeed-Projects/")
    assert len(source["commit"]) == 40  # pinned immutable SHA, not a branch
    assert source["launch"].endswith("driver.launch.py")


def test_official_config_joint_order_matches_profile_group():
    config = load_json(OFFICIAL_CONFIG)
    profile = load_robot_profile(OFFICIAL_PROFILE)
    assert config["state"]["joint_order"] == (
        list(profile.groups["arm"]) + list(profile.groups["gripper"])
    )


def test_community_profile_limits_now_match_driver_source():
    profile = load_robot_profile(COMMUNITY_PROFILE)
    joints = profile.joint_map()
    for name, (lower, upper) in COMMUNITY_CORRECTED_LIMITS.items():
        assert joints[name].lower == pytest.approx(lower)
        assert joints[name].upper == pytest.approx(upper)
    assert joints["gripper_joint2"].upper == pytest.approx(0.0715)


def test_community_profile_gripper_joints_remain_prismatic_metres():
    profile = load_robot_profile(COMMUNITY_PROFILE)
    for name in profile.groups["gripper"]:
        assert profile.joint_map()[name].unit == "m"


def test_community_profile_still_rejects_hardware_commands():
    profile = load_robot_profile(COMMUNITY_PROFILE)
    now = 10_000_000_000
    command = JointCommand(
        group="arm",
        positions={joint: 0.0 for joint in profile.groups["arm"]},
        issued_monotonic_ns=now,
        valid_for_s=0.1,
        sequence=1,
    )
    with pytest.raises(ValueError, match="hardware commands are disabled"):
        validate_joint_command(profile, command, now_monotonic_ns=now, hardware=True)


def test_runbook_records_not_run_status_for_hardware():
    runbook = (ROOT / "docs/REBOT_ROS2_RUNBOOK.md").read_text(encoding="utf-8")
    assert "NOT_RUN" in runbook
    for required in (
        "hardware_verified",
        "motor-disable",
        "/rebotarm/joint_states",
        "/rebotarm/follow_joint_trajectory",
        "gripper_command_joint",
    ):
        assert required in runbook
