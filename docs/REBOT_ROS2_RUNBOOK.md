# reBot B601-RS ROS 2 adapter runbook (Module D, uncommissioned)

This runbook is Module D of the robot adapter plan
(`docs/ROBOT_ADAPTER_ROADMAP_AND_ACCEPTANCE.md`). It records only facts
traced from driver source and documentation. No physical reBot arm, CAN bus,
or live ROS graph was connected while producing it.

## Scope and non-goals

- Delivered in this module: two configuration-only ROS 2 artifacts — the
  official-Seeed-driver profile
  (`examples/robot_adapters/profiles/rebot-b601-rs-official-ros2.json`) and
  its adapter config
  (`examples/robot_adapters/configs/rebot-b601-rs-official-ros2.json`) —
  plus corrected limits in the existing community/Isaac profile
  (`examples/robot_adapters/profiles/rebot-b601-rs.json`) and CPU-only
  config/contract tests.
- Not delivered, not attempted, not authorized: any `ros2 launch`, action
  goal, service call, CAN traffic, or physical robot command. There is no
  `Ros2Adapter` transport implementation in this module; the generic ROS 2
  adapter runtime belongs to the separately owned Isaac ROS 2 module.
- `hardware_verified` stays `false` everywhere. Hardware command endpoints
  are deliberately unset (`null`) in the official-driver artifacts.

## Two distinct ROS 2 driver stacks — never interchangeable

Module D found two unrelated ROS 2 implementations for the B601-RS. An
adapter must pin exactly one backend; endpoint names, gripper semantics,
and stop behavior differ.

1. **Official Seeed driver** —
   [Seeed-Projects/reBotArmController_ROS2](https://github.com/Seeed-Projects/reBotArmController_ROS2)
   (pinned commit `87780c6a5c2fb43b55ddebda5eac375ceb9c5a40`, 2026-11-06),
   documented by
   [Seeed's B601-RS ROS 2 integration page](https://wiki.seeedstudio.com/rebot_arm_b601_rs_ros2_integration/).
   A custom driver over the `reBotArm_control_py` SDK; default namespace
   `rebotarm` (launch argument `arm_namespace`).
2. **Community `ros2_control` stack** —
   [xiayip/rebot_arm_rs_ros2](https://github.com/xiayip/rebot_arm_rs_ros2)
   (Jazzy). Standard `ros2_control` controllers with MIT-chained output.
   This is the stack the existing `rebot-b601-rs.json` profile referenced;
   it is not an official Seeed certification.

## Verified official-driver facts (commit `87780c6a…`)

Source files: `src/rebotarm_bringup/launch/driver.launch.py`,
`src/rebotarmcontroller/rebotarmcontroller/ros_actions.py`,
`ros_services.py`, `ros_publishers.py`, `hardware_manager.py`,
`src/rebotarm_bringup/config/rebotarm_hardware.yaml`,
`src/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf`, and SDK
config `reBotArm_control_py/config/rebotarm_rs.yaml`.

| Interface | Exact name | Type |
|---|---|---|
| Joint state (default ns) | `/rebotarm/joint_states` | `sensor_msgs/msg/JointState` |
| Arm status | `/rebotarm/arm_status` | `rebotarm_msgs/msg/ArmStatus` |
| Arm trajectory action | `/rebotarm/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` |
| Gripper action | `/rebotarm/gripper/command` | `control_msgs/action/GripperCommand` |
| Disable | `/rebotarm/disable` | `std_srvs/srv/Trigger` |
| Safe home | `/rebotarm/safe_home` | `std_srvs/srv/Trigger` |
| Enable / gravity comp start/stop, set zero, IK | `/rebotarm/...` | `std_srvs`/`rebotarm_msgs` services |

- **Arm joint order** (hardware config + FJT server): `joint1, joint2,
  joint3, joint4, joint5, joint6`. The FJT server requires goal
  `joint_names` to be an exact ordered match; reordering or truncation is
  rejected. A generic adapter must send that order or reject pre-transport.
- **Units**: arm positions rad, velocities rad/s, efforts N·m; gripper
  motor command in radians — `0.0` closed, `5.0` open on the `gripper`
  axis. `SetGripper` takes `position` in that radian motor coordinate; it
  does **not** take a metre opening width.
- **Motor map**: joint1–6 = RobStride RS-06 (J1–J3) / RS-00 (J4–J6), IDs
  0x01–0x06; gripper motor 0x07; feedback host 0xFD.
- **URDF arm limits** (rad): joint1 ±2.8, joint2 [0, 3.14], joint3
  [0, 3.14], joint4 ±1.57, joint5 ±1.57, joint6 ±3.14. The RS URDF declares
  both prismatic finger joints `gripper_joint1/2` 0–0.05 m, but the pinned
  driver does not publish the raw URDF coordinate verbatim: `HardwareManager`
  returns `gripper_joint1, gripper_joint2` for every non-DM model and
  `ros_publishers.py` appends derived state values using
  `_RS_GRIPPER_JOINT_LIMITS = (0.05, 0.0715)`. The official state profile
  therefore represents its actual eight-name `JointState` sample as
  `gripper_joint1` 0–0.05 m and `gripper_joint2` 0–0.0715 m. This is
  state-only geometry, not an executable gripper command representation.

### Stop behavior (safety-critical)

- `/rebotarm/disable` stops gravity compensation and calls the hardware
  disable path; `safe_home` returns to a safe posture and (default
  `disable_after_safe_home:=true`) disables motors after homing. A
  FollowJointTrajectory cancel is accepted, holds current position, and
  returns a canceled result.
- **No dedicated or safety-rated emergency-stop interface exists in the
  driver source or Seeed documentation.** Software cancellation, disable,
  safe_home, and Ctrl-C are not an E-stop; no source establishes stop
  latency, brake behavior, or torque-removal timing. Any deployment
  requires the physical machine's own safety documentation and a
  power-disconnect procedure.
- Launch defaults: `channel` empty → RS config default `can0`;
  `joint_state_rate` 100 Hz; `cmd_arbitration` `reject`;
  `disable_after_safe_home` true.

## Verified community-driver facts (`xiayip/rebot_arm_rs_ros2`)

Sources: repository README,
`rebot_arm_rs_bringup/config/controllers.yaml`,
`rebot_arm_rs_bringup/launch/bringup.launch.py`,
`rebot_arm_rs_description/urdf/rebot_arm_rs/rebot_arm_rs_macro.urdf.xacro`,
`rebot_arm_rs.ros2_control.xacro`,
`rebot_arm_rs_official_parallel_gripper_macro.urdf.xacro`.

- State topic `/joint_states` (from `joint_state_broadcaster`); arm action
  `/rebot_arm_rs_trajectory_controller/follow_joint_trajectory`
  (`control_msgs/action/FollowJointTrajectory`); gripper action
  `/rebot_arm_rs_gripper_controller/gripper_cmd`
  (`control_msgs/action/ParallelGripperCommand`, normalized 0.0 open /
  1.0 closed). Arm joint order `joint1..joint6`.
- Arm URDF limits (rad, used to correct the existing profile):
  joint2 [0, 3.63], joint3 [0, 4.23], joint4 [-1.88, 1.60] — the checked-in
  profile previously said 3.14/3.14/±1.57. Gripper mimic fingers:
  `gripper_joint1` 0–0.05 m (multiplier −0.05, offset 0.05),
  `gripper_joint2` 0–0.0715 m (multiplier −0.0715, offset 0.0715) — the
  checked-in profile previously claimed 0–0.05 m for both.
- ros2_control interfaces: commands `position, velocity, kp, kd, t_ff`
  (states `position, velocity, effort`); the gripper command joint is the
  normalized revolute `gripper_command_joint` 0–1, motor ID 0x07, and the
  finger joints are URDF mimics, not ros2_control command joints.
- **Stop behavior (community stack)**: Ctrl-C intentionally does **not**
  send motor-disable frames — motors keep executing the last MIT command
  and hold torque; torque is removed only by power loss, motor fault, or
  the configured CAN timeout. `stop()` on any adapter for this stack is
  therefore not a power-off, and process exit is not a safe state.

## Unverified / deliberately unset

- Any live ROS graph on real hardware: **NOT_RUN** — no `ros2 topic list`,
  action discovery, or state observation was performed.
- Executable hardware command endpoints in
  `rebot-b601-rs-official-ros2.json` and its adapter config: left `null`
  even though the source documents them, because documentation is not
  commissioning.
- A safety-rated E-stop behavior for either stack: no source found; unset.
- Torque unit on the community stack's impedance state: not stated in its
  README; unset.
- `SetMode` / `JointVelCmd` listed in the official repo's Chinese API doc
  but absent from the pinned revision's `rebotarm_msgs` tree: treated as
  version-skew, not configured.
- Community-stack per-joint zero offsets and the exact physical B601-RS
  revision installed at the bench: unverified; must be measured during
  commissioning.

## Fail-closed rules for the profile/config

- The official-driver profile keeps `transport.hardware.command = null`
  and `verified = false`; `validate_joint_command(..., hardware=True)`
  therefore rejects any hardware command for it by contract.
- The adapter config mirrors that: `arm_command.action = null`,
  `gripper.action = null`, `stop.command = null`; the source-documented
  names are recorded as evidence-only fields, never as endpoints to dial.
- Any future adapter reading this config must fail closed if state is
  stale, joint names differ from the profile, a command expires, or the
  commissioning gate (`hardware.verified`) is not open.

## Commissioning checklist (all NOT_RUN; outside Module D)

1. Operator-held physical E-stop and power-disconnect procedure from the
   machine's own safety documentation.
2. Confirm the physical B601-RS revision, installed driver (official vs
   community), CAN interface, motor IDs, zero offsets, and joint signs.
3. Live `ros2 topic`/`ros2 action` introspection recording actual names,
   types, and observed `/joint_states` contents against this profile.
4. Low-speed, no-load, workspace-limited first-motion plan with
   independent review.
5. Only then: set an executable command transport, flip
   `hardware_verified`, and record per-action evidence per the roadmap
   evidence section.

## Status

| Check | Status |
|---|---|
| Profile/config schema and contract tests (CPU, mocked) | see test run in module report |
| ROS 2 graph discovery on live driver | NOT_RUN |
| ROS 2 state observation on live driver | NOT_RUN |
| FollowJointTrajectory action discovery | NOT_RUN |
| Gripper action discovery | NOT_RUN |
| Physical hardware command | NOT_RUN |
| Hardware commissioning / `hardware_verified` | NOT_RUN (false) |
