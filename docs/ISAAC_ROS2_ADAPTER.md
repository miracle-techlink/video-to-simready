# Isaac Sim 6.2 ROS 2 control adapter

## Scope and status

`simready.robotics.ros2_control_adapter.IsaacRos2ControlAdapter` is the Module C
simulation-only transport for the existing dependency-free `RobotAdapter`
contract. It sends each validated `JointCommand` as a monitored
`control_msgs/action/FollowJointTrajectory` goal and normalizes
`sensor_msgs/msg/JointState` feedback by **joint name**, not incoming message
order.

The module imports without `rclpy`; its default `RclpyTransport` imports ROS 2
only in `connect()`. Unit tests inject a fake transport, so ordinary CPU CI does
not require ROS, Isaac Sim, or a GPU.

> **Acceptance status at 2026-11-16:** CPU contract tests PASS. Live Isaac ROS
> 2 acceptance is **BLOCKED / NOT_RUN**, not PASS. The configured 6.2 L40
> server's shared lock cannot be checked or acquired: the documented
> ControlMaster socket was absent and direct SSH returned `Permission denied
> (password,publickey)`. The local Isaac container is an existing, foreign
> Isaac Sim 6.1 instance using GPU memory, and is not the configured 6.2 server
> instance. It was not stopped, changed, or used. A prior direct-Python
> articulation smoke is not a ROS 2 action/feedback PASS.

## Supported Isaac Sim 6.2 route

Isaac Sim 6.2 has two relevant paths:

- The **ROS 2 Bridge** provides `JointState` publish/subscribe nodes, suitable
  for position/velocity setpoints, but it does not host a
  `FollowJointTrajectory` action server.
- This adapter targets the 6.x-native
  **`isaacsim.ros2.control`** extension. It hosts a standard ros2_control
  `controller_manager` in the Isaac process and exposes a USD articulation as
  `IsaacSimSystem`. A `joint_trajectory_controller` then owns
  `/<controller_name>/follow_joint_trajectory`.

The second route is required for real FollowJointTrajectory goal acceptance,
feedback/result monitoring, and controller outcome handling.

Authoritative 6.2 references:

- [ROS 2 Control tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/ros2_tutorials/robot_control/tutorial_ros2_control.html)
- [ROS 2 Clock tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/ros2_tutorials/tutorial_series/tutorial_ros2_clock.html)
- [Isaac Sim ROS 2 Control extension overview](https://github.com/isaac-sim/IsaacSim/blob/main/source/extensions/isaacsim.ros2.control/docs/Overview.md)
- [Official UR10 controller configuration](https://github.com/isaac-sim/IsaacSim/blob/main/source/standalone_examples/api/isaacsim.ros2.control/ur10_controllers.yaml)

## Isaac-side bring-up

### 1. Start the right extensions before building the graph

For a standalone 6.2 application, enable the extensions before assembling the
scene/OmniGraph:

```python
app_utils.enable_extension("isaacsim.ros2.core")
app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("isaacsim.ros2.control")
```

For a Kit launch, select the current extension name (not an older `omni.*`
name):

```bash
./isaac-sim.sh \
  --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
```

Use a supported ROS distribution: Isaac Sim 6.2 ros2_control supports Humble
and Jazzy; Jazzy is the intended Ubuntu 24.04 route. ROS library selection
must happen before Isaac starts.

### 2. Load the robot and verify names

Load the B601-RS USD as one articulation root, for example `/World/ReBot`.
The profile at `examples/robot_adapters/profiles/rebot-b601-rs.json` requires
these arm DriveAPI/DOF names exactly:

```text
joint1 joint2 joint3 joint4 joint5 joint6
```

The gripper profile names are `gripper_joint1`, `gripper_joint2`. The shipped
sample controller currently owns the **arm** only. The Controller Manager
synthesizes a URDF from USD drives: a driven joint must expose a position
command interface (USD drive stiffness > 0). Do not use the incoming
`JointState` array order as a command mapping; always use its `name[]` values.

Before a live command, confirm the synthesized robot description and active
controller inventory:

```bash
ros2 topic echo --once /robot_description
ros2 control list_controllers --controller-manager /controller_manager
```

For multi-robot scenes, add a unique `namespace` to each
`ROS2ControlManager`/adapter config. It scopes the Controller Manager,
controllers, action endpoint, and robot description. Do not merge joint names
from different robots into one profile.

### 3. Publish `/clock` and use simulation time

With `Ros2ControlManager.setup(..., use_sim_time=True)` (the default), create
this playback-driven graph:

```text
OnPlaybackTick
  ├─> IsaacReadSimulationTime ─> ROS2PublishClock (/clock)
  └─> ROS2ControlManager
```

The controller and external ROS nodes must use sim time. `/clock` is only
published during playback; stopping the timeline also stops trajectory
progress, so the adapter uses a wall-monotonic watchdog for its client-side
result wait. Controller Manager lifecycle is tied to playback: it is torn down
on Stop, and controllers must be reactivated after the next Play.

### 4. Configure and activate controllers

The checked-in sample is
`examples/robot_adapters/isaac_ros2_controllers.yaml`. It names:

- manager: `/controller_manager`
- state publisher: `joint_state_broadcaster`
- arm action controller: `rebot_arm_rs_isaac_trajectory_controller`
- action: `/rebot_arm_rs_isaac_trajectory_controller/follow_joint_trajectory`
- state topic: `/joint_states`

Set up the Isaac-hosted manager after the first physics tick / Play:

```python
from isaacsim.ros2.control import Ros2ControlManager

Ros2ControlManager.setup(
    "/World/ReBot",
    "/workspace/examples/robot_adapters/isaac_ros2_controllers.yaml",
    namespace="",
    use_sim_time=True,
)
```

Then activate both controllers using the manager's `switch_controller` service
(or `ros2 control load_controller --set-state active ...`). A controller may be
listed but inactive; the action server must be active before the adapter's
`connect()` can succeed.

## Adapter behavior and safety boundary

```python
from simready.robotics.profiles import load_robot_profile
from simready.robotics.ros2_control_adapter import (
    IsaacRos2ControlAdapter,
    Ros2AdapterConfig,
)

profile = load_robot_profile(profile_path)
adapter = IsaacRos2ControlAdapter(
    profile,
    Ros2AdapterConfig(
        controller_name="rebot_arm_rs_isaac_trajectory_controller",
        joint_states_topic="/joint_states",
        max_joint_delta=0.08,
        goal_timeout_s=30.0,
    ),
)
adapter.connect()
```

`connect()` requires both a ready action server and a complete JointState sample
covering the profile's joints. `get_state()` records a local monotonic receipt
time separately from the ROS `/clock` timestamp, rejects stale feedback, and
returns positions in canonical radians/metres. Empty velocity/effort arrays
remain empty; populated arrays must align with names.

`send_joint_position()`:

1. uses the existing `validate_joint_command()` bounds, group, expiry and
   sequence checks;
2. applies the configured per-joint delta cap against measured feedback;
3. sends a one-point trajectory in the profile group order;
4. waits for the action result with a bounded watchdog;
5. on timeout, cancels the in-flight goal before returning the receipt;
6. returns `CommandReceipt(accepted=False, ...)` on goal rejection, abort,
   preemption, or timeout so a policy host can report the error and continue.

Once an action server accepts a goal, its sequence is consumed even if its
terminal result later fails: retries must use a new sequence. A goal rejected
before acceptance does not consume the sequence.

Timeout and cancellation semantics: when the result watchdog expires, the
adapter requests goal cancellation and waits (bounded again by
`goal_timeout_s`) for the goal to reach a terminal state before returning
`accepted=False`. If cancellation succeeds, the next command is accepted
normally. If cancellation cannot be confirmed (the cancel response or the
terminal result never arrives), the adapter holds a fail-closed state and
raises `Ros2TransportError` on further `send_joint_position()` and `stop()`
calls until `close()` (a confirmed `stop()` cancel also re-syncs sends); it
never silently continues a trajectory whose outcome is unknown. The same
protection applies when `wait_for_result` itself raises: the adapter attempts
one bounded cancellation and reports the failure rather than assuming the
goal died.

`stop()` cancels any unresolved in-flight goal (bounded by `goal_timeout_s`).
If cancellation is confirmed, it then sends a short
hold-at-measured-position trajectory re-targeting the freshest measured
positions; if cancellation cannot be confirmed, it raises
`Ros2TransportError` and sends nothing, because a hold trajectory would
compete with a goal that may still be running. It is **not an E-stop**,
controller deactivation, collision response, or hardware safety mechanism.
This adapter is for Isaac simulation only; it never marks the reBot hardware
profile verified.

## Required live-smoke procedure

Only run the optional `examples/robot_adapters/isaac_ros2_smoke.py` after:

1. Reading the live shared GPU coordination record.
2. Updating it with owner/start/expected end.
3. Acquiring the remote lock around the entire GPU/Isaac command:
   `flock /root/gpufree-data/.simready/gpu-coordination.lock -c '<command>'`.
4. Confirming the pre-existing configured Isaac Sim 6.2 server has no foreign
   active workload and never stopping/modifying foreign processes.
5. Ensuring the Controller Manager, `/clock`, state broadcaster, and trajectory
   controller are active as above.

Example (paths are server-specific and must be checked first):

```bash
python examples/robot_adapters/isaac_ros2_smoke.py \
  --asset /server/path/ReBot_Arm_RS.usda \
  --profile examples/robot_adapters/profiles/rebot-b601-rs.json \
  --controllers-yaml examples/robot_adapters/isaac_ros2_controllers.yaml \
  --evidence-dir /server/run/module-c-ros2
```

A PASS requires evidence of a real `FollowJointTrajectory` action result **and**
measured post-action `/joint_states`: source commit, profile hash, observation
timestamp, action request, validation outcome, receipt/result, measured
post-action state, tracking error, stop/timeout result, and independent task
verifier result. `isaac_ros2_smoke.py` emits the action/state evidence but a
caller must add the profile hash, source commit, task verifier and any image
hash without logging credentials or private camera contents.

If server connectivity, ROS runtime, required action server, or the GPU lock is
unavailable, report `BLOCKED` or `NOT_RUN` with the exact reason. Do not run
this against hardware or substitute an in-process direct Isaac articulation
call for ROS 2 acceptance.
