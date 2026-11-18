# Robot adapters: sim and hardware boundary

## Purpose and current status

The `simready.robotics` package defines the common I/O boundary for robot
backends. Policy and task code should consume normalized state and issue
bounded commands through this boundary; it should not depend on vendor SDK
types, ROS message ordering, or Isaac prim paths.

The current feature branch provides canonical joint and Cartesian action
contracts, a typed Qwen policy provider, direct Isaac articulation and
Isaac-hosted `ros2_control` adapters, plus separate reBot driver profiles.
CPU contract tests pass. The ROS 2 adapter has not yet passed a live Isaac
action/feedback smoke; the official reBot profile is configuration-only, and
no real hardware adapter has been commissioned. Profiles and mocked tests do
not prove that a physical robot is safe to operate. Hardware commands remain
disabled unless a profile has passed hardware-specific commissioning and is
explicitly marked verified.

## Layering

```text
task + observation history
          │
Qwen provider (typed tool call; no execution)
          │
Cartesian/joint contract + freshness and bounds checks
          │
RobotAdapter
   ├── Isaac direct articulation → USD/PhysX
   └── Isaac ros2_control simulation adapter → ROS 2 action/state
                                      │
                              future vendor adapter
                                      │
                             manufacturer API/ROS 2
```

The adapter is responsible for joint-name mapping, ordering, unit conversion,
timestamps, limits, transport timeouts, and stop/close behavior. The contract
uses radians for revolute joints and metres for prismatic joints. State receipt
time is monotonic local time so staleness checks do not depend on synchronized
wall clocks. Commands address exactly one declared group and have a sequence
number and finite validity window. A separate safety supervisor should still
enforce workspace, speed, collision, emergency-stop, and operator-enable rules.

## Simulation policy-to-adapter loop

`simready.robotics.policy_loop.PolicyAdapterLoop` supplies the missing
caller-side boundary between the inert Qwen provider and a simulation adapter.
It is deliberately separate from `QwenPolicyProvider`, which continues to parse
only typed data and never imports an adapter or executes a tool. A host injects
all three collaborators: a policy provider, a `RobotAdapter`, and an
independent feedback verifier.

For one `run_once()` step the loop reads and validates fresh pre-command state,
sends the caller observation plus that state to exactly one generated joint-group
tool schema, requires one call to the configured tool, constructs a short-lived
`JointCommand`, runs `validate_joint_command`, and only then invokes the
adapter. Following an accepted receipt it reads and validates fresh post-command
state and passes both states, the command, and the receipt to the verifier. The
verifier's result is the only source of the `verified` status; a model response
is never task-success evidence.

The loop is simulation-only: construction rejects profiles with
`hardware_verified=True`, and it cannot create transports or import ROS/Isaac.
Provider/state/verifier timeouts, policy refusal/no-tool/unknown-tool results,
malformed or invalid arguments, adapter rejection, and bad feedback all return
explicit fail-closed statuses. In particular, no tool/validation/policy failure
sends a command. The injected adapter remains responsible for its own transport
and low-level delta/timeout enforcement.

CPU-only fake tests exercise this sequence and its rejection paths. They are
synthetic integration evidence only, **not** a Qwen live-inference PASS, Isaac
ROS 2 live-action PASS, or the live end-to-end canary required before PR #31
can merge. Those acceptance gates and their evidence requirements remain in
`docs/ROBOT_ADAPTER_ROADMAP_AND_ACCEPTANCE.md`.

`RobotAdapter` exposes connect, state read, joint-position command, stop, and
close. Cartesian target validation exists as a separate pure contract; it does
not yet supply IK, collision checking, trajectory generation, or actuation.
Reset/home, camera, and gripper commands remain explicit capabilities; adapters
must not pretend every robot supports them. This avoids making a lowest-common-
denominator method silently unsafe or meaningless.

## Pilot profile: Seeed reBot Arm B601-RS

`examples/robot_adapters/profiles/rebot-b601-rs.json` declares six arm joints
and two gripper joints. The profile records canonical units and the RS URDF
limits. The referenced community ROS 2 driver exposes `/joint_states` and a
`FollowJointTrajectory` controller; the Isaac simulation controller has a
separate action endpoint. The community driver is not an official Seeed
hardware certification. The manifest therefore keeps hardware unverified and
the common contract rejects hardware commands by default.

Seeed's [reBot-Isaacsim](https://github.com/Seeed-Projects/reBot-Isaacsim)
provides the RS model/Isaac integration. The ROS 2 controller mapping in this
profile references the community
[rebot_arm_rs_ros2](https://github.com/xiayip/rebot_arm_rs_ros2) project. Before
using a physical arm, confirm the exact B601-RS revision, CAN configuration,
joint sign/order, gripper semantics, controller action, emergency stop, and
limits with the device present. Do not reuse this RS profile for B601-DM.

## Pilot profile: AgiBot Genie G2

`examples/robot_adapters/profiles/agibot-genie-g2-crsb-sim.json` is explicitly
a Genie Sim profile for the crsB no-gripper model. It describes body, head,
left arm, and right arm joint groups, and the public simulation ROS topics
`/joint_states` and `/joint_command`. The profile intentionally does not invent
joint limits. At runtime, the adapter must get limits and exact active joint
names from the loaded model/configuration and reject a mismatch.

The [AgiBot Genie Sim](https://github.com/AgibotTech/genie_sim) ROS bridge and
[Genie robot models](https://github.com/AgibotTech/genie_sim_robot_model)
document simulation assets and interfaces. A simulation topic is not evidence
of a physical G2 control API. The hardware command endpoint, authentication,
unit conventions, stop behavior, and safety procedure remain unverified; this
profile has no hardware command transport.

## Relationship to RobotControlStack

We borrow the useful separation: a robot-facing contract, one adapter per
hardware family, and environment construction that composes robot, gripper,
cameras, and action bounds outside policy code. We are not importing its
runtime or copying implementation. RobotControlStack is AGPL-3.0 while this
repository is MIT, so any future code reuse requires a separate license review.
Its MuJoCo-specific simulator wrapper also does not directly provide an Isaac
Sim adapter.

## Astra / Code-as-Policy control loop

The NVIDIA Astranomy study describes its agentic setup as a persistent Codex
session: Astra receives camera frames and current robot state, proposes a
bounded end-effector target, the host validates and executes it, then returns
a fresh observation. The agent keeps prior decisions and feedback in context.
Optional Python/image tools can inspect saved observations. This is a
closed-loop agent harness; it is not a hard real-time joint servo. Their report
measures inference calls in seconds, so a local low-level controller must own
the fast control loop.

For this repository the intended mapping is:

```text
camera RGB + normalized joint state + task/history
                     ↓
             Astra proposes one target
                     ↓
     strict action schema + freshness/limit checks
                     ↓
       Isaac adapter or ROS 2 adapter executes
                     ↓
      measured state + new camera frames + errors
                     └──────── back to Astra
```

The first Isaac implementation is `IsaacSimArticulationAdapter`. It maps named
profile joints into Isaac's articulation indices, applies one position target
for one physics step, and returns measured joint positions/velocities. The
host owns the `SimulationApp`, world, and camera sensors. Each agent action
must be followed by a new observation; a model's `done` response is never a
task-success signal. The host needs a separate task verifier.

## Isaac server pilot

Run `examples/robot_adapters/isaac_rebot_agent_loop.py` on the GPU server after
placing the official RS USD asset at the documented path. It publishes a
small JSON observation (joint state, action feedback and image path) and waits
for a one-group joint target in `command.json`. Upload to a temporary filename
and rename it atomically so the running worker never reads a partial JSON file.
The agent reads the image and state, writes a bounded target, then receives the
next state/image and measured action feedback. The server pilot verified this
loop on the L40/Isaac Sim 6.2 with two small joint1 moves; the measured target
error was below 0.002 rad. It stores RGB observations, JSON state, and the exact
command/event trace, not a video. An out-of-limit test target was rejected and
returned with the reason; a subsequent valid target was accepted, so one bad
action does not kill the session. This file handoff is intended for the first
Astra-in-the-loop canary; replace it with a scoped tool service for concurrent
users. The current canary demonstrates the Code-as-Policy control boundary, but
does not launch an Astra model/API session itself or solve a task autonomously.
It operates only the Isaac model and never connects to CAN, a vendor SDK, or
real hardware.

## Environment composition

The next integration step is a workflow/environment factory that receives a
profile and explicit backend, then composes optional gripper, camera, action
scaling, and safety wrappers. Device discovery and credentials belong in
runtime configuration, never in the checked-in profile. A hardware backend
must remain opt-in and fail closed when state is stale, joints differ, a
command expires, or commissioning is incomplete.

## Current validation scope

CPU tests cover profile schemas, command/state contracts, typed Qwen tool-call
parsing, Cartesian action freshness and limits, and the ROS 2 transport using
an injected fake client. The Isaac direct-articulation canary has been run on
the GPU server. The optional Isaac `ros2_control` adapter still needs a live
server-side ROS action and measured-state acceptance run. The official Seeed
reBot profile records documented interfaces but deliberately leaves executable
command endpoints unset; no physical robot command or hardware commissioning
has occurred. AgiBot G2 remains a simulation profile only.
