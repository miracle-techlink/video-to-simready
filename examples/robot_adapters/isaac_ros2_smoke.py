#!/usr/bin/env python3
"""Isaac-hosted ros2_control live smoke for the reBot B601-RS (Module C).

Runs ONLY where Isaac Sim, the in-process ControllerManager, and a ROS 2
runtime coexist (the GPU server instance per the shared GPU lock, or an Isaac
container with isaacsim.ros2.control enabled). It is NOT part of the default
test suite and a direct-API articulation run can never substitute for it.

Server bring-up before this script (see docs/ISAAC_ROS2_ADAPTER.md):
  - shared GPU lock acquired (flock gpu-coordination.lock, gpu-coordination.json updated)
  - reBot USD loaded as an articulation at /World/ReBot
  - /clock publisher (bridge clock graph) running
  - Ros2ControlManager.setup("/World/ReBot", controllers_yaml, namespace="")
  - joint_state_broadcaster + trajectory controller activated

The script commands one small bounded joint1 move through
IsaacRos2ControlAdapter (a real FollowJointTrajectory action goal) and records
measured state before/after, the action outcome, and tracking error into an
evidence directory. Exit code 0 only on a real ROS 2 action result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--controllers-yaml", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True, help="ReBot_Arm_RS.usda")
    parser.add_argument(
        "--prim-path", default="/World/ReBot", help="articulation prim path"
    )
    parser.add_argument("--joint", default="joint1")
    parser.add_argument("--delta", type=float, default=0.05, help="bounded move (rad)")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from simready.robotics.contracts import JointCommand
    from simready.robotics.profiles import load_robot_profile
    from simready.robotics.ros2_control_adapter import (
        IsaacRos2ControlAdapter,
        Ros2AdapterConfig,
    )

    profile = load_robot_profile(args.profile)
    if args.joint not in profile.groups.get("arm", ()):  # arm-group smoke only
        raise SystemExit(f"joint {args.joint!r} is not in profile group 'arm'")
    config = Ros2AdapterConfig(
        controller_name="rebot_arm_rs_isaac_trajectory_controller",
        goal_timeout_s=args.timeout_s,
    )

    adapter = IsaacRos2ControlAdapter(profile, config)
    started = time.time()
    try:
        adapter.connect()
    except Exception as error:
        atomic_json(
            args.evidence_dir / "blocked.json",
            {
                "status": "BLOCKED",
                "reason": f"adapter connect failed: {error!r}",
                "action_server": config.follow_joint_trajectory_action,
                "elapsed_s": time.time() - started,
            },
        )
        return 2

    before = adapter.get_state(timeout_s=5.0)
    target = {joint: before.positions[joint] for joint in profile.groups["arm"]}
    target[args.joint] = min(
        max(
            target[args.joint] + args.delta,
            profile.joint_map()[args.joint].lower or target[args.joint],
        ),
        profile.joint_map()[args.joint].upper or target[args.joint],
    )
    command = JointCommand(
        group="arm",
        positions=target,
        issued_monotonic_ns=time.monotonic_ns(),
        valid_for_s=2.0,
        sequence=1,
    )
    receipt = adapter.send_joint_position(command)
    after = adapter.get_state(timeout_s=5.0)
    error_rad = max(
        abs(target[joint] - after.positions[joint]) for joint in profile.groups["arm"]
    )
    evidence = {
        "schema_version": 1,
        "route": "isaac_ros2_control",
        "action_server": config.follow_joint_trajectory_action,
        "joint_states_topic": config.joint_states_topic,
        "command": {"group": "arm", "positions": dict(target)},
        "receipt": {
            "sequence": receipt.sequence,
            "accepted": receipt.accepted,
            "detail": receipt.detail,
        },
        "measured_before": dict(before.positions),
        "measured_after": dict(after.positions),
        "max_abs_target_error_rad": error_rad,
        "source_before_ns": before.source_timestamp_ns,
        "source_after_ns": after.source_timestamp_ns,
    }
    atomic_json(args.evidence_dir / "smoke.json", evidence)
    print(json.dumps(evidence), flush=True)
    return 0 if receipt.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
