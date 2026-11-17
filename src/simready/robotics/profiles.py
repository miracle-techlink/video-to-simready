"""Load versioned robot profile manifests without importing robot runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simready.robotics.contracts import JointSpec, RobotProfile


def load_robot_profile(path: Path) -> RobotProfile:
    """Read and validate a schema-v1 robot profile JSON file."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return robot_profile_from_dict(document)


def robot_profile_from_dict(document: dict[str, Any]) -> RobotProfile:
    if document.get("schema_version") != 1:
        raise ValueError("robot profile schema_version must be 1")
    if document.get("kind") != "robot_profile":
        raise ValueError("document kind must be robot_profile")
    transport = document.get("transport", {})
    hardware = transport.get("hardware", {})
    simulation = transport.get("simulation", {})
    joints = tuple(
        JointSpec(
            name=item["name"],
            group=item["group"],
            unit=item["unit"],
            lower=item.get("lower"),
            upper=item.get("upper"),
        )
        for item in document["joints"]
    )
    commandable = document.get("commandable_groups")
    if commandable is not None and not isinstance(commandable, list):
        raise ValueError("commandable_groups must be a list of group names")
    return RobotProfile(
        robot_id=document["robot_id"],
        display_name=document["display_name"],
        joints=joints,
        groups={name: tuple(members) for name, members in document["groups"].items()},
        state_transport=document["state_transport"],
        simulation_command_transport=simulation["command"],
        hardware_command_transport=hardware.get("command"),
        hardware_verified=hardware.get("verified", False),
        commandable_groups=tuple(commandable) if commandable is not None else None,
        asset_reference=document.get("asset_reference"),
        source_reference=document.get("source_reference"),
    )
