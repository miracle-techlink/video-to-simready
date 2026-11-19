"""Isaac Sim experimental articulation adapter."""

from __future__ import annotations

import time
from typing import Any

from simready.robotics.contracts import (
    CommandReceipt,
    JointCommand,
    RobotProfile,
    RobotState,
    validate_joint_command,
)


class IsaacSimArticulationAdapter:
    """Map normalized joint commands to one initialized Isaac articulation.

    The adapter does not create or own ``SimulationApp`` or camera sensors.
    The caller owns the Isaac lifecycle and supplies a world per
    accepted action. This keeps agent/tool latency outside simulation time.
    """

    def __init__(
        self,
        profile: RobotProfile,
        articulation: Any,
        world: Any,
    ):
        self.profile = profile
        self._articulation = articulation
        self._world = world
        self._joint_names = list(articulation.dof_names)
        self._indices: dict[str, int] = {}
        self._connected = False
        self._last_sequence = -1

    @property
    def execution_mode(self) -> str:
        """This adapter only ever drives the in-process Isaac simulation."""
        return "simulation"

    def connect(self) -> None:
        missing = set(self.profile.joint_map()) - set(self._joint_names)
        if missing:
            raise ValueError(
                f"Isaac articulation is missing profile joints: {sorted(missing)}"
            )
        self._indices = {
            name: self._joint_names.index(name) for name in self.profile.joint_map()
        }
        self._articulation.initialize()
        self._connected = True

    def get_state(self, *, timeout_s: float = 0.0) -> RobotState:
        import numpy as np

        del timeout_s  # Isaac state reads are in-process and non-blocking.
        self._require_connected()
        raw_positions = self._articulation.get_joint_positions()
        raw_velocities = self._articulation.get_joint_velocities()
        if raw_positions is None or raw_velocities is None:
            raise RuntimeError(
                "Isaac articulation returned no joint feedback; ensure World.step() "
                "has initialized and advanced the physics view"
            )
        positions = np.asarray(raw_positions, dtype=float).reshape(-1)
        velocities = np.asarray(raw_velocities, dtype=float).reshape(-1)
        required = max(self._indices.values(), default=-1)
        if positions.size <= required or velocities.size <= required:
            raise RuntimeError(
                "Isaac articulation joint feedback is shorter than the profile mapping"
            )
        return RobotState(
            positions={
                name: float(positions[index]) for name, index in self._indices.items()
            },
            velocities={
                name: float(velocities[index]) for name, index in self._indices.items()
            },
            received_monotonic_ns=time.monotonic_ns(),
            source_timestamp_ns=None,
            source="isaac_sim",
        )

    def send_joint_position(self, command: JointCommand) -> CommandReceipt:
        import numpy as np

        self._require_connected()
        now = time.monotonic_ns()
        validate_joint_command(self.profile, command, now_monotonic_ns=now)
        if command.sequence <= self._last_sequence:
            raise ValueError("command sequence must increase monotonically")

        ordered = list(self.profile.groups[command.group])
        targets = np.asarray(
            [command.positions[name] for name in ordered], dtype=np.float64
        )
        indices = np.asarray([self._indices[name] for name in ordered], dtype=np.int64)
        from isaacsim.core.utils.types import ArticulationAction

        self._articulation.apply_action(
            ArticulationAction(joint_positions=targets, joint_indices=indices)
        )
        self._world.step(render=True)
        self._last_sequence = command.sequence
        return CommandReceipt(
            sequence=command.sequence,
            accepted=True,
            accepted_monotonic_ns=time.monotonic_ns(),
            detail="one Isaac physics/render step applied; inspect returned state before next action",
        )

    def stop(self) -> None:
        """Hold measured joint positions for one step; this is not an E-stop."""
        import numpy as np

        self._require_connected()
        state = self.get_state()
        from isaacsim.core.utils.types import ArticulationAction

        names = list(self.profile.joint_map())
        values = np.asarray([state.positions[name] for name in names], dtype=np.float64)
        indices = np.asarray([self._indices[name] for name in names], dtype=np.int64)
        self._articulation.apply_action(
            ArticulationAction(joint_positions=values, joint_indices=indices)
        )
        self._world.step(render=True)

    def close(self) -> None:
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "connect() must be called before using the Isaac adapter"
            )
