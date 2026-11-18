"""Fail-closed orchestration from an inert policy call to a simulation adapter.

This module is the caller-side control boundary deliberately excluded from
``qwen_policy``: it reads fresh adapter state, asks a policy for one declared
joint-position tool, validates the resulting command, dispatches it only after
validation, then gives fresh post-command feedback to an independently injected
verifier. It never creates a transport, imports ROS/Isaac, or permits hardware.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from simready.robotics.contracts import (
    CommandReceipt,
    JointCommand,
    RobotAdapter,
    RobotState,
    validate_joint_command,
    validate_robot_state,
)
from simready.robotics.qwen_policy import (
    PolicyDecision,
    QwenPolicyError,
    QwenTimeoutError,
)


@dataclass(frozen=True)
class PolicyLoopConfig:
    """Bound one simulation policy step and name its sole permitted tool."""

    group: str
    tool_name: str
    command_valid_for_s: float = 2.0
    max_joint_step_rad: float = 0.08
    state_timeout_s: float = 0.5
    verification_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if not self.group or not self.tool_name:
            raise ValueError("group and tool_name must be nonempty")
        for name in (
            "command_valid_for_s",
            "max_joint_step_rad",
            "state_timeout_s",
            "verification_timeout_s",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class VerificationResult:
    """Independent assessment of post-command feedback, never a model claim."""

    accepted: bool
    detail: str = ""


@dataclass(frozen=True)
class PolicyLoopResult:
    """One completed or fail-closed policy loop iteration."""

    status: str
    detail: str = ""
    command: JointCommand | None = None
    receipt: CommandReceipt | None = None
    state_before: RobotState | None = None
    state_after: RobotState | None = None
    verification: VerificationResult | None = None

    @property
    def dispatched(self) -> bool:
        """Whether the adapter was called with a validated command."""
        return self.receipt is not None


class PolicyProvider(Protocol):
    """Inert policy boundary; compatible with ``QwenPolicyProvider``."""

    def decide(
        self,
        *,
        instructions: str,
        observation: Mapping[str, Any],
        tools: Sequence[Mapping[str, Any]],
        image_data_urls: Sequence[str] = (),
    ) -> PolicyDecision: ...


class FeedbackVerifier(Protocol):
    """Independent post-command feedback check supplied by the host.

    ``timeout_s`` is a cooperative deadline the verifier must enforce itself;
    the loop reports a ``verification_timeout`` result only if the verifier
    signals it (for example by raising ``TimeoutError``). The loop itself does
    not interrupt a verifier that overruns.
    """

    def verify(
        self,
        *,
        command: JointCommand,
        receipt: CommandReceipt,
        state_before: RobotState,
        state_after: RobotState,
        timeout_s: float,
    ) -> VerificationResult: ...


class PolicyAdapterLoop:
    """Execute one injected-policy simulation step with validation and verification."""

    def __init__(
        self,
        policy: PolicyProvider,
        adapter: RobotAdapter,
        verifier: FeedbackVerifier,
        config: PolicyLoopConfig,
        *,
        now_monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        # Simulation scope is an explicit adapter capability, never inferred from
        # the profile's commissioning flags: a profile carrying an uncommissioned
        # hardware command transport must not make an injected adapter physical.
        execution_mode = getattr(adapter, "execution_mode", None)
        if execution_mode != "simulation":
            raise ValueError(
                "PolicyAdapterLoop requires an adapter that declares "
                "execution_mode='simulation' explicitly"
            )
        if adapter.profile.hardware_verified:
            raise ValueError(
                "PolicyAdapterLoop permits simulation-compatible adapters only"
            )
        if config.group not in adapter.profile.groups:
            raise ValueError(f"unknown joint group: {config.group!r}")
        if not adapter.profile.group_is_commandable(config.group):
            raise ValueError(f"joint group {config.group!r} is not commandable")
        self._policy = policy
        self._adapter = adapter
        self._verifier = verifier
        self._config = config
        self._now_monotonic_ns = now_monotonic_ns
        self._last_sequence = -1

    @property
    def tool_schema(self) -> tuple[Mapping[str, Any], ...]:
        """Return the sole tool schema, exactly matching the configured joint group."""
        names = self._adapter.profile.groups[self._config.group]
        return (
            {
                "type": "function",
                "function": {
                    "name": self._config.tool_name,
                    "description": "Set bounded canonical joint position targets.",
                    "parameters": {
                        "type": "object",
                        "properties": {name: {"type": "number"} for name in names},
                        "required": list(names),
                        "additionalProperties": False,
                    },
                },
            },
        )

    def run_once(
        self,
        *,
        instructions: str,
        observation: Mapping[str, Any],
        image_data_urls: Sequence[str] = (),
    ) -> PolicyLoopResult:
        """Run one bounded step; any failure returns without an unsafe dispatch."""
        try:
            state_before = self._adapter.get_state(
                timeout_s=self._config.state_timeout_s
            )
        except TimeoutError as error:
            return PolicyLoopResult(status="state_timeout", detail=str(error))
        except Exception as error:
            return PolicyLoopResult(status="state_error", detail=str(error))
        try:
            validate_robot_state(
                self._adapter.profile,
                state_before,
                now_monotonic_ns=self._now_monotonic_ns(),
                max_age_s=self._config.state_timeout_s,
            )
        except ValueError as error:
            return PolicyLoopResult(status="state_invalid", detail=str(error))

        policy_observation = {
            "caller_observation": dict(observation),
            "robot_state": {
                "positions": dict(state_before.positions),
                "received_monotonic_ns": state_before.received_monotonic_ns,
                "source": state_before.source,
            },
        }
        try:
            decision = self._policy.decide(
                instructions=instructions,
                observation=policy_observation,
                tools=self.tool_schema,
                image_data_urls=tuple(image_data_urls),
            )
        except QwenTimeoutError as error:
            return PolicyLoopResult(
                status="policy_timeout", detail=str(error), state_before=state_before
            )
        except TimeoutError as error:
            return PolicyLoopResult(
                status="policy_timeout", detail=str(error), state_before=state_before
            )
        except QwenPolicyError as error:
            return PolicyLoopResult(
                status="policy_error", detail=str(error), state_before=state_before
            )
        except Exception as error:
            return PolicyLoopResult(
                status="policy_error", detail=str(error), state_before=state_before
            )

        if decision.refusal:
            return PolicyLoopResult(status="policy_refusal", state_before=state_before)
        if len(decision.calls) != 1:
            return PolicyLoopResult(
                status="no_single_tool_call",
                detail=f"expected one tool call, received {len(decision.calls)}",
                state_before=state_before,
            )
        call = decision.calls[0]
        if call.name != self._config.tool_name:
            return PolicyLoopResult(
                status="unknown_tool",
                detail=f"policy requested {call.name!r}",
                state_before=state_before,
            )

        command = JointCommand(
            group=self._config.group,
            positions=dict(call.arguments),
            issued_monotonic_ns=self._now_monotonic_ns(),
            valid_for_s=self._config.command_valid_for_s,
            sequence=self._last_sequence + 1,
        )
        try:
            validate_joint_command(
                self._adapter.profile,
                command,
                now_monotonic_ns=self._now_monotonic_ns(),
            )
        except (TypeError, ValueError) as error:
            return PolicyLoopResult(
                status="validation_failed",
                detail=str(error),
                command=command,
                state_before=state_before,
            )

        # Profile limits alone do not bound one policy step: reject targets whose
        # per-joint delta from the measured pre-command state exceeds the cap.
        step_error = _joint_step_error(
            command, state_before, self._config.max_joint_step_rad
        )
        if step_error is not None:
            return PolicyLoopResult(
                status="step_limit_exceeded",
                detail=step_error,
                command=command,
                state_before=state_before,
            )

        try:
            receipt = self._adapter.send_joint_position(command)
        except TimeoutError as error:
            return PolicyLoopResult(
                status="adapter_timeout",
                detail=str(error),
                command=command,
                state_before=state_before,
            )
        except Exception as error:
            return PolicyLoopResult(
                status="adapter_error",
                detail=str(error),
                command=command,
                state_before=state_before,
            )
        self._last_sequence = command.sequence
        if not receipt.accepted:
            return PolicyLoopResult(
                status="adapter_rejected",
                detail=receipt.detail,
                command=command,
                receipt=receipt,
                state_before=state_before,
            )

        try:
            state_after = self._adapter.get_state(
                timeout_s=self._config.state_timeout_s
            )
        except TimeoutError as error:
            return PolicyLoopResult(
                status="post_state_timeout",
                detail=str(error),
                command=command,
                receipt=receipt,
                state_before=state_before,
            )
        except Exception as error:
            return PolicyLoopResult(
                status="post_state_error",
                detail=str(error),
                command=command,
                receipt=receipt,
                state_before=state_before,
            )
        try:
            validate_robot_state(
                self._adapter.profile,
                state_after,
                now_monotonic_ns=self._now_monotonic_ns(),
                max_age_s=self._config.state_timeout_s,
            )
        except ValueError as error:
            return PolicyLoopResult(
                status="post_state_invalid",
                detail=str(error),
                command=command,
                receipt=receipt,
                state_before=state_before,
                state_after=state_after,
            )
        try:
            verification = self._verifier.verify(
                command=command,
                receipt=receipt,
                state_before=state_before,
                state_after=state_after,
                timeout_s=self._config.verification_timeout_s,
            )
        except TimeoutError as error:
            return PolicyLoopResult(
                status="verification_timeout",
                detail=str(error),
                command=command,
                receipt=receipt,
                state_before=state_before,
                state_after=state_after,
            )
        except Exception as error:
            return PolicyLoopResult(
                status="verification_error",
                detail=str(error),
                command=command,
                receipt=receipt,
                state_before=state_before,
                state_after=state_after,
            )
        return PolicyLoopResult(
            status="verified" if verification.accepted else "verification_failed",
            detail=verification.detail,
            command=command,
            receipt=receipt,
            state_before=state_before,
            state_after=state_after,
            verification=verification,
        )


def _joint_step_error(
    command: JointCommand, state_before: RobotState, max_joint_step_rad: float
) -> str | None:
    """Return a rejection detail when a target exceeds the measured step cap."""
    deltas = {
        name: abs(command.positions[name] - state_before.positions[name])
        for name in command.positions
    }
    worst = max(deltas.values(), default=0.0)
    if worst <= max_joint_step_rad:
        return None
    return f"command exceeds per-joint step cap {max_joint_step_rad}: {deltas}"
