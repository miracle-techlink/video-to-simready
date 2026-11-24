"""Run one local XPolicyLab-style model through the simulation policy boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from simready.robotics.qwen_policy import PolicyDecision


class LocalModel(Protocol):
    """The synchronous, single-episode subset of XPolicyLab ModelTemplate."""

    def reset(self) -> None: ...

    def update_obs(self, obs: dict[str, Any]) -> None: ...

    def get_action(self) -> Any: ...


class XPolicyLabProvider:
    """Adapt an installed local model with an explicit embodiment-specific codec."""

    def __init__(
        self,
        model: LocalModel,
        build_obs: Callable[[str, Mapping[str, Any], Sequence[str]], dict[str, Any]],
        decode_action: Callable[
            [Any, Mapping[str, Any], Sequence[Mapping[str, Any]]], PolicyDecision
        ],
    ) -> None:
        self._model = model
        self._build_obs = build_obs
        self._decode_action = decode_action
        self._ready = False

    def reset(self) -> None:
        """Start a fresh episode before allowing inference."""
        self._ready = False
        self._model.reset()
        self._ready = True

    def decide(
        self,
        *,
        instructions: str,
        observation: Mapping[str, Any],
        tools: Sequence[Mapping[str, Any]],
        image_data_urls: Sequence[str] = (),
    ) -> PolicyDecision:
        if not self._ready:
            raise ValueError("reset the local model before running an episode")
        try:
            self._model.update_obs(
                self._build_obs(instructions, observation, image_data_urls)
            )
            return self._decode_action(self._model.get_action(), observation, tools)
        except Exception:
            self._ready = False
            raise
