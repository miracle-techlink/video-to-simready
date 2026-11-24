"""Local Astra model and a translation-only subset of the Direct EEF tool."""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

from simready.robotics.policy_loop import PolicyProvider

_TRANSLATION_DIMENSIONS = frozenset(
    f"{arm}_{axis}" for arm in ("left", "right") for axis in "xyz"
)


class AstraMoveEef:
    """Describe and decode named absolute world-frame translations in metres."""

    def __init__(self, dimensions: Sequence[str]) -> None:
        names = frozenset(dimensions)
        if not names or not names <= _TRANSLATION_DIMENSIONS:
            raise ValueError("declare supported Astra translation dimensions")
        self._dimensions = names

    @property
    def tools(self) -> tuple[Mapping[str, Any], ...]:
        return (
            {
                "type": "function",
                "function": {
                    "name": "move_eef",
                    "description": (
                        "Move named grasp-point dimensions to absolute world-frame "
                        "positions in metres. Omitted dimensions hold their observed values."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "targets": {
                                "type": "object",
                                "properties": dict.fromkeys(
                                    sorted(self._dimensions), {"type": "number"}
                                ),
                                "minProperties": 1,
                                "additionalProperties": False,
                            },
                            "note": {"type": "string"},
                        },
                        "required": ["targets", "note"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def parse(self, action: Any) -> dict[str, float]:
        if (
            not isinstance(action, Mapping)
            or set(action) != {"tool", "arguments"}
            or action["tool"] != "move_eef"
        ):
            raise ValueError("expected one move_eef action")
        arguments = action["arguments"]
        if (
            not isinstance(arguments, Mapping)
            or set(arguments) != {"targets", "note"}
            or not isinstance(arguments["note"], str)
        ):
            raise ValueError("move_eef requires targets and a note")
        targets = arguments["targets"]
        if (
            not isinstance(targets, Mapping)
            or not targets
            or not targets.keys() <= self._dimensions
        ):
            raise ValueError("move_eef targets contain unsupported dimensions")
        positions = {}
        for name, value in targets.items():
            if type(value) not in (float, int):
                raise ValueError("move_eef target must be numeric")
            try:
                number = float(value)
            except OverflowError:
                raise ValueError("move_eef target must be finite") from None
            if not math.isfinite(number):
                raise ValueError("move_eef target must be finite")
            positions[name] = number
        return positions


class AstraMoveEefModel:
    """Expose the Astra gateway behind a local one-observation model lifecycle."""

    def __init__(self, provider: PolicyProvider, action: AstraMoveEef) -> None:
        self._provider = provider
        self._action = action
        self._obs: dict[str, Any] | None = None

    def reset(self) -> None:
        self._obs = None

    def update_obs(self, obs: dict[str, Any]) -> None:
        self._obs = obs

    def get_action(self) -> dict[str, Any]:
        from PIL import Image

        obs, self._obs = self._obs, None
        if obs is None:
            raise ValueError("Astra requires a fresh observation")
        images = []
        for camera in obs["vision"].values():
            with io.BytesIO() as buffer:
                Image.fromarray(camera["color"], mode="RGB").save(buffer, format="PNG")
                images.append(
                    "data:image/png;base64,"
                    + base64.b64encode(buffer.getvalue()).decode("ascii")
                )
        if not images:
            raise ValueError("Astra requires at least one RGB image")
        decision = self._provider.decide(
            instructions=obs["instruction"],
            observation={"state": obs["state"], "extra_info": obs["extra_info"]},
            tools=self._action.tools,
            image_data_urls=images,
        )
        if decision.refusal or len(decision.calls) != 1:
            raise ValueError("Astra did not return one move_eef proposal")
        call = decision.calls[0]
        return {"tool": call.name, "arguments": dict(call.arguments)}
