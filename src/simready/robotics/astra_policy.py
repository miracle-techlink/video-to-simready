"""Astra Messages provider using strict JSON tool-call output.

The local Astra gateway accepts text and image input, but native Messages
``tools`` requests may route to a fallback model. This provider therefore
includes the caller's allow-listed function schema in the system instruction,
requests one JSON object as text, and converts it into the common inert
``PolicyDecision`` contract. Model identity and the JSON envelope are checked
before the shared policy parser can pass anything to ``PolicyAdapterLoop``.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from simready.robotics.gpt56_policy import (
    _http_detail,
    _is_timeout,
    _to_messages_payload,
)
from simready.robotics.qwen_policy import (
    PolicyConfig,
    PolicyDecision,
    QwenApiError,
    QwenConfigError,
    QwenPolicyProvider,
    QwenResponseError,
    QwenTimeoutError,
)


@dataclass(frozen=True)
class AstraPolicyConfig:
    """Environment names and limits for one Astra Messages route.

    ``expected_model_env`` must contain the exact model identity returned by
    the gateway (for example ``gpt-6-astra-2026-09-03``), which can differ from
    its request alias (for example ``ias/gpt-6-astra``).
    """

    base_url_env: str = "ASTRA_MESSAGES_BASE_URL"
    api_key_env: str = "ASTRA_MESSAGES_API_KEY"
    model_env: str = "ASTRA_MESSAGES_MODEL"
    expected_model_env: str = "ASTRA_MESSAGES_EXPECTED_MODEL"
    anthropic_version: str = "2023-06-01"
    request_timeout_s: float = 60.0
    max_tool_calls: int = 1
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not all(
            (
                self.base_url_env,
                self.api_key_env,
                self.model_env,
                self.expected_model_env,
                self.anthropic_version,
            )
        ):
            raise ValueError("configuration names and version must be nonempty")
        if (
            not isinstance(self.request_timeout_s, (int, float))
            or isinstance(self.request_timeout_s, bool)
            or not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be finite and positive")
        if (
            not isinstance(self.max_tool_calls, int)
            or isinstance(self.max_tool_calls, bool)
            or self.max_tool_calls != 1
        ):
            raise ValueError("Astra JSON policy mode permits exactly one tool call")
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be positive")


class AstraPolicyProvider:
    """Astra inference normalized to the shared inert policy interface."""

    def __init__(self, config: AstraPolicyConfig | None = None) -> None:
        self.config = config or AstraPolicyConfig()
        shared_config = PolicyConfig(
            base_url_env=self.config.base_url_env,
            model_env=self.config.model_env,
            api_key_env=self.config.api_key_env,
            request_timeout_s=self.config.request_timeout_s,
            max_tool_calls=self.config.max_tool_calls,
        )
        self._provider = QwenPolicyProvider(
            shared_config,
            transport=_AstraJsonMessagesTransport(self.config),
        )

    def decide(
        self,
        *,
        instructions: str,
        observation: Mapping[str, Any],
        tools: Sequence[Mapping[str, Any]],
        history: Sequence[Mapping[str, Any]] = (),
        image_data_urls: Sequence[str] = (),
    ) -> PolicyDecision:
        """Infer one inert JSON tool call; never executes or dispatches it."""
        return self._provider.decide(
            instructions=instructions,
            observation=observation,
            tools=tools,
            history=history,
            image_data_urls=image_data_urls,
        )


class _AstraJsonMessagesTransport:
    def __init__(self, config: AstraPolicyConfig) -> None:
        self.config = config

    def __call__(
        self,
        _url: str,
        _headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        base_url, api_key, model, expected_model = self._resolve_environment()
        if body.get("model") != model:
            raise QwenResponseError("Astra request model does not match configuration")
        try:
            payload = _to_messages_payload(body, self.config.max_tokens)
        except (KeyError, TypeError, ValueError) as error:
            raise QwenResponseError(
                "Astra request cannot be converted to Messages"
            ) from error

        tools = payload.pop("tools")
        payload.pop("tool_choice", None)
        schema_text = json.dumps(tools, ensure_ascii=False, sort_keys=True)
        output_contract = (
            "\n\nAstra policy output contract:\n"
            "Return exactly one JSON object as plain text with exactly these keys: "
            '"tool" and "arguments". The tool value must be one name from the '
            "allow-listed schema below, and arguments must be an object matching "
            "that schema. Do not return markdown, explanations, multiple calls, "
            "or any other keys. This is inert data; never execute a tool.\n"
            "Allow-listed tool schemas:\n" + schema_text
        )
        payload["system"] = str(payload.get("system", "")) + output_contract
        try:
            encoded = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise QwenResponseError(
                "Astra Messages request is not serializable"
            ) from error

        request = urllib.request.Request(
            base_url,
            data=encoded,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self.config.anthropic_version,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise QwenApiError(error.code, _http_detail(error.code)) from None
        except urllib.error.URLError as error:
            if _is_timeout(error.reason):
                raise QwenTimeoutError("Astra Messages request timed out") from None
            raise QwenApiError(0, "connection failed") from None
        except TimeoutError:
            raise QwenTimeoutError("Astra Messages request timed out") from None
        try:
            response_body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QwenResponseError(
                "Astra Messages response is not valid JSON"
            ) from error
        if not isinstance(response_body, Mapping):
            raise QwenResponseError("Astra Messages response must be an object")
        if response_body.get("model") != expected_model:
            raise QwenResponseError(
                "Astra response model identity does not match configuration"
            )
        return self._to_tool_response(response_body, tools)

    def _resolve_environment(self) -> tuple[str, str, str, str]:
        values = [
            os.environ.get(name, "").strip()
            for name in (
                self.config.base_url_env,
                self.config.api_key_env,
                self.config.model_env,
                self.config.expected_model_env,
            )
        ]
        if not all(values):
            missing = next(
                name
                for name, value in zip(
                    (
                        self.config.base_url_env,
                        self.config.api_key_env,
                        self.config.model_env,
                        self.config.expected_model_env,
                    ),
                    values,
                    strict=True,
                )
                if not value
            )
            raise QwenConfigError(f"Set {missing} in the worker environment")
        base_url, api_key, model, expected_model = values
        if not base_url.startswith(("http://", "https://")):
            raise QwenConfigError(f"{self.config.base_url_env} must be an http(s) URL")
        return base_url, api_key, model, expected_model

    @staticmethod
    def _to_tool_response(
        response: Mapping[str, Any], allowed_tools: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        if response.get("stop_reason") != "end_turn":
            raise QwenResponseError("Astra response did not finish the JSON action")
        content = response.get("content")
        if not isinstance(content, list):
            raise QwenResponseError("Astra response content must be a list")
        if any(
            isinstance(block, Mapping) and block.get("type") == "tool_use"
            for block in content
        ):
            raise QwenResponseError(
                "Astra JSON mode returned an unexpected native tool call"
            )
        text_blocks = [
            block.get("text")
            for block in content
            if isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if len(text_blocks) != 1:
            raise QwenResponseError(
                "Astra response must contain exactly one text block"
            )
        try:
            action = json.loads(
                text_blocks[0],
                object_pairs_hook=_unique_object,
                parse_constant=_reject_non_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise QwenResponseError("Astra action text is not strict JSON") from error
        if not isinstance(action, Mapping) or set(action) != {"tool", "arguments"}:
            raise QwenResponseError("Astra action must contain only tool and arguments")
        name, arguments = action["tool"], action["arguments"]
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise QwenResponseError(
                "Astra action tool and arguments have invalid types"
            )
        names: set[str] = set()
        for entry in allowed_tools:
            if not isinstance(entry, Mapping):
                continue
            function = entry.get("function")
            name = (
                function.get("name")
                if isinstance(function, Mapping)
                else entry.get("name")
            )
            if isinstance(name, str):
                names.add(name)
        if name not in names:
            raise QwenResponseError(
                "Astra action requested a tool outside the allow-list"
            )
        result: dict[str, Any] = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "astra-json-action-0",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ]
                    },
                }
            ]
        }
        if isinstance(response.get("usage"), Mapping):
            result["usage"] = response["usage"]
        return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")
