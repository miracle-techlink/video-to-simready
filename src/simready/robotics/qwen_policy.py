"""Qwen policy provider: typed tool-call parsing only, never execution.

The provider is the replaceable model boundary in the robot policy stack.
It accepts caller-supplied observation/history and JSON tool schemas, sends
one OpenAI-compatible chat-completions request to a configured Qwen
endpoint, and parses the response into typed :class:`ToolCall` records.
It never executes a tool, never imports or drives a ``RobotAdapter``, and
never places credential values in code, logs, or exception messages.

Environment resolution is by configurable variable *names*; values live in
the worker environment. Configuration errors and API failures fail closed.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class QwenPolicyError(Exception):
    """Base class for Qwen policy provider failures."""


class QwenConfigError(QwenPolicyError):
    """A required environment variable is missing or malformed."""


class QwenTimeoutError(QwenPolicyError, TimeoutError):
    """The Qwen endpoint did not answer within the configured timeout."""


class QwenApiError(QwenPolicyError):
    """The Qwen endpoint returned an HTTP error status.

    The message carries only the numeric status and a sanitized reason;
    request headers and credential values are never included.
    """

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        message = f"Qwen API request failed with HTTP status {status}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class QwenResponseError(QwenPolicyError):
    """The response envelope or a tool-call payload is malformed."""


class QwenUnknownToolError(QwenPolicyError):
    """The model named a tool that the caller did not supply."""


@dataclass(frozen=True)
class ToolCall:
    """One parsed tool invocation requested by the model; data only."""

    name: str
    arguments: Mapping[str, Any]
    call_id: str | None = None
    index: int = 0


@dataclass(frozen=True)
class PolicyDecision:
    """Parsed model output. ``refusal`` is true when no tool was called.

    A decision is always inert data: executing the calls is the caller's
    responsibility, behind the deterministic validation layer.
    """

    calls: tuple[ToolCall, ...] = ()
    refusal: bool = False
    finish_reason: str | None = None
    usage: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration by environment variable names, not values.

    Default names follow the ``QWEN_*`` convention; deployments may pass
    their own names (for example a secrets-manager injection scheme).
    """

    base_url_env: str = "QWEN_BASE_URL"
    model_env: str = "QWEN_MODEL"
    api_key_env: str = "QWEN_API_KEY"
    request_timeout_s: float = 30.0
    max_tool_calls: int = 4

    def __post_init__(self) -> None:
        if not self.base_url_env or not self.model_env or not self.api_key_env:
            raise ValueError("env variable names must be nonempty")
        if not isinstance(self.request_timeout_s, (int, float)):
            raise ValueError("request_timeout_s must be a number")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")


Transport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]
]
"""Sends ``(url, headers, body, timeout_s)`` and returns the JSON response."""


class QwenPolicyProvider:
    """Client and parser for a Qwen chat-completions tool-call endpoint.

    The provider owns no tool definitions: callers pass generic
    OpenAI-compatible function schemas per request. Responses are parsed
    into :class:`PolicyDecision` records or a typed error; nothing is
    executed and no adapter module is ever imported.
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.config = config if config is not None else PolicyConfig()
        self._transport = transport if transport is not None else _urllib_post_json

    def decide(
        self,
        *,
        instructions: str,
        observation: Mapping[str, Any],
        tools: Sequence[Mapping[str, Any]],
        history: Sequence[Mapping[str, Any]] = (),
        image_data_urls: Sequence[str] = (),
    ) -> PolicyDecision:
        """Request and parse one decision from the Qwen endpoint.

        Raises the typed error hierarchy on any failure; never returns a
        fabricated tool call and never executes anything.
        """
        if not instructions:
            raise ValueError("instructions must be nonempty")
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping")
        for message in history:
            if not isinstance(message, Mapping):
                raise ValueError("history entries must be mappings")
        known_tools = _validate_tools(tools)
        image_blocks = _image_content_blocks(image_data_urls)

        observation_text = json.dumps(observation, default=str, sort_keys=True)
        user_content: str | list[dict[str, Any]] = observation_text
        if image_blocks:
            user_content = [
                {"type": "text", "text": observation_text},
                *image_blocks,
            ]

        base_url, model, api_key = self._resolve_environment()
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": instructions,
                },
                *history,
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "tools": list(tools),
            "tool_choice": "auto",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._transport(
                base_url, headers, body, self.config.request_timeout_s
            )
        except QwenPolicyError:
            raise
        except TimeoutError as error:
            raise QwenTimeoutError("Qwen endpoint timed out") from error
        return _parse_response(response, known_tools, self.config.max_tool_calls)

    def _resolve_environment(self) -> tuple[str, str, str]:
        """Read endpoint, model, and key from the configured env names.

        Errors mention only variable names, never values.
        """
        config = self.config
        base_url = os.environ.get(config.base_url_env, "").strip()
        if not base_url:
            raise QwenConfigError(
                f"Set {config.base_url_env} in the worker environment"
            )
        model = os.environ.get(config.model_env, "").strip()
        if not model:
            raise QwenConfigError(f"Set {config.model_env} in the worker environment")
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise QwenConfigError(f"Set {config.api_key_env} in the worker environment")
        if not base_url.startswith(("https://", "http://")):
            raise QwenConfigError(f"{config.base_url_env} must be an http(s) URL")
        return base_url, model, api_key


_MAX_IMAGE_DATA_URLS = 4
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_DATA_IMAGE_PATTERN = re.compile(
    r"^data:image/(?:png|jpeg|webp);base64,([A-Za-z0-9+/]*={0,2})$"
)


def _image_content_blocks(image_data_urls: Sequence[str]) -> list[dict[str, Any]]:
    """Build OpenAI-compatible vision content from caller-provided data URLs.

    The caller reads/encodes camera frames; this provider never follows local
    paths or remote URLs. That keeps filesystem access and image acquisition
    outside the credential-bearing API client.
    """
    if len(image_data_urls) > _MAX_IMAGE_DATA_URLS:
        raise ValueError(f"at most {_MAX_IMAGE_DATA_URLS} images may be sent")
    blocks: list[dict[str, Any]] = []
    for image in image_data_urls:
        if not isinstance(image, str):
            raise ValueError("image observations must be base64 data URLs")
        match = _DATA_IMAGE_PATTERN.fullmatch(image)
        if not match:
            raise ValueError(
                "image observations must use PNG/JPEG/WEBP base64 data URLs"
            )
        try:
            decoded = base64.b64decode(match.group(1), validate=True)
        except ValueError as error:
            raise ValueError("image data URL has invalid base64") from error
        if not decoded or len(decoded) > _MAX_IMAGE_BYTES:
            raise ValueError("image data URL is empty or exceeds the 10 MiB limit")
        blocks.append({"type": "image_url", "image_url": {"url": image}})
    return blocks


def _urllib_post_json(
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    """Stdlib JSON POST transport; converts errors to the typed hierarchy."""
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise QwenApiError(error.code, str(error.reason)) from error
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", None)
        if isinstance(reason, TimeoutError) or "timeout" in str(reason).lower():
            raise QwenTimeoutError("Qwen endpoint timed out") from error
        raise QwenApiError(0, "connection failed") from error
    except TimeoutError as error:
        raise QwenTimeoutError("Qwen endpoint timed out") from error
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QwenResponseError("response body is not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise QwenResponseError("response body is not a JSON object")
    return parsed


def _validate_tools(tools: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Check caller-supplied OpenAI-compatible function schemas.

    Returns a name-to-function mapping for response validation; raises
    ``ValueError`` for malformed schemas before any request is sent.
    """
    if not tools:
        raise ValueError("tools must declare at least one function")
    known: dict[str, Mapping[str, Any]] = {}
    for entry in tools:
        if not isinstance(entry, Mapping) or entry.get("type") != "function":
            raise ValueError("each tool entry must be of type 'function'")
        function = entry.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("each tool entry needs a 'function' mapping")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool function names must be nonempty strings")
        if name in known:
            raise ValueError(f"duplicate tool function name: {name!r}")
        known[name] = function
    return known


def _parse_response(
    response: Mapping[str, Any],
    known_tools: Mapping[str, Mapping[str, Any]],
    max_tool_calls: int,
) -> PolicyDecision:
    """Parse a chat-completions response into a decision; fail closed."""
    choices = response.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
    ):
        raise QwenResponseError("response has no choices list")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise QwenResponseError("first choice is not a mapping")
    finish_reason = first.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise QwenResponseError("finish_reason must be a string or null")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise QwenResponseError("choice message must be a mapping")
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise QwenResponseError("message tool_calls must be a list")
    if len(raw_calls) > max_tool_calls:
        raise QwenResponseError(
            f"model requested {len(raw_calls)} tool calls; at most "
            f"{max_tool_calls} are accepted"
        )
    parsed: list[ToolCall] = []
    for index, entry in enumerate(raw_calls):
        if not isinstance(entry, Mapping):
            raise QwenResponseError("tool call entry must be a mapping")
        function = entry.get("function")
        if not isinstance(function, Mapping):
            raise QwenResponseError("tool call needs a 'function' mapping")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise QwenResponseError("tool call function name must be nonempty")
        if name not in known_tools:
            raise QwenUnknownToolError(f"model requested unsupplied tool {name!r}")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise QwenResponseError(
                    f"tool {name!r} arguments are not valid JSON"
                ) from error
        if not isinstance(arguments, Mapping):
            raise QwenResponseError(f"tool {name!r} arguments must be a JSON object")
        call_id = entry.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise QwenResponseError("tool call id must be a string or null")
        parsed.append(
            ToolCall(
                name=name,
                arguments=dict(arguments),
                call_id=call_id,
                index=index,
            )
        )
    usage = response.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise QwenResponseError("usage must be a mapping or null")
    refusal = not parsed
    return PolicyDecision(
        calls=tuple(parsed),
        refusal=refusal,
        finish_reason=finish_reason,
        usage=usage,
    )
