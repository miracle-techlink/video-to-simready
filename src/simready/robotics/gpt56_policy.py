"""GPT-5.6 Messages transport for the inert Qwen policy-provider boundary.

This module adapts QwenPolicyProvider's OpenAI-compatible request envelope to a
configured Anthropic Messages-compatible endpoint. It neither executes a tool
nor imports an adapter. The configured model identity is checked on every
successful response so a gateway fallback is rejected before parsed data can
reach PolicyAdapterLoop.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from simready.robotics.qwen_policy import (
    QwenApiError,
    QwenConfigError,
    QwenResponseError,
    QwenTimeoutError,
)


@dataclass(frozen=True)
class Gpt56MessagesConfig:
    """Names of worker environment variables for one explicit Messages route."""

    base_url_env: str = "GPT56_MESSAGES_BASE_URL"
    api_key_env: str = "GPT56_MESSAGES_API_KEY"
    model_env: str = "GPT56_MESSAGES_MODEL"
    expected_model_env: str = "GPT56_MESSAGES_EXPECTED_MODEL"
    anthropic_version: str = "2023-06-01"
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
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


class Gpt56MessagesTransport:
    """Adapt one policy request to Messages API, with strict model identity."""

    def __init__(self, config: Gpt56MessagesConfig | None = None) -> None:
        self.config = config if config is not None else Gpt56MessagesConfig()

    def __call__(
        self,
        _url: str,
        _headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """Send adapted request and return a chat-completions-compatible response."""
        base_url, api_key, expected_model = self._resolve_environment()
        configured_model = os.environ.get(self.config.model_env, "").strip()
        requested_model = body.get("model")
        if not isinstance(requested_model, str) or requested_model != configured_model:
            raise QwenResponseError(
                "request model identity does not match the configured GPT-5.6 model"
            )
        try:
            payload = _to_messages_payload(body, self.config.max_tokens)
        except (KeyError, TypeError, ValueError) as error:
            raise QwenResponseError(
                "policy request cannot be converted to Messages API"
            ) from error
        try:
            encoded = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise QwenResponseError(
                "Messages request is not JSON serializable"
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
            # The upstream exception (headers, URL, reason text) can carry the
            # API key; never chain it into the raised error, not even as the
            # implicit traceback cause.
            raise QwenApiError(error.code, _http_detail(error.code)) from None
        except urllib.error.URLError as error:
            if _is_timeout(error.reason):
                raise QwenTimeoutError("GPT-5.6 Messages endpoint timed out") from None
            raise QwenApiError(0, "connection failed") from None
        except TimeoutError:
            raise QwenTimeoutError("GPT-5.6 Messages endpoint timed out") from None
        try:
            response_body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QwenResponseError(
                "Messages response body is not valid JSON"
            ) from error
        if not isinstance(response_body, Mapping):
            raise QwenResponseError("Messages response body is not a JSON object")
        actual_model = response_body.get("model")
        if not isinstance(actual_model, str) or actual_model != expected_model:
            raise QwenResponseError(
                "GPT-5.6 response model identity does not match configuration"
            )
        return _to_chat_response(response_body)

    def _resolve_environment(self) -> tuple[str, str, str]:
        values: list[str] = []
        for name in (
            self.config.base_url_env,
            self.config.api_key_env,
            self.config.expected_model_env,
        ):
            value = os.environ.get(name, "").strip()
            if not value:
                raise QwenConfigError(f"Set {name} in the worker environment")
            values.append(value)
        base_url, api_key, expected_model = values
        if not base_url.startswith(("https://", "http://")):
            raise QwenConfigError(f"{self.config.base_url_env} must be an http(s) URL")
        configured_model = os.environ.get(self.config.model_env, "").strip()
        if not configured_model:
            raise QwenConfigError(
                f"Set {self.config.model_env} in the worker environment"
            )
        if not self._is_consistent_model(configured_model, expected_model):
            raise QwenConfigError(
                "configured GPT-5.6 model must match the expected model identity"
                " (same name, optionally with a dated snapshot suffix)"
            )
        return base_url, api_key, expected_model

    @staticmethod
    def _is_consistent_model(configured: str, expected: str) -> bool:
        return configured == expected or expected.startswith(configured + "-")


def _to_messages_payload(body: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
    """Convert Qwen-provider request body while preserving images and schemas."""
    messages = body["messages"]
    tools = body["tools"]
    model = body["model"]
    if (
        not isinstance(messages, list)
        or not isinstance(tools, list)
        or not isinstance(model, str)
    ):
        raise ValueError("invalid provider request envelope")
    system: list[str] = []
    output_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("message must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if not isinstance(content, str):
                raise ValueError("system content must be text")
            system.append(content)
            continue
        if role not in ("user", "assistant"):
            raise ValueError("unsupported Messages role")
        output_messages.append({"role": role, "content": _to_messages_content(content)})
    output_tools: list[dict[str, Any]] = []
    for entry in tools:
        if not isinstance(entry, Mapping) or entry.get("type") != "function":
            raise ValueError("invalid tool entry")
        function = entry.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("invalid function entry")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            raise ValueError("invalid function schema")
        output_tools.append(
            {
                "name": name,
                "description": function.get("description", ""),
                "input_schema": dict(parameters),
            }
        )
    return {
        "model": model,
        "system": "\n".join(system),
        "messages": output_messages,
        "tools": output_tools,
        "tool_choice": {"type": "auto"},
        "max_tokens": max_tokens,
    }


def _to_messages_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("content must be text or blocks")
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise ValueError("content block must be a mapping")
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            blocks.append({"type": "text", "text": block["text"]})
            continue
        image_url = block.get("image_url")
        if block.get("type") != "image_url" or not isinstance(image_url, Mapping):
            raise ValueError("unsupported content block")
        url = image_url.get("url")
        if not isinstance(url, str) or not url.startswith("data:image/"):
            raise ValueError("image URL must be a data URL")
        prefix, separator, data = url.partition(",")
        media_type = prefix.removeprefix("data:").removesuffix(";base64")
        if not separator or not media_type.startswith("image/"):
            raise ValueError("image URL must include media type and base64 data")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        )
    return blocks


def _to_chat_response(response: Mapping[str, Any]) -> Mapping[str, Any]:
    content = response.get("content")
    if not isinstance(content, list):
        raise QwenResponseError("Messages response content must be a list")
    calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise QwenResponseError("Messages content block must be a mapping")
        if block.get("type") != "tool_use":
            continue
        name, call_id, arguments = (
            block.get("name"),
            block.get("id"),
            block.get("input"),
        )
        if (
            not isinstance(name, str)
            or not isinstance(call_id, str)
            or not isinstance(arguments, Mapping)
        ):
            raise QwenResponseError("Messages tool-use block is malformed")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        )
    stop_reason = response.get("stop_reason")
    if not isinstance(stop_reason, str) or stop_reason not in ("tool_use", "end_turn"):
        raise QwenResponseError(
            f"Messages response stopped with unsupported reason {stop_reason!r}"
        )
    if calls and stop_reason != "tool_use":
        raise QwenResponseError(
            "Messages response contains tool calls but stopped without completing them"
        )
    finish_reason = "tool_calls" if calls else "stop"
    result: dict[str, Any] = {
        "choices": [{"finish_reason": finish_reason, "message": {"tool_calls": calls}}],
    }
    if isinstance(response.get("usage"), Mapping):
        result["usage"] = response["usage"]
    return result


def _is_timeout(reason: object) -> bool:
    return isinstance(reason, TimeoutError) or "timeout" in str(reason).lower()


_HTTP_STATUS_DETAILS = {
    401: "unauthorized",
    403: "forbidden",
    404: "not found",
    429: "rate limited",
    500: "server error",
    502: "bad gateway",
    503: "service unavailable",
    504: "gateway timeout",
}


def _http_detail(status: int) -> str:
    """Map a numeric status to a fixed reason; upstream text may carry secrets."""
    return _HTTP_STATUS_DETAILS.get(status, "request failed")
