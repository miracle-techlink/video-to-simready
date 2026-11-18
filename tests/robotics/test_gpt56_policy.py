"""CPU-only conversion tests for the GPT-5.6 Messages policy transport."""

from __future__ import annotations

import io
import json
from email.message import Message
from unittest.mock import patch

import pytest

from simready.robotics.gpt56_policy import Gpt56MessagesConfig, Gpt56MessagesTransport
from simready.robotics.qwen_policy import (
    PolicyConfig,
    QwenApiError,
    QwenConfigError,
    QwenPolicyProvider,
    QwenResponseError,
    QwenTimeoutError,
)

KEY = "test-only-key"
MODEL = "gpt-5.6-luna-2026-07-09"
IMAGE = "data:image/png;base64,YWJj"
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "set_joint_positions",
            "description": "Return data only.",
            "parameters": {
                "type": "object",
                "properties": {"joint1": {"type": "number"}},
                "required": ["joint1"],
                "additionalProperties": False,
            },
        },
    },
)


class _Response:
    status = 200

    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _provider(monkeypatch):
    monkeypatch.setenv("GPT56_MESSAGES_BASE_URL", "https://ias.test/v1/messages")
    monkeypatch.setenv("GPT56_MESSAGES_API_KEY", KEY)
    monkeypatch.setenv("GPT56_MESSAGES_MODEL", MODEL)
    monkeypatch.setenv("GPT56_MESSAGES_EXPECTED_MODEL", MODEL)
    return QwenPolicyProvider(
        PolicyConfig(
            base_url_env="GPT56_MESSAGES_BASE_URL",
            api_key_env="GPT56_MESSAGES_API_KEY",
            model_env="GPT56_MESSAGES_MODEL",
        ),
        transport=Gpt56MessagesTransport(),
    )


def _tool_response(model=MODEL):
    return {
        "model": model,
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "set_joint_positions",
                "input": {"joint1": 0.02},
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def test_messages_transport_preserves_image_and_complete_tool_schema(monkeypatch):
    provider = _provider(monkeypatch)
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request, timeout))
        return _Response(_tool_response())

    with patch("urllib.request.urlopen", fake_urlopen):
        decision = provider.decide(
            instructions="Return one inert joint target.",
            observation={"camera": "front"},
            image_data_urls=[IMAGE],
            tools=TOOLS,
        )

    assert decision.calls[0].arguments == {"joint1": 0.02}
    request, timeout = seen[0]
    assert timeout == 30.0
    assert request.full_url == "https://ias.test/v1/messages"
    assert request.get_header("X-api-key") == KEY
    assert request.get_header("Authorization") is None
    body = json.loads(request.data)
    assert body["model"] == MODEL
    assert body["system"] == "Return one inert joint target."
    assert body["tools"] == [
        {
            "name": "set_joint_positions",
            "description": "Return data only.",
            "input_schema": TOOLS[0]["function"]["parameters"],
        }
    ]
    image = body["messages"][-1]["content"][1]
    assert image == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"},
    }


def test_returned_model_identity_mismatch_fails_before_parse(monkeypatch):
    provider = _provider(monkeypatch)
    with patch(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(_tool_response("glm-5")),
    ):
        with pytest.raises(QwenResponseError, match="identity"):
            provider.decide(instructions="x", observation={}, tools=TOOLS)


def test_configured_and_expected_models_must_match(monkeypatch):
    provider = _provider(monkeypatch)
    monkeypatch.setenv("GPT56_MESSAGES_EXPECTED_MODEL", "other-model")
    with pytest.raises(QwenConfigError, match="must match"):
        provider.decide(instructions="x", observation={}, tools=TOOLS)


def test_dated_snapshot_suffix_is_consistent_with_requested_alias(monkeypatch):
    provider = _provider(monkeypatch)
    monkeypatch.setenv("GPT56_MESSAGES_MODEL", MODEL)  # alias only
    with patch(
        "urllib.request.urlopen", lambda *args, **kwargs: _Response(_tool_response())
    ):
        decision = provider.decide(instructions="x", observation={}, tools=TOOLS)
    assert decision.calls[0].name == "set_joint_positions"


def test_http_errors_are_typed_and_do_not_include_credentials(monkeypatch):
    provider = _provider(monkeypatch)
    headers = Message()
    error = __import__("urllib.error").error.HTTPError(
        "https://ias.test",
        429,
        "unauthorized with secret key sk-test-only-key-value",
        headers,
        io.BytesIO(b"ignored"),
    )
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(QwenApiError) as raised:
            provider.decide(instructions="x", observation={}, tools=TOOLS)
    assert raised.value.status == 429
    assert KEY not in str(raised.value)
    assert (
        str(raised.value)
        == "Qwen API request failed with HTTP status 429: rate limited"
    )


def test_timeout_is_typed(monkeypatch):
    provider = _provider(monkeypatch)
    with patch(
        "urllib.request.urlopen", side_effect=TimeoutError("private endpoint timed out")
    ):
        with pytest.raises(QwenTimeoutError) as raised:
            provider.decide(instructions="x", observation={}, tools=TOOLS)
    assert KEY not in str(raised.value)


def test_malformed_messages_tool_block_fails_closed(monkeypatch):
    provider = _provider(monkeypatch)
    response = _tool_response()
    response["content"][0]["input"] = []
    with patch("urllib.request.urlopen", lambda *args, **kwargs: _Response(response)):
        with pytest.raises(QwenResponseError, match="malformed"):
            provider.decide(instructions="x", observation={}, tools=TOOLS)


@pytest.mark.parametrize("stop_reason", ["max_tokens", "pause_turn", "refusal", None])
def test_truncated_or_unknown_stop_reason_is_rejected_even_with_tool_calls(
    monkeypatch, stop_reason
):
    provider = _provider(monkeypatch)
    response = _tool_response()
    response["stop_reason"] = stop_reason
    with patch("urllib.request.urlopen", lambda *args, **kwargs: _Response(response)):
        with pytest.raises(QwenResponseError, match="stop_reason|unsupported reason"):
            provider.decide(instructions="x", observation={}, tools=TOOLS)


def test_tool_calls_without_tool_use_stop_reason_fail_closed(monkeypatch):
    provider = _provider(monkeypatch)
    response = _tool_response()
    response["stop_reason"] = "end_turn"
    with patch("urllib.request.urlopen", lambda *args, **kwargs: _Response(response)):
        with pytest.raises(QwenResponseError, match="without completing"):
            provider.decide(instructions="x", observation={}, tools=TOOLS)


def test_end_turn_without_tool_calls_is_a_refusal_style_decision(monkeypatch):
    provider = _provider(monkeypatch)
    response = _tool_response()
    response["stop_reason"] = "end_turn"
    response["content"] = [{"type": "text", "text": "Done."}]
    with patch("urllib.request.urlopen", lambda *args, **kwargs: _Response(response)):
        decision = provider.decide(instructions="x", observation={}, tools=TOOLS)
    assert decision.calls == ()
    assert decision.refusal is True
    assert decision.finish_reason == "stop"


def test_request_model_must_match_configured_model(monkeypatch):
    _provider(monkeypatch)
    body = {
        "model": "gpt-5.6-terra-2026-07-09",
        "messages": [],
        "tools": [],
    }
    with pytest.raises(QwenResponseError, match="does not match"):
        Gpt56MessagesTransport()("unused", {}, body, 1.0)


def test_config_rejects_empty_expected_model_name():
    with pytest.raises(ValueError, match="nonempty"):
        Gpt56MessagesConfig(expected_model_env="")
