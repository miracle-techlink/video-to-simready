"""CPU-only tests for Astra's text-JSON Messages policy adapter."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from simready.robotics.astra_policy import AstraPolicyConfig, AstraPolicyProvider
from simready.robotics.qwen_policy import QwenResponseError

KEY = "astra-test-key"
REQUEST_MODEL = "ias/gpt-6-astra"
RESPONSE_MODEL = "gpt-6-astra-2026-09-03"
IMAGE = "data:image/png;base64,YWJj"
JOINTS = [f"joint{index}" for index in range(1, 7)]
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "set_arm_joint_targets",
            "description": "Return bounded arm targets as data only.",
            "parameters": {
                "type": "object",
                "properties": {name: {"type": "number"} for name in JOINTS},
                "required": JOINTS,
                "additionalProperties": False,
            },
        },
    },
)
ACTION = {
    "tool": "set_arm_joint_targets",
    "arguments": {name: 0.01 for name in JOINTS},
}


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


def _response(text=None, *, model=RESPONSE_MODEL, stop_reason="end_turn"):
    return {
        "model": model,
        "stop_reason": stop_reason,
        "content": (
            [{"type": "text", "text": json.dumps(ACTION)}] if text is None else text
        ),
        "usage": {"input_tokens": 100, "output_tokens": 30},
    }


def _provider(monkeypatch):
    monkeypatch.setenv("ASTRA_MESSAGES_BASE_URL", "https://gateway.test/v1/messages")
    monkeypatch.setenv("ASTRA_MESSAGES_API_KEY", KEY)
    monkeypatch.setenv("ASTRA_MESSAGES_MODEL", REQUEST_MODEL)
    monkeypatch.setenv("ASTRA_MESSAGES_EXPECTED_MODEL", RESPONSE_MODEL)
    return AstraPolicyProvider()


def test_astra_policy_config_restricts_to_one_action():
    with pytest.raises(ValueError, match="exactly one tool call"):
        AstraPolicyConfig(max_tool_calls=2)


@pytest.mark.parametrize(
    "kwargs", [{"request_timeout_s": float("nan")}, {"max_tokens": True}]
)
def test_astra_policy_config_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        AstraPolicyConfig(**kwargs)


def test_astra_json_transport_preserves_image_schema_and_maps_inert_call(
    monkeypatch,
):
    provider = _provider(monkeypatch)
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request, timeout))
        return _Response(_response())

    with patch("urllib.request.urlopen", fake_urlopen):
        decision = provider.decide(
            instructions="Choose one bounded joint action.",
            observation={"joint_positions": {name: 0.0 for name in JOINTS}},
            image_data_urls=[IMAGE],
            tools=TOOLS,
        )

    assert len(decision.calls) == 1
    assert decision.calls[0].name == "set_arm_joint_targets"
    assert decision.calls[0].arguments == ACTION["arguments"]
    assert decision.usage == {"input_tokens": 100, "output_tokens": 30}
    request, timeout = seen[0]
    assert timeout == 60.0
    assert request.full_url == "https://gateway.test/v1/messages"
    assert request.get_header("X-api-key") == KEY
    body = json.loads(request.data)
    assert body["model"] == REQUEST_MODEL
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "set_arm_joint_targets" in body["system"]
    assert "additionalProperties" in body["system"]
    assert body["messages"][-1]["content"][1]["source"]["data"] == "YWJj"


def test_astra_fallback_model_is_rejected(monkeypatch):
    provider = _provider(monkeypatch)
    with patch(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(_response(model="zhipu/glm-5.3")),
    ):
        with pytest.raises(QwenResponseError, match="identity"):
            provider.decide(instructions="x", observation={}, tools=TOOLS)


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        '{"tool":"set_arm_joint_targets","tool":"set_arm_joint_targets","arguments":{}}',
        '{"tool":"set_arm_joint_targets","arguments":{"joint1":NaN}}',
        json.dumps({"tool": "set_arm_joint_targets", "arguments": {}, "extra": 1}),
        json.dumps({"tool": "unknown", "arguments": {}}),
        json.dumps({"tool": "set_arm_joint_targets", "arguments": []}),
    ],
)
def test_astra_rejects_malformed_or_unallowlisted_action(monkeypatch, text):
    provider = _provider(monkeypatch)
    response = _response(text=[{"type": "text", "text": text}])
    with patch("urllib.request.urlopen", lambda *args, **kwargs: _Response(response)):
        with pytest.raises(QwenResponseError):
            provider.decide(instructions="x", observation={}, tools=TOOLS)


@pytest.mark.parametrize("stop_reason", ["max_tokens", "tool_use", None])
def test_astra_requires_a_complete_plain_text_response(monkeypatch, stop_reason):
    provider = _provider(monkeypatch)
    response = _response(stop_reason=stop_reason)
    with patch("urllib.request.urlopen", lambda *args, **kwargs: _Response(response)):
        with pytest.raises(QwenResponseError, match="finish"):
            provider.decide(instructions="x", observation={}, tools=TOOLS)


def test_astra_native_tool_call_is_rejected_in_json_mode(monkeypatch):
    provider = _provider(monkeypatch)
    response = _response(
        text=[
            {
                "type": "tool_use",
                "id": "fallback",
                "name": "set_arm_joint_targets",
                "input": ACTION["arguments"],
            }
        ],
        stop_reason="end_turn",
    )
    with patch("urllib.request.urlopen", lambda *args, **kwargs: _Response(response)):
        with pytest.raises(QwenResponseError, match="native tool"):
            provider.decide(instructions="x", observation={}, tools=TOOLS)


def test_astra_response_model_is_required(monkeypatch):
    provider = _provider(monkeypatch)
    monkeypatch.setenv("ASTRA_MESSAGES_EXPECTED_MODEL", "gpt-6-astra-2026-09-04")
    with patch(
        "urllib.request.urlopen", lambda *args, **kwargs: _Response(_response())
    ):
        with pytest.raises(QwenResponseError, match="identity"):
            provider.decide(instructions="x", observation={}, tools=TOOLS)
