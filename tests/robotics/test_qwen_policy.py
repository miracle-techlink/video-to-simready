"""Mocked Qwen policy provider tests; never a real network call.

The fake transport stands in for the HTTP seam, and the sentinel module
proves the provider parses typed tool calls without executing anything or
importing robot adapters.
"""

import sys

import pytest

from simready.robotics.qwen_policy import (
    PolicyConfig,
    PolicyDecision,
    QwenApiError,
    QwenConfigError,
    QwenPolicyProvider,
    QwenResponseError,
    QwenTimeoutError,
    QwenUnknownToolError,
    ToolCall,
)

TEST_KEY = "test-only-credential"  # never a real secret
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "move_ee",
            "description": "Move the end effector.",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        },
    },
)
OBSERVATION = {"joint_positions": {"j1": 0.0}, "timestamp_ns": 1}


def _tool_call_response(name, arguments, call_id="call-1", finish_reason="tool_calls"):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class _FakeTransport:
    def __init__(self, response=None, error=None):
        self.seen = []
        self.response = response
        self.error = error

    def __call__(self, url, headers, body, timeout_s):
        self.seen.append(
            {"url": url, "headers": dict(headers), "body": body, "timeout_s": timeout_s}
        )
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.test/v1/chat/completions")
    monkeypatch.setenv("QWEN_MODEL", "qwen-3.8-test")
    monkeypatch.setenv("QWEN_API_KEY", TEST_KEY)


def _provider(transport):
    return QwenPolicyProvider(transport=transport)


def test_valid_tool_call_is_parsed_as_typed_data(env):
    transport = _FakeTransport(_tool_call_response("move_ee", '{"x": 0.25}'))
    decision = _provider(transport).decide(
        instructions="Assist the operator.",
        observation=OBSERVATION,
        tools=TOOLS,
    )
    assert decision == PolicyDecision(
        calls=(
            ToolCall(
                name="move_ee",
                arguments={"x": 0.25},
                call_id="call-1",
                index=0,
            ),
        ),
        refusal=False,
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    request = transport.seen[0]
    assert request["url"] == "https://qwen.test/v1/chat/completions"
    assert request["headers"]["Authorization"] == f"Bearer {TEST_KEY}"
    assert request["body"]["model"] == "qwen-3.8-test"
    assert request["body"]["tool_choice"] == "auto"
    assert request["body"]["tools"] == list(TOOLS)
    assert request["body"]["messages"][0] == {
        "role": "system",
        "content": "Assist the operator.",
    }
    assert '"joint_positions"' in request["body"]["messages"][-1]["content"]
    assert TEST_KEY not in request["body"]["messages"][-1]["content"]


def test_image_observation_is_sent_as_multimodal_content_not_path_text(env):
    transport = _FakeTransport(_tool_call_response("move_ee", '{"x": 0.25}'))
    image = "data:image/png;base64,YWJj"
    _provider(transport).decide(
        instructions="Inspect this camera frame.",
        observation={"joint_positions": {"j1": 0.0}, "rgb_path": "/private/frame.png"},
        image_data_urls=[image],
        tools=TOOLS,
    )
    content = transport.seen[0]["body"]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert '"rgb_path": "/private/frame.png"' in content[0]["text"]
    assert content[1] == {"type": "image_url", "image_url": {"url": image}}


@pytest.mark.parametrize(
    "images",
    [
        ["/tmp/frame.png"],
        ["https://example.test/frame.png"],
        ["data:image/png;base64,%%%"],
        ["data:image/gif;base64,YWJj"],
        ["data:image/png;base64,"],
    ],
)
def test_invalid_image_observation_is_rejected_before_request(env, images):
    transport = _FakeTransport(_tool_call_response("move_ee", '{"x": 0.25}'))
    with pytest.raises(ValueError, match="image"):
        _provider(transport).decide(
            instructions="Inspect this camera frame.",
            observation=OBSERVATION,
            image_data_urls=images,
            tools=TOOLS,
        )
    assert transport.seen == []


def test_image_count_is_bounded(env):
    transport = _FakeTransport(_tool_call_response("move_ee", '{"x": 0.25}'))
    images = ["data:image/png;base64,YWJj"] * 5
    with pytest.raises(ValueError, match="at most 4"):
        _provider(transport).decide(
            instructions="Inspect these camera frames.",
            observation=OBSERVATION,
            image_data_urls=images,
            tools=TOOLS,
        )
    assert transport.seen == []


def test_history_is_forwarded_between_system_and_observation(env):
    transport = _FakeTransport(
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant"}}]}
    )
    history = ({"role": "user", "content": "prior attempt rejected"},)
    _provider(transport).decide(
        instructions="Assist the operator.",
        observation=OBSERVATION,
        tools=TOOLS,
        history=history,
    )
    messages = transport.seen[0]["body"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == history[0]
    assert messages[-1]["role"] == "user"


def test_malformed_arguments_fail_closed(env):
    transport = _FakeTransport(_tool_call_response("move_ee", "{not json"))
    with pytest.raises(QwenResponseError, match="not valid JSON"):
        _provider(transport).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_non_object_arguments_fail_closed(env):
    transport = _FakeTransport(_tool_call_response("move_ee", "[1, 2]"))
    with pytest.raises(QwenResponseError, match="must be a JSON object"):
        _provider(transport).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_unknown_tool_is_rejected(env):
    transport = _FakeTransport(_tool_call_response("fire_lasers", "{}"))
    with pytest.raises(QwenUnknownToolError, match="fire_lasers"):
        _provider(transport).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_timeout_is_typed(env):
    transport = _FakeTransport(error=TimeoutError("timed out"))
    with pytest.raises(QwenTimeoutError):
        _provider(transport).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_api_error_carries_status_without_secrets(env):
    transport = _FakeTransport(error=QwenApiError(500, "internal server error"))
    with pytest.raises(QwenApiError, match="HTTP status 500"):
        _provider(transport).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_refusal_is_a_safe_decision(env):
    transport = _FakeTransport(
        {
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "message": {
                        "role": "assistant",
                        "content": "I cannot help with that.",
                    },
                }
            ]
        }
    )
    decision = _provider(transport).decide(
        instructions="Assist the operator.",
        observation=OBSERVATION,
        tools=TOOLS,
    )
    assert decision.refusal is True
    assert decision.calls == ()
    assert decision.finish_reason == "content_filter"


def test_missing_env_names_are_reported_without_values(monkeypatch):
    for name in ("QWEN_BASE_URL", "QWEN_MODEL", "QWEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    provider = _provider(_FakeTransport())
    with pytest.raises(QwenConfigError, match="QWEN_BASE_URL"):
        provider.decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )
    monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.test")
    with pytest.raises(QwenConfigError, match="QWEN_MODEL"):
        provider.decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )
    monkeypatch.setenv("QWEN_MODEL", "qwen-3.8-test")
    with pytest.raises(QwenConfigError, match="QWEN_API_KEY"):
        provider.decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_configurable_env_names_are_honored(monkeypatch):
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://other.test/v1")
    monkeypatch.setenv("CUSTOM_MODEL_ID", "qwen-3.8-other")
    monkeypatch.setenv("CUSTOM_TOKEN", TEST_KEY)
    transport = _FakeTransport(_tool_call_response("move_ee", "{}"))
    config = PolicyConfig(
        base_url_env="CUSTOM_ENDPOINT",
        model_env="CUSTOM_MODEL_ID",
        api_key_env="CUSTOM_TOKEN",
    )
    QwenPolicyProvider(config, transport=transport).decide(
        instructions="Assist the operator.",
        observation=OBSERVATION,
        tools=TOOLS,
    )
    request = transport.seen[0]
    assert request["url"] == "https://other.test/v1"
    assert request["body"]["model"] == "qwen-3.8-other"
    assert request["headers"]["Authorization"] == f"Bearer {TEST_KEY}"


def test_bad_base_url_is_rejected(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "ftp://not-allowed")
    monkeypatch.setenv("QWEN_MODEL", "qwen-3.8-test")
    monkeypatch.setenv("QWEN_API_KEY", TEST_KEY)
    with pytest.raises(QwenConfigError, match="http"):
        _provider(_FakeTransport()).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_too_many_tool_calls_fail_closed(env):
    calls = [
        {
            "id": f"call-{i}",
            "type": "function",
            "function": {"name": "move_ee", "arguments": "{}"},
        }
        for i in range(5)
    ]
    transport = _FakeTransport(
        {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": calls}}]}
    )
    with pytest.raises(QwenResponseError, match="at most 4"):
        _provider(transport).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_malformed_response_envelope_fails_closed(env):
    transport = _FakeTransport({"error": {"message": "boom"}})
    with pytest.raises(QwenResponseError, match="no choices"):
        _provider(transport).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=TOOLS,
        )


def test_caller_schema_validation_precedes_any_request():
    transport = _FakeTransport(_tool_call_response("move_ee", "{}"))
    with pytest.raises(ValueError, match="type 'function'"):
        _provider(transport).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=({"type": "web_search"},),
        )
    assert transport.seen == []


def test_missing_tools_is_a_caller_error(env):
    with pytest.raises(ValueError, match="at least one function"):
        _provider(_FakeTransport()).decide(
            instructions="Assist the operator.",
            observation=OBSERVATION,
            tools=(),
        )


def test_provider_never_executes_tools_or_imports_adapters(env):
    module = sys.modules["simready.robotics.qwen_policy"]
    public = [name for name in dir(module) if not name.startswith("_")]
    for forbidden in (
        "execute",
        "run_tool",
        "execute_tool",
        "connect",
        "send_joint_position",
        "RobotAdapter",
        "IsaacSimArticulationAdapter",
    ):
        assert forbidden not in public
    source = open(module.__file__).read()
    assert "isaac_sim" not in source
    assert "contracts" not in source
    decision = _provider(_FakeTransport(_tool_call_response("move_ee", "{}"))).decide(
        instructions="Assist the operator.",
        observation=OBSERVATION,
        tools=TOOLS,
    )
    assert isinstance(decision, PolicyDecision)
    assert isinstance(decision.calls[0], ToolCall)
    assert isinstance(decision.calls[0].arguments, dict)
