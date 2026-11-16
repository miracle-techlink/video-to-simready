# Qwen policy provider (Module A)

## Scope

`simready.robotics.qwen_policy.QwenPolicyProvider` is a narrow, replaceable
Qwen chat-completions boundary. It accepts an observation, optional message
history, and **caller-supplied** OpenAI-compatible function schemas. It returns
only `PolicyDecision` / `ToolCall` data records.

It deliberately does **not** execute tools, import an adapter, connect to a
robot, validate a robot command, or make an actuation decision. Deterministic
policy/safety code must validate and execute any accepted action separately.

## Configuration and failures

The default environment-variable names are `QWEN_BASE_URL`, `QWEN_MODEL`, and
`QWEN_API_KEY`. Their **names** are configurable through `PolicyConfig`; the
provider reads values only at request time. It has no logging and does not put
credential values into source, decisions, or its error messages.

The implementation uses stdlib `urllib`, so it adds no production dependency.
It fails closed with typed errors for missing configuration, HTTP/API failures,
timeouts, malformed response/tool arguments, and model-selected tools absent
from the caller schema. A no-tool model response is represented as
`PolicyDecision(refusal=True)`.

## Verification

Mocked CPU tests cover valid typed calls, malformed arguments, unknown tools,
timeouts, API errors, refusals, configurable env names, input/schema failures,
and the no-execution/no-adapter invariant.

- `PYTHONPATH=src python -m pytest tests/robotics -q`: PASS (21 tests)
- `PYTHONPATH=src python -m pytest -q -m "not real_model and not gpu"`: PASS
  (1379 passed, 1 skipped, 20 deselected)
- `ruff check src/simready tests examples scripts deploy`: PASS
- `ruff format --check src/simready tests examples scripts deploy`: PASS
  (with one pre-existing invalid-noqa warning)
- `python scripts/check_architecture.py`: PASS

## Server smoke

Status: **NOT_RUN**. The server/worker presence-only probe found no configured
`QWEN_*` or `DASHSCOPE_*` endpoint, model, or credential variable names. No
values were read, no request was sent, no image was submitted, no GPU was used,
and no tool or robot action occurred. Therefore model ID, image-input support,
function-call support, and latency are all not measured.

Redacted evidence: `/home/nvidia/algotithm/.simready-collaboration/robot-adapters-20261116/policy-qwen/env-presence-probe.json`.
