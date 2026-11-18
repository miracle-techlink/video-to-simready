# Qwen policy provider (Module A)

## Scope

`simready.robotics.qwen_policy.QwenPolicyProvider` is a narrow, replaceable
Qwen chat-completions boundary. It accepts an observation, optional message
history, and **caller-supplied** OpenAI-compatible function schemas. It returns
only `PolicyDecision` / `ToolCall` data records.

It deliberately does **not** execute tools, import an adapter, connect to a
robot, validate a robot command, or make an actuation decision. Deterministic
policy/safety code must validate and execute any accepted action separately.

For visual observations, pass caller-encoded `data:image/png;base64,...`,
`data:image/jpeg;base64,...`, or WebP data URLs through `image_data_urls`.
The provider adds OpenAI-compatible image content blocks to the user message;
it does not open file paths or fetch URLs. A request accepts at most four
images, each no larger than 10 MiB. Whether the configured Qwen model/endpoint
supports image input together with tool calls still requires the server smoke.

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

Mocked CPU tests cover valid typed calls, multimodal image serialization,
invalid image inputs, unknown tools, timeouts, API errors, refusals, configurable
env names, input/schema failures, and the no-execution/no-adapter invariant.

- `PYTHONPATH=src python -m pytest -q tests/robotics/test_qwen_policy.py`:
  PASS (23 tests after multimodal review fix)
- The module author's initial commit reported a full CPU test run (1379 passed,
  1 skipped, 20 deselected) before the multimodal follow-up; the full suite has
  not yet been rerun after that follow-up.

A Messages-API transport for GPT-5.6 models, with strict response model-identity
checking, is documented separately in
[NOTES_GPT56_POLICY_TRANSPORT.md](NOTES_GPT56_POLICY_TRANSPORT.md).

## Server smoke

Status: **BLOCKED / NOT_ACCEPTED**. The configured Spark-compatible endpoint
returned HTTP 502 during connectivity probing; no successful model response
was received and no replacement endpoint/model configuration is available.
An earlier presence-only probe on the server worker found no standard
`QWEN_*` or `DASHSCOPE_*` environment-variable names; it did not read values.
No API credential is recorded here. Model ID, image-input compatibility,
function-call compatibility, and successful inference latency remain
unverified. No tool or robot action was sent.

Redacted evidence: `/home/nvidia/algotithm/.simready-collaboration/robot-adapters-20261116/policy-qwen/env-presence-probe.json`.
