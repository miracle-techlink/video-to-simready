# GPT-5.6 policy transport

## Scope

`simready.robotics.gpt56_policy.Gpt56MessagesTransport` adapts the existing
inert `QwenPolicyProvider` request boundary to an explicitly configured
Anthropic Messages-compatible endpoint. It is a transport only: it neither
executes a returned tool call nor imports a robot adapter. Use it as the
injected `transport` argument to `QwenPolicyProvider`; retain
`PolicyAdapterLoop` as the existing validation/dispatch boundary.

The implementation preserves every accepted `image_url` data URL as an
Messages image block and carries each caller-provided function `name`,
`description`, and full JSON `parameters` schema as a Messages `input_schema`.
It converts returned `tool_use` blocks back into the typed data parsed by
`QwenPolicyProvider`.

## Configuration

All values are read from the worker environment at request time. The defaults
are intentionally explicit and independent of global router configuration:

| Variable | Meaning |
| --- | --- |
| `GPT56_MESSAGES_BASE_URL` | Direct Messages endpoint, including `/v1/messages` |
| `GPT56_MESSAGES_API_KEY` | Endpoint credential |
| `GPT56_MESSAGES_MODEL` | Requested model alias, e.g. `gpt-5.6-luna` |
| `GPT56_MESSAGES_EXPECTED_MODEL` | Exact response identity, e.g. `gpt-5.6-luna-2026-07-09` |

`Gpt56MessagesConfig` permits naming alternative environment variables without
putting credentials, a socket path, or a deployment endpoint into library code.
The requested alias must equal the expected identity or be its prefix before a
dated snapshot suffix, and the model in the outgoing request body must equal
the configured model before any request is sent. More importantly, every
successful response must return an exact `model` equal to
`GPT56_MESSAGES_EXPECTED_MODEL`; a gateway fallback or identity mismatch fails
closed before tool-call parsing.

Only Messages stop reasons `tool_use` and `end_turn` are accepted. A truncated
response (`max_tokens`, unknown or missing stop reason) fails closed even when
`tool_use` blocks are present, and tool calls with a non-`tool_use` stop reason
are rejected as incomplete rather than dispatchable.

HTTP errors carry only their numeric status plus a fixed local status class;
upstream reason or body text is never propagated because it can contain
credentials. Network failures, timeouts, malformed JSON, malformed `tool_use`,
and conversion errors are typed `QwenPolicyError` subclasses and fail closed.
The implementation does not log credential values.

## Tests and live acceptance

CPU-only tests cover preserved multimodal images and tool schema conversion,
returned-model mismatch, request/configured model mismatch, malformed and
truncated tool-use rejection, stop-reason validation, config validation, and
sanitized timeout/HTTP failure:

```bash
PYTHONPATH=src python -m pytest -q tests/robotics/test_gpt56_policy.py
```

Server-side mocked robotics suite (GPUFree, 2026-11-18): **182 passed** before
the review-hardening commit; local mocked robotics after hardening: **189
passed** (server revalidation of the combined integration commit is tracked
separately in the integration acceptance handoff).

Live acceptance is an external, credential-bearing operational script, not a
library file. It reads the local IAS sqlite configuration read-only, forwards
the direct IAS endpoint through the existing SSH master, and sends credentials
only via encrypted stdin. Its exit status is the receipt status (unlike the
superseded exploratory script): `0` only for `PASS`, nonzero for `FAIL`.

The successful Luna run used two distinct color-image/tool-call comparisons and
a structured arm target passed through the existing `JointCommand` validator.
It performed no adapter dispatch or robot command:

- Receipt: `/root/gpufree-data/liuyue/runs/robot-adapter-acceptance-20261118/gpt56-transport-20261118T121438Z/receipt.json`
- Server source snapshot: `/root/gpufree-data/liuyue/runs/robot-adapter-acceptance-20261118/pr31-0ca3a28-gpt56/source`
- Result: `PASS`; returned identity `gpt-5.6-luna-2026-07-09` on all three
  `HTTP 200` Messages responses; `robot_command_sent=false` and
  `adapter_dispatch=false`.

This is model-transport capability evidence only. It is not Isaac/ROS live-loop
acceptance and must not be interpreted as authorization to dispatch an adapter.
