# Astra policy provider

## Interface mismatch and adapter

The local IAS gateway exposes Astra through an Anthropic Messages endpoint and
accepts multimodal text/image requests. Its model catalog lists the request
alias `ias/gpt-6-astra`; the successful response identifies the dated model
`gpt-6-astra-2026-09-03`.

The gateway did not provide usable native tool calls for this route: a request
with Messages `tools` returned HTTP 400, and a text-only native-tool request
fell back to `zhipu/glm-5.3`. The existing `Gpt56MessagesTransport` correctly
rejects unexpected response model identities, so routing Astra through that
native-tool path fails closed.

`simready.robotics.astra_policy.AstraPolicyProvider` uses a different output
shape over the same Messages interface. It sends the allowed tool name and
JSON schema in the system instruction, omits the native `tools` field, and asks
for one plain-text JSON object:

```json
{"tool":"set_arm_joint_targets","arguments":{"joint1":0.01}}
```

The adapter requires the exact configured response model, a complete response,
strict JSON with no duplicate keys, exactly one allow-listed tool, and an
object for its arguments. It then maps that inert data into the shared
`PolicyDecision` type. `PolicyAdapterLoop` remains responsible for exact joint
set, numeric/finite values, profile limits, command freshness, per-step delta,
simulation-only dispatch, and independent post-action verification. The
provider never executes or dispatches a model result. Native tool mode remains
preferred for endpoints that support it; JSON text mode is the Astra-specific
compatibility path.

## Configuration

Set these variables in the environment of the process that runs the provider:

| Variable | Meaning |
| --- | --- |
| `ASTRA_MESSAGES_BASE_URL` | Messages URL including `/v1/messages` |
| `ASTRA_MESSAGES_API_KEY` | Gateway credential; keep out of source and logs |
| `ASTRA_MESSAGES_MODEL` | Request alias, e.g. `ias/gpt-6-astra` |
| `ASTRA_MESSAGES_EXPECTED_MODEL` | Exact response identity, e.g. `gpt-6-astra-2026-09-03` |

Example construction:

```python
from simready.robotics.astra_policy import AstraPolicyProvider
from simready.robotics.policy_loop import PolicyAdapterLoop

policy = AstraPolicyProvider()
loop = PolicyAdapterLoop(policy, simulation_adapter, independent_verifier, config)
```

When the policy loop runs on the GPUFree server, `127.0.0.1` refers to that
server. The workstation's local Astra gateway is not automatically reachable
there. A server-side canary therefore needs an explicitly configured secure
SSH tunnel or another approved private endpoint, and must keep the gateway
credential in a protected server environment. Do not expose the local gateway
on a public interface. The model image/state request and returned action must
be recorded with the source/profile identifiers and verified at the loop
boundary.

## Live model-only canary

On 2026-11-19, the local gateway was called through `AstraPolicyProvider` with
a synthetic image, synthetic six-joint observation, and the real
`set_arm_joint_targets` schema. It returned one parsed call for
`gpt-6-astra-2026-09-03`; the six targets were `0.01 rad`. No adapter was
connected and no robot command was sent. This establishes the multimodal
provider protocol and parser only. It does not establish camera freshness,
Isaac ROS 2 execution, or an independently verified task outcome.
