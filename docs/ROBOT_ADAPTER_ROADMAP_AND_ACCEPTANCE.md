# Robot policy and adapter implementation plan

Baseline: `6a3bc46` on `feat/robot-adapters` (includes the reBot Isaac joint
canary). Implementation tasks are split across isolated worktrees. This plan
does not authorize merging to `main` or operating physical hardware.

## Target architecture

```text
Task + persistent context
          |
Qwen 3.8 provider (initial) / Astra / Fable (replaceable)
          | typed tool call; model never talks to a driver directly
Observation and policy tool API
          | image, calibrated camera data, joint state, EEF pose, feedback
Canonical action contract
          | EEF pose + gripper + short timed action segment
Safety gate and deterministic trajectory/controller layer
          | bounds, freshness, workspace, rate, collision, watchdog/stop
RobotAdapter contract
          +-- Isaac Sim direct API (existing canary)
          +-- Isaac Sim ros2_control / ROS 2 bridge
          +-- reBot ROS 2 / vendor hardware driver (not commissioned)
          +-- AgiBot G2 adapter (future, profile only)
```

The model chooses bounded high-level actions. Deterministic code validates and
executes them; the adapter normalizes feedback. Simulation and hardware use the
same action/state contracts but separate transport implementations. A task
success verifier is independent of the model's `done` response.

## Parallel modules and acceptance gates

| Module | Scope / owner | Acceptance criteria |
|---|---|---|
| A. Replaceable policy providers | Qwen API/native tool calls and Astra Messages JSON-text adapter; no secrets in code/logs | Mocked tests cover valid calls, malformed arguments, timeout, API error, unknown tool, and refusal. Live smoke records exact response model ID and image/action parsing; it must not dispatch an adapter or command hardware. |
| B. Canonical actions and safety | EEF/joint action schemas, action chunks, deterministic validation and adapter conformance suite | Tests cover frames/units, quaternion normalization, finite values, joint/workspace limits, max per-step delta/rate, stale observations, duplicate sequence, malformed and expired actions, and safe rejection. Action-to-trajectory behavior is deterministic and replayable. EEF schema does not claim executable IK unless a tested IK implementation is supplied. |
| C. Isaac ROS 2 simulation adapter | Isaac `ros2_control` or Bridge-backed adapter; determine the supported 6.2 route | On the GPU server, use only the configured existing instance and coordinate via the shared GPU lock. Publish/receive real ROS 2 messages/actions in Isaac, command a small bounded reBot move, and verify feedback and action result. Preserve logs and config. If ROS runtime or GPU is unavailable, report BLOCKED with exact reason; no direct-API smoke may substitute for ROS 2 PASS. |
| D. reBot hardware ROS 2 adapter contract | Robot-specific ROS 2 transport using verified driver documentation; mocks only until separately commissioned | Trace actual driver source/docs for state and command interfaces; implement normalized adapter and mocked contract tests. Keep `hardware_verified=false`; reject hardware commands by default. No physical robot commands. Do not invent topic/action names. Hardware integration remains NOT_RUN without reBot and safety commissioning. |

## Cross-module release gates

1. Each module reports branch, commit, changed files, commands, test results,
   evidence paths, and limitations. No claims based on tests skipped by default.
2. Review all diffs and run focused tests and repository CI locally/CPU-side.
3. Run the selected provider smoke against its configured endpoint after
   confirming model identity and image/action parsing. Keep model-only inference
   separate from GPU coordination; store redacted evidence and never print an
   API key or token. For workstation-local Astra with a GPUFree loop, use only
   an explicitly approved private tunnel; do not expose the gateway publicly.
4. Run Isaac ROS 2 acceptance separately from existing direct Python articulation
   tests. Check observed state and action result, not merely a successful API call.
5. Keep real hardware explicitly disabled until documented commissioning,
   operator-held E-stop, workspace limits, low-speed no-load test, disconnect
   behavior, and independent review have passed.
6. Do not merge to `main`, mark hardware verified, or claim task autonomy from a
   one-step joint move. Codex performs integration review and reports gates as
   PASS / FAIL / BLOCKED / NOT_RUN with evidence.

## 2026-11-17 integrated CPU acceptance

Integrated on `feat/robot-adapters` after isolated review:

- Module A — **PASS (CPU/mock)**: Qwen typed tool-call provider; image data URLs are sent as multimodal API blocks; no execution or adapter imports. Live server smoke is **NOT_RUN** because the configured Spark endpoint currently returns HTTP 502 and no authorized replacement endpoint/model configuration is available.
- Module B — **PASS (CPU)**: canonical external Cartesian action contract uses `orientation_xyzw` and normalized `opening_fraction` (`0.0` closed, `1.0` open). Fresh observation, expiration, workspace, action horizon, sequence, and bounded step rejection tests pass.
- Module C — **PASS (CPU contract), BLOCKED (live Isaac ROS 2)**: injected fake transport confirms the `FollowJointTrajectory` client/state normalization behavior; no live Isaac `ros2_control` action result has been recorded.
- Module D — **PASS (CPU/profile), NOT_RUN (hardware)**: the official reBot profile carries the six arm and two state-only finger joints. Both command transports remain `null`; `hardware_verified=false`, so hardware dispatch is fail-closed.

Verification run from this branch:

```bash
PYTHONPATH=src python -m pytest -q tests/robotics
# 140 passed
PYTHONPATH=src python -m pytest -q -m 'not real_model and not gpu'
# 1498 passed, 1 skipped, 20 deselected
ruff check src/simready tests/robotics examples/robot_adapters
ruff format --check src/simready tests/robotics examples/robot_adapters
python scripts/check_architecture.py
# Architecture PASS: 9 domains, 17 public functions
```

The unqualified `python -m pytest` invocation is not an acceptance command in
this source-layout checkout: it lacks `PYTHONPATH=src` and fails robotics test
collection. This is an environment invocation issue, not a product-test
failure; the project acceptance command above is authoritative.

## 2026-11-18 independent revalidation

Codex reran acceptance on `feat/robot-adapters` at `c2bf687` after reviewing
the separate module branches and their integration:

```bash
PYTHONPATH=src python -m pytest -q tests/robotics
# 142 passed
PYTHONPATH=src python -m pytest -q -m 'not real_model and not gpu'
# 1500 passed, 1 skipped, 20 deselected
ruff check src/simready tests/robotics examples/robot_adapters
# All checks passed
ruff format --check src/simready tests/robotics examples/robot_adapters
# 136 files already formatted
python scripts/check_architecture.py
# Architecture PASS: 9 domains, 17 public functions
```

The live-gate notes in this first revalidation were superseded by the
2026-11-18 follow-up below. The CPU results validate contracts and mock
integration, not an end-to-end Qwen-to-ROS robot task.

## 2026-11-18 PR #31 follow-up acceptance

The policy-loop implementation is now on this PR branch after
`2f0450538d86ad6fd002a584c9b1dc85dbcfe91b`. The production `PolicyAdapterLoop`
connects the Qwen provider, image-bearing observations, fresh joint-state,
canonical validation, per-step bounds, a simulation-marked adapter, and an
independent verifier. Its CPU tests use fake HTTP and ROS2 transports; they do
not demonstrate live inference or live Isaac control.

- **Python 3.12 regression: PASS.** GitHub Actions completed `cpu-tests
  (3.12)` successfully on this head. Static checks, CPU contract tests, package
  smoke, Python 3.10, and merge gate also pass; `main-artifacts` is skipped as
  expected for a PR. The previous pushed head had all required checks green;
  CI for the current freshness-fix head is running. The policy-loop
  implementation passed focused checks (`27` policy-loop tests, `173`
  robotics tests, Ruff, format, and architecture).
- **Qwen 3.8 live image/function-call smoke: BLOCKED.** The server-side
  configuration audit found no configured Qwen, DashScope, or Spark endpoint,
  model, or credential in the permitted environment and user/deployment
  configuration scopes. No request, tool execution, or robot command was
  sent. Evidence:
  `/root/gpufree-data/liuyue/runs/robot-adapter-acceptance-20261118/pr31-9a21f8d/qwen-live-acceptance-receipt.json`.
  To run this gate, configure `QWEN_BASE_URL`, `QWEN_MODEL`, and `QWEN_API_KEY`
  (or the equivalent `DASHSCOPE_*` variables) in the server environment.
- **Isaac Sim 6.2 ROS 2 action/feedback: BLOCKED.** A temporary arm-only USD
  derivative with the articulation root relocated to the model root passed
  composed-stage and joint-limit checks; the original USD and upstream robot
  profiles were unchanged, and the gripper remains excluded/untested. The
  controller-manager and both controllers now load and activate. A bounded
  `FollowJointTrajectory` goal was accepted by the real Isaac controller, whose
  log reports `Goal reached, success!`, but the adapter timed out waiting for
  the terminal action result. Before/after `/joint_states` source timestamps
  were identical, and measured joint motion remained below the requested
  0.02-radian delta, so independent feedback acceptance failed. A prior client
  launch also exposed missing Isaac Jazzy paths in the subprocess environment;
  the acceptance runner now supplies them. Latest evidence:
  `/root/gpufree-data/liuyue/runs/robot-adapter-acceptance-20261118/pr31-9a21f8d/live-isaac-ros2-20261118T092819Z/`;
  initial root relocation validation is
  `/root/gpufree-data/liuyue/runs/robot-adapter-acceptance-20261118/pr31-9a21f8d/arm-only-root-articulation-20261118T090909Z/root-articulation-acceptance.json`.
- **Freshness bug found during live acceptance.** The `rclpy` joint-state
  callback previously left `received_monotonic_ns` unset, causing reads of a
  cached message to be stamped as fresh at read time. Commit `24d9945` records
  the callback's monotonic receive time and adds a regression test. The full
  `tests/robotics` suite passed on the GPUFree server (`174 passed`); GitHub CI
  for this commit is still running.
- **Live policy→validator→Isaac ROS 2→independent verifier: NOT_RUN (status at
  the time of this 2026-11-18 entry).** The 2026-11-19 follow-up below updates
  provider and arm-only Isaac module results; the combined task canary remains
  outstanding. The verifier timeout is cooperative; camera-frame freshness is
  the host caller's responsibility.
- **Physical reBot/G2 hardware: NOT_RUN and fail-closed.** No hardware command
  path is enabled or tested.

## Remaining acceptance gates before PR #31 merge

GitHub CI covers CPU/static/package checks; it does not close the live
integration gate below. Current evidence is:

1. **Policy provider live multimodal action inference: PASS for local Astra JSON
   mode; Qwen live API remains BLOCKED.** Astra returned one parsed
   `set_arm_joint_targets` call for exact response model
   `gpt-6-astra-2026-09-03` from a synthetic image and state. No adapter
   dispatch occurred. Astra native tool mode fell back to GLM and is rejected;
   the JSON-text adapter is the supported path. See
   [`NOTES_ASTRA_POLICY_PROVIDER.md`](NOTES_ASTRA_POLICY_PROVIDER.md).
2. **Isaac ROS 2 live arm action: PASS.** On GPUFree Isaac Sim 6.2, one bounded
   arm-only `FollowJointTrajectory` completed, fresh joint-state source time
   advanced, joint1 moved toward target, and measured error was about
   `0.00098 rad`. The gripper was excluded. Evidence:
   `/root/gpufree-data/liuyue/runs/robot-adapter-acceptance-20261118/pr31-9a21f8d/live-isaac-ros2-20261118T120805Z/evidence/acceptance-verdict.json`.
3. **Closed-loop simulation canary: NOT_RUN.** Connect Astra JSON-mode provider
   to the canonical action validator and Isaac ROS 2 adapter. The workstation
   gateway is not automatically reachable from GPUFree; establish an approved
   private tunnel, then use one fresh camera/state observation, bounded
   simulation command, advancing feedback, and independent task verification.
   Preserve source/profile hashes and action/result evidence. Keep PR #31 open
   until this gate passes.
4. **Merge hygiene:** refresh the feature branch, include this acceptance
   record, rerun required GitHub checks, and review the full diff before merge.

The following are explicitly outside PR #31's merge gate and must remain
fail-closed: physical reBot commissioning (no robot/E-stop procedure has been
validated here) and a physical AgiBot G2 adapter (only a simulation profile is
present). Cartesian validation also does not yet implement IK, collision
checking, trajectory generation, or a physical command path; those require a
separate implementation and acceptance scope.

## 2026-11-19 Astra interface investigation

The local IAS gateway at `127.0.0.1:3456` lists `ias/gpt-6-astra`. A
text/image Messages request returns the dated identity
`gpt-6-astra-2026-09-03`. Native `tools` requests are not usable on this route:
the image-plus-tool request returned HTTP 400, and the text-only native-tool
request returned `zhipu/glm-5.3`. Strict response-model checking therefore
correctly rejects the fallback instead of accepting it as Astra.

`AstraPolicyProvider` adapts this endpoint by placing the allowed tool schema
in the system prompt, requesting exactly one JSON text action, and parsing it
into the shared inert `PolicyDecision`. The local multimodal canary passed with
a synthetic image and synthetic six-joint state: one
`set_arm_joint_targets` call was parsed for `gpt-6-astra-2026-09-03`; all six
targets were `0.01 rad`; no adapter was connected and no command was sent.
Details and configuration are in
[`NOTES_ASTRA_POLICY_PROVIDER.md`](NOTES_ASTRA_POLICY_PROVIDER.md).

The independent Isaac Sim 6.2 arm-only ROS 2 canary also passed on
2026-11-18: one bounded `FollowJointTrajectory` completed, joint-state source
time advanced, joint1 moved toward target, and measured error was about
`0.00098 rad`. Evidence:
`/root/gpufree-data/liuyue/runs/robot-adapter-acceptance-20261118/pr31-9a21f8d/live-isaac-ros2-20261118T120805Z/evidence/acceptance-verdict.json`.
The gripper was excluded.

**The full Astra → policy loop → Isaac ROS 2 → independent task verifier
canary remains NOT_RUN.** Astra inference ran on the workstation while Isaac
runs on GPUFree; the workstation-local gateway is not automatically reachable
from the server. No tunnel or combined run is claimed. PR #31 remains open until
the full canary has one frozen source revision, fresh camera/state evidence,
bounded dispatch, advancing feedback, and an independent task result.

## Evidence required per executed action

Record source commit, robot profile hash, observation timestamp and image hash,
action request, validation outcome, adapter receipt, measured post-action state,
tracking error, stop/timeout outcome, and independent task-verifier result.
API credentials and private camera content must not appear in logs or commits.
