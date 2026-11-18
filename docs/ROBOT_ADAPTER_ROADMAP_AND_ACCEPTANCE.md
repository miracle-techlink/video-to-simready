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
| A. Qwen policy provider | Qwen API client and tool-call parser; no secrets in code/logs | Mocked tests cover valid call, malformed arguments, timeout, API error, unknown tool, and refusal. Server smoke uses the configured Qwen endpoint without printing credentials; records model ID, latency, redacted request/response shape, and whether image input and function calling actually work. It must not command hardware. |
| B. Canonical actions and safety | EEF/joint action schemas, action chunks, deterministic validation and adapter conformance suite | Tests cover frames/units, quaternion normalization, finite values, joint/workspace limits, max per-step delta/rate, stale observations, duplicate sequence, malformed and expired actions, and safe rejection. Action-to-trajectory behavior is deterministic and replayable. EEF schema does not claim executable IK unless a tested IK implementation is supplied. |
| C. Isaac ROS 2 simulation adapter | Isaac `ros2_control` or Bridge-backed adapter; determine the supported 6.2 route | On the GPU server, use only the configured existing instance and coordinate via the shared GPU lock. Publish/receive real ROS 2 messages/actions in Isaac, command a small bounded reBot move, and verify feedback and action result. Preserve logs and config. If ROS runtime or GPU is unavailable, report BLOCKED with exact reason; no direct-API smoke may substitute for ROS 2 PASS. |
| D. reBot hardware ROS 2 adapter contract | Robot-specific ROS 2 transport using verified driver documentation; mocks only until separately commissioned | Trace actual driver source/docs for state and command interfaces; implement normalized adapter and mocked contract tests. Keep `hardware_verified=false`; reject hardware commands by default. No physical robot commands. Do not invent topic/action names. Hardware integration remains NOT_RUN without reBot and safety commissioning. |

## Cross-module release gates

1. Each module reports branch, commit, changed files, commands, test results,
   evidence paths, and limitations. No claims based on tests skipped by default.
2. Review all diffs and run focused tests and repository CI locally/CPU-side.
3. Run Qwen API smoke on the server only after confirming the endpoint/model
   capabilities and GPU coordination state. Store redacted evidence; never print
   an API key or token.
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

Module gates are unchanged: Qwen inference remains blocked by the configured
Spark-compatible endpoint returning HTTP 502; Isaac ROS 2 live acceptance
remains blocked because the configured GPU server SSH/lock could not be
accessed; physical reBot hardware remains NOT_RUN and fail-closed. These
results validate CPU integration and contracts, not an end-to-end Qwen-to-ROS
robot task.

## Remaining acceptance gates before PR #31 merge

The PR's GitHub CPU/static/package checks currently pass, but these checks do
not close the live integration gates below. Keep PR #31 open until Qwen and
Isaac simulation run together through the actual control boundary:

1. **Qwen live inference:** configure the intended Qwen model/endpoint; receive
   a successful function-call response and exercise an image-bearing request.
   Record model ID, endpoint status, latency, parsed tool-call shape, and
   redacted evidence. Do not execute the tool in the Qwen smoke. Current status:
   **BLOCKED** (configured Spark-compatible endpoint returned HTTP 502).
2. **Isaac ROS 2 live action:** on the configured GPUFree Isaac Sim 6.2 server,
   acquire the shared GPU lock, send one small bounded
   `FollowJointTrajectory` goal through the adapter, and record its terminal
   action result plus fresh measured `/joint_states` and tracking error. Current
   status: **BLOCKED** (SSH/lock access unavailable).
3. **Closed-loop simulation canary:** connect the accepted Qwen provider to the
   canonical action validator and Isaac ROS 2 adapter. Send a fresh camera/state
   observation, validate the returned action, execute only the bounded
   simulation command, then verify the post-action state with an independent
   task check. Preserve source/profile hashes and action/result evidence. This
   is currently **NOT_RUN**; isolated provider and adapter tests are not a
   substitute.
4. **Merge hygiene:** refresh the feature branch, ensure the PR head contains
   this acceptance record, rerun required GitHub checks, review the complete
   diff, and merge PR #31 through GitHub only after gates 1–3 pass.

The following are explicitly outside PR #31's merge gate and must remain
fail-closed: physical reBot commissioning (no robot/E-stop procedure has been
validated here) and a physical AgiBot G2 adapter (only a simulation profile is
present). Cartesian validation also does not yet implement IK, collision
checking, trajectory generation, or a physical command path; those require a
separate implementation and acceptance scope.

## Evidence required per executed action

Record source commit, robot profile hash, observation timestamp and image hash,
action request, validation outcome, adapter receipt, measured post-action state,
tracking error, stop/timeout outcome, and independent task-verifier result.
API credentials and private camera content must not appear in logs or commits.
