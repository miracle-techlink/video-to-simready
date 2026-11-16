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

## Evidence required per executed action

Record source commit, robot profile hash, observation timestamp and image hash,
action request, validation outcome, adapter receipt, measured post-action state,
tracking error, stop/timeout outcome, and independent task-verifier result.
API credentials and private camera content must not appear in logs or commits.
