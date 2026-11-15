# tests/fixtures

Shared synthetic inputs and helper workers used by the test suite. None of
these files perform real reconstruction, rendering, model inference, or
scoring. Anything labelled "synthetic" or "fake" here MUST NOT be cited as
evidence of real algorithmic behaviour.

| File | Kind | Synthetic? | Used by | Notes |
|---|---|---|---|---|
| `docker_cli.py` | subprocess stub for the docker CLI | yes | execution/workers tests | intercepts docker invocations so tests don't need a real daemon |
| `pipeline.py` | pytest fixture (`configured`) | yes | workflows/pipeline tests | builds a tiny `examples/pipeline` fixture via `create_fixture` |
| `scene_candidate.py` | builder for candidate + protocol payload | yes | evaluation tests | deterministic RNG image, fixed protocol checks |
| `scene_preparation.py` | scene spec fixture | yes | simulation tests | placeholder scene preparation inputs |
| `synthetic_task.py` | CLI worker emitting `artifact.json` | yes | execution/tasks graph tests | supports `--fail`, `--delay`, seed propagation |
| `task_execution.py` | task dict + executor around `synthetic_task.py` | yes | execution/tests | wraps `run_task_graph` with a tiny inventory |
| `video_worker.py` | writes a synthetic `splats.ply` + `input.json` | yes | ingestion/simulation tests | deterministic PLY bytes |
| `fake_render_worker.py` | **synthetic** fake batch renderer | **yes** | PR 5 CPU CI scaffolding | emits placeholder RGB/depth + receipt; supports `--fail-at-view`, `--timeout-after`, `--corrupt-hash` |
| `fake_scoring_worker.py` | **synthetic** fake scorer | **yes** | PR 5 CPU CI scaffolding | emits placeholder scorecard; verdict derived from candidate+protocol hashes |
| `test_fake_workers.py` | self-tests for the two fake workers | yes | this directory | exercises receipts, fault injection, verdict semantics |

## Provenance discipline

- Every fake worker embeds `synthetic_fixture=True` and a banner string in
  its outputs, so downstream consumers can detect synthetic provenance
  programmatically (not just by reading this README).
- Fake workers' input/output field names are placeholders for the schema
  that PR 1 will freeze. When the real schema lands, replace the dict
  shapes here without changing the worker CLIs.
- Fake worker receipts never satisfy the "real rendering" or "real
  scoring" acceptance criteria from the development plan. They only prove
  that orchestration, hashing, fault injection, and receipt-binding logic
  behave correctly.
