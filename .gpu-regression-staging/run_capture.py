"""GPU regression driver: two-view benchmark observation, local backend.

Usage (venv python, cwd MUST be the inputs dir so the worker resolves the
scene path relative to the manifest root):
    cd <run>/inputs && python <run>/run_capture.py <out-subdir> <timeout-s>

Mirrors simready.simulation.benchmark_observation.run_benchmark_observation
with two adjustments required by the GPUFree host:
  1. observation_plan.json is written next to the manifests (the worker
     re-validates file refs against the plan file's parent directory).
  2. the Isaac interpreter is passed explicitly to launch_worker (the CLI
     'run' subcommand has no --python flag and would use sys.executable).
"""

import sys
import time
from pathlib import Path

REPO = Path("/root/gpufree-data/liuyue/video-to-simready-gpu-regression")
RUN = Path("/root/gpufree-data/liuyue/runs/gpu-regression-20261115")
INPUTS = RUN / "inputs"
# /root/isaacsim install segfaults at omni.kit.async_engine startup on this
# host; the /isaac-sim install (proven by the existing acceptance harness)
# boots cleanly. Wrapper replicates .simready/isaac-python.sh with our venv.
ISAAC_PY = Path("/root/gpufree-data/liuyue/isaac-python-gpu-regression.sh")

from simready.core.artifacts import write_json
from simready.execution.workers.isaac_launcher import launch_worker
from simready.simulation.benchmark_observation import plan_benchmark_observation


def main() -> int:
    out_dir = RUN / sys.argv[1]
    timeout_s = float(sys.argv[2])
    assert Path.cwd().resolve() == INPUTS.resolve(), "run me from the inputs dir"

    plan = plan_benchmark_observation(
        INPUTS / "render_request.json", INPUTS / "view_set.json", out_dir
    )
    mode = plan["render_request"]["image_settings"]["mode"]
    plan_file = INPUTS / "observation_plan.json"
    write_json(
        plan_file,
        {
            "render_request": plan["render_request"],
            "view_set": plan["view_set"],
            "mode": mode,
            "want_depth": mode == "rgbd",
        },
    )

    # PR5 bug A @6a823ef: run_benchmark_observation sends ["--plan", ...] with
    # no subcommand (top-level parser rejects it), while main() re-parses the
    # full argv in _worker_main so inserting "worker" is also rejected. The
    # worker entry is unreachable via CLI; invoke _worker_main directly.
    worker = [
        "-c",
        "import sys; from simready.simulation.benchmark_observation import "
        "_worker_main; sys.exit(_worker_main(sys.argv[1:]))",
        "--plan",
        str(plan_file),
        "--out",
        str(out_dir),
    ]

    started = time.monotonic()
    try:
        code = launch_worker(
            worker,
            repo=REPO,
            assets=REPO,
            output=out_dir,
            backend="local",
            python=ISAAC_PY,
            gpu="0",
            timeout_s=timeout_s,
        )
    except BaseException as error:  # ProcessTimeoutError / ProcessCancelledError
        print(
            f"LAUNCH_EXCEPTION {type(error).__name__}: {error}",
            flush=True,
        )
        print(f"WALL_S {time.monotonic() - started:.1f}", flush=True)
        return 2
    print(f"LAUNCH_RETURN_CODE {code}", flush=True)
    print(f"WALL_S {time.monotonic() - started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
