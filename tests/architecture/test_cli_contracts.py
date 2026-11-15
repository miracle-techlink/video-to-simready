"""CLI surface contracts: exit codes, stderr semantics and unknown-flag rejection."""

import subprocess
import sys

import pytest

PYTHON = sys.executable


def _run(module, *args, timeout=10):
    return subprocess.run(
        [PYTHON, "-m", module, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.parametrize(
    "module,args",
    [
        ("simready", ["--help"]),
        ("simready", ["run", "--help"]),
        ("simready", ["simulation", "--help"]),
        ("simready", ["ingestion", "--help"]),
        ("simready", ["workflows", "--help"]),
        ("simready", ["execution", "--help"]),
        ("simready", ["evaluation", "--help"]),
    ],
)
def test_help_exits_zero_and_prints_usage(module, args):
    result = _run(module, *args)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_bare_simready_requires_a_domain():
    result = _run("simready")
    assert result.returncode == 2
    assert "required" in result.stderr


@pytest.mark.parametrize(
    "module,args",
    [
        ("simready", ["--definitely-unknown-flag"]),
        ("simready", ["simulation", "not-a-worker"]),
        ("simready", ["workflows", "not-a-command"]),
        ("simready", ["execution", "not-a-command"]),
        ("simready", ["evaluation", "not-a-command"]),
        ("simready", ["ingestion", "not-a-command"]),
    ],
)
def test_unknown_subcommand_or_flag_exits_nonzero_with_stderr(module, args):
    result = _run(module, *args)
    assert result.returncode != 0
    assert result.stderr, "expected argparse diagnostics on stderr"


@pytest.mark.parametrize(
    "module,args,missing",
    [
        ("simready", ["execution", "run"], "--plan"),
        ("simready", ["evaluation", "run"], "--protocol"),
        ("simready", ["evaluation", "worker"], "--candidate"),
        ("simready", ["ingestion", "extract"], "--video"),
    ],
)
def test_required_arguments_reported_on_stderr(module, args, missing):
    result = _run(module, *args)
    assert result.returncode == 2
    assert missing in result.stderr


def test_simready_dispatch_rejects_unlisted_domain():
    result = _run("simready", "repair", "--help")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_evaluation_worker_invalid_input_exits_3(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    out = tmp_path / "out"
    result = _run(
        "simready",
        "evaluation",
        "worker",
        "--protocol",
        str(protocol),
        "--candidate",
        str(candidate),
        "--out",
        str(out),
    )
    assert result.returncode == 3
    assert "INVALID" in result.stderr


def test_execution_run_invalid_plan_exits_3(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    result = _run(
        "simready",
        "execution",
        "run",
        "--plan",
        str(plan),
        "--inventory",
        str(inventory),
        "--out",
        str(tmp_path / "out"),
        "--cache",
        str(tmp_path / "cache"),
    )
    assert result.returncode == 3
    assert "INVALID" in result.stdout


def test_simulation_dispatch_imports_only_selected_worker():
    # Dispatching to an unknown worker must fail at the argument layer without
    # importing Isaac worker modules.
    program = (
        "import sys\n"
        "from simready.simulation.cli import main\n"
        "try:\n"
        "    main(['unknown-worker'])\n"
        "except SystemExit as error:\n"
        "    assert error.code != 0, error.code\n"
        "assert 'simready.simulation.scene_video' not in sys.modules\n"
        "assert 'simready.simulation.depth_observation' not in sys.modules\n"
    )
    result = subprocess.run(
        [PYTHON, "-c", program], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
