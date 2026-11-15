"""Wheel-installation smoke test (opt-in).

This test verifies that a built wheel, installed into an isolated virtual
environment OUTSIDE the repository checkout, exposes a usable
``simready`` package. It is skipped by default; CI enables it explicitly
after building a wheel.

Enable locally with::

    SIMREADY_WHEEL_SMOKE=1 SIMREADY_WHEEL=/path/to/dist/simready_pipeline-*.whl \
        python -m pytest tests/architecture/test_wheel_smoke.py

If ``SIMREADY_WHEEL`` is not set, the test builds a fresh wheel via
``python -m build`` into a temporary directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest

from tests.paths import REPOSITORY_ROOT

pytestmark = pytest.mark.skipif(
    not os.environ.get("SIMREADY_WHEEL_SMOKE"),
    reason="wheel smoke requires built wheel; set SIMREADY_WHEEL_SMOKE=1 to enable",
)


def _build_wheel(out_dir: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel in {out_dir}, got {wheels}"
    return wheels[0]


def _make_venv(root: Path) -> Path:
    venv.EnvBuilder(with_pip=True, clear=True).create(root)
    return root / "bin" / "python"


def _run(python: Path, code: str, cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(python), "-I", "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_wheel_installs_and_imports_outside_checkout(tmp_path):
    wheel_path_str = os.environ.get("SIMREADY_WHEEL")
    if wheel_path_str:
        wheel = Path(wheel_path_str)
        assert wheel.exists(), wheel
    else:
        wheel = _build_wheel(tmp_path / "dist")

    env_dir = tmp_path / "venv"
    python = _make_venv(env_dir)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,  # outside the repository checkout
    )

    # Import + version must work in an isolated interpreter.
    result = _run(python, "import simready; print(simready.__version__)", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    version = result.stdout.strip()
    assert version and version[0].isdigit(), version

    # The package must resolve inside the venv, not the repository checkout.
    check = (
        "import pathlib, simready, sys;"
        "assert pathlib.Path(simready.__file__).is_relative_to(sys.prefix), simready.__file__"
    )
    result = _run(python, check, cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    # The runtime must not reach back into examples/ or tests/.
    check = (
        "import simready, sys;"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('examples', 'tests')];"
        "assert not bad, bad"
    )
    result = _run(python, check, cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    # Dev-only top-level packages must not be importable from the wheel.
    check = (
        "import importlib.util;"
        "assert all(importlib.util.find_spec(name) is None for name in ('agents', 'repair', 'experience'))"
    )
    result = _run(python, check, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
