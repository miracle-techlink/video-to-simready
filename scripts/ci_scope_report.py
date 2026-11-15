"""Summarize what CI actually executed versus what stayed NOT_RUN.

Reads the full CPU regression JUnit file and counts the opt-in marker
populations (``real_model`` / ``gpu``) that no Actions job enables, so a
green CPU run cannot be read as model/GPU acceptance. Pure stdlib; the
pytest collect step is the only subprocess.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

OPT_IN_MARKERS = ("real_model", "gpu")
NOT_RUN_REASON = "opt-in markers; no Actions job passes --run-real-model/--run-gpu"


def _junit_tally(path: Path) -> dict:
    """Return per-outcome counts from a JUnit XML file, or an error marker."""
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:  # pragma: no cover - malformed junit
        raise ValueError(f"no testsuite element in {path}")
    return {
        "total": int(suite.get("tests", 0)),
        "failures": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "passed": int(suite.get("tests", 0))
        - int(suite.get("failures", 0))
        - int(suite.get("errors", 0))
        - int(suite.get("skipped", 0)),
    }


def _collect_opt_in(marker: str) -> int | None:
    """Count collected tests for one opt-in marker; None when uncountable."""
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", marker],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    # Exit 5 means "no tests collected": a valid, reportable zero.
    if proc.returncode not in (0, 5):
        return None
    for line in reversed(proc.stdout.splitlines()):
        stripped = line.strip()
        # Shape: "18/1226 tests collected (1208 deselected) in 0.35s"
        if stripped.endswith("tests collected") or " tests collected " in stripped:
            head = stripped.split("/", 1)[0]
            try:
                return int(head)
            except ValueError:
                continue
    return 0 if proc.returncode == 5 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        executed = _junit_tally(args.junit)
    except (OSError, ET.ParseError, ValueError) as exc:
        executed = {"error": f"junit unavailable or malformed: {exc}"}

    not_run = {}
    for marker in OPT_IN_MARKERS:
        count = _collect_opt_in(marker)
        not_run[marker] = (
            {"collected": count, "status": "NOT_RUN", "reason": NOT_RUN_REASON}
            if count is not None
            else {"status": "NOT_RUN", "reason": NOT_RUN_REASON, "collected": None}
        )

    report = {
        "schema_version": 1,
        "executed_cpu_regression": executed,
        "not_run": not_run,
        "boundary": (
            "CPU/contract/package greenness does not validate real-model "
            "inference, GPU rendering, navigation or physical accuracy"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
