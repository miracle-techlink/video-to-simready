"""Synthetic fake worker. Result does NOT count as real rendering or real scoring.

This module produces a placeholder scorecard from a candidate payload
plus a protocol payload. It exists so that CI can exercise the candidate
binding and scoring semantics of PR 3 without depending on the real
``simready.evaluation`` stack or any model outputs. Every emitted payload
carries a ``synthetic_fixture=True`` marker; downstream consumers MUST
NOT treat these outputs as evidence of real quality evaluation.

Inputs are plain dicts. Field names are placeholders for the schema that
PR 1 will freeze; do not couple this worker to PR 1 field details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

SYNTHETIC_BANNER = (
    "Synthetic fake worker. Result does NOT count as real rendering or real scoring."
)


def _check_outcome(check: dict, candidate: dict) -> dict:
    """Compute a deterministic pass/fail for one protocol check.

    The "score" is derived from a hash of (check_id, candidate_id) so
    different candidates produce different but reproducible outcomes. No
    real metric is computed; this is purely a contract placeholder.
    """
    check_id = str(check.get("check_id", "unknown"))
    candidate_id = str(candidate.get("candidate_id", "unknown"))
    digest = hashlib.sha256(f"{check_id}:{candidate_id}".encode()).hexdigest()
    pseudo_score = int(digest[:8], 16) / 0xFFFFFFFF  # 0..1
    limit = float(check.get("limit", 0.5))
    op = str(check.get("op", ">="))

    if op == ">=":
        passed = pseudo_score >= limit
    elif op == "<=":
        passed = pseudo_score <= limit
    else:
        passed = False

    return dict(
        check_id=check_id,
        track=str(check.get("track", "unknown")),
        metric=str(check.get("metric", "unknown")),
        op=op,
        limit=limit,
        synthetic_score=round(pseudo_score, 6),
        passed=passed,
        required=bool(check.get("required", True)),
    )


def run_score_request(candidate: dict, protocol: dict, out_dir: Path) -> dict:
    """Produce a synthetic scorecard from candidate + protocol dicts.

    Returns the scorecard dict written to ``out_dir / "scorecard.json"``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    checks = [_check_outcome(c, candidate) for c in protocol.get("checks", [])]
    failed_required = [c for c in checks if c["required"] and not c["passed"]]

    if not checks:
        verdict = "INCOMPLETE"
    elif failed_required:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    scorecard = dict(
        synthetic_fixture=True,
        banner=SYNTHETIC_BANNER,
        worker="tests/fixtures/fake_scoring_worker.py",
        candidate_id=str(candidate.get("candidate_id", "unknown")),
        protocol_id=str(protocol.get("benchmark_id", "unknown")),
        phase=str(protocol.get("phase", "development")),
        verdict=verdict,
        checks=checks,
        duration_seconds=round(time.monotonic() - started, 6),
    )
    (out_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True)
    )
    return scorecard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=SYNTHETIC_BANNER)
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate JSON (placeholder schema)",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Protocol JSON (placeholder schema)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args(argv)

    candidate = json.loads(args.candidate.read_text())
    protocol = json.loads(args.protocol.read_text())
    scorecard = run_score_request(candidate, protocol, args.out)
    print(
        json.dumps(
            {
                "synthetic_fixture": True,
                "verdict": scorecard["verdict"],
                "checks": len(scorecard["checks"]),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
