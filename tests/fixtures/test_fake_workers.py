"""Self-tests for the synthetic fake workers.

These tests verify the contract semantics (receipts, fault injection,
verdicts) of the fixtures themselves. They do NOT validate any real
rendering or scoring behaviour.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.paths import REPOSITORY_ROOT

RENDER_WORKER = REPOSITORY_ROOT / "tests/fixtures/fake_render_worker.py"
SCORING_WORKER = REPOSITORY_ROOT / "tests/fixtures/fake_scoring_worker.py"


def _render_request(tmp_path: Path, views: int = 2) -> Path:
    payload = dict(
        scene_id="fixture-scene",
        seed=7,
        want_depth=True,
        render_settings={"backend": "fake", "spp": 1},
        views=[
            dict(view_id=f"v{i}", width=32, height=24)
            for i in range(views)
        ],
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload))
    return path


def _candidate_and_protocol(tmp_path: Path):
    candidate = dict(candidate_id="cand-1", artifacts=[], views=[])
    protocol = dict(
        benchmark_id="bench-1",
        phase="development",
        checks=[
            dict(check_id="c1", track="appearance", metric="m1", op=">=", limit=0.5, required=True),
            dict(check_id="c2", track="coverage", metric="m2", op="<=", limit=0.5, required=False),
        ],
    )
    cand_path = tmp_path / "candidate.json"
    prot_path = tmp_path / "protocol.json"
    cand_path.write_text(json.dumps(candidate))
    prot_path.write_text(json.dumps(protocol))
    return cand_path, prot_path


def test_render_worker_emits_receipt_with_synthetic_marker(tmp_path):
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(RENDER_WORKER), "--request", str(_render_request(tmp_path)), "--out", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["synthetic_fixture"] is True
    assert "NOT count as real rendering" in receipt["banner"]
    assert receipt["completed"] == receipt["expected"] == 2
    for view in receipt["views"]:
        assert (out / view["rgb_file"]).exists()
        assert view["rgb_sha256"] != "0" * 64
        assert view["depth_sha256"] is not None


def test_render_worker_fail_at_view_exits_nonzero_without_receipt(tmp_path):
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable, str(RENDER_WORKER),
            "--request", str(_render_request(tmp_path)),
            "--out", str(out),
            "--fail-at-view", "1",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 7
    assert not (out / "receipt.json").exists()
    assert "injected_failure" in result.stdout


def test_render_worker_timeout_after_exits_124(tmp_path):
    # Many views + zero timeout: the second iteration sees elapsed > 0.
    req = _render_request(tmp_path, views=50)
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable, str(RENDER_WORKER),
            "--request", str(req),
            "--out", str(out),
            "--timeout-after", "0",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 124
    assert "timeout" in result.stdout


def test_render_worker_corrupt_hash_produces_zero_digest(tmp_path):
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable, str(RENDER_WORKER),
            "--request", str(_render_request(tmp_path)),
            "--out", str(out),
            "--corrupt-hash",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    receipt = json.loads((out / "receipt.json").read_text())
    assert all(v["rgb_sha256"] == "0" * 64 for v in receipt["views"])


def test_scoring_worker_emits_verdict_with_synthetic_marker(tmp_path):
    cand, prot = _candidate_and_protocol(tmp_path)
    out = tmp_path / "score"
    result = subprocess.run(
        [
            sys.executable, str(SCORING_WORKER),
            "--candidate", str(cand),
            "--protocol", str(prot),
            "--out", str(out),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    scorecard = json.loads((out / "scorecard.json").read_text())
    assert scorecard["synthetic_fixture"] is True
    assert "NOT count as real rendering or real scoring" in scorecard["banner"]
    assert scorecard["verdict"] in {"PASS", "FAIL"}
    assert len(scorecard["checks"]) == 2


def test_scoring_worker_empty_protocol_is_incomplete(tmp_path):
    cand = tmp_path / "candidate.json"
    prot = tmp_path / "protocol.json"
    cand.write_text(json.dumps({"candidate_id": "c"}))
    prot.write_text(json.dumps({"benchmark_id": "b", "checks": []}))
    out = tmp_path / "score"
    subprocess.run(
        [
            sys.executable, str(SCORING_WORKER),
            "--candidate", str(cand),
            "--protocol", str(prot),
            "--out", str(out),
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    scorecard = json.loads((out / "scorecard.json").read_text())
    assert scorecard["verdict"] == "INCOMPLETE"


def test_scoring_worker_deterministic_per_candidate(tmp_path):
    cand, prot = _candidate_and_protocol(tmp_path)
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    for out in (out1, out2):
        subprocess.run(
            [
                sys.executable, str(SCORING_WORKER),
                "--candidate", str(cand),
                "--protocol", str(prot),
                "--out", str(out),
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )
    s1 = json.loads((out1 / "scorecard.json").read_text())
    s2 = json.loads((out2 / "scorecard.json").read_text())
    assert [c["synthetic_score"] for c in s1["checks"]] == [c["synthetic_score"] for c in s2["checks"]]
