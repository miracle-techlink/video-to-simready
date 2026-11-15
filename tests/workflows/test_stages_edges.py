"""Edge cases for semantic_stage and video_execution cancellation / recovery."""

import threading
from types import SimpleNamespace

import pytest

from simready.core.artifacts import digest, read_json, write_json
from simready.workflows.stages import semantic_stage, video_execution
from tests.fixtures.video_worker import worker_inputs

# ---------------------------------------------------------------------------
# video_execution edges
# ---------------------------------------------------------------------------


def test_video_execution_rejects_nonpositive_timeout(tmp_path):
    inputs = worker_inputs(tmp_path)
    with pytest.raises(ValueError, match="timeout_s"):
        video_execution.execute(inputs, tmp_path / "out", timeout_s=0)
    with pytest.raises(ValueError, match="timeout_s"):
        video_execution.execute(inputs, tmp_path / "out", timeout_s=-1)


def test_video_execution_worker_timeout_s_tightens_deadline(tmp_path):
    inputs = worker_inputs(tmp_path)
    source = read_json(inputs / "input.json")
    source["worker_timeout_s"] = 0.05
    write_json(inputs / "input.json", source)
    assert video_execution.execute(inputs, tmp_path / "out", timeout_s=7200) == 2
    status = read_json(tmp_path / "out" / "pipeline-status.json")
    assert status["status"] == "TIMEOUT"


def test_video_execution_cancel_before_first_step(tmp_path):
    event = threading.Event()
    event.set()
    inputs = worker_inputs(tmp_path)
    assert video_execution.execute(inputs, tmp_path / "out", cancel_event=event) == 2
    status = read_json(tmp_path / "out" / "pipeline-status.json")
    assert status["status"] == "CANCELLED"
    assert not (tmp_path / "out" / "scene").exists()


def test_video_execution_second_run_preserves_first_evidence(tmp_path, monkeypatch):
    inputs = worker_inputs(tmp_path)
    output = tmp_path / "out"
    monkeypatch.setattr(
        video_execution,
        "run_process",
        lambda *a, **k: pytest.fail("Old output reran"),
    )
    write_json(output / "pipeline-status.json", {"status": "COMPLETE"})
    before = (output / "pipeline-status.json").read_bytes()
    with pytest.raises(FileExistsError, match="Previous worker evidence"):
        video_execution.execute(inputs, output)
    assert (output / "pipeline-status.json").read_bytes() == before


def test_video_execution_inputs_changed_during_execution_is_failed(
    tmp_path, monkeypatch
):
    inputs = worker_inputs(tmp_path)
    output = tmp_path / "out"
    original_digest = digest

    def corrupting_digest(path):
        if path == inputs / "splats.ply":
            (inputs / "splats.ply").write_bytes(b"corrupted")
        return original_digest(path)

    monkeypatch.setattr(video_execution, "digest", corrupting_digest)
    monkeypatch.setattr(
        video_execution,
        "run_process",
        lambda *a, **k: SimpleNamespace(returncode=0),
    )
    assert video_execution.execute(inputs, output) == 2
    status = read_json(output / "pipeline-status.json")
    assert status["status"] == "FAILED"
    assert "changed" in status["error"].lower()


# ---------------------------------------------------------------------------
# semantic_stage edges
# ---------------------------------------------------------------------------


def test_plan_decomposition_rejects_unknown_provider(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    with pytest.raises(ValueError, match="Unregistered decomposition provider"):
        semantic_stage.plan_decomposition({"provider": "unknown"}, tmp_path, video)


def _decomposition_fixture(tmp_path, video):
    """Write minimal manifest/calibration/evidence files for plan_decomposition."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    calibration = tmp_path / "cal.json"
    calibration.write_text("{}")
    evidence = tmp_path / "ev.json"
    evidence.write_text("{}")
    return {
        "provider": "semantic_projection",
        "manifest": "manifest.json",
        "calibration": "cal.json",
        "association": {
            "source_video_sha256": digest(video),
            "reconstruction_sha256": "a" * 64,
            "reason": "test",
            "evidence": {"path": "ev.json", "sha256": digest(evidence)},
        },
    }


def test_plan_decomposition_rejects_missing_manifest(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    with pytest.raises(ValueError, match="Missing pipeline input"):
        semantic_stage.plan_decomposition(
            {
                "provider": "semantic_projection",
                "manifest": "missing.json",
                "calibration": "cal.json",
                "association": {
                    "source_video_sha256": digest(video),
                    "reconstruction_sha256": "a" * 64,
                    "reason": "test",
                    "evidence": {"path": "ev.json", "sha256": "b" * 64},
                },
            },
            tmp_path,
            video,
        )


def test_plan_decomposition_rejects_nonpositive_timeout(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    config = _decomposition_fixture(tmp_path, video)
    config["timeout_s"] = 0
    with pytest.raises(ValueError, match="decomposition timeout"):
        semantic_stage.plan_decomposition(config, tmp_path, video)


def test_plan_decomposition_rejects_invalid_thresholds(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    config = _decomposition_fixture(tmp_path, video)
    config["min_points"] = 0
    with pytest.raises(ValueError, match="Invalid semantic projection thresholds"):
        semantic_stage.plan_decomposition(config, tmp_path, video)
    config = _decomposition_fixture(tmp_path, video)
    config["min_score"] = 1.5
    with pytest.raises(ValueError, match="Invalid semantic projection thresholds"):
        semantic_stage.plan_decomposition(config, tmp_path, video)


def test_produce_decomposition_rejects_mismatched_reconstruction(tmp_path):
    plan = {
        "provider": "semantic_projection",
        "manifest": str(tmp_path / "manifest.json"),
        "calibration": str(tmp_path / "cal.json"),
        "association": {
            "source_video_sha256": "a" * 64,
            "reconstruction_sha256": "b" * 64,
            "reason": "test",
            "evidence": {"path": str(tmp_path / "ev.json"), "sha256": "c" * 64},
        },
        "min_points": 16,
        "min_score": 0.25,
        "timeout_s": 300,
        "dependencies": [],
    }
    with pytest.raises(ValueError, match="not associated with this reconstruction"):
        semantic_stage.produce_decomposition(
            plan, {"ply_sha256": "d" * 64}, {}, tmp_path / "out"
        )
