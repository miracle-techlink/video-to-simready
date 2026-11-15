"""Multi-scene benchmark suite: 2 scenes × 2 members, failure injection, aggregation."""

import runpy

import pytest

from simready.core.artifacts import read_json, write_json
from simready.core.evidence import reference
from simready.execution.processes.local import ProcessCancelledError
from simready.workflows.pipelines import baseline_matrix as matrix
from simready.workflows.pipelines import benchmark_suite as suite
from tests.paths import REPOSITORY_ROOT


@pytest.fixture
def scene_fixtures(tmp_path):
    """Create two independent scene fixture directories, each with a matrix config."""
    prepare = runpy.run_path(
        str(REPOSITORY_ROOT / "examples/reconstruction/prepare_fixture.py")
    )["prepare"]
    scenes = {}
    for name in ("scene-a", "scene-b"):
        path = prepare(tmp_path / "inputs" / name)
        scenes[name] = {"matrix": path, "root": tmp_path / "inputs" / name}
    return scenes


def _suite_config(tmp_path, scene_fixtures):
    scenes = [
        {"scene_id": sid, "matrix": reference(cfg["matrix"])}
        for sid, cfg in scene_fixtures.items()
    ]
    path = tmp_path / "suite.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "kind": "benchmark_suite",
            "suite_id": "test-suite",
            "scenes": scenes,
            "evaluation": {"timeout_s": 60},
        },
    )
    return path


def test_suite_plan_freezes_all_scene_matrices(scene_fixtures, tmp_path):
    config = _suite_config(tmp_path, scene_fixtures)
    plan = suite.plan_suite(config, tmp_path / "out")
    assert plan["identity"]
    assert len(plan["scenes"]) == 2
    assert len(plan["frozen_references"]) == 3  # suite + 2 matrices
    assert not (tmp_path / "out").exists()


def test_suite_dry_run_is_read_only(scene_fixtures, tmp_path):
    config = _suite_config(tmp_path, scene_fixtures)
    suite.plan_suite(config, tmp_path / "out")
    assert (
        suite.main(
            ["--suite", str(config), "--out", str(tmp_path / "out"), "--dry-run"]
        )
        == 0
    )
    assert not (tmp_path / "out").exists()


def test_suite_executes_two_scenes_and_aggregates(scene_fixtures, tmp_path):
    config = _suite_config(tmp_path, scene_fixtures)
    plan = suite.plan_suite(config, tmp_path / "out")
    report = suite.execute_suite(plan)
    assert report["status"] == "COMPLETE"
    assert len(report["scenes"]) == 2
    agg = report["aggregation"]
    assert agg["cross_scene"]["total_planned"] == 4
    assert agg["cross_scene"]["total_executed"] == 4
    assert agg["cross_scene"]["total_failed"] == 0
    # Both scenes use synthetic fixtures → no real quality measurement.
    assert agg["cross_scene"]["total_quality_not_evaluated"] == 4
    assert agg["cross_scene"]["mean_pass_rate"] == 0
    assert len(agg["rows"]) == 4
    for row in agg["rows"]:
        assert row["execution_status"] == "STAGE_READY"
        assert row["quality_status"] == "NOT_EVALUATED"
        assert row["candidate_mode"] == "external"


def test_suite_member_failure_retains_all_four_rows(
    scene_fixtures, tmp_path, monkeypatch
):
    """Inject a failure in one member of one scene; all 4 rows must remain."""
    config = _suite_config(tmp_path, scene_fixtures)
    plan = suite.plan_suite(config, tmp_path / "out")

    original_execute = matrix.execute_video
    calls = []

    def fail_first(args, plan_dict):
        calls.append(plan_dict["out"])
        if "scene-a" in plan_dict["out"] and "synthetic_cpu" in plan_dict["out"]:
            return {"status": "FAILED", "error": "injected failure"}, 2
        return original_execute(args, plan_dict)

    monkeypatch.setattr(matrix, "execute_video", fail_first)
    report = suite.execute_suite(plan)
    assert report["status"] == "COMPLETE"
    agg = report["aggregation"]
    assert agg["cross_scene"]["total_planned"] == 4
    assert agg["cross_scene"]["total_failed"] == 1
    assert agg["cross_scene"]["total_executed"] == 4
    # Per-scene breakdown.
    scene_a = next(s for s in agg["per_scene"] if s["scene_id"] == "scene-a")
    scene_b = next(s for s in agg["per_scene"] if s["scene_id"] == "scene-b")
    assert scene_a["failed"] == 1
    assert scene_a["failure_rate"] == 0.5
    assert scene_b["failed"] == 0
    assert scene_b["failure_rate"] == 0.0
    assert len(agg["rows"]) == 4
    failed_row = next(r for r in agg["rows"] if r["execution_status"] == "FAILED")
    assert failed_row["error"]["message"] == "injected failure"


def test_suite_cancel_stops_downstream_and_preserves_rows(
    scene_fixtures, tmp_path, monkeypatch
):
    config = _suite_config(tmp_path, scene_fixtures)
    plan = suite.plan_suite(config, tmp_path / "out")

    original_execute = matrix.execute_video
    calls = []

    def cancel_first(args, plan_dict):
        calls.append(plan_dict["out"])
        if len(calls) == 1:
            raise ProcessCancelledError(0)
        return original_execute(args, plan_dict)

    monkeypatch.setattr(matrix, "execute_video", cancel_first)
    with pytest.raises(ProcessCancelledError):
        suite.execute_suite(plan)
    report = read_json(tmp_path / "out" / "suite-result.json")
    assert report["status"] == "CANCELLED"
    # First scene was attempted; second scene may or may not have started,
    # but aggregation must only count planned samples.
    assert report["aggregation"]["cross_scene"]["total_planned"] == 4


def test_suite_result_never_ranks_or_promotes(scene_fixtures, tmp_path):
    config = _suite_config(tmp_path, scene_fixtures)
    report = suite.execute_suite(suite.plan_suite(config, tmp_path / "out"))
    assert report["ranking"] is None
    assert report["default_provider_promotion"] is None


def test_suite_aggregation_missing_quality_not_filled(
    scene_fixtures, tmp_path, monkeypatch
):
    """A member with no quality evidence must NOT have its missing metric filled with zero."""
    config = _suite_config(tmp_path, scene_fixtures)
    plan = suite.plan_suite(config, tmp_path / "out")
    report = suite.execute_suite(plan)
    agg = report["aggregation"]
    # All members are synthetic fixtures → quality_not_evaluated = 4, measured = 0.
    assert agg["cross_scene"]["total_quality_measured"] == 0
    assert agg["cross_scene"]["total_quality_not_evaluated"] == 4
    for row in agg["rows"]:
        assert row["quality_status"] == "NOT_EVALUATED"
        assert row["verifier_status"] is None
        assert row["missing_quality"] is True
        # No artifact existed, so no artifact path may be fabricated.
        assert row["candidate_path"] is None
        assert row["scorecard_path"] is None
        assert row["receipt_path"] is None


def test_suite_output_must_not_contain_frozen_inputs(scene_fixtures, tmp_path):
    config = _suite_config(tmp_path, scene_fixtures)
    suite.plan_suite(config, tmp_path / "out")
    # Place a frozen input inside the output directory.
    (tmp_path / "out").mkdir(parents=True)
    (tmp_path / "out" / "suite.json").write_text((tmp_path / "suite.json").read_text())
    with pytest.raises(ValueError, match="must not contain frozen inputs"):
        suite.plan_suite(tmp_path / "out" / "suite.json", tmp_path / "out")


def test_suite_rejects_unknown_scene_id_duplicates(scene_fixtures, tmp_path):
    config = _suite_config(tmp_path, scene_fixtures)
    doc = read_json(config)
    doc["scenes"][1]["scene_id"] = doc["scenes"][0]["scene_id"]
    write_json(config, doc)
    with pytest.raises(ValueError, match="unique"):
        suite.plan_suite(config, tmp_path / "out")
