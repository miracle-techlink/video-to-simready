"""Protocol schema validation: version compat, unknown protocol rejection, threshold separation."""

import pytest

from simready.core.artifacts import digest
from simready.evaluation.protocol import validate_candidate, validate_protocol
from tests.fixtures.scene_candidate import make_scene_candidate


class TestValidateProtocol:
    def test_valid_protocol_accepted(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        result = validate_protocol(protocol)
        assert result["benchmark_id"] == "fault-fixture"

    def test_wrong_schema_version_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["schema_version"] = 2
        with pytest.raises(ValueError, match="[Ss]chema"):
            validate_protocol(protocol)

    def test_unknown_profile_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["profile"] = "unknown_profile"
        with pytest.raises(ValueError, match="[Pp]rofile"):
            validate_protocol(protocol)

    def test_invalid_phase_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["phase"] = "production"
        with pytest.raises(ValueError, match="[Pp]hase"):
            validate_protocol(protocol)

    def test_empty_views_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["views"] = []
        with pytest.raises(ValueError, match="[Ee]mpty"):
            validate_protocol(protocol)

    def test_empty_checks_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["checks"] = []
        with pytest.raises(ValueError, match="[Ee]mpty"):
            validate_protocol(protocol)

    def test_no_required_checks_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        for check in protocol["checks"]:
            check["required"] = False
        with pytest.raises(ValueError, match="[Nn]o required"):
            validate_protocol(protocol)

    def test_missing_required_tracks_rejected(self, tmp_path):
        """Simready profile requires coverage, appearance, source, geometry,
        instances, physics, tasks, delivery tracks."""
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["profile"] = "simready"
        with pytest.raises(ValueError, match="[Mm]issing required tracks"):
            validate_protocol(protocol)

    def test_fit_eval_frame_overlap_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["fit_frame_ids"] = ["test-frame"]  # same as view frame_id
        with pytest.raises(ValueError, match="overlap"):
            validate_protocol(protocol)

    def test_fit_eval_image_overlap_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["fit_image_sha256"] = [protocol["views"][0]["rgb"]["sha256"]]
        with pytest.raises(ValueError, match="overlap"):
            validate_protocol(protocol)

    def test_duplicate_view_ids_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["views"] = protocol["views"] * 2
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            validate_protocol(protocol)

    def test_duplicate_source_frames_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        view2 = dict(protocol["views"][0])
        view2["view_id"] = "different-id"
        protocol["views"].append(view2)
        with pytest.raises(ValueError, match="[Dd]uplicate source"):
            validate_protocol(protocol)

    def test_invalid_check_operator_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["checks"][0]["op"] = "!="
        with pytest.raises(ValueError, match="[Oo]perator"):
            validate_protocol(protocol)

    def test_invalid_severity_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["checks"][0]["severity"] = "catastrophic"
        with pytest.raises(ValueError, match="[Ss]everity"):
            validate_protocol(protocol)

    def test_unknown_calibration_status_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["checks"][0]["calibration"] = "unknown"
        with pytest.raises(ValueError, match="[Cc]alibration"):
            validate_protocol(protocol)

    def test_nonfinite_limit_rejected(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["checks"][0]["limit"] = float("nan")
        with pytest.raises(ValueError, match="[Ff]inite"):
            validate_protocol(protocol)

    def test_null_limit_accepted_for_pending_calibration(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["checks"][1]["limit"] = None
        protocol["checks"][1]["calibration"] = "pending"
        result = validate_protocol(protocol)
        assert result is not None

    def test_coverage_check_required(self, tmp_path):
        """All declared observations must have a coverage check requiring 100%."""
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        protocol["checks"] = [
            c
            for c in protocol["checks"]
            if c["metric"] != "coverage.observation_fraction"
        ]
        with pytest.raises(ValueError, match="[Mm]issing required tracks|[Cc]overage"):
            validate_protocol(protocol)


class TestValidateCandidate:
    def test_valid_candidate_accepted(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        result = validate_candidate(candidate, digest(p), protocol)
        assert result["candidate_id"] == "fixture"

    def test_protocol_hash_mismatch_rejected(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        with pytest.raises(ValueError, match="binding mismatch"):
            validate_candidate(candidate, "0" * 64, protocol)

    def test_undeclared_view_rejected(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        candidate["views"].append(
            {
                "view_id": "undeclared-view",
                "rgb": {"path": "x.png", "sha256": "a" * 64},
            }
        )
        with pytest.raises(ValueError, match="[Uu]ndeclared"):
            validate_candidate(candidate, digest(p), protocol)

    def test_fit_frame_overlap_rejected(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        candidate["fit_frame_ids"] = ["test-frame"]
        with pytest.raises(ValueError, match="overlap"):
            validate_candidate(candidate, digest(p), protocol)

    def test_fit_image_overlap_rejected(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        candidate["fit_image_sha256"] = [protocol["views"][0]["rgb"]["sha256"]]
        with pytest.raises(ValueError, match="overlap"):
            validate_candidate(candidate, digest(p), protocol)

    def test_empty_artifacts_rejected(self, tmp_path):
        p, c, protocol, candidate, _ = make_scene_candidate(tmp_path)
        candidate["artifacts"] = []
        with pytest.raises(ValueError, match="[Nn]o artifacts"):
            validate_candidate(candidate, digest(p), protocol)


class TestThresholdRenderSeparation:
    """Thresholds are part of the protocol, not the render configuration.

    Changing a threshold should only invalidate scoring, not rendering.
    This is a structural test: the protocol schema has checks with limits,
    while the render request schema (in benchmark_contract) has no threshold
    fields. The separation is enforced by the schema definitions.
    """

    def test_protocol_contains_thresholds(self, tmp_path):
        _, _, protocol, _, _ = make_scene_candidate(tmp_path)
        assert any("limit" in c for c in protocol["checks"])

    def test_render_request_has_no_thresholds(self):
        import inspect

        from simready.core.benchmark_contract import validate_render_request

        source = inspect.getsource(validate_render_request)
        # The render request validator should not reference thresholds or limits
        # in the context of quality checks.
        assert "check" not in source.lower() or "threshold" not in source.lower()
