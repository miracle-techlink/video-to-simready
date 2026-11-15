"""Direct coverage for the frozen reconstruction bundle contract."""

import copy
import json

import pytest

from simready.core.artifacts import digest
from simready.core.reconstruction_contract import checked_ref, declarations, fields, verify_bundle


def _ref(root, name, content=b"x"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": name, "sha256": digest(path)}


def _bundle(root):
    primary = _ref(root, "model.ply", b"ply")
    primary_size = (root / "model.ply").stat().st_size
    return {
        "schema_version": 1,
        "artifact_type": "reconstruction_bundle",
        "status": "COMPLETE",
        "representation": "gaussian_ply",
        "source_video_sha256": "a" * 64,
        "algorithm": {
            "name": "gsplat",
            "version": "1.0",
            "implementation_sha256": None,
        },
        "frames": {
            "status": "declared",
            "training": [{"id": "f0", "sha256": "b" * 64}],
            "heldout": [{"id": "f1", "sha256": "c" * 64}],
            "evidence": _ref(root, "frames.json", b"frames"),
        },
        "coordinates": {
            "frame": "opencv",
            "unit": "m",
            "scale_status": "declared",
            "evidence": _ref(root, "coords.json", b"coords"),
        },
        "artifacts": [
            {
                "role": "primary",
                "kind": "ply",
                "path": primary["path"],
                "sha256": primary["sha256"],
                "bytes": primary_size,
            }
        ],
        "provenance": {
            "provider": "command",
            "execution_mode": "fresh_command",
            "config_sha256": "d" * 64,
            "dependencies": [],
        },
        "declaration_verification": "operator_evidence_not_independent_certification",
    }


def _write_bundle(root, bundle):
    path = root / "bundle.json"
    path.write_text(json.dumps(bundle))
    return path


def test_verify_bundle_accepts_minimal_valid_bundle(tmp_path):
    path = _write_bundle(tmp_path, _bundle(tmp_path))
    assert verify_bundle(path)["artifact_type"] == "reconstruction_bundle"


def test_verify_bundle_rejects_unknown_top_level_field(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["unexpected"] = True
    with pytest.raises(ValueError, match="Unknown or invalid reconstruction bundle fields"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_wrong_schema_version(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["schema_version"] = 2
    with pytest.raises(ValueError, match="reconstruction_bundle v1"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_overlapping_training_heldout(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["frames"]["training"].append({"id": "shared", "sha256": "e" * 64})
    bundle["frames"]["heldout"].append({"id": "shared", "sha256": "e" * 64})
    with pytest.raises(ValueError, match="overlapping training/heldout"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_same_hash_in_both_splits(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["frames"]["training"].append({"id": "t2", "sha256": "c" * 64})
    with pytest.raises(ValueError, match="overlapping training/heldout"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_declared_frames_without_evidence(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["frames"]["evidence"] = None
    with pytest.raises(ValueError, match="require evidence"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_unknown_status_claiming_frames(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["frames"]["status"] = "unknown"
    bundle["frames"]["evidence"] = None
    with pytest.raises(ValueError, match="must not claim frames"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_missing_evidence_file(tmp_path):
    bundle = _bundle(tmp_path)
    (tmp_path / "frames.json").unlink()
    with pytest.raises(ValueError, match="missing or hash changed"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_tampered_artifact(tmp_path):
    bundle = _bundle(tmp_path)
    (tmp_path / "model.ply").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="missing or hash changed"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_artifact_outside_bundle_directory(tmp_path):
    bundle = _bundle(tmp_path)
    outside = tmp_path.parent / "outside.ply"
    outside.write_bytes(b"escape")
    bundle["artifacts"][0]["path"] = "../outside.ply"
    bundle["artifacts"][0]["sha256"] = digest(outside)
    bundle["artifacts"][0]["bytes"] = outside.stat().st_size
    with pytest.raises(ValueError, match="inside the bundle directory"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_absolute_artifact_path(tmp_path):
    bundle = _bundle(tmp_path)
    absolute = tmp_path / "model.ply"
    bundle["artifacts"][0]["path"] = str(absolute)
    with pytest.raises(ValueError, match="inside the bundle directory"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_duplicate_artifact_paths(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["artifacts"].append(copy.deepcopy(bundle["artifacts"][0]))
    bundle["artifacts"][1]["role"] = "supporting"
    with pytest.raises(ValueError, match="unique"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_byte_count_drift(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["artifacts"][0]["bytes"] += 1
    with pytest.raises(ValueError, match="byte count changed"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_provider_mode_contradiction(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["provenance"]["provider"] = "artifact"
    bundle["provenance"]["execution_mode"] = "cloud_new"
    with pytest.raises(ValueError, match="contradicts its provider"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_non_gaussian_primary_kind(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["artifacts"][0]["kind"] = "usd"
    with pytest.raises(ValueError, match="matching its representation"):
        verify_bundle(_write_bundle(tmp_path, bundle))


def test_verify_bundle_rejects_source_video_mismatch(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="another source video"):
        verify_bundle(_write_bundle(tmp_path, bundle), source_video_sha256="f" * 64)


def test_fields_rejects_missing_required():
    with pytest.raises(ValueError, match="Missing .* fields"):
        fields({"a": 1}, {"a", "b"}, "thing", {"a", "b"})


def test_checked_ref_rejects_non_sha_digest(tmp_path):
    (tmp_path / "x").write_bytes(b"x")
    with pytest.raises(ValueError, match="lowercase SHA256"):
        checked_ref({"path": "x", "sha256": "zz"}, tmp_path)


def test_declarations_reject_verified_scale_without_metric_units(tmp_path):
    evidence_file = tmp_path / "frames.json"
    evidence_file.write_bytes(b"frames")
    with pytest.raises(ValueError, match="Verified scale requires metric units"):
        declarations(
            {"name": "n", "version": "1", "implementation_sha256": None},
            {
                "status": "declared",
                "training": [{"id": "f", "sha256": "a" * 64}],
                "heldout": [],
                "evidence": {"path": "frames.json", "sha256": digest(evidence_file)},
            },
            {"frame": "opencv", "unit": "arbitrary", "scale_status": "verified",
             "evidence": None},
            tmp_path,
        )
