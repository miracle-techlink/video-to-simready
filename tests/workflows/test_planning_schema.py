"""pipeline_spec schema evolution and unknown-version rejection."""

import sys

import pytest

from simready.core.artifacts import digest, read_json, write_json
from simready.workflows.planning.pipeline_spec import load_pipeline


def _minimal_pipeline(tmp_path, **overrides):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    raw = {
        "schema_version": 1,
        "input": {"video": str(video)},
        "stages": {
            "reconstruction": {
                "provider": "command",
                "base_dir": str(tmp_path),
                "algorithm": {
                    "name": "synthetic",
                    "version": "1",
                    "implementation_sha256": digest(tmp_path / "video.mp4"),
                },
                "argv": ["{python}", "-c", "pass"],
                "input_files": {},
                "runtime_identity": {
                    "path": str(tmp_path / "runtime.json"),
                    "sha256": digest(tmp_path / "video.mp4"),
                },
                "environment": {"inherit": [], "secrets": []},
                "resources": {"kind": "cpu"},
                "frames": {
                    "status": "declared",
                    "training": [{"id": "t", "sha256": "1" * 64}],
                    "heldout": [{"id": "h", "sha256": "2" * 64}],
                },
                "coordinates": {
                    "frame": "z_up",
                    "unit": "m",
                    "scale_status": "verified",
                },
            },
        },
        "execution": {"backend": "local", "python": sys.executable},
    }
    raw.update(overrides)
    path = tmp_path / "pipeline.json"
    write_json(path, raw)
    return path


def test_unknown_top_level_field_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path, unknown_field=True)
    with pytest.raises(ValueError, match="Unknown or malformed pipeline fields"):
        load_pipeline(path, tmp_path / "out")


def test_unknown_stage_field_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path)
    doc = read_json(path)
    doc["stages"]["unknown_stage"] = {}
    write_json(path, doc)
    with pytest.raises(ValueError, match="Unknown or malformed stages fields"):
        load_pipeline(path, tmp_path / "out")


def test_unknown_observation_provider_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path)
    doc = read_json(path)
    doc["stages"]["observations"] = {"provider": "unknown"}
    write_json(path, doc)
    with pytest.raises(ValueError, match="Unregistered observation provider"):
        load_pipeline(path, tmp_path / "out")


def test_unknown_delivery_provider_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path)
    doc = read_json(path)
    doc["stages"]["delivery"] = {"provider": "unknown"}
    write_json(path, doc)
    with pytest.raises(ValueError, match="Unsupported delivery provider"):
        load_pipeline(path, tmp_path / "out")


def test_unknown_backend_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path)
    doc = read_json(path)
    doc["execution"]["backend"] = "kubernetes"
    write_json(path, doc)
    with pytest.raises(ValueError, match="Pipeline backend must be local or docker"):
        load_pipeline(path, tmp_path / "out")


def test_missing_schema_version_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path)
    doc = read_json(path)
    del doc["schema_version"]
    write_json(path, doc)
    with pytest.raises(ValueError, match="Expected pipeline schema_version=1"):
        load_pipeline(path, tmp_path / "out")


def test_schema_version_2_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path, schema_version=2)
    with pytest.raises(ValueError, match="Expected pipeline schema_version=1"):
        load_pipeline(path, tmp_path / "out")


def test_schema_version_string_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path, schema_version="1")
    with pytest.raises(ValueError, match="Expected pipeline schema_version=1"):
        load_pipeline(path, tmp_path / "out")


def test_pipeline_path_inside_output_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    nested = out / "pipeline.json"
    nested.write_text(path.read_text())
    with pytest.raises(ValueError, match="Pipeline configuration must live outside"):
        load_pipeline(nested, out)


def test_docker_backend_without_environment_manifest_rejected(tmp_path):
    path = _minimal_pipeline(tmp_path)
    doc = read_json(path)
    doc["execution"]["backend"] = "docker"
    write_json(path, doc)
    with pytest.raises(
        ValueError,
        match="External worker environment requires a pinned environment_manifest",
    ):
        load_pipeline(path, tmp_path / "out")


def test_load_pipeline_is_deterministic(tmp_path):
    path = _minimal_pipeline(tmp_path)
    first = load_pipeline(path, tmp_path / "out")
    second = load_pipeline(path, tmp_path / "out")
    assert first == second
