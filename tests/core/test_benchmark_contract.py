"""Positive and negative coverage for the frozen benchmark input contracts."""

import copy
import json
from pathlib import Path

import pytest

from simready.core import benchmark_contract as contract
from simready.core.artifacts import digest


def _write(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": str(path), "sha256": digest(path)}


def _ref(root, name, content=b"x"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": name, "sha256": digest(path)}


def _split(status="claimed", evidence=None):
    return {
        "status": status,
        "access_evidence": evidence,
        "fitted_components": ["reconstruction"],
    }


def _dataset(root, **overrides):
    manifest = {
        "schema_version": 1,
        "kind": "dataset",
        "dataset_id": "fixture",
        "version": "1.0",
        "source": {"type": "local", "reference": "fixture"},
        "license": "CC0-1.0",
        "scenes": [
            {
                "scene_id": "scene-0",
                "captures": [
                    {
                        "capture_id": "cap-0",
                        "frames": [
                            {
                                "frame_id": "fit-0",
                                "split": "fit",
                                "image": _ref(root, "fit0.png", b"fit0"),
                            },
                            {
                                "frame_id": "dev-0",
                                "split": "development",
                                "image": _ref(root, "dev0.png", b"dev0"),
                                "depth": _ref(root, "dev0.npy", b"depth0"),
                            },
                        ],
                    }
                ],
            }
        ],
        "split_access": {
            "fit": _split(),
            "development": _split(),
            "sealed": _split(),
        },
    }
    manifest.update(overrides)
    return manifest


def _camera(root, name="cam.json"):
    return _ref(root, name, b"camera")


def _view_set(root, **overrides):
    manifest = {
        "schema_version": 1,
        "kind": "view_set",
        "view_set_id": "views-0",
        "coordinate_frame": "opencv",
        "unit": "m",
        "views": [
            {
                "view_id": "view-0",
                "resolution": {"width": 64, "height": 48},
                "intrinsics": [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]],
                "camera_to_world": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "distortion": {"model": "none", "coefficients": []},
                "preprocessing": "none",
                "camera_evidence": _camera(root),
            }
        ],
    }
    manifest.update(overrides)
    return manifest


def _render_request(root, **overrides):
    manifest = {
        "schema_version": 1,
        "kind": "render_request",
        "request_id": "render-0",
        "scene": _ref(root, "scene.usd", b"scene"),
        "reconstruction": _ref(root, "model.ply", b"model"),
        "view_set": _ref(root, "views.json", json.dumps(_view_set(root)).encode()),
        "renderer": {
            "name": "gaussian-usd",
            "version": "0.1",
            "code_sha256": "a" * 64,
            "environment": {"image": "isaac:6.1.0", "python": "3.10"},
        },
        "image_settings": {"width": 64, "height": 48, "mode": "rgbd"},
        "lighting": {"policy": "frozen", "exposure": 1.0, "seed": 7},
        "seed": 13,
        "budget": {"max_wall_seconds": 600},
    }
    manifest.update(overrides)
    return manifest


def _evaluation_request(root, **overrides):
    render = _render_request(root)
    path = root / "render.json"
    path.write_text(json.dumps(render))
    manifest = {
        "schema_version": 1,
        "kind": "evaluation_request",
        "request_id": "eval-0",
        "render_request": {"path": "render.json", "sha256": digest(path)},
        "protocol": {"name": "reconstruction", "version": "1", "sha256": "b" * 64},
        "candidate_source": {"mode": "external", "reference": _ref(root, "cand.bin", b"c")},
        "scorer": {"name": "pixel", "version": "1", "code_sha256": "c" * 64},
        "deadline": "2026-12-31",
    }
    manifest.update(overrides)
    return manifest


def test_dataset_accepts_minimal_valid_manifest(tmp_path):
    assert contract.validate_dataset(_dataset(tmp_path), tmp_path)["kind"] == "dataset"


def test_dataset_rejects_unknown_field(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["extra"] = True
    with pytest.raises(contract.ContractError, match="Unknown dataset fields"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_rejects_missing_version(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["schema_version"] = 2
    with pytest.raises(contract.ContractError, match="schema_version"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_rejects_duplicate_frame_id(tmp_path):
    manifest = _dataset(tmp_path)
    frame = copy.deepcopy(manifest["scenes"][0]["captures"][0]["frames"][0])
    manifest["scenes"][0]["captures"][0]["frames"].append(frame)
    with pytest.raises(contract.ContractError, match="Duplicate frame id"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_rejects_missing_image_file(tmp_path):
    manifest = _dataset(tmp_path)
    (tmp_path / "fit0.png").unlink()
    with pytest.raises(contract.ContractError, match="Missing frame image file"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_rejects_path_escape(tmp_path):
    manifest = _dataset(tmp_path)
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"escape")
    manifest["scenes"][0]["captures"][0]["frames"][0]["image"] = {
        "path": "../outside.bin",
        "sha256": digest(outside),
    }
    with pytest.raises(contract.ContractError, match="escapes the manifest root"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_rejects_hash_mismatch(tmp_path):
    manifest = _dataset(tmp_path)
    (tmp_path / "fit0.png").write_bytes(b"tampered")
    with pytest.raises(contract.ContractError, match="Hash mismatch"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_rejects_audited_split_without_evidence(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["split_access"]["sealed"] = _split(status="audited")
    with pytest.raises(contract.ContractError, match="audited split"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_accepts_audited_split_with_evidence(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["split_access"]["sealed"] = _split(
        status="audited",
        evidence={
            "restricted_inputs": ["sealed-bucket"],
            "access_log": "s3://audit/log.json",
        },
    )
    assert contract.validate_dataset(manifest, tmp_path)


def test_dataset_rejects_texture_fitting_component_typo(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["split_access"]["fit"]["fitted_components"] = ["texture-fitting"]
    with pytest.raises(contract.ContractError, match="unknown fitted components"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_rejects_derived_frame_pointing_at_unknown_capture(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["scenes"][0]["captures"][0]["frames"][0]["derived_from_capture"] = "cap-x"
    with pytest.raises(contract.ContractError, match="unknown"):
        contract.validate_dataset(manifest, tmp_path)


def test_dataset_accepts_derived_fit_frames_with_capture_mapping(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["scenes"][0]["captures"][0]["frames"][0]["derived_from_capture"] = "cap-0"
    assert contract.validate_dataset(manifest, tmp_path)


def test_view_set_accepts_valid_manifest(tmp_path):
    assert contract.validate_view_set(_view_set(tmp_path), tmp_path)["kind"] == "view_set"


def test_view_set_rejects_duplicate_view_id(tmp_path):
    manifest = _view_set(tmp_path)
    manifest["views"].append(copy.deepcopy(manifest["views"][0]))
    with pytest.raises(contract.ContractError, match="Duplicate view id"):
        contract.validate_view_set(manifest, tmp_path)


def test_view_set_rejects_nan_intrinsics(tmp_path):
    manifest = _view_set(tmp_path)
    manifest["views"][0]["intrinsics"][0][0] = float("nan")
    with pytest.raises(contract.ContractError, match="finite"):
        contract.validate_view_set(manifest, tmp_path)


def test_view_set_rejects_distortion_coefficients_with_none_model(tmp_path):
    manifest = _view_set(tmp_path)
    manifest["views"][0]["distortion"] = {"model": "none", "coefficients": [0.1]}
    with pytest.raises(contract.ContractError, match="must not carry coefficients"):
        contract.validate_view_set(manifest, tmp_path)


def test_view_set_rejects_bad_camera_evidence_hash(tmp_path):
    manifest = _view_set(tmp_path)
    (tmp_path / "cam.json").write_bytes(b"tampered")
    with pytest.raises(contract.ContractError, match="Hash mismatch"):
        contract.validate_view_set(manifest, tmp_path)


def test_render_request_accepts_valid_manifest(tmp_path):
    assert (
        contract.validate_render_request(_render_request(tmp_path), tmp_path)["kind"]
        == "render_request"
    )


def test_render_request_rejects_inf_exposure(tmp_path):
    manifest = _render_request(tmp_path)
    manifest["lighting"]["exposure"] = float("inf")
    with pytest.raises(contract.ContractError, match="finite"):
        contract.validate_render_request(manifest, tmp_path)


def test_render_request_rejects_frozen_lighting_without_seed(tmp_path):
    manifest = _render_request(tmp_path)
    manifest["lighting"] = {"policy": "frozen", "exposure": 1.0}
    with pytest.raises(contract.ContractError, match="requires an explicit seed"):
        contract.validate_render_request(manifest, tmp_path)


def test_render_request_rejects_negative_budget(tmp_path):
    manifest = _render_request(tmp_path)
    manifest["budget"]["max_wall_seconds"] = -1
    with pytest.raises(contract.ContractError, match=">="):
        contract.validate_render_request(manifest, tmp_path)


def test_evaluation_request_accepts_valid_manifest(tmp_path):
    assert (
        contract.validate_evaluation_request(_evaluation_request(tmp_path), tmp_path)[
            "kind"
        ]
        == "evaluation_request"
    )


def test_evaluation_request_rejects_external_candidate_without_reference(tmp_path):
    manifest = _evaluation_request(tmp_path)
    manifest["candidate_source"] = {"mode": "external"}
    with pytest.raises(contract.ContractError, match="requires a frozen reference"):
        contract.validate_evaluation_request(manifest, tmp_path)


def test_evaluation_request_rejects_auto_candidate_with_reference(tmp_path):
    manifest = _evaluation_request(tmp_path)
    manifest["candidate_source"] = {
        "mode": "auto",
        "reference": _ref(tmp_path, "cand.bin", b"c"),
    }
    with pytest.raises(contract.ContractError, match="must not bind"):
        contract.validate_evaluation_request(manifest, tmp_path)


def test_evaluation_request_rejects_render_request_hash_mismatch(tmp_path):
    manifest = _evaluation_request(tmp_path)
    (tmp_path / "render.json").write_text("{}")
    with pytest.raises(contract.ContractError, match="Hash mismatch"):
        contract.validate_evaluation_request(manifest, tmp_path)


def test_validate_manifest_dispatches_and_rejects_unknown_kind(tmp_path):
    with pytest.raises(contract.ContractError, match="Unknown benchmark manifest kind"):
        contract.validate_manifest({"kind": "other"}, tmp_path)
    assert (
        contract.validate_manifest(_dataset(tmp_path), tmp_path)["kind"] == "dataset"
    )
