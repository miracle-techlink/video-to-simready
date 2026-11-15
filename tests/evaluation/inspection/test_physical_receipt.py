"""Physical receipt verification: hash binding, path safety, asset inventory."""

import json
from pathlib import Path

import pytest

from simready.core.artifacts import digest, write_json
from simready.evaluation.inspection.physical_receipt import verify_physical_receipt


def _make_receipt(tmp_path, objects=None, status="PASS", passed=True):
    """Create a minimal passing physical receipt with real files."""
    if objects is None:
        objects = [{"id": "box1", "type": "box"}]

    # Create required files.
    scene_path = tmp_path / "scene.usda"
    scene_path.write_text("#usda 1.0\n")
    catalog_path = tmp_path / "asset_catalog.json"
    write_json(catalog_path, {"schema_version": 1, "objects": objects})
    spec_path = tmp_path / "scene_spec.json"
    write_json(spec_path, {"schema_version": 1, "objects": objects})

    receipt = {
        "schema_version": 1,
        "artifact_type": "physical_asset_inspection",
        "status": status,
        "passed": passed,
        "checks": [{"id": "c1", "status": "PASS", "passed": True}],
        "inputs": {
            "scene": {"sha256": digest(scene_path)},
            "catalog": {"sha256": digest(catalog_path)},
            "scene_spec": {"sha256": digest(spec_path)},
        },
        "assets": [{"id": obj["id"], "status": "PASS", "passed": True} for obj in objects],
        "dependencies": [
            {"path": "scene.usda", "sha256": digest(scene_path)},
        ],
    }
    return receipt, spec_path


class TestVerifyPhysicalReceipt:
    def test_valid_receipt_passes(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_failed_status_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path, status="FAIL", passed=False)
        with pytest.raises(ValueError, match="did not pass"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_empty_checks_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        receipt["checks"] = []
        with pytest.raises(ValueError, match="nonempty"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_failing_check_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        receipt["checks"] = [{"id": "c1", "status": "FAIL", "passed": False}]
        with pytest.raises(ValueError, match="nonempty passing"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_tampered_scene_hash_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        (tmp_path / "scene.usda").write_text("tampered")
        with pytest.raises(ValueError, match="input changed"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_tampered_spec_hash_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        write_json(spec_path, {"schema_version": 1, "objects": [{"id": "changed"}]})
        with pytest.raises(ValueError, match="input changed"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_asset_inventory_mismatch_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        receipt["assets"] = [{"id": "box1", "status": "PASS", "passed": True},
                              {"id": "box2", "status": "PASS", "passed": True}]
        with pytest.raises(ValueError, match="inventory"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_failing_asset_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        receipt["assets"][0]["passed"] = False
        with pytest.raises(ValueError, match="inventory"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_missing_dependency_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        receipt["dependencies"] = []
        with pytest.raises(ValueError, match="dependencies"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_absolute_dependency_path_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        receipt["dependencies"] = [{"path": "/etc/passwd", "sha256": "a" * 64}]
        with pytest.raises(ValueError, match="invalid|absolute"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_missing_root_layer_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        receipt["dependencies"] = [{"path": "other.usda", "sha256": digest(tmp_path / "scene.usda")}]
        # Create the file so it exists but isn't scene.usda
        (tmp_path / "other.usda").write_text("#usda 1.0\n")
        receipt["dependencies"] = [{"path": "other.usda", "sha256": digest(tmp_path / "other.usda")}]
        with pytest.raises(ValueError, match="root USD layer"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)

    def test_duplicate_dependency_rejected(self, tmp_path):
        receipt, spec_path = _make_receipt(tmp_path)
        dep = {"path": "scene.usda", "sha256": digest(tmp_path / "scene.usda")}
        receipt["dependencies"] = [dep, dict(dep)]
        with pytest.raises(ValueError, match="repeated"):
            verify_physical_receipt(receipt, directory=tmp_path, scene_spec=spec_path)
