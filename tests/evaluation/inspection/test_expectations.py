"""Expected physics interpretation: defaults, overrides, and contract validation."""

import pytest

from simready.evaluation.inspection.expectations import (
    _CONTACT_DEFAULTS,
    _expected_physics,
    _objects,
)


class TestObjects:
    def test_valid_box(self):
        doc = {"objects": [{"id": "box1", "type": "box", "bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]}}]}
        result = _objects(doc, "test")
        assert "box1" in result

    def test_duplicate_id_rejected(self):
        doc = {"objects": [
            {"id": "box1", "type": "box", "bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]}},
            {"id": "box1", "type": "box", "bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]}},
        ]}
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            _objects(doc, "test")

    def test_invalid_type_rejected(self):
        doc = {"objects": [{"id": "obj", "type": "sphere"}]}
        with pytest.raises(ValueError, match="Unsupported"):
            _objects(doc, "test")

    def test_invalid_id_rejected(self):
        doc = {"objects": [{"id": "not valid!", "type": "box"}]}
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            _objects(doc, "test")

    def test_too_many_objects_rejected(self):
        doc = {"objects": [{"id": f"obj{i}", "type": "box"} for i in range(257)]}
        with pytest.raises(ValueError, match="at most 256"):
            _objects(doc, "test")


class TestExpectedPhysics:
    def test_box_defaults_to_static(self):
        obj = {"id": "b", "type": "box", "physics": {}}
        spec = {"collision": {}}
        result = _expected_physics(obj, spec)
        assert result["motion_type"] == "static"
        assert result["contact"]["static_friction"] == _CONTACT_DEFAULTS["static_friction"]

    def test_button_defaults_to_dynamic(self):
        obj = {"id": "btn", "type": "button", "physics": {}}
        spec = {"collision": {}}
        result = _expected_physics(obj, spec)
        assert result["motion_type"] == "dynamic"
        assert result["mass_kg"] == 0.02

    def test_explicit_motion_type_respected(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "kinematic"}}
        spec = {"collision": {}}
        result = _expected_physics(obj, spec)
        assert result["motion_type"] == "kinematic"

    def test_contact_override_from_collision_spec(self):
        obj = {"id": "b", "type": "box", "physics": {}}
        spec = {"collision": {"static_friction": 0.5, "dynamic_friction": 0.4}}
        result = _expected_physics(obj, spec)
        assert result["contact"]["static_friction"] == 0.5
        assert result["contact"]["dynamic_friction"] == 0.4

    def test_contact_override_from_object(self):
        obj = {"id": "b", "type": "box", "physics": {"contact": {"restitution": 0.3}}}
        spec = {"collision": {}}
        result = _expected_physics(obj, spec)
        assert result["contact"]["restitution"] == 0.3

    def test_button_mass_default(self):
        obj = {"id": "btn", "type": "button", "mass": 0.05, "physics": {}}
        spec = {"collision": {}}
        result = _expected_physics(obj, spec)
        assert result["mass_kg"] == 0.05

    def test_explicit_mass_overrides_default(self):
        obj = {"id": "btn", "type": "button", "mass": 0.05, "physics": {"mass_kg": 0.1}}
        spec = {"collision": {}}
        result = _expected_physics(obj, spec)
        assert result["mass_kg"] == 0.1
