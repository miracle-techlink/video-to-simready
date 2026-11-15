"""Physical contract checks: friction bounds, mass positivity, inertia triangle."""

import pytest

from simready.evaluation.inspection.expectations import _physical_contract


def _run_contract(obj, spec):
    """Run _physical_contract and return the checks list."""
    checks = []
    physics = obj.get("physics", {})
    contact = dict({
        "static_friction": 0.8,
        "dynamic_friction": 0.7,
        "restitution": 0.0,
    })
    contact.update(spec.get("collision", {}))
    contact.update(physics.get("contact", {}))
    expected = {
        "motion_type": physics.get(
            "motion_type", "dynamic" if obj["type"] == "button" else "static"
        ),
        "contact": contact,
    }
    if obj["type"] == "button":
        expected["mass_kg"] = physics.get("mass_kg", obj.get("mass", 0.02))
    for key in ("mass_kg", "center_of_mass_m", "inertia_kg_m2"):
        if key in physics:
            expected[key] = physics[key]
    _physical_contract(checks, obj, expected, spec)
    return checks


def _check_map(checks):
    return {c["id"]: c for c in checks}


class TestMotionContract:
    def test_box_static_passes(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "static"}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.motion"]["passed"] is True

    def test_box_dynamic_passes(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "dynamic", "mass_kg": 1.0}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.motion"]["passed"] is True

    def test_button_non_dynamic_fails(self):
        obj = {"id": "btn", "type": "button", "physics": {"motion_type": "static"}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["btn.input_contract.motion"]["passed"] is False

    def test_invalid_motion_fails(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "teleport"}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.motion"]["passed"] is False


class TestFrictionContract:
    def test_valid_friction_passes(self):
        obj = {"id": "b", "type": "box", "physics": {}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.static_friction"]["passed"] is True
        assert cm["b.input_contract.dynamic_friction"]["passed"] is True
        assert cm["b.input_contract.friction_order"]["passed"] is True

    def test_excessive_friction_fails(self):
        obj = {"id": "b", "type": "box", "physics": {"contact": {"static_friction": 15.0}}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.static_friction"]["passed"] is False

    def test_negative_friction_fails(self):
        obj = {"id": "b", "type": "box", "physics": {"contact": {"static_friction": -0.1}}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.static_friction"]["passed"] is False

    def test_dynamic_exceeds_static_fails(self):
        obj = {"id": "b", "type": "box", "physics": {
            "contact": {"static_friction": 0.5, "dynamic_friction": 0.8}
        }}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.friction_order"]["passed"] is False


class TestMassContract:
    def test_dynamic_box_positive_mass_passes(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "dynamic", "mass_kg": 1.0}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.positive_mass"]["passed"] is True

    def test_zero_mass_fails(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "dynamic", "mass_kg": 0}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.positive_mass"]["passed"] is False

    def test_negative_mass_fails(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "dynamic", "mass_kg": -1}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.positive_mass"]["passed"] is False


class TestInertiaContract:
    def test_valid_inertia_passes(self):
        obj = {"id": "b", "type": "box", "physics": {
            "motion_type": "dynamic", "mass_kg": 1.0,
            "inertia_kg_m2": [0.1, 0.1, 0.1],
        }}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.inertia_kg_m2"]["passed"] is True

    def test_inertia_triangle_violation_fails(self):
        obj = {"id": "b", "type": "box", "physics": {
            "motion_type": "dynamic", "mass_kg": 1.0,
            "inertia_kg_m2": [10.0, 0.1, 0.1],
        }}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.inertia_kg_m2"]["passed"] is False


class TestScaleProvenance:
    def test_dynamic_box_without_scale_declaration_flagged(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "dynamic", "mass_kg": 1.0}}
        checks = _run_contract(obj, {"collision": {}})
        cm = _check_map(checks)
        assert cm["b.input_contract.metric_scale_declaration"]["passed"] is False

    def test_dynamic_box_with_scale_declaration_passes(self):
        obj = {"id": "b", "type": "box", "physics": {"motion_type": "dynamic", "mass_kg": 1.0}}
        spec = {
            "collision": {},
            "source_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "scale_provenance": {
                "status": "verified",
                "method": "measured",
                "evidence": "ruler photo",
            },
        }
        checks = _run_contract(obj, spec)
        cm = _check_map(checks)
        assert cm["b.input_contract.metric_scale_declaration"]["passed"] is True
