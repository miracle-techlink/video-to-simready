"""Physical spec edge cases: metric scale honesty, metersPerUnit semantics."""

import numpy as np
import pytest

from simready.assets.physics.physical_spec import normalize_physics


class TestMetricScaleHonesty:
    """metersPerUnit=1 only establishes a numeric unit relationship;
    it does not prove real-world metric scale."""

    def test_unverified_scale_blocks_dynamic_box(self):
        """Dynamic boxes require declared metric scale."""
        with pytest.raises(ValueError, match="metric scale"):
            normalize_physics(
                {"mass_kg": 1.0, "motion_type": "dynamic"},
                geometry_type="box",
                fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
                metric_scale_declared=False,
            )

    def test_unverified_scale_allows_static_box(self):
        """Static boxes don't need metric scale."""
        result = normalize_physics(
            {},
            geometry_type="box",
            fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
            metric_scale_declared=False,
        )
        assert result["motion_type"] == "static"

    def test_verified_scale_enables_dynamic_box(self):
        """Verified metric scale allows dynamic box with explicit mass."""
        result = normalize_physics(
            {"mass_kg": 1.0, "motion_type": "dynamic"},
            geometry_type="box",
            fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
            metric_scale_declared=True,
        )
        assert result["motion_type"] == "dynamic"
        assert result["mass_kg"] == 1.0

    def test_verified_scale_allows_button_overrides(self):
        """Verified scale allows button physics overrides."""
        result = normalize_physics(
            {"mass_kg": 0.1, "motion_type": "dynamic"},
            geometry_type="button",
            fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
            metric_scale_declared=True,
        )
        assert result["mass_kg"] == 0.1

    def test_unverified_scale_blocks_button_mass_override(self):
        """Button mass override requires declared metric scale."""
        with pytest.raises(ValueError, match="metric scale"):
            normalize_physics(
                {"mass_kg": 0.1},
                geometry_type="button",
                fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
                metric_scale_declared=False,
            )

    def test_scale_status_in_validation_output(self):
        """The validation output must honestly report scale status."""
        result = normalize_physics(
            {},
            geometry_type="box",
            fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
            metric_scale_declared=False,
        )
        assert result["validation"]["scale_status"] == "unverified"
        assert result["validation"]["independently_verified"] is False

    def test_operator_declared_scale_status(self):
        """Operator-declared scale is labeled, not independently verified."""
        result = normalize_physics(
            {"mass_kg": 1.0, "motion_type": "dynamic"},
            geometry_type="box",
            fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
            metric_scale_declared=True,
        )
        assert result["validation"]["scale_status"] == "operator_declared_metric"
        assert result["validation"]["independently_verified"] is False
        assert result["validation"]["real_world_validated"] is False


class TestContactParameterBounds:
    """Contact parameters must respect physical bounds."""

    def test_friction_bounds(self):
        with pytest.raises(ValueError, match="static_friction"):
            normalize_physics(
                {"contact": {"static_friction": -0.1}},
                geometry_type="box",
                fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
                metric_scale_declared=False,
            )

    def test_restitution_bounds(self):
        with pytest.raises(ValueError, match="restitution"):
            normalize_physics(
                {"contact": {"restitution": 1.1}},
                geometry_type="box",
                fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
                metric_scale_declared=False,
            )

    def test_dynamic_friction_must_not_exceed_static(self):
        with pytest.raises(ValueError, match="dynamic_friction"):
            normalize_physics(
                {"contact": {"static_friction": 0.5, "dynamic_friction": 0.7}},
                geometry_type="box",
                fallback_contact={"static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0},
                metric_scale_declared=False,
            )
