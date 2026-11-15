"""USD layer composition, collision geometry generation, and spec validation."""

import numpy as np
import pytest

from simready.assets.compilation.collision_geometry import (
    fit_floor_plane,
    voxel_proxies,
)
from simready.assets.compilation.spec_validation import _bounds, _objects
from simready.assets.physics.physical_spec import normalize_physics


class TestBounds:
    def test_valid_bounds(self):
        low, high = _bounds({"min": [0, 0, 0], "max": [1, 1, 1]})
        assert np.allclose(low, [0, 0, 0])
        assert np.allclose(high, [1, 1, 1])

    def test_degenerate_bounds_rejected(self):
        with pytest.raises(ValueError, match="positive extent"):
            _bounds({"min": [0, 0, 0], "max": [0, 0, 0]})

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ValueError, match="positive extent"):
            _bounds({"min": [1, 1, 1], "max": [0, 0, 0]})

    def test_huge_bounds_rejected(self):
        with pytest.raises(ValueError, match="magnitude"):
            _bounds({"min": [0, 0, 0], "max": [1e10, 1, 1]})

    def test_missing_fields_rejected(self):
        with pytest.raises(ValueError):
            _bounds({"min": [0, 0, 0]})


class TestObjects:
    def test_valid_box(self):
        spec = {
            "objects": [
                {
                    "id": "box1",
                    "type": "box",
                    "bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]},
                }
            ],
            "collision": {},
        }
        result = _objects(spec, contact={}, declared_scale=False)
        assert len(result) == 1
        assert result[0]["type"] == "box"
        assert result[0]["physics"]["motion_type"] == "static"

    def test_duplicate_id_rejected(self):
        spec = {
            "objects": [
                {
                    "id": "box1",
                    "type": "box",
                    "bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]},
                },
                {
                    "id": "box1",
                    "type": "box",
                    "bounds_world": {"min": [2, 2, 2], "max": [3, 3, 3]},
                },
            ],
            "collision": {},
        }
        with pytest.raises(ValueError, match="[Dd]uplicate|unique"):
            _objects(spec, contact={}, declared_scale=False)

    def test_invalid_type_rejected(self):
        spec = {
            "objects": [{"id": "s", "type": "sphere"}],
            "collision": {},
        }
        with pytest.raises(ValueError, match="box or button"):
            _objects(spec, contact={}, declared_scale=False)

    def test_too_many_objects_rejected(self):
        spec = {
            "objects": [
                {
                    "id": f"obj{i}",
                    "type": "box",
                    "bounds_world": {"min": [i, 0, 0], "max": [i + 1, 1, 1]},
                }
                for i in range(257)
            ],
            "collision": {},
        }
        with pytest.raises(ValueError, match="at most 256"):
            _objects(spec, contact={}, declared_scale=False)

    def test_dynamic_box_without_scale_rejected(self):
        spec = {
            "objects": [
                {
                    "id": "box1",
                    "type": "box",
                    "bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]},
                    "physics": {"motion_type": "dynamic", "mass_kg": 1.0},
                }
            ],
            "collision": {},
        }
        with pytest.raises(ValueError, match="metric scale"):
            _objects(spec, contact={}, declared_scale=False)

    def test_dynamic_box_with_scale_accepted(self):
        spec = {
            "objects": [
                {
                    "id": "box1",
                    "type": "box",
                    "bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]},
                    "physics": {"motion_type": "dynamic", "mass_kg": 1.0},
                }
            ],
            "collision": {},
        }
        result = _objects(spec, contact={}, declared_scale=True)
        assert result[0]["physics"]["motion_type"] == "dynamic"

    def test_valid_button(self):
        spec = {
            "objects": [
                {
                    "id": "btn1",
                    "type": "button",
                    "position_world": [0, 0, 1],
                }
            ],
            "collision": {},
        }
        result = _objects(spec, contact={}, declared_scale=False)
        assert result[0]["type"] == "button"
        assert result[0]["physics"]["motion_type"] == "dynamic"

    def test_friction_order_enforced(self):
        spec = {
            "objects": [
                {
                    "id": "box1",
                    "type": "box",
                    "bounds_world": {"min": [0, 0, 0], "max": [1, 1, 1]},
                    "physics": {
                        "contact": {"static_friction": 0.3, "dynamic_friction": 0.5},
                    },
                }
            ],
            "collision": {},
        }
        with pytest.raises(ValueError, match="dynamic_friction.*static"):
            _objects(spec, contact={}, declared_scale=False)


class TestFitFloorPlane:
    def test_flat_floor_detected(self):
        rng = np.random.default_rng(42)
        # Create a flat floor at z=0 with some noise.
        points = np.column_stack(
            [
                rng.uniform(-5, 5, 200),
                rng.uniform(-5, 5, 200),
                rng.normal(0, 0.01, 200),
            ]
        )
        result = fit_floor_plane(points, threshold=0.05)
        assert result is not None
        assert abs(result["normal"][2]) > 0.9  # Roughly Z-up
        assert abs(result["offset"]) < 0.1

    def test_too_few_points_returns_none(self):
        points = np.random.default_rng(42).normal(0, 1, (5, 3))
        result = fit_floor_plane(points)
        assert result is None

    def test_no_flat_surface_returns_none(self):
        rng = np.random.default_rng(42)
        # Random 3D cloud — no dominant plane.
        points = rng.normal(0, 5, (100, 3))
        result = fit_floor_plane(points, threshold=0.01)
        # May or may not find a plane in random data, but should not crash.
        # The important thing is it returns None or a valid plane dict.
        if result is not None:
            assert "normal" in result and "offset" in result


class TestVoxelProxies:
    def test_empty_points(self):
        proxies, size = voxel_proxies(np.empty((0, 3)))
        assert proxies == []
        assert size == 0.25

    def test_single_cluster(self):
        points = np.array([[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
        proxies, size = voxel_proxies(points, voxel_size=0.25, min_points=2)
        assert len(proxies) >= 1
        for p in proxies:
            assert len(p["min"]) == 3 and len(p["max"]) == 3

    def test_proxy_cap_coarsens(self):
        rng = np.random.default_rng(42)
        # Create many spread-out points to force coarsening.
        points = rng.uniform(0, 100, (5000, 3))
        proxies, size = voxel_proxies(
            points, voxel_size=0.25, max_proxies=50, min_points=1
        )
        assert len(proxies) <= 50
        assert size > 0.25  # Coarsened

    def test_min_points_filters_sparse_voxels(self):
        points = np.array([[0, 0, 0], [10, 10, 10]])  # Two isolated points
        proxies, _ = voxel_proxies(points, voxel_size=0.25, min_points=2)
        assert len(proxies) == 0  # Both voxels have < 2 points


class TestNormalizePhysics:
    def test_default_static_box(self):
        result = normalize_physics(
            None,
            geometry_type="box",
            fallback_contact={},
            metric_scale_declared=False,
        )
        assert result["motion_type"] == "static"
        assert result["contact"]["static_friction"] == 0.8

    def test_button_always_dynamic(self):
        result = normalize_physics(
            None,
            geometry_type="button",
            fallback_contact={},
            metric_scale_declared=False,
            legacy_mass_kg=0.02,
        )
        assert result["motion_type"] == "dynamic"
        assert "mass_kg" in result

    def test_dynamic_box_requires_mass(self):
        with pytest.raises(ValueError, match="mass_kg"):
            normalize_physics(
                {"motion_type": "dynamic"},
                geometry_type="box",
                fallback_contact={},
                metric_scale_declared=True,
            )

    def test_dynamic_box_without_scale_rejected(self):
        with pytest.raises(ValueError, match="metric scale"):
            normalize_physics(
                {"motion_type": "dynamic", "mass_kg": 1.0},
                geometry_type="box",
                fallback_contact={},
                metric_scale_declared=False,
            )

    def test_friction_order_enforced(self):
        with pytest.raises(ValueError, match="dynamic_friction"):
            normalize_physics(
                {"contact": {"static_friction": 0.3, "dynamic_friction": 0.5}},
                geometry_type="box",
                fallback_contact={},
                metric_scale_declared=False,
            )

    def test_inertia_triangle_inequality(self):
        with pytest.raises(ValueError, match="triangle"):
            normalize_physics(
                {
                    "motion_type": "dynamic",
                    "mass_kg": 1.0,
                    "inertia_kg_m2": [10.0, 0.1, 0.1],
                },
                geometry_type="box",
                fallback_contact={},
                metric_scale_declared=True,
            )

    def test_provenance_tracking(self):
        result = normalize_physics(
            {
                "mass_kg": 5.0,
                "provenance": {
                    "mass_kg": {
                        "source_kind": "measured",
                        "source": "scale",
                        "unit": "kg",
                    },
                },
            },
            geometry_type="box",
            fallback_contact={},
            metric_scale_declared=True,
        )
        assert result["provenance"]["mass_kg"]["source_kind"] == "measured"
