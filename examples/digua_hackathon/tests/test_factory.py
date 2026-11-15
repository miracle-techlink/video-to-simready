"""CPU tests for the factory demo math (real Iris hub mixer + route)."""

from __future__ import annotations

import numpy as np

from digua_hackathon.controller import PositionAttitudeController
from digua_hackathon.iris_spec import IRIS_SPEC, mixer_from_hubs, rotor_hubs


def test_hubs_match_pinned_asset():
    """Hub centers from the astra-integration-audit table (body frame)."""
    hubs = rotor_hubs()
    assert hubs.shape == (4, 3)
    assert np.isclose(hubs[0][0], 0.137985, atol=1e-6)
    assert np.isclose(hubs[1][1], 0.218754, atol=1e-6)
    assert np.isclose(hubs[3][1], -0.222346, atol=1e-6)
    span = np.abs(hubs[:, :2]).max()
    assert 0.1 < span < 0.3, "Iris wheelbase ~0.5 m expected"


def test_iris_mixer_roundtrip_unsaturated():
    ctrl = PositionAttitudeController(spec=IRIS_SPEC, mixer_matrix=mixer_from_hubs(rotor_hubs()))
    force = IRIS_SPEC.mass_kg * 9.81
    torque = np.array([0.015, -0.02, 0.008])  # small torque inside authority
    thrusts, saturated = ctrl.mixer(force, torque)
    assert not saturated
    f_back, t_back = ctrl.wrench_from_thrusts(thrusts)
    assert np.isclose(f_back, force, atol=1e-9)
    assert np.allclose(t_back, torque, atol=1e-9)


def test_iris_hover_never_saturates():
    ctrl = PositionAttitudeController(spec=IRIS_SPEC, mixer_matrix=mixer_from_hubs(rotor_hubs()))
    cmd = ctrl.step(np.zeros(3) + [0, 0, 2.0], np.zeros(3), np.array([1.0, 0, 0, 0]),
                    np.zeros(3), np.zeros(3) + [0, 0, 2.0], np.zeros(3), 0.0)
    assert abs(cmd["total_thrust_n"] - IRIS_SPEC.mass_kg * 9.81) < 0.15
    assert not cmd["saturated"]


def test_route_tracker_inside_factory_bounds():
    from digua_hackathon.factory_demo import FACTORY_ROUTE, RouteTracker

    tracker = RouteTracker(FACTORY_ROUTE, 3.0, 2.5)
    t = 0.0
    while t <= tracker.duration:
        p, v, label = tracker.at(t)
        assert -17.0 < p[0] < 19.4 and -30.6 < p[1] < 30.5 and 0.5 < p[2] < 12.5
        assert np.linalg.norm(v) <= 3.0 + 1e-6
        t += 0.2
    end, v_end, _ = tracker.at(tracker.duration)
    target = np.array(FACTORY_ROUTE[-1]["pos"])
    assert np.linalg.norm(end - target) < 0.3
    assert np.linalg.norm(v_end) < 0.5
