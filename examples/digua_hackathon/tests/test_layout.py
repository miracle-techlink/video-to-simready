"""CPU tests for the synthetic substation layout and mission route."""

from __future__ import annotations

import numpy as np
import pytest

from digua_hackathon.controller import SPEC
from digua_hackathon.layout import (
    FENCE_RECT,
    GROUND_SIZE,
    MISSION_WAYPOINTS,
    mission_duration_s,
    obstacles_for_clearance,
    reference_trajectory,
    substation_layout,
    clearance_to_obstacles,
)

ENVELOPE_RADIUS = float(np.linalg.norm(np.array(SPEC.envelope)) / 2.0)


def test_layout_has_recognizable_substation_kinds():
    kinds = {p.kind for p in substation_layout()}
    for expected in (
        "transformer_tank", "bushing", "gantry_beam", "busbar",
        "breaker_column", "fence_post", "control_building",
    ):
        assert expected in kinds, f"missing recognizable kind: {expected}"


def test_layout_inside_ground_and_fence():
    xmin, ymin, xmax, ymax = FENCE_RECT
    for prim in substation_layout():
        cx, cy, _ = prim.center
        sx = prim.radius if hasattr(prim, "radius") else prim.size[0] / 2
        assert xmin - 0.1 <= cx - sx and cx + sx <= xmax + 0.1
        assert ymin - 0.1 <= cy and cy <= ymax + 0.1
    assert GROUND_SIZE[0] / 2 >= xmax and GROUND_SIZE[1] / 2 >= ymax


def test_route_clearance_positive_everywhere():
    """The pre-planned route must stay clear of every authored obstacle."""
    obstacles = obstacles_for_clearance()
    duration = mission_duration_s()
    worst = np.inf
    worst_kind = ""
    t = 0.0
    while t <= duration + 0.5:
        pos, _, _ = reference_trajectory(t)
        clear, kind = clearance_to_obstacles(pos, obstacles, ENVELOPE_RADIUS)
        if clear < worst:
            worst, worst_kind = clear, kind
        t += 0.05
    assert worst > 0.05, (
        f"route penetrates/brushes obstacle {worst_kind!r}: clearance {worst:.3f} m"
    )


def test_route_clearance_with_tracking_lag_margin():
    """Waypoint tracking lags the reference along-travel; shift the sample
    backwards along the velocity direction and require clearance holds."""
    obstacles = obstacles_for_clearance()
    duration = mission_duration_s()
    worst = np.inf
    t = 0.0
    while t <= duration + 0.5:
        pos, vel, _ = reference_trajectory(t)
        speed = np.linalg.norm(vel)
        if speed > 0.1:
            # lag ~ speed/kp with kp=4 -> 0.25*speed; probe that point too
            lagged = pos - vel / speed * (0.25 * speed)
        else:
            lagged = pos
        clear, _ = clearance_to_obstacles(lagged, obstacles, ENVELOPE_RADIUS)
        worst = min(worst, clear)
        t += 0.05
    assert worst > 0.0, f"lagged positions brush obstacles: {worst:.3f} m"


def test_reference_trajectory_continuous():
    duration = mission_duration_s()
    prev, _, _ = reference_trajectory(0.0)
    t = 0.0
    while t <= duration:
        t += 0.1
        pos, _, _ = reference_trajectory(min(t, duration))
        step = np.linalg.norm(pos - prev)
        assert step < 1.0, f"reference jumps {step:.2f} m in 0.1 s"
        prev = pos


def test_mission_duration_in_demo_window():
    duration = mission_duration_s()
    assert 18.0 <= duration <= 35.0, f"mission {duration:.1f}s outside demo window"


def test_reference_trajectory_bounds():
    duration = mission_duration_s()
    for t in np.linspace(0, duration + 2.0, 200):
        pos, vel, label = reference_trajectory(float(t))
        assert np.isfinite(pos).all() and np.isfinite(vel).all()
        assert isinstance(label, str) and label
        xmin, ymin, xmax, ymax = FENCE_RECT
        assert xmin < pos[0] < xmax and ymin < pos[1] < ymax
        assert 0.5 < pos[2] < 12.0


def test_smooth_reference_speed_bounded_and_settles():
    from digua_hackathon.layout import MISSION_SPEED_MS, smooth_reference

    duration = mission_duration_s() + 3.0
    t = 0.0
    while t <= duration:
        _, v, _ = smooth_reference(t)
        assert np.linalg.norm(v) <= MISSION_SPEED_MS + 1e-6
        t += 0.1
    p_end, v_end, label = smooth_reference(duration)
    target = np.array(MISSION_WAYPOINTS[-1].position)
    assert np.linalg.norm(p_end - target) < 0.2, f"not settled: {np.linalg.norm(p_end - target):.3f}"
    assert np.linalg.norm(v_end) < 0.5


def test_smooth_reference_clearance_positive():
    """The smoothed reference is what actually flies; it must stay clear too."""
    from digua_hackathon.layout import smooth_reference

    obstacles = obstacles_for_clearance()
    duration = mission_duration_s() + 2.0
    t = 0.0
    worst = np.inf
    while t <= duration:
        pos, _, _ = smooth_reference(t)
        clear, _ = clearance_to_obstacles(pos, obstacles, ENVELOPE_RADIUS)
        worst = min(worst, clear)
        t += 0.05
    assert worst > 0.05, f"smoothed route brushes obstacle: {worst:.3f} m"
