"""Synthetic substation scene layout, obstacle boxes, and inspection waypoints.

Everything in this module is pure Python/numpy and CPU-testable: the USD
authoring in ``usd_builder.py`` renders this declarative layout, and the
clearance monitor consumes the same obstacle boxes at runtime.

Honesty contract: this layout is a SYNTHETIC substation for the Digua
hackathon drone demo. It is not a real-site reconstruction and carries no
real2sim claim. The inspection route below is a pre-planned collision-free
waypoint list; flying it demonstrates waypoint tracking, not autonomous
obstacle avoidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic substation layout (meters, Z-up, metersPerUnit=1)
# ---------------------------------------------------------------------------

GROUND_SIZE = (70.0, 50.0)  # x, y extent of gravel ground plane
CORRIDOR_WIDTH = 5.0  # inspection corridor strip along the x axis
FENCE_RECT = (-28.0, -18.0, 28.0, 18.0)  # perimeter fence (xmin, ymin, xmax, ymax)
FENCE_HEIGHT = 2.2

GRAVEL_COLOR = (0.42, 0.40, 0.36)
CORRIDOR_COLOR = (0.30, 0.30, 0.31)
CONCRETE_COLOR = (0.55, 0.55, 0.53)
TRANSFORMER_COLOR = (0.23, 0.32, 0.28)
STEEL_COLOR = (0.62, 0.64, 0.66)
PORCELAIN_COLOR = (0.78, 0.76, 0.70)
BUILDING_COLOR = (0.72, 0.70, 0.66)
DARK_METAL_COLOR = (0.16, 0.16, 0.18)


@dataclass(frozen=True)
class Box:
    """Axis-aligned box center/size/color used both for USD meshes and OBBs."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]
    kind: str  # semantic label, e.g. "transformer", "gantry_beam"
    collision: bool = True


@dataclass(frozen=True)
class Cylinder:
    """Vertical cylinder center/radius/height/color (disc stacks approximate)."""

    center: tuple[float, float, float]
    radius: float
    height: float
    color: tuple[float, float, float]
    kind: str
    collision: bool = True


def _transformer(x: float, y: float) -> list[Box | Cylinder]:
    """One transformer unit: plinth, tank with radiator fins, three bushings."""
    prims: list[Box | Cylinder] = []
    prims.append(
        Box((x, y, 0.15), (4.4, 3.0, 0.3), CONCRETE_COLOR, "transformer_plinth")
    )
    prims.append(
        Box((x, y, 1.7), (3.6, 2.4, 2.6), TRANSFORMER_COLOR, "transformer_tank")
    )
    for side in (-1.0, 1.0):
        for i in range(4):
            fy = y + side * (1.35 + 0.10)
            prims.append(
                Box(
                    (x - 1.35 + i * 0.9, fy, 1.5),
                    (0.55, 0.20, 2.2),
                    TRANSFORMER_COLOR,
                    "transformer_radiator",
                )
            )
    for k in (-1.0, 0.0, 1.0):
        bx = x + k * 1.0
        # porcelain bushing approximated as a disc stack cylinder
        prims.append(
            Cylinder((bx, y, 4.15), 0.16, 1.3, PORCELAIN_COLOR, "bushing")
        )
        prims.append(
            Cylinder((bx, y, 4.90), 0.05, 0.3, DARK_METAL_COLOR, "bushing_terminal")
        )
    return prims


def _gantry(x: float) -> list[Box | Cylinder]:
    """Steel portal frame spanning y=-10..10 with hanging busbar insulators."""
    prims: list[Box | Cylinder] = []
    for gy in (-10.0, 10.0):
        prims.append(Box((x, gy, 3.25), (0.5, 0.5, 6.5), STEEL_COLOR, "gantry_column"))
    prims.append(Box((x, 0.0, 6.6), (0.5, 21.0, 0.5), STEEL_COLOR, "gantry_beam"))
    return prims


def _busbars(x: float) -> list[Cylinder]:
    """Three phase busbars (along y) under one gantry, on insulator strings."""
    prims: list[Cylinder] = []
    heights = (5.35, 5.75, 6.15)
    for z in heights:
        prims.append(Cylinder((x, 0.0, z), 0.05, 20.0, DARK_METAL_COLOR, "busbar"))
    for gy in (-10.0, -3.5, 3.5, 10.0):
        for z in heights:
            prims.append(
                Cylinder((x, gy, z + 0.28), 0.09, 0.45, PORCELAIN_COLOR, "busbar_insulator")
            )
    return prims


def _breaker(x: float, y: float) -> list[Box | Cylinder]:
    """Circuit breaker column: base, porcelain column, top interrupter."""
    return [
        Box((x, y, 0.2), (1.4, 1.4, 0.4), CONCRETE_COLOR, "breaker_base"),
        Cylinder((x, y, 1.9), 0.28, 3.0, PORCELAIN_COLOR, "breaker_column"),
        Cylinder((x, y, 3.6), 0.35, 0.7, DARK_METAL_COLOR, "breaker_head"),
    ]


def substation_layout() -> list[Box | Cylinder]:
    """Full declarative synthetic substation layout (static, no drone)."""
    prims: list[Box | Cylinder] = []
    # transformer bank A (three units) west side, bank B (three units) east side
    for ty in (-6.0, 0.0, 6.0):
        prims.extend(_transformer(-14.0, ty))
        prims.extend(_transformer(14.0, ty))
    # two portal gantries with busbars crossing the yard
    prims.extend(_gantry(-4.0))
    prims.extend(_gantry(4.0))
    prims.extend(_busbars(-4.0))
    prims.extend(_busbars(4.0))
    # breaker columns down the middle corridor edge
    for by in (-6.0, 0.0, 6.0):
        prims.extend(_breaker(0.0, by))
    # control building near the east fence
    prims.append(Box((23.0, 12.0, 1.5), (6.0, 4.0, 3.0), BUILDING_COLOR, "control_building"))
    prims.append(Box((23.0, 9.95, 1.1), (1.0, 0.1, 2.2), DARK_METAL_COLOR, "building_door"))
    # coupling capacitor voltage transformer stubs flanking bank A
    # (placed at |y|=12.5, well clear of the y=±9 patrol lines)
    for cy in (-12.5, 12.5):
        prims.append(Cylinder((-19.5, cy, 1.7), 0.22, 3.0, PORCELAIN_COLOR, "ccvt_column"))
        prims.append(Cylinder((-19.5, cy, 3.4), 0.10, 0.6, DARK_METAL_COLOR, "ccvt_head"))
    # perimeter fence posts + rails (thin boxes, collision enabled)
    xmin, ymin, xmax, ymax = FENCE_RECT
    for fx in np.arange(xmin, xmax + 0.01, 4.0):
        for fy in (ymin, ymax):
            prims.append(Box((fx, fy, FENCE_HEIGHT / 2), (0.08, 0.08, FENCE_HEIGHT), STEEL_COLOR, "fence_post"))
    for fy in np.arange(ymin + 4.0, ymax - 3.99, 4.0):
        for fx in (xmin, xmax):
            prims.append(Box((fx, fy, FENCE_HEIGHT / 2), (0.08, 0.08, FENCE_HEIGHT), STEEL_COLOR, "fence_post"))
    for z in (0.5, 1.3, 2.05):
        prims.append(Box((0.0, ymin, z), (xmax - xmin, 0.05, 0.06), STEEL_COLOR, "fence_rail"))
        prims.append(Box((0.0, ymax, z), (xmax - xmin, 0.05, 0.06), STEEL_COLOR, "fence_rail"))
        prims.append(Box((xmin, 0.0, z), (0.05, ymax - ymin, 0.06), STEEL_COLOR, "fence_rail"))
        prims.append(Box((xmax, 0.0, z), (0.05, ymax - ymin, 0.06), STEEL_COLOR, "fence_rail"))
    return prims


# ---------------------------------------------------------------------------
# Obstacle OBBs for the runtime clearance monitor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned obstacle box (center, half-extents) with semantic kind."""

    center: np.ndarray
    half: np.ndarray
    kind: str


def obstacles_for_clearance() -> list[Obstacle]:
    """Collision-relevant obstacles as inflated OBBs (generous, includes rails)."""
    out: list[Obstacle] = []
    for prim in substation_layout():
        if not prim.collision:
            continue
        if isinstance(prim, Box):
            half = np.array(prim.size) / 2
        else:
            half = np.array([prim.radius, prim.radius, prim.height / 2])
        out.append(Obstacle(np.array(prim.center), half, prim.kind))
    return out


def clearance_to_obstacles(
    position: np.ndarray, obstacles: list[Obstacle], envelope_radius: float
) -> tuple[float, str]:
    """Distance from a sphere (drone envelope) to the closest obstacle box.

    Positive result means free space remaining; negative means penetration.
    This is geometric bookkeeping against the KNOWN authored layout, not a
    sensor measurement and not an autonomy capability.
    """
    best = math.inf
    best_kind = "none"
    for obs in obstacles:
        delta = np.abs(position - obs.center) - obs.half
        outside = np.linalg.norm(np.maximum(delta, 0.0))
        inside = min(max(delta.max(), 0.0), 0.0)  # 0 unless fully inside
        distance = outside + inside - envelope_radius
        if distance < best:
            best, best_kind = float(distance), obs.kind
    return best, best_kind


# ---------------------------------------------------------------------------
# Inspection mission waypoints (pre-planned, collision-free by construction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Waypoint:
    position: tuple[float, float, float]
    hold_s: float = 0.0  # hover hold at the waypoint
    label: str = ""


# Route facts: yard corridor runs along x; equipment rows sit at x=-14/0/+14.
# The drone patrols at y=+9 (north of the breaker row, clear of banks whose
# north edge is y=7.3, south of gantry columns at y=10 minus half-width 0.25)
# and crosses the yard at the open east end (x=+20) where no gantry exists.
# Cruise 5 m/s with short holds keeps the run near 30 s of video.
MISSION_WAYPOINTS: list[Waypoint] = [
    Waypoint((-24.0, 9.0, 2.0), hold_s=2.5, label="start_hover"),
    Waypoint((-24.0, 9.0, 4.0), label="climb"),
    Waypoint((-14.0, 9.0, 4.2), hold_s=1.0, label="inspect_bank_A"),
    Waypoint((-4.0, 9.0, 4.2), hold_s=1.0, label="inspect_gantry_w"),
    Waypoint((4.0, 9.0, 4.2), hold_s=1.0, label="inspect_breakers"),
    Waypoint((14.0, 9.0, 4.2), hold_s=1.0, label="inspect_bank_B"),
    Waypoint((20.0, 9.0, 7.0), label="climb_overview"),
    Waypoint((20.0, -9.0, 7.0), label="cross_yard"),
    Waypoint((0.0, -9.0, 5.0), hold_s=1.5, label="south_pass"),
    Waypoint((-24.0, -9.0, 3.0), label="exit_west"),
    Waypoint((-24.0, -4.0, 1.5), hold_s=2.0, label="final_hover"),
]

MISSION_SPEED_MS = 5.0  # cruise speed between waypoints
MISSION_NAME = "substation_corridor_patrol_v1"


def mission_duration_s() -> float:
    """Total nominal mission time (travel + holds)."""
    total = 0.0
    for a, b in zip(MISSION_WAYPOINTS, MISSION_WAYPOINTS[1:]):
        leg = np.linalg.norm(np.array(b.position) - np.array(a.position))
        total += leg / MISSION_SPEED_MS + b.hold_s
    return total + MISSION_WAYPOINTS[0].hold_s


def reference_trajectory(t: float) -> tuple[np.ndarray, np.ndarray, str]:
    """Position and velocity of the reference at mission time t (trapezoid-free).

    Moves at constant cruise speed along each leg, pausing for holds. Velocity
    is discontinuous at waypoints; the position PD controller smooths it.
    """
    if t < 0:
        raise ValueError("mission time must be non-negative")
    time_left = MISSION_WAYPOINTS[0].hold_s
    if t < time_left:
        p = np.array(MISSION_WAYPOINTS[0].position, dtype=float)
        return p, np.zeros(3), MISSION_WAYPOINTS[0].label
    for a, b in zip(MISSION_WAYPOINTS, MISSION_WAYPOINTS[1:]):
        pa = np.array(a.position, dtype=float)
        pb = np.array(b.position, dtype=float)
        leg = float(np.linalg.norm(pb - pa))
        travel = leg / MISSION_SPEED_MS
        if t < time_left + travel:
            direction = (pb - pa) / leg
            return pa + direction * (t - time_left) * MISSION_SPEED_MS, direction * MISSION_SPEED_MS, b.label
        time_left += travel
        if t < time_left + b.hold_s:
            return pb.copy(), np.zeros(3), b.label
        time_left += b.hold_s
    p = np.array(MISSION_WAYPOINTS[-1].position, dtype=float)
    return p, np.zeros(3), MISSION_WAYPOINTS[-1].label
