"""Deterministic neutral/left/right route condition generator and fixture evaluation.

Issue #8033 (parent research #7883).

This module provides pure, typed, deterministic utilities to generate route alternatives
under explicit side biases (``neutral``, ``left``, ``right``) across canonical
multi-homotopy environments (corridor, doorway, crossing).

It integrates directly with the ``route_choice_observability.v1`` contract
(:mod:`robot_sf.benchmark.route_choice_observability`) to enable reproducible passing-side
and route predictability diagnostics.

Plain-language summary:
- Generates smooth, parameter-bounded left-, right-, and neutral-biased routes.
- Provides canonical benchmark fixtures for corridors, doorways, and crossings.
- Verifies route-side classification and temporal consistency via the observability contract.
- Purely deterministic and diagnostic; makes no human-behavior or social-compliance claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np

from robot_sf.benchmark.route_choice_observability import (
    DEFAULT_NEUTRAL_BAND_M,
    DEFAULT_SIDE_TOLERANCE_M,
    HomotopyObservation,
    RouteSideReport,
    classify_route_side,
    diagnostic_record,
)

RouteBiasMode = Literal["neutral", "left", "right"]
HomotopyEnvironmentType = Literal["corridor", "doorway", "crossing", "free_space"]
ProfileShape = Literal["smooth_sine", "hann", "cubic", "trapezoid"]

ROUTE_BIAS_MODES: frozenset[RouteBiasMode] = frozenset({"neutral", "left", "right"})
HOMOTOPY_ENV_TYPES: frozenset[HomotopyEnvironmentType] = frozenset(
    {"corridor", "doorway", "crossing", "free_space"}
)
PROFILE_SHAPES: frozenset[ProfileShape] = frozenset({"smooth_sine", "hann", "cubic", "trapezoid"})

_EPS: float = 1e-9


@dataclass(frozen=True)
class BiasedRouteConfig:
    """Configuration for deterministic biased route generation."""

    bias_mode: RouteBiasMode = "neutral"
    lateral_bias_m: float = 1.0
    num_points: int = 50
    profile: ProfileShape = "smooth_sine"
    neutral_band_m: float = DEFAULT_NEUTRAL_BAND_M
    tolerance_m: float = DEFAULT_SIDE_TOLERANCE_M
    coordinate_frame: str = "global_xy"
    units: str = "m"
    progress_interval: tuple[float, float] = (0.1, 0.9)

    def __post_init__(self) -> None:
        """Validate configuration parameters fail-closed."""
        if self.bias_mode not in ROUTE_BIAS_MODES:
            msg = (
                f"Invalid bias_mode: {self.bias_mode!r}, must be one of {sorted(ROUTE_BIAS_MODES)}"
            )
            raise ValueError(msg)
        if not math.isfinite(self.lateral_bias_m) or self.lateral_bias_m < 0.0:
            msg = f"lateral_bias_m must be non-negative and finite, got {self.lateral_bias_m}"
            raise ValueError(msg)
        if self.num_points < 2:
            msg = f"num_points must be >= 2, got {self.num_points}"
            raise ValueError(msg)
        if self.profile not in PROFILE_SHAPES:
            msg = f"Invalid profile: {self.profile!r}, must be one of {sorted(PROFILE_SHAPES)}"
            raise ValueError(msg)
        if not math.isfinite(self.neutral_band_m) or self.neutral_band_m < 0.0:
            msg = f"neutral_band_m must be non-negative and finite, got {self.neutral_band_m}"
            raise ValueError(msg)
        if not math.isfinite(self.tolerance_m) or self.tolerance_m < 0.0:
            msg = f"tolerance_m must be non-negative and finite, got {self.tolerance_m}"
            raise ValueError(msg)
        if (
            len(self.progress_interval) != 2
            or not (0.0 <= self.progress_interval[0] < self.progress_interval[1] <= 1.0)
            or not all(math.isfinite(x) for x in self.progress_interval)
        ):
            msg = f"progress_interval must be 0 <= start < end <= 1, got {self.progress_interval}"
            raise ValueError(msg)
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame.strip():
            msg = "coordinate_frame must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(self.units, str) or not self.units.strip():
            msg = "units must be a non-empty string"
            raise ValueError(msg)


@dataclass(frozen=True)
class BiasedRouteResult:
    """Generated biased route geometry and classification report."""

    bias_mode: RouteBiasMode
    path: list[tuple[float, float]]
    start: tuple[float, float]
    goal: tuple[float, float]
    length_m: float
    max_lateral_offset_m: float
    mean_lateral_offset_m: float
    side_report: RouteSideReport

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary representation.

        Returns:
            Dictionary with serialized route attributes and side report.
        """
        return {
            "bias_mode": self.bias_mode,
            "path": [list(pt) for pt in self.path],
            "start": list(self.start),
            "goal": list(self.goal),
            "length_m": self.length_m,
            "max_lateral_offset_m": self.max_lateral_offset_m,
            "mean_lateral_offset_m": self.mean_lateral_offset_m,
            "side_report": self.side_report.as_dict(),
        }


@dataclass(frozen=True)
class CanonicalFixtureTopology:
    """Canonical multi-homotopy benchmark fixture definition."""

    name: str
    environment_type: HomotopyEnvironmentType
    start: tuple[float, float]
    goal: tuple[float, float]
    obstacles: list[tuple[float, float, float, float]]  # (min_x, min_y, max_x, max_y)
    reference_lateral_bias_m: float
    description: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary representation.

        Returns:
            Dictionary with serialized fixture attributes.
        """
        return {
            "name": self.name,
            "environment_type": self.environment_type,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": [list(box) for box in self.obstacles],
            "reference_lateral_bias_m": self.reference_lateral_bias_m,
            "description": self.description,
        }


def _evaluate_profile(t: float, profile: ProfileShape) -> float:
    """Evaluate normalized profile weight in [0, 1] for progress t in [0, 1].

    Parameters:
        t: Progress along path from 0.0 to 1.0.
        profile: Selected shape curve.

    Returns:
        Profile displacement factor in [0.0, 1.0].
    """
    if t <= 0.0 or t >= 1.0:
        return 0.0
    if profile == "smooth_sine":
        return math.sin(math.pi * t)
    if profile == "hann":
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
    if profile == "cubic":
        return 4.0 * t * (1.0 - t)
    if profile == "trapezoid":
        if t < 0.2:
            return t / 0.2
        if t > 0.8:
            return (1.0 - t) / 0.2
        return 1.0
    return math.sin(math.pi * t)


def _validate_point(pt: Any, name: str) -> tuple[float, float]:
    """Validate a 2D finite coordinate pair.

    Parameters:
        pt: Point sequence (x, y).
        name: Name for error reporting.

    Returns:
        Validated 2D float coordinate tuple.
    """
    if not isinstance(pt, (tuple, list)) or len(pt) != 2:
        msg = f"{name} must be a 2-element sequence, got {pt!r}"
        raise ValueError(msg)
    x, y = float(pt[0]), float(pt[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        msg = f"{name} coordinates must be finite, got ({x}, {y})"
        raise ValueError(msg)
    return (x, y)


def generate_biased_route(
    start: tuple[float, float],
    goal: tuple[float, float],
    config: BiasedRouteConfig | None = None,
    **kwargs: Any,
) -> BiasedRouteResult:
    """Generate a deterministic biased route from start to goal.

    Parameters:
        start: Directed reference start point (x, y).
        goal: Directed reference goal point (x, y).
        config: Optional :class:`BiasedRouteConfig` instance.
        **kwargs: Overrides for configuration parameters if config is None.

    Returns:
        A :class:`BiasedRouteResult` containing the sampled path and classification report.
    """
    valid_start = _validate_point(start, "start")
    valid_goal = _validate_point(goal, "goal")

    if config is None:
        cfg = BiasedRouteConfig(**kwargs) if kwargs else BiasedRouteConfig()
    else:
        cfg = config

    dx = valid_goal[0] - valid_start[0]
    dy = valid_goal[1] - valid_start[1]
    distance = math.hypot(dx, dy)

    if distance < cfg.tolerance_m or distance < _EPS:
        msg = (
            f"Start and goal are degenerate (distance {distance:.4f} < tolerance {cfg.tolerance_m})"
        )
        raise ValueError(msg)

    # Unit forward vector u and normal vector n (standard CCW 90 deg rotation: (-u_y, u_x))
    ux = dx / distance
    uy = dy / distance
    nx = -uy
    ny = ux

    # Determine lateral multiplier based on bias mode
    if cfg.bias_mode == "neutral":
        sign = 0.0
    elif cfg.bias_mode == "left":
        sign = 1.0
    elif cfg.bias_mode == "right":
        sign = -1.0
    else:
        sign = 0.0

    target_lateral_offset = sign * cfg.lateral_bias_m

    # Sample points along path
    n_pts = cfg.num_points
    path: list[tuple[float, float]] = []
    lateral_offsets: list[float] = []

    for i in range(n_pts):
        t = float(i) / float(n_pts - 1)
        profile_weight = _evaluate_profile(t, cfg.profile)
        lat_offset = target_lateral_offset * profile_weight

        # Centerline position
        cx = valid_start[0] + t * dx
        cy = valid_start[1] + t * dy

        # Displaced position
        px = cx + lat_offset * nx
        py = cy + lat_offset * ny

        path.append((px, py))
        lateral_offsets.append(lat_offset)

    total_length = sum(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in pairwise(path))

    max_lateral_offset = max(abs(off) for off in lateral_offsets)
    mean_lateral_offset = float(np.mean(np.abs(lateral_offsets)))

    side_report = classify_route_side(
        path,
        start=valid_start,
        goal=valid_goal,
        coordinate_frame=cfg.coordinate_frame,
        units=cfg.units,
        tolerance_m=cfg.tolerance_m,
        neutral_band_m=cfg.neutral_band_m,
        progress_interval=cfg.progress_interval,
    )

    return BiasedRouteResult(
        bias_mode=cfg.bias_mode,
        path=path,
        start=valid_start,
        goal=valid_goal,
        length_m=total_length,
        max_lateral_offset_m=max_lateral_offset,
        mean_lateral_offset_m=mean_lateral_offset,
        side_report=side_report,
    )


def build_corridor_fixture(
    length: float = 20.0,
    width: float = 4.0,
    obstacle_center: tuple[float, float] | None = (10.0, 0.0),
    obstacle_radius: float = 0.8,
) -> CanonicalFixtureTopology:
    """Construct a canonical straight corridor benchmark fixture.

    Parameters:
        length: Total corridor length in metres.
        width: Usable corridor width in metres.
        obstacle_center: Optional (x, y) center for static obstacle.
        obstacle_radius: Half-extent of center barrier.

    Returns:
        CanonicalFixtureTopology for the corridor split scenario.
    """
    start = (0.0, 0.0)
    goal = (length, 0.0)
    obstacles: list[tuple[float, float, float, float]] = []

    half_w = width / 2.0
    wall_thickness = 1.0
    obstacles.append((0.0, -half_w - wall_thickness, length, -half_w))
    obstacles.append((0.0, half_w, length, half_w + wall_thickness))

    if obstacle_center is not None:
        cx, cy = obstacle_center
        r = obstacle_radius
        obstacles.append((cx - r, cy - r, cx + r, cy + r))

    return CanonicalFixtureTopology(
        name="canonical_corridor_split",
        environment_type="corridor",
        start=start,
        goal=goal,
        obstacles=obstacles,
        reference_lateral_bias_m=half_w * 0.6,
        description="Corridor with symmetric left and right passage around a central obstacle.",
    )


def build_doorway_fixture(
    wall_x: float = 10.0,
    corridor_width: float = 6.0,
    door_width: float = 1.2,
    door_offset: float = 1.8,
) -> CanonicalFixtureTopology:
    """Construct a canonical doorway benchmark fixture with dual symmetric doors.

    Parameters:
        wall_x: Longitudinal position of dividing wall.
        corridor_width: Total corridor width in metres.
        door_width: Width of each door opening.
        door_offset: Lateral offset from centerline to door centers.

    Returns:
        CanonicalFixtureTopology for the dual doorway scenario.
    """
    start = (0.0, 0.0)
    goal = (20.0, 0.0)
    obstacles: list[tuple[float, float, float, float]] = []
    half_w = corridor_width / 2.0

    center_min_y = -(door_offset - door_width / 2.0)
    center_max_y = +(door_offset - door_width / 2.0)
    if center_max_y > center_min_y:
        obstacles.append((wall_x - 0.2, center_min_y, wall_x + 0.2, center_max_y))

    top_min_y = door_offset + door_width / 2.0
    if half_w > top_min_y:
        obstacles.append((wall_x - 0.2, top_min_y, wall_x + 0.2, half_w))

    bot_max_y = -(door_offset + door_width / 2.0)
    if bot_max_y > -half_w:
        obstacles.append((wall_x - 0.2, -half_w, wall_x + 0.2, bot_max_y))

    return CanonicalFixtureTopology(
        name="canonical_doorway_split",
        environment_type="doorway",
        start=start,
        goal=goal,
        obstacles=obstacles,
        reference_lateral_bias_m=door_offset,
        description="Wall divider featuring distinct left and right doorway passages.",
    )


def build_crossing_fixture(
    crossing_point: tuple[float, float] = (10.0, 0.0),
    zone_radius: float = 1.5,
) -> CanonicalFixtureTopology:
    """Construct a canonical open crossing benchmark fixture.

    Parameters:
        crossing_point: (x, y) center of crossing interaction zone.
        zone_radius: Radius of interaction zone to avoid.

    Returns:
        CanonicalFixtureTopology for the crossing interaction scenario.
    """
    start = (0.0, 0.0)
    goal = (20.0, 0.0)
    cx, cy = crossing_point
    r = zone_radius
    obstacles = [(cx - r, cy - r, cx + r, cy + r)]

    return CanonicalFixtureTopology(
        name="canonical_crossing_interaction",
        environment_type="crossing",
        start=start,
        goal=goal,
        obstacles=obstacles,
        reference_lateral_bias_m=zone_radius * 1.2,
        description="Crossing interaction zone requiring left or right lateral diversion.",
    )


def generate_corridor_homotopy_routes(
    fixture: CanonicalFixtureTopology | None = None,
    lateral_bias_m: float | None = None,
    num_points: int = 50,
) -> dict[RouteBiasMode, BiasedRouteResult]:
    """Generate all three canonical route variants (neutral, left, right) on a corridor fixture.

    Parameters:
        fixture: Optional corridor fixture topology.
        lateral_bias_m: Optional lateral bias displacement.
        num_points: Waypoint count.

    Returns:
        Mapping from RouteBiasMode to BiasedRouteResult.
    """
    fix = fixture if fixture is not None else build_corridor_fixture()
    bias = lateral_bias_m if lateral_bias_m is not None else fix.reference_lateral_bias_m

    results: dict[RouteBiasMode, BiasedRouteResult] = {}
    for mode in ("neutral", "left", "right"):
        cfg = BiasedRouteConfig(
            bias_mode=cast("RouteBiasMode", mode),
            lateral_bias_m=bias,
            num_points=num_points,
        )
        results[cast("RouteBiasMode", mode)] = generate_biased_route(
            start=fix.start,
            goal=fix.goal,
            config=cfg,
        )
    return results


def generate_doorway_homotopy_routes(
    fixture: CanonicalFixtureTopology | None = None,
    lateral_bias_m: float | None = None,
    num_points: int = 50,
) -> dict[RouteBiasMode, BiasedRouteResult]:
    """Generate all three canonical route variants on a dual-doorway fixture.

    Parameters:
        fixture: Optional doorway fixture topology.
        lateral_bias_m: Optional lateral bias displacement.
        num_points: Waypoint count.

    Returns:
        Mapping from RouteBiasMode to BiasedRouteResult.
    """
    fix = fixture if fixture is not None else build_doorway_fixture()
    bias = lateral_bias_m if lateral_bias_m is not None else fix.reference_lateral_bias_m

    results: dict[RouteBiasMode, BiasedRouteResult] = {}
    for mode in ("neutral", "left", "right"):
        cfg = BiasedRouteConfig(
            bias_mode=cast("RouteBiasMode", mode),
            lateral_bias_m=bias,
            num_points=num_points,
        )
        results[cast("RouteBiasMode", mode)] = generate_biased_route(
            start=fix.start,
            goal=fix.goal,
            config=cfg,
        )
    return results


def rasterize_route_to_grid(
    path: list[tuple[float, float]],
    grid_origin: tuple[float, float],
    grid_resolution: float,
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Rasterize a continuous 2D path into a step-valid 8-connected grid cell sequence.

    Parameters:
        path: Ordered list of (x, y) world coordinates.
        grid_origin: Lower-left world origin (min_x, min_y) of the grid.
        grid_resolution: Metres per grid cell.
        grid_shape: (rows, cols) shape of the grid.

    Returns:
        An 8-connected list of (row, col) integer grid cells without non-adjacent jumps.
    """
    if not path:
        return []
    if grid_resolution <= _EPS or not math.isfinite(grid_resolution):
        msg = f"Invalid grid_resolution: {grid_resolution}"
        raise ValueError(msg)

    n_rows, n_cols = grid_shape
    origin_x, origin_y = grid_origin

    def _to_grid(x: float, y: float) -> tuple[int, int]:
        c = math.floor((x - origin_x) / grid_resolution)
        r = math.floor((y - origin_y) / grid_resolution)
        c = max(0, min(n_cols - 1, c))
        r = max(0, min(n_rows - 1, r))
        return (r, c)

    grid_path: list[tuple[int, int]] = []
    current_cell = _to_grid(path[0][0], path[0][1])
    grid_path.append(current_cell)

    for pt in path[1:]:
        target_cell = _to_grid(pt[0], pt[1])
        r0, c0 = current_cell
        r1, c1 = target_cell

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        steps = max(dr, dc)
        if steps > 0:
            for step in range(1, steps + 1):
                interp_r = r0 + round(step * (r1 - r0) / float(steps))
                interp_c = c0 + round(step * (c1 - c0) / float(steps))
                cell = (interp_r, interp_c)
                if cell != grid_path[-1]:
                    grid_path.append(cell)
        current_cell = target_cell

    return grid_path


def evaluate_route_observability_sequence(
    routes: list[BiasedRouteResult],
    reference_start: tuple[float, float] | None = None,
    reference_goal: tuple[float, float] | None = None,
    tolerance_m: float = DEFAULT_SIDE_TOLERANCE_M,
    neutral_band_m: float = DEFAULT_NEUTRAL_BAND_M,
) -> dict[str, Any]:
    """Evaluate temporal consistency and diagnostic records across a sequence of routes.

    Parameters:
        routes: Chronological sequence of generated :class:`BiasedRouteResult` objects.
        reference_start: Optional fixed reference start point.
        reference_goal: Optional fixed reference goal point.
        tolerance_m: Classification tolerance.
        neutral_band_m: Neutral band half-width.

    Returns:
        A dictionary containing the full diagnostic record emitted by
        :func:`robot_sf.benchmark.route_choice_observability.diagnostic_record`.
    """
    if not routes:
        return diagnostic_record([], [])

    start = reference_start if reference_start is not None else routes[0].start
    goal = reference_goal if reference_goal is not None else routes[0].goal

    side_reports: list[RouteSideReport] = []
    dummy_homotopies: list[HomotopyObservation] = []

    for r in routes:
        report = classify_route_side(
            r.path,
            start=start,
            goal=goal,
            tolerance_m=tolerance_m,
            neutral_band_m=neutral_band_m,
        )
        side_reports.append(report)
        dummy_homotopies.append(
            HomotopyObservation(
                identity=f"mode_{r.bias_mode}",
                unavailable_reason=None,
                identity_coordinate_frame="global_xy",
                identity_units="m",
                identity_points=(start, goal),
                identity_match_tolerance=0.5,
            )
        )

    return diagnostic_record(side_reports, dummy_homotopies)
