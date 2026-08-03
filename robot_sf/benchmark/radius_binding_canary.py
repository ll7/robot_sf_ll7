"""Runtime radius-binding canary for issue #6641 (benchmark 6600 Gate 1).

Plain-language summary
----------------------
This is a small, fail-closed diagnostic that checks the selected robot and
pedestrian collision-envelope radius actually reaches every place that uses it.
A social-navigation benchmark only means what it claims when one chosen radius
drives, consistently, the simulator's obstacle collision geometry, the obstacle
and pedestrian contact logic, the planner-free feasibility oracle, the metric
metadata and output rows, and the planner-facing observation. The canary runs on
one geometry-sensitive scenario, probes each of those five binding surfaces
through its real code path, and emits a machine-readable go/no-go verdict per
surface. It is ``diagnostic-only`` evidence: a pre-campaign binding check, not a
benchmark result, and it never changes the frozen 0.0.3.post1 metric semantics.

Why a differential probe
------------------------
Each surface is tested by varying the radius and checking that the surface's
observable output moves with it. A radius that is silently ignored (hardcoded,
read from the wrong key, or short-circuited) leaves the observable unchanged, so
the corresponding probe records a fail and the overall verdict becomes ``no-go``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import LineString, Point

from robot_sf.benchmark.metrics import human_collisions
from robot_sf.nav.occupancy import ContinuousOccupancy
from robot_sf.scenario_certification.feasibility_oracle import (
    FeasibilityOracleConfig,
    make_envelope_scenario,
    run_feasibility_oracle,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# --- public schema -----------------------------------------------------------

RADIUS_BINDING_CANARY_SCHEMA = "radius_binding_canary.v1"
"""Machine-readable schema tag emitted on every canary verdict."""

CANARY_CLAIM_BOUNDARY = (
    "Issue #6641 (benchmark 6600 Gate 1): prove the selected collision-envelope "
    "radius propagates consistently to simulator collision geometry, obstacle and "
    "pedestrian contact logic, feasibility/oracle calculations, metric metadata and "
    "output rows, and planner inputs. Diagnostic-only pre-campaign binding check; "
    "not benchmark evidence."
)

SURFACE_SIMULATOR_GEOMETRY = "simulator_collision_geometry"
SURFACE_OBSTACLE_PEDESTRIAN_CONTACT = "obstacle_pedestrian_contact"
SURFACE_FEASIBILITY_ORACLE = "feasibility_oracle"
SURFACE_METRIC_METADATA_ROWS = "metric_metadata_and_output_rows"
SURFACE_PLANNER_INPUTS = "planner_inputs"
CANARY_SURFACES: tuple[str, ...] = (
    SURFACE_SIMULATOR_GEOMETRY,
    SURFACE_OBSTACLE_PEDESTRIAN_CONTACT,
    SURFACE_FEASIBILITY_ORACLE,
    SURFACE_METRIC_METADATA_ROWS,
    SURFACE_PLANNER_INPUTS,
)

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
VERDICT_GO = "go"
VERDICT_NO_GO = "no-go"

# A radius binding is accepted when the observed radius-sensitive boundary tracks
# the configured radius within this absolute tolerance (metres).
DEFAULT_RADIUS_TOLERANCE_M = 5e-3
# Step size for the differential radius/distance scans.
DEFAULT_SCAN_STEP_M = 1e-3

# Canonical selected radii for the canary. The robot radius matches the
# simulator's small-envelope benchmark default; the pedestrian radius matches the
# simulation config default. Probes vary these internally to detect silent ignoring.
DEFAULT_SELECTED_ROBOT_RADIUS_M = 0.3
DEFAULT_SELECTED_PED_RADIUS_M = 0.4

DEFAULT_SCENARIO_REL = Path("configs/scenarios/canary_corridor.yaml")
"""Geometry-sensitive default scenario: a 4 m corridor with a 1.7 m wall clearance."""

_REPO_ROOT = Path(__file__).resolve().parents[2]

# These are the expected operational failures from the bounded probe paths. Keep the
# list explicit so the benchmark/script broad-exception ratchet does not need a new
# baseline entry for this diagnostic module.
_CANARY_OPERATIONAL_ERRORS = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


# --- verdict dataclasses -----------------------------------------------------


@dataclass(frozen=True)
class SurfaceVerdict:
    """One binding-surface verdict with the evidence that produced it.

    Attributes:
        surface: Canonical surface key (one of ``CANARY_SURFACES``).
        status: ``pass`` or ``fail``.
        probe: Short human-readable name of the executed probe.
        expected: Human-readable description of the expected radius-sensitive result.
        observed: Human-readable description of the observed result.
        evidence: JSON-safe dict of measured values backing the verdict.
        note: Optional caveat or explanation (e.g. a stubbed sub-component).
    """

    surface: str
    status: str
    probe: str
    expected: str
    observed: str
    evidence: dict[str, Any]
    note: str | None = None


@dataclass(frozen=True)
class CanaryVerdict:
    """Top-level machine-readable go/no-go canary verdict.

    Attributes:
        schema_version: ``RADIUS_BINDING_CANARY_SCHEMA``.
        claim_boundary: Plain-language claim this canary does and does not make.
        evidence_status: Evidence-grade label (``diagnostic-only`` for this canary).
        scenario: Scenario identity and parsed geometry facts used by the probes.
        selected_robot_radius_m: Primary selected robot envelope radius (metres).
        selected_ped_radius_m: Primary selected pedestrian radius (metres).
        surfaces: Per-surface verdicts in ``CANARY_SURFACES`` order.
        verdict: ``go`` only when every surface is ``pass``; otherwise ``no-go``.
        caveats: Caveats and fallback/degraded exclusions.
        generated_at: Timezone-aware ISO timestamp the verdict was assembled.
    """

    schema_version: str
    claim_boundary: str
    evidence_status: str
    scenario: dict[str, Any]
    selected_robot_radius_m: float
    selected_ped_radius_m: float
    surfaces: list[SurfaceVerdict]
    verdict: str
    caveats: list[str]
    generated_at: str


# --- geometry loading --------------------------------------------------------


@dataclass(frozen=True)
class CanaryGeometry:
    """Parsed geometry facts extracted from a geometry-sensitive scenario.

    Attributes:
        scenario_id: Scenario name from the manifest.
        map_name: Parsed map identifier.
        route_point: A robot waypoint used as the probe anchor (world frame).
        goal_point: Route goal supplied to the runtime occupancy component.
        obstacle_lines_runtime: Obstacle segments in runtime ``(x1, y1, x2, y2)``
            ordering, the same array the simulator's collision component receives.
        wall_distance_m: Minimum distance from ``route_point`` to the nearest
            obstacle segment (the geometry-sensitive clearance).
        map_width: Map width in metres.
        map_height: Map height in metres.
        scenario: Raw scenario mapping for the feasibility-oracle probe.
        scenario_path: Resolved scenario manifest path.
        configured_robot_radius_m: Effective robot radius produced by the canonical
            scenario loader.
        configured_ped_radius_m: Effective pedestrian radius produced by the canonical
            scenario loader.
    """

    scenario_id: str
    map_name: str
    route_point: tuple[float, float]
    goal_point: tuple[float, float]
    obstacle_lines_runtime: np.ndarray
    wall_distance_m: float
    map_width: float
    map_height: float
    scenario: Mapping[str, Any]
    scenario_path: Path
    configured_robot_radius_m: float | None = None
    configured_ped_radius_m: float | None = None


def _runtime_obstacle_lines(map_def: Any) -> np.ndarray:
    """Return obstacle segments in the runtime ``(x1, y1, x2, y2)`` ordering.

    ``MapDefinition.obstacles_pysf`` stores the legacy fast-pysf ordering
    ``(start_x, end_x, start_y, end_y)``; the runtime collision component receives
    the normalized ``(start_x, start_y, end_x, end_y)`` array via the same
    ``[0, 2, 1, 3]`` reindex the certifier uses.
    """
    stored = np.asarray(map_def.obstacles_pysf, dtype=float).reshape(-1, 4)
    return stored[:, [0, 2, 1, 3]]


def _min_segment_distance(point_xy: tuple[float, float], segments: np.ndarray) -> float:
    """Return the minimum distance from a point to any obstacle segment."""
    if segments.size == 0:
        return math.inf
    pt = Point(point_xy)
    min_dist = math.inf
    for row in segments:
        line = LineString([(float(row[0]), float(row[1])), (float(row[2]), float(row[3]))])
        dist = float(pt.distance(line))
        min_dist = min(min_dist, dist)
    return min_dist


def load_canary_geometry(scenario_path: Path) -> CanaryGeometry:
    """Load a scenario and extract the geometry facts the surface probes need.

    Args:
        scenario_path: Scenario manifest path (resolved against the repo root when
            relative).

    Returns:
        Parsed :class:`CanaryGeometry`.
    """
    # Local import keeps the module importable without the scenario loader's heavier
    # transitive dependencies until the canary actually runs.
    from robot_sf.training.scenario_loader import (  # noqa: PLC0415
        build_robot_config_from_scenario,
        load_scenarios,
    )

    resolved = scenario_path if scenario_path.is_absolute() else _REPO_ROOT / scenario_path
    scenarios = load_scenarios(resolved)
    if not scenarios:
        raise ValueError(f"scenario manifest {resolved} contained no scenarios")
    scenario = scenarios[0]
    config = build_robot_config_from_scenario(scenario, scenario_path=resolved)
    map_pool = list(config.map_pool.map_defs.items())
    if not map_pool:
        raise ValueError(f"scenario {resolved} produced an empty map pool")
    map_name, map_def = map_pool[0]
    obstacle_lines = _runtime_obstacle_lines(map_def)
    route = map_def.robot_routes[0]
    waypoints = list(route.waypoints)
    route_point = (float(waypoints[0][0]), float(waypoints[0][1]))
    goal_point = (float(waypoints[-1][0]), float(waypoints[-1][1]))
    # Anchor the obstacle probe at the route mid-point, which sits inside the
    # corridor away from the spawn/goal zones, so the nearest wall is a real
    # corridor wall rather than a map edge.
    mid_x = 0.5 * (route_point[0] + goal_point[0])
    mid_y = 0.5 * (route_point[1] + goal_point[1])
    anchor = (mid_x, mid_y)
    wall_distance = _min_segment_distance(anchor, obstacle_lines)
    return CanaryGeometry(
        scenario_id=str(scenario.get("name") or scenario.get("id") or "unknown"),
        map_name=str(map_name),
        route_point=anchor,
        goal_point=goal_point,
        obstacle_lines_runtime=obstacle_lines,
        wall_distance_m=float(wall_distance),
        map_width=float(map_def.width),
        map_height=float(map_def.height),
        scenario=scenario,
        scenario_path=resolved,
        configured_robot_radius_m=float(config.robot_config.radius),
        configured_ped_radius_m=float(config.sim_config.ped_radius),
    )


def _configuration_binding_evidence(
    *,
    selected_robot_radius_m: float,
    selected_ped_radius_m: float,
    configured_robot_radius_m: float | None,
    configured_ped_radius_m: float | None,
    tolerance_m: float,
) -> tuple[bool, dict[str, Any]]:
    """Check that selected radii match the effective loaded scenario configuration.

    Direct unit probes may omit configuration values because they exercise one surface
    with synthetic geometry. The orchestrator always supplies the values loaded from the
    committed scenario, making a CLI/configuration mismatch fail closed.

    Returns:
        A pass flag and JSON-safe evidence for the selected/configured comparison.
    """
    if configured_robot_radius_m is None and configured_ped_radius_m is None:
        return True, {}
    robot_ok = (
        configured_robot_radius_m is not None
        and math.isfinite(float(configured_robot_radius_m))
        and abs(float(configured_robot_radius_m) - float(selected_robot_radius_m)) <= tolerance_m
    )
    ped_ok = (
        configured_ped_radius_m is not None
        and math.isfinite(float(configured_ped_radius_m))
        and abs(float(configured_ped_radius_m) - float(selected_ped_radius_m)) <= tolerance_m
    )
    return robot_ok and ped_ok, {
        "selected_robot_radius_m": float(selected_robot_radius_m),
        "selected_ped_radius_m": float(selected_ped_radius_m),
        "configured_robot_radius_m": configured_robot_radius_m,
        "configured_ped_radius_m": configured_ped_radius_m,
        "selected_configuration_matches": bool(robot_ok and ped_ok),
    }


# --- differential scan helpers ----------------------------------------------


def _scan_flip(
    predicate: Callable[[float], bool],
    *,
    lo: float,
    hi: float,
    step: float,
) -> float | None:
    """Return the smallest value in ``[lo, hi]`` where ``predicate`` becomes True.

    The scan assumes ``predicate(lo) is False`` and searches upward for the first
    flip. Returns ``None`` when the predicate never flips within the range or when
    the initial sample is already True (no detectable boundary).
    """
    if lo >= hi or step <= 0.0:
        return None
    if predicate(lo):
        return None
    value = lo
    while value < hi:
        value = min(value + step, hi)
        if predicate(value):
            return value
    return None


def _make_obstacle_predicate(
    *,
    robot_xy: tuple[float, float],
    obstacle_lines: np.ndarray,
    map_width: float,
    map_height: float,
    ped_positions: np.ndarray | None = None,
):
    """Build a ``radius -> is_collision`` predicate using the runtime component.

    Returns:
        A ``(predicate, state)`` pair; ``predicate(radius)`` runs the runtime
        obstacle-collision component at the supplied robot radius.
    """
    state = {"pos": robot_xy}

    def agent_coords() -> tuple[float, float]:
        return state["pos"]

    def ped_coords() -> np.ndarray:
        return ped_positions if ped_positions is not None else np.empty((0, 2), dtype=float)

    def predicate(radius: float) -> bool:
        occupancy = ContinuousOccupancy(
            width=map_width,
            height=map_height,
            get_agent_coords=agent_coords,
            get_goal_coords=lambda: (0.0, 0.0),
            get_obstacle_coords=lambda: obstacle_lines,
            get_pedestrian_coords=ped_coords,
            agent_radius=float(radius),
            ped_radius=0.0,
        )
        return bool(occupancy.is_obstacle_collision)

    return predicate, state


def _make_pedestrian_predicate(
    *,
    robot_xy: tuple[float, float],
    map_width: float,
    map_height: float,
    robot_radius: float,
    ped_radius: float,
):
    """Build a ``pedestrian-offset -> separated`` predicate.

    The pedestrian is placed at ``robot_xy + (offset, 0)`` so the centre distance
    equals ``offset``. The predicate returns ``True`` when the pair is *separated*
    (not in contact), so the differential scan's False->True boundary lands exactly
    at the contact threshold ``robot_radius + ped_radius``: below it the pair is in
    contact (predicate False), at/above it the pair is separated (predicate True).

    Returns:
        A ``separated(offset)`` predicate (True when the pair is not in contact).
    """
    state = {"ped_x": robot_xy[0]}

    def agent_coords() -> tuple[float, float]:
        return robot_xy

    def ped_coords() -> np.ndarray:
        return np.array([[state["ped_x"], robot_xy[1]]], dtype=float)

    def predicate(offset: float) -> bool:
        state["ped_x"] = robot_xy[0] + float(offset)
        occupancy = ContinuousOccupancy(
            width=map_width,
            height=map_height,
            get_agent_coords=agent_coords,
            get_goal_coords=lambda: (0.0, 0.0),
            get_obstacle_coords=lambda: np.empty((0, 4), dtype=float),
            get_pedestrian_coords=ped_coords,
            agent_radius=float(robot_radius),
            ped_radius=float(ped_radius),
        )
        # Invert so the scan's False->True flip coincides with the contact boundary.
        return not bool(occupancy.is_pedestrian_collision)

    return predicate


# --- surface probes ----------------------------------------------------------


def probe_simulator_collision_geometry(
    geometry: CanaryGeometry,
    *,
    selected_robot_radius_m: float | None = None,
    configured_robot_radius_m: float | None = None,
    tolerance_m: float = DEFAULT_RADIUS_TOLERANCE_M,
    scan_step_m: float = DEFAULT_SCAN_STEP_M,
) -> SurfaceVerdict:
    """Surface 1: the runtime obstacle-collision geometry binds to ``robot_radius``.

    Anchors the robot inside the corridor and scans the configured radius upward
    to find where ``ContinuousOccupancy.is_obstacle_collision`` flips to True. The
    flip must coincide with the measured wall distance: that proves the simulator
    collision geometry is the configured robot disc, not a hardcoded constant.

    Returns:
        SurfaceVerdict: Pass when the collision flip tracks the wall distance.
    """
    wall_distance = float(geometry.wall_distance_m)
    if not math.isfinite(wall_distance) or wall_distance <= 0.0:
        return SurfaceVerdict(
            surface=SURFACE_SIMULATOR_GEOMETRY,
            status=STATUS_FAIL,
            probe="runtime_obstacle_collision_radius_scan",
            expected="A finite positive nearest-wall distance to scan against.",
            observed=f"wall_distance_m={wall_distance!r}",
            evidence={"wall_distance_m": wall_distance},
            note="Geometry-sensitive scenario did not yield a usable wall distance.",
        )

    predicate, _ = _make_obstacle_predicate(
        robot_xy=geometry.route_point,
        obstacle_lines=geometry.obstacle_lines_runtime,
        map_width=geometry.map_width,
        map_height=geometry.map_height,
    )
    # Scan well past the wall distance so a constant radius would be caught.
    scan_hi = wall_distance + max(0.5, 0.25 * wall_distance)
    try:
        flip_radius = _scan_flip(predicate, lo=0.0, hi=scan_hi, step=scan_step_m)
        below = predicate(max(0.0, wall_distance - 0.3))
        above = predicate(wall_distance + 0.3)
    except _CANARY_OPERATIONAL_ERRORS as exc:
        return SurfaceVerdict(
            surface=SURFACE_SIMULATOR_GEOMETRY,
            status=STATUS_FAIL,
            probe="runtime_obstacle_collision_radius_scan",
            expected="Runtime obstacle collision component runs without error.",
            observed=f"probe raised {type(exc).__name__}: {exc}",
            evidence={"wall_distance_m": wall_distance},
            note="Fail-closed: the runtime collision component could not be exercised.",
        )

    config_ok = True
    config_evidence: dict[str, Any] = {}
    if selected_robot_radius_m is not None or configured_robot_radius_m is not None:
        selected_robot = (
            float(selected_robot_radius_m)
            if selected_robot_radius_m is not None
            else float(geometry.configured_robot_radius_m or 0.0)
        )
        config_ok = (
            configured_robot_radius_m is not None
            and math.isfinite(float(configured_robot_radius_m))
            and abs(float(configured_robot_radius_m) - selected_robot) <= tolerance_m
        )
        config_evidence = {
            "selected_robot_radius_m": selected_robot,
            "configured_robot_radius_m": configured_robot_radius_m,
            "selected_configuration_matches": bool(config_ok),
        }
    evidence = {
        "runtime_component": "ContinuousOccupancy.is_obstacle_collision",
        "anchor_point_xy": list(geometry.route_point),
        "wall_distance_m": wall_distance,
        "collision_flip_radius_m": flip_radius,
        "collision_below_wall": bool(below),
        "collision_above_wall": bool(above),
        "tolerance_m": tolerance_m,
        **config_evidence,
    }
    delta = abs((flip_radius or math.inf) - wall_distance)
    ok = (
        flip_radius is not None
        and below is False
        and above is True
        and delta <= tolerance_m
        and config_ok
    )
    return SurfaceVerdict(
        surface=SURFACE_SIMULATOR_GEOMETRY,
        status=STATUS_PASS if ok else STATUS_FAIL,
        probe="runtime_obstacle_collision_radius_scan",
        expected=(
            "Collision flips from False to True exactly at the measured wall "
            f"distance ({wall_distance:.6g} m) as robot_radius increases."
        ),
        observed=(
            f"flip_radius={flip_radius!r}, below_wall={below}, above_wall={above}, "
            f"|flip - wall|={delta:.6g} m."
        ),
        evidence=evidence,
        note=None if ok else "selected radius does not match the effective scenario configuration",
    )


def probe_obstacle_pedestrian_contact(
    geometry: CanaryGeometry,
    *,
    selected_robot_radius_m: float,
    selected_ped_radius_m: float,
    configured_robot_radius_m: float | None = None,
    configured_ped_radius_m: float | None = None,
    tolerance_m: float = DEFAULT_RADIUS_TOLERANCE_M,
    scan_step_m: float = DEFAULT_SCAN_STEP_M,
) -> SurfaceVerdict:
    """Surface 2: obstacle and pedestrian contact logic bind to the radii.

    Pedestrian contact uses the sum ``robot_radius + ped_radius``. The probe scans
    the robot-pedestrian centre distance for two distinct radius pairs and checks
    the contact boundary equals the sum for each pair. Two pairs are required so a
    hardcoded contact distance cannot pass.

    Returns:
        SurfaceVerdict: Pass when both pairs flip at ``robot_radius + ped_radius``.
    """
    pairs = (
        (float(selected_robot_radius_m), float(selected_ped_radius_m)),
        # An alternate pair with a clearly different sum rules out a constant.
        (
            max(0.1, float(selected_robot_radius_m) + 0.2),
            max(0.1, float(selected_ped_radius_m) + 0.1),
        ),
    )
    # Anchor away from walls so obstacle geometry cannot trigger first. Use a large
    # open map region by placing the probe in an empty sub-region of the map.
    anchor = (geometry.map_width * 0.5, geometry.map_height * 0.5)
    observations: list[dict[str, Any]] = []
    all_ok = True
    for robot_radius, ped_radius in pairs:
        expected_sum = robot_radius + ped_radius
        predicate = _make_pedestrian_predicate(
            robot_xy=anchor,
            map_width=geometry.map_width,
            map_height=geometry.map_height,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
        )
        try:
            flip_distance = _scan_flip(
                predicate,
                lo=0.0,
                hi=expected_sum + 0.5,
                step=scan_step_m,
            )
        except _CANARY_OPERATIONAL_ERRORS as exc:
            observations.append(
                {
                    "robot_radius_m": robot_radius,
                    "ped_radius_m": ped_radius,
                    "expected_contact_distance_m": expected_sum,
                    "contact_flip_distance_m": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            all_ok = False
            continue
        delta = abs((flip_distance or math.inf) - expected_sum)
        pair_ok = flip_distance is not None and delta <= tolerance_m
        all_ok = all_ok and pair_ok
        observations.append(
            {
                "robot_radius_m": robot_radius,
                "ped_radius_m": ped_radius,
                "expected_contact_distance_m": expected_sum,
                "contact_flip_distance_m": flip_distance,
                "delta_m": delta,
                "pass": bool(pair_ok),
            }
        )

    config_ok, config_evidence = _configuration_binding_evidence(
        selected_robot_radius_m=selected_robot_radius_m,
        selected_ped_radius_m=selected_ped_radius_m,
        configured_robot_radius_m=configured_robot_radius_m,
        configured_ped_radius_m=configured_ped_radius_m,
        tolerance_m=tolerance_m,
    )
    all_ok = all_ok and config_ok
    return SurfaceVerdict(
        surface=SURFACE_OBSTACLE_PEDESTRIAN_CONTACT,
        status=STATUS_PASS if all_ok else STATUS_FAIL,
        probe="runtime_pedestrian_contact_distance_scan",
        expected=(
            "Pedestrian contact flips at centre distance == robot_radius + ped_radius "
            "for two distinct radius pairs."
        ),
        observed=f"per-pair flip distances: {observations}",
        evidence={
            "runtime_component": "ContinuousOccupancy.is_pedestrian_collision",
            "anchor_point_xy": list(anchor),
            "pairs": observations,
            "tolerance_m": tolerance_m,
            **config_evidence,
        },
        note=(
            None
            if all_ok
            else "selected radius does not match the effective scenario configuration"
        ),
    )


def _stub_completion_runner(
    record: Mapping[str, Any] | None = None,
) -> Callable[[Mapping[str, Any], int, int | None, str], Mapping[str, Any]]:
    """Return a deterministic actor-free rollout runner for the oracle probe.

    The feasibility oracle's radius binding lives in its geometric margin (the
    certifier inflates static obstacles by the envelope radius). The rollout only
    contributes the completion margin, which is not radius-sensitive for this
    probe, so a deterministic stub removes rollout noise without weakening the
    radius-binding claim. The stub reports a completed route.
    """
    payload = dict(record or {})
    payload.setdefault("outcome", {"route_complete": True})
    payload.setdefault("termination_reason", "route_complete")

    def _run(
        _scenario: Mapping[str, Any],
        _seed: int,
        _horizon: int | None,
        _algo: str,
    ) -> Mapping[str, Any]:
        return payload

    return _run


def probe_feasibility_oracle(
    geometry: CanaryGeometry,
    *,
    radius_a_m: float,
    radius_b_m: float,
    selected_robot_radius_m: float | None = None,
    selected_ped_radius_m: float | None = None,
    configured_robot_radius_m: float | None = None,
    configured_ped_radius_m: float | None = None,
    tolerance_m: float = DEFAULT_RADIUS_TOLERANCE_M,
) -> SurfaceVerdict:
    """Surface 3: the planner-free feasibility oracle inflates by the envelope radius.

    Runs the real ``run_feasibility_oracle`` entry point on envelope-overridden
    variants of the scenario at two radii. The geometric
    ``minimum_static_clearance_m`` must decrease by exactly the radius delta
    (clearance = wall_distance - radius), proving the oracle consumes the injected
    envelope radius rather than a fixed default.

    Returns:
        SurfaceVerdict: Pass when the clearance delta equals the radius delta.
    """
    radii = (float(radius_a_m), float(radius_b_m))
    if radii[0] == radii[1]:
        return SurfaceVerdict(
            surface=SURFACE_FEASIBILITY_ORACLE,
            status=STATUS_FAIL,
            probe="envelope_radius_clearance_delta",
            expected="Two distinct envelope radii to compare.",
            observed=f"radius_a_m == radius_b_m == {radii[0]}",
            evidence={"radius_a_m": radii[0], "radius_b_m": radii[1]},
            note="Probe requires two distinct envelope radii.",
        )

    config = FeasibilityOracleConfig(
        scenario_path=geometry.scenario_path,
        envelope_radii_m=radii,
        rollout_algo="goal",
        rollout_seed=0,
    )
    runner = _stub_completion_runner()
    clearances: dict[float, float | None] = {}
    margins: dict[float, float | None] = {}
    error: str | None = None
    try:
        for radius in radii:
            scenario = make_envelope_scenario(geometry.scenario, envelope_radius_m=radius)
            verdict = run_feasibility_oracle(
                scenario,
                config=config,
                envelope_radius_m=radius,
                episode_runner=runner,
            )
            clearances[radius] = verdict.geometric.min_static_clearance_m
            margins[radius] = verdict.geometric.corridor_envelope_margin_m
    except _CANARY_OPERATIONAL_ERRORS as exc:
        error = f"{type(exc).__name__}: {exc}"

    radius_delta = abs(radii[1] - radii[0])
    if error is not None:
        return SurfaceVerdict(
            surface=SURFACE_FEASIBILITY_ORACLE,
            status=STATUS_FAIL,
            probe="envelope_radius_clearance_delta",
            expected="Feasibility oracle runs without error at both envelope radii.",
            observed=f"oracle raised {error}",
            evidence={
                "radius_a_m": radii[0],
                "radius_b_m": radii[1],
                "clearances": clearances,
            },
            note="Fail-closed: the oracle could not produce a geometric margin.",
        )

    clearance_a = clearances.get(radii[0])
    clearance_b = clearances.get(radii[1])
    if clearance_a is None or clearance_b is None:
        return SurfaceVerdict(
            surface=SURFACE_FEASIBILITY_ORACLE,
            status=STATUS_FAIL,
            probe="envelope_radius_clearance_delta",
            expected="Non-None minimum_static_clearance_m at both envelope radii.",
            observed=f"clearances={clearances}",
            evidence={
                "radius_a_m": radii[0],
                "radius_b_m": radii[1],
                "clearances": clearances,
                "margins": margins,
            },
            note="Scenario has no static obstacles on the route; cannot probe clearance.",
        )

    config_ok = True
    config_evidence: dict[str, Any] = {}
    if any(
        value is not None
        for value in (
            selected_robot_radius_m,
            selected_ped_radius_m,
            configured_robot_radius_m,
            configured_ped_radius_m,
        )
    ):
        config_ok, config_evidence = _configuration_binding_evidence(
            selected_robot_radius_m=(
                selected_robot_radius_m
                if selected_robot_radius_m is not None
                else float(geometry.configured_robot_radius_m or 0.0)
            ),
            selected_ped_radius_m=(
                selected_ped_radius_m
                if selected_ped_radius_m is not None
                else float(geometry.configured_ped_radius_m or 0.0)
            ),
            configured_robot_radius_m=configured_robot_radius_m,
            configured_ped_radius_m=configured_ped_radius_m,
            tolerance_m=tolerance_m,
        )

    observed_delta = float(clearance_a - clearance_b)  # clearance shrinks as radius grows
    # Clearance == wall_distance - radius, so clearance(a) - clearance(b) == radius(b) - radius(a)
    # when radius_b > radius_a. Compare absolute values to stay sign-agnostic.
    ok = (
        abs(abs(observed_delta) - radius_delta) <= max(tolerance_m, radius_delta * 1e-6)
        and config_ok
    )
    return SurfaceVerdict(
        surface=SURFACE_FEASIBILITY_ORACLE,
        status=STATUS_PASS if ok else STATUS_FAIL,
        probe="envelope_radius_clearance_delta",
        expected=(
            f"|clearance(a) - clearance(b)| == |radius_b - radius_a| == "
            f"{radius_delta:.6g} m (clearance tracks the envelope radius)."
        ),
        observed=(
            f"clearance(a={radii[0]})={clearance_a:.6g}, "
            f"clearance(b={radii[1]})={clearance_b:.6g}, "
            f"|observed_delta|={abs(observed_delta):.6g} m."
        ),
        evidence={
            "oracle_entry_point": "run_feasibility_oracle",
            "radius_a_m": radii[0],
            "radius_b_m": radii[1],
            "min_static_clearance_a_m": clearance_a,
            "min_static_clearance_b_m": clearance_b,
            "clearance_delta_m": observed_delta,
            "expected_radius_delta_m": radius_delta,
            "corridor_envelope_margin_a_m": margins.get(radii[0]),
            "corridor_envelope_margin_b_m": margins.get(radii[1]),
            "tolerance_m": tolerance_m,
            "rollout_mode": "deterministic_stub",
            **config_evidence,
        },
        note=(
            "Rollout completion is stubbed for determinism; the radius binding is "
            "proven through the oracle's geometric (certifier) margin."
            if config_ok
            else "selected radius does not match the effective scenario configuration",
        ),
    )


def probe_metric_metadata_and_output_rows(
    *,
    selected_robot_radius_m: float,
    selected_ped_radius_m: float,
    scenario: Mapping[str, Any] | None = None,
    configured_robot_radius_m: float | None = None,
    configured_ped_radius_m: float | None = None,
    tolerance_m: float = DEFAULT_RADIUS_TOLERANCE_M,
) -> SurfaceVerdict:
    """Surface 4: metric metadata and output rows consume the recorded radii.

    Two sub-probes:
      1. The same fixed trajectory yields a different ``human_collisions`` count
         when only the recorded ``robot_radius``/``ped_radius`` change, proving the
         metric consumes the recorded radii (clearance = centre distance - sum).
      2. The runner's scenario-radius resolver returns the configured radius from a
         scenario payload, proving the radius is read for row metadata.

    Returns:
        SurfaceVerdict: Pass when the metric and resolver both track the radii.
    """
    from robot_sf.benchmark.runner import (  # noqa: PLC0415
        _build_episode_data,
        _scenario_ped_radius_m,
        _scenario_robot_radius_m,
    )

    # Sub-probe 1: identical trajectory, two recorded radius pairs.
    robot_pos = np.array([[10.0, 9.7], [10.0, 9.7], [10.0, 9.7]], dtype=float)
    peds_pos = np.tile(np.array([[[10.0, 9.7 + 0.9]]], dtype=float), (3, 1, 1))
    centre_distance = 0.9

    def _collisions(robot_radius: float, ped_radius: float) -> float:
        ep = _build_episode_data(
            list(robot_pos),
            [np.zeros(2, dtype=float) for _ in range(len(robot_pos))],
            [np.zeros(2, dtype=float) for _ in range(len(robot_pos))],
            list(peds_pos),
            [np.zeros((1, 2), dtype=float) for _ in range(len(peds_pos))],
            None,
            np.array([17.0, 9.7]),
            0.1,
            None,
            robot_radius=float(robot_radius),
            ped_radius=float(ped_radius),
        )
        return float(human_collisions(ep))

    payload = (
        dict(scenario)
        if scenario is not None
        else {
            "robot_config": {"radius": float(selected_robot_radius_m)},
            "simulation_config": {"ped_radius": float(selected_ped_radius_m)},
        }
    )
    try:
        resolved_robot = float(_scenario_robot_radius_m(payload))
        resolved_ped = float(_scenario_ped_radius_m(payload))
    except _CANARY_OPERATIONAL_ERRORS as exc:
        return SurfaceVerdict(
            surface=SURFACE_METRIC_METADATA_ROWS,
            status=STATUS_FAIL,
            probe="episode_radius_metadata_propagation",
            expected="Runner scenario-radius resolver runs without error.",
            observed=f"resolver raised {type(exc).__name__}: {exc}",
            evidence={"payload": payload},
            note="Fail-closed: the runner radius resolver could not be exercised.",
        )

    pair_a = (resolved_robot, resolved_ped)
    pair_b = (pair_a[0] + 0.2, pair_a[1] + 0.2)
    coll_a = _collisions(*pair_a)
    coll_b = _collisions(*pair_b)
    metrics_ok = coll_a != coll_b

    resolver_ok = (
        abs(resolved_robot - selected_robot_radius_m) <= tolerance_m
        and abs(resolved_ped - selected_ped_radius_m) <= tolerance_m
    )
    config_ok, config_evidence = _configuration_binding_evidence(
        selected_robot_radius_m=selected_robot_radius_m,
        selected_ped_radius_m=selected_ped_radius_m,
        configured_robot_radius_m=configured_robot_radius_m,
        configured_ped_radius_m=configured_ped_radius_m,
        tolerance_m=tolerance_m,
    )
    binding_ok = metrics_ok and resolver_ok and config_ok

    return SurfaceVerdict(
        surface=SURFACE_METRIC_METADATA_ROWS,
        status=STATUS_PASS if binding_ok else STATUS_FAIL,
        probe="episode_radius_metadata_propagation",
        expected=(
            "Same trajectory records different human_collisions when only the radii "
            "change, and the runner resolves the configured radii from the scenario."
        ),
        observed=(
            f"human_collisions(pair_a={pair_a})={coll_a}, "
            f"human_collisions(pair_b={pair_b})={coll_b}; "
            f"resolved_robot={resolved_robot:.6g} (selected {selected_robot_radius_m}), "
            f"resolved_ped={resolved_ped:.6g} (selected {selected_ped_radius_m})."
        ),
        evidence={
            "metric": "human_collisions",
            "episode_builder": "robot_sf.benchmark.runner._build_episode_data",
            "centre_distance_m": centre_distance,
            "pair_a": {"robot_radius_m": pair_a[0], "ped_radius_m": pair_a[1]},
            "pair_b": {"robot_radius_m": pair_b[0], "ped_radius_m": pair_b[1]},
            "human_collisions_a": coll_a,
            "human_collisions_b": coll_b,
            "collisions_responsive_to_radius": bool(metrics_ok),
            "scenario_robot_radius_m": resolved_robot,
            "scenario_ped_radius_m": resolved_ped,
            "resolved_robot_radius_m": resolved_robot,
            "resolved_ped_radius_m": resolved_ped,
            "resolver_responsive": bool(resolver_ok),
            "tolerance_m": tolerance_m,
            **config_evidence,
        },
        note=None if binding_ok else "selected radius did not reach the production output-row path",
    )


def probe_planner_inputs(
    *,
    selected_robot_radius_m: float,
    selected_ped_radius_m: float,
    scenario: Mapping[str, Any] | None = None,
    configured_robot_radius_m: float | None = None,
    configured_ped_radius_m: float | None = None,
    tolerance_m: float = DEFAULT_RADIUS_TOLERANCE_M,
) -> SurfaceVerdict:
    """Surface 5: the planner-facing observation carries the configured radii.

    Builds the canonical planner observation through the runner's
    ``_build_observation`` helper (the same builder the episode loop uses to feed
    baseline planners) and asserts the robot and agent payloads carry the selected
    radii. A planner that never receives the radius cannot consume it, so this is
    the binding surface for planner inputs.

    Returns:
        SurfaceVerdict: Pass when the observation carries the configured radii.
    """
    from robot_sf.baselines.interface import Observation  # noqa: PLC0415
    from robot_sf.benchmark.runner import (  # noqa: PLC0415
        _build_observation,
        _scenario_ped_radius_m,
        _scenario_robot_radius_m,
    )

    robot_pos = np.array([10.0, 9.7], dtype=float)
    robot_vel = np.array([0.0, 0.0], dtype=float)
    robot_goal = np.array([17.0, 9.7], dtype=float)
    ped_positions = np.array([[10.0, 8.7], [11.0, 9.7]], dtype=float)
    dt = 0.1
    effective_robot_radius_m = float(selected_robot_radius_m)
    effective_ped_radius_m = float(selected_ped_radius_m)
    if scenario is not None:
        try:
            effective_robot_radius_m = float(_scenario_robot_radius_m(dict(scenario)))
            effective_ped_radius_m = float(_scenario_ped_radius_m(dict(scenario)))
        except _CANARY_OPERATIONAL_ERRORS as exc:
            return SurfaceVerdict(
                surface=SURFACE_PLANNER_INPUTS,
                status=STATUS_FAIL,
                probe="planner_observation_radius_payload",
                expected="Runner scenario-radius resolver runs without error.",
                observed=f"resolver raised {type(exc).__name__}: {exc}",
                evidence={},
                note="Fail-closed: the runner radius resolver could not be exercised.",
            )
    try:
        observation = _build_observation(
            Observation,
            robot_pos,
            robot_vel,
            robot_goal,
            ped_positions,
            dt,
            robot_radius=effective_robot_radius_m,
            ped_radius=effective_ped_radius_m,
        )
    except _CANARY_OPERATIONAL_ERRORS as exc:
        return SurfaceVerdict(
            surface=SURFACE_PLANNER_INPUTS,
            status=STATUS_FAIL,
            probe="planner_observation_radius_payload",
            expected="Runner _build_observation runs without error.",
            observed=f"builder raised {type(exc).__name__}: {exc}",
            evidence={},
            note="Fail-closed: the planner observation builder could not be exercised.",
        )

    robot_payload = observation.robot
    agent_payloads = observation.agents
    obs_robot_radius = float(robot_payload.get("radius", math.nan))
    obs_ped_radii = [float(agent.get("radius", math.nan)) for agent in agent_payloads]
    robot_ok = abs(obs_robot_radius - effective_robot_radius_m) <= tolerance_m
    agents_ok = bool(agent_payloads) and all(
        abs(r - effective_ped_radius_m) <= tolerance_m for r in obs_ped_radii
    )
    selection_ok, config_evidence = _configuration_binding_evidence(
        selected_robot_radius_m=selected_robot_radius_m,
        selected_ped_radius_m=selected_ped_radius_m,
        configured_robot_radius_m=configured_robot_radius_m,
        configured_ped_radius_m=configured_ped_radius_m,
        tolerance_m=tolerance_m,
    )
    effective_selection_ok = (
        abs(effective_robot_radius_m - selected_robot_radius_m) <= tolerance_m
        and abs(effective_ped_radius_m - selected_ped_radius_m) <= tolerance_m
    )
    ok = robot_ok and agents_ok and selection_ok and effective_selection_ok
    return SurfaceVerdict(
        surface=SURFACE_PLANNER_INPUTS,
        status=STATUS_PASS if ok else STATUS_FAIL,
        probe="planner_observation_radius_payload",
        expected=(
            "Observation.robot.radius == selected_robot_radius_m and every "
            "Observation.agents[].radius == selected_ped_radius_m."
        ),
        observed=(f"obs.robot.radius={obs_robot_radius:.6g}, obs.agents[].radius={obs_ped_radii}."),
        evidence={
            "builder": "robot_sf.benchmark.runner._build_observation",
            "selected_robot_radius_m": float(selected_robot_radius_m),
            "selected_ped_radius_m": float(selected_ped_radius_m),
            "effective_robot_radius_m": effective_robot_radius_m,
            "effective_ped_radius_m": effective_ped_radius_m,
            "observed_robot_radius_m": obs_robot_radius,
            "observed_agent_radii_m": obs_ped_radii,
            "robot_payload_carries_radius": bool(robot_ok),
            "agent_payloads_carry_radius": bool(agents_ok),
            "effective_selection_matches": bool(effective_selection_ok),
            "tolerance_m": tolerance_m,
            **config_evidence,
        },
        note=None if ok else "selected radius did not reach the production planner-input path",
    )


# --- orchestrator ------------------------------------------------------------


def run_radius_binding_canary(
    *,
    scenario_path: Path | str = DEFAULT_SCENARIO_REL,
    selected_robot_radius_m: float = DEFAULT_SELECTED_ROBOT_RADIUS_M,
    selected_ped_radius_m: float = DEFAULT_SELECTED_PED_RADIUS_M,
    tolerance_m: float = DEFAULT_RADIUS_TOLERANCE_M,
    scan_step_m: float = DEFAULT_SCAN_STEP_M,
    geometry: CanaryGeometry | None = None,
) -> CanaryVerdict:
    """Run the full five-surface radius-binding canary and return its verdict.

    Args:
        scenario_path: Geometry-sensitive scenario manifest path.
        selected_robot_radius_m: Primary selected robot envelope radius (metres).
        selected_ped_radius_m: Primary selected pedestrian radius (metres).
        tolerance_m: Absolute radius tolerance for accepting a binding.
        scan_step_m: Step size for the differential radius/distance scans.
        geometry: Optional pre-loaded :class:`CanaryGeometry` (for tests).

    Returns:
        A :class:`CanaryVerdict` with one entry per binding surface.
    """
    caveats: list[str] = [
        "Diagnostic-only pre-campaign binding check; not benchmark evidence.",
        "Frozen 0.0.3.post1 metric semantics are read but never modified.",
        "Feasibility-oracle rollout completion is deterministically stubbed; the "
        "radius binding is proven via the oracle's geometric (certifier) margin.",
    ]
    try:
        resolved_geometry = geometry or load_canary_geometry(Path(scenario_path))
    except _CANARY_OPERATIONAL_ERRORS as exc:
        fail = SurfaceVerdict(
            surface="scenario_geometry",
            status=STATUS_FAIL,
            probe="load_canary_geometry",
            expected="Geometry-sensitive scenario loads and yields a wall distance.",
            observed=f"load raised {type(exc).__name__}: {exc}",
            evidence={"scenario_path": str(scenario_path)},
            note="Fail-closed: cannot run surface probes without scenario geometry.",
        )
        return _assemble_verdict(
            selected_robot_radius_m=selected_robot_radius_m,
            selected_ped_radius_m=selected_ped_radius_m,
            scenario_facts={"scenario_path": str(scenario_path)},
            surfaces=[fail],
            caveats=caveats,
        )

    scenario_facts = {
        "scenario_path": str(resolved_geometry.scenario_path),
        "scenario_id": resolved_geometry.scenario_id,
        "map_name": resolved_geometry.map_name,
        "route_point_xy": list(resolved_geometry.route_point),
        "goal_point_xy": list(resolved_geometry.goal_point),
        "wall_distance_m": resolved_geometry.wall_distance_m,
        "map_width_m": resolved_geometry.map_width,
        "map_height_m": resolved_geometry.map_height,
        "obstacle_segment_count": int(resolved_geometry.obstacle_lines_runtime.shape[0]),
        "configured_robot_radius_m": resolved_geometry.configured_robot_radius_m,
        "configured_ped_radius_m": resolved_geometry.configured_ped_radius_m,
    }

    # Two distinct envelope radii for the feasibility-oracle probe, both below the
    # wall distance so the corridor stays geometrically feasible and the clearance
    # signal is well-defined.
    wall = resolved_geometry.wall_distance_m
    radius_a = min(float(selected_robot_radius_m), 0.5 * wall)
    radius_b = min(radius_a + max(0.2, 0.2 * wall), 0.95 * wall)
    if radius_a <= 0.0 or radius_b <= radius_a:
        radius_a = max(0.1, 0.25 * wall)
        radius_b = max(radius_a + 0.2, 0.6 * wall)

    surfaces: list[SurfaceVerdict] = [
        probe_simulator_collision_geometry(
            resolved_geometry,
            selected_robot_radius_m=selected_robot_radius_m,
            configured_robot_radius_m=resolved_geometry.configured_robot_radius_m,
            tolerance_m=tolerance_m,
            scan_step_m=scan_step_m,
        ),
        probe_obstacle_pedestrian_contact(
            resolved_geometry,
            selected_robot_radius_m=selected_robot_radius_m,
            selected_ped_radius_m=selected_ped_radius_m,
            configured_robot_radius_m=resolved_geometry.configured_robot_radius_m,
            configured_ped_radius_m=resolved_geometry.configured_ped_radius_m,
            tolerance_m=tolerance_m,
            scan_step_m=scan_step_m,
        ),
        probe_feasibility_oracle(
            resolved_geometry,
            radius_a_m=radius_a,
            radius_b_m=radius_b,
            selected_robot_radius_m=selected_robot_radius_m,
            selected_ped_radius_m=selected_ped_radius_m,
            configured_robot_radius_m=resolved_geometry.configured_robot_radius_m,
            configured_ped_radius_m=resolved_geometry.configured_ped_radius_m,
            tolerance_m=tolerance_m,
        ),
        probe_metric_metadata_and_output_rows(
            selected_robot_radius_m=selected_robot_radius_m,
            selected_ped_radius_m=selected_ped_radius_m,
            scenario=resolved_geometry.scenario,
            configured_robot_radius_m=resolved_geometry.configured_robot_radius_m,
            configured_ped_radius_m=resolved_geometry.configured_ped_radius_m,
            tolerance_m=tolerance_m,
        ),
        probe_planner_inputs(
            selected_robot_radius_m=selected_robot_radius_m,
            selected_ped_radius_m=selected_ped_radius_m,
            scenario=resolved_geometry.scenario,
            configured_robot_radius_m=resolved_geometry.configured_robot_radius_m,
            configured_ped_radius_m=resolved_geometry.configured_ped_radius_m,
            tolerance_m=tolerance_m,
        ),
    ]
    return _assemble_verdict(
        selected_robot_radius_m=selected_robot_radius_m,
        selected_ped_radius_m=selected_ped_radius_m,
        scenario_facts=scenario_facts,
        surfaces=surfaces,
        caveats=caveats,
    )


def _assemble_verdict(
    *,
    selected_robot_radius_m: float,
    selected_ped_radius_m: float,
    scenario_facts: Mapping[str, Any],
    surfaces: Sequence[SurfaceVerdict],
    caveats: Sequence[str],
) -> CanaryVerdict:
    """Assemble the top-level verdict from per-surface results.

    Returns:
        CanaryVerdict: ``go`` only when every surface passed.
    """
    verdict = (
        VERDICT_GO if surfaces and all(s.status == STATUS_PASS for s in surfaces) else VERDICT_NO_GO
    )
    return CanaryVerdict(
        schema_version=RADIUS_BINDING_CANARY_SCHEMA,
        claim_boundary=CANARY_CLAIM_BOUNDARY,
        evidence_status="diagnostic-only",
        scenario=dict(scenario_facts),
        selected_robot_radius_m=float(selected_robot_radius_m),
        selected_ped_radius_m=float(selected_ped_radius_m),
        surfaces=list(surfaces),
        verdict=verdict,
        caveats=list(caveats),
        generated_at=datetime.now(UTC).isoformat(),
    )


def surface_verdict_to_dict(verdict: SurfaceVerdict) -> dict[str, Any]:
    """Return a JSON-safe dict for one surface verdict."""
    return asdict(verdict)


def canary_verdict_to_dict(verdict: CanaryVerdict) -> dict[str, Any]:
    """Return a JSON-safe, machine-readable dict for a canary verdict."""
    return {
        "schema_version": verdict.schema_version,
        "claim_boundary": verdict.claim_boundary,
        "evidence_status": verdict.evidence_status,
        "scenario": dict(verdict.scenario),
        "selected_robot_radius_m": verdict.selected_robot_radius_m,
        "selected_ped_radius_m": verdict.selected_ped_radius_m,
        "surfaces": [surface_verdict_to_dict(s) for s in verdict.surfaces],
        "verdict": verdict.verdict,
        "caveats": list(verdict.caveats),
        "generated_at": verdict.generated_at,
    }


__all__ = [
    "CANARY_CLAIM_BOUNDARY",
    "CANARY_SURFACES",
    "DEFAULT_RADIUS_TOLERANCE_M",
    "DEFAULT_SCAN_STEP_M",
    "DEFAULT_SCENARIO_REL",
    "DEFAULT_SELECTED_PED_RADIUS_M",
    "DEFAULT_SELECTED_ROBOT_RADIUS_M",
    "RADIUS_BINDING_CANARY_SCHEMA",
    "STATUS_FAIL",
    "STATUS_PASS",
    "SURFACE_FEASIBILITY_ORACLE",
    "SURFACE_METRIC_METADATA_ROWS",
    "SURFACE_OBSTACLE_PEDESTRIAN_CONTACT",
    "SURFACE_PLANNER_INPUTS",
    "SURFACE_SIMULATOR_GEOMETRY",
    "VERDICT_GO",
    "VERDICT_NO_GO",
    "CanaryGeometry",
    "CanaryVerdict",
    "SurfaceVerdict",
    "canary_verdict_to_dict",
    "load_canary_geometry",
    "probe_feasibility_oracle",
    "probe_metric_metadata_and_output_rows",
    "probe_obstacle_pedestrian_contact",
    "probe_planner_inputs",
    "probe_simulator_collision_geometry",
    "run_radius_binding_canary",
    "surface_verdict_to_dict",
]
