"""Footprint-orientation diagnostics for elongated AMMV bodies (issue #4762).

This module provides a lightweight, CPU-only diagnostic that compares route
clearance under circular, rectangular, and elongated Autonomous Micromobility
Vehicle (AMMV) footprint assumptions. It is a diagnostic proxy, not a full
SE(2) planner implementation (see ``docs/diagnostics/footprint_orientation.md``
and the ``claim_boundary`` in ``configs/diagnostics/footprint_orientation_v1.yaml``).

The diagnostic reports two separate quantities per scenario and footprint:

* ``centerline_clearance_m``: minimum distance from the route centerline
  (a Shapely ``LineString``) to the obstacle polygons, ignoring the footprint.
  This is what the existing circular route-clearance contract reasons about.
* ``footprint_aware_clearance_m``: minimum distance from an oriented rigid
  footprint, sampled along the route and oriented along the local tangent, to
  the obstacle polygons. For circular footprints this is the analytic
  centerline-minus-radius margin; for rectangular footprints it is the min
  oriented-rectangle-to-obstacle distance (0 when overlapping).

The two are reported separately so that a circular body that "clears" can be
compared directly with an elongated body that "collides" on the same route.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from robot_sf.common.types import Vec2D
from robot_sf.errors import RobotSfError

FOOTPRINT_ORIENTATION_SCHEMA_VERSION = "footprint-orientation.v1"
FOOTPRINT_KIND_CIRCULAR = "circular"
FOOTPRINT_KIND_RECTANGULAR = "rectangular"
REQUIRED_SCENARIO_FAMILY_IDS = frozenset(
    {
        "narrow_passage",
        "pedestrian_crossing",
        "occluded_corner",
        "recovery_after_avoidance",
        "blocked_path_turn_around",
    }
)


class FootprintOrientationConfigError(RobotSfError, ValueError):
    """Raised when a footprint-orientation diagnostic config violates the v1 contract."""


@dataclass(frozen=True)
class CircularFootprint:
    """Circular robot footprint (baseline, matches the existing route-clearance contract).

    Attributes:
        id: Stable footprint identifier (lowercase snake_case).
        radius_m: Footprint radius in meters.
        kind: Discriminator constant, always ``"circular"``.
    """

    id: str
    radius_m: float
    kind: str = FOOTPRINT_KIND_CIRCULAR


@dataclass(frozen=True)
class RectangularFootprint:
    """Rectangular AMMV footprint oriented along the local route tangent.

    Attributes:
        id: Stable footprint identifier (lowercase snake_case).
        length_m: Body length in meters, aligned with the travel direction.
        width_m: Body width in meters, perpendicular to the travel direction.
        kind: Discriminator constant, always ``"rectangular"``.
    """

    id: str
    length_m: float
    width_m: float
    kind: str = FOOTPRINT_KIND_RECTANGULAR


FootprintModel = CircularFootprint | RectangularFootprint


@dataclass
class FootprintClearanceResult:
    """One scenario's clearance outcome for a single footprint model.

    Attributes:
        footprint_id: Identifier of the footprint model.
        kind: ``"circular"`` or ``"rectangular"``.
        centerline_clearance_m: Min route-centerline-to-obstacle distance (``None`` if no obstacles).
        footprint_aware_clearance_m: Min oriented-footprint-to-obstacle distance
            (``None`` if no obstacles; ``0.0`` when overlapping).
        collision: True when the oriented footprint intersects any obstacle
            (boundary contact counts as collision, conservative fail-closed).
        sample_count: Number of oriented-footprint samples evaluated (0/1 for the
            analytic circular path).
        method: ``"analytic_margin"`` for circular; ``"oriented_rectangle_sampling"`` for rectangular.
    """

    footprint_id: str
    kind: str
    centerline_clearance_m: float | None
    footprint_aware_clearance_m: float | None
    collision: bool
    sample_count: int
    method: str

    @property
    def status(self) -> str:
        """Return ``"collision"`` when colliding, otherwise ``"clear"``."""

        return "collision" if self.collision else "clear"


@dataclass
class FootprintDiagnosticScenario:
    """A self-contained route-plus-obstacle diagnostic scenario fixture.

    Attributes:
        id: Scenario identifier (lowercase snake_case).
        family: Scenario family identifier from the issue candidate list.
        display_name: Human-readable scenario name.
        description: Plain-language description of what the scenario surfaces.
        mechanism: Diagnostic mechanism tag (e.g. ``"width_driven"``).
        route: Ordered route waypoints.
        obstacles: Obstacle polygons.
    """

    id: str
    family: str
    display_name: str
    description: str
    mechanism: str
    route: list[Vec2D]
    obstacles: list[Polygon]


def load_footprint_orientation_config(path: Path | str) -> dict[str, Any]:
    """Load and validate a footprint-orientation diagnostic YAML config.

    Returns:
        The validated YAML payload as a dictionary.
    """

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate_footprint_orientation_config(payload)


def validate_footprint_orientation_config(payload: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Validate the footprint-orientation diagnostic config contract.

    Returns:
        The validated payload as a dictionary.
    """

    if not isinstance(payload, Mapping):
        raise FootprintOrientationConfigError("config payload must be a mapping")
    _validate_top_level_contract(payload)
    _validate_footprints(payload.get("footprints"))
    _validate_scenario_families(payload.get("scenario_families"))
    _validate_diagnostic_parameters(payload.get("diagnostic_parameters"))
    return dict(payload)


def parse_footprints(payload: Mapping[str, Any]) -> list[FootprintModel]:
    """Parse validated footprint entries into dataclass models.

    Returns:
        Ordered list of ``CircularFootprint`` / ``RectangularFootprint``.
    """

    raw_footprints = payload.get("footprints")
    if not isinstance(raw_footprints, list):
        raise FootprintOrientationConfigError("footprints must be a non-empty list")
    models: list[FootprintModel] = []
    for entry in raw_footprints:
        if not isinstance(entry, Mapping):
            raise FootprintOrientationConfigError("footprint entries must be mappings")
        models.append(_parse_footprint(entry))
    return models


def parse_diagnostic_parameters(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated diagnostic parameters with defaults applied."""

    params = payload.get("diagnostic_parameters")
    if not isinstance(params, Mapping):
        raise FootprintOrientationConfigError("diagnostic_parameters must be a mapping")
    sample_step = float(params.get("sample_step_m", 0.1))
    max_samples = int(params.get("max_samples", 5000))
    pass_threshold = float(params.get("pass_threshold_m", 0.0))
    if not math.isfinite(sample_step) or sample_step <= 0.0:
        raise FootprintOrientationConfigError("diagnostic_parameters.sample_step_m must be > 0")
    if max_samples < 2:
        raise FootprintOrientationConfigError("diagnostic_parameters.max_samples must be >= 2")
    if not math.isfinite(pass_threshold) or pass_threshold < 0.0:
        raise FootprintOrientationConfigError("diagnostic_parameters.pass_threshold_m must be >= 0")
    return {
        "sample_step_m": sample_step,
        "max_samples": max_samples,
        "pass_threshold_m": pass_threshold,
    }


def centerline_clearance_m(
    route: Sequence[Vec2D],
    obstacle_polygons: Sequence[Polygon],
) -> float | None:
    """Compute minimum distance from the route centerline to obstacle polygons.

    Returns:
        Minimum centerline-to-obstacle distance in meters, or ``None`` when the
        route has fewer than two waypoints or there are no obstacles.
    """

    if len(route) < 2 or not obstacle_polygons:
        return None
    line = LineString(list(route))
    if line.is_empty or line.length <= 0.0:
        return None
    obstacles = unary_union(list(obstacle_polygons))
    if obstacles.is_empty:
        return None
    return float(line.distance(obstacles))


def footprint_aware_clearance_m(
    route: Sequence[Vec2D],
    footprint: FootprintModel,
    obstacle_polygons: Sequence[Polygon],
    sample_step_m: float,
    max_samples: int,
    pass_threshold_m: float = 0.0,
) -> tuple[float | None, bool, int]:
    """Compute min oriented-footprint-to-obstacle clearance along a route.

    For circular footprints the clearance is the analytic centerline-minus-radius
    margin (exact, no sampling). For rectangular footprints the route is sampled
    at ``sample_step_m`` spacing and a rigid rectangle is oriented along the
    local tangent at each sample.

    A scenario-footprint pair is reported as a collision when the min
    footprint-to-obstacle clearance is ``<= pass_threshold_m``. The default
    ``0.0`` treats boundary contact (touching or overlapping) as a collision,
    matching a conservative fail-closed diagnostic; a positive threshold flags
    near-misses within that margin.

    Returns:
        ``(clearance_m, collision, sample_count)`` where ``clearance_m`` is the
        min footprint-to-obstacle distance (``0.0`` when overlapping, ``None``
        when there are no obstacles or the route is degenerate), ``collision``
        is True when that clearance is ``<= pass_threshold_m``, and
        ``sample_count`` is the number of evaluated samples.
    """

    if len(route) < 2 or not obstacle_polygons:
        return None, False, 0
    line = LineString(list(route))
    length = float(line.length)
    if length <= 0.0:
        return None, False, 0
    obstacles = unary_union(list(obstacle_polygons))
    if obstacles.is_empty:
        return None, False, 0

    if isinstance(footprint, CircularFootprint):
        center_distance = float(line.distance(obstacles))
        clearance = max(0.0, center_distance - footprint.radius_m)
        # Boundary contact (clearance == 0) counts as collision at the default
        # threshold; a positive threshold also flags near-misses.
        collision = clearance <= pass_threshold_m
        return clearance, collision, 1

    if not isinstance(footprint, RectangularFootprint):
        raise FootprintOrientationConfigError(f"unsupported footprint type: {type(footprint)!r}")

    sample_count = min(max_samples, max(2, math.ceil(length / sample_step_m) + 1))
    eps = max(1e-3, length * 1e-6)
    # Route-level fallback heading from the first non-degenerate segment. The
    # route is guaranteed non-degenerate here (``length > 0``), so this is
    # always available and lets a locally degenerate sample (e.g. duplicate
    # consecutive waypoints) keep the full oriented footprint instead of
    # collapsing to a zero-size point, which would understate collision risk
    # and violate the fail-closed contract.
    fallback_heading = _route_fallback_heading(line)
    min_clearance: float | None = None
    for index in range(sample_count):
        distance = length * index / (sample_count - 1) if sample_count > 1 else 0.0
        point = line.interpolate(distance)
        heading = _route_tangent(line, distance, length, eps)
        if heading is None:
            heading = fallback_heading
        rect = _oriented_rectangle(point.x, point.y, footprint.length_m, footprint.width_m, heading)
        clearance = float(rect.distance(obstacles))
        if min_clearance is None or clearance < min_clearance:
            min_clearance = clearance
    if min_clearance is None:
        return None, False, sample_count
    return float(min_clearance), min_clearance <= pass_threshold_m, sample_count


def run_footprint_diagnostic(
    scenario: FootprintDiagnosticScenario,
    footprints: Sequence[FootprintModel],
    sample_step_m: float,
    max_samples: int,
    pass_threshold_m: float = 0.0,
) -> list[FootprintClearanceResult]:
    """Run all footprint models against one scenario and return per-footprint results.

    Returns:
        One ``FootprintClearanceResult`` per footprint model, in config order.
    """

    centerline = centerline_clearance_m(scenario.route, scenario.obstacles)
    results: list[FootprintClearanceResult] = []
    for footprint in footprints:
        clearance, collision, sample_count = footprint_aware_clearance_m(
            scenario.route,
            footprint,
            scenario.obstacles,
            sample_step_m,
            max_samples,
            pass_threshold_m,
        )
        method = (
            "analytic_margin"
            if isinstance(footprint, CircularFootprint)
            else "oriented_rectangle_sampling"
        )
        results.append(
            FootprintClearanceResult(
                footprint_id=footprint.id,
                kind=footprint.kind,
                centerline_clearance_m=centerline,
                footprint_aware_clearance_m=clearance,
                collision=collision,
                sample_count=sample_count,
                method=method,
            )
        )
    return results


def build_diagnostic_report(
    scenarios: Sequence[FootprintDiagnosticScenario],
    footprints: Sequence[FootprintModel],
    sample_step_m: float,
    max_samples: int,
    *,
    pass_threshold_m: float = 0.0,
    profile_id: str = "footprint_orientation_diagnostic_v1",
) -> dict[str, Any]:
    """Build a JSON-serializable diagnostic report across scenarios and footprints.

    The report distinguishes ``centerline_clearance_m`` from
    ``footprint_aware_clearance_m`` per scenario and footprint, and lists which
    footprints collide vs clear so circular-vs-elongated outcomes are comparable.

    Returns:
        A JSON-serializable report dict with ``profile_id``, ``schema_version``,
        ``claim_boundary_note``, ``diagnostic_parameters``, and a ``scenarios``
        list of per-scenario per-footprint result rows.
    """

    scenario_reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        results = run_footprint_diagnostic(
            scenario, footprints, sample_step_m, max_samples, pass_threshold_m
        )
        rows = [_result_to_row(result) for result in results]
        scenario_reports.append(
            {
                "scenario_id": scenario.id,
                "family": scenario.family,
                "display_name": scenario.display_name,
                "description": scenario.description,
                "mechanism": scenario.mechanism,
                "obstacle_count": len(scenario.obstacles),
                "route_waypoint_count": len(scenario.route),
                "results": rows,
                "collision_footprint_ids": [
                    r["footprint_id"] for r in rows if r["status"] == "collision"
                ],
                "clear_footprint_ids": [r["footprint_id"] for r in rows if r["status"] == "clear"],
            }
        )
    return {
        "profile_id": profile_id,
        "schema_version": FOOTPRINT_ORIENTATION_SCHEMA_VERSION,
        "claim_boundary_note": (
            "Diagnostic proxy only: not a full SE(2) planner implementation, not a "
            "collision-checking runtime, not a benchmark or paper-facing result, and not "
            "calibrated against real-vehicle swept-volume data."
        ),
        "diagnostic_parameters": {
            "sample_step_m": float(sample_step_m),
            "max_samples": int(max_samples),
            "pass_threshold_m": float(pass_threshold_m),
        },
        "scenarios": scenario_reports,
    }


def build_diagnostic_scenarios() -> list[FootprintDiagnosticScenario]:
    """Return the five self-contained scenario-family fixtures from issue #4762.

    These synthetic fixtures make the diagnostic reproducible without external
    map assets. Geometry is chosen so that circular and elongated footprints
    produce contrasting pass/fail outcomes via three distinct mechanisms:
    width-driven (narrow passage / pedestrian offset), turn-overrun
    length-driven (occluded corner), and forward-reach length-driven
    (blocked-path turn-around).
    """

    return [
        FootprintDiagnosticScenario(
            id="narrow_passage_v1",
            family="narrow_passage",
            display_name="Narrow passage",
            description=(
                "Straight 0.9 m corridor. A circular body clears while a "
                "shuttle-pod-class body collides; surfaces width-driven infeasibility."
            ),
            mechanism="width_driven",
            route=[(0.0, 5.0), (10.0, 5.0)],
            obstacles=[
                Polygon([(0.0, 5.45), (10.0, 5.45), (10.0, 10.0), (0.0, 10.0)]),
                Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 4.55), (0.0, 4.55)]),
            ],
        ),
        FootprintDiagnosticScenario(
            id="pedestrian_crossing_v1",
            family="pedestrian_crossing",
            display_name="Pedestrian crossing",
            description=(
                "Straight route passing a pedestrian-sized block offset 0.45 m from "
                "the centerline; surfaces width-driven clearance gradient."
            ),
            mechanism="width_driven",
            route=[(0.0, 5.0), (10.0, 5.0)],
            obstacles=[Polygon([(4.75, 5.45), (5.25, 5.45), (5.25, 5.95), (4.75, 5.95)])],
        ),
        FootprintDiagnosticScenario(
            id="occluded_corner_v1",
            family="occluded_corner",
            display_name="Occluded corner",
            description=(
                "L-shaped route turning away from a wall just east of the corner. An "
                "elongated body oriented along the incoming leg overruns the turn and "
                "pokes the wall; a circular body clears."
            ),
            mechanism="turn_overrun_length_driven",
            route=[(1.0, 1.0), (5.0, 1.0), (5.0, 6.0)],
            obstacles=[Polygon([(5.7, -1.0), (9.0, -1.0), (9.0, 1.4), (5.7, 1.4)])],
        ),
        FootprintDiagnosticScenario(
            id="recovery_after_avoidance_v1",
            family="recovery_after_avoidance",
            display_name="Recovery after avoidance",
            description=(
                "Straight route passing a recovery-zone block offset from the "
                "centerline; width-driven clearance proxy for post-avoidance geometry."
            ),
            mechanism="width_driven",
            route=[(0.0, 5.0), (10.0, 5.0)],
            obstacles=[Polygon([(4.6, 5.6), (5.4, 5.6), (5.4, 6.4), (4.6, 6.4)])],
        ),
        FootprintDiagnosticScenario(
            id="blocked_path_turn_around_v1",
            family="blocked_path_turn_around",
            display_name="Blocked-path turn-around",
            description=(
                "Straight route terminating 1.0 m before a dead-end wall. An "
                "elongated body's forward reach pokes the wall before the centerline "
                "reaches it; a circular body clears."
            ),
            mechanism="forward_reach_length_driven",
            route=[(1.0, 5.0), (8.0, 5.0)],
            obstacles=[Polygon([(9.0, 0.0), (12.0, 0.0), (12.0, 10.0), (9.0, 10.0)])],
        ),
    ]


def _parse_footprint(entry: Mapping[str, Any]) -> FootprintModel:
    kind = entry.get("kind")
    footprint_id = entry.get("id")
    if not isinstance(footprint_id, str) or not _is_lowercase_snake_case(footprint_id):
        raise FootprintOrientationConfigError("footprint id must be a lowercase snake_case string")
    if kind == FOOTPRINT_KIND_CIRCULAR:
        radius = entry.get("radius_m")
        if not _is_positive_finite(radius):
            raise FootprintOrientationConfigError(
                f"footprint {footprint_id!r} radius_m must be a positive finite number"
            )
        return CircularFootprint(id=footprint_id, radius_m=float(radius))
    if kind == FOOTPRINT_KIND_RECTANGULAR:
        length = entry.get("length_m")
        width = entry.get("width_m")
        if not _is_positive_finite(length) or not _is_positive_finite(width):
            raise FootprintOrientationConfigError(
                f"footprint {footprint_id!r} length_m and width_m must be positive finite numbers"
            )
        return RectangularFootprint(id=footprint_id, length_m=float(length), width_m=float(width))
    raise FootprintOrientationConfigError(
        f"footprint {footprint_id!r} kind must be 'circular' or 'rectangular'"
    )


def _validate_top_level_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != FOOTPRINT_ORIENTATION_SCHEMA_VERSION:
        raise FootprintOrientationConfigError(
            f"schema_version must be {FOOTPRINT_ORIENTATION_SCHEMA_VERSION}"
        )
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise FootprintOrientationConfigError("claim_boundary is required")
    _require_boundary_language(claim_boundary)
    _validate_source_literature(payload.get("source_literature"))


def _validate_footprints(raw_footprints: Any) -> list[Mapping[str, Any]]:
    if not isinstance(raw_footprints, list) or not raw_footprints:
        raise FootprintOrientationConfigError("footprints must be a non-empty list")
    kinds: set[str] = set()
    ids: list[str] = []
    for index, entry in enumerate(raw_footprints):
        if not isinstance(entry, Mapping):
            raise FootprintOrientationConfigError(f"footprint {index} must be a mapping")
        if not isinstance(entry.get("id"), str) or not _is_lowercase_snake_case(entry["id"]):
            raise FootprintOrientationConfigError(
                f"footprint {index} id must be lowercase snake_case"
            )
        ids.append(entry["id"])
        kinds.add(entry.get("kind"))
        _validate_footprint_entry(entry)
    duplicate_ids = sorted({footprint_id for footprint_id in ids if ids.count(footprint_id) > 1})
    if duplicate_ids:
        raise FootprintOrientationConfigError(
            f"duplicate footprint ids: {', '.join(duplicate_ids)}"
        )
    if FOOTPRINT_KIND_CIRCULAR not in kinds:
        raise FootprintOrientationConfigError("at least one circular footprint is required")
    if FOOTPRINT_KIND_RECTANGULAR not in kinds:
        raise FootprintOrientationConfigError("at least one rectangular footprint is required")
    return [entry for entry in raw_footprints if isinstance(entry, Mapping)]


def _validate_footprint_entry(entry: Mapping[str, Any]) -> None:
    footprint_id = str(entry["id"])
    kind = entry.get("kind")
    if kind not in (FOOTPRINT_KIND_CIRCULAR, FOOTPRINT_KIND_RECTANGULAR):
        raise FootprintOrientationConfigError(
            f"footprint {footprint_id!r} kind must be 'circular' or 'rectangular'"
        )
    for field_name in ("display_name", "computation_status", "notes"):
        value = entry.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise FootprintOrientationConfigError(
                f"footprint {footprint_id!r} {field_name} must be a non-empty string"
            )
    if kind == FOOTPRINT_KIND_CIRCULAR and not _is_positive_finite(entry.get("radius_m")):
        raise FootprintOrientationConfigError(
            f"footprint {footprint_id!r} radius_m must be a positive finite number"
        )
    if kind == FOOTPRINT_KIND_RECTANGULAR and (
        not _is_positive_finite(entry.get("length_m"))
        or not _is_positive_finite(entry.get("width_m"))
    ):
        raise FootprintOrientationConfigError(
            f"footprint {footprint_id!r} length_m and width_m must be positive finite numbers"
        )


def _validate_scenario_families(raw_families: Any) -> list[Mapping[str, Any]]:
    if not isinstance(raw_families, list) or not raw_families:
        raise FootprintOrientationConfigError("scenario_families must be a non-empty list")
    ids: list[str] = []
    for index, entry in enumerate(raw_families):
        if not isinstance(entry, Mapping):
            raise FootprintOrientationConfigError(f"scenario_family {index} must be a mapping")
        family_id = entry.get("id")
        if not isinstance(family_id, str) or not _is_lowercase_snake_case(family_id):
            raise FootprintOrientationConfigError(
                f"scenario_family {index} id must be lowercase snake_case"
            )
        ids.append(family_id)
        for field_name in ("display_name", "description", "mechanism"):
            value = entry.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise FootprintOrientationConfigError(
                    f"scenario_family {family_id!r} {field_name} must be a non-empty string"
                )
    duplicate_ids = sorted({family_id for family_id in ids if ids.count(family_id) > 1})
    if duplicate_ids:
        raise FootprintOrientationConfigError(
            f"duplicate scenario_family ids: {', '.join(duplicate_ids)}"
        )
    missing_required = sorted(REQUIRED_SCENARIO_FAMILY_IDS.difference(ids))
    if missing_required:
        raise FootprintOrientationConfigError(
            f"missing required scenario_family ids: {', '.join(missing_required)}"
        )
    return [entry for entry in raw_families if isinstance(entry, Mapping)]


def _validate_diagnostic_parameters(raw_params: Any) -> None:
    if not isinstance(raw_params, Mapping):
        raise FootprintOrientationConfigError("diagnostic_parameters must be a mapping")
    sample_step = raw_params.get("sample_step_m")
    if not _is_positive_finite(sample_step):
        raise FootprintOrientationConfigError("diagnostic_parameters.sample_step_m must be > 0")
    max_samples = raw_params.get("max_samples")
    if not isinstance(max_samples, int) or max_samples < 2:
        raise FootprintOrientationConfigError(
            "diagnostic_parameters.max_samples must be an int >= 2"
        )
    pass_threshold = raw_params.get("pass_threshold_m")
    if not _is_nonnegative_finite(pass_threshold):
        raise FootprintOrientationConfigError("diagnostic_parameters.pass_threshold_m must be >= 0")


def _validate_source_literature(source_literature: Any) -> None:
    if not isinstance(source_literature, list) or not source_literature:
        raise FootprintOrientationConfigError("source_literature must be a non-empty list")
    for entry in source_literature:
        if not isinstance(entry, Mapping):
            raise FootprintOrientationConfigError("source_literature entries must be mappings")
        if entry.get("role") != "motivation_only":
            raise FootprintOrientationConfigError(
                "source_literature entries must use role motivation_only"
            )
        if not entry.get("url"):
            raise FootprintOrientationConfigError("source_literature entries require url")


def _require_boundary_language(claim_boundary: str) -> None:
    normalized = claim_boundary.casefold()
    required_phrases = ("diagnostic", "not a full se(2) planner", "not calibrated")
    missing = [phrase for phrase in required_phrases if phrase not in normalized]
    if missing:
        raise FootprintOrientationConfigError(
            "claim_boundary must explicitly state diagnostic, not-a-full-SE(2)-planner, "
            "and not-calibrated status"
        )


def _result_to_row(result: FootprintClearanceResult) -> dict[str, Any]:
    return {
        "footprint_id": result.footprint_id,
        "kind": result.kind,
        "centerline_clearance_m": _round_or_none(result.centerline_clearance_m),
        "footprint_aware_clearance_m": _round_or_none(result.footprint_aware_clearance_m),
        "status": result.status,
        "collision": result.collision,
        "sample_count": result.sample_count,
        "method": result.method,
    }


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _oriented_rectangle(
    cx: float,
    cy: float,
    length: float,
    width: float,
    heading_rad: float,
) -> Polygon:
    """Build an oriented rectangle centered at (cx, cy).

    The rectangle's long axis (``length``) is aligned with ``heading_rad`` and
    its short axis (``width``) is perpendicular.

    Returns:
        A Shapely ``Polygon`` for the oriented footprint centered at (cx, cy).
    """

    half_length = length / 2.0
    half_width = width / 2.0
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    local_corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    world_corners = [
        (cx + x * cos_h - y * sin_h, cy + x * sin_h + y * cos_h) for x, y in local_corners
    ]
    return Polygon(world_corners)


def _route_fallback_heading(line: LineString) -> float:
    """Return a heading (radians) from the first non-degenerate route segment.

    Used as a conservative fallback when a local tangent is degenerate at a
    sample point (e.g. duplicate consecutive waypoints) so the oriented
    footprint never collapses to a zero-size point. The caller guarantees the
    route has positive length, so at least one non-degenerate segment exists;
    ``0.0`` is only returned in the theoretically-unreachable all-degenerate
    case, keeping a full-size (axis-aligned) footprint rather than a point.
    """

    coords = list(line.coords)
    for (x0, y0), (x1, y1) in itertools.pairwise(coords):
        dx = x1 - x0
        dy = y1 - y0
        if dx != 0.0 or dy != 0.0:
            return float(math.atan2(dy, dx))
    return 0.0


def _route_tangent(line: LineString, distance: float, length: float, eps: float) -> float | None:
    """Return the local tangent heading (radians) at a route distance.

    Returns ``None`` when the tangent is degenerate (e.g. a single-point route).
    """

    d_before = max(0.0, distance - eps)
    d_after = min(length, distance + eps)
    if d_after <= d_before:
        return None
    before = line.interpolate(d_before)
    after = line.interpolate(d_after)
    dx = after.x - before.x
    dy = after.y - before.y
    if dx == 0.0 and dy == 0.0:
        return None
    return float(math.atan2(dy, dx))


def _is_positive_finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) > 0.0


def _is_nonnegative_finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) >= 0.0


def _is_lowercase_snake_case(value: str) -> bool:
    if not value:
        return False
    return value[0].islower() and all(
        char.islower() or char.isdigit() or char == "_" for char in value
    )
