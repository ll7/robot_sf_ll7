"""Contract tests for deterministic pedestrian-response observations (Issue #8683)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from robot_sf.benchmark.pedestrian_response_observability import (
    PEDESTRIAN_RESPONSE_SCHEMA_VERSION,
    PedestrianResponseObservation,
    RouteReference,
    build_pedestrian_response_observation,
)
from robot_sf.benchmark.route_choice_observability import classify_route_side
from robot_sf.nav.biased_route_generator import (
    BiasedRouteResult,
    CanonicalFixtureTopology,
    build_corridor_fixture,
    build_doorway_fixture,
    generate_corridor_homotopy_routes,
    generate_doorway_homotopy_routes,
)

RouteSetBuilder = Callable[[CanonicalFixtureTopology], dict[str, BiasedRouteResult]]


@pytest.mark.parametrize(
    ("fixture_builder", "routes_builder", "response_present"),
    [
        (build_corridor_fixture, generate_corridor_homotopy_routes, True),
        (build_doorway_fixture, generate_doorway_homotopy_routes, False),
    ],
)
def test_structured_indoor_fixture_replays_identically(
    fixture_builder: Callable[[], CanonicalFixtureTopology],
    routes_builder: RouteSetBuilder,
    response_present: bool,
) -> None:
    """Identical corridor/doorway inputs produce identical response records."""
    fixture = fixture_builder()
    routes = routes_builder(fixture)
    first = build_pedestrian_response_observation(
        encounter_id=fixture.name,
        offered_route=routes["left"].side_report,
        taken_route=routes["right"].side_report,
        minimum_passing_clearance_m=0.62,
        response_present=response_present,
    )

    replay_fixture = fixture_builder()
    replay_routes = routes_builder(replay_fixture)
    replay = build_pedestrian_response_observation(
        encounter_id=replay_fixture.name,
        offered_route=replay_routes["left"].side_report,
        taken_route=replay_routes["right"].side_report,
        minimum_passing_clearance_m=0.62,
        response_present=response_present,
    )

    assert first == replay
    assert first.status == "available"
    assert first.offered_side == "left"
    assert first.taken_side == "right"
    assert first.minimum_passing_clearance_m == pytest.approx(0.62)
    assert first.route_reference == RouteReference.from_report(routes["left"].side_report)
    assert first.as_dict()["route_reference"] == first.route_reference.as_dict()
    assert first.missing_fields == ()
    assert first.unavailable_fields == ()
    assert json.dumps(first.as_dict(), sort_keys=True) == json.dumps(
        replay.as_dict(), sort_keys=True
    )
    assert first.as_dict()["schema_version"] == PEDESTRIAN_RESPONSE_SCHEMA_VERSION


def test_missing_fields_are_distinct_from_an_observed_false_response() -> None:
    """False means observed absence; None is explicitly missing."""
    routes = generate_corridor_homotopy_routes(build_corridor_fixture(), num_points=24)
    available = build_pedestrian_response_observation(
        encounter_id="false-response",
        offered_route=routes["left"].side_report,
        taken_route=routes["left"].side_report,
        minimum_passing_clearance_m=0.8,
        response_present=False,
    )
    unavailable = build_pedestrian_response_observation(
        encounter_id="missing-response",
        taken_route=routes["left"].side_report,
    )

    assert available.status == "available"
    assert available.response_present is False
    assert available.missing_fields == ()
    assert unavailable.status == "not_available"
    assert unavailable.response_present is None
    assert unavailable.missing_fields == (
        "minimum_passing_clearance_m",
        "offered_side",
        "response_present",
    )
    assert unavailable.unavailable_fields == ()
    assert unavailable.unavailable_reason == "missing_fields"


@pytest.mark.parametrize("invalid_clearance", [True, 10**1000])
def test_builder_marks_invalid_clearance_unavailable(invalid_clearance: object) -> None:
    """Builder normalization rejects booleans and overflowing numeric values."""
    routes = generate_corridor_homotopy_routes(build_corridor_fixture(), num_points=24)
    record = build_pedestrian_response_observation(
        encounter_id="invalid-clearance",
        offered_route=routes["left"].side_report,
        taken_route=routes["right"].side_report,
        minimum_passing_clearance_m=invalid_clearance,  # type: ignore[arg-type]
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.missing_fields == ()
    assert record.unavailable_fields == ("minimum_passing_clearance_m",)
    assert record.unavailable_reason == "minimum_passing_clearance_m:invalid_value"


def test_mismatched_route_references_fail_closed() -> None:
    """Reports from incompatible route frames cannot produce side evidence."""
    routes = generate_corridor_homotopy_routes(build_corridor_fixture(), num_points=24)
    incompatible_taken_route = replace(
        routes["right"].side_report,
        coordinate_frame="ego_xy",
    )
    record = build_pedestrian_response_observation(
        encounter_id="mismatched-reference",
        offered_route=routes["left"].side_report,
        taken_route=incompatible_taken_route,
        minimum_passing_clearance_m=0.8,
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.route_reference is None
    assert record.offered_side == "unavailable"
    assert record.taken_side == "unavailable"
    assert record.missing_fields == ()
    assert record.unavailable_fields == (
        "offered_side",
        "route_reference",
        "taken_side",
    )
    assert record.unavailable_reason == "route_reference:mismatch"
    assert record.as_dict()["route_reference"] is None


def test_direct_record_requires_route_reference_for_available_status() -> None:
    """Direct records without provenance are explicitly not available."""
    record = PedestrianResponseObservation(
        encounter_id="unreferenced",
        minimum_passing_clearance_m=0.5,
        offered_side="left",
        taken_side="right",
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.missing_fields == ("route_reference",)
    assert record.unavailable_fields == ()
    assert record.unavailable_reason == "missing_fields"


def test_unavailable_route_side_preserves_route_observability_reason() -> None:
    """An unavailable predecessor report remains unavailable, never a side guess."""
    unavailable_route = classify_route_side([], start=(0.0, 0.0), goal=(2.0, 0.0))
    record = build_pedestrian_response_observation(
        encounter_id="unavailable-offered-route",
        offered_route=unavailable_route,
        taken_route=unavailable_route,
        minimum_passing_clearance_m=0.8,
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.offered_side == "unavailable"
    assert record.taken_side == "unavailable"
    assert record.missing_fields == ()
    assert record.unavailable_fields == ("offered_side", "taken_side")
    assert record.unavailable_reason == "offered_side:empty_path;taken_side:empty_path"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offered_side": "diagonal"},
        {"minimum_passing_clearance_m": float("nan")},
        {"minimum_passing_clearance_m": 10**1000},
        {"response_present": 1},
    ],
)
def test_typed_record_rejects_invalid_direct_values(kwargs: dict[str, object]) -> None:
    """Direct construction cannot introduce invalid typed observation values."""
    values: dict[str, object] = {
        "encounter_id": "invalid",
        "minimum_passing_clearance_m": 0.5,
        "offered_side": "left",
        "taken_side": "right",
        "response_present": True,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        PedestrianResponseObservation(**values)


def test_direct_record_rejects_missing_and_unavailable_side_overlap() -> None:
    """A side normalized as unavailable cannot also be declared missing."""
    with pytest.raises(ValueError, match="both missing and unavailable"):
        PedestrianResponseObservation(
            encounter_id="contradictory-side-state",
            offered_side="unavailable",
            missing_fields=("offered_side",),
        )


def test_invalid_unavailable_route_reference_fails_closed() -> None:
    """An unavailable predecessor with invalid metadata remains explicit unavailable data."""
    unavailable_route = classify_route_side([], start=(0.0, 0.0), goal=(2.0, 0.0))
    invalid_route = replace(unavailable_route, coordinate_frame="")

    record = build_pedestrian_response_observation(
        encounter_id="invalid-unavailable-reference",
        offered_route=invalid_route,
        taken_route=invalid_route,
        minimum_passing_clearance_m=0.8,
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.route_reference is None
    assert record.offered_side == "unavailable"
    assert record.taken_side == "unavailable"
    assert record.unavailable_fields == (
        "offered_side",
        "route_reference",
        "taken_side",
    )
    assert "route_reference:invalid_reference" in (record.unavailable_reason or "")


def test_mixed_valid_and_unavailable_route_reference_hides_both_sides() -> None:
    """A valid side cannot remain evidence when its paired route reference is invalid."""
    routes = generate_corridor_homotopy_routes(build_corridor_fixture(), num_points=24)
    unavailable_route = classify_route_side([], start=(0.0, 0.0), goal=(2.0, 0.0))
    invalid_unavailable_route = replace(unavailable_route, coordinate_frame="")

    record = build_pedestrian_response_observation(
        encounter_id="mixed-route-reference",
        offered_route=routes["left"].side_report,
        taken_route=invalid_unavailable_route,
        minimum_passing_clearance_m=0.8,
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.route_reference is None
    assert record.offered_side == "unavailable"
    assert record.taken_side == "unavailable"
    assert record.unavailable_fields == (
        "offered_side",
        "route_reference",
        "taken_side",
    )
    assert "route_reference:invalid_reference" in (record.unavailable_reason or "")
