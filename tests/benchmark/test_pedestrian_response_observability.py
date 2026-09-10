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
    """False means observed absence; None is missing by default."""
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


def test_explicitly_unavailable_response_is_not_missing() -> None:
    """An explicit unavailable response flag stays separate from an omitted flag."""
    routes = generate_corridor_homotopy_routes(build_corridor_fixture(), num_points=24)
    record = build_pedestrian_response_observation(
        encounter_id="unavailable-response",
        offered_route=routes["left"].side_report,
        taken_route=routes["right"].side_report,
        minimum_passing_clearance_m=0.8,
        response_present=None,
        unavailable_fields=("response_present",),
    )

    assert record.status == "not_available"
    assert record.response_present is None
    assert record.missing_fields == ()
    assert record.unavailable_fields == ("response_present",)
    assert record.unavailable_reason == "unavailable_fields"


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
    """Direct records hide side labels when route provenance is absent."""
    record = PedestrianResponseObservation(
        encounter_id="unreferenced",
        minimum_passing_clearance_m=0.5,
        offered_side="left",
        taken_side="right",
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.offered_side == "unavailable"
    assert record.taken_side == "unavailable"
    assert record.missing_fields == ("route_reference",)
    assert record.unavailable_fields == ("offered_side", "taken_side")
    assert record.unavailable_reason == "missing_and_unavailable_fields"


@pytest.mark.parametrize("goal_delta", [0.0, 0.025, 0.05])
def test_route_reference_rejects_zero_or_near_zero_start_goal(goal_delta: float) -> None:
    """The route contract's tolerance gate rejects degenerate reference axes."""
    routes = generate_corridor_homotopy_routes(build_corridor_fixture(), num_points=24)
    reference = RouteReference.from_report(routes["left"].side_report)

    with pytest.raises(ValueError, match="non-degenerate reference axis"):
        replace(
            reference,
            goal=(reference.start[0] + goal_delta, reference.start[1]),
        )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"tolerance_m": -1.0}, "invalid_tolerance"),
        ({"neutral_band_m": float("nan")}, "invalid_neutral_band"),
        ({"progress_interval": (0.9, 0.1)}, "invalid_progress_interval"),
        ({"start": (0.0, 0.0), "goal": (0.0, 0.0)}, "degenerate_reference"),
        ({"start": (0.0, 0.0), "goal": (0.025, 0.0)}, "degenerate_reference"),
    ],
)
def test_invalid_upstream_reference_reason_is_propagated(
    kwargs: dict[str, object], reason: str
) -> None:
    """Invalid upstream reference reasons cannot become fallback provenance."""
    reference_kwargs: dict[str, object] = {
        "start": (0.0, 0.0),
        "goal": (2.0, 0.0),
    }
    reference_kwargs.update(kwargs)
    report = classify_route_side(
        [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
        **reference_kwargs,
    )  # type: ignore[arg-type]
    assert report.side == "unavailable"
    assert report.reason == reason

    record = build_pedestrian_response_observation(
        encounter_id=f"invalid-reference-{reason}",
        offered_route=report,
        taken_route=report,
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
    assert f"offered_side:{reason}" in (record.unavailable_reason or "")
    assert f"route_reference:{reason}" in (record.unavailable_reason or "")


def test_builder_hides_side_from_degenerate_reference_without_upstream_reason() -> None:
    """A forged valid-looking side cannot bypass the canonical axis gate."""
    routes = generate_corridor_homotopy_routes(build_corridor_fixture(), num_points=24)
    malformed_route = replace(
        routes["left"].side_report,
        goal=(0.025, 0.0),
        reason=None,
    )

    record = build_pedestrian_response_observation(
        encounter_id="unreasoned-degenerate-reference",
        offered_route=malformed_route,
        taken_route=malformed_route,
        minimum_passing_clearance_m=0.8,
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.route_reference is None
    assert record.offered_side == "unavailable"
    assert record.taken_side == "unavailable"
    assert "route_reference:invalid_reference" in (record.unavailable_reason or "")


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
    "failure_reason",
    [
        "empty_path",
        "single_point",
        "zero_length",
        "non_finite",
        "insufficient_progress",
        "unknown",
        "upstream_failure",
    ],
)
def test_non_none_upstream_reason_hides_non_unavailable_side(failure_reason: str) -> None:
    """A report cannot provide side evidence when it carries a failure reason."""
    routes = generate_corridor_homotopy_routes(build_corridor_fixture(), num_points=24)
    malformed_route = replace(routes["left"].side_report, reason=failure_reason)

    record = build_pedestrian_response_observation(
        encounter_id=f"failed-route-{failure_reason}",
        offered_route=malformed_route,
        taken_route=routes["right"].side_report,
        minimum_passing_clearance_m=0.8,
        response_present=True,
    )

    assert record.status == "not_available"
    assert record.offered_side == "unavailable"
    assert record.taken_side == "right"
    assert record.missing_fields == ()
    assert record.unavailable_fields == ("offered_side",)
    assert record.unavailable_reason == f"offered_side:{failure_reason}"


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
