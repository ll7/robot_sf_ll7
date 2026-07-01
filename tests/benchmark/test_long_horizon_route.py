"""Tests for long-horizon route composition and per-distance metrics."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.long_horizon_route import (
    LongHorizonRouteError,
    LongHorizonRouteSegment,
    aggregate_distance_normalized_route_metrics,
    build_long_horizon_route,
    run_long_horizon_route,
)


def test_build_long_horizon_route_repeats_and_truncates_segments() -> None:
    """Route builder should compose existing scenario segments to a target length."""

    route = build_long_horizon_route(
        "unit-route",
        (
            LongHorizonRouteSegment("clear_sidewalk", 40.0),
            LongHorizonRouteSegment("static_obstacle", 30.0),
            LongHorizonRouteSegment("pedestrian_crossing", 25.0),
        ),
        target_length_m=120.0,
    )

    assert route.length_m == pytest.approx(120.0)
    assert [segment.scenario_id for segment in route.segments] == [
        "clear_sidewalk",
        "static_obstacle",
        "pedestrian_crossing",
        "clear_sidewalk",
    ]
    assert [segment.length_m for segment in route.segments] == pytest.approx(
        [40.0, 30.0, 25.0, 25.0],
        abs=1e-9,
    )
    assert route.to_dict()["segments"][0]["scenario_id"] == "clear_sidewalk"


def test_build_long_horizon_route_without_target_uses_declared_lengths() -> None:
    """Route builder should preserve declared segment repetitions when no target is set."""

    route = build_long_horizon_route(
        "declared-route",
        (
            LongHorizonRouteSegment("clear_sidewalk", 10.0, repetitions=2),
            LongHorizonRouteSegment("blind_corner", 15.0),
        ),
    )

    assert route.length_m == pytest.approx(35.0)
    assert route.to_dict() == {
        "route_id": "declared-route",
        "length_m": 35.0,
        "segments": [
            {
                "scenario_id": "clear_sidewalk",
                "length_m": 10.0,
                "repetitions": 2,
                "parameters": {},
            },
            {
                "scenario_id": "blind_corner",
                "length_m": 15.0,
                "repetitions": 1,
                "parameters": {},
            },
        ],
    }


@pytest.mark.parametrize(
    ("segment_kwargs", "message"),
    [
        ({"scenario_id": "", "length_m": 10.0}, "scenario_id"),
        ({"scenario_id": "clear_sidewalk", "length_m": 0.0}, "length_m"),
        ({"scenario_id": "clear_sidewalk", "length_m": 10.0, "repetitions": 0}, "repetitions"),
    ],
)
def test_route_segment_validation_fails_closed(
    segment_kwargs: dict[str, object],
    message: str,
) -> None:
    """Invalid segment definitions should fail before benchmark use."""

    with pytest.raises(LongHorizonRouteError, match=message):
        LongHorizonRouteSegment(**segment_kwargs)  # type: ignore[arg-type]


def test_distance_normalized_route_metrics_from_synthetic_records() -> None:
    """Aggregator should emit the six requested per-100m/per-km route metrics."""

    records = [
        {
            "episode_id": "route-a-segment-1",
            "metrics": {
                "distance_m": 120.0,
                "route_length_m": 300.0,
                "failures": 1,
                "collisions": 1,
                "near_misses": 3,
                "interventions": 1,
                "resets": 0,
            },
        },
        {
            "episode_id": "route-a-segment-2",
            "metrics": {
                "distance_m": 80.0,
                "route_length_m": 300.0,
                "failures": 0,
                "collisions": 0,
                "near_misses": 1,
                "interventions": 0,
                "resets": 1,
            },
        },
    ]

    metrics = aggregate_distance_normalized_route_metrics(records)

    assert metrics == pytest.approx(
        {
            "failures_per_100m": 0.5,
            "collisions_per_100m": 0.5,
            "near_misses_per_100m": 2.0,
            "interventions_per_km": 5.0,
            "route_completion": 2 / 3,
            "resets_per_km": 5.0,
        }
    )


def test_distance_normalized_route_metrics_accept_existing_runner_path_length() -> None:
    """Aggregator should reuse the benchmark runner's existing path-length metric."""

    metrics = aggregate_distance_normalized_route_metrics(
        [{"metrics": {"socnavbench_path_length": 12.5, "route_length_m": 50.0}}]
    )

    assert metrics["route_completion"] == pytest.approx(0.25)


def test_run_long_horizon_route_executes_segments_headlessly_and_aggregates() -> None:
    """Route runner should compose existing segment episodes without redefining events."""

    route = build_long_horizon_route(
        "short-route",
        (
            LongHorizonRouteSegment(
                "clear_sidewalk",
                10.0,
                parameters={"density": "low", "flow": "uni", "obstacle": "open"},
            ),
            LongHorizonRouteSegment(
                "static_obstacle",
                15.0,
                parameters={"density": "low", "flow": "uni", "obstacle": "bottleneck"},
            ),
        ),
    )
    calls: list[tuple[dict[str, object], int, dict[str, object]]] = []

    def fake_run_episode(
        scenario_params: dict[str, object],
        seed: int,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append((scenario_params, seed, kwargs))
        return {
            "scenario_id": scenario_params["id"],
            "seed": seed,
            "metrics": {
                "socnavbench_path_length": 5.0 * (len(calls)),
                "collisions": 1 if len(calls) == 2 else 0,
                "near_misses": len(calls),
            },
        }

    result = run_long_horizon_route(
        route,
        seed=7,
        horizon=3,
        dt=0.2,
        record_forces=False,
        run_episode_fn=fake_run_episode,
    )

    assert [call[0]["id"] for call in calls] == ["clear_sidewalk", "static_obstacle"]
    assert [call[1] for call in calls] == [7, 8]
    assert all(call[2]["video_enabled"] is False for call in calls)
    assert all(call[2]["record_forces"] is False for call in calls)
    assert result.records[0]["long_horizon_route"] == {
        "route_id": "short-route",
        "segment_index": 0,
        "segment_id": "clear_sidewalk",
        "segment_length_m": 10.0,
    }
    assert result.records[1]["metrics"]["planned_distance_m"] == 15.0
    assert result.metrics == pytest.approx(
        {
            "failures_per_100m": 0.0,
            "collisions_per_100m": 100.0 / 15.0,
            "near_misses_per_100m": 20.0,
            "interventions_per_km": 0.0,
            "route_completion": 15.0 / 25.0,
            "resets_per_km": 0.0,
        }
    )


def test_distance_normalized_route_metrics_fail_closed_without_distance() -> None:
    """Distance-normalized metrics require explicit distance provenance."""

    with pytest.raises(LongHorizonRouteError, match="missing distance_m"):
        aggregate_distance_normalized_route_metrics([{"metrics": {"collisions": 1}}])


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "at least one record"),
        ([{"metrics": {"distance_m": 0.0}}], "total traversed distance_m"),
        ([{"metrics": {"distance_m": -1.0}}], "distance_m must not be negative"),
        ([{"metrics": {"distance_m": 10.0, "collisions": -1}}], "collisions count"),
        ([{"metrics": {"distance_m": "bad"}}], "distance_m must be numeric"),
        ([{"metrics": {"distance_m": 10.0, "route_length_m": 0.0}}], "route_length_m"),
    ],
)
def test_distance_normalized_route_metrics_fail_closed_invalid_inputs(
    records: list[dict[str, object]],
    message: str,
) -> None:
    """Invalid record inputs should fail closed instead of emitting misleading rates."""

    with pytest.raises(LongHorizonRouteError, match=message):
        aggregate_distance_normalized_route_metrics(records)


def test_distance_normalized_route_metrics_support_planned_distance_alias() -> None:
    """Planned segment distances should be summed when no route total is present."""

    metrics = aggregate_distance_normalized_route_metrics(
        [
            {"metrics": {"distance_m": 20.0, "planned_distance_m": 40.0}},
            {"metrics": {"distance_m": 30.0, "planned_distance_m": 60.0}},
        ]
    )

    assert metrics["route_completion"] == pytest.approx(0.5)
