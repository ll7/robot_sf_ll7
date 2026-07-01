"""Tests for issue #3966 canonical benchmark metric layers."""

from __future__ import annotations

import json

import pytest

from robot_sf.benchmark.cli import cli_main as benchmark_cli_main
from robot_sf.benchmark.constraints_first_scoring import build_constraints_first_report
from robot_sf.benchmark.metric_layers import (
    CANONICAL_METRICS,
    LAYER_ORDER,
    METRIC_LAYER_SCHEMA_VERSION,
    build_metric_layer_summary,
)

EXPECTED_METRICS = {
    "collision_rate",
    "human_collision_rate",
    "near_miss_rate",
    "min_time_to_collision",
    "minimum_pedestrian_distance",
    "timeout_rate",
    "stopped_time_ratio",
    "failure_to_progress_rate",
    "deadlock_duration",
    "personal_space_violation_rate",
    "group_intrusion_episode_rate",
    "group_intrusion_time_ratio",
    "pedestrian_path_deviation",
    "time_to_goal",
    "path_length",
    "spl",
    "acceleration",
    "jerk",
    "angular_velocity",
    "intervention_required",
    "tele_assistance_required",
    "tele_driving_required",
}


def _minimal_episode(*, algo: str = "planner_a") -> dict[str, object]:
    """Build a minimal episode record with current schema-level outcome fields."""

    return {
        "algo": algo,
        "scenario_params": {"algo": algo},
        "metrics": {},
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
        "termination_reason": "success",
    }


def test_layer_order_matches_contract() -> None:
    """The canonical layer order is stable and constraints-first."""

    assert LAYER_ORDER == (
        "safety_gate",
        "liveness",
        "social_compliance",
        "efficiency",
        "comfort",
        "operational",
    )


def test_all_issue_metrics_are_registered_once() -> None:
    """Every issue-listed metric has exactly one canonical registry entry."""

    assert set(CANONICAL_METRICS) == EXPECTED_METRICS
    assert len(CANONICAL_METRICS) == len(EXPECTED_METRICS)


def test_missing_metric_is_unavailable_not_zero() -> None:
    """Unavailable metrics must not be silently converted to zero."""

    summary = build_metric_layer_summary([_minimal_episode()])

    metric = summary["layers"]["safety_gate"]["metrics"]["min_time_to_collision"]

    assert metric["status"] == "unavailable"
    assert metric["value"] is None
    assert metric["support_count"] == 0
    assert metric["unavailable_reason"] == "metric_not_present_in_episode_records"


def test_collision_rate_can_be_derived_from_existing_episode_records() -> None:
    """Collision rate derives from current collision count/outcome fields."""

    records = [
        {
            **_minimal_episode(algo="planner_a"),
            "metrics": {"collisions": 0},
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "timeout_event": False,
            },
        },
        {
            **_minimal_episode(algo="planner_a"),
            "metrics": {"collisions": 1},
            "outcome": {
                "route_complete": False,
                "collision_event": True,
                "timeout_event": False,
            },
        },
    ]

    summary = build_metric_layer_summary(records)
    metric = summary["layers"]["safety_gate"]["metrics"]["collision_rate"]

    assert metric["status"] == "available"
    assert metric["value"] == pytest.approx(0.5)
    assert metric["support_count"] == 2
    assert metric["selected_source_keys"] == ["metrics.collisions"]


def test_timeout_and_failure_to_progress_derivations_are_conservative() -> None:
    """Liveness derivations use only explicit route/outcome fields."""

    records = [
        _minimal_episode(algo="planner_a"),
        {
            **_minimal_episode(algo="planner_a"),
            "outcome": {
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            "termination_reason": "max_steps",
        },
        {
            **_minimal_episode(algo="planner_a"),
            "outcome": {
                "route_complete": False,
                "collision_event": False,
                "timeout_event": False,
            },
            "termination_reason": "stalled_without_progress",
        },
    ]

    summary = build_metric_layer_summary(records)
    liveness = summary["layers"]["liveness"]["metrics"]

    assert liveness["timeout_rate"]["value"] == pytest.approx(1 / 3)
    assert liveness["failure_to_progress_rate"]["value"] == pytest.approx(1 / 3)


def test_proxy_social_metrics_are_marked_as_simulation_proxy() -> None:
    """Existing social proxy fields stay explicitly labeled as proxies."""

    record = {
        **_minimal_episode(),
        "metrics": {
            "social_proxemic_intrusion_frac": 0.25,
            "pedestrian_path_deviation_proxy_m": 1.5,
        },
    }

    summary = build_metric_layer_summary([record])
    social = summary["layers"]["social_compliance"]["metrics"]

    assert social["personal_space_violation_rate"]["status"] == "available"
    assert social["personal_space_violation_rate"]["source_kind"] == "simulation_proxy"
    assert social["pedestrian_path_deviation"]["value"] == pytest.approx(1.5)
    assert social["pedestrian_path_deviation"]["source_kind"] == "simulation_proxy"


def test_metric_layer_summary_includes_grouped_views() -> None:
    """The adapter exposes per-planner layer summaries without dropping overall layers."""

    records = [
        {**_minimal_episode(algo="planner_a"), "metrics": {"collisions": 0}},
        {**_minimal_episode(algo="planner_b"), "metrics": {"collisions": 1}},
    ]

    summary = build_metric_layer_summary(records)

    assert summary["n_episodes"] == 2
    assert set(summary["groups"]) == {"planner_a", "planner_b"}
    assert summary["groups"]["planner_a"]["n_episodes"] == 1
    assert summary["layers"]["safety_gate"]["metrics"]["collision_rate"]["value"] == pytest.approx(
        0.5
    )


def test_non_finite_metric_values_are_ignored_with_alias_fallback() -> None:
    """NaN/Inf values are unavailable and do not mask later valid aliases."""

    record = {
        **_minimal_episode(),
        "metrics": {
            "collision_rate": float("nan"),
            "collisions": 1,
            "time_to_goal": float("inf"),
        },
    }

    summary = build_metric_layer_summary([record])
    collision = summary["layers"]["safety_gate"]["metrics"]["collision_rate"]
    time_to_goal = summary["layers"]["efficiency"]["metrics"]["time_to_goal"]

    assert collision["status"] == "available"
    assert collision["value"] == pytest.approx(1.0)
    assert collision["selected_source_keys"] == ["metrics.collisions"]
    assert time_to_goal["status"] == "unavailable"
    assert time_to_goal["support_count"] == 0


def test_constraints_first_report_advertises_metric_layer_contract() -> None:
    """Constraints-first reports name the shared layer contract without reranking changes."""

    report = build_constraints_first_report(
        {"planner_a": [{"collisions": 0, "comfort": 0.8, "efficiency": 0.7, "safe_success": True}]}
    )

    assert report["metric_layer_schema_version"] == METRIC_LAYER_SCHEMA_VERSION
    assert report["metric_layer_order"] == list(LAYER_ORDER)


def test_metric_layers_cli_writes_summary(tmp_path) -> None:
    """The optional benchmark CLI command writes a metric-layer JSON summary."""

    episodes = tmp_path / "episodes.jsonl"
    output = tmp_path / "metric_layers.json"
    episodes.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {**_minimal_episode(algo="planner_a"), "metrics": {"collisions": 0}},
                {**_minimal_episode(algo="planner_a"), "metrics": {"collisions": 1}},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        benchmark_cli_main(
            [
                "metric-layers",
                "--episodes",
                str(episodes),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == METRIC_LAYER_SCHEMA_VERSION
    assert payload["layers"]["safety_gate"]["metrics"]["collision_rate"]["value"] == pytest.approx(
        0.5
    )


def test_metric_layers_cli_filters_non_finite_values_before_json_output(tmp_path) -> None:
    """Metric-layer CLI output remains strict JSON when inputs contain NaN/Inf."""

    episodes = tmp_path / "episodes.jsonl"
    output = tmp_path / "metric_layers.json"
    episodes.write_text(
        json.dumps(
            {
                **_minimal_episode(),
                "metrics": {
                    "collision_rate": float("nan"),
                    "collisions": 0,
                    "time_to_goal": float("inf"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        benchmark_cli_main(
            [
                "metric-layers",
                "--episodes",
                str(episodes),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    output_text = output.read_text(encoding="utf-8")
    assert "NaN" not in output_text
    assert "Infinity" not in output_text
    payload = json.loads(output_text)
    assert payload["layers"]["safety_gate"]["metrics"]["collision_rate"]["value"] == pytest.approx(
        0.0
    )
    assert payload["layers"]["efficiency"]["metrics"]["time_to_goal"]["status"] == "unavailable"
