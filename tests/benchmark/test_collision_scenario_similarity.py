"""Tests for collision scenario similarity analysis reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.collision_scenario_similarity import (
    SCHEMA_VERSION,
    build_collision_scenario_similarity_report,
    describe_collision_scenarios,
)

TRAJECTORY_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "benchmark"
    / "collision_scenario_similarity_trajectory.jsonl"
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _records() -> list[dict]:
    return [
        {
            "episode_id": "alpha-1",
            "scenario_id": "crossing-alpha",
            "seed": 1,
            "map_name": "atrium",
            "termination_reason": "collision",
            "scenario_params": {"algo": "orca", "family": "crossing", "num_pedestrians": 6},
            "metrics": {
                "collisions": 1,
                "near_misses": 1,
                "min_separation": 0.04,
                "time_to_conflict": 1.2,
                "comfort_exposure": 0.3,
                "time_to_goal": 18.0,
            },
        },
        {
            "episode_id": "alpha-2",
            "scenario_id": "crossing-alpha",
            "seed": 2,
            "map_name": "atrium",
            "termination_reason": "collision",
            "scenario_params": {"algo": "orca", "family": "crossing", "num_pedestrians": 6},
            "metrics": {
                "collisions": 1,
                "near_misses": 1,
                "min_separation": 0.05,
                "time_to_conflict": 1.3,
                "comfort_exposure": 0.31,
                "time_to_goal": 18.5,
            },
        },
        {
            "episode_id": "beta-1",
            "scenario_id": "doorway-beta",
            "seed": 3,
            "map_name": "doorway",
            "termination_reason": "near_miss",
            "scenario_params": {"algo": "dwa", "family": "doorway", "num_pedestrians": 12},
            "metrics": {
                "collisions": 0,
                "near_misses": 2,
                "min_separation": 0.18,
                "time_to_conflict": 4.0,
                "comfort_exposure": 0.25,
                "time_to_goal": 35.0,
            },
        },
        {
            "episode_id": "safe-1",
            "scenario_id": "easy",
            "seed": 4,
            "map_name": "atrium",
            "termination_reason": "success",
            "scenario_params": {"algo": "orca", "family": "crossing", "num_pedestrians": 3},
            "metrics": {
                "collisions": 0,
                "near_misses": 0,
                "min_separation": 1.2,
                "time_to_goal": 12.0,
            },
        },
    ]


def test_describe_collision_scenarios_selects_only_unsafe_records() -> None:
    """Descriptor extraction selects only collision, near-miss, or discomfort cases."""
    descriptors = describe_collision_scenarios(_records())

    assert [descriptor.record_id for descriptor in descriptors] == ["alpha-1", "alpha-2", "beta-1"]
    assert descriptors[0].numeric["min_separation"] == 0.04
    assert descriptors[0].categorical["scenario_family"] == "crossing"
    assert descriptors[2].event["near_misses"] == 2


def test_similarity_report_groups_nearby_collision_cases(tmp_path: Path) -> None:
    """Report groups the two similar crossing collisions before the doorway case."""
    episodes = tmp_path / "episodes.jsonl"
    _write_jsonl(episodes, _records())

    report = build_collision_scenario_similarity_report(
        episodes,
        nearest_k=1,
        group_threshold=0.35,
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["selection"]["selected_count"] == 3
    assert report["nearest_neighbors"][0]["record_id"] == "alpha-1"
    assert report["nearest_neighbors"][0]["neighbors"][0]["record_id"] == "alpha-2"
    alpha_group = next(group for group in report["groups"] if "alpha-1" in group["record_ids"])
    assert alpha_group["record_ids"] == ["alpha-1", "alpha-2"]
    assert alpha_group["representative_record_id"] in {"alpha-1", "alpha-2"}
    assert report["validation"]["external_labels"]["status"] == "unavailable"
    assert any("analysis aid" in limitation for limitation in report["limitations"])


def test_similarity_report_summarizes_label_and_trajectory_validation(tmp_path: Path) -> None:
    """Selected unsafe rows report available external labels and trajectory fields."""
    episodes = tmp_path / "episodes.jsonl"
    records = [
        {
            "episode_id": "labeled-near-miss",
            "scenario_id": "crossing",
            "seed": 11,
            "termination_reason": "max_steps",
            "metrics": {"collisions": 0, "near_misses": 1, "min_separation": 0.3},
            "labels": {"near_miss": True, "collision": False},
            "trajectory_features": {"min_rollout_clearance_m": 0.3},
        },
        {
            "episode_id": "labeled-safe",
            "scenario_id": "open",
            "seed": 12,
            "termination_reason": "success",
            "metrics": {"collisions": 0, "near_misses": 0, "min_separation": 2.0},
            "labels": {"near_miss": False, "collision": False},
            "trajectory": {"robot_states": [{"position": [0.0, 0.0]}]},
        },
    ]
    _write_jsonl(episodes, records)

    report = build_collision_scenario_similarity_report(episodes)

    labels = report["validation"]["external_labels"]
    assert labels["status"] == "available"
    assert labels["records_with_labels"] == 2
    assert labels["selected_with_labels"] == 1
    assert labels["selected_positive_labels"] == 1
    assert labels["selected_label_conflicts"] == []
    trajectory = report["validation"]["trajectory_fields"]
    assert trajectory["status"] == "available"
    assert trajectory["records_with_trajectory_fields"] == 2
    assert trajectory["selected_with_trajectory_fields"] == 1
    assert trajectory["selected_fields_observed"] == ["min_rollout_clearance_m"]
    trajectory_metrics = report["validation"]["trajectory_metric_fields"]
    assert trajectory_metrics["status"] == "available"
    assert trajectory_metrics["records_with_trajectory_metric_fields"] == 2
    assert trajectory_metrics["selected_with_trajectory_metric_fields"] == 1
    assert trajectory_metrics["selected_metric_fields_observed"] == ["metrics.min_separation"]


def test_similarity_report_summarizes_trajectory_metric_validation(tmp_path: Path) -> None:
    """Representative benchmark summaries expose trajectory-derived metric evidence."""
    episodes = tmp_path / "episodes.jsonl"
    records = [
        {
            "episode_id": "durable-collision",
            "scenario_id": "classic_head_on_corridor_low",
            "seed": 202,
            "termination_reason": "collision",
            "metrics": {
                "collisions": 1,
                "near_misses": 0,
                "min_distance": 0.2,
                "min_clearance": -0.1,
                "socnavbench_path_length": 0.0,
            },
            "outcome": {"collision_event": True, "route_complete": False},
        },
        {
            "episode_id": "durable-safe",
            "scenario_id": "classic_crossing_low",
            "seed": 203,
            "termination_reason": "success",
            "metrics": {"collisions": 0, "near_misses": 0, "min_distance": 2.0},
            "outcome": {"collision_event": False, "route_complete": True},
        },
    ]
    _write_jsonl(episodes, records)

    report = build_collision_scenario_similarity_report(episodes, nearest_k=1)

    assert report["selection"]["selected_count"] == 1
    trajectory = report["validation"]["trajectory_fields"]
    assert trajectory["status"] == "unavailable"
    trajectory_metrics = report["validation"]["trajectory_metric_fields"]
    assert trajectory_metrics["status"] == "available"
    assert trajectory_metrics["records_with_trajectory_metric_fields"] == 2
    assert trajectory_metrics["selected_with_trajectory_metric_fields"] == 1
    assert trajectory_metrics["selected_metric_fields_observed"] == [
        "metrics.min_clearance",
        "metrics.min_distance",
        "metrics.socnavbench_path_length",
    ]


def test_similarity_report_compares_raw_trajectory_and_action_feature_sets() -> None:
    """Raw actor trajectories produce comparable optional nearest-neighbor feature sets."""
    report = build_collision_scenario_similarity_report(
        TRAJECTORY_FIXTURE,
        nearest_k=1,
        require_trajectory_comparison=True,
    )

    descriptor = report["descriptors"][0]
    assert descriptor["context_categorical"] == {"map_region": "north-crossing"}
    assert descriptor["trajectory_context"] == {
        "raw_actor_trajectories_available": True,
        "robot_samples": 3,
        "pedestrian_tracks": 1,
        "pedestrian_samples": 3,
        "action_samples": 3,
        "dt": 0.5,
    }
    assert descriptor["trajectory_action_numeric"][
        "trajectory_min_center_distance"
    ] == pytest.approx(0.1)
    assert descriptor["trajectory_action_numeric"][
        "trajectory_time_to_min_center_distance"
    ] == pytest.approx(0.5)

    comparison = report["feature_set_comparison"]
    assert comparison["status"] == "available"
    assert comparison["comparison_cohort_count"] == 3
    reports = {row["feature_set_id"]: row for row in comparison["reports"]}
    assert set(reports) == {
        "legacy_summary_v1",
        "trajectory_action_v1",
        "combined_context_v1",
    }
    assert reports["legacy_summary_v1"]["nearest_neighbors"] == report["nearest_neighbors"]
    assert reports["legacy_summary_v1"]["groups"] == report["groups"]
    for feature_report in reports.values():
        assert feature_report["status"] == "available"
        alpha_row = next(
            row
            for row in feature_report["nearest_neighbors"]
            if row["record_id"] == "trajectory-alpha-1"
        )
        assert alpha_row["neighbors"][0]["record_id"] == "trajectory-alpha-2"

    raw_validation = report["validation"]["raw_trajectory_arrays"]
    assert raw_validation["comparison_status"] == "available"
    assert raw_validation["selected_with_raw_trajectory_arrays"] == 3


def test_similarity_report_can_require_raw_trajectory_comparison(tmp_path: Path) -> None:
    """Explicit raw-trajectory comparison fails closed when fewer than two rows qualify."""
    episodes = tmp_path / "episodes.jsonl"
    _write_jsonl(episodes, _records())

    with pytest.raises(ValueError, match="at least two selected records"):
        build_collision_scenario_similarity_report(
            episodes,
            require_trajectory_comparison=True,
        )

    out_json = tmp_path / "similarity.json"
    exit_code = cli_main(
        [
            "collision-scenario-similarity",
            "--episodes-jsonl",
            str(episodes),
            "--out-json",
            str(out_json),
            "--require-trajectory-comparison",
        ]
    )
    assert exit_code == 2
    assert not out_json.exists()


def test_similarity_report_rejects_misaligned_malformed_position_series(tmp_path: Path) -> None:
    """Malformed samples exclude the whole actor series instead of shifting time alignment."""
    records = [
        json.loads(line) for line in TRAJECTORY_FIXTURE.read_text(encoding="utf-8").splitlines()[:2]
    ]
    records[0]["trajectory"]["robot_positions"][1] = ["invalid", 0.0]
    episodes = tmp_path / "malformed_trajectory.jsonl"
    _write_jsonl(episodes, records)

    with pytest.raises(ValueError, match="at least two selected records"):
        build_collision_scenario_similarity_report(
            episodes,
            require_trajectory_comparison=True,
        )


def test_trajectory_fixture_cli_writes_feature_set_table(tmp_path: Path) -> None:
    """Tracked synthetic fixture has a reproducible raw-trajectory CLI path."""
    out_json = tmp_path / "similarity.json"
    out_md = tmp_path / "similarity.md"

    exit_code = cli_main(
        [
            "collision-scenario-similarity",
            "--episodes-jsonl",
            str(TRAJECTORY_FIXTURE),
            "--out-json",
            str(out_json),
            "--out-markdown",
            str(out_md),
            "--nearest-k",
            "1",
            "--require-trajectory-comparison",
        ]
    )

    assert exit_code == 0
    assert (
        json.loads(out_json.read_text(encoding="utf-8"))["feature_set_comparison"]["status"]
        == "available"
    )
    markdown = out_md.read_text(encoding="utf-8")
    assert "Evidence status: `diagnostic-only`" in markdown
    assert "Candidate Feature-Set Comparison" in markdown
    assert "trajectory_action_v1" in markdown


def test_collision_scenario_similarity_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """CLI writes both machine-readable and reviewer-readable reports."""
    episodes = tmp_path / "episodes.jsonl"
    out_json = tmp_path / "similarity.json"
    out_md = tmp_path / "similarity.md"
    _write_jsonl(episodes, _records())

    exit_code = cli_main(
        [
            "collision-scenario-similarity",
            "--episodes-jsonl",
            str(episodes),
            "--out-json",
            str(out_json),
            "--out-markdown",
            str(out_md),
            "--nearest-k",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["nearest_neighbors"][0]["neighbors"]
    markdown = out_md.read_text(encoding="utf-8")
    assert "Collision Scenario Similarity Report" in markdown
    assert "Validation Context" in markdown
    assert "alpha-1" in markdown


def test_collision_scenario_similarity_cli_fails_closed_on_bad_input(tmp_path: Path) -> None:
    """Missing or malformed episode JSONL fails closed with exit code 2 and no output."""
    out_json = tmp_path / "similarity.json"

    missing = tmp_path / "does_not_exist.jsonl"
    missing_exit = cli_main(
        [
            "collision-scenario-similarity",
            "--episodes-jsonl",
            str(missing),
            "--out-json",
            str(out_json),
        ]
    )
    assert missing_exit == 2
    assert not out_json.exists()

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{not valid json}\n", encoding="utf-8")
    malformed_exit = cli_main(
        [
            "collision-scenario-similarity",
            "--episodes-jsonl",
            str(malformed),
            "--out-json",
            str(out_json),
        ]
    )
    assert malformed_exit == 2
    assert not out_json.exists()
