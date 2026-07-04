"""Tests for collision scenario similarity analysis reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.collision_scenario_similarity import (
    SCHEMA_VERSION,
    build_collision_scenario_similarity_report,
    describe_collision_scenarios,
)

if TYPE_CHECKING:
    from pathlib import Path


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
