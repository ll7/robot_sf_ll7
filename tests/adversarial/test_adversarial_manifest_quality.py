"""Tests for compact adversarial manifest quality summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.manifest_quality import (
    MANIFEST_QUALITY_SCHEMA_VERSION,
    _load_records,
    summarize_adversarial_manifest_quality,
)
from robot_sf.adversarial.manifest_quality import main as quality_cli_main
from robot_sf.adversarial.scenario_manifest import compute_control_hash


def _candidate_controls(
    *,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    scenario_seed: int = 7,
) -> dict:
    return {
        "start": {"x": float(start_x), "y": float(start_y)},
        "goal": {"x": float(goal_x), "y": float(goal_y)},
        "spawn_time_s": 0.0,
        "pedestrian_speed_mps": 1.0,
        "pedestrian_delay_s": 0.0,
        "scenario_seed": scenario_seed,
    }


def _manifest_payload(controls: dict, status: str) -> dict:
    candidate = CandidateSpec(
        start=Pose2D(controls["start"]["x"], controls["start"]["y"]),
        goal=Pose2D(controls["goal"]["x"], controls["goal"]["y"]),
        spawn_time_s=float(controls["spawn_time_s"]),
        pedestrian_speed_mps=float(controls["pedestrian_speed_mps"]),
        pedestrian_delay_s=float(controls["pedestrian_delay_s"]),
        scenario_seed=int(controls["scenario_seed"]),
    )
    return {
        "schema_version": "adversarial_scenario_manifest.v1",
        "candidate_controls": controls,
        "validation": {
            "status": status,
            "errors": [],
            "warnings": [],
            "normalized_control_hash": compute_control_hash(candidate),
        },
    }


def _write_manifest(path: Path, controls: dict, status: str) -> None:
    path.write_text(
        yaml.safe_dump(_manifest_payload(controls, status), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_manifest_without_schema(path: Path, controls: dict, status: str) -> None:
    payload = _manifest_payload(controls, status)
    del payload["schema_version"]
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_summarize_manifest_rates_and_novelty(tmp_path: Path) -> None:
    controls_a = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    controls_b = _candidate_controls(start_x=1.5, start_y=2.0, goal_x=5.0, goal_y=2.0)
    controls_c = _candidate_controls(start_x=2.0, start_y=2.0, goal_x=6.0, goal_y=2.0)

    _write_manifest(tmp_path / "a.yaml", controls_a, "valid")
    _write_manifest(tmp_path / "b.yaml", controls_b, "invalid")
    _write_manifest(tmp_path / "c.yaml", controls_c, "degenerate")
    _write_manifest(tmp_path / "d.yaml", controls_a, "valid")

    result = summarize_adversarial_manifest_quality([tmp_path])

    assert result.manifest_count == 4
    assert result.status_counts["valid"] == 2
    assert result.status_counts["invalid"] == 1
    assert result.status_counts["degenerate"] == 1
    assert result.validity_rate == 0.5
    assert result.invalid_rate == 0.25
    assert result.degenerate_rate == 0.25
    assert result.hashable_count == 4
    assert result.duplicate_hash_count == 1
    assert result.unique_hash_count == 3
    assert result.novelty_rate == 0.75
    assert result.duplicate_rate == 0.25


def test_missing_manifest_schema_version_stays_none(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    manifest_path = tmp_path / "a.yaml"
    _write_manifest_without_schema(manifest_path, controls, "valid")

    records, parse_errors = _load_records([manifest_path], reference_vector=None)

    assert parse_errors == []
    assert records[0].schema_version is None


def test_summarize_perturbation_distance(tmp_path: Path) -> None:
    base = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    moved = _candidate_controls(start_x=2.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    far = _candidate_controls(
        start_x=1.0,
        start_y=4.0,
        goal_x=5.0,
        goal_y=4.0,
        scenario_seed=999,
    )

    ref_path = tmp_path / "reference.yaml"
    _write_manifest(ref_path, base, "valid")

    _write_manifest(tmp_path / "moved.yaml", moved, "valid")
    _write_manifest(tmp_path / "far.yaml", far, "valid")

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        reference_manifest=ref_path,
    )

    assert result.perturbation_reference == ref_path.as_posix()
    assert result.perturbation_count == 2
    assert result.perturbation_min == 1.0
    assert result.perturbation_max == 2.828427
    assert result.perturbation_mean == 1.914214


def test_reference_manifest_exclusion_normalizes_equivalent_paths(tmp_path: Path) -> None:
    base = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    moved = _candidate_controls(start_x=2.0, start_y=2.0, goal_x=5.0, goal_y=2.0)

    ref_path = tmp_path / "reference.yaml"
    _write_manifest(ref_path, base, "valid")
    _write_manifest(tmp_path / "moved.yaml", moved, "valid")
    (tmp_path / "subdir").mkdir()

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        reference_manifest=tmp_path / "subdir" / ".." / "reference.yaml",
    )

    assert result.manifest_count == 1
    assert result.perturbation_count == 1
    assert result.perturbation_min == 1.0


def test_summarize_planner_yields_from_smoke_summary(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    episodes = tmp_path / "episodes_goal.jsonl"
    episodes.write_text(
        "\n".join(
            [
                '{"status": "success", "termination_reason": "success", "metrics": {"near_misses": 0}}',
                '{"status": "collision", "termination_reason": "collision", "metrics": {"near_misses": 2}}',
                '{"status": "truncated", "termination_reason": "truncated", "metrics": {"near_misses": 1}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    smoke_summary = tmp_path / "smoke_summary.json"
    smoke_summary.write_text(
        '{"planner_runs": [{"planner": "goal", "out_path": "' + episodes.as_posix() + '"}]}',
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        smoke_summary_json=smoke_summary,
    )

    assert result.planner_outcomes is not None
    planners = result.planner_outcomes.planners
    assert len(planners) == 1
    assert planners[0].failure_count == 2
    assert planners[0].near_miss_count == 2
    assert planners[0].failure_yield == pytest.approx(2 / 3)
    assert planners[0].near_miss_yield == pytest.approx(2 / 3)


def test_summarize_planner_yields_from_aggregate_smoke_summary(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    smoke_summary = tmp_path / "smoke_summary.json"
    smoke_summary.write_text(
        json.dumps(
            {
                "planner_runs": [
                    {
                        "planner": "social_force",
                        "written": 2,
                        "total_jobs": 2,
                        "metrics": {
                            "episodes": 2,
                            "success": {"sum": 0.0},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        smoke_summary_json=smoke_summary,
    )

    assert result.planner_outcomes is not None
    planners = result.planner_outcomes.planners
    assert result.planner_outcomes.available is True
    assert len(planners) == 1
    assert planners[0].source == "aggregate_metrics"
    assert planners[0].episodes == 2
    assert planners[0].failure_count == 2
    assert planners[0].failure_yield == 1.0
    assert planners[0].near_miss_count is None
    assert planners[0].near_miss_yield is None


def test_aggregate_success_yield_requires_count_like_sum(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    smoke_summary = tmp_path / "smoke_summary.json"
    smoke_summary.write_text(
        json.dumps(
            {
                "planner_runs": [
                    {
                        "planner": "ambiguous",
                        "written": 2,
                        "metrics": {
                            "episodes": 2,
                            "success": {"sum": 1.5},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        smoke_summary_json=smoke_summary,
    )

    assert result.planner_outcomes is not None
    assert result.planner_outcomes.available is True
    planner = result.planner_outcomes.planners[0]
    assert planner.source == "aggregate_metrics"
    assert planner.episodes == 2
    assert planner.failure_count is None
    assert planner.failure_yield is None


def test_aggregate_episode_count_preserves_written_zero(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    smoke_summary = tmp_path / "smoke_summary.json"
    smoke_summary.write_text(
        json.dumps(
            {
                "planner_runs": [
                    {
                        "planner": "empty",
                        "written": 0,
                        "total_jobs": 2,
                        "metrics": {
                            "success": {"sum": 0.0},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_adversarial_manifest_quality(
        [tmp_path],
        smoke_summary_json=smoke_summary,
    )

    assert result.planner_outcomes is not None
    planner = result.planner_outcomes.planners[0]
    assert planner.episodes == 0
    assert planner.failure_count == 0
    assert planner.failure_yield == 0.0


def test_manifest_quality_cli_writes_output_json(tmp_path: Path) -> None:
    controls = _candidate_controls(start_x=1.0, start_y=2.0, goal_x=5.0, goal_y=2.0)
    _write_manifest(tmp_path / "a.yaml", controls, "valid")

    output_json = tmp_path / "quality_summary.json"

    exit_code = quality_cli_main([str(tmp_path), "--output-json", str(output_json)])

    assert exit_code == 0
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == MANIFEST_QUALITY_SCHEMA_VERSION
    assert loaded["manifest_count"] == 1
    assert loaded["rates"]["validity_rate"] == 1.0
