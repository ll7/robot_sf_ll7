"""Tests for trace-exemplar interest scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.trace_exemplar_interest import (
    DEFAULT_WEIGHTS,
    discover_episode_dirs,
    score_bundles,
    score_episode,
    write_report_json,
    write_report_markdown,
)

REAL_4891_BUNDLE = Path("docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07")


def test_discovery_finds_nested_and_bare_episode_dirs(tmp_path: Path) -> None:
    """Discovery accepts both bundle-style and bare episode directories."""

    nested = _write_episode(
        tmp_path / "bundle" / "planner_a" / "scenario_seed1_best",
        planner="planner_a",
        status="success",
    )
    bare = _write_episode(tmp_path / "bare_episode", planner="planner_b", status="failure")

    assert discover_episode_dirs([tmp_path / "bundle", bare]) == sorted(
        [nested, bare],
        key=lambda path: path.as_posix(),
    )


def test_feature_bounds_and_composite_determinism(tmp_path: Path) -> None:
    """Episode feature normalization and composite ordering are deterministic."""

    episode_dir = _write_episode(
        tmp_path / "bundle" / "planner_a" / "scenario_seed1_best",
        planner="planner_a",
        status="failure",
        termination_reason="collision",
        distance_start=5.0,
        distance_drop=0.6,
    )

    first = score_episode(episode_dir)
    second = score_episode(episode_dir)

    assert first.composite_score == second.composite_score
    assert 0.0 <= first.composite_score <= 1.0
    assert set(first.features) == set(DEFAULT_WEIGHTS)
    assert all(0.0 <= value <= 1.0 for value in first.features.values())
    assert first.features["outcome_salience"] == 1.0


def test_weight_override_changes_composite_score(tmp_path: Path) -> None:
    """API weight overrides affect the weighted composite score."""

    root = tmp_path / "bundle"
    _write_episode(
        root / "planner_a" / "scenario_seed1_best", planner="planner_a", status="success"
    )

    default_report = score_bundles([root])
    overridden = score_bundles([root], weights={"min_dist_severity": 1.0, "outcome_salience": 0.0})

    assert default_report.weights["min_dist_severity"] == DEFAULT_WEIGHTS["min_dist_severity"]
    assert overridden.weights["min_dist_severity"] == 1.0
    assert default_report.episodes[0].composite_score != overridden.episodes[0].composite_score


def test_comparison_pair_detection_on_same_scenario_seed(tmp_path: Path) -> None:
    """Episodes with the same scenario and seed are paired across planners."""

    root = tmp_path / "bundle"
    _write_episode(
        root / "planner_a" / "scenario_seed7_best", planner="planner_a", status="success"
    )
    _write_episode(
        root / "planner_b" / "scenario_seed7_worst",
        planner="planner_b",
        status="failure",
        offset_x=3.0,
    )
    _write_episode(root / "planner_c" / "scenario_seed8_best", planner="planner_c", seed=8)

    report = score_bundles([root])

    assert len(report.comparison_pairs) == 1
    pair = report.comparison_pairs[0]
    assert pair.scenario_id == "scenario"
    assert pair.seed == 7
    assert pair.outcome_divergence == 1.0
    assert pair.trajectory_divergence > 0.0
    assert 0.0 <= pair.pair_score <= 1.0


def test_markdown_and_json_writers(tmp_path: Path) -> None:
    """Report writers produce deterministic readable artifacts."""

    root = tmp_path / "bundle"
    _write_episode(
        root / "planner_a" / "scenario_seed1_best", planner="planner_a", status="success"
    )
    report = score_bundles([root])
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_report_json(report, json_path)
    write_report_markdown(report, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["episodes"][0]["planner"] == "planner_a"
    assert payload["comparison_pairs"] == []
    assert "Trace-Exemplar Interest Report" in markdown
    assert "Episode Ranking" in markdown


@pytest.mark.skipif(not REAL_4891_BUNDLE.exists(), reason="real 4891 fixture bundle missing")
def test_real_4891_bundle_scores_within_bounds() -> None:
    """Real retained issue 4891 fixture bundle remains scoreable."""

    episode_dirs = discover_episode_dirs([REAL_4891_BUNDLE])
    report = score_bundles([REAL_4891_BUNDLE])

    assert len(episode_dirs) >= 9
    assert len(report.episodes) == len(episode_dirs)
    assert all(0.0 <= episode.composite_score <= 1.0 for episode in report.episodes)
    assert all(
        0.0 <= feature_value <= 1.0
        for episode in report.episodes
        for feature_value in episode.features.values()
    )


def _write_episode(
    episode_dir: Path,
    *,
    planner: str,
    status: str = "success",
    seed: int = 7,
    termination_reason: str = "success",
    distance_start: float = 4.0,
    distance_drop: float = 0.2,
    offset_x: float = 0.0,
) -> Path:
    episode_dir.mkdir(parents=True)
    rows = []
    frames = []
    for step in range(10):
        time_s = step * 0.2
        robot_x = offset_x + step * 0.5
        robot_y = 0.1 * (step % 2)
        row = {
            "step": step,
            "time_s": time_s,
            "robot_x_m": robot_x,
            "robot_y_m": robot_y,
            "robot_heading_rad": step * 0.08,
            "executed_vx_m_s": 0.5,
            "executed_vy_m_s": 0.0,
            "executed_speed_m_s": 0.2 + 0.03 * step,
            "commanded_linear_velocity_m_s": 1.0,
            "commanded_angular_velocity_rad_s": 0.1,
            "nearest_pedestrian_id": "0",
            "min_robot_ped_distance_m": distance_start - step * distance_drop,
            "pedestrian_count": 1,
            "pedestrian_positions_json": "[]",
        }
        rows.append(row)
        frames.append(
            {
                "step": step,
                "time_s": time_s,
                "robot": {
                    "position": [robot_x, robot_y],
                    "velocity": [0.5, 0.0],
                    "heading": step * 0.08,
                },
                "pedestrians": [],
            }
        )

    min_distance = min(row["min_robot_ped_distance_m"] for row in rows)
    metadata = {
        "campaign_id": "synthetic",
        "campaign_job": "local",
        "episode_id": f"scenario--{seed}--{planner}",
        "episode_status": status,
        "planner": planner,
        "scenario_id": "scenario",
        "seed": seed,
        "schema_version": "synthetic-trace.v1",
        "summary": {
            "episode_status": status,
            "global_min_robot_ped_distance_m": min_distance,
            "global_min_distance_step": 9,
            "step_count": len(rows),
            "termination_reason": termination_reason,
            "planner": planner,
            "scenario_id": "scenario",
            "seed": seed,
        },
    }
    trace = {"derived_rows": rows, "frames": frames, "metadata": metadata}
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (episode_dir / "trace_series.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return episode_dir
