"""Focused tests for seed-variability artifact builders."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.seed_variance import (
    build_seed_episode_rows,
    build_seed_variability_csv_rows,
    build_seed_variability_rows,
    build_statistical_sufficiency_rows,
)


def _sample_records() -> list[dict]:
    """Return deterministic episode rows for seed-variance aggregation tests."""
    return [
        {
            "episode_id": "ep-b",
            "scenario_id": "classic_cross_trap_low",
            "seed": 111,
            "algo": "orca",
            "planner_key": "orca",
            "planner_group": "core",
            "benchmark_profile": "baseline-safe",
            "kinematics": "differential_drive",
            "metrics": {
                "success": 1.0,
                "collisions": 0.0,
                "near_misses": 0.0,
                "time_to_goal_norm": 0.30,
                "snqi": 0.10,
            },
        },
        {
            "episode_id": "ep-a",
            "scenario_id": "classic_cross_trap_low",
            "seed": 111,
            "algo": "orca",
            "planner_key": "orca",
            "planner_group": "core",
            "benchmark_profile": "baseline-safe",
            "kinematics": "differential_drive",
            "metrics": {
                "success": 0.0,
                "collisions": 1.0,
                "near_misses": 1.0,
                "time_to_goal_norm": 0.90,
                "snqi": -0.20,
            },
        },
        {
            "episode_id": "ep-c",
            "scenario_id": "classic_cross_trap_low",
            "seed": 112,
            "algo": "orca",
            "planner_key": "orca",
            "planner_group": "core",
            "benchmark_profile": "baseline-safe",
            "kinematics": "differential_drive",
            "metrics": {
                "success": 1.0,
                "collisions": 0.0,
                "near_misses": 0.0,
                "time_to_goal_norm": 0.40,
                "snqi": 0.20,
            },
        },
    ]


def test_build_seed_variability_rows_adds_confidence_metadata() -> None:
    """Per-scenario rows should include bootstrap CI metadata and bounds."""
    rows = build_seed_variability_rows(
        _sample_records(),
        metrics=("success", "collisions", "near_misses", "time_to_goal_norm", "snqi"),
        campaign_id="campaign-1",
        config_hash="cfg-hash",
        git_hash="git-hash",
        seed_policy={"mode": "fixed-list", "resolved_seeds": [111, 112]},
        confidence_settings={
            "method": "bootstrap_mean_over_seed_means",
            "confidence": 0.95,
            "bootstrap_samples": 64,
            "bootstrap_seed": 7,
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 2
    assert row["seed_count"] == 2
    assert row["seed_list"] == [111, 112]
    assert row["provenance"]["confidence"]["method"] == "bootstrap_mean_over_seed_means"
    success_summary = row["summary"]["success"]
    assert success_summary["count"] == pytest.approx(2.0)
    assert success_summary["mean"] == pytest.approx(0.75)
    assert success_summary["ci_low"] <= success_summary["mean"] <= success_summary["ci_high"]
    assert success_summary["ci_half_width"] >= 0.0


def test_build_seed_variability_rows_no_bootstrap_ci_uses_group_mean() -> None:
    """Disabling bootstrap should collapse CI bounds to the across-seed mean."""
    rows = build_seed_variability_rows(
        _sample_records(),
        metrics=("success",),
        campaign_id="campaign-1",
        config_hash="cfg-hash",
        git_hash="git-hash",
        seed_policy={"mode": "fixed-list", "resolved_seeds": [111, 112]},
        confidence_settings={
            "method": "bootstrap_mean_over_seed_means",
            "confidence": 0.95,
            "bootstrap_samples": 0,
            "bootstrap_seed": 7,
        },
    )

    success_summary = rows[0]["summary"]["success"]
    assert success_summary["mean"] == pytest.approx(0.75)
    assert success_summary["ci_low"] == pytest.approx(0.75)
    assert success_summary["ci_high"] == pytest.approx(0.75)
    assert success_summary["ci_half_width"] == pytest.approx(0.0)


def test_build_seed_variability_csv_rows_retains_ci_columns() -> None:
    """CSV export should carry CI-bearing aggregate columns for each metric."""
    rows = build_seed_variability_rows(
        _sample_records(),
        metrics=("success", "collisions", "near_misses", "time_to_goal_norm", "snqi"),
        campaign_id="campaign-1",
        config_hash="cfg-hash",
        git_hash="git-hash",
        seed_policy={"mode": "fixed-list", "resolved_seeds": [111, 112]},
        confidence_settings={
            "method": "bootstrap_mean_over_seed_means",
            "confidence": 0.95,
            "bootstrap_samples": 32,
            "bootstrap_seed": 3,
        },
    )
    csv_rows = build_seed_variability_csv_rows(
        rows,
        metrics=("success", "collisions", "near_misses", "time_to_goal_norm", "snqi"),
    )

    assert len(csv_rows) == 2
    first = csv_rows[0]
    assert "confidence_method" in first
    assert "success_ci_low" in first
    assert "success_ci_high" in first
    assert "success_ci_half_width" in first
    assert first["confidence_method"] == "bootstrap_mean_over_seed_means"


def test_build_seed_episode_rows_assigns_repeat_index_deterministically() -> None:
    """Repeat indices should be stable within each scenario/planner/seed group."""
    rows = build_seed_episode_rows(_sample_records())

    assert [row["episode_id"] for row in rows[:2]] == ["ep-a", "ep-b"]
    assert [row["repeat_index"] for row in rows[:2]] == [0, 1]
    assert rows[0]["planner_key"] == "orca"
    assert rows[0]["algo"] == "orca"
    assert rows[0]["collision"] == pytest.approx(1.0)
    assert rows[0]["near_miss"] == pytest.approx(1.0)
    assert rows[0]["time_to_goal"] == pytest.approx(0.90)


def test_build_seed_episode_rows_collision_remains_legacy_metric_alias() -> None:
    """Compatibility export keeps collision sourced from metrics, not outcome events."""
    rows = build_seed_episode_rows(
        [
            {
                "episode_id": "ep-legacy-collision",
                "scenario_id": "classic_cross_trap_low",
                "seed": 111,
                "algo": "orca",
                "planner_key": "orca",
                "kinematics": "differential_drive",
                "metrics": {
                    "success": 1.0,
                    "collisions": 0.0,
                    "near_misses": 0.0,
                    "time_to_goal_norm": 0.20,
                },
                "outcome": {
                    "collision_event": True,
                },
            }
        ]
    )

    assert rows[0]["collision"] == 0.0


def test_build_seed_episode_rows_groups_repeat_index_by_kinematics() -> None:
    """Repeat indices should restart when the same planner runs under other kinematics."""
    records = _sample_records() + [
        {
            "episode_id": "ep-z",
            "scenario_id": "classic_cross_trap_low",
            "seed": 111,
            "algo": "orca",
            "planner_key": "orca",
            "planner_group": "core",
            "benchmark_profile": "baseline-safe",
            "kinematics": "holonomic",
            "metrics": {
                "success": 1.0,
                "collisions": 0.0,
                "near_misses": 0.0,
                "time_to_goal_norm": 0.20,
                "snqi": 0.30,
            },
        }
    ]

    rows = build_seed_episode_rows(records)
    holonomic_rows = [row for row in rows if row["kinematics"] == "holonomic"]
    differential_rows = [row for row in rows if row["kinematics"] == "differential_drive"]

    assert len(holonomic_rows) == 1
    assert holonomic_rows[0]["episode_id"] == "ep-z"
    assert holonomic_rows[0]["repeat_index"] == 0
    assert holonomic_rows[0]["success"] == pytest.approx(1.0)
    assert holonomic_rows[0]["time_to_goal"] == pytest.approx(0.20)
    assert [row["repeat_index"] for row in differential_rows[:2]] == [0, 1]


def test_build_seed_episode_rows_uses_algorithm_metadata_fallback() -> None:
    """Records with only algorithm metadata should still keep planner traceability."""
    rows = build_seed_episode_rows(
        [
            {
                "episode_id": "ep-meta",
                "scenario_id": "classic_cross_trap_low",
                "seed": 111,
                "kinematics": "differential_drive",
                "algorithm_metadata": {"algorithm": "orca"},
                "metrics": {
                    "success": 1.0,
                    "collisions": 0.0,
                    "near_misses": 0.0,
                    "time_to_goal_norm": 0.20,
                },
            }
        ]
    )

    assert rows[0]["planner_key"] == "orca"
    assert rows[0]["algo"] == "orca"


def test_build_seed_episode_rows_emits_mechanism_taxonomy_and_exposure_fields() -> None:
    """Episode rows expose trace-verified taxonomy plus native exposure fields."""

    rows = build_seed_episode_rows(
        [
            {
                "episode_id": "ep-taxonomy",
                "scenario_id": "classic_cross_trap_low",
                "seed": 111,
                "algo": "orca",
                "planner_key": "orca",
                "kinematics": "differential_drive",
                "failure_mechanism_taxonomy": {
                    "label": "static_deadlock_or_local_minimum",
                    "source": "trace_review_issue_2220",
                    "trace_status": "trace_verified",
                    "confidence": "observed_mechanism",
                },
                "metrics": {
                    "success": 0.0,
                    "collisions": 0.0,
                    "near_misses": 1.0,
                    "time_to_goal_norm": 1.2,
                    "interaction_exposure_share": 0.75,
                    "robot_motion_share_before_first_clearance": 0.5,
                    "first_clearance_step": 42,
                    "low_exposure_success": 0.0,
                },
            },
            {
                "episode_id": "ep-geometry-only",
                "scenario_id": "classic_cross_trap_low",
                "seed": 112,
                "algo": "orca",
                "planner_key": "orca",
                "kinematics": "differential_drive",
                "metrics": {
                    "success": 0.0,
                    "collisions": 0.0,
                    "near_misses": 1.0,
                    "time_to_goal_norm": 1.2,
                    "geometry_label": "static_deadlock_or_local_minimum",
                },
            },
        ]
    )

    trace_row = next(row for row in rows if row["episode_id"] == "ep-taxonomy")
    assert trace_row["failure_mechanism_taxonomy_schema"] == "failure_mechanism_taxonomy.v1"
    assert trace_row["failure_mechanism_label"] == "static_deadlock_or_local_minimum"
    assert trace_row["failure_mechanism_source"] == "trace_review_issue_2220"
    assert trace_row["failure_mechanism_trace_status"] == "trace_verified"
    assert trace_row["failure_mechanism_confidence"] == "observed_mechanism"
    assert trace_row["mechanism_schema_version"] == "failure_mechanism_taxonomy.v1"
    assert trace_row["mechanism_label"] == "static_deadlock_or_local_minimum"
    assert trace_row["mechanism_confidence"] == "observed_mechanism"
    assert trace_row["mechanism_evidence_mode"] == "paired_trace"
    assert trace_row["mechanism_evidence_uri"] == "trace_review_issue_2220"
    assert trace_row["interaction_exposure_share"] == pytest.approx(0.75)
    assert trace_row["interaction_exposure_schema_version"] == "interaction_exposure.v1"
    assert trace_row["interaction_exposure_status"] == "computed"
    assert trace_row["robot_motion_share_before_first_clearance"] == pytest.approx(0.5)
    assert trace_row["first_clearance_step"] == pytest.approx(42)
    assert trace_row["low_exposure_success"] == pytest.approx(0.0)
    assert trace_row["interaction_exposure_source"] == "episode_metrics"

    geometry_row = next(row for row in rows if row["episode_id"] == "ep-geometry-only")
    assert geometry_row["failure_mechanism_label"] is None
    assert geometry_row["failure_mechanism_trace_status"] == "not_derivable"
    assert geometry_row["mechanism_label"] == "unknown"
    assert geometry_row["mechanism_evidence_mode"] == "unknown"
    assert geometry_row["interaction_exposure_source"] == "not_derivable_from_episode_record"
    assert geometry_row["interaction_exposure_status"] == "not_derivable_missing_trace"


def test_build_statistical_sufficiency_rows_exposes_half_widths() -> None:
    """Sufficiency rows should surface per-metric CI half-widths and counts."""
    rows = build_seed_variability_rows(
        _sample_records(),
        metrics=("success", "collisions", "near_misses", "time_to_goal_norm", "snqi"),
        campaign_id="campaign-1",
        config_hash="cfg-hash",
        git_hash="git-hash",
        seed_policy={"mode": "fixed-list", "resolved_seeds": [111, 112]},
        confidence_settings={
            "method": "bootstrap_mean_over_seed_means",
            "confidence": 0.95,
            "bootstrap_samples": 32,
            "bootstrap_seed": 5,
        },
    )
    sufficiency_rows = build_statistical_sufficiency_rows(
        rows,
        metrics=("success", "collisions", "near_misses", "time_to_goal_norm", "snqi"),
    )

    assert len(sufficiency_rows) == 1
    row = sufficiency_rows[0]
    assert row["sufficiency_status"] == "reported"
    assert row["metric_half_widths"]["success"] >= 0.0
    assert row["metric_half_widths"]["collision"] >= 0.0
    assert row["metric_half_widths"]["near_miss"] >= 0.0
    assert row["metric_half_widths"]["time_to_goal"] >= 0.0
    assert row["metrics"]["success"]["n"] == pytest.approx(2.0)
    assert row["kinematics"] == "differential_drive"
