"""Tests for trace-derived cross-benchmark metric wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.cross_benchmark_metrics import (
    CROSS_BENCHMARK_CLAIM_BOUNDARY,
    CROSS_BENCHMARK_METRIC_MAPPING_VERSION,
    CROSS_BENCHMARK_METRIC_REPORT_SCHEMA_VERSION,
    CrossBenchmarkMetricRow,
    build_cross_benchmark_metric_report,
    compute_cross_benchmark_metric_rows,
    cross_benchmark_metric_mappings,
    mapping_ids,
    mapping_table_as_dicts,
    summarize_status_counts,
)
from robot_sf.benchmark.metrics import EpisodeData

REPO_ROOT = Path(__file__).parents[2]
MAPPING_PATH = REPO_ROOT / "configs/benchmarks/cross_benchmark_metric_mapping_v1.yaml"
DOC_PATH = REPO_ROOT / "docs/context/issue_3286_cross_benchmark_metric_wrappers.md"


def _fixture_episode() -> EpisodeData:
    """Return a tiny trace with one approaching robot/pedestrian pair."""
    robot_pos = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    robot_vel = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    peds_pos = np.asarray(
        [
            [[4.0, 0.0]],
            [[4.0, 0.0]],
            [[4.0, 0.0]],
        ],
        dtype=float,
    )
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=np.zeros_like(robot_pos),
        peds_pos=peds_pos,
        ped_forces=np.ones_like(peds_pos),
        goal=np.asarray([2.0, 0.0], dtype=float),
        dt=1.0,
        reached_goal_step=2,
    )


def _rows_by_id() -> dict[str, CrossBenchmarkMetricRow]:
    """Compute fixture rows keyed by metric ID."""
    rows = compute_cross_benchmark_metric_rows(_fixture_episode(), horizon=5)
    return {row.metric_id: row for row in rows}


def test_mapping_config_matches_runtime_mapping_table() -> None:
    """The checked-in YAML mapping should stay aligned with the runtime table."""
    payload = yaml.safe_load(MAPPING_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CROSS_BENCHMARK_METRIC_MAPPING_VERSION
    assert payload["issue"] == 3286
    assert payload["claim_boundary"] == CROSS_BENCHMARK_CLAIM_BOUNDARY

    config_rows = payload["rows"]
    runtime_rows = mapping_table_as_dicts()
    assert mapping_ids(config_rows) == mapping_ids(runtime_rows)
    assert mapping_ids(config_rows) == {row.metric_id for row in cross_benchmark_metric_mappings()}

    for row in config_rows:
        assert {
            "metric_id",
            "benchmark",
            "external_metric",
            "robot_sf_source",
            "units",
            "denominator",
            "mapping_class",
            "evidence_tier",
            "semantic_notes",
        } <= set(row)
        assert row["mapping_class"] in payload["status_vocabulary"]["mapping_class"]


def test_fixture_report_labels_available_approximate_and_unavailable_rows() -> None:
    """Wrapper reports should preserve status instead of collapsing unavailable rows."""
    rows = compute_cross_benchmark_metric_rows(_fixture_episode(), horizon=5)
    counts = summarize_status_counts(rows)

    assert counts["available"] >= 3
    assert counts["approximate"] >= 2
    assert counts["unavailable"] >= 2

    report = build_cross_benchmark_metric_report(_fixture_episode(), horizon=5)
    assert report["schema_version"] == CROSS_BENCHMARK_METRIC_REPORT_SCHEMA_VERSION
    assert report["mapping_version"] == CROSS_BENCHMARK_METRIC_MAPPING_VERSION
    assert report["claim_boundary"] == CROSS_BENCHMARK_CLAIM_BOUNDARY
    assert {row["status"] for row in report["rows"]} == {
        "available",
        "approximate",
        "unavailable",
    }


def test_fixture_computes_trace_derivable_external_style_metrics() -> None:
    """Deterministic fixture values should match the underlying Robot SF trace helpers."""
    rows = _rows_by_id()

    path_ratio = rows["socnavbench.path_length_ratio"]
    assert path_ratio.status == "available"
    assert path_ratio.value == pytest.approx(1.000005)
    assert path_ratio.units == "ratio"

    traversal_time = rows["common.traversal_time_s"]
    assert traversal_time.status == "approximate"
    assert traversal_time.value == pytest.approx(2.0)
    assert "timeout" in traversal_time.semantic_notes

    ttc = rows["common.time_to_collision_min_s"]
    assert ttc.status == "approximate"
    assert ttc.value == pytest.approx(2.0)
    assert ttc.denominator == "minimum_approaching_robot_pedestrian_pair"

    closest = rows["common.closest_pedestrian_distance_m"]
    assert closest.status == "available"
    assert closest.value == pytest.approx(2.0)
    assert "center-to-center" in closest.semantic_notes

    success = rows["robot_sf.success_trace_predicate"]
    assert success.status == "available"
    assert success.value == pytest.approx(1.0)


def test_unavailable_external_only_metrics_keep_reasons() -> None:
    """External simulator-only rows should be explicit unavailable rows, not zeros."""
    rows = _rows_by_id()

    for metric_id in (
        "socnavbench.personal_space_objective",
        "hunavsim.human_behavior_cost",
    ):
        row = rows[metric_id]
        assert row.status == "unavailable"
        assert row.value is None
        assert row.robot_sf_source is None
        assert row.unavailable_reason == "external_metric_not_trace_derivable"


def test_success_predicate_is_unavailable_without_horizon() -> None:
    """Rows that need extra context should fail closed as unavailable."""
    rows = {row.metric_id: row for row in compute_cross_benchmark_metric_rows(_fixture_episode())}

    success = rows["robot_sf.success_trace_predicate"]
    assert success.status == "unavailable"
    assert success.unavailable_reason == "required_context_missing"


def test_context_note_links_mapping_and_wrapper_surfaces() -> None:
    """The compact context note should make the wrapper discoverable."""
    note = DOC_PATH.read_text(encoding="utf-8")
    assert "configs/benchmarks/cross_benchmark_metric_mapping_v1.yaml" in note
    assert "robot_sf/benchmark/cross_benchmark_metrics.py" in note
    assert "not simulator parity or paper-grade evidence" in note
