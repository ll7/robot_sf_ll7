"""Tests for issue #3971 pedestrian flow validation harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_sf.benchmark.pedestrian_flow_validation import (
    PED_FLOW_REPORT_SCHEMA_VERSION,
    FlowGate,
    PedFlowRunConfig,
    PedFlowTrace,
    build_ped_flow_scenarios,
    compute_flow_rate,
    run_ped_flow_trace,
    run_pedestrian_flow_validation,
    write_pedestrian_flow_report,
)


def test_ped_flow_harness_emits_required_metric_keys_on_short_run() -> None:
    """Short no-robot run covers all requested scenarios and metric families."""

    report = run_pedestrian_flow_validation(
        config=PedFlowRunConfig(duration_s=0.3, dt_s=0.1, pedestrian_counts=(2,), seed=3971)
    )

    assert report["schema_version"] == PED_FLOW_REPORT_SCHEMA_VERSION
    assert report["status"]["robot_inserted"] is False
    assert report["status"]["thresholds_applied"] is False
    assert set(report["scenario_ids"]) == {
        "bidirectional_corridor",
        "bottleneck",
        "forked_route",
    }
    for key in (
        "average_speed_vs_density",
        "flow_rate",
        "lane_formation_score",
        "jam_duration",
    ):
        assert key in report["flow_metrics"]
    for key in (
        "speed_mps",
        "acceleration_mps2",
        "curvature_1pm",
        "turning_angle_rad",
        "pairwise_distance_m",
        "stop_frequency_hz",
    ):
        assert key in report["trajectory_quality"]
    for run in report["runs"]:
        assert run["robot_count"] == 0
        assert run["trajectory_quality"]["speed_mps"]["status"] == "ok"


def test_no_robot_trace_produces_finite_pedestrian_states() -> None:
    """The harness composes Simulator directly with robots=[] as required."""

    scenario = build_ped_flow_scenarios()["bidirectional_corridor"]
    trace = run_ped_flow_trace(
        scenario,
        pedestrian_count=2,
        config=PedFlowRunConfig(duration_s=0.2, dt_s=0.1, pedestrian_counts=(2,)),
    )

    assert trace.robot_count == 0
    assert trace.positions.shape == (3, 2, 2)
    assert trace.velocities.shape == (3, 2, 2)
    assert trace.density_ped_per_m2 > 0.0
    assert trace.pedestrian_model


def test_bottleneck_speed_vs_density_is_monotone_ish_on_short_sanity_run() -> None:
    """Tiny bottleneck sanity remains descriptive, not a realism gate."""

    report = run_pedestrian_flow_validation(
        config=PedFlowRunConfig(duration_s=0.4, dt_s=0.1, pedestrian_counts=(2, 6), seed=3971),
        scenarios=("bottleneck",),
    )
    points = report["flow_metrics"]["average_speed_vs_density"]
    low = next(point for point in points if point["pedestrian_count"] == 2)
    high = next(point for point in points if point["pedestrian_count"] == 6)

    assert high["average_speed_mps"] <= low["average_speed_mps"] + 0.25


def test_report_writer_emits_expected_schema_files(tmp_path) -> None:
    """JSON, Markdown, and CSV evidence are compact and schema-labeled."""

    report = run_pedestrian_flow_validation(
        config=PedFlowRunConfig(duration_s=0.2, dt_s=0.1, pedestrian_counts=(2,), seed=3971),
        scenarios=("forked_route",),
    )
    written = write_pedestrian_flow_report(report, tmp_path)

    loaded = json.loads(written["summary_json"].read_text())
    assert loaded["schema_version"] == PED_FLOW_REPORT_SCHEMA_VERSION
    assert "Claim boundary: diagnostic-only" in written["summary_md"].read_text()
    csv_text = written["trajectory_quality_csv"].read_text()
    assert "scenario_id,pedestrian_count" in csv_text
    assert "speed_mps" in csv_text


def test_flow_config_rejects_non_finite_and_bool_controls() -> None:
    """Run controls fail closed on non-finite or bool scalar mistakes."""

    for kwargs in (
        {"duration_s": float("nan")},
        {"dt_s": float("inf")},
        {"speed_mps": True},
        {"jam_speed_threshold_mps": -0.1},
    ):
        with pytest.raises(ValueError):
            PedFlowRunConfig(**kwargs)


def test_flow_gate_crossings_are_strict_one_sided() -> None:
    """Samples starting on the gate boundary do not create duplicate crossings."""

    trace = PedFlowTrace(
        scenario_id="unit",
        pedestrian_count=2,
        density_ped_per_m2=1.0,
        seed=1,
        dt_s=0.1,
        duration_s=0.2,
        measurement_area_m2=2.0,
        pedestrian_model="unit",
        robot_count=0,
        positions=np.asarray(
            [
                [[-1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [2.0, 0.0]],
            ],
            dtype=float,
        ),
        velocities=np.zeros((3, 2, 2), dtype=float),
    )

    boundary_metric = compute_flow_rate(
        trace,
        FlowGate("gate", axis="x", coordinate=0.0, direction="positive"),
    )
    direct_trace = PedFlowTrace(
        scenario_id="unit",
        pedestrian_count=1,
        density_ped_per_m2=0.5,
        seed=1,
        dt_s=0.1,
        duration_s=0.1,
        measurement_area_m2=2.0,
        pedestrian_model="unit",
        robot_count=0,
        positions=np.asarray([[[-1.0, 0.0]], [[1.0, 0.0]]], dtype=float),
        velocities=np.zeros((2, 1, 2), dtype=float),
    )
    direct_metric = compute_flow_rate(
        direct_trace,
        FlowGate("gate", axis="x", coordinate=0.0, direction="positive"),
    )

    assert boundary_metric["crossing_pedestrian_count"] == 0
    assert direct_metric["crossing_pedestrian_count"] == 1
