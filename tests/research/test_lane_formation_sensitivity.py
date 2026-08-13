"""Tests for the issue #6962 lane-formation sensitivity diagnostic."""

from __future__ import annotations

import pytest

from robot_sf.research.emergent_phenomena import (
    RELEASED_DEFAULT_CALIBRATION,
    released_default_config,
)
from robot_sf.research.lane_formation_sensitivity import (
    build_corridor_surface,
    build_threshold_grid,
    diagnostic_manifest,
    run_lane_formation_sensitivity,
    summarize_sensitivity_rows,
)


def test_build_corridor_surface_exposes_geometry_population_duration_axes():
    scenarios = build_corridor_surface(
        seeds=[1, 2],
        lengths_m=[12.0],
        half_widths_m=[1.5, 2.0],
        pedestrian_counts=[8],
        steps=[20],
    )
    assert len(scenarios) == 4
    assert {scenario.length for scenario in scenarios} == {12.0}
    assert {scenario.half_width for scenario in scenarios} == {1.5, 2.0}
    assert {scenario.n_pedestrians for scenario in scenarios} == {8}
    assert {scenario.n_steps for scenario in scenarios} == {20}
    assert {scenario.seed for scenario in scenarios} == {1, 2}
    assert all(scenario.name == "bidirectional_corridor" for scenario in scenarios)
    assert all("density_peds_per_m2" in scenario.extra for scenario in scenarios)


def test_build_corridor_surface_fails_closed_on_empty_or_invalid_axis():
    with pytest.raises(ValueError, match="seeds"):
        build_corridor_surface(seeds=[])
    with pytest.raises(ValueError, match="lengths_m"):
        build_corridor_surface(lengths_m=[0.0])
    with pytest.raises(ValueError, match="pedestrian_counts"):
        build_corridor_surface(pedestrian_counts=[0])
    with pytest.raises(ValueError, match="steps"):
        build_corridor_surface(steps=[1.5])


def test_threshold_grid_is_explicit_and_sorted():
    specs = build_threshold_grid({"lane_purity": [0.8, 0.4]})
    assert [spec.label for spec in specs] == ["lane_purity>=0.4", "lane_purity>=0.8"]
    assert [spec.threshold for spec in specs] == [0.4, 0.8]


def test_threshold_grid_rejects_unknown_metrics():
    with pytest.raises(ValueError, match="unsupported threshold metric"):
        build_threshold_grid({"unknown_metric": [0.5]})


@pytest.mark.parametrize(
    ("execution_mode", "status"),
    [
        ("adapter", "computed"),
        ("fallback", "computed"),
        ("degraded", "computed"),
        ("unavailable", "computed"),
        ("native", "failed"),
    ],
)
def test_manifest_rejects_any_non_native_or_noncomputed_row(execution_mode, status):
    row = {"execution": {"execution_mode": execution_mode, "status": status}}
    manifest = diagnostic_manifest(
        rows=[row],
        threshold_specs=build_threshold_grid({"lane_segregation_index": [0.5]}),
        sim_config=released_default_config(),
        axes={},
    )
    assert manifest["execution_policy"]["non_native_rows"] == {f"{execution_mode}:{status}": 1}


def test_run_lane_formation_sensitivity_smoke_reuses_native_harness():
    payload = run_lane_formation_sensitivity(
        seeds=[7],
        lengths_m=[8.0],
        half_widths_m=[1.5],
        pedestrian_counts=[6],
        steps=[10],
        calibrations=[RELEASED_DEFAULT_CALIBRATION],
        thresholds={"lane_segregation_index": [0.15], "lane_purity": [0.4]},
    )
    assert payload["schema_version"] == "lane_formation_sensitivity_diagnostic.v1"
    assert (
        payload["manifest"]["claim_boundary"] == "diagnostic_only_not_benchmark_or_paper_evidence"
    )
    assert payload["manifest"]["released_defaults_changed"] is False
    assert payload["manifest"]["metric_semantics_changed"] is False
    assert payload["manifest"]["execution_policy"]["non_native_rows"] == {}
    [row] = payload["rows"]
    assert row["execution"]["execution_mode"] == "native"
    assert row["execution"]["status"] == "computed"
    assert set(row["metrics"]) == {"lane_segregation_index", "lane_purity"}
    assert set(row["threshold_evaluations"]) == {
        "lane_purity>=0.4",
        "lane_segregation_index>=0.15",
    }
    assert len(payload["summary"]) == 1


def test_summarize_sensitivity_rows_reports_threshold_hit_rates_by_cell():
    rows = [
        {
            "calibration": "released_default",
            "seed": 1,
            "geometry": {"length_m": 10.0, "half_width_m": 2.0},
            "population": {"n_pedestrians": 8},
            "duration": {"n_steps": 20},
            "metrics": {"lane_segregation_index": 0.2},
            "threshold_evaluations": {"lane_segregation_index>=0.15": {"meets_threshold": True}},
            "execution": {"execution_mode": "native"},
        },
        {
            "calibration": "released_default",
            "seed": 2,
            "geometry": {"length_m": 10.0, "half_width_m": 2.0},
            "population": {"n_pedestrians": 8},
            "duration": {"n_steps": 20},
            "metrics": {"lane_segregation_index": 0.1},
            "threshold_evaluations": {"lane_segregation_index>=0.15": {"meets_threshold": False}},
            "execution": {"execution_mode": "native"},
        },
    ]
    [summary] = summarize_sensitivity_rows(rows)
    assert summary["n_seeds"] == 2
    assert summary["threshold_hit_rates"]["lane_segregation_index>=0.15"] == pytest.approx(0.5)
    assert summary["metric_stats"]["lane_segregation_index"]["mean"] == pytest.approx(0.15)
