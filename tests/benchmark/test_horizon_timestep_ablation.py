"""Tests for the horizon x timestep forecast ablation report."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/build_horizon_timestep_ablation_report.py"


def _load_script_module():
    """Load the ablation script as a module for testing."""
    spec = importlib.util.spec_from_file_location(
        "build_horizon_timestep_ablation_report", _SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_horizon_timestep_ablation_report"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_script_module()

CLAIM_BOUNDARY = _mod.CLAIM_BOUNDARY
DT_LADDER_S = _mod.DT_LADDER_S
HORIZON_LADDER_S = _mod.HORIZON_LADDER_S
MISSING_FAMILIES = _mod.MISSING_FAMILIES
TRACE_CANDIDATES = _mod.TRACE_CANDIDATES

_evaluate_ablation_cell = _mod.evaluate_ablation_cell
_cells_for_candidate = _mod._cells_for_candidate
_build_summary_rows = _mod._build_summary_rows
_preset_recommendations = _mod._preset_recommendations
_resample_trace_steps = _mod._resample_trace_steps
_extract_trace_steps = _mod._extract_trace_steps
_compute_native_dt_s = _mod._compute_native_dt_s
_artifact_size_bytes = _mod._artifact_size_bytes
build_ablation_report = _mod.build_ablation_report


def test_horizon_and_dt_ladders_cover_request() -> None:
    """The ablation ladder includes the issue-requested horizon and dt values."""
    assert 0.5 in HORIZON_LADDER_S
    assert 1.0 in HORIZON_LADDER_S
    assert 1.6 in HORIZON_LADDER_S
    assert 2.0 in HORIZON_LADDER_S
    assert 3.0 in HORIZON_LADDER_S
    assert 0.1 in DT_LADDER_S
    assert 0.2 in DT_LADDER_S


def test_claim_boundary_is_diagnostic_only() -> None:
    """The claim boundary explicitly excludes navigation/benchmark claims."""
    assert "analysis_only_not_navigation_evidence" in CLAIM_BOUNDARY
    assert "navigation" in CLAIM_BOUNDARY
    assert "benchmark" in CLAIM_BOUNDARY


def test_missing_trace_file_fails_closed() -> None:
    """A missing fixture produces an unavailable cell, not fabricated metrics."""
    cell = _evaluate_ablation_cell(
        {
            "family": "test",
            "label": "missing",
            "path": "nonexistent/path.json",
            "scenario_id": "test",
            "planner_id": "test",
            "seed": 0,
        },
        horizon_s=1.0,
        requested_dt_s=0.1,
    )
    assert cell["status"] == "trace_file_missing"
    assert cell["metrics"] == {}
    assert "fabricat" not in cell.get("limitation", "").lower()


def test_resample_trace_steps_preserves_native_dt_for_fine_target() -> None:
    """Resampling to a dt equal to native dt returns the trace unchanged."""
    trace = {
        "frames": [
            {
                "time_s": 0.0,
                "pedestrians": [{"id": 1, "position": [0.0, 0.0], "velocity": [1.0, 0.0]}],
            },
            {
                "time_s": 0.1,
                "pedestrians": [{"id": 1, "position": [0.1, 0.0], "velocity": [1.0, 0.0]}],
            },
            {
                "time_s": 0.2,
                "pedestrians": [{"id": 1, "position": [0.2, 0.0], "velocity": [1.0, 0.0]}],
            },
        ]
    }
    steps = _extract_trace_steps(trace)
    resampled, actual_dt = _resample_trace_steps(steps, 0.1)
    assert actual_dt == pytest.approx(0.1)
    assert len(resampled) == 3


def test_resample_trace_steps_coarsens_native_grid() -> None:
    """Resampling to a coarser dt selects every Nth frame and updates time_s."""
    trace = {
        "frames": [
            {
                "time_s": 0.0,
                "pedestrians": [{"id": 1, "position": [0.0, 0.0], "velocity": [1.0, 0.0]}],
            },
            {
                "time_s": 0.1,
                "pedestrians": [{"id": 1, "position": [0.1, 0.0], "velocity": [1.0, 0.0]}],
            },
            {
                "time_s": 0.2,
                "pedestrians": [{"id": 1, "position": [0.2, 0.0], "velocity": [1.0, 0.0]}],
            },
            {
                "time_s": 0.3,
                "pedestrians": [{"id": 1, "position": [0.3, 0.0], "velocity": [1.0, 0.0]}],
            },
            {
                "time_s": 0.4,
                "pedestrians": [{"id": 1, "position": [0.4, 0.0], "velocity": [1.0, 0.0]}],
            },
        ]
    }
    steps = _extract_trace_steps(trace)
    resampled, actual_dt = _resample_trace_steps(steps, 0.2)
    assert actual_dt == pytest.approx(0.2)
    assert len(resampled) == 3
    assert resampled[0]["time_s"] == pytest.approx(0.0)
    assert resampled[1]["time_s"] == pytest.approx(0.2)
    assert resampled[2]["time_s"] == pytest.approx(0.4)


def test_cells_for_candidate_cover_full_ladder() -> None:
    """One trace candidate generates a cell for every horizon x dt combination."""
    candidate = TRACE_CANDIDATES[0]
    cells = _cells_for_candidate(candidate)
    assert len(cells) == len(HORIZON_LADDER_S) * len(DT_LADDER_S)
    for cell in cells:
        assert cell["horizon_s"] in HORIZON_LADDER_S
        assert cell["requested_dt_s"] in DT_LADDER_S


def test_evaluated_cells_have_provenance_and_runtime() -> None:
    """Evaluated cells carry provenance, runtime, and memory proxies."""
    candidate = next(
        c
        for c in TRACE_CANDIDATES
        if c["family"] == "corridor_interaction" and c["label"] == "default_social_force"
    )
    cell = _evaluate_ablation_cell(
        candidate,
        horizon_s=0.5,
        requested_dt_s=0.1,
    )
    assert cell["status"] == "evaluated"
    assert cell["family"] == candidate["family"]
    assert cell["label"] == candidate["label"]
    assert cell["trace_path"] == candidate["path"]
    assert "runtime_s" in cell
    assert "memory_proxy" in cell
    assert "artifact_size_bytes" in cell
    assert cell["metrics"]["forecast_evaluable_samples"] > 0


def test_horizon_longer_than_trace_is_unavailable() -> None:
    """A horizon beyond the resampled trace length is marked unavailable."""
    candidate = next(c for c in TRACE_CANDIDATES if c["family"] == "signalized_crossing")
    cell = _evaluate_ablation_cell(
        candidate,
        horizon_s=3.0,
        requested_dt_s=0.1,
    )
    assert cell["status"] == "horizon_longer_than_trace"
    assert "limitation" in cell


def test_zero_motion_trace_is_limited_not_error() -> None:
    """A trace without pedestrian motion is recorded as a limitation."""
    candidate = next(c for c in TRACE_CANDIDATES if c["family"] == "bottleneck")
    cell = _evaluate_ablation_cell(
        candidate,
        horizon_s=0.5,
        requested_dt_s=0.1,
    )
    assert cell["status"] == "limited_no_pedestrian_motion"
    assert "limitation" in cell


def test_summary_rows_aggregate_by_horizon_dt() -> None:
    """Summary rows roll cells up by (horizon, dt_s)."""
    candidate = next(
        c
        for c in TRACE_CANDIDATES
        if c["family"] == "corridor_interaction" and c["label"] == "default_social_force"
    )
    cells = _cells_for_candidate(candidate)
    summary = _build_summary_rows(cells)
    assert summary
    for row in summary:
        assert "horizon_s" in row
        assert "dt_s" in row
        assert "evaluated_cells" in row
        assert "total_cells" in row


def test_preset_recommendations_include_short_medium_long() -> None:
    """Preset list contains short, medium, and long entries with statuses."""
    cells = [cell for candidate in TRACE_CANDIDATES for cell in _cells_for_candidate(candidate)]
    presets = _preset_recommendations(cells)
    names = [p["preset"] for p in presets]
    assert "short" in names
    assert "medium" in names
    assert "long" in names


def test_build_ablation_report_has_required_fields() -> None:
    """The full report contains the expected schema and provenance fields."""
    report = build_ablation_report(issue=2837, generated_at_utc="2026-06-15T00:00:00+00:00")
    assert report["issue"] == 2837
    assert report["schema_version"] == "HorizonTimestepAblation.v1"
    assert "analysis_only_not_navigation_evidence" in report["claim_boundary"]
    assert set(report.keys()) >= {
        "issue",
        "schema_version",
        "claim_boundary",
        "reproducibility",
        "ablation_rows",
        "summary_rows",
        "preset_recommendations",
        "missing_families",
    }
    repro = report["reproducibility"]
    assert repro["issue"] == 2837
    assert repro["horizon_ladder_s"] == list(HORIZON_LADDER_S)
    assert repro["dt_ladder_s"] == list(DT_LADDER_S)
    assert repro["command"]


def test_build_ablation_report_distinguishes_evaluated_and_unavailable() -> None:
    """The report contains both evaluated and unavailable cells."""
    report = build_ablation_report(issue=2837, generated_at_utc="2026-06-15T00:00:00+00:00")
    statuses = {r["status"] for r in report["ablation_rows"]}
    assert "evaluated" in statuses
    assert statuses.intersection({"horizon_longer_than_trace", "limited_no_pedestrian_motion"})


def test_markdown_report_is_generated(tmp_path: pathlib.Path) -> None:
    """Markdown generation produces a file containing the claim boundary."""
    report = build_ablation_report(issue=2837, generated_at_utc="2026-06-15T00:00:00+00:00")
    md = _mod._generate_markdown(
        report["ablation_rows"],
        report["summary_rows"],
        report["preset_recommendations"],
        report["reproducibility"],
    )
    assert "Horizon and Timestep Ablation Report" in md
    assert "analysis_only_not_navigation_evidence" in md
    assert "Preset Recommendations" in md
    assert "Missing Trace Families" in md


def test_missing_families_are_documented() -> None:
    """Missing fixture families are listed with reasons."""
    assert MISSING_FAMILIES
    for entry in MISSING_FAMILIES:
        assert "family" in entry
        assert "reason" in entry


def test_artifact_size_bytes_for_missing_file() -> None:
    """Artifact size returns None for a missing trace file."""
    assert _artifact_size_bytes("nonexistent/path.json") is None


def test_artifact_size_bytes_for_existing_file() -> None:
    """Artifact size returns a positive integer for an existing trace file."""
    size = _artifact_size_bytes(TRACE_CANDIDATES[0]["path"])
    assert isinstance(size, int)
    assert size > 0
