"""Tests for the issue #2227 AMMV contrastive mechanism panel builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "build_ammv_mechanism_panel_issue_2227.py"
)


def _load_module():
    """Import the script module by path."""
    spec = importlib.util.spec_from_file_location(
        "build_ammv_mechanism_panel_issue_2227", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def panel_module():
    """Load the panel builder module once per test module."""
    return _load_module()


@pytest.fixture(scope="module")
def panel_run(panel_module, tmp_path_factory):
    """Run the full panel build once and return (summary, output_dir)."""
    output_dir = tmp_path_factory.mktemp("ammv_panel")
    summary = panel_module.run(output_dir)
    return summary, output_dir


def test_both_traces_validate_against_schema(panel_run):
    """Both exported traces load cleanly against simulation_trace_export.v1."""
    summary, _ = panel_run
    control = load_simulation_trace_export(Path(summary["control_trace"]))
    intervention = load_simulation_trace_export(Path(summary["intervention_trace"]))
    assert control.schema_version == "simulation_trace_export.v1"
    assert intervention.schema_version == "simulation_trace_export.v1"
    assert len(control.frames) == summary["steps"]
    assert len(intervention.frames) == summary["steps"]


def test_ammv_force_contrast(panel_run):
    """AMMV-on arm has nonzero force; AMMV-off arm has exactly zero force."""
    summary, _ = panel_run
    assert summary["ammv_off_max_force_magnitude"] == 0.0
    assert summary["ammv_on_max_force_magnitude"] > 0.0


def test_per_frame_force_metadata(panel_run):
    """The off arm carries zero AMMV force in every frame; the on arm carries some."""
    summary, _ = panel_run
    off = load_simulation_trace_export(Path(summary["control_trace"]))
    on = load_simulation_trace_export(Path(summary["intervention_trace"]))
    off_forces = [f.planner["ammv_force_magnitude"] for f in off.frames]
    on_forces = [f.planner["ammv_force_magnitude"] for f in on.frames]
    assert all(value == 0.0 for value in off_forces)
    assert any(value > 0.0 for value in on_forces)


def test_panel_artifacts_produced(panel_run):
    """A PNG+PDF panel artifact pair is produced for each arm."""
    summary, _ = panel_run
    assert summary["panel_pngs"], "expected at least one panel PNG"
    assert summary["panel_pdfs"], "expected at least one panel PDF"
    for png in summary["panel_pngs"]:
        assert Path(png).is_file()
        assert Path(png).stat().st_size > 0
    for pdf in summary["panel_pdfs"]:
        assert Path(pdf).is_file()
        assert Path(pdf).stat().st_size > 0


def test_trajectory_changed(panel_run):
    """The two arms produce divergent robot trajectories at the planner level."""
    summary, _ = panel_run
    assert summary["trajectory_delta"]["final_position_distance_m"] > 1e-6


def test_determinism(panel_module, tmp_path):
    """Re-running with the same seed/scenario reproduces force magnitudes/trajectory."""
    first = panel_module.run(tmp_path / "run_a")
    second = panel_module.run(tmp_path / "run_b")
    assert first["ammv_off_max_force_magnitude"] == second["ammv_off_max_force_magnitude"]
    assert first["ammv_on_max_force_magnitude"] == second["ammv_on_max_force_magnitude"]
    assert (
        first["trajectory_delta"]["final_position_distance_m"]
        == second["trajectory_delta"]["final_position_distance_m"]
    )


def test_claim_boundary_is_diagnostic_only(panel_run):
    """The summary records a diagnostic-only, non-paper-grade claim boundary."""
    summary, _ = panel_run
    assert summary["claim_boundary"] == "diagnostic_only"
    assert summary["paper_grade"] is False
