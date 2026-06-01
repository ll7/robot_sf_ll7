"""Tests for static simulation trace report rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExportValidationError,
    load_simulation_trace_export,
)
from robot_sf.analysis_workbench.trace_annotation import (
    TraceAnnotationSetValidationError,
    load_trace_annotation_set,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)
ANNOTATION_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "trace_annotation_set_v1"
    / "issue_1962_planner_sanity_open_annotations.json"
)


def test_trace_report_renderer_writes_markdown_summary_for_minimal_fixture(
    tmp_path: Path,
) -> None:
    """The minimal trace fixture should render to a readable Markdown report."""

    from scripts.tools.render_trace_report import write_trace_report

    output = tmp_path / "report.md"

    assert write_trace_report(trace_path=FIXTURE_PATH, output=output) == output

    markdown = output.read_text(encoding="utf-8")
    assert "# Simulation Trace Report" in markdown
    assert "analysis/debug artifact" in markdown
    assert "not benchmark evidence" in markdown
    assert "| trace_id | fixture_trace_001 |" in markdown
    assert "| scenario_id | classic_bottleneck_medium |" in markdown
    assert "| planner_id | hybrid_rule_v0_minimal |" in markdown
    assert "| frames | 2 |" in markdown
    assert "| start | 1 |" in markdown
    assert "| advance | 1 |" in markdown
    assert "| 0 | 0.000 | start | 0.100, 0.000 |" in markdown
    assert "ped_1 @ (1.000, 0.500)" in markdown


def test_trace_report_renderer_surfaces_existing_planner_annotations(
    tmp_path: Path,
) -> None:
    """Planner annotation fields already present in the trace should be visible."""

    from scripts.tools.render_trace_report import write_trace_report

    payload = load_simulation_trace_export(FIXTURE_PATH).to_dict()
    payload["frames"][1]["planner"]["annotations"] = {
        "clearance": 0.42,
        "note": "fixture near-pass",
    }
    trace_path = tmp_path / "annotated_trace.json"
    output = tmp_path / "report.md"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    write_trace_report(trace_path=trace_path, output=output)

    markdown = output.read_text(encoding="utf-8")
    assert "## Notable Annotations" in markdown
    assert "clearance: 0.420" in markdown
    assert "note: fixture near-pass" in markdown


def test_trace_report_renderer_uses_existing_trace_validation_path(
    tmp_path: Path,
) -> None:
    """Invalid traces should fail through the existing loader/validator."""

    from scripts.tools.render_trace_report import write_trace_report

    payload = load_simulation_trace_export(FIXTURE_PATH).to_dict()
    payload["evidence_boundary"] = "benchmark_evidence"
    trace_path = tmp_path / "invalid_trace.json"
    output = tmp_path / "report.md"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SimulationTraceExportValidationError, match="/evidence_boundary"):
        write_trace_report(trace_path=trace_path, output=output)

    assert not output.exists()


def test_trace_report_renderer_includes_qualitative_annotations(
    tmp_path: Path,
) -> None:
    """An optional annotation set should render a compact qualitative-annotations section."""

    from scripts.tools.render_trace_report import write_trace_report

    output = tmp_path / "annotated_report.md"
    trace_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "analysis_workbench"
        / "simulation_trace_export_v1"
        / "planner_sanity_open_episode_0000.json"
    )

    assert (
        write_trace_report(
            trace_path=trace_path,
            output=output,
            annotation_path=ANNOTATION_FIXTURE_PATH,
        )
        == output
    )

    markdown = output.read_text(encoding="utf-8")
    assert "## Qualitative Trace Annotations" in markdown
    assert "not benchmark evidence" in markdown
    assert "planner_action" in markdown
    assert "issue_1962_step_1_to_2_goal_action_observed" in markdown
    assert "1-2" in markdown
    assert "robot:robot" in markdown
    assert "none" in markdown


def test_trace_report_renderer_rejects_trace_id_mismatch(
    tmp_path: Path,
) -> None:
    """Annotation sets that reference a different trace_id should fail closed."""

    from scripts.tools.render_trace_report import write_trace_report

    trace_data = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "analysis_workbench"
            / "simulation_trace_export_v1"
            / "planner_sanity_open_episode_0000.json"
        ).read_text(encoding="utf-8")
    )
    fixture_dir = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "analysis_workbench"
        / "trace_annotation_set_v1"
    )
    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["timeline"]["trace_id"] = "different_trace_id"
    trace_data["trace_id"] = "different_trace_id"
    annotation_dir = tmp_path / fixture_dir.name
    annotation_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = tmp_path / "simulation_trace_export_v1"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "planner_sanity_open_episode_0000.json").write_text(
        json.dumps(trace_data), encoding="utf-8"
    )
    mismatched_path = annotation_dir / "mismatched_annotations.json"
    mismatched_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceAnnotationSetValidationError, match="expected report trace_id"):
        write_trace_report(
            trace_path=FIXTURE_PATH,
            output=tmp_path / "report.md",
            annotation_path=mismatched_path,
        )


def test_trace_report_renderer_propagates_annotation_validation_errors(
    tmp_path: Path,
) -> None:
    """Invalid annotation fixtures should raise TraceAnnotationSetValidationError."""

    from scripts.tools.render_trace_report import write_trace_report

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["annotations"][0]["category"] = "storytelling"
    trace_data = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "analysis_workbench"
            / "simulation_trace_export_v1"
            / "planner_sanity_open_episode_0000.json"
        ).read_text(encoding="utf-8")
    )
    annotation_dir = tmp_path / "trace_annotation_set_v1"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = tmp_path / "simulation_trace_export_v1"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "planner_sanity_open_episode_0000.json").write_text(
        json.dumps(trace_data), encoding="utf-8"
    )
    invalid_path = annotation_dir / "invalid_annotations.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceAnnotationSetValidationError, match="category"):
        write_trace_report(
            trace_path=trace_dir / "planner_sanity_open_episode_0000.json",
            output=tmp_path / "report.md",
            annotation_path=invalid_path,
        )
