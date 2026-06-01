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
ANNOTATED_TRACE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "planner_sanity_open_episode_0000.json"
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


def test_trace_report_renderer_writes_annotation_review_report(tmp_path: Path) -> None:
    """Trace and annotation fixtures should render a cited qualitative review report."""

    from scripts.tools.render_trace_report import write_trace_report

    output = tmp_path / "annotation_report.md"

    write_trace_report(
        trace_path=ANNOTATED_TRACE_FIXTURE_PATH,
        annotation_path=ANNOTATION_FIXTURE_PATH,
        output=output,
    )

    markdown = output.read_text(encoding="utf-8")
    assert "## Source Fixtures" in markdown
    assert "| annotation_set_id | issue_1962_planner_sanity_open_annotations |" in markdown
    assert "| annotation_source_issue | #1962 |" in markdown
    assert "## Annotation Review" in markdown
    assert "diagnostic-only and not benchmark evidence" in markdown
    assert (
        "| issue_1962_step_1_to_2_goal_action_observed | observed | planner_action | 1-2 |"
        in markdown
    )
    assert "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0000" in markdown
    assert "| issue_1962_step_3_heading_hypothesis | hypothesis | interaction | 3-3 |" in markdown
    assert "| issue_1962_fixture_scope_commentary | commentary | data_quality | 1-3 |" in markdown
    assert "robot:robot" in markdown


def test_trace_report_renderer_rejects_mismatched_annotation_trace(
    tmp_path: Path,
) -> None:
    """The supplied trace must match the trace referenced by the annotation fixture."""

    from scripts.tools.render_trace_report import write_trace_report

    output = tmp_path / "report.md"

    with pytest.raises(TraceAnnotationSetValidationError, match="/timeline/trace_id"):
        write_trace_report(
            trace_path=FIXTURE_PATH,
            annotation_path=ANNOTATION_FIXTURE_PATH,
            output=output,
        )

    assert not output.exists()


def test_trace_report_renderer_rejects_invalid_annotation_references(
    tmp_path: Path,
) -> None:
    """Invalid annotation-to-trace references should fail closed before output writes."""

    from scripts.tools.render_trace_report import write_trace_report

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["annotations"][0]["anchor"]["event_ids"] = ["missing-event"]
    trace_dir = tmp_path / "simulation_trace_export_v1"
    annotation_dir = tmp_path / "trace_annotation_set_v1"
    trace_dir.mkdir()
    annotation_dir.mkdir()
    trace_copy = trace_dir / ANNOTATED_TRACE_FIXTURE_PATH.name
    trace_copy.write_text(ANNOTATED_TRACE_FIXTURE_PATH.read_text(encoding="utf-8"))
    annotation_path = annotation_dir / "bad_annotations.json"
    output = tmp_path / "report.md"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceAnnotationSetValidationError, match="unknown referenced trace"):
        write_trace_report(
            trace_path=trace_copy,
            annotation_path=annotation_path,
            output=output,
        )

    assert not output.exists()


def test_trace_report_renderer_rejects_output_only_annotation_sources(
    tmp_path: Path,
) -> None:
    """Annotation reports should not depend on generated output-only trace sources."""

    from scripts.tools.render_trace_report import write_trace_report

    output_trace = tmp_path / "output" / "trace.json"
    output_trace.parent.mkdir()
    output_trace.write_text(ANNOTATED_TRACE_FIXTURE_PATH.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="not generated output"):
        write_trace_report(
            trace_path=output_trace,
            annotation_path=ANNOTATION_FIXTURE_PATH,
            output=tmp_path / "report.md",
        )
