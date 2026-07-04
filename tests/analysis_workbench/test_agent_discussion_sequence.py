"""Unit tests for the agent discussion workflow (issue #1646 child)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
)
from robot_sf.analysis_workbench.trace_annotation import (
    TraceAnnotationSetValidationError,
    load_trace_annotation_set,
)
from scripts.analysis.agent_discussion_sequence_issue_1646 import (
    generate_agent_discussion,
    main,
    write_agent_discussion,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "analysis_workbench"
TRACE_FIXTURE_PATH = (
    FIXTURE_DIR / "simulation_trace_export_v1" / "planner_sanity_open_episode_0000.json"
)
ANNOTATION_FIXTURE_PATH = (
    FIXTURE_DIR / "trace_annotation_set_v1" / "issue_1962_planner_sanity_open_annotations.json"
)


def test_generate_agent_discussion_valid() -> None:
    """Validate generating debate Markdown with a valid trace and annotations."""
    trace = load_simulation_trace_export(TRACE_FIXTURE_PATH)
    annotation_set = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH)

    markdown = generate_agent_discussion(
        trace,
        trace_path=TRACE_FIXTURE_PATH,
        annotation_set=annotation_set,
        annotation_path=ANNOTATION_FIXTURE_PATH,
        frame_start=1,
        frame_end=2,
    )

    assert "# Agent Discussion Report" in markdown
    assert "ObserverAgent" in markdown
    assert "TheoristAgent" in markdown
    assert "Empirical Takeaways" in markdown
    assert "Interpretive Takeaways" in markdown
    assert "issue_1962_step_1_to_2_goal_action_observed" in markdown
    assert "The robot stays nearly stationary" in markdown


def test_generate_agent_discussion_no_annotations() -> None:
    """Validate generating debate Markdown without annotations."""
    trace = load_simulation_trace_export(TRACE_FIXTURE_PATH)

    markdown = generate_agent_discussion(
        trace,
        trace_path=TRACE_FIXTURE_PATH,
        frame_start=1,
        frame_end=2,
    )

    assert "# Agent Discussion Report" in markdown
    assert "No overlapping annotations found" in markdown


def test_invalid_frame_ranges() -> None:
    """Assert invalid frame range raises ValueError."""
    trace = load_simulation_trace_export(TRACE_FIXTURE_PATH)

    with pytest.raises(ValueError, match="outside trace step range"):
        generate_agent_discussion(
            trace,
            trace_path=TRACE_FIXTURE_PATH,
            frame_start=1000,
            frame_end=2000,
        )

    with pytest.raises(ValueError, match="must be <= frame_end"):
        generate_agent_discussion(
            trace,
            trace_path=TRACE_FIXTURE_PATH,
            frame_start=3,
            frame_end=2,
        )


def test_write_agent_discussion_rejects_output_dependency(tmp_path: Path) -> None:
    """Assert output paths must not be used as tracked source fixtures."""
    out_file = tmp_path / "debate.md"

    # Try referencing an output folder in the trace path
    fake_output_trace = tmp_path / "output" / "trace.json"
    fake_output_trace.parent.mkdir(parents=True, exist_ok=True)
    fake_output_trace.touch()

    with pytest.raises(ValueError, match="must be a tracked fixture"):
        write_agent_discussion(
            trace_path=fake_output_trace,
            output=out_file,
        )


def test_validate_annotation_matches_trace_mismatch(tmp_path: Path) -> None:
    """Assert mismatched annotation set and trace raises error."""
    out_file = tmp_path / "debate.md"
    annotation_set = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH)

    # Modify trace ID to force mismatch
    payload = annotation_set.to_dict()
    payload["timeline"]["trace_id"] = "mismatching-trace-id"

    mismatched_ann_file = ANNOTATION_FIXTURE_PATH.parent / "temp_mismatched_annotations.json"
    mismatched_ann_file.write_text(json.dumps(payload), encoding="utf-8")

    try:
        with pytest.raises(TraceAnnotationSetValidationError, match="expected referenced trace_id"):
            write_agent_discussion(
                trace_path=TRACE_FIXTURE_PATH,
                annotation_path=mismatched_ann_file,
                output=out_file,
            )
    finally:
        if mismatched_ann_file.exists():
            mismatched_ann_file.unlink()


def test_main_cli_execution(tmp_path: Path) -> None:
    """Assert CLI execution runs successfully."""
    out_file = tmp_path / "cli_debate.md"
    args = [
        "--trace",
        str(TRACE_FIXTURE_PATH),
        "--annotations",
        str(ANNOTATION_FIXTURE_PATH),
        "--frame-start",
        "1",
        "--frame-end",
        "3",
        "--output",
        str(out_file),
    ]

    exit_code = main(args)
    assert exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# Agent Discussion Report" in content
