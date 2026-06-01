"""Contract tests for analysis-workbench trace annotation sets."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.analysis_workbench.trace_annotation import (
    TRACE_ANNOTATION_SET_SCHEMA_VERSION,
    TraceAnnotationSetValidationError,
    load_trace_annotation_set,
    trace_annotation_set_from_dict,
)

ANNOTATION_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "trace_annotation_set_v1"
    / "issue_1962_planner_sanity_open_annotations.json"
)


def test_load_trace_annotation_fixture() -> None:
    """The annotation fixture should validate against its referenced trace."""

    annotation_set = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH)

    assert annotation_set.schema_version == TRACE_ANNOTATION_SET_SCHEMA_VERSION
    assert annotation_set.annotation_set_id == "issue_1962_planner_sanity_open_annotations"
    assert annotation_set.timeline.schema_version == "simulation_trace_export.v1"
    assert annotation_set.provenance.evidence_boundary == ("analysis_workbench_qualitative_only")
    assert [annotation.evidence_type for annotation in annotation_set.annotations] == [
        "observed",
        "hypothesis",
        "commentary",
    ]
    assert annotation_set.annotations[0].anchor.frame_start == 1
    assert annotation_set.annotations[0].anchor.frame_end == 2
    assert annotation_set.annotations[0].anchor.entities[0].id == "robot"


def test_trace_annotation_rejects_missing_trace_fixture() -> None:
    """Annotation sets should fail closed when their source timeline is missing."""

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["timeline"]["path"] = "../simulation_trace_export_v1/missing.json"

    with pytest.raises(TraceAnnotationSetValidationError, match="does not exist"):
        trace_annotation_set_from_dict(payload, source=ANNOTATION_FIXTURE_PATH)


def test_trace_annotation_rejects_invalid_frame_range_order() -> None:
    """Frame ranges should be ordered inclusively by trace step."""

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["annotations"][0]["anchor"]["frame_start"] = 3
    payload["annotations"][0]["anchor"]["frame_end"] = 2

    with pytest.raises(TraceAnnotationSetValidationError, match="expected <= frame_end"):
        trace_annotation_set_from_dict(payload, source=ANNOTATION_FIXTURE_PATH)


def test_trace_annotation_rejects_frame_range_outside_trace() -> None:
    """Frame anchors should point at steps present in the referenced trace."""

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["annotations"][0]["anchor"]["frame_start"] = 0

    with pytest.raises(TraceAnnotationSetValidationError, match="outside referenced trace"):
        trace_annotation_set_from_dict(payload, source=ANNOTATION_FIXTURE_PATH)


def test_trace_annotation_rejects_output_timeline_dependency() -> None:
    """Annotation fixtures should not depend on worktree-local generated output."""

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["timeline"]["path"] = "output/debug/generated_trace.json"

    with pytest.raises(TraceAnnotationSetValidationError, match="not generated output"):
        trace_annotation_set_from_dict(payload, source=ANNOTATION_FIXTURE_PATH)


def test_trace_annotation_rejects_unknown_event_id() -> None:
    """Event references should resolve against the referenced timeline."""

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["annotations"][0]["anchor"]["event_ids"] = ["missing-event"]

    with pytest.raises(TraceAnnotationSetValidationError, match="unknown referenced trace"):
        trace_annotation_set_from_dict(payload, source=ANNOTATION_FIXTURE_PATH)


def test_trace_annotation_rejects_unsupported_category() -> None:
    """Categories are intentionally finite so downstream views can remain stable."""

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["annotations"][0]["category"] = "storytelling"

    with pytest.raises(TraceAnnotationSetValidationError, match="/annotations/0/category"):
        trace_annotation_set_from_dict(payload, source=ANNOTATION_FIXTURE_PATH)


def test_trace_annotation_rejects_unsupported_provenance() -> None:
    """Generated-output provenance should not masquerade as a tracked fixture."""

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["provenance"]["kind"] = "output_run"

    with pytest.raises(TraceAnnotationSetValidationError, match="/provenance/kind"):
        trace_annotation_set_from_dict(payload, source=ANNOTATION_FIXTURE_PATH)


def test_trace_annotation_rejects_benchmark_evidence_boundary() -> None:
    """Qualitative annotations should never be validated as benchmark evidence."""

    payload = load_trace_annotation_set(ANNOTATION_FIXTURE_PATH).to_dict()
    payload["provenance"]["evidence_boundary"] = "benchmark_evidence"

    with pytest.raises(
        TraceAnnotationSetValidationError,
        match="/provenance/evidence_boundary",
    ):
        trace_annotation_set_from_dict(payload, source=ANNOTATION_FIXTURE_PATH)
