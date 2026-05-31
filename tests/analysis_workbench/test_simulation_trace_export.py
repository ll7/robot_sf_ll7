"""Contract tests for analysis-workbench simulation trace exports."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExportValidationError,
    load_simulation_trace_export,
    simulation_trace_export_from_dict,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)
MATERIALIZED_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "planner_sanity_open_episode_0000.json"
)


def test_load_minimal_simulation_trace_export_fixture() -> None:
    """The tiny fixture should validate as analysis input, not benchmark evidence."""

    trace = load_simulation_trace_export(FIXTURE_PATH)

    assert trace.schema_version == SIMULATION_TRACE_EXPORT_SCHEMA_VERSION
    assert trace.trace_id == "fixture_trace_001"
    assert trace.source.scenario_id == "classic_bottleneck_medium"
    assert trace.evidence_boundary == "analysis_workbench_only"
    assert [frame.step for frame in trace.frames] == [0, 1]
    assert trace.frames[0].planner["event"] == "start"


def test_simulation_trace_export_rejects_benchmark_evidence_boundary() -> None:
    """Trace exports should fail closed if presented as benchmark evidence."""

    payload = load_simulation_trace_export(FIXTURE_PATH).to_dict()
    payload["evidence_boundary"] = "benchmark_evidence"

    with pytest.raises(SimulationTraceExportValidationError, match="/evidence_boundary"):
        simulation_trace_export_from_dict(payload, source=FIXTURE_PATH)


def test_simulation_trace_export_requires_monotonic_steps() -> None:
    """Workbench traces should expose ordered frames for deterministic playback."""

    payload = load_simulation_trace_export(FIXTURE_PATH).to_dict()
    payload["frames"][1]["step"] = 0

    with pytest.raises(SimulationTraceExportValidationError, match="/frames/1/step"):
        simulation_trace_export_from_dict(payload, source=FIXTURE_PATH)


def test_simulation_trace_export_requires_monotonic_time() -> None:
    """Workbench traces should expose increasing timestamps for playback."""

    payload = load_simulation_trace_export(FIXTURE_PATH).to_dict()
    payload["frames"][1]["time_s"] = 0.0

    with pytest.raises(SimulationTraceExportValidationError, match="/frames/1/time_s"):
        simulation_trace_export_from_dict(payload, source=FIXTURE_PATH)


def test_simulation_trace_export_schema_errors_keep_numeric_frame_order() -> None:
    """Schema errors should sort array indices numerically, not lexicographically."""

    payload = load_simulation_trace_export(FIXTURE_PATH).to_dict()
    template_frame = payload["frames"][-1]
    while len(payload["frames"]) <= 10:
        next_frame = dict(template_frame)
        next_frame["robot"] = dict(template_frame["robot"])
        next_frame["pedestrians"] = [
            dict(pedestrian) for pedestrian in template_frame["pedestrians"]
        ]
        next_frame["planner"] = dict(template_frame["planner"])
        next_frame["planner"]["selected_action"] = dict(
            template_frame["planner"]["selected_action"]
        )
        next_frame["step"] = len(payload["frames"])
        next_frame["time_s"] = float(len(payload["frames"]))
        payload["frames"].append(next_frame)
    payload["frames"][2]["unexpected"] = True
    payload["frames"][10]["unexpected"] = True

    with pytest.raises(SimulationTraceExportValidationError) as exc_info:
        simulation_trace_export_from_dict(payload, source=FIXTURE_PATH)

    errors = exc_info.value.errors
    assert next(
        index for index, error in enumerate(errors) if error.startswith("/frames/2:")
    ) < next(index for index, error in enumerate(errors) if error.startswith("/frames/10:"))


def test_materialized_trace_fixture_has_source_provenance_and_event_ids() -> None:
    """Materialized campaign slice keeps provenance and stable event identifiers."""

    trace = load_simulation_trace_export(MATERIALIZED_FIXTURE_PATH)

    assert trace.trace_id == "planner_sanity_open-ep0-seed7-source-f6b33f7c"
    assert trace.source.scenario_id == "planner_sanity_open"
    assert trace.source.planner_id == "trace_fixture_gen"
    assert trace.source.seed == 7
    assert (
        "scripts.tools.build_simulation_trace_export from "
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000.jsonl "
        "source_sha256:f6b33f7c4a67" in trace.source.generated_by
    )
    assert [frame.step for frame in trace.frames] == [1, 2, 3]
    assert [frame.planner["event"] for frame in trace.frames] == ["step", "step", "step"]
    assert [frame.planner["event_id"] for frame in trace.frames] == [
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0000",
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0001",
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0002",
    ]
