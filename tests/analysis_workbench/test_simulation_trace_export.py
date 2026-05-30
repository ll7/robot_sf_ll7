"""Contract tests for analysis-workbench simulation trace exports."""

from __future__ import annotations

from copy import deepcopy
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
    invalid = deepcopy(payload)
    invalid["frames"][1]["step"] = 0

    with pytest.raises(SimulationTraceExportValidationError, match="/frames/1/step"):
        simulation_trace_export_from_dict(invalid, source=FIXTURE_PATH)
