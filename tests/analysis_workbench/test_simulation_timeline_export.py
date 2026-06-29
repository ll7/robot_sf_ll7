"""Contract tests for ``simulation_timeline.v1`` trace slices."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.simulation_timeline import (
    SIMULATION_TIMELINE_SCHEMA_FILE,
    SIMULATION_TIMELINE_SCHEMA_VERSION,
    build_simulation_timeline,
    load_simulation_timeline_schema,
    validate_simulation_timeline,
)

TRACE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)
MATERIALIZED_TRACE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "planner_sanity_open_episode_0000.json"
)
TIMELINE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_timeline_v1"
    / "planner_sanity_open_episode_0000_timeline.json"
)
CLI_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "export_trace_timeline_issue_1646.py"
)


def test_simulation_timeline_schema_accepts_converted_trace_fixture() -> None:
    """A renderer-neutral timeline should validate from an existing trace export fixture."""

    timeline = build_simulation_timeline(TRACE_FIXTURE_PATH)

    validate_simulation_timeline(timeline)
    assert timeline["schema_version"] == SIMULATION_TIMELINE_SCHEMA_VERSION
    assert timeline["source_trace"]["schema_version"] == "simulation_trace_export.v1"
    assert timeline["source_trace"]["trace_id"] == "fixture_trace_001"
    assert [frame["frame_index"] for frame in timeline["frames"]] == [0, 1]
    assert timeline["frames"][0]["state"]["planner_decision"] == {
        "selected_action": {"linear_velocity": 0.1, "angular_velocity": 0.0},
        "event": "start",
    }
    assert timeline["frames"][0]["state"]["collision"] == {"status": "unknown"}
    assert timeline["frames"][0]["state"]["pedestrians"][0]["id"] == "ped_1"
    assert timeline["events"][0]["event_type"] == "start"


def test_materialized_timeline_fixture_regenerates_from_trace_export() -> None:
    """The durable timeline artifact should regenerate from its tracked trace source."""

    timeline = build_simulation_timeline(MATERIALIZED_TRACE_FIXTURE_PATH)
    materialized = json.loads(TIMELINE_FIXTURE_PATH.read_text(encoding="utf-8"))

    validate_simulation_timeline(materialized)
    assert timeline == materialized
    assert materialized["source_trace"]["trace_id"] == (
        "planner_sanity_open-ep0-seed7-source-aae87cf6"
    )
    assert [frame["step"] for frame in materialized["frames"]] == [1, 2, 3]


def test_simulation_timeline_preserves_explicit_empty_optional_slots(
    tmp_path: Path,
) -> None:
    """Explicit empty planner mappings should stay explicit instead of becoming null."""

    trace_payload = json.loads(TRACE_FIXTURE_PATH.read_text(encoding="utf-8"))
    trace_payload["frames"][0]["planner"]["policy"] = {}
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(trace_payload), encoding="utf-8")

    timeline = build_simulation_timeline(source)

    assert timeline["frames"][0]["state"]["policy"] == {}
    assert timeline["frames"][1]["state"]["policy"] is None


def test_simulation_timeline_normalizes_blank_event_ids(tmp_path: Path) -> None:
    """Whitespace-only source event IDs should use deterministic fallback IDs."""

    trace_payload = json.loads(TRACE_FIXTURE_PATH.read_text(encoding="utf-8"))
    trace_payload["frames"][0]["planner"]["event_id"] = "  "
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(trace_payload), encoding="utf-8")

    timeline = build_simulation_timeline(source)

    assert timeline["events"][0]["event_id"] == "frame-0000-start"


def test_simulation_timeline_schema_rejects_missing_required_shape() -> None:
    """The schema should fail closed when required top-level fields are absent."""

    schema = load_simulation_timeline_schema()
    errors = sorted(
        Draft202012Validator(schema).iter_errors(
            {
                "schema_version": SIMULATION_TIMELINE_SCHEMA_VERSION,
                "frames": [],
                "events": [],
            }
        ),
        key=lambda error: list(error.absolute_path),
    )

    assert any(error.validator == "required" and "timeline_id" in error.message for error in errors)
    assert any(
        error.validator == "required" and "source_trace" in error.message for error in errors
    )


def test_simulation_timeline_schema_rejects_wrong_source_schema_version() -> None:
    """Timeline provenance should fail closed for non-trace-export sources."""

    schema = load_simulation_timeline_schema()
    timeline = build_simulation_timeline(TRACE_FIXTURE_PATH)
    timeline["source_trace"]["schema_version"] = "other_trace.v1"

    errors = list(Draft202012Validator(schema).iter_errors(timeline))

    assert any(
        list(error.absolute_path) == ["source_trace", "schema_version"]
        and error.validator == "const"
        for error in errors
    )


def test_export_trace_timeline_cli_writes_valid_artifact(tmp_path: Path) -> None:
    """The CLI should write a schema-valid timeline artifact from a trace export."""

    output_path = tmp_path / "timeline.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--input",
            str(TRACE_FIXTURE_PATH),
            "--out",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    schema = json.loads(SIMULATION_TIMELINE_SCHEMA_FILE.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)
    assert payload["timeline_id"] == "fixture_trace_001-timeline"
    assert payload["events"] == [
        {
            "event_id": "frame-0000-start",
            "event_type": "start",
            "frame_index": 0,
            "step": 0,
            "time_s": 0.0,
        },
        {
            "event_id": "frame-0001-advance",
            "event_type": "advance",
            "frame_index": 1,
            "step": 1,
            "time_s": 0.1,
        },
    ]
