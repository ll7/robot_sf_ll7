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
from scripts.tools.build_simulation_trace_export import build_simulation_trace_export

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
SOURCE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "sources"
    / "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000.jsonl"
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

    assert trace.trace_id == "planner_sanity_open-ep0-seed7-source-aae87cf6"
    assert trace.source.scenario_id == "planner_sanity_open"
    assert trace.source.planner_id == "trace_fixture_gen"
    assert trace.source.seed == 7
    assert (
        "scripts.tools.build_simulation_trace_export from "
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000.jsonl "
        "source_sha256:aae87cf69fbf" in trace.source.generated_by
    )
    assert [frame.step for frame in trace.frames] == [1, 2, 3]
    assert [frame.planner["event"] for frame in trace.frames] == ["step", "step", "step"]
    assert [frame.planner["event_id"] for frame in trace.frames] == [
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0000",
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0001",
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0002",
    ]


def test_materialized_trace_fixture_regenerates_from_tracked_source() -> None:
    """The durable timeline fixture should be reproducible from a tracked source slice."""

    assert SOURCE_FIXTURE_PATH.exists()

    payload = build_simulation_trace_export(SOURCE_FIXTURE_PATH)
    materialized = load_simulation_trace_export(MATERIALIZED_FIXTURE_PATH).to_dict()

    assert payload == materialized


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write compact JSONL source records for trace-export builder tests."""
    import json

    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _step_record(
    step: int,
    *,
    timestep: object,
    x: float,
    y: float,
    action: object | None = None,
) -> dict[str, object]:
    """Return one minimal step record accepted by the trace-export builder."""
    state: dict[str, object] = {
        "robot_pose": [[x, y], 0.0],
        "timestep": timestep,
        "pedestrian_positions": [],
    }
    if action is not None:
        state["robot_action"] = action
    return {"event": "step", "step_idx": step, "state": state}


def test_build_trace_export_skips_non_step_records(tmp_path: Path) -> None:
    """Non-step events should not create duplicate playback frames."""
    source = tmp_path / "episode.jsonl"
    _write_jsonl(
        source,
        [
            _step_record(0, timestep=0.0, x=0.0, y=0.0),
            {
                "event": "entity_reset",
                "step_idx": 0,
                "state": {"robot_pose": [[0.0, 0.0], 0.0], "timestep": 0.0},
            },
            _step_record(1, timestep=1.0, x=1.0, y=0.0),
        ],
    )

    payload = build_simulation_trace_export(source)

    assert [frame["step"] for frame in payload["frames"]] == [0, 1]
    assert [frame["time_s"] for frame in payload["frames"]] == [0.0, 1.0]


def test_build_trace_export_uses_aggregate_simulation_step_trace(tmp_path: Path) -> None:
    """Aggregate benchmark rows can carry opt-in AMV trace frames for review packs."""

    source = tmp_path / "aggregate.jsonl"
    _write_jsonl(
        source,
        [
            {
                "episode_id": "s1--7--trace",
                "scenario_id": "s1",
                "seed": 7,
                "scenario_params": {"algo": "goal"},
                "algorithm_metadata": {
                    "algorithm": "goal",
                    "simulation_step_trace": {
                        "schema_version": "simulation-step-trace.v1",
                        "dt": 0.1,
                        "steps": [
                            {
                                "step": 0,
                                "time_s": 0.1,
                                "robot": {
                                    "position": [1.0, 2.0],
                                    "heading": 0.25,
                                    "velocity": [0.0, 0.0],
                                },
                                "pedestrians": [],
                                "planner": {
                                    "event": "step",
                                    "selected_action": {
                                        "linear_velocity": 0.0,
                                        "angular_velocity": 0.0,
                                    },
                                    "amv": {
                                        "requested_linear_m_s": 3.0,
                                        "requested_angular_rad_s": 2.0,
                                        "applied_linear_m_s": 0.0,
                                        "applied_angular_rad_s": 0.0,
                                        "command_clipped": False,
                                        "yaw_rate_saturated": False,
                                    },
                                },
                            }
                        ],
                    },
                },
            }
        ],
    )

    payload = build_simulation_trace_export(source)

    assert payload["source"]["scenario_id"] == "s1"
    assert payload["source"]["seed"] == 7
    assert payload["source"]["planner_id"] == "goal"
    assert payload["source"]["episode_id"] == "s1--7--trace"
    frame = payload["frames"][0]
    assert frame["time_s"] == pytest.approx(0.1)
    assert frame["robot"]["heading"] == pytest.approx(0.25)
    assert frame["planner"]["amv"]["command_clipped"] is False


def test_build_trace_export_rejects_non_object_metadata(tmp_path: Path) -> None:
    """Sidecar metadata must be an object so provenance fields are explicit."""
    source = tmp_path / "episode.jsonl"
    _write_jsonl(source, [_step_record(0, timestep=0.0, x=0.0, y=0.0)])
    source.with_name("episode.meta.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata file .* must contain a JSON object"):
        build_simulation_trace_export(source)


def test_build_trace_export_uses_pose_fallback_for_malformed_dict_action(
    tmp_path: Path,
) -> None:
    """Malformed dict actions should still use pose-derived motion."""
    source = tmp_path / "episode.jsonl"
    _write_jsonl(
        source,
        [
            _step_record(0, timestep=0.0, x=0.0, y=0.0),
            _step_record(
                1,
                timestep=1.0,
                x=2.0,
                y=0.0,
                action={"linear_velocity": "bad", "angular_velocity": None},
            ),
        ],
    )

    payload = build_simulation_trace_export(source)

    assert payload["frames"][1]["planner"]["selected_action"] == {
        "linear_velocity": 2.0,
        "angular_velocity": 0.0,
    }


def test_build_trace_export_wraps_invalid_timestep_error(tmp_path: Path) -> None:
    """Invalid fallback timestamps should report the affected step."""
    source = tmp_path / "episode.jsonl"
    _write_jsonl(source, [_step_record(0, timestep="not-a-time", x=0.0, y=0.0)])

    with pytest.raises(ValueError, match="invalid timestamp for step 0"):
        build_simulation_trace_export(source)
