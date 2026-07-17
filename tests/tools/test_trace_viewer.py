"""Tests for the optional-Rerun edge-case trace viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.tools import trace_viewer


@dataclass(frozen=True)
class _Archetype:
    """Small stand-in retaining constructor arguments for logging assertions."""

    kind: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _archetype(kind: str) -> Any:
    def construct(*args: Any, **kwargs: Any) -> _Archetype:
        return _Archetype(kind=kind, args=args, kwargs=kwargs)

    return construct


class _FakeRecording:
    """Capture the RecordingStream surface used by the viewer."""

    def __init__(self) -> None:
        self.logs: list[tuple[str, _Archetype, bool]] = []
        self.times: list[tuple[str, str, float | int]] = []
        self.blueprints: list[_Archetype] = []

    def set_time_seconds(self, timeline: str, value: float) -> None:
        self.times.append(("duration", timeline, value))

    def set_time_sequence(self, timeline: str, value: int) -> None:
        self.times.append(("sequence", timeline, value))

    def log(self, path: str, payload: _Archetype, *, static: bool = False) -> None:
        self.logs.append((path, payload, static))

    def send_blueprint(self, blueprint: _Archetype) -> None:
        self.blueprints.append(blueprint)


class _ModernRecording:
    """Capture the current Rerun ``set_time`` keyword surface."""

    def __init__(self) -> None:
        self.times: list[tuple[str, int | float | None, int | None]] = []

    def set_time(
        self,
        timeline: str,
        *,
        duration: int | float | None = None,
        sequence: int | None = None,
    ) -> None:
        self.times.append((timeline, duration, sequence))


class _SavedRecording:
    """Capture current Rerun save finalization order."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def flush(self, *, timeout_sec: float = 1e38) -> None:
        self.events.append(f"flush:{timeout_sec}")

    def disconnect(self) -> None:
        self.events.append("disconnect")


FAKE_RERUN = SimpleNamespace(
    LineStrips2D=_archetype("LineStrips2D"),
    Points2D=_archetype("Points2D"),
    Scalars=_archetype("Scalars"),
    SeriesLine=_archetype("SeriesLine"),
    TextLog=_archetype("TextLog"),
)
FAKE_BLUEPRINT = SimpleNamespace(
    Blueprint=_archetype("Blueprint"),
    Horizontal=_archetype("Horizontal"),
    Spatial2DView=_archetype("Spatial2DView"),
    TimeSeriesView=_archetype("TimeSeriesView"),
    Vertical=_archetype("Vertical"),
)


def _frame(
    *,
    step: int,
    time_s: float,
    robot_x: float,
    speed_command: float,
    omega_command: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "time_s": time_s,
        "robot": {
            "position": [robot_x, 10.0],
            "velocity": [speed_command, 0.0],
            "heading": 0.0,
        },
        "pedestrians": [
            {
                "id": 0,
                "position": [7.0, 10.0],
                "velocity": [0.0, 0.0],
            }
        ],
        "planner": {
            "selected_action": {
                "linear_velocity": speed_command,
                "angular_velocity": omega_command,
            }
        },
    }


def _episode_row(seed: int, status: str, robot_positions: list[float]) -> dict[str, Any]:
    adapter = trace_viewer._adapter_module()
    frames = [
        _frame(
            step=index,
            time_s=0.1 * (index + 1),
            robot_x=robot_x,
            speed_command=0.4 + 0.1 * index,
            omega_command=(-0.1 if seed % 2 else 0.1) * index,
        )
        for index, robot_x in enumerate(robot_positions)
    ]
    return {
        "episode_id": f"synthetic-{seed}",
        "scenario_id": "classic_doorway_medium",
        "seed": seed,
        "status": status,
        "termination_reason": status,
        "git_hash": adapter.EXEC_COMMIT,
        "algo": adapter.EXPECTED_ALGO,
        "result_provenance": {
            "repo_commit": adapter.EXEC_COMMIT,
            "scenario_id": adapter.EXPECTED_SCENARIO_ID,
            "seed": seed,
        },
        "algorithm_metadata": {
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "dt": 0.1,
                "initial_goal_distance_m": 10.0,
                "steps": frames,
            }
        },
    }


def test_raw_pair_loader_and_rerun_logging_contract(tmp_path: Path) -> None:
    """Raw adapter, canonical loader, A/B layout, entities, and time ranges agree."""
    episodes_jsonl = tmp_path / "episodes.jsonl"
    rows = [
        _episode_row(7, "success", [5.0, 5.1, 5.3]),
        _episode_row(8, "collision", [5.0, 5.2, 5.5]),
    ]
    episodes_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with trace_viewer.prepared_episode_dirs(
        bundle_dirs=[],
        episodes_jsonl=episodes_jsonl,
        seeds=[7, 8],
    ) as bundle_dirs:
        cases = trace_viewer.load_episode_bundles(bundle_dirs)
        recording = _FakeRecording()
        audit, focal, contrast = trace_viewer.log_cases(
            recording,
            FAKE_RERUN,
            FAKE_BLUEPRINT,
            cases,
        )
        report = trace_viewer.verify_recording_contract(audit, cases)

    assert [case.label for case in cases] == ["A", "B"]
    assert all(
        isinstance(case.scene_trace, trace_viewer.trace_scene.EpisodeTrace) for case in cases
    )
    assert focal.by_label == {"A": "0", "B": "0"}
    assert focal.shared
    assert contrast is not None
    assert "A ended in success" in contrast
    assert "B ended in collision" in contrast

    assert len(recording.blueprints) == 1
    blueprint = recording.blueprints[0]
    assert blueprint.kind == "Blueprint"
    vertical = blueprint.args[0]
    assert vertical.kind == "Vertical"
    assert vertical.args[0].kind == "Horizontal"
    assert [view.kind for view in vertical.args[0].args] == [
        "Spatial2DView",
        "Spatial2DView",
    ]
    assert [track.kind for track in vertical.args[1:]] == [
        "TimeSeriesView",
        "TimeSeriesView",
        "TimeSeriesView",
    ]

    assert "episode_A/scene/map/obstacles" in report.entity_paths
    assert "episode_A/scene/robot/path" in report.entity_paths
    assert "episode_A/summary/provenance" in report.entity_paths
    assert "episode_B/scene/events/collision" in report.entity_paths
    assert "episode_B/metrics/omega_command_rad_s" in report.entity_paths
    assert "comparison/provenance" in report.entity_paths
    assert report.time_ranges_s["episode_A/scene/robot/path"] == pytest.approx((0.1, 0.3))
    assert report.time_ranges_s["episode_B/metrics/min_pedestrian_clearance_m"] == pytest.approx(
        (0.1, 0.3)
    )

    robot_colors = [
        payload.kwargs["colors"]
        for path, payload, _static in recording.logs
        if path == "episode_A/scene/robot/position"
    ]
    assert robot_colors
    assert all(colors == [trace_viewer.COLOR_A] for colors in robot_colors)
    assert {timeline for _kind, timeline, _value in recording.times} == {
        trace_viewer.TIMELINE_NAME,
        trace_viewer.STEP_TIMELINE_NAME,
    }


def test_single_raw_seed_uses_one_spatial_panel(tmp_path: Path) -> None:
    """A single --seed input should not require or fabricate an episode B."""
    episodes_jsonl = tmp_path / "episodes.jsonl"
    episodes_jsonl.write_text(
        json.dumps(_episode_row(7, "success", [5.0, 5.1, 5.3])) + "\n",
        encoding="utf-8",
    )

    with trace_viewer.prepared_episode_dirs(
        bundle_dirs=[],
        episodes_jsonl=episodes_jsonl,
        seeds=[7],
    ) as bundle_dirs:
        cases = trace_viewer.load_episode_bundles(bundle_dirs)
        recording = _FakeRecording()
        audit, focal, contrast = trace_viewer.log_cases(
            recording,
            FAKE_RERUN,
            FAKE_BLUEPRINT,
            cases,
        )
        report = trace_viewer.verify_recording_contract(audit, cases)

    assert [case.label for case in cases] == ["A"]
    assert focal.by_label == {"A": "0"}
    assert contrast is None
    assert not any(path.startswith("episode_B/") for path in report.entity_paths)
    vertical = recording.blueprints[0].args[0]
    assert vertical.args[0].kind == "Spatial2DView"


def test_assumed_radius_overlap_does_not_fabricate_collision_event(tmp_path: Path) -> None:
    """Diagnostic default-radius overlap must not override a successful source outcome."""
    episodes_jsonl = tmp_path / "episodes.jsonl"
    episodes_jsonl.write_text(
        json.dumps(_episode_row(7, "success", [6.8, 6.9, 6.8])) + "\n",
        encoding="utf-8",
    )

    with trace_viewer.prepared_episode_dirs(
        bundle_dirs=[],
        episodes_jsonl=episodes_jsonl,
        seeds=[7],
    ) as bundle_dirs:
        cases = trace_viewer.load_episode_bundles(bundle_dirs)
        assert min(cases[0].surface_clearance_m) < 0.0
        recording = _FakeRecording()
        audit, _focal, _contrast = trace_viewer.log_cases(
            recording,
            FAKE_RERUN,
            FAKE_BLUEPRINT,
            cases,
        )

    assert "episode_A/scene/events/collision" not in audit.entities


def test_current_rerun_time_api_uses_duration_and_sequence_keywords() -> None:
    """Current RecordingStream time calls use the SDK's documented keywords."""
    recording = _ModernRecording()

    trace_viewer._set_recording_time(
        recording,
        timeline=trace_viewer.TIMELINE_NAME,
        value=1.25,
        kind="duration",
    )
    trace_viewer._set_recording_time(
        recording,
        timeline=trace_viewer.STEP_TIMELINE_NAME,
        value=7,
        kind="sequence",
    )

    assert recording.times == [
        (trace_viewer.TIMELINE_NAME, 1.25, None),
        (trace_viewer.STEP_TIMELINE_NAME, None, 7),
    ]


def test_null_metadata_summary_is_treated_as_absent() -> None:
    """Explicit JSON null summary values should not crash metadata fallback."""
    assert trace_viewer._metadata_summary({"summary": None}) == {}

    with pytest.raises(trace_viewer.TraceViewerError, match="object or null"):
        trace_viewer._metadata_summary({"summary": []})


def test_null_frame_time_has_source_aware_error(tmp_path: Path) -> None:
    """Explicit JSON null frame timestamps should fail cleanly."""
    with pytest.raises(trace_viewer.TraceViewerError, match="raw frame 3 has invalid time_s"):
        trace_viewer._frame_time_s({"time_s": None}, bundle_dir=tmp_path, index=3)


def test_saved_recording_is_closed_before_artifact_verification() -> None:
    """Saved recordings must write their footer before size/integrity checks."""
    recording = _SavedRecording()

    trace_viewer._finalize_saved_recording(recording)

    assert recording.events == ["flush:1e+38", "disconnect"]
