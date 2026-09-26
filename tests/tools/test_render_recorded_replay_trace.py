"""Render-only replay-trace tooling must never invoke episode dynamics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from robot_sf.adversarial import replay_gallery
from robot_sf.benchmark import episode_replay_figure
from scripts.tools import render_recorded_replay_trace as render_tool


def _trace_payload() -> dict[str, Any]:
    return {
        "schema_version": "simulation-step-trace.v1",
        "dt": 0.1,
        "steps": [
            {
                "time_s": 0.1,
                "robot": {"position": [0.0, 0.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [
                    {"id": "ped-0", "position": [0.8, 0.0], "surface_clearance_m": 0.2}
                ],
            },
            {
                "time_s": 0.2,
                "robot": {"position": [0.1, 0.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [
                    {"id": "ped-0", "position": [0.2, 0.0], "surface_clearance_m": -0.01}
                ],
            },
            {
                "time_s": 0.3,
                "robot": {"position": [0.2, 0.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [
                    {"id": "ped-0", "position": [0.1, 0.0], "surface_clearance_m": 0.0}
                ],
            },
        ],
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    trace_path = tmp_path / "simulation_step_trace.json"
    trace_path.write_text(
        json.dumps(_trace_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    provenance = {
        "schema_version": "issue_9647_recorded_trace_provenance.v1",
        "trace": {
            "sha256": trace_sha,
            "source_extraction_pointer": "algorithm_metadata.simulation_step_trace",
        },
        "source": {
            "episode_record_sha256": "b" * 64,
            "episode_record_provenance_sha256": "c" * 64,
        },
        "episode": {
            "case_id": "case_recorded_test",
            "episode_id": "episode-recorded-test",
            "scenario_id": "scenario-recorded-test",
            "seed": 12,
            "planner": "goal",
            "runner_revision": "a" * 40,
            "exact_collision": True,
            "collision_events": [
                {
                    "event_time_s": 0.3,
                    "exact_event_source": "runtime.step.meta.is_pedestrian_collision",
                }
            ],
        },
    }
    provenance_path = tmp_path / "trace_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trace_path, provenance_path


def _forbid_dynamics(*_args: Any, **_kwargs: Any) -> Any:
    pytest.fail("render-only trace workflow attempted to run episode dynamics")


def test_recorded_trace_render_checks_digest_and_writes_annotated_figures(
    tmp_path: Path, monkeypatch: Any
) -> None:
    trace_path, provenance_path = _write_inputs(tmp_path)
    monkeypatch.setattr(replay_gallery, "run_batch", _forbid_dynamics)
    monkeypatch.setattr(episode_replay_figure, "_resimulate_episode", _forbid_dynamics)

    result = render_tool.render_recorded_trace_bundle(
        trace_path, provenance_path, tmp_path / "render", video=False
    )

    render = result["render_result"]
    assert render["status"] == "rendered"
    assert render["frame_steps"] == [0, 1, 2]
    assert render["critical_frame_step"] == 1
    assert render["visual_annotations"]["minimum_clearance"]["render_step_index"] == 1
    assert render["visual_annotations"]["collision_events"][0]["render_step_index"] == 2
    assert render["visual_annotations"]["minimum_clearance"]["source_path"].endswith(
        "steps[1].pedestrians[0].surface_clearance_m"
    )
    assert render["visual_annotations"]["collision_events"][0]["source_path"] == (
        "event_ledger.collision_events[0].collision_time"
    )
    assert set(render["artifacts"]) == {
        "cases/case_recorded_test/figures/still_1.png",
        "cases/case_recorded_test/figures/filmstrip.png",
        "cases/case_recorded_test/figures/trajectory.png",
    }
    assert all((tmp_path / "render" / path).is_file() for path in render["artifacts"])
    provenance = json.loads(Path(result["provenance_path"]).read_text(encoding="utf-8"))
    assert provenance["simulation_or_replay_dynamics_executed"] is False
    assert provenance["render_mode"] == "stored_trace_only"


def test_recorded_video_receives_only_converted_stored_samples(
    tmp_path: Path, monkeypatch: Any
) -> None:
    trace_path, provenance_path = _write_inputs(tmp_path)
    monkeypatch.setattr(replay_gallery, "run_batch", _forbid_dynamics)
    monkeypatch.setattr(episode_replay_figure, "_resimulate_episode", _forbid_dynamics)
    seen: dict[str, Any] = {}

    def _fake_video(episode: Any, video_path: str, **kwargs: Any) -> dict[str, Any]:
        seen["times"] = [step.t for step in episode.steps]
        seen["max_frames"] = kwargs["max_frames"]
        Path(video_path).write_bytes(b"video derived from stored samples")
        return {"status": "success", "note": None}

    video_renderer = ModuleType("robot_sf.benchmark.full_classic.render_sim_view")
    video_renderer.generate_video_file = _fake_video  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "robot_sf.benchmark.full_classic.render_sim_view",
        video_renderer,
    )
    result = render_tool.render_recorded_trace_bundle(
        trace_path, provenance_path, tmp_path / "render_video", video=True
    )

    assert seen == {"times": [0.1, 0.2, 0.3], "max_frames": 3}
    assert result["video_result"]["status"] == "rendered_from_stored_trace"
    assert result["video_result"]["frames_requested"] == 3
    assert (
        result["video_result"]["sha256"]
        == hashlib.sha256(b"video derived from stored samples").hexdigest()
    )


def test_recorded_trace_render_rejects_a_mismatched_trace_digest(tmp_path: Path) -> None:
    trace_path, provenance_path = _write_inputs(tmp_path)
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="trace bytes do not match"):
        render_tool.render_recorded_trace_bundle(
            trace_path, provenance_path, tmp_path / "render", video=False
        )
