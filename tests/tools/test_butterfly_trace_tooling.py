"""Focused tests for the prototype A/B trace adapter and playback bridge."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

from robot_sf.render.jsonl_playback import JSONLPlaybackLoader
from scripts.repro import butterfly_hinge_figure_proto as hinge
from scripts.repro import butterfly_reexport_to_trace_series as adapter
from scripts.repro import butterfly_trace_to_video_proto as video

if TYPE_CHECKING:
    from pathlib import Path


def _frame(step: int, robot_x: float, pedestrian_x: float) -> dict[str, Any]:
    """Return one minimal simulation-step-trace frame."""
    return {
        "step": step,
        "time_s": (step + 1) * 0.1,
        "robot": {
            "position": [robot_x, 0.0],
            "velocity": [0.5, 0.0],
            "heading": 0.0,
        },
        "pedestrians": [{"id": 7, "position": [pedestrian_x, 0.0]}],
        "planner": {
            "selected_action": {
                "linear_velocity": 0.5,
                "angular_velocity": 0.0,
            }
        },
    }


def _episode_row() -> dict[str, Any]:
    """Return a minimal row matching the adapter's pinned campaign contract."""
    return {
        "seed": 113,
        "episode_id": "classic_doorway_medium--113--fixture",
        "scenario_id": adapter.EXPECTED_SCENARIO_ID,
        "algo": adapter.EXPECTED_ALGO,
        "git_hash": adapter.EXEC_COMMIT,
        "status": "success",
        "termination_reason": "success",
        "result_provenance": {
            "repo_commit": adapter.EXEC_COMMIT,
            "scenario_id": adapter.EXPECTED_SCENARIO_ID,
            "seed": 113,
        },
        "algorithm_metadata": {
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "steps": [_frame(0, 0.0, 2.0), _frame(1, 0.0, 1.5)],
            }
        },
    }


def _write_episode(path: Path, row: dict[str, Any]) -> None:
    """Write one benchmark episode row as JSON Lines."""
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_build_bundle_round_trips_through_playback_loader(tmp_path: Path) -> None:
    """Exercise the production adapter-to-JSONL playback boundary."""
    episodes = tmp_path / "episodes.jsonl"
    bundle = tmp_path / "bundle"
    _write_episode(episodes, _episode_row())

    summary = adapter.build_bundle(episodes, 113, bundle)
    trace_payload = json.loads((bundle / "trace_series.json").read_text(encoding="utf-8"))

    assert summary["n_steps"] == 2
    assert trace_payload["derived_rows"][1]["nearest_pedestrian_id"] == "7"
    assert trace_payload["metadata"]["git_commit"] == adapter.EXEC_COMMIT

    playback_jsonl = tmp_path / "playback.jsonl"
    assert video.trace_series_to_jsonl(bundle / "trace_series.json", playback_jsonl) == 2
    episode, _map_definition = JSONLPlaybackLoader().load_single_episode(playback_jsonl)
    assert len(episode.states) == 2

    loaded = hinge.load_episode(bundle, "A")
    assert hinge.count_near_miss_steps(loaded) == 1


def test_build_bundle_rejects_mislabeled_source_provenance(tmp_path: Path) -> None:
    """Prevent arbitrary episode files from inheriting pinned job provenance."""
    row = _episode_row()
    row["git_hash"] = "not-the-pinned-commit"
    episodes = tmp_path / "episodes.jsonl"
    _write_episode(episodes, row)

    with pytest.raises(ValueError, match="does not match pinned doorway re-export provenance"):
        adapter.build_bundle(episodes, 113, tmp_path / "bundle")


def test_build_bundle_rejects_empty_trace(tmp_path: Path) -> None:
    """Report an actionable error instead of failing later during minimum reduction."""
    row = copy.deepcopy(_episode_row())
    row["algorithm_metadata"]["simulation_step_trace"]["steps"] = []
    episodes = tmp_path / "episodes.jsonl"
    _write_episode(episodes, row)

    with pytest.raises(ValueError, match="steps must be a non-empty array"):
        adapter.build_bundle(episodes, 113, tmp_path / "bundle")


def test_video_helpers_reject_empty_trace(tmp_path: Path) -> None:
    """Keep both playback conversion and metric extraction fail-closed."""
    trace_series = tmp_path / "trace_series.json"
    trace_series.write_text(json.dumps({"metadata": {}, "frames": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="frames must be a non-empty array"):
        video.trace_series_to_jsonl(trace_series, tmp_path / "playback.jsonl")
    with pytest.raises(ValueError, match="frames must be a non-empty array"):
        video.compute_trace_metrics(trace_series)


def test_print_contrast_renderer_uses_final_width_and_role_colors(tmp_path: Path) -> None:
    """Exercise the renderer delta's print geometry and shared role-color contract."""
    episode_kwargs = {
        "payload": {},
        "metadata": {"summary": {"step_count": 3}},
        "robot_xy": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "robot_vel": np.zeros((3, 2)),
        "time_s": np.array([0.1, 0.2, 0.3]),
        "ped_ids": [7],
        "ped_xy": np.array([[[4.0, 0.0]], [[4.0, 0.0]], [[4.0, 0.0]]]),
        "cmd_v": np.array([0.5, 0.5, 0.5]),
        "cmd_omega": np.zeros(3),
        "metrics": {
            "clearance_m": np.array([4.0, 3.0, 2.0]),
            "nearest_pedestrian_id": np.array([7.0, 7.0, 7.0]),
        },
    }
    ep_a = hinge.EpisodeTrace("A", tmp_path, **episode_kwargs)
    ep_b = hinge.EpisodeTrace("B", tmp_path, **episode_kwargs)
    out_png = tmp_path / "contrast.png"

    hinge.render_hinge_figure(
        ep_a,
        ep_b,
        label_a="A",
        label_b="B",
        divergence={},
        separator={"step": 1},
        gutter={
            "mode": "contrast",
            "min_clearance_focal_m": {"episode_a": 2.0, "episode_b": 2.0},
            "near_miss_steps": {"episode_a": 0, "episode_b": 0},
            "steps_to_termination": {"episode_a": 3, "episode_b": 3},
            "first_braking_time_s": {"episode_a": None, "episode_b": None},
        },
        focal_ped_id=7,
        events_a={},
        events_b={},
        closest_a={"distance_m": 2.0, "time_s": 0.3, "robot_xy": [2.0, 0.0], "ped_xy": [4.0, 0.0]},
        closest_b={"distance_m": 2.0, "time_s": 0.3, "robot_xy": [2.0, 0.0], "ped_xy": [4.0, 0.0]},
        outcome_a="success",
        outcome_b="non_completion",
        b_outcome_step=None,
        headline="diagnostic contrast",
        map_definition=None,
        out_pdf=tmp_path / "contrast.pdf",
        out_png=out_png,
        contrast_mode=True,
        layout="print",
    )

    image = mpimg.imread(out_png)
    assert image.shape[1] == round(hinge.PRINT_FIG_WIDTH_IN * 200)

    legend = hinge._build_legend_elements(
        contrast_mode=True,
        print_layout=True,
        outcome_b="non_completion",
        b_outcome_step=None,
    )
    assert legend[0].get_color() == hinge.tsf.INK
    assert legend[1].get_color() == hinge.tsf.INK
    assert legend[1].get_linestyle() == "--"
    assert hinge.COLOR_PED_FOCAL_OUTLINE == hinge.tsf.ORANGE
    plt.close("all")
