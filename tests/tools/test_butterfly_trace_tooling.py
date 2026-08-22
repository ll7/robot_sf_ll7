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


def _episode_row(
    arm: adapter.ReexportArmSpec = adapter.DOORWAY_ARM, *, seed: int = 113
) -> dict[str, Any]:
    """Return a minimal row matching the adapter's pinned campaign contract."""
    return {
        "seed": seed,
        "episode_id": f"{arm.scenario_id}--{seed}--fixture",
        "scenario_id": arm.scenario_id,
        "algo": arm.planner,
        "git_hash": arm.execution_commit,
        "config_hash": f"fixture-{arm.key}-config",
        "status": "success",
        "termination_reason": "success",
        "result_provenance": {
            "repo_commit": arm.execution_commit,
            "scenario_id": arm.scenario_id,
            "seed": seed,
            "config_hash": f"fixture-{arm.key}-config",
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


@pytest.mark.parametrize(
    "arm",
    [adapter.BOTTLENECK_GOAL_ARM, adapter.BOTTLENECK_PPO_ARM],
    ids=lambda arm: arm.key,
)
def test_bottleneck_arms_propagate_source_provenance(
    tmp_path: Path, arm: adapter.ReexportArmSpec
) -> None:
    """Bottleneck bundles carry source provenance, not arm-invented job labels."""
    row = _episode_row(arm, seed=118)
    row["result_provenance"].update({"campaign_id": f"source-{arm.key}", "slurm_job_id": 9001})
    episodes = tmp_path / f"{arm.key}.jsonl"
    _write_episode(episodes, row)

    adapter.build_bundle(episodes, 118, tmp_path / "bundle", arm=arm)
    metadata = json.loads((tmp_path / "bundle" / "metadata.json").read_text())

    assert metadata["source_arm"] == arm.key
    assert metadata["result_provenance"] == row["result_provenance"]
    assert metadata["campaign_id"] == f"source-{arm.key}"
    assert metadata["campaign_job"] == "9001"


@pytest.mark.parametrize(
    "arm",
    [adapter.BOTTLENECK_GOAL_ARM, adapter.BOTTLENECK_PPO_ARM],
    ids=lambda arm: arm.key,
)
def test_bottleneck_arms_reject_mismatched_result_provenance(
    tmp_path: Path, arm: adapter.ReexportArmSpec
) -> None:
    """A matching top-level row cannot inherit a different arm's provenance."""
    row = _episode_row(arm, seed=118)
    row["result_provenance"]["scenario_id"] = "unrelated-scenario"
    episodes = tmp_path / f"{arm.key}.jsonl"
    _write_episode(episodes, row)

    with pytest.raises(ValueError, match=r"result_provenance\.scenario_id"):
        adapter.build_bundle(episodes, 118, tmp_path / "bundle", arm=arm)


def test_build_bundle_requires_result_provenance(tmp_path: Path) -> None:
    """No arm may label an unproven matching row as a pinned re-export."""
    row = _episode_row(adapter.BOTTLENECK_GOAL_ARM, seed=118)
    del row["result_provenance"]
    episodes = tmp_path / "episodes.jsonl"
    _write_episode(episodes, row)

    with pytest.raises(ValueError, match="result_provenance is missing"):
        adapter.build_bundle(
            episodes,
            118,
            tmp_path / "bundle",
            arm=adapter.BOTTLENECK_GOAL_ARM,
        )


def test_bottleneck_sidecar_uses_selected_arm_source_provenance(tmp_path: Path) -> None:
    """A bottleneck sidecar cannot inherit doorway-only provenance statements."""
    arm = adapter.BOTTLENECK_GOAL_ARM
    row = _episode_row(arm, seed=118)
    row["metrics"] = {"near_misses": 4}
    row["result_provenance"].update({"campaign_id": "goal-source", "slurm_job_id": 13487})
    episodes = tmp_path / "episodes.jsonl"
    provenance_json = tmp_path / "butterfly_hinge_provenance.json"
    _write_episode(episodes, row)
    provenance_json.write_text("{}", encoding="utf-8")

    payload = adapter.augment_provenance_sidecar(provenance_json, episodes, 118, arm=arm)[
        "release_reexport_provenance"
    ]

    assert payload["source_arm"] == arm.key
    assert payload["execution_commit"] == arm.execution_commit
    assert payload["result_provenance"] == row["result_provenance"]
    assert payload["seed_118_near_misses"]["rerun_execution"] == 4
    assert "slurm_job" not in payload
    assert "config" not in payload
    assert "outcome_fidelity_vs_release" not in payload


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


def test_hinge_gutter_font_size_is_capped_for_short_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short custom gutter must not enlarge the adaptive hinge font."""
    fig, ax = plt.subplots()
    monkeypatch.setattr(hinge, "_format_gutter_lines", lambda _gutter: ["short label"])

    hinge._draw_delta_gutter(ax, {})

    assert ax.texts[0].get_fontsize() == pytest.approx(6.3)
    plt.close(fig)


def test_print_contrast_renderer_uses_final_width_and_role_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise print geometry, including the panel-B tick/spine clearance regression."""
    episode_kwargs = {
        "payload": {},
        "metadata": {"summary": {"step_count": 3}},
        # Offset the minimal trace so the synthetic axes use two-digit y ticks (10/15/20),
        # matching the print-layout collision reported by ll7/diss#1409.
        "robot_xy": np.array([[0.0, 15.0], [1.0, 15.0], [2.0, 15.0]]),
        "robot_vel": np.zeros((3, 2)),
        "time_s": np.array([0.1, 0.2, 0.3]),
        "ped_ids": [7],
        "ped_xy": np.array([[[4.0, 15.0]], [[4.0, 15.0]], [[4.0, 15.0]]]),
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

    # ``render_hinge_figure`` closes its figure after saving; capture it at close time so the
    # assertion measures the final post-tight-layout display geometry, not the pre-layout spec.
    rendered_figures: list[tuple[Any, Any]] = []
    original_close = hinge.plt.close

    def capture_close(fig: Any = None) -> None:
        if fig is not None and not isinstance(fig, str):
            fig.canvas.draw()
            rendered_figures.append((fig, fig.canvas.get_renderer()))
        original_close(fig)

    monkeypatch.setattr(hinge.plt, "close", capture_close)
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
        closest_a={
            "distance_m": 2.0,
            "time_s": 0.3,
            "robot_xy": [2.0, 15.0],
            "ped_xy": [4.0, 15.0],
        },
        closest_b={
            "distance_m": 2.0,
            "time_s": 0.3,
            "robot_xy": [2.0, 15.0],
            "ped_xy": [4.0, 15.0],
        },
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

    assert rendered_figures
    fig, renderer = rendered_figures[-1]
    panel_a, panel_b = fig.axes[:2]
    panel_a_right = panel_a.spines["right"].get_window_extent(renderer).x1
    tick_boxes = [
        label.get_window_extent(renderer)
        for label in panel_b.get_yticklabels()
        if label.get_visible()
    ]
    assert tick_boxes
    assert all(
        len(label.get_text()) >= 2 for label in panel_b.get_yticklabels() if label.get_visible()
    )
    assert min(box.x0 for box in tick_boxes) - panel_a_right > 0.0

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


def _render_args(tmp_path: Path, *, b_story_steps: int = 1) -> tuple[list[str], Path]:
    """Build two minimal fixture bundles and return a runnable ``main()`` argv + out dir.

    Both bundles are the same 2-frame doorway fixture row (seed 113); ``--b-story-steps``
    truncates episode B's story window so the rendered ``n_steps_used`` is observable and
    pin-able by the issue #6616 expectation flags.
    """
    episodes = tmp_path / "episodes.jsonl"
    _write_episode(episodes, _episode_row())
    bundle_a = tmp_path / "bundle_a"
    bundle_b = tmp_path / "bundle_b"
    adapter.build_bundle(episodes, 113, bundle_a)
    adapter.build_bundle(episodes, 113, bundle_b)
    out_dir = tmp_path / "out"
    argv = [
        "--episode-a",
        str(bundle_a),
        "--episode-b",
        str(bundle_b),
        "--out-dir",
        str(out_dir),
        "--no-video",
        "--b-story-steps",
        str(b_story_steps),
    ]
    return argv, out_dir


def test_hinge_sidecar_records_full_invocation_with_non_default_flag(tmp_path: Path) -> None:
    """Issue #6616: a render with a non-default ``--b-story-steps`` pins the COMPLETE argv.

    Asserts the regression class: a re-render reading the provenance sidecar can recover
    the exact non-default flag (``--b-story-steps 1``) from ``invocation.argv`` /
    ``invocation.parsed_args``, so a silent fallback to the 220-step default is no longer
    unrecoverable.
    """
    argv, out_dir = _render_args(tmp_path, b_story_steps=1)
    assert hinge.main(argv) == 0

    sidecar = json.loads((out_dir / "butterfly_hinge_provenance.json").read_text(encoding="utf-8"))
    assert sidecar["invocation"]["argv"] == argv
    assert "--b-story-steps" in sidecar["invocation"]["argv"]
    assert "1" in sidecar["invocation"]["argv"]
    assert sidecar["invocation"]["parsed_args"]["b_story_steps"] == 1
    assert sidecar["episode_b"]["n_steps_used"] == 1


def test_hinge_fails_closed_on_b_story_steps_expectation_mismatch(tmp_path: Path) -> None:
    """Issue #6616: ``--expect-b-story-steps`` disagreement fails closed before rendering.

    A re-render that silently dropped ``--b-story-steps`` (falling back to the 220
    default) must exit non-zero with an actionable message instead of writing a
    mismatched figure.
    """
    argv, out_dir = _render_args(tmp_path, b_story_steps=1)
    argv += ["--expect-b-story-steps", "235"]

    with pytest.raises(RuntimeError, match="--expect-b-story-steps 235"):
        hinge.main(argv)

    assert not (out_dir / "butterfly_hinge_figure_proto.png").exists()
    assert not (out_dir / "butterfly_hinge_report.json").exists()
    assert not (out_dir / "butterfly_hinge_provenance.json").exists()


def test_hinge_fails_closed_on_rendered_step_count_expectation_mismatch(tmp_path: Path) -> None:
    """Issue #6616: ``--expect-n-steps-b`` disagreement fails closed before rendering."""
    argv, out_dir = _render_args(tmp_path, b_story_steps=1)
    argv += ["--expect-n-steps-b", "2"]

    with pytest.raises(RuntimeError, match="--expect-n-steps-b 2"):
        hinge.main(argv)

    assert not (out_dir / "butterfly_hinge_figure_proto.png").exists()
    assert not (out_dir / "butterfly_hinge_report.json").exists()
    assert not (out_dir / "butterfly_hinge_provenance.json").exists()


def test_hinge_expectation_agreement_renders_normally(tmp_path: Path) -> None:
    """Issue #6616: matching expectations render normally and land in the sidecar."""
    argv, out_dir = _render_args(tmp_path, b_story_steps=1)
    argv += ["--expect-b-story-steps", "1", "--expect-n-steps-b", "1"]

    assert hinge.main(argv) == 0

    sidecar = json.loads((out_dir / "butterfly_hinge_provenance.json").read_text(encoding="utf-8"))
    assert sidecar["episode_b"]["n_steps_used"] == 1
    assert sidecar["invocation"]["parsed_args"]["expect_b_story_steps"] == 1
    assert sidecar["invocation"]["parsed_args"]["expect_n_steps_b"] == 1
    assert (out_dir / "butterfly_hinge_report.json").exists()


def test_assert_render_expectations_match_guard_semantics() -> None:
    """The guard is a no-op when nothing is pinned and raises an actionable RuntimeError
    for each mismatch class independently."""
    hinge.assert_render_expectations_match(
        expected_b_story_steps=None,
        expected_n_steps_b=None,
        actual_b_story_steps=220,
        actual_n_steps_b=220,
    )

    with pytest.raises(RuntimeError, match="expect-b-story-steps 235"):
        hinge.assert_render_expectations_match(
            expected_b_story_steps=235,
            expected_n_steps_b=None,
            actual_b_story_steps=220,
            actual_n_steps_b=220,
        )

    with pytest.raises(RuntimeError, match="expect-n-steps-b 235"):
        hinge.assert_render_expectations_match(
            expected_b_story_steps=None,
            expected_n_steps_b=235,
            actual_b_story_steps=220,
            actual_n_steps_b=220,
        )
