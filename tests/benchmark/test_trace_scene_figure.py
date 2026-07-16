"""Tests for exemplar trace scene figure rendering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.text import Text

from robot_sf.benchmark.figure_qa import lint_figure
from robot_sf.benchmark.trace_scene_figure import (
    BLUE,
    CONTEXT_PEDESTRIAN_ALPHA,
    CONTEXT_PEDESTRIAN_COLOR,
    FOCAL_PEDESTRIAN_COLOR,
    TEAL,
    EpisodeTrace,
    _choose_scale_bar_corner,
    _clamp_texts_to_canvas,
    _collect_line_obstacles,
    _collect_marker_obstacles,
    _compute_scene_extent,
    _contiguous_segments,
    _draw_robot_time_markers,
    _effective_marker_interval,
    _focal_pedestrian_id,
    _focused_pedestrian_tracks,
    _pedestrian_styles,
    _snap_marker_indices,
    load_episode,
    render_comparison,
    render_scene,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAD_ON_BUNDLE = REPO_ROOT / "docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07"
SOCIAL_FORCE_EPISODE = (
    HEAD_ON_BUNDLE / "social_force" / "classic_head_on_corridor_medium_seed24_worst"
)
ORCA_EPISODE = HEAD_ON_BUNDLE / "orca" / "classic_head_on_corridor_medium_seed24_best"


@pytest.fixture
def synthetic_episode() -> EpisodeTrace:
    """Build a deterministic ten-step trace with one near and one distant pedestrian."""

    steps = tuple(range(10))
    times = tuple(float(step) for step in steps)
    robot_xy = tuple((13.0 + 0.5 * step, 5.0 + step) for step in steps)
    close_track = tuple(
        (time, x + 1.0, y + 0.5) for time, (x, y) in zip(times, robot_xy, strict=True)
    )
    far_track = tuple((time, 30.0, 39.0) for time in times)
    min_distances = tuple(1.0 + abs(step - 5) * 0.1 for step in steps)
    return EpisodeTrace(
        metadata={
            "episode_status": "success",
            "planner": "synthetic",
            "scenario_id": "classic_head_on_corridor_medium",
            "seed": 7,
            "summary": {
                "global_min_robot_ped_distance_m": 1.0,
                "global_min_distance_step": 5,
                "step_count": 10,
                "termination_reason": "goal",
            },
        },
        steps=steps,
        time_s=times,
        robot_xy=robot_xy,
        robot_heading_rad=(0.0,) * 10,
        executed_speed_m_s=(1.0,) * 10,
        min_robot_ped_distance_m=min_distances,
        nearest_pedestrian_id=("close",) * 10,
        pedestrian_tracks={"close": close_track, "far": far_track},
    )


def test_pedestrian_filtering_uses_synchronized_distance(synthetic_episode: EpisodeTrace) -> None:
    """Only tracks entering the strict focus radius are retained."""

    focused, filtered_count = _focused_pedestrian_tracks(synthetic_episode, 2.0)

    assert set(focused) == {"close"}
    assert filtered_count == 1


def test_focal_pedestrian_selection_and_legacy_palette(
    synthetic_episode: EpisodeTrace,
) -> None:
    """Global-minimum pedestrian selection drives highlighting without removing legacy colors."""

    focused, _ = _focused_pedestrian_tracks(synthetic_episode, 50.0)

    assert _focal_pedestrian_id(synthetic_episode) == "close"
    highlighted = _pedestrian_styles(synthetic_episode, focused, highlight_focal=True)
    assert highlighted["close"].color == FOCAL_PEDESTRIAN_COLOR
    assert highlighted["close"].open_markers
    assert highlighted["far"].color == CONTEXT_PEDESTRIAN_COLOR
    assert highlighted["far"].alpha == CONTEXT_PEDESTRIAN_ALPHA
    assert not highlighted["far"].draw_markers

    legacy = _pedestrian_styles(synthetic_episode, focused, highlight_focal=False)
    assert [style.color for style in legacy.values()] == [BLUE, TEAL]
    assert all(style.draw_markers and style.show_label for style in legacy.values())


def test_marker_times_snap_to_nearest_step() -> None:
    """Requested marker timestamps snap deterministically to trace steps."""

    assert _snap_marker_indices((0.1, 0.7, 1.8, 2.2, 3.9), 2.0) == (2, 4)


def test_stationary_marker_run_is_consolidated() -> None:
    """A stopped robot keeps every dot but draws far fewer time labels."""

    steps = tuple(range(21))
    times = tuple(float(step) for step in steps)
    robot_xy = tuple((min(float(step), 6.0), 0.0) for step in steps)
    episode = EpisodeTrace(
        metadata={
            "episode_status": "failure",
            "planner": "synthetic",
            "scenario_id": "classic_head_on_corridor_medium",
            "seed": 9,
            "summary": {
                "global_min_robot_ped_distance_m": 2.0,
                "global_min_distance_step": 6,
                "step_count": 21,
                "termination_reason": "timeout",
            },
        },
        steps=steps,
        time_s=times,
        robot_xy=robot_xy,
        robot_heading_rad=(0.0,) * len(steps),
        executed_speed_m_s=(0.0,) * len(steps),
        min_robot_ped_distance_m=(2.0,) * len(steps),
        nearest_pedestrian_id=(None,) * len(steps),
        pedestrian_tracks={},
    )
    marker_indices = _snap_marker_indices(episode.time_s, 2.0)
    figure, ax = plt.subplots()

    label_specs = _draw_robot_time_markers(ax, episode, marker_indices, 2.0)
    drawn_time_labels = [text.get_text() for text in ax.texts if text.get_text().startswith("t=")]
    marker_dots = [line for line in ax.lines if line.get_marker() == "o"]
    plt.close(figure)

    assert len(marker_dots) == len(marker_indices)
    assert len(drawn_time_labels) == len(label_specs) < len(marker_indices) / 2
    assert "t=8-20 s (stopped)" in drawn_time_labels


def test_scale_bar_prefers_empty_bottom_corner() -> None:
    """Scale-bar selection uses the preferred bottom-right corner until it is occupied."""

    limits = ((0.0, 20.0), (0.0, 20.0))
    assert _choose_scale_bar_corner(limits, 5.0, [], []) == "bottom-right"
    bottom_right_points = [(x, 1.2) for x in range(13, 20)]
    assert _choose_scale_bar_corner(limits, 5.0, bottom_right_points, []) == "bottom-left"


def test_extent_and_synthetic_pdf(synthetic_episode: EpisodeTrace, tmp_path: Path) -> None:
    """Action extents include intersecting obstacles and produce a non-empty PDF."""

    focused, _ = _focused_pedestrian_tracks(synthetic_episode, 2.0)
    map_definition = SimpleNamespace(
        width=40.0,
        height=40.0,
        obstacles=[SimpleNamespace(vertices=[(12.0, 4.0), (14.0, 4.0), (14.0, 6.0), (12.0, 6.0)])],
    )

    assert _compute_scene_extent([synthetic_episode], [focused], map_definition) == (
        (10.0, 20.5),
        (2.0, 16.5),
    )

    output = render_scene(synthetic_episode, tmp_path / "synthetic.pdf", timeline=False)
    assert output.exists()
    assert output.stat().st_size > 5_000


def test_default_render_preserves_screen_layout(
    synthetic_episode: EpisodeTrace, tmp_path: Path
) -> None:
    """Default sizing must not activate print-only wrapping or density adaptations."""

    _, figure = render_scene(
        synthetic_episode,
        tmp_path / "default-screen.pdf",
        return_figure=True,
    )
    try:
        assert figure.get_figwidth() == pytest.approx(7.2)
        assert figure.axes[1].get_ylabel() == "min robot-ped distance (m)"
    finally:
        plt.close(figure)


def test_print_render_uses_requested_width_and_minimum_font(
    synthetic_episode: EpisodeTrace, tmp_path: Path
) -> None:
    """Print sizing should keep the requested canvas width and fonts at least 8.25 pt."""

    output, figure = render_scene(
        synthetic_episode,
        tmp_path / "print-sized.pdf",
        figure_width_in=3.43,
        base_font_pt=9.0,
        return_figure=True,
    )
    try:
        visible_font_sizes = [
            text.get_fontsize()
            for text in figure.findobj(Text)
            if text.get_visible() and text.get_text().strip()
        ]
        assert output.exists()
        assert figure.get_figwidth() == pytest.approx(3.43)
        assert visible_font_sizes
        assert min(visible_font_sizes) >= 8.25
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("sizing", "message"),
    [
        ({"figure_width_in": 0.0}, "figure_width_in must be greater than zero"),
        ({"base_font_pt": -1.0}, "base_font_pt must be greater than zero"),
    ],
)
def test_print_render_rejects_non_positive_sizing(
    synthetic_episode: EpisodeTrace,
    tmp_path: Path,
    sizing: dict[str, float],
    message: str,
) -> None:
    """Print sizing should fail closed before rendering invalid dimensions."""

    with pytest.raises(ValueError, match=message):
        render_scene(synthetic_episode, tmp_path / "invalid.pdf", **sizing)


def test_canvas_clamp_tolerates_missing_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canvas clamping should return safely when a backend has no initialized renderer."""

    figure, _ = plt.subplots()
    monkeypatch.setattr(figure.canvas, "draw", lambda: None)
    monkeypatch.setattr(figure.canvas, "get_renderer", lambda: None)
    try:
        _clamp_texts_to_canvas(figure)
    finally:
        plt.close(figure)


@pytest.mark.skipif(not SOCIAL_FORCE_EPISODE.is_dir(), reason="real exemplar bundle unavailable")
def test_render_real_head_on_episode(tmp_path: Path) -> None:
    """A retained head-on exemplar renders with its real map geometry."""

    episode = load_episode(SOCIAL_FORCE_EPISODE)
    output = render_scene(episode, tmp_path / "head_on.pdf")

    assert output.exists()
    assert output.stat().st_size > 5_000
    assert _effective_marker_interval([episode], None) == 4.0


@pytest.mark.skipif(
    not (SOCIAL_FORCE_EPISODE.is_dir() and ORCA_EPISODE.is_dir()),
    reason="real comparison exemplars unavailable",
)
def test_render_real_comparison(tmp_path: Path) -> None:
    """Two same-scenario retained exemplars render with shared axes."""

    episodes = [load_episode(ORCA_EPISODE), load_episode(SOCIAL_FORCE_EPISODE)]
    output = render_comparison(episodes, tmp_path / "comparison.pdf")

    assert output.exists()
    assert output.stat().st_size > 5_000


@pytest.mark.skipif(
    not (SOCIAL_FORCE_EPISODE.is_dir() and ORCA_EPISODE.is_dir()),
    reason="real exemplar figures unavailable",
)
def test_real_scene_and_comparison_have_no_hard_qa_defects(tmp_path: Path) -> None:
    """Real single and comparison figures carry no label-on-label or out-of-axes defects.

    ``text_line_overlap`` is deliberately not asserted: the canonical ``figure_qa`` linter
    flags reference-line labels (e.g. the ``collision envelope`` label that annotates its own
    dashed line) and time-marker labels adjacent to the trajectory, which are an intended part
    of this renderer's annotation style rather than legibility defects.
    """
    orca_episode = load_episode(ORCA_EPISODE)
    social_force_episode = load_episode(SOCIAL_FORCE_EPISODE)
    _, scene_figure = render_scene(
        orca_episode,
        tmp_path / "qa_scene.pdf",
        return_figure=True,
    )
    _, comparison_figure = render_comparison(
        [orca_episode, social_force_episode],
        tmp_path / "qa_comparison.pdf",
        return_figure=True,
    )
    hard_types = {"text_text_overlap", "text_out_of_axes"}
    try:
        for figure in (scene_figure, comparison_figure):
            hard = [
                d
                for d in lint_figure(figure)
                if d.severity == "error" and d.defect_type in hard_types
            ]
            assert not hard, [f"[{d.defect_type}] {d.message}" for d in hard]
    finally:
        plt.close(scene_figure)
        plt.close(comparison_figure)


def test_contiguous_segments_breaks_on_teleport() -> None:
    """A respawn teleport splits the polyline; a smooth track stays one segment."""
    # smooth walk (small steps) -> one segment
    xs = [0.0, 0.2, 0.4, 0.6]
    ys = [0.0, 0.1, 0.2, 0.3]
    assert len(_contiguous_segments(xs, ys, max_step=3.0)) == 1
    # a ~25 m jump in one step (respawn) -> two segments, no connecting line
    xs = [0.0, 0.2, 25.0, 25.2]
    ys = [0.0, 0.1, 30.0, 30.1]
    segments = _contiguous_segments(xs, ys, max_step=3.0)
    assert len(segments) == 2
    assert segments[0] == ([0.0, 0.2], [0.0, 0.1])
    assert segments[1] == ([25.0, 25.2], [30.0, 30.1])


def test_line_obstacles_split_non_finite_gaps() -> None:
    """NaN/Inf coordinates do not reach transforms or bridge line gaps."""
    figure, ax = plt.subplots()
    try:
        ax.plot([0.0, 1.0, np.nan, 2.0, 3.0], [0.0, 1.0, np.inf, 2.0, 3.0])
        figure.canvas.draw()
        obstacles = _collect_line_obstacles(ax)
    finally:
        plt.close(figure)

    assert len(obstacles) == 2
    assert all(np.isfinite(points).all() for points in obstacles)


def test_marker_obstacles_filter_non_finite_coordinates() -> None:
    """Scatter and line markers expose only finite display-space points."""
    figure, ax = plt.subplots()
    try:
        ax.scatter([0.0, np.nan], [1.0, 2.0])
        ax.plot([2.0, np.inf], [3.0, 4.0], marker="o", linestyle="none")
        figure.canvas.draw()
        markers = _collect_marker_obstacles(ax)
    finally:
        plt.close(figure)

    assert markers.shape == (2, 2)
    assert np.isfinite(markers).all()
