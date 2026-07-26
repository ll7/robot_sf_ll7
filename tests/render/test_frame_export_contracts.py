"""Headless contract tests for :mod:`robot_sf.render.frame_export`.

These tests lock the PNG/filmstrip writers, render lifecycle, pickle export
orchestration, argument validation, and CLI output. They use ``tmp_path`` for
all filesystem effects and mock :class:`SimulationView` and
:func:`load_states` so no GUI window is opened and no real pickle/map asset is
read. Frame arrays are small synthetic ``uint8`` RGB buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

from robot_sf.render import frame_export

if TYPE_CHECKING:
    from pathlib import Path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _frame(fill: int, shape: tuple[int, int, int] = (4, 5, 3)) -> np.ndarray:
    """Return a deterministic ``uint8`` RGB frame filled with ``fill``.

    Default shape is ``(height=4, width=5, channels=3)`` so the resulting
    :class:`PIL.Image` has ``size == (5, 4)``.
    """
    return np.full(shape, fill_value=fill, dtype=np.uint8)


def _fake_view_factory(
    frames: list[np.ndarray] | None,
    *,
    render_raises: BaseException | None = None,
) -> tuple[type, dict[str, list[Any]]]:
    """Build a :class:`SimulationView` stand-in for ``render_selected_frames``.

    The returned view class records construction kwargs, ``render`` calls, and
    ``exit_simulation`` calls, and mirrors the ``is_exit_requested`` flag that
    gates cleanup in the real ``finally`` block.

    Args:
        frames: Frames returned by ``exit_simulation(return_frames=True)``.
            ``None`` models the real method returning ``None``.
        render_raises: If set, ``render`` raises this after recording the call.

    Returns:
        Tuple of the fake view class and a shared state dict whose
        ``"instances"`` list collects every constructed view.
    """
    state: dict[str, list[Any]] = {"instances": [], "init_kwargs": []}

    class _FakeView:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = dict(kwargs)
            self.render_calls: list[Any] = []
            self.exit_calls: list[bool] = []
            self.is_exit_requested = False
            state["instances"].append(self)
            state["init_kwargs"].append(self.init_kwargs)

        def render(self, sim_state: Any) -> None:
            self.render_calls.append(sim_state)
            if render_raises is not None:
                raise render_raises

        def exit_simulation(self, return_frames: bool = False):
            self.exit_calls.append(return_frames)
            # Real exit_simulation flips the flag before doing any work.
            self.is_exit_requested = True
            if return_frames:
                return None if frames is None else list(frames)
            return None

    return _FakeView, state


# --------------------------------------------------------------------------- #
# select_evenly_spaced_indices: argument validation and edge cases
# --------------------------------------------------------------------------- #


def test_select_rejects_negative_total_frames() -> None:
    """Negative totals are rejected before any selection runs."""
    with pytest.raises(ValueError, match="non-negative"):
        frame_export.select_evenly_spaced_indices(-1, 3)


def test_select_rejects_nonpositive_count() -> None:
    """Counts below one are rejected before any selection runs."""
    with pytest.raises(ValueError, match="positive"):
        frame_export.select_evenly_spaced_indices(5, 0)


def test_select_single_count_returns_first_frame_only() -> None:
    """A count of one always selects frame zero, independent of total."""
    assert frame_export.select_evenly_spaced_indices(7, 1) == [0]


def test_select_count_at_least_total_returns_every_frame() -> None:
    """When count meets or exceeds the total, every index is returned in order."""
    assert frame_export.select_evenly_spaced_indices(3, 10) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# render_selected_frames: lifecycle, validation, and cleanup
# --------------------------------------------------------------------------- #


def test_render_skips_view_construction_when_states_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty states short-circuit before any renderer is built."""
    view_cls, state = _fake_view_factory([])
    monkeypatch.setattr(frame_export, "SimulationView", view_cls)

    assert frame_export.render_selected_frames([], object(), [0, 1]) == []
    assert state["instances"] == []


def test_render_skips_view_construction_when_indices_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty indices short-circuit before any renderer is built."""
    view_cls, state = _fake_view_factory([])
    monkeypatch.setattr(frame_export, "SimulationView", view_cls)

    assert frame_export.render_selected_frames(["s0", "s1"], object(), []) == []
    assert state["instances"] == []


def test_render_rejects_invalid_indices_before_view_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-range indices raise before the renderer is constructed."""
    view_cls, state = _fake_view_factory([])
    monkeypatch.setattr(frame_export, "SimulationView", view_cls)

    with pytest.raises(IndexError, match="out of range"):
        frame_export.render_selected_frames(["s0", "s1"], object(), [0, 5, -1])
    assert state["instances"] == []


def test_render_constructs_headless_view_and_renders_each_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renderer is built once with headless/recording kwargs and renders each index."""
    frames = [_frame(11), _frame(22), _frame(33)]
    view_cls, state = _fake_view_factory(frames)
    map_def = object()
    monkeypatch.setattr(frame_export, "SimulationView", view_cls)

    result = frame_export.render_selected_frames(
        ["s0", "s1", "s2", "s3"], map_def, [0, 2, 3], width=100, height=50, scaling=2.5
    )

    assert result == frames
    assert len(state["instances"]) == 1
    view = state["instances"][0]
    assert view.render_calls == ["s0", "s2", "s3"]
    assert view.init_kwargs == {
        "width": 100,
        "height": 50,
        "scaling": 2.5,
        "map_def": map_def,
        "caption": "RobotSF Frame Export",
        "record_video": True,
        "video_path": None,
        "display_text": False,
    }


def test_render_uses_default_render_size_when_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default width/height/scaling flow through to the renderer."""
    view_cls, state = _fake_view_factory([_frame(1)])
    monkeypatch.setattr(frame_export, "SimulationView", view_cls)

    frame_export.render_selected_frames(["s0"], object(), [0])

    kw = state["instances"][0].init_kwargs
    assert kw["width"] == 1280
    assert kw["height"] == 720
    assert kw["scaling"] == 10


def test_render_does_not_double_exit_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """On success exit runs once with return_frames=True; the finally cleanup is skipped."""
    view_cls, state = _fake_view_factory([_frame(7)])
    monkeypatch.setattr(frame_export, "SimulationView", view_cls)

    frame_export.render_selected_frames(["s0"], object(), [0])

    view = state["instances"][0]
    assert view.exit_calls == [True]


def test_render_cleans_up_when_render_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A render exception triggers exactly one cleanup exit before propagating."""
    view_cls, state = _fake_view_factory([], render_raises=RuntimeError("render boom"))
    monkeypatch.setattr(frame_export, "SimulationView", view_cls)

    with pytest.raises(RuntimeError, match="render boom"):
        frame_export.render_selected_frames(["s0", "s1"], object(), [0])

    view = state["instances"][0]
    # render raised before exit flipped the flag -> finally calls cleanup exit once.
    assert view.exit_calls == [False]


def test_render_skips_cleanup_when_exit_already_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If exit_simulation raises after flipping the flag, the finally cleanup is skipped."""
    instances: list[Any] = []

    class _FakeView:
        def __init__(self, **kwargs: Any) -> None:
            self.exit_calls: list[bool] = []
            self.is_exit_requested = False
            instances.append(self)

        def render(self, sim_state: Any) -> None:
            return None

        def exit_simulation(self, return_frames: bool = False):
            self.exit_calls.append(return_frames)
            # Real exit_simulation flips the flag before doing any work.
            self.is_exit_requested = True
            if return_frames:
                raise RuntimeError("video write failed")

    monkeypatch.setattr(frame_export, "SimulationView", _FakeView)

    with pytest.raises(RuntimeError, match="video write failed"):
        frame_export.render_selected_frames(["s0"], object(), [0])

    # Flag was flipped before the raise, so finally must not call exit again.
    assert instances[0].exit_calls == [True]


def test_render_returns_empty_list_when_view_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty captured-frame list falls through the ``or []`` guard unchanged."""
    view_cls, state = _fake_view_factory([])
    monkeypatch.setattr(frame_export, "SimulationView", view_cls)

    assert frame_export.render_selected_frames(["s0"], object(), [0]) == []
    assert state["instances"][0].exit_calls == [True]


# --------------------------------------------------------------------------- #
# write_png_frames: naming, conversion, parent creation, validation
# --------------------------------------------------------------------------- #


def test_write_png_frames_names_by_explicit_indices(tmp_path: Path) -> None:
    """Explicit indices drive zero-padded deterministic filenames."""
    frames = [_frame(10), _frame(20), _frame(30)]
    paths = frame_export.write_png_frames(
        frames, tmp_path / "out", prefix="shot", indices=[0, 5, 12]
    )

    assert [p.name for p in paths] == ["shot_000000.png", "shot_000005.png", "shot_000012.png"]
    assert all(path.exists() for path in paths)


def test_write_png_frames_names_by_ordinal_when_indices_none(tmp_path: Path) -> None:
    """Without explicit indices, the ordinal position drives the filename."""
    paths = frame_export.write_png_frames([_frame(10), _frame(20)], tmp_path / "out")

    assert [p.name for p in paths] == ["frame_000000.png", "frame_000001.png"]


def test_write_png_frames_creates_nested_parent_dirs(tmp_path: Path) -> None:
    """Missing nested output directories are created before writing."""
    out = tmp_path / "a" / "b" / "c"
    assert not out.exists()

    paths = frame_export.write_png_frames([_frame(40)], out)

    assert out.is_dir()
    assert paths[0].exists()


def test_write_png_frames_converts_float_frames_to_uint8(tmp_path: Path) -> None:
    """Float input is truncated (not rounded) to uint8 on write."""
    frame = np.full((2, 2, 3), 200.7, dtype=np.float64)

    paths = frame_export.write_png_frames([frame], tmp_path / "out")

    arr = np.asarray(Image.open(paths[0]))
    assert arr.dtype == np.uint8
    assert (arr == 200).all()


def test_write_png_frames_rejects_mismatched_indices_length(tmp_path: Path) -> None:
    """An indices list whose length differs from frames raises before writing."""
    with pytest.raises(ValueError, match="indices length must match"):
        frame_export.write_png_frames([_frame(1), _frame(2)], tmp_path / "out", indices=[0])


def test_write_png_frames_empty_returns_empty_but_creates_dir(tmp_path: Path) -> None:
    """No frames writes nothing but still ensures the output directory exists."""
    out = tmp_path / "empty"

    paths = frame_export.write_png_frames([], out)

    assert paths == []
    assert out.is_dir()


# --------------------------------------------------------------------------- #
# write_filmstrip: validation, defaults, and grid layout
# --------------------------------------------------------------------------- #


def test_write_filmstrip_rejects_empty_frames(tmp_path: Path) -> None:
    """A filmstrip cannot be written from zero frames."""
    with pytest.raises(ValueError, match="without frames"):
        frame_export.write_filmstrip([], tmp_path / "x.png")


@pytest.mark.parametrize("columns", [0, -3])
def test_write_filmstrip_rejects_nonpositive_columns(tmp_path: Path, columns: int) -> None:
    """Explicit column counts below one are rejected."""
    with pytest.raises(ValueError, match="columns must be positive"):
        frame_export.write_filmstrip([_frame(1)], tmp_path / "x.png", columns=columns)


def test_write_filmstrip_rejects_mismatched_dimensions(tmp_path: Path) -> None:
    """Frames with differing dimensions are rejected before compositing."""
    frames = [_frame(1, (4, 5, 3)), _frame(2, (4, 6, 3))]
    with pytest.raises(ValueError, match="same dimensions"):
        frame_export.write_filmstrip(frames, tmp_path / "x.png")


def test_write_filmstrip_defaults_to_single_row_and_creates_parents(tmp_path: Path) -> None:
    """Without columns the filmstrip is a single row and nested parents are created."""
    frames = [_frame(0, (3, 4, 3)), _frame(255, (3, 4, 3))]  # height=3, width=4
    path = frame_export.write_filmstrip(frames, tmp_path / "nested" / "strip.png")

    assert path == tmp_path / "nested" / "strip.png"
    assert path.exists()
    img = Image.open(path)
    # columns default to len(frames)=2 -> one row; size (2 * width, height) = (8, 3)
    assert img.size == (8, 3)
    arr = np.asarray(img)
    assert (arr[:, :4] == 0).all()  # left cell
    assert (arr[:, 4:] == 255).all()  # right cell


def test_write_filmstrip_grid_layout_with_columns(tmp_path: Path) -> None:
    """A column count below the frame count produces a multi-row grid."""
    frames = [
        np.full((4, 4, 3), 10, dtype=np.uint8),
        np.full((4, 4, 3), 20, dtype=np.uint8),
        np.full((4, 4, 3), 30, dtype=np.uint8),
    ]

    path = frame_export.write_filmstrip(frames, tmp_path / "strip.png", columns=2)

    img = Image.open(path)
    # 2 columns * width 4 -> 8 wide; ceil(3 / 2) = 2 rows * height 4 -> 8 tall.
    assert img.size == (8, 8)
    arr = np.asarray(img)
    assert (arr[:4, :4] == 10).all()  # row 0, col 0
    assert (arr[:4, 4:8] == 20).all()  # row 0, col 1
    assert (arr[4:8, :4] == 30).all()  # row 1, col 0
    assert (arr[4:8, 4:8] == 255).all()  # row 1, col 1 stays white background


# --------------------------------------------------------------------------- #
# export_pickle_frames: orchestration with and without filmstrip
# --------------------------------------------------------------------------- #


def test_export_pickle_frames_orchestrates_without_filmstrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a filmstrip path the writer is skipped and selected indices drive naming."""
    states = ["s0", "s1", "s2", "s3"]
    map_def = object()
    frames = [_frame(1), _frame(2), _frame(3)]

    load_calls: list[str] = []

    def fake_load_states(filename: str) -> tuple[list[str], object]:
        load_calls.append(filename)
        return states, map_def

    monkeypatch.setattr(frame_export, "load_states", fake_load_states)

    render_calls: list[dict[str, Any]] = []

    def fake_render(render_states, render_map, indices, **kwargs):
        render_calls.append(
            {
                "states": list(render_states),
                "map": render_map,
                "indices": list(indices),
                "kwargs": kwargs,
            }
        )
        return frames

    monkeypatch.setattr(frame_export, "render_selected_frames", fake_render)

    filmstrip_calls: list[tuple[tuple, dict]] = []

    def spy_write_filmstrip(*args, **kwargs):
        filmstrip_calls.append((args, kwargs))
        return tmp_path / "should-not-exist.png"

    monkeypatch.setattr(frame_export, "write_filmstrip", spy_write_filmstrip)

    frame_paths, filmstrip = frame_export.export_pickle_frames(
        tmp_path / "rec.pkl", tmp_path / "out", count=3
    )

    assert load_calls == [str(tmp_path / "rec.pkl")]
    # select_evenly_spaced_indices(4, 3) -> [0, 2, 3]
    assert render_calls[0]["indices"] == [0, 2, 3]
    assert render_calls[0]["map"] is map_def
    assert render_calls[0]["states"] == states
    assert render_calls[0]["kwargs"] == {"width": 1280, "height": 720, "scaling": 10}
    assert [p.name for p in frame_paths] == [
        "frame_000000.png",
        "frame_000002.png",
        "frame_000003.png",
    ]
    assert all(path.exists() for path in frame_paths)
    assert filmstrip is None
    assert filmstrip_calls == []


def test_export_pickle_frames_orchestrates_with_filmstrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a filmstrip path the real writer composes a single-row strip from selected frames."""
    states = ["s0", "s1", "s2", "s3"]
    map_def = object()
    frames = [_frame(1), _frame(2), _frame(3)]  # default (h=4, w=5)

    monkeypatch.setattr(frame_export, "load_states", lambda filename: (states, map_def))
    monkeypatch.setattr(frame_export, "render_selected_frames", lambda *args, **kwargs: frames)

    frame_paths, filmstrip = frame_export.export_pickle_frames(
        tmp_path / "rec.pkl",
        tmp_path / "out",
        count=3,
        filmstrip_path=tmp_path / "strip.png",
        filmstrip_columns=3,
    )

    assert all(path.exists() for path in frame_paths)
    assert filmstrip == tmp_path / "strip.png"
    assert filmstrip.exists()
    # 3 columns * width 5 -> 15 wide; 1 row * height 4 -> 4 tall.
    assert Image.open(filmstrip).size == (15, 4)


# --------------------------------------------------------------------------- #
# CLI (main): argument forwarding, output, and validation
# --------------------------------------------------------------------------- #


def test_main_forwards_parsed_args_and_prints_all_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Parsed CLI args forward into export and every produced path is printed."""
    captured: dict[str, Any] = {}

    def fake_export(state_file, output_dir, **kwargs):
        captured["state_file"] = state_file
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        return [
            tmp_path / "frame_000000.png",
            tmp_path / "frame_000005.png",
        ], tmp_path / "strip.png"

    monkeypatch.setattr(frame_export, "export_pickle_frames", fake_export)

    code = frame_export.main(
        [
            str(tmp_path / "rec.pkl"),
            str(tmp_path / "out"),
            "--count",
            "6",
            "--prefix",
            "shot",
            "--filmstrip",
            str(tmp_path / "strip.png"),
            "--filmstrip-columns",
            "3",
            "--width",
            "640",
            "--height",
            "360",
            "--scaling",
            "5",
        ]
    )

    assert code == 0
    assert captured["state_file"] == tmp_path / "rec.pkl"
    assert captured["output_dir"] == tmp_path / "out"
    assert captured["kwargs"] == {
        "count": 6,
        "prefix": "shot",
        "filmstrip_path": tmp_path / "strip.png",
        "filmstrip_columns": 3,
        "render_size": (640, 360),
        "scaling": 5.0,
    }
    out = capsys.readouterr().out
    assert str(tmp_path / "frame_000000.png") in out
    assert str(tmp_path / "frame_000005.png") in out
    assert str(tmp_path / "strip.png") in out


def test_main_prints_only_frame_paths_without_filmstrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a filmstrip the CLI prints exactly the frame paths, one per line."""
    paths = [tmp_path / "frame_000000.png", tmp_path / "frame_000001.png"]
    monkeypatch.setattr(frame_export, "export_pickle_frames", lambda *args, **kwargs: (paths, None))

    code = frame_export.main([str(tmp_path / "rec.pkl"), str(tmp_path / "out")])

    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out == [str(paths[0]), str(paths[1])]


def test_main_requires_both_positional_args(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing a required positional argument makes argparse exit non-zero."""
    monkeypatch.setattr(frame_export, "export_pickle_frames", lambda *args, **kwargs: ([], None))

    with pytest.raises(SystemExit) as exc:
        frame_export.main(["only-one-arg"])

    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()
