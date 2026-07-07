"""Tests for headless frame export from pickle recordings."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from robot_sf.common.safe_pickle import UnsafePickleError
from robot_sf.render import frame_export
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_state import VisualizableSimState
from robot_sf.render.sim_view import _empty_map_definition

if TYPE_CHECKING:
    from pathlib import Path


def _state(timestep: int = 0) -> VisualizableSimState:
    """Return a small renderable state for pickle/export tests."""
    return VisualizableSimState(
        timestep=timestep,
        robot_action=None,
        robot_pose=((0.5, 0.5), 0.0),
        pedestrian_positions=np.array([[0.25, 0.25]], dtype=float),
        ray_vecs=np.zeros((0, 2), dtype=float),
        ped_actions=np.array([[[0.25, 0.25], [0.35, 0.25]]], dtype=float),
    )


def test_load_states_accepts_base_visualizable_sim_state(tmp_path: Path) -> None:
    """Fresh pickle states using the base sim_state dataclass are valid."""
    pickle_path = tmp_path / "states.pkl"
    states = [_state()]
    map_def = _empty_map_definition()
    with pickle_path.open("wb") as handle:
        pickle.dump((states, map_def), handle)

    loaded_states, loaded_map = load_states(str(pickle_path))

    assert isinstance(loaded_states[0], VisualizableSimState)
    assert loaded_states[0].timestep == states[0].timestep
    np.testing.assert_array_equal(
        loaded_states[0].pedestrian_positions,
        states[0].pedestrian_positions,
    )
    assert loaded_map.width == map_def.width
    assert loaded_map.height == map_def.height


def test_load_states_rejects_arbitrary_objects(tmp_path: Path) -> None:
    """Arbitrary objects are rejected by the restricted unpickler.

    Since #4767, ``load_states`` loads through a restricted unpickler that
    allowlists known-safe globals, so an arbitrary ``object()`` is rejected at
    unpickle time (``UnsafePickleError``) — before the post-load type check that
    previously raised ``TypeError('Invalid states')``.
    """
    pickle_path = tmp_path / "states.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(([object()], _empty_map_definition()), handle)

    with pytest.raises(UnsafePickleError, match="builtins.object"):
        load_states(str(pickle_path))


def test_select_evenly_spaced_indices_spans_recording() -> None:
    """Evenly spaced selection spans endpoints and avoids duplicate requests."""
    assert frame_export.select_evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert frame_export.select_evenly_spaced_indices(3, 8) == [0, 1, 2]
    assert frame_export.select_evenly_spaced_indices(0, 4) == []


def test_write_png_frames_and_filmstrip(tmp_path: Path) -> None:
    """PNG frame and filmstrip writers produce inspectable image files."""
    frames = [
        np.full((2, 3, 3), fill_value=0, dtype=np.uint8),
        np.full((2, 3, 3), fill_value=255, dtype=np.uint8),
    ]

    frame_paths = frame_export.write_png_frames(
        frames,
        tmp_path / "frames",
        prefix="sample",
        indices=[0, 9],
    )
    filmstrip_path = frame_export.write_filmstrip(
        frames,
        tmp_path / "filmstrip.png",
        columns=2,
    )

    assert [path.name for path in frame_paths] == ["sample_000000.png", "sample_000009.png"]
    assert all(path.exists() for path in frame_paths)
    assert Image.open(frame_paths[0]).size == (3, 2)
    assert Image.open(filmstrip_path).size == (6, 2)


def test_export_pickle_frames_uses_pickle_map_and_selected_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """High-level export loads pickle payloads and forwards selected states/map."""
    states = [_state(0), _state(1), _state(2)]
    map_def = _empty_map_definition()
    pickle_path = tmp_path / "states.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump((states, map_def), handle)

    calls: dict[str, object] = {}

    def fake_render_selected_frames(render_states, render_map, indices, **kwargs):
        calls["states"] = render_states
        calls["map"] = render_map
        calls["indices"] = list(indices)
        calls["kwargs"] = kwargs
        return [np.zeros((2, 2, 3), dtype=np.uint8) for _ in indices]

    monkeypatch.setattr(frame_export, "render_selected_frames", fake_render_selected_frames)

    frame_paths, filmstrip_path = frame_export.export_pickle_frames(
        pickle_path,
        tmp_path / "frames",
        count=2,
        filmstrip_path=tmp_path / "filmstrip.png",
        render_size=(160, 90),
    )

    render_states = calls["states"]
    assert [state.timestep for state in render_states] == [0, 1, 2]
    render_map = calls["map"]
    assert render_map.width == map_def.width
    assert render_map.height == map_def.height
    assert calls["indices"] == [0, 2]
    assert calls["kwargs"] == {"width": 160, "height": 90, "scaling": 10}
    assert [path.name for path in frame_paths] == ["frame_000000.png", "frame_000002.png"]
    assert filmstrip_path == tmp_path / "filmstrip.png"
