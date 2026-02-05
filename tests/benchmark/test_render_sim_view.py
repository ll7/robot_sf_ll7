"""Tests for SimulationView rendering helpers with lightweight stubs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.full_classic import render_sim_view


def _step(**kwargs):
    defaults = {
        "x": 0.0,
        "y": 0.0,
        "heading": 0.0,
        "ped_positions": [],
        "ray_vecs": None,
        "ped_actions": None,
        "robot_goal": None,
        "action": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _episode(steps, dt=0.1):
    return SimpleNamespace(steps=steps, dt=dt, map_path=None)


def test_build_state_constructs_visualizable_action() -> None:
    """Ensure VisualizableSimState includes robot action when provided."""
    step = _step(robot_goal=(1.0, 1.0), action=np.array([0.1, 0.0]))
    state = render_sim_view._build_state(step, idx=0, dt=0.1)
    assert state.robot_action is not None
    assert state.robot_pose == ((step.x, step.y), step.heading)


def test_generate_frames_uses_dummy_view(monkeypatch) -> None:
    """Verify frame generator yields fixed-size RGB arrays."""

    class _DummyView:
        def __init__(self, **_kwargs) -> None:
            self.kwargs = _kwargs

    monkeypatch.setattr(render_sim_view, "_assert_ready", lambda: None)
    monkeypatch.setattr(render_sim_view, "SimulationView", _DummyView)

    episode = _episode([_step()])
    frames = list(render_sim_view.generate_frames(episode, fps=5, max_frames=1))
    assert len(frames) == 1
    assert frames[0].shape == (360, 640, 3)
    assert frames[0].dtype == np.uint8


def test_generate_video_file_writes_stub_video(tmp_path, monkeypatch) -> None:
    """Check video export helper reports success when file is written."""

    class _DummyView:
        def __init__(self, video_path: str) -> None:
            self._video_path = video_path

        def render(self, _state) -> None:
            return None

        def exit_simulation(self) -> None:
            with open(self._video_path, "wb") as handle:
                handle.write(b"1")

    def _build_view(_episode, _fps, video_path: str):
        return _DummyView(video_path)

    monkeypatch.setattr(render_sim_view, "_assert_ready", lambda: None)
    monkeypatch.setattr(render_sim_view, "_build_view", _build_view)

    episode = _episode([_step()], dt=0.1)
    out_path = tmp_path / "video.mp4"
    result = render_sim_view.generate_video_file(episode, str(out_path), fps=5, max_frames=1)
    assert result["status"] == "success"
    assert out_path.exists()


def test_assert_ready_raises_when_unavailable(monkeypatch) -> None:
    """Ensure readiness guard fails when dependencies are missing."""
    monkeypatch.setattr(render_sim_view, "SimulationView", None)
    monkeypatch.setattr(render_sim_view, "has_pygame", lambda: False)
    monkeypatch.setattr(render_sim_view, "simulation_view_ready", lambda: False)
    with pytest.raises(RuntimeError):
        render_sim_view._assert_ready()


def test_load_map_def_cache(monkeypatch) -> None:
    """Verify map definition caching on episodes."""
    calls = {"count": 0}

    def _convert_map(path: str):
        calls["count"] += 1
        return {"map": path}

    monkeypatch.setattr(render_sim_view, "convert_map", _convert_map)
    episode = SimpleNamespace(steps=[], dt=0.1, map_path="maps/test.svg")
    first = render_sim_view._load_map_def(episode)
    second = render_sim_view._load_map_def(episode)
    assert first == second
    assert calls["count"] == 1


def test_build_view_includes_map_def(monkeypatch) -> None:
    """Ensure build_view passes map_def and obstacles into SimulationView."""

    class _DummyView:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    dummy_map = SimpleNamespace(obstacles=["obs"])
    monkeypatch.setattr(render_sim_view, "_load_map_def", lambda ep: dummy_map)
    monkeypatch.setattr(render_sim_view, "SimulationView", _DummyView)
    episode = SimpleNamespace(steps=[], dt=0.1, map_path="maps/test.svg")
    view = render_sim_view._build_view(episode, fps=5, video_path="out.mp4")
    assert view.kwargs["map_def"] == dummy_map
    assert view.kwargs["obstacles"] == ["obs"]


def test_generate_video_file_skipped_when_empty(tmp_path, monkeypatch) -> None:
    """Verify empty video output reports a skipped status."""

    class _DummyView:
        def __init__(self, video_path: str) -> None:
            self._video_path = video_path

        def render(self, _state) -> None:
            return None

        def exit_simulation(self) -> None:
            Path(self._video_path).touch()

    def _build_view(_episode, _fps, video_path: str):
        return _DummyView(video_path)

    monkeypatch.setattr(render_sim_view, "_assert_ready", lambda: None)
    monkeypatch.setattr(render_sim_view, "_build_view", _build_view)
    monkeypatch.setattr(render_sim_view, "MOVIEPY_AVAILABLE", False)

    episode = _episode([_step()], dt=0.1)
    out_path = tmp_path / "video.mp4"
    result = render_sim_view.generate_video_file(episode, str(out_path), fps=5, max_frames=1)
    assert result["status"] == "skipped"
    assert result["note"] == "moviepy-missing"


def test_load_map_def_returns_none_without_path() -> None:
    """Ensure map loader returns None when no map path is provided."""
    episode = SimpleNamespace(steps=[], dt=0.1, map_path=None)
    assert render_sim_view._load_map_def(episode) is None


def test_generate_frames_uses_screen_surface(monkeypatch) -> None:
    """Exercise the pygame surface capture branch."""

    class DummySurface:
        pass

    class DummyPygame:
        Surface = DummySurface

        class surfarray:
            @staticmethod
            def array3d(_surf):
                return np.zeros((640, 360, 3), dtype=np.uint8)

    class _DummyView:
        def __init__(self, **_kwargs) -> None:
            self.screen = DummySurface()

        def render(self, _state) -> None:
            return None

    original_import = render_sim_view.importlib.import_module

    def _import_module(name: str):
        if name == "pygame":
            return DummyPygame
        return original_import(name)

    monkeypatch.setattr(render_sim_view, "_assert_ready", lambda: None)
    monkeypatch.setattr(render_sim_view, "SimulationView", _DummyView)
    monkeypatch.setattr(render_sim_view.importlib, "import_module", _import_module)

    episode = _episode([_step()])
    frames = list(render_sim_view.generate_frames(episode, fps=5, max_frames=1))
    assert frames[0].shape == (360, 640, 3)
