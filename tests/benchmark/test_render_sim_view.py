"""Tests for SimulationView rendering helpers with lightweight stubs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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
