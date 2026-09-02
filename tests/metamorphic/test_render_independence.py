"""Render-independence metamorphism for the crowd-only environment."""

from __future__ import annotations

import numpy as np

from tests.metamorphic.support import (
    BASE_MAP,
    EpisodeTrace,
    assert_trace_equal,
    make_env,
    run_episode,
)


class _DeterministicView:
    """Tiny renderer double that exercises the environment render call without SDL."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.screen = np.zeros((2, 2, 3), dtype=np.uint8)
        self.closed = False

    def render(self, state, *, target_fps: float) -> None:
        """Store a frame whose value depends only on the rendered timestep."""
        del target_fps
        frame = np.full((2, 2, 3), int(state.timestep), dtype=np.uint8)
        self.frames.append(frame)
        self.screen = frame

    def exit_simulation(self) -> None:
        """Record the normal environment resource-close callback."""
        self.closed = True


def _capture_rendered_episode() -> tuple[EpisodeTrace, _DeterministicView]:
    """Capture the same bounded trace while rendering at every observation boundary."""
    env = make_env(BASE_MAP, render_mode="rgb_array")
    view = _DeterministicView()
    env._sim_ui = view
    try:
        observation, info = env.reset(seed=8244, options={"map_id": "synthetic"})
        observations = [{key: np.array(value, copy=True) for key, value in observation.items()}]
        infos = [dict(info)]
        frame = env.render()
        assert frame is not None and frame.shape == (2, 2, 3)
        for _ in range(3):
            observation, _reward, _terminated, _truncated, info = env.step()
            observations.append(
                {key: np.array(value, copy=True) for key, value in observation.items()}
            )
            infos.append(dict(info))
            frame = env.render()
            assert frame is not None and frame.shape == (2, 2, 3)
        trace = EpisodeTrace(
            tuple(observations),
            tuple(pedestrian.id for pedestrian in BASE_MAP.single_pedestrians),
            tuple(infos),
        )
        return trace, view
    finally:
        env.close()


def test_rendering_does_not_change_simulation_trace() -> None:
    """Rendering each step must leave the dynamic observation trace unchanged."""
    baseline = run_episode(BASE_MAP)
    rendered, view = _capture_rendered_episode()

    assert len(view.frames) == 4
    assert_trace_equal(baseline, rendered)
