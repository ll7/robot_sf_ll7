"""SimulationView frame generator (T031).

Provides a lazy generator that yields RGB numpy arrays for each step of a
`ReplayEpisode`. This isolates SimulationView usage from higher‑level orchestration
so encoding can remain streaming (FR‑001, FR‑009).

Design goals:
- Lazy: do not accumulate all frames in memory.
- Headless friendly: forces offscreen surface via record_video=True.
- Deterministic: fixed FPS tied to provided cfg or default (10).
- Graceful degradation: if pygame or SimulationView missing, raise RuntimeError;
  caller decides fallback (handled upstream).

Future extensions (not in T031 scope):
- Overlay metadata layers (collisions, goals, etc.) once replay enriched.
- Adaptive cropping or scaling.
"""

from __future__ import annotations

from typing import Generator

import numpy as np

from .replay import ReplayEpisode
from .visual_deps import has_pygame, simulation_view_ready

try:  # lightweight import gate
    from robot_sf.render.sim_view import SimulationView  # type: ignore
except Exception as e:  # noqa: BLE001
    SimulationView = None  # type: ignore
    _SIM_VIEW_IMPORT_ERROR = e
else:  # pragma: no cover - trivial branch
    _SIM_VIEW_IMPORT_ERROR = None


def _assert_ready() -> None:
    if not has_pygame() or SimulationView is None or not simulation_view_ready():  # type: ignore
        raise RuntimeError(
            "SimulationView not available (pygame or probe failed); caller should fallback"
        )


def generate_frames(
    episode: ReplayEpisode, *, fps: int = 10, max_frames: int | None = None
) -> Generator[np.ndarray, None, None]:
    """Yield RGB frames for the given replay episode.

    Parameters
    ----------
    episode: ReplayEpisode
        Episode with validated steps.
    fps: int
        Target frames per second encoded into downstream video; currently used
        for potential future pacing logic (no sleep performed here).
    max_frames: int | None
        Optional cap for smoke / budget control; when set, iteration stops
        after producing at most this many frames.
    """
    _assert_ready()
    # Create SimulationView primarily for future integration & to validate that
    # dependencies are functioning. We don't yet render real state, so the
    # instance is intentionally unused beyond lifecycle side effects.
    _sim_view = SimulationView(record_video=True, video_fps=fps, width=640, height=360)  # type: ignore  # noqa: F841

    produced = 0
    for _ in episode.steps:
        # For now we create a simple placeholder visualization: solid color
        # gradient based on timestep index. A richer reconstruction requires
        # environment state enrichment (future task when available).
        arr = np.zeros((360, 640, 3), dtype=np.uint8)
        shade = int(min(255, (produced / max(1, len(episode.steps) - 1)) * 255))
        arr[:, :, 0] = shade  # Red channel gradient
        arr[:, :, 1] = 20
        arr[:, :, 2] = 255 - shade
        produced += 1
        # Future: when replay contains physical state + poses we will construct a
        # VisualizableSimState and call `_sim_view.render(state)` here. For now we
        # simply keep the SimulationView instance alive (ensuring pygame surfaces
        # persist) and emit a synthetic gradient frame to exercise the video pipeline.
        yield arr
        if max_frames is not None and produced >= max_frames:
            break

    # Explicit cleanup: rely on pygame.quit invoked by probe earlier if needed.
    # We purposely do not call view.exit_simulation() to avoid side effects on global state.


__all__ = ["generate_frames"]
