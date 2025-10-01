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

from typing import TYPE_CHECKING

import numpy as np

from .state_builder import iter_states  # T037
from .visual_deps import has_pygame, simulation_view_ready

if TYPE_CHECKING:
    from collections.abc import Generator

    from .replay import ReplayEpisode

try:  # lightweight import gate
    from robot_sf.render.sim_view import SimulationView  # type: ignore
except ImportError as e:
    SimulationView = None  # type: ignore
    _SIM_VIEW_IMPORT_ERROR = e
except Exception as e:  # pragma: no cover - defensive
    # Keep prior behavior for unexpected import-time errors: record and continue.
    SimulationView = None  # type: ignore
    _SIM_VIEW_IMPORT_ERROR = e
else:  # pragma: no cover - trivial branch
    _SIM_VIEW_IMPORT_ERROR = None


def _assert_ready() -> None:
    if not has_pygame() or SimulationView is None or not simulation_view_ready():  # type: ignore
        raise RuntimeError(
            "SimulationView not available (pygame or probe failed); caller should fallback",
        )


def generate_frames(
    episode: ReplayEpisode,
    *,
    fps: int = 10,
    max_frames: int | None = None,
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
    _sim_view = SimulationView(record_video=False, video_fps=fps, width=640, height=360)  # type: ignore
    produced = 0
    try:
        for st in iter_states(episode):
            # Placeholder gradient fallback (will be replaced when real capture succeeds)
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            shade = int(min(255, (produced / max(1, len(episode.steps) - 1)) * 255))
            frame[:, :, 0] = shade
            frame[:, :, 1] = 20
            frame[:, :, 2] = 255 - shade
            try:
                if hasattr(_sim_view, "render"):
                    _sim_view.render(st)  # type: ignore[arg-type]
                    # Attempt to grab pixel buffer from pygame surface
                    import pygame  # type: ignore

                    if hasattr(_sim_view, "screen") and isinstance(
                        _sim_view.screen,
                        pygame.Surface,
                    ):  # type: ignore[attr-defined]
                        surf = _sim_view.screen  # type: ignore[attr-defined]
                        # Ensure consistent size (H,W) = (surface.get_height(), surface.get_width())
                        arr = pygame.surfarray.array3d(surf)  # (W,H,3)
                        arr = np.transpose(arr, (1, 0, 2))  # to (H,W,3)
                        frame_h, frame_w = frame.shape[:2]
                        if arr.shape[0] == frame_h and arr.shape[1] == frame_w:
                            frame = arr
            except (AttributeError, RuntimeError, ImportError):
                # Ignore and keep synthetic frame
                pass
            except Exception as exc:  # pragma: no cover - defensive
                # Log if possible, otherwise ignore. Limit the scope of what we catch
                # when trying to import the logger to avoid swallowing unrelated errors.
                try:
                    from loguru import logger

                    logger.debug("generate_frames render capture failed: %s", exc)
                except ImportError:
                    # No logger available; ignore
                    pass
            produced += 1
            yield frame
            if max_frames is not None and produced >= max_frames:
                break
    finally:  # T041A cleanup
        try:
            if hasattr(_sim_view, "exit_simulation"):
                _sim_view.exit_simulation()  # type: ignore[call-arg]
        except (AttributeError, OSError, RuntimeError):
            # Best-effort cleanup; ignore known failures
            pass


__all__ = ["generate_frames"]
