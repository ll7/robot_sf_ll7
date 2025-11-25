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

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.render.sim_view import MOVIEPY_AVAILABLE, VisualizableSimState

from .visual_deps import has_pygame, simulation_view_ready

if TYPE_CHECKING:
    from collections.abc import Generator

    from robot_sf.nav.map_config import MapDefinition

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
    map_def = None
    if getattr(episode, "map_path", None):
        try:
            map_def = convert_map(str(episode.map_path))  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - fallback path
            try:
                from loguru import logger

                logger.debug("convert_map failed for %s: %s", episode.map_path, exc)
            except Exception:
                pass
            map_def = None
    view_kwargs: dict = {"record_video": False, "video_fps": fps, "width": 640, "height": 360}
    if map_def is not None:
        view_kwargs["map_def"] = map_def
        view_kwargs["obstacles"] = getattr(map_def, "obstacles", [])
    _sim_view = SimulationView(**view_kwargs)  # type: ignore[arg-type]
    produced = 0
    dt = episode.dt if episode.dt is not None else (1.0 / fps if fps > 0 else 0.1)
    try:
        for idx, step in enumerate(episode.steps):
            ped_positions = np.asarray(step.ped_positions or [], dtype=float)
            try:
                from robot_sf.render.sim_view import VisualizableSimState

                state = VisualizableSimState(  # type: ignore[call-arg]
                    timestep=idx,
                    robot_action=None,
                    robot_pose=((step.x, step.y), step.heading),
                    pedestrian_positions=ped_positions,
                    ray_vecs=np.zeros((0, 2)),
                    ped_actions=np.zeros_like(ped_positions),
                    time_per_step_in_secs=dt,
                )
            except Exception:
                state = step  # fallback to replay step if construction fails
            try:
                if hasattr(_sim_view, "render"):
                    _sim_view.render(state)  # type: ignore[arg-type]
                    import pygame  # type: ignore

                    if hasattr(_sim_view, "screen") and isinstance(
                        _sim_view.screen,
                        pygame.Surface,
                    ):  # type: ignore[attr-defined]
                        surf = _sim_view.screen  # type: ignore[attr-defined]
                        arr = pygame.surfarray.array3d(surf)  # (W,H,3)
                        frame = np.transpose(arr, (1, 0, 2))  # to (H,W,3)
                    else:
                        frame = np.zeros((360, 640, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
            except (AttributeError, RuntimeError, ImportError):
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
            except Exception as exc:  # pragma: no cover - defensive
                try:
                    from loguru import logger

                    logger.debug("generate_frames render capture failed: %s", exc)
                except ImportError:
                    pass
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
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


def _build_state(step, idx: int, dt: float) -> VisualizableSimState:
    ped_positions = np.asarray(step.ped_positions or [], dtype=float)
    ray_vecs = (
        np.asarray(step.ray_vecs, dtype=float) if step.ray_vecs is not None else np.zeros((0, 2))
    )
    ped_actions = (
        np.asarray(step.ped_actions, dtype=float)
        if step.ped_actions is not None
        else np.zeros_like(ped_positions)
    )
    from robot_sf.render.sim_view import VisualizableAction

    robot_action = None
    if step.robot_goal is not None and step.action is not None:
        robot_action = VisualizableAction(
            ((step.x, step.y), step.heading),
            step.action,
            step.robot_goal,
        )
    return VisualizableSimState(
        timestep=idx,
        robot_action=robot_action,
        robot_pose=((step.x, step.y), step.heading),
        pedestrian_positions=ped_positions,
        ray_vecs=ray_vecs,
        ped_actions=ped_actions,
        time_per_step_in_secs=dt,
    )


def _load_map_def(ep: ReplayEpisode) -> MapDefinition | None:
    # Try to reuse already converted map from episode if present
    if hasattr(ep, "_map_def_cache"):
        return ep._map_def_cache
    map_def = None
    if getattr(ep, "map_path", None):
        try:
            map_def = convert_map(str(ep.map_path))  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - fallback
            map_def = None
    try:
        ep._map_def_cache = map_def
    except Exception:
        pass
    return map_def


def _build_view(episode: ReplayEpisode, fps: int, video_path: str):
    map_def = _load_map_def(episode)
    view_kwargs: dict = {
        "record_video": True,
        "video_path": video_path,
        "video_fps": fps,
        "width": 640,
        "height": 360,
    }
    if map_def is not None:
        view_kwargs["map_def"] = map_def
        view_kwargs["obstacles"] = getattr(map_def, "obstacles", [])
    return SimulationView(**view_kwargs)  # type: ignore[arg-type]


def generate_video_file(
    episode: ReplayEpisode,
    video_path: str,
    *,
    fps: int = 10,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Render a ReplayEpisode via SimulationView's native recorder and write mp4."""
    _assert_ready()
    dt = episode.dt if episode.dt is not None else (1.0 / fps if fps > 0 else 0.1)
    view = _build_view(episode, fps, video_path)
    produced = 0
    status = "failed"
    note: str | None = None
    encode_time = None
    try:
        for idx, step in enumerate(episode.steps):
            state = _build_state(step, idx, dt)
            view.render(state)  # type: ignore[arg-type]
            produced += 1
            if max_frames is not None and produced >= max_frames:
                break
        import time as _time

        t0 = _time.perf_counter()
        view.exit_simulation()  # writes video when record_video=True
        encode_time = _time.perf_counter() - t0
        size = Path(video_path).stat().st_size if Path(video_path).exists() else 0
        status = "success" if size > 0 else "skipped"
        if size == 0:
            note = "moviepy-missing" if not MOVIEPY_AVAILABLE else "encode-empty"
    except Exception as exc:  # pragma: no cover - defensive
        try:
            view.exit_simulation()
        except Exception:
            pass
        status = "failed"
        note = f"render-error:{exc.__class__.__name__}"
    return {"status": status, "note": note, "encode_time": encode_time}


__all__.append("generate_video_file")
