"""Optional dependency detection utilities for visual artifact generation.

Provides lightweight probes for pygame (SimulationView), moviepy and ffmpeg
availability. These are used to decide renderer path or skip reasons without
raising hard import errors inside the core benchmark logic.
"""

from __future__ import annotations

import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def simulation_view_ready() -> bool:  # T030
    """Return True if SimulationView appears usable for rendering.

    Goes beyond a bare import by instantiating a tiny SimulationView once and
    verifying a non‑zero surface size. All work is cached so subsequent calls
    are effectively free. Any exception results in False (graceful degrade).

    Implementation notes:
    - Uses dummy headless surface automatically when SDL_VIDEODRIVER=dummy.
    - Immediately quits pygame to avoid leaving an extra window/context.
    - Avoids retaining the instance to minimize memory footprint.
    """
    if not has_pygame():  # Quick fail path
        return False
    ready = False
    try:  # noqa: SIM105
        import pygame  # type: ignore

        from robot_sf.render.sim_view import SimulationView  # type: ignore

        view = SimulationView(record_video=False)  # lightweight init
        _ = pygame.time.get_ticks  # touch pygame to silence unused import
        width: int = getattr(view.screen, "get_width", lambda: 0)()  # type: ignore[arg-type]
        height: int = getattr(view.screen, "get_height", lambda: 0)()  # type: ignore[arg-type]
        ready = width > 0 and height > 0
    except Exception:  # noqa: BLE001
        return False
    finally:  # Best effort cleanup (safe if pygame not fully init)
        try:  # noqa: SIM105
            pygame.display.quit()  # type: ignore[name-defined]
            pygame.quit()  # type: ignore[name-defined]
        except Exception:  # noqa: BLE001
            pass
    return ready


@lru_cache(maxsize=1)
def has_pygame() -> bool:
    try:  # noqa: SIM105
        import pygame  # type: ignore

        _ = pygame.time.get_ticks  # access attribute to avoid unused warning
    except Exception:  # noqa: BLE001
        return False
    return True


@lru_cache(maxsize=1)
def has_moviepy() -> bool:
    try:  # noqa: SIM105
        import moviepy  # type: ignore

        _ = getattr(moviepy, "__version__", None)
    except Exception:  # noqa: BLE001
        return False
    return True


@lru_cache(maxsize=1)
def ffmpeg_in_path() -> bool:
    return shutil.which("ffmpeg") is not None


@lru_cache(maxsize=1)
def moviepy_ready() -> bool:
    """Return True if moviepy importable and ffmpeg binary appears in PATH.

    We do not attempt an encode test here to avoid overhead; actual encode
    errors will still be caught and converted into failure/skip notes.
    """
    return has_moviepy() and ffmpeg_in_path()


__all__ = [
    "has_pygame",
    "has_moviepy",
    "ffmpeg_in_path",
    "moviepy_ready",
    "simulation_view_ready",
]
