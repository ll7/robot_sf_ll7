"""Optional dependency detection utilities for visual artifact generation.

Provides lightweight probes for pygame (SimulationView), moviepy and ffmpeg
availability. These are used to decide renderer path or skip reasons without
raising hard import errors inside the core benchmark logic.
"""

from __future__ import annotations

import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def has_pygame() -> bool:
    try:  # noqa: SIM105
        import pygame  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@lru_cache(maxsize=1)
def has_moviepy() -> bool:
    try:  # noqa: SIM105
        import moviepy  # type: ignore  # noqa: F401
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
]
