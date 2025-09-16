"""
Pytest configuration to ensure Pygame runs headless during the test session.

This prevents any real OS window from opening when tests create a SimulationView
or otherwise initialize Pygame displays. It mirrors the headless env vars used
in docs and CI, but enforces them automatically for local test runs as well.
"""

from __future__ import annotations

import os
from typing import Dict, Generator, Optional

import pytest


@pytest.fixture(scope="session", autouse=True)
def headless_pygame_environment() -> Generator[None, None, None]:
    """Force headless graphics for the whole pytest session.

    Sets common environment variables so that Pygame uses the dummy video driver
    and no real window is created. This applies to all tests regardless of
    individual test configuration and ensures running tests locally won't pop up
    a window.
    """
    # Save original values to restore after the session
    originals: Dict[str, Optional[str]] = {
        "DISPLAY": os.environ.get("DISPLAY"),
        "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER"),
        "MPLBACKEND": os.environ.get("MPLBACKEND"),
        "SDL_AUDIODRIVER": os.environ.get("SDL_AUDIODRIVER"),
        "PYGAME_HIDE_SUPPORT_PROMPT": os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT"),
    }

    # Enforce headless
    os.environ["DISPLAY"] = ""  # Treat as no display
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Pygame dummy video driver
    os.environ["MPLBACKEND"] = "Agg"  # Non-GUI matplotlib backend
    os.environ["SDL_AUDIODRIVER"] = "dummy"  # Avoid audio device errors
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"  # Cleaner test logs

    # Yield control to the test session
    yield

    # Restore original environment after session completes
    for key, value in originals.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value
