"""Headless safety test (T010 / FR-012).

Ensures that setting SDL_VIDEODRIVER=dummy allows run_demo to proceed without errors.
TDD: expects episodes list to be non-empty (will fail until implementation returns summaries).
"""

from __future__ import annotations

import importlib
import os


def test_headless_dummy_driver_runs():
    """TODO docstring. Document this function."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    mod = importlib.import_module("examples.classic_interactions_pygame")
    if hasattr(mod, "DRY_RUN"):
        original = mod.DRY_RUN
        mod.DRY_RUN = False  # type: ignore
    else:
        original = None
    try:
        episodes = mod.run_demo()
    finally:
        if hasattr(mod, "DRY_RUN"):
            mod.DRY_RUN = original  # type: ignore
    assert episodes, (
        "Expected episodes under headless dummy driver (TDD failing until implementation)."
    )
