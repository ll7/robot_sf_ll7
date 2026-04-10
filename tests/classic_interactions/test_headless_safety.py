"""Headless safety test (T010 / FR-012).

Ensures that setting SDL_VIDEODRIVER=dummy allows run_demo to proceed without errors.
TDD: expects episodes list to be non-empty (will fail until implementation returns summaries).
"""

from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.slow
def test_headless_dummy_driver_runs(monkeypatch):
    """Run the classic interactions demo in headless mode without crashing."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("ROBOT_SF_FAST_DEMO", "1")
    monkeypatch.setenv("ROBOT_SF_EXAMPLES_MAX_STEPS", "8")
    mod = import_module("examples.classic_interactions_pygame")
    if hasattr(mod, "DRY_RUN"):
        original = mod.DRY_RUN
        mod.DRY_RUN = False  # type: ignore
    else:
        original = None
    try:
        episodes = mod.run_demo(max_episodes=1, enable_recording=False)
    finally:
        if hasattr(mod, "DRY_RUN"):
            mod.DRY_RUN = original  # type: ignore
    assert episodes, (
        "Expected episodes under headless dummy driver (TDD failing until implementation)."
    )
