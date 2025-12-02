"""Outcome summary schema test (T009 / FR-010, FR-011).

Ensures each episode summary contains the required minimal keys.
TDD failure expected until run_demo returns structured summaries.
"""

from __future__ import annotations

import importlib

REQUIRED_KEYS = {"scenario", "seed", "steps", "success", "collision", "timeout"}


def _demo():
    """Demo.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return importlib.import_module("examples.classic_interactions_pygame")


def test_episode_summary_schema():
    """Test episode summary schema.

    Returns:
        Any: Auto-generated placeholder description.
    """
    mod = _demo()
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

    assert episodes, "Expected episodes (TDD failing until implementation)."
    first = episodes[0]
    missing = REQUIRED_KEYS - set(first)
    assert not missing, f"Missing keys in episode summary: {missing}"
