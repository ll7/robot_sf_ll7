"""Smoke test for Classic Interactions PPO visualization demo (T005 / FR-001, FR-004, FR-010).

Current expected behavior (pre-core-impl):
- run_demo() with DRY_RUN=True returns empty list (design placeholder)
We intentionally assert FUTURE behavior (>=1 episode summaries with required keys)
so that this test will initially FAIL (TDD) until core implementation (T016+T019) adds summaries.
"""

from __future__ import annotations

import importlib

REQUIRED_KEYS = {"scenario", "seed", "steps", "outcome", "recorded"}


def _import_demo_module():
    mod = importlib.import_module("examples.classic_interactions_pygame")
    return mod


def test_smoke_run_demo_produces_episode_summaries():
    mod = _import_demo_module()
    # Force non-dry run by temporarily patching DRY_RUN constant if present
    if hasattr(mod, "DRY_RUN"):
        original = mod.DRY_RUN
        mod.DRY_RUN = False  # type: ignore
    else:  # pragma: no cover - defensive
        original = None

    try:
        episodes = mod.run_demo()
    finally:
        if hasattr(mod, "DRY_RUN"):
            mod.DRY_RUN = original  # type: ignore

    # Intentional TDD expectation: we expect at least one episode summary now.
    # This will fail until implementation ensures non-empty results.
    assert episodes, (
        "Expected at least one episode summary (TDD failing state before implementation)."
    )
    # Validate schema keys for first episode
    missing = REQUIRED_KEYS - set(episodes[0].keys())
    assert not missing, f"Episode summary missing keys: {missing}"
