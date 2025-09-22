"""Performance smoke test (T015 / FR-015).

Ensures that a single demo run (with minimal episodes) completes within a soft
performance threshold. TDD: Fails currently because run_demo returns empty list
and we assert episodes exist. Once implementation is in place, also guards against
accidentally slow regressions.
"""

from __future__ import annotations

import importlib
import time

SOFT_THRESHOLD_SECONDS = 3.0  # configurable if needed


def test_demo_runtime_under_threshold():  # noqa: D401
    mod = importlib.import_module("examples.classic_interactions_pygame")
    original_dry = getattr(mod, "DRY_RUN", None)
    original_max = getattr(mod, "MAX_EPISODES", None)
    mod.DRY_RUN = False  # type: ignore
    mod.MAX_EPISODES = 1  # type: ignore
    start = time.time()
    try:
        episodes = mod.run_demo()
    finally:
        if original_dry is not None:
            mod.DRY_RUN = original_dry  # type: ignore
        if original_max is not None:
            mod.MAX_EPISODES = original_max  # type: ignore
    elapsed = time.time() - start
    assert episodes, "Expected at least one episode (TDD failing until implementation)."
    assert elapsed < SOFT_THRESHOLD_SECONDS, (
        f"Demo run exceeded soft threshold {SOFT_THRESHOLD_SECONDS:.1f}s (elapsed={elapsed:.2f}s)."
    )
