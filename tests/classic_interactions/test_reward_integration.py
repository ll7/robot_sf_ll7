"""Reward fallback integration test (T012 / FR-013).

Objective: Ensure that the demo path integrates with the environment reward fallback
mechanism (already implemented at env layer) such that each episode summary will
implicitly reflect that reward computation occurred (no None reward path). Since the
current run_demo() does not expose reward totals yet (future T019 extension), this
test encodes a forward-looking assertion that at least one episode is produced and
the environment didn't crash due to missing reward function.

TDD Failure Expectation: episodes list currently empty; will pass once summaries
are implemented. If a future enhancement adds 'total_reward' key, this test can be
extended to assert it's a float. For now we only assert non-empty episodes.
"""

from __future__ import annotations

import importlib


def test_reward_integration_episode_present():
    """Test reward integration episode present.

    Returns:
        Any: Auto-generated placeholder description.
    """
    mod = importlib.import_module("examples.classic_interactions_pygame")
    original_dry = getattr(mod, "DRY_RUN", None)
    mod.DRY_RUN = False  # type: ignore
    try:
        episodes = mod.run_demo()
    finally:
        if original_dry is not None:
            mod.DRY_RUN = original_dry  # type: ignore
    assert episodes, (
        "Expected at least one episode (reward fallback integration) - TDD failing until implementation."
    )
