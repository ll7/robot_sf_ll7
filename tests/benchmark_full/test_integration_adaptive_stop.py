"""Integration test T018: adaptive early stop.

Expectation (future): configure very low max_episodes and thresholds so
adaptive loop stops early (done_flag True before hitting max) and manifest
records fewer than max episodes.

Current state: run_full_benchmark not implemented -> expect NotImplementedError.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_adaptive_early_stop(config_factory):
    cfg = config_factory(smoke=True, workers=1, max_episodes=20, initial_episodes=5, batch_size=5)
    with pytest.raises(NotImplementedError):  # until T029/T034
        run_full_benchmark(cfg)
