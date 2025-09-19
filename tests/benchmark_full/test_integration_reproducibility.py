"""Integration test T021: reproducibility of episode_ids.

Expectation (future): Two consecutive runs with identical seed and config
produce identical sets of episode_ids (order acceptable). This test will later
compare extracted ids; for now just asserts NotImplementedError.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_reproducibility_same_seed(config_factory):
    cfg = config_factory(smoke=True, master_seed=123)
    with pytest.raises(NotImplementedError):  # until end-to-end implemented
        run_full_benchmark(cfg)
