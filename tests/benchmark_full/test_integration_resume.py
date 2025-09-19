"""Integration test T017: resume behavior.

Expectation (eventual):
  - First run creates some episodes.
  - Second run with same config should skip previously completed episodes (executed_jobs smaller on second invocation).

Current state: `run_full_benchmark` not implemented → expect NotImplementedError.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_resume_skips_existing(config_factory):
    cfg = config_factory(smoke=True, workers=1)
    with pytest.raises(NotImplementedError):  # until T029
        run_full_benchmark(cfg)
    # A second call will also raise until implementation exists; kept for structure.
    with pytest.raises(NotImplementedError):
        run_full_benchmark(cfg)
