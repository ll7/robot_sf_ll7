"""Contract test T015 for `adaptive_sampling_iteration`.

Expectation (final):
  - Returns (done_flag, new_jobs). done_flag True when precision goals met or max episodes reached.

Current state: NotImplementedError expected.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.orchestrator import adaptive_sampling_iteration


def test_adaptive_sampling_iteration():
    class _Scenario:
        def __init__(self):
            self.scenario_id = "scenario_a"
            self.archetype = "crossing"
            self.density = "low"
            self.planned_seeds = [0, 1]

    scenarios = [_Scenario()]

    class _Cfg:
        max_episodes = 10
        batch_size = 5
        smoke = True

    manifest = type("M", (), {})()
    with pytest.raises(NotImplementedError):  # until T028
        adaptive_sampling_iteration([], _Cfg(), scenarios, manifest)
