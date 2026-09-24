"""Contract test T015 for `adaptive_sampling_iteration`.

Expectation (final):
  - Returns (done_flag, new_jobs). done_flag True when precision goals met or max episodes reached.

Current state: NotImplementedError expected.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.orchestrator import adaptive_sampling_iteration


def test_adaptive_sampling_iteration():
    """Verify adaptive_sampling_iteration schedules a batch and stays not done below the cap."""

    class _Scenario:
        """Minimal scenario descriptor with an id, archetype, density, and planned seeds."""

        def __init__(self):
            """Initialize a scenario stub with fixed metadata and two planned seeds."""
            self.scenario_id = "scenario_a"
            self.archetype = "crossing"
            self.density = "low"
            self.planned_seeds = [0, 1]

    scenarios = [_Scenario()]

    class _Cfg:
        """Minimal config stub exposing the episode cap, batch size, and smoke flag."""

        max_episodes = 10
        batch_size = 5
        smoke = True

    manifest = type("M", (), {})()
    done, new_jobs = adaptive_sampling_iteration([], _Cfg(), scenarios, manifest)
    assert done is False
    assert len(new_jobs) > 0
    # Simulate adding returned episodes until cap
    records = [{"scenario_id": scenarios[0].scenario_id} for _ in range(len(new_jobs))]
    done2, _ = adaptive_sampling_iteration(records, _Cfg(), scenarios, manifest)
    assert done2 is False  # Still below max_episodes=10 after one batch of 5
