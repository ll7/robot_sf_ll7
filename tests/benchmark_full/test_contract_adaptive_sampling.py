"""Contract test T015 for `adaptive_sampling_iteration`.

Expectation (final):
  - Returns (done_flag, new_jobs). done_flag True when precision goals met or max episodes reached.

Current state: NotImplementedError expected.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.orchestrator import adaptive_sampling_iteration


def test_adaptive_sampling_iteration():
    """Test adaptive sampling iteration.

    Returns:
        Any: Auto-generated placeholder description.
    """

    class _Scenario:
        """Scenario class."""

        def __init__(self):
            """Init.

            Returns:
                Any: Auto-generated placeholder description.
            """
            self.scenario_id = "scenario_a"
            self.archetype = "crossing"
            self.density = "low"
            self.planned_seeds = [0, 1]

    scenarios = [_Scenario()]

    class _Cfg:
        """Cfg class."""

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
