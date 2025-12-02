"""Contract test T007 for `expand_episode_jobs`.

Expectations:
  - Number of EpisodeJob objects equals total planned seeds across scenarios.
  - Horizon override in cfg is applied to each job's horizon field.

Current state: expand_episode_jobs not implemented (raises NotImplementedError)
so this test will FAIL until task T024 is completed.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot_sf.benchmark.full_classic import planning


@dataclass
class _Cfg:
    """Cfg class."""

    horizon_override: int | None = 300


def _scenario_descriptor(seed_count: int = 3):
    """Scenario descriptor.

    Args:
        seed_count: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return planning.ScenarioDescriptor(
        scenario_id="scenario_a",
        archetype="crossing",
        density="low",
        map_path="maps/svg_maps/simple_crossing.svg",
        params={},
        raw={},
        planned_seeds=list(range(seed_count)),
        max_episode_steps=500,
        hash_fragment="abc123",
    )


def test_expand_jobs_count_and_horizon():
    """Test expand jobs count and horizon.

    Returns:
        Any: Auto-generated placeholder description.
    """
    cfg = _Cfg(horizon_override=250)
    scenarios = [_scenario_descriptor(seed_count=4)]
    jobs = planning.expand_episode_jobs(scenarios, cfg)
    assert len(jobs) == 4
    horizons = {j.horizon for j in jobs}
    assert horizons == {250}
