"""Contract test T006 for `plan_scenarios`.

Expectation (from contracts):
  - Seeds planned count must equal cfg.initial_episodes for each scenario.

Initial TDD state: Function not implemented yet (raises NotImplementedError),
so this test will FAIL until task T023 completes.
"""

from __future__ import annotations

import random

from robot_sf.benchmark.full_classic.planning import plan_scenarios


def test_plan_scenarios_seed_count(config_factory):  # uses test double BenchmarkConfig
    """Test plan scenarios seed count.

    Args:
        config_factory: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    cfg = config_factory(initial_episodes=7)
    raw = [
        {
            "scenario_id": "scenario_a",
            "archetype": "crossing",
            "density": "low",
            "map_path": "maps/svg_maps/simple_crossing.svg",
            "params": {},
            "max_episode_steps": 200,
        },
    ]
    rng = random.Random(cfg.master_seed)
    scenarios = plan_scenarios(raw, cfg, rng=rng)
    assert len(scenarios) == 1
    assert len(scenarios[0].planned_seeds) == cfg.initial_episodes
