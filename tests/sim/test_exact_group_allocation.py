"""Population-level contracts for opt-in nearest-feasible group allocation."""

import hashlib

import pytest

from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_crowded_zones
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import _group_member_probabilities


def _spawn(n, fraction, maximum=3, mode="exact_small_crowd_v1", seed=1001):
    settings = SimulationSettings(groups=fraction, max_peds_per_group=maximum)
    config = PedSpawnConfig(
        0,
        maximum,
        group_member_probs=_group_member_probabilities(settings),
        force_population_size=n,
        route_spawn_seed=seed,
    )
    # Setting fields after construction also reaches the pre-feature implementation:
    # base ignores the mode and therefore fails on realised membership, not imports.
    config.group_allocation_mode = mode
    config.group_fraction = fraction
    return populate_crowded_zones(config, [((0.0, 0.0), (20.0, 0.0), (20.0, 20.0))])


@pytest.mark.parametrize(
    "n,expected", [(0, (0, 0, 0)), (1, (0, 0, 0)), (2, (0, 2, 2)), (3, (0, 2, 3)), (4, (0, 2, 4))]
)
@pytest.mark.parametrize("index,fraction", enumerate((0.0, 0.5, 1.0)))
def test_small_population_table(n, expected, index, fraction):
    """Nearest feasible count, ties upward, preserves all pedestrians without episodes."""
    state, groups, _ = _spawn(n, fraction)
    assert len(state) == n
    assert sum(len(group) for group in groups if len(group) > 1) == expected[index]
    assert sorted(pid for group in groups for pid in group) == list(range(n))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_half_fraction_within_one_pedestrian(n):
    """Every development seed obeys the count bound, rather than only its mean."""
    for seed in range(1001, 1031):
        _, groups, _ = _spawn(n, 0.5, seed=seed)
        realised = sum(len(g) for g in groups if len(g) > 1)
        assert abs(realised - n * 0.5) <= 1


@pytest.mark.parametrize("maximum", [2, 3, 4, 5])
def test_nearest_feasible_partition(maximum):
    """An independent reachable-sum oracle checks rounding and remainder repairs."""
    for n in range(21):
        reachable = {0}
        for count in range(2, n + 1):
            if any(count - size in reachable for size in range(2, maximum + 1)):
                reachable.add(count)
        for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            expected = min(reachable, key=lambda count: (abs(count - n * fraction), -count))
            state, groups, _ = _spawn(n, fraction, maximum)
            assert len(state) == n
            assert all(1 <= len(g) <= maximum for g in groups)
            assert sum(len(g) for g in groups if len(g) > 1) == expected


def test_seeded_replay_and_legacy_bytes():
    """Replay includes state and membership; default output matches fresh-main bytes."""
    for mode in ("legacy", "exact_small_crowd_v1"):
        first = _spawn(12, 0.5, mode=mode)
        second = _spawn(12, 0.5, mode=mode)
        assert first[0].tobytes() == second[0].tobytes()
        assert first[1:] == second[1:]
    state, groups, zones = _spawn(12, 0.5, mode="legacy")
    assert (
        hashlib.sha256(state.tobytes()).hexdigest()
        == "5b97cae6653cbf7d2af25151391760a2eadf651c8e456cd9f7bb57cdc046a247"
    )
    assert [sorted(g) for g in groups] == [[0], [1], [2], [3], [4, 5, 6], [7], [8], [9], [10], [11]]
    assert len(zones) == 12


def test_scenario_override_reaches_spawner():
    """The public scenario setting must select the law used by actual resets."""
    from pathlib import Path

    from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
    from robot_sf.gym_env.environment_factory import make_robot_env
    from robot_sf.training.scenario_loader import load_scenarios

    path = Path("configs/scenarios/archetypes/classic_group_crossing.yaml")
    scenario = dict(load_scenarios(path)[0])
    scenario["simulation_config"] = dict(
        scenario.get("simulation_config", {}),
        groups=0.5,
        group_allocation_mode="exact_small_crowd_v1",
        population_size=4,
    )
    env = make_robot_env(build_env_config(scenario, scenario_path=path), seed=1001)
    try:
        env.reset(seed=1001)
        sim = env.unwrapped.simulator
        assert len(sim.ped_pos) == 4
        assert sum(len(g) for g in sim.groups.groups_as_lists if len(g) > 1) == 2
    finally:
        env.close()


def test_group_mode_preserves_default_bytes_and_explicit_identity():
    """Legacy omits the new key; exact mode survives serialization, copying and hashing."""
    from dataclasses import asdict, replace

    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.gym_env.robot_env import _stable_config_hash

    legacy = SimulationSettings(groups=0.5)
    exact = SimulationSettings(groups=0.5)
    exact.group_allocation_mode = "exact_small_crowd_v1"
    assert "group_allocation_mode" not in asdict(legacy)
    assert "group_allocation_mode" not in legacy.to_dict()
    assert exact.to_dict().get("group_allocation_mode") == "exact_small_crowd_v1"
    assert SimulationSettings(**exact.to_dict()) == exact
    assert replace(exact).group_allocation_mode == "exact_small_crowd_v1"
    assert _stable_config_hash(EnvSettings(sim_config=exact)) != _stable_config_hash(
        EnvSettings(sim_config=legacy)
    )
    explicitly_legacy = replace(legacy, group_allocation_mode="legacy")
    assert _stable_config_hash(EnvSettings(sim_config=explicitly_legacy)) == _stable_config_hash(
        EnvSettings(sim_config=legacy)
    )
