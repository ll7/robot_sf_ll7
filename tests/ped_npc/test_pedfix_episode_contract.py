"""Development-seed regressions for planner-independent crowds and safe resets."""

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios

SCENARIOS = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")


def _env(name, seed):
    """Build the real release scenario on an explicitly permitted development seed."""
    assert 1001 <= seed <= 1030
    scenario = next(s for s in load_scenarios(SCENARIOS) if s["name"] == name)
    config = build_env_config(scenario, scenario_path=SCENARIOS)
    env = make_robot_env(config=config, seed=seed, debug=False)
    env.reset(seed=seed)
    return env


@pytest.mark.parametrize("perturbation", ["draw", "reseed"])
def test_trajectories_ignore_global_numpy_after_respawn(perturbation):
    """A planner's NumPy draws or SICNav-style reseeds cannot change route respawns."""

    def trajectory(mode):
        env = _env("classic_head_on_corridor_medium", 1003)
        try:
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            frames = [env.simulator.pysf_sim.peds.pos().copy()]
            for step in range(400):
                if mode == "draw":
                    np.random.random(17)
                elif mode == "reseed":
                    np.random.seed(1001 + step % 30)
                env.step(action)
                frames.append(env.simulator.pysf_sim.peds.pos().copy())
            return np.asarray(frames)
        finally:
            env.close()

    reference = trajectory("none")
    assert np.any(np.linalg.norm(np.diff(reference, axis=0), axis=2) > 1.0), "must cover respawn"
    np.testing.assert_array_equal(reference, trajectory(perturbation))


def test_groups_override_changes_real_population():
    """groups=1 must produce multi-person groups, unlike the old ignored setting."""
    scenario = next(
        s for s in load_scenarios(SCENARIOS) if s["name"] == "classic_group_crossing_high"
    )
    scenario["simulation_config"] = {**scenario["simulation_config"], "groups": 1.0}
    cfg = build_env_config(scenario, scenario_path=SCENARIOS)
    env = make_robot_env(config=cfg, seed=1001, debug=False)
    try:
        env.reset(seed=1001)
        sizes = [len(g) for g in env.simulator.groups.groups.values()]
        assert len(sizes) > 1
        assert sizes.count(1) <= 1, sizes  # only a final population remainder may be singleton
    finally:
        env.close()


def test_unknown_simulation_key_fails_at_real_loader():
    """A misspelled configuration cannot silently run the default simulation."""
    scenario = next(
        s for s in load_scenarios(SCENARIOS) if s["name"] == "classic_group_crossing_high"
    )
    scenario["simulation_config"] = {**scenario["simulation_config"], "gruops": 0.5}
    with pytest.raises(ValueError, match="simulation_config.*unknown.*gruops"):
        build_env_config(scenario, scenario_path=SCENARIOS)


def test_relocation_has_reaction_clearance_and_route_velocity():
    """Force an overlap to exercise relocation even if future RNG streams move spawns."""
    env = _env("francis2023_circular_crossing", 1001)
    try:
        sim = env.simulator
        state = sim.pysf_state.pysf_states()
        robot = np.asarray(sim.robot_pos[0])
        state[0, :2] = robot + [0.1, 0.0]
        state[0, 2:4] = [0.5, 0.0]
        state[0, 4:6] = robot + [-5.0, 0.0]
        speed_cap = float(sim.pysf_sim.peds.max_speeds[0])
        sim._enforce_reset_spawn_clearance()
        assert 0 in sim.last_spawn_relocation.relocated
        position, velocity, goal = state[0, :2], state[0, 2:4], state[0, 4:6]
        required = sim.robots[0].config.radius + sim.config.ped_radius + 0.1 + max(speed_cap, 0.5)
        assert np.linalg.norm(position - robot) >= required
        expected_velocity = (goal - position) / np.linalg.norm(goal - position) * 0.5
        np.testing.assert_allclose(velocity, expected_velocity)
        assert np.dot(velocity, position - robot) >= -1e-9
    finally:
        env.close()


def test_vendored_population_leaves_global_numpy_untouched():
    """The standalone fast-pysf population API also owns all sampling draws."""
    from pysocialforce.ped_population import PedSpawnConfig, populate_simulation

    zone = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    np.random.seed(1001)
    before = np.random.get_state()
    states, _groups, behaviors = populate_simulation(0.5, PedSpawnConfig(), [], [zone])
    for behavior in behaviors:
        behavior.reset()
    after = np.random.get_state()
    assert len(states.raw_states) > 0
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_relocation_keeps_reaction_buffer_when_goal_is_inside_robot():
    """A closing route heading cannot justify leaving an overlapping pedestrian unmoved."""
    env = _env("francis2023_circular_crossing", 1001)
    try:
        sim = env.simulator
        state = sim.pysf_state.pysf_states()
        robot = np.asarray(sim.robot_pos[0])
        state[0, :2] = robot + [0.1, 0.0]
        state[0, 2:4] = [0.5, 0.0]
        state[0, 4:6] = robot
        sim._enforce_reset_spawn_clearance()
        assert 0 in sim.last_spawn_relocation.relocated
        assert np.linalg.norm(state[0, :2] - robot) >= 2.15
        assert np.dot(state[0, 2:4], robot - state[0, :2]) > 0.0
    finally:
        env.close()
