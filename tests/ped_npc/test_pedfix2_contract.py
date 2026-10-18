"""Review regressions using development seeds and real runtime entry points."""

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.simulator_counterfactual_adapter import SimulatorCounterfactualModel
from robot_sf.gym_env.environment_factory import make_pedestrian_env, make_robot_env
from robot_sf.gym_env.shared_world import SharedWorldRunner
from robot_sf.gym_env.unified_config import MultiRobotConfig, RobotSimulationConfig
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_npc.ped_behavior import FollowRouteBehavior
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import _group_member_probabilities, init_simulators
from robot_sf.training.scenario_loader import load_scenarios

SCENARIOS = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_groups_is_fraction_of_people(fraction):
    """Size-weighted mass, rather than group count, matches the authored fraction."""
    probabilities = np.asarray(_group_member_probabilities(SimulationSettings(groups=fraction)))
    expected_people = np.dot([1, 2, 3], probabilities)
    assert np.dot([0, 2, 3], probabilities) / expected_people == pytest.approx(fraction)
    if fraction > 0:
        assert probabilities[2] / probabilities[1] == pytest.approx(0.3)


@pytest.mark.parametrize("entry", ["robot", "shared_create", "shared_reset", "map", "adapter"])
def test_episode_seed_entry_points_reproduce_crowds(entry, monkeypatch):
    """Each public seeded path repeats real population bytes despite global perturbations."""

    def crowd():
        if entry.startswith("shared"):
            runner = SharedWorldRunner.create(
                world_id="seed-test",
                env_config=MultiRobotConfig(num_robots=2),
                num_robots=2,
                seed=1001,
            )
            if entry == "shared_reset":
                runner.reset(seed=1002)
                first = runner.ped_positions().copy()
                runner.reset(seed=1002)
                np.testing.assert_array_equal(first, runner.ped_positions())
            return runner.ped_positions().copy()
        if entry == "adapter":
            cfg = RobotSimulationConfig(sim_config=SimulationSettings(pedestrian_seed=1001))
            sim = init_simulators(
                cfg, convert_map("maps/svg_maps/classic_doorway.svg"), random_start_pos=False
            )[0]
            model = SimulatorCounterfactualModel(sim)
            return model.snapshot().pysf_state.copy()
        scenario = next(
            s for s in load_scenarios(SCENARIOS) if s["name"] == "classic_group_crossing_high"
        )
        if entry == "map":
            from robot_sf.benchmark.map_runner import map_runner as map_runner_episode
            from robot_sf.benchmark.map_runner.map_runner import _run_map_episode

            frames = []
            factory = map_runner_episode.make_robot_env

            def capture_factory(**kwargs):
                env = factory(**kwargs)
                reset = env.reset

                def capture_reset(**reset_kwargs):
                    result = reset(**reset_kwargs)
                    frames.append(env.simulator.pysf_state.pysf_states().copy())
                    return result

                env.reset = capture_reset
                return env

            with monkeypatch.context() as patch:
                patch.setattr(map_runner_episode, "make_robot_env", capture_factory)
                _run_map_episode(
                    scenario,
                    1001,
                    horizon=1,
                    dt=0.1,
                    record_forces=False,
                    snqi_weights=None,
                    snqi_baseline=None,
                    algo="goal",
                    scenario_path=SCENARIOS,
                )
            return frames[-1]
        env = make_robot_env(
            config=build_env_config(scenario, scenario_path=SCENARIOS), debug=False
        )
        try:
            env.reset(seed=1001)
            return env.simulator.pysf_state.pysf_states().copy()
        finally:
            env.close()

    first = crowd()
    np.random.seed(1003)
    np.random.random(200)
    np.testing.assert_array_equal(first, crowd())


def test_unseeded_ego_resets_advance_but_seeded_reset_replays():
    """Training resets vary the relative spawn while an explicit seed replays it."""
    env = make_pedestrian_env(seed=1001, debug=False)
    try:

        def offset(seed=None):
            env.reset(seed=seed)
            return np.asarray(env.simulator.ego_ped_pos) - env.simulator.robot_pos[0]

        initial = offset(1001)
        repeated = [offset() for _ in range(6)]
        assert len({tuple(np.round(value, 6)) for value in repeated}) == 6
        np.testing.assert_array_equal(initial, offset(1001))
    finally:
        env.close()


def test_snapshot_replays_guarded_respawn_near_robot(monkeypatch):
    """A forced robot overlap consumes the guard stream and restores identical continuation."""
    from robot_sf.ped_npc import ped_behavior

    cfg = RobotSimulationConfig(sim_config=SimulationSettings(pedestrian_seed=1001))
    sim = init_simulators(
        cfg, convert_map("maps/svg_maps/classic_doorway.svg"), random_start_pos=False
    )[0]
    behavior = next(b for b in sim.peds_behaviors if isinstance(b, FollowRouteBehavior))
    gid = next(iter(behavior.navigators))
    center = np.mean(behavior.route_assignments[gid].spawn_zone, axis=0)
    behavior.robot_pose_provider = lambda: [(tuple(center), 0.0)]
    behavior.robot_exclusion_radius = 0.25
    sample = ped_behavior.sample_zone
    guarded_calls = []

    def force_overlap(zone, count, **kwargs):
        if kwargs.get("exclusions"):
            guarded_calls.append(True)
            return sample(zone, count, **kwargs)
        return [tuple(center)] * count

    monkeypatch.setattr(ped_behavior, "sample_zone", force_overlap)
    model = SimulatorCounterfactualModel(sim)
    snapshot = model.snapshot()

    def continuation():
        model.restore(snapshot)
        behavior.respawn_group_at_start(gid)
        frames = [sim.pysf_state.pysf_states().copy()]
        for _ in range(5):
            model.step((0.0, 0.0))
            frames.append(sim.pysf_state.pysf_states().copy())
        return np.asarray(frames)

    first = continuation()
    np.testing.assert_array_equal(first, continuation())
    assert len(guarded_calls) >= 2


def test_shipped_scenario_manifests_build_without_unknown_keys():
    """Build actual scenario manifests; auxiliary contracts/catalogues use other schemas."""
    import yaml

    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    for path in sorted(Path("configs/scenarios").rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict) or not ({"scenarios", "include"} & payload.keys()):
            continue
        for scenario in load_scenarios(path):
            if scenario.get("name") == "pedestrian_emergence_from_spawn_edge":
                # This semantic-boundary validator fixture describes an actor
                # absent from the SVG; it is intentionally not a runnable episode.
                with pytest.raises(ValueError, match="map has no single pedestrians"):
                    build_robot_config_from_scenario(scenario, scenario_path=path)
                continue
            build_robot_config_from_scenario(scenario, scenario_path=path)
    path = Path("tests/fixtures/cli_scenarios/valid_scenario.yaml")
    for scenario in load_scenarios(path):
        build_robot_config_from_scenario(scenario, scenario_path=path)


def test_runtime_written_scenarios_can_be_rebuilt():
    """Real mutation and population-report writers leave replayable loader input."""
    from robot_sf.benchmark.map_runner.map_runner_episode import _prepare_episode_env
    from robot_sf.benchmark.rare_event_sampling import (
        SampledScenarioRow,
        apply_sampled_scenario_mutation,
        parameter_vector_hash,
    )

    scenario = next(
        s for s in load_scenarios(SCENARIOS) if s["name"] == "classic_group_crossing_high"
    )
    parameters = {"crossing_time_offset_s": -0.5, "density": 0.03}
    row = SampledScenarioRow(
        sample_index=0,
        seed=1001,
        parameters=parameters,
        base_probability=1.0,
        proposal_probability=1.0,
        likelihood_ratio=1.0,
        parameter_vector_hash=parameter_vector_hash(parameters),
    )
    scenario = apply_sampled_scenario_mutation(scenario, row)
    scenario["simulation_config"]["population_size"] = 12
    config = build_env_config(scenario, scenario_path=SCENARIOS)
    env = make_robot_env(config=config, seed=1001, debug=False)
    try:
        _prepare_episode_env(
            env,
            seed=1001,
            scenario=scenario,
            planner_bind_env=None,
            planner_reset=None,
            expected_population_size=12,
            pedestrian_control_trace_label_builder=None,
        )
        assert scenario["metadata"]["population_realization"] == {
            "instantiated_population_size": 12,
            "declared_population_size": 12,
        }
        build_env_config(scenario, scenario_path=SCENARIOS)
    finally:
        env.close()


def test_config_hash_excludes_episode_seed_but_retains_groups():
    """Recording identity stays stable across episodes and differs for a group-law change."""
    from robot_sf.gym_env.robot_env import _stable_config_hash

    config = RobotSimulationConfig()
    initial = _stable_config_hash(config)
    for seed in (1001, 1002):
        config.sim_config.pedestrian_seed = seed
        assert _stable_config_hash(config) == initial
    config.sim_config.groups = 0.5
    assert _stable_config_hash(config) != initial


def test_feature_seed_without_episode_seed_avoids_entropy():
    """Legacy route-seeded construction also repeats desired-speed draws."""

    def speeds():
        cfg = RobotSimulationConfig(
            sim_config=SimulationSettings(
                route_spawn_seed=1001,
                desired_speed_mean=0.8,
                desired_speed_std=0.1,
            )
        )
        sim = init_simulators(
            cfg, convert_map("maps/svg_maps/classic_doorway.svg"), random_start_pos=False
        )[0]
        return sim.pysf_sim.peds.max_speeds.copy()

    np.testing.assert_array_equal(speeds(), speeds())


def test_timestep_override_is_normalized_before_episode_horizon():
    """A canonical timestep sets physical duration and stays numeric in the config."""
    scenario = next(
        s for s in load_scenarios(SCENARIOS) if s["name"] == "classic_group_crossing_high"
    )
    scenario["simulation_config"].update(time_per_step_in_secs="0.2", max_episode_steps=5)
    config = build_env_config(scenario, scenario_path=SCENARIOS)
    assert config.sim_config.time_per_step_in_secs == 0.2
    assert config.sim_config.sim_time_in_secs == 1.0
