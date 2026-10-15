"""Exercise the shipped predictor-development models through the real runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
    ORCA_PEDESTRIANS_V1,
    SOCIAL_FORCE_DEFAULT,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

SCENARIO_PATH = (
    Path(__file__).resolve().parents[2] / "configs/scenarios/predictor_pedestrian_models.yaml"
)


def test_orca_selector_bypasses_social_force_runtime(monkeypatch) -> None:
    """The independent option never calls the SFM force or integration providers."""
    scenario = deepcopy(load_scenarios(SCENARIO_PATH)[0])
    scenario["simulation_config"]["pedestrian_model"] = "orca_pedestrians_v1"
    config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)
    env = make_robot_env(config, seed=1001)
    try:
        sim = env.unwrapped.simulator

        def forbidden(*_args, **_kwargs):
            pytest.fail("ORCA called a Social Force dynamics provider")

        monkeypatch.setattr(sim.pysf_sim, "compute_forces", forbidden)
        monkeypatch.setattr(sim.pysf_sim, "compute_force_components", forbidden)
        monkeypatch.setattr(sim.pysf_sim.peds, "step", forbidden)
        before = sim.pysf_sim.peds.state.copy()
        sim.step_once([(0.0, 0.0)])
        assert sim.pedestrian_model == "orca_pedestrians_v1"
        assert np.isfinite(sim.pysf_sim.peds.state).all()
        assert np.max(np.abs(before[:, :2] - sim.pysf_sim.peds.state[:, :2])) > 1e-6
    finally:
        env.close()


def test_shipped_pair_changes_only_pedestrian_dynamics() -> None:
    """The loader preserves matched inputs and activates the alternative per scenario."""
    control, alternative, orca = load_scenarios(SCENARIO_PATH)
    assert control["seeds"] == alternative["seeds"] == [1001]
    assert control["map_file"] == alternative["map_file"]
    configs = [
        build_robot_config_from_scenario(row, scenario_path=SCENARIO_PATH)
        for row in (control, alternative)
    ]
    baseline, variant = [asdict(config.sim_config) for config in configs]
    assert baseline.pop("pedestrian_model") == SOCIAL_FORCE_DEFAULT
    assert variant.pop("pedestrian_model") == HSFM_ZANLUNGO_COLLISION_PREDICTION_V1
    assert baseline.pop("zanlungo_collision_prediction")["enabled"] is False
    assert variant.pop("zanlungo_collision_prediction")["enabled"] is True
    assert baseline == variant
    orca_config = build_robot_config_from_scenario(orca, scenario_path=SCENARIO_PATH)
    orca_settings = asdict(orca_config.sim_config)
    control_settings = asdict(configs[0].sim_config)
    assert orca["seeds"] == control["seeds"]
    assert orca["map_file"] == control["map_file"]
    assert orca_settings.pop("pedestrian_model") == ORCA_PEDESTRIANS_V1
    control_settings.pop("pedestrian_model")
    assert orca_settings == control_settings


def test_omitted_and_explicit_default_equal_legacy_steps() -> None:
    """Default dispatch is bitwise equal to direct PySocialForce integration."""
    explicit = load_scenarios(SCENARIO_PATH)[0]
    omitted = deepcopy(explicit)
    del omitted["simulation_config"]["pedestrian_model"]
    traces = []
    for scenario in (omitted, explicit):
        config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)
        env = make_robot_env(config, seed=1001)
        try:
            sim = env.unwrapped.simulator
            assert sim.pedestrian_model == SOCIAL_FORCE_DEFAULT
            assert sim.pysf_sim.peds.state.shape[0] >= 2
            legacy = deepcopy(sim.pysf_sim.peds)
            trace = [sim.pysf_sim.peds.state.copy()]
            for _ in range(10):
                forces = sim.pysf_sim.compute_forces()
                groups = sim.groups.groups_as_lists
                legacy.step(forces, groups)
                sim._step_pedestrians(forces, groups)
                np.testing.assert_array_equal(sim.pysf_sim.peds.state, legacy.state)
                trace.append(sim.pysf_sim.peds.state.copy())
            traces.append(np.stack(trace))
        finally:
            env.close()
    np.testing.assert_array_equal(traces[0], traces[1])


@pytest.mark.parametrize("alternative_index", [1, 2], ids=["zanlungo", "orca"])
def test_alternative_trajectory_is_distinct_finite_and_repeatable(alternative_index) -> None:
    """Real construction and force dispatch activate a repeatable different trajectory."""
    scenarios = load_scenarios(SCENARIO_PATH)
    control, alternative = scenarios[0], scenarios[alternative_index]
    traces = []
    for scenario in (control, alternative, alternative):
        config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)
        env = make_robot_env(config, seed=1001)
        try:
            sim = env.unwrapped.simulator
            trace = [sim.pysf_sim.peds.state.copy()]
            assert trace[0].shape[0] >= 2
            for _ in range(20):
                sim.step_once([(0.0, 0.0)])
                trace.append(sim.pysf_sim.peds.state.copy())
            traces.append(np.stack(trace))
        finally:
            env.close()
    assert all(np.isfinite(trace).all() for trace in traces)
    np.testing.assert_array_equal(traces[0][0], traces[1][0])
    np.testing.assert_array_equal(traces[1], traces[2])
    assert np.max(np.abs(traces[0][1:, :, :2] - traces[1][1:, :, :2])) > 1e-6


@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
def test_orca_reset_replays_current_population(empty) -> None:
    """ORCA has no stale native positions after resets or zero-count construction."""
    scenario = deepcopy(load_scenarios(SCENARIO_PATH)[2])
    if empty:
        scenario["simulation_config"]["population_size"] = 0
        scenario["simulation_config"]["ped_density"] = 0.0
    config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)
    env = make_robot_env(config, seed=1001)
    try:
        traces = []
        for _ in range(2):
            env.reset(seed=1001)
            sim = env.unwrapped.simulator
            initial = sim.pysf_sim.peds.state.copy()
            sim.step_once([(0.0, 0.0)])
            traces.append(sim.pysf_sim.peds.state.copy())
            assert np.isfinite(traces[-1]).all()
            if empty:
                assert initial.shape[0] == traces[-1].shape[0] == 0
        np.testing.assert_array_equal(traces[0], traces[1])
    finally:
        env.close()


def test_orca_rejects_sfm_tracing_and_ego_pedestrian_env() -> None:
    """Unsupported SFM provenance and ego control cannot silently use another model."""
    from robot_sf.gym_env.environment_factory import make_pedestrian_env
    from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
    from robot_sf.sim.sim_config import SimulationSettings

    scenario = deepcopy(load_scenarios(SCENARIO_PATH)[2])
    scenario["simulation_config"]["oracle_force_trace_enabled"] = True
    config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)
    with pytest.raises(ValueError, match="SFM force tracing or residuals"):
        make_robot_env(config, seed=1001)
    ped_config = PedestrianSimulationConfig(
        sim_config=SimulationSettings(pedestrian_model=ORCA_PEDESTRIANS_V1)
    )
    with pytest.raises(ValueError, match="not ego-pedestrian"):
        make_pedestrian_env(ped_config, seed=1001)
