"""Exercise the shipped predictor-development model pair through the real runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
    SOCIAL_FORCE_DEFAULT,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

SCENARIO_PATH = (
    Path(__file__).resolve().parents[2] / "configs/scenarios/predictor_pedestrian_models.yaml"
)


def test_shipped_pair_changes_only_pedestrian_dynamics() -> None:
    """The loader preserves matched inputs and activates the alternative per scenario."""
    control, alternative = load_scenarios(SCENARIO_PATH)
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


def test_alternative_trajectory_is_distinct_finite_and_repeatable() -> None:
    """Real construction and force dispatch activate a repeatable different trajectory."""
    control, alternative = load_scenarios(SCENARIO_PATH)
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
