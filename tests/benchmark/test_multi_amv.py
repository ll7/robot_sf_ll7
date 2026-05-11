"""Tests for minimal multi-AMV benchmark helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.multi_amv import (
    MultiAmvSettings,
    inter_robot_metrics,
    multi_amv_settings_from_scenario,
)
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.gym_env.unified_config import MultiRobotConfig, RobotSimulationConfig
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_multi_amv_smoke import _multi_robot_config_from_scenario


def test_multi_amv_settings_from_scenario_validates_thresholds() -> None:
    """Scenario multi-AMV settings should fail closed for inconsistent thresholds."""
    settings = multi_amv_settings_from_scenario(
        {
            "multi_amv": {
                "num_robots": 2,
                "collision_distance_m": 0.4,
                "near_miss_distance_m": 1.2,
                "deadlock_speed_mps": 0.05,
                "deadlock_window_steps": 8,
            }
        }
    )

    assert settings.num_robots == 2
    assert settings.near_miss_distance_m == pytest.approx(1.2)

    with pytest.raises(ValueError, match="near_miss_distance_m"):
        multi_amv_settings_from_scenario(
            {"multi_amv": {"collision_distance_m": 1.0, "near_miss_distance_m": 0.5}}
        )


def test_inter_robot_metrics_counts_near_miss_collision_and_deadlock() -> None:
    """Inter-robot metric block should expose discrete pairwise encounters."""
    positions = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.8, 0.0]],
            [[0.0, 0.0], [0.3, 0.0]],
            [[0.0, 0.0], [0.3, 0.0]],
            [[0.0, 0.0], [0.3, 0.0]],
        ],
        dtype=float,
    )
    metrics = inter_robot_metrics(
        positions,
        dt=1.0,
        settings=MultiAmvSettings(
            num_robots=2,
            collision_distance_m=0.4,
            near_miss_distance_m=1.0,
            deadlock_speed_mps=0.05,
            deadlock_window_steps=2,
        ),
    )

    assert metrics["robot_count"] == pytest.approx(2.0)
    assert metrics["pair_count"] == pytest.approx(1.0)
    assert metrics["min_inter_robot_distance_m"] == pytest.approx(0.3)
    assert metrics["inter_robot_near_miss_events"] == pytest.approx(1.0)
    assert metrics["inter_robot_collision_events"] == pytest.approx(1.0)
    assert metrics["deadlock_detected"] is True


def test_inter_robot_metrics_handles_empty_trajectory() -> None:
    """Empty multi-robot trajectories should fail closed without crashing."""

    metrics = inter_robot_metrics(
        np.zeros((0, 2, 2), dtype=float),
        dt=1.0,
        settings=MultiAmvSettings(num_robots=2),
    )

    assert metrics["robot_count"] == pytest.approx(2.0)
    assert metrics["pair_count"] == pytest.approx(1.0)
    assert np.isnan(metrics["min_inter_robot_distance_m"])
    assert metrics["inter_robot_collision_events"] == pytest.approx(0.0)
    assert metrics["inter_robot_near_miss_events"] == pytest.approx(0.0)
    assert metrics["deadlock_detected"] is False


def test_multi_amv_smoke_scenario_passes_schema() -> None:
    """Tracked minimal multi-AMV scenario should be accepted by the benchmark schema."""
    scenario_path = Path("configs/scenarios/single/multi_amv_minimal_smoke.yaml")
    scenarios = [dict(scenario) for scenario in load_scenarios(scenario_path)]

    assert not validate_scenario_list(scenarios)
    assert scenarios[0]["multi_amv"]["num_robots"] == 2


def test_multi_robot_config_from_scenario_preserves_base_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Smoke config conversion should copy base config fields explicitly and override robots."""

    base = RobotSimulationConfig()
    base.map_id = "multi-amv-test-map"
    base.use_planner = True
    base.planner_clearance_margin = 0.75

    monkeypatch.setattr(
        "scripts.validation.run_multi_amv_smoke.build_robot_config_from_scenario",
        lambda scenario, scenario_path: base,
    )

    config = _multi_robot_config_from_scenario(
        {"multi_amv": {"num_robots": 3}},
        tmp_path / "scenario.yaml",
    )

    assert isinstance(config, MultiRobotConfig)
    assert config.map_id == "multi-amv-test-map"
    assert config.use_planner is True
    assert config.planner_clearance_margin == pytest.approx(0.75)
    assert config.num_robots == 3
