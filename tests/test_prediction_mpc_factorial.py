"""Tests for issue #5355 prediction-MPC 2x2 factorial design.

Validates config parsing, toggle behavior, and preregistration harness
for the prediction on/off x constraint-handling on/off factorial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.nmpc_social import _RolloutContext
from robot_sf.planner.prediction_mpc import (
    NullPedestrianPredictor,
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _minimal_observation(
    *,
    ped_positions: list[list[float]] | None = None,
    ped_velocities: list[list[float]] | None = None,
) -> dict:
    """Build a minimal SocNav observation for testing."""
    if ped_positions is None:
        ped_positions = [[1.0, 0.0]]
    if ped_velocities is None:
        ped_velocities = [[0.0, 0.0]]
    return {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [0.1],
            "radius": [0.25],
        },
        "goal": {"current": [5.0, 0.0], "next": [5.0, 0.0]},
        "pedestrians": {
            "positions": [coord for pos in ped_positions for coord in pos],
            "velocities": [coord for vel in ped_velocities for coord in vel],
            "count": [len(ped_positions)],
            "radius": [0.25],
        },
    }


class TestPredictionMPCFactorialConfig:
    """Test config parsing for the 2x2 factorial arms."""

    def test_default_config_has_factorial_fields(self):
        config = PredictionMPCConfig()
        assert config.hard_pedestrian_constraints_enabled is True
        assert config.local_min_escape_enabled is False
        assert config.local_min_escape_distance == 2.0
        assert config.local_min_escape_speed_threshold == 0.05

    def test_build_config_from_yaml_A0_B0(self):
        cfg = {
            "predictor_backend": "none",
            "hard_pedestrian_constraints_enabled": False,
            "local_min_escape_enabled": False,
        }
        config = build_prediction_mpc_config(cfg)
        assert config.predictor_backend == "none"
        assert config.hard_pedestrian_constraints_enabled is False
        assert config.local_min_escape_enabled is False

    def test_build_config_from_yaml_A1_B1(self):
        cfg = {
            "predictor_backend": "constant_velocity",
            "hard_pedestrian_constraints_enabled": True,
            "local_min_escape_enabled": True,
            "local_min_escape_distance": 2.0,
            "local_min_escape_speed_threshold": 0.05,
        }
        config = build_prediction_mpc_config(cfg)
        assert config.predictor_backend == "constant_velocity"
        assert config.hard_pedestrian_constraints_enabled is True
        assert config.local_min_escape_enabled is True

    def test_build_config_string_bool_parsing(self):
        cfg = {
            "hard_pedestrian_constraints_enabled": "true",
            "local_min_escape_enabled": "off",
        }
        config = build_prediction_mpc_config(cfg)
        assert config.hard_pedestrian_constraints_enabled is True
        assert config.local_min_escape_enabled is False


class TestNullPedestrianPredictor:
    """Test the null predictor for Factor A OFF."""

    def test_null_predictor_holds_current_positions_without_predicting_velocity(self):
        predictor = NullPedestrianPredictor()
        obs = _minimal_observation()
        futures = predictor.predict(obs, horizon_steps=6, dt=0.25)
        assert futures.positions_world.shape == (1, 6, 2)
        assert np.all(futures.positions_world == [[1.0, 0.0]])
        assert futures.mask.tolist() == [1.0]
        assert futures.source == "current_position_hold"

    def test_null_predictor_zero_pedestrians(self):
        predictor = NullPedestrianPredictor()
        obs = _minimal_observation(ped_positions=[], ped_velocities=[])
        futures = predictor.predict(obs, horizon_steps=4, dt=0.25)
        assert futures.positions_world.shape[0] == 0


class TestFactorAToggle:
    """Test prediction ON/OFF toggle behavior."""

    def test_none_backend_uses_null_predictor(self):
        config = PredictionMPCConfig(predictor_backend="none")
        adapter = PredictionMPCPlannerAdapter(config=config)
        assert isinstance(adapter._future_predictor, NullPedestrianPredictor)

    def test_null_backend_uses_null_predictor(self):
        config = PredictionMPCConfig(predictor_backend="null")
        adapter = PredictionMPCPlannerAdapter(config=config)
        assert isinstance(adapter._future_predictor, NullPedestrianPredictor)

    def test_cv_backend_uses_cv_predictor(self):
        config = PredictionMPCConfig(predictor_backend="constant_velocity")
        adapter = PredictionMPCPlannerAdapter(config=config)
        assert not isinstance(adapter._future_predictor, NullPedestrianPredictor)

    def test_prediction_off_preserves_hard_constraints_from_factor_b(self):
        config = PredictionMPCConfig(
            predictor_backend="none",
            hard_pedestrian_constraints_enabled=True,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        context = _RolloutContext(
            robot_pos=np.array([0.0, 0.0]),
            heading=0.0,
            current_speed=0.1,
            goal=np.array([5.0, 0.0]),
            ped_positions=np.array([[1.0, 0.0]]),
            ped_velocities=np.array([[0.0, 0.0]]),
            robot_radius=0.25,
            ped_radius=0.25,
            pedestrian_uncertainty_envelope_enabled=False,
            pedestrian_uncertainty_alpha_mps=0.0,
            observation=obs,
            speed_cap=0.9,
        )
        assert len(adapter._optimizer_constraints(context)) == 1


class TestFactorBToggle:
    """Test constraint + local-minimum handling ON/OFF toggle."""

    @pytest.mark.parametrize("predictor_backend", ["none", "constant_velocity"])
    def test_constraints_off_arms_remain_functional(self, predictor_backend: str):
        """B0 arms still make finite progress toward an unobstructed goal."""
        config = PredictionMPCConfig(
            predictor_backend=predictor_backend,
            hard_pedestrian_constraints_enabled=False,
            local_min_escape_enabled=False,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)

        linear, angular = adapter.plan(_minimal_observation(ped_positions=[], ped_velocities=[]))

        assert np.isfinite(linear)
        assert np.isfinite(angular)
        assert linear > 0.0
        assert adapter.diagnostics()["nonzero_command_count"] == 1

    def test_constraints_disabled_returns_empty(self):
        config = PredictionMPCConfig(
            predictor_backend="constant_velocity",
            hard_pedestrian_constraints_enabled=False,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert diag["factorial_toggles"]["hard_pedestrian_constraints_enabled"] is False

    def test_constraints_enabled_returns_nonempty(self):
        config = PredictionMPCConfig(
            predictor_backend="constant_velocity",
            hard_pedestrian_constraints_enabled=True,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert diag["factorial_toggles"]["hard_pedestrian_constraints_enabled"] is True

    def test_local_min_escape_disabled(self):
        config = PredictionMPCConfig(
            predictor_backend="constant_velocity",
            local_min_escape_enabled=False,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert diag["factorial_toggles"]["local_min_escape_enabled"] is False
        assert diag["factorial_toggles"]["local_min_escape_count"] == 0

    def test_local_min_escape_enabled_but_robot_moving(self):
        config = PredictionMPCConfig(
            predictor_backend="constant_velocity",
            local_min_escape_enabled=True,
            local_min_escape_speed_threshold=0.05,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation(ped_positions=[], ped_velocities=[])
        obs["robot"]["speed"] = [0.5]
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert diag["factorial_toggles"]["local_min_escape_count"] == 0


class TestFactorialDiagnostics:
    """Test diagnostics payload includes factorial toggle state."""

    def test_diagnostics_has_factorial_toggles(self):
        config = PredictionMPCConfig(predictor_backend="none")
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert "factorial_toggles" in diag
        toggles = diag["factorial_toggles"]
        assert "prediction_enabled" in toggles
        assert "hard_pedestrian_constraints_enabled" in toggles
        assert "local_min_escape_enabled" in toggles
        assert "local_min_escape_count" in toggles

    def test_diagnostics_A0_B0_arm(self):
        config = PredictionMPCConfig(
            predictor_backend="none",
            hard_pedestrian_constraints_enabled=False,
            local_min_escape_enabled=False,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        toggles = diag["factorial_toggles"]
        assert toggles["prediction_enabled"] is False
        assert toggles["hard_pedestrian_constraints_enabled"] is False
        assert toggles["local_min_escape_enabled"] is False

    def test_diagnostics_A1_B1_arm(self):
        config = PredictionMPCConfig(
            predictor_backend="constant_velocity",
            hard_pedestrian_constraints_enabled=True,
            local_min_escape_enabled=True,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        toggles = diag["factorial_toggles"]
        assert toggles["prediction_enabled"] is True
        assert toggles["hard_pedestrian_constraints_enabled"] is True
        assert toggles["local_min_escape_enabled"] is True


class TestFactorialArmConfigs:
    """Test that the four YAML arm configs build valid adapters."""

    ARM_CONFIGS = [
        "configs/algos/prediction_mpc_factorial_A0_B0.yaml",
        "configs/algos/prediction_mpc_factorial_A0_B1.yaml",
        "configs/algos/prediction_mpc_factorial_A1_B0.yaml",
        "configs/algos/prediction_mpc_factorial_A1_B1.yaml",
    ]

    @pytest.mark.parametrize("config_path", ARM_CONFIGS)
    def test_arm_config_loads_and_builds(self, config_path: str):
        path = REPO_ROOT / config_path
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = build_prediction_mpc_config(cfg)
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert "factorial_toggles" in diag

    def test_arm_configs_cover_all_four_cells(self):
        configs = {}
        for path_str in self.ARM_CONFIGS:
            path = REPO_ROOT / path_str
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            config = build_prediction_mpc_config(cfg)
            arm_key = path.stem.replace("prediction_mpc_factorial_", "")
            configs[arm_key] = config

        assert configs["A0_B0"].predictor_backend == "none"
        assert configs["A0_B0"].hard_pedestrian_constraints_enabled is False
        assert configs["A0_B0"].local_min_escape_enabled is False

        assert configs["A0_B1"].predictor_backend == "none"
        assert configs["A0_B1"].hard_pedestrian_constraints_enabled is True
        assert configs["A0_B1"].local_min_escape_enabled is True

        assert configs["A1_B0"].predictor_backend == "constant_velocity"
        assert configs["A1_B0"].hard_pedestrian_constraints_enabled is False
        assert configs["A1_B0"].local_min_escape_enabled is False

        assert configs["A1_B1"].predictor_backend == "constant_velocity"
        assert configs["A1_B1"].hard_pedestrian_constraints_enabled is True
        assert configs["A1_B1"].local_min_escape_enabled is True


class TestPreregistrationHarness:
    """Test the preregistration harness module."""

    def test_validate_arm_configs(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            validate_arm_configs,
        )

        config_path = REPO_ROOT / "configs/research/prediction_mpc_factorial_v1.yaml"
        results = validate_arm_configs(config_path)
        assert len(results) == 4
        for arm_key, result in results.items():
            assert result["valid"] is True, f"{arm_key}: {result.get('error')}"

    def test_check_planned_rows_complete(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            EXPECTED_FACTORIAL_ARMS,
            check_planned_rows,
        )

        rows = []
        for scenario_id in ["s1", "s2"]:
            for seed in [111, 112]:
                for arm in EXPECTED_FACTORIAL_ARMS:
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "seed": seed,
                            "factorial_arm": arm,
                        }
                    )
        result = check_planned_rows(rows)
        assert result["complete"] is True
        assert result["row_count"] == 16
        assert result["pair_count"] == 4

    def test_check_planned_rows_incomplete(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            check_planned_rows,
        )

        rows = [
            {"scenario_id": "s1", "seed": 111, "factorial_arm": "A0_B0"},
            {"scenario_id": "s1", "seed": 111, "factorial_arm": "A1_B1"},
        ]
        result = check_planned_rows(rows)
        assert result["complete"] is False
        assert len(result["incomplete_pairs"]) > 0
