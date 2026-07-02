"""Tests for the learned short-horizon pedestrian predictor."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.learned_prediction_mpc import build_learned_prediction_mpc_adapter
from robot_sf.planner.learned_short_horizon_predictor import (
    LearnedShortHorizonPedestrianPredictor,
    LearnedShortHorizonPredictorConfig,
    build_learned_short_horizon_predictor_config,
)


def _obs(*, heading: float = 0.0) -> dict[str, object]:
    """Build compact SocNav observation for learned predictor tests."""

    return {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
        },
        "goal": {"current": np.asarray([2.0, 0.0], dtype=float)},
        "pedestrians": {
            "positions": np.asarray([[1.0, 2.0]], dtype=float),
            "velocities": np.asarray([[1.0, 0.0]], dtype=float),
            "count": np.asarray([1.0], dtype=float),
        },
    }


def test_learned_predictor_config_parses_yaml_values() -> None:
    """YAML-style scalar values map into the learned predictor config."""

    cfg = build_learned_short_horizon_predictor_config(
        {
            "allow_untrained_smoke": "true",
            "device": None,
            "max_pedestrians": "4",
            "horizon_steps": "3",
            "rollout_dt": "0.1",
            "hidden_dim": "16",
        }
    )

    assert cfg.allow_untrained_smoke is True
    assert cfg.device == "cpu"
    assert cfg.max_pedestrians == 4
    assert cfg.horizon_steps == 3
    assert cfg.rollout_dt == 0.1
    assert cfg.hidden_dim == 16


def test_learned_predictor_config_rejects_non_positive_dimensions() -> None:
    """Non-positive model dimensions fail before obscure tensor errors."""

    for field in ("max_pedestrians", "horizon_steps", "hidden_dim"):
        config = LearnedShortHorizonPredictorConfig(
            allow_untrained_smoke=True,
            **{field: 0},
        )
        with pytest.raises(ValueError, match=f"{field} must be strictly positive"):
            LearnedShortHorizonPedestrianPredictor(config)


def test_learned_predictor_missing_checkpoint_fails_closed() -> None:
    """Learned predictor requires an artifact unless diagnostic mode is explicit."""

    with pytest.raises(ValueError, match="requires checkpoint_path or model_id"):
        LearnedShortHorizonPedestrianPredictor(LearnedShortHorizonPredictorConfig())


def test_learned_predictor_missing_checkpoint_path_fails_closed(tmp_path) -> None:
    """Missing configured checkpoint path is a hard setup error."""

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        LearnedShortHorizonPedestrianPredictor(
            LearnedShortHorizonPredictorConfig(checkpoint_path=str(tmp_path / "missing.pt"))
        )


def test_untrained_smoke_predictor_is_deterministic_and_diagnostic() -> None:
    """Untrained smoke mode returns deterministic diagnostic futures."""

    pytest.importorskip("torch")
    predictor = LearnedShortHorizonPedestrianPredictor(
        LearnedShortHorizonPredictorConfig(
            allow_untrained_smoke=True,
            max_pedestrians=2,
            horizon_steps=2,
            hidden_dim=8,
        )
    )

    futures = predictor.predict(_obs(heading=np.pi / 2.0), horizon_steps=2, dt=0.5)

    np.testing.assert_allclose(futures.positions_world[0, 0], np.asarray([1.0, 2.5]))
    np.testing.assert_allclose(futures.positions_world[0, 1], np.asarray([1.0, 3.0]))
    assert futures.source == "diagnostic_untrained_smoke"
    diagnostics = predictor.diagnostics()
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["not_full_world_model"] is True
    assert diagnostics["calls"] == 1


def test_fallback_predictor_marks_constant_velocity_non_evidence() -> None:
    """Explicit fallback stays labeled diagnostic rather than success evidence."""

    predictor = LearnedShortHorizonPedestrianPredictor(
        LearnedShortHorizonPredictorConfig(fallback_to_constant_velocity=True)
    )

    futures = predictor.predict(_obs(), horizon_steps=1, dt=0.5)

    np.testing.assert_allclose(futures.positions_world[0, 0], np.asarray([1.5, 2.0]))
    assert futures.source == "diagnostic_constant_velocity_fallback"
    assert predictor.diagnostics()["diagnostic_only"] is True


def test_learned_prediction_mpc_adapter_exposes_predictor_diagnostics() -> None:
    """Learned MPC adapter wires predictor into existing prediction-MPC constraints."""

    pytest.importorskip("torch")
    planner = build_learned_prediction_mpc_adapter(
        {
            "allow_untrained_smoke": True,
            "max_pedestrians": 2,
            "horizon_steps": 2,
            "hidden_dim": 8,
            "solver_max_iterations": 4,
        }
    )

    futures = planner._future_predictor.predict(_obs(), horizon_steps=2, dt=0.2)
    diagnostics = planner.diagnostics()

    assert planner.prediction_config.predictor_backend == "learned_short_horizon"
    assert futures.source == "diagnostic_untrained_smoke"
    assert diagnostics["predictor"]["backend"] == "learned_short_horizon"
    assert diagnostics["predictor"]["diagnostic_only"] is True
