"""Tests for the probabilistic pedestrian prediction interface contract.

These tests validate that the data types (TrajectoryDistribution,
ProbabilisticPrediction) and the ProbabilisticPredictor protocol are
well-formed, importable, and behave as documented. They do **not** test
prediction accuracy, calibration, or planning benefit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_sf.nav.predictive_types import (
    ProbabilisticPrediction,
    ProbabilisticPredictor,
    TrajectoryDistribution,
)


class _DummyPredictor:
    """Minimal ProbabilisticPredictor implementation for protocol testing."""

    def predict(self, observation: dict[str, Any]) -> ProbabilisticPrediction:
        horizon_steps = 8
        ped_count = 2
        mean = np.zeros((horizon_steps, 2), dtype=np.float32)
        std = np.full((horizon_steps, 2), 0.05, dtype=np.float32)
        predictions = [
            TrajectoryDistribution(
                mean=mean.copy(),
                std=std.copy(),
                confidence=0.95,
                pedestrian_id=i,
            )
            for i in range(ped_count)
        ]
        return ProbabilisticPrediction(
            predictions=predictions,
            prediction_horizon=1.6,
            prediction_dt=0.2,
            timestamp=42.0,
            sample_count=8,
            metadata={"model_id": "dummy_v1", "fallback": False},
        )


def _make_socnav_obs(ped_count: int = 2) -> dict[str, Any]:
    """Build a minimal SocNav-structured observation for predictor tests."""
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.5, 0.0], dtype=np.float32),
            "radius": np.array([0.3], dtype=np.float32),
        },
        "goal": {
            "current": np.array([5.0, 0.0], dtype=np.float32),
            "next": np.array([5.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.zeros((ped_count, 2), dtype=np.float32),
            "velocities": np.zeros((ped_count, 2), dtype=np.float32),
            "radius": np.array([0.3], dtype=np.float32),
            "count": np.array([float(ped_count)], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.2], dtype=np.float32)},
    }


class TestTrajectoryDistribution:
    """TrajectoryDistribution dataclass contract."""

    def test_default_confidence_is_one(self) -> None:
        """Default confidence should be 1.0 (deterministic fallback)."""
        mean = np.zeros((8, 2), dtype=np.float32)
        td = TrajectoryDistribution(mean=mean, pedestrian_id=0)
        assert td.confidence == 1.0
        assert td.std is None
        assert td.covariance is None

    def test_full_uncertainty_fields(self) -> None:
        """All optional fields should be settable."""
        mean = np.zeros((8, 2), dtype=np.float32)
        std = np.full((8, 2), 0.05, dtype=np.float32)
        cov = np.tile(np.eye(2, dtype=np.float32) * 0.0025, (8, 1, 1))
        td = TrajectoryDistribution(
            mean=mean,
            std=std,
            covariance=cov,
            confidence=0.85,
            pedestrian_id=3,
        )
        assert td.mean.shape == (8, 2)
        assert td.std is not None
        assert td.std.shape == (8, 2)
        assert td.covariance is not None
        assert td.covariance.shape == (8, 2, 2)
        assert td.confidence == 0.85
        assert td.pedestrian_id == 3

    def test_arrays_and_scalar_fields_are_normalized(self) -> None:
        """Float arrays and numpy scalars should be normalized for consumers."""
        mean = np.zeros((8, 2), dtype=np.float64)
        std = np.full((8, 2), 0.05, dtype=np.float64)
        cov = np.tile(np.eye(2, dtype=np.float64) * 0.0025, (8, 1, 1))

        td = TrajectoryDistribution(
            mean=mean,
            std=std,
            covariance=cov,
            confidence=np.float64(0.75),
            pedestrian_id=np.int64(4),
        )

        assert td.mean.dtype == np.float32
        assert td.std is not None
        assert td.std.dtype == np.float32
        assert td.covariance is not None
        assert td.covariance.dtype == np.float32
        assert type(td.confidence) is float
        assert td.confidence == 0.75
        assert type(td.pedestrian_id) is int
        assert td.pedestrian_id == 4

    def test_invalid_confidence_is_rejected(self) -> None:
        """Confidence must stay in the documented [0, 1] interval."""
        mean = np.zeros((8, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="confidence"):
            TrajectoryDistribution(mean=mean, confidence=1.5)

    def test_invalid_uncertainty_shape_is_rejected(self) -> None:
        """Uncertainty arrays must match the mean trajectory shape."""
        mean = np.zeros((8, 2), dtype=np.float32)
        std = np.zeros((7, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="std"):
            TrajectoryDistribution(mean=mean, std=std)


class TestProbabilisticPrediction:
    """ProbabilisticPrediction dataclass contract."""

    def test_empty_prediction(self) -> None:
        """Empty prediction should be valid with zero pedestrians."""
        pred = ProbabilisticPrediction()
        assert pred.predictions == []
        assert pred.prediction_horizon == 0.0
        assert pred.sample_count == 1

    def test_bundles_predictions(self) -> None:
        """Should bundle multiple pedestrian predictions with shared metadata."""
        peds = [
            TrajectoryDistribution(
                mean=np.zeros((8, 2), dtype=np.float32),
                pedestrian_id=i,
            )
            for i in range(3)
        ]
        pred = ProbabilisticPrediction(
            predictions=peds,
            prediction_horizon=1.6,
            prediction_dt=0.2,
            timestamp=10.0,
            sample_count=16,
            metadata={"mode": "cvar"},
        )
        assert len(pred.predictions) == 3
        assert pred.prediction_horizon == 1.6
        assert pred.prediction_dt == 0.2
        assert pred.timestamp == 10.0
        assert pred.sample_count == 16
        assert pred.metadata["mode"] == "cvar"

    def test_metadata_scalar_fields_are_normalized(self) -> None:
        """Numpy scalar metadata should be normalized to plain Python scalars."""
        pred = ProbabilisticPrediction(
            prediction_horizon=np.float32(1.6),
            prediction_dt=np.float64(0.2),
            timestamp=np.float64(10.0),
            sample_count=np.int64(16),
        )

        assert type(pred.prediction_horizon) is float
        assert type(pred.prediction_dt) is float
        assert type(pred.timestamp) is float
        assert type(pred.sample_count) is int
        assert pred.sample_count == 16

    def test_invalid_prediction_metadata_is_rejected(self) -> None:
        """Prediction horizon, timestep, and sample count should be contract-checked."""
        with pytest.raises(ValueError, match="prediction_dt"):
            ProbabilisticPrediction(prediction_dt=0.0)

        with pytest.raises(ValueError, match="sample_count"):
            ProbabilisticPrediction(sample_count=0)


class TestProbabilisticPredictorProtocol:
    """ProbabilisticPredictor protocol contract."""

    def test_dummy_satisfies_protocol(self) -> None:
        """A class with a predict() method should satisfy the protocol."""
        predictor: ProbabilisticPredictor = _DummyPredictor()
        assert isinstance(predictor, ProbabilisticPredictor)

    def test_predict_returns_expected_structure(self) -> None:
        """predict() should return a ProbabilisticPrediction with correct shapes."""
        predictor: ProbabilisticPredictor = _DummyPredictor()
        obs = _make_socnav_obs(ped_count=2)
        result = predictor.predict(obs)
        assert isinstance(result, ProbabilisticPrediction)
        assert len(result.predictions) == 2
        for ped_pred in result.predictions:
            assert isinstance(ped_pred, TrajectoryDistribution)
            assert ped_pred.mean.shape == (8, 2)
            assert ped_pred.std is not None
            assert ped_pred.std.shape == (8, 2)
            assert 0.0 <= ped_pred.confidence <= 1.0
        assert result.prediction_horizon == 1.6
        assert result.prediction_dt == 0.2
        assert result.sample_count == 8

    def test_protocol_runtime_checkable(self) -> None:
        """@runtime_checkable should allow isinstance checks."""
        predictor = _DummyPredictor()
        assert isinstance(predictor, ProbabilisticPredictor)

    def test_non_predictor_rejected(self) -> None:
        """An object without predict() should not pass isinstance check."""
        assert not isinstance(42, ProbabilisticPredictor)
        assert not isinstance("string", ProbabilisticPredictor)

    def test_protocol_does_not_require_observation_specific_output_count(self) -> None:
        """The protocol shape is independent from a predictor's filtering policy."""
        predictor: ProbabilisticPredictor = _DummyPredictor()
        obs = _make_socnav_obs(ped_count=0)
        result = predictor.predict(obs)
        assert len(result.predictions) == 2  # dummy ignores observation

    def test_predictions_are_independent(self) -> None:
        """Each pedestrian's prediction should be an independent object."""
        predictor: ProbabilisticPredictor = _DummyPredictor()
        result = predictor.predict(_make_socnav_obs())
        orig_1 = float(result.predictions[1].mean[0, 0])
        result.predictions[0].mean[0, 0] = 999.0
        assert float(result.predictions[1].mean[0, 0]) == orig_1
