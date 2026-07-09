"""Tests for the baseline ProbabilisticPredictor implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import robot_sf.nav.baseline_probabilistic_predictor as baseline_predictor_module
from robot_sf.benchmark.pedestrian_forecast import ForecastDistribution, PedestrianForecast
from robot_sf.nav.baseline_probabilistic_predictor import BaselineProbabilisticPredictor
from robot_sf.nav.predictive_types import ProbabilisticPrediction, ProbabilisticPredictor


def _make_socnav_obs(ped_count: int = 2) -> dict[str, Any]:
    """Build a minimal SocNav-structured observation for predictor tests."""

    positions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[:ped_count]
    velocities = np.asarray([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32)[:ped_count]
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.5, 0.0], dtype=np.float32),
            "velocity_xy": np.array([0.5, 0.0], dtype=np.float32),
            "angular_velocity": np.array([0.0], dtype=np.float32),
            "radius": np.array([0.3], dtype=np.float32),
        },
        "goal": {
            "current": np.array([5.0, 0.0], dtype=np.float32),
            "next": np.array([5.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": positions,
            "velocities": velocities,
            "radius": np.array([0.3], dtype=np.float32),
            "count": np.array([float(ped_count)], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {
            "timestep": np.array([0.1], dtype=np.float32),
            "time_s": np.array([12.5], dtype=np.float32),
        },
    }


class TestBaselineProbabilisticPredictor:
    """Baseline predictor satisfies the ProbabilisticPredictor protocol."""

    def test_satisfies_protocol(self) -> None:
        """The predictor should pass runtime isinstance checks."""

        predictor: ProbabilisticPredictor = BaselineProbabilisticPredictor(variant="cv")
        assert isinstance(predictor, ProbabilisticPredictor)

    @pytest.mark.parametrize(
        "variant", ["none", "cv", "semantic", "interaction_aware", "risk_filtered"]
    )
    def test_predict_returns_valid_prediction(self, variant: str) -> None:
        """Every supported variant should produce a valid ProbabilisticPrediction."""

        predictor = BaselineProbabilisticPredictor(variant=variant)
        obs = _make_socnav_obs(ped_count=2)
        result = predictor.predict(obs)

        assert isinstance(result, ProbabilisticPrediction)
        assert result.timestamp == pytest.approx(12.5)
        assert result.prediction_dt == 0.1
        assert len(result.predictions) == 2
        for distribution in result.predictions:
            assert distribution.mean.shape[0] == round(
                result.prediction_horizon / result.prediction_dt
            )
            assert distribution.mean.shape[1] == 2
            assert distribution.pedestrian_id in {0, 1}

    def test_cv_forecast_is_constant_velocity(self) -> None:
        """The cv variant should predict straight-line motion."""

        predictor = BaselineProbabilisticPredictor(variant="cv", horizons_s=(0.5, 1.0), dt_s=0.1)
        obs = _make_socnav_obs(ped_count=1)
        result = predictor.predict(obs)

        assert len(result.predictions) == 1
        mean = result.predictions[0].mean
        # World-frame velocity for the single pedestrian should be [0.1, 0.0]
        # because heading is 0, so ego velocity equals world velocity.
        expected_step_displacement = np.array([0.1, 0.0], dtype=np.float32) * predictor.dt_s
        actual_displacement = mean[1] - mean[0]
        np.testing.assert_allclose(actual_displacement, expected_step_displacement, rtol=1e-5)

    def test_steps_cover_requested_horizon(self) -> None:
        """Prediction steps should cover horizons that are not exact dt multiples."""

        predictor = BaselineProbabilisticPredictor(variant="cv", horizons_s=(0.25,), dt_s=0.1)
        result = predictor.predict(_make_socnav_obs(ped_count=1))

        assert result.predictions[0].mean.shape == (3, 2)
        assert result.prediction_horizon == pytest.approx(0.3)

    def test_invalid_variant_rejected(self) -> None:
        """Unsupported variants should fail closed at construction."""

        with pytest.raises(ValueError, match="variant"):
            BaselineProbabilisticPredictor(variant="unknown_variant")

    def test_none_variant_returns_static_predictions(self) -> None:
        """The none variant should keep pedestrian positions fixed."""

        predictor = BaselineProbabilisticPredictor(variant="none", horizons_s=(0.5, 1.0), dt_s=0.1)
        obs = _make_socnav_obs(ped_count=1)
        result = predictor.predict(obs)

        assert len(result.predictions) == 1
        mean = result.predictions[0].mean
        initial_position = obs["pedestrians"]["positions"][0]
        for step in range(mean.shape[0]):
            np.testing.assert_allclose(mean[step], initial_position, rtol=1e-6)

    def test_missing_optional_sections_are_treated_as_empty(self) -> None:
        """Optional observation sections may be omitted or set to None."""

        predictor = BaselineProbabilisticPredictor(variant="cv")

        result = predictor.predict({"robot": None, "pedestrians": None, "sim": None})

        assert result.predictions == []
        assert result.timestamp == -1.0

    def test_timestep_is_not_used_as_timestamp(self) -> None:
        """Prediction timestamp should use simulation time, not the fixed dt."""

        obs = _make_socnav_obs(ped_count=1)
        obs["sim"] = {"timestep": np.array([0.25], dtype=np.float32)}
        result = BaselineProbabilisticPredictor(variant="cv").predict(obs)

        assert result.prediction_dt == 0.1
        assert result.timestamp == -1.0

    def test_non_dict_forecast_metadata_uses_default_std(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed forecast metadata should not crash std extraction."""

        def fake_baseline(state, horizons_s):
            return PedestrianForecast(
                id=state.id,
                predictions=[
                    ForecastDistribution(
                        horizon_s=float(horizon_s),
                        mean=np.asarray(state.position, dtype=np.float32),
                        covariance=np.eye(2, dtype=np.float32),
                        metadata=({"std_m": 0.1} if index == 0 else None),  # type: ignore[arg-type]
                    )
                    for index, horizon_s in enumerate(horizons_s)
                ],
            )

        monkeypatch.setattr(
            baseline_predictor_module,
            "_baseline_for_variant",
            lambda _variant: fake_baseline,
        )
        predictor = BaselineProbabilisticPredictor(variant="cv", horizons_s=(0.2,), dt_s=0.1)

        result = predictor.predict(_make_socnav_obs(ped_count=1))

        assert result.predictions[0].std is not None
        np.testing.assert_allclose(
            result.predictions[0].std,
            np.array([[0.1, 0.1], [0.3, 0.3]], dtype=np.float32),
        )

    def test_per_step_uncertainty_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each forecast step should keep its own baseline std metadata."""

        def fake_baseline(state, horizons_s):
            return PedestrianForecast(
                id=state.id,
                predictions=[
                    ForecastDistribution(
                        horizon_s=float(horizon_s),
                        mean=np.asarray(state.position, dtype=np.float32),
                        covariance=np.eye(2, dtype=np.float32),
                        metadata={"std_m": 0.1 + 0.1 * index},
                    )
                    for index, horizon_s in enumerate(horizons_s)
                ],
            )

        monkeypatch.setattr(
            baseline_predictor_module,
            "_baseline_for_variant",
            lambda _variant: fake_baseline,
        )
        predictor = BaselineProbabilisticPredictor(variant="cv", horizons_s=(0.2,), dt_s=0.1)
        result = predictor.predict(_make_socnav_obs(ped_count=1))

        np.testing.assert_allclose(
            result.predictions[0].std,
            np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float32),
        )

    def test_ego_frame_velocity_conversion(self) -> None:
        """Pedestrian velocities stored in ego frame should be rotated to world frame."""

        # Robot heading 90 degrees; ego-frame x is world-frame y.
        obs = _make_socnav_obs(ped_count=1)
        obs["robot"]["heading"] = np.array([np.pi / 2], dtype=np.float32)
        obs["pedestrians"]["velocities"] = np.array([[0.1, 0.0]], dtype=np.float32)

        predictor = BaselineProbabilisticPredictor(variant="cv", horizons_s=(0.5, 1.0), dt_s=0.1)
        result = predictor.predict(obs)

        mean = result.predictions[0].mean
        # World velocity should be [0, 0.1]
        expected_displacement = np.array([0.0, 0.1], dtype=np.float32) * predictor.dt_s
        np.testing.assert_allclose(mean[1] - mean[0], expected_displacement, rtol=1e-6)
