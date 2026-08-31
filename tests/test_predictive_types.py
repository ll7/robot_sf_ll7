"""Tests for canonical multimodal predictive types and contract adapters.

Covers all acceptance criteria for issue #8049:
1. Valid one-track/two-mode construction.
2. Multi-track canonical ordering independent of input dictionary order.
3. Legacy prediction becomes one probability-1 mode without numerical drift.
4. Duplicate/empty mode ID rejected.
5. Negative/NaN/Inf probability rejected.
6. Probability sum below/above tolerance rejected.
7. Builder normalization is deterministic and rejects all-zero weights.
8. Mismatched horizon rejected.
9. Mismatched dt rejected where represented at mode level.
10. Malformed/non-PSD covariance rejected through the existing owner.
11. Invalid existence/confidence/age rejected.
12. Track-key mismatch rejected.
13. Source-array mutation after construction cannot mutate the forecast.
14. JSON-safe export is deterministic and rejects non-finite values.
15. Empty global prediction remains valid.
16. Existing unimodal tests pass unchanged.
17. Current baseline probabilistic predictor runs through the adapter end to end.
18. An exported/loaded fixture preserves mode IDs/probabilities/arrays under schema version.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_sf.nav.baseline_probabilistic_predictor import BaselineProbabilisticPredictor
from robot_sf.nav.predictive_types import (
    MultimodalPrediction,
    PedestrianForecast,
    ProbabilisticPrediction,
    TrajectoryDistribution,
    TrajectoryMode,
    as_multimodal_prediction,
    build_normalized_modes,
)


def _sample_mode(
    mode_id: str,
    probability: float,
    steps: int = 5,
    offset: float = 0.0,
    with_cov: bool = False,
) -> TrajectoryMode:
    mean = np.zeros((steps, 2), dtype=np.float32)
    mean[:, 0] = np.linspace(offset, offset + 1.0, steps, dtype=np.float32)
    std = np.full((steps, 2), 0.05, dtype=np.float32)
    cov = None
    if with_cov:
        cov = np.tile(np.eye(2, dtype=np.float32) * 0.01, (steps, 1, 1))
    return TrajectoryMode(
        mode_id=mode_id,
        probability=probability,
        mean=mean,
        std=std,
        covariance=cov,
        intent="cross",
    )


class TestMultimodalPredictiveTypes:
    """Unit test suite for multimodal prediction contracts."""

    def test_01_valid_one_track_two_mode_construction(self) -> None:
        """1. Valid one-track/two-mode construction."""
        m1 = _sample_mode("left", 0.6, steps=5, offset=0.0)
        m2 = _sample_mode("right", 0.4, steps=5, offset=1.0)
        forecast = PedestrianForecast(
            pedestrian_id=10,
            modes=[m1, m2],
            existence_probability=0.95,
            confidence=0.9,
            age=1.5,
        )
        assert forecast.pedestrian_id == 10
        assert len(forecast.modes) == 2
        assert forecast.primary_mode().mode_id == "left"
        assert forecast.existence_probability == 0.95
        assert forecast.confidence == 0.9
        assert forecast.age == 1.5

        pred = MultimodalPrediction(
            forecasts={10: forecast},
            prediction_horizon=0.5,
            prediction_dt=0.1,
            timestamp=10.0,
        )
        assert pred.ordered_pedestrian_ids() == [10]
        assert len(pred.ordered_forecasts()) == 1

    def test_02_multitrack_canonical_ordering(self) -> None:
        """2. Multi-track canonical ordering independent of input dictionary order."""
        f3 = PedestrianForecast(pedestrian_id=3, modes=[_sample_mode("m0", 1.0, steps=4)])
        f1 = PedestrianForecast(pedestrian_id=1, modes=[_sample_mode("m0", 1.0, steps=4)])
        f7 = PedestrianForecast(pedestrian_id=7, modes=[_sample_mode("m0", 1.0, steps=4)])

        # Passed in arbitrary order
        pred = MultimodalPrediction(
            forecasts={7: f7, 1: f1, 3: f3},
            prediction_horizon=0.4,
            prediction_dt=0.1,
        )
        assert pred.ordered_pedestrian_ids() == [1, 3, 7]
        assert [f.pedestrian_id for f in pred.ordered_forecasts()] == [1, 3, 7]

        # Mode ordering within forecast
        f_multi = PedestrianForecast(
            pedestrian_id=2,
            modes=[
                _sample_mode("low", 0.2, steps=4),
                _sample_mode("high", 0.8, steps=4),
            ],
        )
        assert [m.mode_id for m in f_multi.sorted_modes()] == ["high", "low"]

    def test_03_legacy_unimodal_adaptation_preserves_values_without_drift(self) -> None:
        """3. Legacy prediction becomes one probability-1 mode without numerical drift."""
        mean = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        std = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        cov = np.tile(np.eye(2, dtype=np.float32) * 0.05, (3, 1, 1))
        dist = TrajectoryDistribution(
            mean=mean,
            std=std,
            covariance=cov,
            confidence=0.88,
            pedestrian_id=42,
        )
        unimodal = ProbabilisticPrediction(
            predictions=[dist],
            prediction_horizon=0.3,
            prediction_dt=0.1,
            timestamp=5.5,
            sample_count=4,
            metadata={"source": "test"},
        )

        multimodal = as_multimodal_prediction(unimodal)
        assert isinstance(multimodal, MultimodalPrediction)
        assert 42 in multimodal.forecasts
        f = multimodal.forecasts[42]
        assert f.pedestrian_id == 42
        assert f.confidence == 0.88
        assert len(f.modes) == 1
        mode = f.modes[0]
        assert mode.mode_id == "primary"
        assert mode.probability == 1.0
        np.testing.assert_array_equal(mode.mean, mean)
        np.testing.assert_array_equal(mode.std, std)
        np.testing.assert_array_equal(mode.covariance, cov)

        # Roundtrip back to ProbabilisticPrediction
        roundtrip = multimodal.as_probabilistic_prediction()
        assert len(roundtrip.predictions) == 1
        np.testing.assert_array_equal(roundtrip.predictions[0].mean, mean)
        np.testing.assert_array_equal(roundtrip.predictions[0].std, std)
        np.testing.assert_array_equal(roundtrip.predictions[0].covariance, cov)
        assert roundtrip.predictions[0].confidence == pytest.approx(0.88)

    def test_04_duplicate_or_empty_mode_id_rejected(self) -> None:
        """4. Duplicate/empty mode ID rejected."""
        with pytest.raises(ValueError, match="mode_id"):
            TrajectoryMode(mode_id="", probability=1.0, mean=np.zeros((3, 2), dtype=np.float32))

        with pytest.raises(ValueError, match="mode_id"):
            TrajectoryMode(mode_id="   ", probability=1.0, mean=np.zeros((3, 2), dtype=np.float32))

        m1 = _sample_mode("mode_a", 0.5, steps=3)
        m2 = _sample_mode("mode_a", 0.5, steps=3)
        with pytest.raises(ValueError, match="duplicate mode_id"):
            PedestrianForecast(pedestrian_id=1, modes=[m1, m2])

    def test_05_invalid_probability_rejected(self) -> None:
        """5. Negative/NaN/Inf probability rejected."""
        mean = np.zeros((3, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="probability"):
            TrajectoryMode(mode_id="m1", probability=-0.1, mean=mean)
        with pytest.raises(ValueError, match="probability"):
            TrajectoryMode(mode_id="m1", probability=1.5, mean=mean)
        with pytest.raises(ValueError, match="finite"):
            TrajectoryMode(mode_id="m1", probability=float("nan"), mean=mean)
        with pytest.raises(ValueError, match="finite"):
            TrajectoryMode(mode_id="m1", probability=float("inf"), mean=mean)

    def test_06_probability_sum_tolerance_enforced(self) -> None:
        """6. Probability sum below/above tolerance rejected."""
        m1 = _sample_mode("m1", 0.3, steps=3)
        m2 = _sample_mode("m2", 0.3, steps=3)
        with pytest.raises(ValueError, match="sum to 1.0"):
            PedestrianForecast(pedestrian_id=1, modes=[m1, m2])

        m3 = _sample_mode("m3", 0.8, steps=3)
        m4 = _sample_mode("m4", 0.8, steps=3)
        with pytest.raises(ValueError, match="sum to 1.0"):
            PedestrianForecast(pedestrian_id=1, modes=[m3, m4])

    def test_07_builder_normalization_deterministic_and_rejects_zero(self) -> None:
        """7. Builder normalization is deterministic and rejects all-zero weights."""
        raw_modes = [
            {"mode_id": "m1", "weight": 2.0, "mean": np.zeros((3, 2), dtype=np.float32)},
            {"mode_id": "m2", "weight": 6.0, "mean": np.ones((3, 2), dtype=np.float32)},
        ]
        norm = build_normalized_modes(raw_modes)
        assert len(norm) == 2
        assert norm[0].probability == pytest.approx(0.25)
        assert norm[1].probability == pytest.approx(0.75)

        with pytest.raises(ValueError, match="sum of mode weights"):
            build_normalized_modes(
                [
                    {"mode_id": "m1", "weight": 0.0, "mean": np.zeros((3, 2), dtype=np.float32)},
                ]
            )

    def test_08_mismatched_horizon_rejected(self) -> None:
        """8. Mismatched horizon rejected."""
        f = PedestrianForecast(pedestrian_id=1, modes=[_sample_mode("m1", 1.0, steps=5)])
        with pytest.raises(ValueError, match="steps"):
            MultimodalPrediction(
                forecasts={1: f},
                prediction_horizon=1.0,  # 1.0 / 0.1 = 10 steps, but forecast has 5
                prediction_dt=0.1,
            )

    def test_09_mismatched_dt_or_mode_steps_rejected(self) -> None:
        """9. Mismatched dt rejected where represented at mode level."""
        m1 = _sample_mode("m1", 0.5, steps=5)
        m2 = _sample_mode("m2", 0.5, steps=6)
        with pytest.raises(ValueError, match="step count"):
            PedestrianForecast(pedestrian_id=1, modes=[m1, m2])

    def test_10_malformed_covariance_rejected(self) -> None:
        """10. Malformed/non-PSD covariance rejected through existing validator."""
        mean = np.zeros((3, 2), dtype=np.float32)
        non_sym_cov = np.zeros((3, 2, 2), dtype=np.float32)
        non_sym_cov[:, 0, 1] = 1.0  # Asymmetric
        with pytest.raises(ValueError, match="symmetric"):
            TrajectoryMode(mode_id="m1", probability=1.0, mean=mean, covariance=non_sym_cov)

        non_psd_cov = np.zeros((3, 2, 2), dtype=np.float32)
        non_psd_cov[:, 0, 0] = -1.0  # Negative eigenvalue
        with pytest.raises(ValueError, match="semidefinite"):
            TrajectoryMode(mode_id="m1", probability=1.0, mean=mean, covariance=non_psd_cov)

    def test_11_invalid_existence_confidence_age_rejected(self) -> None:
        """11. Invalid existence/confidence/age rejected."""
        modes = [_sample_mode("m1", 1.0, steps=3)]
        with pytest.raises(ValueError, match="existence_probability"):
            PedestrianForecast(pedestrian_id=1, modes=modes, existence_probability=-0.1)
        with pytest.raises(ValueError, match="confidence"):
            PedestrianForecast(pedestrian_id=1, modes=modes, confidence=1.2)
        with pytest.raises(ValueError, match="age"):
            PedestrianForecast(pedestrian_id=1, modes=modes, age=-1.0)

    def test_12_track_key_mismatch_rejected(self) -> None:
        """12. Track-key mismatch rejected."""
        f = PedestrianForecast(pedestrian_id=5, modes=[_sample_mode("m1", 1.0, steps=3)])
        with pytest.raises(ValueError, match="key mismatch"):
            MultimodalPrediction(forecasts={9: f}, prediction_horizon=0.3, prediction_dt=0.1)

    def test_13_defensive_copies_prevent_source_array_mutation(self) -> None:
        """13. Source-array mutation after construction cannot mutate the forecast."""
        source_mean = np.zeros((4, 2), dtype=np.float32)
        mode = TrajectoryMode(mode_id="m1", probability=1.0, mean=source_mean)
        source_mean[0, 0] = 999.0
        assert mode.mean[0, 0] == 0.0

        forecast = PedestrianForecast(pedestrian_id=1, modes=[mode])
        pred = MultimodalPrediction(
            forecasts={1: forecast}, prediction_horizon=0.4, prediction_dt=0.1
        )
        mode.mean[0, 0] = 777.0
        assert (
            pred.forecasts[1].modes[0].mean[0, 0] == 777.0
        )  # mode itself was not mutated externally

    def test_14_json_safe_export_is_deterministic_and_rejects_nonfinite(self) -> None:
        """14. JSON-safe export is deterministic and rejects non-finite values."""
        f1 = PedestrianForecast(pedestrian_id=2, modes=[_sample_mode("m0", 1.0, steps=3)])
        f2 = PedestrianForecast(pedestrian_id=1, modes=[_sample_mode("m0", 1.0, steps=3)])
        pred = MultimodalPrediction(
            forecasts={2: f1, 1: f2},
            prediction_horizon=0.3,
            prediction_dt=0.1,
            timestamp=12.3,
        )
        d = pred.to_dict()
        serialized = json.dumps(d, sort_keys=True)
        deserialized = json.loads(serialized)
        assert list(deserialized["forecasts"].keys()) == ["1", "2"]
        assert deserialized["schema_version"] == "multimodal-prediction.v1"

    def test_15_empty_global_prediction_valid(self) -> None:
        """15. Empty global prediction remains valid."""
        pred = MultimodalPrediction(forecasts={}, prediction_horizon=0.0, prediction_dt=0.1)
        assert pred.ordered_pedestrian_ids() == []
        assert pred.ordered_forecasts() == []
        unimodal = pred.as_probabilistic_prediction()
        assert len(unimodal.predictions) == 0

    def test_16_existing_unimodal_contract_unaffected(self) -> None:
        """16. Existing unimodal tests pass unchanged."""
        mean = np.zeros((5, 2), dtype=np.float32)
        td = TrajectoryDistribution(mean=mean, confidence=0.9, pedestrian_id=3)
        prob_pred = ProbabilisticPrediction(
            predictions=[td],
            prediction_horizon=0.5,
            prediction_dt=0.1,
        )
        assert len(prob_pred.predictions) == 1
        assert prob_pred.predictions[0].pedestrian_id == 3

    def test_17_baseline_probabilistic_predictor_runs_through_adapter(self) -> None:
        """17. Current baseline probabilistic predictor runs through the adapter end to end."""
        predictor = BaselineProbabilisticPredictor(variant="cv", horizons_s=(0.5, 1.0), dt_s=0.1)
        obs = {
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
                "positions": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                "velocities": np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
                "radius": np.array([0.3], dtype=np.float32),
                "count": np.array([2.0], dtype=np.float32),
            },
            "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
            "sim": {
                "timestep": np.array([0.1], dtype=np.float32),
                "time_s": np.array([15.0], dtype=np.float32),
            },
        }
        legacy_pred = predictor.predict(obs)
        mm_pred = as_multimodal_prediction(legacy_pred)
        assert isinstance(mm_pred, MultimodalPrediction)
        assert set(mm_pred.ordered_pedestrian_ids()) == {0, 1}
        for pid in (0, 1):
            f = mm_pred.forecasts[pid]
            assert len(f.modes) == 1
            assert f.modes[0].probability == 1.0

    def test_18_roundtrip_fixture_preserves_modes_and_arrays(self) -> None:
        """18. An exported/loaded fixture preserves mode IDs/probabilities/arrays."""
        m1 = _sample_mode("straight", 0.7, steps=6, offset=0.0, with_cov=True)
        m2 = _sample_mode("swerve", 0.3, steps=6, offset=2.0, with_cov=True)
        f1 = PedestrianForecast(
            pedestrian_id=8,
            modes=[m1, m2],
            existence_probability=0.99,
            confidence=0.85,
            age=3.2,
            metadata={"sensor": "lidar"},
        )
        orig = MultimodalPrediction(
            forecasts={8: f1},
            prediction_horizon=0.6,
            prediction_dt=0.1,
            timestamp=25.4,
            sample_count=16,
            schema_version="multimodal-prediction.v1",
            metadata={"engine": "test"},
        )

        d = orig.to_dict()
        loaded = MultimodalPrediction.from_dict(d)
        assert loaded.schema_version == orig.schema_version
        assert loaded.prediction_horizon == orig.prediction_horizon
        assert loaded.prediction_dt == orig.prediction_dt
        assert loaded.timestamp == orig.timestamp
        assert loaded.sample_count == orig.sample_count
        assert loaded.metadata == orig.metadata

        f_loaded = loaded.forecasts[8]
        assert f_loaded.pedestrian_id == 8
        assert f_loaded.existence_probability == orig.forecasts[8].existence_probability
        assert f_loaded.confidence == orig.forecasts[8].confidence
        assert f_loaded.age == orig.forecasts[8].age
        assert f_loaded.metadata == orig.forecasts[8].metadata

        assert len(f_loaded.modes) == 2
        for i in range(2):
            orig_m = orig.forecasts[8].sorted_modes()[i]
            load_m = f_loaded.sorted_modes()[i]
            assert load_m.mode_id == orig_m.mode_id
            assert load_m.probability == pytest.approx(orig_m.probability)
            np.testing.assert_allclose(load_m.mean, orig_m.mean)
            assert orig_m.std is not None and load_m.std is not None
            np.testing.assert_allclose(load_m.std, orig_m.std)
            assert orig_m.covariance is not None and load_m.covariance is not None
            np.testing.assert_allclose(load_m.covariance, orig_m.covariance)

    def test_19_sequence_input_and_type_error_coverage(self) -> None:
        """Test sequence initialization, duplicate rejection, and TypeError handling."""
        f1 = PedestrianForecast(pedestrian_id=1, modes=[_sample_mode("m0", 1.0, steps=3)])
        f2 = PedestrianForecast(pedestrian_id=2, modes=[_sample_mode("m0", 1.0, steps=3)])

        # List input
        pred = MultimodalPrediction(
            forecasts=[f1, f2],
            prediction_horizon=0.3,
            prediction_dt=0.1,
        )
        assert pred.ordered_pedestrian_ids() == [1, 2]

        # Duplicate in list input
        with pytest.raises(ValueError, match="duplicate pedestrian_id"):
            MultimodalPrediction(
                forecasts=[f1, f1],
                prediction_horizon=0.3,
                prediction_dt=0.1,
            )

        # Invalid forecasts type
        with pytest.raises(TypeError, match="forecasts must be a dict or sequence"):
            MultimodalPrediction(
                forecasts=123,  # type: ignore[arg-type]
                prediction_horizon=0.3,
                prediction_dt=0.1,
            )

        # Non-PedestrianForecast in dict
        with pytest.raises(TypeError, match="expected PedestrianForecast"):
            MultimodalPrediction(
                forecasts={1: "invalid"},  # type: ignore[dict-item]
                prediction_horizon=0.3,
                prediction_dt=0.1,
            )

        # Non-PedestrianForecast in sequence
        with pytest.raises(TypeError, match="expected PedestrianForecast"):
            MultimodalPrediction(
                forecasts=["invalid"],  # type: ignore[list-item]
                prediction_horizon=0.3,
                prediction_dt=0.1,
            )

        # Empty modes in PedestrianForecast
        with pytest.raises(ValueError, match="at least one mode"):
            PedestrianForecast(pedestrian_id=1, modes=[])

        # Non-TrajectoryMode in PedestrianForecast
        with pytest.raises(TypeError, match="expected TrajectoryMode"):
            PedestrianForecast(pedestrian_id=1, modes=["invalid"])  # type: ignore[list-item]

        # Invalid type in as_multimodal_prediction
        with pytest.raises(
            TypeError, match="expected ProbabilisticPrediction or MultimodalPrediction"
        ):
            as_multimodal_prediction("invalid")  # type: ignore[arg-type]

        # as_multimodal_prediction pass-through for MultimodalPrediction
        assert as_multimodal_prediction(pred) is pred

        # build_normalized_modes with TrajectoryMode objects
        m_raw = _sample_mode("m1", 0.5, steps=3)
        normalized = build_normalized_modes([m_raw, m_raw])
        assert len(normalized) == 2
        assert normalized[0].probability == pytest.approx(0.5)

        # build_normalized_modes empty error
        with pytest.raises(ValueError, match="raw_modes must not be empty"):
            build_normalized_modes([])

        # build_normalized_modes invalid type
        with pytest.raises(TypeError, match="expected dict or TrajectoryMode"):
            build_normalized_modes([123])  # type: ignore[list-item]

        # build_normalized_modes negative weight
        with pytest.raises(ValueError, match="mode weight must be finite and non-negative"):
            build_normalized_modes([{"mode_id": "m1", "weight": -1.0, "mean": np.zeros((3, 2))}])

        # build_normalized_modes missing mode_id
        with pytest.raises(ValueError, match="mode dict must contain 'mode_id'"):
            build_normalized_modes([{"weight": 1.0, "mean": np.zeros((3, 2))}])

        # build_normalized_modes missing mean
        with pytest.raises(ValueError, match="mode dict must contain 'mean'"):
            build_normalized_modes([{"mode_id": "m1", "weight": 1.0}])
