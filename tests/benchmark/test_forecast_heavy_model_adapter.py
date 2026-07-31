"""Tests for the heavy forecast-model -> ForecastBatch adapter (issue #2845).

These tests exercise the adapter contract: shape validation, provenance wiring,
fail-closed on bad input, and the convenience ``build_heavy_model_forecast_batch``
entry point. They do NOT require any ML framework or trained model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.forecast_heavy_model_adapter import (
    FORECAST_HEAVY_MODEL_ADAPTER_VERSION,
    HeavyModelAdapterConfig,
    HeavyModelForecastAdapter,
    HeavyModelPredictionPayload,
    build_heavy_model_forecast_batch,
)


class TestHeavyModelAdapterConfig:
    """Tests for HeavyModelAdapterConfig construction and serialization."""

    def test_default_config_is_valid(self):
        cfg = HeavyModelAdapterConfig()
        assert cfg.predictor_id == "heavy_model_offline_study"
        assert cfg.predictor_family == "heavy_model_adapter_placeholder"
        assert cfg.dt_s == 0.1
        assert cfg.horizons_s == (0.5, 1.0, 1.5, 2.0)

    def test_config_to_dict_round_trips(self):
        cfg = HeavyModelAdapterConfig(predictor_family="cvae", seed=42)
        data = cfg.to_dict()
        assert data["predictor_family"] == "cvae"
        assert data["seed"] == 42


class TestHeavyModelForecastAdapter:
    """Tests for HeavyModelForecastAdapter.build_batch validation."""

    def test_build_batch_minimal(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5, 1.0))
        actor_ids = ["ped_0", "ped_1"]
        det = np.zeros((2, 2, 2), dtype=float)
        det[0] = [[1.0, 2.0], [3.0, 4.0]]
        det[1] = [[5.0, 6.0], [7.0, 8.0]]

        batch = adapter.build_batch(
            config=cfg,
            actor_ids=actor_ids,
            deterministic_trajectories=det,
        )

        assert batch.schema_version == "ForecastBatch.v1"
        assert len(batch.forecasts) == 2
        assert batch.provenance.predictor_family == "heavy_model_adapter_placeholder"
        assert batch.provenance.predictor_id == "heavy_model_offline_study"
        assert batch.provenance.actor_ids == actor_ids
        assert batch.provenance.horizons_s == [0.5, 1.0]

    def test_build_batch_with_samples(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5, 1.0))
        actor_ids = ["ped_0"]
        det = np.zeros((1, 2, 2), dtype=float)
        payload = HeavyModelPredictionPayload(
            samples=np.zeros((1, 3, 2, 2), dtype=float),
        )

        batch = adapter.build_batch(
            config=cfg,
            actor_ids=actor_ids,
            deterministic_trajectories=det,
            payload=payload,
        )

        assert len(batch.forecasts) == 1
        f = batch.forecasts[0]
        assert f.actor_id == "ped_0"
        assert f.deterministic is not None
        assert f.samples is not None
        assert f.samples.shape == (3, 2, 2)

    def test_build_batch_with_mode_probabilities(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5,))
        actor_ids = ["ped_0"]
        det = np.zeros((1, 1, 2), dtype=float)
        payload = HeavyModelPredictionPayload(
            samples=np.zeros((1, 2, 1, 2), dtype=float),
            mode_probabilities=[0.3, 0.7],
        )

        batch = adapter.build_batch(
            config=cfg,
            actor_ids=actor_ids,
            deterministic_trajectories=det,
            payload=payload,
        )

        assert batch.forecasts[0].mode_probabilities == [0.3, 0.7]

    def test_build_batch_rejects_wrong_deterministic_shape(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5, 1.0))
        det = np.zeros((2, 3, 2), dtype=float)

        with pytest.raises(ValueError, match="deterministic_trajectories must have shape"):
            adapter.build_batch(
                config=cfg,
                actor_ids=["ped_0", "ped_1"],
                deterministic_trajectories=det,
            )

    def test_build_batch_rejects_non_finite_deterministic(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5,))
        det = np.array([[[np.nan, 0.0]]], dtype=float)

        with pytest.raises(ValueError, match="deterministic_trajectories must contain only finite"):
            adapter.build_batch(
                config=cfg,
                actor_ids=["ped_0"],
                deterministic_trajectories=det,
            )

    def test_build_batch_rejects_non_finite_samples(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5,))
        det = np.zeros((1, 1, 2), dtype=float)
        payload = HeavyModelPredictionPayload(
            samples=np.array([[[[np.inf, 0.0]]]], dtype=float),
        )

        with pytest.raises(ValueError, match="samples must contain only finite"):
            adapter.build_batch(
                config=cfg,
                actor_ids=["ped_0"],
                deterministic_trajectories=det,
                payload=payload,
            )

    def test_build_batch_rejects_wrong_samples_shape(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5, 1.0))
        det = np.zeros((1, 2, 2), dtype=float)
        payload = HeavyModelPredictionPayload(
            samples=np.zeros((1, 3, 5, 2), dtype=float),
        )

        with pytest.raises(ValueError, match="samples must have shape"):
            adapter.build_batch(
                config=cfg,
                actor_ids=["ped_0"],
                deterministic_trajectories=det,
                payload=payload,
            )

    def test_build_batch_rejects_mode_probability_mismatch(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5,))
        det = np.zeros((1, 1, 2), dtype=float)
        payload = HeavyModelPredictionPayload(
            samples=np.zeros((1, 2, 1, 2), dtype=float),
            mode_probabilities=[0.3],
        )

        with pytest.raises(ValueError, match="mode_probabilities"):
            adapter.build_batch(
                config=cfg,
                actor_ids=["ped_0"],
                deterministic_trajectories=det,
                payload=payload,
            )

    def test_build_batch_provenance_wiring(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(
            predictor_id="test_predictor",
            predictor_family="cvae",
            observation_tier="deployable",
            scenario_id="test_scene",
            seed=99,
            horizons_s=(0.5, 1.0),
        )
        det = np.zeros((1, 2, 2), dtype=float)
        payload = HeavyModelPredictionPayload(
            actor_classes={"ped_0": "pedestrian"},
            feature_schema={"pos": "x_y", "vel": "dx_dy"},
            actor_mask_metadata={"reason": "test"},
        )

        batch = adapter.build_batch(
            config=cfg,
            actor_ids=["ped_0"],
            deterministic_trajectories=det,
            payload=payload,
        )

        prov = batch.provenance
        assert prov.predictor_id == "test_predictor"
        assert prov.predictor_family == "cvae"
        assert prov.observation_tier == "deployable"
        assert prov.scenario_id == "test_scene"
        assert prov.seed == 99
        assert prov.actor_classes == {"ped_0": "pedestrian"}
        assert prov.actor_mask_metadata["reason"] == "test"

    def test_build_batch_metadata(self):
        adapter = HeavyModelForecastAdapter()
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5, 1.0))
        det = np.zeros((1, 2, 2), dtype=float)
        payload = HeavyModelPredictionPayload(
            metadata={"run_tag": "offline_smoke", "heavy_family": "diffusion"},
        )

        batch = adapter.build_batch(
            config=cfg,
            actor_ids=["ped_0"],
            deterministic_trajectories=det,
            payload=payload,
        )

        assert batch.metadata["run_tag"] == "offline_smoke"


class TestBuildHeavyModelForecastBatch:
    """Tests for the build_heavy_model_forecast_batch convenience function."""

    def test_convenience_function_produces_valid_batch(self):
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5, 1.0))
        det = np.zeros((1, 2, 2), dtype=float)

        batch = build_heavy_model_forecast_batch(
            config=cfg,
            actor_ids=["ped_0"],
            deterministic_trajectories=det,
        )

        assert batch.schema_version == "ForecastBatch.v1"
        assert len(batch.forecasts) == 1

    def test_convenience_function_forwards_payload(self):
        cfg = HeavyModelAdapterConfig(horizons_s=(0.5,))
        det = np.zeros((1, 1, 2), dtype=float)
        payload = HeavyModelPredictionPayload(
            samples=np.zeros((1, 4, 1, 2), dtype=float),
        )

        batch = build_heavy_model_forecast_batch(
            config=cfg,
            actor_ids=["ped_0"],
            deterministic_trajectories=det,
            payload=payload,
        )

        assert batch.forecasts[0].samples is not None
        assert batch.forecasts[0].samples.shape == (4, 1, 2)

    def test_version_constant_is_defined(self):
        assert FORECAST_HEAVY_MODEL_ADAPTER_VERSION == "forecast_heavy_model_adapter.v1"
