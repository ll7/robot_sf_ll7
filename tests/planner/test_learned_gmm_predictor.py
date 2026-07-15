"""Tests for the learned GMM K-mode pedestrian predictor (#5307 successor slice).

Tests cover the feature encoding, the zero-initialised (untrained) smoke-mode
predictor, config parsing, error handling, and end-to-end integration with the
``ChanceConstrainedMPCPlannerAdapter``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_sf.planner.chance_constrained_mpc import (
    ChanceConstrainedMPCPlannerAdapter,
    GaussianMixturePedestrianForecast,
    build_chance_constrained_mpc_adapter,
    build_chance_constrained_mpc_config,
)
from robot_sf.planner.learned_gmm_predictor import (
    LearnedGmmPedestrianPredictor,
    LearnedGmmPredictorConfig,
    build_learned_gmm_predictor_config,
    decode_gmm_forecast,
    encode_gmm_predictor_features,
    predictor_io_dims,
)

# ── fixture helpers ──────────────────────────────────────────────────────────


def _simple_observation(*, num_pedestrians: int = 3) -> dict[str, Any]:
    """Build a deterministic SocNav observation for testing.

    The robot is at the origin facing +x, one goal is straight ahead at
    (4, 0), and pedestrians are placed at known positions with zero velocity.
    """
    ped_positions = [[2.0 + i, -0.5 + i * 0.3] for i in range(num_pedestrians)]
    ped_velocities = [[0.0, 0.0] for _ in range(num_pedestrians)]
    return {
        "robot": {
            "position": np.asarray([0.0, 0.0]),
            "heading": np.asarray([0.0]),
            "speed": np.asarray([0.0]),
            "radius": np.asarray([0.25]),
        },
        "goal": {
            "current": np.asarray([4.0, 0.0]),
            "next": np.asarray([4.0, 0.0]),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([float(num_pedestrians)]),
            "radius": np.asarray([0.25]),
        },
    }


def _moving_observation(*, num_pedestrians: int = 2) -> dict[str, Any]:
    """Observation with moving pedestrians (inbound) for forecast tests."""
    ped_positions = [[3.0, 0.5], [4.0, -0.5]]
    ped_velocities = [[-0.5, 0.0], [-0.3, 0.1]]
    return {
        "robot": {
            "position": np.asarray([0.0, 0.0]),
            "heading": np.asarray([0.0]),
            "speed": np.asarray([0.0]),
            "radius": np.asarray([0.25]),
        },
        "goal": {
            "current": np.asarray([5.0, 0.0]),
            "next": np.asarray([5.0, 0.0]),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([float(num_pedestrians)]),
            "radius": np.asarray([0.25]),
        },
    }


# ── predictor_io_dims ───────────────────────────────────────────────────────


class TestPredictorIoDims:
    """Dimension contract for the tiny MLP."""

    def test_default_dims(self) -> None:
        """Sanity check dimensions for the default config."""
        cfg = LearnedGmmPredictorConfig()
        input_dim, output_dim = predictor_io_dims(cfg)
        # 4 (robot pos/hdg/speed) + 2 (goal delta) + 16 * 4 (pedestrian features)
        assert input_dim == 4 + 2 + 16 * 4 == 70
        # 16 pedestrians * 6 horizon * 3 modes * 6 parameters
        assert output_dim == 16 * 6 * 3 * 6 == 1728

    def test_custom_dims(self) -> None:
        """Confirm dimensions adapt to a custom config."""
        cfg = LearnedGmmPredictorConfig(
            max_pedestrians=8,
            horizon_steps=4,
            mode_count=2,
        )
        input_dim, output_dim = predictor_io_dims(cfg)
        assert input_dim == 4 + 2 + 8 * 4 == 38
        assert output_dim == 8 * 4 * 2 * 6 == 384


# ── feature encoding ─────────────────────────────────────────────────────────


class TestEncodeGmmPredictorFeatures:
    """Feature-vector structure and content."""

    def test_feature_shape(self) -> None:
        """Fixed-size feature vector for all pedestrian counts."""
        obs = _simple_observation(num_pedestrians=2)
        positions = np.asarray([[2.0, -0.5], [3.0, 0.2]], dtype=float)
        velocities = np.zeros((2, 2), dtype=float)
        features = encode_gmm_predictor_features(
            obs,
            positions,
            velocities,
            max_pedestrians=16,
        )
        assert features.shape == (70,)

    def test_feature_content_zeros(self) -> None:
        """Content sanity: robot at origin, no ped motion."""
        obs = _simple_observation(num_pedestrians=1)
        pos = np.asarray([[2.0, -0.5]], dtype=float)
        vel = np.zeros((1, 2), dtype=float)
        features = encode_gmm_predictor_features(
            obs,
            pos,
            vel,
            max_pedestrians=16,
        )
        # Robot pos = (0, 0), heading = 0, speed = 0
        assert np.allclose(features[:2], [0.0, 0.0])
        assert np.allclose(features[2], 0.0)
        assert np.allclose(features[3], 0.0)
        # Goal delta = (4, 0)
        assert np.allclose(features[4:6], [4.0, 0.0])
        # Pedestrian 0 relative pos = (2, -0.5)
        assert np.allclose(features[6:8], [2.0, -0.5])
        assert np.allclose(features[8:10], [0.0, 0.0])
        # Excess pedestrian slots are zero-filled
        assert np.allclose(features[10:], 0.0)

    def test_empty_pedestrians(self) -> None:
        """No pedestrians yields a feature vector with only robot/goal entries."""
        obs = _simple_observation(num_pedestrians=0)
        positions = np.zeros((0, 2), dtype=float)
        velocities = np.zeros((0, 2), dtype=float)
        features = encode_gmm_predictor_features(
            obs,
            positions,
            velocities,
            max_pedestrians=16,
        )
        assert features.shape == (70,)
        assert np.allclose(features[6:], 0.0)


# ── decode_gmm_forecast ─────────────────────────────────────────────────────


class TestDecodeGmmForecast:
    """GMM forecast validity from known MLP outputs."""

    def test_zero_output_produces_valid_forecast(self) -> None:
        """Zero MLP output (untrained smoke) yields valid forward GMM."""
        pos = np.asarray([[3.0, 0.5], [4.0, -0.5]], dtype=float)
        vel = np.asarray([[-0.5, 0.0], [-0.3, 0.1]], dtype=float)
        raw = np.zeros(16 * 6 * 3 * 6, dtype=float)
        forecast = decode_gmm_forecast(
            raw,
            pos,
            vel,
            dt=0.25,
            horizon_steps=6,
            mode_count=3,
            max_pedestrians=16,
        )
        assert isinstance(forecast, GaussianMixturePedestrianForecast)
        assert forecast.means_world.shape == (2, 3, 6, 2)
        assert forecast.covariances_world.shape == (2, 3, 6, 2, 2)
        assert forecast.mode_weights.shape == (2, 3)
        assert forecast.source == "learned_gmm_predictor"
        # Means should equal CV baseline (zero deltas)
        for step in range(6):
            tau = (step + 1) * 0.25
            expected = pos + vel * tau
            for mode in range(3):
                assert np.allclose(forecast.means_world[:, mode, step, :], expected, atol=1e-6)
        # Covariances should be isotropic identity (zero log-stds)
        expected_cov = np.eye(2)
        for p_idx in range(2):
            for mode in range(3):
                for step in range(6):
                    assert np.allclose(
                        forecast.covariances_world[p_idx, mode, step],
                        expected_cov,
                        atol=1e-6,
                    )
        # Mode weights should be equal
        assert np.allclose(forecast.mode_weights, 1.0 / 3.0, atol=1e-6)

    def test_partial_output_small_ped_count(self) -> None:
        """Only 1 pedestrian but network built for 16; output slices correctly."""
        pos = np.asarray([[2.0, 0.0]], dtype=float)
        vel = np.zeros((1, 2), dtype=float)
        raw = np.zeros(16 * 6 * 2 * 6, dtype=float)
        forecast = decode_gmm_forecast(
            raw,
            pos,
            vel,
            dt=0.25,
            horizon_steps=6,
            mode_count=2,
            max_pedestrians=16,
        )
        assert forecast.means_world.shape == (1, 2, 6, 2)
        assert forecast.covariances_world.shape == (1, 2, 6, 2, 2)
        assert forecast.mode_weights.shape == (1, 2)
        assert np.allclose(forecast.mode_weights, 0.5)

    def test_single_mode_always_weight_one(self) -> None:
        """K=1 means mode weight must be 1.0."""
        pos = np.asarray([[2.0, 0.0]], dtype=float)
        vel = np.zeros((1, 2), dtype=float)
        raw = np.zeros(16 * 6 * 1 * 6, dtype=float)
        forecast = decode_gmm_forecast(
            raw,
            pos,
            vel,
            dt=0.25,
            horizon_steps=6,
            mode_count=1,
            max_pedestrians=16,
        )
        assert forecast.mode_weights.shape == (1, 1)
        assert np.allclose(forecast.mode_weights, 1.0)

    def test_output_size_mismatch_raises(self) -> None:
        """Wrong MLP output size raises a clear error."""
        pos = np.zeros((1, 2), dtype=float)
        vel = np.zeros((1, 2), dtype=float)
        with pytest.raises(ValueError, match="MLP output size"):
            decode_gmm_forecast(
                np.zeros(100, dtype=float),
                pos,
                vel,
                dt=0.25,
                horizon_steps=6,
                mode_count=3,
                max_pedestrians=16,
            )


# ── config parsing ──────────────────────────────────────────────────────────


class TestLearnGmmPredictorConfig:
    """YAML-to-config parsing."""

    def test_defaults(self) -> None:
        """Default config is valid and smoke-test-able."""
        cfg = build_learned_gmm_predictor_config(None)
        assert cfg.max_pedestrians == 16
        assert cfg.mode_count == 3
        assert cfg.horizon_steps == 6
        assert not cfg.allow_untrained_smoke

    def test_custom_values(self) -> None:
        """Known fields are correctly parsed."""
        cfg = build_learned_gmm_predictor_config(
            {
                "max_pedestrians": "8",
                "mode_count": "2",
                "horizon_steps": "4",
                "hidden_dim": "64",
                "allow_untrained_smoke": "true",
            }
        )
        assert cfg.max_pedestrians == 8
        assert cfg.mode_count == 2
        assert cfg.horizon_steps == 4
        assert cfg.hidden_dim == 64
        assert cfg.allow_untrained_smoke

    def test_checkpoint_path_missing_raises(self) -> None:
        """Providing a non-existent checkpoint path without smoke mode raises."""
        cfg = LearnedGmmPredictorConfig(
            checkpoint_path="/nonexistent/model.pt",
            allow_untrained_smoke=False,
        )
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            LearnedGmmPedestrianPredictor(cfg)

    def test_no_checkpoint_no_smoke_raises(self) -> None:
        """Refusing smoke mode and having no checkpoint fails closed."""
        cfg = LearnedGmmPredictorConfig(
            checkpoint_path=None,
            allow_untrained_smoke=False,
        )
        with pytest.raises(ValueError, match="requires checkpoint_path"):
            LearnedGmmPedestrianPredictor(cfg)

    def test_unsupported_model_type(self) -> None:
        """Only 'mlp' is supported."""
        cfg = LearnedGmmPredictorConfig(
            model_type="transformer",
            allow_untrained_smoke=True,
        )
        with pytest.raises(ValueError, match="model_type='mlp' only"):
            LearnedGmmPedestrianPredictor(cfg)

    def test_invalid_values_fallback_to_defaults(self) -> None:
        """Malformed field values produce warnings and fall back."""
        cfg = build_learned_gmm_predictor_config(
            {
                "max_pedestrians": "not-a-number",
            }
        )
        # Falls back to 16
        assert int(cfg.max_pedestrians) == 16


# ── LearnedGmmPedestrianPredictor (untrained smoke mode) ─────────────────────


class TestLearnedGmmPedestrianPredictor:
    """Core predictor contract in untrained smoke mode."""

    def test_smoke_predictor_default(self) -> None:
        """Default untrained smoke predictor produces valid forecasts."""
        cfg = LearnedGmmPredictorConfig(allow_untrained_smoke=True)
        predictor = LearnedGmmPedestrianPredictor(cfg)
        assert predictor._evidence_tier == "diagnostic_untrained_smoke"

        obs = _moving_observation()
        forecast = predictor.predict(obs, horizon_steps=6, dt=0.25)
        assert isinstance(forecast, GaussianMixturePedestrianForecast)
        assert forecast.means_world.shape[0] == 2  # 2 pedestrians
        assert forecast.means_world.shape[1] == 3  # K=3 modes
        assert forecast.means_world.shape[2] == 6  # T=6 steps
        assert forecast.covariances_world.shape[-2:] == (2, 2)
        assert forecast.mode_weights.shape == (2, 3)

    def test_smoke_predictor_zero_pedestrians(self) -> None:
        """Empty scene produces a valid empty forecast."""
        cfg = LearnedGmmPredictorConfig(allow_untrained_smoke=True)
        predictor = LearnedGmmPedestrianPredictor(cfg)
        obs = _simple_observation(num_pedestrians=0)
        forecast = predictor.predict(obs, horizon_steps=6, dt=0.25)
        assert isinstance(forecast, GaussianMixturePedestrianForecast)
        assert forecast.means_world.shape[0] == 0  # No pedestrians

    def test_smoke_predictor_single_mode(self) -> None:
        """Single-mode config works in smoke mode."""
        cfg = LearnedGmmPredictorConfig(mode_count=1, allow_untrained_smoke=True)
        predictor = LearnedGmmPedestrianPredictor(cfg)
        obs = _moving_observation(num_pedestrians=1)
        forecast = predictor.predict(obs, horizon_steps=6, dt=0.25)
        assert isinstance(forecast, GaussianMixturePedestrianForecast)
        assert forecast.means_world.shape[1] == 1  # K=1
        assert np.allclose(forecast.mode_weights, 1.0)

    def test_smoke_predictor_source_tracking(self) -> None:
        """Source and call count are tracked correctly."""
        cfg = LearnedGmmPredictorConfig(allow_untrained_smoke=True)
        predictor = LearnedGmmPedestrianPredictor(cfg)
        assert predictor._calls == 0
        obs = _simple_observation()
        predictor.predict(obs, horizon_steps=6, dt=0.25)
        assert predictor._calls == 1
        predictor.predict(obs, horizon_steps=6, dt=0.25)
        assert predictor._calls == 2
        diag = predictor.diagnostics()
        assert diag["backend"] == "learned_gmm"
        assert diag["diagnostic_only"] is True
        assert diag["evidence_tier"] == "diagnostic_untrained_smoke"

    def test_reset_clears_diagnostics(self) -> None:
        """reset() zeros call counter and source."""
        cfg = LearnedGmmPredictorConfig(allow_untrained_smoke=True)
        predictor = LearnedGmmPedestrianPredictor(cfg)
        obs = _simple_observation()
        predictor.predict(obs, horizon_steps=6, dt=0.25)
        assert predictor._calls == 1
        predictor.reset()
        assert predictor._calls == 0
        assert predictor._last_source == "not_run"


# ── end-to-end integration with ChanceConstrainedMPCPlannerAdapter ───────────


class TestEndToEndIntegration:
    """Learned GMM predictor running inside the chance-constrained MPC."""

    def test_adapter_builds_with_learned_gmm_backend(self) -> None:
        """Adapter builds and runs one plan step with untrained smoke predictor."""
        adapter = build_chance_constrained_mpc_adapter(
            {
                "predictor_backend": "learned_gmm",
                "learned_gmm_allow_untrained_smoke": True,
                "chance_constraint_formulation": "marginal",
                "max_collision_risk": 0.1,
            }
        )
        assert isinstance(adapter, ChanceConstrainedMPCPlannerAdapter)

        obs = _moving_observation(num_pedestrians=2)
        command = adapter.plan(obs)
        assert len(command) == 2
        # Both speed and angular speed should be finite
        assert all(np.isfinite(c) for c in command)

    def test_adapter_joint_horizon_with_learned_gmm(self) -> None:
        """joint_horizon formulation works end-to-end with learned GMM."""
        adapter = build_chance_constrained_mpc_adapter(
            {
                "predictor_backend": "learned_gmm",
                "learned_gmm_allow_untrained_smoke": True,
                "chance_constraint_formulation": "joint_horizon",
                "max_collision_risk": 0.1,
            }
        )
        obs = _simple_observation(num_pedestrians=2)
        command = adapter.plan(obs)
        assert len(command) == 2
        assert all(np.isfinite(c) for c in command)

    def test_adapter_cvar_tail_with_learned_gmm(self) -> None:
        """CVaR tail-risk formulation works end-to-end with learned GMM."""
        adapter = build_chance_constrained_mpc_adapter(
            {
                "predictor_backend": "learned_gmm",
                "learned_gmm_allow_untrained_smoke": True,
                "chance_constraint_formulation": "cvar_tail",
                "max_collision_risk": 0.1,
                "cvar_alpha": 0.95,
            }
        )
        obs = _simple_observation(num_pedestrians=2)
        command = adapter.plan(obs)
        assert len(command) == 2
        assert all(np.isfinite(c) for c in command)

    def test_adapter_diagnostics_show_learned_gmm_source(self) -> None:
        """Diagnostics reflect the learned GMM backend."""
        adapter = build_chance_constrained_mpc_adapter(
            {
                "predictor_backend": "learned_gmm",
                "learned_gmm_allow_untrained_smoke": True,
                "learned_gmm_mode_count": 3,
            }
        )
        obs = _moving_observation(num_pedestrians=2)
        adapter.plan(obs)
        diag = adapter.diagnostics()
        cc = diag.get("chance_constraint", {})
        assert cc.get("forecast_source") == "diagnostic_untrained_smoke"
        assert cc.get("forecast_modes") == 3

    def test_empty_scene_causes_no_crashes(self) -> None:
        """No pedestrians in the observation does not crash the adapter."""
        adapter = build_chance_constrained_mpc_adapter(
            {
                "predictor_backend": "learned_gmm",
                "learned_gmm_allow_untrained_smoke": True,
            }
        )
        obs = _simple_observation(num_pedestrians=0)
        command = adapter.plan(obs)
        assert len(command) == 2
        assert all(np.isfinite(c) for c in command)

    def test_unrecognized_backend_fails_closed(self) -> None:
        """Unknown backend raises a clear error."""
        with pytest.raises(ValueError, match="chance_constrained_mpc is unavailable"):
            build_chance_constrained_mpc_adapter(
                {
                    "predictor_backend": "nonexistent_backend",
                }
            )

    def test_learned_gmm_without_smoke_or_checkpoint_fails(self) -> None:
        """learned_gmm backend with no checkpoint and smoke=False fails."""
        with pytest.raises(ValueError, match="requires checkpoint_path"):
            build_chance_constrained_mpc_adapter(
                {
                    "predictor_backend": "learned_gmm",
                    "learned_gmm_allow_untrained_smoke": False,
                }
            )


# ── config builder end-to-end through chance_constrained_mpc path ────────────


class TestConfigParsingEndToEnd:
    """Learned GMM config fields survive round-trip through the MPC config builder."""

    def test_learned_gmm_fields_in_mpc_config(self) -> None:
        """Learned GMM config fields are parsed correctly by the MPC config builder."""
        config = build_chance_constrained_mpc_config(
            {
                "predictor_backend": "learned_gmm",
                "learned_gmm_checkpoint_path": "/some/path.pt",
                "learned_gmm_hidden_dim": "64",
                "learned_gmm_mode_count": "5",
                "learned_gmm_allow_untrained_smoke": "true",
            }
        )
        assert config.predictor_backend == "learned_gmm"
        assert config.learned_gmm_checkpoint_path == "/some/path.pt"
        assert config.learned_gmm_hidden_dim == 64
        assert config.learned_gmm_mode_count == 5
        assert config.learned_gmm_allow_untrained_smoke

    def test_learned_gmm_fields_default(self) -> None:
        """Default config has sensible learned GMM field defaults."""
        config = build_chance_constrained_mpc_config(
            {
                "predictor_backend": "learned_gmm",
            }
        )
        assert config.learned_gmm_checkpoint_path is None
        assert config.learned_gmm_hidden_dim == 128
        assert config.learned_gmm_mode_count == 3
        assert not config.learned_gmm_allow_untrained_smoke
