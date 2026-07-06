"""Tests for the issue #4013 diagnostic short-horizon predictor trainer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_sf.planner.learned_short_horizon_predictor import (
    LearnedShortHorizonPedestrianPredictor,
    predictor_io_dims,
)
from robot_sf.planner.learned_short_horizon_trainer import (
    ShortHorizonTrainerConfig,
    generate_training_batch,
    train_short_horizon_predictor,
)


def _small_config(tmp_path) -> ShortHorizonTrainerConfig:
    """Build a fast CPU trainer config for tests."""

    return ShortHorizonTrainerConfig(
        max_pedestrians=3,
        horizon_steps=2,
        hidden_dim=16,
        num_samples=64,
        epochs=80,
        seed=4013,
        output_dir=str(tmp_path / "short_horizon"),
    )


def _obs() -> dict[str, object]:
    """Build a compact observation with one pedestrian near the robot."""

    return {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=float),
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
        },
        "goal": {"current": np.asarray([2.0, 0.0], dtype=float)},
        "pedestrians": {
            "positions": np.asarray([[1.0, 0.5]], dtype=float),
            "velocities": np.asarray([[0.5, 0.0]], dtype=float),
            "count": np.asarray([1.0], dtype=float),
        },
    }


def test_generate_training_batch_matches_predictor_dims(tmp_path) -> None:
    """Synthetic batch shapes match the predictor feature/output contract."""

    config = _small_config(tmp_path)
    features, targets = generate_training_batch(config)
    input_dim, output_dim = predictor_io_dims(config.predictor_config())

    assert features.shape == (config.num_samples, input_dim)
    assert targets.shape == (config.num_samples, output_dim)
    # Repulsion residual targets are non-trivial (not an all-zero degenerate task).
    assert float(np.abs(targets).max()) > 0.0


def test_batch_generation_is_deterministic(tmp_path) -> None:
    """Seeded batch generation is reproducible."""

    config = _small_config(tmp_path)
    first_features, first_targets = generate_training_batch(config)
    second_features, second_targets = generate_training_batch(config)

    np.testing.assert_array_equal(first_features, second_features)
    np.testing.assert_array_equal(first_targets, second_targets)


def test_training_reduces_loss_and_writes_artifacts(tmp_path) -> None:
    """Training lowers the loss and emits checkpoint, manifest, and metrics."""

    pytest.importorskip("torch")
    config = _small_config(tmp_path)
    result = train_short_horizon_predictor(config)

    assert result.final_loss < result.initial_loss
    assert result.loss_reduction > 0.0
    assert result.checkpoint_path.exists()
    assert result.manifest_path.exists()
    assert result.metrics_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["evidence_tier"] == "diagnostic-only"
    assert manifest["not_full_world_model"] is True
    assert "claim_boundary" in manifest
    assert manifest["metrics"]["final_loss"] < manifest["metrics"]["initial_loss"]


def test_trained_checkpoint_loads_without_fallback(tmp_path) -> None:
    """The predictor loads the trained checkpoint as checkpoint-loaded, not fallback."""

    pytest.importorskip("torch")
    config = _small_config(tmp_path)
    result = train_short_horizon_predictor(config)

    predictor = LearnedShortHorizonPedestrianPredictor(
        config.predictor_config(str(result.checkpoint_path))
    )
    diagnostics = predictor.diagnostics()
    assert diagnostics["evidence_tier"] == "checkpoint_loaded"
    assert diagnostics["diagnostic_only"] is False

    futures = predictor.predict(_obs(), horizon_steps=2, dt=0.2)
    assert futures.source == "checkpoint_loaded"
    assert futures.positions_world.shape == (1, 2, 2)
    assert np.all(np.isfinite(futures.positions_world))
