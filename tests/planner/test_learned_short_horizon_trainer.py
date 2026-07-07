"""Tests for the issue #4013 diagnostic short-horizon predictor trainer."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from robot_sf.data_ingestion import real_trajectory_contract as contract
from robot_sf.planner.learned_short_horizon_predictor import (
    LearnedShortHorizonPedestrianPredictor,
    predictor_io_dims,
)
from robot_sf.planner.learned_short_horizon_trainer import (
    ShortHorizonTrainerConfig,
    generate_real_trajectory_training_batch,
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


def test_real_trajectory_manifest_batch_requires_validated_data(tmp_path, monkeypatch) -> None:
    """Validated staged trajectory manifests produce real-data trainer batches."""

    external_root = tmp_path / "external"
    staging_dir = external_root / "issue_4013_fixture"
    staging_dir.mkdir(parents=True)
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(external_root))
    (staging_dir / "trajectories.csv").write_text(
        "\n".join(
            ["scene,frame,ped_id,x,y"]
            + [
                f"fixture,{frame},{ped_id},{0.2 * frame + 0.01 * frame * frame + ped_id},{0.1 * ped_id}"
                for frame in range(8)
                for ped_id in range(2)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tree_sha256 = contract._staging_tree_sha256(staging_dir)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "robot_sf_real_trajectory_ingestion_manifest.v1",
                "dataset_id": "issue_4013_fixture",
                "title": "Issue 4013 fixture trajectories",
                "source": {
                    "url": "https://example.org/fixture",
                    "version": "fixture-v1",
                    "citation": "Synthetic fixture for test only.",
                    "access_date": "2026-07-06",
                },
                "license": {
                    "name": "test-fixture",
                    "url": None,
                    "posture": "bring-your-own",
                    "supplier_acknowledgment": True,
                    "redistribution": False,
                },
                "retrieval": {
                    "instructions": "Generated by the unit test.",
                    "download_url": None,
                    "fail_closed": True,
                },
                "checksums": {
                    "algorithm": "SHA-256",
                    "tree_sha256": tree_sha256,
                    "expected_tree_sha256": tree_sha256,
                },
                "conversion": {
                    "frame_rate_hz": 2.5,
                    "length_unit": "meters",
                    "coordinate_frame": "world_meters_xy",
                    "timestamp_field": "frame",
                    "agent_id_field": "ped_id",
                    "position_fields": ["x", "y"],
                    "map_context_field": "scene",
                    "missing_data_behavior": "fail_closed",
                },
                "splits": {"naming": "scene", "members": ["fixture"]},
                "staging": {
                    "staging_dir": "${ROBOT_SF_EXTERNAL_DATA_ROOT}/issue_4013_fixture",
                    "local_only_raw": True,
                    "durable_storage_target": "local-only-byo",
                },
                "privacy": {"pii_reviewed": True, "notes": "Synthetic fixture only."},
                "availability": "validated",
                "benchmark_eligibility": "research_only",
                "related_issues": [4013],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = ShortHorizonTrainerConfig(
        max_pedestrians=2,
        horizon_steps=2,
        hidden_dim=16,
        epochs=10,
        output_dir=str(tmp_path / "real_trajectory_model"),
        training_data_manifest_path=str(manifest_path),
    )
    features, targets = generate_real_trajectory_training_batch(config)
    assert features.shape[0] > 0
    assert features.shape[1:] == (predictor_io_dims(config.predictor_config())[0],)
    assert targets.shape[1:] == (predictor_io_dims(config.predictor_config())[1],)
    assert np.any(np.abs(targets) > 0.0)

    result = train_short_horizon_predictor(config)
    assert result.loss_reduction > 0.0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["evidence_tier"] == "real-trajectory-smoke"
    assert manifest["training_data_manifest_path"] == str(manifest_path)
    assert manifest["metrics"]["data_source"] == "real_trajectory_manifest"
    # Regression: the checkpoint's evidence tier must match the manifest so a
    # downstream consumer inspecting only the .pt does not misread a
    # real-trajectory run as diagnostic-only synthetic data.
    checkpoint = torch.load(result.checkpoint_path, weights_only=False)
    assert checkpoint["evidence_tier"] == "real-trajectory-smoke"


def test_real_trajectory_batch_fails_closed_when_not_validated(tmp_path) -> None:
    """A non-``validated`` manifest must fail closed instead of training on it."""

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "robot_sf_real_trajectory_ingestion_manifest.v1",
                "dataset_id": "issue_4013_fixture",
                "title": "Issue 4013 fixture trajectories",
                "source": {
                    "url": "https://example.org/fixture",
                    "version": "fixture-v1",
                    "citation": "Synthetic fixture for test only.",
                    "access_date": "2026-07-06",
                },
                "license": {
                    "name": "test-fixture",
                    "url": None,
                    "posture": "bring-your-own",
                    "supplier_acknowledgment": True,
                    "redistribution": False,
                },
                "retrieval": {
                    "instructions": "Generated by the unit test.",
                    "download_url": None,
                    "fail_closed": True,
                },
                "checksums": {
                    "algorithm": "SHA-256",
                    "tree_sha256": None,
                    "expected_tree_sha256": "0" * 64,
                },
                "conversion": {
                    "frame_rate_hz": 2.5,
                    "length_unit": "meters",
                    "coordinate_frame": "world_meters_xy",
                    "timestamp_field": "frame",
                    "agent_id_field": "ped_id",
                    "position_fields": ["x", "y"],
                    "map_context_field": "scene",
                    "missing_data_behavior": "fail_closed",
                },
                "splits": {"naming": "scene", "members": ["fixture"]},
                "staging": {
                    "staging_dir": "${ROBOT_SF_EXTERNAL_DATA_ROOT}/issue_4013_fixture",
                    "local_only_raw": True,
                    "durable_storage_target": "local-only-byo",
                },
                "privacy": {"pii_reviewed": True, "notes": "Synthetic fixture only."},
                "availability": "missing",
                "benchmark_eligibility": "research_only",
                "related_issues": [4013],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = ShortHorizonTrainerConfig(
        max_pedestrians=2,
        horizon_steps=2,
        hidden_dim=16,
        epochs=10,
        output_dir=str(tmp_path / "real_trajectory_model"),
        training_data_manifest_path=str(manifest_path),
    )
    with pytest.raises(ValueError, match="requires manifest availability 'validated'"):
        generate_real_trajectory_training_batch(config)
