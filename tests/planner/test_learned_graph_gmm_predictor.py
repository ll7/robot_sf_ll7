"""CPU contract tests for the opt-in #2844 graph-GRU GMM scaffold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_sf.planner.chance_constrained_mpc import build_chance_constrained_mpc_adapter
from robot_sf.planner.learned_gmm_predictor import (
    LearnedGmmPedestrianPredictor,
    LearnedGmmPredictorConfig,
    decode_graph_gmm_forecast,
    encode_graph_predictor_features,
    graph_predictor_io_dims,
)

pytest.importorskip("torch")


def _observation(*, pedestrian_count: int = 2) -> dict[str, Any]:
    """Build a compact SocNav observation with world-frame test values."""
    positions = np.asarray([[2.0, -0.5], [3.0, 0.25]], dtype=float)[:pedestrian_count]
    velocities = np.asarray([[0.2, 0.0], [-0.1, 0.3]], dtype=float)[:pedestrian_count]
    return {
        "robot": {
            "position": np.asarray([1.0, -1.0]),
            "heading": np.asarray([0.0]),
            "speed": np.asarray([0.4]),
        },
        "goal": {"current": np.asarray([5.0, 2.0])},
        "pedestrians": {
            "positions": positions,
            "velocities": velocities,
            "count": np.asarray([float(pedestrian_count)]),
        },
    }


def test_graph_feature_encoder_pads_and_masks_nodes() -> None:
    """The graph input keeps node identity and excludes padded slots."""
    nodes, mask, global_features = encode_graph_predictor_features(
        _observation(),
        np.asarray([[2.0, -0.5], [3.0, 0.25]]),
        np.asarray([[0.2, 0.0], [-0.1, 0.3]]),
        max_pedestrians=3,
    )

    assert nodes.shape == (3, 4)
    assert mask.tolist() == [1.0, 1.0, 0.0]
    np.testing.assert_allclose(nodes[0], [1.0, 0.5, 0.2, 0.0])
    np.testing.assert_allclose(nodes[2], 0.0)
    np.testing.assert_allclose(global_features, [1.0, -1.0, 0.0, 0.4, 4.0, 3.0])


def test_graph_predictor_io_dims_describe_diagonal_head() -> None:
    """The graph head emits four values per mode, not correlation parameters."""
    config = LearnedGmmPredictorConfig(
        max_pedestrians=3,
        horizon_steps=4,
        mode_count=2,
        model_type="graph_gru",
    )

    assert graph_predictor_io_dims(config) == (4, 6, 3 * 4 * 2 * 4)


def test_zero_graph_output_is_cv_with_equal_diagonal_modes() -> None:
    """Zero-initialized graph output is a diagnostic CV baseline with K modes."""
    positions = np.asarray([[2.0, -0.5]], dtype=float)
    velocities = np.asarray([[0.2, 0.0]], dtype=float)
    forecast = decode_graph_gmm_forecast(
        np.zeros(3 * 2 * 2 * 4, dtype=float),
        positions,
        velocities,
        dt=0.25,
        horizon_steps=2,
        mode_count=2,
        max_pedestrians=3,
    )

    assert forecast.means_world.shape == (1, 2, 2, 2)
    assert np.allclose(forecast.mode_weights, 0.5)
    for mode in range(2):
        np.testing.assert_allclose(
            forecast.means_world[0, mode],
            positions + velocities * np.asarray([[0.25], [0.5]]),
        )
        np.testing.assert_allclose(
            forecast.covariances_world[0, mode],
            np.tile(np.eye(2), (2, 1, 1)),
        )


def test_graph_predictor_cpu_smoke_is_explicitly_diagnostic() -> None:
    """The opt-in graph backend runs on CPU without implying trained evidence."""
    config = LearnedGmmPredictorConfig(
        max_pedestrians=3,
        horizon_steps=3,
        mode_count=2,
        hidden_dim=8,
        model_type="graph_gru",
        allow_untrained_smoke=True,
    )
    predictor = LearnedGmmPedestrianPredictor(config)
    forecast = predictor.predict(_observation(), horizon_steps=3, dt=0.25)

    assert forecast.means_world.shape == (2, 2, 3, 2)
    assert forecast.source == "diagnostic_untrained_smoke"
    assert predictor.diagnostics() == {
        "backend": "learned_graph_gru",
        "mode_count": 2,
        "evidence_tier": "diagnostic_untrained_smoke",
        "diagnostic_only": True,
        "checkpoint_path": None,
        "calls": 1,
        "last_source": "diagnostic_untrained_smoke",
        "max_pedestrians": 3,
        "horizon_steps": 3,
        "model_type": "graph_gru",
        "feature_schema": "socnav_graph_nodes_v1",
    }


def test_graph_predictor_slices_shorter_requested_horizon() -> None:
    """A graph model configured for a longer horizon serves shorter MPC requests."""
    predictor = LearnedGmmPedestrianPredictor(
        LearnedGmmPredictorConfig(
            max_pedestrians=3,
            horizon_steps=3,
            mode_count=2,
            hidden_dim=8,
            model_type="graph_gru",
            allow_untrained_smoke=True,
        )
    )

    forecast = predictor.predict(_observation(), horizon_steps=2, dt=0.25)

    assert forecast.means_world.shape == (2, 2, 2, 2)


def test_graph_predictor_rejects_excess_pedestrians() -> None:
    """Scenes over the fixed graph capacity fail closed instead of dropping nodes."""
    predictor = LearnedGmmPedestrianPredictor(
        LearnedGmmPredictorConfig(
            max_pedestrians=1,
            horizon_steps=2,
            mode_count=2,
            model_type="graph_gru",
            allow_untrained_smoke=True,
        )
    )

    with pytest.raises(ValueError, match="exceeds configured max_pedestrians"):
        predictor.predict(_observation(), horizon_steps=2, dt=0.25)


def test_graph_predictor_checkpoint_round_trip_preserves_provenance(tmp_path) -> None:
    """A graph state dict can be reloaded without relabeling smoke as evidence."""
    torch = pytest.importorskip("torch")
    smoke_config = LearnedGmmPredictorConfig(
        max_pedestrians=3,
        horizon_steps=2,
        mode_count=2,
        hidden_dim=8,
        model_type="graph_gru",
        allow_untrained_smoke=True,
    )
    smoke_predictor = LearnedGmmPedestrianPredictor(smoke_config)
    checkpoint = tmp_path / "graph_predictor.pt"
    torch.save({"state_dict": smoke_predictor._model.module.state_dict()}, checkpoint)

    loaded = LearnedGmmPedestrianPredictor(
        LearnedGmmPredictorConfig(
            checkpoint_path=str(checkpoint),
            max_pedestrians=3,
            horizon_steps=2,
            mode_count=2,
            hidden_dim=8,
            model_type="graph_gru",
        )
    )
    loaded.predict(_observation(), horizon_steps=2, dt=0.25)

    diagnostics = loaded.diagnostics()
    assert diagnostics["evidence_tier"] == "checkpoint_loaded"
    assert diagnostics["diagnostic_only"] is False


def test_graph_predictor_empty_scene_is_valid() -> None:
    """A scene without active nodes remains a valid empty GMM forecast."""
    predictor = LearnedGmmPedestrianPredictor(
        LearnedGmmPredictorConfig(
            max_pedestrians=3,
            horizon_steps=2,
            mode_count=2,
            model_type="graph_gru",
            allow_untrained_smoke=True,
        )
    )
    forecast = predictor.predict(_observation(pedestrian_count=0), horizon_steps=2, dt=0.25)

    assert forecast.means_world.shape == (0, 2, 2, 2)
    assert forecast.mode_weights.shape == (0, 2)


def test_graph_backend_is_forwarded_through_chance_mpc() -> None:
    """The existing GMM planner boundary can select the new backend explicitly."""
    adapter = build_chance_constrained_mpc_adapter(
        {
            "predictor_backend": "learned_gmm",
            "learned_gmm_model_type": "graph_gru",
            "learned_gmm_mode_count": 2,
            "learned_gmm_hidden_dim": 8,
            "learned_gmm_allow_untrained_smoke": True,
        }
    )

    command = adapter.plan(_observation(pedestrian_count=1))

    assert len(command) == 2
    assert all(np.isfinite(value) for value in command)
    predictor = adapter._multimodal_predictor
    assert predictor.diagnostics()["backend"] == "learned_graph_gru"
