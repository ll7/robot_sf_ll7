"""Unit tests for predictive planner trajectory model."""

from dataclasses import asdict

import torch

from robot_sf.planner.obstacle_features import (
    PREDICTIVE_EGO_FEATURE_SCHEMA,
    PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE,
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    ObstacleFeatureSchemaError,
    predictive_feature_schema_metadata,
    validate_predictive_feature_schema_metadata,
)
from robot_sf.planner.predictive_model import (
    PredictiveModelConfig,
    PredictiveTrajectoryModel,
    compute_ade_fde,
    load_predictive_checkpoint,
    masked_trajectory_loss,
    save_predictive_checkpoint,
)


def test_predictive_model_forward_shapes() -> None:
    """Model forward pass should return expected tensor shapes."""
    cfg = PredictiveModelConfig(
        max_agents=6, horizon_steps=5, hidden_dim=32, message_passing_steps=1
    )
    model = PredictiveTrajectoryModel(cfg)
    state = torch.randn(4, cfg.max_agents, 4)
    mask = torch.ones(4, cfg.max_agents)
    out = model(state, mask)
    assert out["future_positions"].shape == (4, cfg.max_agents, cfg.horizon_steps, 2)


def test_predictive_model_forward_accepts_ego_conditioned_state() -> None:
    """Model should accept wider ego-conditioned agent features when configured."""
    cfg = PredictiveModelConfig(
        max_agents=5,
        horizon_steps=4,
        input_dim=9,
        hidden_dim=24,
        message_passing_steps=1,
    )
    model = PredictiveTrajectoryModel(cfg)
    state = torch.randn(3, cfg.max_agents, cfg.input_dim)
    mask = torch.ones(3, cfg.max_agents)
    out = model(state, mask)
    assert out["future_positions"].shape == (3, cfg.max_agents, cfg.horizon_steps, 2)


def test_masked_trajectory_loss_respects_mask() -> None:
    """Loss should ignore masked-out trajectory slots."""
    pred = torch.zeros(1, 2, 3, 2)
    target = torch.ones(1, 2, 3, 2)
    mask = torch.tensor([[1.0, 0.0]])
    loss = masked_trajectory_loss(pred, target, mask)
    assert float(loss.item()) > 0.0

    full_mask = torch.tensor([[1.0, 1.0]])
    loss_full = masked_trajectory_loss(pred, target, full_mask)
    assert float(loss_full.item()) == float(loss.item())


def test_compute_ade_fde_zero_on_exact_match() -> None:
    """ADE/FDE should be zero when predictions match targets exactly."""
    target = torch.randn(2, 4, 6, 2)
    pred = target.clone()
    mask = torch.ones(2, 4)
    ade, fde = compute_ade_fde(pred, target, mask)
    assert ade == 0.0
    assert fde == 0.0


def test_load_predictive_checkpoint_ignores_unexpected_aux_head_keys(tmp_path) -> None:
    """Checkpoint loading should ignore extra auxiliary-head keys and keep core model valid."""
    cfg = PredictiveModelConfig(
        max_agents=6, horizon_steps=5, hidden_dim=32, message_passing_steps=1
    )
    model = PredictiveTrajectoryModel(cfg)
    payload = {
        "config": asdict(cfg),
        "state_dict": dict(model.state_dict()),
    }
    payload["state_dict"]["value_head.0.weight"] = torch.randn(8, cfg.hidden_dim)
    payload["state_dict"]["value_head.0.bias"] = torch.randn(8)

    checkpoint = tmp_path / "predictive_model_aux.pt"
    torch.save(payload, checkpoint)

    loaded, _ = load_predictive_checkpoint(checkpoint)

    state = torch.randn(2, cfg.max_agents, 4)
    mask = torch.ones(2, cfg.max_agents)
    out = loaded(state, mask)
    assert out["future_positions"].shape == (2, cfg.max_agents, cfg.horizon_steps, 2)


def test_predictive_checkpoint_records_and_validates_feature_schema(tmp_path) -> None:
    """Checkpoint loading should fail closed when requested schema does not match metadata."""
    cfg = PredictiveModelConfig(
        max_agents=4,
        horizon_steps=3,
        input_dim=10,
        hidden_dim=16,
        message_passing_steps=1,
        feature_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    )
    model = PredictiveTrajectoryModel(cfg)
    checkpoint = tmp_path / "predictive_obstacle.pt"

    save_predictive_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        epoch=1,
        feature_schema_metadata=predictive_feature_schema_metadata(
            model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            ego_conditioning=False,
        ),
    )

    _loaded, payload = load_predictive_checkpoint(
        checkpoint,
        expected_feature_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        expected_input_dim=10,
    )
    assert payload["feature_schema"]["name"] == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA

    try:
        load_predictive_checkpoint(
            checkpoint,
            expected_feature_schema_name="predictive_legacy_v1",
        )
    except ObstacleFeatureSchemaError as exc:
        assert "Predictive feature schema mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ObstacleFeatureSchemaError")


def test_predictive_schema_validation_rejects_legacy_name_with_ego_dimension() -> None:
    """Legacy and ego schema names should remain tied to their exact input widths."""
    try:
        validate_predictive_feature_schema_metadata(
            {
                "name": "predictive_legacy_v1",
                "base_schema": "predictive_legacy_v1",
                "base_feature_dim": 4,
                "obstacle_feature_schema": None,
                "input_dim": 9,
            },
            input_dim=9,
            expected_schema_name="predictive_legacy_v1",
        )
    except ObstacleFeatureSchemaError as exc:
        assert "Predictive legacy feature dimension mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ObstacleFeatureSchemaError")

    validate_predictive_feature_schema_metadata(
        predictive_feature_schema_metadata(model_family=PREDICTIVE_EGO_FEATURE_SCHEMA),
        input_dim=9,
        expected_schema_name=PREDICTIVE_EGO_FEATURE_SCHEMA,
    )


def test_predictive_schema_metadata_records_ego_motion_producers() -> None:
    """Ego-conditioned schema metadata should encode the active motion-channel producer."""
    ego_schema = predictive_feature_schema_metadata(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        ego_motion_channel_producer=PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE,
    )
    ego_obstacle_schema = predictive_feature_schema_metadata(
        model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        ego_conditioning=True,
        ego_motion_channel_producer=PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    )

    assert ego_schema["ego_motion_channel_producer"]["producer_key"] == (
        PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE
    )
    assert ego_obstacle_schema["base_schema"] == PREDICTIVE_EGO_FEATURE_SCHEMA
    assert ego_obstacle_schema["input_dim"] == 15
    assert ego_obstacle_schema["ego_motion_channel_producer"]["producer_key"] == (
        PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME
    )

    malformed_missing_key = predictive_feature_schema_metadata(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        ego_motion_channel_producer=PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    )
    del malformed_missing_key["ego_motion_channel_producer"]["producer_key"]
    try:
        validate_predictive_feature_schema_metadata(
            malformed_missing_key,
            input_dim=9,
            expected_schema_name=PREDICTIVE_EGO_FEATURE_SCHEMA,
        )
    except ObstacleFeatureSchemaError as exc:
        assert "producer_key" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ObstacleFeatureSchemaError")

    non_ego_with_ego_producer = predictive_feature_schema_metadata(
        model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        ego_conditioning=False,
    )
    non_ego_with_ego_producer["ego_motion_channel_producer"] = {
        "producer_key": PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
        "slot_range": [4, 5],
    }
    try:
        validate_predictive_feature_schema_metadata(
            non_ego_with_ego_producer,
            input_dim=10,
            expected_schema_name=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        )
    except ObstacleFeatureSchemaError as exc:
        assert "only valid for" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ObstacleFeatureSchemaError")
