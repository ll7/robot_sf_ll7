"""Unit tests for predictive planner trajectory model."""

from dataclasses import asdict

import torch

from robot_sf.planner.predictive_model import (
    PredictiveModelConfig,
    PredictiveTrajectoryModel,
    compute_ade_fde,
    load_predictive_checkpoint,
    masked_trajectory_loss,
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
