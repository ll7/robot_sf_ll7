"""Tests for issue #4016 QR-DQN primitive utilities."""

from __future__ import annotations

import pytest
import torch

from robot_sf.training.discrete_action_lattice import DiscreteUnicycleActionLattice
from robot_sf.training.distributional_rl import (
    QuantileQNetwork,
    build_qr_dqn_targets,
    fixed_quantile_fractions,
    hard_update_target_network,
    load_quantile_checkpoint_metadata,
    quantile_huber_loss,
    save_quantile_checkpoint,
    select_action_quantiles,
)


def test_quantile_network_output_shape() -> None:
    """Network output should be `[batch, action_count, num_quantiles]`."""

    model = QuantileQNetwork(observation_dim=4, action_count=3, num_quantiles=5, hidden_sizes=(8,))
    observations = torch.zeros((2, 4), dtype=torch.float32)

    assert model(observations).shape == (2, 3, 5)


def test_fixed_quantile_fractions_are_midpoints() -> None:
    """QR-DQN fixed fractions should be quantile-bin midpoints."""

    assert torch.allclose(
        fixed_quantile_fractions(4),
        torch.tensor([0.125, 0.375, 0.625, 0.875]),
    )


def test_quantile_huber_loss_is_finite() -> None:
    """Quantile regression loss should produce a scalar finite tensor."""

    predicted = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    target = torch.tensor([[1.5, 2.5, 3.5], [0.5, 1.5, 2.5]])

    loss = quantile_huber_loss(predicted, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_select_action_quantiles_gathers_per_batch_distribution() -> None:
    """Action selection should gather one quantile distribution per batch row."""

    quantiles = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    selected = select_action_quantiles(quantiles, torch.tensor([2, 1]))

    assert torch.equal(selected, torch.stack([quantiles[0, 2], quantiles[1, 1]]))


def test_select_action_quantiles_rejects_out_of_range_indices() -> None:
    """Action indices fail closed before low-level gather assertions."""
    quantiles = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

    with pytest.raises(ValueError, match=r"action_indices must be in range"):
        select_action_quantiles(quantiles, torch.tensor([3, 1]))

    with pytest.raises(ValueError, match=r"action_indices must be in range"):
        select_action_quantiles(quantiles, torch.tensor([0, -1]))


def test_double_q_target_construction_has_expected_shape() -> None:
    """Bellman target construction should preserve the quantile axis."""

    target_next_quantiles = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    batch = build_qr_dqn_targets(
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([0.0, 1.0]),
        target_next_quantiles=target_next_quantiles,
        next_action_indices=torch.tensor([1, 2]),
        gamma=0.9,
    )

    assert batch.target_quantiles.shape == (2, 4)
    assert torch.allclose(batch.target_quantiles[0], 1.0 + 0.9 * target_next_quantiles[0, 1])
    assert torch.allclose(batch.target_quantiles[1], torch.full((4,), 2.0))


def test_hard_update_target_network_copies_parameters() -> None:
    """Target-network sync should exactly copy online weights."""

    source = QuantileQNetwork(observation_dim=2, action_count=2, num_quantiles=3, hidden_sizes=(4,))
    target = QuantileQNetwork(observation_dim=2, action_count=2, num_quantiles=3, hidden_sizes=(4,))

    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
        for parameter in target.parameters():
            parameter.fill_(-0.25)

    hard_update_target_network(source, target)

    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_param, target_param)


def test_checkpoint_round_trip_restores_action_and_quantile_metadata(tmp_path) -> None:
    """Primitive checkpoints should carry enough metadata for adapter slices."""

    model = QuantileQNetwork(observation_dim=2, action_count=2, num_quantiles=7, hidden_sizes=(4,))
    lattice = DiscreteUnicycleActionLattice(
        linear_values=(0.0, 0.4),
        angular_values=(0.0,),
        max_linear_speed=0.4,
        max_angular_speed=1.0,
    )
    path = tmp_path / "qr_dqn.pt"

    save_quantile_checkpoint(path, model=model, action_lattice=lattice)
    metadata = load_quantile_checkpoint_metadata(path)

    assert metadata["model_metadata"]["action_count"] == 2
    assert metadata["model_metadata"]["num_quantiles"] == 7
    assert metadata["model_metadata"]["claim_boundary"] == "primitive-only; not benchmark evidence"
    assert metadata["action_lattice"]["command_space"] == "unicycle_vw"
