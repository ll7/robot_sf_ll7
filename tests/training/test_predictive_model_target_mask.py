"""Regression tests for predictive trajectory masking semantics."""

from __future__ import annotations

import torch

from robot_sf.planner.predictive_model import compute_ade_fde, masked_trajectory_loss


def test_masked_trajectory_loss_respects_target_mask() -> None:
    """Only entries marked in target_mask should contribute to loss."""
    predicted = torch.tensor([[[[2.0, 0.0], [2.0, 0.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
    mask = torch.tensor([[1.0]], dtype=torch.float32)

    no_target = torch.zeros((1, 1, 2), dtype=torch.float32)
    with_target = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)

    loss_none = float(masked_trajectory_loss(predicted, target, mask, no_target).item())
    loss_some = float(masked_trajectory_loss(predicted, target, mask, with_target).item())

    assert loss_none == 0.0
    assert loss_some > 0.0


def test_compute_ade_fde_respects_target_mask() -> None:
    """ADE/FDE should ignore timesteps where target_mask is zero."""
    predicted = torch.tensor([[[[1.0, 0.0], [5.0, 0.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
    mask = torch.tensor([[1.0]], dtype=torch.float32)

    # Only first horizon step is valid.
    target_mask = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    ade, fde = compute_ade_fde(predicted, target, mask, target_mask)

    assert ade > 0.0
    # FDE uses only final-step validity and should be zero if final step is invalid.
    assert fde == 0.0
