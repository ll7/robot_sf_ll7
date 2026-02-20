"""RGL-inspired predictive model for crowd trajectory forecasting.

This module provides a compact graph-message-passing predictor used by the
prediction planner adapter and training scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn


@lru_cache(maxsize=32)
def _log_unexpected_checkpoint_keys(signature: tuple[str, ...]) -> None:
    """Log one compatibility warning per distinct unexpected-key signature."""
    logger.debug(
        "Ignoring unexpected checkpoint keys when loading PredictiveTrajectoryModel: {}",
        list(signature),
    )


@dataclass
class PredictiveModelConfig:
    """Configuration for the predictive trajectory model."""

    max_agents: int = 16
    horizon_steps: int = 8
    hidden_dim: int = 96
    message_passing_steps: int = 2
    distance_temperature: float = 2.0


class _MessageBlock(nn.Module):
    """Single message-passing block over agent features."""

    def __init__(self, hidden_dim: int) -> None:
        """Initialize MLP layers for message passing."""
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, h: Tensor, msg: Tensor) -> Tensor:
        """Return updated node features from current features and aggregated messages.

        Returns:
            Tensor: Updated hidden representation with shape ``(B, N, H)``.
        """
        return h + self.update(torch.cat([h, msg], dim=-1))


class PredictiveTrajectoryModel(nn.Module):
    """Graph-based predictor that forecasts future pedestrian positions.

    Input state is a tensor of shape ``(B, N, 4)`` containing pedestrian features:
    ``(x_rel, y_rel, vx_rel, vy_rel)`` in the robot frame.
    """

    def __init__(self, config: PredictiveModelConfig) -> None:
        """Initialize encoder, graph blocks, and decoder heads."""
        super().__init__()
        self.config = config
        h = int(config.hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(4, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(_MessageBlock(h) for _ in range(config.message_passing_steps))
        self.decoder = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, int(config.horizon_steps) * 2),
        )

    def _attention_weights(self, state: Tensor, mask: Tensor) -> Tensor:
        """Compute distance-based attention weights for each agent pair.

        Returns:
            Tensor: Pairwise attention matrix with shape ``(B, N, N)``.
        """
        pos = state[:, :, :2]
        rel = pos[:, :, None, :] - pos[:, None, :, :]
        dist_sq = torch.sum(rel * rel, dim=-1)
        logits = -dist_sq / max(float(self.config.distance_temperature), 1e-6)

        pair_mask = mask[:, :, None] * mask[:, None, :]
        logits = logits.masked_fill(pair_mask <= 0.0, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        return attn

    def forward(self, state: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """Predict future pedestrian positions in the robot frame.

        Args:
            state: Agent features ``(B, N, 4)``.
            mask: Agent validity mask ``(B, N)`` with 1 for active slots.

        Returns:
            dict[str, Tensor]:
                - ``future_positions``: ``(B, N, T, 2)`` absolute positions in robot frame.
        """
        mask = mask.float().clamp(0.0, 1.0)
        h = self.encoder(state)
        h = h * mask.unsqueeze(-1)

        attn = self._attention_weights(state, mask)
        for block in self.blocks:
            msg = torch.matmul(attn, h)
            h = block(h, msg)
            h = h * mask.unsqueeze(-1)

        raw = self.decoder(h)
        steps = int(self.config.horizon_steps)
        delta = raw.view(raw.shape[0], raw.shape[1], steps, 2)
        future = state[:, :, None, :2] + torch.cumsum(delta, dim=2)
        future = future * mask[:, :, None, None]

        return {"future_positions": future}


def masked_trajectory_loss(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    horizon_weights: Tensor | None = None,
) -> Tensor:
    """Compute SmoothL1 loss over masked trajectory predictions.

    Returns:
        Tensor: Scalar loss.
    """
    errors = torch.nn.functional.smooth_l1_loss(predicted, target, reduction="none")
    if horizon_weights is not None:
        w = horizon_weights.view(1, 1, -1, 1)
        errors = errors * w

    slot_mask = mask[:, :, None, None].float().clamp(0.0, 1.0)
    errors = errors * slot_mask
    denom = torch.clamp(slot_mask.sum(), min=1.0)
    return errors.sum() / denom


def compute_ade_fde(predicted: Tensor, target: Tensor, mask: Tensor) -> tuple[float, float]:
    """Return average and final displacement error on valid trajectories.

    Returns:
        tuple[float, float]: ``(ADE, FDE)`` in meters.
    """
    with torch.no_grad():
        diff = torch.linalg.norm(predicted - target, dim=-1)
        valid = mask[:, :, None].float().clamp(0.0, 1.0)
        ade = (diff * valid).sum() / torch.clamp(valid.sum(), min=1.0)
        fde = (diff[:, :, -1] * mask.float()).sum() / torch.clamp(mask.sum(), min=1.0)
    return float(ade.item()), float(fde.item())


def save_predictive_checkpoint(
    path: str | Path,
    *,
    model: PredictiveTrajectoryModel,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist model/optimizer/config/metrics to a checkpoint file."""
    payload = {
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


def load_predictive_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PredictiveTrajectoryModel, dict[str, Any]]:
    """Load a predictive model checkpoint.

    Returns:
        tuple[PredictiveTrajectoryModel, dict[str, Any]]: Instantiated model and raw payload.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    config_data = payload.get("config", {})
    config = PredictiveModelConfig(**config_data)
    model = PredictiveTrajectoryModel(config)
    state_dict = payload["state_dict"]
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        signature = tuple(sorted(load_result.unexpected_keys))
        _log_unexpected_checkpoint_keys(signature)
    if load_result.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing required PredictiveTrajectoryModel keys: "
            f"{sorted(load_result.missing_keys)}"
        )
    model.eval()
    return model, payload


__all__ = [
    "PredictiveModelConfig",
    "PredictiveTrajectoryModel",
    "compute_ade_fde",
    "load_predictive_checkpoint",
    "masked_trajectory_loss",
    "save_predictive_checkpoint",
]
