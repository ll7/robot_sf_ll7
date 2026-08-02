"""RGL-inspired predictive model for crowd trajectory forecasting.

This module provides a compact graph-message-passing predictor used by the
prediction planner adapter and training scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.planner.obstacle_features import (
    PREDICTIVE_LEGACY_FEATURE_SCHEMA,
    ObstacleFeatureSchemaError,
    infer_predictive_feature_schema,
    validate_predictive_feature_schema_metadata,
)

if TYPE_CHECKING:
    import torch
    from torch import Tensor


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
    input_dim: int = 4
    hidden_dim: int = 96
    message_passing_steps: int = 2
    distance_temperature: float = 2.0
    feature_schema_name: str = PREDICTIVE_LEGACY_FEATURE_SCHEMA


def _init_classes() -> dict[str, Any]:  # noqa: C901
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class _MessageBlock(nn.Module):
        """Single message-passing block over agent features."""

        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.update = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, h: Tensor, msg: Tensor) -> Tensor:
            return h + self.update(torch.cat([h, msg], dim=-1))

    class PredictiveTrajectoryModel(nn.Module):
        """Graph-based predictor that forecasts future pedestrian positions."""

        def __init__(self, config: PredictiveModelConfig) -> None:
            super().__init__()
            self.config = config
            h = int(config.hidden_dim)
            self.encoder = nn.Sequential(
                nn.Linear(int(config.input_dim), h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
            )
            self.blocks = nn.ModuleList(
                _MessageBlock(h) for _ in range(config.message_passing_steps)
            )
            self.decoder = nn.Sequential(
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, int(config.horizon_steps) * 2),
            )

        def _attention_weights(self, state: Tensor, mask: Tensor) -> Tensor:
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
        target_mask: Tensor | None = None,
        *,
        horizon_weights: Tensor | None = None,
    ) -> Tensor:
        errors = torch.nn.functional.smooth_l1_loss(predicted, target, reduction="none")
        if horizon_weights is not None:
            w = horizon_weights.view(1, 1, -1, 1)
            errors = errors * w

        if target_mask is None:
            valid = mask[:, :, None].float().clamp(0.0, 1.0)
        else:
            valid = target_mask.float().clamp(0.0, 1.0)
        valid_xy = valid[:, :, :, None]
        errors = errors * valid_xy
        denom = torch.clamp(valid_xy.sum(), min=1.0)
        return errors.sum() / denom

    def compute_ade_fde(
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
        target_mask: Tensor | None = None,
    ) -> tuple[float, float]:
        with torch.no_grad():
            diff = torch.linalg.norm(predicted - target, dim=-1)
            if target_mask is None:
                valid = mask[:, :, None].float().clamp(0.0, 1.0)
            else:
                valid = target_mask.float().clamp(0.0, 1.0)
            ade = (diff * valid).sum() / torch.clamp(valid.sum(), min=1.0)
            fde_valid = valid[:, :, -1]
            fde = (diff[:, :, -1] * fde_valid).sum() / torch.clamp(fde_valid.sum(), min=1.0)
        return float(ade.item()), float(fde.item())

    def save_predictive_checkpoint(
        path: str | Path,
        *,
        model: PredictiveTrajectoryModel,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
        feature_schema_metadata: dict[str, object] | None = None,
    ) -> None:
        feature_schema = feature_schema_metadata or infer_predictive_feature_schema(
            int(model.config.input_dim)
        )
        validate_predictive_feature_schema_metadata(
            feature_schema,
            input_dim=int(model.config.input_dim),
            expected_schema_name=str(model.config.feature_schema_name),
        )
        payload = {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics or {},
            "extra": extra or {},
            "feature_schema": feature_schema,
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
        expected_feature_schema_name: str | None = None,
        expected_input_dim: int | None = None,
    ) -> tuple[PredictiveTrajectoryModel, dict[str, Any]]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        config_data = payload.get("config", {})
        config = PredictiveModelConfig(**config_data)
        feature_schema = payload.get("feature_schema")
        if not isinstance(feature_schema, dict):
            feature_schema = infer_predictive_feature_schema(int(config.input_dim))
        validate_predictive_feature_schema_metadata(
            feature_schema,
            input_dim=int(config.input_dim),
            expected_schema_name=expected_feature_schema_name,
        )
        if expected_input_dim is not None and int(expected_input_dim) != int(config.input_dim):
            raise ObstacleFeatureSchemaError(
                "Predictive checkpoint input_dim mismatch: "
                f"expected {int(expected_input_dim)}, got {int(config.input_dim)}"
            )
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

    return {
        "_MessageBlock": _MessageBlock,
        "PredictiveTrajectoryModel": PredictiveTrajectoryModel,
        "masked_trajectory_loss": masked_trajectory_loss,
        "compute_ade_fde": compute_ade_fde,
        "save_predictive_checkpoint": save_predictive_checkpoint,
        "load_predictive_checkpoint": load_predictive_checkpoint,
    }


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {
    "_MessageBlock",
    "PredictiveTrajectoryModel",
    "masked_trajectory_loss",
    "compute_ade_fde",
    "save_predictive_checkpoint",
    "load_predictive_checkpoint",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        global _cache
        if _cache is None:
            _cache = _init_classes()
            for lazy_name, value in _cache.items():
                value.__qualname__ = lazy_name
                globals()[lazy_name] = value
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List lazy exports without importing optional ML dependencies.

    Returns:
        All module globals and deferred export names.
    """
    return sorted(set(globals()) | _LAZY_NAMES)


__all__ = [  # noqa: F822
    "PredictiveModelConfig",
    "PredictiveTrajectoryModel",
    "compute_ade_fde",
    "load_predictive_checkpoint",
    "masked_trajectory_loss",
    "save_predictive_checkpoint",
]
