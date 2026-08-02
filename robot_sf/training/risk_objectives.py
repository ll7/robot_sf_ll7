"""Risk objectives over ordered return quantile estimates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

RISK_OBJECTIVES = ("mean", "var_lower", "cvar_lower", "cvar_blend")


def _validate_quantiles(quantiles: torch.Tensor) -> None:
    """Validate the shared quantile tensor contract."""
    import torch  # noqa: PLC0415

    if quantiles.ndim < 1:
        raise ValueError("quantiles must have at least one dimension")
    if quantiles.shape[-1] < 1:
        raise ValueError("quantiles must include at least one quantile estimate")
    if not torch.is_floating_point(quantiles):
        raise TypeError("quantiles must be a floating-point tensor")


def _lower_count(num_quantiles: int, alpha: float) -> int:
    """Return how many ordered quantiles participate in a lower-tail objective.

    Returns:
        int: Number of quantiles.
    """
    if not 0.0 < alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and in the interval (0, 1]")
    return max(1, math.ceil(num_quantiles * alpha))


def quantile_mean(quantiles: torch.Tensor) -> torch.Tensor:
    """Score each action by arithmetic mean return.

    Returns:
        torch.Tensor: Mean score tensor.
    """
    _validate_quantiles(quantiles)
    return quantiles.mean(dim=-1)


def quantile_var_lower(quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
    """Score each action by its lower-tail value-at-risk estimate.

    Returns:
        torch.Tensor: VaR score tensor.
    """
    _validate_quantiles(quantiles)
    index = _lower_count(quantiles.shape[-1], alpha) - 1
    return quantiles.select(dim=-1, index=index)


def quantile_cvar_lower(quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
    """Score each action by mean lower-tail return.

    Returns:
        torch.Tensor: CVaR score tensor.
    """
    _validate_quantiles(quantiles)
    count = _lower_count(quantiles.shape[-1], alpha)
    return quantiles.narrow(dim=-1, start=0, length=count).mean(dim=-1)


def score_action_quantiles(
    quantiles: torch.Tensor,
    *,
    objective: str,
    alpha: float,
    blend_beta: float = 1.0,
) -> torch.Tensor:
    """Collapse action quantiles to scalar action scores for selection.

    Returns:
        torch.Tensor: Action scores tensor.
    """
    if objective == "mean":
        return quantile_mean(quantiles)
    if objective == "var_lower":
        return quantile_var_lower(quantiles, alpha)
    if objective == "cvar_lower":
        return quantile_cvar_lower(quantiles, alpha)
    if objective == "cvar_blend":
        if not 0.0 <= blend_beta <= 1.0 or not math.isfinite(blend_beta):
            raise ValueError("blend_beta must be finite and in the interval [0, 1]")
        mean_score = quantile_mean(quantiles)
        cvar_score = quantile_cvar_lower(quantiles, alpha)
        return (1.0 - blend_beta) * mean_score + blend_beta * cvar_score
    raise ValueError(f"unknown risk objective {objective!r}; expected one of {RISK_OBJECTIVES}")
