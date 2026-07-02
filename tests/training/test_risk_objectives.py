"""Tests for issue #4016 risk objectives over quantile estimates."""

from __future__ import annotations

import pytest
import torch

from robot_sf.training.risk_objectives import (
    quantile_cvar_lower,
    quantile_mean,
    quantile_var_lower,
    score_action_quantiles,
)


def test_quantile_mean_equals_arithmetic_average() -> None:
    """Mean objective should match standard expected-return action scoring."""

    quantiles = torch.tensor([[[1.0, 3.0, 5.0], [0.0, 2.0, 4.0]]])

    assert torch.equal(quantile_mean(quantiles), torch.tensor([[3.0, 2.0]]))


def test_var_lower_selects_ordered_lower_quantile() -> None:
    """Lower VaR should select the quantile bucket implied by alpha."""

    quantiles = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

    assert torch.equal(quantile_var_lower(quantiles, alpha=0.2), torch.tensor([[1.0]]))
    assert torch.equal(quantile_var_lower(quantiles, alpha=0.4), torch.tensor([[2.0]]))


def test_cvar_lower_averages_bottom_quantiles() -> None:
    """Lower CVaR should average only lower-tail quantile buckets."""

    quantiles = torch.tensor([[[2.0, 4.0, 6.0, 8.0, 10.0]]])

    assert torch.equal(quantile_cvar_lower(quantiles, alpha=0.2), torch.tensor([[2.0]]))
    assert torch.equal(quantile_cvar_lower(quantiles, alpha=0.4), torch.tensor([[3.0]]))


def test_score_action_quantiles_preserves_batch_action_dimensions() -> None:
    """Risk scoring should collapse only the quantile axis."""

    quantiles = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

    for objective in ("mean", "var_lower", "cvar_lower", "cvar_blend"):
        scores = score_action_quantiles(
            quantiles,
            objective=objective,
            alpha=0.5,
            blend_beta=0.25,
        )
        assert scores.shape == (2, 3)


def test_score_action_quantiles_blends_mean_and_cvar() -> None:
    """Blended score should interpolate expected value and lower-tail risk."""

    quantiles = torch.tensor([[[1.0, 3.0, 5.0, 7.0]]])

    assert torch.equal(
        score_action_quantiles(quantiles, objective="cvar_blend", alpha=0.5, blend_beta=0.5),
        torch.tensor([[3.0]]),
    )


def test_invalid_alpha_and_objective_fail_closed() -> None:
    """Risk-objective configuration errors should be explicit."""

    quantiles = torch.tensor([[[1.0, 2.0]]])

    with pytest.raises(ValueError, match="alpha"):
        quantile_var_lower(quantiles, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        quantile_cvar_lower(quantiles, alpha=1.5)
    with pytest.raises(ValueError, match="unknown risk objective"):
        score_action_quantiles(quantiles, objective="optimistic", alpha=0.5)
    with pytest.raises(ValueError, match="blend_beta"):
        score_action_quantiles(quantiles, objective="cvar_blend", alpha=0.5, blend_beta=2.0)
