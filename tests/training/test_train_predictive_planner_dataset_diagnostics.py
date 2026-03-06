"""Tests for predictive training dataset diagnostics."""

from __future__ import annotations

import numpy as np

from scripts.training import train_predictive_planner as trainer


def test_dataset_diagnostics_flags_degenerate_targets() -> None:
    """Zero-spread target trajectories should be flagged as degenerate."""
    n, a, t = 80, 4, 8
    state = np.zeros((n, a, 4), dtype=np.float32)
    target = np.zeros((n, a, t, 2), dtype=np.float32)
    mask = np.ones((n, a), dtype=np.float32)
    target_mask = np.ones((n, a, t), dtype=np.float32)

    diag = trainer._dataset_diagnostics(
        state=state,
        target=target,
        mask=mask,
        target_mask=target_mask,
    )
    assert diag["is_degenerate"] is True
    assert "target_values_near_constant" in diag["fail_reasons"]


def test_dataset_diagnostics_accepts_spread_trajectories() -> None:
    """Moving trajectories with active masks should pass degeneracy checks."""
    n, a, t = 120, 3, 6
    rng = np.random.default_rng(3)
    state = rng.normal(size=(n, a, 4)).astype(np.float32)
    base = rng.normal(size=(n, a, 1, 2)).astype(np.float32)
    delta = rng.normal(size=(n, a, t, 2)).astype(np.float32) * 0.2
    target = base + np.cumsum(delta, axis=2)
    mask = np.ones((n, a), dtype=np.float32)
    target_mask = np.ones((n, a, t), dtype=np.float32)

    diag = trainer._dataset_diagnostics(
        state=state,
        target=target,
        mask=mask,
        target_mask=target_mask,
    )
    assert diag["is_degenerate"] is False
