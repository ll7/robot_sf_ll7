"""Regression tests for vendored SocNavBench cost functions."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _load_cost_functions(monkeypatch: pytest.MonkeyPatch) -> Any:
    socnavbench_root = Path(__file__).resolve().parents[1] / "third_party" / "socnavbench"
    monkeypatch.syspath_prepend(str(socnavbench_root))
    return importlib.import_module("metrics.cost_functions")


def test_path_length_ratio_batch_matches_scalar_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batched path ratios preserve the existing per-trajectory scalar formula."""
    cost_functions = _load_cost_functions(monkeypatch)
    trajectories = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 3.0], [4.0, 3.0]],
        ],
        dtype=float,
    )
    goals = np.array([[2.0, 0.0], [4.0, 3.0]], dtype=float)

    batched = cost_functions.path_length_ratio_batch(trajectories, goals)
    scalar = np.array(
        [
            cost_functions.path_length_ratio(trajectories[0], goals[0]),
            cost_functions.path_length_ratio(trajectories[1], goals[1]),
        ],
        dtype=float,
    )

    assert batched == pytest.approx(scalar)


def test_path_length_ratio_return_batch_keeps_scalar_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in batch flag avoids changing the default scalar API."""
    cost_functions = _load_cost_functions(monkeypatch)
    trajectories = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="single-trajectory batch"):
        cost_functions.path_length_ratio(trajectories)

    batched = cost_functions.path_length_ratio(trajectories, return_batch=True)
    assert batched.shape == (2,)
    assert batched == pytest.approx(np.array([1.0 / 1.00001, 2.0 / 2.00001]))


def test_path_length_ratio_batch_broadcasts_single_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One goal config may be reused across a trajectory batch."""
    cost_functions = _load_cost_functions(monkeypatch)
    trajectories = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
        dtype=float,
    )

    batched = cost_functions.path_length_ratio_batch(trajectories, np.array([1.0, 0.0]))
    assert batched == pytest.approx(np.array([1.0 / 1.00001, np.sqrt(2.0) / 1.00001]))
