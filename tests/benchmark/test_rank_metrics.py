"""Characterization tests for shared rank-correlation helpers."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.rank_metrics import (
    kendall_tau,
    kendall_tau_by_value,
    rank_by,
    rank_order,
    spearman,
    spearman_by_value,
    spearman_from_order,
    spearman_from_rank_maps,
    top_tied,
)


def test_rank_by_uses_average_ties_and_direction() -> None:
    """Average-rank semantics match the legacy benchmark helper convention."""
    values = {"beta": 2.0, "alpha": 2.0, "gamma": 1.0}

    assert rank_by(values, higher_is_better=True) == {
        "alpha": 1.5,
        "beta": 1.5,
        "gamma": 3.0,
    }
    assert rank_by(values, higher_is_better=False) == {
        "gamma": 1.0,
        "alpha": 2.5,
        "beta": 2.5,
    }


def test_rank_order_and_top_tied_preserve_metric_ranking_script_behavior() -> None:
    """Ordering stays deterministic while top-tie detection ignores name tie-breaks."""
    values = {"zulu": 1.0, "alpha": 1.0, "middle": 0.5}

    assert rank_order(values, higher_is_better=True) == ["alpha", "zulu", "middle"]
    assert top_tied(values, higher_is_better=True, tie_abs_tol=1e-12) is True
    assert top_tied({"zulu": 1.0, "alpha": 0.9}, higher_is_better=True) is False


def test_spearman_matches_legacy_degenerate_and_tie_contracts() -> None:
    """Numeric Spearman keeps the SNQI campaign contract degenerate behavior."""
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert spearman([1.0], [1.0], degenerate=0.0) == 0.0
    assert spearman([1.0, 1.0], [1.0, 2.0], degenerate=None) is None


def test_order_correlations_match_prior_order_index_helpers() -> None:
    """Order-based helpers match the prior script-local index comparison."""
    left = ["a", "b", "c"]
    swapped = ["b", "a", "c"]

    assert spearman_from_order(left, swapped, degenerate=0.0) == pytest.approx(0.5)
    assert kendall_tau(left, swapped, degenerate=0.0) == pytest.approx(1.0 / 3.0)
    assert kendall_tau(left, list(reversed(left)), degenerate=0.0) == pytest.approx(-1.0)
    assert kendall_tau(["a"], ["a"], degenerate=1.0) == 1.0


def test_rank_map_spearman_preserves_absolute_rank_position_contract() -> None:
    """Rank-map Spearman keeps scenario-difficulty's prior non-renormalized overlap behavior."""
    assert spearman_from_rank_maps({"b": 2, "c": 3}, {"b": 1, "c": 2}, degenerate=None) == -1.0
    assert spearman_from_rank_maps({"only": 1}, {"only": 1}, degenerate=None) is None


def test_value_correlations_preserve_seed_schedule_tie_handling() -> None:
    """Value-map helpers preserve average Spearman ranks and tie-skipping Kendall tau."""
    base = {("alpha", "dd"): 1.0, ("beta", "dd"): 1.0, ("gamma", "dd"): 3.0}
    candidate = {("alpha", "dd"): 1.0, ("beta", "dd"): 2.0, ("gamma", "dd"): 3.0}

    assert spearman_by_value(
        base,
        candidate,
        higher_is_better=True,
        degenerate=None,
    ) == pytest.approx(0.8660254038)
    assert kendall_tau_by_value(
        base,
        candidate,
        higher_is_better=True,
        degenerate=None,
        tie_abs_tol=1e-12,
    ) == pytest.approx(1.0)
    assert (
        kendall_tau_by_value(
            {"alpha": 1.0, "beta": 1.0},
            {"alpha": 2.0, "beta": 2.0},
            higher_is_better=True,
            degenerate=None,
        )
        is None
    )
