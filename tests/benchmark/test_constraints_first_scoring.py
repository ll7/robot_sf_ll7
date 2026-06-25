"""Tests for the constraints-first scoring layer (issue #3572)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.constraints_first_scoring import (
    CONSTRAINTS_FIRST_SCHEMA,
    AdmissibilityGates,
    collision_upper_confidence_bound,
    constraints_first_planner_summary,
    is_episode_admissible,
    ranking_inversion,
    survivorship_aware_metric,
)


def test_collision_ucb_reproduces_rule_of_three() -> None:
    """Zero observed collisions must still yield a non-trivial upper bound (~3/N)."""
    ub = collision_upper_confidence_bound(0, 100)

    assert ub == pytest.approx(1 - 0.05 ** (1 / 100))
    assert ub == pytest.approx(0.03, abs=0.005)


def test_collision_ucb_is_one_when_all_collide() -> None:
    """If every episode collides the upper bound is 1.0."""
    assert collision_upper_confidence_bound(5, 5) == 1.0


def test_collision_ucb_decreases_with_more_episodes() -> None:
    """The same zero-collision rate must tighten as N grows."""
    assert collision_upper_confidence_bound(0, 1000) < collision_upper_confidence_bound(0, 10)


@pytest.mark.parametrize(
    ("n_events", "n_episodes", "confidence"),
    [(-1, 10, 0.95), (5, 4, 0.95), (0, 0, 0.95), (1, 10, 0.0), (1, 10, 1.0)],
)
def test_collision_ucb_rejects_invalid_inputs(
    n_events: int, n_episodes: int, confidence: float
) -> None:
    """Out-of-range counts or confidence must fail closed."""
    with pytest.raises(ValueError):
        collision_upper_confidence_bound(n_events, n_episodes, confidence=confidence)


def test_admissibility_gates_collision_first() -> None:
    """A collision makes an episode inadmissible regardless of comfort/efficiency."""
    assert not is_episode_admissible({"collisions": 1, "comfort": 1.0})
    assert is_episode_admissible({"collisions": 0, "comfort": 0.0})


def test_admissibility_respects_near_miss_timeout_deadlock() -> None:
    """Near-miss severity, timeout, and deadlock gates must each block admissibility."""
    gates = AdmissibilityGates(max_near_miss_severity=0.5)

    assert not is_episode_admissible({"collisions": 0, "near_miss_severity": 0.9}, gates)
    assert is_episode_admissible({"collisions": 0, "near_miss_severity": 0.3}, gates)
    assert not is_episode_admissible({"collisions": 0, "timeout": True})
    assert not is_episode_admissible({"collisions": 0, "deadlock": True})


def test_survivorship_delta_exposes_conditioning_bias() -> None:
    """Comfort over only-successful episodes must differ from the unconditional mean."""
    episodes = [
        {"comfort": 1.0, "safe_success": True},
        {"comfort": 1.0, "safe_success": True},
        {"comfort": 0.0, "safe_success": False},
    ]
    report = survivorship_aware_metric(episodes, "comfort")

    assert report["unconditional_mean"] == pytest.approx(2 / 3)
    assert report["conditioned_on_safe_success_mean"] == pytest.approx(1.0)
    assert report["survivorship_delta"] == pytest.approx(1 / 3)
    assert report["n_all"] == 3
    assert report["n_safe_success"] == 2


def test_planner_summary_is_versioned_and_complete() -> None:
    """The planner summary must expose admissibility, collision UCB, and survivorship."""
    episodes = [
        {"collisions": 0, "comfort": 0.9, "efficiency": 0.8, "safe_success": True},
        {"collisions": 1, "comfort": 0.2, "efficiency": 0.9, "safe_success": False},
        {"collisions": 0, "comfort": 0.7, "efficiency": 0.6, "safe_success": True},
    ]
    summary = constraints_first_planner_summary(episodes)

    assert summary["schema_version"] == CONSTRAINTS_FIRST_SCHEMA
    assert summary["n_episodes"] == 3
    assert summary["admissible_rate"] == pytest.approx(2 / 3)
    assert summary["collision_rate"] == pytest.approx(1 / 3)
    assert 0.0 < summary["collision_upper_confidence_bound"] <= 1.0
    assert summary["comfort"]["survivorship_delta"] is not None


def test_planner_summary_rejects_empty() -> None:
    """A planner with no episodes cannot be summarized."""
    with pytest.raises(ValueError):
        constraints_first_planner_summary([])


def test_ranking_inversion_detects_order_change() -> None:
    """A planner that looks good only under the soft composite must surface as inverted."""
    compensatory = {"A": 0.9, "B": 0.8, "C": 0.5}
    # Under constraints-first, B (a frequent collider) drops below C.
    constraints_first = {"A": 0.9, "B": 0.3, "C": 0.6}
    result = ranking_inversion(compensatory, constraints_first)

    assert result["any_inversion"] is True
    assert set(result["inverted_planners"]) == {"B", "C"}
    assert result["per_planner"]["B"]["compensatory_rank"] == 2
    assert result["per_planner"]["B"]["constraints_first_rank"] == 3


def test_ranking_inversion_none_when_orders_match() -> None:
    """Identical orderings must report no inversion."""
    scores = {"A": 0.9, "B": 0.5}
    result = ranking_inversion(scores, {"A": 0.8, "B": 0.1})

    assert result["any_inversion"] is False
    assert result["inverted_planners"] == []


def test_ranking_inversion_requires_same_planner_set() -> None:
    """Mismatched planner sets must fail closed."""
    with pytest.raises(ValueError):
        ranking_inversion({"A": 1.0}, {"B": 1.0})
