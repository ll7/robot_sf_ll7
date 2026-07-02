"""Tests for Bayesian pedestrian goal-intention inference."""

from __future__ import annotations

import math

import pytest

from robot_sf.prediction.goal_intention import (
    CandidateGoal,
    GoalPosteriorConfig,
    candidate_goals_from_points,
    planner_goal_posterior_channel,
    update_goal_posterior,
)


def _two_goals() -> tuple[CandidateGoal, CandidateGoal]:
    return (
        CandidateGoal(id="east_exit", position=(10.0, 0.0), source="map_annotations"),
        CandidateGoal(id="north_exit", position=(0.0, 10.0), source="map_annotations"),
    )


def test_posterior_favors_goal_aligned_with_heading() -> None:
    """Aligned motion should concentrate mass on the goal along that heading."""

    posterior = update_goal_posterior(
        pedestrian_id="ped_1",
        candidate_goals=_two_goals(),
        observed_position=(0.0, 0.0),
        observed_velocity=(1.0, 0.0),
        config=GoalPosteriorConfig(heading_kappa=4.0),
    )

    assert posterior.top_goal_id == "east_exit"
    assert posterior.probabilities["east_exit"] > 0.95
    assert math.isclose(sum(posterior.probabilities.values()), 1.0)
    assert posterior.candidate_source == "map_annotations"
    assert posterior.blocker is None


def test_posterior_switches_after_synthetic_goal_switch() -> None:
    """Repeated motion toward another goal should switch the posterior."""

    goals = _two_goals()
    posterior = update_goal_posterior(
        pedestrian_id="ped_1",
        candidate_goals=goals,
        observed_position=(0.0, 0.0),
        observed_velocity=(1.0, 0.0),
    )
    assert posterior.top_goal_id == "east_exit"

    probabilities = posterior.probabilities
    for _step in range(2):
        posterior = update_goal_posterior(
            pedestrian_id="ped_1",
            candidate_goals=goals,
            observed_position=(1.0, 1.0),
            observed_velocity=(0.0, 1.0),
            prior=probabilities,
        )
        probabilities = posterior.probabilities

    assert posterior.top_goal_id == "north_exit"
    assert posterior.probabilities["north_exit"] > 0.9


def test_stationary_pedestrian_returns_prior_without_nan() -> None:
    """Stationary observations should keep the prior and record a blocker."""

    prior = {"east_exit": 0.8, "north_exit": 0.2}

    posterior = update_goal_posterior(
        pedestrian_id="ped_1",
        candidate_goals=_two_goals(),
        observed_position=(0.0, 0.0),
        observed_velocity=(0.0, 0.0),
        prior=prior,
    )

    assert posterior.probabilities == pytest.approx(prior)
    assert posterior.blocker == "stationary_below_velocity_min_mps"
    assert all(math.isfinite(value) for value in posterior.probabilities.values())


def test_zero_speed_blocks_even_when_velocity_minimum_is_zero() -> None:
    """Exactly stationary observations never divide by zero."""

    posterior = update_goal_posterior(
        pedestrian_id="ped_1",
        candidate_goals=_two_goals(),
        observed_position=(0.0, 0.0),
        observed_velocity=(0.0, 0.0),
        config=GoalPosteriorConfig(velocity_min_mps=0.0),
    )

    assert posterior.blocker == "stationary_below_velocity_min_mps"
    assert math.isclose(sum(posterior.probabilities.values()), 1.0)


def test_extreme_heading_kappa_overflow_fails_closed() -> None:
    """Huge likelihood exponents become explicit validation errors."""

    with pytest.raises(ValueError, match="goal likelihood must be finite"):
        update_goal_posterior(
            pedestrian_id="ped_1",
            candidate_goals=_two_goals(),
            observed_position=(0.0, 0.0),
            observed_velocity=(1.0, 0.0),
            config=GoalPosteriorConfig(heading_kappa=1000.0),
        )


def test_duplicate_goal_ids_fail_closed() -> None:
    """Duplicate candidate IDs should fail before producing a posterior."""

    goals = (
        CandidateGoal(id="exit", position=(10.0, 0.0), source="scenario_route_endpoints"),
        CandidateGoal(id="exit", position=(0.0, 10.0), source="scenario_route_endpoints"),
    )

    with pytest.raises(ValueError, match="duplicate candidate goal id"):
        update_goal_posterior(
            pedestrian_id="ped_1",
            candidate_goals=goals,
            observed_position=(0.0, 0.0),
            observed_velocity=(1.0, 0.0),
        )


def test_planner_channel_serializes_enabled_summary_and_disabled_absence() -> None:
    """Planner channel should expose provenance only when explicitly enabled."""

    goals = candidate_goals_from_points(
        {"east_exit": (10.0, 0.0), "north_exit": (0.0, 10.0)},
        source="scenario_route_endpoints",
    )
    posterior = update_goal_posterior(
        pedestrian_id="ped_1",
        candidate_goals=goals,
        observed_position=(0.0, 0.0),
        observed_velocity=(1.0, 0.0),
    )

    enabled_channel = planner_goal_posterior_channel([posterior], enabled=True)
    disabled_channel = planner_goal_posterior_channel([posterior], enabled=False)

    summary = enabled_channel["pedestrian_goal_posteriors"]["ped_1"]
    assert enabled_channel["enabled"] is True
    assert summary["top_goal_id"] == "east_exit"
    assert summary["candidate_source"] == "scenario_route_endpoints"
    assert isinstance(summary["config_hash"], str)
    assert disabled_channel == {"enabled": False, "pedestrian_goal_posteriors": {}}
