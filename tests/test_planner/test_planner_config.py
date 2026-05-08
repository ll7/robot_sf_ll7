"""Unit tests for PlannerConfig and PlanningFailedError."""

import pytest

from robot_sf.planner import PlannerConfig, PlanningFailedError


def test_planner_config_defaults():
    """Verify PlannerConfig defaults match the public planner contract."""
    config = PlannerConfig()

    assert config.robot_radius == 0.4
    assert config.min_safe_clearance == 0.3
    assert config.enable_smoothing is True
    assert config.smoothing_epsilon == 0.1
    assert config.cache_graphs is True
    assert config.fallback_on_failure is False


def test_planner_config_rejects_non_positive_radius():
    """Verify non-positive robot radius values are rejected."""
    with pytest.raises(ValueError):
        PlannerConfig(robot_radius=0)


def test_planner_config_rejects_negative_clearance():
    """Verify negative clearance values are rejected."""
    with pytest.raises(ValueError):
        PlannerConfig(min_safe_clearance=-0.01)


def test_planner_config_requires_positive_smoothing_when_enabled():
    """Verify smoothing requires a positive epsilon when enabled."""
    with pytest.raises(ValueError):
        PlannerConfig(enable_smoothing=True, smoothing_epsilon=0)


def test_planner_config_allows_zero_smoothing_when_disabled():
    """Verify zero smoothing epsilon is allowed when smoothing is disabled."""
    config = PlannerConfig(enable_smoothing=False, smoothing_epsilon=0)

    assert config.smoothing_epsilon == 0
    assert config.enable_smoothing is False


def test_planning_failed_error_captures_context():
    """Verify PlanningFailedError retains start, goal, and reason context."""
    start = (0.0, 0.0)
    goal = (1.0, 1.0)
    reason = "unreachable goal"

    err = PlanningFailedError(start=start, goal=goal, reason=reason)

    assert err.start == start
    assert err.goal == goal
    assert err.reason == reason
    assert reason in str(err)
