"""Unit tests for Alyassi-inspired reward composition and registry wiring."""

from __future__ import annotations

from robot_sf.gym_env.reward import build_reward_function
from robot_sf.gym_env.reward_alyassi import (
    AlyassiRewardWeights,
    alyassi_component_citations,
    alyassi_component_scores,
    alyassi_reward,
)


def _base_meta() -> dict:
    """Create a minimal metadata payload compatible with Alyassi reward defaults.

    Returns:
        Baseline metadata dictionary.
    """
    return {
        "step_of_episode": 10,
        "max_sim_steps": 100,
        "is_pedestrian_collision": False,
        "is_robot_collision": False,
        "is_obstacle_collision": False,
        "is_route_complete": False,
        "is_robot_at_goal": False,
        "near_misses": 0.0,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0.0,
        "action": [0.4, 0.1],
        "last_action": [0.4, 0.1],
    }


def test_alyassi_reward_terminal_goal_bonus_is_positive() -> None:
    """Goal completion should produce a positive total reward signal."""
    meta = _base_meta()
    meta["is_robot_at_goal"] = True
    value = alyassi_reward(meta)
    assert value > 0.0


def test_alyassi_reward_collision_reduces_total_score() -> None:
    """Collision flags should reduce the composite reward compared with clean state."""
    clean = _base_meta()
    collided = _base_meta()
    collided["is_pedestrian_collision"] = True

    assert alyassi_reward(collided) < alyassi_reward(clean)


def test_alyassi_smoothness_penalizes_action_change() -> None:
    """Action changes should reduce the smoothness component."""
    smooth = _base_meta()
    jerky = _base_meta()
    jerky["action"] = [1.0, -1.0]
    jerky["last_action"] = [0.0, 0.0]

    scores_smooth = alyassi_component_scores(smooth)
    scores_jerky = alyassi_component_scores(jerky)
    assert scores_jerky["smoothness"] < scores_smooth["smoothness"]


def test_alyassi_optional_prediction_and_preference_terms_are_used() -> None:
    """Optional meta fields should activate corresponding component terms."""
    meta = _base_meta()
    meta["human_preference_score"] = 0.5
    meta["prediction_zone_intrusion"] = 0.2
    meta["prediction_cov_sqrt_det"] = 0.1

    scores = alyassi_component_scores(meta)
    assert scores["human_preference"] > 0.0
    assert scores["human_prediction"] < 0.0


def test_alyassi_registry_name_builds_callable() -> None:
    """Reward registry should resolve the Alyassi reward name."""
    reward_fn = build_reward_function("alyassi", reward_kwargs={"step_cost": 0.02})
    value = reward_fn(_base_meta())
    assert isinstance(value, float)


def test_alyassi_weights_can_focus_single_component() -> None:
    """Custom weights should isolate the target component behavior."""
    meta = _base_meta()
    meta["near_misses"] = 2.0
    weights = AlyassiRewardWeights(
        w_goal=0.0,
        w_collision=0.0,
        w_efficiency=0.0,
        w_smoothness=0.0,
        w_social=1.0,
        w_geometric_collision=0.0,
        w_human_preference=0.0,
        w_human_prediction=0.0,
        w_exploration=0.0,
        w_task_specific=0.0,
        w_demo_learning=0.0,
    )
    value = alyassi_reward(meta, weights=weights)
    assert value < 0.0


def test_alyassi_component_citations_cover_core_terms() -> None:
    """Citation mapping should include every implemented conceptual component."""
    citations = alyassi_component_citations()
    for term in [
        "goal",
        "collision",
        "efficiency",
        "smoothness",
        "social",
        "geometric_collision",
        "human_preference",
        "human_prediction",
        "exploration",
        "task_specific",
        "demo_learning",
        "weight_learning",
    ]:
        assert term in citations
        assert len(citations[term]) >= 1


def test_alyassi_efficiency_speed_term_is_clamped() -> None:
    """Extreme speed metadata should not create unbounded efficiency penalties."""
    meta = _base_meta()
    meta["speed"] = 100.0
    scores = alyassi_component_scores(meta, step_cost=0.01, speed_target=0.7)
    assert scores["efficiency"] >= -1.01


def test_alyassi_efficiency_ignores_non_numeric_speed_metadata() -> None:
    """Non-numeric speed metadata should safely fall back to step-cost-only efficiency."""
    meta = _base_meta()
    meta["speed"] = "unknown"
    scores = alyassi_component_scores(meta, step_cost=0.05, speed_target=0.7)
    assert scores["efficiency"] == -0.05
