"""Behavioral contract tests for the one-frame actor goal posterior."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
)
from robot_sf.prediction.goal_belief_contract import GoalBeliefMode
from robot_sf.prediction.goal_intention import (
    GoalCandidate,
    GoalCandidateAvailability,
    GoalCandidateRole,
    GoalCandidateSet,
    HeadingGoalPosteriorConfig,
    planner_goal_posterior_channel_from_beliefs,
    planner_oracle_goal_posterior_channel_from_state,
    update_heading_goal_posterior,
)


def _east_north_candidates() -> GoalCandidateSet:
    return GoalCandidateSet(
        candidates=(
            GoalCandidate(
                id="east",
                position=(10.0, 0.0),
                source="public_fixture",
            ),
            GoalCandidate(
                id="north",
                position=(0.0, 10.0),
                source="public_fixture",
            ),
        ),
        source="public_fixture",
    )


def _probabilities(belief) -> dict[str, float]:
    return {
        candidate.candidate_id: candidate.probability
        for candidate in belief.candidate_probabilities
    }


def _rotated(
    point: tuple[float, float], angle: float, offset: tuple[float, float]
) -> tuple[float, float]:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (
        offset[0] + cos_angle * point[0] - sin_angle * point[1],
        offset[1] + sin_angle * point[0] + cos_angle * point[1],
    )


def test_aligned_actor_candidate_wins_with_explicit_unknown_mass() -> None:
    """Heading evidence favors the aligned public candidate without forced confidence."""

    belief = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=_east_north_candidates(),
    )

    probabilities = _probabilities(belief)
    assert belief.source.value == "observation_only"
    assert belief.mode is GoalBeliefMode.CENSORED
    assert probabilities["east"] > probabilities["north"]
    assert 0.0 < belief.unknown_candidate_probability < 0.1
    assert math.isfinite(belief.entropy)
    assert belief.track_confidence is None
    assert belief.force_estimate is None
    assert "arrival_probability_unestimated" in belief.blockers


def test_rotation_and_translation_preserve_id_probabilities() -> None:
    """The circular likelihood depends on relative geometry, not world origin."""

    base = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(1.0, 2.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=GoalCandidateSet.from_points(
            {"east": (11.0, 2.0), "north": (1.0, 12.0)},
            source="public_fixture",
        ),
    )
    angle = math.pi / 3.0
    offset = (20.0, -4.0)
    transformed = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=_rotated((1.0, 2.0), angle, offset),
        observed_velocity_global=_rotated((1.0, 0.0), angle, (0.0, 0.0)),
        candidate_set=GoalCandidateSet.from_points(
            {
                "east": _rotated((11.0, 2.0), angle, offset),
                "north": _rotated((1.0, 12.0), angle, offset),
            },
            source="public_fixture",
        ),
    )

    assert _probabilities(transformed) == pytest.approx(_probabilities(base))
    assert transformed.unknown_candidate_probability == pytest.approx(
        base.unknown_candidate_probability
    )


def test_stationary_actor_preserves_prior_and_records_blocker() -> None:
    """Slow observations do not manufacture a heading likelihood."""

    belief = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(0.0, 0.0),
        candidate_set=_east_north_candidates(),
        prior={"east": 0.8, "north": 0.2},
    )

    probabilities = _probabilities(belief)
    assert probabilities == pytest.approx({"east": 0.72, "north": 0.18})
    assert belief.unknown_candidate_probability == pytest.approx(0.1)
    assert "stationary_below_velocity_min_mps" in belief.blockers
    assert all(math.isfinite(value) for value in probabilities.values())


def test_same_ray_candidates_remain_ambiguous() -> None:
    """A one-frame heading observation does not infer distance along a ray."""

    belief = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=GoalCandidateSet.from_points(
            {"near": (2.0, 0.0), "far": (20.0, 0.0)},
            source="public_fixture",
        ),
    )

    probabilities = _probabilities(belief)
    assert probabilities["near"] == pytest.approx(probabilities["far"])


def test_opposite_and_perpendicular_candidates_follow_circular_ordering() -> None:
    """Aligned, perpendicular, and opposite directions retain interpretable ordering."""

    belief = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=GoalCandidateSet.from_points(
            {
                "aligned": (10.0, 0.0),
                "perpendicular": (0.0, 10.0),
                "opposite": (-10.0, 0.0),
            },
            source="public_fixture",
        ),
    )

    probabilities = _probabilities(belief)
    assert probabilities["aligned"] > probabilities["perpendicular"] > probabilities["opposite"]


def test_missing_candidates_are_unavailable_not_a_true_goal_fallback() -> None:
    """No public provider yields an explicit unknown state."""

    belief = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=GoalCandidateSet(source="public_fixture"),
    )

    assert belief.mode is GoalBeliefMode.UNAVAILABLE
    assert belief.candidate_probabilities == ()
    assert belief.unknown_candidate_probability == 1.0
    assert "no_point_candidates" in belief.blockers


def test_unknown_and_unavailable_candidate_hypotheses_are_explicit() -> None:
    """Non-point and unavailable candidates cannot be silently treated as goals."""

    candidate_set = GoalCandidateSet(
        candidates=(
            GoalCandidate(
                id="open-ray",
                position=None,
                source="public_fixture",
                role=GoalCandidateRole.OPEN_RAY,
            ),
            GoalCandidate(
                id="closed",
                position=(10.0, 0.0),
                source="public_fixture",
                availability=GoalCandidateAvailability.UNAVAILABLE,
            ),
            GoalCandidate(
                id="north",
                position=(0.0, 10.0),
                source="public_fixture",
            ),
        ),
        source="public_fixture",
    )

    belief = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=candidate_set,
    )

    assert set(_probabilities(belief)) == {"north"}
    assert "non_point_candidate_unknown" in belief.blockers
    assert "candidate_unavailable_unknown" in belief.blockers


def test_large_kappa_is_finite_and_candidate_order_is_not_semantic() -> None:
    """Stable log-sum-exp and stable IDs make extreme replay deterministic."""

    config = HeadingGoalPosteriorConfig(heading_kappa=1e300)
    first = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=_east_north_candidates(),
        config=config,
    )
    second = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=GoalCandidateSet(
            candidates=tuple(reversed(_east_north_candidates().candidates)),
            source="public_fixture",
        ),
        config=config,
    )

    assert _probabilities(first) == pytest.approx(_probabilities(second))
    assert first.content_digest == second.content_digest
    assert len(config.config_hash) == 64
    assert all(math.isfinite(value) for value in _probabilities(first).values())


def test_actor_output_does_not_depend_on_unseen_simulator_goal_identity() -> None:
    """The actor API has no true-goal input that could affect its output."""

    candidate_set = _east_north_candidates()
    first = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=candidate_set,
    )
    second = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=candidate_set,
    )

    assert first.to_json() == second.to_json()


def test_malformed_candidates_fail_closed() -> None:
    """Candidate geometry and provenance are validated before inference."""

    with pytest.raises(ValueError):
        GoalCandidate(
            id="bad",
            position=(math.nan, 0.0),
            source="public_fixture",
        )


def test_candidate_and_config_validation_is_strict() -> None:
    """Duplicate IDs, negative priors, frames, and invalid configuration fail closed."""

    with pytest.raises(ValueError, match="unique"):
        GoalCandidateSet(
            candidates=(
                GoalCandidate(id="same", position=(1.0, 0.0), source="public_fixture"),
                GoalCandidate(id="same", position=(2.0, 0.0), source="public_fixture"),
            ),
            source="public_fixture",
        )
    with pytest.raises(ValueError, match="non-negative"):
        GoalCandidate(
            id="bad-prior",
            position=(1.0, 0.0),
            source="public_fixture",
            prior_weight=-1.0,
        )
    with pytest.raises(TypeError, match="coordinate_frame"):
        GoalCandidate(
            id="bad-frame",
            position=(1.0, 0.0),
            source="public_fixture",
            coordinate_frame="global_xy",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown_prior_probability"):
        HeadingGoalPosteriorConfig(unknown_prior_probability=1.1)
    with pytest.raises(ValueError, match="unknown_likelihood"):
        HeadingGoalPosteriorConfig(unknown_likelihood=0.0)
    with pytest.raises(ValueError, match="stationary_prior_policy"):
        HeadingGoalPosteriorConfig(stationary_prior_policy="hidden_state")


def test_actor_rejects_non_candidate_provider_objects() -> None:
    """Simulator/PySocialForce-shaped objects are not accepted by the actor API."""

    with pytest.raises(TypeError, match="GoalCandidateSet"):
        update_heading_goal_posterior(
            track_id="track-1",
            observed_position_global=(0.0, 0.0),
            observed_velocity_global=(1.0, 0.0),
            candidate_set=np.zeros((1, 6)),  # type: ignore[arg-type]
        )


def test_typed_actor_channel_rejects_oracle_beliefs() -> None:
    """The typed planner adapter does not admit upper-bound sources in actor mode."""

    belief = update_heading_goal_posterior(
        track_id="track-1",
        observed_position_global=(0.0, 0.0),
        observed_velocity_global=(1.0, 0.0),
        candidate_set=_east_north_candidates(),
    )

    assert (
        planner_goal_posterior_channel_from_beliefs((belief,), enabled=True, actor_only=True)[
            "source"
        ]
        == "observation_only"
    )

    from dataclasses import replace

    from robot_sf.prediction.goal_belief_contract import GoalBeliefSource

    oracle_belief = replace(belief, source=GoalBeliefSource.SIMULATOR_UPPER_BOUND)
    with pytest.raises(ValueError, match="rejects oracle"):
        planner_goal_posterior_channel_from_beliefs((oracle_belief,), enabled=True, actor_only=True)


def test_legacy_state_helper_is_explicit_oracle_metadata() -> None:
    """Compatibility state plumbing is visibly upper-bound and not actor output."""

    channel = planner_oracle_goal_posterior_channel_from_state(
        enabled=True,
        positions=[(0.0, 0.0)],
        velocities=[(1.0, 0.0)],
        goals=[(5.0, 0.0)],
    )

    assert channel["source"] == "simulator_upper_bound"
    assert channel["oracle_only"] is True
    summary = channel["pedestrian_goal_posteriors"]["ped_0"]
    assert summary["source"] == "simulator_upper_bound"
    assert summary["candidate_source"] == "oracle_true_goal_identity"


def test_planner_actor_only_mode_rejects_oracle_channel() -> None:
    """A configured actor-only planner refuses the compatibility oracle channel."""

    planner = HybridRuleLocalPlannerAdapter(
        HybridRuleLocalPlannerConfig(
            goal_posterior_avoidance_enabled=True,
            goal_posterior_actor_only=True,
        )
    )
    observation = {
        "robot": {
            "position": np.array([0.0, 0.0]),
            "heading": np.array([0.0]),
            "speed": np.array([0.5]),
            "radius": np.array([0.3]),
        },
        "goal": {"current": np.array([4.0, 0.0]), "next": np.array([4.0, 0.0])},
        "pedestrians": {
            "positions": np.array([[1.0, 0.2]]),
            "velocities": np.array([[0.0, 0.8]]),
            "count": np.array([1.0]),
            "radius": 0.25,
        },
        "sim": {"timestep": 0.2},
        "info": {
            "planner_goal_posterior_channel": planner_oracle_goal_posterior_channel_from_state(
                enabled=True,
                positions=[(1.0, 0.2)],
                velocities=[(0.0, 0.8)],
                goals=[(5.0, 0.0)],
            )
        },
    }

    planner.plan(observation)
    assert planner.diagnostics()["last_decision"]["goal_posterior_avoidance"]["blocker"] == (
        "oracle_source_rejected"
    )
