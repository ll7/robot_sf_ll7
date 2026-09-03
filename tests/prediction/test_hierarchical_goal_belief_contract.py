"""Contract tests for the additive hierarchical goal posterior (issue #8075)."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest

from robot_sf.prediction import (
    ACTOR_FORBIDDEN_KEYS,
    HIERARCHICAL_GOAL_POSTERIOR_SCHEMA_VERSION,
    GoalCandidateKind,
    HierarchicalGoalPosteriorV1,
    HierarchicalProbability,
    HierarchicalWaypointConditionalV1,
)

HASH = "a" * 64


def _posterior(
    *,
    destinations: tuple[HierarchicalProbability, ...] | None = None,
    conditionals: tuple[HierarchicalWaypointConditionalV1, ...] | None = None,
    parents: dict[str, str] | None = None,
    evidence_source: str = "upstream_selected",
) -> HierarchicalGoalPosteriorV1:
    """Build a small two-destination hierarchy with explicit unknown states."""
    return HierarchicalGoalPosteriorV1(
        track_id="track-1",
        tracking_epoch_id="epoch-1",
        timestamp_s=1.5,
        step_index=3,
        destination_probabilities=destinations
        or (
            HierarchicalProbability("destination-a", 0.6),
            HierarchicalProbability("destination-b", 0.3),
        ),
        unknown_destination_probability=0.1,
        waypoint_conditionals=conditionals
        or (
            HierarchicalWaypointConditionalV1(
                "destination-a",
                (
                    HierarchicalProbability("waypoint-a-near", 0.5),
                    HierarchicalProbability("waypoint-a-far", 0.25),
                ),
                0.25,
            ),
            HierarchicalWaypointConditionalV1(
                "destination-b",
                (HierarchicalProbability("waypoint-b", 0.8),),
                0.2,
            ),
        ),
        waypoint_parent_destination=parents
        or {
            "waypoint-a-near": "destination-a",
            "waypoint-a-far": "destination-a",
            "waypoint-b": "destination-b",
        },
        evidence_source=evidence_source,
        innovation=0.2,
        blockers=("synthetic_fixture",),
        config_hash=HASH,
        candidate_set_digest="b" * 64,
    )


def test_hierarchy_has_independent_normalization_and_expected_marginal() -> None:
    """Destination and conditional masses remain distinct and marginalize correctly."""
    posterior = _posterior()

    assert math.isclose(
        sum(value.probability for value in posterior.destination_probabilities)
        + posterior.unknown_destination_probability,
        1.0,
    )
    for conditional in posterior.waypoint_conditionals:
        assert math.isclose(
            sum(value.probability for value in conditional.waypoint_probabilities)
            + conditional.unknown_waypoint_probability,
            1.0,
        )

    marginal, unknown = posterior.active_waypoint_marginal()

    assert {value.candidate_id: value.probability for value in marginal} == pytest.approx(
        {
            "waypoint-a-near": 0.30,
            "waypoint-a-far": 0.15,
            "waypoint-b": 0.24,
        }
    )
    assert unknown == pytest.approx(0.31)
    assert sum(value.probability for value in marginal) + unknown == pytest.approx(1.0)


def test_serialization_is_strict_deterministic_and_round_trips() -> None:
    """The actor payload has a stable digest and rejects tampered state."""
    posterior = _posterior()
    payload = posterior.to_dict()

    assert payload["schema_version"] == HIERARCHICAL_GOAL_POSTERIOR_SCHEMA_VERSION
    assert "unknown_destination_probability" in payload
    assert all(
        "unknown_waypoint_probability" in conditional
        for conditional in payload["waypoint_conditionals"]
    )
    parsed = HierarchicalGoalPosteriorV1.from_dict(copy.deepcopy(payload))
    assert parsed.to_json() == posterior.to_json()
    assert parsed.content_digest == posterior.content_digest

    payload["innovation"] = 0.3
    with pytest.raises(ValueError, match="state_digest"):
        HierarchicalGoalPosteriorV1.from_dict(payload)


def test_candidate_and_parent_input_order_does_not_change_bytes() -> None:
    """Candidate order is presentation detail, not posterior identity."""
    reordered = _posterior(
        destinations=(
            HierarchicalProbability("destination-b", 0.3),
            HierarchicalProbability("destination-a", 0.6),
        ),
        conditionals=(
            HierarchicalWaypointConditionalV1(
                "destination-b",
                (HierarchicalProbability("waypoint-b", 0.8),),
                0.2,
            ),
            HierarchicalWaypointConditionalV1(
                "destination-a",
                (
                    HierarchicalProbability("waypoint-a-far", 0.25),
                    HierarchicalProbability("waypoint-a-near", 0.5),
                ),
                0.25,
            ),
        ),
        parents={
            "waypoint-b": "destination-b",
            "waypoint-a-far": "destination-a",
            "waypoint-a-near": "destination-a",
        },
    )

    assert reordered.to_json() == _posterior().to_json()


@pytest.mark.parametrize(
    "level, expected_kind, expected_ids",
    [
        (
            "active_waypoint",
            GoalCandidateKind.ACTIVE_WAYPOINT,
            {"waypoint-a-near", "waypoint-a-far", "waypoint-b"},
        ),
        (
            "final_destination",
            GoalCandidateKind.FINAL_DESTINATION,
            {"destination-a", "destination-b"},
        ),
    ],
)
def test_flat_projection_requires_and_preserves_the_selected_level(
    level: str,
    expected_kind: GoalCandidateKind,
    expected_ids: set[str],
) -> None:
    """Compatibility output never silently mixes destination and waypoint IDs."""
    belief = _posterior().to_goal_belief_v1(level)

    assert {candidate.candidate_id for candidate in belief.candidate_probabilities} == expected_ids
    assert {candidate.kind for candidate in belief.candidate_probabilities} == {expected_kind}
    assert "hierarchical_projection" in belief.blockers
    assert ACTOR_FORBIDDEN_KEYS.isdisjoint(belief.to_dict())


def test_flat_projection_rejects_implicit_or_unknown_level() -> None:
    """Callers must name one supported hierarchy level explicitly."""
    with pytest.raises(ValueError, match="level must be one of"):
        _posterior().to_goal_belief_v1("all")


@pytest.mark.parametrize(
    "bad_evidence_source",
    ["oracle_upper_bound", "simulator_truth", "waypoint_truth"],
)
def test_actor_hierarchy_rejects_privileged_evidence_labels(bad_evidence_source: str) -> None:
    """Oracle, simulator, and truth labels cannot enter actor-side hierarchy state."""
    with pytest.raises(ValueError, match="oracle or simulator"):
        _posterior(evidence_source=bad_evidence_source)


def test_parent_links_and_unknown_external_keys_fail_closed() -> None:
    """Every known waypoint needs one known parent and no silent schema extension."""
    with pytest.raises(ValueError, match="missing or incorrect parent"):
        _posterior(parents={"waypoint-a-near": "destination-b"})

    payload = _posterior().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected key"):
        HierarchicalGoalPosteriorV1.from_dict(payload)


def test_malformed_external_shapes_fail_closed() -> None:
    """Low-level and top-level payload parsers reject malformed shapes and fields."""
    with pytest.raises(TypeError, match="must be an object"):
        HierarchicalProbability.from_dict(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing a required field"):
        HierarchicalProbability.from_dict({"candidate_id": "waypoint"})
    with pytest.raises(TypeError, match="must be an object"):
        HierarchicalWaypointConditionalV1.from_dict(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing a required field"):
        HierarchicalWaypointConditionalV1.from_dict({"destination_id": "destination-a"})
    with pytest.raises(TypeError, match="must be an object"):
        HierarchicalGoalPosteriorV1.from_dict(None)  # type: ignore[arg-type]

    payload = _posterior().to_dict()
    del payload["state_digest"]
    with pytest.raises(ValueError, match="missing a required field"):
        HierarchicalGoalPosteriorV1.from_dict(payload)


def test_hierarchy_constructor_rejects_invalid_vectors_and_lifecycle_shapes() -> None:
    """Constructor validation covers duplicate, non-normalized, and malformed hierarchy state."""
    posterior = _posterior()
    with pytest.raises(TypeError, match="must be an array"):
        replace(posterior, destination_probabilities=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must contain HierarchicalProbability"):
        replace(
            posterior,
            destination_probabilities=("not-a-probability",),
            unknown_destination_probability=1.0,
            waypoint_conditionals=(),
            waypoint_parent_destination={},
        )
    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        replace(
            posterior,
            destination_probabilities=(
                HierarchicalProbability("same", 0.3),
                HierarchicalProbability("same", 0.3),
            ),
            unknown_destination_probability=0.4,
        )
    with pytest.raises(ValueError, match="plus unknown mass must sum to 1"):
        replace(
            posterior,
            destination_probabilities=(HierarchicalProbability("destination-a", 0.2),),
            unknown_destination_probability=0.2,
            waypoint_conditionals=(HierarchicalWaypointConditionalV1("destination-a"),),
            waypoint_parent_destination={},
        )
    with pytest.raises(TypeError, match="waypoint_parent_destination must be an array"):
        replace(posterior, waypoint_parent_destination="not-a-parent-map")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entries must be"):
        replace(
            posterior,
            waypoint_parent_destination=[("waypoint-a-near",)],
        )
    with pytest.raises(ValueError, match="duplicate waypoint parent mapping"):
        replace(
            posterior,
            waypoint_parent_destination=[
                ("waypoint-a-near", "destination-a"),
                ("waypoint-a-near", "destination-a"),
                ("waypoint-a-far", "destination-a"),
                ("waypoint-b", "destination-b"),
            ],
        )
    with pytest.raises(TypeError, match="waypoint_conditionals must contain"):
        replace(posterior, waypoint_conditionals=("not-a-conditional",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate destination IDs"):
        replace(
            posterior,
            waypoint_conditionals=(
                HierarchicalWaypointConditionalV1("destination-a"),
                HierarchicalWaypointConditionalV1("destination-a"),
            ),
        )
    with pytest.raises(ValueError, match="must name exactly"):
        replace(
            posterior,
            waypoint_conditionals=(HierarchicalWaypointConditionalV1("destination-a"),),
            waypoint_parent_destination={},
        )


def test_hierarchy_constructor_rejects_invalid_metadata() -> None:
    """Metadata fields remain finite, versioned, and canonical."""
    posterior = _posterior()
    with pytest.raises(ValueError, match="schema_version"):
        replace(posterior, schema_version="hierarchical_goal_posterior.v2")
    with pytest.raises(ValueError, match="timestamp_s must be non-negative"):
        replace(posterior, timestamp_s=-0.1)
    with pytest.raises(ValueError, match="blockers must be unique"):
        replace(posterior, blockers=("same", "same"))


def test_empty_hierarchy_can_represent_unknown_at_both_levels() -> None:
    """Unknown destination and unknown waypoint are separate valid states."""
    posterior = HierarchicalGoalPosteriorV1(
        track_id="track-empty",
        tracking_epoch_id="epoch-1",
        timestamp_s=0.0,
        step_index=0,
        destination_probabilities=(),
        unknown_destination_probability=1.0,
        waypoint_conditionals=(),
        waypoint_parent_destination={},
        config_hash=HASH,
        candidate_set_digest="b" * 64,
    )

    marginal, unknown = posterior.active_waypoint_marginal()
    assert marginal == ()
    assert unknown == 1.0
    assert posterior.to_goal_belief_v1("active_waypoint").unknown_candidate_probability == 1.0
    assert posterior.to_goal_belief_v1("final_destination").unknown_candidate_probability == 1.0
