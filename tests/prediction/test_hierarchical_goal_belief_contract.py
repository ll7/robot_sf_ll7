"""Contract tests for the additive hierarchical goal posterior (issue #8075)."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest

from robot_sf.prediction import (
    ACTOR_FORBIDDEN_KEYS,
    HIERARCHICAL_GOAL_POSTERIOR_SCHEMA_VERSION,
    GoalBeliefSource,
    GoalCandidate,
    GoalCandidateAvailability,
    GoalCandidateKind,
    GoalCandidateRole,
    GoalCandidateSet,
    HierarchicalGoalPosteriorV1,
    HierarchicalProbability,
    HierarchicalWaypointConditionalV1,
)
from robot_sf.prediction._contract_utils import stable_digest

HASH = "a" * 64


POSTERIOR_CANDIDATE_SET = GoalCandidateSet(
    candidates=(
        GoalCandidate(
            id="destination-a",
            position=(10.0, 0.0),
            source="public_fixture",
            role=GoalCandidateRole.FINAL_DESTINATION,
        ),
        GoalCandidate(
            id="destination-b",
            position=(0.0, 10.0),
            source="public_fixture",
            role=GoalCandidateRole.FINAL_DESTINATION,
        ),
        GoalCandidate(
            id="waypoint-a-near",
            position=(2.0, 0.0),
            source="public_fixture",
            role=GoalCandidateRole.ACTIVE_WAYPOINT,
            parent_destination_id="destination-a",
        ),
        GoalCandidate(
            id="waypoint-a-far",
            position=(6.0, 0.0),
            source="public_fixture",
            role=GoalCandidateRole.ACTIVE_WAYPOINT,
            parent_destination_id="destination-a",
        ),
        GoalCandidate(
            id="waypoint-b",
            position=(0.0, 4.0),
            source="public_fixture",
            role=GoalCandidateRole.ACTIVE_WAYPOINT,
            parent_destination_id="destination-b",
        ),
    ),
    source="public_fixture",
)
POSTERIOR_CANDIDATE_SET_DIGEST = stable_digest(POSTERIOR_CANDIDATE_SET.to_dict())


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
        candidate_set_digest=POSTERIOR_CANDIDATE_SET_DIGEST,
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


def test_tolerated_roundoff_is_canonicalized_before_flat_projection() -> None:
    """Tolerated normalization drift cannot make a derived probability invalid."""
    candidate_set = GoalCandidateSet(
        candidates=(
            GoalCandidate(
                id="destination",
                position=(1.0, 0.0),
                source="public_fixture",
                role=GoalCandidateRole.FINAL_DESTINATION,
            ),
        ),
        source="public_fixture",
    )
    posterior = HierarchicalGoalPosteriorV1(
        track_id="track-roundoff",
        tracking_epoch_id="epoch-1",
        timestamp_s=0.0,
        step_index=0,
        destination_probabilities=(HierarchicalProbability("destination", 0.9999999995),),
        unknown_destination_probability=1e-9,
        waypoint_conditionals=(HierarchicalWaypointConditionalV1("destination"),),
        waypoint_parent_destination={},
        config_hash=HASH,
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    projected = posterior.to_goal_belief_v1("active_waypoint", candidate_set=candidate_set)

    assert projected.unknown_candidate_probability <= 1.0
    assert sum(
        candidate.probability for candidate in projected.candidate_probabilities
    ) + projected.unknown_candidate_probability == pytest.approx(1.0)

    destination_count = 17
    destinations = tuple(
        HierarchicalProbability(f"destination-{index:02d}", 0.9 / destination_count)
        for index in range(destination_count)
    )
    all_unknown_conditionals = tuple(
        HierarchicalWaypointConditionalV1(destination.candidate_id) for destination in destinations
    )
    cardinality_candidate_set = GoalCandidateSet(
        candidates=tuple(
            GoalCandidate(
                id=destination.candidate_id,
                position=(float(index), 0.0),
                source="public_fixture",
                role=GoalCandidateRole.FINAL_DESTINATION,
            )
            for index, destination in enumerate(destinations)
        ),
        source="public_fixture",
    )
    cardinality_posterior = HierarchicalGoalPosteriorV1(
        track_id="track-cardinality-roundoff",
        tracking_epoch_id="epoch-1",
        timestamp_s=0.0,
        step_index=0,
        destination_probabilities=destinations,
        unknown_destination_probability=0.1,
        waypoint_conditionals=all_unknown_conditionals,
        waypoint_parent_destination={},
        config_hash=HASH,
        candidate_set_digest=stable_digest(cardinality_candidate_set.to_dict()),
    )

    marginal, unknown = cardinality_posterior.active_waypoint_marginal()
    cardinality_projected = cardinality_posterior.to_goal_belief_v1(
        "active_waypoint",
        candidate_set=cardinality_candidate_set,
    )

    assert marginal == ()
    assert unknown == 1.0
    assert cardinality_projected.candidate_probabilities == ()
    assert cardinality_projected.unknown_candidate_probability == 1.0

    reordered = replace(
        cardinality_posterior,
        destination_probabilities=tuple(reversed(destinations)),
        waypoint_conditionals=tuple(reversed(all_unknown_conditionals)),
    )
    assert reordered.to_json() == cardinality_posterior.to_json()


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
    belief = _posterior().to_goal_belief_v1(level, candidate_set=POSTERIOR_CANDIDATE_SET)

    assert {candidate.candidate_id for candidate in belief.candidate_probabilities} == expected_ids
    assert {candidate.kind for candidate in belief.candidate_probabilities} == {expected_kind}
    assert "hierarchical_projection" in belief.blockers
    assert "arrival_probability_unestimated" in belief.blockers
    assert "change_probability_unestimated" in belief.blockers
    assert belief.change_probability == 0.0
    assert belief.source is GoalBeliefSource.OBSERVATION_ONLY
    assert ACTOR_FORBIDDEN_KEYS.isdisjoint(belief.to_dict())


def test_flat_projection_rejects_implicit_or_unknown_level() -> None:
    """Callers must name one supported hierarchy level explicitly."""
    with pytest.raises(ValueError, match="level must be one of"):
        _posterior().to_goal_belief_v1("all", candidate_set=POSTERIOR_CANDIDATE_SET)


def test_flat_projection_requires_a_concrete_candidate_set() -> None:
    """Observation-only projection must fail closed when no candidate set is supplied."""
    with pytest.raises(TypeError, match="candidate_set"):
        _posterior().to_goal_belief_v1("final_destination")


def test_flat_projection_rejects_noncanonical_candidate_set_types() -> None:
    """Projection admission accepts only the canonical GoalCandidateSet implementation."""
    with pytest.raises(TypeError, match="GoalCandidateSet"):
        _posterior().to_goal_belief_v1(
            "final_destination",
            candidate_set=object(),  # type: ignore[arg-type]
        )


def test_flat_projection_recomputes_and_checks_candidate_set_digest() -> None:
    """A shape-valid digest cannot admit projection without matching canonical candidate bytes."""
    posterior = replace(_posterior(), candidate_set_digest="c" * 64)

    with pytest.raises(ValueError, match="does not match"):
        posterior.to_goal_belief_v1(
            "final_destination",
            candidate_set=POSTERIOR_CANDIDATE_SET,
        )


def test_flat_projection_rejects_unreferenced_candidate_ids() -> None:
    """Every non-unknown posterior mass must name a candidate in the bound public set."""
    candidate_set = GoalCandidateSet(
        candidates=tuple(
            candidate
            for candidate in POSTERIOR_CANDIDATE_SET.candidates
            if candidate.id != "waypoint-b"
        ),
        source="public_fixture",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="waypoint-b"):
        posterior.to_goal_belief_v1("active_waypoint", candidate_set=candidate_set)


@pytest.mark.parametrize("binding", ["set", "candidate"])
def test_flat_projection_rejects_privileged_candidate_sources(binding: str) -> None:
    """Static source labels cannot smuggle privileged evidence into observation-only output."""
    if binding == "set":
        candidate_set = GoalCandidateSet(
            candidates=POSTERIOR_CANDIDATE_SET.candidates,
            source="simulator_truth",
        )
    else:
        candidate_set = GoalCandidateSet(
            candidates=tuple(
                replace(candidate, source="true_goal")
                if candidate.id == "destination-a"
                else candidate
                for candidate in POSTERIOR_CANDIDATE_SET.candidates
            ),
            source="public_fixture",
        )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="forbidden oracle or simulator source"):
        posterior.to_goal_belief_v1("final_destination", candidate_set=candidate_set)


def test_flat_projection_rejects_parent_metadata_disagreement() -> None:
    """A bound waypoint cannot disagree with the hierarchy's static parent mapping."""
    candidate_set = GoalCandidateSet(
        candidates=tuple(
            replace(candidate, parent_destination_id="destination-b")
            if candidate.id == "waypoint-a-near"
            else candidate
            for candidate in POSTERIOR_CANDIDATE_SET.candidates
        ),
        source="public_fixture",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="parent_destination_id disagrees"):
        posterior.to_goal_belief_v1("active_waypoint", candidate_set=candidate_set)


def test_flat_projection_rejects_missing_parent_metadata() -> None:
    """A waypoint must carry the same explicit parent as the hierarchy mapping."""
    candidate_set = GoalCandidateSet(
        candidates=tuple(
            replace(candidate, parent_destination_id=None)
            if candidate.id == "waypoint-a-near"
            else candidate
            for candidate in POSTERIOR_CANDIDATE_SET.candidates
        ),
        source="public_fixture",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="parent_destination_id is missing"):
        posterior.to_goal_belief_v1("active_waypoint", candidate_set=candidate_set)


@pytest.mark.parametrize(
    ("candidate_id", "role", "message"),
    [
        (
            "destination-a",
            GoalCandidateRole.ACTIVE_WAYPOINT,
            "destination probability",
        ),
        (
            "waypoint-a-near",
            GoalCandidateRole.FINAL_DESTINATION,
            "waypoint probability",
        ),
        (
            "destination-a",
            GoalCandidateRole.OPEN_RAY,
            "destination probability",
        ),
    ],
)
def test_flat_projection_rejects_incompatible_candidate_roles(
    candidate_id: str,
    role: GoalCandidateRole,
    message: str,
) -> None:
    """Known mass cannot bind to an open-ray or opposite-level candidate."""
    if role is GoalCandidateRole.OPEN_RAY:
        replacement = GoalCandidate(
            id=candidate_id,
            position=None,
            source="public_fixture",
            role=role,
            direction=(1.0, 0.0),
        )
    else:
        replacement = replace(
            next(
                candidate
                for candidate in POSTERIOR_CANDIDATE_SET.candidates
                if candidate.id == candidate_id
            ),
            role=role,
        )
    candidate_set = GoalCandidateSet(
        candidates=tuple(
            replacement if candidate.id == candidate_id else candidate
            for candidate in POSTERIOR_CANDIDATE_SET.candidates
        ),
        source="public_fixture",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match=message):
        posterior.to_goal_belief_v1("final_destination", candidate_set=candidate_set)


def test_flat_projection_rejects_unavailable_candidate_mass() -> None:
    """Unavailable candidates cannot be emitted as known posterior mass."""
    candidate_set = GoalCandidateSet(
        candidates=tuple(
            replace(candidate, availability=GoalCandidateAvailability.UNAVAILABLE)
            if candidate.id == "destination-a"
            else candidate
            for candidate in POSTERIOR_CANDIDATE_SET.candidates
        ),
        source="public_fixture",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="unavailable candidate"):
        posterior.to_goal_belief_v1("final_destination", candidate_set=candidate_set)


def test_flat_projection_rejects_unavailable_candidate_set() -> None:
    """An unavailable candidate provider cannot authorize a flat projection."""
    candidate_set = GoalCandidateSet(
        candidates=POSTERIOR_CANDIDATE_SET.candidates,
        source="public_fixture",
        availability=GoalCandidateAvailability.UNAVAILABLE,
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="candidate_set.availability"):
        posterior.to_goal_belief_v1("final_destination", candidate_set=candidate_set)


@pytest.mark.parametrize("binding", ["set", "candidate", "provenance"])
@pytest.mark.parametrize("privileged_provenance", ["oracle:goal", "route truth"])
def test_flat_projection_rejects_extended_privileged_candidate_provenance(
    binding: str, privileged_provenance: str
) -> None:
    """Route/truth labels and provenance references cannot enter actor-side projection."""
    if binding == "set":
        candidate_set = GoalCandidateSet(
            candidates=POSTERIOR_CANDIDATE_SET.candidates,
            source="assigned_route_v2",
        )
    elif binding == "candidate":
        candidate_set = GoalCandidateSet(
            candidates=tuple(
                replace(candidate, source="ground_truth")
                if candidate.id == "destination-a"
                else candidate
                for candidate in POSTERIOR_CANDIDATE_SET.candidates
            ),
            source="public_fixture",
        )
    else:
        candidate_set = GoalCandidateSet(
            candidates=tuple(
                replace(candidate, provenance_refs=(privileged_provenance,))
                if candidate.id == "destination-a"
                else candidate
                for candidate in POSTERIOR_CANDIDATE_SET.candidates
            ),
            source="public_fixture",
        )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="privileged"):
        posterior.to_goal_belief_v1("final_destination", candidate_set=candidate_set)


@pytest.mark.parametrize("source", ["sim_truth", "truth_label", "route_assignment"])
def test_flat_projection_rejects_privileged_source_aliases(source: str) -> None:
    """Near-miss simulator/truth aliases cannot bypass canonical source admission."""
    candidate_set = GoalCandidateSet(
        candidates=tuple(
            replace(candidate, source=source) if candidate.id == "destination-a" else candidate
            for candidate in POSTERIOR_CANDIDATE_SET.candidates
        ),
        source="public_fixture",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="forbidden oracle or simulator source"):
        posterior.to_goal_belief_v1("final_destination", candidate_set=candidate_set)


@pytest.mark.parametrize("status", ["infeasible", "unknown", "unavailable"])
def test_flat_projection_rejects_non_feasible_referenced_candidates(status: str) -> None:
    """Available metadata cannot reclassify an infeasible candidate as known posterior mass."""
    candidate_set = GoalCandidateSet(
        candidates=tuple(
            replace(candidate, feasibility_status=status)
            if candidate.id == "destination-a"
            else candidate
            for candidate in POSTERIOR_CANDIDATE_SET.candidates
        ),
        source="public_fixture",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="non-feasible status"):
        posterior.to_goal_belief_v1("final_destination", candidate_set=candidate_set)


def test_flat_projection_allows_unreferenced_provider_unknown_candidate() -> None:
    """The provider's explicit unknown candidate remains separate from known posterior mass."""
    unknown = GoalCandidate(
        id="unknown",
        position=None,
        source="unknown",
        role=GoalCandidateRole.UNKNOWN,
        feasibility_status="unknown",
    )
    candidate_set = GoalCandidateSet(
        candidates=POSTERIOR_CANDIDATE_SET.candidates + (unknown,),
        source="goal_candidate_provider",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    belief = posterior.to_goal_belief_v1("final_destination", candidate_set=candidate_set)

    assert {item.candidate_id for item in belief.candidate_probabilities} == {
        "destination-a",
        "destination-b",
    }


def test_flat_projection_rejects_unknown_role_used_as_known_mass() -> None:
    """The explicit unknown probability bucket must not be duplicated as candidate mass."""
    candidate_set = GoalCandidateSet(
        candidates=tuple(
            replace(candidate, role=GoalCandidateRole.UNKNOWN)
            if candidate.id == "waypoint-a-near"
            else candidate
            for candidate in POSTERIOR_CANDIDATE_SET.candidates
        ),
        source="public_fixture",
    )
    posterior = replace(
        _posterior(),
        candidate_set_digest=stable_digest(candidate_set.to_dict()),
    )

    with pytest.raises(ValueError, match="UNKNOWN role"):
        posterior.to_goal_belief_v1("active_waypoint", candidate_set=candidate_set)


@pytest.mark.parametrize(
    "bad_evidence_source",
    [
        "oracle_upper_bound",
        "simulator_truth",
        "scenario_assigned_route",
        "assigned_route",
        "true_goal",
        "goal_truth",
        "waypoint_truth",
        "future_trajectory",
        "simulator_goal",
        "simulator_route",
        "ground_truth",
        "expert_demonstration",
        "assigned_route_v2",
    ],
)
def test_actor_hierarchy_rejects_privileged_evidence_labels(bad_evidence_source: str) -> None:
    """Oracle, simulator, and truth labels cannot enter actor-side hierarchy state."""
    with pytest.raises(ValueError, match="actor-safe source"):
        _posterior(evidence_source=bad_evidence_source)


def test_innovation_is_an_unbounded_diagnostic_and_not_change_probability() -> None:
    """Slice A keeps NIS-like innovation separate from calibrated change probability."""
    posterior = replace(_posterior(), innovation=4.2)

    belief = posterior.to_goal_belief_v1(
        "final_destination",
        candidate_set=POSTERIOR_CANDIDATE_SET,
    )

    assert posterior.innovation == pytest.approx(4.2)
    assert belief.change_probability == 0.0
    assert "change_probability_unestimated" in belief.blockers
    with pytest.raises(ValueError, match="innovation must be non-negative"):
        replace(posterior, innovation=-0.1)


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
    for candidate_id in ("unknown", "UNKNOWN"):
        with pytest.raises(ValueError, match="reserved"):
            HierarchicalProbability(candidate_id, 0.5)
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
    with pytest.raises(ValueError, match="duplicate waypoint ID"):
        replace(
            posterior,
            waypoint_conditionals=(
                HierarchicalWaypointConditionalV1(
                    "destination-a",
                    (HierarchicalProbability("shared-waypoint", 0.5),),
                    0.5,
                ),
                HierarchicalWaypointConditionalV1(
                    "destination-b",
                    (HierarchicalProbability("shared-waypoint", 0.5),),
                    0.5,
                ),
            ),
            waypoint_parent_destination={"shared-waypoint": "destination-a"},
        )
    with pytest.raises(ValueError, match="cover every waypoint"):
        replace(
            posterior,
            waypoint_parent_destination={
                "waypoint-a-near": "destination-a",
                "waypoint-a-far": "destination-a",
                "waypoint-b": "destination-b",
                "extra-waypoint": "destination-a",
            },
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
        candidate_set_digest=stable_digest(GoalCandidateSet().to_dict()),
    )

    marginal, unknown = posterior.active_waypoint_marginal()
    assert marginal == ()
    assert unknown == 1.0
    empty_candidate_set = GoalCandidateSet()
    assert (
        posterior.to_goal_belief_v1(
            "active_waypoint",
            candidate_set=empty_candidate_set,
        ).unknown_candidate_probability
        == 1.0
    )
    assert (
        posterior.to_goal_belief_v1(
            "final_destination",
            candidate_set=empty_candidate_set,
        ).unknown_candidate_probability
        == 1.0
    )
