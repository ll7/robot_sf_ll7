"""Fixture tests for the experimental risk-aware trajectory ranker (issue #6567).

Smoke / diagnostic evidence only: these tests pin the ranker's orchestration
contract -- candidate shapes, decomposed score components, hard-gate precedence,
peak-risk timing shape, deterministic ordering, and risk provenance -- against
deterministic zero-risk, certain-collision, uncertain-crossing, and
hard-gate-precedence fixtures. They reuse ``CandidateAction``,
``estimate_action_conditioned_risk``, and the schema types verbatim and never
claim calibrated real-world collision probability or planner improvement.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from robot_sf.benchmark.actuator_feasibility import (
    VERDICT_GEOMETRY_ONLY_CLEAR,
    VERDICT_INFEASIBLE,
    ActuatorLimitsConfig,
)
from robot_sf.benchmark.trajectory_verifier import DECISION_FALLBACK_BRAKE
from robot_sf.nav.predictive_types import PedestrianState
from robot_sf.planner.risk_aware_trajectory_ranker import (
    RANKER_CLAIM_BOUNDARY,
    RANKER_SCHEMA_VERSION,
    PrimitiveGeneratorConfig,
    RankingWeights,
    RBFGeneratorConfig,
    generate_primitive_candidates,
    generate_rbf_candidates,
    rank_trajectories,
)
from robot_sf.research.collision_risk import (
    CandidateAction,
    RiskEstimatorConfig,
    RiskProvenance,
    action_from_constant_velocity,
)

HORIZON_STEPS = 20
DT_S = 0.1


def _risk_config(**overrides) -> RiskEstimatorConfig:
    """Return a deterministic estimator config with the given overrides."""
    base = RiskEstimatorConfig(
        horizon_steps=HORIZON_STEPS,
        dt_s=DT_S,
        n_samples=512,
        velocity_std_m_s=0.0,
        robot_radius_m=0.3,
        pedestrian_radius_m=0.3,
        seed=1,
    )
    return replace(base, **overrides)


def _ped(actor_id: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> PedestrianState:
    """Return a pedestrian state at a position with a velocity."""
    return PedestrianState(id=actor_id, position=np.array([x, y]), velocity=np.array([vx, vy]))


def _components_are_finite(record) -> None:
    """Assert every reported score component is a finite real number."""
    components = record.components
    assert math.isfinite(components.calibrated_collision_risk)
    assert math.isfinite(components.travel_time_s)
    assert math.isfinite(components.integrated_jerk)
    assert math.isfinite(components.path_length_m)
    assert math.isfinite(components.clearance_penalty)
    assert 0.0 <= components.calibrated_collision_risk <= 1.0
    assert components.travel_time_s >= 0.0
    assert components.integrated_jerk >= 0.0
    assert components.path_length_m >= 0.0
    assert components.clearance_penalty >= 0.0


# ---------------------------------------------------------------------------
# Deterministic primitive generator
# ---------------------------------------------------------------------------


def test_primitive_generator_shapes_finiteness_and_action_ids() -> None:
    """Generated primitives are finite, uniquely identified, and speed-limited."""
    config = PrimitiveGeneratorConfig(cruise_speed_mps=1.0)
    candidates = generate_primitive_candidates(
        [0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S, config=config
    )

    assert len(candidates) >= 3
    action_ids = [candidate.action_id for candidate in candidates]
    assert len(action_ids) == len(set(action_ids))  # unique
    for candidate in candidates:
        waypoints = candidate.as_array(horizon_steps=HORIZON_STEPS)
        assert waypoints.shape == (HORIZON_STEPS + 1, 2)
        assert np.all(np.isfinite(waypoints))
        sampled_speeds = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / DT_S
        assert float(np.max(sampled_speeds)) <= config.cruise_speed_mps + 1.0e-12


def test_primitive_generator_is_deterministic() -> None:
    """Identical inputs produce identical waypoint sequences and action ids."""
    first = generate_primitive_candidates(
        [0.5, -0.3], [3.0, 1.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    second = generate_primitive_candidates(
        [0.5, -0.3], [3.0, 1.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    assert [c.action_id for c in first] == [c.action_id for c in second]
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.waypoints, right.waypoints)


def test_primitive_generator_rejects_fewer_than_three() -> None:
    """A config that would yield fewer than three primitives fails closed."""
    config = PrimitiveGeneratorConfig(lateral_offsets_m=(0.0,), include_brake_primitive=False)
    with pytest.raises(ValueError, match="at least three"):
        generate_primitive_candidates(
            [0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S, config=config
        )


def test_primitive_generator_rejects_invalid_horizon() -> None:
    """Non-positive horizon or timestep fails closed."""
    with pytest.raises(ValueError):
        generate_primitive_candidates([0.0, 0.0], [2.0, 0.0], horizon_steps=0, dt_s=DT_S)
    with pytest.raises(ValueError):
        generate_primitive_candidates([0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=0.0)
    with pytest.raises(ValueError):
        generate_primitive_candidates(
            [0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=math.inf
        )
    with pytest.raises(ValueError):
        PrimitiveGeneratorConfig(cruise_speed_mps=math.inf)


def test_rbf_generator_shapes_finiteness_and_action_ids() -> None:
    """RBF candidates satisfy the shared finite waypoint contract."""
    config = RBFGeneratorConfig(cruise_speed_mps=1.0)
    candidates = generate_rbf_candidates(
        [0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S, config=config
    )

    assert len(candidates) >= 3
    action_ids = [candidate.action_id for candidate in candidates]
    assert len(action_ids) == len(set(action_ids))
    assert all(candidate.representation == "rbf" for candidate in candidates)
    for candidate in candidates:
        waypoints = candidate.as_array(horizon_steps=HORIZON_STEPS)
        assert waypoints.shape == (HORIZON_STEPS + 1, 2)
        assert np.all(np.isfinite(waypoints))
        sampled_speeds = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / DT_S
        assert float(np.max(sampled_speeds)) <= config.cruise_speed_mps + 1.0e-12


def test_rbf_generator_is_deterministic_and_rejects_short_budgets() -> None:
    """RBF proposals are repeatable and fail closed below the minimum budget."""
    first = generate_rbf_candidates([0.5, -0.3], [3.0, 1.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S)
    second = generate_rbf_candidates(
        [0.5, -0.3], [3.0, 1.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    assert [c.action_id for c in first] == [c.action_id for c in second]
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.waypoints, right.waypoints)

    config = RBFGeneratorConfig(lateral_offsets_m=(0.0,), include_brake_primitive=False)
    with pytest.raises(ValueError, match="at least three"):
        generate_rbf_candidates(
            [0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S, config=config
        )


def test_rbf_candidates_use_the_existing_ranker_gate_and_risk_schema() -> None:
    """RBF candidates are ranked through the same gates and provenance schema."""
    candidates = generate_rbf_candidates(
        [0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    records = rank_trajectories(candidates, [_ped(1, 1.0, 0.7)], risk_config=_risk_config())

    assert len(records) == len(candidates)
    assert {record.action_id for record in records} == {
        candidate.action_id for candidate in candidates
    }
    assert all(record.provenance.action_representation == "rbf" for record in records)
    assert all(record.claim_boundary == RANKER_CLAIM_BOUNDARY for record in records)


# ---------------------------------------------------------------------------
# Ranking: zero-risk, deterministic collision, uncertain crossing
# ---------------------------------------------------------------------------


def test_rank_zero_risk_is_eligible_with_no_peak() -> None:
    """A robot moving away from a distant actor yields zero risk and eligibility."""
    action = action_from_constant_velocity(
        "away", [0.0, 0.0], [-1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    pedestrian = _ped(1, 5.0, 0.0)
    rankings = rank_trajectories([action], [pedestrian], risk_config=_risk_config())

    assert len(rankings) == 1
    record = rankings[0]
    assert record.eligible is True
    assert record.hard_gate.verifier_decision == "accept"
    assert record.joint_contact_probability == 0.0
    assert record.components.calibrated_collision_risk == 0.0
    assert record.peak_risk.peak_step == -1
    assert record.peak_risk.peak_actor_id is None
    _components_are_finite(record)


def test_rank_no_pedestrian_hazard_is_eligible() -> None:
    """With no pedestrians the verifier is skipped and the candidate stays eligible."""
    action = action_from_constant_velocity(
        "cruise", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    rankings = rank_trajectories([action], [], risk_config=_risk_config())

    record = rankings[0]
    assert record.eligible is True
    assert record.hard_gate.verifier_decision == "skipped_no_hazard"
    assert record.hard_gate.actuator_verdict != VERDICT_INFEASIBLE
    assert record.components.min_clearance_m == float("inf")
    assert record.components.clearance_penalty == 0.0


def test_rank_no_hazard_does_not_apply_fixed_clearance_cap() -> None:
    """No-hazard actuator checks do not reject a high-speed constant trajectory."""
    action = action_from_constant_velocity(
        "fast_cruise", [0.0, 0.0], [200.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    record = rank_trajectories([action], [], risk_config=_risk_config())[0]

    assert record.components.min_clearance_m == float("inf")
    assert record.hard_gate.actuator_verdict == "actuator_feasible"
    assert record.eligible is True


def test_rank_hard_verifier_uses_nominal_pedestrian_forecast() -> None:
    """A future nominal contact is visible to the deterministic hard verifier."""
    action = action_from_constant_velocity(
        "stationary", [0.0, 0.0], [0.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    pedestrian = _ped(1, 0.0, 2.0, vy=-1.0)
    rankings = rank_trajectories(
        [action], [pedestrian], risk_config=_risk_config(velocity_std_m_s=0.0)
    )

    record = rankings[0]
    assert record.components.min_clearance_m < 0.0
    assert record.hard_gate.verifier_decision == DECISION_FALLBACK_BRAKE
    assert "min_clearance_hard" in record.hard_gate.violated_predicates
    assert record.eligible is False


def test_rank_deterministic_collision_is_ineligible() -> None:
    """A head-on drive into a stationary actor is ineligible under both hard gates."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    pedestrian = _ped(1, 1.0, 0.0)
    rankings = rank_trajectories([action], [pedestrian], risk_config=_risk_config())

    record = rankings[0]
    assert record.eligible is False
    assert record.hard_gate.verifier_decision == DECISION_FALLBACK_BRAKE
    assert record.hard_gate.actuator_verdict == VERDICT_INFEASIBLE
    assert record.joint_contact_probability == 1.0
    assert record.components.min_clearance_m < 0.0
    assert record.estimate.deterministic.contact_certain is True
    assert record.peak_risk.peak_step >= 0
    _components_are_finite(record)


def test_rank_uncertain_crossing_has_nontrivial_probability_and_peak() -> None:
    """An uncertain perpendicular crossing yields a non-degenerate risk and peak step."""
    action = action_from_constant_velocity(
        "cross", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    pedestrian = _ped(1, 1.0, 1.0, vx=0.0, vy=-0.6)
    config = _risk_config(velocity_std_m_s=0.4, n_samples=2048, seed=99)

    rankings = rank_trajectories([action], [pedestrian], risk_config=config)
    record = rankings[0]

    # Strictly between 0 and 1: a genuinely uncertain (non-degenerate) estimate.
    assert 0.1 < record.joint_contact_probability < 0.9
    assert 0.1 < record.components.calibrated_collision_risk < 0.9
    assert 0 <= record.peak_risk.peak_step < HORIZON_STEPS
    assert record.peak_risk.peak_actor_id == 1
    assert max(record.peak_risk.first_passage_distribution) > 0.0
    _components_are_finite(record)


# ---------------------------------------------------------------------------
# Hard-gate precedence
# ---------------------------------------------------------------------------


def test_hard_gate_precedence_zero_risk_candidate_still_ineligible() -> None:
    """A zero-probability candidate rejected by a hard gate stays ineligible.

    The deterministic primitive that curves away from the pedestrian has zero
    model contact probability under zero velocity noise, yet the trajectory
    verifier fires ``fallback_brake`` (low TTC while initially heading toward the
    actor). A low collision probability must never override the hard check.
    """
    candidates = generate_primitive_candidates(
        [0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    pedestrian = _ped(1, 1.0, 0.7)
    rankings = rank_trajectories(
        candidates, [pedestrian], risk_config=_risk_config(velocity_std_m_s=0.0)
    )

    ineligible = [r for r in rankings if not r.eligible]
    eligible = [r for r in rankings if r.eligible]
    assert eligible, "expected at least one eligible candidate (the brake primitive)"
    assert ineligible, "expected at least one hard-gate-rejected candidate"

    # At least one ineligible candidate carries zero model collision risk.
    zero_risk_ineligible = [r for r in ineligible if r.joint_contact_probability == 0.0]
    assert zero_risk_ineligible, "a low-risk candidate must still be gated out"
    gated = zero_risk_ineligible[0]
    assert gated.hard_gate.verifier_decision == DECISION_FALLBACK_BRAKE
    assert gated.rank == -1

    # Eligible candidates are ranked (rank >= 1) and listed before ineligible ones.
    assert all(record.rank >= 1 for record in eligible)
    assert all(record.rank == -1 for record in ineligible)
    first_ineligible_index = rankings.index(ineligible[0])
    assert all(rankings.index(record) < first_ineligible_index for record in eligible)


def test_hard_gate_precedence_low_risk_does_not_override_brake_infeasible() -> None:
    """An actuator-infeasible candidate is ineligible regardless of its risk score.

    A candidate whose footprint overlaps the pedestrian (negative clearance) is
    ``infeasible`` at the actuator gate and ``fallback_brake`` at the verifier,
    even when its declared model probability is moderate.
    """
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    pedestrian = _ped(1, 0.5, 0.0)  # close enough to force footprint overlap
    rankings = rank_trajectories([action], [pedestrian], risk_config=_risk_config())

    record = rankings[0]
    assert record.eligible is False
    assert record.hard_gate.actuator_verdict == VERDICT_INFEASIBLE
    assert record.components.min_clearance_m < 0.0


def test_ineligible_candidates_are_not_ordered_by_composite_score() -> None:
    """Rejected candidates stay unranked even when their scores differ."""
    higher_cost = action_from_constant_velocity(
        "a_higher_cost", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    lower_cost = action_from_constant_velocity(
        "z_lower_cost", [0.0, 0.0], [0.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    rankings = rank_trajectories(
        [lower_cost, higher_cost], [_ped(1, 0.1, 0.0)], risk_config=_risk_config()
    )

    assert [record.action_id for record in rankings] == ["a_higher_cost", "z_lower_cost"]
    assert all(record.rank == -1 for record in rankings)
    assert rankings[0].composite_score > rankings[1].composite_score


# ---------------------------------------------------------------------------
# Decomposed components, deterministic ordering, provenance, peak timing
# ---------------------------------------------------------------------------


def test_rank_reports_every_component_separately() -> None:
    """The ranking exposes each score component, not only a composite scalar."""
    action = action_from_constant_velocity(
        "cruise", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    pedestrian = _ped(1, 3.0, 0.0)
    weights = RankingWeights()
    rankings = rank_trajectories(
        [action], [pedestrian], risk_config=_risk_config(), weights=weights
    )
    record = rankings[0]
    components = record.components

    # Every named component is present and distinct from the composite.
    for name in (
        "calibrated_collision_risk",
        "travel_time_s",
        "integrated_jerk",
        "path_length_m",
        "clearance_penalty",
    ):
        assert hasattr(components, name)
    assert components.calibration_applied is False
    assert record.composite_score == pytest.approx(
        weights.w_risk * components.calibrated_collision_risk
        + weights.w_time * components.travel_time_s
        + weights.w_jerk * components.integrated_jerk
        + weights.w_length * components.path_length_m
        + weights.w_clearance * components.clearance_penalty
    )
    assert record.claim_boundary == RANKER_CLAIM_BOUNDARY


def test_ranker_emits_no_safe_verdict() -> None:
    """The ranker never emits a structural ``safe`` verdict, even at zero risk."""
    action = action_from_constant_velocity(
        "away", [0.0, 0.0], [-1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    rankings = rank_trajectories([action], [_ped(1, 5.0, 0.0)], risk_config=_risk_config())
    record = rankings[0]
    payload = record.estimate.to_dict()

    keys: set[str] = set()

    def collect(node: object) -> None:
        """Recursively collect mapping keys from a nested JSON-like structure."""
        if isinstance(node, dict):
            keys.update(str(key) for key in node)
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(payload)
    assert not any("safe" in key.lower() for key in keys)
    assert "experimental" in record.claim_boundary.lower()


def test_rank_ordering_is_deterministic() -> None:
    """Ranking the same candidate set twice yields identical order and scores."""
    candidates = generate_primitive_candidates(
        [0.0, 0.0], [2.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    pedestrians = [_ped(1, 1.0, 0.7), _ped(2, 1.5, -0.5, vx=-0.3)]
    config = _risk_config(velocity_std_m_s=0.2, n_samples=256, seed=42)

    first = rank_trajectories(candidates, pedestrians, risk_config=config)
    second = rank_trajectories(candidates, pedestrians, risk_config=config)

    assert [r.action_id for r in first] == [r.action_id for r in second]
    assert [r.rank for r in first] == [r.rank for r in second]
    for left, right in zip(first, second, strict=True):
        assert left.eligible == right.eligible
        assert left.composite_score == pytest.approx(right.composite_score)
        assert left.joint_contact_probability == pytest.approx(right.joint_contact_probability)


def test_rank_provenance_reuses_the_canonical_estimator_schema() -> None:
    """Each ranking exposes the canonical full estimator provenance unchanged."""
    action = CandidateAction(
        action_id="custom", waypoints=np.zeros((HORIZON_STEPS + 1, 2)), representation="zeros"
    )
    config = _risk_config(seed=7)
    rankings = rank_trajectories([action], [_ped(1, 5.0, 0.0)], risk_config=config)
    record = rankings[0]
    provenance = record.provenance

    assert isinstance(provenance, RiskProvenance)
    assert provenance is record.estimate.provenance
    assert provenance.action_id == "custom"
    assert provenance.estimator_id
    assert provenance.forecast_model
    assert provenance.geometry_version
    assert provenance.config_hash
    assert provenance.seed == 7
    assert provenance.horizon_steps == HORIZON_STEPS
    assert provenance.dt_s == DT_S
    assert record.ranker_schema_version == RANKER_SCHEMA_VERSION


def test_peak_risk_timing_matches_critical_interval_shape() -> None:
    """Peak-risk timing exposes anchor step, time, window, and first-passage mass."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    rankings = rank_trajectories([action], [_ped(1, 1.0, 0.0)], risk_config=_risk_config())
    peak = rankings[0].peak_risk

    assert peak.peak_step >= 0
    assert peak.peak_time_s == pytest.approx(peak.peak_step * DT_S)
    assert 0 <= peak.window_start_step <= peak.peak_step
    assert peak.peak_step < peak.window_end_step <= HORIZON_STEPS
    assert len(peak.first_passage_distribution) == HORIZON_STEPS


def test_rank_rejects_nonintegral_peak_window() -> None:
    """Critical-interval windows must preserve integer step indices."""
    action = action_from_constant_velocity(
        "cruise", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )

    with pytest.raises(ValueError, match="non-negative integer"):
        rank_trajectories(
            [action],
            [],
            risk_config=_risk_config(),
            peak_window_half_steps=0.5,  # ty: ignore
        )


def test_rank_reuses_candidate_action_schema_verbatim() -> None:
    """The ranker consumes CandidateAction produced by the reused estimator helper."""
    action = action_from_constant_velocity(
        "reuse", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    assert isinstance(action, CandidateAction)
    rankings = rank_trajectories([action], [], risk_config=_risk_config())
    assert rankings[0].estimate.provenance.action_id == "reuse"


def test_rank_actuator_geometry_only_clear_is_ineligible() -> None:
    """A physically infeasible ``geometry_only_clear`` candidate is gated out."""
    # A tight actuator limit makes braking physically infeasible while geometry
    # is still clear, producing ``geometry_only_clear`` rather than ``infeasible``.
    action = action_from_constant_velocity(
        "cruise", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    tight_actuator = ActuatorLimitsConfig(
        max_decel_mps2=0.1, command_latency_s=0.5, brake_latency_s=0.5
    )
    pedestrian = _ped(1, 3.0, 0.0)  # no footprint overlap (clearance > 0)
    rankings = rank_trajectories(
        [action],
        [pedestrian],
        risk_config=_risk_config(),
        actuator_config=tight_actuator,
    )
    record = rankings[0]
    # Non-overlapping geometry plus a missed brake deadline yields the
    # ``geometry_only_clear`` is geometrically clear but physically infeasible,
    # so the ranker must not select it despite the non-negative clearance.
    assert record.hard_gate.actuator_verdict == VERDICT_GEOMETRY_ONLY_CLEAR
    assert record.components.min_clearance_m > 0.0
    assert record.eligible is False
    assert record.hard_gate.ineligibility_reason is not None
    assert "physically infeasible" in record.hard_gate.ineligibility_reason
