"""Fixture tests for the action-conditioned online collision-risk API (issue #5444).

These are *API + baseline fixture* tests, not calibrated benchmark risk evidence.
They pin the schema contract and the constant-velocity Monte Carlo / deterministic
baselines against analytic no-contact, certain-contact, grazing, correlated
multi-actor, and action-sensitive scenarios.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_sf.nav.predictive_types import PedestrianState
from robot_sf.research.collision_risk import (
    DETERMINISTIC_FIELD_LABEL,
    RISK_SCHEMA_VERSION,
    CollisionRiskInputError,
    RiskEstimatorConfig,
    RiskSchemaError,
    action_from_constant_velocity,
    estimate_action_conditioned_risk,
    latency_summary_from_samples,
)
from robot_sf.research.collision_risk.estimators import CandidateAction

HORIZON_STEPS = 20
DT_S = 0.1


def _config(**overrides) -> RiskEstimatorConfig:
    base = {
        "horizon_steps": HORIZON_STEPS,
        "dt_s": DT_S,
        "n_samples": 1024,
        "velocity_std_m_s": 0.3,
        "robot_radius_m": 0.3,
        "pedestrian_radius_m": 0.3,
        "seed": 1234,
    }
    base.update(overrides)
    return RiskEstimatorConfig(**base)


def _stationary_ped(actor_id: int, x: float, y: float) -> PedestrianState:
    return PedestrianState(id=actor_id, position=np.array([x, y]), velocity=np.array([0.0, 0.0]))


def test_collision_risk_deterministic_no_contact() -> None:
    """Robot moving away from a distant actor yields zero contact probability."""
    action = action_from_constant_velocity(
        "away", [0.0, 0.0], [-1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    ped = _stationary_ped(1, 5.0, 0.0)
    estimate = estimate_action_conditioned_risk(action, [ped], _config())

    assert estimate.joint_contact_probability == 0.0
    assert estimate.no_contact_probability == 1.0
    assert estimate.deterministic.contact_certain is False
    assert estimate.deterministic.ttc_s == float("inf")
    assert estimate.deterministic.first_contact_step == -1
    assert estimate.deterministic.min_clearance_m > 0.0
    assert sum(estimate.first_passage_distribution) == 0.0


def test_collision_risk_deterministic_certain_contact() -> None:
    """Robot driving into a stationary actor is a certain deterministic contact."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    ped = _stationary_ped(1, 1.0, 0.0)
    estimate = estimate_action_conditioned_risk(action, [ped], _config())

    assert estimate.deterministic.contact_certain is True
    assert estimate.deterministic.first_contact_step >= 0
    assert estimate.deterministic.ttc_s < estimate.provenance.horizon_s
    assert estimate.deterministic.min_clearance_m < 0.0
    # With small velocity noise on a stationary head-on actor, contact is near-certain.
    assert estimate.joint_contact_probability > 0.9
    assert estimate.deterministic.velocity_obstacle_flags == (True,)


def test_collision_risk_grazing_min_clearance_near_zero() -> None:
    """A grazing pass touches at exactly the summed footprint radius."""
    action = action_from_constant_velocity(
        "straight", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    # radii sum = 0.6; place actor exactly 0.6 m off the robot's path abeam x=1.0.
    ped = _stationary_ped(1, 1.0, 0.6)
    estimate = estimate_action_conditioned_risk(action, [ped], _config(velocity_std_m_s=0.0))
    # Noise-free grazing => minimum clearance is ~0 at closest approach.
    assert estimate.deterministic.min_clearance_m == pytest.approx(0.0, abs=1e-6)
    # Zero noise: the joint estimate is degenerate at the boundary (contact <= 0).
    assert estimate.joint_contact_probability == pytest.approx(1.0)


def test_collision_risk_multi_actor_joint_below_union_bound() -> None:
    """Joint contact stays at or below the union bound and differs from independence."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    ped_a = PedestrianState(id=1, position=np.array([1.5, 0.3]), velocity=np.array([-0.6, 0.0]))
    ped_b = PedestrianState(id=2, position=np.array([1.5, -0.3]), velocity=np.array([-0.6, 0.0]))
    estimate = estimate_action_conditioned_risk(
        action, [ped_a, ped_b], _config(cross_actor_correlation=0.8, velocity_std_m_s=0.4)
    )

    joint = estimate.joint_contact_probability
    marginals = [c.marginal_contact_probability for c in estimate.per_actor]
    # Union bound is an upper bound on the true joint probability.
    assert joint <= estimate.union_bound_probability + 1e-9
    # Joint is at least the largest single marginal.
    assert joint >= max(marginals) - 1e-9
    # The independence approximation is a *different* number (intentionally invalid).
    assert estimate.independence_approx_probability != pytest.approx(joint, abs=1e-3)


def test_collision_risk_action_sensitive_direction() -> None:
    """Risk is higher for an action steering into actors than one steering away."""
    ped_a = PedestrianState(id=1, position=np.array([1.5, 0.2]), velocity=np.array([-0.4, 0.0]))
    ped_b = PedestrianState(id=2, position=np.array([1.5, -0.2]), velocity=np.array([-0.4, 0.0]))
    config = _config(velocity_std_m_s=0.35)

    toward = action_from_constant_velocity(
        "toward", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    away = action_from_constant_velocity(
        "sidestep", [0.0, 0.0], [0.2, 1.2], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    risk_toward = estimate_action_conditioned_risk(toward, [ped_a, ped_b], config)
    risk_away = estimate_action_conditioned_risk(away, [ped_a, ped_b], config)

    assert risk_toward.joint_contact_probability > risk_away.joint_contact_probability


def test_collision_risk_first_passage_sums_to_joint_and_hazard_bounded() -> None:
    """First-passage distribution sums to the joint probability with bounded hazard."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [0.8, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    ped = PedestrianState(id=1, position=np.array([1.2, 0.1]), velocity=np.array([-0.3, 0.0]))
    estimate = estimate_action_conditioned_risk(action, [ped], _config())

    assert sum(estimate.first_passage_distribution) == pytest.approx(
        estimate.joint_contact_probability, abs=1e-9
    )
    assert len(estimate.binned_hazard) == HORIZON_STEPS
    assert all(0.0 <= hazard <= 1.0 for hazard in estimate.binned_hazard)


def test_collision_risk_schema_validation_and_json_safe() -> None:
    """The estimate validates, round-trips to JSON, and carries full provenance."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    estimate = estimate_action_conditioned_risk(action, [_stationary_ped(1, 1.0, 0.1)], _config())

    estimate.validate()  # does not raise
    payload = estimate.to_dict()
    dumped = json.dumps(payload)  # strict JSON, no NaN/inf leakage
    assert json.loads(dumped)["schema_version"] == RISK_SCHEMA_VERSION

    prov = estimate.provenance
    assert prov.estimator_id and prov.forecast_model and prov.geometry_version
    assert prov.action_id == "into"
    assert prov.horizon_s == pytest.approx(prov.horizon_steps * prov.dt_s)
    assert prov.config_hash  # non-empty stable hash


def _collect_keys(payload: object) -> set[str]:
    """Recursively collect every mapping key in a nested JSON-like structure."""
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key))
            keys |= _collect_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _collect_keys(item)
    return keys


def test_collision_risk_no_safe_label_emitted() -> None:
    """The API never emits a ``safe`` verdict field, even when risk is low."""
    action = action_from_constant_velocity(
        "away", [0.0, 0.0], [-1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    estimate = estimate_action_conditioned_risk(action, [_stationary_ped(1, 5.0, 0.0)], _config())
    payload = estimate.to_dict()

    # No structural key asserts safety; only probabilities + deterministic warnings exist.
    keys = _collect_keys(payload)
    assert not any("safe" in key.lower() for key in keys)
    assert estimate.deterministic.label == DETERMINISTIC_FIELD_LABEL
    assert "authoritative" in estimate.guard_authority_note.lower()
    assert estimate.joint_contact_probability == 0.0  # low risk, still no safe verdict


def test_collision_risk_latency_summary_classification() -> None:
    """Each estimate carries a latency summary classified against the deadline."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    estimate = estimate_action_conditioned_risk(action, [_stationary_ped(1, 1.0, 0.0)], _config())
    assert estimate.latency is not None
    assert estimate.latency.classification in {"online", "offline_only"}
    assert estimate.latency.p95_ms >= 0.0

    summary = latency_summary_from_samples([10.0, 20.0, 200.0], deadline_ms=100.0)
    assert summary.deadline_misses == 1
    assert summary.classification == "offline_only"
    assert summary.p50_ms == pytest.approx(20.0)


def test_collision_risk_out_of_distribution_abstains() -> None:
    """An actor faster than the model range flags OOD and abstains."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    fast_ped = PedestrianState(id=1, position=np.array([2.0, 0.0]), velocity=np.array([9.0, 0.0]))
    estimate = estimate_action_conditioned_risk(
        action, [fast_ped], _config(max_pedestrian_speed_m_s=2.5)
    )
    assert estimate.uncertainty.ood_actor_flags == (True,)
    assert estimate.uncertainty.abstained is True
    assert estimate.uncertainty.abstention_reasons


def test_collision_risk_fail_closed_on_bad_inputs() -> None:
    """Malformed configuration and actions fail closed rather than silently degrade."""
    with pytest.raises(CollisionRiskInputError):
        RiskEstimatorConfig(horizon_steps=0)
    with pytest.raises(CollisionRiskInputError):
        RiskEstimatorConfig(cross_actor_correlation=1.0)

    bad_action = CandidateAction(action_id="bad", waypoints=np.zeros((3, 2)))
    with pytest.raises(CollisionRiskInputError):
        estimate_action_conditioned_risk(bad_action, [], _config())


def test_collision_risk_schema_rejects_inconsistent_probabilities() -> None:
    """The schema validator rejects a joint/no-contact pair that does not sum to 1."""
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    estimate = estimate_action_conditioned_risk(action, [_stationary_ped(1, 1.0, 0.0)], _config())
    broken = type(estimate)(
        **{
            **{f.name: getattr(estimate, f.name) for f in estimate.__dataclass_fields__.values()},
            "no_contact_probability": 0.0,
        }
    )
    with pytest.raises(RiskSchemaError):
        broken.validate()


def test_public_contact_geometry_matches_private_aliases() -> None:
    """The public contact geometry (issue #5468) is the exact private objects.

    ``segment_min_distance`` / ``pedestrian_arrays`` were promoted from the
    leading-underscore internals so downstream calibration and label-generation
    code can match the estimator geometry through a documented public surface.
    Both handles must resolve to the same function object (not a re-implementation)
    and be re-exported from the package root.
    """
    from robot_sf.research.collision_risk import estimators as est
    from robot_sf.research.collision_risk import pedestrian_arrays, segment_min_distance

    assert segment_min_distance is est.segment_min_distance
    assert pedestrian_arrays is est.pedestrian_arrays
    # Backward-compatible private aliases still point at the promoted objects.
    assert est._segment_min_distance is segment_min_distance
    assert est._pedestrian_arrays is pedestrian_arrays


def test_public_contact_geometry_computes_matched_contact() -> None:
    """Public geometry helpers reproduce the estimator's contact predicate.

    A head-on constant-velocity encounter must yield a non-positive footprint
    clearance (contact) when scored through the public ``pedestrian_arrays`` +
    ``segment_min_distance`` surface, matching the deterministic estimator field.
    """
    from robot_sf.research.collision_risk import pedestrian_arrays, segment_min_distance

    config = _config()
    action = action_from_constant_velocity(
        "into", [0.0, 0.0], [1.0, 0.0], horizon_steps=HORIZON_STEPS, dt_s=DT_S
    )
    ped = PedestrianState(id=1, position=np.array([1.5, 0.0]), velocity=np.array([-1.0, 0.0]))

    robot_xy = action.as_array(horizon_steps=config.horizon_steps)
    ped_pos, ped_vel, radii, ids = pedestrian_arrays([ped], config)
    assert ped_pos.shape == (1, 2)
    assert ids.tolist() == [1]

    steps = np.arange(config.horizon_steps + 1, dtype=float)
    actor_xy = ped_pos[:, None, :] + steps[None, :, None] * config.dt_s * ped_vel[:, None, :]
    clearance = segment_min_distance(robot_xy, actor_xy) - (radii + config.robot_radius_m)[:, None]
    contact = bool((clearance <= 0.0).any())

    estimate = estimate_action_conditioned_risk(action, [ped], config)
    assert contact is True
    assert estimate.deterministic.contact_certain is contact
