"""Contract tests for identity-safe multimodal grid-time arbitration."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import robot_sf.planner.multimodal_trajectory_arbitrator as arb_module
from robot_sf.benchmark.actuator_feasibility import ActuatorLimitsConfig
from robot_sf.nav.predictive_types import MultimodalPrediction, PedestrianForecast, TrajectoryMode
from robot_sf.planner.maneuver_candidates import ManeuverCandidate, ManeuverId
from robot_sf.planner.multimodal_trajectory_arbitrator import (
    ARBITRATION_CLAIM_BOUNDARY,
    MultimodalArbitrationConfig,
    arbitrate_multimodal_trajectories,
    discrete_tail_metrics,
)
from robot_sf.planner.scenario_belief_adapter import (
    IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION,
    TRACK_EXISTENCE_PROBABILITY_SEMANTICS,
    BeliefAwarePlannerInput,
    IdentitySafeProjectionDiagnostics,
    PlannerTrackBelief,
)
from robot_sf.research.collision_risk import (
    CandidateAction,
    RiskEstimatorConfig,
    action_from_constant_velocity,
)
from robot_sf.research.collision_risk.estimators import estimate_trajectory_mode_risk
from robot_sf.research.collision_risk.schema import TrajectoryModeRiskInput

HORIZON = 10
DT_S = 0.1
TRACKER_NAMESPACE = "arbitration-fixture"


class _FixtureCandidateAction(CandidateAction):
    """Annotate the legacy action shape with the required fixture contracts."""

    metadata = {
        "timestamp_s": 0.0,
        "dt_s": DT_S,
        "horizon_steps": HORIZON,
        "robot_radius_m": 0.2,
        "static_feasible": True,
        "generation_mode": "fixture",
    }


def _risk_config(**overrides: object) -> RiskEstimatorConfig:
    """Return a small reproducible #9813 discrete-grid estimator config."""
    config = RiskEstimatorConfig(
        horizon_steps=HORIZON,
        dt_s=DT_S,
        n_samples=128,
        robot_radius_m=0.2,
        pedestrian_radius_m=0.2,
        seed=7,
    )
    return replace(config, **overrides)


def _action(action_id: str, velocity: tuple[float, float] = (0.0, 0.0)) -> CandidateAction:
    """Create a canonical H+1 robot waypoint action."""
    action = action_from_constant_velocity(
        action_id, (0.0, 0.0), velocity, horizon_steps=HORIZON, dt_s=DT_S
    )
    return _FixtureCandidateAction(action.action_id, action.waypoints, action.representation)


def _maneuver_candidate(action_id: str, metadata: dict[str, object]) -> ManeuverCandidate:
    """Build a producer-owned candidate fixture with immutable aligned states."""
    action = _action(action_id)
    positions = action.as_array(horizon_steps=HORIZON)
    states = np.zeros((HORIZON + 1, 5), dtype=float)
    states[:, :2] = positions
    return ManeuverCandidate(
        candidate_id=action_id,
        maneuver=ManeuverId.ROUTE_FOLLOW,
        action=action,
        controls=((0.0, 0.0),) * HORIZON,
        states=states,
        generation_source="test_fixture",
        generation_rank=0,
        metadata=metadata,
    )


def _mode(
    mode_id: str,
    position: tuple[float, float],
    *,
    probability: float = 1.0,
    velocity: tuple[float, float] = (0.0, 0.0),
    covariance: float = 0.0,
) -> TrajectoryMode:
    """Build a future-only trajectory mode with explicit covariance marginals."""
    steps = np.arange(1, HORIZON + 1, dtype=np.float32)[:, None]
    start = np.asarray(position, dtype=np.float32)
    speed = np.asarray(velocity, dtype=np.float32)
    means = start[None, :] + steps * DT_S * speed[None, :]
    cov = np.zeros((HORIZON, 2, 2), dtype=np.float32)
    cov[:, 0, 0] = covariance
    cov[:, 1, 1] = covariance
    return TrajectoryMode(mode_id=mode_id, probability=probability, mean=means, covariance=cov)


def _forecast(
    track_id: int,
    modes: tuple[TrajectoryMode, ...],
    *,
    existence_probability: float = 1.0,
    source_confidence: float = 1.0,
    age_steps: int = 0,
    metadata: dict[str, object] | None = None,
) -> MultimodalPrediction:
    """Build one-track forecast fixture, marking non-tracker existence semantics explicitly."""
    track_metadata: dict[str, object] = {
        "source": "force_residual_intent_predictor",
    }
    if existence_probability != 1.0:
        track_metadata["existence_probability_calibrated"] = True
        track_metadata["existence_probability_semantics"] = "synthetic_fixture_weight"
    if metadata:
        track_metadata.update(metadata)
    return MultimodalPrediction(
        forecasts={
            track_id: PedestrianForecast(
                pedestrian_id=track_id,
                modes=modes,
                existence_probability=existence_probability,
                source_confidence=source_confidence,
                age_steps=age_steps,
                metadata=track_metadata,
            )
        },
        prediction_horizon=HORIZON * DT_S,
        prediction_dt=DT_S,
        timestamp=0.0,
        metadata={"step": max(0, age_steps)},
    )


def _join_belief(
    forecast: MultimodalPrediction,
    *,
    stale_track_ids: frozenset[int] = frozenset(),
) -> tuple[MultimodalPrediction, BeliefAwarePlannerInput]:
    """Attach valid tracker lifecycle identities and build the matching sidecar."""
    step = int(forecast.metadata.get("step", 0))
    tracks: dict[int, PlannerTrackBelief] = {}
    forecasts: dict[int, PedestrianForecast] = {}
    tokens: list[str] = []
    for track_id in sorted(forecast.forecasts):
        forecast_track = forecast.forecasts[track_id]
        ordered_modes = tuple(sorted(forecast_track.modes, key=lambda item: item.mode_id))
        if ordered_modes:
            first = ordered_modes[0]
            if first.mean.shape[0] > 1:
                velocity = (np.asarray(first.mean[1], dtype=float) - first.mean[0]) / DT_S
            else:
                velocity = np.zeros(2)
            position = np.asarray(first.mean[0], dtype=float) - DT_S * velocity
        else:
            velocity = np.zeros(2)
            position = np.zeros(2)
        token = json.dumps(
            [TRACKER_NAMESPACE, 0, track_id], ensure_ascii=False, separators=(",", ":")
        )
        tokens.append(token)
        track_metadata = dict(forecast_track.metadata)
        track_metadata.update(
            {
                "source": "force_residual_intent_predictor",
                "identity_token": token,
                "tracker_namespace": TRACKER_NAMESPACE,
                "reset_epoch": 0,
            }
        )
        joined_modes: list[TrajectoryMode] = []
        for mode in ordered_modes:
            mode_metadata = dict(mode.metadata)
            mode_metadata.update(
                {
                    "identity_token": token,
                    "time_offsets_s": (np.arange(HORIZON, dtype=float) + 1.0) * DT_S,
                }
            )
            joined_modes.append(replace(mode, metadata=mode_metadata))
        forecast_track = replace(
            forecast_track,
            modes=tuple(joined_modes),
            metadata=track_metadata,
        )
        forecasts[track_id] = forecast_track
        is_stale = track_id in stale_track_ids
        age = forecast_track.age_steps
        mean_state = np.array([*position, *velocity, 0.2], dtype=float)
        covariance = np.zeros((4, 4), dtype=float)
        tracks[track_id] = PlannerTrackBelief(
            track_id=track_id,
            mean_state=mean_state,
            covariance=covariance,
            confidence=forecast_track.source_confidence,
            existence_probability=1.0,
            existence_probability_calibrated=False,
            existence_probability_semantics=TRACK_EXISTENCE_PROBABILITY_SEMANTICS,
            visibility=not is_stale,
            age_steps=age,
            source="pedestrian_tracker",
            tracker_namespace=TRACKER_NAMESPACE,
            reset_epoch=0,
            lifecycle_token=token,
            belief_step=step,
            last_observed_step=step,
            missed_steps=1 if is_stale else 0,
            status="lost" if is_stale else "confirmed",
        )
    joined_forecast = replace(
        forecast, forecasts=forecasts, metadata={**forecast.metadata, "step": step}
    )
    diagnostics = IdentitySafeProjectionDiagnostics(
        status="supported" if tracks else "empty",
        planner_name="BeliefGuidedLocalPlanner",
        belief_step=step,
        visible_track_count=sum(track.visibility for track in tracks.values()),
        occluded_track_count=sum(not track.visibility for track in tracks.values()),
        stale_track_count=sum(track.missed_steps > 0 for track in tracks.values()),
        retained_track_count=len(tracks),
        retired_track_count=0,
        dropped_track_count=0,
        per_reason_drop_count=(),
        fallback_reason=None,
        ordered_track_ids=tuple(sorted(tracks)),
        retired_track_ids=(),
        identity_lifecycle_tokens=tuple(tokens),
    )
    belief = BeliefAwarePlannerInput(
        legacy_observation=None,
        tracks=tracks,
        belief_step=step,
        schema_version=IDENTITY_SAFE_PLANNER_INPUT_SCHEMA_VERSION,
        diagnostics=diagnostics,
    )
    return joined_forecast, belief


def _empty_belief(step: int = 0) -> BeliefAwarePlannerInput:
    """Build a valid empty identity-safe belief input."""
    empty = MultimodalPrediction(
        forecasts={},
        prediction_horizon=HORIZON * DT_S,
        prediction_dt=DT_S,
        timestamp=0.0,
        metadata={"step": step},
    )
    return _join_belief(empty)[1]


def _arbitrate(candidates: list[object], forecast: MultimodalPrediction, **kwargs: object):
    """Join fixture identities then run the production evaluator."""
    risk_config = kwargs.pop("risk_config", _risk_config())
    joined, belief = _join_belief(forecast)
    route_progress = kwargs.pop("route_progress", None)
    if route_progress is None:
        route_progress = {str(_candidate_id(item)): 0.0 for item in candidates}
    supplied_belief = kwargs.pop("belief", belief)
    if supplied_belief is not belief:
        joined = forecast
    return arbitrate_multimodal_trajectories(
        candidates,
        joined,
        belief=supplied_belief,
        risk_config=risk_config,
        route_progress=route_progress,
        **kwargs,
    )


def _evaluate_base(candidates: list[object], forecast: MultimodalPrediction, **kwargs: object):
    """Run the additive evaluate-once seam against the shared fixture belief."""
    risk_config = kwargs.pop("risk_config", _risk_config())
    joined, belief = _join_belief(forecast)
    route_progress = kwargs.pop("route_progress", None)
    if route_progress is None:
        route_progress = {str(_candidate_id(item)): 0.0 for item in candidates}
    return arb_module.evaluate_multimodal_trajectories_base(
        candidates,
        joined,
        belief=belief,
        risk_config=risk_config,
        route_progress=route_progress,
        **kwargs,
    )


def _candidate_id(candidate: object) -> str:
    """Read a stable candidate identifier from either public candidate form."""
    return str(
        getattr(
            candidate,
            "candidate_id",
            getattr(
                candidate, "action_id", getattr(getattr(candidate, "action", None), "action_id", "")
            ),
        )
    )


def _replace_mode_risk(estimate, value: float):
    """Return a deterministic synthetic finite-sample result for ordering tests."""
    per_time = (float(value),) + (0.0,) * HORIZON
    return replace(
        estimate,
        per_time_probability=per_time,
        per_time_mc_standard_error=(0.0,) * (HORIZON + 1),
        peak_risk_time_index=0,
        estimated_time_union_bound=float(value),
    )


def _patch_risks(monkeypatch: pytest.MonkeyPatch, values: dict[tuple[str, str], float]) -> None:
    """Replace only the estimator result while preserving its input/provenance checks."""
    original = arb_module.estimate_trajectory_mode_risk

    def fake(action, mode, config=None):
        estimate = original(action, mode, config)
        return _replace_mode_risk(estimate, values[(action.action_id, mode.mode_id)])

    monkeypatch.setattr(arb_module, "estimate_trajectory_mode_risk", fake)


def test_discrete_cvar_splits_boundary_mass_exactly() -> None:
    """The finite upper tail takes only the required part of a boundary atom."""
    expected, var, cvar, worst = discrete_tail_metrics([0.0, 0.2, 1.0], [0.85, 0.10, 0.05], 0.90)
    assert expected == pytest.approx(0.07)
    assert var == pytest.approx(0.2)
    assert cvar == pytest.approx(0.6)
    assert worst == pytest.approx(1.0)


def test_cvar_is_never_below_expected_for_bounded_losses() -> None:
    """Exact upper-tail CVaR dominates each finite-distribution expectation."""
    for losses, probabilities in (
        ((0.0, 0.2, 1.0), (0.85, 0.10, 0.05)),
        ((0.1, 0.1), (0.25, 0.75)),
    ):
        expected, _, cvar, _ = discrete_tail_metrics(losses, probabilities, 0.90)
        assert cvar + 1.0e-12 >= expected


def test_low_probability_catastrophe_is_caught_by_cvar(monkeypatch: pytest.MonkeyPatch) -> None:
    """CVaR can reverse an expectation-favored low-probability catastrophic mode."""
    forecast = _forecast(
        1,
        (
            _mode("catastrophe", (5.0, 3.0), probability=0.05),
            _mode("clear", (8.0, 3.0), probability=0.95),
        ),
    )
    _patch_risks(
        monkeypatch,
        {
            ("catastrophic", "catastrophe"): 1.0,
            ("catastrophic", "clear"): 0.0,
            ("safe", "catastrophe"): 0.1,
            ("safe", "clear"): 0.1,
        },
    )
    result = _arbitrate(
        [_action("catastrophic", (0.5, 0.0)), _action("safe", (0.5, 0.0))],
        forecast,
        arbitration_config=MultimodalArbitrationConfig(cvar_alpha=0.90),
        route_progress={"catastrophic": 0.0, "safe": 0.0},
    )
    catastrophic = next(item for item in result.evaluations if item.candidate_id == "catastrophic")
    safe = next(item for item in result.evaluations if item.candidate_id == "safe")
    assert catastrophic.raw_dynamic_risk < safe.raw_dynamic_risk
    assert catastrophic.cvar_mode_risk > safe.cvar_mode_risk
    assert result.selected_candidate_id == "safe"


def test_probability_one_mode_matches_direct_9813_estimate() -> None:
    """The adapter preserves exact t=0 belief and future-only mode marginals."""
    action = _action("parity", (0.5, 0.0))
    forecast, belief = _join_belief(_forecast(3, (_mode("only", (5.0, 2.0)),)))
    mode = forecast.forecasts[3].modes[0]
    track = belief.tracks[3]
    points = np.concatenate((track.mean_state[None, :2], mode.mean), axis=0)
    covariances = np.concatenate((track.covariance[None, :2, :2], mode.covariance), axis=0)
    risk_input = TrajectoryModeRiskInput(
        actor_id=3,
        mode_id="only",
        actor_radius_m=float(track.mean_state[4]),
        time_offsets_s=np.arange(HORIZON + 1) * DT_S,
        mean_positions=points,
        covariances=covariances,
    )
    direct = estimate_trajectory_mode_risk(action, risk_input, _risk_config())
    result = _arbitrate([action], forecast, belief=belief)
    summary = result.evaluations[0].track_risks[0]
    assert summary.mode_risks == pytest.approx((direct.estimated_time_union_bound,))
    assert summary.mode_per_time_risks[0] == pytest.approx(direct.per_time_probability)
    assert summary.mode_per_time_standard_errors[0] == pytest.approx(
        direct.per_time_mc_standard_error
    )


def test_nonexistence_atom_is_explicit_and_confidence_is_separate() -> None:
    """Existence weighting adds a zero-loss atom; confidence never discounts risk."""
    action = _action("existence", (0.5, 0.0))
    forecast = _forecast(
        4,
        (
            _mode("contact", (0.2, 0.0), probability=0.2),
            _mode("clear", (5.0, 3.0), probability=0.8),
        ),
        existence_probability=0.5,
    )
    result = _arbitrate([action], forecast)
    track = result.evaluations[0].track_risks[0]
    assert track.expected_mode_risk == pytest.approx(0.2)
    assert track.existence_probability == pytest.approx(0.5)
    assert track.existence_probability_calibrated is True
    assert track.existence_probability_semantics == "synthetic_fixture_weight"
    assert track.existence_weighted_expected_risk == pytest.approx(0.1)
    assert result.evaluations[0].raw_dynamic_risk == pytest.approx(0.1)
    assert track.existence_weighted_cvar_risk >= track.existence_weighted_expected_risk


def test_lower_source_confidence_does_not_multiply_raw_risk_down() -> None:
    """Identical marginals produce identical risk under different source confidence."""
    action = _action("confidence", (0.5, 0.0))
    high = _forecast(5, (_mode("only", (2.0, 0.0)),), source_confidence=1.0)
    low = _forecast(5, (_mode("only", (2.0, 0.0)),), source_confidence=0.1)
    high_eval = _arbitrate([action], high).evaluations[0]
    low_eval = _arbitrate([action], low).evaluations[0]
    assert low_eval.raw_dynamic_risk == high_eval.raw_dynamic_risk
    assert (
        low_eval.track_risks[0].mode_per_time_risks == high_eval.track_risks[0].mode_per_time_risks
    )


def test_larger_covariance_worsens_separate_clearance_envelope() -> None:
    """Covariance cannot improve the conservative geometric decision feature."""
    action = _action("uncertainty")
    base = _forecast(6, (_mode("only", (1.0, 0.0), covariance=0.001),))
    uncertain = _forecast(6, (_mode("only", (1.0, 0.0), covariance=0.25),))
    base_eval = _arbitrate([action], base).evaluations[0]
    uncertain_eval = _arbitrate([action], uncertain).evaluations[0]
    assert uncertain_eval.robust_min_clearance_m < base_eval.robust_min_clearance_m
    assert uncertain_eval.uncertainty_radius_m > base_eval.uncertainty_radius_m
    assert uncertain_eval.uncertainty_margin >= base_eval.uncertainty_margin
    assert uncertain_eval.decision_key >= base_eval.decision_key


def test_occluded_stale_track_covariance_cannot_improve_ordering() -> None:
    """A coasting, low-confidence forecast with inflated covariance ranks no safer."""
    action = _action("stale")
    fresh = _forecast(7, (_mode("only", (1.5, 0.0), covariance=0.001),), age_steps=2)
    stale = _forecast(
        7, (_mode("only", (1.5, 0.0), covariance=0.2),), age_steps=2, source_confidence=0.2
    )
    fresh_joined, fresh_belief = _join_belief(fresh)
    stale_joined, stale_belief = _join_belief(stale, stale_track_ids=frozenset({7}))
    fresh_eval = _arbitrate([action], fresh_joined, belief=fresh_belief).evaluations[0]
    stale_eval = _arbitrate([action], stale_joined, belief=stale_belief).evaluations[0]
    assert stale_belief.tracks[7].missed_steps > fresh_belief.tracks[7].missed_steps
    assert stale_eval.robust_min_clearance_m <= fresh_eval.robust_min_clearance_m
    assert stale_eval.decision_key >= fresh_eval.decision_key


def test_cross_track_risk_uses_conservative_sum_without_independence() -> None:
    """The aggregate is a clipped sum of per-track estimates, never a product."""
    forecast = MultimodalPrediction(
        forecasts={
            8: PedestrianForecast(
                8,
                (_mode("only", (0.5, 0.6)),),
                metadata={"source": "force_residual_intent_predictor"},
            ),
            9: PedestrianForecast(
                9,
                (_mode("only", (0.5, -0.6)),),
                metadata={"source": "force_residual_intent_predictor"},
            ),
        },
        prediction_horizon=HORIZON * DT_S,
        prediction_dt=DT_S,
        timestamp=0.0,
        metadata={"step": 0},
    )
    action = _action("two-tracks")
    result = _arbitrate([action], forecast)
    values = [item.existence_weighted_expected_risk for item in result.evaluations[0].track_risks]
    assert result.evaluations[0].raw_dynamic_risk == pytest.approx(min(1.0, sum(values)))
    assert result.evaluations[0].raw_dynamic_risk >= max(values)
    assert result.evaluations[0].max_track_risk >= max(values)
    assert 0.0 <= result.evaluations[0].max_track_risk <= 1.0


def test_max_track_risk_stays_separate_from_cross_track_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worst individual conservative track risk is separately observable."""
    forecast = MultimodalPrediction(
        forecasts={
            8: PedestrianForecast(
                8,
                (_mode("lower", (8.0, 5.0)),),
                metadata={"source": "force_residual_intent_predictor"},
            ),
            9: PedestrianForecast(
                9,
                (_mode("higher", (8.0, -5.0)),),
                metadata={"source": "force_residual_intent_predictor"},
            ),
        },
        prediction_horizon=HORIZON * DT_S,
        prediction_dt=DT_S,
        timestamp=0.0,
        metadata={"step": 0},
    )
    _patch_risks(monkeypatch, {("two-track-risk", "lower"): 0.2, ("two-track-risk", "higher"): 0.4})

    result = _arbitrate([_action("two-track-risk")], forecast)

    evaluation = result.evaluations[0]
    individual = [item.existence_weighted_cvar_risk for item in evaluation.track_risks]
    assert evaluation.max_track_risk == pytest.approx(max(individual))
    assert evaluation.conservative_dynamic_risk == pytest.approx(min(1.0, sum(individual)))


def test_critical_track_mode_and_time_are_exposed() -> None:
    """Diagnostics identify the highest estimated mode and its peak grid time."""
    action = _action("critical")
    forecast = _forecast(
        10,
        (
            _mode("clear", (5.0, 5.0), probability=0.8),
            _mode("contact", (0.4, 0.0), probability=0.2),
        ),
    )
    evaluation = _arbitrate([action], forecast).evaluations[0]
    assert evaluation.critical_track_id == 10
    assert evaluation.critical_mode_id == "contact"
    assert evaluation.critical_step is not None


def test_hard_invalid_candidate_loses_to_clear_candidate_despite_progress() -> None:
    """A canonical per-mode collision gate dominates route progress."""
    forecast = _forecast(11, (_mode("only", (1.0, 0.0)),))
    result = _arbitrate(
        [_action("bad", (1.0, 0.0)), _action("clear", (-0.2, 0.0))],
        forecast,
        route_progress={"bad": 100.0, "clear": 0.0},
    )
    assert result.selected_candidate_id == "clear"
    assert (
        next(item for item in result.evaluations if item.candidate_id == "bad").hard_feasible
        is False
    )


def test_every_retained_mode_is_hard_checked_even_at_low_probability() -> None:
    """A low-weight retained mean path cannot bypass feasibility verification."""
    forecast = _forecast(
        12,
        (
            _mode("rare_collision", (1.0, 0.0), probability=0.001),
            _mode("likely_clear", (8.0, 4.0), probability=0.999),
        ),
    )
    result = _arbitrate([_action("candidate", (1.0, 0.0))], forecast)
    assert result.status == "all_hard_invalid"
    assert "rare_collision" in (result.evaluations[0].hard_gate.ineligibility_reason or "")


def test_actuator_invalid_candidate_loses_regardless_of_progress() -> None:
    """Canonical actuator rate checks remain a hard lexicographic class."""
    bad_positions = np.zeros((HORIZON + 1, 2), dtype=float)
    bad_positions[1:, 0] = 1.0
    bad = _FixtureCandidateAction("actuator_bad", bad_positions)
    safe = _action("actuator_safe", (0.5, 0.0))
    tight = ActuatorLimitsConfig(
        max_accel_mps2=0.1,
        max_decel_mps2=0.1,
        max_yaw_rate_radps=1.0,
        max_steering_rate_radps=1.0,
        command_latency_s=0.0,
        brake_latency_s=0.0,
    )
    result = _arbitrate(
        [safe, bad],
        MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0}),
        route_progress={"actuator_bad": 10.0, "actuator_safe": 0.0},
        actuator_config=tight,
    )
    assert result.selected_candidate_id == "actuator_safe"
    assert not result.evaluations[-1].hard_feasible


def test_risk_limit_cannot_be_offset_by_route_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw-risk threshold rejects a faster candidate before progress scoring."""
    forecast = _forecast(13, (_mode("only", (5.0, 3.0)),))
    _patch_risks(monkeypatch, {("risky", "only"): 0.1, ("safe", "only"): 0.0})
    result = _arbitrate(
        [_action("risky", (0.0, 0.0)), _action("safe", (-0.5, 0.0))],
        forecast,
        arbitration_config=MultimodalArbitrationConfig(raw_risk_limit=0.0),
        route_progress={"risky": 100.0, "safe": 0.0},
    )
    assert result.selected_candidate_id == "safe"
    assert not next(
        item for item in result.evaluations if item.candidate_id == "risky"
    ).risk_limit_ok


def test_progress_wins_within_equal_safety_bucket() -> None:
    """Route progress breaks ties after hard feasibility and risk classes."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    result = _arbitrate(
        [_action("slow", (0.2, 0.0)), _action("fast", (0.8, 0.0))],
        empty,
        route_progress={"slow": 0.2, "fast": 0.8},
    )
    assert result.selected_candidate_id == "fast"


def test_yield_beats_stop_on_liveness_after_equal_safety() -> None:
    """A moving option can beat stopping only after safety fields tie."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    result = _arbitrate(
        [_action("stop"), _action("yield", (0.1, 0.0))],
        empty,
        route_progress={"stop": 0.0, "yield": 0.0},
        stopped_duration_s=10.0,
    )
    assert result.selected_candidate_id == "yield"


def test_comfort_breaks_otherwise_equivalent_tie() -> None:
    """Comfort is scored only after risk and progress match."""
    smooth = _action("smooth", (0.5, 0.0))
    rough_points = np.zeros((HORIZON + 1, 2), dtype=float)
    rough_points[:, 0] = np.linspace(0.0, 0.5, HORIZON + 1)
    rough_points[2::2, 1] = 0.3
    rough = _FixtureCandidateAction("rough", rough_points)
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    generous = ActuatorLimitsConfig(
        max_accel_mps2=100.0,
        max_decel_mps2=100.0,
        max_yaw_rate_radps=100.0,
        max_steering_rate_radps=100.0,
        command_latency_s=0.0,
        brake_latency_s=0.0,
    )
    result = _arbitrate(
        [rough, smooth],
        empty,
        route_progress={"rough": 0.5, "smooth": 0.5},
        actuator_config=generous,
    )
    assert result.selected_candidate_id == "smooth"


def test_supplied_switch_cost_breaks_last_tie() -> None:
    """Switch costs are caller-supplied diagnostics and introduce no state."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    result = _arbitrate(
        [_action("left", (0.5, 0.0)), _action("right", (0.5, 0.0))],
        empty,
        route_progress={"left": 0.5, "right": 0.5},
        switch_costs={"left": 4.0, "right": 0.0},
    )
    assert result.selected_candidate_id == "right"


def test_base_evaluation_forces_zero_switch_cost_and_normalizes_invalid_records() -> None:
    """The base seam hides metadata costs and keeps retained failures finite."""

    class _MetadataCostAction(_FixtureCandidateAction):
        metadata = {**_FixtureCandidateAction.metadata, "switch_cost": 99.0}

    first = _action("first")
    second = _MetadataCostAction("second", _action("second").waypoints)
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    base = _evaluate_base([first, second], empty)

    assert base.status == "selected"
    assert all(item.switch_cost == 0.0 for item in base.evaluations)

    capped = _evaluate_base(
        [first, second],
        empty,
        arbitration_config=MultimodalArbitrationConfig(max_candidates=1),
    )
    assert capped.status == "evaluation_error"
    assert capped.evaluations
    assert all(item.switch_cost == 0.0 for item in capped.evaluations)
    assert all(not item.eligible for item in capped.evaluations)
    combined = _arbitrate(
        [first, second], empty, arbitration_config=MultimodalArbitrationConfig(max_candidates=1)
    )
    original_keys = {item.candidate_id: item.decision_key for item in combined.evaluations}
    assert all(item.decision_key == original_keys[item.candidate_id] for item in capped.evaluations)
    reordered = arb_module.reorder_multimodal_trajectories(
        capped,
        [item.candidate_id for item in capped.evaluations],
        {item.candidate_id: 3.0 for item in capped.evaluations},
        switch_cost_scale=2.0,
    )
    assert reordered.status == "evaluation_error"
    assert reordered.selected_candidate_id is None
    assert all(
        item.decision_key == original_keys[item.candidate_id] for item in reordered.evaluations
    )
    assert all(item.switch_cost == 3.0 for item in reordered.evaluations)


def test_evaluate_once_then_reorder_does_not_recompute_risk_or_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-evaluation seam is pure and does not revisit expensive checks."""
    calls: list[str] = []
    original_risk = arb_module.estimate_trajectory_mode_risk
    original_verify = arb_module.verify_trajectory

    def counted_risk(*args: object, **kwargs: object):
        calls.append("risk")
        return original_risk(*args, **kwargs)

    def counted_verify(*args: object, **kwargs: object):
        calls.append("verify")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(arb_module, "estimate_trajectory_mode_risk", counted_risk)
    monkeypatch.setattr(arb_module, "verify_trajectory", counted_verify)
    base = _evaluate_base([_action("a"), _action("b")], _forecast(99, (_mode("only", (8.0, 5.0)),)))
    evaluation_calls = len(calls)
    assert evaluation_calls > 0

    monkeypatch.setattr(
        arb_module, "estimate_trajectory_mode_risk", lambda *_a, **_k: pytest.fail("risk rerun")
    )
    monkeypatch.setattr(
        arb_module, "verify_trajectory", lambda *_a, **_k: pytest.fail("verify rerun")
    )
    monkeypatch.setattr(
        arb_module, "_coerce_forecast", lambda *_a, **_k: pytest.fail("forecast rerun")
    )
    final = arb_module.reorder_multimodal_trajectories(
        base,
        ["b", "a"],
        {item.candidate_id: float(index) for index, item in enumerate(base.evaluations)},
        switch_cost_scale=1.0,
    )

    assert len(calls) == evaluation_calls
    assert final.candidate_count == 2


def test_reorder_filters_exact_ids_preserves_base_identity_and_inputs() -> None:
    """Filtering is pure and retains the original portfolio identity."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    base = _evaluate_base([_action("a"), _action("b")], empty)
    original_evaluations = base.evaluations
    original_order = base.ordered_candidate_ids
    final = arb_module.reorder_multimodal_trajectories(
        base,
        ["b"],
        {"a": 4.0, "b": 0.5},
        switch_cost_scale=2.0,
    )

    assert tuple(item.candidate_id for item in final.evaluations) == ("b",)
    assert final.ordered_candidate_ids == ("b",)
    assert final.candidate_count == 1
    assert final.candidate_set_id == base.candidate_set_id
    assert final.evaluations[0].switch_cost == 0.5
    assert final.evaluations[0].decision_key[-2] == pytest.approx(0.25)
    assert base.evaluations == original_evaluations
    assert base.ordered_candidate_ids == original_order
    assert base.evaluations[1].switch_cost == 0.0


def test_reorder_rejects_duplicate_unknown_and_incomplete_cost_inputs() -> None:
    """Allowed IDs and costs must form strict, known, complete contracts."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    base = _evaluate_base([_action("a"), _action("b")], empty)
    complete = {"a": 0.0, "b": 0.0}

    with pytest.raises(ValueError, match="unique"):
        arb_module.reorder_multimodal_trajectories(
            base, ["a", "a"], complete, switch_cost_scale=1.0
        )
    with pytest.raises(ValueError, match="unknown IDs"):
        arb_module.reorder_multimodal_trajectories(
            base, ["missing"], complete, switch_cost_scale=1.0
        )
    with pytest.raises(ValueError, match="exactly base evaluation IDs"):
        arb_module.reorder_multimodal_trajectories(base, ["a"], {"a": 0.0}, switch_cost_scale=1.0)
    with pytest.raises(ValueError, match="exactly base evaluation IDs"):
        arb_module.reorder_multimodal_trajectories(
            base,
            ["a"],
            {"a": 0.0, "b": 0.0, "extra": 0.0},
            switch_cost_scale=1.0,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        arb_module.reorder_multimodal_trajectories(
            base,
            ["a"],
            {"a": float("nan"), "b": 0.0},
            switch_cost_scale=1.0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        arb_module.reorder_multimodal_trajectories(base, ["a"], complete, switch_cost_scale=0.0)


def test_reorder_rejects_switch_cost_scaling_overflow() -> None:
    """Finite inputs must not create an infinite tie-break decision value."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    base = _evaluate_base([_action("route")], empty)

    with pytest.raises(ValueError, match="scaled switch cost must be finite"):
        arb_module.reorder_multimodal_trajectories(
            base,
            ["route"],
            {"route": 1e308},
            switch_cost_scale=1e-308,
        )


def test_reorder_preserves_fail_closed_non_selected_status() -> None:
    """Reordering cannot upgrade an invalid base result into a selection."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    base = _evaluate_base([_action("route")], empty, route_progress={})
    assert base.status == "invalid_route_context"
    final = arb_module.reorder_multimodal_trajectories(
        base, [], {"route": 0.0}, switch_cost_scale=1.0
    )
    assert final.status == base.status
    assert final.selected_candidate_id is None
    assert final.candidate_count == 0


def test_reorder_keeps_safety_prefix_ahead_of_switch_cost() -> None:
    """A lower switch cost cannot outrank an earlier safety decision field."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    base = _evaluate_base([_action("safe"), _action("risky")], empty)
    safe = next(item for item in base.evaluations if item.candidate_id == "safe")
    risky = next(item for item in base.evaluations if item.candidate_id == "risky")
    safe = replace(
        safe,
        risk_bucket=0,
        decision_key=(0, 0, 0, 0, *safe.decision_key[4:12], 0.0, "safe"),
    )
    risky = replace(
        risky,
        risk_bucket=3,
        decision_key=(0, 0, 0, 3, *risky.decision_key[4:12], 0.0, "risky"),
    )
    synthetic = replace(
        base,
        selected_candidate_id="safe",
        ordered_candidate_ids=("safe", "risky"),
        evaluations=(safe, risky),
        status="selected",
    )
    final = arb_module.reorder_multimodal_trajectories(
        synthetic,
        ["safe", "risky"],
        {"safe": 100.0, "risky": 0.0},
        switch_cost_scale=1.0,
    )
    assert final.selected_candidate_id == "safe"
    assert final.ordered_candidate_ids == ("safe", "risky")


def test_candidate_input_permutation_keeps_stable_order() -> None:
    """Stable candidate IDs make ordering independent of sequence order."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    first = _arbitrate([_action("b", (0.5, 0.0)), _action("a", (0.5, 0.0))], empty)
    second = _arbitrate([_action("a", (0.5, 0.0)), _action("b", (0.5, 0.0))], empty)
    assert first.ordered_candidate_ids == second.ordered_candidate_ids == ("a", "b")


def test_mode_and_track_permutation_keeps_risks_stable() -> None:
    """Canonical track and mode ID ordering preserves risk under input permutation."""
    modes = (_mode("z", (5.0, 3.0), probability=0.4), _mode("a", (5.0, 4.0), probability=0.6))
    first = _forecast(14, modes)
    second = _forecast(14, tuple(reversed(modes)))
    left = _arbitrate([_action("same", (0.5, 0.0))], first).evaluations[0]
    right = _arbitrate([_action("same", (0.5, 0.0))], second).evaluations[0]
    assert left.raw_dynamic_risk == right.raw_dynamic_risk
    assert left.cvar_mode_risk == right.cvar_mode_risk


def test_reused_numeric_id_with_new_reset_epoch_fails_identity_join() -> None:
    """Lifecycle tokens prevent joining a stale forecast to a reacquired ID."""
    forecast, belief = _join_belief(_forecast(15, (_mode("only", (5.0, 3.0)),)))
    record = forecast.forecasts[15]
    stale_metadata = dict(record.metadata)
    stale_metadata["reset_epoch"] = 1
    stale = replace(record, metadata=stale_metadata)
    invalid = replace(forecast, forecasts={15: stale})
    result = _arbitrate([_action("identity")], invalid, belief=belief)
    assert result.status == "invalid_forecast"
    assert result.selected_candidate_id is None


def test_curved_future_mode_is_scored_without_cv_surrogate() -> None:
    """A non-constant-velocity forecast uses each supplied future mean directly."""
    curved = _mode("curved", (5.0, 3.0)).mean.copy()
    curved[4:, 1] += np.linspace(0.0, 0.8, curved.shape[0] - 4)
    mode = TrajectoryMode(
        "curved", 1.0, curved, covariance=np.zeros((HORIZON, 2, 2), dtype=np.float32)
    )
    result = _arbitrate([_action("curved", (0.5, 0.0))], _forecast(16, (mode,)))
    assert result.status == "selected"
    assert len(result.evaluations[0].track_risks[0].mode_per_time_risks[0]) == HORIZON + 1


def test_between_grid_contact_is_not_promoted_to_swept_probability() -> None:
    """A crossing between sample times stays outside the grid estimator claim."""
    means = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (HORIZON, 1))
    means[0] = (0.0, -1.0)
    mode = TrajectoryMode(
        "between", 1.0, means, covariance=np.zeros((HORIZON, 2, 2), dtype=np.float32)
    )
    result = _arbitrate([_action("stationary")], _forecast(17, (mode,)))
    risk = result.evaluations[0].track_risks[0]
    assert risk.mode_risks == (0.0,)
    assert "not a calibrated probability" in result.claim_boundary
    assert "not a calibrated probability" in ARBITRATION_CLAIM_BOUNDARY


def test_empty_forecast_is_valid_no_pedestrian_input() -> None:
    """A truly empty, identity-safe belief and forecast remain selectable."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    result = _arbitrate([_action("empty", (0.5, 0.0))], empty)
    assert result.status == "selected"
    assert result.evaluations[0].raw_dynamic_risk == 0.0
    assert result.evaluations[0].robust_min_clearance_m == float("inf")


def test_invalid_forecast_grid_returns_explicit_no_selection() -> None:
    """A malformed or mismatched forecast is never silently adapted."""
    invalid = {"forecasts": {}, "prediction_horizon": HORIZON * DT_S, "prediction_dt": 0.2}
    result = arbitrate_multimodal_trajectories(
        [_action("invalid")], invalid, belief=_empty_belief(), risk_config=_risk_config()
    )
    assert result.status == "invalid_forecast"
    assert result.selected_candidate_id is None
    assert result.no_selection_reason

    base = arb_module.evaluate_multimodal_trajectories_base(
        [_action("invalid")], invalid, belief=_empty_belief(), risk_config=_risk_config()
    )
    reordered = arb_module.reorder_multimodal_trajectories(base, [], {}, switch_cost_scale=1.0)
    assert reordered.status == "invalid_forecast"
    assert reordered.selected_candidate_id is None


def test_missing_mode_uncertainty_fails_closed() -> None:
    """A mode without covariance or explicit standard deviation is invalid."""
    mode = TrajectoryMode("unknown", 1.0, np.ones((HORIZON, 2), dtype=np.float32))
    result = _arbitrate([_action("missing-cov")], _forecast(18, (mode,)))
    assert result.status == "invalid_forecast"
    assert "covariance or standard deviation" in (result.no_selection_reason or "")


def test_no_candidates_all_hard_invalid_and_all_above_limit_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-selection status states whether inputs, hard gates, or risk limits failed."""
    safe_forecast = _forecast(19, (_mode("only", (8.0, 5.0)),))
    empty = _arbitrate([], safe_forecast)
    assert empty.status == "no_candidates"
    assert (
        empty.max_total_contact_samples == MultimodalArbitrationConfig().max_total_contact_samples
    )
    collision = _arbitrate(
        [_action("collision", (1.0, 0.0))], _forecast(20, (_mode("only", (1.0, 0.0)),))
    )
    assert collision.status == "all_hard_invalid"
    _patch_risks(monkeypatch, {("limited", "only"): 0.1})
    limited = _arbitrate(
        [_action("limited")],
        safe_forecast,
        arbitration_config=MultimodalArbitrationConfig(raw_risk_limit=0.0),
    )
    assert limited.status == "all_above_risk_limit"


def test_caps_fail_before_dropping_candidates_or_modes() -> None:
    """Candidate, track, and sample-work caps are explicit and fail closed."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    candidate_cap = _arbitrate(
        [_action("a"), _action("b")],
        empty,
        arbitration_config=MultimodalArbitrationConfig(max_candidates=1),
    )
    assert candidate_cap.status == "evaluation_error"
    assert candidate_cap.candidate_count == 2
    assert (
        candidate_cap.max_total_contact_samples
        == MultimodalArbitrationConfig().max_total_contact_samples
    )
    work_cap = _arbitrate(
        [_action("work")],
        _forecast(21, (_mode("only", (5.0, 5.0)),)),
        arbitration_config=MultimodalArbitrationConfig(max_total_contact_samples=1),
    )
    assert work_cap.status == "evaluation_error"
    assert "max_total_contact_samples" in (work_cap.no_selection_reason or "")
    with pytest.raises(ValueError, match="max_total_contact_samples"):
        MultimodalArbitrationConfig(max_total_contact_samples=0)
    with pytest.raises(ValueError, match="conservative_clearance_m"):
        MultimodalArbitrationConfig(conservative_clearance_m=-0.01)


def test_candidate_cap_diagnostics_are_permutation_invariant() -> None:
    """An over-cap portfolio keeps stable diagnostic ordering regardless of input order."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    config = MultimodalArbitrationConfig(max_candidates=1)
    forward = _arbitrate([_action("z"), _action("a")], empty, arbitration_config=config)
    reversed_result = _arbitrate([_action("a"), _action("z")], empty, arbitration_config=config)

    assert forward.status == reversed_result.status == "evaluation_error"
    assert forward.ordered_candidate_ids == reversed_result.ordered_candidate_ids == ("a", "z")
    assert tuple(item.candidate_id for item in forward.evaluations) == ("a", "z")


def test_generation_only_candidates_require_producer_owned_type() -> None:
    """Duck-typed candidates cannot claim the #8057 generated-candidate contract."""

    class _UntrustedGeneratedAction(_FixtureCandidateAction):
        metadata = {**_FixtureCandidateAction.metadata, "generation_mode": "generation_only"}

    fixture_action = _action("untrusted-action")
    untrusted_action = _UntrustedGeneratedAction(
        fixture_action.action_id, fixture_action.waypoints, fixture_action.representation
    )
    untrusted_duck = SimpleNamespace(
        candidate_id="untrusted-duck",
        action=_action("untrusted-duck"),
        maneuver="route_follow",
        states=np.zeros((HORIZON + 1, 5), dtype=float),
        generation_rank=0,
        metadata={
            "generation_mode": "generation_only",
            "timestamp_s": 0.0,
            "dt_s": DT_S,
            "horizon_steps": HORIZON,
            "robot_radius_m": 0.2,
            "static_feasible": True,
        },
    )
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate([untrusted_duck, untrusted_action], empty)

    assert result.status == "evaluation_error"
    assert result.selected_candidate_id is None
    assert all("producer-owned" in item.hard_reasons[0] for item in result.evaluations)


def test_static_gate_runs_before_risk_and_survives_risk_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Risk errors retain the already-computed hard-gate result and reasons."""

    class _BlockedCandidateAction(_FixtureCandidateAction):
        metadata = {**_FixtureCandidateAction.metadata, "static_feasible": False}

    base = _action("static-blocked")
    candidate = _BlockedCandidateAction(base.action_id, base.waypoints, base.representation)
    events: list[str] = []
    original_gate = arb_module._multimodal_hard_gate

    def observed_gate(*args: object, **kwargs: object):
        result = original_gate(*args, **kwargs)
        events.append("hard_gate")
        return result

    def failed_risk(*args: object, **kwargs: object):
        events.append("risk")
        raise ValueError("synthetic risk-estimator failure")

    monkeypatch.setattr(arb_module, "_multimodal_hard_gate", observed_gate)
    monkeypatch.setattr(arb_module, "_candidate_track_summaries", failed_risk)
    forecast = _forecast(30, (_mode("clear", (8.0, 5.0)),))

    result = _arbitrate([candidate], forecast)

    evaluation = result.evaluations[0]
    assert events == ["hard_gate", "risk"]
    assert result.status == "evaluation_error"
    assert evaluation.hard_feasible is False
    assert evaluation.hard_gate is not None
    assert any("static feasibility" in reason for reason in evaluation.hard_reasons)
    assert any("synthetic risk-estimator failure" in reason for reason in evaluation.hard_reasons)


def test_actuator_gate_receives_geometry_only_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covariance support remains in the risk envelope, not actuator geometry input."""
    candidate = _action("geometry-clearance", (0.5, 0.0))
    forecast = _forecast(31, (_mode("uncertain", (8.0, 5.0), covariance=0.25),))
    observed: list[float] = []
    original_actuator = arb_module.evaluate_actuator_feasibility

    def record_clearance(**kwargs: object):
        observed.append(float(kwargs["hazard_clearance_m"]))
        return original_actuator(**kwargs)

    monkeypatch.setattr(arb_module, "evaluate_actuator_feasibility", record_clearance)
    result = _arbitrate([candidate], forecast)
    robot_positions = candidate.as_array(horizon_steps=HORIZON)
    expected_geometry = float(
        np.min(np.linalg.norm(robot_positions - np.asarray((8.0, 5.0)), axis=1)) - 0.4
    )

    assert observed == pytest.approx([expected_geometry])
    assert result.evaluations[0].robust_min_clearance_m < observed[0]


def test_invalid_route_context_is_not_replaced_by_a_default() -> None:
    """Missing route data has its own truthful no-selection status."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    result = _arbitrate([_action("route")], empty, route_progress={})
    assert result.status == "invalid_route_context"
    assert result.selected_candidate_id is None


def test_route_progress_sequence_stays_aligned_after_candidate_coercion_failure() -> None:
    """A malformed earlier candidate cannot shift progress onto a later candidate."""
    malformed = SimpleNamespace(candidate_id="malformed-first", action=object())
    valid = _action("valid-second")
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate([malformed, valid], empty, route_progress=[100.0, 0.25])

    selected = next(item for item in result.evaluations if item.candidate_id == "valid-second")
    assert result.candidate_count == 2
    assert result.selected_candidate_id == "valid-second"
    assert selected.route_progress_m == pytest.approx(0.25)
    assert any(item.candidate_id == "malformed-first" for item in result.evaluations)


def test_route_progress_sequence_length_mismatch_fails_as_invalid_route_context() -> None:
    """A sequence cannot silently broadcast or truncate route progress."""
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate([_action("route-sequence")], empty, route_progress=[])

    assert result.status == "invalid_route_context"
    assert result.selected_candidate_id is None


def test_candidates_from_different_initial_robot_states_are_retained_but_rejected() -> None:
    """One shared candidate set must describe a single initial robot state."""
    first_action = action_from_constant_velocity(
        "state-a", (0.0, 0.0), (0.2, 0.0), horizon_steps=HORIZON, dt_s=DT_S
    )
    second_action = action_from_constant_velocity(
        "state-b", (0.5, 0.0), (0.2, 0.0), horizon_steps=HORIZON, dt_s=DT_S
    )
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate(
        [
            _FixtureCandidateAction("state-a", first_action.waypoints),
            _FixtureCandidateAction("state-b", second_action.waypoints),
        ],
        empty,
    )

    assert result.status == "evaluation_error"
    assert result.selected_candidate_id is None
    assert {item.candidate_id for item in result.evaluations} == {"state-a", "state-b"}
    assert all(
        any("initial XY" in reason for reason in item.hard_reasons) for item in result.evaluations
    )


def test_nonfinite_candidate_is_retained_as_evaluation_error() -> None:
    """NaN waypoints are diagnosed and cannot become a selected fallback."""
    action = _action("nan")
    action.waypoints[2, 0] = np.nan
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    result = _arbitrate([action], empty)
    assert result.status == "evaluation_error"
    assert not result.evaluations[0].hard_feasible


def test_generated_candidate_radius_must_match_risk_config() -> None:
    """The #8057 footprint metadata must agree with the estimator footprint."""
    candidate = _maneuver_candidate(
        "radius",
        {
            "generation_mode": "generation_only",
            "dt_s": DT_S,
            "horizon_steps": HORIZON,
            "robot_radius_m": 0.4,
            "timestamp_s": 0.0,
            "static_feasible": True,
        },
    )
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})
    result = _arbitrate([candidate], empty)
    assert result.status == "evaluation_error"
    assert any("robot radius" in reason for reason in result.evaluations[0].hard_reasons)


def test_raw_candidate_without_static_evidence_is_retained_and_ineligible() -> None:
    """Legacy actions remain observable but cannot bypass the static hard gate."""
    annotated = _action("raw-source")
    raw = CandidateAction("raw-source", annotated.waypoints)
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate([raw], empty)

    assert result.selected_candidate_id is None
    assert len(result.evaluations) == 1
    assert any("static_collision" in reason for reason in result.evaluations[0].hard_reasons)


def test_malformed_candidate_with_recoverable_id_is_retained() -> None:
    """Candidate coercion errors retain the recoverable identifier in diagnostics."""
    malformed = SimpleNamespace(candidate_id="malformed", action=object())
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate([malformed], empty)

    assert result.status == "evaluation_error"
    assert result.ordered_candidate_ids == ("malformed",)
    assert result.evaluations[0].candidate_id == "malformed"


def test_candidate_cycle_timestamp_must_match_forecast() -> None:
    """A candidate from another planning cycle cannot be evaluated or selected."""
    stale = _maneuver_candidate(
        "stale-cycle",
        {
            "generation_mode": "generation_only",
            "timestamp_s": 1.0,
            "dt_s": DT_S,
            "horizon_steps": HORIZON,
            "robot_radius_m": 0.2,
            "static_feasible": True,
        },
    )
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate([stale], empty)

    assert result.status == "evaluation_error"
    assert "timestamp" in result.evaluations[0].hard_reasons[0]


def test_generated_candidate_requires_explicit_common_grid_metadata() -> None:
    """A generated path cannot hide its temporal spacing behind matching shape."""
    candidate = _maneuver_candidate(
        "missing-grid",
        {
            "generation_mode": "generation_only",
            "timestamp_s": 0.0,
            "robot_radius_m": 0.2,
            "static_feasible": True,
        },
    )
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate([candidate], empty)

    assert result.status == "evaluation_error"
    assert "horizon_steps metadata" in result.evaluations[0].hard_reasons[0]


def test_producer_owned_generated_candidate_with_contract_is_selectable() -> None:
    """A complete immutable #8057 candidate crosses the consumer contract."""
    candidate = _maneuver_candidate(
        "producer-owned",
        {
            "generation_mode": "generation_only",
            "timestamp_s": 0.0,
            "dt_s": DT_S,
            "horizon_steps": HORIZON,
            "robot_radius_m": 0.2,
            "static_min_clearance_m": 1.0,
            "static_feasible": True,
        },
    )
    empty = MultimodalPrediction({}, HORIZON * DT_S, DT_S, timestamp=0.0, metadata={"step": 0})

    result = _arbitrate([candidate], empty)

    assert result.status == "selected"
    assert result.selected_candidate_id == "producer-owned"


def test_braking_predicate_is_a_hard_decision_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical braking predicates remain hard even when actuator limits are clean."""
    forecast = _forecast(22, (_mode("only", (8.0, 5.0)),))

    def fake_verify(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            decision="fallback_brake",
            violated_predicates=("braking_infeasible",),
        )

    monkeypatch.setattr(arb_module, "verify_trajectory", fake_verify)
    result = _arbitrate([_action("braking")], forecast)

    evaluation = result.evaluations[0]
    assert evaluation.braking_feasible is False
    assert evaluation.hard_gate is not None
    assert any(
        "braking_infeasible" in predicate for predicate in evaluation.hard_gate.violated_predicates
    )


def test_fallback_brake_without_predicates_is_still_hard_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verifier brake decision cannot become eligible when diagnostics are empty."""
    forecast = _forecast(32, (_mode("only", (8.0, 5.0)),))
    monkeypatch.setattr(
        arb_module,
        "verify_trajectory",
        lambda **_: SimpleNamespace(decision="fallback_brake", violated_predicates=()),
    )

    result = _arbitrate([_action("empty-brake-reasons")], forecast)

    evaluation = result.evaluations[0]
    assert result.status == "all_hard_invalid"
    assert evaluation.hard_feasible is False
    assert evaluation.braking_feasible is False
    assert any(
        "fallback_brake without violated predicates" in reason for reason in evaluation.hard_reasons
    )
