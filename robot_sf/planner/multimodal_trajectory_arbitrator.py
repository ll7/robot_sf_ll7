"""Pure multimodal trajectory risk evaluation and deterministic arbitration.

This experimental planner surface consumes identity-safe belief tracks and
explicit multimodal future paths. Its canonical risk values are finite-sample
marginal estimates over discrete grid-time contact events; they are not
continuous-time collision probabilities or safety verdicts. Canonical hard
verifiers remain authoritative.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from numbers import Integral

import numpy as np

from robot_sf.benchmark.actuator_feasibility import (
    VERDICT_ACTUATOR_FEASIBLE,
    ActuatorLimitsConfig,
    evaluate_actuator_feasibility,
    stopping_distance,
)
from robot_sf.benchmark.trajectory_verifier import (
    DECISION_FALLBACK_BRAKE,
    TrajectoryVerifierConfig,
    verify_trajectory,
)
from robot_sf.nav.predictive_types import MultimodalPrediction, PedestrianForecast, TrajectoryMode
from robot_sf.planner.maneuver_candidates import ManeuverCandidate
from robot_sf.planner.risk_aware_trajectory_ranker import HardGateResult
from robot_sf.planner.scenario_belief_adapter import (
    BeliefAwarePlannerInput,
    PlannerTrackBelief,
)
from robot_sf.research.collision_risk import CandidateAction, RiskEstimatorConfig
from robot_sf.research.collision_risk.estimators import estimate_trajectory_mode_risk
from robot_sf.research.collision_risk.schema import (
    TrajectoryModeRiskEstimate,
    TrajectoryModeRiskInput,
)

# ---------------------------------------------------------------------------
# Pure multimodal arbitration (issue #8062)
# ---------------------------------------------------------------------------


ARBITRATION_SCHEMA_VERSION = "multimodal_trajectory_arbitration.v1"
ARBITRATION_CLAIM_BOUNDARY = (
    "finite-sample per-mode marginal contact estimates on discrete grid times; "
    "the clipped time sum is not a calibrated probability, certified bound, "
    "continuous-time swept-contact estimate, or safety verdict; hard gates remain authoritative"
)
SELECTION_STATUSES = frozenset(
    {
        "selected",
        "no_candidates",
        "all_hard_invalid",
        "all_above_risk_limit",
        "invalid_forecast",
        "invalid_route_context",
        "evaluation_error",
        "deadline_exceeded",
    }
)


@dataclass(frozen=True)
class MultimodalArbitrationConfig:
    """Validated, bounded configuration for multimodal candidate arbitration.

    The raw risk values in this module retain the canonical #6567 estimator
    semantics.  ``uncertainty_*`` fields create a separately named decision
    envelope; they never turn a model score into a calibrated probability.
    ``track_aggregation`` is deliberately restricted to the conservative sum
    (union bound) because pedestrian events are not independent by default.
    """

    cvar_alpha: float = 0.90
    raw_risk_limit: float = 1.0
    conservative_risk_limit: float = 1.0
    risk_bucket_edges: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    uncertainty_quantile: float = 0.99
    track_aggregation: str = "union_bound"
    route_progress_scale_m: float = 1.0
    liveness_scale: float = 1.0
    comfort_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
    switch_cost_scale: float = 1.0
    numeric_tolerance: float = 1.0e-9
    max_tracks: int = 32
    max_modes_per_track: int = 16
    max_candidates: int = 64
    max_total_contact_samples: int = 20_000_000
    conservative_clearance_m: float = 0.0
    safe_clearance_m: float = 0.5
    uncertainty_base_radius_m: float = 0.0

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Reject malformed or unbounded arbitration configurations."""
        for name in (
            "cvar_alpha",
            "uncertainty_quantile",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        for name in ("raw_risk_limit", "conservative_risk_limit"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        edges = tuple(float(value) for value in self.risk_bucket_edges)
        if len(edges) < 2 or edges[0] != 0.0:
            raise ValueError("risk_bucket_edges must start at 0 and contain at least two edges")
        if any(not math.isfinite(value) for value in edges):
            raise ValueError("risk_bucket_edges must be finite")
        if any(right <= left for left, right in pairwise(edges)):
            raise ValueError("risk_bucket_edges must be strictly increasing")
        if edges[-1] < 1.0:
            raise ValueError("risk_bucket_edges must cover risk values through 1")
        object.__setattr__(self, "risk_bucket_edges", edges)
        if self.track_aggregation != "union_bound":
            raise ValueError("track_aggregation must be 'union_bound'")
        for name in (
            "route_progress_scale_m",
            "liveness_scale",
            "switch_cost_scale",
            "numeric_tolerance",
            "safe_clearance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not math.isfinite(self.conservative_clearance_m) or self.conservative_clearance_m < 0.0:
            raise ValueError("conservative_clearance_m must be finite and >= 0")
        for name in ("uncertainty_base_radius_m",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        scales = tuple(float(value) for value in self.comfort_scales)
        if len(scales) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("comfort_scales must contain three finite positive values")
        object.__setattr__(self, "comfort_scales", scales)
        for name in (
            "max_tracks",
            "max_modes_per_track",
            "max_candidates",
            "max_total_contact_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class TrackRiskSummary:
    """Conditional and existence-weighted risk diagnostics for one track."""

    track_id: int
    mode_ids: tuple[str, ...]
    mode_probabilities: tuple[float, ...]
    mode_risks: tuple[float, ...]
    mode_per_time_risks: tuple[tuple[float, ...], ...]
    mode_per_time_standard_errors: tuple[tuple[float, ...], ...]
    expected_mode_risk: float
    var_mode_risk: float
    cvar_mode_risk: float
    worst_mode_risk: float
    existence_probability: float
    existence_probability_calibrated: bool | None
    existence_probability_semantics: str
    existence_weighted_expected_risk: float
    existence_weighted_var_risk: float
    existence_weighted_cvar_risk: float
    robust_min_clearance_m: float
    uncertainty_radius_m: float
    uncertainty_load: float
    critical_mode_id: str | None
    critical_step: int | None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Immutable diagnostic record for one multimodal candidate evaluation."""

    candidate_id: str
    maneuver: object
    hard_feasible: bool
    hard_reasons: tuple[str, ...]
    raw_dynamic_risk: float
    conservative_dynamic_risk: float
    max_track_risk: float
    risk_bucket: int
    critical_track_id: int | None
    critical_mode_id: str | None
    critical_step: int | None
    route_progress_m: float
    liveness_cost: float
    comfort_cost: float
    switch_cost: float
    decision_key: tuple[object, ...]
    expected_mode_risk: float = 0.0
    var_mode_risk: float = 0.0
    cvar_mode_risk: float = 0.0
    worst_mode_risk: float = 0.0
    existence_weighted_expected_risk: float = 0.0
    existence_weighted_var_risk: float = 0.0
    existence_weighted_cvar_risk: float = 0.0
    robust_min_clearance_m: float = float("inf")
    uncertainty_radius_m: float = 0.0
    uncertainty_load: float = 0.0
    uncertainty_margin: float = 0.0
    braking_feasible: bool = True
    risk_limit_ok: bool = True
    rejection_reason: str | None = None
    hard_gate: HardGateResult | None = None
    track_risks: tuple[TrackRiskSummary, ...] = ()

    @property
    def eligible(self) -> bool:
        """Return whether this candidate may be selected."""
        return self.hard_feasible and self.risk_limit_ok and self.rejection_reason is None

    @property
    def conservative_risk(self) -> float:
        """Backward-readable alias for the conservative decision score."""
        return self.conservative_dynamic_risk


@dataclass(frozen=True)
class ArbitrationResult:
    """Pure selector output, including diagnostics for every input candidate."""

    status: str
    selected_candidate_id: str | None
    ordered_candidate_ids: tuple[str, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    no_selection_reason: str | None
    schema_version: str = ARBITRATION_SCHEMA_VERSION
    forecast_schema_version: str | None = None
    candidate_set_id: str | None = None
    evaluation_duration_ms: float = 0.0
    candidate_count: int = 0
    track_count: int = 0
    mode_count: int = 0
    diagnostics: tuple[tuple[str, str], ...] = ()
    risk_config_hash: str | None = None
    max_candidates: int | None = None
    max_tracks: int | None = None
    max_modes_per_track: int | None = None
    max_total_contact_samples: int | None = None
    claim_boundary: str = ARBITRATION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate result status and immutable diagnostic shape."""
        if self.status not in SELECTION_STATUSES:
            raise ValueError(f"unknown arbitration status: {self.status!r}")
        if self.status == "selected" and self.selected_candidate_id is None:
            raise ValueError("selected result requires selected_candidate_id")
        if self.status != "selected" and self.selected_candidate_id is not None:
            raise ValueError("non-selected result cannot contain selected_candidate_id")
        if self.evaluation_duration_ms < 0.0 or not math.isfinite(self.evaluation_duration_ms):
            raise ValueError("evaluation_duration_ms must be finite and >= 0")

    @property
    def selection_status(self) -> str:
        """Return the issue-facing status name used by result consumers."""
        return self.status


def _validate_distribution(
    probabilities: Sequence[float], *, tolerance: float = 1.0e-9
) -> tuple[float, ...]:
    """Validate a finite probability distribution without surprising renormalization.

    Returns:
        The accepted probabilities normalized only for representational round-off.
    """
    values = tuple(float(value) for value in probabilities)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be a non-empty finite non-negative sequence")
    total = sum(values)
    if not math.isfinite(total) or abs(total - 1.0) > tolerance:
        raise ValueError(f"probabilities must sum to one within tolerance; got {total}")
    # Normalization is only used to absorb representational round-off already
    # accepted by the tolerance; a materially mis-specified distribution fails.
    return tuple(value / total for value in values)


def discrete_tail_metrics(
    losses: Sequence[float],
    probabilities: Sequence[float],
    alpha: float,
    *,
    stable_ids: Sequence[str] | None = None,
    tolerance: float = 1.0e-9,
) -> tuple[float, float, float, float]:
    """Return expected loss, VaR, exact discrete upper-tail CVaR, and worst loss.

    ``alpha`` is the confidence level, so CVaR averages exactly the worst
    ``1 - alpha`` probability mass.  The boundary atom is split rather than
    included in full, which preserves the finite-distribution definition.
    """
    values = tuple(float(value) for value in losses)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("losses must be a non-empty finite sequence")
    if len(values) != len(probabilities):
        raise ValueError("losses and probabilities must have equal length")
    if stable_ids is None:
        stable_ids = tuple(str(index) for index in range(len(values)))
    if len(stable_ids) != len(values):
        raise ValueError("stable_ids must match losses")
    if not math.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    weights = _validate_distribution(probabilities, tolerance=tolerance)
    expected = float(
        sum(loss * probability for loss, probability in zip(values, weights, strict=True))
    )
    ascending = sorted(
        zip(values, weights, stable_ids, strict=True), key=lambda item: (item[0], item[2])
    )
    cumulative = 0.0
    var = ascending[-1][0]
    for loss, probability, _ in ascending:
        cumulative += probability
        if cumulative + tolerance >= alpha:
            var = loss
            break
    descending = sorted(
        zip(values, weights, stable_ids, strict=True), key=lambda item: (-item[0], item[2])
    )
    remaining = 1.0 - float(alpha)
    tail_loss = 0.0
    for loss, probability, _ in descending:
        taken = min(probability, remaining)
        tail_loss += loss * taken
        remaining -= taken
        if remaining <= tolerance:
            break
    cvar = float(tail_loss / (1.0 - float(alpha)))
    return expected, float(var), cvar, float(max(values))


def discrete_upper_tail_cvar(
    losses: Sequence[float], probabilities: Sequence[float], alpha: float
) -> float:
    """Return only the exact finite-distribution upper-tail CVaR."""
    return discrete_tail_metrics(losses, probabilities, alpha)[2]


@dataclass(frozen=True)
class _CandidateInput:
    candidate_id: str
    action: CandidateAction
    maneuver: object
    metadata: Mapping[str, object]
    generation_rank: int
    states: np.ndarray | None = None


def _candidate_set_identifier(candidates: Sequence[_CandidateInput]) -> str:
    """Return a stable candidate-set identifier from immutable candidate metadata."""
    supplied = {
        str(candidate.metadata["candidate_set_id"])
        for candidate in candidates
        if candidate.metadata.get("candidate_set_id") is not None
    }
    if len(supplied) == 1:
        return next(iter(supplied))
    payload = "\x1f".join(
        f"{candidate.candidate_id}:{candidate.generation_rank}"
        for candidate in sorted(candidates, key=lambda item: item.candidate_id)
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class _ModeInput:
    """One identity-joined forecast mode on the candidate's complete grid."""

    belief_track: PlannerTrackBelief
    forecast_track: PedestrianForecast
    mode: TrajectoryMode
    points: np.ndarray
    covariance: np.ndarray
    uncertainty_radius: np.ndarray
    risk_input: TrajectoryModeRiskInput


@dataclass(frozen=True)
class _EvaluationContext:
    """Immutable shared inputs for one candidate evaluation cycle."""

    forecast: MultimodalPrediction
    mode_inputs: tuple[_ModeInput, ...]
    risk_config: RiskEstimatorConfig
    arbitration_config: MultimodalArbitrationConfig
    route_progress: Mapping[str, float] | Sequence[float] | Callable[[object], float] | None
    current_command: Sequence[float] | None
    stopped_duration_s: float
    switch_costs: Mapping[str, float] | None
    verifier_config: TrajectoryVerifierConfig | None
    actuator_config: ActuatorLimitsConfig | None


@dataclass(frozen=True)
class _CandidateRiskAggregate:
    """Aggregated candidate risk and uncertainty components used for ordering."""

    expected_mode: float
    var_mode: float
    cvar_mode: float
    worst_mode: float
    raw_dynamic: float
    existence_var: float
    existence_cvar: float
    robust_clearance: float
    uncertainty_radius: float
    uncertainty_load: float
    uncertainty_margin: float
    max_track_risk: float
    conservative: float
    risk_limit_ok: bool
    risk_bucket: int
    critical: tuple[float, int, str, int] | None


@dataclass(frozen=True)
class _PreparedCandidates:
    """Validated candidate collection and bounded work estimate."""

    values: tuple[_CandidateInput, ...]
    candidate_set_id: str
    estimated_contact_samples: int
    route_progress: Mapping[str, float] | Callable[[object], float] | None = None
    route_error: bool = False
    invalid_evaluations: tuple[CandidateEvaluation, ...] = ()
    collection_error: str | None = None


def _coerce_candidate_action(action: CandidateAction, *, horizon_steps: int) -> _CandidateInput:
    """Adapt a canonical #6567 action without copying its trajectory semantics.

    Returns:
        A normalized internal candidate record.
    """
    if not isinstance(action.action_id, str) or not action.action_id.strip():
        raise ValueError("candidate action_id must be a non-empty string")
    raw_metadata = getattr(action, "metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("candidate metadata must be a mapping")
    if raw_metadata.get("generation_mode") == "generation_only":
        raise ValueError(
            "generation-only candidates must use the producer-owned ManeuverCandidate type"
        )
    action.as_array(horizon_steps=horizon_steps)
    return _CandidateInput(
        candidate_id=str(action.action_id),
        action=action,
        maneuver="unknown",
        metadata=dict(raw_metadata),
        generation_rank=0,
        states=None,
    )


def _coerce_maneuver_candidate(
    candidate: ManeuverCandidate, *, horizon_steps: int
) -> _CandidateInput:
    """Adapt an immutable producer-owned #8057 candidate.

    Returns:
        A normalized internal candidate record with its immutable state grid.
    """
    action = candidate.action
    metadata = candidate.metadata
    if metadata.get("generation_mode") != "generation_only":
        raise ValueError("ManeuverCandidate is missing its generation_only producer contract")
    action.as_array(horizon_steps=horizon_steps)
    if "robot_radius_m" not in metadata:
        raise ValueError("maneuver candidate is missing its robot_radius_m contract")
    states = np.asarray(candidate.states, dtype=float)
    if (
        states.ndim != 2
        or states.shape[0] != horizon_steps + 1
        or states.shape[1] < 5
        or not np.all(np.isfinite(states))
        or not np.array_equal(states[:, :2], action.as_array(horizon_steps=horizon_steps))
    ):
        raise ValueError("candidate state and action grids do not agree")
    return _CandidateInput(
        candidate_id=candidate.candidate_id,
        action=action,
        maneuver=candidate.maneuver,
        metadata=dict(metadata),
        generation_rank=candidate.generation_rank,
        states=states,
    )


def _coerce_candidate(candidate: object, *, horizon_steps: int) -> _CandidateInput:
    """Adapt a #8057 candidate or canonical #6567 action.

    Returns:
        A normalized internal candidate record.
    """
    if isinstance(candidate, CandidateAction):
        return _coerce_candidate_action(candidate, horizon_steps=horizon_steps)
    if isinstance(candidate, ManeuverCandidate):
        return _coerce_maneuver_candidate(candidate, horizon_steps=horizon_steps)
    raise ValueError("candidate must be CandidateAction or producer-owned #8057 ManeuverCandidate")


def _candidate_id_hint(candidate: object, *, index: int) -> str:
    """Recover a stable diagnostic ID from a malformed candidate when possible.

    Returns:
        A candidate identifier or deterministic positional fallback.
    """
    if isinstance(candidate, CandidateAction):
        value = candidate.action_id
    else:
        value = getattr(candidate, "candidate_id", None)
        if not isinstance(value, str) or not value.strip():
            action = getattr(candidate, "action", None)
            value = getattr(action, "action_id", None)
    if isinstance(value, str) and value.strip():
        return value
    return f"candidate_{index}"


def _invalid_candidate_evaluation(
    candidate_id: str,
    *,
    maneuver: object = "unknown",
    reason: str,
    risk_bucket: int,
    hard_gate: HardGateResult | None = None,
    braking_feasible: bool = False,
) -> CandidateEvaluation:
    """Build a deterministic retained diagnostic for a candidate that cannot run.

    Returns:
        A fail-closed evaluation record retaining the candidate identity.
    """
    diagnostic = f"evaluation_error: {reason}"
    if hard_gate is None:
        invalid_gate = HardGateResult(
            eligible=False,
            verifier_decision="not_evaluated",
            actuator_verdict="not_evaluated",
            violated_predicates=(diagnostic,),
            violated_limits=(),
            ineligibility_reason=diagnostic,
        )
    else:
        predicates = tuple(dict.fromkeys((*hard_gate.violated_predicates, diagnostic)))
        limits = hard_gate.violated_limits
        invalid_gate = HardGateResult(
            eligible=False,
            verifier_decision=hard_gate.verifier_decision,
            actuator_verdict=hard_gate.actuator_verdict,
            violated_predicates=predicates,
            violated_limits=limits,
            ineligibility_reason="; ".join(dict.fromkeys((*predicates, *limits))),
        )
        braking_feasible = not any("brak" in value.lower() for value in (*predicates, *limits))
    hard_reasons = tuple(
        dict.fromkeys((*invalid_gate.violated_predicates, *invalid_gate.violated_limits))
    )
    return CandidateEvaluation(
        candidate_id=candidate_id,
        maneuver=maneuver,
        hard_feasible=False,
        hard_reasons=hard_reasons,
        raw_dynamic_risk=1.0,
        conservative_dynamic_risk=1.0,
        max_track_risk=1.0,
        risk_bucket=risk_bucket,
        critical_track_id=None,
        critical_mode_id=None,
        critical_step=None,
        route_progress_m=0.0,
        liveness_cost=float("inf"),
        comfort_cost=float("inf"),
        switch_cost=float("inf"),
        decision_key=(2, 2, 1, risk_bucket, float("inf"), 1.0, 1.0, candidate_id),
        robust_min_clearance_m=float("-inf"),
        uncertainty_margin=1.0,
        braking_feasible=braking_feasible,
        risk_limit_ok=False,
        rejection_reason=diagnostic,
        hard_gate=invalid_gate,
    )


def _resolve_selection(
    evaluations: Sequence[CandidateEvaluation],
    *,
    route_error: bool,
    evaluation_error: bool,
) -> tuple[str, str | None, str | None]:
    """Resolve final status after all candidate diagnostics have been retained.

    Returns:
        Status, selected candidate ID, and an optional no-selection reason.
    """
    if route_error:
        return (
            "invalid_route_context",
            None,
            "route progress context is invalid or incomplete",
        )
    eligible = [item for item in evaluations if item.eligible]
    hard_feasible = [item for item in evaluations if item.hard_feasible]
    if evaluation_error and not eligible:
        return "evaluation_error", None, "one or more candidates could not be evaluated"
    if eligible:
        return "selected", eligible[0].candidate_id, None
    if not hard_feasible:
        return "all_hard_invalid", None, "every candidate failed a canonical hard gate"
    return (
        "all_above_risk_limit",
        None,
        "every hard-feasible candidate exceeded the conservative risk or clearance limit",
    )


def _coerce_forecast(forecast: object) -> MultimodalPrediction:
    """Require the canonical forecast object carrying identity and time metadata.

    Returns:
        The supplied canonical multimodal forecast.
    """
    if not isinstance(forecast, MultimodalPrediction):
        raise ValueError("forecast must be a canonical MultimodalPrediction")
    return forecast


def _validate_time_grid(
    values: object,
    *,
    expected_steps: int,
    dt_s: float,
    tolerance: float,
    label: str,
    start_s: float | None = None,
) -> None:
    """Validate an optional explicit timestamp grid against the common dt."""
    if values is None:
        return
    grid = np.asarray(values, dtype=float).reshape(-1)
    if grid.shape != (expected_steps,) or not np.all(np.isfinite(grid)):
        raise ValueError(f"{label} must contain {expected_steps} finite timestamps")
    expected = np.arange(expected_steps, dtype=float) * dt_s
    if start_s is not None:
        expected = expected + float(start_s)
    if not np.allclose(grid, expected, atol=tolerance, rtol=0.0):
        raise ValueError(f"{label} is not aligned to the exact configured time grid")


def _validate_projection_track_counts(
    belief: BeliefAwarePlannerInput,
    *,
    track_ids: tuple[int, ...],
) -> None:
    """Validate projection counts that can be derived from retained tracks."""
    diagnostics = belief.diagnostics
    expected_visible = sum(track.visibility for track in belief.tracks.values())
    expected_stale = sum(track.missed_steps > 0 for track in belief.tracks.values())
    if diagnostics.belief_step != belief.belief_step:
        raise ValueError("identity projection belief step summary does not match tracks")
    if diagnostics.retained_track_count != len(track_ids):
        raise ValueError("identity projection retained track count does not match tracks")
    if diagnostics.visible_track_count != expected_visible:
        raise ValueError("identity projection visible track count does not match tracks")
    if diagnostics.stale_track_count != expected_stale:
        raise ValueError("identity projection stale track count does not match tracks")
    if diagnostics.occluded_track_count < 0 or diagnostics.occluded_track_count > (
        len(track_ids) - expected_visible
    ):
        raise ValueError("identity projection occluded track count is inconsistent")


def _validate_projection_track_identity(
    belief: BeliefAwarePlannerInput,
    *,
    track_ids: tuple[int, ...],
) -> None:
    """Validate ordered track IDs, retired IDs, and lifecycle tokens."""
    diagnostics = belief.diagnostics
    if tuple(diagnostics.ordered_track_ids) != track_ids:
        raise ValueError("identity projection ordered track IDs do not match tracks")
    retired_ids = tuple(diagnostics.retired_track_ids)
    if diagnostics.retired_track_count != len(retired_ids):
        raise ValueError("identity projection retired track summary is inconsistent")
    if set(retired_ids) & set(track_ids):
        raise ValueError("identity projection retired IDs overlap retained tracks")
    active_tokens = {track.lifecycle_token for track in belief.tracks.values()}
    reported_tokens = tuple(diagnostics.identity_lifecycle_tokens)
    if len(reported_tokens) != len(active_tokens) + diagnostics.retired_track_count:
        raise ValueError("identity projection lifecycle token count is inconsistent")
    if not active_tokens.issubset(reported_tokens):
        raise ValueError("identity projection lifecycle tokens do not match tracks")


def _validate_projection_track_summary(belief: BeliefAwarePlannerInput) -> None:
    """Validate diagnostic counters and identities against the retained tracks."""
    track_ids = tuple(sorted(belief.tracks))
    _validate_projection_track_counts(belief, track_ids=track_ids)
    _validate_projection_track_identity(belief, track_ids=track_ids)


def _validate_projection_diagnostics(belief: BeliefAwarePlannerInput) -> None:
    """Reject incomplete or fallback identity projections before forecast joining."""
    diagnostics = belief.diagnostics
    expected_status = "supported" if belief.tracks else "empty"
    if diagnostics.status != expected_status:
        raise ValueError("identity projection status is not a complete supported or empty result")
    if diagnostics.fallback_reason is not None:
        raise ValueError("identity projection contains a fallback reason")
    if diagnostics.dropped_track_count != 0 or diagnostics.per_reason_drop_count:
        raise ValueError("identity projection dropped tracks or recorded drop reasons")
    if diagnostics.planner_name != "BeliefGuidedLocalPlanner":
        raise ValueError("identity projection planner name is not supported")
    _validate_projection_track_summary(belief)


def _validate_forecast_timing(
    forecast: MultimodalPrediction,
    belief: BeliefAwarePlannerInput,
    *,
    risk_config: RiskEstimatorConfig,
    tolerance: float,
    observation_timestamp_s: float | None,
) -> None:
    """Validate forecast and belief cycle timing against the configured grid."""
    if not math.isclose(forecast.prediction_dt, risk_config.dt_s, abs_tol=tolerance, rel_tol=0.0):
        raise ValueError("forecast dt does not match candidate/risk dt")
    if forecast.horizon != risk_config.horizon_steps or not math.isclose(
        forecast.prediction_horizon,
        risk_config.horizon_s,
        abs_tol=tolerance,
        rel_tol=0.0,
    ):
        raise ValueError("forecast horizon does not match candidate horizon")
    if forecast.timestamp < -1.0 or not math.isfinite(forecast.timestamp):
        raise ValueError("forecast timestamp must be finite or the unavailable sentinel -1")
    if observation_timestamp_s is not None:
        if not math.isfinite(observation_timestamp_s) or observation_timestamp_s < 0.0:
            raise ValueError("observation_timestamp_s must be finite and non-negative")
        if forecast.timestamp < 0.0 or not math.isclose(
            forecast.timestamp, observation_timestamp_s, abs_tol=tolerance, rel_tol=0.0
        ):
            raise ValueError("forecast timestamp does not match observation timestamp")
    forecast_step = forecast.metadata.get("step")
    if belief.belief_step is not None and (
        isinstance(forecast_step, bool)
        or not isinstance(forecast_step, Integral)
        or int(forecast_step) != belief.belief_step
    ):
        raise ValueError("forecast step does not match identity-safe belief step")


def _validate_forecast_mode(
    belief_track: PlannerTrackBelief,
    mode: TrajectoryMode,
    *,
    risk_config: RiskEstimatorConfig,
    tolerance: float,
) -> None:
    """Validate one forecast mode's identity, geometry, uncertainty, and time grid."""
    if mode.metadata.get("identity_token") != belief_track.lifecycle_token:
        raise ValueError(f"mode {mode.mode_id} identity token does not match belief")
    means = np.asarray(mode.mean, dtype=float)
    if means.shape != (risk_config.horizon_steps, 2) or not np.all(np.isfinite(means)):
        raise ValueError(f"mode {mode.mode_id!r} has an invalid future grid")
    _validate_time_grid(
        mode.metadata.get("time_offsets_s"),
        expected_steps=risk_config.horizon_steps,
        dt_s=risk_config.dt_s,
        tolerance=tolerance,
        label=f"mode {mode.mode_id} time offsets",
        start_s=risk_config.dt_s,
    )
    if mode.covariance is None and mode.std is None:
        raise ValueError(f"mode {mode.mode_id!r} lacks explicit covariance or standard deviation")
    if mode.covariance is not None and not np.all(np.isfinite(mode.covariance)):
        raise ValueError(f"mode {mode.mode_id!r} covariance is non-finite")
    if mode.std is not None and (
        not np.all(np.isfinite(mode.std)) or np.any(np.asarray(mode.std) < 0.0)
    ):
        raise ValueError(f"mode {mode.mode_id!r} standard deviation is invalid")


def _validate_forecast_track_identity(
    forecast_track: PedestrianForecast,
    belief_track: PlannerTrackBelief,
    *,
    tolerance: float,
) -> None:
    """Validate one forecast track's maintained identity and producer metadata."""
    track_id = belief_track.track_id
    if forecast_track.pedestrian_id != track_id or belief_track.track_id != track_id:
        raise ValueError("forecast and belief track identity mismatch")
    existence_matches = math.isclose(
        forecast_track.existence_probability,
        belief_track.existence_probability,
        abs_tol=tolerance,
        rel_tol=0.0,
    )
    semantics = forecast_track.metadata.get("existence_probability_semantics")
    if not existence_matches and (
        forecast_track.metadata.get("existence_probability_calibrated") is not True
        or not isinstance(semantics, str)
        or not semantics.strip()
    ):
        raise ValueError(
            "forecast existence differs from the tracker keep-alive assumption without explicit semantics"
        )
    if not math.isclose(
        forecast_track.source_confidence,
        belief_track.confidence,
        abs_tol=tolerance,
        rel_tol=0.0,
    ):
        raise ValueError("forecast confidence does not match maintained belief")
    if forecast_track.age_steps != belief_track.age_steps:
        raise ValueError("forecast age does not match maintained belief")
    identity_fields = {
        "identity_token": belief_track.lifecycle_token,
        "tracker_namespace": belief_track.tracker_namespace,
        "reset_epoch": belief_track.reset_epoch,
    }
    for field_name, expected in identity_fields.items():
        if forecast_track.metadata.get(field_name) != expected:
            raise ValueError(f"forecast {field_name} does not match maintained belief")


def _validate_forecast_modes(
    forecast_track: PedestrianForecast,
    belief_track: PlannerTrackBelief,
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
    tolerance: float,
) -> None:
    """Validate one track's bounded conditional mode collection."""
    track_id = belief_track.track_id
    if not forecast_track.modes:
        raise ValueError(f"forecast track {track_id} has no conditional modes")
    if len(forecast_track.modes) > arbitration_config.max_modes_per_track:
        raise ValueError("forecast exceeds max_modes_per_track; omitted modes are not dropped")
    _validate_distribution(
        [mode.probability for mode in forecast_track.modes], tolerance=tolerance * 10.0
    )
    mode_ids: set[str] = set()
    for mode in forecast_track.modes:
        if mode.mode_id in mode_ids:
            raise ValueError(f"forecast track {track_id} has duplicate mode IDs")
        mode_ids.add(mode.mode_id)
        _validate_forecast_mode(
            belief_track,
            mode,
            risk_config=risk_config,
            tolerance=tolerance,
        )


def _validate_forecast_track(
    forecast_track: PedestrianForecast,
    belief_track: PlannerTrackBelief,
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
    tolerance: float,
) -> None:
    """Validate one identity-matched track and all of its conditional modes."""
    _validate_forecast_track_identity(
        forecast_track,
        belief_track,
        tolerance=tolerance,
    )
    _validate_forecast_modes(
        forecast_track,
        belief_track,
        risk_config=risk_config,
        arbitration_config=arbitration_config,
        tolerance=tolerance,
    )


def _validate_forecast_contract(
    forecast: MultimodalPrediction,
    belief: BeliefAwarePlannerInput,
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
    observation_timestamp_s: float | None,
) -> int:
    """Fail closed unless identity, timing, mode probabilities, and grids agree.

    Returns:
        The total number of retained forecast modes.
    """
    tolerance = arbitration_config.numeric_tolerance
    _validate_projection_diagnostics(belief)
    _validate_forecast_timing(
        forecast,
        belief,
        risk_config=risk_config,
        tolerance=tolerance,
        observation_timestamp_s=observation_timestamp_s,
    )
    if len(forecast.forecasts) > arbitration_config.max_tracks:
        raise ValueError("forecast exceeds max_tracks; omitted tracks are not silently dropped")
    forecast_ids = set(forecast.forecasts)
    belief_ids = set(belief.tracks)
    if forecast_ids != belief_ids:
        raise ValueError("forecast track identities do not match maintained belief tracks")

    total_modes = 0
    for track_id in sorted(forecast_ids):
        forecast_track = forecast.forecasts[track_id]
        _validate_forecast_track(
            forecast_track,
            belief.tracks[track_id],
            risk_config=risk_config,
            arbitration_config=arbitration_config,
            tolerance=tolerance,
        )
        total_modes += len(forecast_track.modes)
    return total_modes


def _validate_candidate_timestamp(
    candidate: _CandidateInput,
    forecast: MultimodalPrediction,
    *,
    tolerance: float,
) -> None:
    """Require one finite candidate timestamp for the known forecast cycle."""
    if forecast.timestamp < 0.0 or not math.isfinite(forecast.timestamp):
        raise ValueError("forecast timestamp is required to bind candidates to a planning cycle")
    raw_timestamp = candidate.metadata.get("timestamp_s", candidate.metadata.get("timestamp"))
    if raw_timestamp is None:
        raise ValueError(f"candidate {candidate.candidate_id} is missing timestamp_s metadata")
    try:
        timestamp = float(raw_timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate {candidate.candidate_id} timestamp_s is invalid") from exc
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError(f"candidate {candidate.candidate_id} timestamp_s is invalid")
    if not math.isclose(timestamp, forecast.timestamp, abs_tol=tolerance, rel_tol=0.0):
        raise ValueError(f"candidate {candidate.candidate_id} timestamp does not match forecast")


def _validate_candidate_collection(
    candidates: Sequence[_CandidateInput],
    forecast: MultimodalPrediction,
    *,
    tolerance: float,
) -> None:
    """Validate cycle timestamps and shared initial robot state across candidates."""
    if not candidates:
        return
    reference_xy: np.ndarray | None = None
    reference_state: np.ndarray | None = None
    for candidate in candidates:
        _validate_candidate_timestamp(candidate, forecast, tolerance=tolerance)
        positions = candidate.action.as_array(horizon_steps=len(candidate.action.waypoints) - 1)
        initial_xy = np.asarray(positions[0], dtype=float)
        if reference_xy is None:
            reference_xy = initial_xy
        elif not np.allclose(initial_xy, reference_xy, atol=tolerance, rtol=0.0):
            raise ValueError("candidate initial XY states do not match")
        if candidate.states is None:
            continue
        initial_state = np.asarray(candidate.states[0, :5], dtype=float)
        if reference_state is None:
            reference_state = initial_state
        elif not np.allclose(initial_state, reference_state, atol=tolerance, rtol=0.0):
            raise ValueError("candidate initial full states do not match")


def _coerce_candidate_values(
    candidate_values: Sequence[object],
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
) -> tuple[list[tuple[int, _CandidateInput]], list[CandidateEvaluation]]:
    """Normalize candidate shapes and retain identifiable coercion failures.

    Returns:
        Valid candidate inputs paired with source indices, followed by failures.
    """
    coerced: list[tuple[int, _CandidateInput]] = []
    invalid: list[CandidateEvaluation] = []
    for index, item in enumerate(candidate_values):
        try:
            coerced.append(
                (
                    index,
                    _coerce_candidate(item, horizon_steps=risk_config.horizon_steps),
                )
            )
        except (
            AttributeError,
            IndexError,
            OverflowError,
            TypeError,
            ValueError,
            FloatingPointError,
        ) as exc:
            invalid.append(
                _invalid_candidate_evaluation(
                    _candidate_id_hint(item, index=index),
                    reason=str(exc),
                    risk_bucket=len(arbitration_config.risk_bucket_edges) - 1,
                    maneuver=getattr(item, "maneuver", "unknown"),
                )
            )
    return coerced, invalid


def _duplicate_candidate_inputs(
    candidates: Sequence[tuple[int, _CandidateInput]], *, risk_bucket: int
) -> tuple[list[tuple[int, _CandidateInput]], list[CandidateEvaluation]]:
    """Remove duplicate identities and retain a diagnostic for each occurrence.

    Returns:
        Unique candidate inputs and fail-closed records for duplicate IDs.
    """
    candidate_ids = [candidate.candidate_id for _, candidate in candidates]
    duplicate_ids = {
        candidate_id for candidate_id in candidate_ids if candidate_ids.count(candidate_id) > 1
    }
    unique = [
        (index, candidate)
        for index, candidate in candidates
        if candidate.candidate_id not in duplicate_ids
    ]
    invalid = [
        _invalid_candidate_evaluation(
            candidate.candidate_id,
            maneuver=candidate.maneuver,
            reason="candidate IDs must be unique",
            risk_bucket=risk_bucket,
        )
        for _, candidate in candidates
        if candidate.candidate_id in duplicate_ids
    ]
    return unique, invalid


def _prepare_candidate_collection(
    candidate_values: Sequence[object],
    forecast: MultimodalPrediction,
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
    mode_count: int,
    route_progress: Mapping[str, float] | Sequence[float] | Callable[[object], float] | None,
) -> _PreparedCandidates:
    """Coerce and validate candidates while retaining recoverable failures.

    Returns:
        Validated candidates, aligned route inputs, and retained diagnostics.
    """
    risk_bucket = len(arbitration_config.risk_bucket_edges) - 1
    indexed_candidates, invalid = _coerce_candidate_values(
        candidate_values,
        risk_config=risk_config,
        arbitration_config=arbitration_config,
    )
    indexed_candidates, duplicate_errors = _duplicate_candidate_inputs(
        indexed_candidates, risk_bucket=risk_bucket
    )
    invalid.extend(duplicate_errors)
    candidates = tuple(candidate for _, candidate in indexed_candidates)
    candidate_set_id = _candidate_set_identifier(candidates) if candidates else ""
    collection_error: str | None = None
    try:
        _validate_candidate_collection(
            candidates,
            forecast,
            tolerance=arbitration_config.numeric_tolerance,
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        collection_error = str(exc)
    route_error = False
    aligned_route_progress: Mapping[str, float] | Callable[[object], float] | None
    if (
        route_progress is not None
        and not callable(route_progress)
        and not isinstance(route_progress, Mapping)
    ):
        route_error = len(route_progress) != len(candidate_values)
        aligned_route_progress = (
            {
                candidate.candidate_id: route_progress[index]
                for index, candidate in indexed_candidates
            }
            if not route_error
            else None
        )
    else:
        aligned_route_progress = route_progress
    estimated_contact_samples = (
        len(candidates) * mode_count * (risk_config.horizon_steps + 1) * risk_config.n_samples
    )
    if estimated_contact_samples > arbitration_config.max_total_contact_samples:
        budget_error = (
            "evaluation exceeds max_total_contact_samples; no candidates or modes were dropped"
        )
        invalid.extend(
            _invalid_candidate_evaluation(
                candidate.candidate_id,
                maneuver=candidate.maneuver,
                reason=budget_error,
                risk_bucket=risk_bucket,
            )
            for candidate in candidates
        )
        collection_error = budget_error
        candidates = ()
    if collection_error is not None:
        for candidate in candidates:
            reason = collection_error
            if candidate.metadata.get("static_feasible") is not True:
                reason = f"{reason}; static_collision: candidate static feasibility is unverified"
            invalid.append(
                _invalid_candidate_evaluation(
                    candidate.candidate_id,
                    maneuver=candidate.maneuver,
                    reason=reason,
                    risk_bucket=len(arbitration_config.risk_bucket_edges) - 1,
                )
            )
        candidates = ()
    return _PreparedCandidates(
        values=candidates,
        candidate_set_id=candidate_set_id,
        estimated_contact_samples=estimated_contact_samples,
        route_progress=aligned_route_progress,
        route_error=route_error,
        invalid_evaluations=tuple(invalid),
        collection_error=collection_error,
    )


def _empty_arbitration_result(
    *, risk_config: RiskEstimatorConfig, arbitration_config: MultimodalArbitrationConfig
) -> ArbitrationResult:
    """Return the explicit result for an empty candidate portfolio.

    Returns:
        A no-candidates result with stable configuration provenance.
    """
    return ArbitrationResult(
        status="no_candidates",
        selected_candidate_id=None,
        ordered_candidate_ids=(),
        evaluations=(),
        no_selection_reason="candidate portfolio is empty",
        evaluation_duration_ms=0.0,
        candidate_count=0,
        track_count=0,
        mode_count=0,
        diagnostics=(
            ("risk_config_hash", risk_config.config_hash()),
            ("max_candidates", str(arbitration_config.max_candidates)),
            ("max_tracks", str(arbitration_config.max_tracks)),
            ("max_modes_per_track", str(arbitration_config.max_modes_per_track)),
            ("max_total_contact_samples", str(arbitration_config.max_total_contact_samples)),
        ),
        risk_config_hash=risk_config.config_hash(),
        max_candidates=arbitration_config.max_candidates,
        max_tracks=arbitration_config.max_tracks,
        max_modes_per_track=arbitration_config.max_modes_per_track,
        max_total_contact_samples=arbitration_config.max_total_contact_samples,
    )


def _candidate_cap_result(
    candidates: Sequence[object],
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
) -> ArbitrationResult:
    """Retain every identity when the candidate cap prevents evaluation.

    Returns:
        An evaluation-error result with one fail-closed record per input.
    """
    reason = "candidate portfolio exceeds max_candidates"
    invalid = tuple(
        sorted(
            (
                _invalid_candidate_evaluation(
                    _candidate_id_hint(item, index=index),
                    reason=reason,
                    risk_bucket=len(arbitration_config.risk_bucket_edges) - 1,
                    maneuver=getattr(item, "maneuver", "unknown"),
                )
                for index, item in enumerate(candidates)
            ),
            key=lambda item: item.candidate_id,
        )
    )
    risk_config_hash = risk_config.config_hash()
    return ArbitrationResult(
        status="evaluation_error",
        selected_candidate_id=None,
        ordered_candidate_ids=tuple(item.candidate_id for item in invalid),
        evaluations=invalid,
        no_selection_reason=reason,
        evaluation_duration_ms=0.0,
        candidate_count=len(candidates),
        track_count=0,
        mode_count=0,
        diagnostics=(
            ("risk_config_hash", risk_config_hash),
            ("max_candidates", str(arbitration_config.max_candidates)),
            ("max_tracks", str(arbitration_config.max_tracks)),
            ("max_modes_per_track", str(arbitration_config.max_modes_per_track)),
            ("max_total_contact_samples", str(arbitration_config.max_total_contact_samples)),
            ("error", reason),
        ),
        risk_config_hash=risk_config_hash,
        max_candidates=arbitration_config.max_candidates,
        max_tracks=arbitration_config.max_tracks,
        max_modes_per_track=arbitration_config.max_modes_per_track,
        max_total_contact_samples=arbitration_config.max_total_contact_samples,
    )


def _invalid_input_result(
    error: Exception,
    *,
    validation_phase: str,
    forecast: MultimodalPrediction | None,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
    candidate_count: int,
    evaluation_duration_ms: float,
    candidate_set_id: str | None = None,
) -> ArbitrationResult:
    """Represent a forecast or portfolio contract failure without selection.

    Returns:
        An explicit invalid-forecast or evaluation-error result.
    """
    risk_config_hash = risk_config.config_hash()
    return ArbitrationResult(
        status="invalid_forecast" if validation_phase == "forecast" else "evaluation_error",
        selected_candidate_id=None,
        ordered_candidate_ids=(),
        evaluations=(),
        no_selection_reason=str(error),
        forecast_schema_version=getattr(forecast, "schema_version", None),
        evaluation_duration_ms=evaluation_duration_ms,
        candidate_count=candidate_count,
        track_count=len(forecast.forecasts) if forecast is not None else 0,
        mode_count=(
            sum(len(track.modes) for track in forecast.forecasts.values())
            if forecast is not None
            else 0
        ),
        diagnostics=(
            ("risk_config_hash", risk_config_hash),
            ("max_candidates", str(arbitration_config.max_candidates)),
            ("max_tracks", str(arbitration_config.max_tracks)),
            ("max_modes_per_track", str(arbitration_config.max_modes_per_track)),
            ("max_total_contact_samples", str(arbitration_config.max_total_contact_samples)),
            ("error", str(error)),
        ),
        risk_config_hash=risk_config_hash,
        max_candidates=arbitration_config.max_candidates,
        max_tracks=arbitration_config.max_tracks,
        max_modes_per_track=arbitration_config.max_modes_per_track,
        max_total_contact_samples=arbitration_config.max_total_contact_samples,
        candidate_set_id=candidate_set_id,
    )


def _mode_points_and_covariance(
    mode: TrajectoryMode,
    belief_track: PlannerTrackBelief,
    *,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend maintained t=0 state to exactly H future predictor marginals.

    Returns:
        The complete grid positions and covariance marginals.
    """
    mean = np.asarray(mode.mean, dtype=float)
    if mean.shape != (horizon_steps, 2) or not np.all(np.isfinite(mean)):
        raise ValueError("mode means do not match the candidate's future grid")
    points = np.concatenate((belief_track.mean_state[None, :2], mean), axis=0)
    if mode.covariance is not None:
        future_covariance = np.asarray(mode.covariance, dtype=float)
    elif mode.std is not None:
        std = np.asarray(mode.std, dtype=float)
        future_covariance = np.zeros((horizon_steps, 2, 2), dtype=float)
        future_covariance[:, 0, 0] = np.square(std[:, 0])
        future_covariance[:, 1, 1] = np.square(std[:, 1])
    else:
        raise ValueError("mode must supply covariance or standard deviation")
    covariance = np.concatenate((belief_track.covariance[None, :2, :2], future_covariance), axis=0)
    if covariance.shape != (horizon_steps + 1, 2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("mode covariance does not match the candidate grid")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-10, rtol=0.0):
        raise ValueError("mode covariance must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance) < -1e-10):
        raise ValueError("mode covariance must be positive semidefinite")
    return points, covariance


def _mode_uncertainty_radius(
    track: PedestrianForecast,
    mode: TrajectoryMode,
    covariance: np.ndarray,
    *,
    config: MultimodalArbitrationConfig,
) -> tuple[np.ndarray, float]:
    """Return covariance support and a monotone load scalar without double inflation.

    The canonical #8051 force-residual producer folds missed-step staleness,
    occlusion, and source confidence into each mode covariance.  Its
    "age_steps" field is track tenure, so this layer never interprets it as
    staleness and never adds those producer inflations again.  Unknown
    producers must declare the same covariance provenance explicitly.
    """
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = np.maximum(eigenvalues[:, -1], 0.0)
    # For a 2-D Gaussian, this is the radial chi-square quantile multiplier.
    multiplier = math.sqrt(-2.0 * math.log(max(1.0e-15, 1.0 - config.uncertainty_quantile)))
    canonical_source = track.metadata.get("source") == "force_residual_intent_predictor"
    declared = any(
        bool(metadata.get("covariance_inflation_applied", False))
        or bool(metadata.get("covariance_includes_inflation", False))
        for metadata in (track.metadata, mode.metadata)
    )
    if not canonical_source and not declared:
        raise ValueError(
            "uncertainty inflation provenance is ambiguous for mode; "
            "forecast producer must declare covariance inflation"
        )
    # Age/confidence/occlusion inflation is already represented in covariance
    # for declared producers.  Only this module's explicit base support is
    # added, so the envelope cannot erase or double-count producer uncertainty.
    additive = config.uncertainty_base_radius_m
    radius = multiplier * np.sqrt(largest) + additive
    # Trace is used as an ordering feature as well as a diagnostic.  It is
    # monotone under PSD covariance growth, including covariance directions that
    # leave the largest eigenvalue unchanged.
    load = float(np.sum(np.trace(covariance)) + additive * additive * len(radius))
    return radius, load


def _mode_input(
    forecast_track: PedestrianForecast,
    belief_track: PlannerTrackBelief,
    mode: TrajectoryMode,
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
) -> _ModeInput:
    """Build canonical per-mode risk input and a separately named uncertainty envelope.

    Returns:
        The identity-joined mode input used by risk and hard-gate checks.
    """
    points, covariance = _mode_points_and_covariance(
        mode, belief_track, horizon_steps=risk_config.horizon_steps
    )
    risk_input = TrajectoryModeRiskInput(
        actor_id=belief_track.track_id,
        mode_id=mode.mode_id,
        actor_radius_m=float(belief_track.mean_state[4]),
        time_offsets_s=np.arange(risk_config.horizon_steps + 1, dtype=float) * risk_config.dt_s,
        mean_positions=points,
        covariances=covariance,
    )
    radius, _ = _mode_uncertainty_radius(
        forecast_track, mode, covariance, config=arbitration_config
    )
    return _ModeInput(
        belief_track=belief_track,
        forecast_track=forecast_track,
        mode=mode,
        points=points,
        covariance=covariance,
        uncertainty_radius=radius,
        risk_input=risk_input,
    )


def _risk_metrics_for_track(
    mode_inputs: Sequence[_ModeInput],
    mode_estimates: Sequence[TrajectoryModeRiskEstimate],
    *,
    config: MultimodalArbitrationConfig,
) -> TrackRiskSummary:
    """Aggregate per-mode grid-time estimates without confidence discounting.

    Returns:
        Conditional and existence-mixture diagnostics for the track.
    """
    if not mode_inputs or len(mode_inputs) != len(mode_estimates):
        raise ValueError("track mode estimates are incomplete")
    track = mode_inputs[0].forecast_track
    mode_ids = tuple(item.mode.mode_id for item in mode_inputs)
    mode_probabilities = _validate_distribution(
        tuple(float(item.mode.probability) for item in mode_inputs),
        tolerance=config.numeric_tolerance * 10.0,
    )
    mode_risks = tuple(float(estimate.estimated_time_union_bound) for estimate in mode_estimates)
    if any(not 0.0 <= value <= 1.0 or not math.isfinite(value) for value in mode_risks):
        raise ValueError("canonical discrete-grid mode risk must be finite and in [0, 1]")
    expected, var, cvar, worst = discrete_tail_metrics(
        mode_risks,
        mode_probabilities,
        config.cvar_alpha,
        stable_ids=mode_ids,
        tolerance=config.numeric_tolerance * 10.0,
    )
    existence = float(track.existence_probability)
    weighted_losses = (0.0, *mode_risks)
    weighted_probabilities = (1.0 - existence, *(existence * value for value in mode_probabilities))
    weighted_probabilities = _validate_distribution(
        weighted_probabilities, tolerance=config.numeric_tolerance * 10.0
    )
    weighted_expected, weighted_var, weighted_cvar, _ = discrete_tail_metrics(
        weighted_losses,
        weighted_probabilities,
        config.cvar_alpha,
        stable_ids=("__not_exists__", *mode_ids),
        tolerance=config.numeric_tolerance * 10.0,
    )
    radius_by_mode: list[np.ndarray] = []
    loads: list[float] = []
    for mode_input in mode_inputs:
        radius_by_mode.append(mode_input.uncertainty_radius)
        _, load = _mode_uncertainty_radius(
            mode_input.forecast_track,
            mode_input.mode,
            mode_input.covariance,
            config=config,
        )
        loads.append(load)
    critical_index = max(
        range(len(mode_risks)), key=lambda index: (mode_risks[index], mode_ids[index])
    )
    critical_step = mode_estimates[critical_index].peak_risk_time_index
    return TrackRiskSummary(
        track_id=int(track.pedestrian_id),
        mode_ids=mode_ids,
        mode_probabilities=mode_probabilities,
        mode_risks=mode_risks,
        mode_per_time_risks=tuple(tuple(item.per_time_probability) for item in mode_estimates),
        mode_per_time_standard_errors=tuple(
            tuple(item.per_time_mc_standard_error) for item in mode_estimates
        ),
        expected_mode_risk=float(expected),
        var_mode_risk=float(var),
        cvar_mode_risk=float(cvar),
        worst_mode_risk=float(worst),
        existence_probability=existence,
        existence_probability_calibrated=(
            bool(track.metadata["existence_probability_calibrated"])
            if "existence_probability_calibrated" in track.metadata
            else mode_inputs[0].belief_track.existence_probability_calibrated
        ),
        existence_probability_semantics=str(
            track.metadata.get(
                "existence_probability_semantics",
                mode_inputs[0].belief_track.existence_probability_semantics,
            )
        ),
        existence_weighted_expected_risk=float(weighted_expected),
        existence_weighted_var_risk=float(weighted_var),
        existence_weighted_cvar_risk=float(weighted_cvar),
        robust_min_clearance_m=float("inf"),
        uncertainty_radius_m=float(max(float(np.max(radius)) for radius in radius_by_mode)),
        uncertainty_load=float(
            existence
            * sum(
                probability * load
                for probability, load in zip(mode_probabilities, loads, strict=True)
            )
        ),
        critical_mode_id=mode_ids[critical_index],
        critical_step=int(critical_step),
    )


def _candidate_route_progress(  # noqa: C901
    candidate: _CandidateInput,
    route_progress: Mapping[str, float] | Sequence[float] | Callable[[object], float] | None,
    *,
    candidate_index: int,
) -> float:
    """Read route progress from #8052 output or a finite observation-derived fallback.

    Returns:
        Finite signed route progress in meters.
    """
    if route_progress is not None:
        if isinstance(route_progress, Mapping):
            if candidate.candidate_id not in route_progress:
                raise ValueError(f"missing route progress for candidate {candidate.candidate_id}")
            value = route_progress[candidate.candidate_id]
        elif callable(route_progress):
            value = route_progress(candidate)
        else:
            if candidate_index >= len(route_progress):
                raise ValueError("route progress sequence is shorter than candidates")
            value = route_progress[candidate_index]
        if hasattr(value, "is_valid") and not bool(value.is_valid):
            raise ValueError(f"route progress for candidate {candidate.candidate_id} is invalid")
        if hasattr(value, "signed_progress_m"):
            value = value.signed_progress_m
    elif "route_progress_m" in candidate.metadata:
        route_status = candidate.metadata.get("route_progress_status")
        if route_status is not None and str(route_status) != "ok":
            raise ValueError(f"route progress for candidate {candidate.candidate_id} is invalid")
        value = candidate.metadata["route_progress_m"]
    else:
        raise ValueError(
            f"route progress context is required for candidate {candidate.candidate_id}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("route progress must be finite")
    return result


def _candidate_switch_cost(
    candidate: _CandidateInput,
    switch_costs: Mapping[str, float] | None,
) -> float:
    """Read the future commitment-manager seam without owning mutable state.

    Returns:
        Finite non-negative switch cost.
    """
    value: object
    if switch_costs is not None:
        if candidate.candidate_id not in switch_costs:
            raise ValueError(f"missing switch cost for candidate {candidate.candidate_id}")
        value = switch_costs[candidate.candidate_id]
    else:
        value = candidate.metadata.get(
            "switch_cost", candidate.metadata.get("commitment_compatibility", 0.0)
        )
    cost = float(value)
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("switch cost must be finite and >= 0")
    return cost


def _waypoint_velocities(positions: np.ndarray, dt_s: float) -> np.ndarray:
    """Return one deterministic velocity vector for each aligned waypoint."""
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("positions must have shape (H + 1, 2)")
    differences = np.diff(values, axis=0) / dt_s
    return np.concatenate((differences, differences[-1:]), axis=0)


def _integrated_jerk(positions: np.ndarray, dt_s: float) -> float:
    """Return a bounded squared-jerk integral for deterministic tie-breaking."""
    velocities = _waypoint_velocities(positions, dt_s)
    acceleration = np.diff(velocities, axis=0) / dt_s
    if acceleration.shape[0] < 2:
        return 0.0
    jerk = np.diff(acceleration, axis=0) / dt_s
    return float(np.sum(np.square(jerk)) * dt_s)


def _comfort_cost(
    positions: np.ndarray,
    *,
    dt_s: float,
    scales: tuple[float, float, float],
    current_command: Sequence[float] | None,
) -> float:
    """Compute a bounded observation-derived comfort tie-breaker.

    Returns:
        Non-negative comfort cost.
    """
    velocities = _waypoint_velocities(positions, dt_s)
    acceleration = np.gradient(velocities, dt_s, axis=0)
    speed_term = float(np.max(np.linalg.norm(acceleration, axis=1))) / scales[0]
    headings = np.unwrap(np.arctan2(velocities[:, 1], velocities[:, 0]))
    angular_speed = np.gradient(headings, dt_s)
    angular_accel = np.gradient(angular_speed, dt_s)
    angular_term = float(np.max(np.abs(angular_accel))) / scales[1]
    jerk_term = _integrated_jerk(positions, dt_s) / scales[2]
    discontinuity = 0.0
    if current_command is not None:
        command = np.asarray(current_command, dtype=float).reshape(-1)
        if command.shape != (2,) or not np.all(np.isfinite(command)):
            raise ValueError("current_command must contain two finite values")
        discontinuity = float(np.linalg.norm(velocities[0] - command)) / scales[0]
    value = speed_term + angular_term + jerk_term + discontinuity
    if not math.isfinite(value):
        raise ValueError("comfort cost is non-finite")
    return float(max(0.0, value))


def _risk_bucket(value: float, edges: tuple[float, ...]) -> int:
    """Map a conservative decision score to a deterministic bucket.

    Returns:
        The index of the first edge containing ``value``.
    """
    for index, edge in enumerate(edges[1:]):
        if value <= edge:
            return index
    return len(edges) - 1


def _mode_grid_inputs(
    forecast: MultimodalPrediction,
    belief: BeliefAwarePlannerInput,
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
) -> tuple[_ModeInput, ...]:
    """Join every sorted forecast mode to exactly one maintained identity.

    Returns:
        Deterministically ordered identity-joined mode inputs.
    """
    mode_inputs: list[_ModeInput] = []
    for track_id in sorted(forecast.forecasts):
        forecast_track = forecast.forecasts[track_id]
        belief_track = belief.tracks[track_id]
        for mode in sorted(forecast_track.modes, key=lambda item: item.mode_id):
            mode_inputs.append(
                _mode_input(
                    forecast_track,
                    belief_track,
                    mode,
                    risk_config=risk_config,
                    arbitration_config=arbitration_config,
                )
            )
    return tuple(mode_inputs)


def _update_track_clearance(
    summary: TrackRiskSummary,
    mode_inputs: Sequence[_ModeInput],
    robot_positions: np.ndarray,
    *,
    risk_config: RiskEstimatorConfig,
) -> TrackRiskSummary:
    """Attach a grid-time geometric uncertainty envelope, separate from probability.

    Returns:
        The track summary with robust clearance diagnostics attached.
    """
    clearances: list[float] = []
    uncertainty_radius = 0.0
    for item in mode_inputs:
        if item.forecast_track.existence_probability <= 0.0:
            continue
        distances = np.linalg.norm(robot_positions - item.points, axis=1)
        clearance = (
            distances
            - risk_config.robot_radius_m
            - item.risk_input.actor_radius_m
            - item.uncertainty_radius
        )
        clearances.append(float(np.min(clearance)))
        uncertainty_radius = max(uncertainty_radius, float(np.max(item.uncertainty_radius)))
    robust_min = min(clearances, default=float("inf"))
    return TrackRiskSummary(
        track_id=summary.track_id,
        mode_ids=summary.mode_ids,
        mode_probabilities=summary.mode_probabilities,
        mode_risks=summary.mode_risks,
        mode_per_time_risks=summary.mode_per_time_risks,
        mode_per_time_standard_errors=summary.mode_per_time_standard_errors,
        expected_mode_risk=summary.expected_mode_risk,
        var_mode_risk=summary.var_mode_risk,
        cvar_mode_risk=summary.cvar_mode_risk,
        worst_mode_risk=summary.worst_mode_risk,
        existence_probability=summary.existence_probability,
        existence_probability_calibrated=summary.existence_probability_calibrated,
        existence_probability_semantics=summary.existence_probability_semantics,
        existence_weighted_expected_risk=summary.existence_weighted_expected_risk,
        existence_weighted_var_risk=summary.existence_weighted_var_risk,
        existence_weighted_cvar_risk=summary.existence_weighted_cvar_risk,
        robust_min_clearance_m=float(robust_min),
        uncertainty_radius_m=float(uncertainty_radius),
        uncertainty_load=summary.uncertainty_load,
        critical_mode_id=summary.critical_mode_id,
        critical_step=summary.critical_step,
    )


def _mode_velocities(item: _ModeInput, dt_s: float) -> np.ndarray:
    """Return observation velocity at t=0 and piecewise-grid mode velocities."""
    differences = np.diff(item.points, axis=0) / dt_s
    values = np.concatenate((differences, differences[-1:]), axis=0)
    values[0] = item.belief_track.mean_state[2:4]
    return values


def _verify_retained_modes(
    mode_inputs: Sequence[_ModeInput],
    *,
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    risk_config: RiskEstimatorConfig,
    verifier_config: TrajectoryVerifierConfig | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Run the canonical verifier for each retained, existent forecast mode.

    Returns:
        Violated dynamic predicates and verifier decisions in evaluation order.
    """
    hard_predicates: list[str] = []
    decisions: list[str] = []
    for item in mode_inputs:
        if item.forecast_track.existence_probability <= 0.0:
            continue
        result = verify_trajectory(
            robot_positions=robot_positions,
            robot_velocities=robot_velocities,
            pedestrian_positions=item.points[:, None, :],
            pedestrian_velocities=_mode_velocities(item, risk_config.dt_s)[:, None, :],
            dt_s=risk_config.dt_s,
            robot_radius_m=risk_config.robot_radius_m,
            pedestrian_radius_m=item.risk_input.actor_radius_m,
            config=verifier_config,
        )
        decisions.append(result.decision)
        if result.decision == DECISION_FALLBACK_BRAKE:
            if result.violated_predicates:
                hard_predicates.extend(
                    f"track {item.belief_track.track_id} mode {item.mode.mode_id}: {predicate}"
                    for predicate in result.violated_predicates
                )
            else:
                hard_predicates.append(
                    "trajectory verifier returned fallback_brake without violated predicates "
                    f"for track {item.belief_track.track_id} mode {item.mode.mode_id}"
                )
    return tuple(hard_predicates), tuple(decisions)


def _candidate_static_context(candidate: _CandidateInput) -> tuple[tuple[str, ...], float | None]:
    """Read static feasibility metadata and its optional clearance diagnostic.

    Returns:
        Static hard-gate reasons and the optional finite clearance.
    """
    hard_predicates: list[str] = []
    static_feasible = candidate.metadata.get("static_feasible")
    if static_feasible is not True:
        hard_predicates.append("static_collision: candidate static feasibility is unverified")
    if static_feasible is False:
        hard_predicates.append("static_collision: candidate metadata marks static infeasibility")
    raw_clearance = candidate.metadata.get("static_min_clearance_m")
    if raw_clearance is None:
        return tuple(hard_predicates), None
    static_clearance = float(raw_clearance)
    if not math.isfinite(static_clearance):
        raise ValueError("candidate static_min_clearance_m must be finite")
    if static_clearance < 0.0:
        hard_predicates.append("static_collision: candidate static clearance is negative")
    return tuple(hard_predicates), static_clearance


def _verifier_decision(mode_inputs: Sequence[_ModeInput], decisions: Sequence[str]) -> str:
    """Summarize canonical verifier decisions without changing their authority.

    Returns:
        The deterministic aggregate verifier decision.
    """
    if not mode_inputs:
        return "skipped_no_hazard"
    if DECISION_FALLBACK_BRAKE in decisions:
        return DECISION_FALLBACK_BRAKE
    if "warn" in decisions:
        return "warn"
    return "accept"


def _geometry_only_min_clearance_m(
    mode_inputs: Sequence[_ModeInput],
    robot_positions: np.ndarray,
    *,
    risk_config: RiskEstimatorConfig,
) -> float:
    """Return mean-path clearance without covariance or other uncertainty margins.

    Returns:
        Minimum sampled center-distance clearance over existent forecast modes,
        or positive infinity when no dynamic hazard is present.
    """
    clearances = (
        float(
            np.min(np.linalg.norm(robot_positions - item.points, axis=1))
            - risk_config.robot_radius_m
            - item.risk_input.actor_radius_m
        )
        for item in mode_inputs
        if item.forecast_track.existence_probability > 0.0
    )
    return min(clearances, default=float("inf"))


def _multimodal_hard_gate(
    candidate: _CandidateInput,
    mode_inputs: Sequence[_ModeInput],
    *,
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    geometry_only_min_clearance_m: float,
    risk_config: RiskEstimatorConfig,
    verifier_config: TrajectoryVerifierConfig | None,
    actuator_config: ActuatorLimitsConfig | None,
) -> HardGateResult:
    """Apply canonical trajectory, static, and actuator hard checks.

    Returns:
        The canonical hard-gate result for the candidate.
    """
    dynamic_reasons, decisions = _verify_retained_modes(
        mode_inputs,
        robot_positions=robot_positions,
        robot_velocities=robot_velocities,
        risk_config=risk_config,
        verifier_config=verifier_config,
    )
    static_reasons, static_clearance = _candidate_static_context(candidate)
    hard_predicates = list(dynamic_reasons) + list(static_reasons)

    resolved_actuator = actuator_config if actuator_config is not None else ActuatorLimitsConfig()
    hazard_clearance = float(geometry_only_min_clearance_m)
    if static_clearance is not None:
        hazard_clearance = min(hazard_clearance, static_clearance)
    if not math.isfinite(hazard_clearance):
        max_speed = float(np.max(np.linalg.norm(robot_velocities, axis=1)))
        hazard_clearance = stopping_distance(max_speed, resolved_actuator)
    states = candidate.states
    headings = states[:, 2] if states is not None else None
    angular_velocities = states[:, 4] if states is not None else None
    actuator_report = evaluate_actuator_feasibility(
        robot_positions=robot_positions,
        robot_velocities=robot_velocities,
        dt_s=risk_config.dt_s,
        hazard_clearance_m=hazard_clearance,
        config=resolved_actuator,
        robot_headings=headings,
        robot_angular_velocities=angular_velocities,
    )
    if actuator_report.verdict != VERDICT_ACTUATOR_FEASIBLE:
        actuator_reason = (
            f"actuator feasibility reported a non-feasible verdict ({actuator_report.verdict})"
        )
        hard_predicates.append(actuator_reason)

    verifier_decision = _verifier_decision(mode_inputs, decisions)
    return HardGateResult(
        eligible=not hard_predicates,
        verifier_decision=verifier_decision,
        actuator_verdict=actuator_report.verdict,
        violated_predicates=tuple(hard_predicates),
        violated_limits=tuple(actuator_report.violated_limits),
        ineligibility_reason="; ".join(dict.fromkeys(hard_predicates)) if hard_predicates else None,
    )


def _validate_candidate_metadata(
    candidate: _CandidateInput,
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
) -> None:
    """Validate candidate metadata that defines the shared grid and footprint."""
    metadata = candidate.metadata
    explicit_grid_required = metadata.get("generation_mode") == "generation_only"
    metadata_horizon = metadata.get("horizon_steps")
    if metadata_horizon is None and explicit_grid_required:
        raise ValueError("generated candidate is missing horizon_steps metadata")
    if metadata_horizon is not None and (
        isinstance(metadata_horizon, bool)
        or not isinstance(metadata_horizon, Integral)
        or int(metadata_horizon) != risk_config.horizon_steps
    ):
        raise ValueError("candidate horizon does not match the common grid")
    metadata_dt = metadata.get("dt_s")
    if metadata_dt is None and explicit_grid_required:
        raise ValueError("generated candidate is missing dt_s metadata")
    if metadata_dt is not None:
        if isinstance(metadata_dt, bool):
            raise ValueError("candidate dt must match the common grid")
        if not math.isclose(
            float(metadata_dt),
            risk_config.dt_s,
            abs_tol=arbitration_config.numeric_tolerance,
            rel_tol=0.0,
        ):
            raise ValueError("candidate dt does not match the common grid")
    metadata_radius = metadata.get("robot_radius_m")
    if metadata_radius is not None and not math.isclose(
        float(metadata_radius),
        risk_config.robot_radius_m,
        abs_tol=arbitration_config.numeric_tolerance,
        rel_tol=0.0,
    ):
        raise ValueError("risk robot radius does not match candidate-generation radius")
    _validate_time_grid(
        metadata.get("time_offsets_s"),
        expected_steps=risk_config.horizon_steps + 1,
        dt_s=risk_config.dt_s,
        tolerance=arbitration_config.numeric_tolerance,
        label=f"candidate {candidate.candidate_id} time offsets",
        start_s=0.0,
    )


def _candidate_robot_velocities(
    candidate: _CandidateInput,
    positions: np.ndarray,
    *,
    dt_s: float,
) -> np.ndarray:
    """Resolve candidate robot velocities from states or waypoint geometry.

    Returns:
        Robot XY velocities aligned to every candidate waypoint.
    """
    if candidate.states is None:
        velocities = _waypoint_velocities(positions, dt_s)
        headings = np.unwrap(np.arctan2(velocities[:, 1], velocities[:, 0]))
        speeds = np.linalg.norm(velocities, axis=1)
    else:
        headings = candidate.states[:, 2]
        speeds = candidate.states[:, 3]
    return np.column_stack((speeds * np.cos(headings), speeds * np.sin(headings)))


def _candidate_track_summaries(
    candidate: _CandidateInput,
    mode_inputs: Sequence[_ModeInput],
    positions: np.ndarray,
    *,
    risk_config: RiskEstimatorConfig,
    arbitration_config: MultimodalArbitrationConfig,
) -> tuple[TrackRiskSummary, ...]:
    """Estimate and geometrically summarize every identity-grouped track.

    Returns:
        Deterministically ordered track summaries.
    """
    grouped_inputs: dict[int, list[_ModeInput]] = {}
    for item in mode_inputs:
        grouped_inputs.setdefault(item.belief_track.track_id, []).append(item)
    summaries: list[TrackRiskSummary] = []
    for track_id in sorted(grouped_inputs):
        selected_inputs = grouped_inputs[track_id]
        estimates = tuple(
            estimate_trajectory_mode_risk(candidate.action, item.risk_input, risk_config)
            for item in selected_inputs
        )
        summary = _risk_metrics_for_track(selected_inputs, estimates, config=arbitration_config)
        summaries.append(
            _update_track_clearance(
                summary,
                selected_inputs,
                positions,
                risk_config=risk_config,
            )
        )
    return tuple(summaries)


def _aggregate_candidate_risk(
    summaries: Sequence[TrackRiskSummary],
    *,
    config: MultimodalArbitrationConfig,
) -> _CandidateRiskAggregate:
    """Aggregate track summaries into conservative risk and monotone uncertainty fields.

    Returns:
        Candidate-level risk, clearance, and uncertainty components.
    """
    expected_mode = min(1.0, sum(item.expected_mode_risk for item in summaries))
    var_mode = min(1.0, sum(item.var_mode_risk for item in summaries))
    cvar_mode = min(1.0, sum(item.cvar_mode_risk for item in summaries))
    worst_mode = max((item.worst_mode_risk for item in summaries), default=0.0)
    raw_dynamic = min(1.0, sum(item.existence_weighted_expected_risk for item in summaries))
    existence_var = min(1.0, sum(item.existence_weighted_var_risk for item in summaries))
    existence_cvar = min(1.0, sum(item.existence_weighted_cvar_risk for item in summaries))
    robust_clearance = min(
        (item.robust_min_clearance_m for item in summaries), default=float("inf")
    )
    uncertainty_radius = max((item.uncertainty_radius_m for item in summaries), default=0.0)
    uncertainty_load = sum(item.uncertainty_load for item in summaries)
    per_track_conservative: list[float] = []
    for summary in summaries:
        clearance_shortfall = max(
            0.0,
            (config.safe_clearance_m - summary.robust_min_clearance_m) / config.safe_clearance_m,
        )
        per_track_conservative.append(
            min(1.0, summary.existence_weighted_cvar_risk + min(1.0, clearance_shortfall))
        )
    max_track_risk = max(per_track_conservative, default=0.0)
    uncertainty_margin = max(
        (
            max(0.0, config.safe_clearance_m - item.robust_min_clearance_m)
            / config.safe_clearance_m
            for item in summaries
        ),
        default=0.0,
    )
    uncertainty_margin = min(1.0, uncertainty_margin)
    # Tracks are not independent events.  The clipped sum is a conservative
    # union-bound-style aggregate; ``max_track_risk`` remains the separate
    # worst individual-track diagnostic.
    conservative = min(1.0, sum(per_track_conservative))
    risk_limit_ok = (
        raw_dynamic <= config.raw_risk_limit + config.numeric_tolerance
        and conservative <= config.conservative_risk_limit + config.numeric_tolerance
        and robust_clearance >= config.conservative_clearance_m
    )
    critical: tuple[float, int, str, int] | None = None
    for summary in summaries:
        if summary.critical_mode_id is None:
            continue
        mode_index = summary.mode_ids.index(summary.critical_mode_id)
        key = (
            summary.mode_risks[mode_index],
            summary.track_id,
            summary.critical_mode_id,
            summary.critical_step if summary.critical_step is not None else -1,
        )
        if critical is None or key > critical:
            critical = key
    return _CandidateRiskAggregate(
        expected_mode=expected_mode,
        var_mode=var_mode,
        cvar_mode=cvar_mode,
        worst_mode=worst_mode,
        raw_dynamic=raw_dynamic,
        existence_var=existence_var,
        existence_cvar=existence_cvar,
        robust_clearance=robust_clearance,
        uncertainty_radius=uncertainty_radius,
        uncertainty_load=float(uncertainty_load),
        uncertainty_margin=uncertainty_margin,
        max_track_risk=float(max_track_risk),
        conservative=conservative,
        risk_limit_ok=bool(risk_limit_ok),
        risk_bucket=_risk_bucket(conservative, config.risk_bucket_edges),
        critical=critical,
    )


def _candidate_liveness(
    route_value: float,
    robot_velocities: np.ndarray,
    *,
    stopped_duration_s: float,
    config: MultimodalArbitrationConfig,
) -> float:
    """Compute liveness cost after validating the stopped-duration context.

    Returns:
        Non-negative liveness tie-break cost.
    """
    if not math.isfinite(stopped_duration_s) or stopped_duration_s < 0.0:
        raise ValueError("stopped_duration_s must be finite and >= 0")
    value = max(0.0, -route_value) / config.route_progress_scale_m
    if float(np.linalg.norm(robot_velocities[-1])) <= 0.05:
        value += stopped_duration_s / config.liveness_scale
    return value


def _evaluate_multimodal_candidate(
    candidate: _CandidateInput,
    *,
    candidate_index: int,
    context: _EvaluationContext,
) -> CandidateEvaluation:
    """Evaluate exact supplied mode marginals and all canonical hard gates.

    Returns:
        Immutable diagnostics and ordering fields for one candidate.
    """
    risk_config = context.risk_config
    arbitration_config = context.arbitration_config
    positions = candidate.action.as_array(horizon_steps=risk_config.horizon_steps)
    _validate_candidate_metadata(
        candidate, risk_config=risk_config, arbitration_config=arbitration_config
    )
    robot_velocities = _candidate_robot_velocities(candidate, positions, dt_s=risk_config.dt_s)
    geometry_clearance = _geometry_only_min_clearance_m(
        context.mode_inputs,
        positions,
        risk_config=risk_config,
    )
    hard_gate = _multimodal_hard_gate(
        candidate,
        context.mode_inputs,
        robot_positions=positions,
        robot_velocities=robot_velocities,
        geometry_only_min_clearance_m=geometry_clearance,
        risk_config=risk_config,
        verifier_config=context.verifier_config,
        actuator_config=context.actuator_config,
    )
    braking_feasible = not any(
        "brak" in value.lower()
        for value in (*hard_gate.violated_predicates, *hard_gate.violated_limits)
    )
    try:
        per_track = _candidate_track_summaries(
            candidate,
            context.mode_inputs,
            positions,
            risk_config=risk_config,
            arbitration_config=arbitration_config,
        )
        aggregate = _aggregate_candidate_risk(per_track, config=arbitration_config)
    except (TypeError, ValueError, FloatingPointError) as exc:
        return _invalid_candidate_evaluation(
            candidate.candidate_id,
            maneuver=candidate.maneuver,
            reason=f"dynamic risk evaluation failed: {exc}",
            risk_bucket=len(arbitration_config.risk_bucket_edges) - 1,
            hard_gate=hard_gate,
            braking_feasible=braking_feasible,
        )
    route_value = _candidate_route_progress(
        candidate, context.route_progress, candidate_index=candidate_index
    )
    switch_cost = _candidate_switch_cost(candidate, context.switch_costs)
    comfort = _comfort_cost(
        positions,
        dt_s=risk_config.dt_s,
        scales=arbitration_config.comfort_scales,
        current_command=context.current_command,
    )
    liveness = _candidate_liveness(
        route_value,
        robot_velocities,
        stopped_duration_s=context.stopped_duration_s,
        config=arbitration_config,
    )
    hard_reasons = list(hard_gate.violated_predicates) + list(hard_gate.violated_limits)
    if not aggregate.risk_limit_ok:
        hard_reasons.append("conservative risk or clearance limit exceeded")
    decision_key: tuple[object, ...] = (
        0 if hard_gate.eligible else 1,
        0 if braking_feasible else 1,
        0 if aggregate.risk_limit_ok else 1,
        aggregate.risk_bucket,
        float(aggregate.uncertainty_load),
        float(aggregate.uncertainty_margin),
        float(aggregate.conservative),
        float(aggregate.raw_dynamic),
        float(aggregate.existence_cvar),
        -float(route_value),
        float(liveness),
        float(comfort),
        float(switch_cost / arbitration_config.switch_cost_scale),
        candidate.candidate_id,
    )
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        maneuver=candidate.maneuver,
        hard_feasible=hard_gate.eligible,
        hard_reasons=tuple(hard_reasons),
        raw_dynamic_risk=float(aggregate.raw_dynamic),
        conservative_dynamic_risk=float(aggregate.conservative),
        max_track_risk=float(aggregate.max_track_risk),
        risk_bucket=int(aggregate.risk_bucket),
        critical_track_id=aggregate.critical[1] if aggregate.critical is not None else None,
        critical_mode_id=aggregate.critical[2] if aggregate.critical is not None else None,
        critical_step=(
            aggregate.critical[3]
            if aggregate.critical is not None and aggregate.critical[3] >= 0
            else None
        ),
        route_progress_m=float(route_value),
        liveness_cost=float(liveness),
        comfort_cost=float(comfort),
        switch_cost=float(switch_cost),
        decision_key=decision_key,
        expected_mode_risk=float(aggregate.expected_mode),
        var_mode_risk=float(aggregate.var_mode),
        cvar_mode_risk=float(aggregate.cvar_mode),
        worst_mode_risk=float(aggregate.worst_mode),
        existence_weighted_expected_risk=float(aggregate.raw_dynamic),
        existence_weighted_var_risk=float(aggregate.existence_var),
        existence_weighted_cvar_risk=float(aggregate.existence_cvar),
        robust_min_clearance_m=float(aggregate.robust_clearance),
        uncertainty_radius_m=float(aggregate.uncertainty_radius),
        uncertainty_load=float(aggregate.uncertainty_load),
        uncertainty_margin=float(aggregate.uncertainty_margin),
        braking_feasible=braking_feasible,
        risk_limit_ok=aggregate.risk_limit_ok,
        rejection_reason=None if not hard_reasons else "; ".join(dict.fromkeys(hard_reasons)),
        hard_gate=hard_gate,
        track_risks=tuple(per_track),
    )


def arbitrate_multimodal_trajectories(  # noqa: PLR0913
    candidates: Sequence[CandidateAction | object],
    forecast: object,
    *,
    belief: BeliefAwarePlannerInput,
    risk_config: RiskEstimatorConfig | None = None,
    arbitration_config: MultimodalArbitrationConfig | None = None,
    route_progress: Mapping[str, float] | Sequence[float] | Callable[[object], float] | None = None,
    current_command: Sequence[float] | None = None,
    stopped_duration_s: float = 0.0,
    switch_costs: Mapping[str, float] | None = None,
    verifier_config: TrajectoryVerifierConfig | None = None,
    actuator_config: ActuatorLimitsConfig | None = None,
    observation_timestamp_s: float | None = None,
) -> ArbitrationResult:
    """Evaluate and select a bounded multimodal maneuver portfolio.

    The function is pure and observation-derived. It joins every forecast and
    mode to a maintained lifecycle identity, uses #9813's discrete marginal
    estimator, and applies the canonical deterministic verifier and actuator
    gate to every retained mean path. It does not register a planner, inspect
    simulator futures, or calibrate a threshold. Risk values retain their
    finite-sample grid-time claim boundary.

    Returns:
        An immutable arbitration result containing one diagnostic for every
        candidate that could be identified and a deterministic selection status.
    """
    started_ns = time.perf_counter_ns()
    risk_config = risk_config if risk_config is not None else RiskEstimatorConfig()
    arbitration_config = (
        arbitration_config if arbitration_config is not None else MultimodalArbitrationConfig()
    )
    candidate_values = tuple(candidates)
    if not candidate_values:
        return _empty_arbitration_result(
            risk_config=risk_config,
            arbitration_config=arbitration_config,
        )
    if len(candidate_values) > arbitration_config.max_candidates:
        return _candidate_cap_result(
            candidate_values,
            risk_config=risk_config,
            arbitration_config=arbitration_config,
        )

    forecast_value: MultimodalPrediction | None = None
    mode_count = 0
    prepared: _PreparedCandidates | None = None
    validation_phase = "forecast"
    try:
        forecast_value = _coerce_forecast(forecast)
        if not isinstance(belief, BeliefAwarePlannerInput):
            raise ValueError("belief must be an identity-safe BeliefAwarePlannerInput")
        mode_count = _validate_forecast_contract(
            forecast_value,
            belief,
            risk_config=risk_config,
            arbitration_config=arbitration_config,
            observation_timestamp_s=observation_timestamp_s,
        )
        mode_inputs = _mode_grid_inputs(
            forecast_value,
            belief,
            risk_config=risk_config,
            arbitration_config=arbitration_config,
        )
        validation_phase = "candidate"
        prepared = _prepare_candidate_collection(
            candidate_values,
            forecast_value,
            risk_config=risk_config,
            arbitration_config=arbitration_config,
            mode_count=mode_count,
            route_progress=route_progress,
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        return _invalid_input_result(
            exc,
            validation_phase=validation_phase,
            forecast=forecast_value,
            risk_config=risk_config,
            arbitration_config=arbitration_config,
            candidate_count=len(candidate_values),
            evaluation_duration_ms=(time.perf_counter_ns() - started_ns) / 1e6,
            candidate_set_id=prepared.candidate_set_id if prepared is not None else None,
        )

    if prepared is None:
        return _invalid_input_result(
            ValueError("candidate preparation produced no result"),
            validation_phase="candidate",
            forecast=forecast_value,
            risk_config=risk_config,
            arbitration_config=arbitration_config,
            candidate_count=len(candidate_values),
            evaluation_duration_ms=(time.perf_counter_ns() - started_ns) / 1e6,
            candidate_set_id=None,
        )
    evaluations = list(prepared.invalid_evaluations)
    route_error = prepared.route_error
    evaluation_error = bool(evaluations)
    context = _EvaluationContext(
        forecast=forecast_value,
        mode_inputs=mode_inputs,
        risk_config=risk_config,
        arbitration_config=arbitration_config,
        route_progress=prepared.route_progress,
        current_command=current_command,
        stopped_duration_s=stopped_duration_s,
        switch_costs=switch_costs,
        verifier_config=verifier_config,
        actuator_config=actuator_config,
    )
    for index, candidate in enumerate(prepared.values):
        try:
            evaluation = _evaluate_multimodal_candidate(
                candidate,
                candidate_index=index,
                context=context,
            )
        except (TypeError, ValueError, FloatingPointError) as exc:
            reason = str(exc)
            is_route_error = "route progress" in reason.lower()
            route_error = route_error or is_route_error
            evaluation_error = evaluation_error or not is_route_error
            evaluation = _invalid_candidate_evaluation(
                candidate.candidate_id,
                maneuver=candidate.maneuver,
                reason=reason,
                risk_bucket=len(arbitration_config.risk_bucket_edges) - 1,
            )
        evaluation_error = evaluation_error or (
            evaluation.rejection_reason is not None
            and evaluation.rejection_reason.startswith("evaluation_error:")
        )
        evaluations.append(evaluation)

    evaluations.sort(key=lambda item: item.decision_key)
    ordered_ids = tuple(item.candidate_id for item in evaluations)
    duration = (time.perf_counter_ns() - started_ns) / 1e6
    status, selected_id, reason = _resolve_selection(
        evaluations,
        route_error=route_error,
        evaluation_error=evaluation_error,
    )
    if status == "evaluation_error" and prepared.collection_error is not None:
        reason = prepared.collection_error
    risk_config_hash = risk_config.config_hash()
    diagnostics = (
        ("risk_config_hash", risk_config_hash),
        ("max_candidates", str(arbitration_config.max_candidates)),
        ("max_tracks", str(arbitration_config.max_tracks)),
        ("max_modes_per_track", str(arbitration_config.max_modes_per_track)),
        ("max_total_contact_samples", str(arbitration_config.max_total_contact_samples)),
        ("estimated_contact_samples", str(prepared.estimated_contact_samples)),
    )
    return ArbitrationResult(
        status=status,
        selected_candidate_id=selected_id,
        ordered_candidate_ids=ordered_ids,
        evaluations=tuple(evaluations),
        no_selection_reason=reason,
        forecast_schema_version=forecast_value.schema_version,
        evaluation_duration_ms=float(duration),
        candidate_count=len(candidate_values),
        track_count=len(forecast_value.forecasts),
        mode_count=mode_count,
        diagnostics=diagnostics,
        risk_config_hash=risk_config_hash,
        max_candidates=arbitration_config.max_candidates,
        max_tracks=arbitration_config.max_tracks,
        max_modes_per_track=arbitration_config.max_modes_per_track,
        max_total_contact_samples=arbitration_config.max_total_contact_samples,
        candidate_set_id=prepared.candidate_set_id,
    )


# Names that make the pure evaluator discoverable to callers using either the
# issue vocabulary (arbitration) or the existing ranker vocabulary (ranking).
evaluate_multimodal_trajectories = arbitrate_multimodal_trajectories
rank_multimodal_trajectories = arbitrate_multimodal_trajectories
select_multimodal_trajectory = arbitrate_multimodal_trajectories


__all__ = [
    "ARBITRATION_CLAIM_BOUNDARY",
    "ARBITRATION_SCHEMA_VERSION",
    "ArbitrationResult",
    "CandidateEvaluation",
    "MultimodalArbitrationConfig",
    "TrackRiskSummary",
    "arbitrate_multimodal_trajectories",
    "discrete_tail_metrics",
    "discrete_upper_tail_cvar",
]
