"""Fixture-only decomposition of prediction, planning, and runtime safety evidence.

The benchmark currently has separate forecast, planner, and runtime-safety surfaces.  A
single episode outcome can therefore be difficult to diagnose: a collision may follow a
poor forecast, a poor nominal plan, or an unavailable safety check.  This module provides
the small typed trace contract needed to keep those mechanisms separate.

This is an implementation-integrity and fixture-diagnostic slice for issue #7317.  It
reuses :func:`robot_sf.benchmark.uncertainty_safety.split_conformal_radius` for the
prediction-coverage calculation and records the existing chance-constrained MPC builder
as the planner integration owner.  It does not run a navigation campaign, change a
planner, or provide a per-encounter, deployment, or real-world safety guarantee.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jsonschema
import numpy as np

from robot_sf.benchmark.uncertainty_safety import (
    SPLIT_CONFORMAL_RADIUS_SCHEMA,
    split_conformal_radius,
)

PREDICTION_PLANNING_SAFETY_SCHEMA_VERSION = "prediction_planning_safety.v1"

PredictionRepresentation = Literal["point", "interval", "samples"]
PredictionStatus = Literal["available", "unavailable"]
PlanningStatus = Literal["available", "unavailable"]
RuntimeSafetyStatus = Literal[
    "verified",
    "contingency_invoked",
    "verification_unavailable",
]
LaneId = Literal["baseline", "uncertainty_aware"]
FixtureCase = Literal[
    "calibration_reference",
    "good_prediction_poor_planning",
    "poor_prediction_safe_fallback",
    "verification_unavailable",
]

_PREDICTION_REPRESENTATIONS = {"point", "interval", "samples"}
_PREDICTION_STATUSES = {"available", "unavailable"}
_PLANNING_STATUSES = {"available", "unavailable"}
_RUNTIME_STATUSES = {"verified", "contingency_invoked", "verification_unavailable"}
_LANES = {"baseline", "uncertainty_aware"}
_FIXTURE_CASES = {
    "calibration_reference",
    "good_prediction_poor_planning",
    "poor_prediction_safe_fallback",
    "verification_unavailable",
}
_OUTCOME_FIELDS = (
    "collision",
    "near_miss",
    "path_efficiency",
    "pedestrian_disruption",
    "unnecessary_braking",
)


def _finite_non_negative(name: str, value: float) -> float:
    """Validate and normalize a finite non-negative scalar.

    Returns:
        Normalized scalar value.
    """
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return normalized


def _optional_finite_non_negative(name: str, value: float | None) -> float | None:
    """Validate an optional finite non-negative scalar.

    Returns:
        Normalized scalar value, or ``None`` when unavailable.
    """
    if value is None:
        return None
    return _finite_non_negative(name, value)


def _optional_bool(name: str, value: bool | None) -> bool | None:
    """Validate an optional boolean outcome field.

    Returns:
        The unchanged boolean value, or ``None`` when unavailable.
    """
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool or None")
    return value


@dataclass(frozen=True, slots=True)
class PredictionHorizonTrace:
    """One horizon-specific prediction and realized-error record.

    ``realized_error_m`` is the nonconformity score used by the fixture diagnostic.  The
    source forecast may be a point prediction, an interval, or samples; this record keeps
    the representation explicit instead of converting all three into an unlabeled radius.
    """

    horizon_step: int
    representation: PredictionRepresentation
    status: PredictionStatus
    interval_radius_m: float | None = None
    sample_count: int = 0
    realized_error_m: float | None = None
    out_of_support: bool = False

    def __post_init__(self) -> None:
        """Validate horizon, representation, availability, and residual semantics."""
        if isinstance(self.horizon_step, bool) or int(self.horizon_step) < 0:
            raise ValueError("horizon_step must be a non-negative integer")
        if self.representation not in _PREDICTION_REPRESENTATIONS:
            raise ValueError(f"unsupported prediction representation: {self.representation!r}")
        if self.status not in _PREDICTION_STATUSES:
            raise ValueError(f"unsupported prediction status: {self.status!r}")
        if isinstance(self.sample_count, bool) or int(self.sample_count) < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if (
            self.representation == "samples"
            and self.status == "available"
            and self.sample_count < 1
        ):
            raise ValueError("sample representation requires sample_count >= 1")
        radius = _optional_finite_non_negative("interval_radius_m", self.interval_radius_m)
        error = _optional_finite_non_negative("realized_error_m", self.realized_error_m)
        object.__setattr__(self, "horizon_step", int(self.horizon_step))
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "interval_radius_m", radius)
        object.__setattr__(self, "realized_error_m", error)
        if self.status == "unavailable" and (error is not None or radius is not None):
            raise ValueError("unavailable predictions cannot carry radius or realized error")
        if self.out_of_support and self.status != "available":
            raise ValueError("out_of_support requires an available prediction")


@dataclass(frozen=True, slots=True)
class NominalPlanningTrace:
    """Nominal planner output and the deterministic/uncertainty margin decision."""

    status: PlanningStatus
    planner_source: str
    deterministic_margin_m: float | None = None
    uncertainty_margin_m: float | None = None
    effective_margin_m: float | None = None
    command: tuple[float, float] | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate planning availability and the monotone margin decomposition."""
        if self.status not in _PLANNING_STATUSES:
            raise ValueError(f"unsupported planning status: {self.status!r}")
        if not isinstance(self.planner_source, str) or not self.planner_source.strip():
            raise ValueError("planner_source must be a non-empty string")
        if self.status == "available":
            if self.deterministic_margin_m is None or self.effective_margin_m is None:
                raise ValueError("available planning requires deterministic and effective margins")
            deterministic = _finite_non_negative(
                "deterministic_margin_m", self.deterministic_margin_m
            )
            uncertainty = _finite_non_negative(
                "uncertainty_margin_m", self.uncertainty_margin_m or 0.0
            )
            effective = _finite_non_negative("effective_margin_m", self.effective_margin_m)
            if effective + 1e-12 < deterministic:
                raise ValueError("effective_margin_m cannot weaken deterministic margin")
            object.__setattr__(self, "deterministic_margin_m", deterministic)
            object.__setattr__(self, "uncertainty_margin_m", uncertainty)
            object.__setattr__(self, "effective_margin_m", effective)
        elif any(
            value is not None
            for value in (
                self.deterministic_margin_m,
                self.uncertainty_margin_m,
                self.effective_margin_m,
                self.command,
            )
        ):
            raise ValueError("unavailable planning cannot carry planner output or margins")
        if self.status == "unavailable" and not self.unavailable_reason:
            raise ValueError("unavailable planning requires unavailable_reason")


@dataclass(frozen=True, slots=True)
class RuntimeSafetyTrace:
    """Runtime verification decision and any contingency action."""

    status: RuntimeSafetyStatus
    selected_action: tuple[float, float] | None = None
    contingency_action: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate event-specific action and reason requirements."""
        if self.status not in _RUNTIME_STATUSES:
            raise ValueError(f"unsupported runtime safety status: {self.status!r}")
        if self.status == "contingency_invoked" and not self.contingency_action:
            raise ValueError("contingency_invoked requires contingency_action")
        if self.status == "verification_unavailable" and not self.reason:
            raise ValueError("verification_unavailable requires reason")
        if self.status == "verified" and self.contingency_action is not None:
            raise ValueError("verified records cannot carry contingency_action")


@dataclass(frozen=True, slots=True)
class RealizedOutcomeTrace:
    """Optional realized episode outcomes; absent fields remain unavailable."""

    collision: bool | None = None
    near_miss: bool | None = None
    path_efficiency: float | None = None
    pedestrian_disruption: float | None = None
    unnecessary_braking: bool | None = None

    def __post_init__(self) -> None:
        """Validate optional outcome fields without synthesizing absent measurements."""
        for name in ("collision", "near_miss", "unnecessary_braking"):
            _optional_bool(name, getattr(self, name))
        efficiency = self.path_efficiency
        if efficiency is not None:
            efficiency = float(efficiency)
            if not math.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
                raise ValueError("path_efficiency must be finite and in [0, 1]")
            object.__setattr__(self, "path_efficiency", efficiency)
        disruption = _optional_finite_non_negative(
            "pedestrian_disruption", self.pedestrian_disruption
        )
        object.__setattr__(self, "pedestrian_disruption", disruption)


@dataclass(frozen=True, slots=True)
class PredictionPlanningSafetyTrace:
    """One step/episode row binding prediction, planning, runtime, and outcome layers."""

    trace_id: str
    scenario_id: str
    seed: int
    split: Literal["fit", "calibration", "evaluation"]
    lane_id: LaneId
    fixture_case: FixtureCase
    prediction: tuple[PredictionHorizonTrace, ...]
    planning: NominalPlanningTrace
    runtime_safety: RuntimeSafetyTrace
    outcome: RealizedOutcomeTrace

    def __post_init__(self) -> None:
        """Validate stable identity and keep all mechanism layers present."""
        for name in ("trace_id", "scenario_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if self.split not in {"fit", "calibration", "evaluation"}:
            raise ValueError(f"unsupported split: {self.split!r}")
        if self.lane_id not in _LANES:
            raise ValueError(f"unsupported lane_id: {self.lane_id!r}")
        if self.fixture_case not in _FIXTURE_CASES:
            raise ValueError(f"unsupported fixture_case: {self.fixture_case!r}")
        if not self.prediction:
            raise ValueError("prediction must contain at least one horizon record")
        horizons = [record.horizon_step for record in self.prediction]
        if len(set(horizons)) != len(horizons):
            raise ValueError("prediction horizon_step values must be unique per trace")
        object.__setattr__(self, "seed", int(self.seed))

    def validate_hard_floor(self, hard_floor_m: float) -> None:
        """Reject a trace whose uncertainty term weakens the deterministic hard floor."""
        if self.planning.status != "available":
            return
        floor = _finite_non_negative("hard_floor_m", hard_floor_m)
        deterministic = self.planning.deterministic_margin_m
        effective = self.planning.effective_margin_m
        if deterministic is None or effective is None:
            raise ValueError(f"trace {self.trace_id!r} has incomplete available planning margins")
        if deterministic + 1e-12 < floor:
            raise ValueError(
                f"trace {self.trace_id!r} deterministic margin is below hard floor {floor}"
            )
        if effective + 1e-12 < max(floor, deterministic):
            raise ValueError(f"trace {self.trace_id!r} effective margin weakens hard floor")


@dataclass(frozen=True, slots=True)
class PredictionCoverageSummary:
    """Held-out horizon coverage summary with an explicit unavailable status."""

    horizon_step: int
    calibration_count: int
    evaluation_count: int
    covered_count: int
    miss_count: int
    radius_m: float | None
    radius_status: Literal["finite", "infinite", "unavailable"]
    coverage_target: float
    empirical_coverage: float | None
    status: Literal["available", "under_covered", "unavailable"]


@dataclass(frozen=True, slots=True)
class LaneOutcomeSummary:
    """Outcome and runtime-event summary for one evaluation lane."""

    lane_id: LaneId
    trace_count: int
    collision_rate: float | None
    near_miss_rate: float | None
    path_efficiency_mean: float | None
    pedestrian_disruption_mean: float | None
    unnecessary_braking_rate: float | None
    unavailable_fields: tuple[str, ...]
    event_counts: Mapping[str, int]


def _json_compatible(value: Any) -> Any:
    """Convert nested dataclass output to JSON-native lists and mappings.

    Returns:
        JSON-compatible representation of ``value``.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PredictionPlanningSafetyDiagnosticReport:
    """Complete fixture/trace diagnostic report for issue #7317."""

    schema_version: str
    evidence_tier: str
    hard_floor_m: float
    coverage_target: float
    split_provenance: Mapping[str, Any]
    prediction_coverage: tuple[PredictionCoverageSummary, ...]
    lanes: tuple[LaneOutcomeSummary, ...]
    same_seed_comparison: Mapping[str, Any]
    fixture_case_counts: Mapping[str, int]
    provenance: Mapping[str, Any]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report payload."""
        return _json_compatible(asdict(self))


def _validate_split_assignments(
    traces: Sequence[PredictionPlanningSafetyTrace],
    *,
    fit_trace_ids: Collection[str],
    calibration_trace_ids: Collection[str],
    evaluation_trace_ids: Collection[str],
) -> dict[str, tuple[PredictionPlanningSafetyTrace, ...]]:
    """Validate unique trace identities and declared fit/calibration/evaluation splits.

    Returns:
        Trace rows partitioned under ``fit``, ``calibration``, and ``evaluation``.
    """
    partitions = {
        "fit": set(fit_trace_ids),
        "calibration": set(calibration_trace_ids),
        "evaluation": set(evaluation_trace_ids),
    }
    if any(not values for values in partitions.values()):
        raise ValueError("fit, calibration, and evaluation trace-id sets must be non-empty")
    names = tuple(partitions)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = partitions[left_name] & partitions[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} trace identities overlap: "
                    f"{', '.join(sorted(overlap))}"
                )
    by_id: dict[str, PredictionPlanningSafetyTrace] = {}
    for trace in traces:
        if trace.trace_id in by_id:
            raise ValueError(f"trace_id values must be unique: {trace.trace_id}")
        by_id[trace.trace_id] = trace
    declared = set().union(*partitions.values())
    observed = set(by_id)
    if declared != observed:
        missing = sorted(declared - observed)
        extra = sorted(observed - declared)
        raise ValueError(f"trace identity assignment mismatch: missing={missing}, extra={extra}")
    result: dict[str, tuple[PredictionPlanningSafetyTrace, ...]] = {}
    for split_name, identifiers in partitions.items():
        selected = tuple(by_id[trace_id] for trace_id in sorted(identifiers))
        if any(trace.split != split_name for trace in selected):
            raise ValueError(
                f"declared {split_name} trace identities must carry split={split_name!r}"
            )
        result[split_name] = selected
    return result


def _prediction_records(
    traces: Sequence[PredictionPlanningSafetyTrace],
) -> dict[int, list[PredictionHorizonTrace]]:
    """Group prediction records by horizon step.

    Returns:
        Mapping from horizon step to its trace records.
    """
    grouped: dict[int, list[PredictionHorizonTrace]] = defaultdict(list)
    for trace in traces:
        for record in trace.prediction:
            grouped[record.horizon_step].append(record)
    return grouped


def _coverage_rows(
    calibration: Sequence[PredictionPlanningSafetyTrace],
    evaluation: Sequence[PredictionPlanningSafetyTrace],
    *,
    coverage_target: float,
) -> tuple[PredictionCoverageSummary, ...]:
    """Fit horizon-wise split-conformal radii and evaluate them on held-out traces.

    Returns:
        Ordered coverage summaries, one per observed horizon step.
    """
    calibration_by_horizon = _prediction_records(calibration)
    evaluation_by_horizon = _prediction_records(evaluation)
    rows: list[PredictionCoverageSummary] = []
    for horizon_step in sorted(set(calibration_by_horizon) | set(evaluation_by_horizon)):
        calibration_records = calibration_by_horizon.get(horizon_step, [])
        evaluation_records = evaluation_by_horizon.get(horizon_step, [])
        calibration_scores = [
            record.realized_error_m
            for record in calibration_records
            if record.status == "available" and record.realized_error_m is not None
        ]
        evaluation_scores = [
            record.realized_error_m
            for record in evaluation_records
            if record.status == "available" and record.realized_error_m is not None
        ]
        if not calibration_scores or not evaluation_scores:
            rows.append(
                PredictionCoverageSummary(
                    horizon_step=horizon_step,
                    calibration_count=len(calibration_scores),
                    evaluation_count=len(evaluation_scores),
                    covered_count=0,
                    miss_count=0,
                    radius_m=None,
                    radius_status="unavailable",
                    coverage_target=coverage_target,
                    empirical_coverage=None,
                    status="unavailable",
                )
            )
            continue
        radius = split_conformal_radius(
            np.asarray(calibration_scores, dtype=np.float64),
            coverage_target=coverage_target,
        )
        covered_count = sum(score <= radius for score in evaluation_scores)
        miss_count = len(evaluation_scores) - covered_count
        empirical = covered_count / len(evaluation_scores)
        rows.append(
            PredictionCoverageSummary(
                horizon_step=horizon_step,
                calibration_count=len(calibration_scores),
                evaluation_count=len(evaluation_scores),
                covered_count=covered_count,
                miss_count=miss_count,
                radius_m=None if math.isinf(radius) else float(radius),
                radius_status="infinite" if math.isinf(radius) else "finite",
                coverage_target=coverage_target,
                empirical_coverage=float(empirical),
                status="available" if empirical >= coverage_target else "under_covered",
            )
        )
    return tuple(rows)


def _rate(values: Sequence[bool | None]) -> float | None:
    """Compute a boolean rate over available values, or preserve unavailable status.

    Returns:
        Available-value rate, or ``None`` when no value was recorded.
    """
    available = [value for value in values if value is not None]
    if not available:
        return None
    return float(sum(available) / len(available))


def _mean(values: Sequence[float | None]) -> float | None:
    """Compute a mean over available scalar values, or preserve unavailable status.

    Returns:
        Available-value mean, or ``None`` when no value was recorded.
    """
    available = [float(value) for value in values if value is not None]
    return None if not available else float(np.mean(available))


def _lane_summary(
    lane_id: LaneId,
    traces: Sequence[PredictionPlanningSafetyTrace],
) -> LaneOutcomeSummary:
    """Summarize one lane without imputing absent outcome fields.

    Returns:
        Typed lane summary with event counts and unavailable-field names.
    """
    event_counts = Counter(trace.runtime_safety.status for trace in traces)
    event_counts["prediction_out_of_support"] = sum(
        record.out_of_support for trace in traces for record in trace.prediction
    )
    unavailable_fields = tuple(
        field
        for field in _OUTCOME_FIELDS
        if any(getattr(trace.outcome, field) is None for trace in traces)
    )
    return LaneOutcomeSummary(
        lane_id=lane_id,
        trace_count=len(traces),
        collision_rate=_rate([trace.outcome.collision for trace in traces]),
        near_miss_rate=_rate([trace.outcome.near_miss for trace in traces]),
        path_efficiency_mean=_mean([trace.outcome.path_efficiency for trace in traces]),
        pedestrian_disruption_mean=_mean([trace.outcome.pedestrian_disruption for trace in traces]),
        unnecessary_braking_rate=_rate([trace.outcome.unnecessary_braking for trace in traces]),
        unavailable_fields=unavailable_fields,
        event_counts=dict(sorted(event_counts.items())),
    )


def _same_seed_comparison(
    evaluation: Sequence[PredictionPlanningSafetyTrace],
) -> dict[str, Any]:
    """Compare baseline and uncertainty-aware traces on identical scenario/seed keys.

    Returns:
        JSON-compatible paired comparison, or an explicit unavailable status.
    """
    by_lane: dict[str, dict[tuple[str, int, str], PredictionPlanningSafetyTrace]] = {
        "baseline": {},
        "uncertainty_aware": {},
    }
    for trace in evaluation:
        key = (trace.scenario_id, trace.seed, trace.fixture_case)
        lane_rows = by_lane[trace.lane_id]
        if key in lane_rows:
            raise ValueError(f"duplicate same-seed comparison key: {key}")
        lane_rows[key] = trace
    shared = sorted(set(by_lane["baseline"]) & set(by_lane["uncertainty_aware"]))
    if not shared:
        return {"status": "unavailable", "paired_trace_count": 0, "rows": []}
    rows = []
    for key in shared:
        baseline = by_lane["baseline"][key]
        uncertainty = by_lane["uncertainty_aware"][key]
        rows.append(
            {
                "scenario_id": key[0],
                "seed": key[1],
                "fixture_case": key[2],
                "baseline_trace_id": baseline.trace_id,
                "uncertainty_aware_trace_id": uncertainty.trace_id,
                "baseline_runtime_status": baseline.runtime_safety.status,
                "uncertainty_aware_runtime_status": uncertainty.runtime_safety.status,
                "baseline_collision": baseline.outcome.collision,
                "uncertainty_aware_collision": uncertainty.outcome.collision,
            }
        )
    return {
        "status": "paired_fixture_diagnostic",
        "paired_trace_count": len(rows),
        "rows": rows,
        "claim_boundary": "paired fixture behavior only; not a navigation comparison",
    }


def _chance_constrained_provenance() -> dict[str, Any]:
    """Record the canonical chance-constrained builder without executing a campaign.

    Returns:
        Provenance payload identifying the referenced planner configuration.
    """
    from robot_sf.planner.chance_constrained_mpc import (  # noqa: PLC0415
        build_chance_constrained_mpc_config,
    )

    config = build_chance_constrained_mpc_config(
        {
            "predictor_backend": "constant_velocity_gmm",
            "chance_constraint_formulation": "marginal",
        }
    )
    return {
        "source": "robot_sf.planner.chance_constrained_mpc.build_chance_constrained_mpc_config",
        "status": "referenced_not_executed_fixture_only",
        "predictor_backend": config.predictor_backend,
        "chance_constraint_formulation": config.chance_constraint_formulation,
        "max_collision_risk": float(config.max_collision_risk),
    }


def build_prediction_planning_safety_diagnostic(
    *,
    traces: Sequence[PredictionPlanningSafetyTrace],
    fit_trace_ids: Collection[str],
    calibration_trace_ids: Collection[str],
    evaluation_trace_ids: Collection[str],
    hard_floor_m: float,
    coverage_target: float = 0.9,
    seed: int = 0,
) -> PredictionPlanningSafetyDiagnosticReport:
    """Build the typed prediction/planning/runtime diagnostic report.

    The function requires an explicit disjoint identity assignment for fit, calibration,
    and evaluation traces.  It measures forecast coverage only from calibration/evaluation
    residuals, verifies the hard-floor monotonicity of every available planner record, and
    reports missing outcome fields as unavailable.

    Returns:
        A deterministic, claim-bounded diagnostic report.
    """
    if not 0.0 < float(coverage_target) < 1.0:
        raise ValueError("coverage_target must be between 0 and 1 (exclusive)")
    floor = _finite_non_negative("hard_floor_m", hard_floor_m)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    normalized_traces = tuple(traces)
    partitions = _validate_split_assignments(
        normalized_traces,
        fit_trace_ids=fit_trace_ids,
        calibration_trace_ids=calibration_trace_ids,
        evaluation_trace_ids=evaluation_trace_ids,
    )
    for trace in normalized_traces:
        trace.validate_hard_floor(floor)
    evaluation = partitions["evaluation"]
    lanes = tuple(
        _lane_summary(lane_id, tuple(trace for trace in evaluation if trace.lane_id == lane_id))
        for lane_id in ("baseline", "uncertainty_aware")
        if any(trace.lane_id == lane_id for trace in evaluation)
    )
    coverage_rows = _coverage_rows(
        partitions["calibration"], evaluation, coverage_target=float(coverage_target)
    )
    split_provenance = {
        "fit_trace_ids": sorted(fit_trace_ids),
        "calibration_trace_ids": sorted(calibration_trace_ids),
        "evaluation_trace_ids": sorted(evaluation_trace_ids),
        "pairwise_disjoint": True,
        "fit_trace_count": len(partitions["fit"]),
        "calibration_trace_count": len(partitions["calibration"]),
        "evaluation_trace_count": len(evaluation),
        "identity_leakage_rejected": True,
    }
    fixture_case_counts = dict(sorted(Counter(trace.fixture_case for trace in evaluation).items()))
    return PredictionPlanningSafetyDiagnosticReport(
        schema_version=PREDICTION_PLANNING_SAFETY_SCHEMA_VERSION,
        evidence_tier="smoke/diagnostic",
        hard_floor_m=floor,
        coverage_target=float(coverage_target),
        split_provenance=split_provenance,
        prediction_coverage=coverage_rows,
        lanes=lanes,
        same_seed_comparison=_same_seed_comparison(evaluation),
        fixture_case_counts=fixture_case_counts,
        provenance={
            "seed": int(seed),
            "reused_primitives": {
                "split_conformal_radius": {
                    "schema": SPLIT_CONFORMAL_RADIUS_SCHEMA,
                    "source": "robot_sf.benchmark.uncertainty_safety.split_conformal_radius",
                },
                "chance_constrained_mpc": _chance_constrained_provenance(),
            },
            "runtime_event_vocabulary": sorted(_RUNTIME_STATUSES),
        },
        claim_boundary=(
            "Smoke/diagnostic fixture evidence only. Prediction coverage is empirical and "
            "split-specific; it does not establish a per-encounter or deployment safety "
            "guarantee. The paired lane rows do not establish navigation benefit, collision "
            "reduction, or real-world pedestrian safety. A held-out navigation campaign "
            "requires the approval/preregistration boundary in issue #6647."
        ),
    )


def _fixture_trace(  # noqa: PLR0913
    *,
    trace_id: str,
    scenario_id: str,
    seed: int,
    split: Literal["fit", "calibration", "evaluation"],
    lane_id: LaneId,
    fixture_case: FixtureCase,
    error_m: float,
    runtime: RuntimeSafetyTrace,
    outcome: RealizedOutcomeTrace,
    uncertainty_margin_m: float,
    representation: PredictionRepresentation = "interval",
    out_of_support: bool = False,
) -> PredictionPlanningSafetyTrace:
    """Build one deterministic fixture row with a shared hard-floor baseline.

    Returns:
        A typed fixture trace row.
    """
    return PredictionPlanningSafetyTrace(
        trace_id=trace_id,
        scenario_id=scenario_id,
        seed=seed,
        split=split,
        lane_id=lane_id,
        fixture_case=fixture_case,
        prediction=(
            PredictionHorizonTrace(
                horizon_step=1,
                representation=representation,
                status="available",
                interval_radius_m=0.15 if representation == "interval" else None,
                sample_count=8 if representation == "samples" else 0,
                realized_error_m=error_m,
                out_of_support=out_of_support,
            ),
        ),
        planning=NominalPlanningTrace(
            status="available",
            planner_source=(
                "baseline_nominal_planner"
                if lane_id == "baseline"
                else "chance_constrained_mpc_uncertainty_aware_fixture"
            ),
            deterministic_margin_m=0.3,
            uncertainty_margin_m=uncertainty_margin_m,
            effective_margin_m=0.3 + uncertainty_margin_m,
            command=(0.4, 0.0) if uncertainty_margin_m == 0.0 else (0.2, 0.0),
        ),
        runtime_safety=runtime,
        outcome=outcome,
    )


def build_fixture_traces(*, seed: int = 7317) -> tuple[PredictionPlanningSafetyTrace, ...]:
    """Return a deterministic fixture set for the three mechanism cases in issue #7317."""
    traces: list[PredictionPlanningSafetyTrace] = [
        _fixture_trace(
            trace_id=f"fixture-fit-{seed}",
            scenario_id="fixture_fit_reference",
            seed=seed,
            split="fit",
            lane_id="baseline",
            fixture_case="calibration_reference",
            error_m=0.04,
            runtime=RuntimeSafetyTrace(status="verified", selected_action=(0.4, 0.0)),
            outcome=RealizedOutcomeTrace(),
            uncertainty_margin_m=0.0,
        )
    ]
    for index, error in enumerate((0.05, 0.08, 0.10, 0.12, 0.15)):
        traces.append(
            _fixture_trace(
                trace_id=f"fixture-calibration-{index}-{seed}",
                scenario_id=f"fixture_calibration_{index}",
                seed=seed + index,
                split="calibration",
                lane_id="baseline",
                fixture_case="calibration_reference",
                error_m=error,
                runtime=RuntimeSafetyTrace(status="verified", selected_action=(0.4, 0.0)),
                outcome=RealizedOutcomeTrace(),
                uncertainty_margin_m=0.0,
            )
        )
    case_specs = (
        (
            "good_prediction_poor_planning",
            0.08,
            False,
            RuntimeSafetyTrace(status="verified", selected_action=(0.4, 0.0)),
            RuntimeSafetyTrace(status="verified", selected_action=(0.2, 0.0)),
            RealizedOutcomeTrace(
                collision=True,
                near_miss=True,
                path_efficiency=0.75,
                pedestrian_disruption=0.3,
                unnecessary_braking=False,
            ),
            RealizedOutcomeTrace(
                collision=False,
                near_miss=False,
                path_efficiency=0.70,
                pedestrian_disruption=0.2,
                unnecessary_braking=True,
            ),
            0.10,
        ),
        (
            "poor_prediction_safe_fallback",
            0.60,
            True,
            RuntimeSafetyTrace(status="verified", selected_action=(0.4, 0.0)),
            RuntimeSafetyTrace(
                status="contingency_invoked",
                selected_action=(0.0, 0.5),
                contingency_action="stop_keep_turn",
            ),
            RealizedOutcomeTrace(
                collision=True,
                near_miss=True,
                path_efficiency=0.60,
                pedestrian_disruption=0.4,
                unnecessary_braking=False,
            ),
            RealizedOutcomeTrace(
                collision=False,
                near_miss=True,
                path_efficiency=0.55,
                pedestrian_disruption=0.1,
                unnecessary_braking=True,
            ),
            0.15,
        ),
        (
            "verification_unavailable",
            0.10,
            False,
            RuntimeSafetyTrace(status="verification_unavailable", reason="missing_runtime_trace"),
            RuntimeSafetyTrace(status="verification_unavailable", reason="missing_runtime_trace"),
            RealizedOutcomeTrace(),
            RealizedOutcomeTrace(),
            0.05,
        ),
    )
    for index, (
        case,
        error,
        out_of_support,
        baseline_runtime,
        uncertainty_runtime,
        baseline_outcome,
        uncertainty_outcome,
        uncertainty_margin,
    ) in enumerate(case_specs):
        scenario_id = f"fixture_{case}"
        traces.extend(
            (
                _fixture_trace(
                    trace_id=f"fixture-{case}-baseline-{seed}",
                    scenario_id=scenario_id,
                    seed=seed + 100 + index,
                    split="evaluation",
                    lane_id="baseline",
                    fixture_case=case,
                    error_m=error,
                    runtime=baseline_runtime,
                    outcome=baseline_outcome,
                    uncertainty_margin_m=0.0,
                    representation="point",
                    out_of_support=out_of_support,
                ),
                _fixture_trace(
                    trace_id=f"fixture-{case}-uncertainty-aware-{seed}",
                    scenario_id=scenario_id,
                    seed=seed + 100 + index,
                    split="evaluation",
                    lane_id="uncertainty_aware",
                    fixture_case=case,
                    error_m=error,
                    runtime=uncertainty_runtime,
                    outcome=uncertainty_outcome,
                    uncertainty_margin_m=uncertainty_margin,
                    representation="interval",
                    out_of_support=out_of_support,
                ),
            )
        )
    return tuple(traces)


def build_fixture_diagnostic_report(
    *,
    seed: int = 7317,
    coverage_target: float = 0.8,
    hard_floor_m: float = 0.3,
) -> PredictionPlanningSafetyDiagnosticReport:
    """Build the canonical deterministic fixture report for issue #7317.

    Returns:
        Deterministic fixture-only diagnostic report.
    """
    traces = build_fixture_traces(seed=seed)
    return build_prediction_planning_safety_diagnostic(
        traces=traces,
        fit_trace_ids={trace.trace_id for trace in traces if trace.split == "fit"},
        calibration_trace_ids={trace.trace_id for trace in traces if trace.split == "calibration"},
        evaluation_trace_ids={trace.trace_id for trace in traces if trace.split == "evaluation"},
        hard_floor_m=hard_floor_m,
        coverage_target=coverage_target,
        seed=seed,
    )


def validate_prediction_planning_safety_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized report against the versioned JSON schema.

    Returns:
        The validated mapping copied into a plain dictionary.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("prediction/planning/safety report must be a mapping")
    schema_path = (
        Path(__file__).parent.parent / "schemas" / "prediction_planning_safety.schema.v1.json"
    )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(dict(payload))
    except jsonschema.ValidationError as exc:
        raise ValueError(
            f"prediction/planning/safety report schema validation failed: {exc.message}"
        ) from exc
    except jsonschema.SchemaError as exc:
        raise ValueError("prediction/planning/safety report schema is invalid") from exc
    return dict(payload)


__all__ = [
    "PREDICTION_PLANNING_SAFETY_SCHEMA_VERSION",
    "LaneOutcomeSummary",
    "NominalPlanningTrace",
    "PredictionCoverageSummary",
    "PredictionHorizonTrace",
    "PredictionPlanningSafetyDiagnosticReport",
    "PredictionPlanningSafetyTrace",
    "RealizedOutcomeTrace",
    "RuntimeSafetyTrace",
    "build_fixture_diagnostic_report",
    "build_fixture_traces",
    "build_prediction_planning_safety_diagnostic",
    "validate_prediction_planning_safety_report",
]
