"""Offline calibration and comparison of safety-margin constructions (issue #6581).

A single fixed clearance threshold is unlikely to stay appropriate across pedestrian
density, visibility, speed, footprint, and localization uncertainty. This module is the
offline, deterministic *comparison owner* for three margin constructions over small
synthetic or tracked-fixture traces:

1. **Fixed** -- a single constant clearance, clamped to the hard floor.
2. **Context-adaptive (preferred)** -- a non-negative, context-dependent additive margin
   whose feature normalization is fit on the ``fit`` split only.
3. **Adaptive plus conformal** -- the preferred margin plus a split-conformal tightening
   calibrated on the ``calibration`` split from nominal-to-perturbed residual scores.

It deliberately *reuses* the already-delivered conformal primitives from
:mod:`robot_sf.benchmark.uncertainty_safety` (:func:`split_conformal_radius` and
:func:`adaptive_conformal_buffers`) without modifying them, and carries their
schema/version plus the distributional assumptions in provenance.

.. admonition:: Claim boundary
   :class: note

   This is **smoke/diagnostic only**. It proves implementation integrity: disjoint
   fit/calibration/evaluation identifiers, hard-floor monotonicity, deterministic
   report output, and honest reporting of absent outcome fields. It does *not* prove
   improved navigation safety, calibrated deployment coverage, or real-world risk
   reduction. A held-out navigation campaign and any such claim require a separate
   preregistration and approval. Collision, pedestrian-disruption, braking, and
   efficiency outcome fields are reported as ``unavailable`` when absent and are never
   synthesized; fallback or degraded data never counts as evidence here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.finite_checks import require_finite_array, require_finite_scalar
from robot_sf.benchmark.uncertainty_safety import (
    ADAPTIVE_CONFORMAL_BUFFERS_SCHEMA,
    SPLIT_CONFORMAL_RADIUS_SCHEMA,
    AdaptiveConformalConfig,
    adaptive_conformal_buffers,
    split_conformal_radius,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

SAFETY_MARGIN_CALIBRATION_SCHEMA = "safety_margin_calibration.comparison.v1"

#: Default non-negative weights (metres) for the preferred-margin feature terms. Every
#: term is non-negative, so the preferred margin can only widen the clearance.
DEFAULT_PREFERRED_WEIGHTS: Mapping[str, float] = {
    "pedestrian_density": 0.10,
    "robot_speed_mps": 0.10,
    "localization_uncertainty_m": 0.20,
    "footprint_radius_m": 0.05,
    "low_visibility": 0.10,
}

#: Small floor preventing division by zero when a fit feature is degenerate (all zeros).
_FEATURE_SCALE_FLOOR = 1e-9

METHOD_FIXED = "fixed"
METHOD_ADAPTIVE = "adaptive"
METHOD_ADAPTIVE_CONFORMAL = "adaptive_conformal"

#: Conditioning label for trace-supplied outcomes (not margin-conditioned campaign results).
TRACE_SUPPLIED_CONDITIONING = "trace_supplied_not_margin_conditioned"


# --------------------------------------------------------------------------- context


@dataclass(frozen=True)
class MarginContext:
    """Context descriptor driving the adaptive preferred safety margin.

    All fields are non-negative physical quantities. ``visibility_m`` is the effective
    sensing or line-of-sight range; *lower* visibility increases the preferred margin
    (a robot that sees less keeps more clearance). The remaining fields increase the
    preferred margin monotonically.

    Attributes:
        pedestrian_density: Crowd density (pedestrians per square metre, >= 0).
        visibility_m: Effective visibility in metres (> 0).
        robot_speed_mps: Robot speed in metres per second (>= 0).
        footprint_radius_m: Robot footprint (inscribing-circle) radius in metres (> 0).
        localization_uncertainty_m: Localization 1-sigma uncertainty in metres (>= 0).
    """

    pedestrian_density: float
    visibility_m: float
    robot_speed_mps: float
    footprint_radius_m: float
    localization_uncertainty_m: float

    def __post_init__(self) -> None:
        """Validate that all context fields are finite and within physical bounds."""
        density = require_finite_scalar("pedestrian_density", self.pedestrian_density)
        visibility = require_finite_scalar("visibility_m", self.visibility_m)
        speed = require_finite_scalar("robot_speed_mps", self.robot_speed_mps)
        footprint = require_finite_scalar("footprint_radius_m", self.footprint_radius_m)
        unc = require_finite_scalar("localization_uncertainty_m", self.localization_uncertainty_m)
        if density < 0.0:
            raise ValueError(f"pedestrian_density must be non-negative: {density}")
        if visibility <= 0.0:
            raise ValueError(f"visibility_m must be positive: {visibility}")
        if speed < 0.0:
            raise ValueError(f"robot_speed_mps must be non-negative: {speed}")
        if footprint <= 0.0:
            raise ValueError(f"footprint_radius_m must be positive: {footprint}")
        if unc < 0.0:
            raise ValueError(f"localization_uncertainty_m must be non-negative: {unc}")


@dataclass(frozen=True)
class SafetyMarginTraceSample:
    """One trace observation used for fit, calibration, or evaluation.

    ``residual_m`` is the nonconformity score (nominal-to-perturbed trajectory
    deviation magnitude) used to fit and evaluate conformal margins; it is required on
    calibration and evaluation traces and may be ``None`` on fit traces (which only
    contribute context). The outcome fields are optional *trace-supplied* diagnostics;
    ``None`` means the field was not recorded and is reported as ``unavailable``,
    never synthesized or imputed.

    Attributes:
        split_id: Identifier of the split this trace belongs to (fit/calibration/eval).
        context: Context descriptor active at this trace.
        residual_m: Nonconformity score in metres, or ``None`` for fit-only traces.
        collision: Whether a collision was recorded, or ``None`` if unmeasured.
        near_miss: Whether a near miss was recorded, or ``None`` if unmeasured.
        path_efficiency: Path-efficiency ratio in ``[0, 1]``, or ``None`` if unmeasured.
        pedestrian_disruption: Recorded pedestrian-disruption score, or ``None``.
        unnecessary_braking: Whether unnecessary braking was recorded, or ``None``.
    """

    split_id: str
    context: MarginContext
    residual_m: float | None = None
    collision: bool | None = None
    near_miss: bool | None = None
    path_efficiency: float | None = None
    pedestrian_disruption: float | None = None
    unnecessary_braking: bool | None = None

    def __post_init__(self) -> None:
        """Validate the split id and the optional outcome fields.

        Residual finiteness is intentionally deferred to
        :func:`build_safety_margin_comparison`, which rejects non-finite
        calibration/evaluation residuals via :func:`require_finite_array` so the
        ``finite_checks`` boundary is exercised on the computation path.
        """
        if not isinstance(self.split_id, str) or not self.split_id:
            raise ValueError("split_id must be a non-empty string")
        if self.path_efficiency is not None:
            eff = require_finite_scalar("path_efficiency", self.path_efficiency)
            if not 0.0 <= eff <= 1.0:
                raise ValueError(f"path_efficiency must be in [0, 1]: {eff}")
            object.__setattr__(self, "path_efficiency", eff)
        if self.pedestrian_disruption is not None:
            disrupt = require_finite_scalar("pedestrian_disruption", self.pedestrian_disruption)
            if disrupt < 0.0:
                raise ValueError(f"pedestrian_disruption must be non-negative: {disrupt}")
            object.__setattr__(self, "pedestrian_disruption", disrupt)


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class MarginMethodResult:
    """Per-method margin construction summary over the evaluation split.

    Attributes:
        method: Construction name (``fixed``/``adaptive``/``adaptive_conformal``).
        hard_floor_m: Immutable hard floor echoed for this method.
        preferred_margin_m: Mean preferred additive margin over evaluation contexts
            (``0.0`` for the fixed method).
        conformal_tightening_m: Scalar split-conformal radius (``0.0`` for fixed and
            adaptive methods; may be ``+inf`` when the target cannot be certified).
        effective_margin_m: Mean effective margin over evaluation contexts.
        min_effective_margin_m: Minimum effective margin; never below ``hard_floor_m``.
        target_coverage: Conformal coverage target, or ``None`` for non-conformal methods.
        empirical_coverage: Fraction of evaluation residuals within the margin, or
            ``None`` when no evaluation residuals are available.
        coverage_gap: ``empirical_coverage - target_coverage``, or ``None``.
        coverage_status: Coverage classification string (see ``_coverage_status``).
        efficiency_cost: Efficiency proxy = mean effective margin (more margin is
            safer but less efficient). May be ``+inf`` when the conformal radius is.
        violated_constraints: Hard-floor reductions; empty by construction.
        schema: Versioned schema identifier.
    """

    method: str
    hard_floor_m: float
    preferred_margin_m: float
    conformal_tightening_m: float
    effective_margin_m: float
    min_effective_margin_m: float
    target_coverage: float | None
    empirical_coverage: float | None
    coverage_gap: float | None
    coverage_status: str
    efficiency_cost: float
    violated_constraints: list[str]
    schema: str = SAFETY_MARGIN_CALIBRATION_SCHEMA


@dataclass(frozen=True)
class EvaluationOutcomeSummary:
    """Trace-supplied evaluation outcomes (diagnostics, not margin-conditioned results).

    Each rate or mean is computed over the evaluation traces that supplied the field.
    Fields no trace supplied are reported as ``None`` and listed as ``unavailable`` in
    :attr:`field_availability`; they are never synthesized or imputed. These describe
    the recorded traces only -- a campaign that re-runs each margin is out of scope.

    Attributes:
        conditioning: Always :data:`TRACE_SUPPLIED_CONDITIONING`.
        evaluation_trace_count: Number of traces in the evaluation split.
        collision_rate: Mean collision indicator over supplying traces, or ``None``.
        near_miss_rate: Mean near-miss indicator over supplying traces, or ``None``.
        mean_path_efficiency: Mean path efficiency over supplying traces, or ``None``.
        mean_pedestrian_disruption: Mean disruption over supplying traces, or ``None``.
        unnecessary_braking_rate: Mean braking indicator over supplying traces, or ``None``.
        field_availability: Mapping of field name to ``"available"``/``"unavailable"``.
    """

    conditioning: str
    evaluation_trace_count: int
    collision_rate: float | None
    near_miss_rate: float | None
    mean_path_efficiency: float | None
    mean_pedestrian_disruption: float | None
    unnecessary_braking_rate: float | None
    field_availability: Mapping[str, str]


@dataclass(frozen=True)
class SafetyMarginCalibrationReport:
    """Deterministic comparison report for the three margin constructions.

    Attributes:
        schema_version: Versioned report schema identifier.
        hard_floor_m: Immutable hard floor applied to every method.
        coverage_target: Requested conformal coverage target.
        fixed_margin_m: Fixed-baseline margin used by the ``fixed`` method.
        methods: One :class:`MarginMethodResult` per construction, ordered
            fixed, adaptive, adaptive_conformal.
        evaluation_outcomes: Trace-supplied outcome diagnostics.
        split_provenance: Disjoint split identifiers, trace counts, and leakage check.
        provenance: Reused primitive schemas, distributional assumptions, and seed.
        claim_boundary: Plain-language claim boundary and evidence-tier statement.
    """

    schema_version: str
    hard_floor_m: float
    coverage_target: float
    fixed_margin_m: float
    methods: list[MarginMethodResult]
    evaluation_outcomes: EvaluationOutcomeSummary
    split_provenance: Mapping[str, Any]
    provenance: Mapping[str, Any]
    claim_boundary: str


# --------------------------------------------------------------------------- split guards


def _to_id_set(name: str, ids: Collection[str]) -> set[str]:
    """Validate and return a non-empty set of split identifiers.

    Args:
        name: Parameter name for error messages.
        ids: Collection of split identifiers.

    Returns:
        The de-duplicated set of identifiers.

    Raises:
        ValueError: If the collection is empty or contains non-string/blank identifiers.
    """
    out: set[str] = set()
    for raw in ids:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{name} must contain only non-empty string identifiers")
        out.add(raw)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _validate_disjoint_splits(
    fit_ids: set[str],
    calibration_ids: set[str],
    evaluation_ids: set[str],
) -> None:
    """Reject any overlap between the fit, calibration, and evaluation identifier sets.

    Args:
        fit_ids: Fit split identifiers.
        calibration_ids: Calibration split identifiers.
        evaluation_ids: Evaluation split identifiers.

    Raises:
        ValueError: If any identifier appears in more than one split (leakage).
    """
    pairs = (
        ("fit", fit_ids, "calibration", calibration_ids),
        ("fit", fit_ids, "evaluation", evaluation_ids),
        ("calibration", calibration_ids, "evaluation", evaluation_ids),
    )
    for left_name, left, right_name, right in pairs:
        overlap = left & right
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(
                f"{left_name} and {right_name} split identifiers must be disjoint; "
                f"overlap leaked: {joined}"
            )


def _partition_traces(
    traces: Sequence[SafetyMarginTraceSample],
    *,
    fit_ids: set[str],
    calibration_ids: set[str],
    evaluation_ids: set[str],
) -> tuple[
    list[SafetyMarginTraceSample], list[SafetyMarginTraceSample], list[SafetyMarginTraceSample]
]:
    """Partition traces by split membership, rejecting unknown split ids.

    Args:
        traces: All trace samples.
        fit_ids: Fit split identifiers.
        calibration_ids: Calibration split identifiers.
        evaluation_ids: Evaluation split identifiers.

    Returns:
        ``(fit, calibration, evaluation)`` trace lists.

    Raises:
        ValueError: If a trace carries a ``split_id`` not in any provided set.
    """
    known = fit_ids | calibration_ids | evaluation_ids
    fit: list[SafetyMarginTraceSample] = []
    calibration: list[SafetyMarginTraceSample] = []
    evaluation: list[SafetyMarginTraceSample] = []
    for sample in traces:
        sid = sample.split_id
        if sid not in known:
            raise ValueError(f"trace split_id {sid!r} is not in any fit/calibration/evaluation set")
        if sid in fit_ids:
            fit.append(sample)
        elif sid in calibration_ids:
            calibration.append(sample)
        else:
            evaluation.append(sample)
    return fit, calibration, evaluation


def _require_non_empty_split(name: str, samples: Sequence[SafetyMarginTraceSample]) -> None:
    """Fail closed when a split has no traces.

    Args:
        name: Split name for error messages.
        samples: Traces assigned to the split.

    Raises:
        ValueError: If the split is empty.
    """
    if not samples:
        raise ValueError(f"{name} split must contain at least one trace")


# --------------------------------------------------------------------------- preferred margin


def _feature_scales(contexts: Sequence[MarginContext]) -> dict[str, float]:
    """Per-feature normalization scales derived from the fit split only.

    The scale is the mean of each feature over the fit contexts with a positive floor
    so a degenerate (all-zero) feature does not divide by zero. Using fit statistics
    only keeps calibration/evaluation residuals from leaking into the margin model.

    Args:
        contexts: Fit-split contexts.

    Returns:
        Mapping of feature name to positive scale.

    Raises:
        ValueError: If no fit contexts are supplied.
    """
    if not contexts:
        raise ValueError("fit contexts must not be empty")
    n = float(len(contexts))
    sums = {
        "pedestrian_density": sum(c.pedestrian_density for c in contexts),
        "visibility_m": sum(c.visibility_m for c in contexts),
        "robot_speed_mps": sum(c.robot_speed_mps for c in contexts),
        "footprint_radius_m": sum(c.footprint_radius_m for c in contexts),
        "localization_uncertainty_m": sum(c.localization_uncertainty_m for c in contexts),
    }
    return {key: max(value / n, _FEATURE_SCALE_FLOOR) for key, value in sums.items()}


def _preferred_margin(
    context: MarginContext,
    *,
    scales: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Non-negative additive preferred margin (metres) for one context.

    Every term is non-negative, so the result is ``>= 0`` and can only widen the
    clearance above the hard floor. Lower visibility increases the margin.

    Args:
        context: Evaluation context.
        scales: Fit-derived per-feature normalization scales.
        weights: Non-negative per-feature weights (metres).

    Returns:
        Preferred margin in metres (``>= 0``).

    Raises:
        ValueError: If any weight is negative.
    """
    for key, value in weights.items():
        if require_finite_scalar(f"weights[{key!r}]", value) < 0.0:
            raise ValueError(f"preferred weights must be non-negative: {key}={value}")
    density_term = weights["pedestrian_density"] * (
        context.pedestrian_density / scales["pedestrian_density"]
    )
    speed_term = weights["robot_speed_mps"] * (context.robot_speed_mps / scales["robot_speed_mps"])
    unc_term = weights["localization_uncertainty_m"] * (
        context.localization_uncertainty_m / scales["localization_uncertainty_m"]
    )
    foot_term = weights["footprint_radius_m"] * (
        context.footprint_radius_m / scales["footprint_radius_m"]
    )
    # Visibility below the fit-scale reference widens the margin; at or above it adds 0.
    low_visibility = max(0.0, 1.0 - context.visibility_m / scales["visibility_m"])
    vis_term = weights["low_visibility"] * low_visibility
    return float(max(0.0, density_term + speed_term + unc_term + foot_term + vis_term))


# --------------------------------------------------------------------------- helpers


def _effective_margins(
    method: str,
    *,
    preferred: Sequence[float],
    conformal_tightening: float,
    hard_floor_m: float,
    fixed_margin_m: float,
    evaluation_count: int,
) -> tuple[list[float], list[str]]:
    """Compute per-context effective margins and any hard-floor violations.

    The effective margin is ``max(hard_floor, hard_floor + preferred + conformal)`` for
    adaptive methods and ``max(hard_floor, fixed_margin)`` for the fixed method. Because
    preferred and conformal terms are non-negative, the floor is never reduced; the
    violation list is therefore empty by construction but is computed explicitly so the
    monotonicity contract is observable.

    Args:
        method: Construction name.
        preferred: Per-context preferred margins (empty for the fixed method).
        conformal_tightening: Scalar conformal radius (``0.0`` unless conformal method).
        hard_floor_m: Immutable hard floor.
        fixed_margin_m: Fixed-baseline margin.
        evaluation_count: Number of evaluation contexts (one margin per context).

    Returns:
        ``(margins, violations)`` where violations lists hard-floor reductions.

    Raises:
        ValueError: If adaptive ``preferred`` does not align with ``evaluation_count``.
    """
    violations: list[str] = []
    if method == METHOD_FIXED:
        effective = max(hard_floor_m, fixed_margin_m)
        margins = [effective] * evaluation_count
    else:
        if len(preferred) != evaluation_count:
            raise ValueError("preferred margins must align one-to-one with evaluation contexts")
        conformal = conformal_tightening if method == METHOD_ADAPTIVE_CONFORMAL else 0.0
        margins = []
        for preferred_i in preferred:
            candidate = hard_floor_m + preferred_i + conformal
            effective_i = max(hard_floor_m, candidate)
            if effective_i < hard_floor_m:
                violations.append("hard_floor_reduced")
            margins.append(effective_i)
    return margins, violations


def _coverage_status(
    *,
    method: str,
    conformal_tightening: float,
    empirical_coverage: float | None,
    coverage_target: float | None,
    coverage_gap: float | None,
) -> str:
    """Classify coverage for one method.

    Args:
        method: Construction name.
        conformal_tightening: Split-conformal radius used by the method.
        empirical_coverage: Measured evaluation coverage, or ``None``.
        coverage_target: Target for conformal methods, or ``None``.
        coverage_gap: ``empirical - target``, or ``None``.

    Returns:
        Coverage status string.
    """
    if empirical_coverage is None:
        return "unavailable_no_evaluation_residuals"
    if method == METHOD_ADAPTIVE_CONFORMAL and math.isinf(conformal_tightening):
        # An unbounded radius covers every finite evaluation residual trivially, but it
        # means the calibration sample could not certify the requested target with a
        # finite usable margin.  Do not represent that fail-closed result as coverage
        # success in a smoke report.
        return "uncertifiable_infinite_radius"
    if method != METHOD_ADAPTIVE_CONFORMAL or coverage_target is None or coverage_gap is None:
        return "diagnostic_no_target"
    if coverage_gap < 0.0:
        return "under_covered_evaluation"
    return "covered_evaluation_smoke"


def _evaluation_outcomes(
    evaluation: Sequence[SafetyMarginTraceSample],
) -> EvaluationOutcomeSummary:
    """Summarize trace-supplied evaluation outcomes, marking absent fields unavailable.

    Args:
        evaluation: Evaluation-split traces.

    Returns:
        Populated :class:`EvaluationOutcomeSummary`.
    """
    collisions = [bool(t.collision) for t in evaluation if t.collision is not None]
    near_misses = [bool(t.near_miss) for t in evaluation if t.near_miss is not None]
    efficiencies = [float(t.path_efficiency) for t in evaluation if t.path_efficiency is not None]
    disruptions = [
        float(t.pedestrian_disruption) for t in evaluation if t.pedestrian_disruption is not None
    ]
    braking = [bool(t.unnecessary_braking) for t in evaluation if t.unnecessary_braking is not None]

    def mean(values: Sequence[float]) -> float | None:
        """Return the arithmetic mean, or ``None`` for an empty sequence."""
        return float(np.mean(values)) if values else None

    collision_rate = mean([float(v) for v in collisions])
    near_miss_rate = mean([float(v) for v in near_misses])
    braking_rate = mean([float(v) for v in braking])

    def status(has: bool) -> str:
        """Map presence to an availability label.

        Returns:
            ``"available"`` when the field was supplied, else ``"unavailable"``.
        """
        return "available" if has else "unavailable"

    availability = {
        "collision_rate": status(bool(collisions)),
        "near_miss_rate": status(bool(near_misses)),
        "mean_path_efficiency": status(bool(efficiencies)),
        "mean_pedestrian_disruption": status(bool(disruptions)),
        "unnecessary_braking_rate": status(bool(braking)),
    }
    return EvaluationOutcomeSummary(
        conditioning=TRACE_SUPPLIED_CONDITIONING,
        evaluation_trace_count=len(evaluation),
        collision_rate=collision_rate,
        near_miss_rate=near_miss_rate,
        mean_path_efficiency=mean(efficiencies),
        mean_pedestrian_disruption=mean(disruptions),
        unnecessary_braking_rate=braking_rate,
        field_availability=availability,
    )


def _provenance(  # noqa: PLR0913
    *,
    coverage_target: float,
    calibration_count: int,
    evaluation_count: int,
    conformal_radius: float,
    adaptive_config: AdaptiveConformalConfig,
    adaptive_empirical_coverage: float | None,
    weights: Mapping[str, float],
    feature_scales: Mapping[str, float],
    seed: int,
) -> dict[str, Any]:
    """Build provenance carrying reused primitive schemas and distributional assumptions.

    Args:
        coverage_target: Requested conformal coverage target.
        calibration_count: Number of calibration residuals used to fit the radius.
        evaluation_count: Number of evaluation residuals measured.
        conformal_radius: Fitted split-conformal radius.
        adaptive_config: ACI configuration reused for the diagnostic.
        adaptive_empirical_coverage: ACI empirical coverage over evaluation, or ``None``.
        weights: Preferred-margin weights used.
        feature_scales: Fit-derived feature normalization scales.
        seed: Determinism seed recorded for reproducibility.

    Returns:
        JSON-compatible provenance mapping.
    """
    return {
        "reused_primitives": {
            "split_conformal_radius": {
                "schema": SPLIT_CONFORMAL_RADIUS_SCHEMA,
                "coverage_target": float(coverage_target),
                "calibration_denominator": int(calibration_count),
                "radius_m": _finite_or_inf(conformal_radius),
            },
            "adaptive_conformal_buffers": {
                "schema": ADAPTIVE_CONFORMAL_BUFFERS_SCHEMA,
                "config": {
                    "coverage_target": float(adaptive_config.coverage_target),
                    "step_size": float(adaptive_config.step_size),
                    "window": adaptive_config.window,
                    "min_history": int(adaptive_config.min_history),
                },
                "evaluation_denominator": int(evaluation_count),
                "empirical_coverage": _finite_or_none(adaptive_empirical_coverage),
            },
        },
        "preferred_margin_model": {
            "name": "monotonic_non_negative_weighted_context_terms",
            "weights": {str(k): float(v) for k, v in weights.items()},
            "feature_scales_source": "fit_split_mean_with_positive_floor",
            "feature_scales": {str(k): float(v) for k, v in feature_scales.items()},
            "monotonic_non_negative": True,
            "hard_floor_immutable": True,
        },
        "distributional_assumptions": [
            "split_conformal_radius gives a marginal-coverage guarantee only when "
            "calibration and evaluation residuals are exchangeable.",
            "adaptive_conformal_buffers (ACI) tracks a target under drift but assumes "
            "bounded residuals and a stationary target in the long run.",
            "the preferred-margin feature scales are derived from the fit split only so "
            "calibration/evaluation statistics cannot leak into the margin model.",
            "traces are synthetic or tracked fixtures; a real navigation campaign is out "
            "of scope for this report.",
        ],
        "documented_failure_cases": [
            "when the calibration sample cannot certify the target the radius is +inf "
            "and the conformal margin is unbounded.",
            "absent collision, pedestrian-disruption, braking, or efficiency fields are "
            "reported as unavailable and never synthesized.",
            "fallback or degraded trace data is not accepted as evidence.",
        ],
        "seed": int(seed),
        "determinism": "fully deterministic for a fixed input and seed; no RNG is used.",
    }


def _finite_or_none(value: float | None) -> float | str | None:
    """Return a finite float, ``None``, or the string ``'+inf'`` for an infinite value."""
    if value is None:
        return None
    return _finite_or_inf(value)


def _finite_or_inf(value: float) -> float | str:
    """Return a finite float, or the string ``'+inf'`` when the value is infinite."""
    if math.isinf(value):
        return "+inf"
    return float(value)


# --------------------------------------------------------------------------- public API


def build_safety_margin_comparison(  # noqa: PLR0913
    *,
    fit_split_ids: Collection[str],
    calibration_split_ids: Collection[str],
    evaluation_split_ids: Collection[str],
    traces: Sequence[SafetyMarginTraceSample],
    hard_floor_m: float,
    coverage_target: float = 0.95,
    fixed_margin_m: float | None = None,
    preferred_weights: Mapping[str, float] | None = None,
    adaptive_config: AdaptiveConformalConfig | None = None,
    seed: int = 0,
) -> SafetyMarginCalibrationReport:
    """Build a deterministic offline comparison of the three margin constructions.

    The ``fit`` split establishes preferred-margin feature scales, the ``calibration``
    split fits the split-conformal radius, and the ``evaluation`` split measures
    coverage and trace-supplied outcomes. The three split-identifier sets must be
    explicit and pairwise disjoint; any overlap is rejected before a report is computed.

    Args:
        fit_split_ids: Identifiers of traces used only to fit feature scales.
        calibration_split_ids: Identifiers of traces used only to calibrate the
            conformal radius.
        evaluation_split_ids: Identifiers of held-out traces used only to measure
            coverage and outcomes.
        traces: All trace samples, each carrying one of the split identifiers.
        hard_floor_m: Immutable minimum clearance in metres (``>= 0``).
        coverage_target: Desired conformal coverage in ``(0, 1)``.
        fixed_margin_m: Fixed-baseline margin in metres (``>= 0``); defaults to the
            hard floor when omitted.
        preferred_weights: Optional override of the preferred-margin weights; defaults
            to :data:`DEFAULT_PREFERRED_WEIGHTS`.
        adaptive_config: Optional ACI configuration for the diagnostic; defaults to a
            conservative :class:`AdaptiveConformalConfig`.
        seed: Determinism seed recorded in provenance.

    Returns:
        A :class:`SafetyMarginCalibrationReport` comparing the three constructions.

    Raises:
        ValueError: On bad coverage target, non-finite/negative margins, empty split
            identifier sets, overlapping split identifiers (leakage), unknown trace
            split ids, empty splits, or non-finite calibration/evaluation residuals.

    .. admonition:: Evidence tier
       :class: warning

       Smoke/diagnostic only. This proves implementation integrity, not improved
       navigation safety, deployment coverage, or real-world risk reduction.
    """
    if not 0.0 < coverage_target < 1.0:
        raise ValueError("coverage_target must be between 0 and 1 (exclusive)")
    floor = require_finite_scalar("hard_floor_m", hard_floor_m)
    if floor < 0.0:
        raise ValueError(f"hard_floor_m must be non-negative: {floor}")
    fixed = (
        floor if fixed_margin_m is None else require_finite_scalar("fixed_margin_m", fixed_margin_m)
    )
    if fixed < 0.0:
        raise ValueError(f"fixed_margin_m must be non-negative: {fixed}")
    weights = dict(DEFAULT_PREFERRED_WEIGHTS if preferred_weights is None else preferred_weights)
    if set(weights) != set(DEFAULT_PREFERRED_WEIGHTS):
        raise ValueError(
            "preferred_weights must provide exactly the default weight keys: "
            f"{sorted(DEFAULT_PREFERRED_WEIGHTS)}"
        )
    aci_cfg = adaptive_config or AdaptiveConformalConfig(
        coverage_target=coverage_target, step_size=0.05, window=None, min_history=1
    )

    fit_ids = _to_id_set("fit_split_ids", fit_split_ids)
    calibration_ids = _to_id_set("calibration_split_ids", calibration_split_ids)
    evaluation_ids = _to_id_set("evaluation_split_ids", evaluation_split_ids)
    _validate_disjoint_splits(fit_ids, calibration_ids, evaluation_ids)

    fit, calibration, evaluation = _partition_traces(
        traces, fit_ids=fit_ids, calibration_ids=calibration_ids, evaluation_ids=evaluation_ids
    )
    _require_non_empty_split("fit", fit)
    _require_non_empty_split("calibration", calibration)
    _require_non_empty_split("evaluation", evaluation)

    scales = _feature_scales([t.context for t in fit])

    calibration_residuals = _split_residuals("calibration", calibration)
    evaluation_residuals = _split_residuals("evaluation", evaluation)

    conformal_radius = split_conformal_radius(
        calibration_residuals, coverage_target=coverage_target
    )
    aci_result = adaptive_conformal_buffers(evaluation_residuals, config=aci_cfg)
    aci_coverage: float | None = (
        float(aci_result.empirical_coverage) if aci_result.indices.size else None
    )

    eval_contexts = [t.context for t in evaluation]
    preferred_eval = [_preferred_margin(c, scales=scales, weights=weights) for c in eval_contexts]

    method_specs = (
        (METHOD_FIXED, [], 0.0),
        (METHOD_ADAPTIVE, preferred_eval, 0.0),
        (METHOD_ADAPTIVE_CONFORMAL, preferred_eval, conformal_radius),
    )
    methods: list[MarginMethodResult] = []
    for method, preferred, tightening in method_specs:
        margins, violations = _effective_margins(
            method,
            preferred=preferred,
            conformal_tightening=tightening,
            hard_floor_m=floor,
            fixed_margin_m=fixed,
            evaluation_count=len(eval_contexts),
        )
        empirical_coverage = _empirical_coverage(evaluation_residuals, margins)
        target = coverage_target if method == METHOD_ADAPTIVE_CONFORMAL else None
        gap = None
        if empirical_coverage is not None and target is not None:
            gap = empirical_coverage - target
        status = _coverage_status(
            method=method,
            conformal_tightening=tightening,
            empirical_coverage=empirical_coverage,
            coverage_target=target,
            coverage_gap=gap,
        )
        preferred_mean = float(np.mean(preferred)) if preferred else 0.0
        methods.append(
            MarginMethodResult(
                method=method,
                hard_floor_m=floor,
                preferred_margin_m=preferred_mean,
                conformal_tightening_m=tightening,
                effective_margin_m=float(np.mean(margins)),
                min_effective_margin_m=float(np.min(margins)),
                target_coverage=target,
                empirical_coverage=empirical_coverage,
                coverage_gap=gap,
                coverage_status=status,
                efficiency_cost=float(np.mean(margins)),
                violated_constraints=violations,
            )
        )

    outcomes = _evaluation_outcomes(evaluation)
    split_provenance = {
        "fit_split_ids": sorted(fit_ids),
        "calibration_split_ids": sorted(calibration_ids),
        "evaluation_split_ids": sorted(evaluation_ids),
        "pairwise_disjoint": True,
        "fit_trace_count": len(fit),
        "calibration_trace_count": len(calibration),
        "evaluation_trace_count": len(evaluation),
        "leakage_rejected_before_report": True,
    }
    provenance = _provenance(
        coverage_target=coverage_target,
        calibration_count=int(calibration_residuals.size),
        evaluation_count=int(evaluation_residuals.size),
        conformal_radius=conformal_radius,
        adaptive_config=aci_cfg,
        adaptive_empirical_coverage=aci_coverage,
        weights=weights,
        feature_scales=scales,
        seed=seed,
    )
    return SafetyMarginCalibrationReport(
        schema_version=SAFETY_MARGIN_CALIBRATION_SCHEMA,
        hard_floor_m=floor,
        coverage_target=coverage_target,
        fixed_margin_m=fixed,
        methods=methods,
        evaluation_outcomes=outcomes,
        split_provenance=split_provenance,
        provenance=provenance,
        claim_boundary=(
            "Smoke/diagnostic only. This report proves implementation integrity "
            "(disjoint splits, hard-floor monotonicity, deterministic output, honest "
            "unavailable-field reporting). It does not prove improved navigation safety, "
            "calibrated deployment coverage, or real-world risk reduction; a held-out "
            "navigation campaign requires a separate preregistration and approval."
        ),
    )


def _split_residuals(name: str, samples: Sequence[SafetyMarginTraceSample]) -> np.ndarray:
    """Collect non-finite-rejected nonconformity scores from a split's traces.

    Args:
        name: Split name for error messages.
        samples: Traces in the split.

    Returns:
        1-D float64 residual array.

    Raises:
        ValueError: If any residual is missing, non-finite, or negative, or the split is empty.
    """
    raw: list[float] = []
    for sample in samples:
        if sample.residual_m is None:
            raise ValueError(f"{name} traces must carry a non-null residual_m")
        raw.append(float(sample.residual_m))
    if not raw:
        raise ValueError(f"{name} split must contain at least one residual")
    residuals = require_finite_array(f"{name}_residuals", np.asarray(raw, dtype=np.float64))
    if np.any(residuals < 0.0):
        raise ValueError(f"{name}_residuals must be non-negative deviation magnitudes")
    return residuals


def _empirical_coverage(evaluation_residuals: np.ndarray, margins: Sequence[float]) -> float | None:
    """Fraction of evaluation residuals that fall within the per-context margin.

    Args:
        evaluation_residuals: Held-out nonconformity scores.
        margins: Per-context effective margins (one per evaluation residual).

    Returns:
        Empirical coverage in ``[0, 1]``, or ``None`` when there are no residuals.
    """
    if evaluation_residuals.size == 0:
        return None
    margins_arr = np.asarray(margins, dtype=np.float64)
    if margins_arr.shape[0] != evaluation_residuals.shape[0]:
        raise ValueError("margins must align one-to-one with evaluation residuals")
    covered = evaluation_residuals <= margins_arr
    return float(np.mean(covered))


__all__ = [
    "DEFAULT_PREFERRED_WEIGHTS",
    "METHOD_ADAPTIVE",
    "METHOD_ADAPTIVE_CONFORMAL",
    "METHOD_FIXED",
    "SAFETY_MARGIN_CALIBRATION_SCHEMA",
    "TRACE_SUPPLIED_CONDITIONING",
    "EvaluationOutcomeSummary",
    "MarginContext",
    "MarginMethodResult",
    "SafetyMarginCalibrationReport",
    "SafetyMarginTraceSample",
    "build_safety_margin_comparison",
]
