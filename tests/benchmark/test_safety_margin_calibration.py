"""Tests for offline safety-margin calibration and comparison (issue #6581).

Evidence tier: smoke/diagnostic only. These tests prove implementation integrity
(disjoint splits, hard-floor monotonicity, deterministic output, honest
unavailable-field reporting, and non-finite rejection) -- not improved navigation
safety, deployment coverage, or real-world risk reduction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.safety_margin_calibration import (
    DEFAULT_PREFERRED_WEIGHTS,
    METHOD_ADAPTIVE,
    METHOD_ADAPTIVE_CONFORMAL,
    METHOD_FIXED,
    SAFETY_MARGIN_CALIBRATION_SCHEMA,
    TRACE_SUPPLIED_CONDITIONING,
    MarginContext,
    SafetyMarginTraceSample,
    build_safety_margin_comparison,
)
from robot_sf.benchmark.uncertainty_safety import (
    ADAPTIVE_CONFORMAL_BUFFERS_SCHEMA,
    SPLIT_CONFORMAL_RADIUS_SCHEMA,
)

# --------------------------------------------------------------------------- helpers


def _ctx(
    *,
    density: float = 0.2,
    visibility: float = 8.0,
    speed: float = 1.0,
    footprint: float = 0.3,
    uncertainty: float = 0.05,
) -> MarginContext:
    """Build a valid MarginContext with small representative defaults."""
    return MarginContext(
        pedestrian_density=density,
        visibility_m=visibility,
        robot_speed_mps=speed,
        footprint_radius_m=footprint,
        localization_uncertainty_m=uncertainty,
    )


def _sample(  # noqa: PLR0913
    split_id: str,
    *,
    residual: float | None = 0.1,
    density: float = 0.2,
    visibility: float = 8.0,
    speed: float = 1.0,
    footprint: float = 0.3,
    uncertainty: float = 0.05,
    collision: bool | None = None,
    near_miss: bool | None = None,
    path_efficiency: float | None = None,
    pedestrian_disruption: float | None = None,
    unnecessary_braking: bool | None = None,
) -> SafetyMarginTraceSample:
    """Build a trace sample with optional outcome fields."""
    return SafetyMarginTraceSample(
        split_id=split_id,
        context=_ctx(
            density=density,
            visibility=visibility,
            speed=speed,
            footprint=footprint,
            uncertainty=uncertainty,
        ),
        residual_m=residual,
        collision=collision,
        near_miss=near_miss,
        path_efficiency=path_efficiency,
        pedestrian_disruption=pedestrian_disruption,
        unnecessary_braking=unnecessary_braking,
    )


def _balanced_traces(
    *,
    calibration_residuals: list[float] | None = None,
    evaluation_residuals: list[float] | None = None,
    include_outcomes: bool = False,
) -> list[SafetyMarginTraceSample]:
    """Build a small synthetic trace set with disjoint fit/cal/eval ids."""
    fit = [_sample("fit", residual=0.0) for _ in range(3)]
    cal = [_sample("cal", residual=r) for r in (calibration_residuals or [0.1, 0.2, 0.3])]
    eval_residuals = evaluation_residuals or [0.1, 0.2, 0.3]
    evaluation = [_sample("eval", residual=r) for r in eval_residuals]
    if include_outcomes:
        evaluation = [
            _sample(
                "eval",
                residual=r,
                collision=False,
                near_miss=False,
                path_efficiency=0.95,
                pedestrian_disruption=0.1,
                unnecessary_braking=False,
            )
            for r in eval_residuals
        ]
    return [*fit, *cal, *evaluation]


def _build(
    traces: list[SafetyMarginTraceSample] | None = None,
    **kwargs,
):
    """Build a comparison with the canonical disjoint fit/cal/eval id sets."""
    defaults: dict = {
        "fit_split_ids": {"fit"},
        "calibration_split_ids": {"cal"},
        "evaluation_split_ids": {"eval"},
        "traces": traces if traces is not None else _balanced_traces(),
        "hard_floor_m": 0.5,
        "coverage_target": 0.9,
        "fixed_margin_m": 0.5,
    }
    defaults.update(kwargs)
    return build_safety_margin_comparison(**defaults)


def _method(report, name: str):
    """Return the named method result from a report."""
    for row in report.methods:
        if row.method == name:
            return row
    raise AssertionError(f"method {name!r} missing from report")


# --------------------------------------------------------------------------- split leakage


def test_split_leakage_rejects_fit_calibration_overlap() -> None:
    """An id shared between fit and calibration must fail before any report."""
    with pytest.raises(ValueError, match="fit and calibration.*overlap leaked"):
        build_safety_margin_comparison(
            fit_split_ids={"shared", "fit"},
            calibration_split_ids={"shared", "cal"},
            evaluation_split_ids={"eval"},
            traces=_balanced_traces(),
            hard_floor_m=0.5,
        )


def test_split_leakage_rejects_calibration_evaluation_overlap() -> None:
    """An id shared between calibration and evaluation must fail before any report."""
    with pytest.raises(ValueError, match="calibration and evaluation.*overlap leaked"):
        build_safety_margin_comparison(
            fit_split_ids={"fit"},
            calibration_split_ids={"shared", "cal"},
            evaluation_split_ids={"shared", "eval"},
            traces=_balanced_traces(),
            hard_floor_m=0.5,
        )


def test_split_leakage_rejects_fit_evaluation_overlap() -> None:
    """An id shared between fit and evaluation must fail before any report."""
    with pytest.raises(ValueError, match="fit and evaluation.*overlap leaked"):
        build_safety_margin_comparison(
            fit_split_ids={"shared", "fit"},
            calibration_split_ids={"cal"},
            evaluation_split_ids={"shared", "eval"},
            traces=_balanced_traces(),
            hard_floor_m=0.5,
        )


def test_split_leakage_check_runs_before_residual_computation() -> None:
    """Leakage is rejected before residuals are read from any split."""
    # Evaluation traces lack residuals (which would later fail the eval-residual guard);
    # the leakage guard must fire first, proving split validation precedes computation.
    fit = [_sample("fit", residual=0.0) for _ in range(2)]
    cal = [_sample("cal", residual=0.1) for _ in range(2)]
    evaluation = [_sample("eval", residual=None) for _ in range(2)]
    with pytest.raises(ValueError, match="fit and calibration.*overlap leaked"):
        build_safety_margin_comparison(
            fit_split_ids={"shared", "fit"},
            calibration_split_ids={"shared", "cal"},
            evaluation_split_ids={"eval"},
            traces=[*fit, *cal, *evaluation],
            hard_floor_m=0.5,
        )


def test_unknown_split_id_is_rejected() -> None:
    """A trace whose split id is in none of the sets is rejected."""
    traces = _balanced_traces() + [_sample("rogue")]
    with pytest.raises(ValueError, match="not in any fit/calibration/evaluation set"):
        _build(traces)


def test_empty_split_identifier_set_is_rejected() -> None:
    """Every split identifier collection must be non-empty."""
    with pytest.raises(ValueError, match="evaluation_split_ids must not be empty"):
        build_safety_margin_comparison(
            fit_split_ids={"fit"},
            calibration_split_ids={"cal"},
            evaluation_split_ids=set(),
            traces=_balanced_traces(),
            hard_floor_m=0.5,
        )


def test_empty_trace_partition_is_rejected() -> None:
    """A split identifier present but unpopulated by traces fails closed."""
    traces = [*_balanced_traces()[3:]]  # drop all fit traces
    with pytest.raises(ValueError, match="fit split must contain at least one trace"):
        _build(traces)


# --------------------------------------------------------------------------- hard-floor monotonicity


def test_effective_margin_never_below_hard_floor() -> None:
    """Every method's minimum effective margin respects the immutable hard floor."""
    report = _build(hard_floor_m=0.7)
    for row in report.methods:
        assert row.min_effective_margin_m >= row.hard_floor_m - 1e-12
        assert row.violated_constraints == []


def test_hard_floor_not_reduced_by_preferred_or_conformal() -> None:
    """High density/uncertainty/low visibility widen but never reduce the floor."""
    # Replace evaluation contexts with ones that strongly prefer a wider margin.
    eval_residuals = [0.05, 0.1, 0.15]
    evaluation = [
        _sample("eval", residual=r, density=1.5, visibility=1.0, speed=2.0, uncertainty=0.5)
        for r in eval_residuals
    ]
    fit = [_sample("fit", residual=0.0) for _ in range(3)]
    cal = [_sample("cal", residual=0.1) for _ in range(3)]
    report = build_safety_margin_comparison(
        fit_split_ids={"fit"},
        calibration_split_ids={"cal"},
        evaluation_split_ids={"eval"},
        traces=[*fit, *cal, *evaluation],
        hard_floor_m=0.4,
        coverage_target=0.9,
    )
    adaptive = _method(report, METHOD_ADAPTIVE)
    conformal = _method(report, METHOD_ADAPTIVE_CONFORMAL)
    assert adaptive.preferred_margin_m > 0.0
    assert conformal.conformal_tightening_m >= 0.0
    assert adaptive.min_effective_margin_m >= report.hard_floor_m
    assert conformal.min_effective_margin_m >= adaptive.min_effective_margin_m - 1e-12
    for row in report.methods:
        assert row.violated_constraints == []


def test_fixed_method_uses_max_of_floor_and_fixed_margin() -> None:
    """The fixed method clamps the baseline up to the floor but never below it."""
    report = _build(hard_floor_m=0.5, fixed_margin_m=0.2)
    fixed = _method(report, METHOD_FIXED)
    assert fixed.effective_margin_m == pytest.approx(0.5)
    assert fixed.preferred_margin_m == 0.0
    assert fixed.conformal_tightening_m == 0.0
    assert fixed.target_coverage is None


def test_conformal_term_only_widens_margin() -> None:
    """Adaptive+conformal effective margin is at least the adaptive margin."""
    report = _build()
    adaptive = _method(report, METHOD_ADAPTIVE)
    conformal = _method(report, METHOD_ADAPTIVE_CONFORMAL)
    assert conformal.effective_margin_m >= adaptive.effective_margin_m - 1e-12


# --------------------------------------------------------------------------- determinism


def test_report_is_deterministic_for_fixed_input_and_seed() -> None:
    """Two builds with identical inputs produce identical reports."""
    first = _build(seed=1234)
    second = _build(seed=1234)
    assert first == second
    assert first.schema_version == SAFETY_MARGIN_CALIBRATION_SCHEMA


def test_report_seed_recorded_in_provenance() -> None:
    """The seed is echoed in provenance for reproducibility."""
    report = _build(seed=42)
    assert report.provenance["seed"] == 42
    assert "deterministic" in report.provenance["determinism"]


def test_method_ordering_is_stable() -> None:
    """Methods appear in fixed, adaptive, adaptive_conformal order."""
    report = _build()
    assert [row.method for row in report.methods] == [
        METHOD_FIXED,
        METHOD_ADAPTIVE,
        METHOD_ADAPTIVE_CONFORMAL,
    ]


# --------------------------------------------------------------------------- coverage


def test_conformal_coverage_when_evaluation_within_radius() -> None:
    """Evaluation residuals within the calibrated radius are fully covered."""
    # Calibration residuals 0.1..0.9 with target 0.9 -> radius = 9th smallest = 0.9.
    report = build_safety_margin_comparison(
        fit_split_ids={"fit"},
        calibration_split_ids={"cal"},
        evaluation_split_ids={"eval"},
        traces=_balanced_traces(
            calibration_residuals=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            evaluation_residuals=[0.1, 0.2, 0.3],
        ),
        hard_floor_m=0.0,
        coverage_target=0.9,
        fixed_margin_m=0.0,
    )
    conformal = _method(report, METHOD_ADAPTIVE_CONFORMAL)
    assert conformal.conformal_tightening_m == pytest.approx(0.9)
    assert conformal.target_coverage == 0.9
    assert conformal.empirical_coverage == pytest.approx(1.0)
    assert conformal.coverage_gap == pytest.approx(1.0 - 0.9)
    assert conformal.coverage_status == "covered_evaluation_smoke"


def test_under_coverage_when_evaluation_exceeds_radius() -> None:
    """Held-out residuals beyond the calibrated radius report under-coverage."""
    # Small calibration residuals -> tight radius; large evaluation residual escapes.
    report = build_safety_margin_comparison(
        fit_split_ids={"fit"},
        calibration_split_ids={"cal"},
        evaluation_split_ids={"eval"},
        traces=_balanced_traces(
            calibration_residuals=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            evaluation_residuals=[5.0, 5.0, 5.0],
        ),
        hard_floor_m=0.0,
        coverage_target=0.9,
        fixed_margin_m=0.0,
    )
    conformal = _method(report, METHOD_ADAPTIVE_CONFORMAL)
    assert conformal.empirical_coverage == pytest.approx(0.0)
    assert conformal.coverage_gap is not None and conformal.coverage_gap < 0.0
    assert conformal.coverage_status == "under_covered_evaluation"


def test_coverage_unreachable_returns_infinite_radius() -> None:
    """An infinite radius is uncertifiable, even though it trivially covers the holdout."""
    # n=2, target 0.9 -> k = ceil(3*0.9) = 3 > 2 -> +inf per split_conformal_radius.
    report = build_safety_margin_comparison(
        fit_split_ids={"fit"},
        calibration_split_ids={"cal"},
        evaluation_split_ids={"eval"},
        traces=_balanced_traces(
            calibration_residuals=[0.1, 0.2],
            evaluation_residuals=[10.0, 20.0],
        ),
        hard_floor_m=0.0,
        coverage_target=0.9,
        fixed_margin_m=0.0,
    )
    conformal = _method(report, METHOD_ADAPTIVE_CONFORMAL)
    assert math.isinf(conformal.conformal_tightening_m)
    assert conformal.empirical_coverage == pytest.approx(1.0)
    assert conformal.coverage_status == "uncertifiable_infinite_radius"
    assert report.provenance["reused_primitives"]["split_conformal_radius"]["radius_m"] == "+inf"


def test_non_conformal_methods_report_diagnostic_coverage_status() -> None:
    """Fixed and adaptive methods carry no target and report a diagnostic status."""
    report = _build()
    for name in (METHOD_FIXED, METHOD_ADAPTIVE):
        row = _method(report, name)
        assert row.target_coverage is None
        assert row.coverage_gap is None
        assert row.coverage_status == "diagnostic_no_target"


# --------------------------------------------------------------------------- outcome availability


def test_absent_outcome_fields_reported_unavailable() -> None:
    """Outcome fields no trace supplied are unavailable, never synthesized."""
    report = _build()  # no outcome fields set on any trace
    outcomes = report.evaluation_outcomes
    assert outcomes.conditioning == TRACE_SUPPLIED_CONDITIONING
    assert outcomes.collision_rate is None
    assert outcomes.near_miss_rate is None
    assert outcomes.mean_path_efficiency is None
    assert outcomes.mean_pedestrian_disruption is None
    assert outcomes.unnecessary_braking_rate is None
    assert all(v == "unavailable" for v in outcomes.field_availability.values())


def test_present_outcome_fields_reported_available() -> None:
    """Outcome fields at least one trace supplied are aggregated and marked available."""
    report = _build(traces=_balanced_traces(include_outcomes=True))
    outcomes = report.evaluation_outcomes
    assert outcomes.collision_rate == pytest.approx(0.0)
    assert outcomes.mean_path_efficiency == pytest.approx(0.95)
    assert outcomes.mean_pedestrian_disruption == pytest.approx(0.1)
    assert outcomes.unnecessary_braking_rate == pytest.approx(0.0)
    assert all(v == "available" for v in outcomes.field_availability.values())


def test_partial_outcome_fields_mixed_availability() -> None:
    """A field supplied by only some traces is still aggregated over the suppliers."""
    evaluation = [
        _sample("eval", residual=0.1, collision=True, path_efficiency=0.8),
        _sample("eval", residual=0.2, collision=False, path_efficiency=None),
    ]
    fit = [_sample("fit", residual=0.0) for _ in range(2)]
    cal = [_sample("cal", residual=0.1) for _ in range(2)]
    report = build_safety_margin_comparison(
        fit_split_ids={"fit"},
        calibration_split_ids={"cal"},
        evaluation_split_ids={"eval"},
        traces=[*fit, *cal, *evaluation],
        hard_floor_m=0.5,
        coverage_target=0.9,
    )
    outcomes = report.evaluation_outcomes
    assert outcomes.collision_rate == pytest.approx(0.5)
    assert outcomes.mean_path_efficiency == pytest.approx(0.8)
    assert outcomes.field_availability["collision_rate"] == "available"
    assert outcomes.field_availability["mean_path_efficiency"] == "available"
    assert outcomes.field_availability["near_miss_rate"] == "unavailable"


# --------------------------------------------------------------------------- provenance


def test_provenance_carries_both_reused_primitive_schemas() -> None:
    """Provenance emits the split-conformal and adaptive-conformal schema versions."""
    report = _build()
    primitives = report.provenance["reused_primitives"]
    assert primitives["split_conformal_radius"]["schema"] == SPLIT_CONFORMAL_RADIUS_SCHEMA
    assert primitives["adaptive_conformal_buffers"]["schema"] == ADAPTIVE_CONFORMAL_BUFFERS_SCHEMA
    assert report.provenance["distributional_assumptions"]
    assert report.provenance["documented_failure_cases"]
    assert report.provenance["preferred_margin_model"]["hard_floor_immutable"] is True


def test_split_provenance_records_disjoint_counts() -> None:
    """Split provenance echoes disjoint ids, counts, and the leakage check."""
    report = _build()
    sp = report.split_provenance
    assert sp["fit_split_ids"] == ["fit"]
    assert sp["calibration_split_ids"] == ["cal"]
    assert sp["evaluation_split_ids"] == ["eval"]
    assert sp["pairwise_disjoint"] is True
    assert sp["leakage_rejected_before_report"] is True
    assert sp["fit_trace_count"] == 3
    assert sp["calibration_trace_count"] == 3
    assert sp["evaluation_trace_count"] == 3


def test_claim_boundary_states_smoke_only() -> None:
    """The claim boundary forbids any safety or deployment-coverage claim."""
    report = _build()
    assert "Smoke/diagnostic only" in report.claim_boundary
    assert "does not prove improved navigation safety" in report.claim_boundary


def test_efficiency_cost_is_mean_effective_margin() -> None:
    """The efficiency proxy equals the mean effective margin per method."""
    report = _build()
    for row in report.methods:
        assert row.efficiency_cost == pytest.approx(row.effective_margin_m)


# --------------------------------------------------------------------------- non-finite rejection


def test_non_finite_context_field_rejected() -> None:
    """A NaN context field fails closed at construction."""
    with pytest.raises(ValueError, match="pedestrian_density is not finite"):
        MarginContext(
            pedestrian_density=float("nan"),
            visibility_m=8.0,
            robot_speed_mps=1.0,
            footprint_radius_m=0.3,
            localization_uncertainty_m=0.05,
        )


def test_non_finite_residual_rejected() -> None:
    """A non-finite calibration residual fails closed before any report."""
    traces = _balanced_traces(calibration_residuals=[0.1, float("inf"), 0.3])
    with pytest.raises(ValueError, match="calibration_residuals must contain only finite"):
        _build(traces)


def test_non_finite_evaluation_residual_rejected() -> None:
    """A non-finite evaluation residual fails closed before any report."""
    traces = _balanced_traces(evaluation_residuals=[0.1, float("nan"), 0.3])
    with pytest.raises(ValueError, match="evaluation_residuals must contain only finite"):
        _build(traces)


@pytest.mark.parametrize(
    ("split", "kwargs"),
    [
        ("calibration", {"calibration_residuals": [-0.1, 0.2, 0.3]}),
        ("evaluation", {"evaluation_residuals": [0.1, -0.2, 0.3]}),
    ],
)
def test_negative_residual_rejected(split: str, kwargs: dict[str, list[float]]) -> None:
    """Residuals are non-negative nominal-to-perturbed deviation magnitudes."""
    with pytest.raises(ValueError, match=rf"{split}_residuals must be non-negative"):
        _build(_balanced_traces(**kwargs))


def test_non_finite_hard_floor_rejected() -> None:
    """A non-finite hard floor fails closed."""
    with pytest.raises(ValueError, match="hard_floor_m is not finite"):
        _build(hard_floor_m=float("inf"))


def test_non_finite_fixed_margin_rejected() -> None:
    """A non-finite fixed margin fails closed."""
    with pytest.raises(ValueError, match="fixed_margin_m is not finite"):
        _build(fixed_margin_m=float("nan"))


def test_negative_context_field_rejected() -> None:
    """A negative speed fails closed at construction."""
    with pytest.raises(ValueError, match="robot_speed_mps must be non-negative"):
        MarginContext(
            pedestrian_density=0.1,
            visibility_m=8.0,
            robot_speed_mps=-1.0,
            footprint_radius_m=0.3,
            localization_uncertainty_m=0.05,
        )


def test_bad_coverage_target_rejected() -> None:
    """Coverage targets at or outside the (0, 1) bounds fail closed."""
    with pytest.raises(ValueError, match="coverage_target"):
        _build(coverage_target=1.0)
    with pytest.raises(ValueError, match="coverage_target"):
        _build(coverage_target=0.0)


def test_preferred_weights_must_match_default_keys() -> None:
    """An incomplete preferred-weights override fails closed."""
    with pytest.raises(ValueError, match="preferred_weights must provide exactly"):
        _build(preferred_weights={"pedestrian_density": 0.1})


def test_default_preferred_weights_are_non_negative() -> None:
    """The default preferred-margin weights are all non-negative."""
    assert all(value >= 0.0 for value in DEFAULT_PREFERRED_WEIGHTS.values())
    assert set(DEFAULT_PREFERRED_WEIGHTS) == {
        "pedestrian_density",
        "robot_speed_mps",
        "localization_uncertainty_m",
        "footprint_radius_m",
        "low_visibility",
    }


def test_missing_calibration_residual_rejected() -> None:
    """A calibration trace missing its residual fails closed."""
    traces = _balanced_traces()
    # Drop the residual on the first calibration trace.
    cal_trace = next(t for t in traces if t.split_id == "cal")
    traces[traces.index(cal_trace)] = SafetyMarginTraceSample(
        split_id="cal", context=cal_trace.context, residual_m=None
    )
    with pytest.raises(ValueError, match="calibration traces must carry a non-null residual_m"):
        _build(traces)


def test_numpy_rng_is_not_used_in_build() -> None:
    """The build does not consume RNG state; identical inputs stay identical."""
    # Burn some RNG state between the two builds to prove the result is unaffected.
    _ = np.random.default_rng(999).random(10)
    first = _build()
    _ = np.random.default_rng(1).random(50)
    second = _build()
    assert first == second
