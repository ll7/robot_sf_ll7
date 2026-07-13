"""Social Navigation Quality Index (SNQI) scalarization diagnostics.

This module exports post-hoc analysis artifacts only. It recomputes SNQI rankings
from existing episode records under weight-zero and weight-sweep variants, then
compares those scalar rankings with a constraints-first diagnostic ordering. It
does not change SNQI definitions, benchmark metrics, or claim status.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.rank_metrics import kendall_tau, spearman_from_order
from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES, compute_snqi, normalize_metric

if TYPE_CHECKING:
    from pathlib import Path

SCALARIZATION_SENSITIVITY_SCHEMA = "snqi_scalarization_sensitivity.v1"
SCALARIZATION_SENSITIVITY_PREFLIGHT_SCHEMA = "snqi_scalarization_sensitivity_preflight.v1"
ADMISSIBLE_WEIGHT_FAMILY_SCHEMA = "snqi-admissible-weight-family.v1"
DEFAULT_SWEEP_FACTORS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
SENSITIVITY_PREFLIGHT_READY = "ready"
SENSITIVITY_PREFLIGHT_BLOCKED = "blocked"
SENSITIVITY_PREFLIGHT_MALFORMED = "malformed"
REQUIRED_SENSITIVITY_METRICS = (
    "success",
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "comfort_exposure",
)
OPTIONAL_SENSITIVITY_METRICS = ("force_exceed_events", "jerk_mean")
OPTIONAL_WEIGHTED_SENSITIVITY_METRICS = {
    "force_exceed_events": "w_force_exceed",
    "jerk_mean": "w_jerk",
}
BOUNDED_NORMALIZED_SENSITIVITY_METRICS = (
    "success",
    "time_to_goal_norm",
    "comfort_exposure",
)


@dataclass(frozen=True, slots=True)
class DiagnosticArtifacts:
    """Paths written by ``write_diagnostic_artifacts``."""

    json_path: Path
    csv_path: Path
    decision_disagreement_csv_path: Path
    markdown_path: Path
    svg_path: Path


@dataclass(frozen=True, slots=True)
class SensitivityPreflightIssue:
    """One fail-closed readiness issue for scalarization-sensitivity inputs."""

    code: str
    severity: str
    message: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-ready issue row."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(slots=True)
class _SensitivityPreflightState:
    issues: list[SensitivityPreflightIssue]
    malformed: bool
    grouped: dict[str, list[Mapping[str, Any]]]
    coverage: dict[str, set[tuple[str, str]]]
    scenario_horizons: set[tuple[str, str]]

    @classmethod
    def empty(cls) -> _SensitivityPreflightState:
        return cls(
            issues=[],
            malformed=False,
            grouped={},
            coverage={},
            scenario_horizons=set(),
        )

    def add_issue(self, code: str, severity: str, message: str) -> None:
        self.issues.append(SensitivityPreflightIssue(code, severity, message))
        if severity == SENSITIVITY_PREFLIGHT_MALFORMED:
            self.malformed = True


def classify_scalarization_sensitivity_inputs(
    records: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
    planner_key: str = "planner_key",
    fallback_planner_key: str = "planner",
    scenario_key: str = "scenario_id",
    horizon_key: str = "horizon",
    min_planners: int = 2,
) -> dict[str, Any]:
    """Classify whether SNQI scalarization-sensitivity inputs are ready.

    This is a blocker/readiness preflight only. It intentionally avoids exporting
    sensitivity artifacts or making decision-reversal claims.

    Returns:
        JSON-ready readiness report with ``ready``, ``blocked``, or ``malformed`` status.
    """

    state = _SensitivityPreflightState.empty()
    records = _validate_preflight_inputs(state, records, weights, baseline)
    _collect_preflight_records(
        state, records, weights, planner_key, fallback_planner_key, scenario_key, horizon_key
    )
    missing_cells = _add_global_preflight_issues(state, records, min_planners)
    _add_pareto_preflight_issues(state, weights, baseline, min_planners)
    return _format_preflight_report(state, records, missing_cells)


def _validate_preflight_inputs(
    state: _SensitivityPreflightState,
    records: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> Sequence[Mapping[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        state.add_issue(
            "records_not_sequence",
            SENSITIVITY_PREFLIGHT_MALFORMED,
            "records must be a sequence of mapping objects",
        )
        records = []
    if not isinstance(weights, Mapping):
        state.add_issue(
            "weights_not_mapping",
            SENSITIVITY_PREFLIGHT_MALFORMED,
            "weights must be a mapping keyed by SNQI component",
        )
    if not isinstance(baseline, Mapping):
        state.add_issue(
            "baseline_not_mapping",
            SENSITIVITY_PREFLIGHT_MALFORMED,
            "baseline must be a mapping keyed by normalized metric",
        )

    if isinstance(weights, Mapping):
        missing_weights = [name for name in WEIGHT_NAMES if name not in weights]
        if missing_weights:
            state.add_issue(
                "missing_snqi_weights",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                "missing SNQI weights: " + ", ".join(missing_weights),
            )
    return records


def _collect_preflight_records(
    state: _SensitivityPreflightState,
    records: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    planner_key: str,
    fallback_planner_key: str,
    scenario_key: str,
    horizon_key: str,
) -> None:
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            state.add_issue(
                "record_not_mapping",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                f"record {index} must be a mapping",
            )
            continue
        _collect_preflight_record(
            state,
            index,
            record,
            weights,
            planner_key,
            fallback_planner_key,
            scenario_key,
            horizon_key,
        )


def _collect_preflight_record(
    state: _SensitivityPreflightState,
    index: int,
    record: Mapping[str, Any],
    weights: Mapping[str, float],
    planner_key: str,
    fallback_planner_key: str,
    scenario_key: str,
    horizon_key: str,
) -> None:
    planner = _preflight_planner(record, planner_key, fallback_planner_key)
    if planner in (None, ""):
        state.add_issue(
            "missing_planner",
            SENSITIVITY_PREFLIGHT_MALFORMED,
            f"record {index} is missing planner identity",
        )
        return

    scenario = _get_nested(record, scenario_key)
    horizon = _get_nested(record, horizon_key)
    if horizon in (None, ""):
        horizon = _get_nested(record, "scenario_horizon")
    if scenario in (None, "") or horizon in (None, ""):
        state.add_issue(
            "missing_scenario_horizon",
            SENSITIVITY_PREFLIGHT_BLOCKED,
            f"record {index} lacks scenario-horizon evidence",
        )
        return

    _add_metric_preflight_issues(state, index, _metrics(record), weights)
    planner_name = str(planner)
    scenario_horizon = (str(scenario), str(horizon))
    state.grouped.setdefault(planner_name, []).append(record)
    state.scenario_horizons.add(scenario_horizon)
    state.coverage.setdefault(planner_name, set()).add(scenario_horizon)


def _preflight_planner(
    record: Mapping[str, Any], planner_key: str, fallback_planner_key: str
) -> Any:
    planner = _get_nested(record, planner_key)
    if planner in (None, ""):
        planner = _get_nested(record, fallback_planner_key)
    if planner in (None, ""):
        planner = _get_nested(record, "scenario_params.algo")
    return planner


def _add_metric_preflight_issues(
    state: _SensitivityPreflightState,
    index: int,
    metrics: Mapping[str, Any],
    weights: Mapping[str, float],
) -> None:
    for metric in REQUIRED_SENSITIVITY_METRICS:
        if metric not in metrics:
            state.add_issue(
                "missing_required_term",
                SENSITIVITY_PREFLIGHT_BLOCKED,
                f"record {index} is missing SNQI term {metric!r}",
            )
        elif not _is_finite_metric(metrics.get(metric)):
            state.add_issue(
                "non_finite_required_term",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                f"record {index} has non-finite SNQI term {metric!r}",
            )

        elif metric in BOUNDED_NORMALIZED_SENSITIVITY_METRICS and not _is_unit_interval(
            metrics.get(metric)
        ):
            state.add_issue(
                "out_of_range_normalized_term",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                f"record {index} normalized SNQI term {metric!r} outside [0, 1]",
            )

    for metric in OPTIONAL_SENSITIVITY_METRICS:
        if metric not in metrics:
            if _is_active_optional_weight(weights, metric):
                state.add_issue(
                    "missing_weighted_optional_term",
                    SENSITIVITY_PREFLIGHT_BLOCKED,
                    f"record {index} missing weighted SNQI term {metric!r}",
                )
        elif not _is_finite_metric(metrics.get(metric)):
            state.add_issue(
                "non_finite_optional_term",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                f"record {index} has non-finite SNQI term {metric!r}",
            )


def _add_global_preflight_issues(
    state: _SensitivityPreflightState,
    records: Sequence[Mapping[str, Any]],
    min_planners: int,
) -> dict[str, list[tuple[str, str]]]:
    if not records:
        state.add_issue(
            "no_records",
            SENSITIVITY_PREFLIGHT_BLOCKED,
            "no episode records available for scalarization-sensitivity preflight",
        )
    if len(state.grouped) < min_planners:
        state.add_issue(
            "insufficient_planners",
            SENSITIVITY_PREFLIGHT_BLOCKED,
            f"at least {min_planners} planners are required for rank and Pareto diagnostics",
        )

    missing_cells = {
        planner: sorted(state.scenario_horizons - seen)
        for planner, seen in sorted(state.coverage.items())
        if state.scenario_horizons - seen
    }
    if missing_cells:
        state.add_issue(
            "non_rectangular_planner_table",
            SENSITIVITY_PREFLIGHT_BLOCKED,
            "planner table must cover the same scenario-horizon cells for every planner",
        )
    if not state.scenario_horizons:
        state.add_issue(
            "no_valid_scenario_horizon",
            SENSITIVITY_PREFLIGHT_BLOCKED,
            "no valid scenario-horizon evidence was found",
        )
    return missing_cells


def _add_pareto_preflight_issues(
    state: _SensitivityPreflightState,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
    min_planners: int,
) -> None:
    _add_normalization_preflight_issues(state, baseline)
    if len(state.grouped) < min_planners or state.malformed:
        return
    try:
        snqi_scores = _planner_snqi_scores(state.grouped, weights, baseline)
        constraints = {
            planner: _constraints_first_endpoint(rows) for planner, rows in state.grouped.items()
        }
        rows = _planner_rows(
            snqi_scores,
            constraints,
            _rank_order(snqi_scores, higher_is_better=True),
            _rank_order(
                {
                    planner: endpoint["constraints_first_score"]
                    for planner, endpoint in constraints.items()
                },
                higher_is_better=True,
            ),
        )
        _pareto_points(rows)
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        state.add_issue(
            "pareto_prerequisite_error",
            SENSITIVITY_PREFLIGHT_MALFORMED,
            f"could not derive Pareto prerequisites: {exc}",
        )


def _add_normalization_preflight_issues(
    state: _SensitivityPreflightState, baseline: Mapping[str, Mapping[str, float]]
) -> None:
    for metric in ("collisions", "near_misses", "force_exceed_events", "jerk_mean"):
        stats = baseline.get(metric)
        if not isinstance(stats, Mapping):
            state.add_issue(
                "malformed_baseline_stats",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                f"baseline metric {metric!r} must provide med/p95 mapping",
            )
            continue
        try:
            med = float(stats["med"])
            p95 = float(stats["p95"])
        except (KeyError, TypeError, ValueError) as exc:
            state.add_issue(
                "malformed_baseline_stats",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                f"baseline metric {metric!r} must provide finite med/p95 values: {exc}",
            )
            continue
        if not math.isfinite(med) or not math.isfinite(p95):
            state.add_issue(
                "non_finite_baseline_stats",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                f"baseline metric {metric!r} has non-finite med/p95 values",
            )
            continue
        if p95 <= med:
            state.add_issue(
                "degenerate_baseline_range",
                SENSITIVITY_PREFLIGHT_MALFORMED,
                f"baseline metric {metric!r} must satisfy p95 > med",
            )


def _format_preflight_report(
    state: _SensitivityPreflightState,
    records: Sequence[Mapping[str, Any]],
    missing_cells: Mapping[str, Sequence[tuple[str, str]]],
) -> dict[str, Any]:
    status = SENSITIVITY_PREFLIGHT_READY
    if state.malformed:
        status = SENSITIVITY_PREFLIGHT_MALFORMED
    elif state.issues:
        status = SENSITIVITY_PREFLIGHT_BLOCKED

    return {
        "schema_version": SCALARIZATION_SENSITIVITY_PREFLIGHT_SCHEMA,
        "issue": 3653,
        "status": status,
        "ready": status == SENSITIVITY_PREFLIGHT_READY,
        "planners": sorted(state.grouped),
        "planner_count": len(state.grouped),
        "record_count": len(records),
        "scenario_horizon_count": len(state.scenario_horizons),
        "missing_planner_cells": {
            planner: [{"scenario": cell[0], "horizon": cell[1]} for cell in cells]
            for planner, cells in missing_cells.items()
        },
        "issues": [issue.as_dict() for issue in state.issues],
        "claim_boundary": (
            "Readiness preflight only; no SNQI weight changes, no scalarization export, "
            "no benchmark campaign run, and no decision-reversal claim."
        ),
    }


def build_scalarization_sensitivity_report(
    records: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
    planner_key: str = "planner_key",
    fallback_planner_key: str = "planner",
    sweep_factors: Sequence[float] = DEFAULT_SWEEP_FACTORS,
    admissible_weight_family: Mapping[str, Any] | None = None,
    input_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a SNQI scalarization-sensitivity and Pareto-front report.

    Returns:
        Versioned report payload with rank disagreement, weight variants, term
        dominance, and Pareto-front rows.
    """

    _validate_export_required_terms(records, weights)
    grouped = _group_records(records, planner_key, fallback_planner_key)
    if len(grouped) < 2:
        raise ValueError("at least two planners are required for rank-sensitivity diagnostics")

    clean_weights = {name: float(weights.get(name, 1.0)) for name in WEIGHT_NAMES}
    base_scores = _planner_snqi_scores(grouped, clean_weights, baseline)
    base_order = _rank_order(base_scores, higher_is_better=True)
    constraints = {
        planner: _constraints_first_endpoint(episodes)
        for planner, episodes in sorted(grouped.items(), key=lambda item: item[0])
    }
    constraints_scores = {
        planner: float(endpoint["constraints_first_score"])
        for planner, endpoint in constraints.items()
    }
    constraints_order = _rank_order(constraints_scores, higher_is_better=True)
    disagreement = _rank_disagreement(base_order, constraints_order)
    planner_rows = _planner_rows(base_scores, constraints, base_order, constraints_order)
    pareto_points = _pareto_points(planner_rows)

    weight_zero: dict[str, Any] = {}
    weight_sweep: dict[str, Any] = {}
    family_variants: list[dict[str, Any]] = []
    for weight_name in WEIGHT_NAMES:
        if weight_name not in clean_weights:
            continue
        zero_weights = dict(clean_weights)
        zero_weights[weight_name] = 0.0
        zero_order = _rank_order(
            _planner_snqi_scores(grouped, zero_weights, baseline),
            higher_is_better=True,
        )
        zero_summary = _variant_summary(base_order, zero_order)
        if admissible_weight_family is not None:
            zero_summary["weight_family"] = classify_admissible_weight_vector(
                zero_weights, admissible_weight_family
            )
        weight_zero[weight_name] = zero_summary

        variants = []
        for factor in sweep_factors:
            sweep_weights = dict(clean_weights)
            sweep_weights[weight_name] = clean_weights[weight_name] * float(factor)
            sweep_scores = _planner_snqi_scores(grouped, sweep_weights, baseline)
            sweep_order = _rank_order(sweep_scores, higher_is_better=True)
            variant = {
                "factor": float(factor),
                "weight_value": float(sweep_weights[weight_name]),
                "order": sweep_order,
                **_variant_summary(base_order, sweep_order),
            }
            if admissible_weight_family is not None:
                variant["weight_family"] = classify_admissible_weight_vector(
                    sweep_weights, admissible_weight_family
                )
            variants.append(variant)
            family_variants.append(
                {
                    "variant_id": f"{weight_name}@{float(factor):g}",
                    "weights": sweep_weights,
                    **variant,
                }
            )
        weight_sweep[weight_name] = variants

    dominance = _term_dominance(grouped, clean_weights, baseline)
    max_disagreement_rate = max(
        [
            float(row["pairwise_disagreement_rate_vs_base"])
            for rows in weight_sweep.values()
            for row in rows
        ],
        default=0.0,
    )
    max_zero_reversals = max(
        [int(row["pairwise_reversal_count_vs_base"]) for row in weight_zero.values()],
        default=0,
    )

    report = {
        "schema_version": SCALARIZATION_SENSITIVITY_SCHEMA,
        "evidence_kind": "analysis_artifact_only",
        "claim_boundary": (
            "Diagnostic export for inspecting SNQI scalarization sensitivity; "
            "not benchmark evidence and not a primary-index claim."
        ),
        "inputs": {
            "planner_key": planner_key,
            "fallback_planner_key": fallback_planner_key,
            "planners": len(grouped),
            "episodes": sum(len(rows) for rows in grouped.values()),
            "sweep_factors": [float(factor) for factor in sweep_factors],
            "provenance": dict(input_provenance or {}),
        },
        "base_snqi_order": base_order,
        "constraints_first_order": constraints_order,
        "decision_disagreement": disagreement,
        "planner_rows": planner_rows,
        "term_dominance": dominance,
        "weight_zero_ablation": weight_zero,
        "weight_sweep": weight_sweep,
        "pareto_front": {
            "x": "constraints_first_score",
            "y": "snqi_mean",
            "points": pareto_points,
        },
        "summary": {
            "decision_disagreement_rate": disagreement["pairwise_disagreement_rate"],
            "max_weight_sweep_disagreement_rate_vs_base": max_disagreement_rate,
            "max_weight_zero_pairwise_reversal_count_vs_base": max_zero_reversals,
            "top_term_by_mean_abs_contribution": dominance[0]["component"] if dominance else None,
        },
    }
    if admissible_weight_family is not None:
        family_summary = _weight_family_sensitivity_summary(
            admissible_weight_family, family_variants
        )
        report["admissible_weight_family"] = family_summary
        report["summary"].update(
            {
                "headline_scope": "admissible_family_only",
                "admissible_family_pairwise_reversal_count_vs_base": family_summary[
                    "admissible_family"
                ]["pairwise_reversal_count_vs_base"],
                "full_sweep_pairwise_reversal_count_vs_base": family_summary["full_sweep"][
                    "pairwise_reversal_count_vs_base"
                ],
            }
        )
    return report


def load_admissible_weight_family_config(path: Path) -> dict[str, Any]:
    """Load and validate an ex-ante SNQI weight-family configuration.

    The configuration deliberately classifies weight vectors before ranking
    artifacts are inspected. It does not select canonical SNQI weights.

    Returns:
        Validated family policy suitable for report classification.
    """

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("admissible weight-family config must be a mapping")
    if raw.get("schema_version") != ADMISSIBLE_WEIGHT_FAMILY_SCHEMA:
        raise ValueError(
            f"unsupported admissible weight-family schema_version: {raw.get('schema_version')!r}"
        )
    family = raw.get("admissible_family")
    if not isinstance(family, Mapping):
        raise ValueError("admissible weight-family config requires an admissible_family mapping")
    return _validated_admissible_weight_family(family)


def classify_admissible_weight_vector(
    weights: Mapping[str, float], family: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify one vector as admissible or as a labeled stress probe.

    The vector is normalized onto the simplex solely for family membership.
    SNQI scores continue to use the original supplied weights, so this checker
    does not alter metric semantics.

    Returns:
        JSON-ready membership label, normalized components, group masses, and
        any policy violations.
    """

    validated = _validated_admissible_weight_family(family)
    raw_weights, violations = _coerce_nonnegative_weight_vector(weights)

    total = sum(raw_weights.values())
    if len(raw_weights) != len(WEIGHT_NAMES) or total <= 0.0:
        if total <= 0.0:
            violations.append("non_positive_weight_sum")
        return _weight_family_classification(validated, {}, violations)

    normalized = {name: raw_weights[name] / total for name in WEIGHT_NAMES}
    minimum = validated["component_bounds"]["minimum"]
    maximum = validated["component_bounds"]["maximum"]
    for name, value in normalized.items():
        if value < minimum:
            violations.append(f"below_component_minimum:{name}")
        if value > maximum:
            violations.append(f"above_component_maximum:{name}")

    group_masses = {
        group_name: sum(normalized[name] for name in group_weights)
        for group_name, group_weights in validated["groups"].items()
    }
    order = validated["ordered_group_masses"]
    for stronger, weaker in pairwise(order):
        if group_masses[stronger] < group_masses[weaker]:
            violations.append(f"group_order:{stronger}<{weaker}")
    return _weight_family_classification(validated, normalized, violations, group_masses)


def _validated_admissible_weight_family(family: Mapping[str, Any]) -> dict[str, Any]:
    """Return the narrow, JSON-ready family contract or reject malformed policy."""

    family_id, minimum, maximum = _validate_weight_family_header(family)
    normalized_groups = _validate_weight_family_groups(family.get("groups"))
    normalized_order = _validate_weight_family_order(
        family.get("ordered_group_masses"), normalized_groups
    )
    stress_probe_label = _validate_stress_probe_label(family.get("stress_probe_label"))
    return {
        "id": family_id,
        "normalization": "simplex_l1",
        "component_bounds": {"minimum": minimum, "maximum": maximum},
        "groups": normalized_groups,
        "ordered_group_masses": normalized_order,
        "stress_probe_label": stress_probe_label,
    }


def _validate_weight_family_header(family: Mapping[str, Any]) -> tuple[str, float, float]:
    """Validate the family identity, simplex rule, and component bounds.

    Returns:
        Family identifier plus minimum and maximum normalized component bounds.
    """

    family_id = family.get("id")
    if not isinstance(family_id, str) or not family_id.strip():
        raise ValueError("admissible_family.id must be a non-empty string")
    if family.get("normalization") != "simplex_l1":
        raise ValueError("admissible_family.normalization must be 'simplex_l1'")
    bounds = family.get("component_bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("admissible_family.component_bounds must be a mapping")
    try:
        minimum = float(bounds["minimum"])
        maximum = float(bounds["maximum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("component bounds require numeric minimum and maximum") from exc
    if not (0.0 <= minimum <= maximum <= 1.0):
        raise ValueError("component bounds must satisfy 0 <= minimum <= maximum <= 1")
    component_count = len(WEIGHT_NAMES)
    if minimum * component_count > 1.0 or maximum * component_count < 1.0:
        raise ValueError(
            "component bounds are infeasible for a normalized simplex across all SNQI weights"
        )
    return family_id, minimum, maximum


def _validate_weight_family_groups(value: Any) -> dict[str, tuple[str, ...]]:
    """Validate an exact partition of the current SNQI weight names.

    Returns:
        Group names mapped to their validated component names.
    """

    if not isinstance(value, Mapping) or not value:
        raise ValueError("admissible_family.groups must be a non-empty mapping")
    normalized_groups: dict[str, tuple[str, ...]] = {}
    seen_weights: set[str] = set()
    for group_name, names in value.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("admissible_family group names must be non-empty strings")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names:
            raise ValueError(f"admissible_family group {group_name!r} must contain weight names")
        normalized_names = tuple(str(name) for name in names)
        unknown = set(normalized_names).difference(WEIGHT_NAMES)
        duplicate = seen_weights.intersection(normalized_names)
        if unknown:
            raise ValueError(
                f"admissible_family group {group_name!r} has unknown weights: {unknown}"
            )
        if duplicate:
            raise ValueError(f"admissible_family weights appear in multiple groups: {duplicate}")
        normalized_groups[group_name] = normalized_names
        seen_weights.update(normalized_names)
    missing = set(WEIGHT_NAMES).difference(seen_weights)
    if missing:
        raise ValueError(f"admissible_family groups omit SNQI weights: {missing}")
    return normalized_groups


def _validate_weight_family_order(
    value: Any, groups: Mapping[str, Sequence[str]]
) -> tuple[str, ...]:
    """Validate the strongest-to-weakest group-mass ordering.

    Returns:
        Ordered group names from strongest to weakest allowed mass.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("admissible_family.ordered_group_masses must be a sequence")
    normalized_order = tuple(str(name) for name in value)
    if len(normalized_order) < 2:
        raise ValueError("admissible_family.ordered_group_masses needs at least two groups")
    if len(set(normalized_order)) != len(normalized_order):
        raise ValueError("admissible_family.ordered_group_masses cannot repeat groups")
    unknown_groups = set(normalized_order).difference(groups)
    if unknown_groups:
        raise ValueError(f"admissible_family ordering has unknown groups: {unknown_groups}")
    missing_groups = set(groups).difference(normalized_order)
    if missing_groups:
        raise ValueError(f"admissible_family ordering must include every group: {missing_groups}")
    return normalized_order


def _validate_stress_probe_label(value: Any) -> str:
    """Validate the durable label assigned to out-of-family vectors.

    Returns:
        The non-empty stress-probe label.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("admissible_family.stress_probe_label must be a non-empty string")
    return value


def _coerce_nonnegative_weight_vector(
    weights: Mapping[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Collect finite nonnegative weights while retaining all invalid-field reasons.

    Returns:
        Accepted component values and rejection reasons for the rest.
    """

    raw_weights: dict[str, float] = {}
    violations: list[str] = []
    for name in WEIGHT_NAMES:
        try:
            value = float(weights[name])
        except (KeyError, TypeError, ValueError):
            violations.append(f"missing_or_non_numeric:{name}")
            continue
        if not math.isfinite(value) or value < 0.0:
            violations.append(f"non_finite_or_negative:{name}")
            continue
        raw_weights[name] = value
    return raw_weights, violations


def _weight_family_classification(
    family: Mapping[str, Any],
    normalized_weights: Mapping[str, float],
    violations: Sequence[str],
    group_masses: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    admissible = not violations
    return {
        "family_id": family["id"],
        "classification": "admissible" if admissible else "stress_probe",
        "stress_probe_label": None if admissible else family["stress_probe_label"],
        "violations": list(violations),
        "normalized_weights": dict(normalized_weights),
        "group_masses": dict(group_masses or {}),
    }


def _weight_family_sensitivity_summary(
    family: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Summarize unique sweep vectors without letting stress probes set the headline.

    Returns:
        Separate full-sweep, admissible-family, and stress-probe inversion totals.
    """

    unique_variants: list[Mapping[str, Any]] = []
    seen_vectors: set[tuple[float, ...]] = set()
    for variant in variants:
        weights = variant["weights"]
        signature = tuple(float(weights[name]) for name in WEIGHT_NAMES)
        if signature not in seen_vectors:
            seen_vectors.add(signature)
            unique_variants.append(variant)

    admissible = [
        variant
        for variant in unique_variants
        if variant.get("weight_family", {}).get("classification") == "admissible"
    ]
    stress_probes = [variant for variant in unique_variants if variant not in admissible]
    return {
        "family": dict(family),
        "headline_scope": "admissible_family_only",
        "full_sweep": _inversion_summary(unique_variants),
        "admissible_family": _inversion_summary(admissible),
        "stress_probes": {
            **_inversion_summary(stress_probes),
            "label": family["stress_probe_label"],
        },
    }


def _inversion_summary(variants: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    reversals = [int(variant["pairwise_reversal_count_vs_base"]) for variant in variants]
    return {
        "vector_count": len(variants),
        "vectors_with_inversions": sum(value > 0 for value in reversals),
        "pairwise_reversal_count_vs_base": sum(reversals),
        "max_pairwise_reversal_count_vs_base": max(reversals, default=0),
    }


def write_diagnostic_artifacts(
    report: Mapping[str, Any],
    output_dir: Path,
    *,
    stem: str = "snqi_scalarization_sensitivity",
) -> DiagnosticArtifacts:
    """Write JSON, CSV, Markdown, and SVG diagnostic artifacts.

    Returns:
        Paths for the written diagnostic artifact files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}_planner_rows.csv"
    decision_disagreement_csv_path = output_dir / f"{stem}_decision_disagreement.csv"
    markdown_path = output_dir / f"{stem}.md"
    svg_path = output_dir / f"{stem}_pareto.svg"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_planner_csv(csv_path, report)
    _write_decision_disagreement_csv(decision_disagreement_csv_path, report)
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    svg_path.write_text(format_pareto_svg(report), encoding="utf-8")
    return DiagnosticArtifacts(
        json_path, csv_path, decision_disagreement_csv_path, markdown_path, svg_path
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load episode records from JSON Lines.

    Returns:
        Episode records as dictionaries.
    """

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(record)
    return records


def load_weight_mapping(path: Path | None) -> dict[str, float]:
    """Load SNQI weight mapping, accepting either raw or nested ``weights`` JSON.

    Returns:
        Complete SNQI weight mapping.
    """

    if path is None:
        return dict.fromkeys(WEIGHT_NAMES, 1.0)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("weights file must contain a JSON object")
    source = raw.get("weights", raw)
    if not isinstance(source, Mapping):
        raise ValueError("weights file 'weights' field must be a JSON object")
    return {name: float(source.get(name, 1.0)) for name in WEIGHT_NAMES}


def load_baseline_mapping(path: Path | None) -> dict[str, dict[str, float]]:
    """Load SNQI baseline normalization stats.

    Returns:
        Baseline mapping keyed by metric name.
    """

    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("baseline file must contain a JSON object")
    baseline: dict[str, dict[str, float]] = {}
    for metric in ("collisions", "near_misses", "force_exceed_events", "jerk_mean"):
        if metric not in raw:
            raise ValueError(f"baseline file missing required normalized metric {metric!r}")
    for metric, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"baseline metric {metric!r} must provide med/p95 mapping")
        try:
            med = float(entry["med"])
            p95 = float(entry["p95"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"baseline metric {metric!r} must provide finite med/p95 values"
            ) from exc
        if not math.isfinite(med) or not math.isfinite(p95):
            raise ValueError(f"baseline metric {metric!r} has non-finite med/p95 values")
        if p95 <= med:
            raise ValueError(f"baseline metric {metric!r} must satisfy p95 > med")
        baseline[str(metric)] = {"med": med, "p95": p95}
    return baseline


def input_file_provenance(path: Path | None) -> dict[str, str | None]:
    """Return path and SHA-256 provenance for an optional diagnostic input file."""

    if path is None:
        return {"path": None, "sha256": None}
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def missing_input_preflight(
    path: Path,
    *,
    input_kind: str,
    application_packet: Path | None = None,
) -> dict[str, Any]:
    """Build a blocked SNQI sensitivity preflight for a missing input artifact.

    This is for application-packet gates where the next valid step is to
    hydrate or promote the referenced input, not to emit diagnostic artifacts
    from an incomplete campaign surface.

    Returns:
        JSON-ready blocked preflight payload with missing-input provenance.
    """
    provenance: dict[str, Any] = {input_kind: {"path": str(path), "sha256": None, "exists": False}}
    if application_packet is not None:
        if application_packet.exists():
            provenance["application_packet"] = {
                **input_file_provenance(application_packet),
                "exists": True,
            }
        else:
            provenance["application_packet"] = {
                "path": str(application_packet),
                "sha256": None,
                "exists": False,
            }
    return {
        "schema_version": SCALARIZATION_SENSITIVITY_PREFLIGHT_SCHEMA,
        "ready": False,
        "status": SENSITIVITY_PREFLIGHT_BLOCKED,
        "record_count": 0,
        "planner_count": 0,
        "scenario_horizon_count": 0,
        "missing_scenario_horizon_cells": {},
        "inputs": {"provenance": provenance},
        "issues": [
            {
                "code": f"missing_{input_kind}_file",
                "severity": SENSITIVITY_PREFLIGHT_BLOCKED,
                "message": (
                    f"missing {input_kind} input file {path}; hydrate or promote the "
                    "referenced campaign artifact before exporting SNQI scalarization "
                    "sensitivity diagnostics"
                ),
            }
        ],
    }


def format_markdown(report: Mapping[str, Any]) -> str:
    """Render the report as Markdown.

    Returns:
        Markdown report text.
    """

    summary = report.get("summary", {})
    disagreement = report.get("decision_disagreement", {})
    lines = [
        "# SNQI Scalarization Sensitivity Diagnostic",
        "",
        "This is a diagnostic export for Social Navigation Quality Index (SNQI) "
        "scalarization sensitivity; it is not benchmark evidence and does not "
        "establish SNQI as a primary index.",
        "",
        "## Summary",
        "",
        f"- Decision disagreement rate: `{float(disagreement.get('pairwise_disagreement_rate', 0.0)):.6f}`",
        "- Max full weight-sweep disagreement rate vs base: "
        f"`{float(summary.get('max_weight_sweep_disagreement_rate_vs_base', 0.0)):.6f}`",
        "- Max weight-zero pairwise reversals vs base: "
        f"`{int(summary.get('max_weight_zero_pairwise_reversal_count_vs_base', 0))}`",
        f"- Top term by mean absolute contribution: `{summary.get('top_term_by_mean_abs_contribution')}`",
        "",
        "## Planner Rows",
        "",
        "| Planner | SNQI rank | Constraints-first rank | Rank delta | SNQI mean | Constraints-first score | Pareto front |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    weight_family = report.get("admissible_weight_family")
    if isinstance(weight_family, Mapping):
        admissible = weight_family.get("admissible_family", {})
        full_sweep = weight_family.get("full_sweep", {})
        stress_probes = weight_family.get("stress_probes", {})
        lines.extend(
            [
                "",
                "## Ex-Ante Admissible Weight Family",
                "",
                "Headline sensitivity statements use only the admissible family below. "
                "Nonconforming vectors remain visible as labeled stress probes.",
                "",
                "| Scope | Vectors | Vectors with inversions | Pairwise reversals vs base |",
                "|---|---:|---:|---:|",
                "| Admissible family | {vectors} | {with_inversions} | {reversals} |".format(
                    vectors=int(admissible.get("vector_count", 0)),
                    with_inversions=int(admissible.get("vectors_with_inversions", 0)),
                    reversals=int(admissible.get("pairwise_reversal_count_vs_base", 0)),
                ),
                "| Full sweep | {vectors} | {with_inversions} | {reversals} |".format(
                    vectors=int(full_sweep.get("vector_count", 0)),
                    with_inversions=int(full_sweep.get("vectors_with_inversions", 0)),
                    reversals=int(full_sweep.get("pairwise_reversal_count_vs_base", 0)),
                ),
                "| Stress probes ({label}) | {vectors} | {with_inversions} | {reversals} |".format(
                    label=str(stress_probes.get("label", "stress_probe")),
                    vectors=int(stress_probes.get("vector_count", 0)),
                    with_inversions=int(stress_probes.get("vectors_with_inversions", 0)),
                    reversals=int(stress_probes.get("pairwise_reversal_count_vs_base", 0)),
                ),
            ]
        )
    for row in report.get("planner_rows", []):
        lines.append(
            "| {planner} | {snqi_rank} | {constraints_rank} | {rank_delta:+d} | "
            "{snqi_mean:.6f} | {constraints_first_score:.6f} | {front} |".format(
                planner=str(row.get("planner", "unknown")),
                snqi_rank=int(row.get("snqi_rank", 0)),
                constraints_rank=int(row.get("constraints_first_rank", 0)),
                rank_delta=int(row.get("rank_delta", 0)),
                snqi_mean=float(row.get("snqi_mean", 0.0)),
                constraints_first_score=float(row.get("constraints_first_score", 0.0)),
                front="yes" if bool(row.get("pareto_front")) else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Term Dominance",
            "",
            "| Component | Mean abs contribution | Share |",
            "|---|---:|---:|",
        ]
    )
    for row in report.get("term_dominance", []):
        lines.append(
            "| {component} | {mean_abs:.6f} | {share:.6f} |".format(
                component=str(row.get("component", "unknown")),
                mean_abs=float(row.get("mean_abs_contribution", 0.0)),
                share=float(row.get("mean_abs_share", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Out Of Scope",
            "",
            "- No full benchmark campaign run.",
            "- No Slurm or GPU submission.",
            "- No paper or dissertation claim edits.",
        ]
    )
    return "\n".join(lines) + "\n"


def _nice_ticks(low: float, high: float, *, target_count: int = 6) -> tuple[float, ...]:
    """Return round tick values inside a numeric domain.

    Returns:
        Tick values at a step in ``{1, 2, 2.5, 5} x 10^k``.
    """

    span = high - low
    raw_step = span / max(target_count - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized_step = raw_step / magnitude
    factor = min((1.0, 2.0, 2.5, 5.0, 10.0), key=lambda value: abs(value - normalized_step))
    step = factor * magnitude
    epsilon = step * 1e-9
    first = math.ceil((low - epsilon) / step) * step
    last = math.floor((high + epsilon) / step) * step
    count = max(0, round((last - first) / step) + 1)
    precision = max(0, -math.floor(math.log10(step))) + 2
    return tuple(round(first + index * step, precision) for index in range(count))


def _format_tick(value: float, ticks: Sequence[float]) -> str:
    """Format a tick value without floating-point noise.

    Returns:
        A fixed-point tick label with enough decimals for the selected step.
    """

    step = abs(ticks[1] - ticks[0]) if len(ticks) > 1 else 1.0
    decimals = max(0, -math.floor(math.log10(step)))
    while not math.isclose(step, round(step, decimals), rel_tol=0.0, abs_tol=step * 1e-9):
        decimals += 1
    if math.isclose(value, 0.0, abs_tol=step * 1e-9):
        value = 0.0
    return f"{value:.{decimals}f}"


def _point_label_position(
    cx: float,
    cy: float,
    index: int,
    *,
    plot_left: float,
    plot_right: float,
    plot_top: float,
    plot_bottom: float,
) -> tuple[float, float, str]:
    """Place point indices in alternating quadrants while respecting plot edges.

    Returns:
        Label x/y coordinates and SVG text-anchor value.
    """

    label_dx, label_dy = ((9.0, -8.0), (-9.0, -8.0), (9.0, 14.0), (-9.0, 14.0))[(index - 1) % 4]
    if cx > plot_right - 24:
        label_dx = -9.0
    elif cx < plot_left + 24:
        label_dx = 9.0
    if cy < plot_top + 18:
        label_dy = 14.0
    elif cy > plot_bottom - 18:
        label_dy = -8.0
    anchor = "end" if label_dx < 0.0 else "start"
    return cx + label_dx, cy + label_dy, anchor


def format_pareto_svg(report: Mapping[str, Any], *, width: int = 900, height: int = 560) -> str:
    """Render a dependency-free SVG Pareto figure.

    Numeric axes use data-derived round tick intervals. Points use compact
    numeric labels, with full planner names listed in a deterministic legend.

    Returns:
        SVG figure text.
    """

    rows = [dict(row) for row in report.get("planner_rows", []) if isinstance(row, Mapping)]
    margin_left = 90
    margin_right = 30
    margin_top = 40
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    xs = [float(row.get("constraints_first_score", 0.0)) for row in rows] or [0.0]
    ys = [float(row.get("snqi_mean", 0.0)) for row in rows] or [0.0]
    x_min, x_max = _padded_domain(xs)
    y_min, y_max = _padded_domain(ys)
    x_ticks = _nice_ticks(x_min, x_max)
    y_ticks = _nice_ticks(y_min, y_max)

    def x_pos(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_height

    front = [
        row
        for row in sorted(rows, key=lambda item: float(item.get("constraints_first_score", 0.0)))
        if bool(row.get("pareto_front"))
    ]
    polyline = " ".join(
        f"{x_pos(float(row.get('constraints_first_score', 0.0))):.1f},"
        f"{y_pos(float(row.get('snqi_mean', 0.0))):.1f}"
        for row in front
    )

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            not bool(row.get("pareto_front")),
            -float(row.get("constraints_first_score", 0.0)),
            str(row.get("planner", "unknown")),
        ),
    )

    # --- build SVG ---
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}" role="img">',
        "<title>SNQI scalarization sensitivity Pareto diagnostic</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for tick in x_ticks:
        x = x_pos(tick)
        parts.extend(
            [
                f'<line class="gridline x-gridline" x1="{x:.1f}" y1="{margin_top}"'
                f' x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#E5E5E5"/>',
                f'<line class="axis-tick x-axis-tick" x1="{x:.1f}"'
                f' y1="{height - margin_bottom}" x2="{x:.1f}"'
                f' y2="{height - margin_bottom + 6}" stroke="#333"/>',
                f'<text class="tick-label x-tick-label" x="{x:.1f}"'
                f' y="{height - margin_bottom + 22}" text-anchor="middle"'
                f' font-family="sans-serif" font-size="11">'
                f"{_format_tick(tick, x_ticks)}</text>",
            ]
        )
    for tick in y_ticks:
        y = y_pos(tick)
        parts.extend(
            [
                f'<line class="gridline y-gridline" x1="{margin_left}" y1="{y:.1f}"'
                f' x2="{width - margin_right}" y2="{y:.1f}" stroke="#E5E5E5"/>',
                f'<line class="axis-tick y-axis-tick" x1="{margin_left - 6}" y1="{y:.1f}"'
                f' x2="{margin_left}" y2="{y:.1f}" stroke="#333"/>',
                f'<text class="tick-label y-tick-label" x="{margin_left - 10}" y="{y + 4:.1f}"'
                f' text-anchor="end" font-family="sans-serif" font-size="11">'
                f"{_format_tick(tick, y_ticks)}</text>",
            ]
        )

    parts.extend(
        [
            f'<line class="axis x-axis" x1="{margin_left}" y1="{height - margin_bottom}"'
            f' x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333"/>',
            f'<line class="axis y-axis" x1="{margin_left}" y1="{margin_top}"'
            f' x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
            f'<text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle"'
            f' font-family="sans-serif" font-size="15">'
            "Constraints-first score (higher is better)</text>",
            f'<text x="24" y="{height / 2:.1f}" text-anchor="middle"'
            f' transform="rotate(-90 24 {height / 2:.1f})" font-family="sans-serif"'
            f' font-size="15">SNQI mean (higher is better)</text>',
        ]
    )

    if polyline:
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>'
        )

    for index, row in enumerate(ranked_rows, start=1):
        cx = x_pos(float(row.get("constraints_first_score", 0.0)))
        cy = y_pos(float(row.get("snqi_mean", 0.0)))
        is_front = bool(row.get("pareto_front"))
        color = "#1f77b4" if is_front else "#777777"
        name = str(row.get("planner", "unknown"))
        point_class = "front" if is_front else "non-front"
        label_x, label_y, label_anchor = _point_label_position(
            cx,
            cy,
            index,
            plot_left=margin_left,
            plot_right=width - margin_right,
            plot_top=margin_top,
            plot_bottom=height - margin_bottom,
        )
        parts.extend(
            [
                f'<circle class="point-halo {point_class}" cx="{cx:.1f}" cy="{cy:.1f}"'
                ' r="7" fill="white"/>',
                f'<circle class="pareto-point {point_class}" data-planner="{html.escape(name)}"'
                f' cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"/>',
                f'<text class="point-index {point_class}" x="{label_x:.1f}" y="{label_y:.1f}"'
                f' text-anchor="{label_anchor}" font-family="sans-serif" font-size="11"'
                f' font-weight="600" fill="{color}">{index}</text>',
            ]
        )

    if ranked_rows:
        legend_x = margin_left + 12
        legend_y = margin_top + 12
        legend_width = min(
            plot_width - 24,
            max(
                230.0,
                max(len(str(row.get("planner", "unknown"))) for row in ranked_rows) * 5.8 + 48,
            ),
        )
        legend_height = 30 + len(ranked_rows) * 16
        parts.extend(
            [
                f'<rect class="planner-legend" x="{legend_x}" y="{legend_y}"'
                f' width="{legend_width:.1f}" height="{legend_height}" rx="4"'
                ' fill="#ffffff" fill-opacity="0.94" stroke="#CCCCCC"/>',
                f'<text x="{legend_x + 10}" y="{legend_y + 17}" font-family="sans-serif"'
                ' font-size="11" font-weight="600" fill="#333">Planner legend</text>',
            ]
        )
        for index, row in enumerate(ranked_rows, start=1):
            entry_y = legend_y + 26 + index * 16
            is_front = bool(row.get("pareto_front"))
            color = "#1f77b4" if is_front else "#777777"
            name = html.escape(str(row.get("planner", "unknown")))
            parts.extend(
                [
                    f'<circle class="legend-swatch" cx="{legend_x + 12}" cy="{entry_y - 4}"'
                    f' r="4" fill="{color}"/>',
                    f'<text class="legend-entry" x="{legend_x + 22}" y="{entry_y}"'
                    f' font-family="sans-serif" font-size="10" fill="#333">'
                    f"{index}. {name}</text>",
                ]
            )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _group_records(
    records: Iterable[Mapping[str, Any]], planner_key: str, fallback_planner_key: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, record in enumerate(records, start=1):
        planner = _get_nested(record, planner_key)
        if planner in (None, ""):
            planner = _get_nested(record, fallback_planner_key)
        if planner in (None, ""):
            planner = _get_nested(record, "scenario_params.algo")
        if planner in (None, ""):
            raise ValueError(f"record {index} missing planner key {planner_key!r}")
        grouped.setdefault(str(planner), []).append(record)
    return grouped


def _validate_export_required_terms(
    records: Sequence[Mapping[str, Any]], weights: Mapping[str, float]
) -> None:
    """Fail closed before export when required SNQI terms would otherwise default."""

    for index, record in enumerate(records, start=1):
        metrics = _metrics(record)
        for metric in REQUIRED_SENSITIVITY_METRICS:
            if metric not in metrics:
                raise ValueError(
                    f"record {index} missing required SNQI term {metric!r}; "
                    "run scalarization-sensitivity preflight before export"
                )
            if not _is_finite_metric(metrics.get(metric)):
                raise ValueError(
                    f"record {index} non-finite required SNQI term {metric!r}; "
                    "run scalarization-sensitivity preflight before export"
                )
            if metric in BOUNDED_NORMALIZED_SENSITIVITY_METRICS and not _is_unit_interval(
                metrics.get(metric)
            ):
                raise ValueError(
                    f"record {index} normalized SNQI term {metric!r} outside [0, 1]; "
                    "run scalarization-sensitivity preflight before export"
                )
        for metric in OPTIONAL_SENSITIVITY_METRICS:
            if metric not in metrics:
                if _is_active_optional_weight(weights, metric):
                    raise ValueError(
                        f"record {index} missing weighted SNQI term {metric!r}; "
                        "run scalarization-sensitivity preflight before export"
                    )
            elif not _is_finite_metric(metrics.get(metric)):
                raise ValueError(
                    f"record {index} non-finite optional SNQI term {metric!r}; "
                    "run scalarization-sensitivity preflight before export"
                )


def _planner_snqi_scores(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for planner, rows in grouped.items():
        episode_scores = [compute_snqi(_metrics(row), weights, baseline) for row in rows]
        scores[planner] = _mean(episode_scores)
    return scores


def _constraints_first_endpoint(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    n = len(rows)
    if n == 0:
        raise ValueError("planner must have at least one episode")
    metrics = [_metrics(row) for row in rows]
    success_rate = _mean(_metric_float(row, "success", 0.0) for row in metrics)
    collision_rate = _mean(
        1.0 if _metric_float(row, "collisions", 0.0) > 0.0 else 0.0 for row in metrics
    )
    near_miss_rate = _mean(
        1.0 if _metric_float(row, "near_misses", 0.0) > 0.0 else 0.0 for row in metrics
    )
    timeout_rate = _mean(
        1.0 if _metric_float(row, "timeout", 0.0) > 0.0 else 0.0 for row in metrics
    )
    deadlock_rate = _mean(
        1.0 if _metric_float(row, "deadlock", 0.0) > 0.0 else 0.0 for row in metrics
    )
    time_mean = _mean(_metric_float(row, "time_to_goal_norm", 1.0) for row in metrics)
    constraints_first_score = (
        success_rate
        - collision_rate
        - (0.5 * near_miss_rate)
        - (0.25 * timeout_rate)
        - (0.25 * deadlock_rate)
        - (0.01 * time_mean)
    )
    return {
        "episodes": n,
        "success_rate": success_rate,
        "collision_event_rate": collision_rate,
        "near_miss_event_rate": near_miss_rate,
        "timeout_rate": timeout_rate,
        "deadlock_rate": deadlock_rate,
        "time_to_goal_norm_mean": time_mean,
        "constraints_first_score": constraints_first_score,
    }


def _planner_rows(
    snqi_scores: Mapping[str, float],
    constraints: Mapping[str, Mapping[str, float | int]],
    snqi_order: Sequence[str],
    constraints_order: Sequence[str],
) -> list[dict[str, Any]]:
    snqi_ranks = {planner: index + 1 for index, planner in enumerate(snqi_order)}
    constraints_ranks = {planner: index + 1 for index, planner in enumerate(constraints_order)}
    rows: list[dict[str, Any]] = []
    for planner in snqi_order:
        endpoint = constraints[planner]
        rows.append(
            {
                "planner": planner,
                "snqi_rank": snqi_ranks[planner],
                "constraints_first_rank": constraints_ranks[planner],
                "rank_delta": constraints_ranks[planner] - snqi_ranks[planner],
                "snqi_mean": float(snqi_scores[planner]),
                **endpoint,
            }
        )
    front_names = {row["planner"] for row in _pareto_points(rows)}
    for row in rows:
        row["pareto_front"] = row["planner"] in front_names
    return rows


def _pareto_points(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        x = float(row.get("constraints_first_score", 0.0))
        y = float(row.get("snqi_mean", 0.0))
        dominated = False
        for other in rows:
            if other is row:
                continue
            ox = float(other.get("constraints_first_score", 0.0))
            oy = float(other.get("snqi_mean", 0.0))
            if ox >= x and oy >= y and (ox > x or oy > y):
                dominated = True
                break
        if not dominated:
            points.append(
                {
                    "planner": str(row.get("planner", "unknown")),
                    "constraints_first_score": x,
                    "snqi_mean": y,
                }
            )
    return sorted(points, key=lambda item: (item["constraints_first_score"], item["snqi_mean"]))


def _variant_summary(base_order: Sequence[str], variant_order: Sequence[str]) -> dict[str, Any]:
    return {
        "order": list(variant_order),
        "winner_changed": bool(base_order and variant_order and base_order[0] != variant_order[0]),
        "pairwise_reversal_count_vs_base": _pairwise_reversal_count(base_order, variant_order),
        "pairwise_disagreement_rate_vs_base": _pairwise_disagreement_rate(
            base_order, variant_order
        ),
        "kendall_tau_vs_base": kendall_tau(base_order, variant_order, degenerate=0.0),
        "spearman_rho_vs_base": spearman_from_order(base_order, variant_order, degenerate=0.0),
    }


def _rank_disagreement(
    snqi_order: Sequence[str], constraints_order: Sequence[str]
) -> dict[str, Any]:
    return {
        "pairwise_reversal_count": _pairwise_reversal_count(snqi_order, constraints_order),
        "pairwise_disagreement_rate": _pairwise_disagreement_rate(snqi_order, constraints_order),
        "kendall_tau": kendall_tau(snqi_order, constraints_order, degenerate=0.0),
        "spearman_rho": spearman_from_order(snqi_order, constraints_order, degenerate=0.0),
        "snqi_winner": snqi_order[0] if snqi_order else None,
        "constraints_first_winner": constraints_order[0] if constraints_order else None,
        "winner_disagreement": bool(
            snqi_order and constraints_order and snqi_order[0] != constraints_order[0]
        ),
    }


def _pairwise_reversal_count(left_order: Sequence[str], right_order: Sequence[str]) -> int:
    if set(left_order) != set(right_order):
        raise ValueError("rank orders must contain the same planners")
    right_pos = {planner: index for index, planner in enumerate(right_order)}
    reversals = 0
    for left_index, planner in enumerate(left_order):
        for other in left_order[left_index + 1 :]:
            if right_pos[planner] > right_pos[other]:
                reversals += 1
    return reversals


def _pairwise_disagreement_rate(left_order: Sequence[str], right_order: Sequence[str]) -> float:
    possible = len(left_order) * (len(left_order) - 1) / 2
    if possible <= 0:
        return 0.0
    return float(_pairwise_reversal_count(left_order, right_order) / possible)


def _term_dominance(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    contributions: dict[str, list[float]] = {name: [] for name in WEIGHT_NAMES}
    for rows in grouped.values():
        for row in rows:
            metrics = _metrics(row)
            contributions["w_success"].append(
                weights.get("w_success", 1.0) * _metric_float(metrics, "success", 0.0)
            )
            contributions["w_time"].append(
                -weights.get("w_time", 1.0) * _metric_float(metrics, "time_to_goal_norm", 1.0)
            )
            contributions["w_collisions"].append(
                -weights.get("w_collisions", 1.0)
                * normalize_metric(
                    "collisions", _metric_float(metrics, "collisions", 0.0), baseline
                )
            )
            contributions["w_near"].append(
                -weights.get("w_near", 1.0)
                * normalize_metric(
                    "near_misses", _metric_float(metrics, "near_misses", 0.0), baseline
                )
            )
            contributions["w_comfort"].append(
                -weights.get("w_comfort", 1.0) * _metric_float(metrics, "comfort_exposure", 0.0)
            )
            contributions["w_force_exceed"].append(
                -weights.get("w_force_exceed", 1.0)
                * normalize_metric(
                    "force_exceed_events",
                    _metric_float(metrics, "force_exceed_events", 0.0),
                    baseline,
                )
            )
            contributions["w_jerk"].append(
                -weights.get("w_jerk", 1.0)
                * normalize_metric("jerk_mean", _metric_float(metrics, "jerk_mean", 0.0), baseline)
            )
    mean_abs = {
        name: _mean(abs(value) for value in values) if values else 0.0
        for name, values in contributions.items()
    }
    total = sum(mean_abs.values())
    rows = [
        {
            "component": name,
            "mean_abs_contribution": value,
            "mean_abs_share": value / total if total > 0.0 else 0.0,
        }
        for name, value in mean_abs.items()
    ]
    return sorted(rows, key=lambda row: (-row["mean_abs_contribution"], row["component"]))


def _write_planner_csv(path: Path, report: Mapping[str, Any]) -> None:
    headers = [
        "planner",
        "snqi_rank",
        "constraints_first_rank",
        "rank_delta",
        "snqi_mean",
        "constraints_first_score",
        "success_rate",
        "collision_event_rate",
        "near_miss_event_rate",
        "pareto_front",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in report.get("planner_rows", []):
            writer.writerow({header: row.get(header, "") for header in headers})


def _write_decision_disagreement_csv(path: Path, report: Mapping[str, Any]) -> None:
    """Write the scalar-vs-constraints decision-disagreement table."""

    disagreement = report.get("decision_disagreement", {})
    rows = []
    if isinstance(disagreement, Mapping):
        rows.append(
            {
                "comparison": "base_snqi_vs_constraints_first",
                "left_order": " > ".join(str(item) for item in report.get("base_snqi_order", [])),
                "right_order": " > ".join(
                    str(item) for item in report.get("constraints_first_order", [])
                ),
                "pairwise_reversal_count": disagreement.get("pairwise_reversal_count", ""),
                "pairwise_disagreement_rate": disagreement.get("pairwise_disagreement_rate", ""),
                "claim_boundary": report.get("claim_boundary", ""),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison",
                "left_order",
                "right_order",
                "pairwise_reversal_count",
                "pairwise_disagreement_rate",
                "claim_boundary",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _rank_order(scores: Mapping[str, float], *, higher_is_better: bool) -> list[str]:
    return [
        key
        for key, _value in sorted(
            scores.items(),
            key=lambda item: (
                -float(item[1]) if higher_is_better else float(item[1]),
                str(item[0]),
            ),
        )
    ]


def _metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = record.get("metrics")
    return metrics if isinstance(metrics, Mapping) else record


def _metric_float(metrics: Mapping[str, Any], key: str, default: float) -> float:
    value = metrics.get(key, default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _is_finite_metric(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_unit_interval(value: Any) -> bool:
    return 0.0 <= float(value) <= 1.0


def _is_active_weight(value: Any) -> bool:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(weight) and weight != 0.0


def _is_active_optional_weight(weights: Any, metric: str) -> bool:
    """Whether ``metric``'s SNQI weight is present and active in ``weights``.

    Returns:
        ``True`` when ``weights`` is a mapping carrying an active (finite,
        nonzero) weight for ``metric``. Returns ``False`` when ``weights`` is
        not a mapping, so malformed-weight inputs are reported by the dedicated
        ``weights_not_mapping`` check instead of crashing the preflight/export
        guards with ``AttributeError``.
    """

    if not isinstance(weights, Mapping):
        return False
    weight_name = OPTIONAL_WEIGHTED_SENSITIVITY_METRICS[metric]
    return _is_active_weight(weights.get(weight_name, 0.0))


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return 0.0
    return float(sum(clean) / len(clean))


def _get_nested(record: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = record
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _padded_domain(values: Sequence[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        pad = max(abs(low) * 0.05, 1.0)
    else:
        pad = (high - low) * 0.08
    return low - pad, high + pad


__all__ = [
    "ADMISSIBLE_WEIGHT_FAMILY_SCHEMA",
    "DEFAULT_SWEEP_FACTORS",
    "OPTIONAL_SENSITIVITY_METRICS",
    "REQUIRED_SENSITIVITY_METRICS",
    "SCALARIZATION_SENSITIVITY_PREFLIGHT_SCHEMA",
    "SCALARIZATION_SENSITIVITY_SCHEMA",
    "SENSITIVITY_PREFLIGHT_BLOCKED",
    "SENSITIVITY_PREFLIGHT_MALFORMED",
    "SENSITIVITY_PREFLIGHT_READY",
    "DiagnosticArtifacts",
    "SensitivityPreflightIssue",
    "build_scalarization_sensitivity_report",
    "classify_admissible_weight_vector",
    "classify_scalarization_sensitivity_inputs",
    "format_markdown",
    "format_pareto_svg",
    "input_file_provenance",
    "load_admissible_weight_family_config",
    "load_baseline_mapping",
    "load_jsonl",
    "load_weight_mapping",
    "missing_input_preflight",
    "write_diagnostic_artifacts",
]
