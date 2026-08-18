"""Collision-envelope radius rank-stability analysis (issue #6643, Gate 3 of #6600).

Given the Gate 2 production radius sweep (planner metric tables measured at robot
collision-envelope radii 0.5 m, 0.8 m, and the 1.0 m release baseline), this module
reports whether the planner **ranking** is identifiable and, when it is, stable
across the tested radii. That is the evidence behind the #6600 validity-boundary
verdict and its propagation to the parent validity study (#3207).

The module produces, without running any simulation:

- planner-ranking tables for success, typed collisions, and SNQI;
- Kendall rank correlation and rank-flip counts versus the 1.0 m baseline;
- per-planner paired changes with a deterministic paired-bootstrap uncertainty;
- scenario-family and feasibility transitions, including the narrow-doorway family;
- a fail-closed missingness/degradation ledger;
- exactly one verdict from the preregistered vocabulary.

Pure and deterministic: it operates on an already-measured sweep summary and runs no
simulation. It is analysis tooling and makes no benchmark, realism, sim-to-real, or
safety claim; radius perturbations are deliberate within-simulator sensitivity probes,
not fallback/degraded runs. The Gate 2 sweep *execution* lives with child issue #6642.

Fail-closed gate precedence (mirrors the #6600 stop rules):

1. No sweep summary available -> ``blocked_pending_gate2`` (a pre-analysis gate status,
   not a scientific verdict). The analysis does not run.
2. Incomplete row-identity accounting, or any fallback/degraded/failed/missing/
   duplicate/provenance-invalid row -> ``invalid_missing_or_inconsistent_evidence`` and
   interpretation stops (no ranking claim is promoted).
3. Otherwise a non-identifiable ranking -> ``non_identifiable``.
4. Otherwise any ranking flip versus baseline -> ``radius_dependent`` (a ranking flip is
   a valid boundary result, not a failed experiment).
5. Otherwise -> ``stable_within_tested_radii``.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.fidelity_rank_stability import (
    count_rank_flips,
    kendall_tau,
    rank_planners,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.radius_sweep_manifest import (
    EXPECTED_ROWS_PER_ARM,
    EXPECTED_SCENARIO_MATRIX,
    EXPECTED_SCENARIO_NAMES,
    EXPECTED_TOTAL_ROWS,
)
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# The report envelope gained required paired-inference/support blocks in #7139.
# Bump both the report and durable bundle identifiers before Gate 2 can produce
# retained artifacts; no prior Gate 2 bundle exists to migrate.
RADIUS_RANK_STABILITY_SCHEMA = "radius_rank_stability.v2"
RADIUS_EVIDENCE_BUNDLE_SCHEMA = "issue_6643_radius_rank_stability_bundle.v2"
SWEEP_SUMMARY_SCHEMA = "issue_6642_radius_sweep_summary.v1"

# Preregistered scientific verdicts (exactly one is emitted once the gate passes).
VERDICT_STABLE = "stable_within_tested_radii"
VERDICT_RADIUS_DEPENDENT = "radius_dependent"
VERDICT_NON_IDENTIFIABLE = "non_identifiable"
VERDICT_INVALID = "invalid_missing_or_inconsistent_evidence"
SCIENTIFIC_VERDICTS = frozenset(
    {VERDICT_STABLE, VERDICT_RADIUS_DEPENDENT, VERDICT_NON_IDENTIFIABLE, VERDICT_INVALID}
)

# Pre-analysis gate status emitted when the Gate 2 sweep is not yet available. This is
# deliberately NOT one of the scientific verdicts: a planned, not-yet-run sweep is not
# the same as a failed or invalid experiment.
ANALYSIS_BLOCKED_PENDING_GATE2 = "blocked_pending_gate2"

# Rank metrics reported as planner-ranking tables.
RANK_METRIC_SUCCESS = "success"
RANK_METRIC_TYPED_COLLISIONS = "typed_collisions"
RANK_METRIC_SNQI = "snqi"
DEFAULT_RANK_METRICS: tuple[str, ...] = (
    RANK_METRIC_SUCCESS,
    RANK_METRIC_TYPED_COLLISIONS,
    RANK_METRIC_SNQI,
)
EXPECTED_RADIUS_ARMS: tuple[float, ...] = (0.5, 0.8, 1.0)
EXPECTED_SCENARIO_CELL_COUNT = 48
EXPECTED_SEED_ROSTER: tuple[int, ...] = tuple(range(111, 141))
EXPECTED_PLANNER_ROSTER: tuple[str, ...] = (
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
)
# Success and SNQI are higher-is-better; typed collisions are lower-is-better.
LOWER_IS_BETTER_METRICS = frozenset({RANK_METRIC_TYPED_COLLISIONS})

# Fail-closed row-exclusion reasons. Any such row is removed from evidence and, when it
# breaks the matched design or its accounting, forces the invalid-evidence verdict.
EXCLUSION_FALLBACK = "fallback"
EXCLUSION_DEGRADED = "degraded"
EXCLUSION_FAILED = "failed"
EXCLUSION_MISSING = "missing"
EXCLUSION_DUPLICATE = "duplicate"
EXCLUSION_PROVENANCE_INVALID = "provenance_invalid"
EXCLUSION_REASONS = frozenset(
    {
        EXCLUSION_FALLBACK,
        EXCLUSION_DEGRADED,
        EXCLUSION_FAILED,
        EXCLUSION_MISSING,
        EXCLUSION_DUPLICATE,
        EXCLUSION_PROVENANCE_INVALID,
    }
)

# Identifiability reasons (shared vocabulary with fidelity_rank_stability).
PRIMARY_METRIC_ZERO_VARIANCE_REASON = "primary_metric_zero_variance"
PRIMARY_METRIC_INSUFFICIENT_REASON = "primary_metric_insufficient_finite_values"

# Feasibility status vocabulary for scenario-family transitions.
FEASIBILITY_FEASIBLE = "feasible"
FEASIBILITY_INFEASIBLE = "infeasible"
FEASIBILITY_DEGRADED = "degraded"

NARROW_DOORWAY_FAMILY = "narrow_doorway"

# Claim-boundary phrases that every emitted bundle and verdict comment must carry so the
# result cannot be read as a realism, sim-to-real, or safety claim.
REQUIRED_CLAIM_BOUNDARY_PHRASES: tuple[str, ...] = (
    "within-simulator radius sensitivity only",
    "not physical-footprint validation",
    "not simulator-realism evidence",
    "not sim-to-real evidence",
    "not a safety guarantee",
)

# Gate 1 receipt identity. A matching digest is necessary but not sufficient: the
# receipt must be the machine-readable all-surface ``go`` report for the complete
# 0.5/0.8/1.0 m canary treatment.
GATE1_CANARY_REPORT_SCHEMA = "radius_binding_canary_report.v1"
GATE1_CANARY_VERDICT_SCHEMA = "radius_binding_canary.v1"
GATE1_CANARY_ISSUE = 6641
GATE1_CANARY_PARENT_ISSUE = 6600
GATE1_CANARY_RADII: tuple[float, ...] = (0.5, 0.8, 1.0)
GATE1_CANARY_SURFACES = frozenset(
    {
        "simulator_collision_geometry",
        "obstacle_pedestrian_contact_logic",
        "feasibility_oracle",
        "metric_metadata_and_output_rows",
        "planner_inputs",
    }
)


def _radius_key(radius: float) -> str:
    """Return the canonical string key used for a radius in serialized output."""
    return f"{float(radius):g}"


def _float_keyed(mapping: object) -> dict[float, object]:
    """Return a radius-keyed mapping normalized to numeric radius keys.

    Sweep summaries may key radius arms as ``"1.0"``, ``"1"``, or ``"0.5"``; normalizing
    to ``float`` on input makes lookups robust to the author's string spelling. Keys that
    do not parse as finite floats are dropped.
    """
    if not isinstance(mapping, Mapping):
        return {}
    normalized: dict[float, object] = {}
    for key, value in mapping.items():
        try:
            radius = float(key)
        except (TypeError, ValueError):
            continue
        if math.isfinite(radius):
            normalized[radius] = value
    return normalized


def _finite_metric_value(metrics: Mapping[str, object], metric: str) -> float | None:
    """Return a finite numeric metric value, or ``None`` when unavailable."""
    raw_value = metrics.get(metric)
    if raw_value is None or isinstance(raw_value, bool):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _metric_identifiable(
    table: Mapping[str, Mapping[str, object]],
    metric: str,
    *,
    expected_planners: Iterable[str] | None = None,
) -> tuple[bool, str | None]:
    """Return whether planner ranks are identifiable from ``metric`` values.

    A ranking is identifiable when at least two planners carry finite metric values and
    those values are not all tied. All-tied or insufficient values are non-identifiable
    because any reported order would be an artifact of the deterministic tie-break only.
    """
    expected = set(expected_planners or ())
    if expected and set(table) != expected:
        return False, PRIMARY_METRIC_INSUFFICIENT_REASON

    values = [
        value
        for planner in sorted(table)
        if (value := _finite_metric_value(table[planner], metric)) is not None
    ]
    if len(values) < 2 or (expected and len(values) != len(expected)):
        return False, PRIMARY_METRIC_INSUFFICIENT_REASON
    first = values[0]
    if all(value == first for value in values[1:]):
        return False, PRIMARY_METRIC_ZERO_VARIANCE_REASON
    return True, None


# --- row admission and missingness ledger ---------------------------------


@dataclass(frozen=True)
class RadiusRowAdmission:
    """Fail-closed row-identity admission result for one radius arm."""

    radius: float
    declared: int
    present: int
    excluded_by_reason: dict[str, int]
    accounting_complete: bool
    excluded_total: int
    disqualifying_reasons: tuple[str, ...]
    accounting_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation.

        Returns:
            Mapping of radius, declared/present counts, exclusions, and admission flags.
        """
        return {
            "radius": self.radius,
            "declared": self.declared,
            "present": self.present,
            "excluded_by_reason": dict(self.excluded_by_reason),
            "excluded_total": self.excluded_total,
            "accounting_complete": self.accounting_complete,
            "disqualifying_reasons": list(self.disqualifying_reasons),
            "accounting_errors": list(self.accounting_errors),
        }


@dataclass(frozen=True)
class MissingnessLedger:
    """Aggregate fail-closed missingness/degradation ledger across all radius arms."""

    radii: tuple[float, ...]
    per_radius: tuple[RadiusRowAdmission, ...]
    declared_total: int
    present_total: int
    excluded_total: int
    excluded_by_reason: dict[str, int]
    complete: bool
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation.

        Returns:
            Mapping of aggregate counts, per-radius admissions, and blocking reasons.
        """
        return {
            "radii": list(self.radii),
            "declared_total": self.declared_total,
            "present_total": self.present_total,
            "excluded_total": self.excluded_total,
            "excluded_by_reason": dict(self.excluded_by_reason),
            "complete": self.complete,
            "blocking_reasons": list(self.blocking_reasons),
            "per_radius": [entry.to_dict() for entry in self.per_radius],
        }


def _parse_nonnegative_count(
    accounting: Mapping[str, object],
    field: str,
) -> tuple[int, str | None]:
    """Parse one JSON row-accounting count without coercing malformed values.

    Returns:
        The non-negative integer and an error label, or zero and an error label.
    """
    if field not in accounting:
        return 0, f"missing_{field}"
    value = accounting[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0, f"invalid_{field}"
    return value, None


def _admit_radius_rows(radius: float, accounting: Mapping[str, object]) -> RadiusRowAdmission:
    """Build the fail-closed admission result for one radius arm.

    Accounting is complete only when every declared row is reconciled as either present
    or excluded for a known reason. Any exclusion reason is disqualifying for evidence
    because the matched design requires every declared row to be valid native output.

    Returns:
        The per-radius admission result with completeness and disqualifying reasons.
    """
    declared, declared_error = _parse_nonnegative_count(accounting, "declared")
    present, present_error = _parse_nonnegative_count(accounting, "present")
    accounting_errors = [error for error in (declared_error, present_error) if error is not None]
    raw_excluded = accounting.get("excluded_by_reason")
    excluded_by_reason: dict[str, int] = {}
    if not isinstance(raw_excluded, Mapping):
        accounting_errors.append("missing_or_invalid_excluded_by_reason")
    else:
        for reason, count in raw_excluded.items():
            reason_text = str(reason)
            parsed_count, count_error = _parse_nonnegative_count({"count": count}, "count")
            if count_error is not None:
                accounting_errors.append(f"{count_error}:{reason_text}")
                continue
            excluded_by_reason[reason_text] = parsed_count
            if reason_text not in EXCLUSION_REASONS:
                accounting_errors.append(f"unknown_exclusion_reason:{reason_text}")
    excluded_total = sum(excluded_by_reason.values())
    if declared != EXPECTED_ROWS_PER_ARM:
        accounting_errors.append(
            f"unexpected_declared_row_count:{declared}:expected_{EXPECTED_ROWS_PER_ARM}"
        )
    accounting_complete = (
        not accounting_errors
        and declared == present + excluded_total
        and declared > 0
        and present <= declared
    )
    disqualifying = tuple(
        reason
        for reason in sorted(excluded_by_reason)
        if excluded_by_reason[reason] > 0 and reason in EXCLUSION_REASONS
    )
    return RadiusRowAdmission(
        radius=radius,
        declared=declared,
        present=present,
        excluded_by_reason=excluded_by_reason,
        accounting_complete=accounting_complete,
        excluded_total=excluded_total,
        disqualifying_reasons=disqualifying,
        accounting_errors=tuple(accounting_errors),
    )


def _declared_planner_roster(
    sweep_summary: Mapping[str, object],
) -> tuple[tuple[str, ...], list[str]]:
    """Return the declared planner roster and structural roster errors."""
    raw_planners = sweep_summary.get("planners")
    if not isinstance(raw_planners, (list, tuple)):
        return (), ["missing_planner_roster"]
    planners = tuple(planner for planner in raw_planners if isinstance(planner, str) and planner)
    errors: list[str] = []
    if len(planners) != len(raw_planners):
        errors.append("invalid_planner_roster")
    if len(set(planners)) != len(planners):
        errors.append("duplicate_planner_roster_entry")
    if set(planners) != set(EXPECTED_PLANNER_ROSTER):
        errors.append("unexpected_planner_roster")
    return planners, errors


def _metric_table_blockers(
    sweep_summary: Mapping[str, object],
    radii: Sequence[float],
    declared_planners: tuple[str, ...],
) -> tuple[dict[float, Mapping[str, Mapping[str, object]]], list[str]]:
    """Return normalized metric tables and coverage errors."""
    raw_tables = sweep_summary.get("metric_tables")
    if not isinstance(raw_tables, Mapping):
        return {}, ["missing_metric_tables"]

    tables = _metric_tables_by_radius(sweep_summary)
    blockers = _metric_table_radius_blockers(raw_tables, radii)
    blockers.extend(_metric_table_coverage_blockers(raw_tables, tables, radii, declared_planners))
    return tables, blockers


def _metric_table_radius_blockers(
    raw_tables: Mapping[object, object],
    radii: Sequence[float],
) -> list[str]:
    """Return errors in the metric-table radius keys."""
    blockers: list[str] = []
    seen_radii: set[float] = set()
    for raw_radius in raw_tables:
        try:
            radius = float(raw_radius)
        except (TypeError, ValueError):
            blockers.append(f"invalid_metric_table_radius:{raw_radius}")
            continue
        if not math.isfinite(radius):
            blockers.append(f"invalid_metric_table_radius:{raw_radius}")
        elif radius not in radii:
            blockers.append(f"undeclared_metric_table_radius:{_radius_key(radius)}")
        elif radius in seen_radii:
            blockers.append(f"duplicate_metric_table_radius:{_radius_key(radius)}")
        else:
            seen_radii.add(radius)
    return blockers


def _metric_table_coverage_blockers(
    raw_tables: Mapping[object, object],
    tables: Mapping[float, object],
    radii: Sequence[float],
    declared_planners: tuple[str, ...],
) -> list[str]:
    """Return errors in per-radius planner and row coverage."""
    blockers: list[str] = []
    expected_planners = set(declared_planners)
    for radius in radii:
        raw_table = _raw_metric_table_for_radius(raw_tables, radius)
        table = tables.get(radius)
        if not isinstance(raw_table, Mapping) or not isinstance(table, Mapping):
            blockers.append(f"radius_{_radius_key(radius)}_missing_metric_table")
            continue
        if declared_planners and set(table) != expected_planners:
            blockers.append(f"radius_{_radius_key(radius)}_planner_roster_mismatch")
        if any(
            not isinstance(planner, str) or not planner or not isinstance(row, Mapping)
            for planner, row in raw_table.items()
        ):
            blockers.append(f"radius_{_radius_key(radius)}_invalid_metric_row")
            continue
        for planner, row in sorted(table.items()):
            for metric in DEFAULT_RANK_METRICS:
                if _finite_metric_value(row, metric) is None:
                    blockers.append(
                        f"radius_{_radius_key(radius)}_planner_{planner}_invalid_metric:{metric}"
                    )
    return blockers


def _raw_metric_table_for_radius(
    raw_tables: Mapping[object, object],
    radius: float,
) -> object:
    """Return the original metric table for a normalized radius key."""
    for raw_radius, table in raw_tables.items():
        try:
            parsed_radius = float(raw_radius)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed_radius) and parsed_radius == radius:
            return table
    return None


def _row_accounting_radius_blockers(
    sweep_summary: Mapping[str, object],
    radii: Sequence[float],
) -> list[str]:
    """Return accounting entries that do not correspond to declared radii."""
    raw_accounting = sweep_summary.get("row_accounting")
    if not isinstance(raw_accounting, Mapping):
        return []
    blockers: list[str] = []
    seen_radii: set[float] = set()
    for raw_radius in raw_accounting:
        try:
            radius = float(raw_radius)
        except (TypeError, ValueError):
            blockers.append(f"invalid_row_accounting_radius:{raw_radius}")
            continue
        if not math.isfinite(radius) or radius not in radii:
            blockers.append(f"undeclared_row_accounting_radius:{raw_radius}")
        elif radius in seen_radii:
            blockers.append(f"duplicate_row_accounting_radius:{_radius_key(radius)}")
        else:
            seen_radii.add(radius)
    return blockers


def _fixed_scope_blockers(sweep_summary: Mapping[str, object]) -> list[str]:
    """Return blockers for the Gate 2 scenario and seed scope."""
    blockers: list[str] = []
    if sweep_summary.get("scenario_matrix") != EXPECTED_SCENARIO_MATRIX:
        blockers.append("scenario_matrix_mismatch")
    raw_cells = sweep_summary.get("scenario_cells")
    if not isinstance(raw_cells, (list, tuple)):
        blockers.append("missing_scenario_cells")
    elif len(raw_cells) != EXPECTED_SCENARIO_CELL_COUNT:
        blockers.append(f"unexpected_scenario_cell_count:{len(raw_cells)}")
    elif not all(isinstance(cell, str) and cell for cell in raw_cells):
        blockers.append("invalid_scenario_cell_identity")
    elif len(set(raw_cells)) != len(raw_cells):
        blockers.append("duplicate_scenario_cell")
    elif set(raw_cells) != set(EXPECTED_SCENARIO_NAMES):
        blockers.append("scenario_cell_roster_mismatch")

    raw_seeds = sweep_summary.get("seeds")
    if not isinstance(raw_seeds, (list, tuple)):
        blockers.append("missing_seed_roster")
    elif tuple(raw_seeds) != EXPECTED_SEED_ROSTER:
        blockers.append("unexpected_seed_roster")
    return blockers


def _is_hex_digest(value: object, *, length: int) -> bool:
    """Return whether ``value`` is a lowercase-or-uppercase hexadecimal digest."""
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _seed_keyed_observations(container: object) -> dict[int, float] | None:
    """Return finite observations keyed by their declared integer seed, or ``None``.

    Positional arrays are deliberately not accepted for promotable Gate 3 evidence: their
    alignment cannot be audited once a row is absent or reordered.
    """
    if not isinstance(container, Mapping):
        return None
    observations: dict[int, float] = {}
    for raw_seed, raw_value in container.items():
        if isinstance(raw_seed, bool):
            return None
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            return None
        if str(seed) != str(raw_seed):
            return None
        if seed in observations or isinstance(raw_value, bool):
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        observations[seed] = value
    return observations


def _radius_mapping_blockers(
    mapping: Mapping[object, object], radii: Sequence[float], label: str
) -> list[str]:
    """Return invalid, undeclared, or duplicate normalized radius keys for one payload."""
    blockers: list[str] = []
    seen_radii: set[float] = set()
    for raw_radius in mapping:
        try:
            radius = float(raw_radius)
        except (TypeError, ValueError):
            blockers.append(f"invalid_{label}_radius:{raw_radius}")
            continue
        if not math.isfinite(radius) or radius not in radii:
            blockers.append(f"undeclared_{label}_radius:{raw_radius}")
        elif radius in seen_radii:
            blockers.append(f"duplicate_{label}_radius:{_radius_key(radius)}")
        else:
            seen_radii.add(radius)
    return blockers


def _family_feasibility_blockers(
    sweep_summary: Mapping[str, object], radii: Sequence[float]
) -> list[str]:
    """Return fail-closed blockers for matched per-arm family feasibility evidence."""
    raw_families = sweep_summary.get("family_feasibility")
    if not isinstance(raw_families, Mapping):
        return ["missing_family_feasibility"]
    normalized = _float_keyed(raw_families)
    blockers = _radius_mapping_blockers(raw_families, radii, "family_feasibility")
    reference_families: set[str] | None = None
    for radius in radii:
        families = normalized.get(radius)
        if not isinstance(families, Mapping):
            blockers.append(f"radius_{_radius_key(radius)}_missing_family_feasibility")
            continue
        names = {name for name in families if isinstance(name, str) and name}
        if len(names) != len(families):
            blockers.append(f"radius_{_radius_key(radius)}_invalid_family_identity")
        if NARROW_DOORWAY_FAMILY not in names:
            blockers.append(f"radius_{_radius_key(radius)}_missing_narrow_doorway_feasibility")
        for family in names:
            if families[family] not in {FEASIBILITY_FEASIBLE, FEASIBILITY_INFEASIBLE}:
                blockers.append(f"radius_{_radius_key(radius)}_invalid_family_feasibility:{family}")
        if reference_families is None:
            reference_families = names
        elif names != reference_families:
            blockers.append(f"radius_{_radius_key(radius)}_family_feasibility_mismatch")
    return blockers


def _paired_observation_blockers(
    sweep_summary: Mapping[str, object],
    radii: Sequence[float],
    declared_planners: tuple[str, ...],
) -> list[str]:
    """Return blockers unless every arm/planner/metric has the declared seed keys."""
    raw_pairs = sweep_summary.get("paired_observations")
    if not isinstance(raw_pairs, Mapping):
        return ["missing_paired_observations"]
    paired_by_radius = _float_keyed(raw_pairs)
    expected_planners = set(declared_planners)
    expected_seeds = set(EXPECTED_SEED_ROSTER)
    blockers = _radius_mapping_blockers(raw_pairs, radii, "paired_observations")
    for radius in radii:
        arm_pairs = paired_by_radius.get(radius)
        if not isinstance(arm_pairs, Mapping):
            blockers.append(f"radius_{_radius_key(radius)}_missing_paired_observations")
            continue
        if set(arm_pairs) != expected_planners:
            blockers.append(f"radius_{_radius_key(radius)}_paired_planner_roster_mismatch")
        for planner in declared_planners:
            planner_pairs = arm_pairs.get(planner)
            if not isinstance(planner_pairs, Mapping):
                blockers.append(
                    f"radius_{_radius_key(radius)}_planner_{planner}_missing_paired_data"
                )
                continue
            if set(planner_pairs) != set(DEFAULT_RANK_METRICS):
                blockers.append(
                    f"radius_{_radius_key(radius)}_planner_{planner}_paired_metric_mismatch"
                )
            for metric in DEFAULT_RANK_METRICS:
                observations = _seed_keyed_observations(planner_pairs.get(metric))
                if observations is None or set(observations) != expected_seeds:
                    blockers.append(
                        f"radius_{_radius_key(radius)}_planner_{planner}_invalid_paired_seeds:{metric}"
                    )
    return blockers


def _campaign_provenance_blockers(
    sweep_summary: Mapping[str, object], radii: Sequence[float]
) -> tuple[list[str], tuple[str, str, str] | None]:
    """Validate the per-arm Gate 2 commit, config digest, and Gate 1 receipt binding.

    Returns:
        Fail-closed blockers and the shared immutable binding when all arms match.
    """
    raw_provenance = sweep_summary.get("campaign_provenance")
    if not isinstance(raw_provenance, Mapping):
        return ["missing_campaign_provenance"], None
    provenance_by_radius = _float_keyed(raw_provenance)
    blockers = _radius_mapping_blockers(raw_provenance, radii, "campaign_provenance")
    bindings: list[tuple[str, str, str]] = []
    for radius in radii:
        arm_provenance = provenance_by_radius.get(radius)
        if not isinstance(arm_provenance, Mapping):
            blockers.append(f"radius_{_radius_key(radius)}_missing_campaign_provenance")
            continue
        campaign_commit = arm_provenance.get("campaign_commit")
        config_sha256 = arm_provenance.get("config_sha256")
        canary_receipt_sha256 = arm_provenance.get("gate1_canary_receipt_sha256")
        if not _is_hex_digest(campaign_commit, length=40):
            blockers.append(f"radius_{_radius_key(radius)}_invalid_campaign_commit")
        if not _is_hex_digest(config_sha256, length=64):
            blockers.append(f"radius_{_radius_key(radius)}_invalid_config_sha256")
        if not _is_hex_digest(canary_receipt_sha256, length=64):
            blockers.append(f"radius_{_radius_key(radius)}_invalid_gate1_canary_receipt")
        if (
            _is_hex_digest(campaign_commit, length=40)
            and _is_hex_digest(config_sha256, length=64)
            and _is_hex_digest(canary_receipt_sha256, length=64)
        ):
            bindings.append((campaign_commit, config_sha256, canary_receipt_sha256))
    if bindings and len(bindings) != len(radii):
        blockers.append("incomplete_campaign_provenance")
    if len(set(bindings)) > 1:
        blockers.append("mixed_campaign_provenance")
    return blockers, bindings[0] if len(bindings) == len(radii) and len(
        set(bindings)
    ) == 1 else None


def _read_json_mapping(path: Path | None) -> Mapping[str, object] | None:
    """Read a JSON object, returning ``None`` for an unavailable or malformed file.

    Returns:
        The parsed JSON mapping, or ``None`` when the receipt cannot be read.
    """
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _gate1_receipt_header_is_passing(payload: Mapping[str, object]) -> bool:
    """Validate the Gate 1 report identity and top-level go flag.

    Returns:
        ``True`` only for the expected report schema, scope, and passing flag.
    """
    return (
        payload.get("schema") == GATE1_CANARY_REPORT_SCHEMA
        and payload.get("canary_schema") == GATE1_CANARY_VERDICT_SCHEMA
        and payload.get("issue") == GATE1_CANARY_ISSUE
        and payload.get("parent_issue") == GATE1_CANARY_PARENT_ISSUE
        and payload.get("go") is True
        and _gate1_receipt_radii(payload) == GATE1_CANARY_RADII
    )


def _gate1_receipt_radii(payload: Mapping[str, object]) -> tuple[float, ...]:
    """Parse the declared Gate 1 radii without allowing malformed input to escape.

    Returns:
        The parsed radii, or an empty tuple for malformed input.
    """
    raw_radii = payload.get("radii_m")
    if not isinstance(raw_radii, (list, tuple)):
        return ()
    try:
        return tuple(float(radius) for radius in raw_radii)
    except (TypeError, ValueError):
        return ()


def _gate1_surface_names_are_passing(surfaces: object) -> bool:
    """Validate that all five Gate 1 binding surfaces reported bound.

    Returns:
        ``True`` only when each required surface appears once and is bound.
    """
    if not isinstance(surfaces, list) or len(surfaces) != len(GATE1_CANARY_SURFACES):
        return False
    names: set[str] = set()
    for surface in surfaces:
        if not isinstance(surface, Mapping) or surface.get("bound") is not True:
            return False
        name = surface.get("surface")
        if not isinstance(name, str) or name in names:
            return False
        names.add(name)
    return names == GATE1_CANARY_SURFACES


def _gate1_verdict_is_passing(verdict: object, seen_radii: set[float]) -> bool:
    """Validate one per-radius Gate 1 verdict and add its radius to ``seen_radii``.

    Returns:
        ``True`` when the verdict is valid and passing.
    """
    if not isinstance(verdict, Mapping):
        return False
    if verdict.get("schema") != GATE1_CANARY_VERDICT_SCHEMA or verdict.get("go") is not True:
        return False
    try:
        radius = float(verdict.get("target_radius_m"))
    except (TypeError, ValueError):
        return False
    if radius not in GATE1_CANARY_RADII or radius in seen_radii:
        return False
    if not _gate1_surface_names_are_passing(verdict.get("surfaces")):
        return False
    seen_radii.add(radius)
    return True


def _gate1_canary_receipt_is_passing(path: Path | None) -> bool:
    """Return whether a Gate 1 receipt proves every required surface passed.

    The receipt digest binds the sweep to one artifact, but the digest alone does not
    prove that the artifact is a passing canary. This check validates the report schema,
    complete radius treatment, top-level go flag, and all five per-radius surfaces.
    """
    payload = _read_json_mapping(path)
    if payload is None or not _gate1_receipt_header_is_passing(payload):
        return False
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != len(GATE1_CANARY_RADII):
        return False
    seen_radii: set[float] = set()
    if not all(_gate1_verdict_is_passing(verdict, seen_radii) for verdict in verdicts):
        return False
    return seen_radii == set(GATE1_CANARY_RADII)


def _missing_radius_admission(radius: float) -> RadiusRowAdmission:
    """Return the admission record used when one radius lacks accounting."""
    return RadiusRowAdmission(
        radius=radius,
        declared=0,
        present=0,
        excluded_by_reason={},
        accounting_complete=False,
        excluded_total=0,
        disqualifying_reasons=(),
    )


def build_missingness_ledger(
    sweep_summary: Mapping[str, object],
    *,
    baseline_radius: float | None = None,
) -> MissingnessLedger:
    """Build the aggregate fail-closed missingness/degradation ledger.

    The ledger is ``complete`` only when every radius arm reconciles all declared rows
    and no arm carries a disqualifying exclusion. When incomplete, ``blocking_reasons``
    names each gap so the verdict can stop interpretation honestly.

    Returns:
        The aggregate missingness ledger across all declared radius arms.
    """
    radii = _normalized_radii(sweep_summary)
    accounting_by_radius = _float_keyed(sweep_summary.get("row_accounting"))
    blocking_reasons: list[str] = list(_radius_declaration_issues(sweep_summary))
    if tuple(radii) != EXPECTED_RADIUS_ARMS:
        blocking_reasons.append(
            "unexpected_radius_arms:" + ",".join(_radius_key(radius) for radius in radii)
        )
    if baseline_radius is not None and not math.isclose(baseline_radius, 1.0):
        blocking_reasons.append(f"unexpected_baseline_radius:{_radius_key(baseline_radius)}")

    declared_planners, roster_blockers = _declared_planner_roster(sweep_summary)
    blocking_reasons.extend(roster_blockers)
    blocking_reasons.extend(_fixed_scope_blockers(sweep_summary))
    _, table_blockers = _metric_table_blockers(sweep_summary, radii, declared_planners)
    blocking_reasons.extend(table_blockers)
    blocking_reasons.extend(_family_feasibility_blockers(sweep_summary, radii))
    blocking_reasons.extend(_paired_observation_blockers(sweep_summary, radii, declared_planners))
    provenance_blockers, _ = _campaign_provenance_blockers(sweep_summary, radii)
    blocking_reasons.extend(provenance_blockers)
    blocking_reasons.extend(_row_accounting_radius_blockers(sweep_summary, radii))

    per_radius: list[RadiusRowAdmission] = []
    excluded_by_reason: dict[str, int] = {}
    for radius in radii:
        accounting = accounting_by_radius.get(radius)
        if not isinstance(accounting, Mapping):
            per_radius.append(_missing_radius_admission(radius))
            blocking_reasons.append(f"radius_{_radius_key(radius)}_missing_row_accounting")
        else:
            admission = _admit_radius_rows(radius, accounting)
            per_radius.append(admission)
            if not admission.accounting_complete:
                blocking_reasons.append(f"radius_{_radius_key(radius)}_incomplete_accounting")
            for error in admission.accounting_errors:
                blocking_reasons.append(f"radius_{_radius_key(radius)}_accounting_error:{error}")
            for reason in admission.disqualifying_reasons:
                blocking_reasons.append(f"radius_{_radius_key(radius)}_excluded_{reason}")
            for reason, count in admission.excluded_by_reason.items():
                excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + count

    declared_total = sum(entry.declared for entry in per_radius)
    if declared_total != EXPECTED_TOTAL_ROWS:
        blocking_reasons.append(
            f"unexpected_declared_total:{declared_total}:expected_{EXPECTED_TOTAL_ROWS}"
        )
    complete = not blocking_reasons
    return MissingnessLedger(
        radii=tuple(radii),
        per_radius=tuple(per_radius),
        declared_total=declared_total,
        present_total=sum(entry.present for entry in per_radius),
        excluded_total=sum(entry.excluded_total for entry in per_radius),
        excluded_by_reason=excluded_by_reason,
        complete=complete,
        blocking_reasons=tuple(blocking_reasons),
    )


# --- ranking tables, correlation, and rank flips --------------------------


@dataclass(frozen=True)
class MetricRankStability:
    """Rank-stability result for one rank metric across the non-baseline radii."""

    metric: str
    higher_is_better: bool
    baseline_ranking: list[str]
    baseline_identifiable: bool
    baseline_identifiability_reason: str | None
    rankings_by_radius: dict[float, list[str]]
    kendall_tau_by_radius: dict[float, float | None]
    rank_flips_by_radius: dict[float, int | None]
    top1_changed_by_radius: dict[float, bool | None]
    identifiable: bool
    identifiability_reason: str | None
    flipping_radii: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation keyed by canonical radius strings.

        Returns:
            Mapping of metric, baseline order, per-radius ranks/tau/flips, and identifiability.
        """
        return {
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "baseline_ranking": list(self.baseline_ranking),
            "baseline_identifiable": self.baseline_identifiable,
            "baseline_identifiability_reason": self.baseline_identifiability_reason,
            "rankings_by_radius": {
                _radius_key(r): list(o) for r, o in self.rankings_by_radius.items()
            },
            "kendall_tau_by_radius": {
                _radius_key(r): v for r, v in self.kendall_tau_by_radius.items()
            },
            "rank_flips_by_radius": {
                _radius_key(r): v for r, v in self.rank_flips_by_radius.items()
            },
            "top1_changed_by_radius": {
                _radius_key(r): v for r, v in self.top1_changed_by_radius.items()
            },
            "identifiable": self.identifiable,
            "identifiability_reason": self.identifiability_reason,
            "flipping_radii": [_radius_key(r) for r in self.flipping_radii],
        }


def _metric_tables_by_radius(
    sweep_summary: Mapping[str, object],
) -> dict[float, Mapping[str, Mapping[str, object]]]:
    """Return planner metric tables keyed by numeric radius."""
    raw_tables = sweep_summary.get("metric_tables")
    tables: dict[float, Mapping[str, Mapping[str, object]]] = {}
    if isinstance(raw_tables, Mapping):
        for radius_key, table in raw_tables.items():
            try:
                radius = float(radius_key)
            except (TypeError, ValueError):
                continue
            if math.isfinite(radius) and isinstance(table, Mapping):
                tables[radius] = {
                    planner: row
                    for planner, row in table.items()
                    if isinstance(planner, str) and isinstance(row, Mapping)
                }
    return tables


def analyze_metric_rank_stability(
    sweep_summary: Mapping[str, object],
    metric: str,
    *,
    baseline_radius: float,
    higher_is_better: bool | None = None,
) -> MetricRankStability:
    """Analyze planner-ranking stability for one rank metric versus the baseline radius.

    Reports the baseline ranking, the per-radius ranking, Kendall tau, rank-flip count,
    and top-1 change for each non-baseline radius. When the baseline or a radius metric
    is non-identifiable, the rank-evidence fields for that radius are null and the
    non-identifiable reason is recorded.

    Returns:
        The per-metric rank-stability result versus the baseline radius.
    """
    if higher_is_better is None:
        higher_is_better = metric not in LOWER_IS_BETTER_METRICS
    tables = _metric_tables_by_radius(sweep_summary)
    expected_planners = sweep_summary.get("planners")
    expected_planners = (
        tuple(expected_planners)
        if isinstance(expected_planners, (list, tuple))
        and all(isinstance(planner, str) for planner in expected_planners)
        else None
    )
    baseline_table = tables.get(float(baseline_radius), {})
    baseline_ranking = rank_planners(baseline_table, metric, higher_is_better=higher_is_better)
    baseline_identifiable, baseline_reason = _metric_identifiable(
        baseline_table, metric, expected_planners=expected_planners
    )

    rankings: dict[float, list[str]] = {}
    taus: dict[float, float | None] = {}
    flips: dict[float, int | None] = {}
    top1: dict[float, bool | None] = {}
    flipping_radii: list[float] = []
    identifiable = baseline_identifiable
    identifiability_reason = baseline_reason
    for radius in sorted(tables):
        if radius == float(baseline_radius):
            continue
        table = tables[radius]
        ranking = rank_planners(table, metric, higher_is_better=higher_is_better)
        rankings[radius] = ranking
        radius_identifiable, radius_reason = _metric_identifiable(
            table, metric, expected_planners=expected_planners
        )
        pair_identifiable = baseline_identifiable and radius_identifiable
        if pair_identifiable and set(ranking) == set(baseline_ranking):
            tau: float | None = kendall_tau(baseline_ranking, ranking)
            flip: int | None = count_rank_flips(baseline_ranking, ranking)
            top1_changed: bool | None = bool(
                baseline_ranking and ranking and baseline_ranking[0] != ranking[0]
            )
        else:
            tau = None
            flip = None
            top1_changed = None
        taus[radius] = tau
        flips[radius] = flip
        top1[radius] = top1_changed
        if not radius_identifiable:
            identifiable = False
            if identifiability_reason is None:
                identifiability_reason = radius_reason
        if flip is not None and flip > 0:
            flipping_radii.append(radius)

    return MetricRankStability(
        metric=metric,
        higher_is_better=higher_is_better,
        baseline_ranking=baseline_ranking,
        baseline_identifiable=baseline_identifiable,
        baseline_identifiability_reason=baseline_reason,
        rankings_by_radius=rankings,
        kendall_tau_by_radius=taus,
        rank_flips_by_radius=flips,
        top1_changed_by_radius=top1,
        identifiable=identifiable,
        identifiability_reason=identifiability_reason,
        flipping_radii=tuple(flipping_radii),
    )


# --- per-planner paired changes with uncertainty --------------------------


@dataclass(frozen=True)
class PairedChange:
    """Per-planner paired metric change for one radius versus the baseline."""

    planner: str
    metric: str
    radius: float
    baseline_value: float | None
    radius_value: float | None
    delta: float | None
    ci_low: float | None
    ci_high: float | None
    n_pairs: int
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation.

        Returns:
            Mapping of planner, metric, radius, point delta, and confidence interval.
        """
        return {
            "planner": self.planner,
            "metric": self.metric,
            "radius": self.radius,
            "baseline_value": self.baseline_value,
            "radius_value": self.radius_value,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_pairs": self.n_pairs,
            "reason": self.reason,
        }


def _paired_bootstrap_ci(
    baseline_values: Sequence[float],
    radius_values: Sequence[float],
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    """Return (point delta, ci_low, ci_high, n_pairs) for paired per-seed observations.

    Uses a deterministic paired bootstrap (fixed seed) over the finite paired deltas so
    repeated invocations produce identical intervals.
    """
    pairs = [
        (float(base), float(current))
        for base, current in zip(baseline_values, radius_values, strict=False)
        if math.isfinite(float(base)) and math.isfinite(float(current))
    ]
    deltas = [current - base for base, current in pairs]
    point = sum(deltas) / len(deltas)
    if len(deltas) < 2 or n_resamples < 1:
        return point, point, point, len(deltas)
    rng = random.Random(seed)
    count = len(deltas)
    means = []
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(count)] for _ in range(count)]
        means.append(sum(sample) / count)
    means.sort()
    low_index = min(int((alpha / 2.0) * n_resamples), n_resamples - 1)
    high_index = min(int((1.0 - alpha / 2.0) * n_resamples), n_resamples) - 1
    return point, means[low_index], means[high_index], len(deltas)


def compute_paired_changes(
    sweep_summary: Mapping[str, object],
    metric: str,
    *,
    baseline_radius: float,
    radii: Iterable[float],
    n_resamples: int = 1000,
    seed: int = 123,
) -> dict[float, list[PairedChange]]:
    """Compute per-planner paired metric changes versus the baseline radius.

    Seed-keyed paired observations produce a deterministic paired-bootstrap confidence
    interval. Malformed or missing pair data is rendered diagnostically here, while the
    top-level missingness ledger prevents it from promoting an interpretation.

    Returns:
        Mapping of radius to the list of per-planner paired changes versus baseline.
    """
    tables = _metric_tables_by_radius(sweep_summary)
    baseline_table = tables.get(float(baseline_radius), {})
    paired_by_radius = _float_keyed(sweep_summary.get("paired_observations"))
    baseline_pairs = paired_by_radius.get(float(baseline_radius))
    baseline_pairs = baseline_pairs if isinstance(baseline_pairs, Mapping) else {}

    changes: dict[float, list[PairedChange]] = {}
    for radius in radii:
        if float(radius) == float(baseline_radius):
            continue
        table = tables.get(float(radius), {})
        radius_pairs = paired_by_radius.get(float(radius))
        radius_pairs = radius_pairs if isinstance(radius_pairs, Mapping) else {}
        radius_changes: list[PairedChange] = []
        for planner in sorted(set(baseline_table) | set(table)):
            base_value = _finite_metric_value(baseline_table.get(planner, {}), metric)
            radius_value = _finite_metric_value(table.get(planner, {}), metric)
            base_planner_pairs = baseline_pairs.get(planner)
            base_planner_pairs = (
                base_planner_pairs if isinstance(base_planner_pairs, Mapping) else {}
            )
            radius_planner_pairs = radius_pairs.get(planner)
            radius_planner_pairs = (
                radius_planner_pairs if isinstance(radius_planner_pairs, Mapping) else {}
            )
            base_obs = _seed_keyed_observations(base_planner_pairs.get(metric))
            radius_obs = _seed_keyed_observations(radius_planner_pairs.get(metric))
            paired_base: list[float] = []
            paired_radius: list[float] = []
            if base_obs and radius_obs:
                for paired_seed in sorted(set(base_obs) & set(radius_obs)):
                    paired_base.append(base_obs[paired_seed])
                    paired_radius.append(radius_obs[paired_seed])
            if len(paired_base) >= 2:
                point, low, high, n_pairs = _paired_bootstrap_ci(
                    paired_base,
                    paired_radius,
                    n_resamples=n_resamples,
                    seed=seed,
                )
                radius_changes.append(
                    PairedChange(
                        planner=planner,
                        metric=metric,
                        radius=float(radius),
                        baseline_value=base_value,
                        radius_value=radius_value,
                        delta=point,
                        ci_low=low,
                        ci_high=high,
                        n_pairs=n_pairs,
                        reason=None,
                    )
                )
            else:
                delta = (
                    radius_value - base_value
                    if base_value is not None and radius_value is not None
                    else None
                )
                radius_changes.append(
                    PairedChange(
                        planner=planner,
                        metric=metric,
                        radius=float(radius),
                        baseline_value=base_value,
                        radius_value=radius_value,
                        delta=delta,
                        ci_low=None,
                        ci_high=None,
                        n_pairs=0,
                        reason=(
                            "insufficient_paired_observations"
                            if base_obs or radius_obs
                            else "no_paired_observations"
                        ),
                    )
                )
        changes[float(radius)] = radius_changes
    return changes


def _finite_sequence(container: object, metric: str) -> list[float]:
    """Return finite per-seed observations for ``metric`` from a paired-observations cell."""
    if not isinstance(container, Mapping):
        return []
    raw = container.get(metric)
    if not isinstance(raw, (list, tuple)):
        return []
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


# --- scenario-family and feasibility transitions --------------------------


@dataclass(frozen=True)
class FamilyTransition:
    """Feasibility transition for one scenario family across the tested radii."""

    family: str
    status_by_radius: dict[float, str | None]
    changed_vs_baseline: dict[float, bool]
    is_narrow_doorway: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation keyed by canonical radius strings.

        Returns:
            Mapping of family, per-radius feasibility status, and baseline transitions.
        """
        return {
            "family": self.family,
            "is_narrow_doorway": self.is_narrow_doorway,
            "status_by_radius": {_radius_key(r): s for r, s in self.status_by_radius.items()},
            "changed_vs_baseline": {_radius_key(r): c for r, c in self.changed_vs_baseline.items()},
        }


def compute_family_transitions(
    sweep_summary: Mapping[str, object],
    *,
    baseline_radius: float,
    radii: Iterable[float],
) -> list[FamilyTransition]:
    """Compute scenario-family feasibility transitions versus the baseline radius.

    The narrow-doorway family is flagged explicitly because the parent issue identifies
    its 2.0 m doorway gap as geometry-coupled to the 1.0 m collision-envelope diameter.

    Returns:
        The list of per-family feasibility transitions across the tested radii.
    """
    families_by_radius = _float_keyed(sweep_summary.get("family_feasibility"))
    family_names: set[str] = set()
    status_by_radius: dict[float, Mapping[str, object]] = {}
    for radius in radii:
        families = families_by_radius.get(float(radius))
        families = families if isinstance(families, Mapping) else {}
        status_by_radius[float(radius)] = families
        family_names.update(str(name) for name in families)

    baseline_status = status_by_radius.get(float(baseline_radius), {})
    transitions: list[FamilyTransition] = []
    for family in sorted(family_names):
        per_radius_status: dict[float, str | None] = {}
        changed: dict[float, bool] = {}
        base_value = baseline_status.get(family)
        for radius in radii:
            status = status_by_radius.get(float(radius), {}).get(family)
            status_str = str(status) if status is not None else None
            per_radius_status[float(radius)] = status_str
            if float(radius) != float(baseline_radius):
                changed[float(radius)] = status_str != (
                    str(base_value) if base_value is not None else None
                )
        transitions.append(
            FamilyTransition(
                family=family,
                status_by_radius=per_radius_status,
                changed_vs_baseline=changed,
                is_narrow_doorway=family == NARROW_DOORWAY_FAMILY,
            )
        )
    return transitions


# --- verdict decision ------------------------------------------------------


@dataclass(frozen=True)
class VerdictDecision:
    """The preregistered radius-sensitivity verdict and its decision audit trail."""

    verdict: str
    is_scientific_verdict: bool
    analysis_status: str
    reasons: tuple[str, ...]
    interpretation_promoted: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation.

        Returns:
            Mapping of verdict, scientific-verdict flag, status, reasons, and promotion.
        """
        return {
            "verdict": self.verdict,
            "is_scientific_verdict": self.is_scientific_verdict,
            "analysis_status": self.analysis_status,
            "reasons": list(self.reasons),
            "interpretation_promoted": self.interpretation_promoted,
        }


def decide_radius_verdict(
    *,
    sweep_available: bool,
    missingness: MissingnessLedger | None,
    metric_stability: Sequence[MetricRankStability],
) -> VerdictDecision:
    """Apply the fail-closed gate precedence to select exactly one verdict.

    Precedence: blocked (no sweep) > invalid (incomplete accounting / excluded rows) >
    non_identifiable > radius_dependent (any flip) > stable_within_tested_radii.
    Interpretation is promoted only for the identifiable, complete-evidence verdicts.

    Returns:
        The selected verdict decision with its audit trail of reasons.
    """
    if not sweep_available:
        return VerdictDecision(
            verdict=ANALYSIS_BLOCKED_PENDING_GATE2,
            is_scientific_verdict=False,
            analysis_status=ANALYSIS_BLOCKED_PENDING_GATE2,
            reasons=("gate_2_sweep_summary_unavailable",),
            interpretation_promoted=False,
        )

    if missingness is None or not missingness.complete:
        return VerdictDecision(
            verdict=VERDICT_INVALID,
            is_scientific_verdict=True,
            analysis_status="fail_closed_invalid_evidence",
            reasons=(
                tuple(missingness.blocking_reasons)
                if missingness is not None and missingness.blocking_reasons
                else ("missingness_ledger_unavailable",)
            ),
            interpretation_promoted=False,
        )

    if not metric_stability:
        return VerdictDecision(
            verdict=VERDICT_INVALID,
            is_scientific_verdict=True,
            analysis_status="fail_closed_missing_rank_metrics",
            reasons=("no_rank_metrics_analyzed",),
            interpretation_promoted=False,
        )

    non_identifiable_metrics = tuple(
        stability.metric for stability in metric_stability if not stability.identifiable
    )
    if non_identifiable_metrics:
        reasons = tuple(f"non_identifiable_metric:{metric}" for metric in non_identifiable_metrics)
        return VerdictDecision(
            verdict=VERDICT_NON_IDENTIFIABLE,
            is_scientific_verdict=True,
            analysis_status="rank_non_identifiable",
            reasons=reasons,
            interpretation_promoted=False,
        )

    flipping: list[str] = []
    for stability in metric_stability:
        for radius in stability.flipping_radii:
            flipping.append(f"{stability.metric}@{_radius_key(radius)}")
    if flipping:
        return VerdictDecision(
            verdict=VERDICT_RADIUS_DEPENDENT,
            is_scientific_verdict=True,
            analysis_status="ranking_flip_boundary",
            reasons=tuple(f"rank_flip:{entry}" for entry in flipping),
            interpretation_promoted=True,
        )

    return VerdictDecision(
        verdict=VERDICT_STABLE,
        is_scientific_verdict=True,
        analysis_status="rank_stable_within_tested_radii",
        reasons=("no_rank_flips_across_tested_radii",),
        interpretation_promoted=True,
    )


# --- top-level report ------------------------------------------------------


@dataclass(frozen=True)
class RadiusSensitivityReport:
    """Complete Gate 3 radius rank-stability report."""

    baseline_radius: float
    radii: tuple[float, ...]
    rank_metrics: tuple[str, ...]
    planners: tuple[str, ...]
    scenario_cell_count: int
    seed_roster: tuple[int, ...]
    sweep_available: bool
    missingness: MissingnessLedger | None
    metric_stability: tuple[MetricRankStability, ...]
    paired_changes: dict[str, dict[float, list[PairedChange]]]
    family_transitions: tuple[FamilyTransition, ...]
    verdict: VerdictDecision

    def to_dict(self) -> dict[str, object]:
        """Return the ``radius_rank_stability.v2`` JSON-safe payload.

        Returns:
            Mapping with schema, configuration, missingness, per-metric stability,
            paired changes, family transitions, verdict, and claim boundary.
        """
        return {
            "schema_version": RADIUS_RANK_STABILITY_SCHEMA,
            "baseline_radius_m": self.baseline_radius,
            "radii_m": list(self.radii),
            "rank_metrics": list(self.rank_metrics),
            "planners": list(self.planners),
            "scenario_cell_count": self.scenario_cell_count,
            "seed_roster": list(self.seed_roster),
            "sweep_available": self.sweep_available,
            "missingness": self.missingness.to_dict() if self.missingness is not None else None,
            "metric_stability": [entry.to_dict() for entry in self.metric_stability],
            "paired_changes": {
                metric: {
                    _radius_key(radius): [change.to_dict() for change in changes]
                    for radius, changes in by_radius.items()
                }
                for metric, by_radius in self.paired_changes.items()
            },
            "family_transitions": [entry.to_dict() for entry in self.family_transitions],
            "verdict": self.verdict.to_dict(),
            "claim_boundary": list(REQUIRED_CLAIM_BOUNDARY_PHRASES),
        }


def _normalized_radii(sweep_summary: Mapping[str, object]) -> list[float]:
    """Return the declared radii as sorted floats."""
    raw_radii = sweep_summary.get("radii_m")
    if not isinstance(raw_radii, (list, tuple)):
        return []
    radii: list[float] = []
    for item in raw_radii:
        try:
            radius = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(radius) and radius > 0.0:
            radii.append(radius)
    return sorted(radii)


def _radius_declaration_issues(sweep_summary: Mapping[str, object]) -> tuple[str, ...]:
    """Return structural errors in the declared radius list."""
    raw_radii = sweep_summary.get("radii_m")
    if not isinstance(raw_radii, (list, tuple)):
        return ("missing_or_invalid_radii_m",)
    issues: list[str] = []
    parsed: list[float] = []
    for index, item in enumerate(raw_radii):
        try:
            radius = float(item)
        except (TypeError, ValueError):
            issues.append(f"invalid_radius_at_index:{index}")
            continue
        if not math.isfinite(radius) or radius <= 0.0:
            issues.append(f"invalid_radius_at_index:{index}")
            continue
        parsed.append(radius)
    if len(set(parsed)) != len(parsed):
        issues.append("duplicate_radius_arm")
    return tuple(issues)


def _int_sequence(container: object) -> tuple[int, ...]:
    """Return a tuple of ints from a JSON list, skipping non-integer entries."""
    if not isinstance(container, (list, tuple)):
        return ()
    values: list[int] = []
    for item in container:
        if isinstance(item, bool):
            continue
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(values)


def analyze_radius_sensitivity(
    sweep_summary: Mapping[str, object] | None,
    *,
    baseline_radius: float = 1.0,
    rank_metrics: Iterable[str] | None = None,
    n_resamples: int = 1000,
    seed: int = 123,
) -> RadiusSensitivityReport:
    """Run the full Gate 3 radius rank-stability analysis on a Gate 2 sweep summary.

    When ``sweep_summary`` is ``None`` or empty, the report carries the
    ``blocked_pending_gate2`` pre-analysis status and no scientific verdict. Otherwise
    the fail-closed missingness ledger, per-metric rank stability, paired changes, and
    family transitions feed the verdict decision.

    Returns:
        The complete radius rank-stability report including the verdict decision.
    """
    metrics = tuple(rank_metrics) if rank_metrics is not None else DEFAULT_RANK_METRICS
    summary: Mapping[str, object] = sweep_summary if isinstance(sweep_summary, Mapping) else {}
    if sweep_summary is not None and not isinstance(sweep_summary, Mapping):
        raise ValueError("sweep summary must be a mapping")
    if summary and summary.get("schema_version") != SWEEP_SUMMARY_SCHEMA:
        raise ValueError(f"sweep summary schema_version must be {SWEEP_SUMMARY_SCHEMA}")
    # ``None`` means that the operator has not supplied a Gate 2 artifact. An existing
    # mapping, including an empty or incomplete JSON object, is an attempted analysis
    # input and must reach the invalid-evidence path rather than masquerading as pending.
    sweep_available = sweep_summary is not None

    planners = tuple(str(name) for name in summary.get("planners", []) if isinstance(name, str))
    cells = summary.get("scenario_cells")
    scenario_cell_count = len(cells) if isinstance(cells, (list, tuple)) else 0
    seed_roster = _int_sequence(summary.get("seeds"))
    radii = tuple(_normalized_radii(summary)) if sweep_available else ()

    missingness: MissingnessLedger | None = None
    metric_stability: list[MetricRankStability] = []
    paired_changes: dict[str, dict[float, list[PairedChange]]] = {}
    family_transitions: list[FamilyTransition] = []

    if sweep_available:
        missingness = build_missingness_ledger(summary, baseline_radius=baseline_radius)
        for metric in metrics:
            metric_stability.append(
                analyze_metric_rank_stability(summary, metric, baseline_radius=baseline_radius)
            )
        # Paired changes and family transitions are only meaningful on complete evidence;
        # compute them always but the verdict gates whether interpretation is promoted.
        for metric in metrics:
            paired_changes[metric] = compute_paired_changes(
                summary,
                metric,
                baseline_radius=baseline_radius,
                radii=radii,
                n_resamples=n_resamples,
                seed=seed,
            )
        family_transitions = compute_family_transitions(
            summary, baseline_radius=baseline_radius, radii=radii
        )

    verdict = decide_radius_verdict(
        sweep_available=sweep_available,
        missingness=missingness,
        metric_stability=metric_stability,
    )

    return RadiusSensitivityReport(
        baseline_radius=float(baseline_radius),
        radii=radii,
        rank_metrics=metrics,
        planners=planners,
        scenario_cell_count=scenario_cell_count,
        seed_roster=seed_roster,
        sweep_available=sweep_available,
        missingness=missingness,
        metric_stability=tuple(metric_stability),
        paired_changes=paired_changes,
        family_transitions=tuple(family_transitions),
        verdict=verdict,
    )


# --- durable evidence bundle ----------------------------------------------


def current_git_sha() -> str:
    """Return the current Git commit SHA, or ``unknown`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_repository_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def evidence_tier_for_verdict(verdict: str) -> str:
    """Return the honest evidence tier for a verdict.

    Only a complete, identifiable verdict on valid native rows is nominal benchmark
    evidence for radius sensitivity. The blocked, invalid, and non-identifiable outcomes
    are diagnostic-only by construction.
    """
    if verdict in {VERDICT_STABLE, VERDICT_RADIUS_DEPENDENT}:
        return "nominal_benchmark_radius_sensitivity"
    return "diagnostic-only"


@dataclass(frozen=True)
class EvidenceProvenance:
    """Immutable provenance for a Gate 3 durable evidence bundle."""

    config_path: str
    config_sha256: str | None
    command: str
    campaign_commit: str | None
    analysis_commit: str
    seed_roster: tuple[int, ...]
    radii_m: tuple[float, ...]
    planners: tuple[str, ...]
    scenario_cell_count: int
    input_sha256: dict[str, str] = field(default_factory=dict)
    gate1_canary_receipt_sha256: str | None = None
    gate1_canary_receipt_verified: bool = False
    campaign_provenance_verified: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe representation.

        Returns:
            Mapping of immutable config, command, commits, roster, and input checksums.
        """
        return {
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "command": self.command,
            "campaign_commit": self.campaign_commit,
            "analysis_commit": self.analysis_commit,
            "seed_roster": list(self.seed_roster),
            "radii_m": list(self.radii_m),
            "planners": list(self.planners),
            "scenario_cell_count": self.scenario_cell_count,
            "input_sha256": dict(self.input_sha256),
            "gate1_canary_receipt_sha256": self.gate1_canary_receipt_sha256,
            "gate1_canary_receipt_verified": self.gate1_canary_receipt_verified,
            "campaign_provenance_verified": self.campaign_provenance_verified,
        }


def build_evidence_provenance(
    report: RadiusSensitivityReport,
    *,
    config_path: str,
    command: str,
    campaign_commit: str | None,
    analysis_commit: str | None = None,
    config_sha256: str | None = None,
    input_paths: Mapping[str, Path] | None = None,
    sweep_summary: Mapping[str, object] | None = None,
) -> EvidenceProvenance:
    """Assemble immutable provenance, checksumming any supplied input artifacts.

    Blocked Gate 3 reports do not have a campaign commit. Normalize that field here so
    programmatic bundle writers preserve the same fail-closed boundary as the CLI.

    Returns:
        The immutable provenance record for the durable evidence bundle.
    """
    if report.verdict.verdict == ANALYSIS_BLOCKED_PENDING_GATE2:
        campaign_commit = None
    if config_sha256 is None:
        config_file = Path(config_path)
        if config_file.is_file():
            config_sha256 = sha256_file(config_file)
    input_sha256: dict[str, str] = {}
    if input_paths:
        for label, path in input_paths.items():
            resolved = Path(path)
            if resolved.is_file():
                input_sha256[label] = sha256_file(resolved)
    gate1_receipt_path = input_paths.get("gate1_canary_receipt.json") if input_paths else None
    gate1_receipt_verified = _gate1_canary_receipt_is_passing(gate1_receipt_path)
    summary = sweep_summary if isinstance(sweep_summary, Mapping) else {}
    provenance_blockers, source_binding = _campaign_provenance_blockers(
        summary, _normalized_radii(summary)
    )
    source_campaign_commit = source_config_sha256 = source_canary_receipt_sha256 = None
    if source_binding is not None and not provenance_blockers:
        source_campaign_commit, source_config_sha256, source_canary_receipt_sha256 = source_binding
    provenance_verified = bool(
        source_binding
        and not provenance_blockers
        and campaign_commit == source_campaign_commit
        and config_sha256 == source_config_sha256
        and input_sha256.get("gate1_canary_receipt.json") == source_canary_receipt_sha256
        and gate1_receipt_verified
    )
    return EvidenceProvenance(
        config_path=config_path,
        config_sha256=config_sha256,
        command=command,
        campaign_commit=campaign_commit,
        analysis_commit=analysis_commit or current_git_sha(),
        seed_roster=report.seed_roster,
        radii_m=report.radii,
        planners=report.planners,
        scenario_cell_count=report.scenario_cell_count,
        input_sha256=input_sha256,
        gate1_canary_receipt_sha256=source_canary_receipt_sha256,
        gate1_canary_receipt_verified=gate1_receipt_verified,
        campaign_provenance_verified=provenance_verified,
    )


def build_analysis_provenance_payload(
    report: RadiusSensitivityReport,
    provenance: EvidenceProvenance,
    *,
    output_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return the ``analysis_provenance.json`` payload for the durable bundle.

    Returns:
        Mapping with schema, evidence tier, claim boundary, verdict, provenance, and hashes.
    """
    return {
        "schema_version": RADIUS_EVIDENCE_BUNDLE_SCHEMA,
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "evidence_status": evidence_tier_for_verdict(report.verdict.verdict),
        "claim_boundary": " ".join(
            [
                "Within-simulator radius sensitivity only:",
                "not physical-footprint validation,",
                "not simulator-realism evidence,",
                "not sim-to-real evidence,",
                "not a safety guarantee.",
            ]
        ),
        "verdict": report.verdict.to_dict(),
        "provenance": provenance.to_dict(),
        "output_sha256": dict(output_sha256 or {}),
    }


# --- markdown rendering and verdict propagation ---------------------------


def _verdict_headline(report: RadiusSensitivityReport) -> str:
    """Return a one-line verdict headline for markdown surfaces."""
    verdict = report.verdict
    if verdict.verdict == ANALYSIS_BLOCKED_PENDING_GATE2:
        return "analysis blocked: Gate 2 production sweep summary unavailable"
    return f"verdict: `{verdict.verdict}`"


def _render_missingness_section(missingness: MissingnessLedger | None) -> list[str]:
    """Return the missingness/degradation ledger markdown section lines."""
    if missingness is None:
        return []
    lines = [
        "## Missingness / degradation ledger",
        "",
        f"- Accounting complete: {missingness.complete}",
        f"- Declared rows: {missingness.declared_total}",
        f"- Present rows: {missingness.present_total}",
        f"- Excluded rows: {missingness.excluded_total}",
    ]
    for reason in sorted(missingness.excluded_by_reason):
        lines.append(f"  - {reason}: {missingness.excluded_by_reason[reason]}")
    if missingness.blocking_reasons:
        lines.append(f"- Blocking reasons: {', '.join(missingness.blocking_reasons)}")
    lines.append("")
    return lines


def _render_metric_stability_section(metric_stability: Sequence[MetricRankStability]) -> list[str]:
    """Return the per-metric rank-stability markdown section lines."""
    if not metric_stability:
        return []
    lines = ["## Planner-ranking stability versus baseline", ""]
    for stability in metric_stability:
        direction = "higher is better" if stability.higher_is_better else "lower is better"
        lines.append(f"### {stability.metric} ({direction})")
        lines.append("")
        lines.append(f"- Identifiable: {stability.identifiable}")
        if stability.identifiability_reason:
            lines.append(f"- Non-identifiable reason: `{stability.identifiability_reason}`")
        lines.append(f"- Baseline ranking: {' > '.join(stability.baseline_ranking) or 'none'}")
        for radius in sorted(stability.kendall_tau_by_radius):
            tau = stability.kendall_tau_by_radius[radius]
            flips = stability.rank_flips_by_radius.get(radius)
            top1 = stability.top1_changed_by_radius.get(radius)
            tau_text = f"{tau:.3f}" if tau is not None else "null"
            lines.append(
                f"- radius {radius:g} m: kendall_tau={tau_text}, "
                f"rank_flips={flips}, top1_changed={top1}"
            )
        lines.append("")
    return lines


def _render_family_section(family_transitions: Sequence[FamilyTransition]) -> list[str]:
    """Return the scenario-family feasibility-transition markdown section lines."""
    if not family_transitions:
        return []
    lines = ["## Scenario-family feasibility transitions", ""]
    for transition in family_transitions:
        flag = " (narrow-doorway family)" if transition.is_narrow_doorway else ""
        lines.append(f"### {transition.family}{flag}")
        lines.append("")
        for radius in sorted(transition.status_by_radius):
            status = transition.status_by_radius[radius]
            changed = transition.changed_vs_baseline.get(radius)
            changed_text = "" if changed is None else f", changed_vs_baseline={changed}"
            lines.append(f"- radius {radius:g} m: {status}{changed_text}")
        lines.append("")
    return lines


def render_report_markdown(report: RadiusSensitivityReport) -> str:
    """Render the human-readable ``report.md`` for the durable bundle.

    Returns:
        The rendered markdown text for the bundle report.
    """
    lines: list[str] = [
        "<!-- AI-GENERATED: radius rank-stability analysis; NEEDS-REVIEW before reuse. -->",
        "# Radius Rank-Stability Analysis (issue #6643, Gate 3 of #6600)",
        "",
        f"**{_verdict_headline(report)}**",
        "",
        "## Claim boundary",
        "",
        "Within-simulator radius sensitivity only. This is not physical-footprint validation,",
        "not simulator-realism evidence, not sim-to-real evidence, and not a safety guarantee.",
        "",
        "## Configuration",
        "",
        f"- Baseline radius: {report.baseline_radius:g} m",
        f"- Tested radii (m): {', '.join(f'{r:g}' for r in report.radii) or 'none'}",
        f"- Rank metrics: {', '.join(report.rank_metrics)}",
        f"- Declared planners: {len(report.planners)}",
        f"- Declared scenario cells: {report.scenario_cell_count}",
        f"- Seed roster: {len(report.seed_roster)} seeds",
        "",
        "## Verdict decision",
        "",
        f"- Verdict: `{report.verdict.verdict}`",
        f"- Analysis status: `{report.verdict.analysis_status}`",
        f"- Interpretation promoted: {report.verdict.interpretation_promoted}",
        f"- Reasons: {', '.join(report.verdict.reasons) or 'none'}",
        "",
    ]
    lines += _render_missingness_section(report.missingness)
    lines += _render_metric_stability_section(report.metric_stability)
    lines += _render_family_section(report.family_transitions)
    return "\n".join(lines).rstrip() + "\n"


def render_claim_decision(report: RadiusSensitivityReport) -> str:
    """Render the ``claim_decision.md`` claim card for the durable bundle.

    Returns:
        The rendered claim-card markdown text.
    """
    verdict = report.verdict
    promoted = verdict.interpretation_promoted
    lines: list[str] = [
        "<!-- AI-GENERATED: radius rank-stability claim card; NEEDS-REVIEW before reuse. -->",
        "# Radius Rank-Stability Claim Decision",
        "",
        "## Decision",
        "",
        f"- **Verdict:** `{verdict.verdict}`",
        f"- **Evidence tier:** `{evidence_tier_for_verdict(verdict.verdict)}`",
        f"- **Interpretation promoted:** {promoted}",
        "",
        "## Claim card",
        "",
        "### Supported for review",
        "",
    ]
    if promoted:
        lines += [
            "- The radius rank-stability verdict above is reproducible from the checksum-covered",
            "  bundle inputs for the tested radii, planners, scenario cells, seeds, and metric",
            "  contract.",
            "- A ranking flip, when present, is reported as a valid boundary result.",
        ]
    else:
        lines += [
            "- No radius-sensitivity ranking claim is promoted: the analysis is blocked or the",
            "  evidence is incomplete, non-identifiable, or invalid under the fail-closed gate.",
        ]
    lines += [
        "",
        "### Intentionally not supported",
        "",
        "- No physical-footprint validation, simulator-realism, sim-to-real, or safety claim.",
        "- No manuscript claim: manuscript admission is a separate author step and is not",
        "  triggered by this bundle or by issue closure.",
        "",
        "## Claim boundary",
        "",
        "Within-simulator radius sensitivity only. Not physical-footprint validation, not",
        "simulator-realism evidence, not sim-to-real evidence, not a safety guarantee.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_readme(report: RadiusSensitivityReport, provenance: EvidenceProvenance) -> str:
    """Render the bundle ``README.md`` with immutable reproduction instructions.

    Returns:
        The rendered bundle README markdown text.
    """
    lines: list[str] = [
        "<!-- AI-GENERATED: radius rank-stability evidence bundle; NEEDS-REVIEW before reuse. -->",
        "# Radius Rank-Stability Evidence Bundle (issue #6643)",
        "",
        f"**{_verdict_headline(report)}**",
        "",
        "## Claim boundary",
        "",
        f"**Evidence tier: {evidence_tier_for_verdict(report.verdict.verdict)}.** ",
        "Within-simulator radius sensitivity only. Not physical-footprint validation, not",
        "simulator-realism evidence, not sim-to-real evidence, not a safety guarantee.",
        "",
        "## Provenance",
        "",
        f"- Campaign commit: `{provenance.campaign_commit}`"
        if provenance.campaign_commit
        else "- Campaign commit: `not available (Gate 2 pending)`",
        f"- Analysis commit: `{provenance.analysis_commit}`",
        f"- Config: `{provenance.config_path}`",
    ]
    if provenance.config_sha256:
        lines.append(f"- Config SHA-256: `{provenance.config_sha256}`")
    lines += [
        f"- Radii (m): {', '.join(f'{r:g}' for r in provenance.radii_m)}",
        f"- Baseline radius (m): {report.baseline_radius:g}",
        f"- Planners: {len(provenance.planners)}",
        f"- Scenario cells: {provenance.scenario_cell_count}",
        f"- Seeds: {len(provenance.seed_roster)}",
        "",
        "## Reproduction",
        "",
        "```bash",
        provenance.command,
        "```",
        "",
        "## Artifact policy",
        "",
        "- `result.json` is the canonical machine-readable radius rank-stability result.",
        "- `analysis_provenance.json` records provenance, checksums, and the verdict.",
        "- `report.md` is the generated human-readable rendering of the same result.",
        "- `claim_decision.md` records the claim boundary without promoting a manuscript claim.",
        "- Raw Gate 2 episode rows remain ignored local artifacts and are not copied into Git.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_verdict_comment(report: RadiusSensitivityReport) -> str:
    """Render the verdict comment body for the parent campaign issue (#6600).

    Returns:
        The rendered markdown comment body for the parent campaign issue.
    """
    verdict = report.verdict
    lines: list[str] = [
        "## Gate 3 radius rank-stability verdict (issue #6643)",
        "",
        f"**Verdict:** `{verdict.verdict}`",
        f"**Evidence tier:** `{evidence_tier_for_verdict(verdict.verdict)}`",
        f"**Analysis status:** `{verdict.analysis_status}`",
        "",
        "Claim boundary: within-simulator radius sensitivity only. Not physical-footprint",
        "validation, not simulator-realism evidence, not sim-to-real evidence. It is",
        "not a safety guarantee.",
        "",
        f"Reasons: {', '.join(verdict.reasons) or 'none'}.",
        "",
    ]
    if verdict.verdict == ANALYSIS_BLOCKED_PENDING_GATE2:
        lines += [
            "This is a pre-analysis gate status, not a scientific verdict: the Gate 2 production",
            "sweep (#6642) has not yielded complete row identities or a fail-closed missingness",
            "ledger yet. The analysis tooling is in place and fails closed; rerun once Gate 2",
            "lands. No ranking interpretation is promoted.",
            "",
        ]
    lines += [
        "Manuscript admission is a separate author step and is not triggered by this verdict",
        "or by issue closure.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_propagation_comment(report: RadiusSensitivityReport) -> str:
    """Render the propagation comment body for the parent validity study (#3207).

    Returns:
        The rendered markdown comment body for the parent validity study.
    """
    verdict = report.verdict
    if verdict.verdict == ANALYSIS_BLOCKED_PENDING_GATE2:
        return "\n".join(
            [
                "## Radius-sensitivity analysis status (issue #6643)",
                "",
                "No radius-axis result is available for propagation to #3207.",
                "The Gate 2 production sweep (#6642) has not supplied a summary, so this is",
                "a pre-analysis gate status, not a scientific verdict. No validity-boundary",
                "decision or ranking interpretation is promoted.",
                "",
                "Claim boundary: within-simulator radius sensitivity only. Not physical-footprint",
                "validation, not simulator-realism evidence, not sim-to-real evidence, not a",
                "safety guarantee.",
                "",
            ]
        )
    lines: list[str] = [
        "## Radius-sensitivity validity-boundary propagation (from #6600 Gate 3, issue #6643)",
        "",
        f"The collision-envelope radius axis (#6600) recorded verdict `{verdict.verdict}` ",
        f"(evidence tier `{evidence_tier_for_verdict(verdict.verdict)}`).",
        "",
        "Claim boundary: within-simulator radius sensitivity only. Not physical-footprint",
        "validation, not simulator-realism evidence, not sim-to-real evidence.",
        "It is not a safety guarantee. This propagates the radius-axis result into the",
        "parent validity-boundary decision; it does not promote a manuscript claim.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_evidence_bundle(
    report: RadiusSensitivityReport,
    provenance: EvidenceProvenance,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the durable evidence bundle and return the written file paths.

    Writes ``result.json``, ``report.md``, and ``claim_decision.md`` first, checksums them,
    then writes ``analysis_provenance.json`` (carrying those output hashes) and ``README.md``.

    Returns:
        Mapping of bundle file name to the written absolute path.
    """
    if report.verdict.interpretation_promoted:
        if not provenance.config_sha256:
            raise ValueError("promoted evidence requires a checksum-covered config")
        if "sweep_summary.json" not in provenance.input_sha256:
            raise ValueError("promoted evidence requires a checksum-covered sweep summary")
        if "gate1_canary_receipt.json" not in provenance.input_sha256:
            raise ValueError("promoted evidence requires a checksum-covered Gate 1 canary receipt")
        if not provenance.gate1_canary_receipt_verified:
            raise ValueError("promoted evidence requires a passing Gate 1 canary receipt")
        if not provenance.campaign_provenance_verified:
            raise ValueError(
                "promoted evidence requires verified matching Gate 2 campaign/config/canary provenance"
            )
        if not _is_hex_digest(provenance.gate1_canary_receipt_sha256, length=64):
            raise ValueError("promoted evidence requires a Gate 1 canary receipt digest")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result_path = out / "result.json"
    result_payload = report.to_dict()
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_path = out / "report.md"
    report_path.write_text(render_report_markdown(report), encoding="utf-8")

    claim_path = out / "claim_decision.md"
    claim_path.write_text(render_claim_decision(report), encoding="utf-8")

    output_sha256 = {
        "result.json": sha256_file(result_path),
        "report.md": sha256_file(report_path),
        "claim_decision.md": sha256_file(claim_path),
    }
    provenance_path = out / "analysis_provenance.json"
    provenance_payload = build_analysis_provenance_payload(
        report, provenance, output_sha256=output_sha256
    )
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    readme_path = out / "README.md"
    readme_path.write_text(render_readme(report, provenance), encoding="utf-8")

    return {
        "result.json": result_path,
        "report.md": report_path,
        "claim_decision.md": claim_path,
        "analysis_provenance.json": provenance_path,
        "README.md": readme_path,
    }


# --- sweep summary loading and gate ---------------------------------------


def sweep_summary_available(path: str | Path | None) -> bool:
    """Return whether a Gate 2 sweep summary file exists at ``path``."""
    if path is None:
        return False
    return Path(path).is_file()


def load_sweep_summary(path: str | Path) -> dict[str, object]:
    """Load a Gate 2 sweep summary JSON object, failing closed on any problem.

    Returns:
        The parsed sweep summary as a JSON object.

    Raises:
        FileNotFoundError: when the sweep summary file does not exist.
        ValueError: when the file is not a JSON object.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Gate 2 sweep summary not found: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gate 2 sweep summary is not valid JSON: {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Gate 2 sweep summary must be a JSON object: {resolved}")
    return payload
