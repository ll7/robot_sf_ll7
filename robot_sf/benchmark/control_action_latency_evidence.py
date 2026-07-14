"""Promote control-action-latency sweep episode rows into durable evidence (issue #5034).

Issue #5034 asks to *execute* the control-action-latency fidelity sweep (wired by
PR #5026, scoped by parent #4977) and record whether the 0/100/300 ms-equivalent
delays change the configured safety metrics. PR #5061 merged the fail-closed
launch/readiness preflight plus a historical "blocked" packet. Once the native
campaign runs and emits raw episode rows under ``output/``, this module is the
**metric-evidence promotion** step that remains: it reads those rows, isolates the
``control_action_latency`` axis, reports the latency metadata plus success,
collision, and minimum-clearance metrics for each completed cell, and classifies
every non-native / fallback / degraded row as an **exclusion** rather than a
result (per the issue #691 benchmark fallback policy).

This module runs **no episodes** and makes **no benchmark / simulator-realism /
sim-to-real / paper-facing claim**. It is the deterministic promoter that turns a
raw campaign row file into a durable compact evidence bundle. It fails closed when
the raw rows do not cover the required action-latency step set (0, 1, 3) among
native result rows, so a partial or non-latency run cannot be silently promoted as
the latency sweep. When given the serialized fixed-scope run plan, it additionally
requires exact planner-group / variant / seed / scenario coverage before promotion.

Episode row contract (the shape :func:`run_episode` in
``scripts/benchmark/run_fidelity_sensitivity_campaign.py`` emits)::

    {
        "axis": "control_action_latency",
        "variant": "control_action_latency__three_step_300ms",
        "variant_source_key": "three_step_300ms",
        "baseline_variant": false,
        "runtime_binding": "sim_config.action_latency_steps",
        "action_latency": {  # sim_config.action_latency_metadata()
            "configured_steps": 3,
            "configured_ms": null,
            "effective_steps": 3,
            "effective_ms": 300.0,
        },
        "planner": "baseline_social_force",
        "scenario_id": "...",
        "seed": 111,
        "success": true,
        "collision": false,
        "metrics": {"success_rate": 1.0, "collision_rate": 0.0, "min_clearance": 0.42},
    }
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.control_action_latency_preflight import (
    AXIS_KEY,
    DECISION_READY,
    REQUIRED_LATENCY_STEPS,
    check_control_action_latency_axis,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.errors import RobotSfError
from robot_sf.evidence.distance_convention import DistanceConvention
from robot_sf.evidence.writers import review_marker, review_marker_comment, review_marker_json

PROMOTION_SCHEMA_VERSION = "control-action-latency-sweep-evidence-promotion.v2"
FIXED_SCOPE_COVERAGE_CONTRACT = "control_action_latency_fixed_scope_coverage.v1"
FIXED_SCOPE_PLAN_SCHEMA_VERSION = "issue_3207_fidelity_fixed_scope_run_plan.v2"
ISSUE = 5034
PARENT_ISSUE = 4977
DEPENDENCY_PR = "#5026"

#: Metrics the issue #5034 evidence summary must report per latency cell.
REQUIRED_RESULT_METRICS: tuple[str, ...] = ("success_rate", "collision_rate", "min_clearance")

#: Execution modes that count as native benchmark-success rows (issue #691 policy).
NATIVE_EXECUTION_MODES: frozenset[str] = frozenset({"native", "adapter"})
AVAILABLE_AVAILABILITY_STATUSES: frozenset[str] = frozenset({"available"})

CLAIM_BOUNDARY = (
    "control-action-latency metric-evidence promotion only: reads raw fidelity-campaign episode "
    "rows, isolates the control_action_latency axis, and reports action-latency metadata plus "
    "success / collision / minimum-clearance metrics per native latency cell. It runs no episode "
    "and promotes no claim beyond the declared campaign evidence tier; it is not "
    "simulator-realism evidence, not sim-to-real evidence, and not paper-facing evidence."
)

EXCLUSION_POLICY = (
    "Per the issue #691 benchmark fallback policy, any row whose execution_mode is not "
    "native/adapter or whose availability_status is not available, plus any latency-axis row "
    "missing action_latency metadata or a valid seed, is recorded as an exclusion and never "
    "contributes to the result metrics. This keeps fallback/degraded execution out of the "
    "latency result set."
)

EVIDENCE_TIER = "targeted smoke"
RESULT_CLASSIFICATION = "diagnostic-only"
DISTANCE_CONVENTION = DistanceConvention.SURFACE_CLEARANCE.value


class LatencyEvidenceError(RobotSfError, RuntimeError):
    """Raised when raw rows are not promotable as latency-sweep evidence (fail closed)."""


@dataclass(frozen=True)
class LatencyCell:
    """One classified control-action-latency episode row.

    ``classification`` is ``result`` for native, latency-metadata-bearing rows with
    a valid seed and ``exclusion`` for everything else (fallback / degraded /
    non-native / missing latency metadata). Only ``result`` cells contribute to the
    aggregate metrics.
    """

    planner: str
    planner_group: str | None
    latency_step: int | None
    latency_ms: float | None
    variant: str
    variant_source_key: str | None
    baseline_variant: bool
    seed: int | None
    scenario_id: str
    success_rate: float | None
    collision_rate: float | None
    min_clearance: float | None
    classification: str
    exclusion_reason: str | None
    execution_mode: str
    availability_status: str


def _coerce_int(value: Any) -> int | None:
    """Return an int for numeric values, rejecting bools and non-numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _row_execution_mode(row: Mapping[str, Any]) -> str:
    """Return a row's execution mode, defaulting to ``native`` when unset.

    The fail-closed campaign runner (``allow_fallback=False``) emits no fallback
    rows, so a completed row without an explicit marker is native. Rows that
    *do* carry a non-native marker are honored so the promoter degrades safely if
    a future runner path emits one.
    """
    mode = row.get("execution_mode")
    return str(mode) if isinstance(mode, str) and mode else "native"


def _row_availability_status(row: Mapping[str, Any]) -> str:
    """Return a row's availability status, defaulting to ``available`` when unset."""
    status = row.get("availability_status")
    return str(status) if isinstance(status, str) and status else "available"


def _row_seed(row: Mapping[str, Any]) -> int | None:
    """Return the integer seed marker, or ``None`` when it is missing/invalid."""
    return _coerce_int(row.get("seed"))


def _numeric_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    """Return a finite numeric metric value, or ``None`` when it is unusable."""
    value = metrics.get(name)
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _latency_step_from_row(row: Mapping[str, Any]) -> int | None:
    """Extract the effective action-latency step for a row.

    Prefers the structured ``action_latency.effective_steps`` metadata; falls back
    to ``action_latency.configured_steps`` then to the row-level
    ``action_latency_steps`` marker so rows produced before the metadata dict
    landed still resolve.

    Returns:
        The effective action-latency step, or ``None`` when no marker resolves.
    """
    metadata = row.get("action_latency")
    if isinstance(metadata, Mapping):
        for key in ("effective_steps", "configured_steps"):
            step = _coerce_int(metadata.get(key))
            if step is not None:
                return step
    return _coerce_int(row.get("action_latency_steps"))


def _latency_ms_from_row(row: Mapping[str, Any]) -> float | None:
    """Return the effective action-latency milliseconds for a row, if recorded.

    Returns:
        The effective latency in ms, or ``None`` when not recorded.
    """
    metadata = row.get("action_latency")
    if isinstance(metadata, Mapping):
        for key in ("effective_ms", "configured_ms"):
            value = metadata.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
    return None


def classify_latency_row(row: Mapping[str, Any]) -> LatencyCell:
    """Classify one episode row as a latency ``result`` cell or an ``exclusion``.

    The row must belong to the ``control_action_latency`` axis. A row is a
    ``result`` only when it carries action-latency metadata, a valid seed, finite
    required outcome metrics, a native/adapter execution mode, and an available
    status. Anything else (fallback / degraded / non-native / missing metadata,
    seed, or required metrics) becomes an ``exclusion`` with a precise reason, so
    malformed or fallback execution can never enter the latency result set.

    Returns:
        The classified :class:`LatencyCell` for the row.
    """
    metrics = row.get("metrics")
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    success_rate = _numeric_metric(metric_map, "success_rate")
    collision_rate = _numeric_metric(metric_map, "collision_rate")
    raw_min_clearance = metric_map.get("min_clearance")
    min_clearance = (
        float(raw_min_clearance)
        if isinstance(raw_min_clearance, int | float) and not isinstance(raw_min_clearance, bool)
        else None
    )

    execution_mode = _row_execution_mode(row)
    availability_status = _row_availability_status(row)
    latency_step = _latency_step_from_row(row)
    latency_ms = _latency_ms_from_row(row)
    seed = _row_seed(row)

    reasons: list[str] = []
    if latency_step is None:
        reasons.append("missing_action_latency_metadata")
    if seed is None:
        reasons.append("missing_or_invalid_seed")
    if execution_mode not in NATIVE_EXECUTION_MODES:
        reasons.append(f"non_native_execution_mode:{execution_mode}")
    if availability_status not in AVAILABLE_AVAILABILITY_STATUSES:
        reasons.append(f"unavailable:{availability_status}")
    for name, value in (("success_rate", success_rate), ("collision_rate", collision_rate)):
        if value is None:
            reasons.append(f"missing_or_invalid_metric:{name}")

    return LatencyCell(
        planner=str(row.get("planner") or "unknown"),
        planner_group=(
            str(row["planner_group"])
            if isinstance(row.get("planner_group"), str) and row["planner_group"]
            else None
        ),
        latency_step=latency_step,
        latency_ms=latency_ms,
        variant=str(row.get("variant") or row.get("variant_source_key") or "unknown"),
        variant_source_key=(
            str(row["variant_source_key"])
            if isinstance(row.get("variant_source_key"), str) and row["variant_source_key"]
            else None
        ),
        baseline_variant=bool(row.get("baseline_variant", False)),
        seed=seed,
        scenario_id=str(row.get("scenario_id") or "unknown"),
        success_rate=success_rate,
        collision_rate=collision_rate,
        min_clearance=min_clearance,
        classification="result" if not reasons else "exclusion",
        exclusion_reason="; ".join(reasons) if reasons else None,
        execution_mode=execution_mode,
        availability_status=availability_status,
    )


def extract_latency_cells(rows: Sequence[Mapping[str, Any]]) -> list[LatencyCell]:
    """Return latency cells for every ``control_action_latency`` axis row.

    Rows on other fidelity axes (clearance_radius, integration_timestep, ...) are
    ignored: they are not part of the latency sweep and must never contribute to
    its result metrics.
    """
    cells: list[LatencyCell] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("axis") or "") != AXIS_KEY:
            continue
        cells.append(classify_latency_row(row))
    return cells


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean of a non-empty sequence, else ``None``."""
    if not values:
        return None
    return float(sum(values) / len(values))


def aggregate_latency_metrics(cells: Sequence[LatencyCell]) -> list[dict[str, Any]]:
    """Aggregate result cells into per (planner, latency_step) metric rows.

    Exclusion cells never contribute. Each aggregate row reports the issue-#5034
    required metrics (success_rate, collision_rate, min_clearance) plus the cell
    count and the latency metadata (steps and ms). Rows are ordered by planner
    then latency step for deterministic output.

    Returns:
        One aggregate metric row per ``(planner, latency_step)`` result bucket.
    """
    buckets: dict[tuple[str, int], list[LatencyCell]] = {}
    for cell in cells:
        if cell.classification != "result" or cell.latency_step is None:
            continue
        buckets.setdefault((cell.planner, cell.latency_step), []).append(cell)

    aggregates: list[dict[str, Any]] = []
    for planner, latency_step in sorted(buckets, key=lambda item: (item[0][0], item[0][1])):
        bucket = buckets[(planner, latency_step)]
        min_clearances = [c.min_clearance for c in bucket if c.min_clearance is not None]
        aggregates.append(
            {
                "planner": planner,
                "action_latency_steps": latency_step,
                "action_latency_ms": _mean(
                    [c.latency_ms for c in bucket if c.latency_ms is not None]
                ),
                "cell_count": len(bucket),
                "seeds": sorted({c.seed for c in bucket}),
                "success_rate": _mean(
                    [c.success_rate for c in bucket if c.success_rate is not None]
                ),
                "collision_rate": _mean(
                    [c.collision_rate for c in bucket if c.collision_rate is not None]
                ),
                "min_clearance": _mean(min_clearances) if min_clearances else None,
            }
        )
    return aggregates


def _required_step_coverage(
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return which required latency steps are covered by native result aggregates.

    Returns:
        Coverage dict with ``required_latency_steps``, ``observed_latency_steps``,
        ``missing_latency_steps``, and ``coverage_complete``.
    """
    observed_steps = sorted({int(row["action_latency_steps"]) for row in aggregates})
    missing = [step for step in REQUIRED_LATENCY_STEPS if step not in observed_steps]
    return {
        "required_latency_steps": list(REQUIRED_LATENCY_STEPS),
        "observed_latency_steps": observed_steps,
        "missing_latency_steps": missing,
        "coverage_complete": not missing,
    }


def _latency_variant_steps(config: Mapping[str, Any]) -> dict[str, int]:
    """Return the configured action-latency step for every latency variant.

    Returns:
        Mapping from source variant key to configured action-latency steps.
    """
    axes = config.get("axes")
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
        raise LatencyEvidenceError("fidelity config has no valid axes list")
    axis = next(
        (
            candidate
            for candidate in axes
            if isinstance(candidate, Mapping) and candidate.get("key") == AXIS_KEY
        ),
        None,
    )
    if not isinstance(axis, Mapping):
        raise LatencyEvidenceError(f"fidelity config has no '{AXIS_KEY}' axis")
    variants = axis.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        raise LatencyEvidenceError(f"'{AXIS_KEY}' axis has no valid variants list")

    result: dict[str, int] = {}
    for raw_variant in variants:
        if not isinstance(raw_variant, Mapping):
            continue
        key = raw_variant.get("key")
        patch = raw_variant.get("patch")
        sim_config = patch.get("sim_config") if isinstance(patch, Mapping) else None
        step = (
            _coerce_int(sim_config.get("action_latency_steps"))
            if isinstance(sim_config, Mapping)
            else None
        )
        if isinstance(key, str) and key and step is not None:
            result[key] = step
    if not result:
        raise LatencyEvidenceError(f"'{AXIS_KEY}' axis has no action-latency variant steps")
    return result


def _scope_key_label(key: tuple[Any, Any, Any, Any]) -> str:
    """Render one fixed-scope identity key compactly for fail-closed errors.

    Returns:
        A bounded human-readable identity string.
    """
    planner_group, variant, seed, scenario_id = key
    return (
        f"planner_group={planner_group!r},variant={variant!r},seed={seed!r},"
        f"scenario_id={scenario_id!r}"
    )


def _sample_scope_keys(keys: Sequence[tuple[Any, Any, Any, Any]], limit: int = 3) -> str:
    """Render a bounded sample of fixed-scope identity keys.

    Returns:
        A compact list containing at most ``limit`` identity strings.
    """
    ordered = sorted(
        keys,
        key=lambda key: tuple("" if value is None else str(value) for value in key),
    )
    sample = [_scope_key_label(key) for key in ordered[:limit]]
    suffix = "..." if len(ordered) > limit else ""
    return "[" + "; ".join(sample) + suffix + "]"


def _fixed_scope_scenario_ids(plan: Mapping[str, Any]) -> list[str]:
    """Validate the plan header and return its resolved scenario identifiers.

    Returns:
        Unique scenario identifiers from a ready fixed-scope plan.
    """
    if plan.get("schema_version") != FIXED_SCOPE_PLAN_SCHEMA_VERSION:
        raise LatencyEvidenceError(
            "fixed-scope plan schema is unsupported; expected "
            f"{FIXED_SCOPE_PLAN_SCHEMA_VERSION!r}, got {plan.get('schema_version')!r}"
        )
    if plan.get("preflight_ready") is not True:
        raise LatencyEvidenceError(
            "fixed-scope plan is not preflight-ready; refusing to validate campaign evidence"
        )

    resolution = plan.get("scenario_resolution")
    if not isinstance(resolution, Mapping) or resolution.get("status") != "ready":
        raise LatencyEvidenceError(
            "fixed-scope plan has no ready scenario resolution; refusing to validate evidence"
        )
    raw_scenario_ids = resolution.get("scenario_ids")
    if not isinstance(raw_scenario_ids, Sequence) or isinstance(raw_scenario_ids, (str, bytes)):
        raise LatencyEvidenceError("fixed-scope plan has no scenario_ids list")
    if not all(isinstance(value, str) and value for value in raw_scenario_ids):
        raise LatencyEvidenceError("fixed-scope plan scenario_ids must be non-empty strings")
    scenario_ids = list(raw_scenario_ids)
    if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
        raise LatencyEvidenceError("fixed-scope plan scenario_ids must be non-empty and unique")
    return scenario_ids


def _fixed_scope_expected_cells(
    plan: Mapping[str, Any],
    scenario_ids: Sequence[str],
    variant_steps: Mapping[str, int],
) -> tuple[dict[tuple[Any, Any, Any, Any], int], int]:
    """Expand latency run cells across the plan's resolved scenarios.

    Returns:
        A map from expected row identity to configured latency steps, plus the
        number of latency cells per scenario declared by the plan.
    """
    run_cells = plan.get("run_cells")
    if not isinstance(run_cells, Sequence) or isinstance(run_cells, (str, bytes)):
        raise LatencyEvidenceError("fixed-scope plan has no run_cells list")
    expected: dict[tuple[Any, Any, Any, Any], int] = {}
    latency_cell_count = 0
    for raw_cell in run_cells:
        if not isinstance(raw_cell, Mapping) or raw_cell.get("axis") != AXIS_KEY:
            continue
        latency_cell_count += 1
        planner_group = raw_cell.get("planner_group")
        variant = raw_cell.get("variant")
        seed = _coerce_int(raw_cell.get("seed"))
        if not isinstance(planner_group, str) or not planner_group:
            raise LatencyEvidenceError("fixed-scope latency plan cell is missing planner_group")
        if not isinstance(variant, str) or variant not in variant_steps:
            raise LatencyEvidenceError(
                f"fixed-scope latency plan cell names an unknown variant: {variant!r}"
            )
        if seed is None:
            raise LatencyEvidenceError(
                f"fixed-scope latency plan cell has an invalid seed: {raw_cell.get('seed')!r}"
            )
        for scenario_id in scenario_ids:
            key = (planner_group, variant, seed, scenario_id)
            if key in expected:
                raise LatencyEvidenceError(
                    "fixed-scope latency plan contains duplicate expected cell: "
                    + _scope_key_label(key)
                )
            expected[key] = variant_steps[variant]

    if not expected:
        raise LatencyEvidenceError(f"fixed-scope plan contains no '{AXIS_KEY}' run cells")
    return expected, latency_cell_count


def _fixed_scope_expectations(
    plan: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[tuple[Any, Any, Any, Any], int], dict[str, Any]]:
    """Build expected latency-row identities from a serialized fixed-scope plan.

    Returns:
        Expected identity-to-step mapping and a compact scope summary.
    """
    scenario_ids = _fixed_scope_scenario_ids(plan)
    variant_steps = _latency_variant_steps(config)
    expected, latency_cell_count = _fixed_scope_expected_cells(plan, scenario_ids, variant_steps)
    return expected, {
        "scenario_count": len(scenario_ids),
        "planner_group_count": len({key[0] for key in expected}),
        "variant_count": len({key[1] for key in expected}),
        "seed_count": len({key[2] for key in expected}),
        "latency_cell_count_per_scenario": latency_cell_count,
        "expected_row_count": len(expected),
        "expected_latency_steps": sorted(set(expected.values())),
    }


def _fixed_scope_observed_rows(
    rows: Sequence[Mapping[str, Any]], cells: Sequence[LatencyCell]
) -> dict[tuple[Any, Any, Any, Any], list[tuple[LatencyCell, Mapping[str, Any]]]]:
    """Group classified latency rows by their fixed-scope identity.

    Returns:
        Observed identity keys mapped to their raw row and classified-cell entries.
    """
    latency_rows = [
        row for row in rows if isinstance(row, Mapping) and str(row.get("axis") or "") == AXIS_KEY
    ]
    if len(latency_rows) != len(cells):
        raise LatencyEvidenceError(
            "fixed-scope latency coverage input is internally inconsistent: "
            f"{len(latency_rows)} latency rows produced {len(cells)} classified cells"
        )

    observed: dict[tuple[Any, Any, Any, Any], list[tuple[LatencyCell, Mapping[str, Any]]]] = {}
    for row, cell in zip(latency_rows, cells, strict=True):
        key = (cell.planner_group, cell.variant_source_key, cell.seed, cell.scenario_id)
        observed.setdefault(key, []).append((cell, row))
    return observed


def _fixed_scope_coverage_failures(
    expected: Mapping[tuple[Any, Any, Any, Any], int],
    observed: Mapping[tuple[Any, Any, Any, Any], Sequence[tuple[LatencyCell, Mapping[str, Any]]]],
) -> dict[str, Any]:
    """Classify missing, duplicate, malformed, and unexpected fixed-scope rows.

    Returns:
        Failure buckets plus the set of valid expected identity keys.
    """
    valid_keys: set[tuple[Any, Any, Any, Any]] = set()
    missing_keys: list[tuple[Any, Any, Any, Any]] = []
    duplicate_keys: list[tuple[Any, Any, Any, Any]] = []
    invalid_keys: list[tuple[Any, Any, Any, Any]] = []
    for key, expected_step in expected.items():
        entries = observed.get(key, [])
        if not entries:
            missing_keys.append(key)
            continue
        if len(entries) > 1:
            duplicate_keys.append(key)
            continue
        cell, row = entries[0]
        metadata = row.get("action_latency")
        if (
            cell.classification != "result"
            or not isinstance(metadata, Mapping)
            or cell.latency_step != expected_step
        ):
            invalid_keys.append(key)
            continue
        valid_keys.add(key)

    return {
        "valid_keys": valid_keys,
        "missing_keys": missing_keys,
        "duplicate_keys": duplicate_keys,
        "invalid_keys": invalid_keys,
        "unexpected_keys": [key for key in observed if key not in expected],
    }


def validate_fixed_scope_latency_coverage(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[LatencyCell],
    *,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact fixed-scope identity coverage before promoting real rows.

    The ordinary promoter intentionally accepts compact representative fixtures.
    Supplying the issue #3207 fixed-scope plan enables the stricter #5034 contract:
    every expected planner-group/variant/seed/scenario cell must appear exactly once,
    carry structured action-latency metadata, and classify as a native result. Any
    missing, duplicate, unexpected, fallback, degraded, or malformed expected cell
    fails closed before a durable summary can be written.

    Returns:
        A compact verified coverage summary for the fixed-scope packet.
    """
    expected, summary = _fixed_scope_expectations(plan, config)
    observed = _fixed_scope_observed_rows(rows, cells)
    failures = _fixed_scope_coverage_failures(expected, observed)
    missing_keys = failures["missing_keys"]
    duplicate_keys = failures["duplicate_keys"]
    invalid_keys = failures["invalid_keys"]
    unexpected_keys = failures["unexpected_keys"]
    if missing_keys or duplicate_keys or invalid_keys or unexpected_keys:
        reasons: list[str] = []
        if missing_keys:
            reasons.append(
                f"missing_expected={len(missing_keys)} sample={_sample_scope_keys(missing_keys)}"
            )
        if duplicate_keys:
            reasons.append(
                f"duplicate_expected={len(duplicate_keys)} sample={_sample_scope_keys(duplicate_keys)}"
            )
        if invalid_keys:
            reasons.append(
                f"non_native_or_malformed_expected={len(invalid_keys)} "
                f"sample={_sample_scope_keys(invalid_keys)}"
            )
        if unexpected_keys:
            reasons.append(
                f"unexpected_rows={len(unexpected_keys)} sample={_sample_scope_keys(unexpected_keys)}"
            )
        raise LatencyEvidenceError(f"{FIXED_SCOPE_COVERAGE_CONTRACT} failed: " + "; ".join(reasons))

    summary.update(
        {
            "contract": FIXED_SCOPE_COVERAGE_CONTRACT,
            "status": "verified",
            "observed_row_count": len(cells),
            "observed_result_row_count": len(failures["valid_keys"]),
            "observed_planner_groups": sorted({key[0] for key in failures["valid_keys"]}),
            "observed_variant_source_keys": sorted({key[1] for key in failures["valid_keys"]}),
            "observed_seeds": sorted({key[2] for key in failures["valid_keys"]}),
        }
    )
    return summary


def _exclusion_summary(cells: Sequence[LatencyCell]) -> dict[str, Any]:
    """Return a summary of excluded latency rows and their reasons (#691 discipline).

    Returns:
        Summary dict with ``excluded_row_count``, ``reason_counts``, and a small
        ``sample_exclusions`` list.
    """
    exclusions = [cell for cell in cells if cell.classification == "exclusion"]
    reason_counts: dict[str, int] = {}
    for exclusion in exclusions:
        for reason in (exclusion.exclusion_reason or "unknown").split("; "):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "excluded_row_count": len(exclusions),
        "reason_counts": dict(sorted(reason_counts.items())),
        "sample_exclusions": [asdict(cell) for cell in exclusions[:5]],
    }


def build_latency_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_path: str,
    git_head: str,
    date: str | None,
    raw_rows_path: str,
    fixed_scope_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the durable compact control-action-latency evidence packet.

    Fails closed (:class:`LatencyEvidenceError`) when the latency preflight is not
    ``ready`` or when native result rows do not cover every required action-latency
    step (0, 1, 3), so a partial or non-latency run cannot be promoted as the
    latency sweep.

    Args:
        rows: Raw episode rows from the fidelity campaign runner.
        config: Raw fidelity-sensitivity study config mapping.
        config_path: Repo-relative config path recorded for provenance.
        git_head: Git head recorded for provenance.
        date: ISO date string recorded for provenance.
        raw_rows_path: Repo-relative path of the source raw row file.
        fixed_scope_plan: Optional serialized fixed-scope run plan. When supplied,
            exact scenario/planner-group/variant/seed coverage is required before
            the packet can be promoted.

    Returns:
        JSON-serializable evidence packet with per-cell and aggregate latency
        metrics plus the exclusion classification.
    """
    preflight = check_control_action_latency_axis(
        config, config_path=config_path, git_head=git_head, date=date
    )
    if preflight["decision"] != DECISION_READY:
        raise LatencyEvidenceError(
            "control-action-latency preflight is not ready; refusing to promote. Blockers: "
            + "; ".join(preflight.get("blockers") or ["unknown"])
        )

    cells = extract_latency_cells(rows)
    if not cells:
        raise LatencyEvidenceError(
            f"raw rows contain no '{AXIS_KEY}' axis rows; cannot promote as the latency sweep"
        )

    fixed_scope_coverage = None
    if fixed_scope_plan is not None:
        fixed_scope_coverage = validate_fixed_scope_latency_coverage(
            rows,
            cells,
            plan=fixed_scope_plan,
            config=config,
        )

    aggregates = aggregate_latency_metrics(cells)
    coverage = _required_step_coverage(aggregates)
    if not coverage["coverage_complete"]:
        raise LatencyEvidenceError(
            "native latency result rows do not cover every required action-latency step. "
            f"required={list(REQUIRED_LATENCY_STEPS)} "
            f"observed={coverage['observed_latency_steps']} "
            f"missing={coverage['missing_latency_steps']}"
        )

    result_cells = [cell for cell in cells if cell.classification == "result"]
    return {
        "review_marker": review_marker_json(),
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "depends_on_pr": DEPENDENCY_PR,
        "date": date,
        "git_head": git_head,
        "config_path": config_path,
        "raw_rows_path": raw_rows_path,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": EVIDENCE_TIER,
        "result_classification": RESULT_CLASSIFICATION,
        "distance_convention": DISTANCE_CONVENTION,
        "exclusion_policy": EXCLUSION_POLICY,
        "preflight_decision": preflight["decision"],
        "preflight_axis_present": preflight["axis_present"],
        "preflight_observed_latency_steps": preflight["observed_latency_steps"],
        "required_result_metrics": list(REQUIRED_RESULT_METRICS),
        "latency_coverage": coverage,
        "fixed_scope_coverage": fixed_scope_coverage,
        "scope": {
            "latency_row_count": len(cells),
            "result_row_count": len(result_cells),
            "excluded_row_count": len(cells) - len(result_cells),
            "planners": sorted({cell.planner for cell in result_cells}),
            "seeds": sorted({cell.seed for cell in result_cells}),
            "scenario_ids": sorted({cell.scenario_id for cell in result_cells}),
        },
        "aggregate_metrics": aggregates,
        "per_cell_metrics": [asdict(cell) for cell in cells],
        "exclusions": _exclusion_summary(cells),
        "artifact_policy": (
            "Compact promotion artifacts are tracked here. Raw episode JSONL remains ignored under "
            "output/ and needs a durable external storage pointer before paper-facing use."
        ),
    }


def _format_markdown(packet: Mapping[str, Any]) -> str:
    """Return a compact human-readable evidence Markdown string.

    Returns:
        Markdown rendering of the promotion packet.
    """
    scope = packet["scope"]
    coverage = packet["latency_coverage"]
    exclusions = packet["exclusions"]
    lines = [
        review_marker("robot_sf#5034", marker_date=str(packet.get("date") or "") or None),
        f"# Issue #{ISSUE} Control-action-latency sweep evidence {packet.get('date') or ''}",
        "",
        "Plain-language summary: this bundle promotes raw fidelity-campaign episode rows into a "
        "compact control-action-latency evidence summary. It reports the 0/100/300 ms-equivalent "
        "delay cells' success, collision, and minimum-clearance metrics and excludes any "
        "fallback/degraded/non-native rows. It is not paper-facing evidence.",
        "",
        f"- Schema: `{packet['schema_version']}`",
        f"- Git head: `{packet.get('git_head')}`",
        f"- Raw rows: `{packet.get('raw_rows_path')}`",
        f"- Preflight decision: `{packet['preflight_decision']}`",
        f"- Evidence tier: `{packet['evidence_tier']}`",
        f"- Result classification: `{packet['result_classification']}`",
        f"- Distance convention: `{packet['distance_convention']}`",
        f"- Claim boundary: {packet['claim_boundary']}",
        "",
        "## Scope",
        "",
        f"- Latency rows: `{scope['latency_row_count']}` (results `{scope['result_row_count']}`, "
        f"excluded `{scope['excluded_row_count']}`)",
        f"- Planners: `{', '.join(scope['planners']) or 'none'}`",
        f"- Seeds: `{', '.join(str(seed) for seed in scope['seeds']) or 'none'}`",
        f"- Latency-step coverage: required `{coverage['required_latency_steps']}`, "
        f"observed `{coverage['observed_latency_steps']}`, "
        f"missing `{coverage['missing_latency_steps'] or 'none'}`",
        "",
        "## Aggregate metrics per latency cell",
        "",
        "| Planner | Latency steps | Latency ms | Cells | Success | Collision | Min clearance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fixed_scope_coverage = packet.get("fixed_scope_coverage")
    if isinstance(fixed_scope_coverage, Mapping):
        aggregate_header = lines.index("## Aggregate metrics per latency cell")
        lines[aggregate_header:aggregate_header] = [
            "- Fixed-scope coverage: "
            f"`{fixed_scope_coverage['status']}` "
            f"({fixed_scope_coverage['observed_result_row_count']}/"
            f"{fixed_scope_coverage['expected_row_count']} expected rows)",
            "",
        ]
    for row in packet["aggregate_metrics"]:
        ms = row.get("action_latency_ms")
        min_clearance = row.get("min_clearance")
        lines.append(
            f"| `{row['planner']}` | {row['action_latency_steps']} | "
            f"{ms if ms is not None else 'NA'} | {row['cell_count']} | "
            f"{row['success_rate']:.6g} | {row['collision_rate']:.6g} | "
            f"{min_clearance if min_clearance is not None else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## Exclusions (fallback / degraded / non-native)",
            "",
            f"- Excluded rows: `{exclusions['excluded_row_count']}`",
            f"- Reasons: `{exclusions['reason_counts'] or 'none'}`",
            "",
            "Per the issue #691 benchmark fallback policy, excluded rows never contribute to "
            "the result metrics above.",
            "",
            "## Files",
            "",
            "- `summary.json`: full promotion packet (aggregate + per-cell + exclusions).",
            "- `per_cell_metrics.csv`: compact per-cell latency metrics table.",
            "- `manifest.sha256`: checksums for promoted compact artifacts.",
            "- `README.md`: this human-readable summary.",
            "",
        ]
    )
    return "\n".join(lines)


def write_latency_evidence(packet: Mapping[str, Any], evidence_dir: str | Path) -> list[Path]:
    """Write the durable compact latency evidence bundle.

    Writes ``summary.json``, ``per_cell_metrics.csv``, ``README.md``, and
    ``manifest.sha256`` into ``evidence_dir``.

    Returns:
        The list of written artifact paths.
    """
    out = Path(evidence_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps({"review_marker": review_marker_json(), **packet}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    csv_path = out / "per_cell_metrics.csv"
    fieldnames = [
        "planner",
        "planner_group",
        "latency_step",
        "latency_ms",
        "variant",
        "variant_source_key",
        "baseline_variant",
        "seed",
        "scenario_id",
        "success_rate",
        "collision_rate",
        "min_clearance",
        "classification",
        "exclusion_reason",
        "execution_mode",
        "availability_status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(review_marker_comment() + "\n")
        handle.write(f"# distance_convention: {DISTANCE_CONVENTION}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for cell in packet["per_cell_metrics"]:
            writer.writerow(
                {
                    "planner": cell["planner"],
                    "planner_group": cell["planner_group"] or "",
                    "latency_step": cell["latency_step"]
                    if cell["latency_step"] is not None
                    else "",
                    "latency_ms": cell["latency_ms"] if cell["latency_ms"] is not None else "",
                    "variant": cell["variant"],
                    "variant_source_key": cell["variant_source_key"] or "",
                    "baseline_variant": cell["baseline_variant"],
                    "seed": cell["seed"],
                    "scenario_id": cell["scenario_id"],
                    "success_rate": cell["success_rate"],
                    "collision_rate": cell["collision_rate"],
                    "min_clearance": cell["min_clearance"]
                    if cell["min_clearance"] is not None
                    else "",
                    "classification": cell["classification"],
                    "exclusion_reason": cell["exclusion_reason"] or "",
                    "execution_mode": cell["execution_mode"],
                    "availability_status": cell["availability_status"],
                }
            )

    readme_path = out / "README.md"
    readme_path.write_text(_format_markdown(packet), encoding="utf-8")

    copied = [summary_path, csv_path, readme_path]
    manifest_path = out / "manifest.sha256"
    manifest_path.write_text(
        review_marker_comment()
        + "\n"
        + "\n".join(f"{sha256_file(path)}  {path.name}" for path in copied)
        + "\n",
        encoding="utf-8",
    )
    copied.append(manifest_path)
    return copied


def load_latency_rows(raw_rows_path: str | Path) -> list[dict[str, Any]]:
    """Return newline-delimited JSON episode rows emitted by the campaign runner.

    Returns:
        The list of parsed row mappings.
    """
    path = Path(raw_rows_path)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise LatencyEvidenceError(
                        f"raw rows file {path} has invalid JSON on line {line_number}: {exc}"
                    ) from exc
                if isinstance(row, dict):
                    rows.append(row)
    except OSError as exc:
        raise LatencyEvidenceError(f"raw rows file {path} cannot be read: {exc}") from exc
    return rows


def load_fixed_scope_plan(plan_path: str | Path) -> dict[str, Any]:
    """Load a serialized fixed-scope plan for strict evidence promotion.

    Returns:
        The decoded fixed-scope plan mapping.
    """
    path = Path(plan_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LatencyEvidenceError(f"fixed-scope plan {path} cannot be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise LatencyEvidenceError(f"fixed-scope plan {path} must contain a JSON object")
    return payload


def promote_latency_evidence(
    raw_rows_path: str | Path,
    evidence_dir: str | Path,
    *,
    config: Mapping[str, Any],
    config_path: str,
    git_head: str,
    date: str | None,
    fixed_scope_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load raw rows, build the latency evidence packet, and write the bundle.

    Convenience entry point that combines :func:`load_latency_rows`,
    :func:`build_latency_evidence`, and :func:`write_latency_evidence`. Raises
    :class:`LatencyEvidenceError` (fail closed) when the rows cannot be promoted
    as the latency sweep.

    Returns:
        A promotion result dict with status, evidence dir, written files, and
        coverage summary.
    """
    rows = load_latency_rows(raw_rows_path)
    packet = build_latency_evidence(
        rows,
        config=config,
        config_path=config_path,
        git_head=git_head,
        date=date,
        raw_rows_path=str(raw_rows_path),
        fixed_scope_plan=fixed_scope_plan,
    )
    written = write_latency_evidence(packet, evidence_dir)
    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "promoted",
        "issue": ISSUE,
        "evidence_dir": str(evidence_dir),
        "written_files": [str(path) for path in written],
        "result_row_count": packet["scope"]["result_row_count"],
        "excluded_row_count": packet["scope"]["excluded_row_count"],
        "latency_coverage": packet["latency_coverage"],
        "fixed_scope_coverage": packet["fixed_scope_coverage"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
