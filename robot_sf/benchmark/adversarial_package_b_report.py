"""Fail-closed validation for issue #3079 Package B comparison reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.adversarial_package_b_preflight import (
    EXPECTED_BUDGETS,
    EXPECTED_OBJECTIVE,
    EXPECTED_REPORTING_FIELDS,
    EXPECTED_SAMPLERS,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = "adversarial-package-b-report-gate.v1"
REPORT_SCHEMA_VERSION = "adversarial-sampler-comparison.v3"
EXPECTED_ISSUE = 3079
EXPECTED_CLAIM_SCOPE = "not_paper_facing_benchmark_evidence"
EXPECTED_REPORT_STATUS = "diagnostic_local_nominal"
EXPECTED_HELD_OUT_STATUS = "not_evaluated_narrow_archive"
EXPECTED_SEEDS = (1101, 2202, 3303)
EXPECTED_ROW_COUNT = len(EXPECTED_BUDGETS) * len(EXPECTED_SEEDS) * len(EXPECTED_SAMPLERS)
_REQUIRED_ROW_METADATA_FIELDS = {
    "manifest_path",
    "best_bundle_path",
    "best_objective_value",
    "best_valid_objective",
    "num_candidates",
    "num_valid_candidates",
    "num_invalid_candidates",
    "num_failed_evaluations",
    "held_out_family_status",
    "caveats",
}
REQUIRED_ROW_FIELDS = frozenset(EXPECTED_REPORTING_FIELDS | _REQUIRED_ROW_METADATA_FIELDS)


@dataclass(frozen=True)
class PackageBReportGate:
    """Validation result for one generated Package B report."""

    report_path: str
    ready: bool
    status: str
    errors: tuple[str, ...]
    blockers: dict[str, tuple[str, ...]]
    matrix: dict[str, Any]
    next_empirical_action: str

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON payload for review and downstream checks."""
        return {
            "schema_version": SCHEMA_VERSION,
            "issue": EXPECTED_ISSUE,
            "report_path": self.report_path,
            "ready": self.ready,
            "status": self.status,
            "errors": list(self.errors),
            "blockers": {key: list(value) for key, value in self.blockers.items()},
            "matrix": self.matrix,
            "next_empirical_action": self.next_empirical_action,
            "claim_boundary": (
                "diagnostic-only report validation; this gate does not promote benchmark or "
                "paper-facing evidence"
            ),
        }


def validate_package_b_report(report_path: Path) -> PackageBReportGate:
    """Validate the complete Package B matrix and preserve its claim boundary.

    The gate validates report structure and per-row arithmetic only. It does not
    trust the report as proof of certification or replay: those claims still
    require inspection of the referenced artifacts and independent confirmation.

    Returns:
        A fail-closed gate result with matrix diagnostics and review blockers.
    """
    payload, errors = _load_payload(report_path)
    _validate_header(payload, errors)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        errors.append("rows must be a list")
        rows = []
    observed_keys, fallback_count, degraded_count = _validate_rows(rows, errors)
    _validate_matrix_shape(rows, observed_keys, errors)
    return _build_gate(
        report_path=report_path,
        errors=errors,
        row_count=len(rows),
        fallback_count=fallback_count,
        degraded_count=degraded_count,
    )


def _load_payload(report_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Load a JSON report while converting file and shape errors to blockers.

    Returns:
        The parsed mapping and accumulated load errors.
    """
    errors: list[str] = []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"report could not be read as JSON: {exc}")
        payload = {}

    if not isinstance(payload, dict):
        errors.append("report payload must be a mapping")
        payload = {}
    return payload, errors


def _validate_header(payload: dict[str, Any], errors: list[str]) -> None:
    """Validate the fixed Package B report header and matrix declarations."""
    _check_equal(payload, "schema_version", REPORT_SCHEMA_VERSION, errors)
    _check_equal(payload, "report_status", EXPECTED_REPORT_STATUS, errors)
    _check_equal(payload, "claim_scope", EXPECTED_CLAIM_SCOPE, errors)
    _check_list(payload, "objectives", (EXPECTED_OBJECTIVE,), errors)
    _check_list(payload, "budget_grid", EXPECTED_BUDGETS, errors)
    _check_list(payload, "seeds", EXPECTED_SEEDS, errors)


def _validate_rows(
    rows: list[Any], errors: list[str]
) -> tuple[list[tuple[str, int, int]], int, int]:
    """Validate row fields and return observed keys plus limited-row counts.

    Returns:
        Observed matrix keys, fallback-candidate count, and degraded-candidate count.
    """
    observed_keys: list[tuple[str, int, int]] = []
    seen_keys: set[tuple[str, int, int]] = set()
    fallback_count = 0
    degraded_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"rows[{index}] must be a mapping")
            continue
        missing = sorted(REQUIRED_ROW_FIELDS - row.keys())
        if missing:
            errors.append(f"rows[{index}] missing required fields: {missing}")
            continue
        sampler = row.get("sampler")
        budget = row.get("budget")
        seed = row.get("seed")
        if sampler not in EXPECTED_SAMPLERS:
            errors.append(f"rows[{index}].sampler is not a Package B sampler: {sampler!r}")
        if budget not in EXPECTED_BUDGETS:
            errors.append(f"rows[{index}].budget is not in the Package B grid: {budget!r}")
        if seed not in EXPECTED_SEEDS:
            errors.append(f"rows[{index}].seed is not a declared repeated seed: {seed!r}")
        if (
            isinstance(sampler, str)
            and isinstance(budget, int)
            and not isinstance(budget, bool)
            and isinstance(seed, int)
            and not isinstance(seed, bool)
        ):
            key = (sampler, budget, seed)
            if key in seen_keys:
                errors.append(f"rows[{index}] duplicates matrix cell {key!r}")
            else:
                seen_keys.add(key)
                observed_keys.append(key)
        _validate_row(row, index=index, errors=errors)
        fallback_count += _non_negative_count(row.get("fallback_candidate_count"))
        degraded_count += _non_negative_count(row.get("degraded_candidate_count"))
    return observed_keys, fallback_count, degraded_count


def _validate_matrix_shape(
    rows: list[Any], observed_keys: list[tuple[str, int, int]], errors: list[str]
) -> None:
    """Require every sampler/budget/seed cell exactly once."""
    expected_keys = {
        (sampler, budget, seed)
        for sampler in EXPECTED_SAMPLERS
        for budget in EXPECTED_BUDGETS
        for seed in EXPECTED_SEEDS
    }
    observed_key_set = set(observed_keys)
    missing_keys = sorted(expected_keys - observed_key_set)
    if missing_keys:
        errors.append(f"rows missing Package B matrix cells: {missing_keys}")
    if len(rows) != EXPECTED_ROW_COUNT:
        errors.append(f"rows must contain exactly {EXPECTED_ROW_COUNT} matrix cells")


def _build_gate(
    *,
    report_path: Path,
    errors: list[str],
    row_count: int,
    fallback_count: int,
    degraded_count: int,
) -> PackageBReportGate:
    """Build the claim-boundary result from structural and limitation checks.

    Returns:
        A structured Package B report gate result.
    """
    limited_rows = fallback_count + degraded_count
    intentional = (
        "held-out-family yield remains unevaluated against the narrow certified archive",
        "learned failure proposal #2921 remains out of scope until the base comparison succeeds",
        "paper-facing and dissertation claims remain forbidden",
    )
    remaining = (
        "certification, deterministic replay, independent-seed confirmation, and mechanism "
        "attribution still require artifact-level review",
    )
    new_blockers = list(errors)
    if limited_rows:
        new_blockers.append(
            "fallback/degraded candidate rows are diagnostic limitations and cannot enter "
            "success evidence"
        )
    blockers = {
        "remaining": remaining,
        "new": tuple(new_blockers),
        "intentional": intentional,
    }
    ready = not errors and limited_rows == 0
    status = (
        "ready_for_empirical_review"
        if ready
        else "diagnostic_only_limited_rows"
        if not errors
        else "blocked_on_report_contract"
    )
    return PackageBReportGate(
        report_path=report_path.as_posix(),
        ready=ready,
        status=status,
        errors=tuple(errors),
        blockers=blockers,
        matrix={
            "expected_row_count": EXPECTED_ROW_COUNT,
            "observed_row_count": row_count,
            "expected_samplers": list(EXPECTED_SAMPLERS),
            "expected_budgets": list(EXPECTED_BUDGETS),
            "expected_seeds": list(EXPECTED_SEEDS),
            "fallback_candidate_count": fallback_count,
            "degraded_candidate_count": degraded_count,
        },
        next_empirical_action=(
            "Run the exact manifest command on an approved compute-capable path, then review "
            "this gate together with certification and independent replay artifacts before "
            "interpreting any sampler difference."
        ),
    )


def _check_equal(payload: dict[str, Any], key: str, expected: Any, errors: list[str]) -> None:
    """Append an error when a scalar report field drifts."""
    if payload.get(key) != expected:
        errors.append(f"{key} must equal {expected!r}; found {payload.get(key)!r}")


def _check_list(
    payload: dict[str, Any], key: str, expected: tuple[Any, ...], errors: list[str]
) -> None:
    """Append an error when an ordered report list drifts."""
    value = payload.get(key)
    if value != list(expected):
        errors.append(f"{key} must equal {list(expected)!r}; found {value!r}")


def _validate_row(row: dict[str, Any], *, index: int, errors: list[str]) -> None:
    """Validate row counts, rates, and the certified/replayable denominator contract."""
    counts = _validate_row_counts(row, index=index, errors=errors)
    _validate_count_relationships(row, counts, index=index, errors=errors)
    _validate_row_rates(row, counts, index=index, errors=errors)
    _validate_row_boundaries(row, counts["candidate"], index=index, errors=errors)


def _validate_row_counts(row: dict[str, Any], *, index: int, errors: list[str]) -> dict[str, int]:
    """Validate and normalize the integer counts in one report row.

    Returns:
        Normalized counts used by the remaining row checks.
    """
    count_fields = (
        "num_candidates",
        "num_valid_candidates",
        "num_invalid_candidates",
        "num_failed_evaluations",
        "certified_valid_failure_count",
        "replayable_valid_failure_count",
        "fallback_candidate_count",
        "degraded_candidate_count",
    )
    for field in count_fields:
        if not _is_non_negative_count(row.get(field)):
            errors.append(f"rows[{index}].{field} must be a non-negative integer")
    return {
        "candidate": _non_negative_count(row.get("num_candidates")),
        "valid": _non_negative_count(row.get("num_valid_candidates")),
        "invalid": _non_negative_count(row.get("num_invalid_candidates")),
        "failed": _non_negative_count(row.get("num_failed_evaluations")),
        "certified": _non_negative_count(row.get("certified_valid_failure_count")),
        "replayable": _non_negative_count(row.get("replayable_valid_failure_count")),
        "fallback": _non_negative_count(row.get("fallback_candidate_count")),
        "degraded": _non_negative_count(row.get("degraded_candidate_count")),
    }


def _validate_count_relationships(
    row: dict[str, Any], counts: dict[str, int], *, index: int, errors: list[str]
) -> None:
    """Validate candidate, failure, and limitation count relationships."""
    candidate_count = counts["candidate"]
    if candidate_count != row.get("budget"):
        errors.append(f"rows[{index}].num_candidates must equal budget")
    if counts["valid"] + counts["invalid"] != candidate_count:
        errors.append(f"rows[{index}] valid and invalid candidate counts do not sum to budget")
    if counts["certified"] > counts["valid"]:
        errors.append(f"rows[{index}] certified failures exceed valid candidates")
    if counts["replayable"] > counts["certified"]:
        errors.append(f"rows[{index}] replayable failures exceed certified failures")
    if counts["failed"] > candidate_count:
        errors.append(f"rows[{index}] failed evaluations exceed candidates")
    if counts["fallback"] > candidate_count:
        errors.append(f"rows[{index}] fallback candidates exceed candidates")
    if counts["degraded"] > candidate_count:
        errors.append(f"rows[{index}] degraded candidates exceed candidates")


def _validate_row_rates(
    row: dict[str, Any], counts: dict[str, int], *, index: int, errors: list[str]
) -> None:
    """Validate invalid-candidate and replay-success rates against denominators."""
    invalid_rate = _finite_number(row.get("invalid_candidate_rate"))
    if invalid_rate is None or not 0.0 <= invalid_rate <= 1.0:
        errors.append(f"rows[{index}].invalid_candidate_rate must be finite in [0, 1]")
    elif not math.isclose(
        invalid_rate,
        counts["invalid"] / counts["candidate"] if counts["candidate"] else 0.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        errors.append(f"rows[{index}].invalid_candidate_rate disagrees with its denominator")
    replay_rate = row.get("replay_success_rate")
    expected_replay_rate = (
        counts["replayable"] / counts["certified"] if counts["certified"] else None
    )
    if replay_rate is None:
        if expected_replay_rate is not None:
            errors.append(f"rows[{index}].replay_success_rate is missing for certified failures")
    elif expected_replay_rate is None:
        errors.append(f"rows[{index}].replay_success_rate must be null without certified failures")
    elif _finite_number(replay_rate) is None or not math.isclose(
        float(replay_rate), expected_replay_rate, rel_tol=1e-9, abs_tol=1e-9
    ):
        errors.append(f"rows[{index}].replay_success_rate disagrees with its denominator")


def _validate_row_boundaries(
    row: dict[str, Any], candidate_count: int, *, index: int, errors: list[str]
) -> None:
    """Validate first-failure and held-out-family claim-boundary fields."""
    first_failure = row.get("first_failure_iteration")
    if first_failure is not None and (
        isinstance(first_failure, bool)
        or not isinstance(first_failure, int)
        or not 1 <= first_failure <= candidate_count
    ):
        errors.append(f"rows[{index}].first_failure_iteration must be null or 1..budget")
    if row.get("held_out_family_yield") is not None:
        errors.append(f"rows[{index}].held_out_family_yield must remain null")
    if row.get("held_out_family_status") != EXPECTED_HELD_OUT_STATUS:
        errors.append(f"rows[{index}].held_out_family_status has an unexpected value")


def _finite_number(value: Any) -> float | None:
    """Return a finite numeric value, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _is_non_negative_count(value: Any) -> bool:
    """Return whether a value is an actual non-negative integer count."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _non_negative_count(value: Any) -> int:
    """Return a valid non-negative count or zero for error accumulation."""
    return value if _is_non_negative_count(value) else 0
