"""Fail-closed validation for Package B confirmed-failure evidence.

The Package B report gate validates matrix shape and row arithmetic. This
second gate validates the stronger discovery boundary from issue #3079:
certification, deterministic replay, independent-seed confirmation, and stable
failure-mechanism attribution must all be present before a failure is counted
as confirmed. It consumes post-run JSON artifacts and never executes a
campaign or promotes benchmark evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.adversarial_package_b_report import (
    EXPECTED_BUDGETS,
    EXPECTED_CLAIM_SCOPE,
    EXPECTED_SAMPLERS,
    EXPECTED_SEEDS,
    REPORT_SCHEMA_VERSION,
    validate_package_b_report,
)

SCHEMA_VERSION = "adversarial-package-b-confirmation.v1"
EXPECTED_ISSUE = 3079
EXPECTED_ROW_COUNT = len(EXPECTED_BUDGETS) * len(EXPECTED_SEEDS) * len(EXPECTED_SAMPLERS)
_CONFIRMABLE_FAILURES = frozenset(
    {"collision", "timeout", "near_miss", "comfort_violation", "incomplete"}
)
_REQUIRED_ROW_FIELDS = frozenset(
    {
        "sampler",
        "budget",
        "seed",
        "confirmation_status",
        "certified_failure_count",
        "confirmed_failure_count",
        "unconfirmed_certified_failure_count",
        "time_to_first_confirmed_failure_s",
        "time_to_first_confirmed_failure_censored",
        "simulator_seconds",
        "simulator_seconds_per_confirmed_failure",
        "evidence",
    }
)


@dataclass(frozen=True)
class PackageBConfirmationGate:
    """Validation result for one Package B confirmation sidecar."""

    report_path: str
    confirmation_path: str
    ready: bool
    status: str
    errors: tuple[str, ...]
    blockers: dict[str, tuple[str, ...]]
    matrix: dict[str, Any]
    next_empirical_action: str
    source_report_sha256: str | None

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON payload for review and downstream checks."""
        return {
            "schema_version": SCHEMA_VERSION,
            "issue": EXPECTED_ISSUE,
            "report_path": self.report_path,
            "confirmation_path": self.confirmation_path,
            "source_report_sha256": self.source_report_sha256,
            "ready": self.ready,
            "status": self.status,
            "errors": list(self.errors),
            "blockers": {key: list(value) for key, value in self.blockers.items()},
            "matrix": self.matrix,
            "next_empirical_action": self.next_empirical_action,
            "claim_boundary": (
                "diagnostic-only confirmation-chain validation; this gate does not promote "
                "benchmark, paper, or dissertation evidence"
            ),
        }


@dataclass(frozen=True)
class _StageValidationContext:
    """Shared context for one candidate's confirmation-stage checks."""

    prefix: str
    candidate_index: int
    report_path: Path
    artifact_root: Path | None
    errors: list[str]


def validate_package_b_confirmation(
    report_path: Path,
    confirmation_path: Path,
    *,
    artifact_root: Path | None = None,
) -> PackageBConfirmationGate:
    """Validate confirmed-failure evidence against one Package B report.

    The confirmation sidecar must account for every certified failure candidate
    in every report row. A candidate may be explicitly ``not_confirmed`` and is
    then excluded from confirmed-failure metrics; silently dropping it is a
    contract error. Confirmed candidates must reference existing replay,
    independent-seed, and attribution artifacts.

    Returns:
        A diagnostic-only gate result. ``ready`` means the sidecar is
        structurally complete and all referenced artifacts exist; it does not
        mean the underlying campaign result is benchmark evidence.
    """
    base_gate = validate_package_b_report(report_path)
    report_payload, report_errors = _load_mapping(report_path)
    confirmation_payload, confirmation_errors = _load_mapping(confirmation_path)
    errors = list(report_errors) + list(confirmation_errors)
    if not base_gate.ready:
        errors.append(f"base Package B report gate is not ready: {base_gate.status}")
        errors.extend(f"base report: {error}" for error in base_gate.errors)

    source_report_sha256 = _sha256(report_path)
    _validate_header(
        confirmation_payload,
        source_report_sha256=source_report_sha256,
        errors=errors,
    )

    base_rows = _index_rows(report_payload.get("rows"), label="base report", errors=errors)
    row_count, observed_keys, confirmed_count, unconfirmed_count, censored_rows = (
        _validate_confirmation_rows(
            confirmation_payload.get("rows"),
            base_rows=base_rows,
            report_path=report_path,
            artifact_root=artifact_root,
            errors=errors,
        )
    )

    expected_keys = {
        (sampler, budget, seed)
        for sampler in EXPECTED_SAMPLERS
        for budget in EXPECTED_BUDGETS
        for seed in EXPECTED_SEEDS
    }
    missing_keys = sorted(expected_keys - observed_keys)
    if missing_keys:
        errors.append(f"confirmation rows missing Package B matrix cells: {missing_keys}")
    if row_count != EXPECTED_ROW_COUNT:
        errors.append(f"confirmation rows must contain exactly {EXPECTED_ROW_COUNT} matrix cells")

    ready = not errors
    status = "ready_for_confirmed_failure_review" if ready else "blocked_on_confirmation_contract"
    return PackageBConfirmationGate(
        report_path=report_path.as_posix(),
        confirmation_path=confirmation_path.as_posix(),
        ready=ready,
        status=status,
        errors=tuple(errors),
        blockers={
            "remaining": (
                "The exact Package B campaign and durable artifact review remain external to "
                "this CPU-only checker.",
            ),
            "new": tuple(errors),
            "intentional": (
                "not_confirmed candidates are excluded from confirmed-failure counts",
                "held-out-family yield remains subject to the narrow certified-archive caveat",
                "learned failure proposal #2921 remains out of scope",
                "paper-facing and dissertation claims remain forbidden",
            ),
        },
        matrix={
            "expected_row_count": EXPECTED_ROW_COUNT,
            "observed_row_count": row_count,
            "expected_samplers": list(EXPECTED_SAMPLERS),
            "expected_budgets": list(EXPECTED_BUDGETS),
            "expected_seeds": list(EXPECTED_SEEDS),
            "confirmed_failure_count": confirmed_count,
            "unconfirmed_certified_failure_count": unconfirmed_count,
            "censored_row_count": censored_rows,
        },
        next_empirical_action=(
            "Run the exact manifest command on an approved compute-capable path, emit one "
            "confirmation row per report cell, then review the referenced artifacts before "
            "interpreting any sampler difference."
        ),
        source_report_sha256=source_report_sha256,
    )


def _validate_confirmation_rows(
    rows: Any,
    *,
    base_rows: dict[tuple[str, int, int], dict[str, Any]],
    report_path: Path,
    artifact_root: Path | None,
    errors: list[str],
) -> tuple[int, set[tuple[str, int, int]], int, int, int]:
    """Validate all sidecar rows and aggregate their derived counts.

    Returns:
        ``(row_count, observed_keys, confirmed_count, unconfirmed_count,
        censored_row_count)``.
    """
    if not isinstance(rows, list):
        errors.append("confirmation rows must be a list")
        return 0, set(), 0, 0, 0
    observed_keys: set[tuple[str, int, int]] = set()
    confirmed_count = 0
    unconfirmed_count = 0
    censored_rows = 0
    for row_index, row in enumerate(rows):
        result = _validate_confirmation_row(
            row,
            row_index=row_index,
            base_rows=base_rows,
            report_path=report_path,
            artifact_root=artifact_root,
            observed_keys=observed_keys,
            errors=errors,
        )
        if result is None:
            continue
        row_confirmed, row_unconfirmed, row_censored = result
        confirmed_count += row_confirmed
        unconfirmed_count += row_unconfirmed
        censored_rows += int(row_censored)
    return len(rows), observed_keys, confirmed_count, unconfirmed_count, censored_rows


def _validate_confirmation_row(
    row: Any,
    *,
    row_index: int,
    base_rows: dict[tuple[str, int, int], dict[str, Any]],
    report_path: Path,
    artifact_root: Path | None,
    observed_keys: set[tuple[str, int, int]],
    errors: list[str],
) -> tuple[int, int, bool] | None:
    """Validate sidecar row shape and delegate a complete row to the contract checker.

    Returns:
        Derived row counts, or ``None`` when the row cannot be associated with
        a source-report matrix cell.
    """
    if not isinstance(row, dict):
        errors.append(f"rows[{row_index}] must be a mapping")
        return None
    missing = sorted(_REQUIRED_ROW_FIELDS - row.keys())
    if missing:
        errors.append(f"rows[{row_index}] missing required fields: {missing}")
        return None
    key = _row_key(row)
    if key is None:
        errors.append(f"rows[{row_index}] has invalid sampler/budget/seed types")
        return None
    if key in observed_keys:
        errors.append(f"rows[{row_index}] duplicates matrix cell {key!r}")
        return None
    observed_keys.add(key)
    base_row = base_rows.get(key)
    if base_row is None:
        errors.append(f"rows[{row_index}] does not match a base report matrix cell: {key!r}")
        return None
    return _validate_row(
        row,
        base_row=base_row,
        row_index=row_index,
        report_path=report_path,
        artifact_root=artifact_root,
        errors=errors,
    )


def _validate_header(
    payload: dict[str, Any],
    *,
    source_report_sha256: str | None,
    errors: list[str],
) -> None:
    """Validate sidecar identity and its binding to the source report."""
    _check_equal(payload, "schema_version", SCHEMA_VERSION, errors)
    _check_equal(payload, "issue", EXPECTED_ISSUE, errors)
    _check_equal(payload, "source_report_schema_version", REPORT_SCHEMA_VERSION, errors)
    _check_equal(payload, "claim_scope", EXPECTED_CLAIM_SCOPE, errors)
    if source_report_sha256 is None:
        errors.append("source report could not be hashed")
    elif payload.get("source_report_sha256") != source_report_sha256:
        errors.append("source_report_sha256 does not match the supplied report")


def _validate_row(
    row: dict[str, Any],
    *,
    base_row: dict[str, Any],
    row_index: int,
    report_path: Path,
    artifact_root: Path | None,
    errors: list[str],
) -> tuple[int, int, bool]:
    """Validate one sidecar row and return its derived confirmation counts.

    Returns:
        ``(confirmed_count, unconfirmed_count, is_censored)`` for the row.
    """
    prefix = f"rows[{row_index}]"
    if row.get("confirmation_status") != "complete":
        errors.append(f"{prefix}.confirmation_status must equal 'complete'")

    certified_candidates = _load_certified_candidates(
        base_row=base_row,
        prefix=prefix,
        report_path=report_path,
        artifact_root=artifact_root,
        errors=errors,
    )
    expected_certified_count = len(certified_candidates)
    if expected_certified_count != base_row.get("certified_valid_failure_count"):
        errors.append(
            f"{prefix} manifest certified failure count {expected_certified_count} does not "
            "match base report"
        )
    _check_count(row, "certified_failure_count", expected_certified_count, prefix, errors)

    evidence_by_index = _index_evidence(row, prefix=prefix, errors=errors)
    expected_indices = {index for index, _candidate in certified_candidates}
    observed_indices = set(evidence_by_index)
    if observed_indices != expected_indices:
        errors.append(
            f"{prefix}.evidence must account for certified candidate indexes "
            f"{sorted(expected_indices)}; found {sorted(observed_indices)}"
        )

    confirmed, elapsed_values = _validate_evidence_entries(
        certified_candidates=certified_candidates,
        evidence_by_index=evidence_by_index,
        prefix=prefix,
        report_path=report_path,
        artifact_root=artifact_root,
        errors=errors,
    )

    _check_count(row, "confirmed_failure_count", confirmed, prefix, errors)
    unconfirmed = expected_certified_count - confirmed
    _check_count(row, "unconfirmed_certified_failure_count", unconfirmed, prefix, errors)
    _validate_row_metrics(
        row,
        confirmed=confirmed,
        elapsed_values=elapsed_values,
        prefix=prefix,
        errors=errors,
    )
    return confirmed, unconfirmed, not confirmed


def _load_certified_candidates(
    *,
    base_row: dict[str, Any],
    prefix: str,
    report_path: Path,
    artifact_root: Path | None,
    errors: list[str],
) -> list[tuple[int, Any]]:
    """Load one source manifest and select its certified valid failures.

    Returns:
        One-based candidate indexes paired with their manifest payloads.
    """
    manifest_path = _resolve_existing_path(
        base_row.get("manifest_path"),
        report_path=report_path,
        artifact_root=artifact_root,
    )
    if manifest_path is None:
        errors.append(f"{prefix} base manifest_path does not resolve to an existing file")
        return []
    manifest, manifest_errors = _load_mapping(manifest_path)
    errors.extend(f"{prefix} manifest: {error}" for error in manifest_errors)
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        errors.append(f"{prefix} manifest candidates must be a list")
        return []
    if _is_int(base_row.get("num_candidates")) and len(candidates) != base_row["num_candidates"]:
        errors.append(
            f"{prefix} manifest candidate count {len(candidates)} does not match "
            f"base report count {base_row['num_candidates']}"
        )
    return [
        (index, candidate)
        for index, candidate in enumerate(candidates, start=1)
        if _is_certified_valid_failure(candidate)
    ]


def _index_evidence(
    row: dict[str, Any],
    *,
    prefix: str,
    errors: list[str],
) -> dict[int, dict[str, Any]]:
    """Index sidecar evidence entries by their one-based candidate index.

    Returns:
        Validly indexed evidence entries. Malformed entries are omitted after
        recording a fail-closed error.
    """
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        errors.append(f"{prefix}.evidence must be a list")
        return {}
    indexed: dict[int, dict[str, Any]] = {}
    for evidence_index, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            errors.append(f"{prefix}.evidence[{evidence_index}] must be a mapping")
            continue
        candidate_index = entry.get("candidate_index")
        if not _is_positive_int(candidate_index):
            errors.append(f"{prefix}.evidence[{evidence_index}].candidate_index must be positive")
            continue
        if candidate_index in indexed:
            errors.append(f"{prefix}.evidence duplicates candidate index {candidate_index}")
            continue
        indexed[candidate_index] = entry
    return indexed


def _validate_evidence_entries(
    *,
    certified_candidates: list[tuple[int, Any]],
    evidence_by_index: dict[int, dict[str, Any]],
    prefix: str,
    report_path: Path,
    artifact_root: Path | None,
    errors: list[str],
) -> tuple[int, list[float]]:
    """Validate each accounted-for failure and return confirmed elapsed times.

    Returns:
        ``(confirmed_count, elapsed_seconds_for_confirmed_entries)``.
    """
    confirmed = 0
    elapsed_values: list[float] = []
    for candidate_index, candidate in certified_candidates:
        entry = evidence_by_index.get(candidate_index)
        if entry is None:
            continue
        status = entry.get("status")
        if status == "not_confirmed":
            if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
                errors.append(
                    f"{prefix}.evidence candidate {candidate_index} needs a not_confirmed reason"
                )
            continue
        if status != "confirmed":
            errors.append(
                f"{prefix}.evidence candidate {candidate_index}.status must be confirmed or "
                "not_confirmed"
            )
            continue
        confirmed += 1
        elapsed = _finite_number(entry.get("simulator_seconds_elapsed"))
        if elapsed is None or elapsed < 0.0:
            errors.append(
                f"{prefix}.evidence candidate {candidate_index}.simulator_seconds_elapsed "
                "must be a finite non-negative number"
            )
        else:
            elapsed_values.append(elapsed)
        _validate_confirmed_entry(
            entry,
            candidate=candidate,
            candidate_index=candidate_index,
            prefix=prefix,
            report_path=report_path,
            artifact_root=artifact_root,
            errors=errors,
        )
    return confirmed, elapsed_values


def _validate_row_metrics(
    row: dict[str, Any],
    *,
    confirmed: int,
    elapsed_values: list[float],
    prefix: str,
    errors: list[str],
) -> None:
    """Validate censored time-to-confirmation and simulator-time rates."""
    total_seconds = _validated_total_seconds(row, prefix=prefix, errors=errors)
    _validate_time_to_confirmation(
        row,
        confirmed=confirmed,
        elapsed_values=elapsed_values,
        total_seconds=total_seconds,
        prefix=prefix,
        errors=errors,
    )
    _validate_rate(
        row,
        confirmed=confirmed,
        total_seconds=total_seconds,
        prefix=prefix,
        errors=errors,
    )


def _validated_total_seconds(row: dict[str, Any], *, prefix: str, errors: list[str]) -> float:
    """Return the row's simulator-time total after a finite non-negative check."""
    total_seconds = _finite_number(row.get("simulator_seconds"))
    if total_seconds is not None and total_seconds >= 0.0:
        return total_seconds
    errors.append(f"{prefix}.simulator_seconds must be a finite non-negative number")
    return 0.0


def _validate_time_to_confirmation(
    row: dict[str, Any],
    *,
    confirmed: int,
    elapsed_values: list[float],
    total_seconds: float,
    prefix: str,
    errors: list[str],
) -> None:
    """Validate the first-confirmed time and its right-censoring flag."""
    if elapsed_values and max(elapsed_values) > total_seconds:
        errors.append(f"{prefix} confirmation elapsed time exceeds simulator_seconds")
    reported_time = row.get("time_to_first_confirmed_failure_s")
    censored = row.get("time_to_first_confirmed_failure_censored")
    if not isinstance(censored, bool):
        errors.append(f"{prefix}.time_to_first_confirmed_failure_censored must be boolean")
    if confirmed:
        expected_time = min(elapsed_values) if elapsed_values else None
        if expected_time is None or _finite_number(reported_time) is None:
            errors.append(f"{prefix} confirmed rows need a finite first-confirmed time")
        elif not math.isclose(float(reported_time), expected_time, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{prefix}.time_to_first_confirmed_failure_s disagrees with evidence")
        if censored is not False:
            errors.append(f"{prefix} confirmed rows must not be censored")
    else:
        if reported_time is not None:
            errors.append(f"{prefix}.time_to_first_confirmed_failure_s must be null when censored")
        if censored is not True:
            errors.append(f"{prefix} rows without confirmed failures must be censored")


def _validate_rate(
    row: dict[str, Any],
    *,
    confirmed: int,
    total_seconds: float,
    prefix: str,
    errors: list[str],
) -> None:
    """Validate simulator-seconds per confirmed failure against its denominator."""
    reported_rate = row.get("simulator_seconds_per_confirmed_failure")
    expected_rate = total_seconds / confirmed if confirmed else None
    if expected_rate is None:
        if reported_rate is not None:
            errors.append(
                f"{prefix}.simulator_seconds_per_confirmed_failure must be null without "
                "confirmed failures"
            )
    elif _finite_number(reported_rate) is None or not math.isclose(
        float(reported_rate), expected_rate, rel_tol=1e-9, abs_tol=1e-9
    ):
        errors.append(
            f"{prefix}.simulator_seconds_per_confirmed_failure disagrees with its denominator"
        )


def _validate_confirmed_entry(
    entry: dict[str, Any],
    *,
    candidate: Any,
    candidate_index: int,
    prefix: str,
    report_path: Path,
    artifact_root: Path | None,
    errors: list[str],
) -> None:
    """Validate the four-stage confirmation chain for one candidate."""
    candidate_payload = candidate.get("candidate") if isinstance(candidate, dict) else None
    candidate_seed = (
        candidate_payload.get("scenario_seed") if isinstance(candidate_payload, dict) else None
    )
    failure_attribution = (
        candidate.get("failure_attribution") if isinstance(candidate, dict) else None
    )
    primary_failure = (
        failure_attribution.get("primary_failure")
        if isinstance(failure_attribution, dict)
        else None
    )
    if entry.get("candidate_seed") != candidate_seed:
        errors.append(f"{prefix}.evidence candidate {candidate_index} candidate_seed drifted")
    if entry.get("primary_failure") != primary_failure:
        errors.append(f"{prefix}.evidence candidate {candidate_index} primary_failure drifted")

    context = _StageValidationContext(
        prefix=prefix,
        candidate_index=candidate_index,
        report_path=report_path,
        artifact_root=artifact_root,
        errors=errors,
    )
    _validate_stage(
        entry,
        stage_name="replay",
        expected_status="passed",
        candidate_seed=candidate_seed,
        primary_failure=primary_failure,
        context=context,
    )
    _validate_stage(
        entry,
        stage_name="independent_seed_confirmation",
        expected_status="passed",
        candidate_seed=candidate_seed,
        primary_failure=primary_failure,
        context=context,
    )
    _validate_stage(
        entry,
        stage_name="mechanism_attribution",
        expected_status="stable",
        candidate_seed=candidate_seed,
        primary_failure=primary_failure,
        context=context,
    )


def _validate_stage(
    entry: dict[str, Any],
    *,
    stage_name: str,
    expected_status: str,
    candidate_seed: Any,
    primary_failure: Any,
    context: _StageValidationContext,
) -> None:
    """Validate one confirmation stage and its artifact reference."""
    stage = _mapping_field(
        entry,
        stage_name,
        context.prefix,
        context.candidate_index,
        context.errors,
    )
    if stage is None:
        return
    field_context = f"{context.prefix}.evidence candidate {context.candidate_index}.{stage_name}"
    if stage.get("status") != expected_status:
        context.errors.append(f"{field_context}.status must equal {expected_status!r}")
    if stage_name == "replay" and stage.get("deterministic") is not True:
        context.errors.append(f"{field_context}.deterministic must be true")
    if stage_name == "independent_seed_confirmation":
        _validate_independent_fields(
            stage,
            candidate_seed=candidate_seed,
            context=field_context,
            errors=context.errors,
        )
    if stage_name == "mechanism_attribution" and stage.get("primary_failure") != primary_failure:
        context.errors.append(
            f"{field_context}.primary_failure does not match the candidate failure"
        )
    _require_artifact(
        stage.get("artifact_path"),
        field=f"{field_context}.artifact_path",
        report_path=context.report_path,
        artifact_root=context.artifact_root,
        errors=context.errors,
    )


def _validate_independent_fields(
    stage: dict[str, Any],
    *,
    candidate_seed: Any,
    context: str,
    errors: list[str],
) -> None:
    """Validate independent-seed identity and persistence-rate fields."""
    seeds = stage.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(not _is_int(seed) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        errors.append(f"{context}.seeds needs two distinct integer independent seeds")
    elif candidate_seed in seeds:
        errors.append(f"{context}.seeds must not include the search candidate seed")
    persistence = _finite_number(stage.get("failure_persistence_rate"))
    if persistence is None or not 0.0 < persistence <= 1.0:
        errors.append(f"{context}.failure_persistence_rate must be finite and in (0, 1]")


def _is_certified_valid_failure(candidate: Any) -> bool:
    """Return whether a manifest candidate is eligible for confirmation review."""
    if not isinstance(candidate, dict) or candidate.get("error") is not None:
        return False
    certification = candidate.get("certification_status")
    attribution = candidate.get("failure_attribution")
    return (
        isinstance(certification, dict)
        and certification.get("status") == "passed"
        and isinstance(attribution, dict)
        and attribution.get("primary_failure") in _CONFIRMABLE_FAILURES
    )


def _mapping_field(
    payload: dict[str, Any],
    field: str,
    prefix: str,
    candidate_index: int,
    errors: list[str],
) -> dict[str, Any] | None:
    """Read a required nested mapping and append a contextual error when absent.

    Returns:
        The nested mapping, or ``None`` after recording a contract error.
    """
    value = payload.get(field)
    if not isinstance(value, dict):
        errors.append(f"{prefix}.evidence candidate {candidate_index}.{field} must be a mapping")
        return None
    return value


def _require_artifact(
    raw_path: Any,
    *,
    field: str,
    report_path: Path,
    artifact_root: Path | None,
    errors: list[str],
) -> None:
    """Require an artifact path that resolves to an existing file."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        errors.append(f"{field} must be a non-empty artifact path")
        return
    if (
        _resolve_existing_path(raw_path, report_path=report_path, artifact_root=artifact_root)
        is None
    ):
        errors.append(f"{field} does not resolve to an existing file")


def _index_rows(
    rows: Any,
    *,
    label: str,
    errors: list[str],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Index report rows by matrix key without silently accepting duplicates.

    Returns:
        A mapping from matrix key to its last structurally indexable row.
    """
    if not isinstance(rows, list):
        errors.append(f"{label} rows must be a list")
        return {}
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{label} rows[{index}] must be a mapping")
            continue
        key = _row_key(row)
        if key is None:
            continue
        if key in indexed:
            errors.append(f"{label} rows[{index}] duplicates matrix cell {key!r}")
        indexed[key] = row
    return indexed


def _row_key(row: dict[str, Any]) -> tuple[str, int, int] | None:
    """Return a matrix key when sampler, budget, and seed have native types."""
    sampler = row.get("sampler")
    budget = row.get("budget")
    seed = row.get("seed")
    if not isinstance(sampler, str) or not _is_int(budget) or not _is_int(seed):
        return None
    return sampler, int(budget), int(seed)


def _resolve_existing_path(
    raw_path: Any,
    *,
    report_path: Path,
    artifact_root: Path | None,
) -> Path | None:
    """Resolve a report or artifact path against common post-run roots.

    Returns:
        The first existing file candidate, or ``None`` when no candidate exists.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    trusted_roots = _trusted_search_roots(report_path, artifact_root)
    candidates = [path] if path.is_absolute() else [root / path for root in trusted_roots]
    for candidate in candidates:
        resolved = _safe_existing_file(candidate, trusted_roots)
        if resolved is not None:
            return resolved
    return None


def _trusted_search_roots(report_path: Path, artifact_root: Path | None) -> tuple[Path, ...]:
    """Return resolved roots allowed for user-supplied artifact references."""
    roots: list[Path] = []
    for root in (artifact_root, Path.cwd(), report_path.parent):
        if root is None:
            continue
        try:
            resolved = root.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _safe_existing_file(candidate: Path, trusted_roots: tuple[Path, ...]) -> Path | None:
    """Resolve a candidate only when it is a regular non-symlink file in a trusted root.

    Returns:
        The resolved file path, or ``None`` when the candidate is unsafe/unavailable.
    """
    try:
        current = candidate
        while current != current.parent:
            if current.is_symlink():
                return None
            current = current.parent
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None
    if not any(resolved.is_relative_to(root) for root in trusted_roots):
        return None
    return resolved


def _load_mapping(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Load a JSON object and preserve parse failures as validation errors.

    Returns:
        ``(mapping, errors)`` where parse or shape failures are represented in
        the error list instead of raised to the caller.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"{path} could not be read as JSON: {exc}"]
    if not isinstance(payload, dict):
        return {}, [f"{path} JSON payload must be a mapping"]
    return payload, []


def _sha256(path: Path) -> str | None:
    """Return a file digest or ``None`` when the source is unavailable."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _check_equal(payload: dict[str, Any], field: str, expected: Any, errors: list[str]) -> None:
    """Append an error when a header field drifts."""
    if payload.get(field) != expected:
        errors.append(f"{field} must equal {expected!r}; found {payload.get(field)!r}")


def _check_count(
    payload: dict[str, Any],
    field: str,
    expected: int,
    prefix: str,
    errors: list[str],
) -> None:
    """Append an error when a derived count is not represented exactly."""
    value = payload.get(field)
    if not _is_int(value) or int(value) < 0:
        errors.append(f"{prefix}.{field} must be a non-negative integer")
    elif value != expected:
        errors.append(f"{prefix}.{field} must equal {expected}; found {value}")


def _is_int(value: Any) -> bool:
    """Return whether a value is a native integer rather than a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value: Any) -> bool:
    """Return whether a value is a native positive integer."""
    return _is_int(value) and value > 0


def _finite_number(value: Any) -> float | None:
    """Return a finite number while rejecting booleans and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class PackageBConfirmationSidecar:
    """Producer for one Package-B confirmation sidecar bound to a source report."""

    source_report_path: Path
    confirmation_path: Path
    confirmation_rows: tuple[dict[str, Any], ...]
    source_report_sha256: str | None

    def to_payload(self) -> dict[str, Any]:
        """Return the durable sidecar JSON payload."""
        return {
            "schema_version": SCHEMA_VERSION,
            "issue": EXPECTED_ISSUE,
            "source_report_schema_version": REPORT_SCHEMA_VERSION,
            "claim_scope": EXPECTED_CLAIM_SCOPE,
            "source_report_sha256": self.source_report_sha256,
            "rows": [dict(row) for row in self.confirmation_rows],
        }

    def write(self) -> Path:
        """Write the sidecar JSON to ``confirmation_path``.

        Returns:
            The confirmation artifact path.
        """
        self.confirmation_path.parent.mkdir(parents=True, exist_ok=True)
        self.confirmation_path.write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.confirmation_path


def build_package_b_confirmation_sidecar(
    report_path: Path,
    *,
    confirmation_path: Path,
) -> PackageBConfirmationSidecar:
    """Build a confirmation sidecar from one Package-B report.

    The sidecar conservatively marks every certified failure as ``not_confirmed``:
    no row yet carries replay, independent-seed, or mechanism-attribution artifacts,
    so ``time_to_first_confirmed_failure`` is reported as censored (null) exactly as
    issue #3079 requires. The sidecar is structurally complete (27 cells, sha-bound
    to the source report) and can be validated by ``validate_package_b_confirmation``.

    Returns:
        A sidecar whose rows are account for every report cell without overclaiming.
    """
    source_report_sha256 = _sha256(report_path)
    report_payload = _load_mapping(report_path)[0]
    rows = report_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("source Package-B report rows must be a list")

    base_rows = _index_rows(rows, label="base report", errors=[])
    sidecar_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _row_key(row)
        if key is None:
            continue
        base_row = base_rows.get(key)
        if base_row is None:
            continue
        certified = int(base_row.get("certified_valid_failure_count") or 0)
        confirmed = 0
        is_censored = confirmed == 0
        sidecar_rows.append(
            {
                "sampler": row.get("sampler"),
                "budget": row.get("budget"),
                "seed": row.get("seed"),
                "confirmation_status": "complete",
                "certified_failure_count": certified,
                "confirmed_failure_count": confirmed,
                "unconfirmed_certified_failure_count": certified - confirmed,
                "time_to_first_confirmed_failure_s": None,
                "time_to_first_confirmed_failure_censored": is_censored,
                "simulator_seconds": 0.0,
                "simulator_seconds_per_confirmed_failure": None,
                "evidence": [
                    {
                        "candidate_index": index,
                        "status": "not_confirmed",
                        "reason": (
                            "awaiting deterministic replay, independent-seed confirmation, "
                            "and stable mechanism attribution before a discovery is counted"
                        ),
                    }
                    for index in _certified_candidate_indexes(
                        base_row=base_row,
                        report_path=report_path,
                    )
                ],
            }
        )
    return PackageBConfirmationSidecar(
        source_report_path=report_path,
        confirmation_path=confirmation_path,
        confirmation_rows=tuple(sidecar_rows),
        source_report_sha256=source_report_sha256,
    )


def _certified_candidate_indexes(
    *,
    base_row: dict[str, Any],
    report_path: Path,
) -> list[int]:
    """Return one-based manifest indexes of certified valid failures for one row.

    Returns:
        The certified candidate indexes, matching the contract the confirmation
        gate enforces against the source manifest.
    """
    manifest_path = _resolve_existing_path(
        base_row.get("manifest_path"),
        report_path=report_path,
        artifact_root=None,
    )
    if manifest_path is None:
        return []
    manifest = _load_mapping(manifest_path)[0]
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else None
    if not isinstance(candidates, list):
        return []
    return [
        index
        for index, candidate in enumerate(candidates, start=1)
        if _is_certified_valid_failure(candidate)
    ]
