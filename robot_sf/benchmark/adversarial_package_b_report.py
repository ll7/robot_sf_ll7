"""Fail-closed validation for issue #3079 Package B comparison reports."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from robot_sf.benchmark.adversarial_package_b_preflight import (
    EXPECTED_BUDGETS,
    EXPECTED_OBJECTIVE,
    EXPECTED_REPORTING_FIELDS,
    EXPECTED_SAMPLERS,
)

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
RAW_ARTIFACT_METADATA_FILENAME = "raw_artifact_bundle.json"
RAW_ARTIFACT_SCHEMA_VERSION = "package-b-raw-artifact-bundle.v1"


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


@dataclass(frozen=True)
class PackageBCandidateReplayVerificationResult:
    """Verification result for Package B candidate/replay artifact inventory."""

    inventory_path: str
    total_entries: int
    verified_entries: int
    missing_entries: int
    mismatched_entries: int
    is_valid: bool
    errors: tuple[str, ...]
    claim_boundary: str

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON representation of the verification result."""
        return {
            "inventory_path": self.inventory_path,
            "total_entries": self.total_entries,
            "verified_entries": self.verified_entries,
            "missing_entries": self.missing_entries,
            "mismatched_entries": self.mismatched_entries,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PackageBRawArtifactRetrievalResult:
    """Result of retrieving and unpacking a Package B raw-artifact archive."""

    metadata_path: str
    archive_uri: str | None
    archive_sha256: str | None
    raw_tree_dir: str | None
    verified_log_entries: int
    is_valid: bool
    errors: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON representation of the retrieval result."""
        return {
            "metadata_path": self.metadata_path,
            "archive_uri": self.archive_uri,
            "archive_sha256": self.archive_sha256,
            "raw_tree_dir": self.raw_tree_dir,
            "verified_log_entries": self.verified_log_entries,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
        }


def _is_sha256_digest(value: Any) -> bool:
    """Return whether *value* is a 64-character hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _portable_relative_path(value: Any, *, field: str) -> tuple[Path | None, str | None]:
    """Parse one strictly portable, traversal-free POSIX relative path.

    Returns:
        The parsed path and no error, or no path and an explanatory error.
    """
    if not isinstance(value, str) or not value:
        return None, f"{field} must be a non-empty relative path"
    if "\\" in value:
        return None, f"{field} must use POSIX separators: {value!r}"
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        return None, f"{field} must not contain traversal components: {value!r}"
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        return None, f"{field} must be a non-absolute relative path: {value!r}"
    return Path(*path.parts), None


def _parse_inventory_entry(
    line_no: int,
    line: str,
    seen_paths: set[str],
) -> tuple[tuple[str, Path] | None, str | None]:
    digest, sep, name = line.partition("  ")
    name = name.strip()
    if not sep or not _is_sha256_digest(digest):
        return None, f"line {line_no}: malformed SHA-256 digest or separator: {line!r}"
    rel_path, path_error = _portable_relative_path(name, field=f"line {line_no} path")
    if path_error is not None or rel_path is None:
        return None, path_error
    if "output" in rel_path.parts:
        return None, f"line {line_no}: ignored output/ path prohibited: {name}"
    if rel_path.parts[0] != "worst_case_snqi":
        return None, f"line {line_no}: path must start with worst_case_snqi/: {name}"
    posix_name = rel_path.as_posix()
    if posix_name in seen_paths:
        return None, f"line {line_no}: duplicate path entry: {name}"
    seen_paths.add(posix_name)
    return (digest.lower(), rel_path), None


def _verify_raw_tree_bytes(
    parsed_entries: list[tuple[str, Path]],
    raw_root: Path,
) -> tuple[int, int, int, list[str]]:
    errors: list[str] = []
    verified_entries = 0
    missing_entries = 0
    mismatched_entries = 0

    if not raw_root.is_dir():
        errors.append(f"raw_tree_dir directory not found: {raw_root}")
        return 0, 0, 0, errors

    try:
        resolved_root = raw_root.resolve(strict=True)
    except OSError as exc:
        errors.append(f"could not resolve raw_tree_dir {raw_root}: {exc}")
        return 0, 0, 0, errors

    for digest, rel_path in parsed_entries:
        target_file = raw_root / rel_path
        try:
            resolved_target = target_file.resolve(strict=False)
        except OSError as exc:
            missing_entries += 1
            errors.append(f"could not resolve raw artifact file {rel_path}: {exc}")
            continue
        if not resolved_target.is_relative_to(resolved_root):
            missing_entries += 1
            errors.append(f"raw artifact path escapes raw_tree_dir: {rel_path}")
            continue
        if not resolved_target.is_file():
            missing_entries += 1
            errors.append(f"missing raw artifact file: {rel_path}")
            continue
        actual_digest = hashlib.sha256(resolved_target.read_bytes()).hexdigest()
        if actual_digest != digest:
            mismatched_entries += 1
            errors.append(
                f"SHA-256 mismatch for {rel_path}: expected {digest}, got {actual_digest}"
            )
        else:
            verified_entries += 1

    return verified_entries, missing_entries, mismatched_entries, errors


def _read_raw_artifact_metadata(  # noqa: C901, PLR0912
    metadata_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Read and validate the portable Package B raw-artifact retrieval metadata.

    Returns:
        Normalized retrieval metadata and no errors, or no metadata and validation errors.
    """
    if not metadata_path.is_file():
        return None, [f"raw-artifact metadata not found: {metadata_path}"]
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"could not parse raw-artifact metadata {metadata_path}: {exc}"]
    if not isinstance(payload, dict):
        return None, ["raw-artifact metadata must be a JSON object"]
    if payload.get("schema_version") != RAW_ARTIFACT_SCHEMA_VERSION:
        return None, [
            f"raw-artifact metadata schema_version must equal {RAW_ARTIFACT_SCHEMA_VERSION!r}"
        ]

    archive = payload.get("archive")
    if not isinstance(archive, dict):
        return None, ["raw-artifact metadata.archive must be an object"]
    archive_uri = archive.get("uri")
    parsed_uri = urlsplit(archive_uri) if isinstance(archive_uri, str) else None
    if parsed_uri is None or parsed_uri.scheme not in {"file", "https"}:
        return None, ["raw-artifact metadata.archive.uri must use file:// or https://"]
    if not _is_sha256_digest(archive.get("sha256")):
        return None, ["raw-artifact metadata.archive.sha256 must be a SHA-256 digest"]
    if archive.get("format") != "tar.gz":
        return None, ["raw-artifact metadata.archive.format must equal 'tar.gz'"]

    archive_root, root_error = _portable_relative_path(
        payload.get("archive_root"), field="raw-artifact metadata.archive_root"
    )
    raw_tree_path, raw_tree_error = _portable_relative_path(
        payload.get("raw_tree_path"), field="raw-artifact metadata.raw_tree_path"
    )
    if root_error is not None:
        return None, [root_error]
    if raw_tree_error is not None:
        return None, [raw_tree_error]
    if raw_tree_path != Path("worst_case_snqi"):
        return None, ["raw-artifact metadata.raw_tree_path must equal 'worst_case_snqi'"]

    raw_logs = payload.get("logs")
    if not isinstance(raw_logs, list):
        return None, ["raw-artifact metadata.logs must be a list"]
    logs: list[dict[str, str]] = []
    streams: set[str] = set()
    for index, raw_log in enumerate(raw_logs):
        if not isinstance(raw_log, dict):
            return None, [f"raw-artifact metadata.logs[{index}] must be an object"]
        stream = raw_log.get("stream")
        log_path, log_error = _portable_relative_path(
            raw_log.get("path"), field=f"raw-artifact metadata.logs[{index}].path"
        )
        if stream not in {"stdout", "stderr"} or stream in streams:
            return None, ["raw-artifact metadata.logs must register one stdout and one stderr log"]
        if log_error is not None or log_path is None:
            return None, [log_error or f"raw-artifact metadata.logs[{index}].path is invalid"]
        if not _is_sha256_digest(raw_log.get("sha256")):
            return None, [f"raw-artifact metadata.logs[{index}].sha256 must be a SHA-256 digest"]
        streams.add(stream)
        logs.append(
            {
                "stream": stream,
                "path": log_path.as_posix(),
                "sha256": str(raw_log["sha256"]).lower(),
            }
        )
    if streams != {"stdout", "stderr"}:
        return None, ["raw-artifact metadata.logs must register one stdout and one stderr log"]

    return {
        "archive_uri": archive_uri,
        "archive_sha256": str(archive["sha256"]).lower(),
        "archive_root": archive_root,
        "raw_tree_path": raw_tree_path,
        "logs": logs,
    }, []


def _extract_tar_gz_safely(archive_path: Path, destination: Path) -> list[str]:  # noqa: C901
    """Extract regular archive members only, rejecting traversal and link members.

    Returns:
        A list of extraction errors. An empty list means extraction completed safely.
    """
    errors: list[str] = []
    try:
        resolved_destination = destination.resolve(strict=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            members: list[tuple[tarfile.TarInfo, Path]] = []
            seen_names: set[str] = set()
            for member in archive.getmembers():
                member_path, path_error = _portable_relative_path(
                    member.name, field="raw-artifact archive member"
                )
                if path_error is not None or member_path is None:
                    errors.append(path_error or "invalid raw-artifact archive member")
                    continue
                if member_path.as_posix() in seen_names:
                    errors.append(f"duplicate raw-artifact archive member: {member.name}")
                    continue
                seen_names.add(member_path.as_posix())
                if not (member.isdir() or member.isfile()):
                    errors.append(f"unsupported raw-artifact archive member type: {member.name}")
                    continue
                target = destination / member_path
                if not target.resolve(strict=False).is_relative_to(resolved_destination):
                    errors.append(f"raw-artifact archive member escapes destination: {member.name}")
                    continue
                members.append((member, target))
            if errors:
                return errors
            for member, target in members:
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    errors.append(f"could not read raw-artifact archive member: {member.name}")
                    continue
                with source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"could not extract raw-artifact archive {archive_path}: {exc}")
    return errors


def retrieve_package_b_raw_artifacts(  # noqa: C901
    bundle_dir: Path | str,
    destination: Path | str,
    *,
    metadata_path: Path | str | None = None,
) -> PackageBRawArtifactRetrievalResult:
    """Retrieve, checksum, safely extract, and log-verify the Package B raw archive.

    The committed metadata must pin an HTTPS (or test-only local ``file://``) archive, its
    SHA-256, archive layout, and SHA-256 digests for both process streams. Any absent, changed,
    malformed, or unsafe input is reported as invalid without extracting untrusted bytes.

    Returns:
        Retrieval status including the extracted raw-tree path only when all archive and log
        checks pass.
    """
    bundle_path = Path(bundle_dir)
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else bundle_path / RAW_ARTIFACT_METADATA_FILENAME
    )
    metadata, errors = _read_raw_artifact_metadata(resolved_metadata_path)
    if metadata is None:
        return PackageBRawArtifactRetrievalResult(
            metadata_path=str(resolved_metadata_path),
            archive_uri=None,
            archive_sha256=None,
            raw_tree_dir=None,
            verified_log_entries=0,
            is_valid=False,
            errors=tuple(errors),
        )

    archive_uri = str(metadata["archive_uri"])
    archive_sha256 = str(metadata["archive_sha256"])
    destination_path = Path(destination)
    try:
        destination_path.mkdir(parents=True, exist_ok=True)
        if any(destination_path.iterdir()):
            errors.append(f"raw-artifact retrieval destination must be empty: {destination_path}")
    except OSError as exc:
        errors.append(
            f"could not create raw-artifact retrieval destination {destination_path}: {exc}"
        )
    if errors:
        return PackageBRawArtifactRetrievalResult(
            metadata_path=str(resolved_metadata_path),
            archive_uri=archive_uri,
            archive_sha256=archive_sha256,
            raw_tree_dir=None,
            verified_log_entries=0,
            is_valid=False,
            errors=tuple(errors),
        )

    archive_path = destination_path / "package_b_raw_artifacts.tar.gz"
    try:
        with urlopen(archive_uri, timeout=60) as response, archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (OSError, URLError, ValueError) as exc:
        errors.append(f"could not retrieve raw-artifact archive {archive_uri}: {exc}")
    if not errors:
        actual_archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual_archive_sha256 != archive_sha256:
            errors.append(
                "raw-artifact archive SHA-256 mismatch: "
                f"expected {archive_sha256}, got {actual_archive_sha256}"
            )
    extraction_path = destination_path / "extracted"
    if not errors:
        extraction_path.mkdir()
        errors.extend(_extract_tar_gz_safely(archive_path, extraction_path))
    archive_root = metadata["archive_root"]
    raw_tree_path = metadata["raw_tree_path"]
    raw_artifact_root = extraction_path / archive_root
    raw_tree_dir = raw_artifact_root / raw_tree_path
    verified_logs = 0
    if not errors:
        if not raw_tree_dir.is_dir():
            errors.append(f"retrieved raw candidate/replay tree not found: {raw_tree_dir}")
        for log in metadata["logs"]:
            log_path = extraction_path / archive_root / Path(log["path"])
            if not log_path.is_file():
                errors.append(f"retrieved {log['stream']} log not found: {log['path']}")
                continue
            actual_log_sha256 = hashlib.sha256(log_path.read_bytes()).hexdigest()
            if actual_log_sha256 != log["sha256"]:
                errors.append(
                    f"retrieved {log['stream']} log SHA-256 mismatch: expected {log['sha256']}, "
                    f"got {actual_log_sha256}"
                )
                continue
            verified_logs += 1

    return PackageBRawArtifactRetrievalResult(
        metadata_path=str(resolved_metadata_path),
        archive_uri=archive_uri,
        archive_sha256=archive_sha256,
        raw_tree_dir=str(raw_artifact_root) if raw_tree_dir.is_dir() else None,
        verified_log_entries=verified_logs,
        is_valid=not errors,
        errors=tuple(errors),
    )


def verify_package_b_candidate_replay_inventory(
    bundle_dir: Path | str,
    raw_tree_dir: Path | str | None = None,
) -> PackageBCandidateReplayVerificationResult:
    """Verify Package B raw candidate/replay SHA-256 inventory against portable rules and disk bytes.

    Validates that:
    1. candidate_replay_SHA256SUMS.txt exists and contains 4,761 unique, portable entries.
    2. Every digest is a 64-character SHA-256 hex string.
    3. Every path is relative, starts with worst_case_snqi/, and contains no 'output' or absolute components.
    4. Requires ``raw_tree_dir`` and computes actual SHA-256 digests for all 4,761 files.

    Returns:
        A structured result object with counts, validity, errors, and claim boundary.
    """
    bundle_path = Path(bundle_dir)
    inventory_path = (
        bundle_path if bundle_path.is_file() else bundle_path / "candidate_replay_SHA256SUMS.txt"
    )

    errors: list[str] = []
    if not inventory_path.is_file():
        errors.append(f"inventory manifest not found: {inventory_path}")
        return PackageBCandidateReplayVerificationResult(
            inventory_path=str(inventory_path),
            total_entries=0,
            verified_entries=0,
            missing_entries=0,
            mismatched_entries=0,
            is_valid=False,
            errors=tuple(errors),
            claim_boundary=(
                "diagnostic-only candidate/replay inventory verification; "
                "not paper-facing benchmark evidence"
            ),
        )

    lines = inventory_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 4761:
        errors.append(f"expected 4761 inventory entries, found {len(lines)}")

    seen_paths: set[str] = set()
    parsed_entries: list[tuple[str, Path]] = []
    for line_no, line in enumerate(lines, start=1):
        entry, err = _parse_inventory_entry(line_no, line, seen_paths)
        if err:
            errors.append(err)
        elif entry:
            parsed_entries.append(entry)

    verified_entries = 0
    missing_entries = 0
    mismatched_entries = 0

    if raw_tree_dir is not None:
        v_entries, m_entries, mm_entries, disk_errs = _verify_raw_tree_bytes(
            parsed_entries, Path(raw_tree_dir)
        )
        verified_entries = v_entries
        missing_entries = m_entries
        mismatched_entries = mm_entries
        errors.extend(disk_errs)
    else:
        errors.append(
            "raw_tree_dir is required for byte verification; retrieve the pinned archive or "
            "supply --raw-tree-dir"
        )

    return PackageBCandidateReplayVerificationResult(
        inventory_path=str(inventory_path),
        total_entries=len(lines),
        verified_entries=verified_entries,
        missing_entries=missing_entries,
        mismatched_entries=mismatched_entries,
        is_valid=(len(errors) == 0),
        errors=tuple(errors),
        claim_boundary=(
            "diagnostic-only candidate/replay inventory verification; "
            "not paper-facing benchmark evidence"
        ),
    )
