"""Benchmark row claim validation for static leaderboards and summary tables.

A row claim is the compact, schema-checked boundary between a single leaderboard
row and the durable evidence it is allowed to reference.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.errors import RobotSfError

BENCHMARK_ROW_CLAIM_SCHEMA_VERSION = "benchmark_row_claim.v1"
BENCHMARK_ROW_CLAIM_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/benchmark_row_claim.v1.json")

#: Row statuses that are allowed to carry benchmark-strength success wording.
_SUCCESS_STATUSES = {"successful_evidence", "pass"}
#: Row statuses that explicitly denote non-success or unavailable evidence.
_NON_SUCCESS_STATUSES = {
    "accepted_unavailable",
    "unexpected_failure",
    "fallback",
    "degraded",
    "blocked",
    "excluded",
    "revise",
    "completed_smoke_not_benchmark_evidence",
    "not_yet_populated",
}
#: Planner modes that cannot be promoted as successful benchmark evidence.
_NON_SUCCESS_MODES = {"fallback", "degraded", "not_available"}
#: Evidence tiers ordered from weakest to strongest.
_EVIDENCE_TIER_ORDER = ["smoke", "diagnostic", "benchmark", "paper_facing"]
#: Wording markers that are only acceptable for paper-facing rows with success status.
_PAPER_ONLY_SUPERLATIVES = {
    "outperforms",
    "outperformed",
    "superior",
    "superiority",
    "state-of-the-art",
    "state of the art",
    "sota",
    "best",
    "better than",
    "improves upon",
}
_NON_SUCCESS_WORDING = {
    "prove",
    "proves",
    "proven",
    "proof",
    "benchmark readiness",
    "successful evidence",
}


class BenchmarkRowClaimError(RobotSfError, ValueError):
    """Raised when a leaderboard row claim violates the v1 contract."""


@functools.lru_cache(maxsize=1)
def load_benchmark_row_claim_schema() -> dict[str, Any]:
    """Load the benchmark row claim v1 JSON schema.

    Returns:
        Parsed benchmark row claim JSON schema.
    """
    schema_path = get_repository_root() / BENCHMARK_ROW_CLAIM_SCHEMA_PATH
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkRowClaimError("benchmark_row_claim schema must be a JSON object")
    return payload


def _validate_schema(claim: dict[str, Any]) -> None:
    """Validate a row claim payload against the v1 JSON schema."""
    try:
        Draft202012Validator(load_benchmark_row_claim_schema()).validate(claim)
    except JsonSchemaValidationError as exc:
        raise BenchmarkRowClaimError(f"schema validation failed: {exc.message}") from exc


def _validate_artifact_uri(artifact_uri: str, repo_root: Path) -> Path:
    """Fail closed on worktree-local output/ paths and missing artifacts.

    Returns:
        Resolved artifact path.
    """
    normalized = artifact_uri.strip()
    lower = normalized.lower()
    if lower.startswith("output/") or lower.startswith("output\\"):
        raise BenchmarkRowClaimError(
            f"artifact_uri must not point at worktree-local output/: {artifact_uri}"
        )
    if Path(normalized).is_absolute():
        raise BenchmarkRowClaimError(
            f"artifact_uri must be repository-relative, not absolute: {artifact_uri}"
        )
    root = repo_root.resolve()
    unresolved = root / normalized
    if unresolved.is_symlink():
        raise BenchmarkRowClaimError(f"artifact_uri must not be a symlink: {artifact_uri}")
    resolved = unresolved.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise BenchmarkRowClaimError(
            f"artifact_uri must stay within the repository: {artifact_uri}"
        ) from exc
    if relative.parts and relative.parts[0].lower() == "output":
        raise BenchmarkRowClaimError(
            f"artifact_uri must not resolve under worktree-local output/: {artifact_uri}"
        )
    if not resolved.exists():
        raise BenchmarkRowClaimError(f"artifact_uri not found: {artifact_uri}")
    if not resolved.is_file():
        raise BenchmarkRowClaimError(f"artifact_uri must point to a file: {artifact_uri}")
    return resolved


def _normalize(value: str) -> str:
    """Return a lowercase, punctuation-stripped string for wording checks."""
    return re.sub(r"[^a-z0-9\s-]", "", value.lower())


def _validate_claim_wording(claim: dict[str, Any]) -> None:
    """Reject deterministic wording that overshoots the evidence tier or status."""
    wording = str(claim.get("claim_wording", "")).strip()
    if not wording:
        raise BenchmarkRowClaimError("claim_wording must be a non-empty string")

    row_status = str(claim.get("row_status", ""))
    planner_mode = str(claim.get("planner_mode", ""))
    evidence_tier = str(claim.get("evidence_tier", "diagnostic"))
    normalized = _normalize(wording)

    if evidence_tier not in _EVIDENCE_TIER_ORDER:
        raise BenchmarkRowClaimError(f"evidence_tier must be one of {_EVIDENCE_TIER_ORDER}")

    # Fallback/degraded/not_available mode must not be described as successful evidence.
    if planner_mode in _NON_SUCCESS_MODES and row_status in _SUCCESS_STATUSES:
        raise BenchmarkRowClaimError(
            f"planner_mode {planner_mode} cannot have row_status {row_status}: {wording!r}"
        )

    # Superlatives and ranking claims are only allowed for paper-facing successful rows.
    if any(marker in normalized for marker in _PAPER_ONLY_SUPERLATIVES):
        if evidence_tier != "paper_facing" or row_status not in _SUCCESS_STATUSES:
            raise BenchmarkRowClaimError(
                f"ranking/superlative wording requires paper_facing tier and success status: {wording!r}"
            )

    if row_status in _NON_SUCCESS_STATUSES and any(
        marker in normalized for marker in _NON_SUCCESS_WORDING
    ):
        raise BenchmarkRowClaimError(
            f"row_status {row_status} must not carry proof/success wording: {wording!r}"
        )

    # Benchmark-strength success wording requires benchmark or paper_facing tier.
    if row_status in _SUCCESS_STATUSES and evidence_tier in {"smoke", "diagnostic"}:
        if re.search(
            r"\b(benchmark|ranking|performance)\s+(result|claim|evidence|conclusion)\b", normalized
        ):
            raise BenchmarkRowClaimError(
                f"success status with benchmark-ranking wording requires benchmark/paper_facing tier: {wording!r}"
            )


def validate_benchmark_row_claim(
    claim: dict[str, Any],
    *,
    check_artifact_exists: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a single benchmark row claim record.

    Args:
        claim: Row claim payload.
        check_artifact_exists: When True, fail closed if ``artifact_uri`` does not exist.
        repo_root: Repository root for resolving relative artifact URIs.

    Returns:
        Normalized claim payload.

    Raises:
        BenchmarkRowClaimError: When the claim violates the v1 contract.
    """
    _validate_schema(claim)
    _validate_claim_wording(claim)

    if check_artifact_exists:
        root = repo_root or get_repository_root()
        _validate_artifact_uri(str(claim["artifact_uri"]), root)

    return claim


def load_leaderboard_claims(path: Path) -> list[dict[str, Any]]:
    """Load a leaderboard sidecar claim file.

    The sidecar must be a JSON object containing a ``rows`` array of claim records.

    Returns:
        List of row claim payloads.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkRowClaimError(f"leaderboard claims must be a JSON object: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise BenchmarkRowClaimError(f"leaderboard claims must contain a 'rows' list: {path}")
    return rows


def _strip_markdown_cell(value: str) -> str:
    """Normalize a Markdown table cell for sidecar comparison.

    Returns:
        Cell text with lightweight Markdown code delimiters removed.
    """
    return value.strip().strip("`").replace("`", "")


def _markdown_rows_for_sidecar(path: Path) -> list[dict[str, str]]:
    """Read visible leaderboard Markdown rows that correspond to a sidecar.

    Returns:
        Parsed visible row descriptors. Missing Markdown files return an empty list so fixture
        sidecars outside ``docs/leaderboards`` can still be validated directly.
    """
    if path.name.endswith(".rows.json"):
        markdown_path = path.with_name(f"{path.name.removesuffix('.rows.json')}.md")
    else:
        markdown_path = path.with_suffix(".md")
    if not markdown_path.exists():
        return []

    rows: list[dict[str, str]] = []
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 12:
            continue
        first_cell = cells[0].lower()
        second_cell = cells[1].lower()
        if first_cell == "planner" and second_cell == "suite":
            continue
        if set(first_cell) <= {"-", ":"} and set(second_cell) <= {"-", ":"}:
            continue
        rows.append(
            {
                "planner_id": _strip_markdown_cell(cells[0]),
                "suite_id": _strip_markdown_cell(cells[1]),
                "claim_wording": _strip_markdown_cell(cells[-1]),
            }
        )
    return rows


def validate_leaderboard_claims(
    path: Path,
    *,
    check_artifact_exists: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate all row claims in a leaderboard sidecar file.

    Returns:
        Validation report payload.
    """
    root = repo_root or get_repository_root()
    rows = load_leaderboard_claims(path)
    errors: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0

    markdown_rows = _markdown_rows_for_sidecar(path)
    if markdown_rows:
        if len(markdown_rows) != len(rows):
            errors.append(
                {
                    "index": None,
                    "planner_id": None,
                    "error": (
                        f"Markdown row count {len(markdown_rows)} does not match sidecar row "
                        f"count {len(rows)}"
                    ),
                }
            )
        for index, (markdown_row, row) in enumerate(zip(markdown_rows, rows, strict=False)):
            expected = {
                "planner_id": str(row.get("planner_id", "")),
                "suite_id": str(row.get("suite_id", "")),
                "claim_wording": str(row.get("claim_wording", "")),
            }
            for field_name, expected_value in expected.items():
                if markdown_row[field_name] != expected_value:
                    errors.append(
                        {
                            "index": index,
                            "planner_id": row.get("planner_id"),
                            "error": (
                                f"Markdown {field_name} {markdown_row[field_name]!r} does not "
                                f"match sidecar {expected_value!r}"
                            ),
                        }
                    )

    for index, row in enumerate(rows):
        try:
            validate_benchmark_row_claim(
                row, check_artifact_exists=check_artifact_exists, repo_root=root
            )
            accepted += 1
        except BenchmarkRowClaimError as exc:
            rejected += 1
            errors.append({"index": index, "planner_id": row.get("planner_id"), "error": str(exc)})

    return {
        "schema_version": "benchmark_row_claim_validation.v1",
        "leaderboard": path.as_posix(),
        "accepted": accepted,
        "rejected": rejected,
        "total": len(rows),
        "valid": rejected == 0 and not errors,
        "errors": errors,
    }


def validate_all_leaderboards(
    leaderboards_dir: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate every ``*.rows.json`` sidecar under the leaderboards directory.

    Returns:
        Aggregate validation report.
    """
    root = repo_root or get_repository_root()
    directory = leaderboards_dir or (root / "docs" / "leaderboards")
    if not directory.exists():
        raise BenchmarkRowClaimError(f"leaderboards directory not found: {directory}")

    reports: list[dict[str, Any]] = []
    overall_valid = True
    for sidecar in sorted(directory.glob("*.rows.json")):
        report = validate_leaderboard_claims(sidecar, repo_root=root)
        reports.append(report)
        if not report["valid"]:
            overall_valid = False

    return {
        "schema_version": "benchmark_row_claim_validation.v1",
        "overall_valid": overall_valid,
        "leaderboards_checked": len(reports),
        "reports": reports,
    }


__all__ = [
    "BENCHMARK_ROW_CLAIM_SCHEMA_PATH",
    "BENCHMARK_ROW_CLAIM_SCHEMA_VERSION",
    "BenchmarkRowClaimError",
    "load_benchmark_row_claim_schema",
    "load_leaderboard_claims",
    "validate_all_leaderboards",
    "validate_benchmark_row_claim",
    "validate_leaderboard_claims",
]
