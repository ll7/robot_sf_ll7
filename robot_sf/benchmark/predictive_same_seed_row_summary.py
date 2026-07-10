"""Validation helpers for predictive same-seed comparison outcome rows."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.errors import RobotSfError
from robot_sf.planner.obstacle_features import (
    ObstacleFeatureSchemaError,
    predictive_ego_motion_channel_producer_key,
    validate_predictive_feature_schema_metadata,
)

ROW_STATUSES = frozenset({"ok", "failed", "degraded", "unavailable", "unknown"})
ROW_REQUIRED_FIELDS = frozenset(
    {
        "source_issue",
        "campaign",
        "variant",
        "scenario",
        "seed",
        "planner_grid_row",
        "planner_grid_key",
        "status",
        "success",
        "collision_event",
        "near_miss",
        "low_progress",
        "timeout",
        "min_distance",
        "artifact_pointer",
        "commit_artifact",
    }
)
ROW_OPTIONAL_FIELDS = frozenset(
    {
        "comparison_group",
        "feature_schema",
        "scenario_matrix",
        "seed_manifest",
        "source_note",
    }
)
DURABLE_URI_PREFIXES = (
    "wandb://",
    "wandb-artifact://",
    "artifact://",
    "s3://",
    "gs://",
    "https://",
)
_ISSUE_RE = re.compile(r"^#\d+$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class PredictiveSameSeedRowSummaryValidationError(RobotSfError, ValueError):
    """Raised when a predictive same-seed row-summary payload cannot be loaded."""


def load_predictive_same_seed_row_summary_input(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load a row-summary file as one or more row mappings.

    Returns:
        Tuple containing the detected input format label and the loaded row list.
    """
    if not path.is_file():
        raise PredictiveSameSeedRowSummaryValidationError(f"input file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
        input_format = "rows"
    elif isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
        input_format = "matrix"
    elif isinstance(payload, dict):
        rows = [payload]
        input_format = "row"
    else:
        raise PredictiveSameSeedRowSummaryValidationError(
            "input payload must be a mapping, a list of rows, or a mapping with a 'rows' list"
        )
    if not isinstance(rows, list):
        raise PredictiveSameSeedRowSummaryValidationError("rows must be a list of mappings")
    return input_format, rows


def validate_predictive_same_seed_row_summary_file(
    path: Path,
    *,
    repo_root: Path | None = None,
    check_git_history: bool = False,
) -> dict[str, Any]:
    """Load and validate a predictive same-seed row-summary file.

    Returns:
        Structured validation report with aggregate counts and per-row details.
    """
    input_format, rows = load_predictive_same_seed_row_summary_input(path)
    report = validate_predictive_same_seed_row_summary_rows(
        rows,
        repo_root=repo_root,
        check_git_history=check_git_history,
    )
    report["input_format"] = input_format
    report["input_path"] = _repo_relative_or_absolute(
        path.resolve(), root=(repo_root or get_repository_root())
    )
    return report


def validate_predictive_same_seed_row_summary_rows(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path | None = None,
    check_git_history: bool = False,
) -> dict[str, Any]:
    """Validate predictive same-seed comparison outcome rows.

    Returns:
        Structured validation report with aggregate counts and per-row details.
    """
    root = (repo_root or get_repository_root()).resolve()
    provenance_validation = "git_history" if check_git_history else "format_only"
    git_commit_cache: dict[str, bool] = {}
    row_reports = [
        _validate_row(
            row,
            index=index,
            repo_root=root,
            provenance_validation=provenance_validation,
            git_commit_cache=git_commit_cache,
        )
        for index, row in enumerate(rows)
    ]
    _mark_duplicate_row_keys(row_reports)
    _guard_mixed_ego_motion_producer_keys(row_reports)
    _strip_internal_row_metadata(row_reports)
    invalid_row_count = sum(1 for row in row_reports if row["status"] == "invalid")
    return {
        "status": "valid" if invalid_row_count == 0 else "invalid",
        "provenance_validation": provenance_validation,
        "row_count": len(row_reports),
        "valid_row_count": len(row_reports) - invalid_row_count,
        "invalid_row_count": invalid_row_count,
        "rows": row_reports,
    }


def _mark_duplicate_row_keys(row_reports: list[dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    for row_report in row_reports:
        row_key = row_report.get("row_key")
        if not row_key:
            continue
        first_index = seen.get(row_key)
        if first_index is None:
            seen[row_key] = int(row_report["index"])
            continue
        _append_problem(
            row_report["errors"],
            "row_key",
            f"duplicates row key {row_key!r} first seen at row index {first_index}",
        )
        row_report["status"] = "invalid"


def _validate_row(
    row: object,
    *,
    index: int,
    repo_root: Path,
    provenance_validation: str,
    git_commit_cache: dict[str, bool],
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not isinstance(row, dict):
        _append_problem(errors, "row", "must be a mapping")
        return {
            "index": index,
            "row_key": None,
            "row_status": None,
            "status": "invalid",
            "errors": errors,
            "warnings": warnings,
        }

    _check_expected_fields(
        row,
        required=ROW_REQUIRED_FIELDS,
        optional=ROW_OPTIONAL_FIELDS,
        prefix="row",
        errors=errors,
    )

    source_issue = _require_non_empty_string(row, "source_issue", errors)
    if source_issue is not None and not _ISSUE_RE.fullmatch(source_issue):
        _append_problem(errors, "source_issue", "must match '#<number>'")
    campaign = _require_non_empty_string(row, "campaign", errors)
    variant = _require_non_empty_string(row, "variant", errors)
    scenario = _require_non_empty_string(row, "scenario", errors)
    planner_grid_row = _require_non_empty_string(row, "planner_grid_row", errors)
    planner_grid_key = _require_non_empty_string(row, "planner_grid_key", errors)
    seed = _require_non_negative_int(row, "seed", errors)
    row_status = _require_enum(row, "status", ROW_STATUSES, errors)
    success = _require_nullable_bool(row, "success", errors)
    collision_event = _require_nullable_bool(row, "collision_event", errors)
    near_miss = _require_nullable_bool(row, "near_miss", errors)
    low_progress = _require_nullable_bool(row, "low_progress", errors)
    timeout = _require_nullable_bool(row, "timeout", errors)
    min_distance = _require_nullable_non_negative_number(row, "min_distance", errors)

    _validate_artifact_pointer(
        row.get("artifact_pointer"), field="artifact_pointer", repo_root=repo_root, errors=errors
    )
    _validate_commit_artifact(
        row.get("commit_artifact"),
        field="commit_artifact",
        repo_root=repo_root,
        provenance_validation=provenance_validation,
        git_commit_cache=git_commit_cache,
        errors=errors,
    )
    _validate_optional_reference(row, "scenario_matrix", repo_root=repo_root, errors=errors)
    _validate_optional_reference(row, "seed_manifest", repo_root=repo_root, errors=errors)
    _validate_optional_reference(row, "source_note", repo_root=repo_root, errors=errors)
    if "comparison_group" in row:
        _require_non_empty_string(row, "comparison_group", errors)
    feature_schema = _validate_optional_feature_schema(row, errors=errors)

    _validate_semantics(
        row_status=row_status,
        outcomes={
            "success": success,
            "collision_event": collision_event,
            "near_miss": near_miss,
            "low_progress": low_progress,
            "timeout": timeout,
        },
        min_distance=min_distance,
        errors=errors,
        warnings=warnings,
    )

    row_key = None
    if (
        variant is not None
        and scenario is not None
        and seed is not None
        and planner_grid_key is not None
    ):
        row_key = f"{variant}:{scenario}:{seed}:{planner_grid_key}"
    return {
        "index": index,
        "row_key": row_key,
        "source_issue": source_issue,
        "campaign": campaign,
        "variant": variant,
        "scenario": scenario,
        "seed": seed,
        "planner_grid_row": planner_grid_row,
        "planner_grid_key": planner_grid_key,
        "row_status": row_status,
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "_comparison_context": _comparison_context(
            campaign=campaign,
            comparison_group=row.get("comparison_group"),
            scenario=scenario,
            seed=seed,
            planner_grid_key=planner_grid_key,
        ),
        "_feature_schema_present": feature_schema is not None,
        "_producer_key": (
            predictive_ego_motion_channel_producer_key(feature_schema)
            if isinstance(feature_schema, dict)
            else None
        ),
    }


def _validate_optional_feature_schema(
    mapping: dict[str, Any],
    *,
    errors: list[dict[str, str]],
) -> dict[str, Any] | None:
    if "feature_schema" not in mapping or mapping["feature_schema"] is None:
        return None
    feature_schema = mapping.get("feature_schema")
    if not isinstance(feature_schema, dict):
        _append_problem(errors, "feature_schema", "must be a mapping when provided")
        return None
    try:
        validate_predictive_feature_schema_metadata(
            feature_schema,
            input_dim=_feature_schema_input_dim(feature_schema),
        )
    except (ObstacleFeatureSchemaError, TypeError, ValueError) as exc:
        _append_problem(
            errors, "feature_schema", f"invalid predictive feature schema metadata: {exc}"
        )
        return None
    return feature_schema


def _feature_schema_input_dim(feature_schema: dict[str, Any]) -> int:
    input_dim = feature_schema.get("input_dim")
    if isinstance(input_dim, bool) or not isinstance(input_dim, int):
        raise ValueError("feature_schema.input_dim must be an integer")
    return int(input_dim)


def _comparison_context(
    *,
    campaign: str | None,
    comparison_group: object,
    scenario: str | None,
    seed: int | None,
    planner_grid_key: str | None,
) -> tuple[str | None, str | None, str | None, int | None, str | None] | None:
    if scenario is None or seed is None or planner_grid_key is None:
        return None
    normalized_group = None
    if isinstance(comparison_group, str) and comparison_group.strip():
        normalized_group = comparison_group.strip()
    return (campaign, normalized_group, scenario, seed, planner_grid_key)


def _guard_mixed_ego_motion_producer_keys(row_reports: list[dict[str, Any]]) -> None:
    grouped_reports: dict[
        tuple[str | None, str | None, str | None, int | None, str | None],
        list[dict[str, Any]],
    ] = {}
    for row_report in row_reports:
        context = row_report.get("_comparison_context")
        if context is None:
            continue
        grouped_reports.setdefault(context, []).append(row_report)
    for context, grouped in grouped_reports.items():
        if len(grouped) < 2:
            continue
        if not any(row_report.get("_feature_schema_present") for row_report in grouped):
            continue
        _apply_ego_motion_producer_guard(grouped, context=context)


def _apply_ego_motion_producer_guard(
    grouped: list[dict[str, Any]],
    *,
    context: tuple[str | None, str | None, str | None, int | None, str | None],
) -> None:
    producer_keys = sorted(
        {
            str(producer_key)
            for producer_key in (row_report.get("_producer_key") for row_report in grouped)
            if producer_key
        }
    )
    rows_without_producer = [
        row_report for row_report in grouped if row_report.get("_producer_key") is None
    ]
    if len(producer_keys) > 1:
        _mark_grouped_rows_invalid_for_mixed_producers(
            grouped, context=context, producer_keys=producer_keys
        )
    if rows_without_producer and producer_keys:
        _warn_grouped_rows_for_missing_producers(
            grouped,
            context=context,
            producer_keys=producer_keys,
        )
        return
    if rows_without_producer:
        _warn_grouped_rows_for_unknown_comparability(grouped, context=context)


def _mark_grouped_rows_invalid_for_mixed_producers(
    grouped: list[dict[str, Any]],
    *,
    context: tuple[str | None, str | None, str | None, int | None, str | None],
    producer_keys: list[str],
) -> None:
    producer_list = ", ".join(repr(key) for key in producer_keys)
    message = (
        f"{_format_comparison_context(context)} mixes "
        "ego_motion_channel_producer.producer_key values "
        f"[{producer_list}]; rows are not directly comparable without grouping or an explicit caveat"
    )
    for row_report in grouped:
        _append_problem(
            row_report["errors"],
            "feature_schema.ego_motion_channel_producer.producer_key",
            message,
        )
        row_report["status"] = "invalid"


def _warn_grouped_rows_for_missing_producers(
    grouped: list[dict[str, Any]],
    *,
    context: tuple[str | None, str | None, str | None, int | None, str | None],
    producer_keys: list[str],
) -> None:
    producer_list = ", ".join(repr(key) for key in producer_keys)
    message = (
        f"{_format_comparison_context(context)} includes legacy/no-metadata rows without "
        "ego_motion_channel_producer.producer_key alongside producer-stamped rows "
        f"[{producer_list}]; direct comparability is not proven"
    )
    for row_report in grouped:
        _append_problem(
            row_report["warnings"],
            "feature_schema.ego_motion_channel_producer.producer_key",
            message,
        )


def _warn_grouped_rows_for_unknown_comparability(
    grouped: list[dict[str, Any]],
    *,
    context: tuple[str | None, str | None, str | None, int | None, str | None],
) -> None:
    message = (
        f"{_format_comparison_context(context)} has rows without "
        "ego_motion_channel_producer.producer_key metadata; direct comparability is not proven"
    )
    for row_report in grouped:
        _append_problem(
            row_report["warnings"],
            "feature_schema.ego_motion_channel_producer.producer_key",
            message,
        )


def _format_comparison_context(
    context: tuple[str | None, str | None, str | None, int | None, str | None],
) -> str:
    campaign, comparison_group, scenario, seed, planner_grid_key = context
    parts = []
    if campaign:
        parts.append(f"campaign={campaign!r}")
    if comparison_group:
        parts.append(f"comparison_group={comparison_group!r}")
    if scenario:
        parts.append(f"scenario={scenario!r}")
    if seed is not None:
        parts.append(f"seed={seed}")
    if planner_grid_key:
        parts.append(f"planner_grid_key={planner_grid_key!r}")
    return "comparison context " + ", ".join(parts)


def _strip_internal_row_metadata(row_reports: list[dict[str, Any]]) -> None:
    for row_report in row_reports:
        for field in (
            "_comparison_context",
            "_feature_schema_present",
            "_producer_key",
        ):
            row_report.pop(field, None)


def _validate_semantics(
    *,
    row_status: str | None,
    outcomes: dict[str, bool | None],
    min_distance: float | None,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    _validate_status_specific_nullability(
        row_status=row_status,
        outcomes=outcomes,
        min_distance=min_distance,
        errors=errors,
    )
    success = outcomes["success"]
    if success is True:
        for field in ("collision_event", "low_progress", "timeout"):
            if outcomes[field] is True:
                _append_problem(errors, field, "cannot be true when success is true")
    if row_status == "failed" and success is True:
        _append_problem(errors, "success", "cannot be true when status is 'failed'")
    if outcomes["low_progress"] is True and outcomes["timeout"] is True:
        _append_problem(
            errors,
            "timeout",
            "cannot be true when low_progress is already true for the same row",
        )
    if row_status == "degraded" and all(
        value is None for value in (*outcomes.values(), min_distance)
    ):
        _append_problem(
            warnings,
            "status",
            "degraded rows should preserve any known outcome flags instead of leaving every field null",
        )


def _validate_status_specific_nullability(
    *,
    row_status: str | None,
    outcomes: dict[str, bool | None],
    min_distance: float | None,
    errors: list[dict[str, str]],
) -> None:
    if row_status == "ok":
        for field, value in outcomes.items():
            if value is None:
                _append_problem(errors, field, "must be a boolean when status is 'ok'")
        if min_distance is None:
            _append_problem(errors, "min_distance", "must be a number when status is 'ok'")
        return
    if row_status not in {"unavailable", "unknown"}:
        return
    for field, value in outcomes.items():
        if value is not None:
            _append_problem(errors, field, f"must be null when status is {row_status!r}")
    if min_distance is not None:
        _append_problem(errors, "min_distance", f"must be null when status is {row_status!r}")


def _validate_artifact_pointer(
    value: object,
    *,
    field: str,
    repo_root: Path,
    errors: list[dict[str, str]],
) -> None:
    if not isinstance(value, str) or not value.strip():
        _append_problem(errors, field, "must be a non-empty string")
        return
    if len(_split_reference_tokens(value)) != 1:
        _append_problem(errors, field, "must be a single durable pointer token")
        return
    _validate_reference_token(value.strip(), field, repo_root=repo_root, errors=errors)


def _validate_commit_artifact(
    value: object,
    *,
    field: str,
    repo_root: Path,
    provenance_validation: str,
    git_commit_cache: dict[str, bool],
    errors: list[dict[str, str]],
) -> None:
    if not isinstance(value, str) or not value.strip():
        _append_problem(errors, field, "must be a non-empty string")
        return
    tokens = [token for token in _split_reference_tokens(value) if token]
    if not tokens:
        _append_problem(
            errors,
            field,
            "must include a git SHA token plus one or more provenance tokens",
        )
        return
    has_git_sha = False
    has_provenance_token = False
    git_sha_tokens: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if _GIT_SHA_RE.fullmatch(normalized):
            has_git_sha = True
            git_sha_tokens.append(normalized)
            continue
        has_provenance_token = True
        _validate_reference_token(token, field, repo_root=repo_root, errors=errors)
    if not has_git_sha:
        _append_problem(errors, field, "must include a 7-40 character git SHA token")
    if not has_provenance_token:
        _append_problem(errors, field, "must include at least one provenance pointer token")
    if provenance_validation == "git_history":
        for sha in git_sha_tokens:
            if _git_commit_exists(sha, repo_root=repo_root, cache=git_commit_cache):
                continue
            _append_problem(
                errors,
                field,
                f"references unknown git commit SHA in repository history: {sha!r}",
            )


def _git_commit_exists(sha: str, *, repo_root: Path, cache: dict[str, bool]) -> bool:
    cached = cache.get(sha)
    if cached is not None:
        return cached
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    exists = result.returncode == 0
    cache[sha] = exists
    return exists


def _validate_optional_reference(
    mapping: dict[str, Any],
    field: str,
    *,
    repo_root: Path,
    errors: list[dict[str, str]],
) -> None:
    if field not in mapping or mapping[field] is None:
        return
    value = _require_non_empty_string(mapping, field, errors)
    if value is not None:
        _validate_reference_token(value, field, repo_root=repo_root, errors=errors)


def _validate_reference_token(
    token: str,
    field: str,
    *,
    repo_root: Path,
    errors: list[dict[str, str]],
) -> None:
    if token.startswith(DURABLE_URI_PREFIXES):
        return
    path = Path(token)
    if path.is_absolute():
        _append_problem(
            errors, field, f"must use repository-root-relative paths, not absolute path {token!r}"
        )
        return
    if ".." in path.parts:
        _append_problem(errors, field, f"must not escape the repository root: {token!r}")
        return
    resolved = (repo_root / path).resolve()
    output_dir = (repo_root / "output").resolve()
    if resolved == output_dir or output_dir in resolved.parents:
        _append_problem(errors, field, f"must not depend on worktree-local output paths: {token!r}")
        return
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        _append_problem(errors, field, f"must resolve inside the repository root: {token!r}")
        return
    if not resolved.exists():
        _append_problem(errors, field, f"references a missing repository path: {token!r}")


def _check_expected_fields(
    mapping: dict[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    prefix: str,
    errors: list[dict[str, str]],
) -> None:
    missing = sorted(required - mapping.keys())
    for field in missing:
        _append_problem(errors, f"{prefix}.{field}" if prefix != "row" else field, "is required")
    allowed = required | optional
    unexpected = sorted(set(mapping) - allowed)
    for field in unexpected:
        _append_problem(
            errors,
            f"{prefix}.{field}" if prefix != "row" else field,
            "is not part of the canonical schema",
        )


def _require_non_empty_string(
    mapping: dict[str, Any], field: str, errors: list[dict[str, str]]
) -> str | None:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        _append_problem(errors, field, "must be a non-empty string")
        return None
    return value.strip()


def _require_enum(
    mapping: dict[str, Any],
    field: str,
    allowed: frozenset[str],
    errors: list[dict[str, str]],
) -> str | None:
    value = mapping.get(field)
    if not isinstance(value, str) or value not in allowed:
        _append_problem(errors, field, f"must be one of {sorted(allowed)!r}")
        return None
    return value


def _require_non_negative_int(
    mapping: dict[str, Any], field: str, errors: list[dict[str, str]]
) -> int | None:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        _append_problem(errors, field, "must be a non-negative integer")
        return None
    if value < 0:
        _append_problem(errors, field, "must be a non-negative integer")
        return None
    return value


def _require_nullable_bool(
    mapping: dict[str, Any], field: str, errors: list[dict[str, str]]
) -> bool | None:
    value = mapping.get(field)
    if value is None:
        return None
    if not isinstance(value, bool):
        _append_problem(errors, field, "must be a boolean or null")
        return None
    return value


def _require_nullable_non_negative_number(
    mapping: dict[str, Any], field: str, errors: list[dict[str, str]]
) -> float | None:
    value = mapping.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _append_problem(errors, field, "must be a finite number or null")
        return None
    normalized = float(value)
    if normalized < 0.0:
        _append_problem(errors, field, "must be greater than or equal to 0.0")
    return normalized


def _split_reference_tokens(value: str) -> list[str]:
    return [token.strip() for token in re.split(r"[\n,]+", value) if token.strip()]


def _append_problem(problems: list[dict[str, str]], field: str, message: str) -> None:
    problems.append({"field": field, "message": message})


def _repo_relative_or_absolute(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "PredictiveSameSeedRowSummaryValidationError",
    "load_predictive_same_seed_row_summary_input",
    "validate_predictive_same_seed_row_summary_file",
    "validate_predictive_same_seed_row_summary_rows",
]
