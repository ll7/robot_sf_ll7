#!/usr/bin/env python3
"""Validate the tracked asset-rights inventory without changing repository state.

The inventory is a classification and provenance boundary, not a rights decision.  A row with
``blocked`` or ``external-pointer-only`` status is intentionally retained in the report and
keeps a release fail-closed until the named evidence is available.  The CI workflow may permit
those known blockers while still rejecting malformed, overlapping, or unclassified rows.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPO_ROOT / "scripts" / "validation" / "asset_rights_inventory.v1.yaml"
REPORT_SCHEMA = "robot-sf-asset-rights-inventory.v1"
INVENTORY_SCHEMA = "robot_sf.asset_rights_inventory.v1"
ALLOWED_STATUSES = {
    "blocked",
    "cleared",
    "evidence-only",
    "external-pointer-only",
    "fixture-only",
    "project-authored",
}
REQUIRED_ROW_FIELDS = (
    "source",
    "source_revision_or_access_date",
    "license_or_rights",
    "attribution",
    "checksum_policy",
    "modification_status",
    "evidence",
)
KNOWN_BLOCKING_STATUSES = {"blocked", "external-pointer-only"}
ASSET_SUFFIXES = {
    ".bag",
    ".geojson",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".mov",
    ".osm",
    ".pbf",
    ".pkl",
    ".png",
    ".svg",
    ".wav",
}
ASSET_PATH_HINTS = {"assets", "data", "datasets", "maps", "media", "recordings"}
NON_ASSET_SUFFIXES = {".md", ".py", ".pyi", ".rst", ".sh", ".toml", ".txt"}


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    """Return one deterministic validation issue."""
    return {"code": code, "message": message, **details}


def _normalise_path(value: str) -> str:
    """Normalise a repository-relative POSIX path and reject traversal."""
    normalised = value.replace("\\", "/")
    if normalised.startswith("/") or any(part == ".." for part in normalised.split("/")):
        raise ValueError(f"repository-relative path required: {value!r}")
    return normalised.lstrip("./")


def _normalise_pattern(value: str) -> str:
    """Normalise and validate one repository-relative glob."""
    pattern = _normalise_path(value)
    if not pattern or pattern == ".":
        raise ValueError("glob must not be empty")
    return pattern.rstrip("/")


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a small POSIX glob dialect where ``*`` does not cross directories."""
    chunks: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            chunks.append("(?:.*/)?")
            index += 3
            continue
        if pattern.startswith("**", index):
            chunks.append(".*")
            index += 2
            continue
        character = pattern[index]
        if character == "*":
            chunks.append("[^/]*")
        elif character == "?":
            chunks.append("[^/]")
        else:
            chunks.append(re.escape(character))
        index += 1
    return re.compile("^" + "".join(chunks) + "$")


def _matches(path: str, pattern: str) -> bool:
    """Return whether a normalised path matches a normalised inventory glob."""
    return bool(_glob_regex(pattern).match(path))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return [item for item in value if item.strip()]


def _load_inventory(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load YAML and return its object plus structural load issues."""
    issues: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        return None, [_issue("inventory_load_error", f"cannot load inventory: {exc}")]
    if not isinstance(payload, dict):
        return None, [_issue("inventory_not_object", "inventory YAML must contain a mapping")]
    return payload, issues


def _tracked_paths(repo_root: Path) -> list[str]:
    """Read tracked paths from Git without touching the worktree."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return sorted(_normalise_path(raw) for raw in result.stdout.decode("utf-8").split("\0") if raw)


def _path_is_inside(repo_root: Path, relative_path: str) -> bool:
    """Return whether an evidence path stays within the repository."""
    candidate = (repo_root / relative_path).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        return False
    return True


def _row_matches(row: dict[str, Any], path: str) -> bool:
    """Return whether a row covers a path, honoring optional exclusions."""
    patterns = row.get("globs", [])
    exclusions = row.get("exclude_globs", [])
    return any(_matches(path, pattern) for pattern in patterns) and not any(
        _matches(path, pattern) for pattern in exclusions
    )


def _normalise_scope(
    raw_scope: Any,
    index: int,
    scope_ids: set[str],
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate and normalise one tracked scope."""
    if not isinstance(raw_scope, dict):
        issues.append(_issue("invalid_scope", "scope must be a mapping", index=index))
        return None
    scope_id = raw_scope.get("id")
    if not _nonempty_string(scope_id):
        issues.append(
            _issue("invalid_scope_id", "scope id must be a non-empty string", index=index)
        )
        return None
    if scope_id in scope_ids:
        issues.append(
            _issue("duplicate_scope_id", f"duplicate scope id: {scope_id}", scope=scope_id)
        )
        return None
    scope_ids.add(scope_id)
    raw_globs = _string_list(raw_scope.get("globs"))
    if not raw_globs:
        issues.append(
            _issue("invalid_scope_globs", f"scope has no globs: {scope_id}", scope=scope_id)
        )
        return None
    try:
        globs = [_normalise_pattern(pattern) for pattern in raw_globs]
    except ValueError as exc:
        issues.append(_issue("invalid_scope_glob", str(exc), scope=scope_id))
        return None
    release_relevant = raw_scope.get("release_relevant")
    if not isinstance(release_relevant, bool):
        issues.append(
            _issue(
                "invalid_scope_release_flag",
                f"scope release_relevant must be boolean: {scope_id}",
                scope=scope_id,
            )
        )
        return None
    if not release_relevant and not _nonempty_string(raw_scope.get("exclusion_reason")):
        issues.append(
            _issue(
                "missing_scope_exclusion_reason",
                f"non-release scope needs an exclusion reason: {scope_id}",
                scope=scope_id,
            )
        )
    return {
        **raw_scope,
        "id": scope_id,
        "globs": globs,
        "release_relevant": release_relevant,
    }


def _normalise_row(
    raw_row: Any,
    index: int,
    row_ids: set[str],
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate and normalise one inventory row."""
    if not isinstance(raw_row, dict):
        issues.append(_issue("invalid_row", "row must be a mapping", index=index))
        return None
    row_id = raw_row.get("id")
    if not _nonempty_string(row_id):
        issues.append(_issue("invalid_row_id", "row id must be a non-empty string", index=index))
        return None
    if row_id in row_ids:
        issues.append(_issue("duplicate_row_id", f"duplicate row id: {row_id}", row=row_id))
        return None
    row_ids.add(row_id)
    raw_globs = _string_list(raw_row.get("globs"))
    if not raw_globs:
        issues.append(_issue("invalid_row_globs", f"row has no globs: {row_id}", row=row_id))
        return None
    try:
        globs = [_normalise_pattern(pattern) for pattern in raw_globs]
        exclusions = [
            _normalise_pattern(pattern)
            for pattern in (_string_list(raw_row.get("exclude_globs")) or [])
        ]
    except ValueError as exc:
        issues.append(_issue("invalid_row_glob", str(exc), row=row_id))
        return None
    return {**raw_row, "id": row_id, "globs": globs, "exclude_globs": exclusions}


def _normalise_inventory_patterns(
    payload: dict[str, Any],
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and normalise scopes and rows enough for coverage checking."""
    raw_scopes = payload.get("tracked_scopes")
    raw_rows = payload.get("rows")
    if not isinstance(raw_scopes, list):
        issues.append(_issue("missing_scopes", "tracked_scopes must be a list"))
        raw_scopes = []
    if not isinstance(raw_rows, list):
        issues.append(_issue("missing_rows", "rows must be a list"))
        raw_rows = []
    scope_ids: set[str] = set()
    scopes = [
        scope
        for index, raw_scope in enumerate(raw_scopes)
        if (scope := _normalise_scope(raw_scope, index, scope_ids, issues)) is not None
    ]
    row_ids: set[str] = set()
    rows = [
        row
        for index, raw_row in enumerate(raw_rows)
        if (row := _normalise_row(raw_row, index, row_ids, issues)) is not None
    ]
    return scopes, rows


def _validate_row_metadata(
    row: dict[str, Any],
    scopes_by_id: dict[str, dict[str, Any]],
    repo_root: Path,
    issues: list[dict[str, Any]],
) -> None:
    """Validate fields that do not depend on tracked-path coverage."""
    row_id = row["id"]
    scope_id = row.get("scope")
    if scope_id not in scopes_by_id:
        issues.append(
            _issue(
                "row_unknown_scope", f"row references an unknown scope: {scope_id!r}", row=row_id
            )
        )
    status = row.get("status")
    if status not in ALLOWED_STATUSES:
        issues.append(
            _issue("invalid_row_status", f"row has unsupported status: {status!r}", row=row_id)
        )
    for field in REQUIRED_ROW_FIELDS:
        value = row.get(field)
        valid = _string_list(value) is not None if field == "evidence" else _nonempty_string(value)
        if not valid:
            issues.append(
                _issue(
                    "missing_row_field", f"row field {field!r} is required", row=row_id, field=field
                )
            )
    if status in KNOWN_BLOCKING_STATUSES and not _nonempty_string(row.get("unblock_condition")):
        issues.append(
            _issue(
                "missing_unblock_condition",
                f"known-blocked row needs an unblock condition: {row_id}",
                row=row_id,
            )
        )
    if status == "cleared" and not row.get("evidence"):
        issues.append(
            _issue(
                "cleared_row_without_evidence",
                f"cleared row needs evidence paths: {row_id}",
                row=row_id,
            )
        )
    scope = scopes_by_id.get(scope_id)
    if scope is not None:
        _validate_row_scope_status(row, scope, issues)
    for evidence_path in _string_list(row.get("evidence")) or []:
        _validate_evidence_path(repo_root, row_id, evidence_path, issues)


def _validate_row_scope_status(
    row: dict[str, Any],
    scope: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    """Ensure row status agrees with whether its scope is release-relevant."""
    status = row.get("status")
    if not scope["release_relevant"] and status not in {"evidence-only", "fixture-only"}:
        issues.append(
            _issue(
                "nonrelease_row_status",
                f"non-release scope must use an exclusion status: {row['id']}",
                row=row["id"],
                scope=scope["id"],
            )
        )
    if scope["release_relevant"] and status in {"evidence-only", "fixture-only"}:
        issues.append(
            _issue(
                "release_row_excluded",
                f"release-relevant scope cannot exclude an asset row: {row['id']}",
                row=row["id"],
                scope=scope["id"],
            )
        )


def _validate_evidence_path(
    repo_root: Path,
    row_id: str,
    evidence_path: str,
    issues: list[dict[str, Any]],
) -> None:
    """Ensure one evidence path is repository-relative and present."""
    try:
        normalised_evidence = _normalise_path(evidence_path)
    except ValueError as exc:
        issues.append(_issue("invalid_evidence_path", str(exc), row=row_id))
        return
    if not _path_is_inside(repo_root, normalised_evidence):
        issues.append(
            _issue(
                "evidence_path_escape",
                f"evidence path escapes repository: {normalised_evidence}",
                row=row_id,
            )
        )
    elif not (repo_root / normalised_evidence).exists():
        issues.append(
            _issue(
                "missing_evidence_path",
                f"evidence path does not exist: {normalised_evidence}",
                row=row_id,
                path=normalised_evidence,
            )
        )


def _looks_like_asset(path: str) -> bool:
    """Identify a likely asset that should not remain silently outside a declared scope."""
    suffix = Path(path).suffix.lower()
    return suffix in ASSET_SUFFIXES or (
        suffix not in NON_ASSET_SUFFIXES and bool(ASSET_PATH_HINTS.intersection(path.split("/")))
    )


def _collect_scope_paths(
    paths: list[str],
    scopes: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], int]:
    """Partition tracked paths by declared scope and detect scope overlaps."""
    scope_paths: dict[str, list[str]] = {scope["id"]: [] for scope in scopes}
    unscoped_count = 0
    for path in paths:
        matching_scopes = [
            scope for scope in scopes if any(_matches(path, pattern) for pattern in scope["globs"])
        ]
        if not matching_scopes:
            unscoped_count += 1
            if _looks_like_asset(path):
                issues.append(
                    _issue(
                        "unscoped_asset_path",
                        f"asset-like tracked path has no declared scope: {path}",
                        path=path,
                    )
                )
            continue
        if len(matching_scopes) > 1:
            issues.append(
                _issue(
                    "scope_overlap",
                    f"tracked path belongs to multiple scopes: {path}",
                    path=path,
                    scopes=[scope["id"] for scope in matching_scopes],
                )
            )
        for scope in matching_scopes:
            scope_paths[scope["id"]].append(path)
    return scope_paths, unscoped_count


def _classify_paths(
    paths: list[str],
    scopes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    scope_paths: dict[str, list[str]],
    issues: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], int]:
    """Match scoped paths to exactly one row and reject rows leaking across scopes."""
    row_paths: dict[str, list[str]] = {row["id"]: [] for row in rows}
    classified_count = 0
    rows_by_scope = {
        scope["id"]: [row for row in rows if row.get("scope") == scope["id"]] for scope in scopes
    }
    for scope in scopes:
        scoped_rows = rows_by_scope[scope["id"]]
        for path in scope_paths[scope["id"]]:
            matching_rows = [row for row in scoped_rows if _row_matches(row, path)]
            if len(matching_rows) == 1:
                row_paths[matching_rows[0]["id"]].append(path)
                classified_count += 1
            elif not matching_rows:
                issues.append(
                    _issue(
                        "unclassified_path",
                        f"tracked path has no inventory row: {path}",
                        path=path,
                        scope=scope["id"],
                    )
                )
            else:
                issues.append(
                    _issue(
                        "row_overlap",
                        f"tracked path matches multiple inventory rows: {path}",
                        path=path,
                        scope=scope["id"],
                        rows=[row["id"] for row in matching_rows],
                    )
                )
    for path in paths:
        for row in rows:
            if not _row_matches(row, path):
                continue
            scope_id = row.get("scope")
            if scope_id in scope_paths and path not in scope_paths[scope_id]:
                issues.append(
                    _issue(
                        "row_outside_scope",
                        f"row matches a path outside its scope: {path}",
                        path=path,
                        row=row["id"],
                        scope=scope_id,
                    )
                )
    return row_paths, classified_count


def _finalise_rows(
    rows: list[dict[str, Any]],
    row_paths: dict[str, list[str]],
    known_blockers: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    """Validate non-empty row coverage and collect explicit legal blockers."""
    for row in rows:
        row_id = row["id"]
        if not row_paths[row_id] and not row.get("allow_empty", False):
            issues.append(
                _issue(
                    "empty_inventory_row",
                    f"inventory row matches no tracked path: {row_id}",
                    row=row_id,
                )
            )
        if row.get("status") in KNOWN_BLOCKING_STATUSES:
            known_blockers.append(
                {
                    "row": row_id,
                    "status": row["status"],
                    "path_count": len(row_paths[row_id]),
                    "unblock_condition": row.get("unblock_condition"),
                }
            )


def build_report(
    repo_root: Path = REPO_ROOT,
    inventory_path: Path = DEFAULT_INVENTORY,
    *,
    tracked_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build the deterministic, read-only asset-rights inventory report."""
    repo_root = repo_root.resolve()
    inventory_path = inventory_path.resolve()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "blocked",
        "read_only": True,
        "inventory_path": str(inventory_path),
        "claim_boundary": None,
        "tracked_path_count": 0,
        "release_relevant_path_count": 0,
        "classified_path_count": 0,
        "unscoped_tracked_path_count": 0,
        "issues": [],
        "known_blockers": [],
    }
    issues: list[dict[str, Any]] = report["issues"]
    payload, load_issues = _load_inventory(inventory_path)
    issues.extend(load_issues)
    if payload is None:
        return report

    report["claim_boundary"] = payload.get("claim_boundary")
    if payload.get("schema_version") != INVENTORY_SCHEMA:
        issues.append(
            _issue(
                "invalid_inventory_schema",
                f"expected {INVENTORY_SCHEMA!r}",
                actual=payload.get("schema_version"),
            )
        )
    if not _nonempty_string(payload.get("claim_boundary")):
        issues.append(_issue("missing_claim_boundary", "claim_boundary must be a non-empty string"))

    try:
        source_paths = _tracked_paths(repo_root) if tracked_paths is None else tracked_paths
        paths = sorted(_normalise_path(path) for path in source_paths)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError, ValueError) as exc:
        issues.append(_issue("tracked_path_error", f"cannot enumerate tracked paths: {exc}"))
        return report
    report["tracked_path_count"] = len(paths)
    scopes, rows = _normalise_inventory_patterns(payload, issues)
    scopes_by_id = {scope["id"]: scope for scope in scopes}
    for row in rows:
        _validate_row_metadata(row, scopes_by_id, repo_root, issues)
    scope_paths, unscoped_count = _collect_scope_paths(paths, scopes, issues)
    report["unscoped_tracked_path_count"] = unscoped_count
    report["release_relevant_path_count"] = sum(
        len(scope_paths[scope["id"]]) for scope in scopes if scope["release_relevant"]
    )
    row_paths, classified_count = _classify_paths(paths, scopes, rows, scope_paths, issues)
    report["classified_path_count"] = classified_count
    _finalise_rows(rows, row_paths, report["known_blockers"], issues)

    report["known_blockers"].sort(key=lambda item: item["row"])
    report["issues"] = sorted(
        issues,
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("path", "")),
            str(item.get("row", "")),
        ),
    )
    report["counts"] = {
        "issues": len(report["issues"]),
        "known_blocker_rows": len(report["known_blockers"]),
        "unclassified_paths": sum(
            1 for issue in report["issues"] if issue["code"] == "unclassified_path"
        ),
    }
    report["status"] = (
        "passed" if not report["issues"] and not report["known_blockers"] else "blocked"
    )
    return report


def exit_code(report: dict[str, Any], *, allow_known_blockers: bool = False) -> int:
    """Return a release-fail-closed or classification-only exit code."""
    if report.get("issues"):
        return 2
    if report.get("known_blockers") and not allow_known_blockers:
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--allow-known-blockers",
        action="store_true",
        help="Allow explicit blocked/external-pointer rows after structural validation.",
    )
    parser.add_argument(
        "--tracked-path",
        action="append",
        help="Override Git path enumeration; repeat for deterministic fixture checks.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the read-only inventory checker."""
    args = build_arg_parser().parse_args(argv)
    report = build_report(
        args.repo_root,
        args.inventory,
        tracked_paths=args.tracked_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code(report, allow_known_blockers=args.allow_known_blockers)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
