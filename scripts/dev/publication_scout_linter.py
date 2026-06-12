#!/usr/bin/env python3
"""Conformance checks for publication-scout issue candidate readiness.

The script is intentionally read-only and deterministic; all checks are driven from fixture or
preloaded JSON payloads so tests can cover edge cases without contacting GitHub.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_RECENT_COMMENT_HOURS = 24
SCHEMA = "publication_scout_linter.v1"
COMMENT_WINDOW_FIELDS = ("createdAt", "created_at", "created")


@dataclass(frozen=True)
class LinterFinding:
    """Single validation finding from issue candidate readiness checks."""

    code: str
    detail: str

    def to_payload(self) -> dict[str, str]:
        """Serialize to a JSON-friendly payload."""
        return {"code": self.code, "detail": self.detail}


def parse_iso_datetime(raw: object) -> datetime | None:
    """Parse common GitHub timestamp shapes into timezone-aware datetimes."""
    if not isinstance(raw, str):
        return None

    candidate = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _normalize_repo(value: object) -> str | None:
    """Normalize a repository URL or owner/repo string to ``owner/repo``."""
    if not isinstance(value, str):
        return None

    candidate = value.strip().rstrip("/")
    if not candidate:
        return None

    lowered = candidate.lower()
    if lowered.startswith("https://api.github.com/repos/"):
        candidate = candidate[len("https://api.github.com/repos/") :]
    elif lowered.startswith("https://github.com/"):
        candidate = candidate[len("https://github.com/") :]

    if "://" in candidate:
        return None

    parts = [part for part in candidate.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def extract_issue_repo(issue: dict[str, Any]) -> str | None:
    """Extract repository owner/name from a typical GH issue payload."""
    repository = issue.get("repository")
    if isinstance(repository, dict):
        owner = repository.get("owner")
        if isinstance(owner, dict):
            owner_value = owner.get("login")
        else:
            owner_value = owner
        name = repository.get("name")
        if isinstance(owner_value, str) and isinstance(name, str):
            return f"{owner_value}/{name}"
    for field in ("repository_url", "html_url", "url", "repo"):
        normalized = _normalize_repo(issue.get(field))
        if normalized is not None:
            return normalized
    return None


def _extract_comment_time(comment: dict[str, Any]) -> datetime | None:
    """Return a parsed comment timestamp if available."""
    for field in COMMENT_WINDOW_FIELDS:
        value = comment.get(field)
        parsed = parse_iso_datetime(value)
        if parsed is not None:
            return parsed
    return None


def has_recent_comment_evidence(
    comments: object,
    *,
    recent_window_hours: int,
    now: datetime | None = None,
) -> bool:
    """Return true when at least one comment is newer than the recency window."""
    if not isinstance(recent_window_hours, int) or recent_window_hours < 0:
        raise ValueError("recent_window_hours must be a non-negative integer")

    if not isinstance(comments, list):
        return False

    now_dt = now or datetime.now(UTC)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=UTC)

    threshold = now_dt - timedelta(hours=recent_window_hours)

    for raw_comment in comments:
        if not isinstance(raw_comment, dict):
            continue
        created_at = _extract_comment_time(raw_comment)
        if created_at is None:
            continue
        if created_at >= threshold:
            return True
    return False


def validate_candidate(
    issue: dict[str, Any],
    comments: object,
    *,
    expected_repo: str = DEFAULT_REPO,
    recent_comment_hours: int = DEFAULT_RECENT_COMMENT_HOURS,
    now: datetime | None = None,
) -> list[LinterFinding]:
    """Validate a candidate issue for publication-scout handoff readiness."""
    findings: list[LinterFinding] = []

    if str(issue.get("state", "")).lower() != "open":
        findings.append(
            LinterFinding(
                code="issue_not_open",
                detail="candidate issue is not open; skip until the issue is open in source of truth",
            )
        )

    actual_repo = extract_issue_repo(issue)
    if actual_repo != expected_repo:
        findings.append(
            LinterFinding(
                code="issue_repo_mismatch",
                detail=(f"candidate repo {actual_repo or '<unknown>'} != expected {expected_repo}"),
            )
        )

    if not has_recent_comment_evidence(
        comments,
        recent_window_hours=recent_comment_hours,
        now=now,
    ):
        findings.append(
            LinterFinding(
                code="missing_recent_comments",
                detail="no recent comments found within the configured recency window",
            )
        )

    return findings


def classify_comment_publication_result(raw_payload: object) -> dict[str, Any]:
    """Classify GraphQL addComment result payloads for deterministic script checks."""
    if not isinstance(raw_payload, dict):
        return {
            "ok": False,
            "status": "invalid_payload",
            "detail": "payload is not a JSON object",
            "error_type": "invalid_payload",
        }

    errors = raw_payload.get("errors")
    if not errors:
        return {"ok": True, "status": "ok", "detail": "no GraphQL errors"}

    if not isinstance(errors, list) or not errors or not isinstance(errors[0], dict):
        return {
            "ok": False,
            "status": "invalid_payload",
            "detail": "errors must be a non-empty list of objects",
            "error_type": "invalid_payload",
        }

    first_error = errors[0]
    extensions = first_error.get("extensions")
    extension_type = extensions.get("type", "") if isinstance(extensions, dict) else ""
    error_type = str(first_error.get("type", "") or extension_type)
    message = str(first_error.get("message", "")).strip()
    if not error_type:
        error_type = "UNKNOWN"

    normalized = error_type.upper()
    status_map = {
        "FORBIDDEN": "forbidden",
        "UNAUTHENTICATED": "unauthenticated",
        "NOT_FOUND": "target_not_found",
        "UNPROCESSABLE": "unprocessable",
        "RATE_LIMITED": "rate_limited",
        "RATELIMITED": "rate_limited",
        "BAD_REQUEST": "bad_request",
    }

    return {
        "ok": False,
        "status": status_map.get(normalized, "graphql_error"),
        "detail": message or "GraphQL returned an error",
        "error_type": normalized,
        "raw_error": first_error,
    }


def _build_report(
    *,
    issue: dict[str, Any],
    findings: list[LinterFinding],
    expected_repo: str,
    recent_comment_hours: int,
    has_recent_comments: bool,
) -> dict[str, Any]:
    """Build the machine-readable CLI report."""
    return {
        "schema": SCHEMA,
        "ok": not findings,
        "read_only": True,
        "project_writes": False,
        "expected_repo": expected_repo,
        "recent_comment_hours": recent_comment_hours,
        "checks": [
            {"name": "issue_state_open", "ok": str(issue.get("state", "")).lower() == "open"},
            {
                "name": "issue_repo_match",
                "ok": extract_issue_repo(issue) == expected_repo,
                "actual_repo": extract_issue_repo(issue),
            },
            {"name": "recent_comment_evidence", "ok": has_recent_comments},
        ],
        "issue_number": issue.get("number"),
        "issue_url": issue.get("url"),
        "findings": [finding.to_payload() for finding in findings],
        "failure_summary": None
        if not findings
        else {
            "count": len(findings),
            "codes": [finding.code for finding in findings],
        },
    }


def _load_json(payload_path: Path) -> Any:
    """Load one JSON payload from disk."""
    data = payload_path.read_text(encoding="utf-8")
    return json.loads(data)


def _build_parser() -> argparse.ArgumentParser:
    """Build a deterministic CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-json", type=Path, required=True)
    parser.add_argument("--comments-json", type=Path, required=True)
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="Expected owner/repo (e.g. ll7/robot_sf_ll7)"
    )
    parser.add_argument(
        "--recent-comment-hours",
        type=int,
        default=DEFAULT_RECENT_COMMENT_HOURS,
        help="Window in hours for recency checks on comments.",
    )
    parser.add_argument(
        "--publish-result-json",
        type=Path,
        default=None,
        help="Optional GraphQL publication result payload for classify-only checks.",
    )
    return parser


def _dump_json(payload: dict[str, Any]) -> None:
    """Write sorted JSON for stable, machine-readable output."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)

    try:
        issue = _load_json(args.issue_json)
        comments = _load_json(args.comments_json)
    except (OSError, json.JSONDecodeError) as exc:
        _dump_json(
            {
                "schema": SCHEMA,
                "ok": False,
                "read_only": True,
                "project_writes": False,
                "error": str(exc),
            }
        )
        return 2

    if not isinstance(issue, dict):
        _dump_json(
            {
                "schema": SCHEMA,
                "ok": False,
                "read_only": True,
                "project_writes": False,
                "error": "issue-json payload must be an object",
            }
        )
        return 2

    findings = validate_candidate(
        issue,
        comments,
        expected_repo=args.repo,
        recent_comment_hours=args.recent_comment_hours,
    )
    report = _build_report(
        issue=issue,
        findings=findings,
        expected_repo=args.repo,
        recent_comment_hours=args.recent_comment_hours,
        has_recent_comments=not any(f.code == "missing_recent_comments" for f in findings),
    )

    if args.publish_result_json is not None:
        try:
            publish_payload = _load_json(args.publish_result_json)
        except (OSError, json.JSONDecodeError) as exc:
            report["publish_classification"] = {
                "ok": False,
                "status": "invalid_payload",
                "detail": str(exc),
                "error_type": "invalid_payload",
            }
        else:
            report["publish_classification"] = classify_comment_publication_result(publish_payload)

    _dump_json(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
