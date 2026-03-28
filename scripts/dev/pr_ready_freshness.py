#!/usr/bin/env python3
"""Record and verify local PR-readiness evidence for the current branch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _run_git(args: list[str]) -> str:
    """Return stdout for a git command."""
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _current_branch() -> str:
    return _run_git(["branch", "--show-current"])


def _head_sha() -> str:
    return _run_git(["rev-parse", "HEAD"])


def _sanitize_branch(branch: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", branch).strip("-") or "detached-head"


def _default_stamp_path(branch: str) -> Path:
    return Path("output/validation/pr_ready") / f"{_sanitize_branch(branch)}.json"


def _resolve_stamp_path(stamp_path: str | None, branch: str) -> Path:
    return Path(stamp_path) if stamp_path else _default_stamp_path(branch)


def _json_dump(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_stamp(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_stamp(
    *,
    path: Path,
    branch: str,
    base_ref: str,
    head_sha: str,
    status: str,
) -> dict[str, Any]:
    payload = {
        "branch": branch,
        "base_ref": base_ref,
        "head_sha": head_sha,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _freshness_result(
    *,
    path: Path,
    branch: str,
    base_ref: str,
    head_sha: str,
    max_age_hours: float,
) -> tuple[bool, dict[str, Any]]:
    result: dict[str, Any] = {
        "fresh": False,
        "stamp_path": str(path),
        "branch": branch,
        "base_ref": base_ref,
        "head_sha": head_sha,
        "max_age_hours": max_age_hours,
    }
    if not path.exists():
        result["reason"] = "missing"
        return False, result

    try:
        stamp = _load_stamp(path)
    except (OSError, json.JSONDecodeError) as exc:
        result["reason"] = "unreadable"
        result["error"] = str(exc)
        return False, result

    result["stamp"] = stamp
    if str(stamp.get("status")) != "passed":
        result["reason"] = "status_not_passed"
        return False, result
    if str(stamp.get("branch")) != branch:
        result["reason"] = "branch_mismatch"
        return False, result
    if str(stamp.get("base_ref")) != base_ref:
        result["reason"] = "base_ref_mismatch"
        return False, result
    if str(stamp.get("head_sha")) != head_sha:
        result["reason"] = "head_sha_mismatch"
        return False, result

    try:
        recorded_at = datetime.fromisoformat(str(stamp.get("recorded_at_utc")))
    except ValueError:
        result["reason"] = "invalid_timestamp"
        return False, result

    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=UTC)

    age = datetime.now(UTC) - recorded_at.astimezone(UTC)
    result["age_seconds"] = age.total_seconds()
    if age > timedelta(hours=max_age_hours):
        result["reason"] = "expired"
        return False, result

    result["fresh"] = True
    result["reason"] = "fresh"
    return True, result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and verify local PR readiness evidence for the current branch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser(
        "status",
        help="Check whether a recent successful readiness run exists for this branch and HEAD.",
    )
    status_parser.add_argument("--base-ref", default="origin/main")
    status_parser.add_argument("--branch")
    status_parser.add_argument("--head-sha")
    status_parser.add_argument("--stamp-path")
    status_parser.add_argument("--max-age-hours", type=float, default=24.0)

    write_parser = subparsers.add_parser(
        "write",
        help="Record a successful readiness run for this branch and HEAD.",
    )
    write_parser.add_argument("--base-ref", default="origin/main")
    write_parser.add_argument("--branch")
    write_parser.add_argument("--head-sha")
    write_parser.add_argument("--stamp-path")
    write_parser.add_argument("--status", default="passed")

    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    branch = args.branch or _current_branch()
    head_sha = args.head_sha or _head_sha()
    stamp_path = _resolve_stamp_path(args.stamp_path, branch)

    if args.command == "write":
        payload = _write_stamp(
            path=stamp_path,
            branch=branch,
            base_ref=args.base_ref,
            head_sha=head_sha,
            status=args.status,
        )
        _json_dump(
            {
                "ok": True,
                "stamp_path": str(stamp_path),
                "stamp": payload,
            }
        )
        return 0

    is_fresh, payload = _freshness_result(
        path=stamp_path,
        branch=branch,
        base_ref=args.base_ref,
        head_sha=head_sha,
        max_age_hours=args.max_age_hours,
    )
    _json_dump(payload)
    return 0 if is_fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
