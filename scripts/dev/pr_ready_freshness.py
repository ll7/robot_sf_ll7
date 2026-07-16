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
    """Read the current git branch name.

    Returns:
        Active branch name, or an empty string when git reports detached HEAD.
    """
    return _run_git(["branch", "--show-current"])


def _head_sha() -> str:
    """Read the current HEAD commit SHA.

    Returns:
        Full git SHA for HEAD.
    """
    return _run_git(["rev-parse", "HEAD"])


def _resolve_base_sha(base_ref: str) -> str | None:
    """Resolve *base_ref* to a concrete commit SHA.

    Returns:
        Full git SHA for *base_ref*, or ``None`` when the ref does not resolve to
        a local commit (e.g. a fresh checkout where ``origin/main`` was never
        fetched).  A ``None`` result is intentional: callers must skip the
        resolved-SHA comparison rather than fail closed on a ref they cannot see.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{base_ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _tree_state() -> str:
    """Return whether the current non-ignored worktree has uncommitted changes.

    Returns:
        ``"clean"`` when tracked, staged, and untracked non-ignored files are absent;
        otherwise ``"dirty"``.
    """
    status = _run_git(["status", "--porcelain", "--untracked-files=normal"])
    return "dirty" if status else "clean"


def _sanitize_branch(branch: str) -> str:
    """Convert a branch name into a safe readiness-stamp filename stem.

    Returns:
        Sanitized branch name or ``detached-head`` for empty input.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", branch).strip("-") or "detached-head"


def _default_stamp_path(branch: str) -> Path:
    """Build the default readiness stamp path for a branch.

    Returns:
        Path under ``output/validation/pr_ready`` for the branch stamp.
    """
    return Path("output/validation/pr_ready") / f"{_sanitize_branch(branch)}.json"


def _resolve_stamp_path(stamp_path: str | None, branch: str) -> Path:
    """Resolve an explicit or default readiness stamp path.

    Returns:
        Caller-provided path, or the branch-specific default stamp path.
    """
    return Path(stamp_path) if stamp_path else _default_stamp_path(branch)


def _json_dump(payload: dict[str, Any]) -> None:
    """Print a JSON payload with stable formatting."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_stamp(path: Path) -> dict[str, Any]:
    """Load a readiness stamp from disk.

    Returns:
        Parsed readiness stamp payload.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_stamp(
    *,
    path: Path,
    branch: str,
    base_ref: str,
    base_sha: str | None,
    head_sha: str,
    tree_state: str,
    status: str,
    require_clean_tree: bool,
) -> dict[str, Any]:
    """Write a readiness stamp for the current branch and HEAD.

    Returns:
        Stamp payload written to disk.
    """
    if require_clean_tree and tree_state != "clean":
        return {
            "ok": False,
            "reason": "dirty_worktree",
            "branch": branch,
            "base_ref": base_ref,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "tree_state": tree_state,
            "stamp_path": str(path),
        }

    payload = {
        "branch": branch,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "tree_state": tree_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _base_sha_drift_advisory(
    stamp: dict[str, Any],
    base_ref: str,
    current_base_sha: str | None,
) -> dict[str, str] | None:
    """Return an advisory describing base drift between the stamped and current base.

    The stamp records the concrete base SHA that the readiness run actually validated
    against (issue #5782).  When ``origin/main`` (or another moving base) advances
    after the stamp was recorded, that SHA no longer matches the current base.  This
    is reported as a reviewable advisory rather than a freshness failure: the
    readiness gate already ran a base-drift recheck before recording the stamp and
    records the stamp only when the drift is current or unrelated to the PR's changed
    paths, so failing status here would recreate the re-run friction the gate-level
    check exists to break.

    Returns:
        ``None`` when the base SHA is still current or cannot be compared; otherwise
        an advisory dict with the stamped and current base SHAs.
    """
    stamped_base_sha = stamp.get("base_sha")
    if not stamped_base_sha or current_base_sha is None:
        # Either the stamp predates base-SHA capture or the ref cannot be resolved
        # locally; nothing to compare, so there is no drift advisory.
        return None
    if stamped_base_sha == current_base_sha:
        return None
    return {
        "base_ref": base_ref,
        "stamped_base_sha": str(stamped_base_sha),
        "current_base_sha": str(current_base_sha),
    }


def _equality_mismatch_reason(
    stamp: dict[str, Any],
    *,
    branch: str,
    base_ref: str,
    head_sha: str,
) -> str | None:
    """Return the reason string for the first non-matching equality field, else None."""
    checks = [
        ("status", "passed", "status_not_passed"),
        ("branch", branch, "branch_mismatch"),
        ("base_ref", base_ref, "base_ref_mismatch"),
        ("head_sha", head_sha, "head_sha_mismatch"),
    ]
    for key, expected, reason in checks:
        if str(stamp.get(key)) != str(expected):
            return reason
    return None


def _freshness_result(
    *,
    path: Path,
    branch: str,
    base_ref: str,
    base_sha: str | None,
    head_sha: str,
    tree_state: str | None,
    require_clean_tree: bool,
    max_age_hours: float,
) -> tuple[bool, dict[str, Any]]:
    """Check whether an existing readiness stamp matches the current context.

    Returns:
        Tuple of freshness boolean and detailed status payload.
    """
    result: dict[str, Any] = {
        "fresh": False,
        "stamp_path": str(path),
        "branch": branch,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "tree_state": tree_state,
        "require_clean_tree": require_clean_tree,
        "max_age_hours": max_age_hours,
    }
    if require_clean_tree and tree_state is not None and tree_state != "clean":
        result["reason"] = "dirty_worktree"
        return False, result
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
    if require_clean_tree and stamp.get("tree_state") != "clean":
        result["reason"] = "stamp_tree_state_not_clean"
        return False, result

    equality_reason = _equality_mismatch_reason(
        stamp, branch=branch, base_ref=base_ref, head_sha=head_sha
    )
    if equality_reason is not None:
        result["reason"] = equality_reason
        return False, result

    # The validated base SHA is captured for provenance and reported as an advisory
    # when origin/main (or another moving base) has advanced since the gate recorded
    # the stamp. It is intentionally NOT a freshness failure here: the readiness gate
    # already ran a base-drift recheck before recording the stamp (issue #5782) and
    # records the stamp only when the drift is current or unrelated to the PR's
    # changed paths. Failing status on any base movement would force the opener to
    # re-run the gate every time main advances during an active merge window -- the
    # exact friction the gate-level drift check exists to break. The advisory keeps
    # the drift visible and reviewable without invalidating a gate-approved stamp.
    drift_advisory = _base_sha_drift_advisory(stamp, base_ref, base_sha)
    if drift_advisory is not None:
        result["base_drift"] = drift_advisory

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
    """Build the readiness freshness CLI parser.

    Returns:
        Configured argument parser with ``status`` and ``write`` subcommands.
    """
    parser = argparse.ArgumentParser(
        description="Record and verify local PR readiness evidence for the current branch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser(
        "status",
        help="Check whether a recent successful readiness run exists for this branch and HEAD.",
    )
    status_parser.add_argument("--base-ref", default="origin/main")
    status_parser.add_argument("--base-sha")
    status_parser.add_argument("--branch")
    status_parser.add_argument("--head-sha")
    status_parser.add_argument("--stamp-path")
    status_parser.add_argument("--tree-state", choices=("clean", "dirty"))
    status_parser.add_argument(
        "--require-clean-tree",
        action="store_true",
        help="Require both the current worktree and recorded stamp to be clean.",
    )
    status_parser.add_argument("--max-age-hours", type=float, default=24.0)

    write_parser = subparsers.add_parser(
        "write",
        help="Record a successful readiness run for this branch and HEAD.",
    )
    write_parser.add_argument("--base-ref", default="origin/main")
    write_parser.add_argument("--base-sha")
    write_parser.add_argument("--branch")
    write_parser.add_argument("--head-sha")
    write_parser.add_argument("--stamp-path")
    write_parser.add_argument("--tree-state", choices=("clean", "dirty"))
    write_parser.add_argument(
        "--require-clean-tree",
        action="store_true",
        help="Fail instead of writing final readiness evidence from a dirty worktree.",
    )
    write_parser.add_argument("--status", default="passed")

    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    branch = args.branch or _current_branch()
    head_sha = args.head_sha or _head_sha()
    tree_state = args.tree_state or _tree_state()
    stamp_path = _resolve_stamp_path(args.stamp_path, branch)
    # Resolve the concrete base SHA the run validates against.  When the caller
    # passed an explicit SHA (e.g. a gate that just resolved it), honour it,
    # otherwise resolve the base_ref to its current commit.  A None result means
    # the ref is not visible locally; the drift check then falls back to the
    # base_ref string comparison and stays fail-open on resolution failure.
    base_sha = args.base_sha if args.base_sha else _resolve_base_sha(args.base_ref)

    if args.command == "write":
        payload = _write_stamp(
            path=stamp_path,
            branch=branch,
            base_ref=args.base_ref,
            base_sha=base_sha,
            head_sha=head_sha,
            tree_state=tree_state,
            status=args.status,
            require_clean_tree=args.require_clean_tree,
        )
        if not payload.get("ok", True):
            _json_dump(payload)
            return 2
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
        base_sha=base_sha,
        head_sha=head_sha,
        tree_state=tree_state,
        require_clean_tree=args.require_clean_tree,
        max_age_hours=args.max_age_hours,
    )
    _json_dump(payload)
    return 0 if is_fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
