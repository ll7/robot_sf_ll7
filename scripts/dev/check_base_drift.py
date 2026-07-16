#!/usr/bin/env python3
"""Lightweight base-drift check for the PR-readiness gate (issue #5782).

The costly readiness lanes (pytest, coverage) validate a branch against a
specific base commit.  When ``origin/main`` advances during that long run, the
base the run validated against is no longer current, so any readiness stamp
recorded at the end is immediately stale relative to the moving base.

This check supports the readiness gate's base-drift handling for issue #5782:
the gate captures the concrete base SHA before the expensive lanes start and then
invokes this check immediately before recording the readiness stamp, to fail
closed when ``origin/main`` moved *during* those lanes.

When the base *has* drifted but the drifted commits touch **none** of the
files this PR changes, the drift is unrelated to the changed paths and a
reviewable reuse path is recommended (exit 0, ``reuse_recommended``).  When the
drift intersects the PR's changed files, the check fails closed (exit 1) and
names the exact base SHA that must be revalidated.

Exit codes:
    0  No base drift, or drift unrelated to changed paths (reuse recommended).
    1  Base drift affects the PR's changed paths; revalidation required.
    2  Could not resolve the base ref or compute the drift (indeterminate).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_git(args: list[str]) -> tuple[int, str]:
    """Return (returncode, stdout) for a git command; never raise on failure."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip()


def _resolve_base_sha(base_ref: str) -> tuple[str | None, str | None]:
    """Resolve *base_ref* to a concrete commit SHA.

    Returns:
        (sha, None) on success, or (None, error_message) when the ref does not
        resolve to a local commit.
    """
    rc, stdout = _run_git(["rev-parse", "--verify", "--quiet", f"{base_ref}^{{commit}}"])
    if rc != 0:
        return None, f"base ref {base_ref!r} does not resolve to a local commit"
    sha = stdout.strip()
    if not sha:
        return None, f"base ref {base_ref!r} resolved to an empty SHA"
    return sha, None


def _merge_base(commit_a: str, commit_b: str) -> tuple[str | None, str | None]:
    """Return the merge base of two commits, or (None, error) on failure."""
    rc, stdout = _run_git(["merge-base", commit_a, commit_b])
    if rc != 0:
        return None, "could not compute merge-base"
    sha = stdout.strip()
    if not sha:
        return None, "merge-base resolved to an empty SHA"
    return sha, None


def _diff_name_only(base: str, tip: str) -> tuple[list[str] | None, str | None]:
    """Return the sorted, unique changed paths between *base* and *tip*.

    Returns:
        (paths, None) on success, or (None, error) when the diff cannot be read.
    """
    rc, stdout = _run_git(["diff", "--name-only", base, tip])
    if rc != 0:
        return None, f"could not diff {base}..{tip}"
    paths = [line.strip() for line in stdout.splitlines() if line.strip()]
    return sorted(set(paths)), None


def _pr_changed_files(
    validated_base_sha: str,
    *,
    changed_files_override: list[str] | None = None,
) -> tuple[list[str] | None, str | None]:
    """Return the files this PR changes relative to the validated base.

    Uses the three-dot diff against HEAD (changes on the HEAD side since the
    merge base with the validated base), matching the changed-file scope the
    readiness gate already uses.  When *changed_files_override* is provided it
    is used verbatim (tests and callers that precomputed the set).
    """
    if changed_files_override is not None:
        return sorted(set(changed_files_override)), None
    merge_base, err = _merge_base(validated_base_sha, "HEAD")
    if err is not None:
        return None, err
    return _diff_name_only(merge_base, "HEAD")


def check_base_drift(
    *,
    base_ref: str,
    validated_base_sha: str | None,
    changed_files: list[str] | None = None,
) -> dict[str, Any]:
    """Compare the validated base SHA against the current base and classify drift.

    Returns:
        A result dict with keys including ``status`` (one of ``current``,
        ``reuse_recommended``, ``revalidate_required``) and ``exit_code``.
    """
    result: dict[str, Any] = {
        "base_ref": base_ref,
        "validated_base_sha": validated_base_sha,
    }

    current_base_sha, err = _resolve_base_sha(base_ref)
    if current_base_sha is None:
        result["status"] = "indeterminate"
        result["error"] = err
        result["exit_code"] = 2
        return result
    result["current_base_sha"] = current_base_sha

    if validated_base_sha is None or validated_base_sha == current_base_sha:
        result["status"] = "current"
        result["exit_code"] = 0
        return result

    # Base has drifted between validation start and now.
    drift_files, err = _diff_name_only(validated_base_sha, current_base_sha)
    if drift_files is None:
        result["status"] = "indeterminate"
        result["error"] = err
        result["exit_code"] = 2
        return result
    result["drift_file_count"] = len(drift_files)
    result["drift_files"] = drift_files

    pr_files, err = _pr_changed_files(validated_base_sha, changed_files_override=changed_files)
    if pr_files is None:
        result["status"] = "indeterminate"
        result["error"] = err
        result["exit_code"] = 2
        return result
    result["pr_changed_file_count"] = len(pr_files)
    result["pr_changed_files"] = pr_files

    affected = sorted(set(drift_files) & set(pr_files))
    result["affected_file_count"] = len(affected)
    result["affected_files"] = affected

    if affected:
        result["status"] = "revalidate_required"
        result["exit_code"] = 1
        result["message"] = (
            f"Base drifted to {current_base_sha[:8]}; revalidate against "
            f"{base_ref} (was {validated_base_sha[:8]}). Drift touches "
            f"{len(affected)} PR-changed file(s): {', '.join(affected[:10])}"
        )
    else:
        result["status"] = "reuse_recommended"
        result["exit_code"] = 0
        result["message"] = (
            f"Base drifted to {current_base_sha[:8]} but drift is unrelated to the "
            f"{len(pr_files)} PR-changed file(s); reuse the passing run after noting "
            f"the base moved from {validated_base_sha[:8]}."
        )
    return result


def _read_changed_files_file(path: str) -> list[str]:
    """Read one changed file path per line from *path*."""
    return [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _emit(result: dict[str, Any], *, as_json: bool) -> None:
    """Print the result and return via sys.exit with the result's exit code."""
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = result.get("status")
        if status == "current":
            print(f"Base {result['base_ref']} is current at {result['current_base_sha'][:8]}.")
        elif status in ("reuse_recommended", "revalidate_required"):
            print(result["message"])
        else:
            print(f"Error: {result.get('error', 'indeterminate')}", file=sys.stderr)
    raise SystemExit(result["exit_code"])


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the base-drift check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base ref to compare against (default: origin/main).",
    )
    parser.add_argument(
        "--validated-base-sha",
        default=None,
        help="Concrete base SHA the readiness run started validating against.",
    )
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Path to a file with one PR-changed file path per line (otherwise computed via git).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    changed_files = _read_changed_files_file(args.changed_files) if args.changed_files else None
    result = check_base_drift(
        base_ref=args.base_ref,
        validated_base_sha=args.validated_base_sha,
        changed_files=changed_files,
    )
    _emit(result, as_json=args.json)


if __name__ == "__main__":
    main()
