#!/usr/bin/env python3
"""Base-sensitive gate: run snapshot/invariant tests when base-sensitive files change.

Prevents green-alone-red-together merge races by requiring PRs that touch
base-sensitive test surfaces to be up-to-date with main and to pass the
base_sensitive marker subset before merge.

Usage:
    # Check if a PR touches base-sensitive files and needs the gate:
    python scripts/dev/check_base_sensitive_gates.py --pr 123

    # Run the base_sensitive test subset (cheap, <2 min):
    python scripts/dev/check_base_sensitive_gates.py --run-subset

    # Full gate: check staleness + run subset:
    python scripts/dev/check_base_sensitive_gates.py --pr 123 --run-subset

Exit codes:
    0  Gate passed.
    1  Gate failed (stale base or subset failure).
    2  Error (could not determine).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the completed process."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=check,
        timeout=60,
    )


def _find_base_sensitive_test_files() -> list[Path]:
    """Find test files containing the base_sensitive marker."""
    matches: list[Path] = []
    ignored_dirs = {".git", ".venv", "venv", "build", "dist", "__pycache__", "third_party"}
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [directory for directory in dirs if directory not in ignored_dirs]
        for filename in files:
            if not (filename.startswith("test_") and filename.endswith(".py")):
                continue
            path = Path(root) / filename
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "base_sensitive" in text:
                matches.append(path)
    return sorted(matches)


def _get_pr_changed_files(pr_number: str) -> list[str] | None:
    """Return changed file paths for a PR via gh CLI, or None on error."""
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "diff",
                str(pr_number),
                "--name-only",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return None
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.TimeoutExpired, OSError):
        return None


def _branch_is_current_with_main(
    commit_or_branch: str, base_ref: str = "origin/main"
) -> bool | None:
    """Check whether a commit contains base_ref (merge-base test).

    Returns True if current, False if stale, None on error.
    """
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base_ref, commit_or_branch],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _report_gate_error(overall: dict[str, Any], message: str, *, as_json: bool) -> int:
    """Report an indeterminate gate state and return its fail-closed exit code."""
    overall["error"] = message
    print(f"Error: {message}", file=sys.stderr)
    if as_json:
        _json_dump(overall)
    return 2


def check_pr_touches_base_sensitive(pr_number: str) -> dict[str, Any]:
    """Determine if a PR changes any base-sensitive test files.

    Returns a dict with:
      - needs_gate: bool
      - changed_sensitive_files: list of relative paths
      - all_sensitive_files: full list of base-sensitive test files
    """
    sensitive_files = {str(p.relative_to(REPO_ROOT)) for p in _find_base_sensitive_test_files()}
    changed = _get_pr_changed_files(pr_number)
    if changed is None:
        return {
            "needs_gate": None,
            "error": "Could not fetch PR changed files",
            "changed_sensitive_files": [],
            "all_sensitive_files": sorted(sensitive_files),
        }

    changed_sensitive = sorted(f for f in changed if f in sensitive_files)
    return {
        "needs_gate": len(changed_sensitive) > 0,
        "changed_sensitive_files": changed_sensitive,
        "all_sensitive_files": sorted(sensitive_files),
    }


def run_base_sensitive_subset(timeout_minutes: int = 2) -> dict[str, Any]:
    """Run the base_sensitive test subset and return results.

    Returns a dict with:
      - passed: bool
      - duration_seconds: float
      - test_count: int | None
      - output: str (truncated)
    """
    import time

    start = time.monotonic()
    cmd = [
        sys.executable or "python3",
        "-m",
        "pytest",
        "-m",
        "base_sensitive",
        "-v",
        "--timeout",
        str(timeout_minutes * 60),
        "--tb=short",
        "--no-header",
        "-q",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            cwd=str(REPO_ROOT),
            check=False,
        )
        elapsed = time.monotonic() - start
        # Parse test count from output
        test_count: int | None = None
        for line in result.stdout.splitlines()[-3:]:
            if "passed" in line:
                try:
                    # Format like "5 passed in 12.3s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            test_count = int(parts[i - 1])
                            break
                    break
                except (ValueError, IndexError):
                    pass

        passed = result.returncode == 0
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"

        return {
            "passed": passed,
            "duration_seconds": round(elapsed, 1),
            "test_count": test_count,
            "output": output,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "duration_seconds": timeout_minutes * 60,
            "test_count": None,
            "output": f"base_sensitive subset timed out after {timeout_minutes} min",
        }


def _json_dump(data: dict[str, Any]) -> None:
    """Print a JSON payload with stable formatting."""
    print(json.dumps(data, indent=2, sort_keys=True))


def _handle_list_files(args: argparse.Namespace) -> int:
    """Handle --list-files mode."""
    sensitive = _find_base_sensitive_test_files()
    if args.as_json:
        _json_dump(
            {
                "base_sensitive_files": [str(p.relative_to(REPO_ROOT)) for p in sensitive],
                "count": len(sensitive),
            }
        )
    else:
        print(f"Base-sensitive test files ({len(sensitive)}):")
        for p in sensitive:
            print(f"  {p.relative_to(REPO_ROOT)}")
    return 0


def _check_pr_gate_staleness(
    pr_number: str,
    pr_check: dict[str, Any],
    overall: dict[str, Any],
    *,
    as_json: bool,
) -> int | None:
    """Check PR gate staleness. Returns exit code or None to continue."""
    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(pr_number), "--json", "headRefOid", "--jq", ".headRefOid"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _report_gate_error(overall, f"Could not fetch PR head SHA: {exc}", as_json=as_json)

    if result.returncode != 0:
        return _report_gate_error(
            overall,
            f"Could not fetch PR head SHA via gh CLI (exit code {result.returncode})",
            as_json=as_json,
        )

    sha = result.stdout.strip()
    if not sha:
        return _report_gate_error(overall, "PR head SHA is empty", as_json=as_json)

    is_current = _branch_is_current_with_main(sha)
    if is_current is None:
        return _report_gate_error(
            overall,
            "Could not determine whether the PR head contains origin/main",
            as_json=as_json,
        )
    if is_current is False:
        print(
            f"PR #{pr_number}: base-sensitive files changed but commit "
            f"'{sha[:8]}' is stale relative to origin/main.\n"
            f"Changed sensitive files:\n"
            f"  "
            + "\n  ".join(pr_check["changed_sensitive_files"])
            + "\nRun: gh pr edit <number> --base main && "
            "gh pr update-branch <number>",
            file=sys.stderr,
        )
        overall["stale"] = True
        overall["sha"] = sha
        if as_json:
            _json_dump(overall)
        return 1
    return None


def _run_subset_and_check(
    overall: dict[str, Any],
    *,
    timeout_minutes: int,
    as_json: bool,
) -> int:
    """Run base_sensitive subset and check result."""
    result = run_base_sensitive_subset(timeout_minutes=timeout_minutes)
    overall["subset_run"] = result
    if not result["passed"]:
        print("base_sensitive subset FAILED:")
        print(result["output"])
        if as_json:
            _json_dump(overall)
        return 1
    if as_json:
        overall["gate_passed"] = True
        _json_dump(overall)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the base-sensitive gate."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pr",
        metavar="NUMBER",
        help="GitHub PR number to check for base-sensitive file changes",
    )
    parser.add_argument(
        "--run-subset",
        action="store_true",
        help="Run the base_sensitive test subset",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit JSON output",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=2,
        help="Timeout for base_sensitive subset in minutes (default: 2)",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List all base-sensitive test files and exit",
    )
    args = parser.parse_args(argv)

    if args.list_files:
        return _handle_list_files(args)

    overall: dict[str, Any] = {}

    if args.pr:
        pr_check = check_pr_touches_base_sensitive(args.pr)
        overall["pr_check"] = pr_check
        if pr_check.get("needs_gate") is None:
            return 2

        if not pr_check["needs_gate"]:
            print(f"PR #{args.pr}: no base-sensitive files changed; gate not required.")
            if args.as_json:
                _json_dump({"gate_required": False, "pr": args.pr})
            return 0

        stale_exit = _check_pr_gate_staleness(
            args.pr,
            pr_check,
            overall,
            as_json=args.as_json,
        )
        if stale_exit is not None:
            return stale_exit

    if args.run_subset or (args.pr and pr_check.get("needs_gate", False)):
        return _run_subset_and_check(
            overall,
            timeout_minutes=args.timeout_minutes,
            as_json=args.as_json,
        )

    if args.as_json:
        overall["gate_passed"] = True
        _json_dump(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
