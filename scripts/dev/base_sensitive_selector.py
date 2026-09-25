#!/usr/bin/env python3
"""Deterministic selector for the repository's base-sensitive test surfaces."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

SELECTOR_VERSION = "pytest-marker-files.v2"
BASELINE_SCHEMA = "base-sensitive-baseline.v1"
BASE_SENSITIVE = "base_sensitive"
ORDINARY = "ordinary"
UNKNOWN = "unknown"
DEFAULT_BASE_REF = "origin/main"
DEFAULT_TIMEOUT_SECONDS = 120
_MAX_OUTPUT_CHARS = 4000


def _normalize_repo_path(path: str) -> str:
    """Normalize only repeated ``./`` prefixes without changing hidden directories."""
    normalized = path
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _tracked_test_files(repo_root: Path) -> list[str]:
    """Return tracked ``test_*.py`` repository paths via the Git index.

    Tracked-file enumeration is repository-bound: ignored nested worktrees,
    output caches, and other untracked copies (for example ``.emdash/``,
    ``.worktrees/``, and ``output/``) can never enter the selection. The call
    fails closed on Git errors instead of silently widening the selection.
    """
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--", "*.py"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "base-sensitive selection requires the Git tracked-file index; "
            f"git ls-files failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return [
        tracked
        for tracked in completed.stdout.splitlines()
        if PurePosixPath(tracked).name.startswith("test_") and tracked.endswith(".py")
    ]


def find_base_sensitive_test_files(repo_root: Path) -> list[str]:
    """Return repository-relative test files declaring the ``base_sensitive`` marker.

    The marker is the deliberately small, explicit contract from issue #5559.  Keeping the
    selector in one module lets the gate and queue policy use identical file classification.
    Selection is repository-bound: only Git-tracked ``test_*.py`` files are inspected, so
    ignored nested worktree and cache copies never contaminate the result (issue #8025).
    """
    matches: list[str] = []
    for tracked in _tracked_test_files(repo_root):
        path = repo_root / tracked
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if BASE_SENSITIVE in text:
            matches.append(tracked)
    return sorted(matches)


def classify_changed_files(
    changed_files: Sequence[str] | None,
    *,
    sensitive_files: Sequence[str],
) -> dict[str, Any]:
    """Classify a changed-file list using the explicit marker-file selector.

    ``None`` means the changed-file inventory was unavailable and therefore cannot be treated as
    an ordinary PR.  A present inventory with no marker-file intersection is ordinary and may use
    the immediate current-base compare-and-swap path at merge time.
    """
    normalized_sensitive = {_normalize_repo_path(str(path)) for path in sensitive_files}
    if changed_files is None:
        return {
            "status": UNKNOWN,
            "selector": SELECTOR_VERSION,
            "changed_files": [],
            "changed_sensitive_files": [],
            "all_sensitive_files": sorted(normalized_sensitive),
            "reason": "changed_file_inventory_unavailable",
        }

    normalized_changed = sorted(
        {_normalize_repo_path(str(path)) for path in changed_files if str(path)}
    )
    changed_sensitive = sorted(path for path in normalized_changed if path in normalized_sensitive)
    return {
        "status": BASE_SENSITIVE if changed_sensitive else ORDINARY,
        "selector": SELECTOR_VERSION,
        "changed_files": normalized_changed,
        "changed_sensitive_files": changed_sensitive,
        "all_sensitive_files": sorted(normalized_sensitive),
        "reason": (
            "changed_files_intersect_marker_selector"
            if changed_sensitive
            else "no_changed_file_intersects_marker_selector"
        ),
    }


def _resolve_base_sha(repo_root: Path, base_ref: str) -> str:
    """Resolve ``base_ref`` to a commit SHA, failing closed on Git errors."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            f"{base_ref}^{{commit}}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    sha = result.stdout.strip()
    if result.returncode != 0 or not sha:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"could not resolve base ref {base_ref!r}: {detail}")
    return sha


def _truncate_output(output: str) -> str:
    """Keep command output useful without allowing a failed baseline to flood callers."""
    if len(output) <= _MAX_OUTPUT_CHARS:
        return output
    return output[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"


def _clean_checkout_problem(repo_root: Path) -> dict[str, Any] | None:
    """Return a fail-closed report when the checkout contains staged, dirty, or untracked files."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "status": "blocked",
            "reason": "checkout_status_unavailable",
            "error": str(exc),
        }
    if result.returncode != 0:
        return {
            "status": "blocked",
            "reason": "checkout_status_unavailable",
            "error": result.stderr.strip() or f"git status exited with code {result.returncode}",
        }
    dirty_status = [line for line in result.stdout.splitlines() if line.strip()]
    if dirty_status:
        return {
            "status": "blocked",
            "reason": "checkout_dirty",
            "dirty_paths": [line[3:] if len(line) > 3 else line for line in dirty_status],
            "error": "baseline execution requires a clean checkout; changes were found",
        }
    return None


def _parse_junit_outcomes(report_path: Path) -> dict[str, int]:
    """Read aggregate pytest outcomes from its structured JUnit report."""
    root = ET.parse(report_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    if not suites:
        raise ValueError("pytest JUnit report has no testsuite entries")
    counts = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    if passed < 0:
        raise ValueError("pytest JUnit report contains inconsistent outcome counts")
    return {
        "test_count": counts["tests"],
        "passed": passed,
        "failed": counts["failures"],
        "errors": counts["errors"],
        "skipped": counts["skipped"],
    }


def _pytest_result_report(
    *,
    command: list[str],
    result: subprocess.CompletedProcess[str],
    outcomes: dict[str, int],
    duration: float,
) -> dict[str, Any]:
    """Classify pytest results, requiring at least one passing test for success."""
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    report: dict[str, Any] = {
        "pytest_command": command,
        "duration_seconds": round(duration, 3),
        "test_result": {
            "returncode": result.returncode,
            **outcomes,
            "stdout": _truncate_output(stdout),
            "stderr": _truncate_output(stderr),
        },
    }
    if outcomes["test_count"] == 0:
        report.update(
            {
                "status": "blocked",
                "reason": "no_tests_executed",
                "error": "pytest produced no collected test outcomes for the selected marker suite",
            }
        )
    elif outcomes["failed"] > 0 or outcomes["errors"] > 0 or result.returncode != 0:
        report.update({"status": "failed", "reason": "selected_tests_failed"})
    elif outcomes["passed"] == 0:
        report.update(
            {
                "status": "blocked",
                "reason": "no_tests_passed",
                "error": "pytest completed without any passing base-sensitive tests",
            }
        )
    else:
        report.update({"status": "passed", "reason": "selected_tests_passed"})
    return report


def _base_test_environment(base_root: Path, caller_root: Path) -> dict[str, str]:
    """Run pytest with base-commit imports, excluding caller-checkout source paths."""
    environment = os.environ.copy()
    resolved_base_root = base_root.resolve()
    resolved_caller_root = caller_root.resolve()
    python_paths = [str(resolved_base_root)]
    fast_pysf = resolved_base_root / "fast-pysf"
    if fast_pysf.is_dir():
        python_paths.append(str(fast_pysf))

    for entry in environment.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        if not path.is_absolute():
            path = resolved_base_root / path
        resolved_path = path.resolve()
        if resolved_path == resolved_caller_root or resolved_caller_root in resolved_path.parents:
            continue
        if str(resolved_path) not in python_paths:
            python_paths.append(str(resolved_path))

    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def _run_selected_tests(
    root: Path,
    caller_root: Path,
    selected: list[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Execute selected marker tests and require a parseable outcome report."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        BASE_SENSITIVE,
        "--no-header",
        "--tb=short",
        "-q",
        "--timeout",
        str(timeout_seconds),
        *selected,
    ]
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="base-sensitive-baseline-") as temp_dir:
            report_path = Path(temp_dir) / "pytest-junit.xml"
            pytest_command = [*command, "--junitxml", str(report_path)]
            try:
                result = subprocess.run(
                    pytest_command,
                    cwd=str(root),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=_base_test_environment(root, caller_root),
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "status": "blocked",
                    "reason": "pytest_timeout",
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "pytest_command": pytest_command,
                    "test_result": {
                        "returncode": None,
                        "stdout": _truncate_output(str(exc.stdout or "")),
                        "stderr": _truncate_output(str(exc.stderr or "")),
                    },
                    "error": f"pytest timed out after {timeout_seconds} seconds",
                }
            try:
                outcomes = _parse_junit_outcomes(report_path)
            except (OSError, ET.ParseError, ValueError) as exc:
                return {
                    "status": "blocked",
                    "reason": "pytest_report_unavailable",
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "pytest_command": pytest_command,
                    "test_result": {
                        "returncode": result.returncode,
                        "stdout": _truncate_output(result.stdout or ""),
                        "stderr": _truncate_output(result.stderr or ""),
                    },
                    "error": f"could not read pytest's structured outcome report: {exc}",
                }
            return _pytest_result_report(
                command=pytest_command,
                result=result,
                outcomes=outcomes,
                duration=time.monotonic() - started,
            )
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "status": "blocked",
            "reason": "pytest_unavailable",
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": str(exc),
        }


def _create_base_worktree(repo_root: Path, base_sha: str, path: Path) -> dict[str, Any] | None:
    """Materialize the exact commit into an isolated temporary worktree."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "--quiet",
                "--detach",
                str(path),
                base_sha,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "status": "blocked",
            "reason": "base_worktree_unavailable",
            "error": str(exc),
        }
    if result.returncode != 0:
        return {
            "status": "blocked",
            "reason": "base_worktree_unavailable",
            "error": result.stderr.strip()
            or f"git worktree add exited with code {result.returncode}",
        }
    return None


def _remove_base_worktree(repo_root: Path, path: Path) -> str | None:
    """Remove the temporary worktree and return a cleanup error, if any."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return str(exc)
    if result.returncode != 0:
        return result.stderr.strip() or f"git worktree remove exited with code {result.returncode}"
    return None


def _run_base_sensitive_suite(
    repo_root: Path,
    base_sha: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Select and run tests from a pristine worktree materialized at ``base_sha``."""
    report: dict[str, Any] = {
        "status": "blocked",
        "selected_test_files": [],
        "selection_count": 0,
    }
    with tempfile.TemporaryDirectory(prefix="base-sensitive-checkout-") as temp_dir:
        base_worktree = Path(temp_dir) / "checkout"
        creation_problem = _create_base_worktree(repo_root, base_sha, base_worktree)
        if creation_problem:
            report.update(creation_problem)
            return report

        report["test_source"] = "materialized_commit_worktree"
        report["test_source_sha"] = base_sha
        try:
            try:
                selected = find_base_sensitive_test_files(base_worktree)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
                report.update(
                    {
                        "status": "blocked",
                        "reason": "selection_unavailable",
                        "error": str(exc),
                    }
                )
            else:
                report["selected_test_files"] = selected
                report["selection_count"] = len(selected)
                if not selected:
                    report.update(
                        {
                            "status": "blocked",
                            "reason": "selection_empty",
                            "error": "no tracked test files declare the base_sensitive marker",
                        }
                    )
                else:
                    missing = [path for path in selected if not (base_worktree / path).is_file()]
                    if missing:
                        report.update(
                            {
                                "status": "blocked",
                                "reason": "selection_unavailable",
                                "missing_test_files": missing,
                                "error": "selected test files are unavailable",
                            }
                        )
                    else:
                        report.update(
                            _run_selected_tests(
                                base_worktree,
                                repo_root,
                                selected,
                                timeout_seconds,
                            )
                        )
        finally:
            cleanup_error = _remove_base_worktree(repo_root, base_worktree)
        if cleanup_error:
            report.update(
                {
                    "status": "blocked",
                    "reason": "base_worktree_cleanup_failed",
                    "cleanup_error": cleanup_error,
                }
            )
    return report


def _baseline_report(
    *,
    repo_root: Path,
    base_ref: str,
    selected_test_files: list[str],
) -> dict[str, Any]:
    """Build the stable report envelope shared by success and fail-closed results."""
    return {
        "schema": BASELINE_SCHEMA,
        "selector": SELECTOR_VERSION,
        "repository_root": str(repo_root),
        "base_ref": base_ref,
        "branch": None,
        "observed_base_sha": None,
        "observed_head_sha": None,
        "selected_test_files": selected_test_files,
        "selection_count": len(selected_test_files),
    }


def _record_base_observation(report: dict[str, Any], root: Path, base_ref: str) -> bool:
    """Record the requested baseline and require the checkout to be at that commit."""
    try:
        branch = subprocess.run(
            ["git", "-C", str(root), "symbolic-ref", "--quiet", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        report.update(
            {
                "status": "blocked",
                "reason": "checkout_branch_unavailable",
                "error": str(exc),
            }
        )
        return False
    if branch.returncode != 0 or not branch.stdout.strip():
        report.update(
            {
                "status": "blocked",
                "reason": "checkout_detached",
                "error": "baseline execution requires a branch-attached checkout",
            }
        )
        return False
    report["branch"] = branch.stdout.strip()

    checkout_problem = _clean_checkout_problem(root)
    if checkout_problem:
        report.update(checkout_problem)
        return False

    try:
        report["observed_base_sha"] = _resolve_base_sha(root, base_ref)
        report["observed_head_sha"] = _resolve_base_sha(root, "HEAD")
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        report.update(
            {
                "status": "blocked",
                "reason": "base_ref_unavailable",
                "error": str(exc),
            }
        )
        return False

    if report["observed_base_sha"] != report["observed_head_sha"]:
        report.update(
            {
                "status": "blocked",
                "reason": "checkout_not_at_base_ref",
                "error": (
                    "the checkout HEAD does not match the requested baseline ref; "
                    "run the command from the exact branch-attached baseline worktree"
                ),
            }
        )
        return False
    return True


def run_base_sensitive_baseline(
    repo_root: Path,
    *,
    base_ref: str = DEFAULT_BASE_REF,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Select and execute the base-sensitive suite with an observable result.

    The returned report is deliberately useful for both human and machine callers. A report is
    successful only when pytest returns zero, JUnit reports no failures/errors, and at least one
    test passes. Selection and execution use a temporary checkout of the exact baseline commit.
    """
    try:
        root = repo_root.resolve()
    except (OSError, RuntimeError) as exc:
        report = _baseline_report(
            repo_root=repo_root,
            base_ref=base_ref,
            selected_test_files=[],
        )
        report.update(
            {
                "status": "blocked",
                "reason": "repository_unavailable",
                "error": str(exc),
            }
        )
        return report
    report = _baseline_report(
        repo_root=root,
        base_ref=base_ref,
        selected_test_files=[],
    )

    if not _record_base_observation(report, root, base_ref):
        return report

    base_sha = report["observed_base_sha"]
    report.update(_run_base_sensitive_suite(root, base_sha, timeout_seconds))
    return report


def _print_report(report: dict[str, Any], *, as_json: bool) -> None:
    """Print a baseline report in stable JSON or concise human-readable form."""
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"Base-sensitive baseline: {report.get('status', 'blocked')}")
    print(f"Base ref: {report['base_ref']}")
    print(f"Observed base SHA: {report.get('observed_base_sha') or 'unavailable'}")
    selected = report.get("selected_test_files", [])
    print(f"Selected test files ({len(selected)}):")
    for path in selected:
        print(f"  {path}")
    command = report.get("pytest_command")
    if command:
        print("Pytest command: " + " ".join(str(part) for part in command))
    test_result = report.get("test_result")
    if isinstance(test_result, dict):
        print(
            "Test result: "
            f"returncode={test_result.get('returncode')} "
            f"outcomes={test_result.get('test_count')}"
        )
        output = "\n".join(
            part for part in (test_result.get("stdout", ""), test_result.get("stderr", "")) if part
        )
        if output:
            print(output)
    if report.get("error"):
        print(f"Error: {report['error']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Run the marker-selected baseline suite and fail closed on unusable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="checkout whose tracked marker files and tests should be run (default: cwd)",
    )
    parser.add_argument(
        "--base-ref",
        default=DEFAULT_BASE_REF,
        help="Git commit/ref to record as the observed baseline (default: origin/main)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"pytest timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit a machine-readable baseline report",
    )
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    report = run_base_sensitive_baseline(
        args.repo_root,
        base_ref=args.base_ref,
        timeout_seconds=args.timeout_seconds,
    )
    _print_report(report, as_json=args.as_json)
    if report["status"] == "passed":
        return 0
    if report["status"] == "failed":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
