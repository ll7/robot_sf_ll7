#!/usr/bin/env python3
"""Validate the JSON/exit-code contract of the optional dependency preflight."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

SCHEMA = "robot_sf.worktree_optional_deps.v1"
_EXPECTED_EXIT_CODES = {
    "ready": 0,
    "check_failed": 1,
    "missing_optional": 2,
}


def validate_report(  # noqa: C901, PLR0912 - explicit fail-closed contract branches.
    report: Any, *, observed_exit_code: int
) -> tuple[int, list[str]]:
    """Return the validated semantic exit code and deterministic contract errors."""
    errors: list[str] = []
    if not isinstance(report, dict):
        return 1, ["report must be a JSON object"]
    if report.get("schema") != SCHEMA:
        errors.append("schema must be robot_sf.worktree_optional_deps.v1")
    if report.get("profile") != "all-extras":
        errors.append("profile must be all-extras")
    status = report.get("status")
    status_is_known = isinstance(status, str) and status in _EXPECTED_EXIT_CODES
    if not status_is_known:
        errors.append("status must be ready, check_failed, or missing_optional")
    reported_exit_code = report.get("exit_code")
    if isinstance(reported_exit_code, bool) or not isinstance(reported_exit_code, int):
        errors.append("exit_code must be an integer")
    elif reported_exit_code != observed_exit_code:
        errors.append(
            f"reported exit_code {reported_exit_code} disagrees with observed exit "
            f"{observed_exit_code}"
        )
    if observed_exit_code not in _EXPECTED_EXIT_CODES.values():
        errors.append(f"observed exit code {observed_exit_code} is not recognized")
    if status_is_known and observed_exit_code in _EXPECTED_EXIT_CODES.values():
        expected_exit_code = _EXPECTED_EXIT_CODES[status]
        if observed_exit_code != expected_exit_code:
            errors.append(f"status {status} requires exit {expected_exit_code}")
    if report.get("project_imports_performed") is not False:
        errors.append("project_imports_performed must be false")

    missing_optional = report.get("missing_optional")
    if not isinstance(missing_optional, list) or not all(
        isinstance(module, str) for module in missing_optional
    ):
        errors.append("missing_optional must be a list of strings")
        missing_optional = []
    check_failures = report.get("check_failures")
    if not isinstance(check_failures, list) or not all(
        isinstance(module, str) for module in check_failures
    ):
        errors.append("check_failures must be a list of strings")
        check_failures = []
    if status == "ready" and (missing_optional or check_failures):
        errors.append("ready reports cannot contain missing_optional or check_failures")
    elif status == "missing_optional" and (not missing_optional or check_failures):
        errors.append("missing_optional reports require missing modules and no probe failures")
    elif status == "check_failed" and not check_failures:
        errors.append("check_failed reports require at least one probe failure")

    if errors:
        return 1, errors
    return _EXPECTED_EXIT_CODES[str(status)], []


def main(argv: list[str] | None = None) -> int:
    """Validate a JSON report read from stdin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        report = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            json.dumps({"status": "invalid", "errors": [f"malformed JSON: {exc}"]}), file=sys.stderr
        )
        return 1

    exit_code, errors = validate_report(report, observed_exit_code=args.observed_exit_code)
    if errors:
        print(json.dumps({"status": "invalid", "errors": errors}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
