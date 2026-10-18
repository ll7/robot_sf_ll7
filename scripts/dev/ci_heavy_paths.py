#!/usr/bin/env python3
"""Select expensive PR CI lanes from a complete changed-file list.

An unreadable or empty list is an error: missing path evidence must never skip
tests. Main, merge-group, and manual CI do not call this selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LANES = ("compat_macos", "examples_smoke", "notebooks_smoke", "xdist_scratch_isolation")
SHARED = (
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/ci.yml",
    ".github/actions/setup-ci-python/",
    "scripts/dev/ci_driver.sh",
)
PATHS = {
    "compat_macos": SHARED + ("robot_sf/", "fast-pysf/", "tests/", "scripts/", "configs/"),
    "examples_smoke": SHARED
    + ("examples/", "robot_sf/", "tests/examples/", "scripts/validation/run_examples_smoke.py"),
    "notebooks_smoke": SHARED
    + ("notebooks/", "robot_sf/", "tests/notebooks/", "scripts/validation/run_notebooks_smoke.py"),
    "xdist_scratch_isolation": SHARED
    + (
        "robot_sf/",
        "tests/conftest.py",
        "tests/test_runner_policy_timeout.py",
        "tests/tools/test_release_evidence_gate.py",
        "tests/benchmark/test_scenario_criticality_optimization.py",
        "tests/analysis/test_recenter_topology_panels_issue_2227.py",
        "tests/validation/test_evaluate_predictive_planner_smoke.py",
        "tests/validation/test_predictive_success_campaign_tags.py",
    ),
}


def select_lanes(files: list[str]) -> dict[str, bool]:
    """Return the heavy lanes that a complete, nonempty PR file set requires."""
    if not files or any(
        not path
        or path.strip() != path
        or path.startswith(("/", "./", "../"))
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        for path in files
    ):
        raise ValueError("changed-file list is empty or contains an unsafe path")
    return {
        lane: any(
            path == pattern or (pattern.endswith("/") and path.startswith(pattern))
            for path in files
            for pattern in patterns
        )
        for lane, patterns in PATHS.items()
    }


def main() -> int:
    """Write selected lane flags for GitHub Actions job outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files", required=True, type=Path)
    parser.add_argument("--files-status", required=True, type=Path)
    parser.add_argument("--github-output", required=True, type=Path)
    args = parser.parse_args()
    files = args.changed_files.read_text(encoding="utf-8").splitlines()
    selected = select_lanes(files)
    statuses = [
        line.split("\t", 1) for line in args.files_status.read_text(encoding="utf-8").splitlines()
    ]
    if len(statuses) != len(files) or any(
        len(row) != 2 or row[1] != path for row, path in zip(statuses, files, strict=True)
    ):
        raise ValueError("status list does not match the complete changed-file list")
    # GitHub's filename field contains the destination of a rename. The source
    # may have been relevant to any lane. Unknown statuses are also unsafe to
    # classify as irrelevant, so run every lane for either case.
    known_statuses = {"added", "modified", "removed", "renamed", "copied", "changed", "unchanged"}
    if any(row[0] == "renamed" or row[0] not in known_statuses for row in statuses):
        selected = dict.fromkeys(LANES, True)
    selected["compat_os"] = (
        '["ubuntu-latest","macos-latest"]' if selected["compat_macos"] else '["ubuntu-latest"]'
    )
    with args.github_output.open("a", encoding="utf-8") as output:
        for key, value in selected.items():
            output.write(f"{key}={value if isinstance(value, str) else str(value).lower()}\n")
    print(json.dumps(selected, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
