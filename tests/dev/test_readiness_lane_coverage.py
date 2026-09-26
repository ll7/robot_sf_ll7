"""Readiness lane coverage must not silently omit a test root (issue #9754).

``scripts/dev/pr_ready_check.sh`` selects test paths from a fixed core list, the
shared optional allowlist, and the changed test files. A ``tests/*`` root that is
in none of those is never executed by a readiness run, so a green readiness
report silently omits it and a PR body can overstate what was proven.

These tests assert the coverage *property* rather than one implementation, so a
future edit that adds a root to a lane, or drops a lane, fails here instead of
only surfacing as a surprising CI failure.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DEV = REPO_ROOT / "scripts" / "dev"
RUN_TESTS_PARALLEL = SCRIPTS_DEV / "run_tests_parallel.sh"
PR_READY_CHECK = SCRIPTS_DEV / "pr_ready_check.sh"
ALLOWLIST = REPO_ROOT / "tests" / "support" / "optional_test_allowlist.txt"
TESTS_ROOT = REPO_ROOT / "tests"


def _core_test_paths() -> set[str]:
    """Return the core lane's fixed path list from the runner script."""
    text = RUN_TESTS_PARALLEL.read_text(encoding="utf-8")
    match = re.search(r"^core_test_paths=\(\n(.*?)^\)", text, re.S | re.M)
    assert match is not None, "core_test_paths array not found in run_tests_parallel.sh"
    paths = set()
    for line in match.group(1).splitlines():
        entry = line.strip()
        if entry:
            paths.add(entry.rstrip("/"))
    return paths


def _optional_roots() -> set[str]:
    """Return the optional lane's covered roots from the shared allowlist.

    Only allowlist entries written as directory patterns (trailing slash) cover a
    whole ``tests/<root>`` subtree. An individual file entry covers only itself, so
    one optional file inside a root must not be read as covering that root.
    """
    roots: set[str] = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if not entry.endswith("/"):
            continue
        parts = entry.rstrip("/").split("/")
        # A directory pattern covers its whole subtree, so the covered top-level
        # root is tests/<first segment below tests/> for both tests/benchmark/ and
        # the deeper tests/unit/benchmark/.
        if entry.startswith("tests/") and len(parts) > 1:
            roots.add(f"tests/{parts[1]}")
    return roots


def _extended_roots() -> set[str]:
    """Return the roots pr_ready_check.sh runs in its extended lane.

    Returns an empty set when no extended lane exists yet, so the coverage test
    reports which roots are actually uncovered instead of failing on a missing
    marker string.
    """
    text = PR_READY_CHECK.read_text(encoding="utf-8")
    match = re.search(r"for pr_ready_candidate_root in ([^\n]+); do", text)
    if match is None:
        return set()
    return set(re.findall(r"tests/[A-Za-z0-9_]+", match.group(1)))


def _on_disk_roots() -> set[str]:
    """Return every top-level ``tests/<name>`` directory that holds tests."""
    roots = set()
    for child in sorted(TESTS_ROOT.iterdir()):
        if not child.is_dir() or child.name == "support":
            continue
        if any(child.glob("test_*.py")) or any(child.glob("*_test.py")):
            roots.add(f"tests/{child.name}")
    return roots


# The roots issue #9754 names as omitted from the readiness lanes. Scoped to these
# rather than every on-disk root: other roots are executed by separate CI phases
# (scenario-validation, smoke, examples), and asserting on all of them here would
# claim a readiness-lane contract the repository does not make.
ISSUE_NAMED_ROOTS = ("tests/validation", "tests/maps", "tests/benchmark", "tests/training")


def test_issue_named_test_roots_are_covered_by_a_readiness_lane() -> None:
    """Each root from issue #9754 must be reachable by some pr_ready_check lane."""
    assert set(ISSUE_NAMED_ROOTS) <= _on_disk_roots(), "expected test roots moved; update this test"
    covered = _core_test_paths() | _optional_roots() | _extended_roots()
    uncovered = sorted(set(ISSUE_NAMED_ROOTS) - covered)
    assert not uncovered, (
        "these test roots belong to no pr_ready_check lane, so a green readiness "
        f"run never executes them: {uncovered}. Add them to core_test_paths in "
        "run_tests_parallel.sh, to tests/support/optional_test_allowlist.txt, or "
        "to the extended lane list in pr_ready_check.sh."
    )


def test_extended_lane_roots_are_not_already_covered_elsewhere() -> None:
    """The extended lane must not duplicate work another lane already does."""
    extended = _extended_roots()
    assert extended, "pr_ready_check.sh must declare an extended lane root list"
    redundant = sorted(extended & (_core_test_paths() | _optional_roots()))
    assert not redundant, f"extended lane duplicates existing lane coverage: {redundant}"


def test_readiness_reports_lane_coverage_including_omissions() -> None:
    """The readiness run must state which lanes ran and which roots it skipped."""
    text = PR_READY_CHECK.read_text(encoding="utf-8")
    assert "Readiness lane coverage summary" in text, (
        "pr_ready_check.sh must print a readiness lane coverage summary so a PR "
        "body cannot claim coverage the run did not produce"
    )
    assert "NOT COVERED by this readiness run" in text, (
        "the summary must name roots this run did not cover, not just the lanes that ran"
    )


def test_extended_lane_triggers_on_shipped_code_changes() -> None:
    """Changes to shipped behavior must pull the uncovered roots into the run."""
    text = PR_READY_CHECK.read_text(encoding="utf-8")
    assert re.search(r"robot_sf/\*\|configs/\*\|maps/\*", text), (
        "extended lane must trigger for robot_sf/, configs/ and maps/ changes"
    )
    assert "run_pr_ready_lane extended" in text, (
        "pr_ready_check.sh must actually run the extended lane when required"
    )
