#!/usr/bin/env python3
"""Mutation-testing downward ratchet for benchmark-critical semantics (issue #5508).

This helper runs ``mutmut`` on a configurable set of production source paths,
compares the list of surviving mutants against a committed baseline of
tolerated survivors, and fails the CI job if any **new** or **un-baselined**
mutants survive.  This is a *bounded* mutation-testing lane: it never blocks on
the absolute number of survivors, only on *new* regressions beyond the
explicitly grandfathered list.

Ratchet contract
----------------
* A survivor NOT present in the committed baseline -> FAIL.
* A survivor present in the baseline is tolerated (no failure even if the
  list stays the same or shrinks).
* A **decrease** (fewer survivors than the baseline) never fails; the helper
  prints a "ratchet opportunity" notice so the baseline can be refreshed to
  lock in the improvement (``--write-baseline``).
* If ``mutmut`` cannot be run or its output is unparseable the script exits
  with code 2 (infra error).

Exit codes
----------
* ``0`` — ratchet holds (no new un-baselined survivors).
* ``1`` — at least one un-baselined survivor found.
* ``2`` — mutmut could not be run or produced unparseable output.

Usage
-----
::

    # Run mutmut and check against the committed baseline (CI / local gate).
    uv run python scripts/dev/mutation_ratchet.py --check

    # Refresh the baseline after intentionally reducing survivors.
    uv run python scripts/dev/mutation_ratchet.py --write-baseline

    # Run mutmut and print the current survivor list without checking.
    uv run python scripts/dev/mutation_ratchet.py --aggregate-only

The committed baseline lives at ``scripts/validation/mutation_baseline.json``.
The mutmut CI/CD stats file is written to ``mutants/mutmut-cicd-stats.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_BASELINE = Path("scripts/validation/mutation_baseline.json")
# The mutmut CI/CD stats file is written by ``mutmut export-cicd-stats``.
DEFAULT_MUTMUT_STATS = Path("mutants/mutmut-cicd-stats.json")
DEFAULT_MUTMUT_CACHE = Path("mutants")

# The mutmut configuration lives in setup.cfg under the [mutmut] section.
# This script does not duplicate it; it just invokes mutmut and parses the
# CI/CD stats output.


def _repo_root() -> Path:
    """Return the current Git repository root."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("Could not determine git repository root.")
    return Path(proc.stdout.strip())


def run_mutmut(repo_root: Path) -> dict[str, Any]:
    """Run ``mutmut run`` then ``mutmut export-cicd-stats`` and return stats.

    Raises ``RuntimeError`` if either command fails or the stats JSON is
    unparseable.
    """
    mutmut_cache = repo_root / DEFAULT_MUTMUT_CACHE
    stats_path = repo_root / DEFAULT_MUTMUT_STATS

    # Clean any stale mutmut cache to ensure a fresh run.
    if mutmut_cache.exists():
        proc = subprocess.run(
            ["rm", "-rf", str(mutmut_cache)],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Could not clean mutmut cache at {mutmut_cache}: {proc.stderr[:1000]}"
            )

    cmd_run = ["uv", "run", "mutmut", "run"]
    try:
        proc = subprocess.run(
            cmd_run,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not invoke mutmut: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"mutmut run exited {proc.returncode}.\n"
            f"stdout:\n{proc.stdout[:2000]}\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )

    cmd_export = ["uv", "run", "mutmut", "export-cicd-stats"]
    try:
        proc = subprocess.run(
            cmd_export,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not invoke mutmut export-cicd-stats: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"mutmut export-cicd-stats exited {proc.returncode}.\n"
            f"stdout:\n{proc.stdout[:2000]}\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )

    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load mutmut CI/CD stats from {stats_path}: {exc}") from exc

    if not isinstance(stats, dict):
        raise RuntimeError(f"mutmut CI/CD stats must be a JSON object, got {type(stats).__name__}.")
    return stats


def get_survivor_ids(stats: dict[str, Any]) -> list[str]:
    """Return the sorted list of surviving mutant IDs from mutmut stats.

    The stats dict has keys like ``killed``, ``survived``, ``total``, etc.
    The actual surviving-mutant IDs are NOT in the CI/CD stats JSON — they
    come from the ``mutmut results`` command.

    This function runs ``mutmut results`` and parses the "survived" lines.
    """
    return []


def run_mutmut_results(repo_root: Path) -> list[str]:
    """Run ``mutmut results`` and return sorted surviving mutant IDs.

    Parses lines like::

        robot_sf.research.aggregation.x_aggregate_metrics__mutmut_18: survived

    Raises ``RuntimeError`` if the command fails.
    """
    cmd = ["uv", "run", "mutmut", "results"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not invoke mutmut results: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"mutmut results exited {proc.returncode}.\nstderr:\n{proc.stderr[:2000]}"
        )

    survivors: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.endswith(": survived"):
            mutant_id = line.split(":")[0].strip()
            if mutant_id:
                survivors.append(mutant_id)
    return sorted(survivors)


def load_baseline(path: Path) -> dict[str, Any]:
    """Load and minimally validate a baseline file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Baseline {path} must be a JSON object, got {type(data).__name__}.")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported baseline schema_version in {path}: "
            f"got {data.get('schema_version')}, expected {SCHEMA_VERSION}"
        )
    surv = data.get("surviving_mutants")
    if not isinstance(surv, list):
        raise ValueError(f"Baseline {path} is missing a valid 'surviving_mutants' list.")
    return data


def build_baseline_payload(
    survivor_ids: list[str],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Build the versioned baseline JSON payload from mutmut results."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": (
            "Mutation-testing downward ratchet for benchmark-critical semantics "
            "(issue #5508). The ratchet gates on net-new surviving mutants: any "
            "survivor NOT in this list fails the CI gate. To reduce the list, "
            "improve the test coverage and refresh the baseline with "
            "`scripts/dev/mutation_ratchet.py --write-baseline`."
        ),
        "source_paths": ["robot_sf/research/aggregation.py"],
        "test_selection": "tests/research/test_aggregation.py",
        "stats": {
            "total": stats.get("total", 0),
            "killed": stats.get("killed", 0),
            "survived": stats.get("survived", 0),
            "timeout": stats.get("timeout", 0),
        },
        "surviving_mutants": survivor_ids,
    }


def check_against_baseline(
    current_survivors: list[str],
    baseline: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return (failures, notices) for the downward ratchet.

    ``failures`` is non-empty -> the ratchet is broken (exit 1). ``notices``
    are informational ratchet-opportunity hints (count decreased) and are
    always advisory.
    """
    baseline_survivors: list[str] = sorted(str(m) for m in baseline.get("surviving_mutants", []))
    current_set = set(current_survivors)
    baseline_set = set(baseline_survivors)

    failures: list[str] = []
    notices: list[str] = []

    new_survivors = sorted(current_set - baseline_set)
    if new_survivors:
        failures.append(f"Found {len(new_survivors)} new un-baselined surviving mutant(s):")
        for mutant_id in new_survivors:
            failures.append(f"  {mutant_id}")

    if len(current_survivors) < len(baseline_survivors):
        removed = len(baseline_survivors) - len(current_survivors)
        notices.append(
            f"Ratchet opportunity: survivor count dropped from "
            f"{len(baseline_survivors)} to {len(current_survivors)} "
            f"({removed} fewer). Refresh the baseline with "
            f"--write-baseline to lock in the improvement."
        )

    return failures, notices


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write stable, reviewable, sort-keyed JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Run the ratchet gate.")
    mode.add_argument(
        "--write-baseline",
        action="store_true",
        help="Recompute survivors and (re)write the baseline file.",
    )
    mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Print the survivor list without reading/writing a baseline.",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (defaults to git toplevel).",
    )
    return parser.parse_args(argv)


def _gather_results(args: argparse.Namespace, repo_root: Path) -> tuple[list[str], dict[str, Any]]:
    """Run mutmut, get survivors and stats."""
    stats = run_mutmut(repo_root)
    survivors = run_mutmut_results(repo_root)
    return survivors, stats


def _print_aggregate(survivors: list[str], stats: dict[str, Any]) -> None:
    """Print aggregate info for ``--aggregate-only``."""
    print(
        f"mutation testing: total={stats.get('total', 0)} "
        f"killed={stats.get('killed', 0)} "
        f"survived={len(survivors)} "
        f"(stats say survived={stats.get('survived', 0)})"
    )
    for mutant_id in survivors:
        print(f"  SURVIVED  {mutant_id}")


def _report_check(
    survivors: list[str],
    stats: dict[str, Any],
    baseline: dict[str, Any],
    failures: list[str],
    notices: list[str],
) -> int:
    """Print the ``--check`` ratchet result and return the exit code."""
    baseline_count = len(baseline.get("surviving_mutants", []))
    print(
        f"mutation ratchet: survivors={len(survivors)} "
        f"(baseline={baseline_count}), "
        f"killed={stats.get('killed', 0)}, "
        f"total={stats.get('total', 0)}."
    )
    for notice in notices:
        print(f"NOTICE: {notice}")
    if failures:
        print("\nmutation ratchet FAILED (new un-baselined survivors):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nAdd tests to kill the new mutants, or refresh the baseline with "
            "`scripts/dev/mutation_ratchet.py --write-baseline` if the "
            "increase is intentional and reviewed.",
            file=sys.stderr,
        )
        return 1
    print("mutation ratchet passed: no new un-baselined survivors.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ratchet gate, baseline refresh, or aggregate report."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.root.resolve() if args.root is not None else _repo_root()
    baseline_path = args.baseline if args.baseline.is_absolute() else repo_root / args.baseline

    try:
        survivors, stats = _gather_results(args, repo_root)
    except RuntimeError as exc:
        print(f"ERROR: could not obtain mutmut results: {exc}", file=sys.stderr)
        return 2

    if args.aggregate_only:
        _print_aggregate(survivors, stats)
        return 0

    payload = build_baseline_payload(survivors, stats)

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(baseline_path, payload)
        print(
            f"Wrote mutation baseline to {baseline_path}: "
            f"{len(survivors)} survivors out of "
            f"{stats.get('total', 0)} total mutations "
            f"({stats.get('killed', 0)} killed)."
        )
        return 0

    # --check
    if not baseline_path.exists():
        print(
            f"ERROR: baseline not found at {baseline_path}. "
            f"Generate it with --write-baseline first.",
            file=sys.stderr,
        )
        return 2
    baseline = load_baseline(baseline_path)
    failures, notices = check_against_baseline(survivors, baseline)
    return _report_check(survivors, stats, baseline, failures, notices)


if __name__ == "__main__":
    raise SystemExit(main())
