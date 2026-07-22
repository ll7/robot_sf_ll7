#!/usr/bin/env python3
"""Mutation-testing ratchet for benchmark-critical semantics (issue #5508).

Compares surviving mutants from mutmut against a committed baseline,
failing when new un-baselined mutants survive.

Exit codes
----------
* ``0`` — ratchet holds (no new surviving mutants beyond baseline).
* ``1`` — new un-baselined mutants found.
* ``2`` — mutmut execution or parsing error (infra failure).

Usage
-----
::

    # Run mutmut and check against baseline (CI / local gate).
    uv run python scripts/dev/mutation_ratchet.py --check

    # Refresh baseline after triaging new mutants.
    uv run python scripts/dev/mutation_ratchet.py --write-baseline

    # Dry-run to see what would fail without failing.
    uv run python scripts/dev/mutation_ratchet.py --check --dry-run

The committed baseline lives at ``scripts/validation/mutation_baseline.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path("scripts/validation/mutation_baseline.json")
DEFAULT_MODULE = "robot_sf/research/aggregation.py"
DEFAULT_TEST_FILE = "tests/research/test_aggregation.py"
MUTMUT_CACHE_DIR = ".mutmut-cache"


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
    """Run mutmut on the target module and return parsed results.

    Returns:
        Dict with keys: ``surviving_mutants`` (list of mutant dicts),
        ``killed_count`` (int), ``surviving_count`` (int), ``total_count`` (int).

    Raises:
        RuntimeError: If mutmut fails or produces unparseable output.
    """
    cmd = [
        sys.executable,
        "-m",
        "mutmut",
        "run",
        "--paths-to-mutate",
        DEFAULT_MODULE,
        "--tests-dir",
        DEFAULT_TEST_FILE,
    ]

    subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    # mutmut exits 0 on success (all mutants killed) or non-zero if survivors
    # We always try to get results regardless of exit code
    result_cmd = [sys.executable, "-m", "mutmut", "results"]
    result_proc = subprocess.run(
        result_cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result_proc.returncode != 0:
        raise RuntimeError(
            f"mutmut results failed (exit {result_proc.returncode}):\n"
            f"stdout: {result_proc.stdout}\n"
            f"stderr: {result_proc.stderr}"
        )

    return _parse_mutmut_results(result_proc.stdout)


def _parse_mutmut_results(output: str) -> dict[str, Any]:
    """Parse mutmut results output into structured data.

    mutmut results outputs lines like:
        Survived: 1-5, 8, 10-12
        Killed: 6, 7, 9

    Returns:
        Dict with surviving_mutants list and counts.
    """
    surviving_mutants: list[dict[str, Any]] = []
    killed_count = 0
    surviving_count = 0

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Survived:"):
            ids_str = line.split(":", 1)[1].strip()
            surviving_ids = _parse_id_range(ids_str)
            surviving_count = len(surviving_ids)
            for mutant_id in surviving_ids:
                surviving_mutants.append(
                    {
                        "id": mutant_id,
                        "status": "survived",
                    }
                )
        elif line.startswith("Killed:"):
            ids_str = line.split(":", 1)[1].strip()
            killed_ids = _parse_id_range(ids_str)
            killed_count = len(killed_ids)

    total_count = killed_count + surviving_count

    return {
        "surviving_mutants": surviving_mutants,
        "killed_count": killed_count,
        "surviving_count": surviving_count,
        "total_count": total_count,
    }


def _parse_id_range(ids_str: str) -> list[int]:
    """Parse a mutmut ID range string like '1-5, 8, 10-12' into a list of ints."""
    if not ids_str:
        return []

    result = []
    for part in ids_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def load_baseline(path: Path) -> dict[str, Any]:
    """Load and validate the mutation baseline file.

    Returns:
        Parsed baseline dict with ``tolerated_mutants`` list.

    Raises:
        FileNotFoundError: If baseline file doesn't exist.
        ValueError: If baseline format is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "tolerated_mutants" not in data:
        raise ValueError("Baseline missing 'tolerated_mutants' key")

    return data


def save_baseline(path: Path, data: dict[str, Any]) -> None:
    """Save the baseline file with updated metadata."""
    data["metadata"]["updated"] = datetime.now(UTC).strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def compare_results(
    surviving_mutants: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare surviving mutants against baseline.

    Returns:
        Tuple of (new_mutants, tolerated_mutants).
        New mutants are those NOT in baseline.
        Tolerated mutants are those IN baseline.
    """
    tolerated_ids = {m["id"] for m in baseline.get("tolerated_mutants", [])}

    new_mutants = []
    tolerated_mutants = []

    for mutant in surviving_mutants:
        if mutant["id"] in tolerated_ids:
            tolerated_mutants.append(mutant)
        else:
            new_mutants.append(mutant)

    return new_mutants, tolerated_mutants


def run_check(
    baseline_path: Path,
    dry_run: bool = False,
    report_path: Path | None = None,
) -> int:
    """Run mutation testing and check against baseline.

    Returns:
        Exit code (0=pass, 1=new mutants, 2=infra error).
    """
    repo_root = _repo_root()
    baseline_full = repo_root / baseline_path

    # Load baseline
    try:
        baseline = load_baseline(baseline_full)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: Failed to load baseline: {exc}", file=sys.stderr)
        return 2

    # Run mutmut
    try:
        results = run_mutmut(repo_root)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: mutmut execution failed: {exc}", file=sys.stderr)
        return 2

    surviving = results["surviving_mutants"]
    new_mutants, tolerated = compare_results(surviving, baseline)

    # Print summary
    print("Mutation testing summary:")
    print(f"  Total mutants: {results['total_count']}")
    print(f"  Killed: {results['killed_count']}")
    print(f"  Surviving: {results['surviving_count']}")
    print(f"  Tolerated (in baseline): {len(tolerated)}")
    print(f"  NEW (un-baselined): {len(new_mutants)}")

    if new_mutants:
        print("\nNEW un-baselined surviving mutants:")
        for m in new_mutants:
            print(f"  - Mutant #{m['id']}")

        if dry_run:
            print("\n[DRY-RUN] Would fail but --dry-run specified")
            return 0

        print("\nFAIL: New un-baselined mutants found.")
        print("Options:")
        print("  1. Fix the code/tests to kill these mutants")
        print("  2. Add to tolerated_mutants in baseline if equivalent mutant")
        return 1

    print("\nPASS: No new un-baselined mutants.")
    return 0


def run_write_baseline(baseline_path: Path) -> int:
    """Run mutmut and update baseline with current surviving mutants.

    Returns:
        Exit code (0=success, 2=infra error).
    """
    repo_root = _repo_root()
    baseline_full = repo_root / baseline_path

    # Load existing baseline or create new
    try:
        baseline = load_baseline(baseline_full)
    except FileNotFoundError:
        baseline = {
            "schema_version": "1.0.0",
            "module": DEFAULT_MODULE,
            "test_file": DEFAULT_TEST_FILE,
            "description": "Baseline of tolerated surviving mutants.",
            "tolerated_mutants": [],
            "metadata": {
                "created": datetime.now(UTC).strftime("%Y-%m-%d"),
                "issue": "#5508",
            },
        }

    # Run mutmut
    try:
        results = run_mutmut(repo_root)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: mutmut execution failed: {exc}", file=sys.stderr)
        return 2

    # Update baseline with current surviving mutants
    baseline["tolerated_mutants"] = results["surviving_mutants"]
    save_baseline(baseline_full, baseline)

    print(f"Baseline updated: {len(results['surviving_mutants'])} tolerated mutants")
    print(f"Saved to: {baseline_full}")
    return 0


def main() -> int:
    """Entry point for mutation ratchet CLI."""
    parser = argparse.ArgumentParser(
        description="Mutation-testing ratchet for benchmark-critical semantics."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run mutmut and check against baseline (fail on new mutants).",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Run mutmut and update baseline with current surviving mutants.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Path to baseline file (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would fail without actually failing.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to write JSON report (optional).",
    )

    args = parser.parse_args()

    if not args.check and not args.write_baseline:
        parser.error("Must specify either --check or --write-baseline")

    if args.check:
        return run_check(args.baseline, dry_run=args.dry_run, report_path=args.report)
    else:
        return run_write_baseline(args.baseline)


if __name__ == "__main__":
    sys.exit(main())
