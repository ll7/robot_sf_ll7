#!/usr/bin/env python3
"""Bounded mutation-testing downward ratchet for benchmark-critical semantics (issue #5508).

This helper turns the ``mutmut`` mutation-testing tool into a **monotone
downward ratchet** over the surviving mutants for one benchmark-critical module,
mirroring the ``scripts/dev/evidence_registry_ratchet.py`` and
``scripts/dev/ty_advisory_ratchet.py`` precedents.

What this owns (issue #5508)
----------------------------
The scheduled-diagnostic decision for mutation testing on metric aggregation.
Line coverage (the 85% CI floor) does not establish that assertions *detect*
incorrect metric, provenance, or publication-gate behavior. Mutation testing
does. This ratchet limits that cost to **one** module
(``robot_sf/research/aggregation.py``) exercised by
``tests/research/test_aggregation.py`` and makes the result a downward ratchet:
once a surviving mutant is baselined it is tolerated, but **no new** surviving
mutant can land unnoticed.

The issue is explicit that this is a **scheduled diagnostic**, never a required
PR gate. The GitHub Actions workflow (``.github/workflows/mutation-testing.yml``)
runs weekly and on ``workflow_dispatch`` and is intentionally non-gating on pull
requests.

Ratchet contract
----------------
Surviving mutants are identified by their stable mutmut name, e.g.
``robot_sf.research.aggregation.x_aggregate_metrics__mutmut_18``. The name embeds
the module, the mutated function, and the mutant index, so it is stable across
runs as long as the source and mutmut version do not change.

* A **new survivor** (in the current run, absent from the baseline) -> FAIL.
  This is the primary value: no NEW surviving mutant can slip through.
* A **baselined survivor that disappears** (killed by a new/improved test) ->
  informational "ratchet opportunity" notice so the baseline can be refreshed
  with ``--write-baseline`` to lock in the improvement.
* The baseline only ever shrinks: a fully-killed survivor is dropped on
  ``--write-baseline``.

Mutant names are deliberately not validated for staleness against the current
source: a refactor that shifts indices will surface as many "new" and many
"removed" mutants, which is the correct signal to re-triage and refresh the
baseline.

Exit codes
----------
* ``0`` — ratchet holds (no new surviving mutant beyond the baseline).
* ``1`` — at least one new, un-baselined surviving mutant was found.
* ``2`` — mutmut could not be run / produced unparseable output (infra error).

Usage
-----
::

    # Re-run mutmut and check against the committed baseline (scheduled lane).
    uv run python scripts/dev/mutation_ratchet.py --check

    # Refresh the baseline after killing survivors with new tests.
    uv run python scripts/dev/mutation_ratchet.py --write-baseline

    # Parse a pre-rendered `mutmut results` report (offline / test / no-network).
    uv run python scripts/dev/mutation_ratchet.py --check --report /tmp/results.txt

    # Print the current surviving mutants without a baseline comparison.
    uv run python scripts/dev/mutation_ratchet.py --aggregate-only

The committed baseline lives at ``scripts/validation/mutation_baseline.json``.
The ``[mutmut]`` configuration in ``setup.cfg`` pins the source module and the
test selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = 1
DEFAULT_BASELINE = Path("scripts/validation/mutation_baseline.json")
DEFAULT_SOURCE_PATH = "robot_sf/research/aggregation.py"
DEFAULT_TEST_PATH = "tests/research/test_aggregation.py"
# mutmut stores its summary counts here (relative to the mutants workspace) when
# `debug=true` is set in setup.cfg and `mutmut export-cicd-stats` is run.
DEFAULT_CICD_STATS = Path("mutants/mutmut-cicd-stats.json")
SURVIVED_SUFFIX = ": survived"


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


def run_mutmut(repo_root: Path) -> tuple[list[str], dict[str, Any]]:
    """Run mutmut and return ``(survivors, stats)``.

    Runs ``mutmut run`` (full mutation pass over the configured source module),
    then ``mutmut results`` to enumerate the surviving mutants, and
    ``mutmut export-cicd-stats`` to capture the summary counts. The survivor
    names are returned sorted; ``stats`` is the parsed cicd-stats mapping (empty
    when the stats file is unavailable, which is non-fatal).

    Raises ``RuntimeError`` on a non-zero mutmut exit (infra error) or on
    unparseable ``mutmut results`` output.
    """
    survivors_raw = _run_mutmut_results(repo_root)
    survivors = parse_survivors(survivors_raw)
    stats = _load_cicd_stats(repo_root)
    return survivors, stats


def _run_mutmut_results(repo_root: Path) -> str:
    """Run ``mutmut run`` then ``mutmut results`` and return the results text."""
    run_cmd = [sys.executable, "-m", "mutmut", "run"]
    try:
        run_proc = subprocess.run(
            run_cmd, cwd=repo_root, check=False, capture_output=True, text=True
        )
    except OSError as exc:
        raise RuntimeError(f"Could not invoke 'mutmut run': {exc}") from exc
    if run_proc.returncode != 0:
        raise RuntimeError(
            f"'mutmut run' exited {run_proc.returncode}.\nstderr tail:\n{run_proc.stderr[-2000:]}"
        )

    results_cmd = [sys.executable, "-m", "mutmut", "results"]
    try:
        results_proc = subprocess.run(
            results_cmd, cwd=repo_root, check=False, capture_output=True, text=True
        )
    except OSError as exc:
        raise RuntimeError(f"Could not invoke 'mutmut results': {exc}") from exc
    if results_proc.returncode != 0:
        raise RuntimeError(
            f"'mutmut results' exited {results_proc.returncode}.\nstderr tail:\n"
            f"{results_proc.stderr[-2000:]}"
        )
    return results_proc.stdout


def _load_cicd_stats(repo_root: Path) -> dict[str, Any]:
    """Load the mutmut CI/CD summary counts, generating them if absent.

    Runs ``mutmut export-cicd-stats`` (which writes
    ``mutants/mutmut-cicd-stats.json``) and parses the result. Any failure is
    non-fatal: the counts are informational and the ratchet gates on the
    survivor list, not the counts.
    """
    stats_cmd = [sys.executable, "-m", "mutmut", "export-cicd-stats"]
    try:
        subprocess.run(stats_cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    except OSError:
        return {}
    stats_path = repo_root / DEFAULT_CICD_STATS
    if not stats_path.exists():
        return {}
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_survivors(text: str) -> list[str]:
    """Parse ``mutmut results`` text into a sorted list of survivor names.

    ``mutmut results`` prints one line per survivor in the form::

        <indent>robot_sf.research.aggregation.x_func__mutmut_N: survived

    Only ``survived`` lines are collected; ``timeout``/``suspicious``/``skipped``
    states are intentionally excluded from the baseline because they are flaky or
    environment-dependent rather than a stable "the tests do not catch this"
    signal. ``--all`` is not passed to ``mutmut results`` so only survivors are
    reported.
    """
    survivors: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.endswith(SURVIVED_SUFFIX):
            continue
        name = stripped[: -len(SURVIVED_SUFFIX)].strip()
        if name:
            survivors.append(name)
    return sorted(set(survivors))


def load_report(path: Path) -> str:
    """Load a pre-rendered ``mutmut results`` text report from ``path``.

    Raises ``RuntimeError`` on a missing file so the CLI maps a bad ``--report``
    to the infra-error exit code (2) instead of an uncaught traceback.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read report file '{path}': {exc}") from exc


def build_baseline_payload(survivors: list[str], stats: dict[str, Any]) -> dict[str, Any]:
    """Build the versioned baseline JSON payload from survivors and stats."""
    by_function: dict[str, int] = {}
    for name in survivors:
        # Strip the trailing __mutmut_N to group by mutated function.
        func = name.rsplit("__mutmut_", 1)[0] if "__mutmut_" in name else name
        by_function[func] = by_function.get(func, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "mutmut",
        "source_path": DEFAULT_SOURCE_PATH,
        "test_path": DEFAULT_TEST_PATH,
        "description": (
            "Mutation-testing downward ratchet for benchmark-critical semantics "
            "(issue #5508). The ratchet gates on the surviving-mutant set for "
            f"{DEFAULT_SOURCE_PATH} exercised by {DEFAULT_TEST_PATH}: a survivor "
            "absent from this baseline fails, and a baselined survivor that "
            "disappears is a ratchet opportunity. This is a scheduled diagnostic, "
            "never a required PR gate. Refresh this baseline with "
            "`scripts/dev/mutation_ratchet.py --write-baseline` after killing "
            "survivors with new assertions; triage guidance lives in "
            "`mutation_testing_triage.md`."
        ),
        "summary": {
            "total_mutants": int(stats.get("total", 0)),
            "killed": int(stats.get("killed", 0)),
            "survived": len(survivors),
            "timeout": int(stats.get("timeout", 0)),
            "suspicious": int(stats.get("suspicious", 0)),
            "skipped": int(stats.get("skipped", 0)),
            "survivors_by_function": dict(sorted(by_function.items())),
        },
        "survivors": survivors,
    }


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
    survivors = data.get("survivors")
    if not isinstance(survivors, list) or any(not isinstance(name, str) for name in survivors):
        raise ValueError(f"Baseline {path} is missing a valid 'survivors' string list.")
    return data


def check_against_baseline(
    current: list[str], baseline: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return ``(failures, notices)`` for the downward ratchet.

    ``failures`` is non-empty -> the ratchet is broken (exit 1): there are new,
    un-baselined surviving mutants. ``notices`` are informational ratchet
    opportunities (baselined survivors that were killed) and are always advisory.
    """
    baseline_set = set(baseline.get("survivors", []))
    current_set = set(current)

    new_survivors = sorted(current_set - baseline_set)
    killed = sorted(baseline_set - current_set)

    failures: list[str] = []
    notices: list[str] = []

    if new_survivors:
        failures.append(f"{len(new_survivors)} new surviving mutant(s) not in the baseline:")
        for name in new_survivors:
            failures.append(f"  + {name}")

    if killed:
        notices.append(
            f"ratchet opportunity: {len(killed)} baselined survivor(s) were "
            "killed; refresh the baseline to lock in the improvement:"
        )
        for name in killed:
            notices.append(f"  - {name}")

    return failures, notices


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write stable, reviewable, sort-keyed JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
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
        help="Print the surviving mutants without a baseline comparison.",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--root", type=Path, default=None, help="Repository root (defaults to git toplevel)."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Path to a pre-rendered `mutmut results` text report. When set, "
            "mutmut is NOT re-run; the report is parsed instead (offline / test "
            "mode)."
        ),
    )
    return parser.parse_args(argv)


def _gather_survivors(
    args: argparse.Namespace, repo_root: Path
) -> tuple[list[str], dict[str, Any]]:
    """Resolve survivors either by running mutmut or parsing --report."""
    if args.report is not None:
        text = load_report(args.report)
        return parse_survivors(text), {}
    return run_mutmut(repo_root)


def _print_aggregate(survivors: list[str], stats: dict[str, Any]) -> None:
    """Print the survivor aggregate report for ``--aggregate-only``."""
    total = stats.get("total", "?")
    killed = stats.get("killed", "?")
    print(f"mutation ratchet: survivors={len(survivors)} (mutmut total={total} killed={killed})")
    by_function: dict[str, int] = {}
    for name in survivors:
        func = name.rsplit("__mutmut_", 1)[0] if "__mutmut_" in name else name
        by_function[func] = by_function.get(func, 0) + 1
    for func, count in sorted(by_function.items()):
        print(f"  {count:5d}  {func}")
    for name in survivors:
        print(f"    {name}")


def _report_check(
    survivors: list[str],
    stats: dict[str, Any],
    baseline: dict[str, Any],
    failures: list[str],
    notices: list[str],
) -> int:
    """Print the ``--check`` ratchet result and return the exit code."""
    baseline_count = len(baseline.get("survivors", []))
    total = stats.get("total", "?")
    killed = stats.get("killed", "?")
    print(
        f"mutation ratchet: survivors={len(survivors)} (baseline={baseline_count}, "
        f"mutmut total={total} killed={killed})."
    )
    for notice in notices:
        print(f"NOTICE: {notice}")
    if failures:
        print("\nmutation ratchet FAILED (new surviving mutants):", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        print(
            "\nEither add assertions that kill the new survivors, or refresh the "
            "baseline with `scripts/dev/mutation_ratchet.py --write-baseline` after "
            "triaging them in `mutation_testing_triage.md`.",
            file=sys.stderr,
        )
        return 1
    print("mutation ratchet passed: no new surviving mutants; all survivors are baselined.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ratchet gate, baseline refresh, or aggregate report."""
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    repo_root = args.root.resolve() if args.root is not None else _repo_root()
    baseline_path = args.baseline if args.baseline.is_absolute() else repo_root / args.baseline

    try:
        survivors, stats = _gather_survivors(args, repo_root)
    except RuntimeError as exc:
        print(f"ERROR: could not obtain mutmut survivors: {exc}", file=sys.stderr)
        return 2

    if args.aggregate_only:
        _print_aggregate(survivors, stats)
        return 0

    payload = build_baseline_payload(survivors, stats)

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(baseline_path, payload)
        print(
            f"Wrote mutation baseline to {baseline_path}: {len(survivors)} survivors "
            f"(killed={stats.get('killed', '?')} of {stats.get('total', '?')})."
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
