#!/usr/bin/env python3
"""Branch-coverage threshold analysis report.

Reads a coverage.json produced by ``pytest --cov-branch --cov-report=json``
and prints:

* Overall branch-coverage summary
* Per-package (``robot_sf/<pkg>``) branch-coverage table
* The 10 worst-covered evidence-critical modules
* A proposed phased threshold schedule

Usage::

    uv run python scripts/dev/branch_coverage_report.py [--json path/to/coverage.json]

Default ``--json`` is ``output/coverage/coverage.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EVIDENCE_CRITICAL_PREFIXES: list[str] = [
    "robot_sf/benchmark/",
    "robot_sf/scenario_certification/",
    "robot_sf/analysis/",
    "robot_sf/analysis_workbench/",
    "robot_sf/coverage_tools/",
    "robot_sf/research/",
    "robot_sf/planner/",
    "robot_sf/sensor/",
    "robot_sf/gym_env/",
    "robot_sf/nav/",
]


def _branch_pct(summary: dict[str, int]) -> float:
    covered = summary.get("covered_branches", 0)
    total = covered + summary.get("missing_branches", 0)
    return (covered / total * 100) if total > 0 else 100.0


def load_coverage(path: Path) -> dict:
    """Load coverage.json from disk."""
    with path.open() as f:
        return json.load(f)


def overall_summary(data: dict) -> str:
    """Return overall branch-coverage summary line."""
    t = data["totals"]
    total_b = t["covered_branches"] + t["missing_branches"]
    pct = t["covered_branches"] / total_b * 100 if total_b > 0 else 100.0
    lines = (
        f"Overall branch coverage: {pct:.1f}%  "
        f"({t['covered_branches']}/{total_b} branches)\n"
        f"Line coverage: {t['covered_lines']}/{t['covered_lines'] + t['missing_lines']} "
        f"({t['percent_covered']:.1f}%)"
    )
    return lines


def per_package_table(data: dict) -> str:
    """Return Markdown table of per-package branch coverage."""
    pkg_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"covered": 0, "missing": 0})
    for fpath, info in data["files"].items():
        if not fpath.startswith("robot_sf/"):
            continue
        parts = fpath.split("/")
        if len(parts) >= 3 and not parts[1].endswith(".py"):
            pkg = "/".join(parts[:2])
        elif len(parts) >= 2:
            pkg = "robot_sf/_root"
        else:
            pkg = "robot_sf/_root"
        s = info["summary"]
        pkg_stats[pkg]["covered"] += s.get("covered_branches", 0)
        pkg_stats[pkg]["missing"] += s.get("missing_branches", 0)

    rows: list[tuple[str, float, int, int]] = []
    for pkg, s in sorted(pkg_stats.items()):
        total = s["covered"] + s["missing"]
        pct = s["covered"] / total * 100 if total > 0 else 100.0
        rows.append((pkg, pct, s["covered"], total))

    lines = ["| Package | Branch % | Covered | Total |", "|---|---|---|---|"]
    for pkg, pct, cov, total in rows:
        lines.append(f"| {pkg} | {pct:.1f}% | {cov} | {total} |")
    return "\n".join(lines)


def worst_evidence_critical(data: dict, n: int = 10) -> str:
    """Return Markdown table of the n worst-covered evidence-critical modules."""
    rows: list[tuple[str, float, int, int, int, int]] = []
    for fpath, info in data["files"].items():
        if not fpath.startswith("robot_sf/"):
            continue
        is_ec = any(fpath.startswith(p) for p in EVIDENCE_CRITICAL_PREFIXES)
        if not is_ec:
            continue
        s = info["summary"]
        total_b = s.get("covered_branches", 0) + s.get("missing_branches", 0)
        pct = _branch_pct(s)
        rows.append(
            (
                fpath,
                pct,
                s.get("covered_branches", 0),
                total_b,
                s["covered_lines"],
                s["covered_lines"] + s["missing_lines"],
            )
        )

    rows.sort(key=lambda r: (r[1], -r[3]))
    lines = [
        "| # | Module | Branch % | Branches | Lines |",
        "|---|---|---|---|---|",
    ]
    for i, (path, pct, bc, bt, lc, lt) in enumerate(rows[:n], 1):
        lines.append(f"| {i} | {path} | {pct:.1f}% | {bc}/{bt} | {lc}/{lt} |")
    return "\n".join(lines)


def threshold_proposal(data: dict) -> str:
    """Return Markdown text with a proposed phased threshold schedule."""
    t = data["totals"]
    total_b = t["covered_branches"] + t["missing_branches"]
    current_pct = t["covered_branches"] / total_b * 100 if total_b > 0 else 100.0

    zero_count = 0
    for fpath, info in data["files"].items():
        if not fpath.startswith("robot_sf/"):
            continue
        s = info["summary"]
        total = s.get("covered_branches", 0) + s.get("missing_branches", 0)
        if total > 0 and _branch_pct(s) == 0.0:
            zero_count += 1

    proposal = f"""## Proposed Phased Threshold Schedule

**Current baseline** (2026-07-09, CPU-only pytest with optional-dep tests excluded):
{current_pct:.1f}% branch coverage across {total_b} branches.

**Context caveat**: 57 test files could not collect (missing torch, stable-baselines3,
optuna, duckdb, pyarrow). Modules exercised only by those tests report 0% branch
coverage even though they may have tests in CI. This report reflects the CPU-only
subset.

### Phase 1 — Measure-only (current)
- No threshold enforcement.
- Goal: establish a reproducible baseline and identify blind spots.

### Phase 2 — Floor at current baseline (~{current_pct:.0f}%)
- Enforce: ``--cov-fail-under={int(current_pct)}`` on the CPU test subset.
- Prevent regressions on already-tested code paths.
- Target: Q3 2026, after optional-dep tests are integrated into CPU CI.

### Phase 3 — Raise to 35%
- Focus on the 10 worst evidence-critical modules listed above.
- Priority: ``benchmark/map_runner.py``, ``planner/socnav.py``,
  ``scenario_certification/perturbation_preflight.py``.
- Target: Q4 2026.

### Phase 4 — Raise to 50%
- Requires branch-coverage tests for planner, benchmark, and scenario_certification.
- Address the {zero_count} robot_sf files with 0% branch coverage.
- Target: Q1 2027.

### Phase 5 — Target 70%+ (evidence-critical only)
- Enforce per-package thresholds on evidence-critical packages only.
- Non-evidence packages (render, manual_control, telemetry) excluded.
- Target: Q2 2027.

### Recommended immediate actions
1. Add ``--cov-fail-under={int(current_pct)}`` to the CPU test runner to lock the
   baseline (must match the Phase 2 floor; a value above the current {current_pct:.1f}%
   baseline would fail CI immediately).
2. Open follow-up issues for the 10 worst evidence-critical modules.
3. Integrate optional-dep tests into CPU CI with ``--ignore`` guards for GPU-only tests.
"""
    return proposal


def main() -> None:
    """Parse args and print the branch-coverage report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("output/coverage/coverage.json"),
        help="Path to coverage.json",
    )
    args = parser.parse_args()

    if not args.json.exists():
        print(
            f"ERROR: {args.json} not found. Run pytest with --cov-report=json first.",
            file=sys.stderr,
        )
        sys.exit(1)

    data = load_coverage(args.json)

    print("# Branch-Coverage Threshold Analysis Report")
    print()
    print(overall_summary(data))
    print()
    print("## Per-Package Branch Coverage")
    print(per_package_table(data))
    print()
    print("## 10 Worst-Covered Evidence-Critical Modules")
    print(worst_evidence_critical(data))
    print()
    print(threshold_proposal(data))


if __name__ == "__main__":
    main()
