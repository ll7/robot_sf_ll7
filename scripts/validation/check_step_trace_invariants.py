#!/usr/bin/env python3
"""Check benchmark episode rows against step-trace invariants (#9979).

Usage:
    uv run python scripts/validation/check_step_trace_invariants.py PATH [PATH ...] \
        --out-json report.json --out-md report.md

PATH is an ``episodes.jsonl`` file or a directory searched recursively for
``episodes*.jsonl``. The arm name is the row field ``arm``/``planner_key`` when present,
otherwise the parent directory name when it has the ``<arm>__<kinematics>`` form,
otherwise ``algo``. Rows are only read, never rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from robot_sf.benchmark.step_trace_invariants import (
    INVARIANTS,
    Tolerances,
    check_episode,
    invariant_coverage,
    summarize,
    to_markdown,
)


def _iter_files(paths: list[str]) -> Iterator[Path]:
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            yield from sorted(p.rglob("episodes*.jsonl"))
        else:
            yield p


def _arm(row: dict[str, Any], path: Path) -> str:
    for key in ("arm", "planner_key"):
        if row.get(key):
            return str(row[key])
    if "__" in path.parent.name:
        return path.parent.name.split("__")[0]
    return str(row.get("algo") or "unknown")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    tol_kwargs = {
        f.name: getattr(args, f"tol_{f.name}")
        for f in fields(Tolerances)
        if getattr(args, f"tol_{f.name}") is not None
    }
    tol = Tolerances(**tol_kwargs)
    overrides = json.loads(args.limits_json) if args.limits_json else None
    holdout = set(range(111, 141))
    trace_versions: set[str] = set()
    results = []
    coverage = []
    seen_holdout = 0
    files = list(_iter_files(args.paths))
    for path in files:
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError("episode row must be an object")
                    eligibility = invariant_coverage(row, args.invariants, tol)
                    viols, has_trace = check_episode(
                        row, tol=tol, overrides=overrides, enabled=args.invariants
                    )
                except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
                    row = {"episode_id": f"{path.name}:invalid-row-{len(results) + 1}"}
                    eligibility = {
                        name: {
                            "eligible": False,
                            "checked_steps": 0,
                            "issues": [f"invalid_row:{type(error).__name__}"],
                        }
                        for name in args.invariants
                    }
                    viols, has_trace = [], False
                if isinstance(row.get("seed"), int) and row["seed"] in holdout:
                    seen_holdout += 1  # existing published rows may be read; only counted
                metadata = row.get("algorithm_metadata")
                trace = (
                    metadata.get("simulation_step_trace") if isinstance(metadata, dict) else None
                )
                if isinstance(trace, dict):
                    version = trace.get("schema_version")
                    trace_versions.add(str(version))
                    if len(trace_versions) > 1:
                        raise ValueError(
                            "mixed trace schema versions: " + ", ".join(sorted(trace_versions))
                        )
                coverage.append(
                    {"episode_id": str(row.get("episode_id", "?")), "invariants": eligibility}
                )
                results.append(
                    (
                        _arm(row, path),
                        str(row.get("scenario_id", "?")),
                        str(row.get("episode_id", "?")),
                        viols,
                        has_trace,
                    )
                )
    summary = summarize(results, args.max_examples)
    summary["coverage"] = coverage
    coverage_complete = (
        bool(results)
        and bool(args.invariants)
        and all(c["eligible"] for r in coverage for c in r["invariants"].values())
    )
    summary["meta"] = {
        "coverage_complete": coverage_complete,
        "trace_schema_versions": sorted(trace_versions),
        "files": [str(f) for f in files],
        "episodes": len(results),
        "episodes_with_trace": sum(1 for r in results if r[4]),
        "rows_on_evaluation_holdout_seeds_read_only": seen_holdout,
        "invariants": list(args.invariants),
        "tolerances": {f.name: getattr(tol, f.name) for f in fields(Tolerances)},
        "limits_override": overrides,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0, or 1 with ``--fail-on-violation`` when any violation exists.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--max-examples", type=int, default=5)
    ap.add_argument("--invariants", nargs="*", default=list(INVARIANTS), choices=INVARIANTS)
    ap.add_argument("--limits-json", help="JSON overrides for DriveLimits fields (radii, speeds).")
    ap.add_argument("--fail-on-violation", action="store_true")
    for f in fields(Tolerances):
        ap.add_argument(f"--tol-{f.name}", dest=f"tol_{f.name}", type=type(f.default), default=None)
    args = ap.parse_args(argv)
    summary = _run(args)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str) + "\n")
    md = to_markdown(
        summary,
        {k: summary["meta"][k] for k in ("episodes", "episodes_with_trace", "coverage_complete")},
    )
    if args.out_md:
        Path(args.out_md).write_text(md)
    else:
        sys.stdout.write(md)
    any_violation = any(t["flagged"][i] for t in summary["arms"].values() for i in INVARIANTS)
    return (
        1
        if args.fail_on_violation and (any_violation or not summary["meta"]["coverage_complete"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
