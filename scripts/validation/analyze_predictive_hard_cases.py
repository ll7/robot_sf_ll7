#!/usr/bin/env python3
"""Analyze predictive planner failures and produce hard-case taxonomy report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/tmp/predictive_planner/reports/hard_case_taxonomy_report.md"),
    )
    return parser.parse_args()


def _episode_success(row: dict) -> bool:
    """Resolve episode success with collision-aware fallback semantics."""
    metrics = row.get("metrics", {})
    if "success_rate" in metrics:
        return float(metrics.get("success_rate", 0.0)) >= 0.5
    value = metrics.get("success", False)
    if isinstance(value, bool):
        return value
    return float(value) >= 0.5


def main() -> int:
    """Read benchmark JSONL and write hard-case taxonomy markdown report."""
    args = parse_args()
    rows = [
        json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line
    ]
    failed = [row for row in rows if not _episode_success(row)]

    by_status = Counter(str(row.get("status", "unknown")) for row in failed)
    by_reason = Counter(str(row.get("termination_reason", "unknown")) for row in failed)
    by_scenario = Counter(str(row.get("scenario_id", "unknown")) for row in failed)

    failures_by_scenario_seed: dict[str, list[int]] = defaultdict(list)
    for row in failed:
        sid = str(row.get("scenario_id", "unknown"))
        failures_by_scenario_seed[sid].append(int(row.get("seed", -1)))

    lines = [
        "# Predictive Hard-Case Taxonomy",
        "",
        f"- Source JSONL: `{args.jsonl}`",
        f"- Episodes: `{len(rows)}`",
        f"- Failed episodes: `{len(failed)}`",
        "",
        "## Failure Status Counts",
        "",
    ]
    for key, value in sorted(by_status.items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Failure Termination Reasons", ""])
    for key, value in sorted(by_reason.items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Failure Scenarios", ""])
    for key, value in sorted(by_scenario.items(), key=lambda kv: kv[1], reverse=True):
        seeds = sorted(set(failures_by_scenario_seed[key]))
        lines.append(f"- `{key}`: `{value}` failures, seeds={seeds}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "failed": len(failed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
