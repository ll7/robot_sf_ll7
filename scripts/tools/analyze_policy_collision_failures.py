#!/usr/bin/env python3
"""Summarize collision failures from policy-analysis run artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_COLLISION_METRICS = (
    "ped_collision_count",
    "obstacle_collision_count",
    "agent_collision_count",
)


@dataclass
class ScenarioStats:
    """Per-scenario collision and outcome counters."""

    episodes: int = 0
    successes: int = 0
    collisions: int = 0
    ped_collisions: int = 0
    obstacle_collisions: int = 0
    agent_collisions: int = 0
    unclassified_collisions: int = 0
    seeds: set[str] = field(default_factory=set)
    collision_seeds: set[str] = field(default_factory=set)
    termination_reasons: Counter[str] = field(default_factory=Counter)

    def update(self, record: dict[str, Any]) -> None:
        """Add one policy-analysis episode record."""
        self.episodes += 1
        seed = _seed(record)
        if seed != "":
            self.seeds.add(seed)
        reason = _termination_reason(record)
        self.termination_reasons[reason] += 1
        if reason == "success":
            self.successes += 1
        if _is_collision(record):
            self.collisions += 1
            if seed != "":
                self.collision_seeds.add(seed)
            metrics = _metrics(record)
            ped = _metric_value(metrics, "ped_collision_count")
            obstacle = _metric_value(metrics, "obstacle_collision_count")
            agent = _metric_value(metrics, "agent_collision_count")
            self.ped_collisions += int(ped > 0)
            self.obstacle_collisions += int(obstacle > 0)
            self.agent_collisions += int(agent > 0)
            if ped <= 0 and obstacle <= 0 and agent <= 0:
                self.unclassified_collisions += 1

    def as_dict(self, scenario: str) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "scenario": scenario,
            "episodes": self.episodes,
            "successes": self.successes,
            "collisions": self.collisions,
            "collision_rate": _rate(self.collisions, self.episodes),
            "ped_collisions": self.ped_collisions,
            "obstacle_collisions": self.obstacle_collisions,
            "agent_collisions": self.agent_collisions,
            "unclassified_collisions": self.unclassified_collisions,
            "seeds": sorted(self.seeds),
            "collision_seeds": sorted(self.collision_seeds),
            "termination_reasons": dict(sorted(self.termination_reasons.items())),
        }


def _build_parser() -> argparse.ArgumentParser:
    """Build the collision-analysis CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_roots",
        nargs="+",
        type=Path,
        help="Policy-analysis output directories containing episodes.jsonl.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Optional markdown report path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of worst scenarios to include in the markdown hotspot table.",
    )
    return parser


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the metrics mapping from one episode record.

    Returns:
        Metrics dictionary, or an empty mapping for malformed records.
    """
    metrics = record.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    """Read one numeric metric with a forgiving default.

    Returns:
        Metric value as float, or ``0.0`` when missing or invalid.
    """
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _termination_reason(record: dict[str, Any]) -> str:
    """Extract the episode termination reason.

    Returns:
        Termination reason string, falling back to status or ``unknown``.
    """
    reason = record.get("termination_reason") or record.get("status") or "unknown"
    return str(reason)


def _scenario(record: dict[str, Any]) -> str:
    """Extract a stable scenario identifier from an episode record.

    Returns:
        Scenario id/name string, or ``unknown`` when absent.
    """
    for key in ("scenario_id", "scenario", "scenario_name"):
        value = record.get(key)
        if value:
            return str(value)
    return "unknown"


def _seed(record: dict[str, Any]) -> str:
    """Extract a seed identifier from an episode record.

    Returns:
        Seed string, or an empty string when absent.
    """
    for key in ("seed", "scenario_seed", "episode_seed"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _is_collision(record: dict[str, Any]) -> bool:
    """Determine whether an episode record represents a collision.

    Returns:
        True when termination or collision metrics indicate a collision.
    """
    if _termination_reason(record) == "collision":
        return True
    metrics = _metrics(record)
    total = _metric_value(metrics, "collisions")
    split_total = sum(_metric_value(metrics, key) for key in _COLLISION_METRICS)
    return total > 0 or split_total > 0


def _rate(numerator: int, denominator: int) -> float:
    """Compute a safe ratio for report rates.

    Returns:
        ``numerator / denominator`` or ``0.0`` for empty denominators.
    """
    return float(numerator / denominator) if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read episode records from a JSONL file.

    Returns:
        List of JSON object records.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    return records


def analyze_run(run_root: Path) -> dict[str, Any]:
    """Analyze one policy-analysis output directory."""
    episodes_path = run_root / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"missing episodes.jsonl under {run_root}")
    records = _read_jsonl(episodes_path)
    scenario_stats: defaultdict[str, ScenarioStats] = defaultdict(ScenarioStats)
    totals = ScenarioStats()
    for record in records:
        totals.update(record)
        scenario_stats[_scenario(record)].update(record)
    scenario_rows = [stats.as_dict(scenario) for scenario, stats in scenario_stats.items()]
    scenario_rows.sort(
        key=lambda row: (
            -int(row["collisions"]),
            -float(row["collision_rate"]),
            str(row["scenario"]),
        )
    )
    return {
        "run": run_root.name,
        "run_root": str(run_root),
        "episodes_path": str(episodes_path),
        "totals": totals.as_dict("ALL"),
        "scenarios": scenario_rows,
    }


def analyze_runs(run_roots: list[Path]) -> dict[str, Any]:
    """Analyze multiple policy-analysis output directories."""
    runs = [analyze_run(root) for root in run_roots]
    return {"runs": runs}


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Format rows as a Markdown table.

    Returns:
        Markdown table lines.
    """
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(payload: dict[str, Any], *, top_n: int) -> str:
    """Render an issue-ready markdown collision analysis report."""
    lines = ["# Policy Collision Failure Analysis", ""]
    summary_rows: list[list[str]] = []
    for run in payload["runs"]:
        totals = run["totals"]
        summary_rows.append(
            [
                f"`{run['run']}`",
                str(totals["episodes"]),
                f"{totals['successes'] / max(totals['episodes'], 1):.3f}",
                f"{totals['collision_rate']:.3f}",
                str(totals["ped_collisions"]),
                str(totals["obstacle_collisions"]),
                str(totals["agent_collisions"]),
                str(totals["unclassified_collisions"]),
            ]
        )
    lines.extend(
        _format_table(
            [
                "Run",
                "Episodes",
                "Success",
                "Collision",
                "Ped",
                "Obstacle",
                "Agent",
                "Unclassified",
            ],
            summary_rows,
        )
    )
    for run in payload["runs"]:
        lines.extend(["", f"## `{run['run']}`", ""])
        rows: list[list[str]] = []
        for row in run["scenarios"][:top_n]:
            rows.append(
                [
                    f"`{row['scenario']}`",
                    str(row["episodes"]),
                    str(row["collisions"]),
                    f"{row['collision_rate']:.3f}",
                    str(row["ped_collisions"]),
                    str(row["obstacle_collisions"]),
                    ", ".join(row["collision_seeds"]) or "-",
                ]
            )
        lines.extend(
            _format_table(
                ["Scenario", "Episodes", "Collisions", "Rate", "Ped", "Obstacle", "Seeds"],
                rows,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    payload = analyze_runs(args.run_roots)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(payload, top_n=args.top_n)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
