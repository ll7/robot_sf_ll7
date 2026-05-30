#!/usr/bin/env python3
"""Analyze scenario-level seed sensitivity for a durable benchmark campaign.

The issue #1608 analysis deliberately reads compact benchmark artifacts instead of rerunning a
campaign. It selects the top planners from ``reports/campaign_table.csv`` and classifies each
scenario from ``reports/seed_episode_rows.csv`` by how much average top-planner success varies
across fixed seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SUCCESS_RANGE_THRESHOLD = 0.5
DEFAULT_HARD_SEED_THRESHOLD = 0.5
DEFAULT_EASY_SEED_THRESHOLD = 0.75
DEFAULT_TOP_PLANNER_COUNT = 4


@dataclass(frozen=True)
class PlannerSelection:
    """Planner row selected from the campaign table."""

    planner_key: str
    success_mean: float
    collisions_mean: float
    time_to_goal_norm_mean: float
    near_misses_mean: float
    execution_mode: str
    planner_group: str
    benchmark_success: bool


@dataclass(frozen=True)
class EpisodeRow:
    """Per-scenario/per-seed episode row for one selected planner."""

    scenario_id: str
    planner_key: str
    seed: int
    success: float
    collision: float
    near_miss: float | None
    time_to_goal_norm: float | None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Campaign root containing reports/campaign_table.csv and seed_episode_rows.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-planner-count", type=int, default=DEFAULT_TOP_PLANNER_COUNT)
    parser.add_argument(
        "--success-range-threshold",
        type=float,
        default=DEFAULT_SUCCESS_RANGE_THRESHOLD,
        help="Minimum range of mean top-planner success across seeds for seed-sensitive status.",
    )
    parser.add_argument(
        "--hard-seed-threshold",
        type=float,
        default=DEFAULT_HARD_SEED_THRESHOLD,
        help="A scenario seed is hard when mean top-planner success is at or below this value.",
    )
    parser.add_argument(
        "--easy-seed-threshold",
        type=float,
        default=DEFAULT_EASY_SEED_THRESHOLD,
        help="A scenario seed is easy when mean top-planner success is at or above this value.",
    )
    return parser.parse_args(argv)


def _parse_float(value: str | None) -> float | None:
    """Parse a finite float from a CSV cell."""
    if value is None:
        return None
    cleaned = value.strip().strip("'\"")
    if cleaned == "":
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_bool(value: str | None) -> bool:
    """Parse CSV boolean values written by the benchmark reporter."""
    return str(value or "").strip().lower() == "true"


def select_top_planners(campaign_table_csv: Path, *, top_count: int) -> list[PlannerSelection]:
    """Select top benchmark-success planners by success, then safety/performance tie-breakers."""
    if top_count < 1:
        raise ValueError("top_count must be positive")
    if not campaign_table_csv.is_file():
        raise FileNotFoundError(f"campaign table not found: {campaign_table_csv}")

    rows: list[PlannerSelection] = []
    with campaign_table_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not _parse_bool(raw.get("benchmark_success")):
                continue
            parsed = {
                "success_mean": _parse_float(raw.get("success_mean")),
                "collisions_mean": _parse_float(raw.get("collisions_mean")),
                "time_to_goal_norm_mean": _parse_float(raw.get("time_to_goal_norm_mean")),
                "near_misses_mean": _parse_float(raw.get("near_misses_mean")),
            }
            if any(value is None for value in parsed.values()):
                continue
            rows.append(
                PlannerSelection(
                    planner_key=str(raw.get("planner_key", "")).strip(),
                    success_mean=float(parsed["success_mean"]),
                    collisions_mean=float(parsed["collisions_mean"]),
                    time_to_goal_norm_mean=float(parsed["time_to_goal_norm_mean"]),
                    near_misses_mean=float(parsed["near_misses_mean"]),
                    execution_mode=str(raw.get("execution_mode", "")).strip(),
                    planner_group=str(raw.get("planner_group", "")).strip(),
                    benchmark_success=True,
                )
            )

    rows.sort(
        key=lambda row: (
            -row.success_mean,
            row.collisions_mean,
            row.time_to_goal_norm_mean,
            row.near_misses_mean,
            row.planner_key,
        )
    )
    if len(rows) < top_count:
        raise ValueError(f"only {len(rows)} benchmark-success rows available for top {top_count}")
    return rows[:top_count]


def load_selected_episode_rows(
    seed_episode_rows_csv: Path, selected_planners: set[str]
) -> list[EpisodeRow]:
    """Load per-seed rows for selected planners."""
    if not seed_episode_rows_csv.is_file():
        raise FileNotFoundError(f"seed episode rows not found: {seed_episode_rows_csv}")

    rows: list[EpisodeRow] = []
    with seed_episode_rows_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            planner_key = str(raw.get("planner_key", "")).strip()
            if planner_key not in selected_planners:
                continue
            success = _parse_float(raw.get("success"))
            collision = _parse_float(raw.get("collision"))
            seed = raw.get("seed")
            scenario_id = str(raw.get("scenario_id", "")).strip()
            if success is None or collision is None or seed is None or scenario_id == "":
                continue
            ttg_norm = _parse_float(raw.get("time_to_goal_norm"))
            if ttg_norm is None:
                ttg_norm = _parse_float(raw.get("time_to_goal"))
            rows.append(
                EpisodeRow(
                    scenario_id=scenario_id,
                    planner_key=planner_key,
                    seed=int(seed),
                    success=success,
                    collision=collision,
                    near_miss=_parse_float(raw.get("near_miss")),
                    time_to_goal_norm=ttg_norm,
                )
            )
    if not rows:
        raise ValueError("no selected planner rows found in seed episode rows")
    return rows


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean for a non-empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_seed_sensitivity_analysis(
    *,
    campaign_root: Path,
    top_planner_count: int = DEFAULT_TOP_PLANNER_COUNT,
    success_range_threshold: float = DEFAULT_SUCCESS_RANGE_THRESHOLD,
    hard_seed_threshold: float = DEFAULT_HARD_SEED_THRESHOLD,
    easy_seed_threshold: float = DEFAULT_EASY_SEED_THRESHOLD,
) -> dict[str, Any]:
    """Build the issue #1608 seed-sensitivity analysis payload."""
    reports_dir = campaign_root / "reports"
    selected = select_top_planners(reports_dir / "campaign_table.csv", top_count=top_planner_count)
    selected_keys = {row.planner_key for row in selected}
    episode_rows = load_selected_episode_rows(reports_dir / "seed_episode_rows.csv", selected_keys)

    by_scenario_seed: dict[tuple[str, int], list[EpisodeRow]] = defaultdict(list)
    by_scenario_planner: dict[tuple[str, str], list[EpisodeRow]] = defaultdict(list)
    for row in episode_rows:
        by_scenario_seed[(row.scenario_id, row.seed)].append(row)
        by_scenario_planner[(row.scenario_id, row.planner_key)].append(row)

    scenario_ids = sorted({row.scenario_id for row in episode_rows})
    scenario_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    selected_key_order = [row.planner_key for row in selected]

    for scenario_id in scenario_ids:
        scenario_seed_items = sorted(
            (seed, rows) for (sid, seed), rows in by_scenario_seed.items() if sid == scenario_id
        )
        seed_scores: list[tuple[int, float, float, int]] = []
        missing_cells: list[str] = []
        for seed, rows in scenario_seed_items:
            present = {row.planner_key for row in rows}
            missing = sorted(selected_keys - present)
            if missing:
                missing_cells.extend(f"{seed}:{planner}" for planner in missing)
            success_mean = _mean([row.success for row in rows])
            collision_mean = _mean([row.collision for row in rows])
            seed_scores.append((seed, success_mean, collision_mean, len(rows)))
            seed_rows.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "mean_success": success_mean,
                    "mean_collision": collision_mean,
                    "planner_count": len(rows),
                    "classification": "hard"
                    if success_mean <= hard_seed_threshold
                    else "easy"
                    if success_mean >= easy_seed_threshold
                    else "mixed",
                }
            )

        seed_success_values = [score[1] for score in seed_scores]
        hard_seeds = [
            seed for seed, score, _collision, _count in seed_scores if score <= hard_seed_threshold
        ]
        easy_seeds = [
            seed for seed, score, _collision, _count in seed_scores if score >= easy_seed_threshold
        ]
        success_range = max(seed_success_values) - min(seed_success_values)
        planner_ranges = []
        for planner_key in selected_key_order:
            planner_rows = by_scenario_planner[(scenario_id, planner_key)]
            planner_success_values = [row.success for row in planner_rows]
            if not planner_success_values:
                planner_ranges.append(
                    {
                        "planner_key": planner_key,
                        "success_range": 0.0,
                        "mean_success": None,
                        "seed_count": 0,
                    }
                )
                continue
            planner_ranges.append(
                {
                    "planner_key": planner_key,
                    "success_range": max(planner_success_values) - min(planner_success_values),
                    "mean_success": _mean(planner_success_values),
                    "seed_count": len({row.seed for row in planner_rows}),
                }
            )
        most_brittle = max(
            planner_ranges, key=lambda row: (row["success_range"], row["planner_key"])
        )

        if missing_cells or len(seed_scores) < 3:
            classification = "inconclusive"
        elif (
            success_range >= success_range_threshold
            and len(hard_seeds) >= 1
            and len(easy_seeds) >= 1
        ):
            classification = "seed_sensitive"
        else:
            classification = "not_seed_sensitive"

        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "classification": classification,
                "seed_count": len(seed_scores),
                "planner_count": len(selected_keys),
                "mean_success": _mean(seed_success_values),
                "min_seed_success": min(seed_success_values),
                "max_seed_success": max(seed_success_values),
                "seed_success_range": success_range,
                "hard_seeds": hard_seeds,
                "easy_seeds": easy_seeds,
                "hard_seed_count": len(hard_seeds),
                "easy_seed_count": len(easy_seeds),
                "missing_cell_count": len(missing_cells),
                "missing_cells": missing_cells,
                "most_brittle_planner": most_brittle["planner_key"],
                "most_brittle_planner_success_range": most_brittle["success_range"],
                "planner_ranges": planner_ranges,
            }
        )

    scenario_rows.sort(
        key=lambda row: (
            row["classification"] != "seed_sensitive",
            -row["seed_success_range"],
            row["scenario_id"],
        )
    )
    seed_rows.sort(key=lambda row: (row["scenario_id"], row["mean_success"], row["seed"]))
    classification_counts = {
        name: sum(row["classification"] == name for row in scenario_rows)
        for name in ("seed_sensitive", "not_seed_sensitive", "inconclusive")
    }
    systematically_hard_seeds = _summarize_systematic_seeds(seed_rows)

    return {
        "schema_version": "scenario-seed-sensitivity.v1",
        "campaign_root": str(campaign_root),
        "inputs": {
            "campaign_table_csv": str(reports_dir / "campaign_table.csv"),
            "seed_episode_rows_csv": str(reports_dir / "seed_episode_rows.csv"),
        },
        "planner_selection_rule": (
            "benchmark_success rows ranked by success_mean descending, then collisions_mean, "
            "time_to_goal_norm_mean, near_misses_mean, and planner_key ascending"
        ),
        "selected_planners": [row.__dict__ for row in selected],
        "thresholds": {
            "success_range_threshold": success_range_threshold,
            "hard_seed_threshold": hard_seed_threshold,
            "easy_seed_threshold": easy_seed_threshold,
            "minimum_seed_count": 3,
        },
        "classification_counts": classification_counts,
        "scenario_count": len(scenario_rows),
        "seed_rows": seed_rows,
        "scenario_rows": scenario_rows,
        "systematically_hard_seeds": systematically_hard_seeds,
        "claim_boundary": (
            "derived analysis over durable compact artifacts; diagnostic scenario prioritization, "
            "not causal mechanism proof or paper-facing significance"
        ),
    }


def _summarize_systematic_seeds(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize whether particular seed ids are often hard across scenarios."""
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_seed[int(row["seed"])].append(row)
    summaries = []
    for seed, rows in sorted(by_seed.items()):
        success_values = [float(row["mean_success"]) for row in rows]
        hard_scenarios = [
            str(row["scenario_id"]) for row in rows if row["classification"] == "hard"
        ]
        summaries.append(
            {
                "seed": seed,
                "scenario_count": len(rows),
                "mean_success": _mean(success_values),
                "hard_scenario_count": len(hard_scenarios),
                "hard_scenarios": hard_scenarios,
            }
        )
    summaries.sort(key=lambda row: (row["mean_success"], -row["hard_scenario_count"], row["seed"]))
    return summaries


def _format_float(value: float) -> str:
    """Format compact CSV/Markdown float output."""
    return f"{value:.4f}"


def write_outputs(analysis: dict[str, Any], output_dir: Path) -> None:
    """Write JSON, CSV, and Markdown analysis artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed_sensitivity_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    _write_scenario_csv(output_dir / "scenario_seed_sensitivity.csv", analysis["scenario_rows"])
    _write_seed_csv(output_dir / "seed_difficulty_summary.csv", analysis["seed_rows"])
    _write_markdown(output_dir / "seed_sensitivity_analysis.md", analysis)


def _write_scenario_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write scenario-level classification rows."""
    fieldnames = [
        "scenario_id",
        "classification",
        "seed_count",
        "planner_count",
        "mean_success",
        "min_seed_success",
        "max_seed_success",
        "seed_success_range",
        "hard_seeds",
        "easy_seeds",
        "most_brittle_planner",
        "most_brittle_planner_success_range",
        "missing_cell_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row},
                    "mean_success": _format_float(row["mean_success"]),
                    "min_seed_success": _format_float(row["min_seed_success"]),
                    "max_seed_success": _format_float(row["max_seed_success"]),
                    "seed_success_range": _format_float(row["seed_success_range"]),
                    "hard_seeds": " ".join(str(seed) for seed in row["hard_seeds"]),
                    "easy_seeds": " ".join(str(seed) for seed in row["easy_seeds"]),
                    "most_brittle_planner_success_range": _format_float(
                        row["most_brittle_planner_success_range"]
                    ),
                }
            )


def _write_seed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write scenario-seed difficulty rows."""
    fieldnames = [
        "scenario_id",
        "seed",
        "classification",
        "mean_success",
        "mean_collision",
        "planner_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row},
                    "mean_success": _format_float(row["mean_success"]),
                    "mean_collision": _format_float(row["mean_collision"]),
                }
            )


def _write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    """Write a compact Markdown report."""
    counts = analysis["classification_counts"]
    lines = [
        "# Scenario Seed Sensitivity Analysis",
        "",
        "## Contract",
        "",
        f"- Campaign root: `{analysis['campaign_root']}`",
        f"- Claim boundary: {analysis['claim_boundary']}.",
        f"- Planner selection: {analysis['planner_selection_rule']}.",
        "",
        "## Selected Planners",
        "",
        "| Rank | Planner | Success | Collision | Time to goal norm | Near misses | Mode |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(analysis["selected_planners"], start=1):
        lines.append(
            "| {rank} | `{planner}` | {success} | {collision} | {ttg} | {near} | `{mode}` |".format(
                rank=rank,
                planner=row["planner_key"],
                success=_format_float(row["success_mean"]),
                collision=_format_float(row["collisions_mean"]),
                ttg=_format_float(row["time_to_goal_norm_mean"]),
                near=_format_float(row["near_misses_mean"]),
                mode=row["execution_mode"],
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Scenarios classified: `{analysis['scenario_count']}`.",
            f"- Seed-sensitive: `{counts['seed_sensitive']}`.",
            f"- Not seed-sensitive: `{counts['not_seed_sensitive']}`.",
            f"- Inconclusive: `{counts['inconclusive']}`.",
            "",
            "## Seed-Sensitive Scenarios",
            "",
            "| Scenario | Range | Mean success | Hard seeds | Easy seeds | Most brittle planner |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    sensitive_rows = [
        row for row in analysis["scenario_rows"] if row["classification"] == "seed_sensitive"
    ]
    for row in sensitive_rows[:20]:
        lines.append(
            "| `{scenario}` | {range_} | {mean} | `{hard}` | `{easy}` | `{planner}` |".format(
                scenario=row["scenario_id"],
                range_=_format_float(row["seed_success_range"]),
                mean=_format_float(row["mean_success"]),
                hard=" ".join(str(seed) for seed in row["hard_seeds"]) or "none",
                easy=" ".join(str(seed) for seed in row["easy_seeds"]) or "none",
                planner=row["most_brittle_planner"],
            )
        )
    if not sensitive_rows:
        lines.append("| none | | | | | |")
    lines.extend(
        [
            "",
            "## Hardest Seed IDs Across Scenarios",
            "",
            "| Seed | Mean success | Hard scenario count |",
            "|---:|---:|---:|",
        ]
    )
    for row in analysis["systematically_hard_seeds"][:10]:
        lines.append(
            f"| {row['seed']} | {_format_float(row['mean_success'])} | "
            f"{row['hard_scenario_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Limit",
            "",
            "This derived analysis can prioritize follow-up scenario inspection. It does not prove a "
            "causal mechanism for any scenario, and it should not be reused as paper-facing "
            "significance evidence without a pre-specified larger seed budget.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    analysis = build_seed_sensitivity_analysis(
        campaign_root=args.campaign_root,
        top_planner_count=args.top_planner_count,
        success_range_threshold=args.success_range_threshold,
        hard_seed_threshold=args.hard_seed_threshold,
        easy_seed_threshold=args.easy_seed_threshold,
    )
    write_outputs(analysis, args.output_dir)
    counts = analysis["classification_counts"]
    print(
        "scenario_seed_sensitivity: "
        f"{counts['seed_sensitive']} seed-sensitive, "
        f"{counts['not_seed_sensitive']} not seed-sensitive, "
        f"{counts['inconclusive']} inconclusive"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
