#!/usr/bin/env python3
"""Select S10/h500 cells for issue #1477 mechanism-claim review.

The issue #1454 archive contains per-episode summaries, not step traces or videos. This helper
therefore produces a compact, reproducible cell table and marks mechanism claims as unresolved
unless future trace/video artifacts are supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

TARGET_SCENARIOS = {
    "francis2023_narrow_doorway": "weak_spot",
    "classic_station_platform_medium": "weak_spot",
    "francis2023_robot_crowding": "near_miss_heavy_win",
    "francis2023_narrow_hallway": "near_miss_heavy_win",
    "classic_bottleneck_high": "optional_large_success_gain",
}

CANDIDATE_PLANNERS = {
    "hybrid_rule_v3_fast_progress",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
}

FIELDNAMES = [
    "review_reason",
    "scenario_id",
    "planner_key",
    "planner_group",
    "seed",
    "episode_id",
    "status",
    "termination_reason",
    "success",
    "collisions",
    "near_misses",
    "clearing_distance_min",
    "time_to_goal_norm",
    "summary_observation",
    "mechanism_claim",
    "support_status",
    "source_episode_file",
]


@dataclass(frozen=True)
class EpisodeCell:
    """A selected episode-summary row for mechanism-claim audit."""

    review_reason: str
    scenario_id: str
    planner_key: str
    seed: int
    episode_id: str
    status: str
    termination_reason: str
    success: bool
    collisions: int
    near_misses: int
    clearing_distance_min: float | None
    time_to_goal_norm: float | None
    source_episode_file: Path

    @property
    def planner_group(self) -> str:
        """Return the issue #1462 comparison group for this planner."""
        return "candidate" if self.planner_key in CANDIDATE_PLANNERS else "core"

    def to_row(self, raw_campaign_dir: Path) -> dict[str, str]:
        """Serialize the selected cell for a compact evidence table."""
        summary_observation = _summary_observation(self)
        return {
            "review_reason": self.review_reason,
            "scenario_id": self.scenario_id,
            "planner_key": self.planner_key,
            "planner_group": self.planner_group,
            "seed": str(self.seed),
            "episode_id": self.episode_id,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "success": str(self.success).lower(),
            "collisions": str(self.collisions),
            "near_misses": str(self.near_misses),
            "clearing_distance_min": _optional_float(self.clearing_distance_min),
            "time_to_goal_norm": _optional_float(self.time_to_goal_norm),
            "summary_observation": summary_observation,
            "mechanism_claim": "waiting/yielding/hesitation/squeezing/risk-taking",
            "support_status": "not_resolved_summary_only_no_trace_or_video",
            "source_episode_file": str(self.source_episode_file.relative_to(raw_campaign_dir)),
        }


def _optional_float(value: float | None) -> str:
    """Format an optional floating-point metric for CSV output."""
    return "" if value is None else f"{value:.6f}"


def _summary_observation(cell: EpisodeCell) -> str:
    """Build a conservative summary-only observation for a selected cell."""
    if cell.review_reason == "weak_spot":
        return (
            f"{cell.termination_reason} outcome with success={cell.success}, "
            f"collisions={cell.collisions}, near_misses={cell.near_misses}; summary metrics show "
            "the weak cell but do not identify the behavior that caused it."
        )
    return (
        f"{cell.termination_reason} outcome with success={cell.success}, "
        f"collisions={cell.collisions}, near_misses={cell.near_misses}; summary metrics show "
        "near-miss exposure but not whether the planner waited, yielded, squeezed, or took risk."
    )


def _planner_key(run_dir: Path) -> str:
    """Convert a run directory name to the planner key used in issue #1462 tables."""
    suffix = "__differential_drive"
    name = run_dir.name
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _load_cells(raw_campaign_dir: Path) -> list[EpisodeCell]:
    """Load all target scenario rows from campaign episode summaries."""
    cells: list[EpisodeCell] = []
    for episode_file in sorted(raw_campaign_dir.glob("runs/*__differential_drive/episodes.jsonl")):
        planner_key = _planner_key(episode_file.parent)
        with episode_file.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                scenario_id = record.get("scenario_id")
                if scenario_id not in TARGET_SCENARIOS:
                    continue
                metrics = record.get("metrics", {})
                if not isinstance(metrics, dict):
                    metrics = {}
                cells.append(
                    EpisodeCell(
                        review_reason=TARGET_SCENARIOS[str(scenario_id)],
                        scenario_id=str(scenario_id),
                        planner_key=planner_key,
                        seed=int(record["seed"]),
                        episode_id=str(record["episode_id"]),
                        status=str(record.get("status", "")),
                        termination_reason=str(record.get("termination_reason", "")),
                        success=bool(metrics.get("success", False)),
                        collisions=int(metrics.get("collisions", 0) or 0),
                        near_misses=int(metrics.get("near_misses", 0) or 0),
                        clearing_distance_min=_maybe_float(metrics.get("clearing_distance_min")),
                        time_to_goal_norm=_maybe_float(metrics.get("time_to_goal_norm")),
                        source_episode_file=episode_file,
                    )
                )
    return cells


def _maybe_float(value: object) -> float | None:
    """Return a float for numeric metric values."""
    if isinstance(value, int | float):
        return float(value)
    return None


def _select_cells(cells: list[EpisodeCell], per_scenario: int) -> list[EpisodeCell]:
    """Select representative candidate cells for each issue #1477 target scenario."""
    selected: list[EpisodeCell] = []
    for scenario_id, reason in TARGET_SCENARIOS.items():
        candidates = [
            cell
            for cell in cells
            if cell.scenario_id == scenario_id and cell.planner_key in CANDIDATE_PLANNERS
        ]
        if reason == "weak_spot":
            candidates.sort(key=lambda cell: (cell.success, -cell.near_misses, cell.seed))
        else:
            candidates.sort(key=lambda cell: (not cell.success, -cell.near_misses, cell.seed))
        selected.extend(candidates[:per_scenario])
    return selected


def _sha256(path: Path) -> str:
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write selected cell rows as CSV."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a compact Markdown table for human review."""
    columns = [
        "review_reason",
        "scenario_id",
        "planner_key",
        "seed",
        "success",
        "collisions",
        "near_misses",
        "support_status",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[column] for column in columns) + " |\n")


def main() -> int:
    """Run the issue #1477 summary-only mechanism review."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-campaign-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-scenario", default=2, type=int)
    args = parser.parse_args()

    archive_sha256 = _sha256(args.archive)
    if archive_sha256 != args.expected_archive_sha256:
        raise SystemExit(
            f"archive SHA256 mismatch: expected {args.expected_archive_sha256}, got {archive_sha256}"
        )

    cells = _load_cells(args.raw_campaign_dir)
    selected = _select_cells(cells, args.per_scenario)
    rows = [cell.to_row(args.raw_campaign_dir) for cell in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "reviewed_cells.csv", rows)
    _write_markdown(args.output_dir / "reviewed_cells.md", rows)
    summary = {
        "status": "summary_only_no_trace_or_video",
        "archive_sha256": archive_sha256,
        "raw_campaign_archive_member": args.raw_campaign_dir.name,
        "selected_cell_count": len(rows),
        "target_scenarios": TARGET_SCENARIOS,
        "candidate_planners": sorted(CANDIDATE_PLANNERS),
        "mechanism_claim_result": "unsupported_without_step_trace_or_video",
        "reviewed_cells_csv": "reviewed_cells.csv",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
