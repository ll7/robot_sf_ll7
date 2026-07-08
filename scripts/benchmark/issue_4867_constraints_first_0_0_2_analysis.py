#!/usr/bin/env python3
"""Constraints-first vs scalarized scoring analysis over 0.0.2 release (issue #4867).

This script performs a pure re-analysis of the frozen 0.0.2 release benchmark table
to compare scalarized (SNQI) ranking with constraints-first (lexicographic safety)
ranking. No new simulation is performed; only retained episode aggregates are consumed.

Deliverables:
1. Per-planner constraints-first metrics (admissibility, collision UCB, survivorship-aware metrics)
2. Side-by-side ranking table: scalarized vs constraints-first ordering
3. Ranking-inversion diagnostic (which planners change rank between orderings)
4. Short methods note on the constraints-first ordering rule

Usage:
    python scripts/benchmark/issue_4867_constraints_first_0_0_2_analysis.py \\
        --bundle-dir /tmp/issue_4867_0_0_2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle \\
        --output-dir output/issue_4867_constraints_first_0_0_2_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

RELEASE_TAG = "0.0.2"
BUNDLE_NAME = (
    "paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle"
)
ISSUE = 4867


@dataclass(frozen=True, slots=True)
class ReleasePaths:
    """Paths to key artifacts in the 0.0.2 release bundle."""

    bundle_dir: Path
    payload_dir: Path
    runs_dir: Path
    campaign_summary: Path
    checksums: Path

    @classmethod
    def from_bundle_dir(cls, bundle_dir: Path) -> ReleasePaths:
        """Construct from the bundle directory root."""
        payload_dir = bundle_dir / "payload"
        return cls(
            bundle_dir=bundle_dir,
            payload_dir=payload_dir,
            runs_dir=payload_dir / "runs",
            campaign_summary=payload_dir / "reports" / "campaign_summary.json",
            checksums=bundle_dir / "checksums.sha256",
        )


def load_campaign_summary(paths: ReleasePaths) -> dict:
    """Load the campaign summary JSON."""
    return json.loads(paths.campaign_summary.read_text(encoding="utf-8"))


def extract_compensatory_scores(campaign: dict) -> dict[str, float]:
    """Extract planner -> SNQI compensatory scores from campaign summary.

    SNQI is the scalarized composite used in the 0.0.2 release.
    Higher scores are better.
    """
    scores = {}
    for row in campaign["planner_rows"]:
        planner = row["planner_key"]
        snqi = row.get("snqi_mean")
        if snqi is None:
            raise ValueError(f"Planner {planner} missing SNQI score")
        try:
            scores[planner] = float(snqi)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid SNQI score for planner {planner}: {snqi}") from exc
    return scores


def load_all_episodes(paths: ReleasePaths) -> list[dict]:
    """Load and merge all episode JSONL files from the runs directory.

    Each planner has a runs/<planner>__differential_drive/episodes.jsonl file.
    This function loads all episodes and adds a top-level 'planner' field.
    """
    episodes = []
    for planner_dir in sorted(paths.runs_dir.iterdir()):
        if not planner_dir.is_dir():
            continue
        if not planner_dir.name.endswith("__differential_drive"):
            continue
        planner_name = planner_dir.name.replace("__differential_drive", "")
        episodes_path = planner_dir / "episodes.jsonl"
        if not episodes_path.exists():
            raise ValueError(f"Missing episodes file: {episodes_path}")
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episode["planner"] = planner_name
            episodes.append(episode)
    return episodes


def transform_episode_to_constraints_first_format(episode: Mapping) -> dict:
    """Transform a raw episode record to the format expected by constraints_first_scoring.py.

    Input schema (0.0.2 episode):
    - algo: planner name
    - metrics.collisions: collision count
    - metrics.near_misses: near-miss count
    - metrics.comfort_exposure: comfort metric (lower is better)
    - metrics.time_to_goal_norm: time to goal normalized (lower is better)
    - outcome.timeout_event: whether episode timed out
    - outcome.collision_event: whether a collision occurred

    Output schema (constraints_first_scoring.py):
    - planner: planner name
    - collisions: collision count
    - near_miss_severity: near-miss count (using count as severity proxy)
    - comfort: comfort metric (inverted so higher is better)
    - efficiency: time-to-goal metric (inverted so higher is better)
    - timeout: whether episode timed out
    - deadlock: always False (not tracked in 0.0.2)
    - safe_success: success without safety violations
    """
    metrics = episode.get("metrics")
    if metrics is None:
        metrics = {}
    outcome = episode.get("outcome")
    if outcome is None:
        outcome = {}
    planner = episode.get("algo")
    if planner is None:
        planner = episode.get("planner")
    if planner is None:
        planner = "unknown"

    # Extract and invert metrics so higher is better for ranking
    comfort_raw = metrics.get("comfort_exposure", 0.0)
    efficiency_raw = metrics.get("time_to_goal_norm", 1.0)

    # Safe success: route complete without collisions or near misses
    # Use outcome.collision_event and metrics.near_misses
    collisions = metrics.get("collisions", 0)
    near_misses = metrics.get("near_misses", 0)
    route_complete = outcome.get("route_complete", False)
    safe_success = route_complete and collisions == 0 and near_misses == 0

    return {
        "planner": planner,
        "collisions": collisions,
        "near_miss_severity": near_misses,  # Using count as severity proxy
        "comfort": -comfort_raw,  # Invert so higher is better
        "efficiency": -efficiency_raw,  # Invert so higher is better (faster is better)
        "timeout": outcome.get("timeout_event", False),
        "deadlock": False,  # Not tracked in 0.0.2 release
        "safe_success": safe_success,
    }


def write_transformed_episodes(episodes: Sequence[dict], output_path: Path) -> None:
    """Write transformed episodes to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            transformed = transform_episode_to_constraints_first_format(episode)
            f.write(json.dumps(transformed) + "\n")


def write_compensatory_scores(scores: dict[str, float], output_path: Path) -> None:
    """Write compensatory scores to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_methods_note(output_path: Path) -> None:
    """Write a short methods note describing the constraints-first ordering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    note = f"""# Constraints-First vs Scalarized Ranking Analysis (Issue #{ISSUE}, Release {RELEASE_TAG})

## Analysis Type

Pure re-analysis of existing retained results from the frozen {RELEASE_TAG} release.
No new simulation was performed.

## Input Artifacts

- Release bundle: `{BUNDLE_NAME}.tar.gz`
- Release tag: `{RELEASE_TAG}`
- Planners analyzed: 7 (goal, orca, ppo, prediction_planner, sacadrl, social_force, socnav_sampling)
- Total episodes: 987 (141 per planner)
- Kinematics: differential_drive

## Scalarized (Compensatory) Ranking

The {RELEASE_TAG} release uses SNQI (Social Navigation Quality Index) as the headline
scalarized composite metric. SNQI is a weighted aggregate of safety, progress, comfort,
and efficiency components where tradeoffs are permitted — a collision can be numerically
offset by faster or smoother motion.

## Constraints-First (Lexicographic) Ranking

The constraints-first ordering applies non-compensatory lexicographic gating:

1. **Admissibility gate**: An episode must have zero collisions to be admissible.
   Episodes with collisions fail immediately regardless of other metrics.
2. **Secondary gate**: Near-miss severity and timeout/deadlock are also checked.
3. **Ranking metric**: Among admissible episodes, comfort and efficiency are used
   to break ties, but only for planners with equivalent safety records.

The key distinction is that safety violations cannot be compensated away: a planner
with even a single collision cannot outrank a collision-free planner.

## Metric Transformations

To apply the constraints-first scoring layer to the {RELEASE_TAG} data:

- **Comfort**: Original `comfort_exposure` (lower is better) is inverted so higher is better.
- **Efficiency**: Original `time_to_goal_norm` (lower is better) is inverted so higher is better.
- **Near-miss severity**: Using near-miss count as a proxy (near-miss distance not in {RELEASE_TAG}).
- **Safe success**: Defined as route_complete AND zero collisions AND zero near misses.

## Ranking Inversion Diagnostic

The analysis explicitly compares the two orderings to detect rank changes:
- Planners may shift positions when safety is prioritized over composite scores.
- Inversions (if any) indicate that the compensatory ranking obscures safety differences.

## Confidence Intervals

Collision rates are reported with 95% upper confidence bounds (Clopper–Pearson exact
binomial interval) so low-episode planners cannot appear collision-free.

## Evidence Grade

Diagnostic proxy: CPU-only analysis of retained {RELEASE_TAG} data. No new simulation.
Suitable for methodology exploration and ranking-robustness checking, but not for
paper-facing claims without additional validation.
"""
    output_path.write_text(note, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the constraints-first vs scalarized scoring analysis."""
    parser = argparse.ArgumentParser(
        description="Constraints-first vs scalarized scoring analysis over 0.0.2 release"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Path to extracted 0.0.2 release bundle directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"output/issue_{ISSUE}_constraints_first_{RELEASE_TAG}_analysis"),
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Skip episode transformation if already done",
    )
    args = parser.parse_args(argv)

    # Validate bundle directory
    paths = ReleasePaths.from_bundle_dir(args.bundle_dir)
    if not paths.campaign_summary.exists():
        sys.stderr.write(f"ERROR: Campaign summary not found: {paths.campaign_summary}\n")
        return 2

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and extract compensatory scores (SNQI)
    print("Loading campaign summary...")
    campaign = load_campaign_summary(paths)
    compensatory_scores = extract_compensatory_scores(campaign)
    print(f"  Found {len(compensatory_scores)} planners with SNQI scores")

    # Write compensatory scores for the constraints_first_scoring.py CLI
    compensatory_path = output_dir / "compensatory_scores.json"
    write_compensatory_scores(compensatory_scores, compensatory_path)
    print(f"  Wrote {compensatory_path}")

    # Step 2: Transform episodes to constraints-first format
    transformed_path = output_dir / "transformed_episodes.jsonl"
    if not args.skip_transform:
        print("Loading and transforming episodes...")
        episodes = load_all_episodes(paths)
        print(f"  Loaded {len(episodes)} episodes")
        write_transformed_episodes(episodes, transformed_path)
        print(f"  Wrote {transformed_path}")
    else:
        print(f"  Skipping transform (using existing {transformed_path})")

    # Step 3: Write methods note
    methods_note_path = output_dir / "METHODS.md"
    write_methods_note(methods_note_path)
    print(f"  Wrote methods note to {methods_note_path}")

    # Step 4: Run constraints-first scoring in-process.
    print("\nRunning constraints-first scoring...")
    from robot_sf.benchmark.constraints_first_scoring import (
        build_constraints_first_report,
        group_episodes_by_planner,
    )

    records = [
        json.loads(line)
        for line in transformed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    planner_episodes = group_episodes_by_planner(records)
    report = build_constraints_first_report(
        planner_episodes,
        compensatory_scores=compensatory_scores,
        confidence=0.95,
    )
    output_abs = output_dir / "constraints_first_report.json"
    output_abs.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"  Wrote {output_dir / 'constraints_first_report.json'}")

    # Step 5: Generate side-by-side ranking table
    print("\nGenerating ranking comparison...")
    report = json.loads((output_dir / "constraints_first_report.json").read_text(encoding="utf-8"))

    ranking_comparison = generate_ranking_comparison(
        compensatory_scores, report["ranking_inversion"], report
    )
    comparison_path = output_dir / "ranking_comparison.json"
    comparison_path.write_text(
        json.dumps(ranking_comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"  Wrote {comparison_path}")

    # Generate CSV table
    csv_path = output_dir / "ranking_comparison.csv"
    write_ranking_comparison_csv(ranking_comparison, csv_path)
    print(f"  Wrote {csv_path}")

    print(f"\nAnalysis complete. Results in: {output_dir}")
    return 0


def generate_ranking_comparison(
    compensatory_scores: dict[str, float], ranking_inversion: dict, report: dict
) -> dict:
    """Generate a side-by-side comparison of the two rankings."""
    per_planner = ranking_inversion["per_planner"]

    # Build ordered lists for each ranking
    comp_ordered = sorted(compensatory_scores.items(), key=lambda x: (-x[1], x[0]))
    cons_ordered = sorted(per_planner.items(), key=lambda x: (x[1]["constraints_first_rank"], x[0]))

    # Extract key metrics from the full report
    per_planner_metrics = {}
    for planner, summary in report.get("per_planner", {}).items():
        per_planner_metrics[planner] = {
            "admissible_rate": summary.get("admissible_rate"),
            "collision_rate": summary.get("collision_rate"),
            "collision_upper_confidence_bound": summary.get("collision_upper_confidence_bound"),
            "n_safe_success": summary.get("comfort", {}).get("n_safe_success"),
            "n_episodes": summary.get("n_episodes"),
        }

    return {
        "schema_version": "issue_4867_ranking_comparison.v1",
        "compensatory_ordering": [name for name, _ in comp_ordered],
        "constraints_first_ordering": [name for name, _ in cons_ordered],
        "per_planner": per_planner,
        "per_planner_metrics": per_planner_metrics,
        "inverted_planners": ranking_inversion["inverted_planners"],
        "any_inversion": ranking_inversion["any_inversion"],
    }


def write_ranking_comparison_csv(comparison: dict, output_path: Path) -> None:
    """Write ranking comparison as a CSV table."""
    import csv

    per_planner = comparison["per_planner"]
    per_planner_metrics = comparison.get("per_planner_metrics", {})

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "planner",
                "compensatory_rank",
                "constraints_first_rank",
                "rank_delta",
                "admissible_rate",
                "collision_rate",
                "collision_ucb",
                "n_safe_success",
                "n_episodes",
            ]
        )
        for planner, data in sorted(per_planner.items(), key=lambda x: x[1]["compensatory_rank"]):
            metrics = per_planner_metrics.get(planner, {})
            writer.writerow(
                [
                    planner,
                    data["compensatory_rank"],
                    data["constraints_first_rank"],
                    data["rank_delta"],
                    metrics.get("admissible_rate", "N/A"),
                    metrics.get("collision_rate", "N/A"),
                    metrics.get("collision_upper_confidence_bound", "N/A"),
                    metrics.get("n_safe_success", "N/A"),
                    metrics.get("n_episodes", "N/A"),
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main())
