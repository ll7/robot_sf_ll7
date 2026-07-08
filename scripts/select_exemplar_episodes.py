"""Select exemplar episodes from campaign JSONL.

Reads a campaign episodes.jsonl, groups by planner/mechanism/outcome cells,
selects median/best/worst by a configured metric, and writes a selection
manifest.

Usage:
  uv run python scripts/select_exemplar_episodes.py \\
    --episodes <episodes.jsonl> \\
    --group-by planner_key,outcome \\
    --metric path_efficiency \\
    --modes median,best,worst \\
    --output-manifest output/exemplars/selection_manifest.json

Part of issue #4778.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.errors import EpisodeRecordInputError
from robot_sf.benchmark.exemplar_selection import (
    METRIC_DIRECTIONS,
    ExemplarSelectionError,
    build_manifest,
    save_manifest,
    select_exemplars,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Select exemplar episodes from campaign JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--episodes",
        required=True,
        type=Path,
        help="Path to episodes.jsonl file.",
    )
    parser.add_argument(
        "--group-by",
        default="planner_key,outcome",
        help="Comma-separated grouping keys (default: planner_key,outcome).",
    )
    parser.add_argument(
        "--metric",
        default="path_efficiency",
        help="Metric to rank by (default: path_efficiency).",
    )
    parser.add_argument(
        "--metric-direction",
        choices=["lower", "higher"],
        default=None,
        help="Metric direction. Auto-detected for known metrics.",
    )
    parser.add_argument(
        "--modes",
        default="median,best,worst",
        help="Comma-separated selection modes (default: median,best,worst).",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        type=Path,
        help="Output path for selection_manifest.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run exemplar selection CLI."""
    args = parse_args(argv)

    episodes_path: Path = args.episodes
    if not episodes_path.exists():
        print(f"Error: episodes file not found: {episodes_path}", file=sys.stderr)
        return 1

    group_by = [k.strip() for k in args.group_by.split(",") if k.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    metric = args.metric
    metric_direction = args.metric_direction

    # Show direction hint
    if metric_direction is None:
        auto = METRIC_DIRECTIONS.get(metric, "lower")
        print(f"Auto-detected metric direction for '{metric}': {auto}")

    try:
        episodes = read_jsonl(episodes_path)
    except (EpisodeRecordInputError, FileNotFoundError) as exc:
        print(f"Error reading episodes: {exc}", file=sys.stderr)
        return 1

    print(f"Read {len(episodes)} episodes from {episodes_path}")

    try:
        selected, skipped = select_exemplars(
            episodes,
            group_by=group_by,
            metric=metric,
            metric_direction=metric_direction,
            modes=modes,
        )
    except ExemplarSelectionError as exc:
        print(f"Selection error: {exc}", file=sys.stderr)
        return 1

    manifest = build_manifest(
        source_episodes_path=episodes_path,
        group_by=group_by,
        metric=metric,
        metric_direction=metric_direction,
        selected=selected,
        skipped_cells=skipped,
    )

    output_path = save_manifest(manifest, args.output_manifest)
    print(f"Selected {len(selected)} exemplars across {len(group_by)} grouping keys")
    if skipped:
        print(f"Skipped {len(skipped)} cells (metric missing or non-finite)")
    print(f"Manifest written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
