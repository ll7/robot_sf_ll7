#!/usr/bin/env python3
"""Render multi-planner trajectory overlay figures.

Overlay robot trajectories from multiple planners on the same scenario+seed
into a single provenance-stamped figure.  Uses the shared colorblind-safe
planner palette from robot_sf.benchmark.figures.style.

This is a visual comparison tool — not benchmark evidence.

Usage:
    uv run python scripts/render_multi_planner_trajectory_overlay.py \\
      --episodes campaign/episodes.jsonl \\
      --scenario-id corridor \\
      --seed 42 \\
      --planners orca,ppo \\
      --out-dir output/overlays

Exit codes:
    0  Success
    1  Missing or invalid inputs
    2  Planner trajectory not found (use --allow-missing to continue)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.multi_planner_overlay import (
    MultiPlannerOverlayError,
    build_overlay_figure,
    load_episodes,
    select_episodes_for_overlay,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render multi-planner trajectory overlay figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--episodes",
        required=True,
        help="Path to campaign episodes.jsonl.",
    )
    parser.add_argument(
        "--scenario-id",
        required=True,
        help="Scenario identifier to filter on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed to filter on.",
    )
    parser.add_argument(
        "--planners",
        required=True,
        help="Comma-separated planner keys to overlay.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats (pdf, png, svg). Default: pdf,png.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Continue even if a planner has no trajectory data.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats. Default: 300.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the multi-planner trajectory overlay CLI.

    Returns:
        Exit code: 0 for success, 1 for errors, 2 for missing planner data.
    """
    args = _parse_args(argv)

    planner_keys = [p.strip() for p in args.planners.split(",") if p.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    if not planner_keys:
        print("Error: --planners must not be empty", file=sys.stderr)
        return 1

    if not formats:
        print("Error: --formats must not be empty", file=sys.stderr)
        return 1

    # Load episodes
    try:
        episodes = load_episodes(args.episodes)
    except MultiPlannerOverlayError as exc:
        print(f"Error loading episodes: {exc}", file=sys.stderr)
        return 1

    # Select episodes for overlay
    try:
        trajectory_rows = select_episodes_for_overlay(
            episodes,
            scenario_id=args.scenario_id,
            seed=args.seed,
            planner_keys=planner_keys,
        )
    except MultiPlannerOverlayError as exc:
        if args.allow_missing:
            return _handle_missing_planners(episodes, args, formats)
        print(f"Error: {exc}", file=sys.stderr)
        print("Use --allow-missing to continue with available planners.", file=sys.stderr)
        return 2

    # Build output path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_base = out_dir / f"{args.scenario_id}_seed{args.seed}"

    # Render
    try:
        saved = build_overlay_figure(
            trajectory_rows,
            output_base=output_base,
            formats=formats,
            dpi=args.dpi,
        )
    except (MultiPlannerOverlayError, ValueError) as exc:
        print(f"Error rendering overlay: {exc}", file=sys.stderr)
        return 1

    # Print results
    print(f"Overlay rendered: {len(saved)} file(s)")
    for path in saved:
        print(f"  {path}")

    return 0


def _handle_missing_planners(
    episodes: list[dict],
    args: argparse.Namespace,
    formats: list[str],
) -> int:
    """Handle missing planners when --allow-missing is set.

    Returns:
        Exit code.
    """
    from robot_sf.benchmark.multi_planner_overlay import _find_trajectory_row

    available: list[str] = []
    for pk in args.planners.split(","):
        pk = pk.strip()
        if not pk:
            continue
        # Reuse the canonical matcher so a planner counts as available only
        # when it actually resolves to valid trajectory data (shared seed
        # coercion + fail-closed extraction).
        if _find_trajectory_row(episodes, args.scenario_id, args.seed, pk) is not None:
            available.append(pk)

    if not available:
        print("Error: no trajectory data available", file=sys.stderr)
        return 2

    # Select with available planners
    try:
        trajectory_rows = select_episodes_for_overlay(
            episodes,
            scenario_id=args.scenario_id,
            seed=args.seed,
            planner_keys=available,
        )
    except MultiPlannerOverlayError:
        print("Error: no trajectories available after filtering", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_base = out_dir / f"{args.scenario_id}_seed{args.seed}"

    try:
        saved = build_overlay_figure(
            trajectory_rows,
            output_base=output_base,
            formats=formats,
            dpi=args.dpi,
        )
    except (MultiPlannerOverlayError, ValueError) as exc:
        print(f"Error rendering overlay: {exc}", file=sys.stderr)
        return 1

    print(f"Overlay rendered: {len(saved)} file(s)")
    for path in saved:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
