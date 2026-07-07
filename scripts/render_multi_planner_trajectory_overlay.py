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

Or with a selection manifest from the exemplar selector (#4778):
    uv run python scripts/render_multi_planner_trajectory_overlay.py \\
      --episodes campaign/episodes.jsonl \\
      --selection-manifest output/exemplars/selection_manifest.json \\
      --out-dir output/overlays

Exit codes:
    0  Success
    1  Missing or invalid inputs
    2  Planner trajectory not found (use --allow-missing to continue)
"""

from __future__ import annotations

import argparse
import json
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
        default=None,
        help="Scenario identifier to filter on (required unless --selection-manifest).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to filter on (required unless --selection-manifest).",
    )
    parser.add_argument(
        "--planners",
        default=None,
        help="Comma-separated planner keys to overlay (required unless --selection-manifest).",
    )
    parser.add_argument(
        "--selection-manifest",
        default=None,
        help="Path to exemplar selection_manifest.json (from select_exemplar_episodes.py). "
        "When provided, --scenario-id, --seed, and --planners are derived from the manifest.",
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


def _load_selection_manifest(path: str) -> dict:
    """Load and validate a selection manifest file.

    Returns:
        Parsed manifest dict.

    Raises:
        SystemExit: On missing file or invalid schema.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        print(f"Error: selection manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error reading selection manifest: {exc}", file=sys.stderr)
        sys.exit(1)
    if manifest.get("schema_version") != "exemplar-selection.v1":
        print(
            f"Error: unsupported manifest schema: {manifest.get('schema_version')}",
            file=sys.stderr,
        )
        sys.exit(1)
    return manifest


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    """Run the multi-planner trajectory overlay CLI.

    Returns:
        Exit code: 0 for success, 1 for errors, 2 for missing planner data.
    """
    args = _parse_args(argv)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    if not formats:
        print("Error: --formats must not be empty", file=sys.stderr)
        return 1

    # Load episodes
    try:
        episodes = load_episodes(args.episodes)
    except MultiPlannerOverlayError as exc:
        print(f"Error loading episodes: {exc}", file=sys.stderr)
        return 1

    # Resolve overlay parameters: from manifest or CLI args
    manifest = None
    if args.selection_manifest:
        manifest = _load_selection_manifest(args.selection_manifest)
        selected = manifest.get("selected", [])
        if not selected:
            print("Error: selection manifest contains no selected episodes", file=sys.stderr)
            return 1

        # Group by (scenario_id, seed) and collect planners
        cells: dict[tuple[str, int], list[str]] = {}
        for entry in selected:
            scenario = entry.get("scenario_id", "")
            seed_val = entry.get("seed", 0)
            planner = entry.get("planner_key", "")
            key = (scenario, seed_val)
            cells.setdefault(key, []).append(planner)

        if args.scenario_id is not None and args.seed is not None:
            override_key = (args.scenario_id, args.seed)
            if override_key in cells:
                cells = {override_key: cells[override_key]}
            else:
                print(
                    f"Warning: scenario={args.scenario_id} seed={args.seed} not in manifest; "
                    "using all manifest cells",
                    file=sys.stderr,
                )
        # If neither --scenario-id nor --seed is provided, render all cells
    elif args.scenario_id is None or args.seed is None or args.planners is None:
        print(
            "Error: --scenario-id, --seed, and --planners are required "
            "unless --selection-manifest is provided",
            file=sys.stderr,
        )
        return 1
    else:
        planner_keys = [p.strip() for p in args.planners.split(",") if p.strip()]
        if not planner_keys:
            print("Error: --planners must not be empty", file=sys.stderr)
            return 1
        cells = {(args.scenario_id, args.seed): planner_keys}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for (scenario_id, seed), planner_keys in cells.items():
        exit_code = max(
            exit_code,
            _render_one_cell(
                episodes=episodes,
                scenario_id=scenario_id,
                seed=seed,
                planner_keys=planner_keys,
                out_dir=out_dir,
                formats=formats,
                dpi=args.dpi,
                allow_missing=args.allow_missing,
                manifest_path=args.selection_manifest,
            ),
        )

    return exit_code


def _render_one_cell(  # noqa: PLR0913
    *,
    episodes: list[dict],
    scenario_id: str,
    seed: int,
    planner_keys: list[str],
    out_dir: Path,
    formats: list[str],
    dpi: int,
    allow_missing: bool,
    manifest_path: str | None,
) -> int:
    """Render overlay for one (scenario, seed) cell.

    Returns:
        Exit code.
    """
    try:
        trajectory_rows = select_episodes_for_overlay(
            episodes,
            scenario_id=scenario_id,
            seed=seed,
            planner_keys=planner_keys,
        )
    except MultiPlannerOverlayError as exc:
        if allow_missing:
            return _handle_missing_planners(
                episodes=episodes,
                scenario_id=scenario_id,
                seed=seed,
                planner_keys=planner_keys,
                out_dir=out_dir,
                formats=formats,
                dpi=dpi,
            )
        print(f"Error: {exc}", file=sys.stderr)
        print("Use --allow-missing to continue with available planners.", file=sys.stderr)
        return 2

    output_base = out_dir / f"{scenario_id}_seed{seed}"

    try:
        saved = build_overlay_figure(
            trajectory_rows,
            output_base=output_base,
            formats=formats,
            dpi=dpi,
        )
    except (MultiPlannerOverlayError, ValueError) as exc:
        print(f"Error rendering overlay: {exc}", file=sys.stderr)
        return 1

    print(f"Overlay rendered ({scenario_id}, seed {seed}): {len(saved)} file(s)")
    for path in saved:
        print(f"  {path}")

    return 0


def _handle_missing_planners(
    *,
    episodes: list[dict],
    scenario_id: str,
    seed: int,
    planner_keys: list[str],
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> int:
    """Handle missing planners when --allow-missing is set.

    Returns:
        Exit code.
    """
    from robot_sf.benchmark.multi_planner_overlay import _find_trajectory_row

    available: list[str] = []
    for pk in planner_keys:
        if _find_trajectory_row(episodes, scenario_id, seed, pk) is not None:
            available.append(pk)

    if not available:
        print("Error: no trajectory data available", file=sys.stderr)
        return 2

    try:
        trajectory_rows = select_episodes_for_overlay(
            episodes,
            scenario_id=scenario_id,
            seed=seed,
            planner_keys=available,
        )
    except MultiPlannerOverlayError:
        print("Error: no trajectories available after filtering", file=sys.stderr)
        return 2

    output_base = out_dir / f"{scenario_id}_seed{seed}"

    try:
        saved = build_overlay_figure(
            trajectory_rows,
            output_base=output_base,
            formats=formats,
            dpi=dpi,
        )
    except (MultiPlannerOverlayError, ValueError) as exc:
        print(f"Error rendering overlay: {exc}", file=sys.stderr)
        return 1

    print(f"Overlay rendered ({scenario_id}, seed {seed}): {len(saved)} file(s)")
    for path in saved:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
