#!/usr/bin/env python3
"""CLI for generating replay-derived figure artifacts from campaign episode rows.

This tool maps a persisted campaign episode row to replay-derived figure
artifacts: stills, filmstrip, and trajectory plots, with deterministic replay
checks and provenance sidecars.

Claim boundary: figure artifact generation only. Does not run campaigns,
reinterpret metrics, or promote replay outputs as new benchmark evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from robot_sf.benchmark.episode_replay_figure import (
    EpisodeRow,
    load_episode_row,
    replay_episode_and_generate_figures,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate replay-derived figure artifacts from campaign episode rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/replay_episode_figure.py \\
    --episodes output/campaign/episodes.jsonl \\
    --episode-id ep_001 \\
    --outputs still,filmstrip,trajectory \\
    --out-dir output/replay_episode_figure/ep_001

  uv run python scripts/replay_episode_figure.py \\
    --episodes episodes.jsonl \\
    --episode-id ep_002 \\
    --outputs trajectory \\
    --out-dir output/trajectory_only \\
    --tolerance-m 0.05

  uv run python scripts/replay_episode_figure.py \\
    --episodes episodes.jsonl \\
    --episode-id ep_003 \\
    --outputs still,filmstrip \\
    --out-dir output/stills \\
    --frame-steps 0,10,20,30 \\
    --format pdf
        """,
    )

    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="Path to episodes JSONL file containing episode rows",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        required=True,
        help="Episode ID to generate figures for",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        required=True,
        help="Comma-separated list of output types: still,filmstrip,trajectory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated artifacts",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=None,
        help="Campaign root path for resolving manifest/run metadata",
    )
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=None,
        help="Path to scenario matrix file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        dest="config_hash",
        help="Campaign config hash",
    )
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=0.1,
        help="Determinism check tolerance in meters (default: 0.1)",
    )
    parser.add_argument(
        "--frame-steps",
        type=str,
        default=None,
        help="Comma-separated step indices for filmstrip (e.g., 0,10,20,30)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for figures (default: png)",
    )
    parser.add_argument(
        "--no-determinism-check",
        action="store_true",
        help="Skip determinism check (diagnostic escape hatch only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="{time:HH:mm:ss} | {level} | {message}")

    try:
        output_types = [o.strip() for o in args.outputs.split(",")]
        valid_outputs = {"still", "filmstrip", "trajectory"}
        for ot in output_types:
            if ot not in valid_outputs:
                logger.error(f"Invalid output type '{ot}'; must be one of {valid_outputs}")
                return 1

        if not args.episodes.exists():
            logger.error(f"Episodes file not found: {args.episodes}")
            return 1

        logger.info(f"Loading episode '{args.episode_id}' from {args.episodes}")
        episode_row = load_episode_row(args.episodes, args.episode_id)
        logger.info(f"  scenario_id: {episode_row.scenario_id}")
        logger.info(f"  seed: {episode_row.seed}")
        logger.info(f"  planner: {episode_row.planner or episode_row.planner_key or 'unknown'}")

        frame_steps = None
        if args.frame_steps:
            try:
                frame_steps = [int(s.strip()) for s in args.frame_steps.split(",")]
            except ValueError as e:
                logger.error(f"Invalid frame-steps format: {e}")
                return 1

        logger.info(f"Generating outputs: {output_types}")
        logger.info(f"Output directory: {args.out_dir}")
        logger.info(f"Format: {args.format}")
        logger.info(f"Determinism tolerance: {args.tolerance_m}m")

        result = replay_episode_and_generate_figures(
            episode_row=episode_row,
            outputs=output_types,
            out_dir=args.out_dir,
            tolerance_m=args.tolerance_m,
            frame_steps=frame_steps,
            fmt=args.format,
            episodes_jsonl_path=args.episodes,
            campaign_root=args.campaign_root,
            scenario_matrix_path=args.scenario_matrix,
            config_hash=args.config_hash,
            no_determinism_check=args.no_determinism_check,
        )

        logger.info(f"Determinism check: {result['determinism_check_status']}")
        logger.info(f"Artifacts generated: {result['artifacts_generated']}")
        for path in result["artifact_paths"]:
            logger.info(f"  {path}")
        logger.info(f"Provenance sidecar: {result['provenance_sidecar']}")
        logger.info(f"Caption fragment: {result['caption_fragment']}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Execution failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
