#!/usr/bin/env python3
"""Render representative trajectory panels from simulation trace exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.trajectory_panels import generate_trajectory_panel_bundle


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the trajectory panel CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        action="append",
        default=[],
        help="Path to a simulation_trace_export.v1 JSON file. Repeat for multiple traces.",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Optional manual representative episode CSV override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where panel artifacts and metadata will be written.",
    )
    parser.add_argument(
        "--command",
        default=None,
        help="Generation command to record in the manifest.",
    )
    parser.add_argument(
        "--commit",
        required=True,
        help="Git commit or short commit recorded in the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run representative trajectory panel generation."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.trace and args.selection_csv is None:
        parser.error("at least one --trace or --selection-csv is required")

    command = args.command or "uv run python scripts/tools/render_trajectory_panels.py"
    try:
        bundle = generate_trajectory_panel_bundle(
            list(args.trace),
            output_dir=args.output_dir,
            command=command,
            commit=args.commit,
            override_csv=args.selection_csv,
        )
    except Exception as exc:  # pragma: no cover - CLI defensive path
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote trajectory panel manifest to {bundle.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
