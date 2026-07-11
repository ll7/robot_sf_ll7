#!/usr/bin/env python3
"""Run the review-pending data-driven scenario-generation MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.scenario_generation.pipeline import run_generation_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample CPU map episodes, distill critical windows, and write a generated-only "
            "scenario hypothesis catalog. This does not update benchmark release matrices."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Tracked MVP YAML config")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the config output directory; it must be empty",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured pipeline and print its compact artifact summary."""

    args = _parse_args(argv)
    manifest = run_generation_pipeline(args.config, output_root=args.output_root)
    print(
        json.dumps(
            {
                "status": "complete",
                "claim_boundary": manifest["claim_boundary"],
                "episode_count": manifest["episode_count"],
                "catalog": manifest["catalog"],
                "artifacts": manifest["artifacts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
