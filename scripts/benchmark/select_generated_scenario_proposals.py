#!/usr/bin/env python3
"""Prioritize generated scenario hypotheses for manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.scenario_generation.adaptive_selector import run_adaptive_selection

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically rank generated scenario hypotheses with configurable, "
            "archive-adaptive scores. Output remains review-only and is not benchmark evidence."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Tracked selector YAML config")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the config output path; the target must not already exist",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run adaptive proposal selection and print a compact non-claim summary."""

    args = _parse_args(argv)
    result = run_adaptive_selection(args.config, output_path=args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "claim_boundary": result["claim_boundary"],
                "source_entry_count": result["source_archive"]["entry_count"],
                "selected_scenario_ids": [row["scenario_id"] for row in result["selected"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
