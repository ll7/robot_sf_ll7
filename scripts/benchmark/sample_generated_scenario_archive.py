#!/usr/bin/env python3
"""Select criticality-biased hypotheses from a generated scenario archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.scenario_generation.archive_sampler import run_archive_sampling

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically sample an existing generated-scenario archive with higher "
            "probability for lower-clearance records. Output remains review-pending and is "
            "not benchmark evidence."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Tracked sampler YAML config")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the config output path; the target must not already exist",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the archive sampler and print a compact non-claim summary."""

    args = _parse_args(argv)
    result = run_archive_sampling(args.config, output_path=args.output)
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
