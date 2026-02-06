#!/usr/bin/env python3
"""Split scenarios into train/holdout groups based on the ``split`` field.

Usage:
  uv run python scripts/tools/split_scenarios.py --scenario configs/scenarios/classic_interactions.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from robot_sf.common.logging import configure_logging
from robot_sf.training.scenario_loader import load_scenarios
from robot_sf.training.scenario_split import split_scenarios


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Scenario YAML file to split.",
    )
    parser.add_argument(
        "--default-split",
        type=str,
        default="train",
        help="Default split name when missing (train or holdout).",
    )
    return parser


def main() -> int:
    """Run the scenario split checker CLI."""
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    scenarios = list(load_scenarios(args.scenario))
    splits = split_scenarios(scenarios, default_split=args.default_split)

    for key, items in splits.items():
        logger.info("Split '{split}': {count} scenario(s)", split=key, count=len(items))

    non_empty = [key for key, items in splits.items() if items]
    if len(non_empty) < 2:
        logger.error("Scenario file does not contain both train and holdout splits.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
