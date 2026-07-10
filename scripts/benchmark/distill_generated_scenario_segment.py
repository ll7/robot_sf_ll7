"""Distill one JSON episode trace into a review-pending generated catalog entry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.scenario_generation import extract_critical_segment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-json", type=Path, required=True, help="Input episode trace JSON file."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output catalog-entry JSON file."
    )
    parser.add_argument("--pre-margin-s", type=float, default=None)
    parser.add_argument("--post-margin-s", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Write one validated catalog entry and no benchmark or replay result."""

    args = _parse_args()
    episode: Any = json.loads(args.episode_json.read_text(encoding="utf-8"))
    entry = extract_critical_segment(
        episode,
        pre_margin_s=args.pre_margin_s,
        post_margin_s=args.post_margin_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
