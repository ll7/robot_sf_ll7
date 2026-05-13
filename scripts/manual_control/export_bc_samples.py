"""Export compact behavior-cloning samples from manual-control JSONL recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.manual_control.export import (
    export_demonstration_samples_from_jsonl,
    write_demonstration_samples_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Manual-control session JSONL produced by the manual recorder.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL path for compact BC samples.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the BC sample export command."""
    args = parse_args()
    samples = export_demonstration_samples_from_jsonl(args.input)
    write_demonstration_samples_jsonl(samples, args.output)
    print(f"wrote {len(samples)} manual-control BC samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
