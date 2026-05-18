"""Curate compact adversarial failure archives from search manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.adversarial.archive import curate_failure_archive

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="+",
        type=Path,
        help="Adversarial search manifest path(s) to curate.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to write adversarial_failure_archive.v1 JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run archive curation and print the compact summary."""
    args = parse_args(argv)
    archive = curate_failure_archive(args.manifest, output_path=args.out)
    print(json.dumps({"path": args.out.as_posix(), "summary": archive["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
