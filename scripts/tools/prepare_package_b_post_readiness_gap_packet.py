#!/usr/bin/env python3
"""Emit the dry-run Package B post-readiness evidence-gap packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.adversarial_package_b_gap_packet import (
    DEFAULT_MANIFEST,
    build_package_b_gap_packet,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Package B gap-packet CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Package B readiness manifest to consume.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving relative manifest paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the dry-run packet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the gap packet and return non-zero while readiness blockers remain."""
    args = parse_args(argv)
    packet = build_package_b_gap_packet(args.manifest, repo_root=args.repo_root)
    payload = packet.to_payload()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if packet.readiness_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
