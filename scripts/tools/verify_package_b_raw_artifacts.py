#!/usr/bin/env python3
"""Verify Package B raw candidate/replay artifact inventory and file digests.

Provides a deterministic verification and retrieval status check for the 4,761-entry
candidate/replay artifact tree (Issue #6131).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.adversarial_package_b_report import (
    verify_package_b_candidate_replay_inventory,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_BUNDLE = Path("docs/context/evidence/issue_5785_package_b_27cell_replication_2026-07-15")


def build_parser() -> argparse.ArgumentParser:
    """Build parser for verify_package_b_raw_artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Path to Package B evidence bundle directory or candidate_replay_SHA256SUMS.txt inventory file.",
    )
    parser.add_argument(
        "--raw-tree-dir",
        type=Path,
        default=None,
        help="Optional directory containing actual raw candidate/replay files to verify on disk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the verification summary payload.",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="Return non-zero exit code if verification fails.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run candidate/replay inventory verification."""
    args = build_parser().parse_args(argv)
    result = verify_package_b_candidate_replay_inventory(
        bundle_dir=args.bundle,
        raw_tree_dir=args.raw_tree_dir,
    )
    payload = result.to_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    if args.fail_closed and not result.is_valid:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
