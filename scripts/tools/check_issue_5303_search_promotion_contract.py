#!/usr/bin/env python3
"""Reproduce the issue #5303 search-promotion contract hash and assert the frozen fields.

This is the side-effect-free, check-only invocation of the frozen #5303 config. It
does not execute planners, run a search campaign, replay or confirm anything, submit
Slurm jobs, or read evaluation outcomes. It only loads the frozen contract, the #6139
recertification receipt, and the preregistration manifest, recomputes SHA-256 hashes,
and asserts the frozen design, the power analysis, and the diagnostic declaration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.issue_5303_search_promotion_preflight import (
    DEFAULT_CONTRACT_PATH,
    DEFAULT_MANIFEST_PATH,
    dump_preflight_payload,
    preflight_issue_5303_contract,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the contract-check CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n  uv run python scripts/tools/check_issue_5303_search_promotion_contract.py"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Frozen issue #5303 search-promotion contract YAML.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help=("Override the contract-declared issue #6139 corrected recertification receipt JSON."),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Frozen-contract hash manifest for the preregistration.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving relative paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the check report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the contract check and return non-zero when blockers are present."""
    args = parse_args(argv)
    try:
        result = preflight_issue_5303_contract(
            args.contract,
            receipt_path=args.receipt,
            manifest_path=args.manifest,
            repo_root=args.repo_root,
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    dump_preflight_payload(result, args.output)
    print(json.dumps(result.to_payload(), sort_keys=True))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
