#!/usr/bin/env python3
"""Run and check the force-coupled potential-field paired diagnostic comparator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.force_coupled_comparator import (
    CANONICAL_CONFIG_PATH,
    run_force_coupled_comparator,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CANONICAL_CONFIG_PATH),
        help="Path to force-coupled potential field configuration YAML.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write receipt JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print receipt JSON to stdout.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke check and assert status is ok.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run comparator and report results.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    args = parse_args(argv)
    receipt = run_force_coupled_comparator(
        config_path=args.config if args.config.exists() else None,
        repo_root=args.repo_root,
    )

    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    if args.json or not args.smoke:
        print(text)

    if args.smoke:
        status = receipt.get("status")
        digest = receipt.get("receipt_digest")
        runs = len(receipt.get("results", []))
        if status != "ok" or not digest or runs == 0:
            print(
                f"FAIL: force-coupled comparator check failed (status={status}, runs={runs})",
                file=sys.stderr,
            )
            return 1
        print(f"PASS: force-coupled comparator check passed (runs={runs}, digest={digest[:16]}...)")

    return 0 if receipt.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
