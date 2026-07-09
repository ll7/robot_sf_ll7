#!/usr/bin/env python3
"""Run the fail-closed package-B manifest preflight for issue #3079."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.adversarial_package_b_preflight import (
    dump_preflight_payload,
    preflight_package_b_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse package-B preflight CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  uv run python scripts/tools/preflight_adversarial_package_b.py [--manifest <path>]"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/adversarial/issue_3079_package_b_budget_matched.yaml"),
        help="Package-B budget-matched comparison manifest to preflight.",
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
        help="Optional JSON path for the preflight report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run preflight and return non-zero when blockers are present."""
    args = parse_args(argv)
    result = preflight_package_b_manifest(args.manifest, repo_root=args.repo_root)
    dump_preflight_payload(result, args.output)
    print(json.dumps(result.to_payload(), sort_keys=True))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
