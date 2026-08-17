"""Build a context-separated, analysis-only Flint atlas manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.flint_chart import (
    FlintChartContractError,
    build_atlas_manifest,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the atlas-manifest argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface",
        type=Path,
        action="append",
        required=True,
        help="candidate surface JSON; repeat for release/replay entries",
    )
    parser.add_argument("--output", type=Path, required=True, help="atlas manifest output")
    parser.add_argument("--atlas-id", default="flint_chart_atlas_v1")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and summarize without writing the manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build an atlas manifest or return a fail-closed diagnostic."""
    args = build_parser().parse_args(argv)
    try:
        manifest = build_atlas_manifest(args.surface, atlas_id=args.atlas_id)
        if not args.check_only:
            write_json(args.output, manifest)
        print(
            json.dumps(
                {
                    "status": "validated_not_promoted",
                    "atlas_id": manifest["atlas_id"],
                    "surface_count": manifest["coverage"]["surface_count"],
                    "contexts": manifest["contexts"],
                    "output": None if args.check_only else str(args.output),
                    "claim_boundary": manifest["claim_boundary"],
                },
                sort_keys=True,
            )
        )
        return 0
    except FlintChartContractError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
