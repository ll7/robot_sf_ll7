#!/usr/bin/env python3
"""Search a bounded existing planner configuration and write replayable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validation.planner_optimizer import run_planner_optimization


def parse_args() -> argparse.Namespace:
    """Parse the run config and fresh evidence output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/policy_search/planner_optimizer_issue9650.yaml"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Execute one bounded planner optimization and print its artifact receipt."""
    args = parse_args()
    manifest = run_planner_optimization(args.config, args.output_dir)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "run_id": manifest["run_id"],
                "selected": manifest["selected"],
                "heldout_comparison": manifest["heldout_comparison"],
                "manifest": str((args.output_dir / "run_manifest.json").resolve()),
                "report": str((args.output_dir / "report.md").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
