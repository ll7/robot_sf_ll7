#!/usr/bin/env python3
"""Write a fail-closed input and claim-gate report for issue #5351."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.hierarchical_paired_release_inputs import (
    INPUTS_READY_ANALYSIS_NOT_RUN,
    evaluate_hierarchical_paired_release_inputs,
    load_hierarchical_paired_release_input_manifest,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Issue #5351 YAML manifest.")
    parser.add_argument("--output", type=Path, required=True, help="JSON report output path.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve durable row paths (default: current directory).",
    )
    return parser.parse_args()


def main() -> int:
    """Evaluate release inputs and write the machine-readable report."""

    args = parse_args()
    manifest = load_hierarchical_paired_release_input_manifest(args.manifest)
    report = evaluate_hierarchical_paired_release_inputs(manifest, repo_root=args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote hierarchical paired release input report {args.output}: {report['status']}")
    return 0 if report["status"] == INPUTS_READY_ANALYSIS_NOT_RUN else 2


if __name__ == "__main__":
    raise SystemExit(main())
