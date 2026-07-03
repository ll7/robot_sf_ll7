#!/usr/bin/env python3
"""Build the issue #4165 paired proxemic-layer ablation report."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.proxemic_ablation_report import (
    build_proxemic_ablation_report,
    load_records,
    write_report_artifacts,
)

DEFAULT_SMOKE_CONFIG = Path("configs/benchmarks/issue_4165_proxemic_costmap_smoke.yaml")
DEFAULT_PROXEMIC_CONFIG = Path("configs/planners/proxemic_costmap_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_4165_proxemic_ablation_report")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-episodes", type=Path, required=True)
    parser.add_argument("--proxemic-episodes", type=Path, required=True)
    parser.add_argument("--smoke-config", type=Path, default=DEFAULT_SMOKE_CONFIG)
    parser.add_argument("--proxemic-config", type=Path, default=DEFAULT_PROXEMIC_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the report builder."""

    args = parse_args(argv)
    report = build_proxemic_ablation_report(
        baseline_records=load_records(args.baseline_episodes),
        proxemic_records=load_records(args.proxemic_episodes),
        smoke_config_path=args.smoke_config,
        proxemic_config_path=args.proxemic_config,
        repo_root=args.repo_root,
    )
    write_report_artifacts(report, args.output_dir)
    print(args.output_dir / "summary.json")
    return 1 if report["report_status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
