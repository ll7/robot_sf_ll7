#!/usr/bin/env python3
"""Build the issue #4183 hybrid_global_rl diagnostic evidence packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.hybrid_global_rl_diagnostic import (
    build_diagnostic_report,
    load_jsonl_records,
    preflight_configs,
)

DEFAULT_ROUTE_CONFIG = Path("configs/benchmarks/issue_4183_hybrid_global_rl_route_conditioned.yaml")
DEFAULT_BASELINE_CONFIG = Path("configs/benchmarks/issue_4183_learned_local_unconditioned.yaml")
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_4183_hybrid_global_rl_diagnostic")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE_CONFIG)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--route-episodes", type=Path)
    parser.add_argument("--baseline-episodes", type=Path)
    parser.add_argument(
        "--run-failures-json",
        type=Path,
        help="Optional JSON list of fail-closed runner failures to record in summary.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paired config/checkpoint prerequisites without requiring episode JSONL.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic packet builder."""

    args = parse_args(argv)
    if args.preflight_only:
        print(
            json.dumps(
                preflight_configs(
                    args.route_config,
                    args.baseline_config,
                    repo_root=args.repo_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.route_episodes is None or args.baseline_episodes is None:
        raise SystemExit(
            "--route-episodes and --baseline-episodes are required without --preflight-only"
        )
    summary = build_diagnostic_report(
        route_records=load_jsonl_records(args.route_episodes),
        baseline_records=load_jsonl_records(args.baseline_episodes),
        route_config_path=args.route_config,
        baseline_config_path=args.baseline_config,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        run_failures=(
            json.loads(args.run_failures_json.read_text(encoding="utf-8"))
            if args.run_failures_json
            else None
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
