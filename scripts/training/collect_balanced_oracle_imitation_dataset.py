#!/usr/bin/env python3
"""CLI for balanced oracle imitation dataset collection and preflight planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.training.balanced_oracle_dataset_collector import BalancedOracleCollector

DEFAULT_PACKET = Path(
    "configs/training/ppo_imitation/oracle_dataset_issue_6127_balanced_launch_packet.yaml"
)
DEFAULT_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for balanced dataset collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PACKET,
        help="Path to launch packet config YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for dataset and manifest output.",
    )
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to candidate registry YAML.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Perform preflight planning only (no simulation) and exit 0.",
    )
    parser.add_argument(
        "--min-usable-transitions",
        type=int,
        default=10000,
        help="Minimum required usable training transitions (default 10000).",
    )
    parser.add_argument(
        "--min-episodes-per-stratum",
        type=int,
        default=10,
        help="Minimum required usable nondegenerate episodes per training stratum (default 10).",
    )
    parser.add_argument(
        "--allow-insufficient-yield",
        action="store_true",
        help="Bypass fail-closed yield threshold checks for testing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result/plan report in JSON format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    collector = BalancedOracleCollector(
        args.config,
        output_root=args.output_root,
        candidate_registry=args.candidate_registry,
        min_usable_transitions=args.min_usable_transitions,
        min_episodes_per_stratum=args.min_episodes_per_stratum,
    )

    if args.preflight:
        plan = collector.build_preflight_plan()
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            print(
                f"Preflight completed successfully: plan written to {plan['manifest_destination']}"
            )
        return 0

    manifest = collector.collect_dataset(
        allow_insufficient_yield=args.allow_insufficient_yield,
        cli_command=" ".join(sys.argv),
    )

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Dataset collection complete. Manifest written to {manifest['manifest_path']}")
        print(f"NPZ SHA-256: {manifest['exact_public_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
