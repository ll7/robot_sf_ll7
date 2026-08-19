#!/usr/bin/env python3
"""CLI for balanced oracle imitation dataset collection and preflight planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.training.balanced_oracle_dataset_collector import (
    BalancedOracleCollector,
    check_yield_status,
)

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
        help=(
            "Write a diagnostic, non-promotable manifest when yield gates fail. "
            "The validator will reject this artifact."
        ),
    )
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result/plan report in JSON format.",
    )
    parser.add_argument(
        "--yield-check",
        action="store_true",
        help=(
            "Perform a deterministic, side-effect-free yield status check on an "
            "existing manifest and exit. Returns exactly one of: eligible_complete, "
            "blocked_scientific_yield, blocked_integrity_or_lineage, or "
            "inconclusive_missing_input."
        ),
    )
    parser.add_argument(
        "--exhausted-attempts-file",
        type=Path,
        help=(
            "JSON file containing prior exhausted-attempt records with packet fingerprints. "
            "An unchanged packet is rejected before preflight or collection."
        ),
    )
    return parser


def _load_exhausted_attempts(path: Path | None) -> list[dict[str, object]] | None:
    """Load the bounded exhausted-attempt fingerprint ledger."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    attempts = payload.get("attempts") if isinstance(payload, dict) else payload
    if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
        raise ValueError(
            "exhausted attempts file must contain a list or an {attempts: [...]} mapping"
        )
    return attempts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    exhausted_attempts = _load_exhausted_attempts(args.exhausted_attempts_file)

    if args.yield_check:
        config_path = args.config if args.config.exists() else None
        result = check_yield_status(args.output_root, config_path=config_path)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"Yield check status: {result['check_status']}")
            if "reason" in result:
                print(f"Reason: {result['reason']}")
        return 0

    collector = BalancedOracleCollector(
        args.config,
        output_root=args.output_root,
        candidate_registry=args.candidate_registry,
        min_usable_transitions=args.min_usable_transitions,
        min_episodes_per_stratum=args.min_episodes_per_stratum,
    )

    if args.preflight:
        plan = collector.build_preflight_plan(exhausted_attempts=exhausted_attempts)
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            print(
                "Preflight completed successfully: plan written to "
                f"{args.output_root / 'balanced_oracle_collection_plan.json'}"
            )
        return 0

    manifest = collector.collect_dataset(
        exhausted_attempts=exhausted_attempts,
        allow_insufficient_yield=args.allow_insufficient_yield,
        cli_command=" ".join(sys.argv),
        horizon=args.horizon,
        dt=args.dt,
    )

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Dataset collection complete. Manifest written to {manifest['manifest_path']}")
        print(f"NPZ SHA-256: {manifest['dataset_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
