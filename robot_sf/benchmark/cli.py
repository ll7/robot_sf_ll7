from __future__ import annotations

import argparse
import json
import sys
from typing import List

from robot_sf.benchmark.baseline_stats import run_and_compute_baseline

DEFAULT_SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _add_baseline_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "baseline",
        help="Run batch and compute baseline med/p95 stats for SNQI",
    )
    p.add_argument(
        "--matrix",
        required=True,
        help="Path to scenario matrix YAML (or JSONL path will be ignored if provided)",
    )
    p.add_argument("--out", required=True, help="Path to write baseline stats JSON")
    p.add_argument(
        "--jsonl",
        default=None,
        help="Optional path to write intermediate episode JSONL (default results/baseline_episodes.jsonl)",
    )
    p.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, help="Schema path for validation")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--record-forces", action="store_true", default=False)
    p.set_defaults(cmd="baseline")


def cli_main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robot_sf_bench", description="Social Navigation Benchmark CLI"
    )
    subparsers = parser.add_subparsers(dest="cmd")
    _add_baseline_subparser(subparsers)

    args = parser.parse_args(argv)
    if args.cmd == "baseline":
        try:
            stats = run_and_compute_baseline(
                args.matrix,
                out_json=args.out,
                out_jsonl=args.jsonl,
                schema_path=args.schema,
                base_seed=args.base_seed,
                repeats_override=args.repeats,
                horizon=args.horizon,
                dt=args.dt,
                record_forces=args.record_forces,
            )
            # Print brief summary to stdout for convenience
            print(json.dumps({"out": args.out, "keys": sorted(stats.keys())}, indent=2))
            return 0
        except Exception as e:  # pragma: no cover - error path
            print(f"Error: {e}", file=sys.stderr)
            return 2
    parser.print_help()
    return 1


def main() -> None:  # pragma: no cover - thin wrapper
    raise SystemExit(cli_main())


__all__ = ["cli_main", "main"]
