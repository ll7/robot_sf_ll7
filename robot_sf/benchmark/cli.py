from __future__ import annotations

import argparse
import json
import sys
from typing import List

from robot_sf.benchmark.baseline_stats import run_and_compute_baseline
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.summary import summarize_to_plots

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
    p.add_argument(
        "--algo",
        default="simple_policy",
        help="Algorithm to use for robot policy (simple_policy, baseline_sf, etc.)"
    )
    p.add_argument(
        "--algo-config",
        help="Path to algorithm configuration YAML file"
    )
    p.set_defaults(cmd="baseline")


def _add_run_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "run",
        help="Run a batch of episodes from a scenario matrix and write JSONL",
    )
    p.add_argument("--matrix", required=True, help="Path to scenario matrix YAML")
    p.add_argument("--out", required=True, help="Path to write episode JSONL")
    p.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, help="Schema path for validation")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=None, help="Override repeats in matrix")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--record-forces", action="store_true", default=False)
    p.add_argument("--append", action="store_true", default=False, help="Append to existing JSONL")
    p.add_argument(
        "--algo",
        default="simple_policy",
        help="Algorithm to use for robot policy (simple_policy, baseline_sf, etc.)"
    )
    p.add_argument(
        "--algo-config",
        help="Path to algorithm configuration YAML file"
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop on first failure instead of collecting errors",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-episode progress output",
    )
    p.set_defaults(cmd="run")


def _add_list_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "list-algorithms",
        help="List available baseline algorithms",
    )
    p.set_defaults(cmd="list-algorithms")


def _add_summary_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "summary",
        help="Generate simple histograms (min_distance, avg_speed) from episode JSONL",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    p.add_argument("--out-dir", required=True, help="Output directory for PNGs")
    p.set_defaults(cmd="summary")


def cli_main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robot_sf_bench", description="Social Navigation Benchmark CLI"
    )
    subparsers = parser.add_subparsers(dest="cmd")
    _add_baseline_subparser(subparsers)
    _add_run_subparser(subparsers)
    _add_summary_subparser(subparsers)
    _add_list_subparser(subparsers)

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
                algo=args.algo,
                algo_config_path=args.algo_config,
            )
            # Print brief summary to stdout for convenience
            print(json.dumps({"out": args.out, "keys": sorted(stats.keys())}, indent=2))
            return 0
        except Exception as e:  # pragma: no cover - error path
            print(f"Error: {e}", file=sys.stderr)
            return 2
    if args.cmd == "list-algorithms":
        try:
            # Show built-in simple policy
            algorithms = ["simple_policy"]
            
            # Try to load baseline algorithms
            try:
                from robot_sf.baselines import list_baselines
                baseline_algos = list_baselines()
                algorithms.extend(baseline_algos)
            except ImportError:
                print("Warning: Could not load baseline algorithms", file=sys.stderr)
            
            print("Available algorithms:")
            for algo in algorithms:
                print(f"  - {algo}")
            
            return 0
        except Exception as e:  # pragma: no cover - error path
            print(f"Error: {e}", file=sys.stderr)
            return 2
    if args.cmd == "run":
        try:

            def _progress(i, total, sc, seed, ok, err):
                if args.quiet:
                    return
                status = "ok" if ok else "FAIL"
                sid = sc.get("id", "unknown")
                msg = f"[{i}/{total}] {sid} seed={seed}: {status}"
                if err:
                    msg += f" ({err})"
                print(msg)

            summary = run_batch(
                scenarios_or_path=args.matrix,
                out_path=args.out,
                schema_path=args.schema,
                base_seed=args.base_seed,
                repeats_override=args.repeats,
                horizon=args.horizon,
                dt=args.dt,
                record_forces=args.record_forces,
                append=args.append,
                fail_fast=args.fail_fast,
                progress_cb=_progress,
                algo=args.algo,
                algo_config_path=args.algo_config,
            )
            print(json.dumps(summary, indent=2))
            return 0
        except Exception as e:  # pragma: no cover - error path
            print(f"Error: {e}", file=sys.stderr)
            return 2
    if args.cmd == "summary":
        try:
            outs = summarize_to_plots(args.in_path, args.out_dir)
            print(json.dumps({"wrote": outs}, indent=2))
            return 0
        except Exception as e:  # pragma: no cover - error path
            print(f"Error: {e}", file=sys.stderr)
            return 2
    parser.print_help()
    return 1


def main() -> None:  # pragma: no cover - thin wrapper
    raise SystemExit(cli_main())


__all__ = ["cli_main", "main"]
