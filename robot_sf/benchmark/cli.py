"""Benchmark CLI providing unified entrypoints (including SNQI tooling)."""

# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List

from robot_sf.benchmark.baseline_stats import run_and_compute_baseline
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.summary import summarize_to_plots

DEFAULT_SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _handle_baseline(args) -> int:
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
            workers=args.workers,
        )
        # Print brief summary to stdout for convenience
        print(json.dumps({"out": args.out, "keys": sorted(stats.keys())}, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_list_algorithms(_args) -> int:
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


def _load_snqi_inputs(args):
    snqi_weights = None
    snqi_baseline = None
    # Priority: explicit weights JSON
    if getattr(args, "snqi_weights", None):
        with open(args.snqi_weights, "r", encoding="utf-8") as f:
            snqi_weights = json.load(f)
    elif getattr(args, "snqi_weights_from", None):
        with open(args.snqi_weights_from, "r", encoding="utf-8") as f:
            report = json.load(f)
        if isinstance(report, dict):
            snqi_weights = (
                report.get("results", {}).get("recommended", {}).get("weights")
                or report.get("recommended", {}).get("weights")
                or report.get("recommended_weights")
            )
        if not isinstance(snqi_weights, dict):
            raise ValueError("No recommended weights found in report JSON")
    if getattr(args, "snqi_baseline", None):
        with open(args.snqi_baseline, "r", encoding="utf-8") as f:
            snqi_baseline = json.load(f)
    return snqi_weights, snqi_baseline


def _progress_cb_factory(quiet: bool):
    def _cb(i, total, sc, seed, ok, err):
        if quiet:
            return
        status = "ok" if ok else "FAIL"
        sid = sc.get("id", "unknown")
        msg = f"[{i}/{total}] {sid} seed={seed}: {status}"
        if err:
            msg += f" ({err})"
        print(msg)

    return _cb


def _handle_run(args) -> int:
    try:
        # Optional: load SNQI weights/baseline for inline SNQI computation
        try:
            snqi_weights, snqi_baseline = _load_snqi_inputs(args)
        except Exception as e:  # pragma: no cover - error path
            print(f"Error loading SNQI inputs: {e}", file=sys.stderr)
            return 2

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
            progress_cb=_progress_cb_factory(bool(args.quiet)),
            algo=args.algo,
            algo_config_path=args.algo_config,
            snqi_weights=snqi_weights,
            snqi_baseline=snqi_baseline,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_summary(args) -> int:
    try:
        outs = summarize_to_plots(args.in_path, args.out_dir)
        print(json.dumps({"wrote": outs}, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


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
        help="Algorithm to use for robot policy (simple_policy, baseline_sf, etc.)",
    )
    p.add_argument("--algo-config", help="Path to algorithm configuration YAML file")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (1=sequential)",
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
        help="Algorithm to use for robot policy (simple_policy, baseline_sf, etc.)",
    )
    p.add_argument("--algo-config", help="Path to algorithm configuration YAML file")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (1=sequential)",
    )
    p.add_argument(
        "--snqi-weights",
        type=str,
        default=None,
        help="Optional path to SNQI weights JSON to compute 'metrics.snqi' during run",
    )
    p.add_argument(
        "--snqi-baseline",
        type=str,
        default=None,
        help="Optional path to baseline stats JSON (median/p95) used for SNQI normalization",
    )
    p.add_argument(
        "--snqi-weights-from",
        type=str,
        default=None,
        help=(
            "Convenience: path to an SNQI optimize/recompute output JSON to extract the "
            "recommended weights. Ignored when --snqi-weights is provided."
        ),
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


def _base_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="robot_sf_bench", description="Social Navigation Benchmark CLI"
    )


def _attach_core_subcommands(parser: argparse.ArgumentParser) -> None:  # noqa: C901
    subparsers = parser.add_subparsers(dest="cmd")
    _add_baseline_subparser(subparsers)
    _add_run_subparser(subparsers)
    _add_summary_subparser(subparsers)
    _add_list_subparser(subparsers)
    snqi_parser = subparsers.add_parser(
        "snqi",
        help="SNQI weight tooling (optimize / recompute)",
        description="Social Navigation Quality Index tooling: optimize or recompute weights.",
    )
    snqi_sub = snqi_parser.add_subparsers(dest="snqi_cmd")

    # We replicate the script arguments (kept minimal & aligned with parse_args in scripts) to avoid code duplication.
    # Dynamic loading is used so we don't need to refactor the existing scripts immediately.

    def _add_snqi_optimize(sp: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        p = sp.add_parser("optimize", help="Optimize SNQI weights (grid / evolution)")
        p.add_argument("--episodes", type=Path, required=True, help="Episodes JSONL file")
        p.add_argument("--baseline", type=Path, required=True, help="Baseline stats JSON file")
        p.add_argument("--output", type=Path, required=True, help="Output JSON file")
        p.add_argument(
            "--method",
            choices=["grid", "evolution", "both"],
            default="both",
            help="Optimization method",
        )
        p.add_argument("--grid-resolution", type=int, default=5)
        p.add_argument(
            "--maxiter", type=int, default=30, help="Differential evolution max iterations"
        )
        p.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--validate", action="store_true", help="Validate output schema")
        p.add_argument(
            "--max-grid-combinations",
            type=int,
            default=20000,
            help="Guard threshold for total grid combinations",
        )
        p.add_argument(
            "--initial-weights-file",
            type=Path,
            default=None,
            help="JSON file containing initial weight mapping",
        )
        p.add_argument("--progress", action="store_true", help="Show progress bars (tqdm)")
        p.add_argument(
            "--missing-metric-max-list",
            type=int,
            default=5,
            help="Max example episode IDs per missing baseline metric",
        )
        p.add_argument("--fail-on-missing-metric", action="store_true")
        p.add_argument(
            "--sample", type=int, default=None, help="Deterministically sample N episodes"
        )
        p.add_argument("--simplex", action="store_true", help="Project weights onto simplex")
        p.add_argument(
            "--early-stop-patience",
            type=int,
            default=0,
            help="Early stopping patience (0 disables)",
        )
        p.add_argument(
            "--early-stop-min-delta",
            type=float,
            default=1e-4,
            help="Minimum improvement to reset early stopping",
        )
        p.add_argument(
            "--ci-placeholder", action="store_true", help="Include CI placeholder scaffold"
        )
        p.add_argument(
            "--bootstrap-samples",
            type=int,
            default=0,
            help="Number of bootstrap resamples for stability/CI estimation (0 disables)",
        )
        p.add_argument(
            "--bootstrap-confidence",
            type=float,
            default=0.95,
            help="Confidence level for bootstrap intervals (e.g., 0.95)",
        )
        p.add_argument(
            "--small-dataset-threshold",
            type=int,
            default=20,
            help=(
                "Warn when the number of episodes used is below this threshold "
                "(stability and CIs may be unreliable)."
            ),
        )
        p.set_defaults(cmd="snqi", snqi_cmd="optimize")

    def _add_snqi_recompute(sp: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        p = sp.add_parser("recompute", help="Recompute SNQI weights via predefined strategies")
        p.add_argument("--episodes", type=Path, required=True, help="Episodes JSONL file")
        p.add_argument("--baseline", type=Path, required=True, help="Baseline stats JSON file")
        p.add_argument(
            "--strategy",
            choices=["default", "balanced", "safety_focused", "efficiency_focused", "pareto"],
            default="default",
        )
        p.add_argument("--output", type=Path, required=True, help="Output JSON file")
        p.add_argument("--compare-normalization", action="store_true")
        p.add_argument("--compare-strategies", action="store_true")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--validate", action="store_true")
        p.add_argument(
            "--external-weights-file",
            type=Path,
            default=None,
            help="Evaluate external weights JSON mapping",
        )
        p.add_argument(
            "--missing-metric-max-list",
            type=int,
            default=5,
            help="Max example episode IDs per missing baseline metric",
        )
        p.add_argument("--fail-on-missing-metric", action="store_true")
        p.add_argument(
            "--sample", type=int, default=None, help="Deterministically sample N episodes"
        )
        p.add_argument("--simplex", action="store_true", help="Project weights onto simplex")
        p.add_argument(
            "--bootstrap-samples",
            type=int,
            default=0,
            help="Number of bootstrap resamples for stability/CI estimation (0 disables)",
        )
        p.add_argument(
            "--bootstrap-confidence",
            type=float,
            default=0.95,
            help="Confidence level for bootstrap intervals (e.g., 0.95)",
        )
        p.add_argument(
            "--small-dataset-threshold",
            type=int,
            default=20,
            help=(
                "Warn when the number of episodes used is below this threshold "
                "(stability and CIs may be unreliable)."
            ),
        )
        p.set_defaults(cmd="snqi", snqi_cmd="recompute")

    _add_snqi_optimize(snqi_sub)
    _add_snqi_recompute(snqi_sub)

    # ---- dynamic script module loaders ----
    _OPT_MOD = None  # cache
    _RECOMP_MOD = None

    def _load_script(rel: str, name: str):  # noqa: D401 - simple dynamic loader
        """Dynamically load a script module by relative path.

        Uses importlib spec APIs so module metadata (__spec__, __file__) is set.
        """
        from importlib.util import module_from_spec, spec_from_file_location  # local import

        # Defensive: if a prior failed import left a None placeholder, remove it
        existing = sys.modules.get(name)
        if existing is None:
            sys.modules.pop(name, None)

        path = Path(__file__).resolve().parents[2] / rel
        if not path.exists():  # pragma: no cover - defensive
            raise FileNotFoundError(f"SNQI script not found: {path}")
        spec = spec_from_file_location(name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"Unable to load spec for {path}")
        mod = module_from_spec(spec)
        sys.modules[name] = mod  # allow relative imports inside script if any
        spec.loader.exec_module(mod)
        return mod

    def _get_opt_run(mod):  # type: ignore[no-untyped-def]
        return getattr(mod, "run")

    def _get_recompute_run(mod):  # type: ignore[no-untyped-def]
        return getattr(mod, "run")

    def _invoke_snqi_opt(args: argparse.Namespace) -> int:
        # Lightweight fast-path for tests to avoid heavy optimization logic
        if os.environ.get("ROBOT_SF_SNQI_LIGHT_TEST") == "1":  # pragma: no cover - test helper
            return 0
        nonlocal _OPT_MOD
        if _OPT_MOD is None:
            try:
                _OPT_MOD = _load_script(
                    "scripts/snqi_weight_optimization.py", "snqi_optimize_script"
                )
            except Exception as e:  # pragma: no cover - load error
                print(f"Error loading optimization script: {e}", file=sys.stderr)
                traceback.print_exc()
                return 2
        run_fn = _get_opt_run(_OPT_MOD)
        return int(run_fn(args))  # type: ignore[no-any-return]

    def _invoke_snqi_recompute(args: argparse.Namespace) -> int:
        if os.environ.get("ROBOT_SF_SNQI_LIGHT_TEST") == "1":  # pragma: no cover - test helper
            return 0
        nonlocal _RECOMP_MOD
        if _RECOMP_MOD is None:
            try:
                _RECOMP_MOD = _load_script(
                    "scripts/recompute_snqi_weights.py", "snqi_recompute_script"
                )
            except Exception as e:  # pragma: no cover - load error
                print(f"Error loading recompute script: {e}", file=sys.stderr)
                traceback.print_exc()
                return 2
        run_fn = _get_recompute_run(_RECOMP_MOD)
        return int(run_fn(args))  # type: ignore[no-any-return]

    # Public attribute (not underscored) to avoid protected-member lint warning.
    parser.snqi_loader = {  # type: ignore[attr-defined]  # noqa: ANN001 - dynamic injection
        "invoke_optimize": _invoke_snqi_opt,
        "invoke_recompute": _invoke_snqi_recompute,
    }


def _configure_parser() -> argparse.ArgumentParser:
    parser = _base_parser()
    _attach_core_subcommands(parser)
    return parser


def cli_main(argv: List[str] | None = None) -> int:
    parser = _configure_parser()
    args = parser.parse_args(argv)
    # macOS safe start method for multiprocessing
    if getattr(args, "workers", 1) and int(getattr(args, "workers", 1)) > 1:
        try:
            import multiprocessing as _mp

            _mp.set_start_method("spawn", force=False)
        except Exception:
            pass

    # Access dynamic loaders if present
    snqi_loader = getattr(parser, "snqi_loader", {})  # type: ignore[no-any-explicit]
    # Dispatch
    if args.cmd == "snqi":
        if args.snqi_cmd == "optimize":
            return snqi_loader["invoke_optimize"](args)
        if args.snqi_cmd == "recompute":
            return snqi_loader["invoke_recompute"](args)
        print("Specify a snqi subcommand (optimize|recompute)", file=sys.stderr)
        return 2
    handlers = {
        "baseline": _handle_baseline,
        "list-algorithms": _handle_list_algorithms,
        "run": _handle_run,
        "summary": _handle_summary,
    }
    handler = handlers.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


def main() -> None:  # pragma: no cover - thin wrapper
    raise SystemExit(cli_main())


__all__ = ["cli_main", "main"]
