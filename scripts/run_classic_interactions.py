"""Run the classic interaction scenario matrix and write JSONL episodes.

Usage (programmatic):
    uv run python scripts/run_classic_interactions.py

This script is a thin wrapper around `robot_sf.benchmark.runner.run_batch` that
loads the scenario matrix at `configs/scenarios/classic_interactions.yaml` and
writes a JSONL file containing one record per (scenario, seed) expansion.

It respects resume semantics by default (won't duplicate existing episodes)
unless `--no-resume` is passed. By default uses the baseline `simple_policy`.

Outputs:
    results/classic_interactions/episodes.jsonl
    results/classic_interactions/episodes.jsonl.manifest.json (resume manifest)

Optional flags:
    --algo <name>                 Baseline algorithm (default: simple_policy)
    --algo-config <path>          YAML config for the algorithm
    --output <path>               Override output JSONL path
    --workers <int>               Parallel workers (default: 1)
    --horizon <int>               Episode horizon (default: 100)
    --dt <float>                  Simulation timestep (default: 0.1)
    --no-resume                   Disable resume (recompute all)
    --fail-fast                   Stop on first failure
    --record-forces/--no-record-forces  Toggle force recording (default: on)
    --snqi-weights <path.json>    JSON of weights to recompute SNQI (optional)
    --snqi-baseline <path.json>   JSON of baseline metrics for SNQI normalization

Example:
    uv run python scripts/run_classic_interactions.py --workers 4 --algo social_force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.benchmark.runner import run_batch

SCENARIO_MATRIX = Path("configs/scenarios/classic_interactions.yaml")
SCHEMA_PATH = Path("docs/dev/issues/social-navigation-benchmark/episode_schema.json")
DEFAULT_OUT = Path("results/classic_interactions/episodes.jsonl")


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():  # pragma: no cover simple guard
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classic interaction scenarios")
    parser.add_argument("--algo", default="simple_policy", help="Baseline algorithm name")
    parser.add_argument(
        "--algo-config",
        dest="algo_config",
        default=None,
        help="Algo YAML config path",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=str(DEFAULT_OUT),
        help="Output JSONL path",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--horizon", type=int, default=100, help="Episode horizon")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation timestep")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (re-run all episodes)",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Abort on first episode failure")
    parser.add_argument("--record-forces", dest="record_forces", action="store_true", default=True)
    parser.add_argument(
        "--no-record-forces",
        dest="record_forces",
        action="store_false",
        help="Do not record social forces",
    )
    parser.add_argument(
        "--snqi-weights",
        dest="snqi_weights",
        default=None,
        help="Path to SNQI weights JSON",
    )
    parser.add_argument(
        "--snqi-baseline",
        dest="snqi_baseline",
        default=None,
        help="Path to SNQI baseline JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    snqi_weights = _load_json(args.snqi_weights)
    snqi_baseline = _load_json(args.snqi_baseline)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running classic interaction scenarios: {}", SCENARIO_MATRIX)

    summary = run_batch(
        SCENARIO_MATRIX,
        out_path=out_path,
        schema_path=SCHEMA_PATH,
        algo=args.algo,
        algo_config_path=args.algo_config,
        workers=args.workers,
        horizon=args.horizon,
        dt=args.dt,
        record_forces=args.record_forces,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )

    logger.info("Completed classic interactions run: {}", summary)

    # Write summary sidecar
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary: {}", summary_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
