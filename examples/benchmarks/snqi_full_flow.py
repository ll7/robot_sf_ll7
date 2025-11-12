"""End-to-end SNQI figure generation flow.

This script demonstrates a complete reproducible pipeline:
1. (Optional) Run a small scenario matrix to produce episodes JSONL (skip if you already have one)
2. Compute baseline median/p95 stats for normalization
3. (Optional) Run SNQI weight optimization/recompute step (placeholder; user can supply their own)
4. Generate canonical figures with SNQI injected using the orchestrator

Usage:
    uv run python examples/benchmarks/snqi_full_flow.py \
        --episodes results/episodes_sf_long_fix1.jsonl \
        --matrix configs/baselines/example_matrix.yaml \
        --baseline-json results/baseline_stats.json \
        --weights-json examples/snqi_weights_example.json

If --episodes does not exist but --matrix is provided, a batch run will be executed first.
If --weights-json does not exist a default weights file is created.

This keeps everything scriptable and reproducible per the development guide.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_SCHEMA = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd)


def _ensure_weights(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {
        "w_success": 1.0,
        "w_time": 0.5,
        "w_collisions": 3.0,
        "w_near": 1.0,
        "w_comfort": 1.0,
        "w_force_exceed": 0.5,
        "w_jerk": 0.5,
        "w_curvature": 0.5,
    }
    path.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    print(f"[info] Wrote default weights JSON -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Full SNQI pipeline example")
    ap.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="Episodes JSONL path (input or to create)",
    )
    ap.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Scenario matrix YAML for batch run if episodes missing",
    )
    ap.add_argument(
        "--baseline-json",
        type=Path,
        required=True,
        help="Output baseline stats JSON path",
    )
    ap.add_argument("--weights-json", type=Path, required=True, help="SNQI weights JSON path")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--schema", type=Path, default=Path(DEFAULT_SCHEMA))
    args = ap.parse_args()

    episodes = args.episodes

    # Step 1: Generate episodes if needed
    if not episodes.exists():
        if args.matrix is None:
            raise SystemExit("episodes JSONL missing and --matrix not provided to generate it")
        _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "robot_sf.benchmark.cli",
                "run",
                "--matrix",
                str(args.matrix),
                "--out",
                str(episodes),
                "--schema",
                str(args.schema),
                "--base-seed",
                str(args.base_seed),
                "--horizon",
                str(args.horizon),
                "--dt",
                str(args.dt),
            ],
        )

    # Step 2: Compute baseline stats (med/p95)
    _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "robot_sf.benchmark.cli",
            "baseline",
            "--matrix",
            str(args.matrix) if args.matrix else str(args.episodes),
            "--out",
            str(args.baseline_json),
            "--schema",
            str(args.schema),
            "--base-seed",
            str(args.base_seed),
            "--horizon",
            str(args.horizon),
            "--dt",
            str(args.dt),
        ],
    )

    # Step 3: Ensure weights (user could alternatively run optimization and point here)
    _ensure_weights(args.weights_json)

    # Step 4: Generate figures (SNQI injected)
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/generate_figures.py",
            "--episodes",
            str(episodes),
            "--auto-out-dir",
            "--set-latest",
            "--pareto-pdf",
            "--dmetrics",
            "collisions,comfort_exposure,near_misses,snqi",
            "--dists-pdf",
            "--table-metrics",
            "collisions,comfort_exposure,near_misses,snqi",
            "--snqi-weights",
            str(args.weights_json),
            "--snqi-baseline",
            str(args.baseline_json),
        ],
    )

    print("[done] Full SNQI pipeline complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
