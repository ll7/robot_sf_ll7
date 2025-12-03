"""Generate publication-ready SNQI figures via the orchestrator script.

This example demonstrates how to generate publication-ready figures with SNQI
included, using the orchestrator script. It assumes you have an episodes JSONL
and (optionally) an SNQI weights JSON and baseline stats JSON.

Usage (from repo root):
    uv run python examples/plotting/snqi_figures_example.py \
        --episodes results/episodes_sf_long_fix1.jsonl \
        --weights examples/snqi_weights_example.json \
        --baseline results/baseline_stats.json

Notes:
- If you don't provide --baseline, SNQI still computes but normalization of
  penalties will default to 0.0 where med/p95 are unknown.
- The script uses --auto-out-dir to write under docs/figures/<stem>__<sha>__v<schema>/
  and updates _latest.txt for convenience.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from robot_sf.common.artifact_paths import resolve_artifact_path


def _ensure_file(path: Path, content: str) -> Path:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        content: TODO docstring.

    Returns:
        TODO docstring.
    """
    resolved = resolve_artifact_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not resolved.exists():
        resolved.write_text(content + "\n", encoding="utf-8")
    return resolved


def main() -> int:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    ap = argparse.ArgumentParser(description="SNQI figures example runner")
    ap.add_argument("--episodes", type=Path, required=True, help="Episodes JSONL path")
    ap.add_argument("--weights", type=Path, default=None, help="SNQI weights JSON path")
    ap.add_argument("--baseline", type=Path, default=None, help="Baseline stats JSON path")
    args = ap.parse_args()

    # Provide a tiny default weights JSON if none is supplied
    weights = args.weights
    if weights is None:
        weights = Path("examples/snqi_weights_example.json")
        weights = _ensure_file(
            weights,
            json.dumps(
                {
                    "w_success": 1.0,
                    "w_time": 0.5,
                    "w_collisions": 3.0,
                    "w_near": 1.0,
                    "w_comfort": 1.0,
                    "w_force_exceed": 0.5,
                    "w_jerk": 0.5,
                    "w_curvature": 0.5,
                },
                indent=2,
            ),
        )

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/generate_figures.py",
        "--episodes",
        str(args.episodes),
        "--auto-out-dir",
        "--set-latest",
        "--pareto-pdf",
        "--dmetrics",
        "collisions,comfort_exposure,near_misses,snqi",
        "--dists-pdf",
        "--table-metrics",
        "collisions,comfort_exposure,near_misses,snqi",
        "--snqi-weights",
        str(weights),
    ]
    if args.baseline is not None:
        cmd += ["--snqi-baseline", str(args.baseline)]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
