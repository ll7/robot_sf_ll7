"""Generate SNQI figures from real episodes or a diagnostic fixture.

This example demonstrates how to generate publication-ready figures with SNQI
included, using the orchestrator script. Use ``--fixture`` for a compact,
headless diagnostic run; real episode inputs remain available through the
existing CLI arguments.

Usage (from repo root):
    uv run python examples/plotting/snqi_figures_example.py \
        --episodes outputresults/episodes_sf_long_fix1.jsonl \
        --weights examples/snqi_weights_example.json \
        --baseline outputresults/baseline_stats.json

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

from examples.fixtures.snqi.v1.loader import materialize_inputs, write_summary
from robot_sf.common.artifact_paths import resolve_artifact_path


def _ensure_file(path: Path, content: str) -> Path:
    """Ensure a file exists at the resolved artifact path.

    Args:
        path: Relative path under artifact root.
        content: Text content to write if file doesn't exist.

    Returns:
        Resolved absolute path to the file.
    """
    resolved = resolve_artifact_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not resolved.exists():
        resolved.write_text(content + "\n", encoding="utf-8")
    return resolved


def main() -> int:
    """Run the SNQI figures generation pipeline.

    Returns:
        Exit code (0 for success).
    """
    ap = argparse.ArgumentParser(description="SNQI figures example runner")
    ap.add_argument("--fixture", action="store_true", help="Run the versioned diagnostic fixture")
    ap.add_argument("--episodes", type=Path, required=False, help="Episodes JSONL path")
    ap.add_argument("--out-dir", type=Path, default=None, help="Caller-owned output directory")
    ap.add_argument("--weights", type=Path, default=None, help="SNQI weights JSON path")
    ap.add_argument("--baseline", type=Path, default=None, help="Baseline stats JSON path")
    args = ap.parse_args()

    if args.fixture:
        if args.out_dir is None:
            raise SystemExit("--out-dir is required with --fixture")
        out_dir = args.out_dir.resolve()
        episodes, weights, baseline = materialize_inputs(out_dir)
    else:
        if args.episodes is None:
            raise SystemExit("--episodes is required unless --fixture is set")
        episodes = args.episodes
        weights = args.weights
        baseline = args.baseline

    # Provide a tiny default weights JSON if none is supplied
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
        str(episodes),
        "--pareto-pdf",
        "--dmetrics",
        "collisions,comfort_exposure,near_misses,snqi",
        "--dists-pdf",
        "--table-metrics",
        "collisions,comfort_exposure,near_misses,snqi",
        "--snqi-weights",
        str(weights),
    ]
    if args.fixture:
        cmd += ["--out-dir", str(args.out_dir)]
    else:
        cmd += ["--auto-out-dir", "--set-latest"]
    if baseline is not None:
        cmd += ["--snqi-baseline", str(baseline)]
    if args.fixture:
        cmd += ["--publication-style", "--format", "png,svg"]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    if args.fixture:
        outputs = [
            str(p.relative_to(args.out_dir.resolve()))
            for p in args.out_dir.resolve().rglob("*")
            if p.suffix in {".png", ".svg"}
        ]
        write_summary(args.out_dir.resolve(), example="snqi_figures_example", output_files=outputs)
        print("[diagnostic-only] Synthetic fixture output is not benchmark or scientific evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
