"""Run the Robot SF benchmark with the pre-trained PPO baseline.

Usage:
    uv run python examples/quickstart/02_trained_model.py

Prerequisites:
    - None (uses bundled scenario matrix and PPO baseline config)

Expected Output:
    - Prints available algorithms and benchmark progress messages to stdout.
    - Writes PPO episode metrics to results/episodes_demo_ppo.jsonl.

Limitations:
    - Requires several seconds to execute the benchmark even with horizon=50.
    - Falls back to a goal-seeking policy if the PPO weights are unavailable.

References:
    - docs/dev_guide.md#quickstart
    - docs/benchmark.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from robot_sf.benchmark.cli import cli_main
from robot_sf.render.helper_catalog import ensure_output_dir


def project_root() -> Path:
    """Return the repository root regardless of the quickstart subdirectory."""

    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = project_root()
    matrix = root / "configs/baselines/example_matrix.yaml"
    ppo_cfg = root / "configs/baselines/ppo.yaml"
    out = root / "results/episodes_demo_ppo.jsonl"
    out_dir = ensure_output_dir(out.parent)
    out = out_dir / out.name

    print("[quickstart] Listing available benchmark algorithms...", flush=True)
    code = cli_main(["list-algorithms"])
    if code != 0:
        print("[quickstart] list-algorithms failed", file=sys.stderr)
        return code

    argv = [
        "run",
        "--matrix",
        str(matrix),
        "--out",
        str(out),
        "--algo",
        "ppo",
        "--algo-config",
        str(ppo_cfg),
        "--repeats",
        "1",
        "--horizon",
        "50",
        "--dt",
        "0.1",
        "--record-forces",
    ]

    print("[quickstart] Running benchmark with the PPO baseline...", flush=True)
    code = cli_main(argv)
    if code == 0:
        print(f"[quickstart] Done. Episodes written to: {out}")
    else:
        print("[quickstart] Benchmark run failed", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
