"""Demo: Run the Social Navigation Benchmark via robot_sf_bench (PPO).

This example invokes the unified CLI programmatically to:
  1) List available algorithms
  2) Run a short batch using the example scenario matrix and PPO baseline config

Defaults:
  - Matrix: configs/baselines/example_matrix.yaml
  - Algo config: configs/baselines/ppo.yaml
  - Output: results/episodes_demo_ppo.jsonl

Notes:
  - If the PPO model file is missing, the PPO baseline adapter will fall back
    to a simple goal-seeking action (configurable in the YAML).
  - Adjust --repeats/--horizon for faster/slower demo runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from robot_sf.benchmark.cli import cli_main


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = project_root()
    matrix = root / "configs/baselines/example_matrix.yaml"
    ppo_cfg = root / "configs/baselines/ppo.yaml"
    out = root / "results/episodes_demo_ppo.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[demo] Listing algorithms...", flush=True)
    code = cli_main(["list-algorithms"])
    if code != 0:
        print("[demo] list-algorithms failed", file=sys.stderr)
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

    print("[demo] Running benchmark with PPO...", flush=True)
    code = cli_main(argv)
    if code == 0:
        print(f"[demo] Done. Episodes written to: {out}")
    else:
        print("[demo] Benchmark run failed", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
