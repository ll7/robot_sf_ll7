"""Run the Robot SF benchmark with the pre-trained PPO baseline.

Usage:
    uv run python examples/quickstart/02_trained_model.py

Prerequisites:
    - None (uses bundled scenario matrix and PPO baseline config)

Expected Output:
    - Prints benchmark progress messages to stdout.
    - Prints available algorithms first when not running in fast-demo mode.
    - In fast-demo mode, skips algorithm listing and runs a short horizon-4 smoke benchmark.
    - Writes PPO episode metrics to output/results/episodes_demo_ppo.jsonl.

Limitations:
    - Fast-demo mode skips algorithm listing and reduces the benchmark horizon to keep smoke
      execution short.
    - The normal interactive path still takes several seconds to execute with horizon=50.
    - `ROBOT_SF_EXAMPLES_MAX_STEPS` overrides both the default and fast-demo horizon settings.
    - Falls back to a goal-seeking policy if the PPO weights are unavailable.

References:
    - docs/dev_guide.md#quickstart
    - docs/benchmark.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from robot_sf.benchmark.cli import cli_main
from robot_sf.render.helper_catalog import ensure_output_dir


def project_root() -> Path:
    """Return the repository root regardless of the quickstart subdirectory."""

    return Path(__file__).resolve().parents[2]


def _fast_demo_enabled() -> bool:
    """Return whether the benchmark demo should prefer smoke-sized settings."""
    return os.environ.get("ROBOT_SF_FAST_DEMO", "0") == "1"


def _horizon() -> str:
    """Return the benchmark horizon override for the current execution mode."""
    override = os.environ.get("ROBOT_SF_EXAMPLES_MAX_STEPS")
    if override:
        try:
            return str(max(1, int(override)))
        except ValueError:  # pragma: no cover - defensive guard
            pass
    # Fast demo keeps the quickstart responsive when examples are exercised in CI.
    if _fast_demo_enabled():
        return "4"
    return "50"


def main() -> int:
    """Run the quickstart PPO benchmark demo and return the CLI exit code."""
    root = project_root()
    matrix = root / "configs/baselines/example_matrix.yaml"
    ppo_cfg = root / "configs/baselines/ppo.yaml"
    out = root / "output/results/episodes_demo_ppo.jsonl"
    out_dir = ensure_output_dir(out.parent)
    out = out_dir / out.name

    # Keep the full quickstart output for normal runs, but skip extra discovery work in smoke mode.
    if not _fast_demo_enabled():
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
        # The horizon helper centralizes env-driven fast-demo and explicit override handling.
        "--horizon",
        _horizon(),
        "--dt",
        "0.1",
    ]
    # Force-heavy metrics are useful in the default demo, but not needed in fast smoke mode.
    if not _fast_demo_enabled():
        argv.append("--record-forces")

    print("[quickstart] Running benchmark with the PPO baseline...", flush=True)
    code = cli_main(argv)
    if code == 0:
        print(f"[quickstart] Done. Episodes written to: {out}")
    else:
        print("[quickstart] Benchmark run failed", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
