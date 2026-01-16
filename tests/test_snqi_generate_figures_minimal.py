"""Minimal end-to-end SNQI figure generation smoke test.

This test constructs a tiny synthetic episodes JSONL with the minimal
metric fields required for SNQI computation, writes lightweight
baseline stats + weights, and invokes the figure orchestrator with
SNQI flags. It validates that:

1. The orchestrator runs without raising.
2. A canonical output directory is created.
3. SNQI is appended to the distribution/table metric selection without error.

We keep runtime small by:
- Only 2 synthetic episodes
- Narrow distribution metric list (collisions,snqi)
- Skipping heavy plots like force-field (default path should be fast)

If this becomes slow in CI we can parametrize to skip with env var.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")


@pytest.mark.parametrize("success_vals", [[1.0, 0.0]])
def test_snqi_minimal_generate_figures(tmp_path: Path, success_vals):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        success_vals: TODO docstring.
    """
    episodes_path = tmp_path / "episodes_test.jsonl"
    weights_path = tmp_path / "weights.json"
    baseline_path = tmp_path / "baseline.json"

    # Minimal weights
    weights = {
        "w_success": 1.0,
        "w_time": 0.5,
        "w_collisions": 2.0,
        "w_near": 1.0,
        "w_comfort": 0.5,
        "w_force_exceed": 0.5,
        "w_jerk": 0.5,
        "w_curvature": 0.5,
    }
    weights_path.write_text(json.dumps(weights), encoding="utf-8")

    # Baseline stats (median/p95) for the metrics used by snqi normalization
    baseline = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
        "curvature_mean": {"med": 0.0, "p95": 1.0},
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    # Two synthetic episodes with necessary metric fields. Non-required
    # metrics default inside SNQI to optimistic values when absent from baseline.
    for i, succ in enumerate(success_vals):
        rec = {
            "episode_id": f"ep{i}",
            "scenario_id": "scnA",
            "scenario_params": {"algo": "dummy"},
            "algo": "dummy",
            "seed": i,
            # metrics required / referenced in snqi logic
            "metrics": {
                "success": succ,
                "time_to_goal_norm": 0.5 + 0.1 * i,
                "collisions": float(i),
                "near_misses": 0.0,
                "comfort_exposure": 0.0,
                "force_exceed_events": 0.0,
                "jerk_mean": 0.0,
                "curvature_mean": 0.0,
            },
        }
        with episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    assert episodes_path.exists()

    # Run orchestrator with only a couple distribution metrics to keep it fast.
    out_dir = tmp_path / "figs_out"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/generate_figures.py",
        "--episodes",
        str(episodes_path),
        "--out-dir",
        str(out_dir),
        "--dmetrics",
        "collisions,snqi",
        "--table-metrics",
        "collisions,snqi",
        "--snqi-weights",
        str(weights_path),
        "--snqi-baseline",
        str(baseline_path),
        "--no-pareto",
    ]

    # Provide schema path only if it exists (some dev envs may not have it); skip if missing.
    # Schema not required by generate_figures; skip conditional.

    subprocess.check_call(cmd, env={**os.environ, "MPLBACKEND": "Agg"})

    # Expect output directory created with at least meta.json present.
    assert out_dir.exists(), "Output directory not created"
    assert (out_dir / "meta.json").exists(), "meta.json missing in output directory"
