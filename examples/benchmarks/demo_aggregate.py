"""Aggregate episode metrics with optional bootstrap confidence intervals.

Purpose:
    Demonstrate how to generate a minimal scenario matrix, run a smoke batch,
    and compute grouped aggregates with bootstrap CIs for reporting.

Usage:
    uv run python examples/benchmarks/demo_aggregate.py

Prerequisites:
    - None (matrix is generated inline for a synthetic policy)

Expected Output:
    - `output/results/demo_aggregate/episodes.jsonl` (raw episodes)
    - `output/results/demo_aggregate/summary_ci.json` (aggregate stats with CIs)

Limitations:
    - Uses a toy scenario; replace the matrix path for production analyses.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.aggregate import compute_aggregates_with_ci, read_jsonl
from robot_sf.benchmark.runner import run_batch
from robot_sf.render.helper_catalog import ensure_output_dir


def _write_demo_matrix(path: Path) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
    """
    scenarios = [
        {
            "id": "demo-agg-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 3,
        },
    ]
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


def main() -> None:
    # Use helper catalog to ensure output directory exists
    """TODO docstring. Document this function."""
    out_dir = ensure_output_dir(Path("output/results/demo_aggregate"))

    matrix_path = out_dir / "matrix.yaml"
    episodes_path = out_dir / "episodes.jsonl"

    _write_demo_matrix(matrix_path)

    # Run a tiny batch
    print("Running a tiny batch to produce episodes.jsonl ...")
    run_batch(
        scenarios_or_path=str(matrix_path),
        out_path=str(episodes_path),
        schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json",
        base_seed=0,
        repeats_override=None,
        horizon=8,
        dt=0.1,
        record_forces=False,
        append=False,
        fail_fast=False,
        progress_cb=None,
        algo="simple_policy",
        algo_config_path=None,
        snqi_weights=None,
        snqi_baseline=None,
        workers=1,
        resume=True,
    )

    # Read episodes and compute aggregates with CIs
    print("Computing grouped aggregates with bootstrap CIs ...")
    records = read_jsonl(episodes_path)
    summary = compute_aggregates_with_ci(
        records,
        group_by="scenario_params.algo",
        fallback_group_by="scenario_id",
        bootstrap_samples=500,
        bootstrap_confidence=0.95,
        bootstrap_seed=123,
    )

    summary_path = out_dir / "summary_ci.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {summary_path}")

    # Print a compact preview to stdout
    group = next(iter(summary.keys()))
    metric, stats = next(iter(summary[group].items()))
    preview = {
        "group": group,
        "metric": metric,
        "mean": stats.get("mean"),
        "mean_ci": stats.get("mean_ci"),
    }
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
