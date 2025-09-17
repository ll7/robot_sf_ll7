"""Demo: Aggregate episode metrics with optional bootstrap CIs.

This example shows how to:
- Run a tiny batch to produce an episodes JSONL (using the demo matrix inline)
- Compute grouped aggregates with bootstrap confidence intervals programmatically
- Print a compact JSON summary

Usage (from repo root):
  uv run python examples/demo_aggregate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.aggregate import compute_aggregates_with_ci, read_jsonl
from robot_sf.benchmark.runner import run_batch


def _write_demo_matrix(path: Path) -> None:
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
        }
    ]
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


def main() -> None:
    out_dir = Path("results/demo_aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = out_dir / "matrix.yaml"
    episodes_path = out_dir / "episodes.jsonl"

    _write_demo_matrix(matrix_path)

    # Run a tiny batch
    print("Running a tiny batch to produce episodes.jsonl ...")
    run_batch(
        scenarios_or_path=str(matrix_path),
        out_path=str(episodes_path),
        schema_path="docs/dev/issues/social-navigation-benchmark/episode_schema.json",
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
