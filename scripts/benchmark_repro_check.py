#!/usr/bin/env python3
"""
Benchmark Reproducibility Check

Purpose: End-to-end validation of benchmark reproducibility including episode generation,
aggregation, and figure regeneration with different seeds.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

from robot_sf.render.helper_catalog import ensure_output_dir


def create_minimal_scenario():
    """Create a minimal test scenario for reproducibility testing."""
    return {
        "scenario_id": "repro_test",
        "scenario_params": {
            "algo": "simple_policy",
            "map_name": "narrow_corridor",
            "ped_count": 2,
            "robot_start": [0.0, 0.0, 0.0],
            "robot_goal": [5.0, 0.0],
            "horizon": 50,  # Short episode for fast testing
        },
        "repeats": 2,  # Minimal repeats for statistical validity
    }


def run_benchmark_pipeline(work_dir: Path, seed: int = 123):
    """Run complete benchmark pipeline in isolated directory."""
    from robot_sf.benchmark.aggregate import compute_aggregates_with_ci, read_jsonl
    from robot_sf.benchmark.runner import run_batch

    print(f"Running benchmark pipeline with seed {seed} in {work_dir}")

    # Create scenario configuration
    scenario = create_minimal_scenario()
    episodes_path = work_dir / "episodes.jsonl"
    schema_path = work_dir / "schema.json"

    # Create minimal schema (for validation)
    schema = {
        "version": "v1",
        "required_fields": ["episode_id", "scenario_id", "metrics"],
        "metrics_fields": ["collision_rate", "efficiency", "jerk"],
    }

    with schema_path.open("w") as f:
        json.dump(schema, f, indent=2)

    print("Phase 1: Generate episodes...")
    start_time = time.time()

    try:
        run_results = run_batch(
            scenarios_or_path=[scenario],
            out_path=episodes_path,
            schema_path=schema_path,
            base_seed=seed,
            horizon=50,
            workers=1,
            resume=False,  # Fresh run for reproducibility test
        )

        episode_gen_time = time.time() - start_time
        print(
            f"  Generated {run_results.get('total_episodes', 0)} episodes in {episode_gen_time:.1f}s",
        )

    except Exception as e:
        print(f"  ERROR in episode generation: {e}")
        return None

    if not episodes_path.exists() or episodes_path.stat().st_size == 0:
        print("  ERROR: No episodes generated")
        return None

    print("Phase 2: Aggregate metrics...")
    start_time = time.time()

    try:
        records = read_jsonl(episodes_path)
        if not records:
            print("  ERROR: No records found in episodes file")
            return None

        summary = compute_aggregates_with_ci(
            records,
            group_by="scenario_params.algo",
            bootstrap_samples=100,  # Small sample for speed
            bootstrap_confidence=0.95,
            bootstrap_seed=seed + 1000,  # Different seed for bootstrap
        )

        aggregate_time = time.time() - start_time
        print(f"  Aggregated {len(records)} episodes in {aggregate_time:.1f}s")

        # Save aggregation results
        summary_path = work_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

    except Exception as e:
        print(f"  ERROR in aggregation: {e}")
        return None

    print("Phase 3: Validation checks...")

    # Basic validation: check for key metrics
    required_metrics = ["collision_rate", "efficiency"]
    for group_name, group_data in summary.items():
        if not isinstance(group_data, dict):
            continue

        for metric in required_metrics:
            if metric not in group_data:
                print(f"  ERROR: Missing metric {metric} in group {group_name}")
                return None

        print(f"  Group {group_name}: {len(group_data)} metrics present")

    return {
        "episodes_file": episodes_path,
        "summary_file": summary_path,
        "episodes_count": len(records),
        "summary_groups": len(summary),
        "episode_gen_time": episode_gen_time,
        "aggregate_time": aggregate_time,
    }


def compare_reproducibility(results1, results2, tolerance=0.05):
    """Compare two benchmark runs for reproducibility."""
    print("\n=== Reproducibility Analysis ===")

    # Load and compare summaries
    with results1["summary_file"].open() as f:
        summary1 = json.load(f)

    with results2["summary_file"].open() as f:
        summary2 = json.load(f)

    # Check episode counts match
    if results1["episodes_count"] != results2["episodes_count"]:
        print(
            f"❌ Episode count mismatch: {results1['episodes_count']} vs {results2['episodes_count']}",
        )
        return False

    print(f"✅ Episode counts match: {results1['episodes_count']}")

    # Check groups match
    groups1 = set(summary1.keys())
    groups2 = set(summary2.keys())

    if groups1 != groups2:
        print(f"❌ Group mismatch: {groups1} vs {groups2}")
        return False

    print(f"✅ Groups match: {list(groups1)}")

    # Compare metric values (should be identical for deterministic algorithms)
    reproducible = True

    for group_name in groups1:
        if not isinstance(summary1[group_name], dict) or not isinstance(summary2[group_name], dict):
            continue

        group1 = summary1[group_name]
        group2 = summary2[group_name]

        common_metrics = set(group1.keys()) & set(group2.keys())

        for metric in common_metrics:
            if metric.endswith("_ci"):
                continue  # Skip CI comparisons (bootstrap variability expected)

            val1 = group1.get(metric)
            val2 = group2.get(metric)

            if val1 is None or val2 is None:
                continue

            if isinstance(val1, int | float) and isinstance(val2, int | float):
                if abs(val1 - val2) > tolerance:
                    print(
                        f"❌ {group_name}.{metric}: {val1} vs {val2} (diff: {abs(val1 - val2):.4f})",
                    )
                    reproducible = False
                else:
                    print(f"✅ {group_name}.{metric}: {val1} ≈ {val2}")

    return reproducible


def main():
    """Run end-to-end reproducibility validation."""
    print("Social Navigation Benchmark - Reproducibility Check")
    print("=" * 60)

    results_base = Path("results")
    ensure_output_dir(results_base)

    # Create temporary working directories
    with (
        tempfile.TemporaryDirectory(prefix="repro_test1_", dir=results_base) as tmp_dir1,
        tempfile.TemporaryDirectory(prefix="repro_test2_", dir=results_base) as tmp_dir2,
    ):
        work_dir1 = Path(tmp_dir1)
        work_dir2 = Path(tmp_dir2)

        print(f"Working directories: {work_dir1.name}, {work_dir2.name}")

        # Run same pipeline with different seeds
        print("\n=== Run 1 (seed=123) ===")
        results1 = run_benchmark_pipeline(work_dir1, seed=123)

        print("\n=== Run 2 (seed=456) ===")
        results2 = run_benchmark_pipeline(work_dir2, seed=456)

        if results1 is None or results2 is None:
            print("❌ Pipeline execution failed")
            return 1

        # Compare for reproducibility
        is_reproducible = compare_reproducibility(results1, results2)

        # Performance summary
        print("\n=== Performance Summary ===")
        print(
            f"Run 1: {results1['episodes_count']} episodes in {results1['episode_gen_time']:.1f}s",
        )
        print(
            f"Run 2: {results2['episodes_count']} episodes in {results2['episode_gen_time']:.1f}s",
        )
        print(f"Aggregation time: {results1['aggregate_time']:.1f}s avg")

        # Save reproducibility report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reproducible": is_reproducible,
            "run1": {
                k: v for k, v in results1.items() if k not in ["episodes_file", "summary_file"]
            },
            "run2": {
                k: v for k, v in results2.items() if k not in ["episodes_file", "summary_file"]
            },
        }

        report_path = results_base / "reproducibility_check.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

        print(f"\nReproducibility report saved to: {report_path}")

        if is_reproducible:
            print("🎉 Reproducibility check PASSED")
            return 0
        else:
            print("❌ Reproducibility check FAILED")
            return 1


if __name__ == "__main__":
    sys.exit(main())
