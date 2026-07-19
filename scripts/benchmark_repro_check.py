#!/usr/bin/env python3
"""
Benchmark Reproducibility Check

Purpose: End-to-end validation of benchmark reproducibility including episode generation
and aggregation in fresh runs with the same seed.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path
from robot_sf.render.helper_catalog import ensure_output_dir

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EPISODE_SCHEMA_PATH = (
    REPOSITORY_ROOT / "robot_sf" / "benchmark" / "schemas" / "episode.schema.v1.json"
)
REQUIRED_AGGREGATE_METRICS = ("collisions", "path_efficiency", "success")
REQUIRED_AGGREGATE_STATISTICS = ("mean", "median", "p95")


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


def validate_simple_policy_aggregate(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate the required simple-policy aggregate shape without changing metric semantics.

    Returns:
        Structured pass/fail diagnostic suitable for a fail-closed compatibility check.
    """
    diagnostic: dict[str, Any] = {
        "schema": "benchmark_repro_check.aggregate_validation.v1",
        "group": "simple_policy",
        "required_metrics": list(REQUIRED_AGGREGATE_METRICS),
        "required_statistics": list(REQUIRED_AGGREGATE_STATISTICS),
        "missing_metrics": [],
        "missing_statistics": [],
    }
    group_data = summary.get("simple_policy")
    if not isinstance(group_data, dict):
        diagnostic["status"] = "failed"
        diagnostic["missing_group"] = "simple_policy"
        return diagnostic

    for metric in REQUIRED_AGGREGATE_METRICS:
        metric_data = group_data.get(metric)
        if not isinstance(metric_data, dict):
            diagnostic["missing_metrics"].append(metric)
            continue
        for statistic in REQUIRED_AGGREGATE_STATISTICS:
            if not isinstance(metric_data.get(statistic), int | float):
                diagnostic["missing_statistics"].append(f"{metric}.{statistic}")

    diagnostic["status"] = (
        "passed"
        if not diagnostic["missing_metrics"] and not diagnostic["missing_statistics"]
        else "failed"
    )
    return diagnostic


def _pipeline_failure(stage: str, error: str | dict[str, Any]) -> dict[str, Any]:
    """Return and print a structured fail-closed pipeline diagnostic."""
    result = {
        "status": "failed",
        "stage": stage,
        "error": error,
    }
    print(f"  ERROR: {json.dumps(result, sort_keys=True)}")
    return result


def _report_run(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the JSON-safe subset of a pipeline result."""
    if result is None:
        return None
    return {
        key: value for key, value in result.items() if key not in {"episodes_file", "summary_file"}
    }


def _write_reproducibility_report(
    results_base: Path,
    *,
    status: str,
    stage: str,
    error: str | dict[str, Any] | None,
    reproducible: bool,
    run1: dict[str, Any] | None = None,
    run2: dict[str, Any] | None = None,
) -> Path:
    """Write the uploadable report for both successful and failed checks."""
    report = {
        "schema": "benchmark_repro_check.report.v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "stage": stage,
        "error": error,
        "reproducible": reproducible,
        "run1": _report_run(run1),
        "run2": _report_run(run2),
    }
    results_base.mkdir(parents=True, exist_ok=True)
    report_path = results_base / "reproducibility_check.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\nReproducibility report saved to: {report_path}")
    return report_path


def run_benchmark_pipeline(work_dir: Path, seed: int = 123):
    """Run complete benchmark pipeline in isolated directory."""
    from robot_sf.benchmark.aggregate import compute_aggregates_with_ci, read_jsonl
    from robot_sf.benchmark.runner import run_batch

    print(f"Running benchmark pipeline with seed {seed} in {work_dir}")

    # Create scenario configuration
    scenario = create_minimal_scenario()
    episodes_path = work_dir / "episodes.jsonl"
    schema_path = EPISODE_SCHEMA_PATH

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
            f"  Generated {run_results.get('written', 0)} episodes in {episode_gen_time:.1f}s",
        )

    except Exception as e:
        return _pipeline_failure("episode_generation", str(e))

    if not episodes_path.exists() or episodes_path.stat().st_size == 0:
        return _pipeline_failure("episode_generation", "No episodes generated")

    print("Phase 2: Aggregate metrics...")
    start_time = time.time()

    try:
        records = read_jsonl(episodes_path)
        if not records:
            return _pipeline_failure("aggregation", "No records found in episodes file")

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
        return _pipeline_failure("aggregation", str(e))

    print("Phase 3: Validation checks...")

    validation = validate_simple_policy_aggregate(summary)
    if validation["status"] != "passed":
        return _pipeline_failure("aggregate_validation", validation)

    group_data = summary["simple_policy"]
    print(f"  Group simple_policy: {len(group_data)} aggregate metrics present")

    return {
        "status": "passed",
        "stage": "completed",
        "error": None,
        "episodes_file": episodes_path,
        "summary_file": summary_path,
        "episodes_count": len(records),
        "summary_groups": len(summary),
        "episode_gen_time": episode_gen_time,
        "aggregate_time": aggregate_time,
    }


def compare_reproducibility(results1, results2):
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

    for metric in REQUIRED_AGGREGATE_METRICS:
        metric1 = summary1.get("simple_policy", {}).get(metric)
        metric2 = summary2.get("simple_policy", {}).get(metric)
        if not isinstance(metric1, dict) or not isinstance(metric2, dict):
            print(f"❌ Missing aggregate metric for reproducibility comparison: {metric}")
            reproducible = False
            continue

        for statistic in REQUIRED_AGGREGATE_STATISTICS:
            val1 = metric1.get(statistic)
            val2 = metric2.get(statistic)
            if not isinstance(val1, int | float) or not isinstance(val2, int | float):
                print(f"❌ Missing numeric aggregate statistic: {metric}.{statistic}")
                reproducible = False
                continue
            if val1 != val2:
                print(
                    f"❌ simple_policy.{metric}.{statistic}: {val1} vs {val2} "
                    f"(diff: {abs(val1 - val2):.17g})",
                )
                reproducible = False
            else:
                print(f"✅ simple_policy.{metric}.{statistic}: {val1} == {val2}")

    return reproducible


def main():
    """Run end-to-end reproducibility validation."""
    print("Social Navigation Benchmark - Reproducibility Check")
    print("=" * 60)

    results_base = REPOSITORY_ROOT / "output" / "benchmarks"
    results1: dict[str, Any] | None = None
    results2: dict[str, Any] | None = None
    try:
        ensure_canonical_tree(categories=("benchmarks",))
        results_base = get_artifact_category_path("benchmarks")
        ensure_output_dir(results_base)

        with (
            tempfile.TemporaryDirectory(prefix="repro_test1_", dir=results_base) as tmp_dir1,
            tempfile.TemporaryDirectory(prefix="repro_test2_", dir=results_base) as tmp_dir2,
        ):
            work_dir1 = Path(tmp_dir1)
            work_dir2 = Path(tmp_dir2)

            print(f"Working directories: {work_dir1.name}, {work_dir2.name}")

            # Run the same seeded pipeline twice in fresh directories. Different seeds
            # represent different stochastic episodes, not a reproducibility failure.
            print("\n=== Run 1 (seed=123) ===")
            results1 = run_benchmark_pipeline(work_dir1, seed=123)
            if results1["status"] != "passed":
                _write_reproducibility_report(
                    results_base,
                    status="failed",
                    stage=f"run1.{results1['stage']}",
                    error=results1["error"],
                    reproducible=False,
                    run1=results1,
                )
                return 1

            print("\n=== Run 2 (seed=123) ===")
            results2 = run_benchmark_pipeline(work_dir2, seed=123)
            if results2["status"] != "passed":
                _write_reproducibility_report(
                    results_base,
                    status="failed",
                    stage=f"run2.{results2['stage']}",
                    error=results2["error"],
                    reproducible=False,
                    run1=results1,
                    run2=results2,
                )
                return 1

            is_reproducible = compare_reproducibility(results1, results2)

            print("\n=== Performance Summary ===")
            print(
                f"Run 1: {results1['episodes_count']} episodes "
                f"in {results1['episode_gen_time']:.1f}s",
            )
            print(
                f"Run 2: {results2['episodes_count']} episodes "
                f"in {results2['episode_gen_time']:.1f}s",
            )
            print(f"Aggregation time: {results1['aggregate_time']:.1f}s avg")

            if not is_reproducible:
                _write_reproducibility_report(
                    results_base,
                    status="failed",
                    stage="comparison",
                    error="Same-seed aggregate statistics differ",
                    reproducible=False,
                    run1=results1,
                    run2=results2,
                )
                print("❌ Reproducibility check FAILED")
                return 1

            _write_reproducibility_report(
                results_base,
                status="passed",
                stage="completed",
                error=None,
                reproducible=True,
                run1=results1,
                run2=results2,
            )
            print("🎉 Reproducibility check PASSED")
            return 0
    except (KeyError, OSError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"❌ Reproducibility check failed during setup or execution: {error}")
        try:
            _write_reproducibility_report(
                results_base,
                status="failed",
                stage="setup_or_execution",
                error=error,
                reproducible=False,
                run1=results1,
                run2=results2,
            )
        except (OSError, TypeError, ValueError) as report_exc:
            print(f"❌ Could not write reproducibility report: {report_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
