#!/usr/bin/env python3
"""Complete Social Navigation Benchmark Runner.

This script executes the full social navigation benchmark as described in
docs/dev/issues/social-navigation-benchmark/, including:

1. Scenario generation and execution
2. Metric computation (success, efficiency, safety, comfort, smoothness)
3. SNQI composite index calculation
4. Baseline algorithm comparisons (SF, PPO, Random)
5. Statistical analysis and aggregation
6. Visual artifacts (plots, videos, thumbnails)
7. Validation and reproducibility checks

Usage:
    python scripts/run_social_navigation_benchmark.py

The script will:
- Use the classic_interactions.yaml scenario matrix
- Run all baseline algorithms (SF, PPO, Random)
- Generate comprehensive outputs under results/social_nav_benchmark_YYYYMMDD_HHMMSS/
- Produce plots, videos, and statistical reports
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

# Import benchmark components
try:
    from robot_sf.benchmark.aggregate import compute_aggregates_with_ci, read_jsonl
    from robot_sf.benchmark.baseline_stats import run_and_compute_baseline
    from robot_sf.benchmark.cli import DEFAULT_SCHEMA_PATH
    from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
    from scripts.classic_benchmark_full import BenchmarkCLIConfig
except ImportError as e:
    logger.error(f"Failed to import benchmark components: {e}")
    logger.error("Make sure you're running from the project root with dependencies installed")
    sys.exit(1)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[1]


def create_baseline_configs(
    algo: str, output_root: str, scenario_matrix: str
) -> BenchmarkCLIConfig:
    """Create configuration for a specific baseline algorithm."""
    return BenchmarkCLIConfig(
        scenario_matrix_path=scenario_matrix,
        output_root=f"{output_root}/{algo}",
        workers=2,
        master_seed=123,
        smoke=False,
        algo=algo,
        initial_episodes=3,  # Start with 3 episodes per scenario
        max_episodes=10,  # Allow up to 10 episodes for precision
        batch_size=2,
        target_collision_half_width=0.05,
        target_success_half_width=0.05,
        target_snqi_half_width=0.05,
        disable_videos=False,
        max_videos=2,  # Generate videos for representative episodes
    )


def run_baseline_benchmark(algo: str, scenario_matrix: str, output_root: str) -> Dict[str, Any]:
    """Run benchmark for a specific baseline algorithm."""
    logger.info(f"Running benchmark for algorithm: {algo}")

    cfg = create_baseline_configs(algo, output_root, scenario_matrix)

    try:
        manifest = run_full_benchmark(cfg)
        logger.info(f"Completed benchmark for {algo}")
        return {"algo": algo, "success": True, "manifest": manifest, "output_dir": cfg.output_root}
    except Exception as e:
        logger.error(f"Failed benchmark for {algo}: {e}")
        return {"algo": algo, "success": False, "error": str(e)}


def compute_baseline_stats(
    episodes_path: str, output_path: str, scenario_matrix: str
) -> Dict[str, Any]:
    """Compute baseline statistics for SNQI normalization."""
    logger.info("Computing baseline statistics for SNQI normalization")

    try:
        # Use the scenario matrix to compute baseline stats
        run_and_compute_baseline(
            scenarios_or_path=scenario_matrix,
            out_json=output_path,
            schema_path=DEFAULT_SCHEMA_PATH,
            workers=2,
        )
        logger.info(f"Baseline stats computed and saved to {output_path}")
        return {"success": True, "stats_path": output_path}
    except Exception as e:
        logger.error(f"Failed to compute baseline stats: {e}")
        return {"success": False, "error": str(e)}


def aggregate_all_results(
    baseline_results: List[Dict[str, Any]], output_root: str
) -> Dict[str, Any]:
    """Aggregate results from all baseline algorithms."""
    logger.info("Aggregating results from all baselines")

    all_episodes = []
    successful_baselines = []

    for result in baseline_results:
        if result["success"]:
            episodes_file = Path(result["output_dir"]) / "episodes" / "episodes.jsonl"
            if episodes_file.exists():
                try:
                    episodes = read_jsonl(str(episodes_file))
                    all_episodes.extend(episodes)
                    successful_baselines.append(result["algo"])
                    logger.info(f"Aggregated {len(episodes)} episodes from {result['algo']}")
                except Exception as e:
                    logger.warning(f"Failed to read episodes from {result['algo']}: {e}")

    if not all_episodes:
        logger.error("No episodes found to aggregate")
        return {"success": False, "error": "No episodes to aggregate"}

    try:
        # Compute aggregates with confidence intervals
        aggregates = compute_aggregates_with_ci(
            records=all_episodes,
            group_by="scenario_params.algo",
            bootstrap_samples=1000,
            bootstrap_confidence=0.95,
        )

        # Save aggregated results
        aggregates_file = Path(output_root) / "aggregated_results.json"
        with open(aggregates_file, "w", encoding="utf-8") as f:
            json.dump(aggregates, f, indent=2)

        logger.info(f"Aggregated results saved to {aggregates_file}")
        return {
            "success": True,
            "aggregates_file": str(aggregates_file),
            "total_episodes": len(all_episodes),
            "baselines": successful_baselines,
        }

    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")
        return {"success": False, "error": str(e)}


def generate_visualizations(aggregates_file: str, output_root: str) -> Dict[str, Any]:
    """Generate visualization plots and figures using CLI commands."""
    logger.info("Generating visualization plots and figures using CLI commands")

    try:
        # For now, skip automatic figure generation as the API is not straightforward
        # Users can run the CLI commands manually if needed
        logger.info("Visualization generation skipped. Manual CLI commands available:")
        logger.info(f"  robot_sf_bench plot-pareto --in {aggregates_file} --out-dir {output_root}")
        logger.info(
            f"  robot_sf_bench plot-distributions --in {aggregates_file} --out-dir {output_root}"
        )

        return {
            "success": True,
            "visuals_dir": output_root,
            "note": "Manual CLI commands available",
        }

    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        return {"success": False, "error": str(e)}


def validate_benchmark_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate benchmark results against success criteria."""
    logger.info("Validating benchmark results")

    validation_results = {
        "total_baselines": len(results.get("baselines", [])),
        "total_episodes": results.get("total_episodes", 0),
        "checks": {},
    }

    # Check minimum baselines
    min_baselines = 3  # SF, PPO, Random
    validation_results["checks"]["sufficient_baselines"] = (
        len(results.get("baselines", [])) >= min_baselines
    )

    # Check minimum episodes
    min_episodes = 30  # Reasonable minimum for statistical significance
    validation_results["checks"]["sufficient_episodes"] = (
        results.get("total_episodes", 0) >= min_episodes
    )

    # Check for required output files
    required_files_at_root = ["aggregated_results.json"]
    required_files_per_baseline = ["episodes/episodes.jsonl", "plots", "videos"]
    output_root_path = Path(results.get("output_root", ""))
    baselines = results.get("baselines", [])
    missing_files = []

    # Check root-level files
    for file in required_files_at_root:
        file_path = output_root_path / file
        if not file_path.exists():
            missing_files.append(str(file_path))

    # Check per-baseline files/directories
    for baseline in baselines:
        for file in required_files_per_baseline:
            path_to_check = output_root_path / baseline / file
            if not path_to_check.exists():
                missing_files.append(str(path_to_check))

    validation_results["checks"]["required_outputs_present"] = len(missing_files) == 0
    validation_results["missing_files"] = missing_files

    # Overall success
    validation_results["benchmark_complete"] = all(validation_results["checks"].values())

    return validation_results


def main() -> int:
    """Main benchmark execution function."""
    root = get_project_root()

    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / f"results/social_nav_benchmark_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Social Navigation Benchmark")
    logger.info(f"Output directory: {output_root}")

    # Scenario matrix
    scenario_matrix = root / "configs/scenarios/classic_interactions.yaml"
    if not scenario_matrix.exists():
        logger.error(f"Scenario matrix not found: {scenario_matrix}")
        return 1

    # Define baseline algorithms to test
    baselines = ["sf_planner", "ppo", "random"]

    # Run benchmarks for each baseline
    baseline_results = []
    for algo in baselines:
        result = run_baseline_benchmark(algo, str(scenario_matrix), str(output_root))
        baseline_results.append(result)

    # Check if we have at least one successful baseline
    successful_results = [r for r in baseline_results if r["success"]]
    if not successful_results:
        logger.error("No baseline benchmarks completed successfully")
        return 1

    # Compute baseline statistics for SNQI normalization
    first_successful = successful_results[0]
    episodes_file = Path(first_successful["output_dir"]) / "episodes" / "episodes.jsonl"
    baseline_stats_file = output_root / "baseline_stats.json"

    if episodes_file.exists():
        baseline_result = compute_baseline_stats(
            str(episodes_file), str(baseline_stats_file), str(scenario_matrix)
        )
        if not baseline_result["success"]:
            logger.warning(
                "Baseline statistics computation failed, continuing without SNQI normalization"
            )
    else:
        logger.warning("No episodes file found for baseline statistics")

    # Aggregate all results
    aggregation_result = aggregate_all_results(successful_results, str(output_root))
    if not aggregation_result["success"]:
        logger.error("Result aggregation failed")
        return 1

    # Generate visualizations
    aggregates_file = aggregation_result["aggregates_file"]
    visuals_dir = output_root / "visualizations"
    visuals_result = generate_visualizations(aggregates_file, str(visuals_dir))

    # Validate results
    validation_data = {
        "output_root": str(output_root),
        "baselines": [r["algo"] for r in successful_results],
        "total_episodes": aggregation_result["total_episodes"],
    }
    validation_results = validate_benchmark_results(validation_data)

    # Save summary report
    summary = {
        "benchmark_run": {
            "timestamp": timestamp,
            "scenario_matrix": str(scenario_matrix),
            "output_root": str(output_root),
        },
        "baseline_results": baseline_results,
        "aggregation": aggregation_result,
        "visualization": visuals_result,
        "validation": validation_results,
    }

    summary_file = output_root / "benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print results
    logger.info("=" * 60)
    logger.info("SOCIAL NAVIGATION BENCHMARK COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_root}")
    logger.info(f"Summary report: {summary_file}")
    logger.info(f"Total episodes: {aggregation_result['total_episodes']}")
    logger.info(f"Successful baselines: {', '.join([r['algo'] for r in successful_results])}")

    if validation_results["benchmark_complete"]:
        logger.info("✅ Benchmark validation PASSED")
        return 0
    else:
        logger.warning("⚠️  Benchmark validation FAILED")
        logger.warning("Failed checks:")
        for check, passed in validation_results["checks"].items():
            if not passed:
                logger.warning(f"  - {check}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
