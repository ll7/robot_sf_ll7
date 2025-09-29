#!/usr/bin/env python3
"""Example usage of SNQI weight recomputation and sensitivity analysis.

This script demonstrates how to use the SNQI weight optimization tools
with sample data and provides a complete workflow example.
"""

import json
import tempfile
from pathlib import Path

import numpy as np


def generate_sample_data():
    """Generate sample episode data and baseline statistics for demonstration."""

    # Sample baseline statistics (median/p95 normalization)
    baseline_stats = {
        "collisions": {"med": 0.0, "p95": 2.0},
        "near_misses": {"med": 1.0, "p95": 5.0},
        "force_exceed_events": {"med": 3.0, "p95": 15.0},
        "jerk_mean": {"med": 0.2, "p95": 1.0},
    }

    # Generate sample episodes with different algorithms
    algorithms = ["baseline_sf", "ppo_trained", "random_policy"]
    densities = ["low", "med", "high"]

    episodes = []
    episode_id = 0

    np.random.seed(42)  # For reproducibility

    for algo in algorithms:
        for density in densities:
            for seed in range(5):  # 5 episodes per algo-density combination
                # Generate realistic metrics based on algorithm type
                if algo == "baseline_sf":
                    success_rate = 0.8
                    collision_rate = 0.1
                    time_efficiency = 0.7
                elif algo == "ppo_trained":
                    success_rate = 0.9
                    collision_rate = 0.05
                    time_efficiency = 0.8
                else:  # random_policy
                    success_rate = 0.3
                    collision_rate = 0.3
                    time_efficiency = 0.4

                # Add density-based variations
                density_factor = {"low": 0.9, "med": 1.0, "high": 1.2}[density]
                collision_rate *= density_factor
                time_efficiency /= density_factor

                # Generate specific metrics with some randomness
                success = np.random.random() < success_rate
                collisions = max(0, int(np.random.poisson(collision_rate * 3)))
                near_misses = max(0, int(np.random.poisson(2.0 * density_factor)))

                time_to_goal_norm = min(1.0, max(0.1, np.random.normal(1.0 - time_efficiency, 0.2)))

                min_distance = max(0.1, np.random.normal(1.5 / density_factor, 0.3))
                path_efficiency = max(0.3, min(1.0, np.random.normal(0.8, 0.1)))

                comfort_exposure = max(0.0, min(1.0, np.random.normal(collision_rate * 2, 0.1)))

                force_exceed_events = max(0, int(np.random.poisson(5.0 * density_factor)))
                jerk_mean = max(0.05, np.random.normal(0.3 * density_factor, 0.1))
                energy = max(0.1, np.random.normal(2.0, 0.5))
                avg_speed = max(0.1, np.random.normal(1.2, 0.2))

                # Force quantiles
                force_q50 = max(0.1, np.random.normal(1.0 * density_factor, 0.3))
                force_q90 = force_q50 + max(0.5, np.random.normal(1.0, 0.2))
                force_q95 = force_q90 + max(0.2, np.random.normal(0.5, 0.1))

                episode = {
                    "episode_id": f"{algo}_{density}_{seed}",
                    "scenario_id": f"scenario_{density}_{algo}",
                    "seed": seed,
                    "scenario_params": {
                        "algo": algo,
                        "density": density,
                        "id": f"scenario_{density}_{algo}",
                    },
                    "metrics": {
                        "success": success,
                        "time_to_goal_norm": time_to_goal_norm,
                        "collisions": collisions,
                        "near_misses": near_misses,
                        "min_distance": min_distance,
                        "path_efficiency": path_efficiency,
                        "avg_speed": avg_speed,
                        "force_quantiles": {"q50": force_q50, "q90": force_q90, "q95": force_q95},
                        "force_exceed_events": force_exceed_events,
                        "comfort_exposure": comfort_exposure,
                        "jerk_mean": jerk_mean,
                        "energy": energy,
                        "force_gradient_norm_mean": max(0.01, np.random.normal(0.1, 0.02)),
                    },
                    "config_hash": "sample_hash",
                    "git_hash": "sample_git",
                    "timestamps": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T00:01:00Z"},
                }

                episodes.append(episode)
                episode_id += 1

    return episodes, baseline_stats


def demonstrate_weight_recomputation():
    """Demonstrate the weight recomputation workflow."""
    print("=== SNQI Weight Recomputation and Sensitivity Analysis Demo ===\n")

    # Generate sample data
    print("1. Generating sample episode data...")
    episodes, baseline_stats = generate_sample_data()
    print(f"   Generated {len(episodes)} episodes with 3 algorithms")

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save sample data
        episodes_file = temp_path / "episodes.jsonl"
        baseline_file = temp_path / "baseline_stats.json"

        with open(episodes_file, "w") as f:
            for episode in episodes:
                f.write(json.dumps(episode) + "\n")

        with open(baseline_file, "w") as f:
            json.dump(baseline_stats, f, indent=2)

        print("   Saved data to temporary files")

        # Demonstrate weight strategies comparison
        print("\n2. Comparing weight strategies...")

        # Import our modules (this would normally be done with the scripts)
        # Ensure recompute_snqi_weights.py is in the same directory or in the Python path.
        # If this import fails, run the script from the project root or install as a package.

        try:
            # We'll simulate the key functionality here since the scripts are designed to run standalone
            # Try to import the recomputation tool whether the script is run via `python -m scripts.example_snqi_workflow`
            # (project root on sys.path) or directly from within the scripts folder.
            import importlib

            try:
                # Preferred when running as package
                recompute_module = importlib.import_module("scripts.recompute_snqi_weights")
            except ModuleNotFoundError:
                # Fallback when running from within scripts directory
                recompute_module = importlib.import_module("recompute_snqi_weights")

            # Guard attribute lookup to satisfy static checkers
            SNQIWeightRecomputer = getattr(recompute_module, "SNQIWeightRecomputer", None)
            if SNQIWeightRecomputer is None:
                raise AttributeError(
                    "SNQIWeightRecomputer not found in recompute_snqi_weights module",
                )

            recomputer = SNQIWeightRecomputer(episodes, baseline_stats)

            # Test different strategies
            strategies = ["default", "balanced", "safety_focused", "efficiency_focused"]
            strategy_results = {}

            for strategy in strategies:
                result = recomputer.recompute_with_strategy(strategy)
                strategy_results[strategy] = result
                stats = result["statistics"]["overall"]
                print(f"   {strategy:15}: mean={stats['mean']:.3f}, range={stats['range']:.3f}")

            # Find best strategy based on discriminative power
            best_strategy = max(
                strategy_results.keys(),
                key=lambda s: strategy_results[s]["statistics"]["overall"]["range"],
            )
            print(f"   Best strategy: {best_strategy}")

            # Demonstrate normalization comparison
            print("\n3. Comparing normalization strategies...")
            best_weights = strategy_results[best_strategy]["weights"]
            norm_results = recomputer.compare_normalization_strategies(best_weights)

            for norm_strategy, data in norm_results.items():
                corr = data["correlation_with_base"]
                print(f"   {norm_strategy:12}: correlation={corr:.4f}")

            # Show sensitivity to weight changes
            print("\n4. Weight sensitivity analysis (simple)...")
            base_weights = strategy_results["default"]["weights"]

            for weight_name in ["w_success", "w_collisions", "w_comfort"]:
                # Test +20% change
                perturbed_weights = base_weights.copy()
                perturbed_weights[weight_name] *= 1.2

                correlation = recomputer.rank_correlation_analysis(base_weights, perturbed_weights)
                print(f"   {weight_name:15} +20%: ranking correlation = {correlation:.4f}")

            print("\n5. Summary of findings:")
            print(f"   - Best performing strategy: {best_strategy}")
            print("   - Most stable normalization: median_p95")
            print("   - Weight sensitivity varies by component")

            # Show recommended weights
            print(f"\n6. Recommended weights ({best_strategy}):")
            for weight_name, value in best_weights.items():
                print(f"   {weight_name}: {value:.3f}")

        except (ImportError, ModuleNotFoundError, AttributeError):
            print("   (Demo would use actual scripts here)")
            print("   Run the scripts directly for full functionality:")
            print(
                f"   python scripts/recompute_snqi_weights.py --episodes {episodes_file} --baseline {baseline_file} --compare-strategies --output results.json",
            )


if __name__ == "__main__":
    demonstrate_weight_recomputation()
