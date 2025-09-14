#!/usr/bin/env python3
"""SNQI Sensitivity Analysis Script

This script performs detailed sensitivity analysis for SNQI weights, analyzing how
changes in weights affect rankings, score distributions, and discriminative power.
It supports the median/p95 normalization strategy and provides various visualizations.

Usage:
    python scripts/snqi_sensitivity_analysis.py --episodes episodes.jsonl --baseline baseline_stats.json --weights weights.json --output analysis/

Requirements:
    - Episode data in JSONL format
    - Baseline statistics for normalization
    - Weight configuration (can be default or optimized weights)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

# Import canonical SNQI computation utilities
from robot_sf.benchmark.snqi import WEIGHT_NAMES, compute_snqi

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import importlib.util

    PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
except ImportError:
    PANDAS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SNQISensitivityAnalyzer:
    """Comprehensive SNQI sensitivity analysis."""

    def __init__(self, episodes_data: List[Dict], baseline_stats: Dict[str, Dict[str, float]]):
        """Initialize analyzer with episode data and baseline statistics."""
        self.episodes = episodes_data
        self.baseline_stats = baseline_stats
        self.weight_names = list(WEIGHT_NAMES)

    def _episode_snqi(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """Wrapper calling canonical compute_snqi with this instance's baseline statistics."""
        return compute_snqi(metrics, weights, self.baseline_stats)

    def weight_sweep_analysis(
        self,
        base_weights: Dict[str, float],
        weight_ranges: Dict[str, Tuple[float, float]] = None,
        n_points: int = 20,
    ) -> Dict[str, Any]:
        """Perform sweep analysis for each weight individually."""
        if weight_ranges is None:
            weight_ranges = {name: (0.1, 3.0) for name in self.weight_names}

        results = {}

        for weight_name in self.weight_names:
            logger.info(f"Analyzing sensitivity of {weight_name}")

            min_val, max_val = weight_ranges.get(weight_name, (0.1, 3.0))
            weight_values = np.linspace(min_val, max_val, n_points)

            sweep_data = {
                "weight_values": weight_values.tolist(),
                "mean_snqi": [],
                "std_snqi": [],
                "ranking_correlation": [],
                "score_distribution": [],
            }

            # Compute base ranking for correlation analysis
            base_scores = [
                self._episode_snqi(ep.get("metrics", {}), base_weights) for ep in self.episodes
            ]
            base_ranking = np.argsort(np.argsort(base_scores))

            for weight_val in weight_values:
                # Create modified weights
                modified_weights = base_weights.copy()
                modified_weights[weight_name] = weight_val

                # Compute SNQI scores
                scores = [
                    self._episode_snqi(ep.get("metrics", {}), modified_weights)
                    for ep in self.episodes
                ]

                # Statistics
                sweep_data["mean_snqi"].append(np.mean(scores))
                sweep_data["std_snqi"].append(np.std(scores))

                # Ranking correlation
                new_ranking = np.argsort(np.argsort(scores))
                corr, _ = spearmanr(base_ranking, new_ranking)
                sweep_data["ranking_correlation"].append(corr if not np.isnan(corr) else 1.0)

                # Store score distribution for later analysis
                sweep_data["score_distribution"].append(scores)

            results[weight_name] = sweep_data

        return results

    def pairwise_weight_analysis(
        self,
        base_weights: Dict[str, float],
        weight_pairs: List[Tuple[str, str]] = None,
        n_points: int = 15,
    ) -> Dict[str, Any]:
        """Analyze sensitivity of weight pairs (2D analysis)."""
        if weight_pairs is None:
            # Analyze most important pairs
            weight_pairs = [
                ("w_success", "w_collisions"),
                ("w_time", "w_comfort"),
                ("w_collisions", "w_near"),
                ("w_comfort", "w_force_exceed"),
            ]

        results = {}

        for weight1, weight2 in weight_pairs:
            logger.info(f"Analyzing pair ({weight1}, {weight2})")

            # Create 2D grid
            w1_values = np.linspace(0.1, 3.0, n_points)
            w2_values = np.linspace(0.1, 3.0, n_points)
            W1, W2 = np.meshgrid(w1_values, w2_values)

            snqi_surface = np.zeros_like(W1)
            stability_surface = np.zeros_like(W1)

            base_scores = [
                self._episode_snqi(ep.get("metrics", {}), base_weights) for ep in self.episodes
            ]
            base_ranking = np.argsort(np.argsort(base_scores))

            for i in range(n_points):
                for j in range(n_points):
                    modified_weights = base_weights.copy()
                    modified_weights[weight1] = W1[i, j]
                    modified_weights[weight2] = W2[i, j]

                    scores = [
                        self._episode_snqi(ep.get("metrics", {}), modified_weights)
                        for ep in self.episodes
                    ]
                    snqi_surface[i, j] = np.mean(scores)

                    # Ranking stability
                    new_ranking = np.argsort(np.argsort(scores))
                    corr, _ = spearmanr(base_ranking, new_ranking)
                    stability_surface[i, j] = corr if not np.isnan(corr) else 1.0

            results[f"{weight1}_vs_{weight2}"] = {
                "weight1_name": weight1,
                "weight2_name": weight2,
                "weight1_values": w1_values.tolist(),
                "weight2_values": w2_values.tolist(),
                "snqi_surface": snqi_surface.tolist(),
                "stability_surface": stability_surface.tolist(),
            }

        return results

    def ablation_analysis(self, base_weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze effect of removing each weight component (ablation study)."""
        logger.info("Performing ablation analysis")

        results = {"base_performance": {}, "ablated_performance": {}, "importance_ranking": []}

        # Base performance
        base_scores = [
            self._episode_snqi(ep.get("metrics", {}), base_weights) for ep in self.episodes
        ]
        base_ranking = np.argsort(np.argsort(base_scores))

        results["base_performance"] = {
            "mean_snqi": np.mean(base_scores),
            "std_snqi": np.std(base_scores),
            "score_range": np.max(base_scores) - np.min(base_scores),
        }

        importance_scores = []

        for weight_name in self.weight_names:
            # Create weights with this component zeroed out
            ablated_weights = base_weights.copy()
            ablated_weights[weight_name] = 0.0

            ablated_scores = [
                self._episode_snqi(ep.get("metrics", {}), ablated_weights) for ep in self.episodes
            ]
            ablated_ranking = np.argsort(np.argsort(ablated_scores))

            # Measure impact
            score_change = abs(np.mean(ablated_scores) - np.mean(base_scores))
            ranking_change = 1.0 - spearmanr(base_ranking, ablated_ranking)[0]
            if np.isnan(ranking_change):
                ranking_change = 0.0

            variance_change = abs(np.std(ablated_scores) - np.std(base_scores))

            # Combined importance score
            importance = 0.4 * score_change + 0.4 * ranking_change + 0.2 * variance_change
            importance_scores.append((weight_name, importance))

            results["ablated_performance"][weight_name] = {
                "mean_snqi": np.mean(ablated_scores),
                "std_snqi": np.std(ablated_scores),
                "score_change": score_change,
                "ranking_correlation_loss": ranking_change,
                "variance_change": variance_change,
                "importance_score": importance,
            }

        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        results["importance_ranking"] = importance_scores

        return results

    def normalization_strategy_analysis(self, base_weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze impact of different normalization strategies (median vs p95)."""
        logger.info("Analyzing normalization strategy impact")

        # Collect raw metric values
        metric_values = {
            metric: []
            for metric in ["collisions", "near_misses", "force_exceed_events", "jerk_mean"]
        }

        for ep in self.episodes:
            metrics = ep.get("metrics", {})
            for metric_name in metric_values.keys():
                value = metrics.get(metric_name)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    metric_values[metric_name].append(float(value))

        # Create alternative strategies
        strategies = {
            "median_p95": self.baseline_stats,  # Original
            "median_p90": {},
            "median_max": {},
            "p25_p75": {},
        }

        for metric_name, values in metric_values.items():
            if len(values) > 0:
                arr = np.array(values)
                strategies["median_p90"][metric_name] = {
                    "med": float(np.median(arr)),
                    "p95": float(np.percentile(arr, 90)),
                }
                strategies["median_max"][metric_name] = {
                    "med": float(np.median(arr)),
                    "p95": float(np.max(arr)),
                }
                strategies["p25_p75"][metric_name] = {
                    "med": float(np.percentile(arr, 25)),
                    "p95": float(np.percentile(arr, 75)),
                }

        results = {}
        base_scores = None

        for strategy_name, strategy_baseline in strategies.items():
            # Temporarily replace baseline stats
            original_baseline = self.baseline_stats
            self.baseline_stats = strategy_baseline

            scores = [
                self._episode_snqi(ep.get("metrics", {}), base_weights) for ep in self.episodes
            ]

            if base_scores is None:
                base_scores = scores
                base_ranking = np.argsort(np.argsort(scores))

            current_ranking = np.argsort(np.argsort(scores))
            ranking_corr, _ = spearmanr(base_ranking, current_ranking)

            results[strategy_name] = {
                "mean_snqi": np.mean(scores),
                "std_snqi": np.std(scores),
                "score_range": np.max(scores) - np.min(scores),
                "ranking_correlation": ranking_corr if not np.isnan(ranking_corr) else 1.0,
            }

            # Restore original baseline
            self.baseline_stats = original_baseline

        return results

    def generate_visualizations(self, analysis_results: Dict[str, Any], output_dir: Path):
        """Generate visualization plots for sensitivity analysis."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping visualizations")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use("default")
            if "sns" in globals():
                sns.set_palette("husl")

        # 1. Weight sweep sensitivity plots
        if "weight_sweep" in analysis_results:
            self._plot_weight_sweeps(analysis_results["weight_sweep"], output_dir)

        # 2. Pairwise analysis heatmaps
        if "pairwise" in analysis_results:
            self._plot_pairwise_analysis(analysis_results["pairwise"], output_dir)

        # 3. Ablation analysis bar chart
        if "ablation" in analysis_results:
            self._plot_ablation_analysis(analysis_results["ablation"], output_dir)

        # 4. Normalization strategy comparison
        if "normalization" in analysis_results:
            self._plot_normalization_analysis(analysis_results["normalization"], output_dir)

    def _plot_weight_sweeps(self, sweep_data: Dict, output_dir: Path):
        """Plot weight sweep sensitivity analysis."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, (weight_name, data) in enumerate(sweep_data.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            # Plot mean SNQI vs weight value
            ax.plot(data["weight_values"], data["mean_snqi"], "b-", linewidth=2, label="Mean SNQI")
            ax.fill_between(
                data["weight_values"],
                np.array(data["mean_snqi"]) - np.array(data["std_snqi"]),
                np.array(data["mean_snqi"]) + np.array(data["std_snqi"]),
                alpha=0.3,
                color="blue",
            )

            # Plot ranking correlation
            ax2 = ax.twinx()
            ax2.plot(
                data["weight_values"],
                data["ranking_correlation"],
                "r--",
                linewidth=2,
                label="Ranking Corr",
            )

            ax.set_xlabel(weight_name.replace("_", " ").title())
            ax.set_ylabel("Mean SNQI", color="blue")
            ax2.set_ylabel("Ranking Correlation", color="red")
            ax.set_title(f"Sensitivity: {weight_name}")
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(len(sweep_data), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(output_dir / "weight_sweep_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_pairwise_analysis(self, pairwise_data: Dict, output_dir: Path):
        """Plot pairwise weight analysis heatmaps."""
        n_pairs = len(pairwise_data)
        fig, axes = plt.subplots(2, n_pairs, figsize=(5 * n_pairs, 10))

        if n_pairs == 1:
            axes = axes.reshape(2, 1)

        for i, (pair_name, data) in enumerate(pairwise_data.items()):
            # SNQI surface
            im1 = axes[0, i].imshow(data["snqi_surface"], cmap="viridis", aspect="auto")
            axes[0, i].set_title(f"SNQI Surface: {data['weight1_name']} vs {data['weight2_name']}")
            axes[0, i].set_xlabel(data["weight1_name"])
            axes[0, i].set_ylabel(data["weight2_name"])
            plt.colorbar(im1, ax=axes[0, i])

            # Stability surface
            im2 = axes[1, i].imshow(data["stability_surface"], cmap="plasma", aspect="auto")
            axes[1, i].set_title(
                f"Ranking Stability: {data['weight1_name']} vs {data['weight2_name']}"
            )
            axes[1, i].set_xlabel(data["weight1_name"])
            axes[1, i].set_ylabel(data["weight2_name"])
            plt.colorbar(im2, ax=axes[1, i])

        plt.tight_layout()
        plt.savefig(output_dir / "pairwise_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_ablation_analysis(self, ablation_data: Dict, output_dir: Path):
        """Plot ablation analysis results."""
        importance_ranking = ablation_data["importance_ranking"]
        weights = [item[0] for item in importance_ranking]
        scores = [item[1] for item in importance_ranking]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(weights)), scores, color="skyblue", edgecolor="navy", alpha=0.7)
        plt.xlabel("Weight Components")
        plt.ylabel("Importance Score")
        plt.title("SNQI Component Importance (Ablation Study)")
        plt.xticks(range(len(weights)), [w.replace("_", " ").title() for w in weights], rotation=45)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(output_dir / "ablation_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_normalization_analysis(self, norm_data: Dict, output_dir: Path):
        """Plot normalization strategy comparison."""
        strategies = list(norm_data.keys())
        metrics = ["mean_snqi", "std_snqi", "score_range", "ranking_correlation"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [norm_data[strategy][metric] for strategy in strategies]
            axes[i].bar(strategies, values, color="lightcoral", alpha=0.7)
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "normalization_strategy_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def load_episodes_data(file_path: Path) -> List[Dict]:
    """Load episode data from JSONL file."""
    episodes = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    logger.info(f"Loaded {len(episodes)} episodes from {file_path}")
    return episodes


def load_baseline_stats(file_path: Path) -> Dict[str, Dict[str, float]]:
    """Load baseline statistics from JSON file."""
    with open(file_path, "r") as f:
        stats = json.load(f)
    logger.info(f"Loaded baseline statistics from {file_path}")
    return stats


def load_weights(file_path: Path) -> Dict[str, float]:
    """Load weights from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Handle different formats (direct weights dict or nested structure)
    if "recommended" in data and "weights" in data["recommended"]:
        weights = data["recommended"]["weights"]
    elif "weights" in data:
        weights = data["weights"]
    else:
        weights = data

    logger.info(f"Loaded weights from {file_path}")
    return weights


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="SNQI Sensitivity Analysis")
    parser.add_argument(
        "--episodes", type=Path, required=True, help="Path to episode data JSONL file"
    )
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Path to baseline statistics JSON file"
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to weights JSON file")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for analysis results"
    )
    parser.add_argument(
        "--sweep-points", type=int, default=20, help="Number of points for weight sweep analysis"
    )
    parser.add_argument(
        "--pairwise-points",
        type=int,
        default=15,
        help="Number of points per dimension for pairwise analysis",
    )
    parser.add_argument(
        "--skip-visualizations", action="store_true", help="Skip generating visualization plots"
    )

    args = parser.parse_args()

    # Load data
    try:
        episodes = load_episodes_data(args.episodes)
        baseline_stats = load_baseline_stats(args.baseline)
        weights = load_weights(args.weights)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return 1

    # Initialize analyzer
    analyzer = SNQISensitivityAnalyzer(episodes, baseline_stats)

    # Run analyses
    results = {}

    logger.info("Running weight sweep analysis")
    results["weight_sweep"] = analyzer.weight_sweep_analysis(weights, n_points=args.sweep_points)

    logger.info("Running pairwise weight analysis")
    results["pairwise"] = analyzer.pairwise_weight_analysis(weights, n_points=args.pairwise_points)

    logger.info("Running ablation analysis")
    results["ablation"] = analyzer.ablation_analysis(weights)

    logger.info("Running normalization strategy analysis")
    results["normalization"] = analyzer.normalization_strategy_analysis(weights)

    # Save detailed results
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "sensitivity_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate summary report
    summary = {
        "analysis_summary": {
            "total_episodes": len(episodes),
            "base_weights": weights,
            "most_sensitive_weights": [],
            "least_sensitive_weights": [],
            "normalization_impact": {},
        }
    }

    # Extract key insights
    if "ablation" in results:
        importance_ranking = results["ablation"]["importance_ranking"]
        summary["analysis_summary"]["most_sensitive_weights"] = importance_ranking[:3]
        summary["analysis_summary"]["least_sensitive_weights"] = importance_ranking[-3:]

    if "normalization" in results:
        norm_results = results["normalization"]
        base_strategy = "median_p95"
        for strategy, data in norm_results.items():
            if strategy != base_strategy:
                corr_change = abs(
                    data["ranking_correlation"] - norm_results[base_strategy]["ranking_correlation"]
                )
                summary["analysis_summary"]["normalization_impact"][strategy] = {
                    "ranking_correlation_change": corr_change,
                    "significant_impact": corr_change > 0.1,
                }

    with open(args.output / "sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate visualizations
    if not args.skip_visualizations:
        try:
            logger.info("Generating visualizations")
            analyzer.generate_visualizations(results, args.output)
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

    logger.info(f"Sensitivity analysis completed. Results saved to {args.output}")

    # Print key findings
    print("\nSensitivity Analysis Summary:")
    print(f"Episodes analyzed: {len(episodes)}")

    if "ablation" in results:
        importance_ranking = results["ablation"]["importance_ranking"]
        print("\nMost influential weight components:")
        for i, (weight_name, importance) in enumerate(importance_ranking[:3], 1):
            print(f"  {i}. {weight_name}: {importance:.4f}")

    if "normalization" in results:
        print("\nNormalization strategy impact:")
        norm_results = results["normalization"]
        for strategy, data in norm_results.items():
            if strategy != "median_p95":
                corr = data["ranking_correlation"]
                print(f"  {strategy}: ranking correlation = {corr:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
