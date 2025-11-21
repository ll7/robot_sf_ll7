"""
Figure generator for research reporting (User Story 1)
Implements: learning_curve, sample_efficiency_bar, distribution_plots, _generate_caption
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib_backend(headless: bool = False) -> None:
    """Configure matplotlib backend for headless or interactive rendering."""
    if headless:
        matplotlib.use("Agg")
    # Apply dev_guide.md figure guidelines
    matplotlib.rcParams.update(
        {
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "font.size": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
        }
    )


def save_figure(fig: matplotlib.figure.Figure, base_path: Path, name: str) -> dict[str, Path]:
    """Save figure as both PDF (vector) and PNG (raster) for publication."""
    pdf_path = base_path / f"{name}.pdf"
    png_path = base_path / f"{name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    return {"pdf": pdf_path, "png": png_path}


def _generate_caption(figure_type: str, metadata: dict[str, Any]) -> str:
    """Generate descriptive caption for a figure."""
    captions = {
        "learning_curve": f"Learning curve showing episode rewards over training timesteps. Aggregated across {metadata.get('n_seeds', 'N')} seeds with 95% confidence intervals.",
        "sample_efficiency": f"Sample efficiency comparison: PPO timesteps to convergence for baseline vs. pretrained policies. Bars show mean across {metadata.get('n_seeds', 'N')} seeds; error bars indicate 95% bootstrap CI.",
        "success_distribution": f"Distribution of success rates across {metadata.get('n_seeds', 'N')} seeds for baseline and pretrained conditions.",
        "collision_distribution": f"Distribution of collision rates across {metadata.get('n_seeds', 'N')} seeds for baseline and pretrained conditions.",
        "improvement_summary": f"Summary of key performance improvements (%) with 95% confidence intervals across {metadata.get('n_seeds', 'N')} seeds.",
    }
    return captions.get(figure_type, f"Figure: {figure_type}")


def plot_learning_curve(
    timesteps: list[float],
    rewards_baseline: list[list[float]],
    rewards_pretrained: list[list[float]],
    output_dir: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generate learning curve plot comparing baseline vs pretrained.
    Args:
        timesteps: Common x-axis values (training steps)
        rewards_baseline: List of reward trajectories (one per seed) for baseline
        rewards_pretrained: List of reward trajectories (one per seed) for pretrained
        output_dir: Directory to save figure
        metadata: Optional metadata for caption generation
    Returns:
        Dict with paths and caption
    """
    configure_matplotlib_backend(headless=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Compute mean and CI for baseline
    rewards_baseline_array = np.array(rewards_baseline)
    mean_baseline = np.mean(rewards_baseline_array, axis=0)
    std_baseline = np.std(rewards_baseline_array, axis=0, ddof=1)

    # Compute mean and CI for pretrained
    rewards_pretrained_array = np.array(rewards_pretrained)
    mean_pretrained = np.mean(rewards_pretrained_array, axis=0)
    std_pretrained = np.std(rewards_pretrained_array, axis=0, ddof=1)

    ax.plot(timesteps, mean_baseline, label="Baseline", color="C0")
    ax.fill_between(
        timesteps,
        mean_baseline - 1.96 * std_baseline,
        mean_baseline + 1.96 * std_baseline,
        alpha=0.3,
        color="C0",
    )

    ax.plot(timesteps, mean_pretrained, label="Pretrained", color="C1")
    ax.fill_between(
        timesteps,
        mean_pretrained - 1.96 * std_pretrained,
        mean_pretrained + 1.96 * std_pretrained,
        alpha=0.3,
        color="C1",
    )

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    paths = save_figure(fig, output_dir, "fig-learning-curve")
    plt.close(fig)

    metadata = metadata or {}
    metadata["n_seeds"] = len(rewards_baseline)
    caption = _generate_caption("learning_curve", metadata)

    return {"paths": paths, "caption": caption, "figure_type": "learning_curve"}


def plot_sample_efficiency(
    baseline_timesteps: list[float],
    pretrained_timesteps: list[float],
    output_dir: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generate bar chart comparing sample efficiency (timesteps to convergence).
    """
    configure_matplotlib_backend(headless=True)
    fig, ax = plt.subplots(figsize=(5, 4))

    mean_baseline = np.mean(baseline_timesteps)
    mean_pretrained = np.mean(pretrained_timesteps)

    # Bootstrap CI
    from robot_sf.research.aggregation import bootstrap_ci

    ci_low_b, ci_high_b = bootstrap_ci(baseline_timesteps) or (mean_baseline, mean_baseline)
    ci_low_p, ci_high_p = bootstrap_ci(pretrained_timesteps) or (mean_pretrained, mean_pretrained)

    labels = ["Baseline", "Pretrained"]
    means = [mean_baseline, mean_pretrained]
    errors = [
        [mean_baseline - ci_low_b, mean_pretrained - ci_low_p],
        [ci_high_b - mean_baseline, ci_high_p - mean_pretrained],
    ]

    ax.bar(labels, means, yerr=errors, capsize=5, color=["C0", "C1"], alpha=0.7)
    ax.set_ylabel("Timesteps to Convergence")
    ax.set_title("Sample Efficiency Comparison")
    ax.grid(axis="y", alpha=0.3)

    paths = save_figure(fig, output_dir, "fig-sample-efficiency")
    plt.close(fig)

    metadata = metadata or {}
    metadata["n_seeds"] = len(baseline_timesteps)
    caption = _generate_caption("sample_efficiency", metadata)

    return {"paths": paths, "caption": caption, "figure_type": "sample_efficiency"}


def plot_distributions(
    baseline_values: list[float],
    pretrained_values: list[float],
    metric_name: str,
    output_dir: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generate distribution comparison plot (histograms/violin) for a metric.
    """
    configure_matplotlib_backend(headless=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(baseline_values, bins=10, alpha=0.5, label="Baseline", color="C0")
    ax.hist(pretrained_values, bins=10, alpha=0.5, label="Pretrained", color="C1")

    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    figure_type_map = {
        "success_rate": "success_distribution",
        "collision_rate": "collision_distribution",
    }
    figure_type = figure_type_map.get(metric_name, f"{metric_name}_distribution")

    paths = save_figure(fig, output_dir, f"fig-{metric_name}-distribution")
    plt.close(fig)

    metadata = metadata or {}
    metadata["n_seeds"] = len(baseline_values)
    caption = _generate_caption(figure_type, metadata)

    return {"paths": paths, "caption": caption, "figure_type": figure_type}
