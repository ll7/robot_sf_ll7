"""
Figure generator for research reporting (User Story 1)
Implements: learning_curve, sample_efficiency_bar, distribution_plots, _generate_caption
"""

import importlib
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from robot_sf.research.aggregation import bootstrap_ci


def _get_pyplot():
    """TODO docstring. Document this function.

    Returns:
        matplotlib.pyplot module.
    """
    return importlib.import_module("matplotlib.pyplot")


def configure_matplotlib_backend(headless: bool = True) -> None:
    """Configure matplotlib backend for headless or interactive rendering.

    Default is headless (Agg) to align with CI/test expectations.
    Pass headless=False only for local interactive exploration.
    """
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


def save_figure(fig, base_path: Path, name: str) -> dict[str, Path]:
    """Save figure as both PDF (vector) and PNG (raster) for publication.

    Args:
        fig: Matplotlib figure object to save
        base_path: Directory where figure files will be saved (created if missing)
        name: Base filename without extension (e.g., "fig-learning-curve")

    Returns:
        Dictionary with 'pdf' and 'png' keys mapping to Path objects

    Note:
        PDF uses fonttype 42 for LaTeX compatibility.
        PNG exports at 300 DPI for print quality.
        Both formats use bbox_inches='tight' to minimize whitespace.
    """
    pdf_path = base_path / f"{name}.pdf"
    png_path = base_path / f"{name}.png"
    base_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    return {"pdf": pdf_path, "png": png_path}


def _generate_caption(figure_type: str, metadata: dict[str, Any]) -> str:
    """Generate descriptive caption for a figure matching publication standards.

    Args:
        figure_type: Type identifier (e.g., 'learning_curve', 'sample_efficiency')
        metadata: Dictionary containing caption metadata (e.g., n_seeds)

    Returns:
        Publication-ready caption string with sample size and confidence level noted

    Note:
        Captions follow dev_guide.md figure guidelines:
            - Descriptive title
            - Axis labels explained
            - Sample size noted (e.g., "n=3 seeds")
            - Confidence level specified where applicable
    """
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
    metadata: dict[str, Any] | None = None,
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
    plt = _get_pyplot()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Compute mean and CI for baseline
    rewards_baseline_array = np.array(rewards_baseline)
    baseline_seed_count = rewards_baseline_array.shape[0]
    mean_baseline = np.mean(rewards_baseline_array, axis=0)
    std_baseline = (
        np.std(rewards_baseline_array, axis=0, ddof=1)
        if baseline_seed_count > 1
        else np.zeros_like(mean_baseline)
    )

    # Compute mean and CI for pretrained
    rewards_pretrained_array = np.array(rewards_pretrained)
    pretrained_seed_count = rewards_pretrained_array.shape[0]
    mean_pretrained = np.mean(rewards_pretrained_array, axis=0)
    std_pretrained = (
        np.std(rewards_pretrained_array, axis=0, ddof=1)
        if pretrained_seed_count > 1
        else np.zeros_like(mean_pretrained)
    )

    ax.plot(timesteps, mean_baseline, label="Baseline", color="C0")
    if baseline_seed_count > 1:
        ax.fill_between(
            timesteps,
            mean_baseline - 1.96 * std_baseline,
            mean_baseline + 1.96 * std_baseline,
            alpha=0.3,
            color="C0",
        )

    ax.plot(timesteps, mean_pretrained, label="Pretrained", color="C1")
    if pretrained_seed_count > 1:
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
    metadata["n_seeds"] = baseline_seed_count
    metadata["n_seeds_pretrained"] = pretrained_seed_count
    caption = _generate_caption("learning_curve", metadata)

    return {"paths": paths, "caption": caption, "figure_type": "learning_curve"}


def plot_sample_efficiency(
    baseline_timesteps: list[float],
    pretrained_timesteps: list[float],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate bar chart comparing sample efficiency (timesteps to convergence).

    Args:
        baseline_timesteps: Per-seed timesteps to convergence for baseline policy
        pretrained_timesteps: Per-seed timesteps to convergence for pretrained policy
        output_dir: Directory to save output figures
        metadata: Optional metadata dict for caption generation

    Returns:
        Dictionary containing:
            - paths: Dict with 'pdf' and 'png' Path objects
            - caption: Publication-ready caption string
            - figure_type: Always 'sample_efficiency'

    Note:
        Error bars show 95% bootstrap confidence intervals.
        Bars are colored C0 (baseline) and C1 (pretrained) from matplotlib cycle.
    """
    configure_matplotlib_backend(headless=True)
    plt = _get_pyplot()

    fig, ax = plt.subplots(figsize=(5, 4))

    mean_baseline = np.mean(baseline_timesteps)
    mean_pretrained = np.mean(pretrained_timesteps)

    # Bootstrap CI
    ci_low_b, ci_high_b = bootstrap_ci(baseline_timesteps) or (
        mean_baseline,
        mean_baseline,
    )
    ci_low_p, ci_high_p = bootstrap_ci(pretrained_timesteps) or (
        mean_pretrained,
        mean_pretrained,
    )
    # Normalize None outputs to means
    ci_low_b = ci_low_b if ci_low_b is not None else mean_baseline
    ci_high_b = ci_high_b if ci_high_b is not None else mean_baseline
    ci_low_p = ci_low_p if ci_low_p is not None else mean_pretrained
    ci_high_p = ci_high_p if ci_high_p is not None else mean_pretrained

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
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate distribution comparison plot (histograms/violin) for a metric.

    Returns:
        Dict with paths and caption for the generated figure.
    """
    configure_matplotlib_backend(headless=True)
    plt = _get_pyplot()

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


def plot_effect_sizes(
    effect_sizes: dict[str, float],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plot effect sizes (Cohen's d) as horizontal bar chart.

    Args:
        effect_sizes: Mapping metric_name -> effect size (Cohen's d)
        output_dir: Directory to save figure
        metadata: Optional metadata (n_seeds etc.)
    Returns:
        Dict with paths, caption, figure_type
    """
    if not effect_sizes:
        return {
            "paths": {},
            "caption": "No effect sizes available",
            "figure_type": "effect_sizes",
        }
    configure_matplotlib_backend(headless=True)
    plt = _get_pyplot()

    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = list(effect_sizes.keys())
    values = [effect_sizes[m] for m in metrics]
    y_pos = np.arange(len(metrics))
    ax.barh(y_pos, values, color="C2", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_xlabel("Effect Size (Cohen's d)")
    ax.set_title("Effect Size Summary")
    ax.axvline(0.0, color="black", linewidth=0.8)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    paths = save_figure(fig, output_dir, "fig-effect-sizes")
    plt.close(fig)
    metadata = metadata or {}
    caption = _generate_caption("improvement_summary", metadata)
    return {"paths": paths, "caption": caption, "figure_type": "effect_sizes"}


def plot_improvement_summary(
    baseline_metrics: dict[str, float],
    pretrained_metrics: dict[str, float],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plot percentage improvements for selected metrics.

    Improvement is defined as ((baseline - pretrained)/baseline)*100 for metrics where
    lower is better (e.g., timesteps_to_convergence) and (pretrained - baseline)/baseline*100
    where higher is better (e.g., success_rate). Heuristic: metrics containing 'timesteps' or
    'collision' treated as lower-is-better, otherwise higher-is-better.

    Returns:
        Dict with paths and caption for the improvement summary figure.
    """
    if not baseline_metrics or not pretrained_metrics:
        return {
            "paths": {},
            "caption": "No metrics available",
            "figure_type": "improvement_summary",
        }
    configure_matplotlib_backend(headless=True)
    plt = _get_pyplot()

    fig, ax = plt.subplots(figsize=(6, 4))
    improvements: list[float] = []
    labels: list[str] = []
    for name, base_val in baseline_metrics.items():
        if name not in pretrained_metrics:
            continue
        treat_val = pretrained_metrics[name]
        if base_val is None or treat_val is None:
            continue
        lower_is_better = any(k in name for k in ["timesteps", "collision"])
        if base_val == 0:
            continue
        if lower_is_better:
            imp = 100.0 * (base_val - treat_val) / base_val
        else:
            imp = 100.0 * (treat_val - base_val) / base_val
        improvements.append(imp)
        labels.append(name.replace("_", " ").title())
    if not improvements:
        return {
            "paths": {},
            "caption": "No comparable metric pairs",
            "figure_type": "improvement_summary",
        }
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, improvements, color="C3", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Improvement Summary")
    for i, v in enumerate(improvements):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    paths = save_figure(fig, output_dir, "fig-improvement-summary")
    plt.close(fig)
    metadata = metadata or {}
    caption = _generate_caption("improvement_summary", metadata)
    return {"paths": paths, "caption": caption, "figure_type": "improvement_summary"}


def plot_sensitivity(
    variants: list[dict[str, Any]],
    param_name: str,
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plot sensitivity of a single parameter against improvement percentage.

    variants: list of variant dicts having keys param_name and improvement_pct.

    Returns:
        Dict with paths and caption for the sensitivity analysis figure.
    """
    if not variants:
        return {"paths": {}, "caption": "No variants", "figure_type": "sensitivity"}
    configure_matplotlib_backend(headless=True)
    plt = _get_pyplot()

    fig, ax = plt.subplots(figsize=(6, 4))
    xs: list[float] = []
    ys: list[float] = []
    for v in variants:
        if v.get("improvement_pct") is None:
            continue
        if param_name not in v:
            continue
        xs.append(float(v[param_name]))
        ys.append(float(v["improvement_pct"]))
    if not xs:
        return {
            "paths": {},
            "caption": "No complete variants",
            "figure_type": "sensitivity",
        }
    ax.plot(xs, ys, marker="o", color="C4")
    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel("Improvement (%)")
    ax.set_title(f"Sensitivity: {param_name} vs Improvement")
    for x, y in zip(xs, ys, strict=False):
        ax.text(x, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3)
    paths = save_figure(fig, output_dir, f"fig-sensitivity-{param_name}")
    plt.close(fig)
    caption = _generate_caption("improvement_summary", metadata or {})
    return {"paths": paths, "caption": caption, "figure_type": "sensitivity"}
