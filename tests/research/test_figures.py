"""Unit tests for figure generation module."""

import pytest

from robot_sf.research.figures import (
    configure_matplotlib_backend,
    plot_distributions,
    plot_effect_sizes,
    plot_improvement_summary,
    plot_learning_curve,
    plot_sample_efficiency,
    plot_sensitivity,
    save_figure,
)


@pytest.fixture
def output_dir(tmp_path):  # fixture reused across tests
    """Create temporary output directory."""
    output_path = tmp_path / "figures"
    output_path.mkdir(exist_ok=True)
    return output_path


def test_configure_matplotlib_backend():
    """Test matplotlib backend configuration."""
    configure_matplotlib_backend()
    import matplotlib

    assert matplotlib.get_backend() == "Agg"


def test_save_figure_dual_export(output_dir):
    """Test save_figure creates both PDF and PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    paths = save_figure(fig, output_dir, "test_figure")

    assert "pdf" in paths
    assert "png" in paths
    assert paths["pdf"].exists()
    assert paths["png"].exists()
    assert paths["pdf"].name == "test_figure.pdf"
    assert paths["png"].name == "test_figure.png"

    plt.close(fig)


def test_plot_effect_sizes(output_dir):
    """Test effect size summary plotting."""
    effect_sizes = {"timesteps_to_convergence": 0.75, "success_rate": 0.35}
    result = plot_effect_sizes(effect_sizes, output_dir)
    # PDF/PNG should exist
    assert result["paths"]["pdf"].exists()
    assert result["paths"]["png"].exists()
    assert result["figure_type"] == "effect_sizes"


def test_plot_improvement_summary(output_dir):
    """Test improvement summary plotting."""
    baseline = {"timesteps_to_convergence": 500000, "success_rate": 0.65}
    pretrained = {"timesteps_to_convergence": 300000, "success_rate": 0.78}
    result = plot_improvement_summary(baseline, pretrained, output_dir)
    assert result["paths"]["pdf"].exists()
    assert result["paths"]["png"].exists()
    assert result["figure_type"] == "improvement_summary"


def test_learning_curve_plot(output_dir):
    """Test learning curve plot generation."""
    timesteps = [0, 100, 200, 300]
    # Two seeds baseline & pretrained
    rewards_baseline = [[1, 2, 3, 4], [1.1, 1.9, 3.1, 3.9]]
    rewards_pretrained = [[1.5, 2.5, 3.5, 4.5], [1.4, 2.6, 3.6, 4.4]]
    result = plot_learning_curve(timesteps, rewards_baseline, rewards_pretrained, output_dir)
    assert result["paths"]["pdf"].exists()
    assert result["paths"]["png"].exists()
    assert result["figure_type"] == "learning_curve"


def test_sample_efficiency_plot(output_dir):
    """Test sample efficiency plot generation."""
    baseline_timesteps = [500000, 520000, 510000]
    pretrained_timesteps = [300000, 310000, 290000]
    result = plot_sample_efficiency(baseline_timesteps, pretrained_timesteps, output_dir)
    assert result["paths"]["pdf"].exists()
    assert result["paths"]["png"].exists()
    assert result["figure_type"] == "sample_efficiency"


def test_distribution_plots(output_dir):
    """Test distribution plot generation for success_rate."""
    baseline_values = [0.6, 0.7, 0.65, 0.62]
    pretrained_values = [0.75, 0.78, 0.8, 0.77]
    result = plot_distributions(baseline_values, pretrained_values, "success_rate", output_dir)
    assert result["paths"]["pdf"].exists()
    assert result["paths"]["png"].exists()
    assert "success" in result["figure_type"]


def test_sensitivity_analysis(output_dir):
    """Test sensitivity plot generation (T071)."""
    variants = [
        {"variant_id": "bc5_ds100", "bc_epochs": 5, "dataset_size": 100, "improvement_pct": 10.0},
        {"variant_id": "bc10_ds100", "bc_epochs": 10, "dataset_size": 100, "improvement_pct": 22.0},
        {"variant_id": "bc20_ds100", "bc_epochs": 20, "dataset_size": 100, "improvement_pct": 38.5},
    ]
    result = plot_sensitivity(variants, "bc_epochs", output_dir)
    assert result["paths"]["pdf"].exists()
    assert result["paths"]["png"].exists()
    assert result["figure_type"] == "sensitivity"
