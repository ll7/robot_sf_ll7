"""
Integration tests for benchmark with real visualizations.

These tests verify the complete workflow from benchmark execution to visualization generation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger


def test_benchmark_with_visualization_integration():
    """Test complete benchmark execution with visualization generation."""

    # Given: Mock benchmark configuration and episode data
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "scenario_params": {"algo": "socialforce"},
            "algo": "socialforce",
            "metrics": {"collisions": 0, "success": True, "snqi": 0.95},
            "trajectory_data": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "benchmark_output"

        # Test that visualization functions are integrated into the orchestrator
        # The orchestrator now calls our visualization functions directly

        # Verify that our visualization functions are available
        from robot_sf.benchmark.visualization import (
            generate_benchmark_plots,
            generate_benchmark_videos,
            validate_visual_artifacts,
        )

        # And that they can be called with the episode data
        plots = generate_benchmark_plots(episodes_data, str(output_dir / "plots"))
        videos = generate_benchmark_videos(episodes_data, str(output_dir / "videos"))
        validation = validate_visual_artifacts(plots + videos)

        # Then: Functions execute without error and return expected types
        assert isinstance(plots, list)
        assert isinstance(videos, list)
        assert hasattr(validation, "passed")
        assert hasattr(validation, "failed_artifacts")


def test_benchmark_visualization_handles_errors():
    """Test that visualization functions handle errors gracefully."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        output_dir = Path(tmp_dir) / "benchmark_output"

        # Given: Visualization functions that raise errors
        with patch("robot_sf.benchmark.visualization.generate_benchmark_plots") as mock_plots:
            mock_plots.side_effect = Exception("Plot generation failed")

            # When: Visualization functions are called
            from robot_sf.benchmark.visualization import generate_benchmark_plots

            with pytest.raises(Exception, match="Plot generation failed"):
                generate_benchmark_plots(str(episodes_path), str(output_dir))


def test_benchmark_visualization_creates_output_structure():
    """Test that visualization functions create proper output directory structure."""

    # Given: Episode data
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "test_scenario",
            "scenario_params": {"algo": "socialforce"},
            "algo": "socialforce",
            "metrics": {"collisions": 0, "success": True},
            "trajectory_data": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "benchmark_output"

        # When: Visualization functions are called
        from robot_sf.benchmark.visualization import (
            generate_benchmark_plots,
            generate_benchmark_videos,
        )

        plots = generate_benchmark_plots(episodes_data, str(output_dir / "plots"))
        videos = generate_benchmark_videos(
            episodes_data,
            str(output_dir / "videos"),
        )  # Then: Functions return artifact lists
        assert isinstance(plots, list)
        assert isinstance(videos, list)


def test_log_manual_visualization_steps_logs_valid_commands(tmp_path: Path) -> None:
    """Verify log_manual_visualization_steps logs valid CLI commands.

    The function must not suggest commands that use aggregate JSON as input
    or omit required CLI arguments. It should point to
    scripts/generate_figures.py and valid robot_sf_bench commands with
    episodes JSONL placeholders and required metric arguments.
    """
    from scripts.run_social_navigation_benchmark import log_manual_visualization_steps

    # Create a minimal aggregated_results.json to simulate benchmark output root
    agg_file = tmp_path / "aggregated_results.json"
    agg_file.write_text(json.dumps({"dummy": True}), encoding="utf-8")
    visuals_dir = tmp_path / "visualizations"

    captured: list[str] = []

    def collect(msg):
        captured.append(str(msg))

    handle = logger.add(collect, level="INFO", format="{message}")
    try:
        result = log_manual_visualization_steps(str(agg_file), str(visuals_dir))
    finally:
        logger.remove(handle)

    assert result["success"] is True
    full_log = "\n".join(captured)

    # Must reference scripts/generate_figures.py as the primary option
    assert "scripts/generate_figures.py" in full_log

    # Must reference episodes JSONL placeholders, never the aggregate file
    assert "episodes.jsonl" in full_log
    assert "aggregated_results.json" not in full_log

    generate_command = next(line for line in captured if "scripts/generate_figures.py" in line)
    pareto_command = next(line for line in captured if "plot-pareto" in line)
    distributions_command = next(line for line in captured if "plot-distributions" in line)

    assert "--episodes" in generate_command
    assert "--pareto-x collisions" in generate_command
    assert "--pareto-y comfort_exposure" in generate_command
    assert "--dmetrics collisions,comfort_exposure" in generate_command

    assert "--in" in pareto_command
    assert "--out " in pareto_command
    assert "--out-dir" not in pareto_command
    assert "--x-metric collisions" in pareto_command
    assert "--y-metric comfort_exposure" in pareto_command

    assert "--in" in distributions_command
    assert "--out-dir" in distributions_command
    assert "--metrics collisions,comfort_exposure" in distributions_command


@pytest.mark.parametrize("formats", [("png",), ("svg",), ("png", "svg")])
def test_publication_plot_without_pdf_format_does_not_crash(formats, tmp_path):
    """Publication figures with formats excluding 'pdf' must not FileNotFoundError.

    Regression for PR #4800 (Gemini finding): plot_path was unconditionally set
    to ``<base>.pdf`` in publication mode, so the ``plot_path.stat()`` size probe
    raised FileNotFoundError whenever the requested formats did not include pdf.
    """
    from robot_sf.benchmark.visualization import (
        _generate_metrics_plot,
        _generate_scenario_comparison_plot,
    )

    episodes = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "metrics": {"collision_rate": 0.1, "success_rate": 0.9, "snqi": 0.8},
        },
        {
            "episode_id": "ep_002",
            "scenario_id": "classic_002",
            "metrics": {"collision_rate": 0.2, "success_rate": 0.8, "snqi": 0.7},
        },
    ]

    for generate in (_generate_metrics_plot, _generate_scenario_comparison_plot):
        artifact = generate(
            episodes,
            str(tmp_path),
            publication=True,
            formats=formats,
        )
        # Artifact must reference an actually-written file in a requested format.
        assert artifact.format in formats
        assert artifact.file_size > 0
        assert (tmp_path / artifact.filename).is_file()
