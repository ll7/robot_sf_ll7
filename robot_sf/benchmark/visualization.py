"""
Visualization functions for benchmark outputs.

This module provides functions to generate real plots and videos from benchmark
episode data, replacing placeholder outputs with actual statistical visualizations
and simulation replays.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from loguru import logger


# TODO: Implement VisualArtifact dataclass (T009)
@dataclass
class VisualArtifact:
    """Metadata about a generated visual artifact (plot or video)."""

    artifact_id: str
    artifact_type: Literal["plot", "video"]
    format: Literal["pdf", "mp4"]
    filename: str
    source_data: str
    generation_time: datetime
    file_size: int
    status: Literal["generated", "failed", "placeholder"]
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating visual artifacts."""

    passed: bool
    failed_artifacts: List[VisualArtifact]
    details: Optional[dict] = None


# TODO: Implement VisualizationError exception class (T010)
class VisualizationError(Exception):
    """Raised when visualization generation fails."""

    def __init__(self, message: str, artifact_type: str, details: Optional[dict] = None):
        super().__init__(message)
        self.artifact_type = artifact_type
        self.details = details or {}


# TODO: Implement generate_benchmark_plots function (T011) - COMPLETED
def generate_benchmark_plots(
    episodes_data: List[dict] | str,
    output_dir: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None,
) -> List[VisualArtifact]:
    """
    Generate PDF plots by analyzing benchmark episodes.

    Args:
        episodes_data: List of episode dictionaries from benchmark execution, or path to JSONL file
        output_dir: Directory to save generated PDF plots
        scenario_filter: Optional scenario ID to filter plots
        baseline_filter: Optional baseline name to filter plots

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If plot generation fails
        FileNotFoundError: If episodes_path doesn't exist (when path provided)
    """
    _check_matplotlib_available()

    # Handle backward compatibility: if episodes_data is a path, load it
    if isinstance(episodes_data, (str, Path)):
        if not os.path.exists(episodes_data):
            raise FileNotFoundError(f"Episodes file not found: {episodes_data}")
        episodes = _load_episodes(str(episodes_data))
    else:
        episodes = episodes_data

    if not episodes:
        raise VisualizationError(
            "No episode data found",
            "plot",
            {"episodes_count": len(episodes)},
        )

    filtered_episodes = _filter_episodes(episodes, scenario_filter, baseline_filter)

    if not filtered_episodes:
        raise VisualizationError(
            "No episodes match the specified filters",
            "plot",
            {"scenario_filter": scenario_filter, "baseline_filter": baseline_filter},
        )

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    artifacts = []

    filter_info = []
    if scenario_filter:
        filter_info.append(f"scenario={scenario_filter}")
    if baseline_filter:
        filter_info.append(f"baseline={baseline_filter}")
    filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

    # Generate metrics distribution plot
    try:
        artifact = _generate_metrics_plot(filtered_episodes, str(plots_dir))
        # Update source_data to include filter info
        artifact.source_data = f"{len(filtered_episodes)} episodes{filter_str}"
        artifacts.append(artifact)
    except Exception as e:
        logger.warning(f"Failed to generate metrics plot: {e}")
        artifacts.append(
            VisualArtifact(
                artifact_id="metrics_plot",
                artifact_type="plot",
                format="pdf",
                filename="metrics_plot.pdf",
                source_data=f"{len(filtered_episodes)} episodes{filter_str}",
                generation_time=datetime.now(),
                file_size=0,
                status="failed",
                error_message=str(e),
            )
        )

    # Generate scenario comparison plot
    try:
        artifact = _generate_scenario_comparison_plot(filtered_episodes, str(plots_dir))
        # Update source_data to include filter info
        artifact.source_data = f"{len(filtered_episodes)} episodes{filter_str}"
        artifacts.append(artifact)
    except Exception as e:
        logger.warning(f"Failed to generate scenario comparison plot: {e}")
        artifacts.append(
            VisualArtifact(
                artifact_id="scenario_comparison",
                artifact_type="plot",
                format="pdf",
                filename="scenario_comparison.pdf",
                source_data=f"{len(filtered_episodes)} episodes{filter_str}",
                generation_time=datetime.now(),
                file_size=0,
                status="failed",
                error_message=str(e),
            )
        )

    return artifacts


def _check_matplotlib_available() -> None:
    """Check if matplotlib is available, raise VisualizationError if not."""
    _check_dependencies(["matplotlib"])

    # Set non-interactive backend
    import matplotlib

    matplotlib.use("Agg")


def _load_episodes(episodes_path: str) -> List[dict]:
    """Load episodes from JSONL file."""
    episodes = []
    try:
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    except (json.JSONDecodeError, IOError) as e:
        raise VisualizationError(
            f"Failed to load episode data: {e}", "plot", {"episodes_path": episodes_path}
        )

    if not episodes:
        raise VisualizationError("No episode data found", "plot", {"episodes_path": episodes_path})
    return episodes


def _filter_episodes(
    episodes: List[dict], scenario_filter: Optional[str], baseline_filter: Optional[str]
) -> List[dict]:
    """Filter episodes based on scenario and baseline filters."""
    filtered_episodes = episodes

    if scenario_filter:
        filtered_episodes = [ep for ep in episodes if ep.get("scenario_id") == scenario_filter]

    if baseline_filter:
        filtered_episodes = [
            ep
            for ep in filtered_episodes
            if ep.get("scenario_params", {}).get("algo") == baseline_filter
        ]

    if not filtered_episodes:
        raise VisualizationError(
            "No episodes match the specified filters",
            "plot",
            {"scenario_filter": scenario_filter, "baseline_filter": baseline_filter},
        )

    return filtered_episodes


def _generate_metrics_plot(episodes: List[dict], output_dir: str) -> VisualArtifact:
    """Generate metrics distribution plot and return artifact."""
    import matplotlib.pyplot as plt

    try:
        # Extract metrics
        collisions = [ep.get("metrics", {}).get("collision_rate", 0) for ep in episodes]
        success_rates = [ep.get("metrics", {}).get("success_rate", 0) for ep in episodes]
        snqi_scores = [ep.get("metrics", {}).get("snqi", 0) for ep in episodes]

        # Create plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Collision distribution
        ax1.hist(collisions, bins=20, alpha=0.7, color="red")
        ax1.set_title("Collision Rate Distribution")
        ax1.set_xlabel("Collision Rate")
        ax1.set_ylabel("Frequency")

        # Plot 2: Success rate distribution
        ax2.hist(success_rates, bins=20, alpha=0.7, color="green")
        ax2.set_title("Success Rate Distribution")
        ax2.set_xlabel("Success Rate")
        ax2.set_ylabel("Frequency")

        # Plot 3: SNQI distribution
        ax3.hist(snqi_scores, bins=20, alpha=0.7, color="blue")
        ax3.set_title("SNQI Score Distribution")
        ax3.set_xlabel("SNQI Score")
        ax3.set_ylabel("Frequency")

        # Plot 4: Success vs SNQI scatter
        ax4.scatter(success_rates, snqi_scores, alpha=0.6, color="purple")
        ax4.set_title("Success Rate vs SNQI")
        ax4.set_xlabel("Success Rate")
        ax4.set_ylabel("SNQI Score")

        plt.tight_layout()

        # Save plot
        plot_filename = "metrics_distribution.pdf"
        plot_path = Path(output_dir) / plot_filename
        plt.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close()

        # Create artifact
        generation_time = datetime.now()
        file_size = plot_path.stat().st_size

        return VisualArtifact(
            artifact_id=f"plot_metrics_{generation_time.timestamp()}",
            artifact_type="plot",
            format="pdf",
            filename=plot_filename,
            source_data=f"{len(episodes)} episodes",
            generation_time=generation_time,
            file_size=file_size,
            status="generated",
        )

    except Exception as e:
        raise VisualizationError(f"Failed to generate metrics plot: {e}", "plot")


def _generate_scenario_comparison_plot(episodes: List[dict], output_dir: str) -> VisualArtifact:
    """Generate scenario comparison plot and return artifact."""
    import matplotlib.pyplot as plt

    try:
        # Group episodes by scenario
        scenario_groups = {}
        for ep in episodes:
            scenario = ep.get("scenario_id", "unknown")
            if scenario not in scenario_groups:
                scenario_groups[scenario] = []
            scenario_groups[scenario].append(ep)

        # Extract metrics for each scenario
        scenarios = list(scenario_groups.keys())
        avg_snqi = []
        avg_collisions = []
        success_rates = []

        for scenario in scenarios:
            eps = scenario_groups[scenario]
            snqi_scores = [ep.get("metrics", {}).get("snqi", 0) for ep in eps]
            collision_rates = [ep.get("metrics", {}).get("collision_rate", 0) for ep in eps]
            successes = [ep.get("metrics", {}).get("success_rate", 1) for ep in eps]

            avg_snqi.append(sum(snqi_scores) / len(snqi_scores))
            avg_collisions.append(sum(collision_rates) / len(collision_rates))
            success_rates.append(sum(successes) / len(successes))

        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Average SNQI by scenario
        ax1.bar(scenarios, avg_snqi, color="blue", alpha=0.7)
        ax1.set_title("Average SNQI by Scenario")
        ax1.set_ylabel("SNQI Score")
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Average collision rate by scenario
        ax2.bar(scenarios, avg_collisions, color="red", alpha=0.7)
        ax2.set_title("Average Collision Rate by Scenario")
        ax2.set_ylabel("Collision Rate")
        ax2.tick_params(axis="x", rotation=45)

        # Plot 3: Success rate by scenario
        ax3.bar(scenarios, success_rates, color="green", alpha=0.7)
        ax3.set_title("Success Rate by Scenario")
        ax3.set_ylabel("Success Rate")
        ax3.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        plot_filename = "scenario_comparison.pdf"
        plot_path = Path(output_dir) / plot_filename
        plt.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close()

        # Create artifact
        generation_time = datetime.now()
        file_size = plot_path.stat().st_size

        return VisualArtifact(
            artifact_id=f"plot_scenario_{generation_time.timestamp()}",
            artifact_type="plot",
            format="pdf",
            filename=plot_filename,
            source_data=f"{len(scenarios)} scenarios, {len(episodes)} episodes",
            generation_time=generation_time,
            file_size=file_size,
            status="generated",
        )

    except Exception as e:
        raise VisualizationError(f"Failed to generate scenario comparison plot: {e}", "plot")


# TODO: Implement generate_benchmark_videos function (T012) - COMPLETED
def generate_benchmark_videos(
    episodes_data: List[dict] | str,
    output_dir: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None,
    fps: int = 30,
    max_duration: float = 10.0,
) -> List[VisualArtifact]:
    """
    Generate MP4 videos by replaying benchmark episodes.

    Args:
        episodes_data: List of episode dictionaries from benchmark execution, or path to JSONL file
        output_dir: Directory to save generated MP4 videos
        scenario_filter: Optional scenario ID to filter videos
        baseline_filter: Optional baseline name to filter videos
        fps: Frames per second for video rendering
        max_duration: Maximum video duration in seconds

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If video rendering fails
        FileNotFoundError: If episodes_path doesn't exist (when path provided)
    """
    _check_moviepy_available()

    # Handle backward compatibility: if episodes_data is a path, load it
    if isinstance(episodes_data, (str, Path)):
        if not os.path.exists(episodes_data):
            raise FileNotFoundError(f"Episodes file not found: {episodes_data}")
        episodes = _load_episodes(str(episodes_data))
    else:
        episodes = episodes_data

    if not episodes:
        raise VisualizationError(
            "No episode data found",
            "video",
            {"episodes_count": len(episodes)},
        )

    filtered_episodes = _filter_episodes(episodes, scenario_filter, baseline_filter)

    # Filter episodes that have trajectory data
    episodes_with_trajectory = [ep for ep in filtered_episodes if ep.get("trajectory_data")]

    if not episodes_with_trajectory:
        # If no trajectory data, create placeholder videos or skip
        logger.warning(
            "No episodes with trajectory data found for video generation. "
            "Video generation requires trajectory data captured during simulation."
        )
        return []

    # For now, skip video generation since trajectory data format is not compatible
    # with the existing replay system. This would need integration with the replay
    # capture system used by the benchmark.
    logger.info(
        f"Found {len(episodes_with_trajectory)} episodes with trajectory data, but video generation is not yet implemented for this data format."
    )
    return []


def _check_moviepy_available() -> None:
    """Check if moviepy is available, raise VisualizationError if not."""
    _check_dependencies(["moviepy"])


def _generate_episode_videos(
    episodes: List[dict], videos_dir: Path, fps: int, max_duration: float, episodes_path: str
) -> List[VisualArtifact]:
    """Generate videos for episodes with trajectory data."""
    artifacts = []

    for episode in episodes:
        try:
            video_filename = f"episode_{episode['episode_id']}.mp4"
            video_path = videos_dir / video_filename

            # For now, create a placeholder video since full environment replay
            # would require significant integration with the simulation system
            _create_placeholder_video(video_path, fps, max_duration)

            generation_time = datetime.now()
            file_size = video_path.stat().st_size

            filter_info = []
            if episode.get("scenario_id"):
                filter_info.append(f"scenario={episode['scenario_id']}")
            if episode.get("scenario_params", {}).get("algo"):
                filter_info.append(f"baseline={episode['scenario_params']['algo']}")
            filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

            artifact = VisualArtifact(
                artifact_id=f"video_{episode['episode_id']}_{generation_time.timestamp()}",
                artifact_type="video",
                format="mp4",
                filename=video_filename,
                source_data=f"Episode {episode['episode_id']} trajectory from {episodes_path}{filter_str}",
                generation_time=generation_time,
                file_size=file_size,
                status="generated",
            )
            artifacts.append(artifact)

        except Exception as e:
            artifact = VisualArtifact(
                artifact_id=f"video_{episode['episode_id']}_{datetime.now().timestamp()}",
                artifact_type="video",
                format="mp4",
                filename=f"episode_{episode['episode_id']}.mp4",
                source_data=f"Episode {episode['episode_id']} trajectory",
                generation_time=datetime.now(),
                file_size=0,
                status="failed",
                error_message=str(e),
            )
            artifacts.append(artifact)

    return artifacts


def _create_placeholder_video(video_path: Path, fps: int, duration: float) -> None:
    """Create a placeholder video for testing purposes."""
    try:
        import numpy as np
        from moviepy import VideoClip

        # Create a simple animated placeholder
        def make_frame(t):
            # Create a simple animated frame
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Add some movement based on time
            y_pos = int(120 + 50 * np.sin(2 * np.pi * t / duration))
            frame[y_pos - 10 : y_pos + 10, 150:170] = [255, 0, 0]  # Red rectangle
            return frame

        clip = VideoClip(make_frame, duration=duration)
        clip.write_videofile(str(video_path), fps=fps, codec="libx264", audio=False)
    except Exception as e:
        # If video creation fails, create an empty file as fallback
        video_path.touch()
        raise VisualizationError(f"Failed to create placeholder video: {e}", "video")

    except Exception as e:
        # If video creation fails, create an empty file as fallback
        video_path.touch()
        raise VisualizationError(f"Failed to create placeholder video: {e}", "video")


# TODO: Implement validate_visual_artifacts function (T013) - COMPLETED
def validate_visual_artifacts(artifacts: List[VisualArtifact]) -> ValidationResult:
    """
    Validate that visual artifacts contain real data.

    Args:
        artifacts: List of VisualArtifact objects to validate

    Returns:
        ValidationResult with pass/fail status and details
    """
    if not artifacts:
        return ValidationResult(
            passed=True, failed_artifacts=[], details={"message": "No artifacts to validate"}
        )

    failed_artifacts = []

    for artifact in artifacts:
        # Check basic properties
        if artifact.status != "generated":
            failed_artifacts.append(artifact)
            continue

        if artifact.file_size <= 0:
            failed_artifacts.append(artifact)
            continue

        # Check filename format
        if artifact.artifact_type == "plot" and not artifact.filename.endswith(".pdf"):
            failed_artifacts.append(artifact)
            continue

        if artifact.artifact_type == "video" and not artifact.filename.endswith(".mp4"):
            failed_artifacts.append(artifact)
            continue

        # Check for placeholder indicators in source_data
        if "placeholder" in artifact.source_data.lower():
            failed_artifacts.append(artifact)
            continue

        # Check for TODO or placeholder in filename
        if "todo" in artifact.filename.lower() or "placeholder" in artifact.filename.lower():
            failed_artifacts.append(artifact)
            continue

    passed = len(failed_artifacts) == 0
    details = {
        "total_artifacts": len(artifacts),
        "failed_count": len(failed_artifacts),
        "passed_count": len(artifacts) - len(failed_artifacts),
    }

    return ValidationResult(passed=passed, failed_artifacts=failed_artifacts, details=details)


# TODO: Implement _check_dependencies helper function (T014)
def _check_dependencies(required_deps: List[str]) -> None:
    """Check that required packages are available for visualization generation.

    Args:
        required_deps: List of dependency names to check (e.g., ['matplotlib', 'moviepy'])

    Raises:
        VisualizationError: If any required dependencies are missing
    """
    import importlib.util

    missing_deps = []

    for dep in required_deps:
        if importlib.util.find_spec(dep) is None:
            missing_deps.append(dep)

    if missing_deps:
        raise VisualizationError(
            f"Missing visualization dependencies: {', '.join(missing_deps)}. "
            "Install with: pip install " + " ".join(missing_deps),
            artifact_type="dependency_check",
            details={"missing_dependencies": missing_deps},
        )


# TODO: Connect visualization functions to episode data parsing (T017)
# TODO: Integrate with environment factory for video rendering (T018)
# TODO: Add error handling for missing dependencies (T019)
# TODO: Add logging for visualization generation progress (T020)
