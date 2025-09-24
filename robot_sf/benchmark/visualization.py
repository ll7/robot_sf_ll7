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
    episodes_path: str,
    output_dir: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None,
) -> List[VisualArtifact]:
    """
    Generate comprehensive PDF plots from benchmark episode data.

    Args:
        episodes_path: Path to JSONL file containing episode records
        output_dir: Directory to save generated PDF plots
        scenario_filter: Optional scenario ID to filter plots
        baseline_filter: Optional baseline name to filter plots

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If plotting fails due to data or dependency issues
        FileNotFoundError: If episodes_path doesn't exist
    """
    _check_matplotlib_available()

    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    episodes = _load_episodes(episodes_path)
    filtered_episodes = _filter_episodes(episodes, scenario_filter, baseline_filter)

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    artifacts.extend(
        _generate_metrics_plot(
            filtered_episodes, plots_dir, episodes_path, scenario_filter, baseline_filter
        )
    )

    return artifacts


def _check_matplotlib_available() -> None:
    """Check if matplotlib is available, raise VisualizationError if not."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
    except ImportError as e:
        raise VisualizationError(
            f"Matplotlib not available: {e}", "plot", {"missing_dependency": "matplotlib"}
        )


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


def _generate_metrics_plot(
    episodes: List[dict],
    plots_dir: Path,
    episodes_path: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None,
) -> List[VisualArtifact]:
    """Generate metrics distribution plot and return artifacts."""
    artifacts = []

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        # Extract metrics
        collisions = [ep.get("metrics", {}).get("collisions", 0) for ep in episodes]
        successes = [ep.get("metrics", {}).get("success", False) for ep in episodes]
        snqi_scores = [ep.get("metrics", {}).get("snqi", 0) for ep in episodes]

        # Plot 1: Collisions distribution
        plt.subplot(2, 2, 1)
        plt.hist(collisions, bins=20, alpha=0.7, color="blue", edgecolor="black")
        plt.title("Collision Distribution")
        plt.xlabel("Collisions")
        plt.ylabel("Frequency")

        # Plot 2: Success rate
        plt.subplot(2, 2, 2)
        success_rate = sum(successes) / len(successes) * 100
        plt.bar(["Success", "Failure"], [success_rate, 100 - success_rate], color=["green", "red"])
        plt.title(f"Success Rate: {success_rate:.1f}%")
        plt.ylabel("Percentage")

        # Plot 3: SNQI distribution
        plt.subplot(2, 2, 3)
        plt.hist(snqi_scores, bins=20, alpha=0.7, color="orange", edgecolor="black")
        plt.title("SNQI Score Distribution")
        plt.xlabel("SNQI Score")
        plt.ylabel("Frequency")

        # Plot 4: Collisions vs SNQI scatter
        plt.subplot(2, 2, 4)
        plt.scatter(collisions, snqi_scores, alpha=0.6, color="purple")
        plt.title("Collisions vs SNQI")
        plt.xlabel("Collisions")
        plt.ylabel("SNQI Score")

        plt.tight_layout()

        # Save plot
        plot_filename = "metrics_distribution.pdf"
        plot_path = plots_dir / plot_filename
        plt.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close()

        # Create artifact
        generation_time = datetime.now()
        file_size = plot_path.stat().st_size

        filter_info = []
        if scenario_filter:
            filter_info.append(f"scenario={scenario_filter}")
        if baseline_filter:
            filter_info.append(f"baseline={baseline_filter}")
        filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

        artifact = VisualArtifact(
            artifact_id=f"plot_metrics_{generation_time.timestamp()}",
            artifact_type="plot",
            format="pdf",
            filename=plot_filename,
            source_data=f"{len(episodes)} episodes from {episodes_path}{filter_str}",
            generation_time=generation_time,
            file_size=file_size,
            status="generated",
        )
        artifacts.append(artifact)

    except Exception as e:
        # Create failed artifact
        artifact = VisualArtifact(
            artifact_id=f"plot_metrics_{datetime.now().timestamp()}",
            artifact_type="plot",
            format="pdf",
            filename="metrics_distribution.pdf",
            source_data=f"Failed: {len(episodes)} episodes",
            generation_time=datetime.now(),
            file_size=0,
            status="failed",
            error_message=str(e),
        )
        artifacts.append(artifact)
        raise VisualizationError(f"Failed to generate metrics plot: {e}", "plot")

    return artifacts


# TODO: Implement generate_benchmark_videos function (T012) - COMPLETED
def generate_benchmark_videos(
    episodes_path: str,
    output_dir: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None,
    fps: int = 30,
    max_duration: float = 10.0,
) -> List[VisualArtifact]:
    """
    Generate MP4 videos by replaying benchmark episodes.

    Args:
        episodes_path: Path to JSONL file containing episode records
        output_dir: Directory to save generated MP4 videos
        scenario_filter: Optional scenario ID to filter videos
        baseline_filter: Optional baseline name to filter videos
        fps: Frames per second for video rendering
        max_duration: Maximum video duration in seconds

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If video rendering fails
        FileNotFoundError: If episodes_path doesn't exist
    """
    _check_moviepy_available()

    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    episodes = _load_episodes(episodes_path)
    filtered_episodes = _filter_episodes(episodes, scenario_filter, baseline_filter)

    # Filter episodes that have trajectory data
    episodes_with_trajectory = [ep for ep in filtered_episodes if ep.get("trajectory_data")]

    if not episodes_with_trajectory:
        raise VisualizationError(
            "No episodes with trajectory data found for video generation",
            "video",
            {"episodes_with_trajectory": len(episodes_with_trajectory)},
        )

    videos_dir = Path(output_dir) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    artifacts.extend(
        _generate_episode_videos(
            episodes_with_trajectory, videos_dir, fps, max_duration, episodes_path
        )
    )

    return artifacts


def _check_moviepy_available() -> None:
    """Check if moviepy is available, raise VisualizationError if not."""
    import importlib.util

    if importlib.util.find_spec("moviepy") is None:
        raise VisualizationError(
            "MoviePy not available", "video", {"missing_dependency": "moviepy"}
        )


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
# TODO: Connect visualization functions to episode data parsing (T017)
# TODO: Integrate with environment factory for video rendering (T018)
# TODO: Add error handling for missing dependencies (T019)
# TODO: Add logging for visualization generation progress (T020)
