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
from typing import List, Literal, Optional, Union

import numpy as np
from loguru import logger


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


class VisualizationError(Exception):
    """Raised when visualization generation fails."""

    def __init__(self, message: str, artifact_type: str, details: Optional[dict] = None):
        super().__init__(message)
        self.artifact_type = artifact_type
        self.details = details or {}


def generate_benchmark_plots(
    episodes_data: Union[List[dict], str, Path],
    output_dir: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None,
) -> List[VisualArtifact]:
    """
    Generate PDF plots by analyzing benchmark episodes.

    Args:
        episodes_data: List of episode dictionaries from benchmark execution, or path (str | Path) to JSONL file
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

    except (ValueError, KeyError, TypeError) as e:
        raise VisualizationError(
            f"Failed to generate metrics plot due to a data error: {e}", "plot"
        )


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

    except (ValueError, KeyError, TypeError) as e:
        raise VisualizationError(
            f"Failed to generate scenario comparison plot due to a data error: {e}", "plot"
        )


def generate_benchmark_videos(
    episodes_data: Union[List[dict], str, Path],
    output_dir: str,
    scenario_filter: Optional[str] = None,
    baseline_filter: Optional[str] = None,
    fps: int = 30,
    max_duration: float = 10.0,
) -> List[VisualArtifact]:
    """
    Generate MP4 videos by replaying benchmark episodes.

    Args:
        episodes_data: List of episode dictionaries from benchmark execution, or path (str | Path) to JSONL file
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
        episodes_path = str(episodes_data)
        episodes = _load_episodes(episodes_path)
    else:
        episodes_path = "episodes_data"
        episodes = episodes_data

    if not episodes:
        raise VisualizationError(
            "No episode data found",
            "video",
            {"episodes_count": len(episodes)},
        )

    filtered_episodes = _filter_episodes(episodes, scenario_filter, baseline_filter)

    # Filter episodes that have replay data
    episodes_with_replay = [ep for ep in filtered_episodes if ep.get("replay_steps")]

    if not episodes_with_replay:
        logger.warning(
            "No episodes with replay data found for video generation. "
            "Video generation requires replay capture to be enabled during benchmark execution "
            "(set capture_replay=True in benchmark configuration)."
        )
        return []

    os.makedirs(output_dir, exist_ok=True)
    videos_dir = Path(output_dir) / "videos"
    videos_dir.mkdir(exist_ok=True)

    artifacts = []
    max_frames = int(max_duration * fps)

    for episode in episodes_with_replay:
        try:
            # Convert episode record to ReplayEpisode
            replay_episode = _episode_record_to_replay_episode(episode)

            # Generate video filename
            video_filename = f"episode_{episode['episode_id']}.mp4"
            video_path = videos_dir / video_filename

            # Generate frames using the existing render_sim_view system
            frames = _generate_frames_from_replay(replay_episode, fps, max_frames)

            if not frames:
                logger.warning(f"No frames generated for episode {episode['episode_id']}")
                continue

            # Encode frames to video
            _encode_frames_to_video(frames, str(video_path), fps)

            # Create artifact
            generation_time = datetime.now()
            file_size = video_path.stat().st_size

            filter_info = []
            if scenario_filter:
                filter_info.append(f"scenario={scenario_filter}")
            if baseline_filter:
                filter_info.append(f"baseline={baseline_filter}")
            filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

            artifact = VisualArtifact(
                artifact_id=f"video_{episode['episode_id']}_{generation_time.timestamp()}",
                artifact_type="video",
                format="mp4",
                filename=video_filename,
                source_data=f"Episode {episode['episode_id']} replay from {episodes_path}{filter_str}",
                generation_time=generation_time,
                file_size=file_size,
                status="generated",
            )
            artifacts.append(artifact)

        except VisualizationError as e:
            logger.warning(f"Failed to generate video for episode {episode['episode_id']}: {e}")
            artifact = VisualArtifact(
                artifact_id=f"video_{episode['episode_id']}_{datetime.now().timestamp()}",
                artifact_type="video",
                format="mp4",
                filename=f"episode_{episode['episode_id']}.mp4",
                source_data=f"Episode {episode['episode_id']} replay",
                generation_time=datetime.now(),
                file_size=0,
                status="failed",
                error_message=str(e),
            )
            artifacts.append(artifact)

    return artifacts


def _check_moviepy_available() -> None:
    """Check if moviepy is available, raise VisualizationError if not."""
    _check_dependencies(["moviepy"])


def _episode_record_to_replay_episode(episode: dict):
    """Convert an episode record with replay data to a ReplayEpisode object."""
    try:
        from robot_sf.benchmark.full_classic.replay import ReplayEpisode, ReplayStep
    except ImportError:
        raise VisualizationError("ReplayEpisode not available", "video")

    episode_id = episode["episode_id"]
    scenario_id = episode["scenario_id"]

    replay_steps = []
    replay_steps_data = episode.get("replay_steps", [])
    replay_peds = episode.get("replay_peds", [])
    replay_actions = episode.get("replay_actions", [])

    for i, (t, x, y, heading) in enumerate(replay_steps_data):
        # Get pedestrian positions and actions for this timestep
        ped_positions = replay_peds[i] if i < len(replay_peds) else None
        action = replay_actions[i] if i < len(replay_actions) else None

        step = ReplayStep(
            t=t,
            x=x,
            y=y,
            heading=heading,
            speed=0.5,  # Default speed, could be enhanced
            ped_positions=ped_positions,
            action=action,
        )
        replay_steps.append(step)

    replay_episode = ReplayEpisode(
        episode_id=episode_id,
        scenario_id=scenario_id,
        steps=replay_steps,
    )

    return replay_episode


def _generate_frames_from_replay(replay_episode, fps: int, max_frames: int) -> List[np.ndarray]:
    """Generate frames from a ReplayEpisode using the render_sim_view system."""
    try:
        from robot_sf.benchmark.full_classic.render_sim_view import generate_frames
    except ImportError:
        logger.warning("SimulationView rendering not available, cannot generate real video frames")
        return []

    frames = []
    for frame in generate_frames(replay_episode, fps=fps, max_frames=max_frames):
        frames.append(frame)

    return frames


def _encode_frames_to_video(frames: List[np.ndarray], video_path: str, fps: int) -> None:
    """Encode a list of frames to an MP4 video file."""
    if not frames:
        raise VisualizationError("No frames to encode", "video")

    try:
        from moviepy import ImageSequenceClip

        # Convert frames to the format expected by moviepy (list of numpy arrays)
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(video_path, codec="libx264", audio=False, verbose=False, logger=None)
    except (ValueError, TypeError, OSError) as e:
        raise VisualizationError(f"Failed to encode video: {e}", "video")


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
