"""Legacy visualization helpers moved from examples/classic_interactions_pygame.py.

This module is kept for backward compatibility with pre–Full Classic flows that
directly called ``generate_benchmark_plots/videos``. The canonical benchmark
visualization pipeline now lives in ``robot_sf.benchmark.full_classic.visuals``
and is invoked by ``run_full_benchmark`` to produce manifest files. Prefer that
pipeline for new work; this module does not emit manifests and will be retired
once downstream consumers migrate.
"""

from __future__ import annotations

# Standard library
import gc
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

# Third-party
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from multiprocessing.connection import Connection


def frame_shape_from_map(map_svg_path: str) -> tuple[int, int]:
    """Return (width_px, height_px) parsed from an SVG file.

    Looks for width/height attributes on the root <svg> element. If missing,
    attempts to parse viewBox and derive width/height. Raises FileNotFoundError
    when the file doesn't exist and ValueError for invalid SVG content.

    Returns:
        Tuple of (width_pixels, height_pixels) parsed from the SVG.
    """
    with open(map_svg_path, encoding="utf-8") as f:
        data = f.read()

    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ValueError("Invalid SVG content") from e

    # Try width/height attributes first
    width = root.get("width")
    height = root.get("height")
    if width and height:
        try:
            return int(float(width)), int(float(height))
        except Exception:
            # fallthrough to try viewBox
            pass

    viewbox = root.get("viewBox") or root.get("viewbox")
    if viewbox:
        parts = viewbox.strip().split()
        if len(parts) == 4:
            try:
                _, _, w, h = parts
                return int(float(w)), int(float(h))
            except Exception:
                pass

    raise ValueError("SVG missing width/height or valid viewBox")


def overlay_text(canvas, text: str, pos: tuple[int, int], font: str | None = None) -> None:
    """Draw text on a duck-typed canvas object.

    The canvas must provide a `draw_text(text, pos, font=None)` method. This
    function calls that method and raises TypeError if unavailable.
    """
    draw = getattr(canvas, "draw_text", None)
    if not callable(draw):
        raise TypeError("canvas does not implement draw_text(text, pos, font)")

    # Keep simple: pass font through as-is; canvas is responsible for font handling
    draw(text, pos, font)


"""
Visualization functions for benchmark outputs.

This module provides higher-level functions used by examples and the
benchmarking pipeline. The heavy plotting code is implemented lazily to
avoid importing matplotlib at module import time (see Constitution XII).
"""

# Default robot speed for replay episodes when speed data is not available
DEFAULT_ROBOT_SPEED = 0.5
_MATPLOTLIB_INITIALIZED = False


def _ensure_matplotlib_backend() -> None:
    """TODO docstring. Document this function."""
    global _MATPLOTLIB_INITIALIZED
    if _MATPLOTLIB_INITIALIZED:
        return
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    _MATPLOTLIB_INITIALIZED = True


def _get_pyplot():
    """TODO docstring. Document this function.

    Returns:
        The matplotlib.pyplot module.
    """
    _ensure_matplotlib_backend()
    return importlib.import_module("matplotlib.pyplot")


def _load_replay_types():
    """TODO docstring. Document this function.

    Returns:
        Tuple of (ReplayEpisode, ReplayStep) classes from replay module.
    """
    module = importlib.import_module("robot_sf.benchmark.full_classic.replay")
    return module.ReplayEpisode, module.ReplayStep


def _load_image_sequence_clip():
    """TODO docstring. Document this function.

    Returns:
        The ImageSequenceClip class from moviepy.
    """
    module = importlib.import_module("moviepy.video.io.ImageSequenceClip")
    return module.ImageSequenceClip


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
    error_message: str | None = None


@dataclass
class ValidationResult:
    """Result of validating visual artifacts."""

    passed: bool
    failed_artifacts: list[VisualArtifact]
    details: dict | None = None


class VisualizationError(Exception):
    """Raised when visualization generation fails."""

    def __init__(self, message: str, artifact_type: str, details: dict | None = None):
        """TODO docstring. Document this function.

        Args:
            message: TODO docstring.
            artifact_type: TODO docstring.
            details: TODO docstring.
        """
        super().__init__(message)
        self.artifact_type = artifact_type
        self.details = details or {}


def generate_benchmark_plots_from_data(
    episodes: list[dict],
    output_dir: str,
    scenario_filter: str | None = None,
    baseline_filter: str | None = None,
) -> list[VisualArtifact]:
    """
    Generate PDF plots by analyzing benchmark episodes from data.

    Args:
        episodes: List of episode dictionaries from benchmark execution
        output_dir: Directory to save generated PDF plots
        scenario_filter: Optional scenario ID to filter plots
        baseline_filter: Optional baseline name to filter plots

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If plot generation fails
    """
    _check_matplotlib_available()

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
            ),
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
            ),
        )

    return artifacts


def generate_benchmark_plots_from_file(
    episodes_path: str | Path,
    output_dir: str,
    scenario_filter: str | None = None,
    baseline_filter: str | None = None,
) -> list[VisualArtifact]:
    """
    Generate PDF plots by analyzing benchmark episodes from a file.

    Args:
        episodes_path: Path to JSONL file containing episode data
        output_dir: Directory to save generated PDF plots
        scenario_filter: Optional scenario ID to filter plots
        baseline_filter: Optional baseline name to filter plots

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If plot generation fails
        FileNotFoundError: If episodes_path doesn't exist
    """
    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    if _should_use_plot_subprocess():
        return _generate_plots_subprocess(
            str(episodes_path),
            output_dir,
            scenario_filter,
            baseline_filter,
        )

    episodes = _load_episodes(str(episodes_path))
    return generate_benchmark_plots_from_data(
        episodes,
        output_dir,
        scenario_filter,
        baseline_filter,
    )


def generate_benchmark_plots(
    episodes_data: list[dict] | str | Path,
    output_dir: str,
    scenario_filter: str | None = None,
    baseline_filter: str | None = None,
) -> list[VisualArtifact]:
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
    # Handle backward compatibility: if episodes_data is a path, load it
    if isinstance(episodes_data, str | Path):
        return generate_benchmark_plots_from_file(
            cast("str | Path", episodes_data),
            output_dir,
            scenario_filter,
            baseline_filter,
        )
    else:
        return generate_benchmark_plots_from_data(
            cast("list[dict]", episodes_data),
            output_dir,
            scenario_filter,
            baseline_filter,
        )


def _should_use_plot_subprocess() -> bool:
    """Return True when plot generation should run in a subprocess."""
    return os.environ.get("ROBOT_SF_VISUALIZATION_SUBPROCESS", "1") != "0"


def _plot_subprocess_worker(
    episodes_path: str,
    output_dir: str,
    scenario_filter: str | None,
    baseline_filter: str | None,
    conn: Connection,
) -> None:
    """Generate plots in a child process and return artifacts via a pipe."""
    try:
        episodes = _load_episodes(episodes_path)
        artifacts = generate_benchmark_plots_from_data(
            episodes,
            output_dir,
            scenario_filter,
            baseline_filter,
        )
        conn.send(("ok", artifacts))
    except Exception as exc:  # pragma: no cover - defensive fallback
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _generate_plots_subprocess(
    episodes_path: str,
    output_dir: str,
    scenario_filter: str | None,
    baseline_filter: str | None,
) -> list[VisualArtifact]:
    """Run plot generation in a subprocess to keep parent RSS stable.

    Returns:
        List of generated VisualArtifact metadata from the subprocess.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_plot_subprocess_worker,
        args=(episodes_path, output_dir, scenario_filter, baseline_filter, child_conn),
    )
    process.start()
    child_conn.close()

    try:
        if not parent_conn.poll(timeout=300):
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            raise VisualizationError(
                "Plot subprocess timed out after 300 seconds",
                "plot",
            )
        status, payload = parent_conn.recv()
    except EOFError as exc:  # pragma: no cover - subprocess crashed before sending
        process.join()
        raise VisualizationError(
            "Plot subprocess exited without returning results",
            "plot",
        ) from exc
    finally:
        parent_conn.close()

    process.join()

    if process.exitcode not in (0, None):
        raise VisualizationError(
            f"Plot subprocess exited with code {process.exitcode}",
            "plot",
            {"exitcode": process.exitcode},
        )

    if status != "ok":
        raise VisualizationError(f"Plot subprocess failed: {payload}", "plot")

    return payload


def _check_matplotlib_available() -> None:
    """Check if matplotlib is available, raise VisualizationError if not."""
    _check_dependencies(["matplotlib"])
    _ensure_matplotlib_backend()


def _load_episodes(episodes_path: str) -> list[dict]:
    """Load episodes from JSONL file.

    Returns:
        List of episode dictionaries loaded from JSONL.
    """
    episodes = []
    try:
        with open(episodes_path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise VisualizationError(
                        f"Failed to decode JSON on line {i} of {episodes_path}: {e}",
                        "plot",
                        {"episodes_path": episodes_path, "line": i},
                    ) from e
    except OSError as e:
        raise VisualizationError(
            f"Failed to read episode file {episodes_path}: {e}",
            "plot",
            {"episodes_path": episodes_path},
        ) from e

    if not episodes:
        raise VisualizationError("No episode data found", "plot", {"episodes_path": episodes_path})
    return episodes


def _filter_episodes(
    episodes: list[dict],
    scenario_filter: str | None,
    baseline_filter: str | None,
) -> list[dict]:
    """Filter episodes based on scenario and baseline filters.

    Returns:
        Filtered list of episode dictionaries.
    """
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


def _generate_metrics_plot(episodes: list[dict], output_dir: str) -> VisualArtifact:
    """Generate metrics distribution plot and return artifact.

    Returns:
        VisualArtifact metadata object for the generated plot.
    """
    plt = _get_pyplot()

    try:
        # Extract metrics
        collisions = [ep.get("metrics", {}).get("collision_rate", 0) for ep in episodes]
        success_rates = [ep.get("metrics", {}).get("success_rate", 0) for ep in episodes]
        snqi_scores = [ep.get("metrics", {}).get("snqi", 0) for ep in episodes]

        # Create plot
        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

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
        plt.close(_fig)
        # Ensure figures are fully released and memory returned to OS where possible
        try:
            gc.collect()
        except Exception:
            pass

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
            f"Failed to generate metrics plot due to a data error: {e}",
            "plot",
        )


def _generate_scenario_comparison_plot(episodes: list[dict], output_dir: str) -> VisualArtifact:
    """Generate scenario comparison plot and return artifact.

    Returns:
        VisualArtifact metadata object for the generated comparison plot.
    """
    plt = _get_pyplot()

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
        _fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

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
        plt.close(_fig)
        # Ensure figures are fully released and memory returned to OS where possible
        try:
            gc.collect()
        except Exception:
            pass

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
            f"Failed to generate scenario comparison plot due to a data error: {e}",
            "plot",
        )


def generate_benchmark_videos_from_data(
    episodes: list[dict],
    output_dir: str,
    scenario_filter: str | None = None,
    baseline_filter: str | None = None,
    fps: int = 30,
    max_duration: float = 10.0,
) -> list[VisualArtifact]:
    """
    Generate MP4 videos by replaying benchmark episodes from data.

    Args:
        episodes: List of episode dictionaries from benchmark execution
        output_dir: Directory to save generated MP4 videos
        scenario_filter: Optional scenario ID to filter videos
        baseline_filter: Optional baseline name to filter videos
        fps: Frames per second for video rendering
        max_duration: Maximum video duration in seconds

    Returns:
        List of generated VisualArtifact objects with metadata

    Raises:
        VisualizationError: If video rendering fails
    """
    _check_moviepy_available()

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
            "(set capture_replay=True in benchmark configuration).",
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
                source_data=f"Episode {episode['episode_id']} replay from episodes_data{filter_str}",
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


def generate_benchmark_videos_from_file(
    episodes_path: str | Path,
    output_dir: str,
    scenario_filter: str | None = None,
    baseline_filter: str | None = None,
    fps: int = 30,
    max_duration: float = 10.0,
) -> list[VisualArtifact]:
    """
    Generate MP4 videos by replaying benchmark episodes from a file.

    Args:
        episodes_path: Path to JSONL file containing episode data
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
    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    episodes = _load_episodes(str(episodes_path))
    return generate_benchmark_videos_from_data(
        episodes,
        output_dir,
        scenario_filter,
        baseline_filter,
        fps,
        max_duration,
    )


def generate_benchmark_videos(
    episodes_data: list[dict] | str | Path,
    output_dir: str,
    scenario_filter: str | None = None,
    baseline_filter: str | None = None,
    fps: int = 30,
    max_duration: float = 10.0,
) -> list[VisualArtifact]:
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
    # Handle backward compatibility: if episodes_data is a path, load it
    if isinstance(episodes_data, str | Path):
        return generate_benchmark_videos_from_file(
            cast("str | Path", episodes_data),
            output_dir,
            scenario_filter,
            baseline_filter,
            fps,
            max_duration,
        )
    else:
        return generate_benchmark_videos_from_data(
            cast("list[dict]", episodes_data),
            output_dir,
            scenario_filter,
            baseline_filter,
            fps,
            max_duration,
        )


def _check_moviepy_available() -> None:
    """Check if moviepy is available, raise VisualizationError if not."""
    _check_dependencies(["moviepy"])


def _episode_record_to_replay_episode(episode: dict):
    """Convert an episode record with replay data to a ReplayEpisode object.

    Returns:
        ReplayEpisode object constructed from the episode dictionary.
    """
    try:
        ReplayEpisode, ReplayStep = _load_replay_types()
    except (ImportError, ModuleNotFoundError) as exc:
        raise VisualizationError("ReplayEpisode not available", "video") from exc

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
            speed=DEFAULT_ROBOT_SPEED,  # Default speed, could be enhanced
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


def _generate_frames_from_replay(replay_episode, fps: int, max_frames: int) -> list[np.ndarray]:
    """Generate synthetic frames from a replay episode for video creation.

    Args:
        replay_episode: The ReplayEpisode object containing trajectory data
        fps: Target frames per second for the video (informational)
        max_frames: Maximum number of frames to generate

    Returns:
        List of numpy arrays representing video frames
    """
    steps = list(getattr(replay_episode, "steps", []))
    if not steps:
        logger.warning("Replay episode %s has no steps", getattr(replay_episode, "episode_id", "?"))
        return []

    # Extract positions and calculate bounds
    bounds = _calculate_trajectory_bounds(steps)
    pixel_converter = _create_pixel_converter(bounds)

    # Convert all positions to pixels
    robot_pixels = [pixel_converter(step.x, step.y) for step in steps]
    ped_pixels = [
        [pixel_converter(px, py) for px, py in (step.ped_positions or [])] for step in steps
    ]

    # Generate frames
    return _generate_frames_from_pixels(robot_pixels, ped_pixels, max_frames)


def _calculate_trajectory_bounds(steps) -> tuple[float, float, float, float]:
    """Calculate the bounding box for all trajectory positions.

    Returns:
        Tuple of (min_x, max_x, min_y, max_y) bounding box coordinates with padding.
    """
    positions: list[tuple[float, float]] = []
    for step in steps:
        positions.append((step.x, step.y))
        if step.ped_positions:
            positions.extend([(float(px), float(py)) for px, py in step.ped_positions])

    if not positions:
        positions.append((0.0, 0.0))

    xs, ys = zip(*positions, strict=False)
    min_x, max_x = float(min(xs)), float(max(xs))
    min_y, max_y = float(min(ys)), float(max(ys))

    # Add padding
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    pad_x = span_x * 0.1
    pad_y = span_y * 0.1

    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def _create_pixel_converter(bounds: tuple[float, float, float, float]):
    """Create a function to convert world coordinates to pixel coordinates.

    Returns:
        Callable that converts (x, y) world coordinates to (row, col) pixel coordinates.
    """
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    height, width = 320, 320

    def to_pixel(x: float, y: float) -> tuple[int, int]:
        """TODO docstring. Document this function.

        Args:
            x: TODO docstring.
            y: TODO docstring.

        Returns:
            TODO docstring.
        """
        norm_x = (x - min_x) / span_x
        norm_y = (y - min_y) / span_y
        col = int(np.clip(norm_x * (width - 1), 0, width - 1))
        row = int(np.clip((1.0 - norm_y) * (height - 1), 0, height - 1))
        return row, col

    return to_pixel


def _generate_frames_from_pixels(
    robot_pixels: list[tuple[int, int]],
    ped_pixels: list[list[tuple[int, int]]],
    max_frames: int,
) -> list[np.ndarray]:
    """Generate video frames from pre-computed pixel positions.

    Returns:
        List of numpy arrays representing video frames (height x width x 3 RGB).
    """
    height, width = 320, 320
    background = np.array([22, 26, 30], dtype=np.uint8)
    trail_color = (72, 188, 110)
    robot_color = (245, 228, 92)
    ped_color = (214, 72, 72)

    frames: list[np.ndarray] = []
    limit = max_frames if max_frames and max_frames > 0 else None

    for idx, (robot_row, robot_col) in enumerate(robot_pixels):
        if limit is not None and len(frames) >= limit:
            break

        # Create fresh frame with background
        frame = np.full((height, width, 3), background, dtype=np.uint8)

        # Draw trail up to current position (efficiently, only drawing what's needed)
        trail_length = min(idx + 1, 50)  # Limit trail length to prevent excessive drawing
        start_idx = max(0, idx - trail_length + 1)
        for trail_idx in range(start_idx, idx + 1):
            trail_row, trail_col = robot_pixels[trail_idx]
            _draw_disk(frame, trail_row, trail_col, 2, trail_color)

        # Draw current robot position
        _draw_disk(frame, robot_row, robot_col, 4, robot_color)

        # Draw pedestrian markers if available
        for ped_row, ped_col in ped_pixels[idx]:
            _draw_disk(frame, ped_row, ped_col, 2, ped_color)

        # Progress bar for quick visual feedback
        progress = int(((idx + 1) / len(robot_pixels)) * width)
        frame[-4:, :progress] = robot_color

        frames.append(frame)

    return frames


def _draw_disk(
    frame: np.ndarray,
    row: int,
    col: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled disk on the frame at the specified position."""
    height, width = frame.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            rr = row + dy
            cc = col + dx
            if 0 <= rr < height and 0 <= cc < width:
                frame[rr, cc] = color


def _encode_frames_to_video(frames: list[np.ndarray], video_path: str, fps: int) -> None:
    """Encode a list of frames to an MP4 video file, optimized for memory usage."""
    if not frames:
        raise VisualizationError("No frames to encode", "video")

    try:
        ImageSequenceClip = _load_image_sequence_clip()

        # Use a generator to avoid loading all frames into memory at once
        def frame_generator():
            """TODO docstring. Document this function."""
            yield from frames

        # Create a temporary audio file path (even though audio=False, moviepy expects it)
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio:
            temp_audiofile = temp_audio.name

        clip = ImageSequenceClip(list(frame_generator()), fps=fps)
        clip.write_videofile(
            video_path,
            codec="libx264",
            audio=False,
            temp_audiofile=temp_audiofile,
            remove_temp=True,
            logger=None,
        )
    except (ValueError, TypeError, OSError) as e:
        raise VisualizationError(f"Failed to encode video: {e}", "video")


def validate_visual_artifacts(artifacts: list[VisualArtifact]) -> ValidationResult:
    """
    Validate that visual artifacts contain real data.

    Args:
        artifacts: List of VisualArtifact objects to validate

    Returns:
        ValidationResult with pass/fail status and details
    """
    if not artifacts:
        return ValidationResult(
            passed=True,
            failed_artifacts=[],
            details={"message": "No artifacts to validate"},
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


def _check_dependencies(required_deps: list[str]) -> None:
    """Check that required packages are available for visualization generation.

    Args:
        required_deps: List of dependency names to check (e.g., ['matplotlib', 'moviepy'])

    Raises:
        VisualizationError: If any required dependencies are missing
    """
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
