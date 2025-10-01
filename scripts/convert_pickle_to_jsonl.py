#!/usr/bin/env python3
"""
Conversion script for legacy multi-episode pickle files to per-episode JSONL format.

This script converts existing multi-episode pickle recordings into the new
per-episode JSONL format with proper episode segmentation and metadata.

Usage:
    python scripts/convert_pickle_to_jsonl.py <input.pkl> [output_dir]
    python scripts/convert_pickle_to_jsonl.py recordings/ [output_dir]

Features:
    - Automatic episode boundary detection using position jump heuristics
    - Per-episode JSONL file generation with metadata
    - Configurable output directory structure
    - Validation of converted files
    - Summary statistics
"""

import argparse
import pickle
import sys
from pathlib import Path

import loguru

from robot_sf.render.jsonl_playback import JSONLPlaybackLoader
from robot_sf.render.jsonl_recording import JSONLRecorder
from robot_sf.render.sim_view import VisualizableSimState

logger = loguru.logger


def detect_episode_boundaries(states: list[VisualizableSimState]) -> list[int]:
    """
    Detect episode boundaries in a sequence of states using position jump heuristics.

    Args:
        states: List of simulation states

    Returns:
        List of indices where new episodes likely begin
    """
    if len(states) < 2:
        return [0]

    boundaries = [0]  # First state is always a boundary

    # Look for sudden position jumps indicating resets
    position_jump_threshold = 10.0  # Threshold for considering a position jump as reset

    for i in range(1, len(states)):
        prev_state = states[i - 1]
        curr_state = states[i]

        reset_detected = False

        # Check robot position jumps
        if (
            hasattr(prev_state, "robot_pose")
            and prev_state.robot_pose
            and hasattr(curr_state, "robot_pose")
            and curr_state.robot_pose
        ):
            prev_pos = prev_state.robot_pose[0]
            curr_pos = curr_state.robot_pose[0]

            distance = ((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2) ** 0.5

            if distance > position_jump_threshold:
                reset_detected = True
                logger.debug(f"Robot position jump detected at step {i}: {distance:.2f}")

        # Check pedestrian position jumps if robot data is not available
        if not reset_detected and (
            hasattr(prev_state, "pedestrian_positions")
            and prev_state.pedestrian_positions is not None
            and hasattr(curr_state, "pedestrian_positions")
            and curr_state.pedestrian_positions is not None
            and len(prev_state.pedestrian_positions) > 0
            and len(curr_state.pedestrian_positions) > 0
        ):
            # Check first pedestrian position jump
            prev_ped_pos = prev_state.pedestrian_positions[0]
            curr_ped_pos = curr_state.pedestrian_positions[0]

            ped_distance = (
                (curr_ped_pos[0] - prev_ped_pos[0]) ** 2 + (curr_ped_pos[1] - prev_ped_pos[1]) ** 2
            ) ** 0.5

            if ped_distance > position_jump_threshold:
                reset_detected = True
                logger.debug(f"Pedestrian position jump detected at step {i}: {ped_distance:.2f}")

        if reset_detected:
            boundaries.append(i)

    logger.info(f"Detected {len(boundaries)} episode boundaries at indices: {boundaries}")
    return boundaries


def segment_states_by_episodes(
    states: list[VisualizableSimState], boundaries: list[int]
) -> list[list[VisualizableSimState]]:
    """
    Segment states into episodes based on boundary indices.

    Args:
        states: List of all simulation states
        boundaries: List of indices where episodes begin

    Returns:
        List of episodes, each containing a list of states
    """
    episodes = []

    for i, start_idx in enumerate(boundaries):
        end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(states)
        episode_states = states[start_idx:end_idx]
        episodes.append(episode_states)

    logger.info(f"Segmented {len(states)} states into {len(episodes)} episodes")
    for i, episode in enumerate(episodes):
        logger.info(f"  Episode {i}: {len(episode)} states")

    return episodes


def convert_pickle_to_jsonl(
    pickle_file: Path,
    output_dir: Path,
    suite: str = "converted",
    scenario: str = "legacy",
    algorithm: str = "unknown",
    seed: int = 0,
) -> int:
    """
    Convert a single pickle file to JSONL format.

    Args:
        pickle_file: Path to input pickle file
        output_dir: Directory for output JSONL files
        suite: Suite name for file naming
        scenario: Scenario name for file naming
        algorithm: Algorithm name for file naming
        seed: Seed for file naming

    Returns:
        Number of episodes converted
    """
    logger.info(f"Converting {pickle_file} to JSONL format")

    # Load pickle file
    loader = JSONLPlaybackLoader()
    episode, _ = loader.load_single_episode(pickle_file)

    if not episode.states:
        logger.warning(f"No states found in {pickle_file}")
        return 0

    # Detect episode boundaries
    boundaries = detect_episode_boundaries(episode.states)

    # Segment states into episodes
    episodes = segment_states_by_episodes(episode.states, boundaries)

    # Convert each episode to JSONL
    converted_count = 0

    for ep_idx, episode_states in enumerate(episodes):
        if not episode_states:
            continue

        # Create JSONL recorder for this episode
        recorder = JSONLRecorder(
            output_dir=str(output_dir),
            suite=suite,
            scenario=scenario,
            algorithm=algorithm,
            seed=seed,
        )

        recorder.current_episode_id = ep_idx
        config_hash = f"converted_from_{pickle_file.stem}"
        recorder.start_episode(config_hash=config_hash)

        # Record all states in the episode
        for state in episode_states:
            recorder.record_step(state)

        recorder.end_episode()
        converted_count += 1

        logger.info(f"Converted episode {ep_idx} with {len(episode_states)} states")

    logger.info(f"Successfully converted {converted_count} episodes from {pickle_file}")
    return converted_count


def convert_directory(
    input_dir: Path, output_dir: Path, file_pattern: str = "*.pkl"
) -> tuple[int, int]:
    """
    Convert all pickle files in a directory to JSONL format.

    Args:
        input_dir: Directory containing pickle files
        output_dir: Directory for output JSONL files
        file_pattern: Glob pattern for finding pickle files

    Returns:
        Tuple of (files_converted, total_episodes)
    """
    pickle_files = list(input_dir.glob(file_pattern))

    if not pickle_files:
        logger.error(f"No pickle files found in {input_dir} matching pattern {file_pattern}")
        return 0, 0

    files_converted = 0
    total_episodes = 0

    for pickle_file in pickle_files:
        try:
            # Extract naming info from filename if possible
            stem = pickle_file.stem
            parts = stem.split("_") if "_" in stem else [stem]

            suite = parts[0] if len(parts) > 0 else "converted"
            scenario = parts[1] if len(parts) > 1 else "legacy"
            algorithm = parts[2] if len(parts) > 2 else "unknown"

            episodes_converted = convert_pickle_to_jsonl(
                pickle_file,
                output_dir,
                suite=suite,
                scenario=scenario,
                algorithm=algorithm,
                seed=files_converted,
            )

            if episodes_converted > 0:
                files_converted += 1
                total_episodes += episodes_converted

        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            logger.exception(f"Failed to convert {pickle_file}")

    return files_converted, total_episodes


def validate_conversion(output_dir: Path) -> bool:
    """
    Validate converted JSONL files by attempting to load them.

    Args:
        output_dir: Directory containing converted files

    Returns:
        True if all files are valid, False otherwise
    """
    jsonl_files = list(output_dir.glob("*.jsonl"))

    if not jsonl_files:
        logger.error("No JSONL files found for validation")
        return False

    loader = JSONLPlaybackLoader()
    valid_files = 0

    for jsonl_file in jsonl_files:
        try:
            episode, _ = loader.load_single_episode(jsonl_file)
            if episode.states:
                valid_files += 1
                logger.debug(f"Validated {jsonl_file}: {len(episode.states)} states")
            else:
                logger.warning(f"No states found in {jsonl_file}")
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            logger.exception(f"Validation failed for {jsonl_file}")

    logger.info(f"Validation: {valid_files}/{len(jsonl_files)} files are valid")
    return valid_files == len(jsonl_files)


def main():
    """Main conversion script entry point."""
    parser = argparse.ArgumentParser(
        description="Convert legacy multi-episode pickle files to per-episode JSONL format"
    )
    parser.add_argument("input", help="Input pickle file or directory containing pickle files")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="converted_jsonl",
        help="Output directory for JSONL files (default: converted_jsonl)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate converted files by loading them"
    )
    parser.add_argument(
        "--pattern", default="*.pkl", help="File pattern for directory processing (default: *.pkl)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    input_path = Path(args.input)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Perform conversion
    if input_path.is_file():
        # Convert single file
        episodes_converted = convert_pickle_to_jsonl(input_path, output_path)
        files_converted = 1 if episodes_converted > 0 else 0
        total_episodes = episodes_converted
    else:
        # Convert directory
        files_converted, total_episodes = convert_directory(input_path, output_path, args.pattern)

    # Print summary
    logger.info("=" * 50)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Files converted: {files_converted}")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Output directory: {output_path}")

    # Validate if requested
    if args.validate:
        logger.info("Validating converted files...")
        if validate_conversion(output_path):
            logger.info("✅ All converted files are valid!")
        else:
            logger.error("❌ Some converted files failed validation")
            sys.exit(1)

    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
