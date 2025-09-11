"""JSONL playback loader for robot simulation recordings.

This module provides functionality to load and parse JSONL-format simulation
recordings, supporting both single-episode files and batch loading from
directories or manifest files. Includes backward compatibility with pickle format.

Key Features:
    - Load single JSONL episode files
    - Batch load from directories with glob patterns
    - Manifest-based batch loading
    - Episode boundary detection for trajectory clearing
    - Backward compatibility with existing pickle files
    - Streaming playback support

Schema Support:
    - JSONL records with episode_id, step_idx, event fields
    - Episode metadata from sidecar files
    - Entity reset event detection
    - Legacy multi-episode pickle file segmentation
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import loguru

from robot_sf.nav.map_config import MapDefinition
from robot_sf.render.sim_view import VisualizableSimState

logger = loguru.logger


@dataclass
class PlaybackEpisode:
    """Container for a single episode's playback data."""

    episode_id: int
    states: List[VisualizableSimState]
    metadata: Optional[Dict[str, Any]] = None
    reset_points: Optional[List[int]] = None  # Step indices where resets occurred


@dataclass
class BatchPlayback:
    """Container for batch playback data."""

    episodes: List[PlaybackEpisode]
    map_def: MapDefinition
    total_episodes: int
    total_steps: int


class JSONLPlaybackLoader:
    """Loads JSONL-format simulation recordings for playback.

    Supports loading single episodes, batch loading from directories,
    and manifest-based loading. Provides episode boundary information
    for proper trajectory visualization.
    """

    def __init__(self):
        """Initialize the playback loader."""
        self.schema_version = "1.0"

    def _deserialize_state(
        self, state_dict: Dict[str, Any], timestep: int = 0
    ) -> VisualizableSimState:
        """Deserialize a state dictionary to VisualizableSimState.

        Args:
            state_dict: Dictionary representation of state
            timestep: Timestep for the state

        Returns:
            Reconstructed VisualizableSimState
        """
        import numpy as np

        # Extract timestep
        timestep = state_dict.get("timestep", timestep)

        # Extract robot pose
        robot_pose = None
        if "robot_pose" in state_dict and state_dict["robot_pose"]:
            pos = state_dict["robot_pose"][0]
            ori = state_dict["robot_pose"][1]
            robot_pose = ((pos[0], pos[1]), ori)

        # Extract pedestrian positions as numpy array
        pedestrian_positions = np.array([])
        if "pedestrian_positions" in state_dict and state_dict["pedestrian_positions"]:
            pedestrian_positions = np.array(
                [[pos[0], pos[1]] for pos in state_dict["pedestrian_positions"]]
            )

        # Extract ego pedestrian pose
        ego_ped_pose = None
        if "ego_ped_pose" in state_dict and state_dict["ego_ped_pose"]:
            pos = state_dict["ego_ped_pose"][0]
            ori = state_dict["ego_ped_pose"][1]
            ego_ped_pose = ((pos[0], pos[1]), ori)

        # Extract ray vectors as numpy array
        ray_vecs = np.array([])
        if "ray_vecs" in state_dict and state_dict["ray_vecs"]:
            try:
                ray_vecs = np.array(state_dict["ray_vecs"])
            except (ValueError, TypeError):
                ray_vecs = np.array([])

        # For now, create empty arrays for required parameters we don't serialize
        ped_actions = np.array([])

        # Extract robot action
        robot_action = None
        if "robot_action" in state_dict and state_dict["robot_action"]:
            # For now, we'll leave robot_action as None since it's complex to reconstruct
            pass

        # Create the state object with required parameters
        state = VisualizableSimState(
            timestep=timestep,
            robot_action=robot_action,
            robot_pose=robot_pose,
            pedestrian_positions=pedestrian_positions,
            ray_vecs=ray_vecs,
            ped_actions=ped_actions,
            ego_ped_pose=ego_ped_pose,
        )

        return state

    def _load_episode_metadata(self, episode_file: Path) -> Optional[Dict[str, Any]]:
        """Load metadata for an episode from sidecar file.

        Args:
            episode_file: Path to the episode JSONL file

        Returns:
            Metadata dictionary or None if not found
        """
        # Generate metadata filename
        metadata_file = episode_file.with_suffix(".meta.json")

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
            return None

    def load_single_episode(
        self, file_path: Union[str, Path]
    ) -> Tuple[PlaybackEpisode, MapDefinition]:
        """Load a single episode from JSONL file.

        Args:
            file_path: Path to JSONL episode file

        Returns:
            Tuple of (PlaybackEpisode, MapDefinition)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Episode file not found: {file_path}")

        # Check if it's a pickle file (backward compatibility)
        if file_path.suffix == ".pkl":
            return self._load_pickle_file(file_path)

        # Load JSONL file
        states = []
        reset_points = []
        episode_id = 0

        with open(file_path, "r") as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())

                    # Skip empty lines
                    if not record:
                        continue

                    # Extract episode info
                    episode_id = record.get("episode_id", 0)
                    event = record.get("event", "step")

                    # Track reset points for trajectory clearing
                    if event in ["entity_reset", "episode_start"]:
                        reset_points.append(len(states))

                    # Skip metadata-only records
                    if event in ["episode_start", "episode_end"]:
                        continue

                    # Deserialize state
                    state_dict = record.get("state", {})
                    if state_dict:  # Only process records with actual state data
                        timestep = record.get("step_idx", len(states))
                        state = self._deserialize_state(state_dict, timestep)
                        states.append(state)

                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num + 1} in {file_path}: {e}")
                    continue

        if not states:
            raise ValueError(f"No valid states found in {file_path}")

        # Load metadata
        metadata = self._load_episode_metadata(file_path)

        # Create episode
        episode = PlaybackEpisode(
            episode_id=episode_id, states=states, metadata=metadata, reset_points=reset_points
        )

        # For now, create a dummy MapDefinition - in practice, this would be stored separately
        # or reconstructed from the first state's context

        map_def = MapDefinition(
            width=20.0,
            height=20.0,
            obstacles=[],
            robot_spawn_zones=[],
            ped_spawn_zones=[],
            robot_goal_zones=[],
            bounds=[],
            robot_routes=[],
            ped_goal_zones=[],
            ped_crowded_zones=[],
            ped_routes=[],
        )

        logger.info(f"Loaded episode {episode_id} with {len(states)} states from {file_path}")

        return episode, map_def

    def _load_pickle_file(self, file_path: Path) -> Tuple[PlaybackEpisode, MapDefinition]:
        """Load legacy pickle file and convert to episode format.

        Args:
            file_path: Path to pickle file

        Returns:
            Tuple of (PlaybackEpisode, MapDefinition)
        """
        logger.info(f"Loading legacy pickle file: {file_path}")

        with open(file_path, "rb") as f:
            states, map_def = pickle.load(f)

        # Detect episode boundaries in multi-episode pickle files
        reset_points = self._detect_reset_points(states)

        # Create episode (use first episode if multiple detected)
        episode = PlaybackEpisode(
            episode_id=0,
            states=states,
            metadata={"source": "pickle", "legacy": True},
            reset_points=reset_points,
        )

        return episode, map_def

    def _detect_reset_points(self, states: List[VisualizableSimState]) -> List[int]:
        """Detect reset points in a sequence of states (for legacy files).

        Args:
            states: List of simulation states

        Returns:
            List of indices where resets likely occurred
        """
        reset_points = []

        if len(states) < 2:
            return reset_points

        # Look for sudden position jumps (indicating resets)
        for i in range(1, len(states)):
            prev_state = states[i - 1]
            curr_state = states[i]

            # Check robot position jumps
            if (
                hasattr(prev_state, "robot_pose")
                and prev_state.robot_pose
                and hasattr(curr_state, "robot_pose")
                and curr_state.robot_pose
            ):
                prev_pos = prev_state.robot_pose[0]
                curr_pos = curr_state.robot_pose[0]

                # Calculate distance
                distance = (
                    (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
                ) ** 0.5

                # If distance is very large, likely a reset
                if distance > 10.0:  # Threshold for reset detection
                    reset_points.append(i)
                    logger.debug(f"Detected reset at step {i} (distance: {distance:.2f})")

        return reset_points

    def load_directory(self, directory: Union[str, Path]) -> BatchPlayback:
        """Load all JSONL episodes from a directory.

        Args:
            directory: Path to directory containing episode files

        Returns:
            BatchPlayback object with all episodes
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        # Find all episode files
        episode_files = list(directory.glob("*.jsonl"))
        episode_files.extend(directory.glob("*.pkl"))  # Include legacy files
        episode_files.sort()  # Ensure consistent ordering

        if not episode_files:
            raise ValueError(f"No episode files found in {directory}")

        # Load all episodes
        episodes = []
        map_def = None
        total_steps = 0

        for file_path in episode_files:
            try:
                episode, episode_map_def = self.load_single_episode(file_path)
                episodes.append(episode)
                total_steps += len(episode.states)

                # Use first valid map definition
                if map_def is None:
                    map_def = episode_map_def

            except Exception as e:
                logger.warning(f"Failed to load episode from {file_path}: {e}")
                continue

        if not episodes:
            raise ValueError(f"No valid episodes loaded from {directory}")

        # Create batch playback
        batch = BatchPlayback(
            episodes=episodes,
            map_def=map_def,
            total_episodes=len(episodes),
            total_steps=total_steps,
        )

        logger.info(
            f"Loaded {len(episodes)} episodes with {total_steps} total steps from {directory}"
        )

        return batch

    def load_manifest(self, manifest_file: Union[str, Path]) -> BatchPlayback:
        """Load episodes from a manifest file.

        Args:
            manifest_file: Path to manifest JSON file listing episode files

        Returns:
            BatchPlayback object with all episodes
        """
        manifest_file = Path(manifest_file)

        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        # Load manifest
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        episode_files = manifest.get("episodes", [])
        if not episode_files:
            raise ValueError(f"No episodes listed in manifest: {manifest_file}")

        # Load all episodes
        episodes = []
        map_def = None
        total_steps = 0

        base_dir = manifest_file.parent

        for file_path in episode_files:
            try:
                full_path = base_dir / file_path
                episode, episode_map_def = self.load_single_episode(full_path)
                episodes.append(episode)
                total_steps += len(episode.states)

                # Use first valid map definition
                if map_def is None:
                    map_def = episode_map_def

            except Exception as e:
                logger.warning(f"Failed to load episode from {file_path}: {e}")
                continue

        if not episodes:
            raise ValueError(f"No valid episodes loaded from manifest: {manifest_file}")

        # Create batch playback
        batch = BatchPlayback(
            episodes=episodes,
            map_def=map_def,
            total_episodes=len(episodes),
            total_steps=total_steps,
        )

        logger.info(f"Loaded {len(episodes)} episodes with {total_steps} total steps from manifest")

        return batch

    def load_batch(self, source: Union[str, Path]) -> BatchPlayback:
        """Load batch playback from directory or manifest file.

        Args:
            source: Path to directory or manifest file

        Returns:
            BatchPlayback object
        """
        source = Path(source)

        if source.is_dir():
            return self.load_directory(source)
        elif source.is_file():
            if source.suffix == ".json":
                return self.load_manifest(source)
            else:
                # Single file - convert to batch
                episode, map_def = self.load_single_episode(source)
                return BatchPlayback(
                    episodes=[episode],
                    map_def=map_def,
                    total_episodes=1,
                    total_steps=len(episode.states),
                )
        else:
            raise FileNotFoundError(f"Source not found: {source}")


def create_playback_loader() -> JSONLPlaybackLoader:
    """Factory function to create a JSONL playback loader.

    Returns:
        Configured JSONLPlaybackLoader instance
    """
    return JSONLPlaybackLoader()
