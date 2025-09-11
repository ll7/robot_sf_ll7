"""JSONL-based recording system for robot simulation states.

This module provides functionality to record robot simulation states in JSONL format
with per-episode files and episode metadata. Supports streaming, episode boundaries,
and detailed event tracking.

Key Features:
    - Per-episode JSONL files with standardized naming
    - Episode metadata with step indices and event types
    - Entity reset tracking
    - Backward compatibility with pickle format
    - Sidecar metadata files for episode configuration

Schema:
    Each JSONL record contains:
    - episode_id: int - Episode identifier
    - step_idx: int - Step within episode (0-based)
    - event: str - Event type ("episode_start", "step", "episode_end", "entity_reset")
    - timestamp: float - Simulation timestamp
    - state: dict - Serialized VisualizableSimState
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import loguru

from robot_sf.render.sim_view import VisualizableSimState

logger = loguru.logger


@dataclass
class EpisodeMetadata:
    """Metadata for an episode recording."""

    episode_id: int
    algorithm: str
    scenario: str
    seed: int
    config_hash: str
    schema_version: str = "1.0"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_steps: Optional[int] = None


@dataclass
class JSONLRecord:
    """A single JSONL record for simulation state."""

    episode_id: int
    step_idx: int
    event: str  # "episode_start", "step", "episode_end", "entity_reset"
    timestamp: float
    state: Dict[str, Any]
    entity_ids: Optional[List[int]] = None  # For entity_reset events


class JSONLRecorder:
    """Records simulation states in JSONL format with per-episode files.

    This recorder creates one JSONL file per episode with standardized naming
    and includes episode metadata in sidecar files. Supports streaming writes
    and proper episode boundary handling.
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        suite: str = "robot_sim",
        scenario: str = "default",
        algorithm: str = "manual",
        seed: int = 0,
    ):
        """Initialize the JSONL recorder.

        Args:
            output_dir: Directory to save recordings
            suite: Test suite name for file naming
            scenario: Scenario name for file naming
            algorithm: Algorithm name for file naming
            seed: Random seed for file naming
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.suite = suite
        self.scenario = scenario
        self.algorithm = algorithm
        self.seed = seed

        self.current_episode_id = 0
        self.current_step_idx = 0
        self.current_file = None
        self.current_metadata = None

        self.schema_version = "1.0"

    def _get_episode_filename(self, episode_id: int) -> Path:
        """Generate standardized filename for episode."""
        filename = (
            f"{self.suite}_{self.scenario}_{self.algorithm}_{self.seed}_ep{episode_id:04d}.jsonl"
        )
        return self.output_dir / filename

    def _get_metadata_filename(self, episode_id: int) -> Path:
        """Generate sidecar metadata filename for episode."""
        filename = f"{self.suite}_{self.scenario}_{self.algorithm}_{self.seed}_ep{episode_id:04d}.meta.json"
        return self.output_dir / filename

    def _serialize_state(self, state: VisualizableSimState) -> Dict[str, Any]:
        """Serialize a VisualizableSimState to dictionary format."""
        state_dict = {}

        # Serialize core attributes
        state_dict["timestep"] = state.timestep

        if hasattr(state, "robot_pose") and state.robot_pose is not None:
            state_dict["robot_pose"] = [
                [float(state.robot_pose[0][0]), float(state.robot_pose[0][1])],  # position
                float(state.robot_pose[1]),  # orientation
            ]

        if hasattr(state, "pedestrian_positions") and state.pedestrian_positions is not None:
            # Handle numpy array
            if (
                hasattr(state.pedestrian_positions, "__len__")
                and len(state.pedestrian_positions) > 0
            ):
                state_dict["pedestrian_positions"] = [
                    [float(pos[0]), float(pos[1])] for pos in state.pedestrian_positions
                ]
            else:
                state_dict["pedestrian_positions"] = []

        if hasattr(state, "ego_ped_pose") and state.ego_ped_pose is not None:
            state_dict["ego_ped_pose"] = [
                [float(state.ego_ped_pose[0][0]), float(state.ego_ped_pose[0][1])],  # position
                float(state.ego_ped_pose[1]),  # orientation
            ]

        if hasattr(state, "ray_vecs") and state.ray_vecs is not None:
            # Handle numpy array for LIDAR rays
            if hasattr(state.ray_vecs, "__len__") and len(state.ray_vecs) > 0:
                try:
                    # Handle multi-dimensional arrays properly
                    if hasattr(state.ray_vecs, "tolist"):
                        state_dict["ray_vecs"] = state.ray_vecs.tolist()
                    else:
                        # Fallback for other array types
                        state_dict["ray_vecs"] = [[float(x) for x in ray] for ray in state.ray_vecs]
                except (TypeError, ValueError):
                    # If conversion fails, skip ray_vecs
                    state_dict["ray_vecs"] = []
            else:
                state_dict["ray_vecs"] = []

        if hasattr(state, "robot_action") and state.robot_action is not None:
            # Serialize robot action if present
            action_dict = {}
            if hasattr(state.robot_action, "linear_velocity"):
                action_dict["linear_velocity"] = float(state.robot_action.linear_velocity)
            if hasattr(state.robot_action, "angular_velocity"):
                action_dict["angular_velocity"] = float(state.robot_action.angular_velocity)
            if action_dict:
                state_dict["robot_action"] = action_dict

        return state_dict

    def start_episode(self, config_hash: str = "unknown") -> None:
        """Start recording a new episode.

        Args:
            config_hash: Hash of environment configuration
        """
        # Close previous episode if open
        if self.current_file is not None:
            self.end_episode()

        # Initialize new episode
        self.current_step_idx = 0
        self.current_metadata = EpisodeMetadata(
            episode_id=self.current_episode_id,
            algorithm=self.algorithm,
            scenario=self.scenario,
            seed=self.seed,
            config_hash=config_hash,
            schema_version=self.schema_version,
            start_time=time.time(),
        )

        # Open new episode file
        episode_file = self._get_episode_filename(self.current_episode_id)
        self.current_file = open(episode_file, "w")

        # Write episode start record
        start_record = JSONLRecord(
            episode_id=self.current_episode_id,
            step_idx=self.current_step_idx,
            event="episode_start",
            timestamp=time.time(),
            state={},
        )
        self.current_file.write(json.dumps(asdict(start_record)) + "\n")
        self.current_file.flush()

        logger.info(f"Started recording episode {self.current_episode_id} to {episode_file}")

    def record_step(self, state: VisualizableSimState) -> None:
        """Record a simulation step.

        Args:
            state: Current simulation state
        """
        if self.current_file is None:
            logger.warning("No episode started, starting new episode")
            self.start_episode()

        self.current_step_idx += 1

        # Serialize state
        state_dict = self._serialize_state(state)

        # Create step record
        step_record = JSONLRecord(
            episode_id=self.current_episode_id,
            step_idx=self.current_step_idx,
            event="step",
            timestamp=time.time(),
            state=state_dict,
        )

        # Write to file
        self.current_file.write(json.dumps(asdict(step_record)) + "\n")
        self.current_file.flush()

    def record_entity_reset(self, entity_ids: List[int], state: VisualizableSimState) -> None:
        """Record an entity reset event.

        Args:
            entity_ids: IDs of entities that were reset
            state: Current simulation state after reset
        """
        if self.current_file is None:
            logger.warning("No episode started, starting new episode")
            self.start_episode()

        # Serialize state
        state_dict = self._serialize_state(state)

        # Create entity reset record
        reset_record = JSONLRecord(
            episode_id=self.current_episode_id,
            step_idx=self.current_step_idx,
            event="entity_reset",
            timestamp=time.time(),
            state=state_dict,
            entity_ids=entity_ids,
        )

        # Write to file
        self.current_file.write(json.dumps(asdict(reset_record)) + "\n")
        self.current_file.flush()

        logger.info(f"Recorded entity reset for entities {entity_ids}")

    def end_episode(self) -> None:
        """End the current episode recording."""
        if self.current_file is None:
            logger.warning("No episode to end")
            return

        # Write episode end record
        end_record = JSONLRecord(
            episode_id=self.current_episode_id,
            step_idx=self.current_step_idx,
            event="episode_end",
            timestamp=time.time(),
            state={},
        )
        self.current_file.write(json.dumps(asdict(end_record)) + "\n")
        self.current_file.flush()

        # Update metadata
        if self.current_metadata:
            self.current_metadata.end_time = time.time()
            self.current_metadata.total_steps = self.current_step_idx

        # Save metadata to sidecar file
        metadata_file = self._get_metadata_filename(self.current_episode_id)
        with open(metadata_file, "w") as f:
            json.dump(asdict(self.current_metadata), f, indent=2)

        # Close episode file
        self.current_file.close()
        self.current_file = None

        logger.info(f"Ended episode {self.current_episode_id} with {self.current_step_idx} steps")

        # Prepare for next episode
        self.current_episode_id += 1
        self.current_step_idx = 0
        self.current_metadata = None

    def close(self) -> None:
        """Close the recorder and end any open episode."""
        if self.current_file is not None:
            self.end_episode()


def create_jsonl_recorder(output_dir: str = "recordings", **kwargs) -> JSONLRecorder:
    """Factory function to create a JSONL recorder.

    Args:
        output_dir: Directory to save recordings
        **kwargs: Additional arguments for JSONLRecorder

    Returns:
        Configured JSONLRecorder instance
    """
    return JSONLRecorder(output_dir=output_dir, **kwargs)
