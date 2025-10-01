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
    - timestamp: float - Simulation timestamp (0.0 for episode_start, state.timestep for others)
    - state: dict - Serialized VisualizableSimState
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

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
    state: dict[str, Any]
    entity_ids: Optional[list[int]] = None  # For entity_reset events


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
        self.last_simulation_timestep = 0.0

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

    def _serialize_state(self, state: VisualizableSimState) -> dict[str, Any]:
        """Serialize a VisualizableSimState to dictionary format."""
        state_dict: dict[str, Any] = {"timestep": getattr(state, "timestep", 0.0)}

        # Delegate serialization of optional components to small helpers to keep
        # cyclomatic complexity low (ruff C901 compliance) while retaining
        # defensive checks.
        robot_pose = self._serialize_robot_pose(state)
        if robot_pose is not None:
            state_dict["robot_pose"] = robot_pose

        ped_positions = self._serialize_pedestrian_positions(state)
        if ped_positions is not None:
            state_dict["pedestrian_positions"] = ped_positions

        ego_ped_pose = self._serialize_ego_ped_pose(state)
        if ego_ped_pose is not None:
            state_dict["ego_ped_pose"] = ego_ped_pose

        ray_vecs = self._serialize_ray_vecs(state)
        if ray_vecs is not None:
            state_dict["ray_vecs"] = ray_vecs

        robot_action = self._serialize_robot_action(state)
        if robot_action is not None:
            state_dict["robot_action"] = robot_action

        return state_dict

    # --- Serialization helpers (each intentionally small & focused) ---
    def _serialize_robot_pose(self, state: VisualizableSimState) -> Optional[list[Any]]:
        """Return robot pose [[x, y], theta] if available else None."""
        if not hasattr(state, "robot_pose") or state.robot_pose is None:
            return None
        try:
            return [
                [float(state.robot_pose[0][0]), float(state.robot_pose[0][1])],
                float(state.robot_pose[1]),
            ]
        except (TypeError, ValueError, IndexError):  # Defensive: malformed pose
            return None

    def _serialize_pedestrian_positions(
        self, state: VisualizableSimState
    ) -> Optional[list[list[float]]]:
        """Return list of pedestrian positions [[x,y], ...] or None if absent."""
        if not hasattr(state, "pedestrian_positions") or state.pedestrian_positions is None:
            return None
        positions = state.pedestrian_positions
        try:
            if hasattr(positions, "__len__") and len(positions) > 0:
                return [[float(p[0]), float(p[1])] for p in positions]
            return []
        except (TypeError, ValueError, IndexError):
            return []

    def _serialize_ego_ped_pose(self, state: VisualizableSimState) -> Optional[list[Any]]:
        """Return ego pedestrian pose [[x,y], theta] if available else None."""
        if not hasattr(state, "ego_ped_pose") or state.ego_ped_pose is None:
            return None
        try:
            return [
                [float(state.ego_ped_pose[0][0]), float(state.ego_ped_pose[0][1])],
                float(state.ego_ped_pose[1]),
            ]
        except (TypeError, ValueError, IndexError):
            return None

    def _serialize_ray_vecs(self, state: VisualizableSimState) -> Optional[list[Any]]:
        """Return ray vectors list representation or None if absent."""
        if not hasattr(state, "ray_vecs") or state.ray_vecs is None:
            return None
        ray_vecs = state.ray_vecs
        if not hasattr(ray_vecs, "__len__") or len(ray_vecs) == 0:  # empty container
            return []
        try:
            if hasattr(ray_vecs, "tolist"):
                return ray_vecs.tolist()
            return [[float(x) for x in ray] for ray in ray_vecs]
        except (TypeError, ValueError):
            return []

    def _serialize_robot_action(self, state: VisualizableSimState) -> Optional[dict[str, float]]:
        """Return robot action dict {'linear_velocity':..,'angular_velocity':..} or None."""
        if not hasattr(state, "robot_action") or state.robot_action is None:
            return None
        action = state.robot_action
        action_dict: dict[str, float] = {}
        if hasattr(action, "linear_velocity"):
            try:
                action_dict["linear_velocity"] = float(action.linear_velocity)
            except (TypeError, ValueError):  # ignore malformed values
                pass
        if hasattr(action, "angular_velocity"):
            try:
                action_dict["angular_velocity"] = float(action.angular_velocity)
            except (TypeError, ValueError):
                pass
        return action_dict or None

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
        self.last_simulation_timestep = 0.0
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
        self.current_file = open(episode_file, "w", encoding="utf-8")

        # Write episode start record
        start_record = JSONLRecord(
            episode_id=self.current_episode_id,
            step_idx=self.current_step_idx,
            event="episode_start",
            timestamp=0.0,
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

        # Extract simulation timestep and track it
        simulation_timestep = float(getattr(state, "timestep", 0.0))
        self.last_simulation_timestep = simulation_timestep

        # Create step record
        step_record = JSONLRecord(
            episode_id=self.current_episode_id,
            step_idx=self.current_step_idx,
            event="step",
            timestamp=simulation_timestep,
            state=state_dict,
        )

        # Write to file
        self.current_file.write(json.dumps(asdict(step_record)) + "\n")
        self.current_file.flush()

    def record_entity_reset(self, entity_ids: list[int], state: VisualizableSimState) -> None:
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

        # Extract simulation timestep and track it
        simulation_timestep = float(getattr(state, "timestep", 0.0))
        self.last_simulation_timestep = simulation_timestep

        # Create entity reset record
        reset_record = JSONLRecord(
            episode_id=self.current_episode_id,
            step_idx=self.current_step_idx,
            event="entity_reset",
            timestamp=simulation_timestep,
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
            timestamp=self.last_simulation_timestep,
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
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.current_metadata), f, indent=2)

        # Close episode file
        self.current_file.close()
        self.current_file = None

        logger.info(f"Ended episode {self.current_episode_id} with {self.current_step_idx} steps")

        # Prepare for next episode
        self.current_episode_id += 1
        self.current_step_idx = 0
        self.current_metadata = None
        self.last_simulation_timestep = 0.0

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
