"""
`robot_env.py` is a module that defines the simulation environment for a robot or multiple robots.
It includes classes and protocols for defining the robot's state, actions, and
observations within the environment.

`RobotEnv`: A class that represents the robot's environment. It inherits from `VectorEnv`
from the `gymnasium` library, which is a base class for environments that operate over
vectorized actions and observations. It includes methods for stepping through the environment,
resetting it, rendering it, and closing it.
It also defines the action and observation spaces for the robot.
"""

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from loguru import logger
from shapely.geometry import Polygon as ShapelyPolygon

from robot_sf.common.types import Line2D
from robot_sf.gym_env.base_env import BaseEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.env_util import (
    global_reset_seed,
    init_collision_and_sensors,
    init_spaces,
    make_grid_observation_spaces,
    prepare_pedestrian_actions,
    reset_episode_counter_for_seed,
)
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.reset_metadata import build_reset_metadata
from robot_sf.gym_env.reward import route_completion_v2_reward
from robot_sf.gym_env.snqi_proxy import StepSNQIProxy
from robot_sf.nav.obstacle import Obstacle
from robot_sf.nav.occupancy_grid import OccupancyGrid
from robot_sf.render.lidar_visual import render_lidar
from robot_sf.render.sim_state import VisualizableAction, VisualizableSimState
from robot_sf.robot.robot_state import RobotState
from robot_sf.robot.rollover_proxy import RolloverProxyParams, rollover_proxy_telemetry
from robot_sf.sensor.range_sensor import lidar_ray_scan
from robot_sf.sensor.sensor_fusion import OBS_IMAGE, SensorFusion
from robot_sf.sensor.socnav_observation import (
    DEFAULT_MAX_PEDS,
    SocNavObservationFusion,
    socnav_observation_space,
)
from robot_sf.sim.simulator import (
    init_simulators,  # noqa: F401 (retained for backwards compatibility; may be removed later)
)
from robot_sf.telemetry.pane import TelemetrySession

DEFAULT_PANE_WIDTH = 320
DEFAULT_PANE_HEIGHT = 240
MIN_PANE_WIDTH = 200
MIN_PANE_HEIGHT = 160
_TELEMETRY_ANALYZER_STEP_METRIC_KEYS: tuple[str, ...] = (
    "near_misses",
    "force_exceed_events",
    "comfort_exposure",
    "jerk_mean",
    "rollover_stability_margin",
    "rollover_lateral_acceleration",
    "rollover_critical_lateral_acceleration",
)
_ASYMMETRIC_CRITIC_STATE_KEY = "critic_privileged_state"
_GridObstacleCacheKey = tuple[int, int, int]
_GridObstacleCacheValue = tuple[list[Line2D], list[ShapelyPolygon]]


# Helper to compute a stable, short hash for env_config
# Placed near imports for reuse and clarity
def _stable_config_hash(cfg: EnvSettings) -> str:
    """Build a stable short hash of the environment settings.

    Args:
        cfg: Environment settings to serialize for identity.

    Returns:
        16-character hexadecimal hash string representing the configuration.
    """
    try:
        payload = json.dumps(
            asdict(cfg) if is_dataclass(cfg) else cfg.__dict__,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        payload = repr(cfg)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _make_telemetry_run_id() -> str:
    """Generate a unique telemetry run identifier to avoid artifact collisions.

    Returns:
        str: High-entropy run identifier for telemetry artifacts.
    """
    return f"telemetry-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def _flatten_nested_dict_spaces(obs_space: spaces.Dict) -> spaces.Dict:
    """Flatten nested Dict spaces to top-level for StableBaselines3 compatibility.

    StableBaselines3's DummyVecEnv does not support nested Dict spaces. This helper
    flattens any nested Dict/Tuple structures to the top level by promoting leaf spaces.

    Args:
        obs_space: The observation space to flatten (typically from socnav_observation_space).

    Returns:
        A new spaces.Dict with all nested structures flattened to the top level using
        underscore-separated naming (e.g., "robot_position", "goal_current").
    """
    flattened = {}

    def _flatten_recursive(space_dict: dict, prefix: str = ""):
        """Recursively flatten nested spaces."""
        for key, space in space_dict.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"

            if isinstance(space, spaces.Dict):
                # Recursively flatten nested Dict
                _flatten_recursive(space.spaces, full_key)
            else:
                # Leaf space - add to flattened dict
                flattened[full_key] = space

    _flatten_recursive(obs_space.spaces)
    return spaces.Dict(flattened)


def _flatten_nested_dict_obs(obs: dict) -> dict:
    """Flatten nested dict observations to top-level for flattened observation spaces.

    Mirrors the structure flattening done in _flatten_nested_dict_spaces, converting
    nested observation dicts to flat dicts with underscore-separated keys.

    Args:
        obs: The nested observation dict (typically from SocNavObservationFusion).

    Returns:
        A flat observation dict matching the flattened observation space structure.
    """
    flattened = {}

    def _flatten_recursive(obs_dict: dict, prefix: str = ""):
        """Recursively flatten nested observation dicts."""
        for key, value in obs_dict.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"

            if isinstance(value, dict):
                # Recursively flatten nested dict
                _flatten_recursive(value, full_key)
            else:
                # Leaf value - add to flattened dict
                flattened[full_key] = value

    _flatten_recursive(obs)
    return flattened


def _flatten_space_leaves(
    space: spaces.Space,
    prefix: str = "",
) -> list[tuple[str, spaces.Space]]:
    """Return leaf spaces in deterministic traversal order."""
    if isinstance(space, spaces.Dict):
        leaves: list[tuple[str, spaces.Space]] = []
        for key, child_space in space.spaces.items():
            if prefix:
                child_prefix = f"{prefix}_{key}"
            else:
                child_prefix = key
            leaves.extend(_flatten_space_leaves(child_space, child_prefix))
        return leaves
    return [(prefix, space)]


def _flatten_obs_from_space(space: spaces.Space, obs: Any) -> list[np.ndarray]:
    """Flatten observations using the declared space ordering.

    Returns:
        list[np.ndarray]: Flattened leaf arrays aligned to the
            traversal order of ``space``.
    """
    if isinstance(space, spaces.Dict):
        leaves: list[np.ndarray] = []
        for key, child_space in space.spaces.items():
            leaves.extend(_flatten_obs_from_space(child_space, obs[key]))
        return leaves
    return [np.asarray(obs, dtype=np.float32).reshape(-1)]


def _asymmetric_critic_state_spec(
    obs_space: spaces.Dict,
    *,
    sim_time_limit: float,
    max_sim_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low/high bounds for the critic-only privileged state vector.

    Returns:
        tuple[np.ndarray, np.ndarray]: Low and high vectors for the
            privileged state.
    """
    low_parts: list[np.ndarray] = []
    high_parts: list[np.ndarray] = []
    for _key, space in _flatten_space_leaves(obs_space):
        if not isinstance(space, spaces.Box):
            raise TypeError(f"asymmetric_critic requires Box leaves; got {type(space).__name__}")
        low_parts.append(np.asarray(space.low, dtype=np.float32).reshape(-1))
        high_parts.append(np.asarray(space.high, dtype=np.float32).reshape(-1))
    # Bounds for the privileged time/step scalars use np.finfo(np.float32).max so that
    # scenario-level overrides extending max_sim_steps or sim_time_limit do not violate
    # the declared observation space. Emitted values are clipped to the configured limits
    # in _build_asymmetric_critic_state.
    finfo_max = float(np.finfo(np.float32).max)
    low_parts.extend(
        [
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.zeros(5, dtype=np.float32),
        ]
    )
    high_parts.extend(
        [
            np.array([finfo_max], dtype=np.float32),
            np.array([finfo_max], dtype=np.float32),
            np.array([finfo_max], dtype=np.float32),
            np.array([finfo_max], dtype=np.float32),
            np.array([finfo_max], dtype=np.float32),
            np.ones(5, dtype=np.float32),
        ]
    )
    return (
        np.concatenate(low_parts).astype(np.float32),
        np.concatenate(high_parts).astype(np.float32),
    )


class _FlatteningObservationWrapper:
    """Wraps a sensor fusion adapter to flatten nested observations.

    This adapter wraps SocNavObservationFusion to automatically flatten its nested
    dict observations to a flat structure compatible with StableBaselines3's DummyVecEnv,
    which does not support nested Dict observation spaces.
    """

    def __init__(self, wrapped_adapter):
        """Initialize wrapper with the sensor fusion adapter to wrap.

        Args:
            wrapped_adapter: The sensor fusion adapter (typically SocNavObservationFusion).
        """
        self.wrapped_adapter = wrapped_adapter

    def reset_cache(self) -> None:
        """Delegate to wrapped adapter."""
        return self.wrapped_adapter.reset_cache()

    def next_obs(self) -> dict:
        """Get next observation and flatten nested structure.

        Returns:
            A flat observation dict with underscore-separated keys matching the
            flattened observation space structure.
        """
        obs = self.wrapped_adapter.next_obs()
        return _flatten_nested_dict_obs(obs)


def _flatten_occupancy_grid_metadata(
    metadata: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Flatten occupancy grid metadata into prefixed field names.

    StableBaselines3 requires a flat Dict observation space without nesting.
    This converts the nested metadata dict (with 'origin', 'resolution', etc.)
    into flattened fields with 'occupancy_grid_meta_' prefix.

    Args:
        metadata: Dict from occupancy_grid.metadata_observation() with keys like
            'origin', 'resolution', etc.

    Returns:
        Dict with keys like 'occupancy_grid_meta_origin', 'occupancy_grid_meta_resolution',
        etc. for inclusion in the top-level observation dict.
    """
    return {f"occupancy_grid_meta_{key}": value for key, value in metadata.items()}


def _build_step_info(meta: dict[str, Any]) -> dict[str, Any]:
    """Construct the info dict with collision/success flags for downstream consumers.

    Returns:
        Dictionary containing step, meta, collision, success, and is_success flags.
        Success is defined strictly as full route completion.
    """

    collision = bool(
        meta.get("is_pedestrian_collision")
        or meta.get("is_obstacle_collision")
        or meta.get("is_robot_collision")
    )
    success = bool(meta.get("is_route_complete"))
    info = {
        "step": meta.get("step"),
        "meta": meta,
        "collision": collision,
        "success": success,
        "is_success": success,
    }
    if "termination_reason" in meta:
        info["termination_reason"] = meta["termination_reason"]
    if "rollover_proxy" in meta:
        info["rollover_proxy"] = meta["rollover_proxy"]
        info["rollover_critical"] = bool(meta.get("rollover_critical", False))
    return info


def _coerce_finite_float(value: Any) -> float | None:
    """Return ``value`` as a finite float when possible."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _build_reset_info(
    env_config: EnvSettings, *, map_def, applied_seed: int | None
) -> dict[str, Any]:
    """Return a stable, non-placeholder reset metadata payload.

    The payload intentionally keeps only lightweight, serializable fields useful for
    downstream debugging and reproducibility.
    """

    return build_reset_metadata(env_config, map_def=map_def, seed=applied_seed)


def _extract_reward_terms(meta: dict[str, Any]) -> dict[str, float]:
    """Extract finite reward-term scalars for telemetry replay.

    Returns:
        dict[str, float]: Weighted reward-term values suitable for analyzer replay.
    """

    reward_terms = meta.get("reward_terms")
    if not isinstance(reward_terms, dict):
        return {}
    extracted: dict[str, float] = {}
    for key, value in reward_terms.items():
        if not isinstance(key, str):
            continue
        numeric = _coerce_finite_float(value)
        if numeric is not None:
            extracted[key] = numeric
    return extracted


def _extract_step_metrics(meta: dict[str, Any]) -> dict[str, float]:
    """Select analyzer-friendly numeric step metrics from reward metadata.

    Returns:
        dict[str, float]: Numeric step metrics exposed to the recorded-step analyzer.
    """

    extracted: dict[str, float] = {}
    for key in _TELEMETRY_ANALYZER_STEP_METRIC_KEYS:
        numeric = _coerce_finite_float(meta.get(key))
        if numeric is not None:
            extracted[key] = numeric
    return extracted


class RobotEnv(BaseEnv):
    """Gymnasium environment facade for Robot SF simulation episodes.

    ``RobotEnv`` remains the public environment API. Focused runtime collaborators own
    extractable concerns such as step-level SNQI proxy metadata, while this class coordinates
    simulator lifecycle, observation assembly, reward evaluation, recording, and rendering.
    """

    def __init__(  # noqa: PLR0913
        self,
        env_config: EnvSettings | None = None,
        reward_func: Callable[[dict], float] | None = None,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str | None = None,
        video_fps: float | None = None,
        peds_have_obstacle_forces: bool = True,
        # New JSONL recording parameters
        use_jsonl_recording: bool = False,
        recording_dir: str = "recordings",
        suite_name: str = "robot_sim",
        scenario_name: str = "default",
        algorithm_name: str = "manual",
        recording_seed: int | None = None,
        asymmetric_critic: bool = False,
    ):
        """Initialize the robot environment.

        Args:
            env_config: Environment settings describing maps, sensors, and simulator behavior.
            reward_func: Optional callable used to compute rewards; defaults to
                ``route_completion_v2_reward``.
            debug: Enables ``SimulationView`` visualization and rendering hooks.
            recording_enabled: When ``True``, record ``VisualizableSimState`` snapshots.
            record_video: Save simulator frames as a video via ``SimulationView``.
            video_path: Output path for the recorded video (when ``record_video`` is enabled).
            video_fps: Override frames-per-second for recorded videos.
            peds_have_obstacle_forces: Deprecated. Controls static obstacle forces for pedestrians.
            use_jsonl_recording: Enable structured JSONL recording instead of pickles.
            recording_dir: Directory for recordings.
            suite_name: Logical suite name stored in recording metadata.
            scenario_name: Scenario identifier stored in metadata.
            algorithm_name: Algorithm identifier stored in metadata.
            recording_seed: Optional seed stored alongside the recording metadata.
            asymmetric_critic: When True, add a critic-only privileged state vector to
                observations for asymmetric actor-critic training.
        """
        if env_config is None:
            env_config = EnvSettings()
        self._asymmetric_critic_enabled = bool(asymmetric_critic)
        self._critic_privileged_state_key = _ASYMMETRIC_CRITIC_STATE_KEY
        self._critic_obs_space: spaces.Dict | None = None
        super().__init__(
            env_config=env_config,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
            use_jsonl_recording=use_jsonl_recording,
            recording_dir=recording_dir,
            suite_name=suite_name,
            scenario_name=scenario_name,
            algorithm_name=algorithm_name,
            recording_seed=recording_seed,
        )

        # Initialize spaces based on the environment configuration and map
        self.action_space, self.observation_space, orig_obs_space = init_spaces(
            env_config,
            self.map_def,
        )

        # Debug help
        self.debug_without_robot_movement: bool = bool(
            env_config.sim_config.debug_without_robot_movement
        )
        if self.debug_without_robot_movement:
            logger.warning("Debug mode: Robot will not move!")

        # Assign the reward function; ensure a valid callable even if None passed via factory
        if reward_func is None:  # defensive: factory allows Optional
            logger.debug(
                "No reward_func provided to RobotEnv; falling back to route_completion_v2_reward.",
            )
        self.reward_func = reward_func or route_completion_v2_reward

        # BaseEnv has already created self.simulator; avoid redundant initialization.
        # Initialize collision detectors and sensor data processors using existing simulator.
        occupancies, sensors = init_collision_and_sensors(
            self.simulator,
            env_config,
            orig_obs_space,
        )

        # Store configuration for factory pattern compatibility
        self.config = env_config
        self.grid_config = env_config.grid_config

        # Initialize optional occupancy grid
        self.occupancy_grid = self._build_occupancy_grid(env_config)

        # Configure observation space and sensor adapter per observation mode
        if env_config.observation_mode == ObservationMode.SOCNAV_STRUCT:
            sensor_adapter = self._setup_socnav_observation(env_config)
        else:
            sensor_adapter = self._setup_default_observation(env_config, sensors)

        self._apply_asymmetric_critic_observation_space(env_config)

        # Pre-compute critic leaf traversal to avoid per-step space reconstruction
        self._critic_obs_traversal: list[tuple[str, ...]] = (
            self._build_critic_traversal() if self._asymmetric_critic_enabled else []
        )

        # Setup initial state of the robot
        self.state = RobotState(
            self.simulator.robot_navs[0],
            occupancies[0],
            sensor_adapter,
            env_config.sim_config.time_per_step_in_secs,
            env_config.sim_config.sim_time_in_secs,
        )

        # Store last action executed by the robot
        self.last_action = None
        self.applied_seed: int | None = None
        self._latest_observation: Any = None
        # Enable occupancy grid overlay visualization if requested
        if self.sim_ui and getattr(env_config, "show_occupancy_grid", False):
            self.sim_ui.show_occupancy_grid = True
        self._configure_observation_overlay_mode()
        # Initialize telemetry session/pane when requested
        self._init_telemetry(env_config)
        self._last_wall_time = time.perf_counter()
        self._frame_idx = 0
        self._telemetry_episode_id = -1
        self._snqi_proxy = StepSNQIProxy()
        self._grid_obstacle_cache_key: _GridObstacleCacheKey | None = None
        self._grid_obstacle_cache_value: _GridObstacleCacheValue | None = None
        self._prime_snqi_proxy_state()

    def _prime_snqi_proxy_state(self) -> None:
        """Reset and prime step-level SNQI proxy state for a fresh episode."""
        self._snqi_proxy.prime(self.simulator)

    def _build_occupancy_grid(self, env_config: EnvSettings) -> OccupancyGrid | None:
        """Initialize occupancy grid if configured and return the instance.

        Returns:
            OccupancyGrid | None: Initialized grid when enabled; otherwise ``None``.
        """
        self.include_grid_in_observation = bool(
            getattr(env_config, "include_grid_in_observation", False)
        )
        if env_config.use_occupancy_grid and env_config.grid_config is not None:
            grid = OccupancyGrid(config=env_config.grid_config)
            logger.debug(
                "Occupancy grid initialized (observe={observe}, visualize={visualize}): shape={shape}, resolution={resolution:.3f}m",
                observe=self.include_grid_in_observation,
                visualize=env_config.show_occupancy_grid,
                shape=grid.shape,
                resolution=env_config.grid_config.resolution,
            )
            return grid
        return None

    def _configure_observation_overlay_mode(self) -> None:
        """Configure visualization overlay mode to match the active observation pipeline."""
        if self.sim_ui is None:
            return
        if getattr(self.env_config, "use_image_obs", False):
            self.sim_ui.observation_space_mode = "image"
            return
        if bool(self.include_grid_in_observation and self.occupancy_grid is not None):
            self.sim_ui.observation_space_mode = "grid"
            return
        self.sim_ui.observation_space_mode = "lidar"

    def _apply_asymmetric_critic_observation_space(self, env_config: EnvSettings) -> None:
        """Augment the active observation space with the critic-only privileged vector."""
        if not self._asymmetric_critic_enabled:
            return
        if not isinstance(self.observation_space, spaces.Dict):
            raise RuntimeError(
                "asymmetric_critic requires Dict observations "
                f"with key '{self._critic_privileged_state_key}'."
            )
        sim_time_limit = float(getattr(env_config.sim_config, "sim_time_in_secs", 0.0) or 0.0)
        dt = float(getattr(env_config.sim_config, "time_per_step_in_secs", 0.0) or 0.0)
        max_sim_steps = int(np.ceil(sim_time_limit / dt)) if dt > 0.0 else 0
        critic_obs_space = spaces.Dict(dict(self.observation_space.spaces))
        low, high = _asymmetric_critic_state_spec(
            critic_obs_space,
            sim_time_limit=sim_time_limit,
            max_sim_steps=max_sim_steps,
        )
        self._critic_obs_space = critic_obs_space
        obs_dict = dict(critic_obs_space.spaces)
        obs_dict[self._critic_privileged_state_key] = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(obs_dict)

    def _build_critic_traversal(self) -> list[tuple[str, ...]]:
        """Pre-compute leaf key-paths for the critic observation space (no privileged state).

        Returns:
            list[tuple[str, ...]]: Deterministic-order key paths into the obs dict.
        """
        paths: list[tuple[str, ...]] = []

        def _traverse(spaces_dict: dict, prefix: tuple[str, ...] = ()) -> None:
            for key, child in spaces_dict.items():
                path = prefix + (key,)
                if isinstance(child, spaces.Dict):
                    _traverse(child.spaces, path)
                else:
                    paths.append(path)

        if self._critic_obs_space is None:
            raise RuntimeError("asymmetric critic observation space has not been initialized")

        _traverse(self._critic_obs_space.spaces)
        return paths

    def _extract_obs_leaf(self, obs: dict, path: tuple[str, ...]) -> np.ndarray:
        """Navigate the nested obs dict along *path* and flatten the leaf value.

        Returns:
            np.ndarray: 1-D float32 array matching a single observation leaf.
        """
        value = obs
        for key in path:
            value = value[key]
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _build_asymmetric_critic_state(self, obs: Any) -> np.ndarray:
        """Build the critic-only privileged state vector from the current observation payload.

        Uses the cached ``_critic_obs_traversal`` to avoid per-step space reconstruction
        and batches metadata into a single array instead of ten tiny allocations.

        Returns:
            np.ndarray: Flattened privileged state for critic-only consumption.
        """
        if not self._asymmetric_critic_enabled:
            raise RuntimeError("asymmetric critic state requested when disabled")

        meta = self.state.meta_dict()
        parts = [self._extract_obs_leaf(obs, path) for path in self._critic_obs_traversal]

        finfo_max = float(np.finfo(np.float32).max)

        def _nonneg(value: float) -> float:
            """Clamp privileged scalar metadata into the declared non-negative Box range.

            Returns:
                float: Metadata value bounded to ``[0, finfo.max]``.
            """
            return max(0.0, min(float(value), finfo_max))

        metadata = np.empty(10, dtype=np.float32)
        metadata[0] = _nonneg(meta.get("step_of_episode", 0) or 0)
        metadata[1] = _nonneg(self.state.sim_time_elapsed)
        metadata[2] = _nonneg(meta.get("max_sim_steps", self.state.max_sim_steps))
        metadata[3] = _nonneg(meta.get("distance_to_goal", 0.0))
        metadata[4] = _nonneg(meta.get("prev_distance_to_goal", 0.0))
        metadata[5] = float(bool(meta.get("is_route_complete")))
        metadata[6] = float(bool(meta.get("is_timesteps_exceeded")))
        metadata[7] = float(bool(meta.get("is_pedestrian_collision")))
        metadata[8] = float(bool(meta.get("is_robot_collision")))
        metadata[9] = float(bool(meta.get("is_obstacle_collision")))

        parts.append(metadata)
        return np.concatenate(parts)

    def _attach_asymmetric_critic_state(self, obs: Any) -> Any:
        """Attach the privileged critic state to dict observations when enabled.

        Returns:
            Any: Observation payload with the critic-only privileged state attached.
        """
        if not self._asymmetric_critic_enabled:
            return obs
        if not isinstance(obs, dict):
            raise RuntimeError(
                "asymmetric_critic requires dict observations for the privileged state."
            )
        obs[self._critic_privileged_state_key] = self._build_asymmetric_critic_state(obs)
        return obs

    @staticmethod
    def _extract_observation_image(obs: Any) -> np.ndarray | None:
        """Extract an image-like tensor from an observation payload when present.

        Returns:
            np.ndarray | None: Image tensor when present in the observation payload.
        """
        if isinstance(obs, dict):
            image_obs = obs.get(OBS_IMAGE)
            if isinstance(image_obs, np.ndarray):
                return image_obs
            agent_obs = obs.get("agent_obs")
            if isinstance(agent_obs, dict):
                nested = agent_obs.get(OBS_IMAGE)
                if isinstance(nested, np.ndarray):
                    return nested
        return None

    def _setup_socnav_observation(self, env_config: EnvSettings):
        """Configure SocNav observation space and sensor adapter.

        Returns:
            SensorFusion: Adapter producing SocNav structured observations (flattened when needed).
        """
        ped_count = getattr(self.simulator, "ped_pos", np.zeros((0, 2))).shape[0]
        max_peds = getattr(env_config.sim_config, "max_total_pedestrians", None)
        if max_peds is None:
            max_peds = max(DEFAULT_MAX_PEDS, ped_count)
        self.observation_space = socnav_observation_space(
            self.map_def,
            env_config,
            max_peds,
        )
        socnav_fusion = SocNavObservationFusion(
            simulator=self.simulator,
            env_config=env_config,
            max_pedestrians=max_peds,
            robot_index=0,
        )

        # T044: When adding occupancy grid, flatten the nested observation space
        # for StableBaselines3 compatibility (SB3 doesn't support nested Dict spaces)
        self._flatten_obs_space = False
        self._wrap_obs_as_dict = False
        if self.include_grid_in_observation and self.occupancy_grid is not None:
            grid_box, meta_spaces = make_grid_observation_spaces(self.occupancy_grid.config)

            # Flatten the nested SocNav structure before adding grid fields
            self.observation_space = _flatten_nested_dict_spaces(self.observation_space)
            self._flatten_obs_space = True

            # Now add grid fields to the flattened space
            obs_dict = dict(self.observation_space.spaces)
            obs_dict["occupancy_grid"] = grid_box
            # Add all flattened metadata fields (not a nested Dict!)
            obs_dict.update(meta_spaces)
            self.observation_space = spaces.Dict(obs_dict)

        # Wrap sensor adapter to flatten observations when needed
        if self._flatten_obs_space:
            return _FlatteningObservationWrapper(socnav_fusion)
        return socnav_fusion

    def _setup_default_observation(
        self, env_config: EnvSettings, sensors: list[Any]
    ) -> SensorFusion:
        """Configure DEFAULT_GYM observation handling and optional grid fields.

        Returns:
            SensorFusion: Adapter used for the default observation mode.
        """
        sensor_adapter = sensors[0]
        self._flatten_obs_space = False
        self._wrap_obs_as_dict = False

        if self.include_grid_in_observation and self.occupancy_grid is not None:
            grid_box, meta_spaces = make_grid_observation_spaces(self.occupancy_grid.config)

            # Convert observation space to dict if needed (for grid fields)
            if not isinstance(self.observation_space, spaces.Dict):
                obs_dict = {"agent_obs": self.observation_space}
                self._wrap_obs_as_dict = True
            else:
                obs_dict = dict(self.observation_space.spaces)
                self._wrap_obs_as_dict = False

            # Add grid fields (grid box + flattened metadata)
            obs_dict["occupancy_grid"] = grid_box
            obs_dict.update(meta_spaces)
            self.observation_space = spaces.Dict(obs_dict)

        return sensor_adapter

    def _init_telemetry(self, env_config: EnvSettings) -> None:
        """Initialize telemetry session and hook into sim UI when requested."""
        self._telemetry_session = None
        if not (env_config.enable_telemetry_panel or env_config.telemetry_record):
            return

        run_id = _make_telemetry_run_id()
        pane_width = DEFAULT_PANE_WIDTH
        pane_height = DEFAULT_PANE_HEIGHT
        if self.sim_ui:
            self.sim_ui.show_telemetry_panel = env_config.enable_telemetry_panel
            self.sim_ui.telemetry_layout = env_config.telemetry_pane_layout
            pane_width = min(DEFAULT_PANE_WIDTH, max(MIN_PANE_WIDTH, self.sim_ui.width // 3))
            pane_height = min(DEFAULT_PANE_HEIGHT, max(MIN_PANE_HEIGHT, self.sim_ui.height // 3))

        self._telemetry_session = TelemetrySession(
            run_id=run_id,
            record=env_config.telemetry_record,
            metrics=env_config.telemetry_metrics,
            refresh_hz=env_config.telemetry_refresh_hz,
            decimation=env_config.telemetry_decimation,
            pane_size=(pane_width, pane_height),
        )
        if self.sim_ui:
            self.sim_ui.telemetry_session = self._telemetry_session

    def _rollover_proxy_record(self) -> dict[str, Any] | None:
        """Return opt-in rollover proxy telemetry for the executed robot state.

        Reads ``robot.current_yaw_rate`` when the robot exposes it (e.g.
        ``BicycleDriveRobot``); otherwise it assumes ``robot.current_speed`` is
        ``(linear_velocity, yaw_rate)``. The fallback is reserved for drive models
        that do not expose an explicit yaw-rate accessor at all. When a drive does
        expose ``current_yaw_rate`` but it is non-finite, the proxy fails closed
        rather than silently falling back to ``current_speed[1]`` (which is the
        heading angle for bicycle drive, the original #3683 mis-read).
        """
        if not getattr(self.env_config, "rollover_proxy_enabled", False):
            return None
        robot = self.simulator.robots[0]
        current_speed = getattr(robot, "current_speed", None)
        current_speed = current_speed() if callable(current_speed) else current_speed
        explicit_yaw_rate = getattr(robot, "current_yaw_rate", None)
        explicit_yaw_rate = (
            explicit_yaw_rate() if callable(explicit_yaw_rate) else explicit_yaw_rate
        )
        try:
            linear_velocity = float(current_speed[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                "rollover_proxy_enabled requires robot.current_speed as "
                "(linear_velocity, yaw_rate)."
            ) from exc
        if explicit_yaw_rate is not None:
            # The drive exposes a dedicated yaw-rate accessor: trust it as the
            # sole source. Fail closed on a non-finite value instead of falling
            # back to the heading-bearing ``current_speed[1]``.
            yaw_rate = _coerce_finite_float(explicit_yaw_rate)
            if yaw_rate is None:
                raise RuntimeError(
                    "rollover_proxy_enabled requires a finite robot.current_yaw_rate."
                )
        else:
            try:
                yaw_rate = float(current_speed[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise RuntimeError(
                    "rollover_proxy_enabled requires robot.current_yaw_rate "
                    "or robot.current_speed (linear_velocity, yaw_rate)."
                ) from exc
        return rollover_proxy_telemetry(
            linear_velocity,
            yaw_rate,
            params=getattr(self.env_config, "rollover_proxy_params", RolloverProxyParams()),
        )

    def _apply_rollover_proxy_metadata(self, reward_dict: dict[str, Any]) -> bool:
        """Add opt-in rollover proxy step metadata.

        Returns:
            True when the proxy classified this step as ``ROLLOVER_CRITICAL``.
        """
        rollover_record = self._rollover_proxy_record()
        if rollover_record is None:
            return False
        reward_dict["rollover_proxy"] = rollover_record
        reward_dict["rollover_stability_margin"] = rollover_record["stability_margin"]
        reward_dict["rollover_lateral_acceleration"] = rollover_record["lateral_acceleration"]
        reward_dict["rollover_critical_lateral_acceleration"] = rollover_record[
            "critical_lateral_acceleration"
        ]
        rollover_critical = bool(rollover_record["rollover_critical"])
        reward_dict["rollover_critical"] = rollover_critical
        if rollover_critical:
            reward_dict["termination_reason"] = "ROLLOVER_CRITICAL"
        return rollover_critical

    def _apply_rollover_proxy_reward(
        self,
        reward: float,
        reward_dict: dict[str, Any],
        *,
        rollover_critical: bool,
    ) -> float:
        """Apply configured rollover penalty when the opt-in proxy trips.

        Returns:
            Reward with the configured rollover penalty applied when critical.
        """
        if not rollover_critical:
            return reward
        penalty = float(getattr(self.env_config, "rollover_proxy_penalty", 0.0))
        # Defensively normalize reward_terms before assignment: custom reward
        # functions may leave it absent or set it to a non-dict value, mirroring
        # the guard in `_extract_reward_terms`.
        reward_terms = reward_dict.get("reward_terms")
        if not isinstance(reward_terms, dict):
            reward_terms = {}
            reward_dict["reward_terms"] = reward_terms
        reward_terms["rollover_proxy_penalty"] = penalty
        return reward + penalty

    def step(self, action):
        """Execute one environment step.

        Args:
            action: Action sampled from ``action_space`` for the controlled robot.

        Returns:
            tuple: ``(obs, reward, terminated, truncated, info)`` per Gymnasium API.
        """
        if self.debug_without_robot_movement:
            action = (0.0, 0.0)
        else:
            # Process the action through the simulator only when debug mode is disabled.
            action = self.simulator.robots[0].parse_action(action)

        # Perform simulation step
        self.simulator.step_once([action])
        # Get updated observation
        obs = self.state.step()

        # T044: Wrap observation as dict if observation space was converted for grid inclusion
        if getattr(self, "_wrap_obs_as_dict", False) and not isinstance(obs, dict):
            obs = {"agent_obs": obs}

        step_ped_positions = getattr(self.simulator, "ped_pos", np.zeros((0, 2)))

        # T044: Update occupancy grid if enabled
        if self.occupancy_grid is not None:
            obstacles, obstacle_polygons = self._get_static_grid_obstacles()
            # Extract updated pedestrian positions and radii
            ped_positions = step_ped_positions
            ped_radii = getattr(self.simulator, "ped_radii", None)
            if ped_radii is None:
                ped_radii = [0.35] * len(ped_positions)
            pedestrians = [
                (tuple(pos), radius) for pos, radius in zip(ped_positions, ped_radii, strict=True)
            ]
            # Get updated robot pose (already in RobotPose format: ((x, y), theta))
            robot_pose = self.simulator.robot_poses[0]
            # Regenerate grid (allow grid config to opt into ego frame)
            self.occupancy_grid.generate(
                obstacles=obstacles,
                pedestrians=pedestrians,
                robot_pose=robot_pose,
                ego_frame=False,
                obstacle_polygons=obstacle_polygons,
            )
            # Update observation with new grid
            if self.include_grid_in_observation:
                obs["occupancy_grid"] = self.occupancy_grid.to_observation()
                # Flatten metadata into individual fields for SB3 compatibility
                obs.update(
                    _flatten_occupancy_grid_metadata(self.occupancy_grid.metadata_observation())
                )

        obs = self._attach_asymmetric_critic_state(obs)

        # Fetch metadata about the current state
        reward_dict = self.state.meta_dict()
        reward_dict.update(
            self._snqi_proxy.compute_step_metrics(
                self.simulator,
                dt=float(self.state.d_t),
                ped_positions_override=step_ped_positions,
            )
        )
        # add the action space to dict
        rollover_critical = self._apply_rollover_proxy_metadata(reward_dict)
        reward_dict["action_space"] = self.action_space
        # add action to dict
        reward_dict["action"] = action
        # Add last_action to reward_dict
        reward_dict["last_action"] = self.last_action
        # Determine if the episode has reached terminal state
        term = self.state.is_terminal
        # Compute the reward using the provided reward function
        term = term or rollover_critical
        reward = self.reward_func(reward_dict)
        if "reward_terms" not in reward_dict:
            reward_dict["reward_terms"] = {"scalar_reward": float(reward)}
        reward = self._apply_rollover_proxy_reward(
            reward,
            reward_dict,
            rollover_critical=rollover_critical,
        )
        reward_dict["reward_total"] = float(reward)
        # Update last_action for next step
        self.last_action = action

        # Telemetry update
        self._emit_telemetry(
            reward,
            term,
            False,
            action,
            reward_dict,
            ped_positions_override=step_ped_positions,
        )
        self._latest_observation = obs

        # if recording is enabled, record the state
        if self.recording_enabled:
            self.record()

        # observation, reward, terminal, truncated,info
        info = _build_step_info(reward_dict)
        return (
            obs,
            reward,
            term,
            False,
            info,
        )

    def reset(self, seed=None, options=None):
        """Reset the environment and start a new episode.

        Args:
            seed: Optional random seed applied to Gymnasium and reset route sampling.
            options: Optional Gymnasium reset options.

        Returns:
            tuple: ``(obs, info)`` with the initial observation and placeholder info dict.
        """
        if seed is not None:
            self.applied_seed = int(seed)

        with global_reset_seed(seed):
            super().reset(seed=seed, options=options)
            self._telemetry_episode_id += 1
            # Reset last_action
            self.last_action = None
            # Reset internal simulator state
            self.simulator.reset_state()
            # Reset the environment's state and return the initial observation
            reset_episode_counter_for_seed(self.state, seed)
            obs = self.state.reset()
            self._prime_snqi_proxy_state()

            # T044: Wrap observation as dict if observation space was converted for grid inclusion
            if getattr(self, "_wrap_obs_as_dict", False) and not isinstance(obs, dict):
                obs = {"agent_obs": obs}

            # T043: Generate initial occupancy grid if enabled
            if self.occupancy_grid is not None:
                self._clear_grid_obstacle_cache()
                obstacles, obstacle_polygons = self._get_static_grid_obstacles()
                # Extract pedestrian positions and radii from simulator
                ped_positions = self.simulator.ped_pos
                ped_radii = getattr(self.simulator, "ped_radii", None)
                if ped_radii is None:
                    # Default pedestrian radius if not available
                    ped_radii = [0.35] * len(ped_positions)
                pedestrians = [
                    (tuple(pos), radius)
                    for pos, radius in zip(ped_positions, ped_radii, strict=True)
                ]
                # Get robot pose (already in RobotPose format: ((x, y), theta))
                robot_pose = self.simulator.robot_poses[0]
                # Generate grid (allow grid config to opt into ego frame)
                self.occupancy_grid.generate(
                    obstacles=obstacles,
                    pedestrians=pedestrians,
                    robot_pose=robot_pose,
                    ego_frame=False,
                    obstacle_polygons=obstacle_polygons,
                )
                # Add grid to observation
                if self.include_grid_in_observation:
                    obs["occupancy_grid"] = self.occupancy_grid.to_observation()
                    # Flatten metadata into individual fields for SB3 compatibility
                    obs.update(
                        _flatten_occupancy_grid_metadata(self.occupancy_grid.metadata_observation())
                    )
                    logger.debug(
                        f"Initial occupancy grid generated: "
                        f"obstacles={len(obstacles)}, pedestrians={len(pedestrians)}"
                    )
            obs = self._attach_asymmetric_critic_state(obs)
            self._latest_observation = obs

            # Handle recording for both systems
            if self.recording_enabled:
                if self.use_jsonl_recording:
                    # End previous episode if active, then start a new one
                    try:
                        self.end_episode_recording()
                    except (
                        RuntimeError,
                        ValueError,
                        AttributeError,
                    ):  # pragma: no cover - safe if none active
                        pass
                    config_hash = _stable_config_hash(self.env_config)
                    telemetry_path = None
                    if self._telemetry_session is not None and self.env_config.telemetry_record:
                        telemetry_path = str(self._telemetry_session.telemetry_path)
                    telemetry_episode_id = (
                        self._telemetry_episode_id if telemetry_path is not None else None
                    )
                    self.start_episode_recording(
                        config_hash=config_hash,
                        telemetry_path=telemetry_path,
                        telemetry_episode_id=telemetry_episode_id,
                    )
                else:
                    # Legacy pickle recording
                    self.save_recording()

            info = _build_reset_info(
                self.config,
                map_def=self.map_def,
                applied_seed=self.applied_seed,
            )
            # Reset telemetry timing on new episode
            self._last_wall_time = time.perf_counter()
            self._frame_idx = 0
            return obs, info

    def _emit_telemetry(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        action: Any,
        meta: dict[str, Any] | None = None,
        ped_positions_override: Any | None = None,
    ) -> None:
        """Record telemetry and update live pane if enabled."""
        if self._telemetry_session is None:
            return
        now = time.perf_counter()
        dt = max(now - self._last_wall_time, 1e-6)
        self._last_wall_time = now

        collision = bool(
            self.state.is_collision_with_ped
            or self.state.is_collision_with_obst
            or self.state.is_collision_with_robot
        )
        ped_positions = np.asarray(
            ped_positions_override
            if ped_positions_override is not None
            else getattr(self.simulator, "ped_pos", np.zeros((0, 2)))
        )
        robot_pos = np.asarray(self.simulator.robot_poses[0][0], dtype=float)
        min_ped_distance = None
        if ped_positions.size > 0:
            try:
                deltas = ped_positions - robot_pos
                dists = np.linalg.norm(deltas, axis=1)
                min_ped_distance = float(np.min(dists))
            except (TypeError, ValueError):
                min_ped_distance = None
        try:
            action_norm = float(np.linalg.norm(action))
        except (TypeError, ValueError):
            action_norm = None

        metrics = {
            "fps": 1.0 / dt if dt > 0 else None,
            "reward": float(reward) if reward is not None else None,
            "collisions": 1 if collision else 0,
            "min_ped_distance": min_ped_distance,
            "action_norm": action_norm,
        }
        status = "terminated" if terminated else ("truncated" if truncated else "running")
        payload = {
            "timestamp_ms": int(time.time() * 1000),
            "frame_idx": self._frame_idx,
            "episode_id": self._telemetry_episode_id,
            "status": status,
            "metrics": metrics,
            "reward_total": float(reward) if reward is not None else None,
        }
        if meta is not None:
            reward_terms = _extract_reward_terms(meta)
            if reward_terms:
                payload["reward_terms"] = reward_terms
            step_metrics = _extract_step_metrics(meta)
            if step_metrics:
                payload["step_metrics"] = step_metrics
        self._frame_idx += 1
        self._telemetry_session.append(payload)

    @staticmethod
    def _normalize_obstacles_for_grid(
        obstacles: list[Obstacle] | list[Line2D], bounds: list[Line2D]
    ) -> tuple[list[Line2D], list[ShapelyPolygon]]:
        """Convert obstacles/bounds into grid-friendly primitives.

        Returns:
            tuple: (line segments, polygons) where polygons preserve compound obstacle geometry.
        """
        line_segments: list[Line2D] = []
        polygons: list[ShapelyPolygon] = []

        def _add_line(line) -> None:
            """Append a malformed-tolerant line segment to the grid primitive list."""
            try:
                start, end = line
                line_segments.append((tuple(start), tuple(end)))
            except (TypeError, ValueError):
                pass  # Skip malformed lines

        for obstacle in obstacles:
            if isinstance(obstacle, Obstacle):
                polygons.extend(obstacle.iter_polygons())
                for line in obstacle.lines:
                    if len(line) == 4:
                        # Obstacle.lines stores (x1, x2, y1, y2); convert to Line2D ((x1, y1), (x2, y2))
                        x1, x2, y1, y2 = line
                        _add_line(((x1, y1), (x2, y2)))
                    else:
                        _add_line(line)
            else:
                _add_line(obstacle)

        for bound in bounds:
            _add_line(bound)

        return line_segments, polygons

    def _clear_grid_obstacle_cache(self) -> None:
        """Invalidate cached static grid geometry before a new episode begins."""

        self._grid_obstacle_cache_key = None
        self._grid_obstacle_cache_value = None

    def _get_static_grid_obstacles(self) -> _GridObstacleCacheValue:
        """Return normalized static map geometry for occupancy-grid generation.

        Map obstacle and bound objects are static during a RobotEnv episode, so step() can reuse the
        reset-time normalized primitives. The identity key still refreshes the cache if a caller
        swaps the map, obstacle list, or bounds list before the next grid generation.
        """

        cache_key = (id(self.map_def), id(self.map_def.obstacles), id(self.map_def.bounds))
        if (
            self._grid_obstacle_cache_key == cache_key
            and self._grid_obstacle_cache_value is not None
        ):
            return self._grid_obstacle_cache_value

        value = self._normalize_obstacles_for_grid(
            self.map_def.obstacles,
            self.map_def.bounds,
        )
        self._grid_obstacle_cache_key = cache_key
        self._grid_obstacle_cache_value = value
        return value

    def _prepare_visualizable_state(self):
        """Build a renderer-friendly simulation snapshot from the current environment state.

        Returns:
            VisualizableSimState: Current state payload consumed by ``SimulationView``.
        """
        action = (
            None
            if not self.last_action
            else VisualizableAction(
                self.simulator.robot_poses[0],
                self.last_action,
                self.simulator.goal_pos[0],
            )
        )

        # Robot position and LIDAR scanning visualization preparation
        robot_pos = self.simulator.robot_poses[0][0]
        distances, directions = lidar_ray_scan(
            self.simulator.robot_poses[0],
            self.state.occupancy,
            self.env_config.lidar_config,
        )

        # Construct ray vectors for visualization
        ray_vecs_np = render_lidar(robot_pos, distances, directions)

        # Prepare pedestrian action visualization
        ped_actions_np = prepare_pedestrian_actions(self.simulator)

        nav = self.simulator.robot_navs[0]
        remaining_waypoints = nav.waypoints[nav.waypoint_id :] if nav.waypoints else []
        planned_path = [robot_pos, *remaining_waypoints] if remaining_waypoints else None

        # Package the state for visualization
        state = VisualizableSimState(
            timestep=self.state.timestep,
            robot_action=action,
            robot_pose=self.simulator.robot_poses[0],
            pedestrian_positions=deepcopy(self.simulator.ped_pos),
            ray_vecs=ray_vecs_np,
            ped_actions=ped_actions_np,
            time_per_step_in_secs=self.env_config.sim_config.time_per_step_in_secs,
            planned_path=planned_path,
            observation_image=self._extract_observation_image(self._latest_observation),
        )

        return state

    def render(self):
        """
        Render the environment visually if in debug mode.

        Raises RuntimeError if debug mode is not enabled.
        """
        if not self.sim_ui:
            raise RuntimeError(
                "Render unavailable: environment was created with debug=False (no sim_ui). "
                "Recreate via make_robot_env(..., debug=True) to enable visualization and frame capture.",
            )

        state = self._prepare_visualizable_state()
        # Provide the latest occupancy grid to the renderer
        if self.sim_ui is not None:
            self.sim_ui.occupancy_grid = self.occupancy_grid

        # Execute rendering of the state through the simulation UI
        self.sim_ui.render(state)

    def record(self):
        """
        Records the current state as visualizable state and stores it in the list.
        """
        state = self._prepare_visualizable_state()

        # Use the new unified recording method
        self.record_simulation_step(state)

    def set_pedestrian_velocity_scale(self, scale: float = 1.0):
        """
        Set the pedestrian velocity visualization scaling factor.

        Args:
            scale (float): Scaling factor for pedestrian velocity arrows in visualization.
                          1.0 = actual size, 2.0 = double size for better visibility, etc.
        """
        if self.sim_ui:
            self.sim_ui.ped_velocity_scale = scale
        else:
            logger.warning("Cannot set velocity scale: debug mode not enabled")

    def get_telemetry_session(self):
        """
        Get the telemetry session for accessing recorded metrics and artifacts.

        Returns:
            TelemetrySession or None: The active telemetry session if telemetry recording is
                enabled (enable_telemetry_panel=True or telemetry_record=True), or None
                if telemetry is disabled or the session has not been initialized.

        Example:
            >>> env = make_robot_env(debug=True, enable_telemetry_panel=True, telemetry_record=True)
            >>> # ... run simulation ...
            >>> env.close()
            >>> session = env.get_telemetry_session()
            >>> if session is not None:
            ...     paths = session.write_summary()
        """
        return self._telemetry_session

    def write_telemetry_summary(self) -> tuple[Path, ...] | None:
        """Write telemetry summary artifacts if telemetry is enabled.

        Returns:
            tuple[Path, ...] | None: Paths to written artifacts when telemetry is active,
                otherwise ``None`` when telemetry is disabled.
        """
        session = self.get_telemetry_session()
        if session is None:
            return None
        return session.write_summary()
