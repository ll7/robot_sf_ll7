"""Fail-fast experimental adapter for upstream CrowdNav_HEIGHT checkpoints."""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

if TYPE_CHECKING:
    from collections.abc import Iterator


_DEFAULT_MODEL_DIR = Path("output/external_checkpoints/crowdnav_height_extracted/HEIGHT/HEIGHT")
_DEFAULT_REPO_ROOT = Path("output/repos/CrowdNav_HEIGHT")
_DEFAULT_CHECKPOINT_NAME = "237800.pt"


@dataclass(frozen=True)
class CrowdNavHeightConfig:
    """Configuration for loading one upstream CrowdNav_HEIGHT checkpoint."""

    repo_root: Path = _DEFAULT_REPO_ROOT
    model_dir: Path = _DEFAULT_MODEL_DIR
    checkpoint_name: str = _DEFAULT_CHECKPOINT_NAME
    device: str = "cpu"
    max_linear_speed: float = 0.5
    max_angular_speed: float = 1.0


def build_crowdnav_height_config(data: dict[str, Any] | None) -> CrowdNavHeightConfig:
    """Build adapter config from benchmark algo config.

    Returns:
        CrowdNavHeightConfig with repository, checkpoint, and projection limits.
    """
    payload = data or {}
    repo_root = Path(str(payload.get("repo_root", _DEFAULT_REPO_ROOT)))
    model_dir = Path(str(payload.get("model_dir", _DEFAULT_MODEL_DIR)))
    checkpoint_name = str(payload.get("checkpoint_name", _DEFAULT_CHECKPOINT_NAME))
    device = str(payload.get("device", "cpu")).strip() or "cpu"
    max_linear_speed = float(payload.get("max_linear_speed", 0.5))
    max_angular_speed = float(payload.get("max_angular_speed", 1.0))
    if max_linear_speed < 0.0 or max_angular_speed < 0.0:
        raise ValueError("max_linear_speed and max_angular_speed must be non-negative")
    return CrowdNavHeightConfig(
        repo_root=repo_root,
        model_dir=model_dir,
        checkpoint_name=checkpoint_name,
        device=device,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )


def _normalize_angle(angle: float) -> float:
    """Wrap angle to ``[-pi, pi)``.

    Returns:
        Wrapped angle in radians.
    """
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _require_array(value: Any, *, size: int, field: str) -> np.ndarray:
    """Return a required float array slice or raise a contract error."""
    arr = np.asarray([] if value is None else value, dtype=float).reshape(-1)
    if arr.size < size:
        raise ValueError(f"Missing or malformed required field: {field}")
    return arr[:size]


def _xy_rows(value: Any) -> np.ndarray:
    """Normalize arbitrary XY payloads to an ``(N, 2)`` array.

    Returns:
        Two-column float array with one row per vector.
    """
    arr = np.asarray([] if value is None else value, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    arr = arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1)
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    return arr[:, :2]


def _robot_world_to_height_robot_frame(vector_xy: np.ndarray, heading: float) -> np.ndarray:
    """Match upstream `world_to_robot()` used by CrowdNav_HEIGHT Turtlebot observations.

    Returns:
        The vector rotated into the upstream robot frame.
    """
    rot_angle = -(float(heading) - math.pi / 2.0)
    cos_a = math.cos(rot_angle)
    sin_a = math.sin(rot_angle)
    x = float(vector_xy[0])
    y = float(vector_xy[1])
    return np.array(
        [
            cos_a * x - sin_a * y,
            sin_a * x + cos_a * y,
        ],
        dtype=float,
    )


def _ray_segment_intersection_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> float | None:
    """Return the distance from a ray origin to a finite line segment, if they intersect.

    Returns:
        Intersection distance, or ``None`` if no intersection exists.
    """
    segment = seg_end - seg_start
    denom = direction[0] * segment[1] - direction[1] * segment[0]
    if abs(denom) <= 1e-9:
        return None
    delta = seg_start - origin
    t = (delta[0] * segment[1] - delta[1] * segment[0]) / denom
    u = (delta[0] * direction[1] - delta[1] * direction[0]) / denom
    if t < 0.0 or not (0.0 <= u <= 1.0):
        return None
    return float(t)


def _load_config_class(config_path: Path) -> Any:
    """Load the checkpoint-side config module without mutating the package search path globally.

    Returns:
        The upstream ``Config`` class from the checkpoint bundle.
    """
    module_name = f"_crowdnav_height_checkpoint_config_{abs(hash(str(config_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        config_class = getattr(module, "Config", None)
    finally:
        sys.modules.pop(module_name, None)
    if config_class is None:
        raise AttributeError(f"Config class not found in {config_path}")
    return config_class


@contextmanager
def _height_import_context(repo_root: Path) -> Iterator[None]:  # noqa: C901, PLR0915
    """Temporarily expose the upstream checkout and clear conflicting cached modules."""
    repo_str = str(repo_root)
    original_path = list(sys.path)
    prefixes = ("crowd_nav", "crowd_sim", "training")
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    sys.path.insert(0, repo_str)
    try:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                sys.modules.pop(name, None)
        if "gym" not in sys.modules:
            sys.modules["gym"] = gymnasium
            sys.modules["gym.spaces"] = gymnasium.spaces
            sys.modules["gym.spaces.box"] = importlib.import_module("gymnasium.spaces.box")
            sys.modules["gym.spaces.dict"] = importlib.import_module("gymnasium.spaces.dict")
        if "baselines" not in sys.modules:
            baselines = ModuleType("baselines")
            bench = ModuleType("baselines.bench")

            class _Monitor:
                def __init__(self, env, *_args, **_kwargs):
                    self.env = env

                def __getattr__(self, name):
                    return getattr(self.env, name)

            bench.Monitor = _Monitor
            common = ModuleType("baselines.common")
            atari_wrappers = ModuleType("baselines.common.atari_wrappers")
            atari_wrappers.make_atari = lambda env_id: env_id
            atari_wrappers.wrap_deepmind = lambda env: env
            vec_env = ModuleType("baselines.common.vec_env")

            class _VecEnvWrapper:
                def __init__(self, venv, observation_space=None):
                    self.venv = venv
                    self.observation_space = observation_space

            class _VecEnv:
                def __init__(self, num_envs, observation_space, action_space):
                    self.num_envs = num_envs
                    self.observation_space = observation_space
                    self.action_space = action_space

            class _CloudpickleWrapper:
                def __init__(self, x):
                    self.x = x

            def _clear_mpi_env_vars():
                return nullcontext()

            class _DummyVecEnv:
                def __init__(self, env_fns):
                    self.envs = [fn() for fn in env_fns]
                    self.num_envs = len(self.envs)
                    self.observation_space = self.envs[0].observation_space
                    self.action_space = self.envs[0].action_space

                def reset(self):
                    return self.envs[0].reset()

                def step_async(self, _actions):
                    return None

                def step_wait(self):
                    raise NotImplementedError

            vec_env.VecEnvWrapper = _VecEnvWrapper
            vec_env.VecEnv = _VecEnv
            vec_env.CloudpickleWrapper = _CloudpickleWrapper
            vec_env.clear_mpi_env_vars = _clear_mpi_env_vars
            dummy_vec_env = ModuleType("baselines.common.vec_env.dummy_vec_env")
            dummy_vec_env.DummyVecEnv = _DummyVecEnv
            vec_env.dummy_vec_env = dummy_vec_env
            vec_normalize = ModuleType("baselines.common.vec_env.vec_normalize")

            class _VecNormalize:
                def __init__(self, *args, **kwargs):
                    del args, kwargs
                    self.training = True

            vec_normalize.VecNormalize = _VecNormalize
            util = ModuleType("baselines.common.vec_env.util")

            def _obs_to_dict(obs):
                return obs

            def _dict_to_obs(obs):
                return obs

            def _obs_space_info(space):
                if hasattr(space, "spaces"):
                    keys = list(space.spaces.keys())
                    shapes = {k: space.spaces[k].shape for k in keys}
                    dtypes = {k: space.spaces[k].dtype for k in keys}
                    return keys, shapes, dtypes
                return [], {}, {}

            util.obs_to_dict = _obs_to_dict
            util.dict_to_obs = _dict_to_obs
            util.obs_space_info = _obs_space_info
            common.atari_wrappers = atari_wrappers
            common.vec_env = vec_env
            baselines.bench = bench
            baselines.common = common
            baselines.logger = ModuleType("baselines.logger")
            baselines.logger.log = lambda *args, **kwargs: None
            baselines.logger.warn = lambda *args, **kwargs: None
            baselines.logger.scoped_configure = lambda *args, **kwargs: nullcontext()
            sys.modules["baselines"] = baselines
            sys.modules["baselines.bench"] = bench
            sys.modules["baselines.common"] = common
            sys.modules["baselines.common.atari_wrappers"] = atari_wrappers
            sys.modules["baselines.common.vec_env"] = vec_env
            sys.modules["baselines.common.vec_env.dummy_vec_env"] = dummy_vec_env
            sys.modules["baselines.common.vec_env.vec_normalize"] = vec_normalize
            sys.modules["baselines.common.vec_env.vec_env"] = vec_env
            sys.modules["baselines.common.vec_env.util"] = util
            sys.modules["baselines.logger"] = baselines.logger
        if "torchvision" not in sys.modules:
            torchvision = ModuleType("torchvision")
            torchvision_models = ModuleType("torchvision.models")
            torchvision.models = torchvision_models  # type: ignore[attr-defined]
            torchvision_models.resnet18 = lambda *args, **kwargs: None
            torchvision_models.resnet34 = lambda *args, **kwargs: None
            torchvision_models.resnet50 = lambda *args, **kwargs: None
            sys.modules["torchvision"] = torchvision
            sys.modules["torchvision.models"] = torchvision_models
        training_root = repo_root / "training"
        if training_root.exists() and "training" not in sys.modules:
            training_pkg = ModuleType("training")
            training_pkg.__path__ = [str(training_root)]  # type: ignore[attr-defined]
            sys.modules["training"] = training_pkg
        yield
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_path


class CrowdNavHeightAdapter:
    """Stateful model-only adapter around one upstream CrowdNav_HEIGHT checkpoint."""

    projection_policy = "upstream_discrete_delta_vw_to_unicycle_vw_stateful"
    upstream_policy = "training.networks.model.Policy[selfAttn_merge_srnn_lidar]"
    _LINEAR_VELOCITY_INDEX = 0
    _ANGULAR_VELOCITY_INDEX = 1

    def __init__(self, config: CrowdNavHeightConfig | None = None) -> None:
        """Initialize the adapter and load the upstream checkpoint."""
        self.config = config or CrowdNavHeightConfig()
        self.repo_root = self.config.repo_root.resolve()
        self.model_dir = self.config.model_dir.resolve()
        self.checkpoint_path = (
            self.model_dir / "checkpoints" / self.config.checkpoint_name
        ).resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(
                "CrowdNav_HEIGHT checkout not found: "
                f"{self.config.repo_root}. Clone the upstream repo under output/repos/ first."
            )
        if not self.model_dir.exists():
            raise FileNotFoundError(
                "CrowdNav_HEIGHT model directory not found: "
                f"{self.config.model_dir}. Extract a checkpoint bundle under output/external_checkpoints/ first."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "CrowdNav_HEIGHT checkpoint not found: "
                f"{self.checkpoint_path}. Download/extract the upstream model bundle first."
            )

        config_path = self.model_dir / "configs" / "config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"CrowdNav_HEIGHT checkpoint config missing: {config_path}")
        config_class = _load_config_class(config_path)
        self._checkpoint_config = config_class()
        self._device = torch.device(self.config.device)
        if hasattr(self._checkpoint_config, "training") and hasattr(
            self._checkpoint_config.training, "cuda"
        ):
            self._checkpoint_config.training.cuda = self._device.type == "cuda"
        if hasattr(self._checkpoint_config, "training") and hasattr(
            self._checkpoint_config.training, "cuda_deterministic"
        ):
            self._checkpoint_config.training.cuda_deterministic = False
        self._obstacle_segments = np.zeros((0, 4), dtype=float)
        self._desired_velocity = np.zeros(2, dtype=float)
        self._last_model_inputs: dict[str, np.ndarray] | None = None
        self._mask = torch.zeros((1, 1), dtype=torch.float32, device=self._device)
        self._hidden_state: dict[str, torch.Tensor] | None = None
        self._action_table = {
            0: np.array([0.05, 0.1], dtype=float),
            1: np.array([0.05, 0.0], dtype=float),
            2: np.array([0.05, -0.1], dtype=float),
            3: np.array([0.0, 0.1], dtype=float),
            4: np.array([0.0, 0.0], dtype=float),
            5: np.array([0.0, -0.1], dtype=float),
            6: np.array([-0.05, 0.1], dtype=float),
            7: np.array([-0.05, 0.0], dtype=float),
            8: np.array([-0.05, -0.1], dtype=float),
        }

        observation_space = self._build_observation_space()
        with _height_import_context(self.repo_root):
            policy_mod = importlib.import_module("training.networks.model")
            self._policy = policy_mod.Policy(
                observation_space.spaces,
                spaces.Discrete(len(self._action_table)),
                base=self._checkpoint_config.robot.policy,
                base_kwargs=self._checkpoint_config,
            )
        state_dict = torch.load(self.checkpoint_path, map_location=self._device)
        self._policy.load_state_dict(state_dict)
        self._policy.base.nenv = 1
        self._policy.to(self._device)
        self._policy.eval()
        self.reset()

    def _build_observation_space(self) -> spaces.Dict:
        """Mirror the upstream dict observation layout required by the checkpoint.

        Returns:
            Gymnasium dict space matching the checkpoint input contract.
        """
        max_humans = int(
            self._checkpoint_config.sim.human_num + self._checkpoint_config.sim.human_num_range
        )
        max_obs = int(
            self._checkpoint_config.sim.static_obs_num
            + self._checkpoint_config.sim.static_obs_num_range
        )
        ray_num = int(360.0 / float(self._checkpoint_config.lidar.angular_res))
        return spaces.Dict(
            {
                "robot_node": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, 5),
                    dtype=np.float32,
                ),
                "temporal_edges": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, 2),
                    dtype=np.float32,
                ),
                "spatial_edges": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max(1, max_humans), 4),
                    dtype=np.float32,
                ),
                "detected_human_num": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "obstacle_vertices": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max(1, max_obs), 8),
                    dtype=np.float32,
                ),
                "obstacle_num": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "point_clouds": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, ray_num),
                    dtype=np.float32,
                ),
            }
        )

    def bind_env(self, env: Any) -> None:
        """Cache static obstacle line segments from the live Robot SF environment."""
        simulator = getattr(env, "simulator", None)
        if simulator is None or not hasattr(simulator, "get_obstacle_lines"):
            raise RuntimeError(
                "CrowdNav_HEIGHT adapter requires an environment with simulator.get_obstacle_lines()."
            )
        self.bind_obstacle_segments(simulator.get_obstacle_lines())

    def bind_obstacle_segments(self, segments: Any) -> None:
        """Bind obstacle segments explicitly for tests or direct use outside the benchmark runner."""
        arr = np.asarray([] if segments is None else segments, dtype=float)
        if arr.size == 0:
            self._obstacle_segments = np.zeros((0, 4), dtype=float)
            return
        self._obstacle_segments = arr.reshape(-1, 4)[:, :4]

    def reset(self, seed: int | None = None) -> None:
        """Reset recurrent state and upstream Turtlebot desired velocities."""
        del seed
        hidden_size = int(self._checkpoint_config.SRNN.human_node_rnn_size)
        self._hidden_state = {
            "rnn": torch.zeros((1, 1, hidden_size), dtype=torch.float32, device=self._device)
        }
        self._mask = torch.zeros((1, 1), dtype=torch.float32, device=self._device)
        self._desired_velocity = np.zeros(2, dtype=float)

    def _extract_socnav_fields(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, int]:
        """Extract Robot SF structured observation fields required by the upstream checkpoint.

        Returns:
            Parsed robot, goal, pedestrian, and time-step fields.
        """
        robot = observation.get("robot", {})
        goal = observation.get("goal", {})
        pedestrians = observation.get("pedestrians", {})
        if robot or goal or pedestrians:
            robot_pos = _require_array(robot.get("position"), size=2, field="robot.position")
            goal_pos = _require_array(goal.get("current"), size=2, field="goal.current")
            heading = float(_require_array(robot.get("heading"), size=1, field="robot.heading")[0])
            velocity_xy = _require_array(
                robot.get("velocity_xy"),
                size=2,
                field="robot.velocity_xy",
            )
            robot_radius = float(
                _require_array(robot.get("radius"), size=1, field="robot.radius")[0]
            )
            ped_positions = _xy_rows(pedestrians.get("positions"))
            ped_velocities_robot = _xy_rows(pedestrians.get("velocities"))
            ped_count_arr = np.asarray(
                pedestrians.get("count", [ped_positions.shape[0]]),
                dtype=float,
            ).reshape(-1)
            ped_count = int(ped_count_arr[0]) if ped_count_arr.size else int(ped_positions.shape[0])
            ped_count = max(
                0, min(ped_count, ped_positions.shape[0], ped_velocities_robot.shape[0])
            )
            ped_positions = ped_positions[:ped_count]
            ped_velocities_robot = ped_velocities_robot[:ped_count]
            return (
                robot_pos,
                heading,
                goal_pos,
                velocity_xy,
                robot_radius,
                ped_positions,
                ped_velocities_robot,
                ped_count,
            )

        robot_pos = _require_array(
            observation.get("robot_position"), size=2, field="robot_position"
        )
        goal_pos = _require_array(observation.get("goal_current"), size=2, field="goal_current")
        heading = float(
            _require_array(observation.get("robot_heading"), size=1, field="robot_heading")[0]
        )
        velocity_xy = _require_array(
            observation.get("robot_velocity_xy"),
            size=2,
            field="robot_velocity_xy",
        )
        robot_radius = float(
            _require_array(observation.get("robot_radius"), size=1, field="robot_radius")[0]
        )
        ped_positions = _xy_rows(observation.get("pedestrians_positions"))
        ped_velocities_robot = _xy_rows(observation.get("pedestrians_velocities"))
        ped_count_arr = np.asarray(
            observation.get("pedestrians_count", [ped_positions.shape[0]]),
            dtype=float,
        ).reshape(-1)
        ped_count = int(ped_count_arr[0]) if ped_count_arr.size else int(ped_positions.shape[0])
        ped_count = max(0, min(ped_count, ped_positions.shape[0], ped_velocities_robot.shape[0]))
        ped_positions = ped_positions[:ped_count]
        ped_velocities_robot = ped_velocities_robot[:ped_count]
        return (
            robot_pos,
            heading,
            goal_pos,
            velocity_xy,
            robot_radius,
            ped_positions,
            ped_velocities_robot,
            ped_count,
        )

    def _build_spatial_edges(
        self,
        robot_pos: np.ndarray,
        heading: float,
        ped_positions: np.ndarray,
        ped_velocities_robot: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Rebuild the upstream sorted human-edge tensor in the Height robot frame.

        Returns:
            Tuple of spatial edge tensor and detected human count.
        """
        max_humans = int(
            self._checkpoint_config.sim.human_num + self._checkpoint_config.sim.human_num_range
        )
        spatial = np.full((max(1, max_humans), 4), 15.0, dtype=np.float32)
        if ped_positions.size == 0:
            return spatial, 1
        rows: list[np.ndarray] = []
        for ped_pos, ped_vel in zip(ped_positions, ped_velocities_robot, strict=False):
            rel_world = np.asarray(ped_pos, dtype=float) - np.asarray(robot_pos, dtype=float)
            rel_robot = _robot_world_to_height_robot_frame(rel_world, heading)
            rows.append(
                np.array(
                    [
                        float(rel_robot[0]),
                        float(rel_robot[1]),
                        float(ped_vel[0]),
                        float(ped_vel[1]),
                    ],
                    dtype=np.float32,
                )
            )
        rows.sort(key=lambda row: float(np.linalg.norm(row[:2])))
        used = min(len(rows), spatial.shape[0])
        if used:
            spatial[:used] = np.stack(rows[:used], axis=0)
        return spatial, max(1, used)

    def _raycast_obstacles(self, robot_pos: np.ndarray, heading: float) -> np.ndarray:
        """Approximate the upstream obstacle-only lidar scan from static obstacle line segments.

        Returns:
            One lidar range value per angular ray.
        """
        if self._obstacle_segments is None or self._obstacle_segments.size == 0:
            raise RuntimeError("CrowdNav_HEIGHT obstacle segments are not initialized.")
        sensor_range = float(self._checkpoint_config.lidar.sensor_range)
        angular_res = float(self._checkpoint_config.lidar.angular_res)
        ray_num = int(360.0 / angular_res)
        distances = np.full((ray_num,), sensor_range, dtype=np.float32)
        origin = np.asarray(robot_pos, dtype=float)
        ray_angles = np.linspace(0.0, 2.0 * math.pi, ray_num, endpoint=False, dtype=float)
        angles = heading + ray_angles
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        segments = self._obstacle_segments
        for idx, direction in enumerate(directions):
            best = sensor_range
            for seg in segments:
                hit = _ray_segment_intersection_distance(
                    origin,
                    direction,
                    seg[:2],
                    seg[2:4],
                )
                if hit is not None and hit < best:
                    best = hit
            distances[idx] = float(min(best, sensor_range))
        return distances

    def _build_model_inputs(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reconstruct the upstream dict observation and convert it to batched torch tensors.

        Returns:
            Batched torch tensors and a metadata dict for tracing.
        """
        has_nested = "robot" in observation or "goal" in observation or "pedestrians" in observation
        has_flat = "robot_position" in observation or "goal_current" in observation
        if not has_nested and not has_flat:
            raise ValueError(
                "CrowdNav_HEIGHT requires either nested SocNav observations or flat benchmark observations."
            )
        if self._obstacle_segments is None or self._obstacle_segments.size == 0:
            raise RuntimeError(
                "CrowdNav_HEIGHT requires obstacle segments; bind the live env first."
            )
        (
            robot_pos,
            heading,
            goal_pos,
            velocity_xy,
            _robot_radius,
            ped_positions,
            ped_velocities_robot,
            ped_count,
        ) = self._extract_socnav_fields(observation)
        spatial_edges, detected_human_num = self._build_spatial_edges(
            robot_pos,
            heading,
            ped_positions,
            ped_velocities_robot,
        )
        point_clouds = self._raycast_obstacles(robot_pos, heading)
        max_obs = int(
            self._checkpoint_config.sim.static_obs_num
            + self._checkpoint_config.sim.static_obs_num_range
        )
        obstacle_vertices = np.full((max(1, max_obs), 8), 15.0, dtype=np.float32)
        obstacle_num = 0.0
        payload = {
            "robot_node": np.asarray(
                [
                    [
                        float(robot_pos[0]),
                        float(robot_pos[1]),
                        float(goal_pos[0]),
                        float(goal_pos[1]),
                        heading,
                    ]
                ],
                dtype=np.float32,
            ),
            "temporal_edges": np.asarray(
                [[float(velocity_xy[0]), float(velocity_xy[1])]],
                dtype=np.float32,
            ),
            "spatial_edges": spatial_edges.astype(np.float32),
            "detected_human_num": np.asarray([float(detected_human_num)], dtype=np.float32),
            "obstacle_vertices": obstacle_vertices,
            "obstacle_num": np.asarray([float(obstacle_num)], dtype=np.float32),
            "point_clouds": np.asarray([point_clouds], dtype=np.float32),
        }
        self._last_model_inputs = {
            key: np.array(value, copy=True) for key, value in payload.items()
        }
        tensors = {
            key: torch.from_numpy(value).unsqueeze(0).to(self._device)
            for key, value in payload.items()
        }
        meta = {
            "detected_human_num": int(detected_human_num),
            "human_count": int(ped_count),
            "lidar_min_range_m": float(np.min(point_clouds)) if point_clouds.size else 0.0,
            "checkpoint_path": str(self.checkpoint_path),
        }
        return tensors, meta

    def act(
        self, observation: dict[str, Any], *, time_step: float
    ) -> tuple[float, float, dict[str, Any]]:
        """Run one upstream inference step and return a fail-fast experimental `(v, w)` command.

        Returns:
            Projected linear/angular command plus debug metadata.
        """
        if self._hidden_state is None:
            self.reset()
        obs_tensors, meta = self._build_model_inputs(observation)
        assert self._hidden_state is not None
        with torch.no_grad():
            _value, action, _log_prob, self._hidden_state = self._policy.act(
                obs_tensors,
                self._hidden_state,
                self._mask,
                deterministic=True,
            )
        self._mask = torch.ones((1, 1), dtype=torch.float32, device=self._device)
        action_idx = int(action.reshape(-1)[0].item())
        if action_idx not in self._action_table:
            raise ValueError(f"Unexpected CrowdNav_HEIGHT discrete action index: {action_idx}")
        delta_v, delta_theta = self._action_table[action_idx]
        robot_cfg = self._checkpoint_config.robot
        linear_min = max(float(robot_cfg.v_min), -self.config.max_linear_speed)
        linear_max = min(float(robot_cfg.v_max), self.config.max_linear_speed)
        angular_min = max(float(robot_cfg.w_min), -self.config.max_angular_speed)
        angular_max = min(float(robot_cfg.w_max), self.config.max_angular_speed)
        if linear_min > linear_max or angular_min > angular_max:
            raise ValueError(
                "CrowdNav_HEIGHT projection limits are inconsistent with the checkpoint robot limits"
            )
        self._desired_velocity[0] = float(
            np.clip(
                self._desired_velocity[self._LINEAR_VELOCITY_INDEX] + delta_v,
                linear_min,
                linear_max,
            )
        )
        self._desired_velocity[1] = float(
            np.clip(
                self._desired_velocity[self._ANGULAR_VELOCITY_INDEX] + delta_theta,
                angular_min,
                angular_max,
            )
        )
        linear = float(self._desired_velocity[self._LINEAR_VELOCITY_INDEX])
        angular = float(self._desired_velocity[self._ANGULAR_VELOCITY_INDEX])
        meta.update(
            {
                "action_index": action_idx,
                "upstream_delta_command": [float(delta_v), float(delta_theta)],
                "projected_command_vw": [linear, angular],
                "projection_policy": self.projection_policy,
            }
        )
        return linear, angular, meta

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Map-runner entrypoint that returns only the projected `(v, w)` command.

        Returns:
            Projected linear/angular command pair.
        """
        dt_source = observation.get("dt", 0.1)
        if "sim" in observation and isinstance(observation["sim"], dict):
            dt_source = observation["sim"].get("timestep", dt_source)
        dt = float(np.asarray(dt_source, dtype=float).reshape(-1)[0])
        linear, angular, _meta = self.act(observation, time_step=dt)
        return linear, angular


__all__ = [
    "CrowdNavHeightAdapter",
    "CrowdNavHeightConfig",
    "build_crowdnav_height_config",
]
