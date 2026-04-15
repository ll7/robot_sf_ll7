"""Fail-fast experimental adapter for upstream SoNIC/CrowdNav-family checkpoints."""

from __future__ import annotations

import importlib
import math
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

if TYPE_CHECKING:
    from collections.abc import Iterator


_DEFAULT_REPO_ROOT = Path("output/repos/SoNIC-Social-Nav")
_DEFAULT_MODEL_NAME = "SoNIC_GST"
_DEFAULT_CHECKPOINT_NAME = "05207.pt"


@dataclass(frozen=True)
class SonicCrowdNavConfig:
    """Configuration for one upstream SoNIC model-only checkpoint wrapper."""

    repo_root: Path = _DEFAULT_REPO_ROOT
    model_name: str = _DEFAULT_MODEL_NAME
    checkpoint_name: str = _DEFAULT_CHECKPOINT_NAME
    device: str = "cpu"
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0


def build_sonic_crowdnav_config(data: dict[str, Any] | None) -> SonicCrowdNavConfig:
    """Build adapter config from benchmark algo config.

    Returns:
        SonicCrowdNavConfig with repository, model, checkpoint, and projection limits.
    """
    payload = data or {}
    repo_root = Path(str(payload.get("repo_root", _DEFAULT_REPO_ROOT)))
    model_name = str(payload.get("model_name", _DEFAULT_MODEL_NAME)).strip() or _DEFAULT_MODEL_NAME
    checkpoint_name = (
        str(payload.get("checkpoint_name", _DEFAULT_CHECKPOINT_NAME)).strip()
        or _DEFAULT_CHECKPOINT_NAME
    )
    device = str(payload.get("device", "cpu")).strip() or "cpu"
    max_linear_speed = float(payload.get("max_linear_speed", 1.0))
    max_angular_speed = float(payload.get("max_angular_speed", 1.0))
    if max_linear_speed < 0.0 or max_angular_speed < 0.0:
        raise ValueError("max_linear_speed and max_angular_speed must be non-negative")
    return SonicCrowdNavConfig(
        repo_root=repo_root,
        model_name=model_name,
        checkpoint_name=checkpoint_name,
        device=device,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``.

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
    if arr.ndim == 1:
        if arr.size < 2:
            return np.zeros((0, 2), dtype=float)
        if arr.size == 2:
            return arr.reshape(1, 2)
        if arr.size % 2 != 0:
            return np.zeros((0, 2), dtype=float)
        return arr.reshape(-1, 2)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    return arr[:, :2]


def _extract_socnav_fields(
    observation: dict[str, Any],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, int]:
    """Extract Robot SF structured observation fields for SoNIC translation.

    Returns:
        Parsed robot, goal, and pedestrian fields.
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
        robot_radius = float(_require_array(robot.get("radius"), size=1, field="robot.radius")[0])
        ped_positions = _xy_rows(pedestrians.get("positions"))
        ped_velocities = _xy_rows(pedestrians.get("velocities"))
        ped_count_arr = np.asarray(
            pedestrians.get("count", [ped_positions.shape[0]]),
            dtype=float,
        ).reshape(-1)
        ped_count = int(ped_count_arr[0]) if ped_count_arr.size else int(ped_positions.shape[0])
        ped_count = max(0, min(ped_count, ped_positions.shape[0], ped_velocities.shape[0]))
        ped_positions = ped_positions[:ped_count]
        ped_velocities = ped_velocities[:ped_count]
        return (
            robot_pos,
            heading,
            goal_pos,
            velocity_xy,
            robot_radius,
            ped_positions,
            ped_velocities,
            ped_count,
        )

    robot_pos = _require_array(observation.get("robot_position"), size=2, field="robot_position")
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
    ped_velocities = _xy_rows(observation.get("pedestrians_velocities"))
    ped_count_arr = np.asarray(
        observation.get("pedestrians_count", [ped_positions.shape[0]]),
        dtype=float,
    ).reshape(-1)
    ped_count = int(ped_count_arr[0]) if ped_count_arr.size else int(ped_positions.shape[0])
    ped_count = max(0, min(ped_count, ped_positions.shape[0], ped_velocities.shape[0]))
    ped_positions = ped_positions[:ped_count]
    ped_velocities = ped_velocities[:ped_count]
    return (
        robot_pos,
        heading,
        goal_pos,
        velocity_xy,
        robot_radius,
        ped_positions,
        ped_velocities,
        ped_count,
    )


@contextmanager
def _sonic_import_context(repo_root: Path) -> Iterator[None]:
    """Temporarily expose the upstream checkout with minimal import shims."""
    repo_str = str(repo_root)
    original_path = list(sys.path)
    original_modules = dict(sys.modules)
    sys.path.insert(0, repo_str)
    try:
        sys.modules["gym"] = gymnasium
        fake_envs = types.ModuleType("rl.networks.envs")

        class VecNormalize:
            """Minimal stub to satisfy SoNIC imports during model-only inference."""

        fake_envs.VecNormalize = VecNormalize
        sys.modules["rl.networks.envs"] = fake_envs
        yield
    finally:
        current_keys = list(sys.modules.keys())
        for key in current_keys:
            if key not in original_modules:
                sys.modules.pop(key, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_path


def _load_model_modules(model_name: str) -> tuple[Any, Any, Any, Any]:
    """Import SoNIC training arguments, config, model module, and parsed args safely.

    Returns:
        Imported modules plus parsed upstream CLI arguments.
    """
    original_argv = sys.argv[:]
    try:
        sys.argv = [f"{model_name}_arguments"]
        args_mod = importlib.import_module(f"trained_models.{model_name}.arguments")
        config_mod = importlib.import_module(f"trained_models.{model_name}.configs.config")
        model_mod = importlib.import_module("rl.networks.model")
        args = args_mod.get_args()
    finally:
        sys.argv = original_argv
    return args_mod, config_mod, model_mod, args


class SonicCrowdNavAdapter:
    """Stateful model-only adapter around one upstream SoNIC checkpoint."""

    projection_policy = "heading_safe_velocity_to_unicycle_vw"
    upstream_policy = "rl.networks.model.Policy[selfAttn_merge_srnn]"
    translation_policy = "constant_velocity_future_extrapolation"
    parity_gaps = (
        "No upstream CrowdSim wrapper/runtime; model-only inference shim is used.",
        "Future pedestrian trajectories are approximated via constant-velocity extrapolation.",
        "Conformity scores are zeroed because the upstream predictor side-channel is absent.",
    )

    def __init__(self, config: SonicCrowdNavConfig | None = None) -> None:
        """Initialize adapter and load one upstream checkpoint with fail-fast preflight."""
        self.config = config or SonicCrowdNavConfig()
        self.repo_root = self.config.repo_root.resolve()
        self.model_name = self.config.model_name
        self.model_root = (self.repo_root / "trained_models" / self.model_name).resolve()
        self.checkpoint_path = (
            self.model_root / "checkpoints" / self.config.checkpoint_name
        ).resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(
                "SoNIC-Social-Nav checkout not found: "
                f"{self.config.repo_root}. Clone the upstream repo under output/repos/ first."
            )
        if not self.model_root.exists():
            raise FileNotFoundError(
                "SoNIC model directory not found: "
                f"{self.model_root}. Expected trained_models/<model_name> in the checkout."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "SoNIC checkpoint not found: "
                f"{self.checkpoint_path}. Download or select an available upstream checkpoint."
            )

        self._device = torch.device(self.config.device)
        with _sonic_import_context(self.repo_root):
            _args_mod, config_mod, model_mod, args = _load_model_modules(self.model_name)
            self._args = args
            self._checkpoint_config = config_mod.Config()
            self._args.num_processes = 1
            self._args.no_cuda = True
            self._args.cuda = False
            self._checkpoint_config.policy.constant_std = False
            if (
                str(getattr(self._checkpoint_config.action_space, "kinematics", "")).lower()
                != "holonomic"
            ):
                raise RuntimeError(
                    "SoNIC wrapper currently supports only holonomic upstream checkpoints. "
                    f"Found kinematics={self._checkpoint_config.action_space.kinematics!r}."
                )
            self._policy = model_mod.Policy(
                self._build_observation_space().spaces,
                spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float),
                self._checkpoint_config,
                base=self._checkpoint_config.robot.policy,
                base_kwargs=self._args,
            )
        state = torch.load(self.checkpoint_path, map_location=self._device)
        missing, unexpected = self._policy.load_state_dict(state, strict=False)
        self._missing_state_keys = list(missing)
        self._unexpected_state_keys = list(unexpected)
        self._policy.to(self._device)
        self._policy.eval()
        self._mask = torch.zeros((1, 1), dtype=torch.float32, device=self._device)
        self._hidden_state: dict[str, torch.Tensor] | None = None
        self.reset()

    def _build_observation_space(self) -> spaces.Dict:
        """Build minimal model-only observation space matching upstream contract.

        Returns:
            Gymnasium dict space for upstream policy forward calls.
        """
        human_num = int(
            self._checkpoint_config.sim.human_num + self._checkpoint_config.sim.human_num_range
        )
        predict_steps = int(self._checkpoint_config.sim.predict_steps)
        spatial_edge_dim = int(2 * (predict_steps + 1))
        return spaces.Dict(
            {
                "robot_node": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32),
                "temporal_edges": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, 2),
                    dtype=np.float32,
                ),
                "spatial_edges": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max(1, human_num), spatial_edge_dim),
                    dtype=np.float32,
                ),
                "conformity_scores": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max(1, human_num), predict_steps),
                    dtype=np.float32,
                ),
                "visible_masks": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(max(1, human_num),),
                    dtype=np.float32,
                ),
                "detected_human_num": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "aggressiveness_factor": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, 1),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, seed: int | None = None) -> None:
        """Reset recurrent state and masks for one deterministic rollout."""
        del seed
        self._hidden_state = {
            "human_node_rnn": torch.zeros(
                (1, 1, self._policy.base.human_node_rnn_size),
                dtype=torch.float32,
                device=self._device,
            ),
            "human_human_edge_rnn": torch.zeros(
                (
                    1,
                    1 + int(self._checkpoint_config.sim.human_num),
                    self._policy.base.human_human_edge_rnn_size,
                ),
                dtype=torch.float32,
                device=self._device,
            ),
        }
        self._mask = torch.zeros((1, 1), dtype=torch.float32, device=self._device)

    def _build_model_inputs(
        self,
        observation: dict[str, Any],
        *,
        time_step: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Translate Robot SF observation into the SoNIC policy input contract.

        Returns:
            Batched torch tensors plus metadata for traceability.
        """
        (
            robot_pos,
            heading,
            goal_pos,
            velocity_xy,
            robot_radius,
            ped_positions,
            ped_velocities,
            ped_count,
        ) = _extract_socnav_fields(observation)
        human_num = int(
            self._checkpoint_config.sim.human_num + self._checkpoint_config.sim.human_num_range
        )
        predict_steps = int(self._checkpoint_config.sim.predict_steps)
        spatial_edge_dim = int(2 * (predict_steps + 1))

        robot_node = np.asarray(
            [
                [
                    float(robot_pos[0]),
                    float(robot_pos[1]),
                    float(robot_radius),
                    float(goal_pos[0]),
                    float(goal_pos[1]),
                    float(self._checkpoint_config.robot.v_pref),
                    float(heading),
                ]
            ],
            dtype=np.float32,
        )
        temporal_edges = np.asarray(
            [[float(velocity_xy[0]), float(velocity_xy[1])]],
            dtype=np.float32,
        )
        spatial_edges = np.full((max(1, human_num), spatial_edge_dim), 15.0, dtype=np.float32)
        conformity_scores = np.zeros((max(1, human_num), predict_steps), dtype=np.float32)
        visible_masks = np.zeros((max(1, human_num),), dtype=np.float32)

        used = min(ped_count, spatial_edges.shape[0])
        for i in range(used):
            rel = np.asarray(ped_positions[i], dtype=float) - np.asarray(robot_pos, dtype=float)
            vel = np.asarray(ped_velocities[i], dtype=float)
            seq: list[float] = []
            for step_idx in range(predict_steps + 1):
                horizon_rel = rel + vel * (float(step_idx) * float(time_step))
                seq.extend([float(horizon_rel[0]), float(horizon_rel[1])])
            spatial_edges[i] = np.asarray(seq, dtype=np.float32)
            visible_masks[i] = 1.0

        if used > 0:
            distances = np.linalg.norm(spatial_edges[:, :2], axis=1)
            order = np.argsort(distances, kind="stable")
            spatial_edges = spatial_edges[order]
            conformity_scores = conformity_scores[order]
            visible_masks = visible_masks[order]

        detected_human_num = max(1, used)
        payload = {
            "robot_node": robot_node,
            "temporal_edges": temporal_edges,
            "spatial_edges": spatial_edges,
            "conformity_scores": conformity_scores,
            "visible_masks": visible_masks,
            "detected_human_num": np.asarray([float(detected_human_num)], dtype=np.float32),
            "aggressiveness_factor": np.zeros((1, 1), dtype=np.float32),
        }
        tensors = {
            key: torch.from_numpy(value).unsqueeze(0).to(self._device)
            for key, value in payload.items()
        }
        meta = {
            "checkpoint_path": str(self.checkpoint_path),
            "model_name": self.model_name,
            "detected_human_num": int(detected_human_num),
            "human_count": int(used),
            "translation_policy": self.translation_policy,
        }
        return tensors, meta

    def act(
        self,
        observation: dict[str, Any],
        *,
        time_step: float,
    ) -> tuple[float, float, dict[str, Any]]:
        """Run one upstream inference step and project ActionXY into `(v, w)`.

        Returns:
            Projected linear/angular command and rich metadata.
        """
        if self._hidden_state is None:
            self.reset()
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError(f"Invalid SoNIC time_step: {time_step}")
        obs_tensors, meta = self._build_model_inputs(observation, time_step=float(time_step))
        assert self._hidden_state is not None
        with torch.no_grad():
            _value, action, _log_prob, self._hidden_state = self._policy.act(
                obs_tensors,
                self._hidden_state,
                self._mask,
                deterministic=True,
            )
        self._mask = torch.ones((1, 1), dtype=torch.float32, device=self._device)

        raw_action = np.asarray(action.reshape(-1).detach().cpu().numpy(), dtype=float)
        if raw_action.size < 2:
            raise ValueError(f"Unexpected SoNIC action payload shape: {raw_action.shape}")
        raw_xy = raw_action[:2]
        speed_norm = float(np.linalg.norm(raw_xy))
        v_pref = float(self._checkpoint_config.robot.v_pref)
        if speed_norm > max(v_pref, 1e-8):
            action_xy = raw_xy / speed_norm * v_pref
        else:
            action_xy = raw_xy

        vx = float(action_xy[0])
        vy = float(action_xy[1])
        speed = float(np.hypot(vx, vy))
        heading = float(_extract_socnav_fields(observation)[1])
        desired_heading = float(math.atan2(vy, vx)) if speed > 1e-8 else heading
        heading_error = _normalize_angle(desired_heading - heading)
        dt = max(float(time_step), 1e-6)
        linear = float(
            np.clip(
                speed * max(0.0, math.cos(heading_error)),
                0.0,
                float(self.config.max_linear_speed),
            )
        )
        angular = float(
            np.clip(
                heading_error / dt,
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        meta.update(
            {
                "upstream_action_xy": [vx, vy],
                "projected_command_vw": [linear, angular],
                "projection_policy": self.projection_policy,
                "upstream_policy": self.upstream_policy,
                "source_action_kinematics": str(self._checkpoint_config.action_space.kinematics),
                "source_contract": {
                    "robot_policy": str(self._checkpoint_config.robot.policy),
                    "human_policy": str(self._checkpoint_config.humans.policy),
                    "predict_method": str(self._checkpoint_config.sim.predict_method),
                    "robot_sensor": str(self._checkpoint_config.robot.sensor),
                    "env_use_wrapper": bool(self._checkpoint_config.env.use_wrapper),
                    "env_time_step": float(self._checkpoint_config.env.time_step),
                },
                "parity_gaps": list(self.parity_gaps),
                "missing_state_keys": list(self._missing_state_keys),
                "unexpected_state_keys": list(self._unexpected_state_keys),
            }
        )
        return linear, angular, meta

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Map-runner entrypoint that returns only the projected command.

        Returns:
            Projected Robot SF linear and angular velocity commands.
        """
        dt_source = observation.get("dt", 0.1)
        if "sim" in observation and isinstance(observation["sim"], dict):
            dt_source = observation["sim"].get("timestep", dt_source)
        dt = float(np.asarray(dt_source, dtype=float).reshape(-1)[0])
        linear, angular, _meta = self.act(observation, time_step=dt)
        return linear, angular


__all__ = [
    "SonicCrowdNavAdapter",
    "SonicCrowdNavConfig",
    "build_sonic_crowdnav_config",
]
