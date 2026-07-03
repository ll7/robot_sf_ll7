"""Robot SF-native diffusion-policy local planner skeleton for issue #4010.

This module implements the first diagnostic slice of a COLSON-style diffusion policy:
Robot SF state observations, a PyTorch-only robot/pedestrian encoder, bounded one-step
action sampling, and inference-time candidate guidance. It intentionally does not claim
training success or COLSON reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised only in environments without torch installed.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


CLAIM_BOUNDARY = (
    "diagnostic Robot SF-native diffusion local planner; not a COLSON reproduction; "
    "not benchmark or paper evidence"
)
EVIDENCE_TIER = "diagnostic-only"
_NODE_FEATURES = 8
_DEFAULT_GUIDANCE = {
    "enabled": True,
    "goal_progress_weight": 1.0,
    "smooth_weight": 0.15,
    "clearance_weight": 0.25,
    "limit_weight": 0.05,
}


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    """Configuration for :class:`DiffusionPolicyAdapter`."""

    checkpoint_path: str | None = None
    model_id: str | None = None
    normalizer_path: str | None = None
    device: str = "cpu"
    max_pedestrians: int = 16
    action_horizon: int = 1
    action_dim: int = 2
    denoising_steps: int = 8
    num_action_samples: int = 8
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0
    deterministic: bool = False
    seed: int | None = None
    allow_untrained_smoke: bool = False
    guidance: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields that define the first issue #4010 runtime contract."""
        if self.max_pedestrians < 0:
            raise ValueError("max_pedestrians must be non-negative")
        if self.action_horizon != 1:
            raise ValueError("issue #4010 first slice supports action_horizon=1 only")
        if self.action_dim != 2:
            raise ValueError("issue #4010 first slice supports action_dim=2 only")
        if self.denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        if self.num_action_samples <= 0:
            raise ValueError("num_action_samples must be positive")
        if self.max_linear_speed <= 0.0:
            raise ValueError("max_linear_speed must be positive")
        if self.max_angular_speed <= 0.0:
            raise ValueError("max_angular_speed must be positive")


def _require_torch() -> None:
    """Raise a fail-closed dependency error when PyTorch is unavailable."""
    if torch is None or nn is None:
        raise RuntimeError(
            "DiffusionPolicyAdapter requires PyTorch. Install project training dependencies "
            "with `uv sync --all-extras`, or choose a non-diffusion policy."
        )


def _build_generator(device: Any) -> Any:
    """Create a sampling generator on the configured runtime device.

    Returns:
        Any: PyTorch generator bound to the configured device.
    """
    _require_torch()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DiffusionPolicyAdapter configured device 'cuda' is not available in this "
            "PyTorch runtime."
        )
    try:
        return torch.Generator(device=device)
    except RuntimeError as exc:
        raise RuntimeError(
            "DiffusionPolicyAdapter cannot create a sampler generator for configured "
            f"device '{device}'."
        ) from exc


def build_diffusion_policy_config(
    config: dict[str, Any] | DiffusionPolicyConfig,
) -> DiffusionPolicyConfig:
    """Build and validate a diffusion-policy config from YAML-style values.

    Returns:
        DiffusionPolicyConfig: Validated planner configuration.
    """
    if isinstance(config, DiffusionPolicyConfig):
        return config
    unknown = set(config) - set(DiffusionPolicyConfig.__dataclass_fields__)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported diffusion policy config fields: {joined}")
    return DiffusionPolicyConfig(**config)


class RobotPedestrianGraphEncoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Lightweight graph-style encoder over robot and visible pedestrian nodes."""

    def __init__(self, *, max_pedestrians: int, hidden_dim: int = 64) -> None:
        """Create the PyTorch-only message-passing encoder."""
        _require_torch()
        super().__init__()
        self.max_pedestrians = int(max_pedestrians)
        self.hidden_dim = int(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(_NODE_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def output_dim(self) -> int:
        """Return encoded condition width."""
        return self.hidden_dim

    def encode_observation(self, observation: dict[str, Any]) -> tuple[Any, Any]:
        """Convert a Robot SF policy observation into padded node features and a mask.

        Returns:
            tuple[Any, Any]: Node feature tensor and boolean mask tensor.
        """
        _require_torch()
        robot = observation.get("robot", {}) if isinstance(observation.get("robot"), dict) else {}
        agents = (
            observation.get("agents", []) if isinstance(observation.get("agents"), list) else []
        )

        robot_pos = _as_xy(robot.get("position"), default=(0.0, 0.0))
        robot_vel = _as_xy(robot.get("velocity"), default=(0.0, 0.0))
        goal = _as_xy(robot.get("goal"), default=(0.0, 0.0))
        heading = _as_float(robot.get("heading"), default=0.0)
        radius = _as_float(robot.get("radius"), default=0.3)
        rel_goal = _rotate_world_to_robot(goal - robot_pos, heading)

        features = np.zeros((self.max_pedestrians + 1, _NODE_FEATURES), dtype=np.float32)
        mask = np.zeros((self.max_pedestrians + 1,), dtype=bool)
        features[0] = np.array(
            [
                rel_goal[0],
                rel_goal[1],
                robot_vel[0],
                robot_vel[1],
                np.cos(heading),
                np.sin(heading),
                radius,
                1.0,
            ],
            dtype=np.float32,
        )
        mask[0] = True

        for index, agent in enumerate(agents[: self.max_pedestrians], start=1):
            if not isinstance(agent, dict):
                continue
            ped_pos = _as_xy(agent.get("position"), default=(0.0, 0.0))
            ped_vel = _as_xy(agent.get("velocity"), default=(0.0, 0.0))
            ped_radius = _as_float(agent.get("radius"), default=0.3)
            rel_pos = _rotate_world_to_robot(ped_pos - robot_pos, heading)
            rel_vel = _rotate_world_to_robot(ped_vel - robot_vel, heading)
            distance = float(np.linalg.norm(rel_pos))
            features[index] = np.array(
                [rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], ped_radius, distance, 1.0, 0.0],
                dtype=np.float32,
            )
            mask[index] = True

        device = torch.device("cpu")
        return torch.as_tensor(features, dtype=torch.float32, device=device), torch.as_tensor(mask)

    def forward(self, node_features: Any, mask: Any) -> Any:
        """Encode masked graph nodes into one fixed-width condition vector.

        Returns:
            Any: Encoded condition tensor.
        """
        encoded = self.node_mlp(node_features)
        robot = encoded[..., 0:1, :].expand_as(encoded)
        messages = self.message_mlp(torch.cat([robot, encoded], dim=-1))
        masked_messages = messages * mask.to(dtype=messages.dtype).unsqueeze(-1)
        denom = mask.to(dtype=messages.dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = masked_messages.sum(dim=-2) / denom
        return self.out_mlp(torch.cat([encoded[..., 0, :], pooled], dim=-1))


class DiffusionActionSampler(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Small conditional one-step action diffusion sampler."""

    def __init__(
        self,
        *,
        condition_dim: int,
        action_dim: int,
        max_linear_speed: float,
        max_angular_speed: float,
        hidden_dim: int = 64,
    ) -> None:
        """Create a bounded epsilon-prediction network."""
        _require_torch()
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.net = nn.Sequential(
            nn.Linear(condition_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def sample(
        self,
        condition: Any,
        *,
        num_samples: int,
        denoising_steps: int,
        generator: Any,
        deterministic: bool,
    ) -> Any:
        """Sample bounded candidate ``(v, omega)`` actions.

        Returns:
            Any: Tensor of bounded candidate actions.
        """
        _require_torch()
        device = condition.device
        dtype = condition.dtype
        if deterministic:
            actions = torch.zeros((num_samples, self.action_dim), dtype=dtype, device=device)
        else:
            actions = torch.randn(
                (num_samples, self.action_dim),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        condition_batch = condition.unsqueeze(0).expand(num_samples, -1)
        for step in reversed(range(denoising_steps)):
            t = torch.full(
                (num_samples, 1),
                float(step + 1) / float(denoising_steps),
                dtype=dtype,
                device=device,
            )
            eps = self.net(torch.cat([actions, t, condition_batch], dim=-1))
            actions = actions - eps / float(denoising_steps)
            actions = self._bounded(actions)
        return actions

    def _bounded(self, actions: Any) -> Any:
        """Project raw model outputs into the configured command range.

        Returns:
            Any: Tensor with configured linear and angular velocity bounds.
        """
        linear = torch.sigmoid(actions[:, 0:1]) * self.max_linear_speed
        angular = torch.tanh(actions[:, 1:2]) * self.max_angular_speed
        return torch.cat([linear, angular], dim=-1)


class DiffusionGuidanceSelector:
    """Rank sampled actions with simple inference-time guidance terms."""

    def __init__(self, guidance: dict[str, Any] | None = None) -> None:
        """Create guidance scorer."""
        merged = dict(_DEFAULT_GUIDANCE)
        merged.update(guidance or {})
        self.guidance = merged

    def select(
        self,
        candidates: np.ndarray,
        observation: dict[str, Any],
        previous_action: tuple[float, float],
    ) -> tuple[tuple[float, float], dict[str, Any]]:
        """Return the best candidate and compact scoring diagnostics.

        Returns:
            tuple[tuple[float, float], dict[str, Any]]: Selected command and scoring metadata.
        """
        if candidates.size == 0:
            return (0.0, 0.0), {"status": "empty_candidates"}
        if not bool(self.guidance.get("enabled", True)):
            selected = candidates[0]
            return (float(selected[0]), float(selected[1])), {
                "status": "disabled",
                "selected_index": 0,
            }

        goal_y = _goal_in_robot_frame(observation)[1]
        distances = _pedestrian_distances(observation)
        min_distance = float(np.min(distances)) if distances.size else float("inf")
        scores = []
        for action in candidates:
            linear = float(action[0])
            angular = float(action[1])
            goal_progress = linear - 0.1 * abs(goal_y) * abs(angular)
            smooth = float(np.sum((action - np.asarray(previous_action, dtype=float)) ** 2))
            clearance = max(0.0, 1.0 - min_distance) * abs(angular)
            limit = max(0.0, linear - 1.0) ** 2 + max(0.0, abs(angular) - 1.0) ** 2
            score = (
                float(self.guidance["goal_progress_weight"]) * goal_progress
                - float(self.guidance["smooth_weight"]) * smooth
                + float(self.guidance["clearance_weight"]) * clearance
                - float(self.guidance["limit_weight"]) * limit
            )
            scores.append(score)
        selected_index = int(np.argmax(scores))
        selected = candidates[selected_index]
        return (float(selected[0]), float(selected[1])), {
            "status": "ok",
            "selected_index": selected_index,
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "terms": ["goal_progress", "smooth_previous_action", "current_clearance_proxy"],
        }


class DiffusionPolicyAdapter:
    """Adapter exposing the map-runner planner contract for diffusion_policy."""

    def __init__(self, config: dict[str, Any] | DiffusionPolicyConfig) -> None:
        """Create an adapter, failing closed for unproven runtime inputs."""
        _require_torch()
        self.config = build_diffusion_policy_config(config)
        self._validate_runtime_source()
        self._device = torch.device(self.config.device)
        self._generator = _build_generator(self._device)
        self._previous_action = (0.0, 0.0)
        self._last_candidates: np.ndarray | None = None
        self._last_guidance: dict[str, Any] = {"status": "not_run"}
        self.encoder = RobotPedestrianGraphEncoder(max_pedestrians=self.config.max_pedestrians).to(
            self._device
        )
        self.sampler = DiffusionActionSampler(
            condition_dim=self.encoder.output_dim,
            action_dim=self.config.action_dim,
            max_linear_speed=self.config.max_linear_speed,
            max_angular_speed=self.config.max_angular_speed,
        ).to(self._device)
        self.selector = DiffusionGuidanceSelector(self.config.guidance)
        self.reset(seed=self.config.seed)
        self._load_checkpoint_if_present()
        self.encoder.eval()
        self.sampler.eval()

    def reset(self, *, seed: int | None = None) -> None:
        """Reset deterministic sampling state."""
        actual_seed = self.config.seed if seed is None else seed
        if actual_seed is not None:
            self._generator.manual_seed(int(actual_seed))
            torch.manual_seed(int(actual_seed))
        self._previous_action = (0.0, 0.0)
        self._last_candidates = None
        self._last_guidance = {"status": "not_run"}

    def close(self) -> None:
        """Release planner resources."""
        self._last_candidates = None

    def plan(self, obs: dict[str, Any]) -> tuple[float, float]:
        """Return one bounded ``(linear, angular)`` command.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        node_features, mask = self.encoder.encode_observation(obs)
        node_features = node_features.to(self._device)
        mask = mask.to(self._device)
        with torch.no_grad():
            condition = self.encoder(node_features, mask)
            candidates_tensor = self.sampler.sample(
                condition,
                num_samples=self.config.num_action_samples,
                denoising_steps=self.config.denoising_steps,
                generator=self._generator,
                deterministic=self.config.deterministic,
            )
        candidates = candidates_tensor.detach().cpu().numpy()
        selected, guidance_diag = self.selector.select(candidates, obs, self._previous_action)
        self._previous_action = selected
        self._last_candidates = candidates
        self._last_guidance = guidance_diag
        return selected

    def diagnostics(self) -> dict[str, Any]:
        """Expose compact diagnostic metadata without large tensors by default.

        Returns:
            dict[str, Any]: Diagnostic claim boundary and optional sample metadata.
        """
        diag: dict[str, Any] = {
            "diffusion_policy": {
                "status": "ok",
                "evidence_tier": EVIDENCE_TIER,
                "denoising_steps": self.config.denoising_steps,
                "num_action_samples": self.config.num_action_samples,
                "device": str(self._device),
                "allow_untrained_smoke": self.config.allow_untrained_smoke,
                "checkpoint_status": (
                    "checkpoint_loaded" if self.config.checkpoint_path else "untrained_smoke"
                ),
                "normalizer_status": "loaded" if self.config.checkpoint_path else "not_required",
                "guidance": {
                    "enabled": bool(self.selector.guidance.get("enabled", True)),
                    "terms": ["smooth_previous_action", "current_clearance_proxy"],
                    "last": self._last_guidance,
                },
                "claim_boundary": CLAIM_BOUNDARY,
            }
        }
        if (
            bool(self.config.diagnostics.get("record_raw_samples", False))
            and self._last_candidates is not None
        ):
            diag["diffusion_policy"]["raw_samples"] = self._last_candidates.tolist()
        return diag

    def _validate_runtime_source(self) -> None:
        """Fail closed unless a checkpoint exists or smoke mode is explicit."""
        if self.config.checkpoint_path:
            checkpoint = Path(self.config.checkpoint_path).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Diffusion policy checkpoint not found: {checkpoint}")
            if not self.config.normalizer_path:
                raise RuntimeError(
                    "diffusion_policy checkpoint_path requires normalizer_path for "
                    "issue #4010 smoke checkpoint provenance."
                )
        elif not self.config.allow_untrained_smoke:
            raise RuntimeError(
                "diffusion_policy requires checkpoint_path unless allow_untrained_smoke=true. "
                "Untrained random inference is diagnostic-only and must not be counted as success evidence."
            )
        if self.config.normalizer_path:
            normalizer = Path(self.config.normalizer_path).expanduser()
            if not normalizer.is_file():
                raise FileNotFoundError(f"Diffusion policy normalizer not found: {normalizer}")

    def _load_checkpoint_if_present(self) -> None:
        """Load a checkpoint when provided; smoke mode keeps random initialization."""
        if not self.config.checkpoint_path:
            return
        checkpoint = Path(self.config.checkpoint_path).expanduser()
        payload = torch.load(checkpoint, map_location=self._device)
        if isinstance(payload, dict) and "encoder" in payload and "sampler" in payload:
            self.encoder.load_state_dict(payload["encoder"])
            self.sampler.load_state_dict(payload["sampler"])
            return
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state = payload["model_state_dict"]
            self.encoder.load_state_dict(state.get("encoder", {}), strict=False)
            self.sampler.load_state_dict(state.get("sampler", {}), strict=False)
            return
        raise RuntimeError(
            "Unsupported diffusion policy checkpoint format; expected encoder/sampler state dictionaries"
        )


def _as_xy(value: Any, *, default: tuple[float, float]) -> np.ndarray:
    """Normalize a value to an ``(x, y)`` float vector.

    Returns:
        np.ndarray: Two-element float vector.
    """
    if value is None:
        return np.asarray(default, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return np.asarray(default, dtype=float)
    return arr[:2].astype(float)


def _as_float(value: Any, *, default: float) -> float:
    """Normalize scalar-like values.

    Returns:
        float: Parsed scalar value.
    """
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return float(default)
    return float(arr[0])


def _rotate_world_to_robot(vector: np.ndarray, heading: float) -> np.ndarray:
    """Rotate a world-frame vector into the robot heading frame.

    Returns:
        np.ndarray: Two-element robot-frame vector.
    """
    c = float(np.cos(-heading))
    s = float(np.sin(-heading))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]], dtype=float)


def _goal_in_robot_frame(observation: dict[str, Any]) -> np.ndarray:
    """Return relative goal vector in robot frame.

    Returns:
        np.ndarray: Two-element robot-frame goal vector.
    """
    robot = observation.get("robot", {}) if isinstance(observation.get("robot"), dict) else {}
    robot_pos = _as_xy(robot.get("position"), default=(0.0, 0.0))
    goal = _as_xy(robot.get("goal"), default=(0.0, 0.0))
    heading = _as_float(robot.get("heading"), default=0.0)
    return _rotate_world_to_robot(goal - robot_pos, heading)


def _pedestrian_distances(observation: dict[str, Any]) -> np.ndarray:
    """Return current pedestrian distances from the robot.

    Returns:
        np.ndarray: Current pedestrian distances in meters.
    """
    robot = observation.get("robot", {}) if isinstance(observation.get("robot"), dict) else {}
    robot_pos = _as_xy(robot.get("position"), default=(0.0, 0.0))
    agents = observation.get("agents", []) if isinstance(observation.get("agents"), list) else []
    distances = []
    for agent in agents:
        if isinstance(agent, dict):
            distance = float(
                np.linalg.norm(_as_xy(agent.get("position"), default=(0.0, 0.0)) - robot_pos)
            )
            if np.isfinite(distance):
                distances.append(distance)
    return np.asarray(distances, dtype=float)


__all__ = [
    "CLAIM_BOUNDARY",
    "EVIDENCE_TIER",
    "DiffusionActionSampler",
    "DiffusionGuidanceSelector",
    "DiffusionPolicyAdapter",
    "DiffusionPolicyConfig",
    "RobotPedestrianGraphEncoder",
    "build_diffusion_policy_config",
]
