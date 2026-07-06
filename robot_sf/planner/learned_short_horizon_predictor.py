"""Learned short-horizon pedestrian prediction backend for prediction MPC."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.models.registry import resolve_model_path
from robot_sf.planner.nmpc_social import _parse_bool
from robot_sf.planner.prediction_mpc import (
    ConstantVelocityPedestrianPredictor,
    PredictedPedestrianFutures,
)


@dataclass(frozen=True)
class LearnedShortHorizonPredictorConfig:
    """Configuration for the lightweight learned pedestrian predictor."""

    checkpoint_path: str | None = None
    model_id: str | None = None
    normalizer_path: str | None = None
    device: str = "cpu"
    max_pedestrians: int = 16
    history_steps: int = 4
    horizon_steps: int = 6
    rollout_dt: float = 0.2
    hidden_dim: int = 128
    model_type: str = "mlp"
    fallback_to_constant_velocity: bool = False
    allow_untrained_smoke: bool = False


def build_predictor_module(*, input_dim: int, output_dim: int, hidden_dim: int, device: str) -> Any:
    """Build the canonical short-horizon predictor network.

    This is the single source of truth for the predictor architecture so the
    inference wrapper and any offline trainer stay checkpoint-compatible.

    Returns:
        Any: A ``torch.nn.Sequential`` module on the requested device.
    """

    _, nn = _load_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    ).to(device)


def predictor_io_dims(config: LearnedShortHorizonPredictorConfig) -> tuple[int, int]:
    """Return the ``(input_dim, output_dim)`` implied by a predictor config.

    Returns:
        tuple[int, int]: Feature vector length and flattened prediction length.
    """

    input_dim = 4 + 2 + int(config.max_pedestrians) * 4
    output_dim = int(config.max_pedestrians) * int(config.horizon_steps) * 2
    return input_dim, output_dim


class _TinyMlp:
    """Lazy PyTorch MLP wrapper so module import remains cheap."""

    def __init__(self, *, input_dim: int, output_dim: int, hidden_dim: int, device: str) -> None:
        """Initialize the tiny state predictor network."""

        torch, _ = _load_torch()
        self.torch = torch
        self.module = build_predictor_module(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

    def zero_initialize(self) -> None:
        """Make untrained-smoke predictions deterministic and stationary."""

        for param in self.module.parameters():
            param.data.zero_()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load checkpoint weights into the wrapped module."""

        self.module.load_state_dict(state_dict)

    def predict_numpy(self, features: np.ndarray) -> np.ndarray:
        """Run a single feature vector through the model.

        Returns:
            np.ndarray: Flattened model output vector.
        """

        torch = self.torch
        self.module.eval()
        device = next(self.module.parameters()).device
        with torch.no_grad():
            tensor = torch.as_tensor(features[None, :], dtype=torch.float32, device=device)
            output = self.module(tensor).detach().cpu().numpy()
        return np.asarray(output[0], dtype=float)


class LearnedShortHorizonPedestrianPredictor:
    """State-based learned predictor conforming to ``PedestrianFuturePredictor``."""

    def __init__(self, config: LearnedShortHorizonPredictorConfig) -> None:
        """Initialize a fail-closed or diagnostic learned predictor."""

        if config.max_pedestrians <= 0:
            raise ValueError("max_pedestrians must be strictly positive")
        if config.horizon_steps <= 0:
            raise ValueError("horizon_steps must be strictly positive")
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be strictly positive")
        self.config = config
        self._cv_predictor = ConstantVelocityPedestrianPredictor()
        self._calls = 0
        self._last_source = "not_run"
        self._checkpoint_path = self._resolve_checkpoint_path(config)
        self._model: _TinyMlp | None = None
        if self._checkpoint_path is not None:
            self._model = self._build_model()
            self._load_checkpoint(self._checkpoint_path)
            self._evidence_tier = "checkpoint_loaded"
        elif config.allow_untrained_smoke:
            self._model = self._build_model()
            self._model.zero_initialize()
            self._evidence_tier = "diagnostic_untrained_smoke"
        elif config.fallback_to_constant_velocity:
            self._evidence_tier = "diagnostic_constant_velocity_fallback"
        else:
            raise ValueError(
                "learned short-horizon predictor requires checkpoint_path or model_id; "
                "set allow_untrained_smoke=true only for diagnostic smoke tests."
            )

    def predict(
        self,
        observation: dict[str, Any],
        *,
        horizon_steps: int,
        dt: float,
    ) -> PredictedPedestrianFutures:
        """Return predicted pedestrian futures in world coordinates."""

        self._calls += 1
        if self._checkpoint_path is None and self.config.fallback_to_constant_velocity:
            futures = self._cv_predictor.predict(observation, horizon_steps=horizon_steps, dt=dt)
            self._last_source = "diagnostic_constant_velocity_fallback"
            return PredictedPedestrianFutures(
                positions_world=futures.positions_world,
                mask=futures.mask,
                dt=futures.dt,
                source=self._last_source,
            )

        ped_positions, ped_velocities_world = self._pedestrian_state_world(observation)
        count = min(ped_positions.shape[0], max(int(self.config.max_pedestrians), 0))
        horizon = max(1, min(int(horizon_steps), int(self.config.horizon_steps)))
        if count == 0:
            self._last_source = self._evidence_tier
            return PredictedPedestrianFutures(
                positions_world=np.zeros((0, horizon, 2), dtype=float),
                mask=np.zeros((0,), dtype=float),
                dt=float(dt),
                source=self._last_source,
            )

        features = self._features(observation, ped_positions, ped_velocities_world)
        if self._model is None:
            raise RuntimeError("learned predictor model is unavailable outside fallback mode")
        output = self._model.predict_numpy(features)
        deltas = output.reshape(self.config.max_pedestrians, self.config.horizon_steps, 2)
        futures = np.zeros((count, horizon, 2), dtype=float)
        for step_idx in range(horizon):
            tau = float(step_idx + 1) * float(dt)
            learned_delta = deltas[:count, step_idx, :]
            futures[:, step_idx, :] = (
                ped_positions[:count] + ped_velocities_world[:count] * tau + learned_delta
            )
        self._last_source = self._evidence_tier
        return PredictedPedestrianFutures(
            positions_world=futures,
            mask=np.ones((count,), dtype=float),
            dt=float(dt),
            source=self._last_source,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return claim-boundary metadata for benchmark rows."""

        return {
            "backend": "learned_short_horizon",
            "model_type": self.config.model_type,
            "evidence_tier": self._evidence_tier,
            "diagnostic_only": self._evidence_tier.startswith("diagnostic_"),
            "not_full_world_model": True,
            "checkpoint_path": str(self._checkpoint_path) if self._checkpoint_path else None,
            "calls": self._calls,
            "last_source": self._last_source,
            "max_pedestrians": self.config.max_pedestrians,
            "horizon_steps": self.config.horizon_steps,
        }

    def reset(self) -> None:
        """Reset per-episode diagnostics."""

        self._calls = 0
        self._last_source = "not_run"

    def _resolve_checkpoint_path(self, config: LearnedShortHorizonPredictorConfig) -> Path | None:
        """Resolve configured checkpoint or model registry ID.

        Returns:
            Path | None: Local checkpoint path, or ``None`` when no artifact is configured.
        """

        if config.checkpoint_path:
            path = Path(config.checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"learned predictor checkpoint not found: {path}")
            return path
        if config.model_id:
            return resolve_model_path(config.model_id, allow_download=False)
        return None

    def _build_model(self) -> _TinyMlp:
        """Build the configured small predictor network.

        Returns:
            _TinyMlp: Tiny PyTorch MLP wrapper.
        """

        if self.config.model_type.strip().lower() != "mlp":
            raise ValueError(
                "learned short-horizon predictor currently supports model_type='mlp' only"
            )
        input_dim, output_dim = predictor_io_dims(self.config)
        return _TinyMlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(self.config.hidden_dim),
            device=self.config.device,
        )

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a PyTorch state dict checkpoint."""

        torch, _ = _load_torch()
        payload = torch.load(checkpoint_path, map_location=self.config.device)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"learned predictor checkpoint has unsupported format: {checkpoint_path}"
            )
        self._model.load_state_dict(state_dict)

    def _features(
        self,
        observation: dict[str, Any],
        ped_positions: np.ndarray,
        ped_velocities_world: np.ndarray,
    ) -> np.ndarray:
        """Encode current state into a fixed-size feature vector.

        Returns:
            np.ndarray: Fixed-width predictor feature vector.
        """

        return encode_predictor_features(
            observation,
            ped_positions,
            ped_velocities_world,
            max_pedestrians=int(self.config.max_pedestrians),
        )

    def _pedestrian_state_world(self, observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Extract pedestrian positions and rotate ego-frame velocities into world frame.

        Returns:
            tuple[np.ndarray, np.ndarray]: Pedestrian positions and world-frame velocities.
        """

        return pedestrian_world_state(observation)


def encode_predictor_features(
    observation: dict[str, Any],
    ped_positions: np.ndarray,
    ped_velocities_world: np.ndarray,
    *,
    max_pedestrians: int,
) -> np.ndarray:
    """Encode observation and world-frame pedestrian state into a feature vector.

    Shared by the inference wrapper and the offline trainer so both encode
    features identically.

    Returns:
        np.ndarray: Fixed-width predictor feature vector.
    """

    robot = observation.get("robot", {})
    goal = observation.get("goal", {})
    robot_pos = _as_xy(robot.get("position", [0.0, 0.0]))
    heading = float(_as_1d(robot.get("heading", [0.0]), pad=1)[0])
    speed = float(_as_1d(robot.get("speed", [0.0]), pad=1)[0])
    goal_pos = _as_xy(goal.get("current", goal.get("next", [0.0, 0.0])))

    features = np.zeros((4 + 2 + int(max_pedestrians) * 4,), dtype=float)
    features[:4] = np.asarray([robot_pos[0], robot_pos[1], heading, speed], dtype=float)
    features[4:6] = goal_pos - robot_pos
    offset = 6
    count = min(ped_positions.shape[0], int(max_pedestrians))
    for idx in range(count):
        features[offset : offset + 2] = ped_positions[idx] - robot_pos
        features[offset + 2 : offset + 4] = ped_velocities_world[idx]
        offset += 4
    return features


def pedestrian_world_state(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract pedestrian positions and rotate ego-frame velocities into world frame.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pedestrian positions and world-frame velocities.
    """

    ped_state = observation.get("pedestrians", {})
    robot = observation.get("robot", {})
    robot_heading = float(_as_1d(robot.get("heading", [0.0]), pad=1)[0])
    positions = _as_xy_matrix(ped_state.get("positions", []))
    velocities_ego = _as_xy_matrix(ped_state.get("velocities", []))
    count = int(_as_1d(ped_state.get("count", [positions.shape[0]]), pad=1)[0])
    count = max(0, min(count, positions.shape[0]))
    positions = positions[:count]
    if velocities_ego.shape[0] < count:
        velocities_ego = np.zeros_like(positions)
    else:
        velocities_ego = velocities_ego[:count]
    cos_h = float(np.cos(robot_heading))
    sin_h = float(np.sin(robot_heading))
    velocities_world = np.empty_like(velocities_ego)
    velocities_world[:, 0] = cos_h * velocities_ego[:, 0] - sin_h * velocities_ego[:, 1]
    velocities_world[:, 1] = sin_h * velocities_ego[:, 0] + cos_h * velocities_ego[:, 1]
    return positions, velocities_world


def build_learned_short_horizon_predictor_config(
    cfg: dict[str, Any] | None,
) -> LearnedShortHorizonPredictorConfig:
    """Build predictor config from algorithm YAML mapping.

    Returns:
        LearnedShortHorizonPredictorConfig: Parsed predictor configuration.
    """

    raw = dict(cfg or {})
    defaults = LearnedShortHorizonPredictorConfig()
    converters = {
        "checkpoint_path": _optional_str,
        "model_id": _optional_str,
        "normalizer_path": _optional_str,
        "device": lambda value: str(value).strip() if value else "cpu",
        "max_pedestrians": int,
        "history_steps": int,
        "horizon_steps": int,
        "rollout_dt": float,
        "hidden_dim": int,
        "model_type": str,
        "fallback_to_constant_velocity": _parse_bool,
        "allow_untrained_smoke": _parse_bool,
    }
    kwargs: dict[str, Any] = {}
    for field in fields(LearnedShortHorizonPredictorConfig):
        value = raw.get(field.name, getattr(defaults, field.name))
        try:
            kwargs[field.name] = converters[field.name](value)
        except (TypeError, ValueError):
            default_value = getattr(defaults, field.name)
            warnings.warn(
                (
                    f"Invalid learned predictor config value '{field.name}': {value!r}; "
                    f"falling back default {default_value!r}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            kwargs[field.name] = default_value
    return LearnedShortHorizonPredictorConfig(**kwargs)


def _optional_str(value: Any) -> str | None:
    """Normalize optional string values from YAML.

    Returns:
        str | None: Stripped string value, or ``None`` for empty inputs.
    """

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_torch() -> tuple[Any, Any]:
    """Load PyTorch lazily so ordinary planner imports stay dependency-light.

    Returns:
        tuple[Any, Any]: Imported ``torch`` module and ``torch.nn`` module.
    """

    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "learned short-horizon predictor requires torch for checkpoint or smoke mode"
        ) from exc
    return torch, nn


def _as_1d(value: Any, *, pad: int) -> np.ndarray:
    """Return one-dimensional float array padded to ``pad`` length.

    Returns:
        np.ndarray: One-dimensional padded float array.
    """

    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < pad:
        arr = np.pad(arr, (0, pad - arr.size), mode="constant")
    return arr


def _as_xy(value: Any) -> np.ndarray:
    """Return first two numeric values as an ``xy`` vector.

    Returns:
        np.ndarray: Two-element float vector.
    """

    return _as_1d(value, pad=2)[:2].astype(float)


def _as_xy_matrix(value: Any) -> np.ndarray:
    """Return ``(N, 2)`` float matrix or an empty matrix for invalid inputs.

    Returns:
        np.ndarray: Two-column float matrix.
    """

    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[-1] != 2:
        return np.zeros((0, 2), dtype=float)
    return arr.astype(float)
