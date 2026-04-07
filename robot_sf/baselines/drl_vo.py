"""DRL-VO baseline adapter for the Social Navigation Benchmark.

This adapter provides a lightweight integration point for DRL-VO-style policies.
It supports optional checkpoint loading and falls back safely to a simple goal
seeking motion when the upstream model is unavailable or cannot be loaded.

The planner is intentionally robust: it can be selected by the benchmark runner
via `--algo drl_vo` and will still produce valid velocity commands even on
machines without PyTorch or a DRL-VO checkpoint.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import torch
except ImportError:  # pragma: no cover - envs without PyTorch
    torch = None  # type: ignore[assignment]

from robot_sf.baselines.social_force import Observation
from robot_sf.common.errors import raise_fatal_with_remedy, warn_soft_degrade
from robot_sf.models import resolve_model_path


@dataclass
class DrlVoPlannerConfig:
    """Configuration for the DRL-VO planner adapter."""

    model_path: str = "model/drl_vo_default.pt"
    model_id: str | None = None
    device: str = "auto"
    deterministic: bool = True
    nearest_k: int = 5
    action_space: str = "velocity"  # "velocity" | "unicycle"
    v_max: float = 2.0
    omega_max: float = 1.0
    fallback_to_goal: bool = True
    goal_speed: float = 1.0


class DrlVoPlanner:
    """Baseline wrapper for DRL-VO policies."""

    def __init__(self, config: DrlVoPlannerConfig | dict[str, Any], *, seed: int | None = None):
        """Initialize the DRL-VO planner adapter.

        Args:
            config: Planner configuration or dict payload.
            seed: Optional random seed for deterministic behavior.
        """
        self.config = self._parse_config(config)
        self._seed = seed
        self._model: Any | None = None
        self._status = "ok"
        self._fallback_reason: str | None = None
        self._load_model()

    def _parse_config(self, cfg: DrlVoPlannerConfig | dict[str, Any]) -> DrlVoPlannerConfig:
        if isinstance(cfg, DrlVoPlannerConfig):
            return cfg
        if isinstance(cfg, dict):
            valid_keys = {field.name for field in fields(DrlVoPlannerConfig)}
            filtered_cfg = {key: value for key, value in cfg.items() if key in valid_keys}
            return DrlVoPlannerConfig(**filtered_cfg)
        raise TypeError(f"Invalid config type: {type(cfg)}")

    def _torch_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_model_path(self) -> Path:
        if self.config.model_id:
            return resolve_model_path(self.config.model_id)
        return Path(self.config.model_path)

    def _load_model(self) -> None:
        if torch is None:
            warn_soft_degrade(
                "DRL-VO",
                "PyTorch is not installed",
                "falling back to goal-seeking navigation",
            )
            self._model = None
            self._status = "fallback"
            self._fallback_reason = "torch_missing"
            return

        try:
            model_path = self._resolve_model_path()
        except (KeyError, FileNotFoundError, RuntimeError, ValueError) as exc:
            if self.config.fallback_to_goal:
                warn_soft_degrade(
                    "DRL-VO model",
                    f"Failed to resolve model path: {exc}",
                    "falling back to goal-seeking navigation",
                )
                self._model = None
                self._status = "fallback"
                self._fallback_reason = "model_resolution_failed"
                return
            raise

        if not model_path.exists():
            if self.config.fallback_to_goal:
                warn_soft_degrade(
                    "DRL-VO model",
                    f"Model file not found: {model_path}",
                    "falling back to goal-seeking navigation",
                )
                self._model = None
                self._status = "fallback"
                self._fallback_reason = "model_missing"
                return
            raise_fatal_with_remedy(
                f"DRL-VO model file not found: {model_path}",
                "Place a compatible DRL-VO checkpoint at the model path or set fallback_to_goal to true.",
            )

        try:
            model = torch.load(str(model_path), map_location=self._torch_device())
        except (RuntimeError, ValueError, OSError) as exc:
            if self.config.fallback_to_goal:
                warn_soft_degrade(
                    "DRL-VO model",
                    f"Failed to load checkpoint: {exc}",
                    "falling back to goal-seeking navigation",
                )
                self._model = None
                self._status = "fallback"
                self._fallback_reason = "model_load_failed"
                return
            raise_fatal_with_remedy(
                f"Failed to load DRL-VO checkpoint from {model_path}: {exc}",
                "Check the checkpoint file format and the installed PyTorch version.",
            )

        if hasattr(model, "predict") or hasattr(model, "forward") or callable(model):
            self._model = model
            self._status = "ok"
            self._fallback_reason = None
        else:
            warn_soft_degrade(
                "DRL-VO model",
                "Loaded checkpoint is not a callable model",
                "falling back to goal-seeking navigation",
            )
            self._model = None
            self._status = "fallback"
            self._fallback_reason = "invalid_model"

    def reset(self, *, seed: int | None = None) -> None:
        """Reset planner state and optionally update the RNG seed."""
        if seed is not None:
            self._seed = seed

    def configure(self, config: DrlVoPlannerConfig | dict[str, Any]) -> None:
        """Update the planner configuration and reload the model if needed."""
        self.config = self._parse_config(config)
        self._load_model()

    def close(self) -> None:
        """Release any loaded model resources."""
        self._model = None

    def step(self, obs: Observation | dict[str, Any]) -> dict[str, float]:
        """Compute the next action from the current observation.

        Args:
            obs: Current observation from the benchmark runner.

        Returns:
            Action dictionary in the configured action space.
        """
        if isinstance(obs, dict):
            obs = Observation(**obs)  # type: ignore[arg-type]
        if not isinstance(obs, Observation):
            raise TypeError(f"Unsupported observation type: {type(obs)}")

        if self._model is not None:
            try:
                action = self._predict_action(obs)
                return action
            except Exception as exc:
                logger.warning("DRL-VO model prediction failed: {}", exc)
                if not self.config.fallback_to_goal:
                    raise
        return self._goal_seeking_action(obs)

    def _predict_action(self, obs: Observation) -> dict[str, float]:
        model_input = self._build_model_input(obs)
        if torch is None:
            raise RuntimeError("PyTorch is required for DRL-VO model inference")

        tensor = torch.tensor(model_input, dtype=torch.float32, device=self._torch_device())
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        if hasattr(self._model, "predict"):
            predict = self._model.predict
            try:
                signature = inspect.signature(predict)
            except (TypeError, ValueError):
                raw_action = predict(tensor)
            else:
                accepts_deterministic = "deterministic" in signature.parameters or any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in signature.parameters.values()
                )
                if accepts_deterministic:
                    raw_action = predict(tensor, deterministic=self.config.deterministic)
                else:
                    raw_action = predict(tensor)
        else:
            raw_action = self._model(tensor)

        # Some models return (action, state) tuples
        if isinstance(raw_action, tuple) and raw_action:
            raw_action = raw_action[0]

        return self._parse_model_action(raw_action)

    def _build_model_input(self, obs: Observation) -> np.ndarray:
        robot_pos = np.asarray(obs.robot["position"], dtype=float)
        robot_vel = np.asarray(obs.robot["velocity"], dtype=float)
        robot_goal = np.asarray(obs.robot["goal"], dtype=float)
        goal_rel = robot_goal - robot_pos

        parts = [goal_rel, robot_vel]
        agents = sorted(
            obs.agents,
            key=lambda a: np.linalg.norm(np.asarray(a.get("position", [0.0, 0.0])) - robot_pos),
        )[: self.config.nearest_k]
        for agent in agents:
            rel_pos = np.asarray(agent.get("position", [0.0, 0.0]), dtype=float) - robot_pos
            rel_vel = np.asarray(agent.get("velocity", [0.0, 0.0]), dtype=float)
            parts.append(rel_pos)
            parts.append(rel_vel)

        for _ in range(self.config.nearest_k - len(agents)):
            parts.append(np.zeros(2, dtype=float))
            parts.append(np.zeros(2, dtype=float))

        return np.concatenate(parts, axis=0)

    def _parse_model_action(self, raw_action: Any) -> dict[str, float]:
        if isinstance(raw_action, dict):
            if self.config.action_space == "velocity" and {"vx", "vy"}.issubset(raw_action):
                return {"vx": float(raw_action["vx"]), "vy": float(raw_action["vy"])}
            if self.config.action_space == "unicycle" and {"v", "omega"}.issubset(raw_action):
                return {"v": float(raw_action["v"]), "omega": float(raw_action["omega"])}

        if torch is not None and isinstance(raw_action, torch.Tensor):
            raw_action = raw_action.detach().cpu().numpy()

        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.flatten()

        if isinstance(raw_action, (list, tuple)) and len(raw_action) == 2:
            first, second = raw_action
            if self.config.action_space == "velocity":
                return {"vx": float(first), "vy": float(second)}
            return {"v": float(first), "omega": float(second)}

        raise ValueError(
            "DRL-VO model returned unsupported action format. "
            "Expected dict with velocity/unicycle keys or a length-2 vector."
        )

    def _goal_seeking_action(self, obs: Observation) -> dict[str, float]:
        robot_pos = np.asarray(obs.robot["position"], dtype=float)
        goal = np.asarray(obs.robot["goal"], dtype=float)
        delta = goal - robot_pos
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            if self.config.action_space == "velocity":
                return {"vx": 0.0, "vy": 0.0}
            return {"v": 0.0, "omega": 0.0}

        direction = delta / distance
        speed = min(self.config.goal_speed, self.config.v_max, distance)

        if self.config.action_space == "velocity":
            return {"vx": float(direction[0] * speed), "vy": float(direction[1] * speed)}

        heading_source = obs.robot.get("heading", [0.0])
        heading = float(np.asarray(heading_source, dtype=float).reshape(-1)[0])
        desired_heading = float(np.arctan2(direction[1], direction[0]))
        heading_error = float((desired_heading - heading + np.pi) % (2.0 * np.pi) - np.pi)
        omega = float(np.clip(heading_error, -self.config.omega_max, self.config.omega_max))
        return {"v": float(speed), "omega": omega}

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the planner status and config."""
        metadata = {
            "algorithm": "drl_vo",
            "status": self._status,
            "config": asdict(self.config),
        }
        if self._fallback_reason:
            metadata["fallback_reason"] = self._fallback_reason
        return metadata


__all__ = ["DrlVoPlanner", "DrlVoPlannerConfig"]
