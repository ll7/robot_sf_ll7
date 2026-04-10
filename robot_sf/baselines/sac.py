"""SAC baseline adapter for the Social Navigation Benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    from stable_baselines3 import SAC
except ImportError:  # pragma: no cover - runtime-only dependency
    SAC = None  # type: ignore

from robot_sf.baselines.social_force import Observation
from robot_sf.common.errors import raise_fatal_with_remedy, warn_soft_degrade
from robot_sf.models import resolve_model_path
from robot_sf.sensor.socnav_observation import SOCNAV_POSITION_CAP_M


@dataclass
class SACPlannerConfig:
    """Configuration for the SAC planner adapter."""

    model_path: str = "output/models/sac/sac_gate_socnav_struct_v1.zip"
    model_id: str | None = None
    device: str = "auto"
    deterministic: bool = True
    obs_mode: str = "dict"  # "dict" for flattened socnav_struct checkpoints
    nearest_k: int = 5
    action_space: str = "unicycle"  # "velocity" | "unicycle"
    v_max: float = 2.0
    omega_max: float = 1.0
    relative_obs: bool = True
    obs_transform: str = "none"
    fallback_to_goal: bool = True


class SACPlanner:
    """Benchmark wrapper for SB3 SAC checkpoints."""

    EPS: float = 1e-9

    def __init__(self, config: SACPlannerConfig | dict[str, Any], *, seed: int | None = None):
        """Create a SAC planner wrapper and load the configured checkpoint."""
        self.config = self._parse_config(config)
        self._seed = seed
        self._model = None
        self._status = "ok"
        self._fallback_reason: str | None = None
        self._load_model()

    def _parse_config(self, cfg: SACPlannerConfig | dict[str, Any]) -> SACPlannerConfig:
        if isinstance(cfg, SACPlannerConfig):
            return cfg
        if isinstance(cfg, dict):
            allowed = {field.name for field in fields(SACPlannerConfig)}
            return SACPlannerConfig(**{key: value for key, value in cfg.items() if key in allowed})
        raise TypeError(f"Invalid config type: {type(cfg)}")

    def _load_model(self) -> None:
        if SAC is None:  # pragma: no cover - runtime dependency
            warn_soft_degrade(
                "SAC",
                "stable_baselines3 not installed",
                "will use fallback-to-goal if enabled",
            )
            self._model = None
            self._status = "fallback"
            self._fallback_reason = "sb3_missing"
            return
        try:
            mp = (
                resolve_model_path(self.config.model_id)
                if self.config.model_id
                else Path(self.config.model_path)
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            if self.config.fallback_to_goal:
                warn_soft_degrade(
                    "SAC model",
                    f"Failed to resolve model: {exc}",
                    "will use fallback-to-goal navigation",
                )
                self._model = None
                self._status = "fallback"
                self._fallback_reason = "model_resolution_failed"
                return
            raise
        if not mp.exists():
            if self.config.fallback_to_goal:
                warn_soft_degrade(
                    "SAC model",
                    f"Model not found at {mp}",
                    "will use fallback-to-goal navigation",
                )
                self._model = None
                self._status = "fallback"
                self._fallback_reason = "model_missing"
                return
            raise_fatal_with_remedy(
                f"SAC model file not found: {mp}",
                "Train a benchmark-compatible checkpoint with "
                "scripts/training/train_sac_sb3.py --config configs/training/sac/gate_socnav_struct.yaml",
            )
        try:
            self._model = SAC.load(str(mp), device=self.config.device, print_system_info=False)
            self._status = "ok"
            self._fallback_reason = None
        except (RuntimeError, ValueError, OSError) as exc:
            if self.config.fallback_to_goal:
                warn_soft_degrade(
                    "SAC model",
                    f"Failed to load model: {exc}",
                    "will use fallback-to-goal navigation",
                )
                self._model = None
                self._status = "fallback"
                self._fallback_reason = "model_load_failed"
                return
            raise_fatal_with_remedy(
                f"Failed to load SAC model from {mp}: {exc}",
                "Check model compatibility with current stable_baselines3 version or retrain it.",
            )

    def reset(self, *, seed: int | None = None) -> None:
        """Reset planner state between episodes."""
        if seed is not None:
            self._seed = seed

    def close(self) -> None:
        """Release the loaded SAC model."""
        self._model = None

    def step(self, obs: Observation | dict[str, Any]) -> dict[str, float]:
        """Predict one action for either the structured or flattened obs contract.

        Returns:
            dict[str, float]: SAC action in the configured output space.
        """
        if isinstance(obs, dict) and self._uses_dict_observation():
            return self._step_dict_obs(obs)
        if isinstance(obs, dict):
            obs = Observation(**obs)  # type: ignore[arg-type]
        if not isinstance(obs, Observation):
            raise TypeError(
                f"Expected Observation or dict input, got {type(obs).__name__}"
            )
        try:
            action_vec = self._predict_action(self._vectorize(obs))
            if action_vec is None:
                raise RuntimeError("SAC model unavailable or prediction failed")
            return self._action_vec_to_dict(action_vec)
        except (RuntimeError, ValueError, OSError):
            if self.config.fallback_to_goal:
                if self._status != "fallback":
                    self._status = "fallback"
                if self._fallback_reason is None:
                    self._fallback_reason = "prediction_failed"
                return self._fallback_action(obs)
            raise

    def _uses_dict_observation(self) -> bool:
        return str(self.config.obs_mode).strip().lower() in {"dict", "native_dict", "multi_input"}

    def _step_dict_obs(self, obs: dict[str, Any]) -> dict[str, float]:
        try:
            model_obs = self._build_model_obs_dict(obs)
            action_vec = self._predict_action(model_obs)
            if action_vec is None:
                raise RuntimeError("SAC model unavailable or prediction failed")
            return self._action_vec_to_dict(action_vec)
        except (RuntimeError, ValueError, OSError):
            if self.config.fallback_to_goal:
                if self._status != "fallback":
                    self._status = "fallback"
                if self._fallback_reason is None:
                    self._fallback_reason = "prediction_failed"
                return self._fallback_action_dict(obs)
            raise

    def _predict_action(self, model_obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | None:
        if self._model is None:
            return None
        try:
            act, _ = self._model.predict(model_obs, deterministic=self.config.deterministic)
            return np.asarray(act, dtype=float).squeeze()
        except (RuntimeError, ValueError, OSError, IndexError) as exc:
            logger.debug("SAC model prediction failed: {}", exc)
            return None

    def _build_model_obs_dict(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        obs_transform = self._effective_obs_transform()
        if self._model is None:
            converted = {str(k): np.asarray(v) for k, v in obs.items()}
            return self._apply_obs_transform(converted, obs_transform)
        space = getattr(self._model, "observation_space", None)
        spaces = getattr(space, "spaces", None)
        if not isinstance(spaces, dict):
            converted = {str(k): np.asarray(v) for k, v in obs.items()}
            return self._apply_obs_transform(converted, obs_transform)
        converted, missing = self._coerce_dict_observation(obs, spaces)
        if missing:
            preview = ", ".join(missing[:6])
            raise ValueError(f"Missing required dict observation keys: {preview}")
        return self._apply_obs_transform(converted, obs_transform)

    def _effective_obs_transform(self) -> str:
        """Resolve the observation transform mode from config.

        Returns:
            str: ``relative`` or ``ego`` when configured explicitly, otherwise ``none``.
        """
        raw = str(getattr(self.config, "obs_transform", "none")).strip().lower()
        if raw == "none" and self.config.relative_obs:
            return "relative"
        return raw

    def _apply_obs_transform(
        self, obs: dict[str, np.ndarray], obs_transform: str
    ) -> dict[str, np.ndarray]:
        """Apply the configured SAC observation transform.

        Returns:
            dict[str, np.ndarray]: The transformed observation mapping.
        """
        if obs_transform == "relative":
            return self._apply_relative_socnav_obs(obs)
        if obs_transform == "ego":
            return self._apply_ego_socnav_obs(obs)
        return obs

    def _apply_relative_socnav_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Translate position-like SocNav keys into a robot-relative frame.

        Returns:
            dict[str, np.ndarray]: A copy of ``obs`` with position-like keys rebased.
        """
        if "robot_position" not in obs:
            return obs

        robot_position = np.asarray(obs["robot_position"], dtype=np.float32).reshape(-1)
        if robot_position.size < 2:
            return obs
        robot_xy = robot_position[:2]
        converted = dict(obs)

        def _shift_xy(key: str) -> None:
            if key not in converted:
                return
            arr = np.asarray(converted[key], dtype=np.float32)
            if arr.ndim == 1 and arr.size >= 2:
                rel = arr.copy()
                rel[:2] = np.clip(rel[:2] - robot_xy, -SOCNAV_POSITION_CAP_M, SOCNAV_POSITION_CAP_M)
                converted[key] = rel
                return
            if arr.ndim >= 2 and arr.shape[-1] >= 2:
                rel = arr.copy()
                mask = np.any(np.abs(rel[..., :2]) > 1e-8, axis=-1)
                rel[..., :2][mask] = np.clip(
                    rel[..., :2][mask] - robot_xy,
                    -SOCNAV_POSITION_CAP_M,
                    SOCNAV_POSITION_CAP_M,
                )
                converted[key] = rel

        converted["robot_position"] = np.zeros_like(
            np.asarray(converted["robot_position"], dtype=np.float32)
        )
        _shift_xy("goal_current")
        _shift_xy("goal_next")
        _shift_xy("pedestrians_positions")
        return converted

    def _apply_ego_socnav_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Rotate position-like SocNav keys into the robot ego frame.

        Returns:
            dict[str, np.ndarray]: A copy of ``obs`` with translated coordinates rotated.
        """
        converted = self._apply_relative_socnav_obs(obs)
        if "robot_heading" not in converted:
            return converted

        heading_arr = np.asarray(converted["robot_heading"], dtype=np.float32).reshape(-1)
        if heading_arr.size == 0:
            return converted
        heading = float(heading_arr[0])
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))

        def _rotate_key(key: str) -> None:
            if key not in converted:
                return
            arr = np.asarray(converted[key], dtype=np.float32)
            if arr.ndim == 1 and arr.size >= 2:
                rel = arr.copy()
                x_val = float(rel[0])
                y_val = float(rel[1])
                rel[0] = np.clip(
                    cos_h * x_val + sin_h * y_val,
                    -SOCNAV_POSITION_CAP_M,
                    SOCNAV_POSITION_CAP_M,
                )
                rel[1] = np.clip(
                    -sin_h * x_val + cos_h * y_val,
                    -SOCNAV_POSITION_CAP_M,
                    SOCNAV_POSITION_CAP_M,
                )
                converted[key] = rel
                return
            if arr.ndim >= 2 and arr.shape[-1] >= 2:
                rel = arr.copy()
                mask = np.any(np.abs(rel[..., :2]) > 1e-8, axis=-1)
                if not np.any(mask):
                    converted[key] = rel
                    return
                x_vals = rel[..., 0][mask]
                y_vals = rel[..., 1][mask]
                rel[..., 0][mask] = np.clip(
                    cos_h * x_vals + sin_h * y_vals,
                    -SOCNAV_POSITION_CAP_M,
                    SOCNAV_POSITION_CAP_M,
                )
                rel[..., 1][mask] = np.clip(
                    -sin_h * x_vals + cos_h * y_vals,
                    -SOCNAV_POSITION_CAP_M,
                    SOCNAV_POSITION_CAP_M,
                )
                converted[key] = rel

        _rotate_key("goal_current")
        _rotate_key("goal_next")
        _rotate_key("pedestrians_positions")
        return converted

    def _coerce_dict_observation(
        self,
        obs: dict[str, Any],
        spaces: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Coerce a flat dict observation into the model's expected dict space.

        Returns:
            tuple[dict[str, np.ndarray], list[str]]: Coerced values plus missing keys.
        """
        converted: dict[str, np.ndarray] = {}
        missing: list[str] = []
        aliases: dict[str, tuple[str, ...]] = {
            "robot_speed": ("robot_velocity_xy",),
            "robot_velocity_xy": ("robot_speed",),
        }
        for key, sub_space in spaces.items():
            source = self._resolve_dict_observation_source(obs, str(key), aliases)
            if source is None:
                missing.append(str(key))
                continue
            converted[key] = self._coerce_dict_observation_value(
                source, key=str(key), sub_space=sub_space
            )
        return converted, missing

    @staticmethod
    def _resolve_dict_observation_source(
        obs: dict[str, Any],
        key: str,
        aliases: dict[str, tuple[str, ...]],
    ) -> Any | None:
        """Resolve a dict observation field, falling back to known aliases.

        Returns:
            Any | None: The first matching source value, or ``None`` when missing.
        """
        source = obs.get(key)
        if source is not None:
            return source
        for alias in aliases.get(key, ()):
            if alias in obs:
                return obs[alias]
        return None

    @staticmethod
    def _coerce_dict_observation_value(source: Any, *, key: str, sub_space: Any) -> np.ndarray:
        """Cast one observation value to the dtype and shape required by the model.

        Returns:
            np.ndarray: Array converted to the target dtype and shape.
        """
        target_shape = getattr(sub_space, "shape", None)
        target_dtype = getattr(sub_space, "dtype", None)
        arr = np.asarray(source, dtype=target_dtype)
        if target_shape is not None and tuple(arr.shape) != tuple(target_shape):
            if int(arr.size) != int(np.prod(target_shape)):
                raise ValueError(
                    f"Observation key '{key}' shape mismatch: got {arr.shape}, expected {target_shape}",
                )
            arr = arr.reshape(target_shape)
        return arr

    def _vectorize(self, obs: Observation) -> np.ndarray:
        rp = np.asarray(obs.robot["position"], dtype=float)
        rv = np.asarray(obs.robot["velocity"], dtype=float)
        rg = np.asarray(obs.robot["goal"], dtype=float)
        rel_goal = rg - rp
        ped_rel: list[np.ndarray] = []
        for agent in obs.agents:
            ap = np.asarray(agent.get("position", [0.0, 0.0]), dtype=float)
            ped_rel.append(ap - rp)
        if ped_rel:
            dists = np.linalg.norm(np.stack(ped_rel), axis=1)
            idx = np.argsort(dists)[: max(0, int(self.config.nearest_k))]
            ped_rel_sorted = [ped_rel[i] for i in idx]
        else:
            ped_rel_sorted = []
        while len(ped_rel_sorted) < int(self.config.nearest_k):
            ped_rel_sorted.append(np.zeros(2, dtype=float))
        ped_flat = (
            np.concatenate(ped_rel_sorted[: int(self.config.nearest_k)])
            if self.config.nearest_k > 0
            else np.zeros(0, dtype=float)
        )
        return np.concatenate([rel_goal, rv, ped_flat]).astype(float)

    def _action_vec_to_dict(self, act: np.ndarray) -> dict[str, float]:
        if self.config.action_space == "unicycle":
            v = float(act[0]) if act.size >= 1 else 0.0
            w = float(act[1]) if act.size >= 2 else 0.0
            v = max(0.0, min(v, self.config.v_max))
            w = max(-self.config.omega_max, min(w, self.config.omega_max))
            return {"v": v, "omega": w}
        vx = float(act[0]) if act.size >= 1 else 0.0
        vy = float(act[1]) if act.size >= 2 else 0.0
        speed = float(np.hypot(vx, vy))
        if speed > self.config.v_max and speed > self.EPS:
            scale = self.config.v_max / (speed + self.EPS)
            vx *= scale
            vy *= scale
        return {"vx": vx, "vy": vy}

    def _fallback_action(self, obs: Observation) -> dict[str, float]:
        rp = np.asarray(obs.robot["position"], dtype=float)
        rg = np.asarray(obs.robot["goal"], dtype=float)
        vec = rg - rp
        dist = float(np.linalg.norm(vec))
        if dist < self.EPS:
            return {"v": 0.0, "omega": 0.0}
        direction = vec / max(dist, self.EPS)
        if self.config.action_space == "unicycle":
            heading = float(obs.robot.get("heading", 0.0))
            desired = float(np.arctan2(direction[1], direction[0]))
            error = float(np.arctan2(np.sin(desired - heading), np.cos(desired - heading)))
            return {
                "v": float(min(self.config.v_max, dist)),
                "omega": float(np.clip(error, -self.config.omega_max, self.config.omega_max)),
            }
        return {
            "vx": float(direction[0] * self.config.v_max),
            "vy": float(direction[1] * self.config.v_max),
        }

    def _fallback_action_dict(self, obs: dict[str, Any]) -> dict[str, float]:
        position = np.asarray(obs.get("robot_position", [0.0, 0.0]), dtype=float)
        goal = np.asarray(obs.get("goal_current", [0.0, 0.0]), dtype=float)
        vec = goal - position
        dist = float(np.linalg.norm(vec))
        if dist < self.EPS:
            return {"v": 0.0, "omega": 0.0}
        direction = vec / max(dist, self.EPS)
        if self.config.action_space == "unicycle":
            heading_arr = np.asarray(obs.get("robot_heading", [0.0]), dtype=float).reshape(-1)
            heading = float(heading_arr[0]) if heading_arr.size else 0.0
            desired = float(np.arctan2(direction[1], direction[0]))
            error = float(np.arctan2(np.sin(desired - heading), np.cos(desired - heading)))
            return {
                "v": float(min(self.config.v_max, dist)),
                "omega": float(np.clip(error, -self.config.omega_max, self.config.omega_max)),
            }
        return {
            "vx": float(direction[0] * self.config.v_max),
            "vy": float(direction[1] * self.config.v_max),
        }

    def get_metadata(self) -> dict[str, Any]:
        """Return a serializable description of the loaded SAC planner state."""
        cfg = asdict(self.config)
        cfg["model_path"] = Path(self.config.model_path).name
        payload: dict[str, Any] = {"algorithm": "sac", "status": self._status, "config": cfg}
        if self._fallback_reason is not None:
            payload["fallback_reason"] = self._fallback_reason
        return payload


__all__ = ["SACPlanner", "SACPlannerConfig"]
