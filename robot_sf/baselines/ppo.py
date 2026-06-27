"""PPO baseline adapter for the Social Navigation Benchmark.

This adapter wraps a Stable-Baselines3 PPO policy (.zip) and exposes the
same simple interface as other baselines:

- init(config, seed)
- step(Observation|dict) -> action dict
- reset(), close(), get_metadata()

Observation comes from `robot_sf.baselines.interface.Observation` and is converted
to the model's expected form. We support three modes:

- vector: derive a compact vector from the Observation (relative goal, robot
          velocity, nearest-K pedestrian relative positions). If the loaded
          model expects a different shape, we catch errors and optionally
          fallback to a simple goal-seeking action.
- image: pass-through of an image found under obs.robot["image"]; if missing,
         we raise unless fallback is enabled.
- dict: pass-through of flattened dict observations (for MultiInput PPO models
        trained on SocNav structured keys like `occupancy_grid`, `goal_current`,
        `robot_position`, etc.). Values are cast/reshaped to model space.

The adapter aims to be robust: if prediction fails (shape mismatch, device
issues), we return a goal-seeking fallback action when `fallback_to_goal` is
enabled (default True) so benchmarks can still run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces as gym_spaces
from gymnasium.spaces.utils import flatdim, flatten
from loguru import logger

try:  # Lazy import; not required for type-check only
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover - envs without SB3 installed
    PPO = None  # type: ignore

from robot_sf.baselines.interface import (
    Observation,
    is_observation_mapping,
    observation_from_mapping,
)
from robot_sf.benchmark.local_model_artifacts import validate_no_local_model_path_value
from robot_sf.common.errors import raise_fatal_with_remedy, warn_soft_degrade
from robot_sf.models import resolve_model_path
from robot_sf.planner.predictive_foresight import (
    PredictiveForesightEncoder,
    predictive_foresight_config_from_source,
)


@dataclass
class PPOPlannerConfig:
    """Configuration for the PPO planner adapter."""

    # Required
    model_path: str = "model/ppo_model_retrained_10m_2025-02-01.zip"
    model_id: str | None = None

    # Device handling: "auto" | "cpu" | "cuda" | "cuda:0" etc.
    device: str = "auto"
    deterministic: bool = True

    # Observation handling
    obs_mode: str = "vector"  # "vector" | "image" | "dict"
    nearest_k: int = 5  # K for nearest pedestrian features

    # Action space formatting for benchmark
    action_space: str = "velocity"  # "velocity" | "unicycle"
    v_max: float = 2.0
    omega_max: float = 1.0

    # Robustness
    fallback_to_goal: bool = True
    predictive_foresight_enabled: bool = False
    predictive_foresight_model_id: str = "predictive_proxy_selected_v2_full"
    predictive_foresight_checkpoint_path: str | None = None
    predictive_foresight_device: str = "cpu"
    predictive_foresight_max_agents: int = 16
    predictive_foresight_horizon_steps: int = 8
    predictive_foresight_rollout_dt: float = 0.2
    predictive_foresight_ego_conditioning: bool = False
    predictive_foresight_near_distance: float = 0.7
    predictive_foresight_front_corridor_length: float = 3.0
    predictive_foresight_front_corridor_half_width: float = 1.0


class PPOPlanner:
    """Baseline wrapper for SB3 PPO policies.

    Contract:
    - Inputs: benchmark Observation (robot pos/vel/goal, ped positions, dt)
    - Output: dict with either {"vx","vy"} or {"v","omega"}
    - Errors: On predict failure, returns fallback action when enabled
    """

    EPS: float = 1e-9

    def __init__(
        self,
        config: PPOPlannerConfig | dict[str, Any],
        *,
        seed: int | None = None,
    ):
        """Initialize the PPO planner and load the model if available.

        Args:
            config: Planner configuration or dict payload.
            seed: Optional seed for reproducibility.
        """
        self.config = self._parse_config(config)
        self._seed = seed
        self._model = None
        self._status = "ok"
        self._fallback_reason: str | None = None
        self._predictive_foresight: PredictiveForesightEncoder | None = None
        self._runtime_observation_space: gym_spaces.Space | None = None
        self._load_model()
        self._init_predictive_foresight()

    # --- Lifecycle -----------------------------------------------------
    def _parse_config(self, cfg: PPOPlannerConfig | dict[str, Any]) -> PPOPlannerConfig:
        """Normalize config input into a PPOPlannerConfig instance.

        Args:
            cfg: Configuration object or dict.

        Returns:
            Parsed PPOPlannerConfig.
        """
        if isinstance(cfg, PPOPlannerConfig):
            return cfg
        if isinstance(cfg, dict):
            return PPOPlannerConfig(**cfg)
        raise TypeError(f"Invalid config type: {type(cfg)}")

    def _load_model(self) -> None:
        """Load the PPO model from disk or enter fallback mode."""
        if self.config.model_id is None:
            validate_no_local_model_path_value(
                self.config.model_path,
                owner="PPOPlannerConfig",
            )
        if PPO is None:  # pragma: no cover - missing sb3 at runtime
            warn_soft_degrade(
                "PPO",
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
                    "PPO model",
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
                    "PPO model",
                    f"Model not found at {mp}",
                    "will use fallback-to-goal navigation",
                )
                self._model = None
                self._status = "fallback"
                self._fallback_reason = "model_missing"
                return
            raise_fatal_with_remedy(
                f"PPO model file not found: {mp}",
                f"Place model at '{mp}' or check available models in model/ directory. "
                "Download from releases or train with scripts/training/train_ppo.py --config ...",
            )
        try:
            # Avoid printing system info in CI/test logs
            self._model = PPO.load(str(mp), device=self.config.device, print_system_info=False)
            self._status = "ok"
            self._fallback_reason = None
        except (RuntimeError, ValueError, OSError) as e:
            if self.config.fallback_to_goal:
                warn_soft_degrade(
                    "PPO model",
                    f"Failed to load model: {e}",
                    "will use fallback-to-goal navigation",
                )
                self._model = None
                self._status = "fallback"
                self._fallback_reason = "model_load_failed"
                return
            raise_fatal_with_remedy(
                f"Failed to load PPO model from {mp}: {e}",
                "Check model compatibility with current stable_baselines3 version. "
                "Re-train if needed using scripts/training/train_ppo.py --config ...",
            )

    def reset(self, *, seed: int | None = None) -> None:
        # No RNN state; just update seed and keep model
        """Reset planner state (currently only updates seed).

        Args:
            seed: Optional new seed value.
        """
        if seed is not None:
            self._seed = seed

    def bind_env(self, env: Any) -> None:
        """Bind runtime observation space for dict-to-Box checkpoint adapters."""

        obs_space = getattr(env, "observation_space", None)
        if isinstance(obs_space, gym_spaces.Space):
            self._runtime_observation_space = obs_space
            self._validate_runtime_observation_space()

    def close(self) -> None:
        """Release the loaded PPO model."""
        self._model = None

    def configure(self, config: PPOPlannerConfig | dict[str, Any]) -> None:
        """Update the planner's configuration."""
        self.config = self._parse_config(config)
        # Need to reload the model if model_path changed
        self._load_model()
        self._init_predictive_foresight()

    def _init_predictive_foresight(self) -> None:
        """(Re)build the optional predictive foresight encoder from current config."""
        self._predictive_foresight = None
        foresight_cfg = predictive_foresight_config_from_source(
            self.config,
            default_max_agents=self.config.predictive_foresight_max_agents,
        )
        if not foresight_cfg.enabled:
            return
        self._predictive_foresight = PredictiveForesightEncoder(foresight_cfg)

    # --- API -----------------------------------------------------------
    def step(self, obs: Observation | dict[str, Any]) -> dict[str, float]:
        """Compute a planner action for the given observation.

        Args:
            obs: SocNav-style observation or Observation instance.

        Returns:
            Action dict in either velocity or unicycle format.
        """
        if is_observation_mapping(obs) and self._uses_dict_observation():
            return self._step_dict_obs(obs)

        if is_observation_mapping(obs):
            obs = observation_from_mapping(obs)
        assert isinstance(obs, Observation)

        # Try model predict
        try:
            action_vec = self._predict_action(obs)
            if action_vec is None:
                raise RuntimeError("PPO model unavailable or prediction failed")
            return self._action_vec_to_dict(action_vec, obs)
        except (RuntimeError, ValueError, OSError):
            # Fallback for robustness on common prediction errors
            if self.config.fallback_to_goal:
                if self._status != "fallback":
                    self._status = "fallback"
                if self._fallback_reason is None:
                    self._fallback_reason = "prediction_failed"
                return self._fallback_action(obs)
            raise

    def _uses_dict_observation(self) -> bool:
        """Return whether planner is configured for native dict observations."""
        return str(self.config.obs_mode).strip().lower() in {"dict", "native_dict", "multi_input"}

    def _step_dict_obs(self, obs: dict[str, Any]) -> dict[str, float]:
        """Predict an action from flattened dict observations expected by MultiInput PPO.

        Returns:
            Action dict in configured output space.
        """
        try:
            model_obs = self._build_model_obs_dict(obs)
            action_vec = self._predict_action(model_obs)
            if action_vec is None:
                raise RuntimeError("PPO model unavailable or prediction failed")
            return self._action_vec_to_dict_from_array(action_vec)
        except (RuntimeError, ValueError, OSError):
            if self.config.fallback_to_goal:
                if self._status != "fallback":
                    self._status = "fallback"
                if self._fallback_reason is None:
                    self._fallback_reason = "prediction_failed"
                return self._fallback_action_dict(obs)
            raise

    # --- Helpers -------------------------------------------------------
    def _predict_action(self, model_obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | None:
        """Run PPO inference and return the raw action vector.

        Args:
            model_obs: Model-ready observation payload.

        Returns:
            Action vector or None when unavailable.
        """
        if self._model is None:
            return None

        # SB3 supports batch and single obs; ensure correct shape for vector
        if isinstance(model_obs, np.ndarray) and model_obs.ndim == 1:
            model_obs_in = model_obs  # SB3 accepts 1D for single obs
        else:
            model_obs_in = model_obs
        try:
            act, _ = self._model.predict(model_obs_in, deterministic=self.config.deterministic)
            act = np.asarray(act, dtype=float).squeeze()
            return act
        except (
            RuntimeError,
            ValueError,
            OSError,
            IndexError,
        ) as exc:  # predict-time errors we can recover from
            # Log at debug level for diagnostics; fall back to goal if enabled
            logger.debug("PPO model prediction failed: %s", exc, exc_info=True)
            return None

    def _build_model_obs_dict(self, obs: dict[str, Any]) -> dict[str, np.ndarray] | np.ndarray:
        """Build model-ready observation payload and align it to the model space.

        Returns:
            Payload shaped and typed to match the loaded model's observation space.
        """
        if self._model is None:
            return {str(k): np.asarray(v) for k, v in obs.items()}

        space = getattr(self._model, "observation_space", None)
        spaces = getattr(space, "spaces", None)
        if not isinstance(spaces, dict):
            if isinstance(space, gym_spaces.Box):
                return self._build_model_obs_flat_box(obs, space)
            return {str(k): np.asarray(v) for k, v in obs.items()}

        source_obs = self._source_obs_with_predictive_backfill(obs, spaces)
        return self._align_model_obs_dict(source_obs, spaces)

    def _source_obs_with_predictive_backfill(
        self,
        obs: dict[str, Any],
        spaces: dict[str, Any],
    ) -> dict[str, Any]:
        """Return dict observations plus any missing predictive feature payload."""

        source_obs = self._flatten_nested_observation(obs)
        source_obs = self._expand_flat_observation_for_spaces(source_obs, spaces)
        if self._predictive_foresight is not None:
            missing_predictive = [
                key for key in spaces if key.startswith("predictive_") and key not in source_obs
            ]
            if missing_predictive:
                payload = self._predictive_feature_payload(source_obs)
                source_obs.update(
                    {key: value for key, value in payload.items() if key in missing_predictive}
                )
        return source_obs

    @staticmethod
    def _flatten_nested_observation(obs: dict[str, Any]) -> dict[str, Any]:
        """Return observation values with nested SocNav leaves promoted to flat keys."""

        flattened: dict[str, Any] = dict(obs)

        def _flatten_recursive(payload: dict[str, Any], prefix: str = "") -> None:
            for key, value in payload.items():
                full_key = f"{prefix}_{key}" if prefix else str(key)
                if isinstance(value, dict):
                    _flatten_recursive(value, full_key)
                elif full_key not in flattened:
                    flattened[full_key] = value

        _flatten_recursive(obs)
        return flattened

    @classmethod
    def _expand_flat_observation_for_spaces(
        cls,
        obs: dict[str, Any],
        spaces: dict[str, Any],
    ) -> dict[str, Any]:
        """Return observation with flat map-runner leaves exposed as nested Dict keys."""

        expanded: dict[str, Any] = dict(obs)
        for key, sub_space in spaces.items():
            key_str = str(key)
            if key_str in expanded:
                continue
            sub_spaces = getattr(sub_space, "spaces", None)
            if not isinstance(sub_spaces, dict):
                continue
            nested = cls._nested_observation_from_flat_prefix(expanded, key_str, sub_spaces)
            if nested:
                expanded[key_str] = nested
        return expanded

    @classmethod
    def _nested_observation_from_flat_prefix(
        cls,
        obs: dict[str, Any],
        prefix: str,
        spaces: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a nested observation block from ``prefix_child`` flat keys.

        Returns:
            Nested observation payload containing the available declared leaves.
        """

        nested: dict[str, Any] = {}
        for key, sub_space in spaces.items():
            key_str = str(key)
            flat_key = f"{prefix}_{key_str}"
            sub_spaces = getattr(sub_space, "spaces", None)
            if isinstance(sub_spaces, dict):
                child = cls._nested_observation_from_flat_prefix(obs, flat_key, sub_spaces)
                if child:
                    nested[key_str] = child
                continue
            if flat_key in obs:
                nested[key_str] = obs[flat_key]
        return nested

    def _align_model_obs_dict(
        self,
        source_obs: dict[str, Any],
        spaces: dict[str, Any],
    ) -> dict[str, Any]:
        """Align source observation fields to a model-declared Dict space.

        Keys declared by the model space but absent from ``source_obs`` (after
        flattening, expansion, and alias resolution) are backfilled with an
        in-bounds default derived from the target subspace rather than raising.
        This keeps PPO evaluation running when a runner emits a subset of the
        keys a checkpoint declares (see issue #3704); the backfill is logged so
        the substitution stays visible, and callers should treat heavily
        backfilled runs as degraded rather than faithful evidence.

        Returns:
            Dict payload shaped and typed to match the model-declared subspaces.
        """

        converted: dict[str, Any] = {}
        backfilled: list[str] = []
        aliases: dict[str, tuple[str, ...]] = {
            "robot_speed": ("robot_velocity_xy",),
            "robot_velocity_xy": ("robot_speed",),
        }
        for key, sub_space in spaces.items():
            source = self._resolve_dict_observation_source(source_obs, str(key), aliases)
            if source is None:
                converted[key] = self._default_for_space(sub_space)
                backfilled.append(str(key))
                continue
            sub_spaces = getattr(sub_space, "spaces", None)
            if isinstance(sub_spaces, dict):
                if not isinstance(source, dict):
                    raise ValueError(
                        f"Observation key '{key}' expected nested Dict payload, "
                        f"got {type(source).__name__}",
                    )
                converted[key] = self._align_model_obs_dict(source, sub_spaces)
                continue
            target_shape = getattr(sub_space, "shape", None)
            target_dtype = getattr(sub_space, "dtype", None)
            arr = np.asarray(source, dtype=target_dtype)
            if target_shape is not None and tuple(arr.shape) != tuple(target_shape):
                target_size = int(np.prod(target_shape))
                if int(arr.size) != target_size:
                    raise ValueError(
                        f"Observation key '{key}' shape mismatch: got {arr.shape}, "
                        f"expected {target_shape}",
                    )
                arr = arr.reshape(target_shape)
            converted[key] = arr
        if backfilled:
            logger.debug(
                "PPO dict observation backfilled {} missing key(s) with space defaults: {}",
                len(backfilled),
                ", ".join(backfilled[:6]),
            )
        return converted

    @classmethod
    def _default_for_space(cls, sub_space: Any) -> Any:
        """Build an in-bounds default payload for a missing model observation key.

        Nested Dict subspaces recurse so each declared leaf is materialized;
        leaf Box subspaces yield zeros clipped to the (finite) box bounds, so the
        default never falls outside a declared low/high range. Unknown subspace
        types fall back to a zero array matching the declared shape.

        Returns:
            A nested dict, or an ``np.ndarray`` default for the subspace.
        """

        sub_spaces = getattr(sub_space, "spaces", None)
        if isinstance(sub_spaces, dict):
            return {str(key): cls._default_for_space(leaf) for key, leaf in sub_spaces.items()}

        target_shape = getattr(sub_space, "shape", None)
        target_dtype = getattr(sub_space, "dtype", None) or np.float32
        shape = tuple(target_shape) if target_shape is not None else ()
        default = np.zeros(shape, dtype=target_dtype)
        low = getattr(sub_space, "low", None)
        high = getattr(sub_space, "high", None)
        if low is not None and high is not None:
            # Clip the zero default into the declared range so a non-zero lower
            # bound (e.g. counts/radii bounded away from 0) stays in-bounds.
            default = np.clip(default, low, high).astype(target_dtype)
        return default

    @staticmethod
    def _resolve_dict_observation_source(
        obs: dict[str, Any],
        key: str,
        aliases: dict[str, tuple[str, ...]],
    ) -> Any | None:
        """Resolve a model observation field from direct keys or known aliases.

        Returns:
            The resolved observation value, or None when the key is unavailable.
        """

        source = obs.get(key)
        if source is not None:
            return source
        for alias in aliases.get(key, ()):
            if alias in obs:
                return obs[alias]
        return None

    def _validate_runtime_observation_space(self) -> None:
        """Fail closed when a flat checkpoint cannot match runtime observations."""

        if self._model is None:
            return
        model_space = getattr(self._model, "observation_space", None)
        if not isinstance(model_space, gym_spaces.Box) or not self._uses_dict_observation():
            return

        runtime_space = self._runtime_observation_space
        model_size = int(np.prod(model_space.shape))
        if isinstance(runtime_space, gym_spaces.Dict):
            runtime_size = int(flatdim(runtime_space))
            if runtime_size != model_size:
                raise ValueError(
                    "Runtime Dict observation space does not flatten to the checkpoint "
                    f"Box shape: runtime flatdim={runtime_size}, "
                    f"checkpoint shape={tuple(model_space.shape)}."
                )
            return
        if isinstance(runtime_space, gym_spaces.Box):
            runtime_size = int(np.prod(runtime_space.shape))
            if runtime_size != model_size:
                raise ValueError(
                    "Runtime Box observation space does not match checkpoint Box shape: "
                    f"runtime shape={tuple(runtime_space.shape)}, "
                    f"checkpoint shape={tuple(model_space.shape)}."
                )
            return
        raise ValueError(
            "PPO checkpoint expects a flat Box observation but runtime observation_space "
            f"is {type(runtime_space).__name__}; cannot prove adapter compatibility."
        )

    def _build_model_obs_flat_box(
        self,
        obs: dict[str, Any],
        model_space: gym_spaces.Box,
    ) -> np.ndarray:
        """Flatten a runtime Dict observation for a flat-Box SB3 checkpoint.

        Returns:
            Model-ready flat observation matching the checkpoint shape.
        """

        runtime_space = self._runtime_observation_space
        if isinstance(runtime_space, gym_spaces.Dict):
            try:
                flat_obs = np.asarray(flatten(runtime_space, obs), dtype=model_space.dtype)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Failed to flatten runtime Dict observation for PPO checkpoint."
                ) from exc
        else:
            flat_obs = np.asarray(obs, dtype=model_space.dtype)

        target_shape = tuple(model_space.shape)
        if tuple(flat_obs.shape) != target_shape:
            target_size = int(np.prod(target_shape))
            if int(flat_obs.size) != target_size:
                raise ValueError(
                    "Flattened PPO observation size mismatch: "
                    f"got shape {tuple(flat_obs.shape)}, expected {target_shape}."
                )
            flat_obs = flat_obs.reshape(target_shape)
        return flat_obs

    def _predictive_feature_payload(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Map nested predictive foresight outputs to flat PPO observation keys.

        Returns:
            dict[str, np.ndarray]: Flat `predictive_*` feature payload.
        """
        if self._predictive_foresight is None:
            return {}
        feature_block = self._predictive_foresight.encode(
            self._normalize_predictive_foresight_obs(obs)
        )
        return {f"predictive_{key}": value for key, value in feature_block.items()}

    def _normalize_predictive_foresight_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Adapt flat PPO dict observations into the structured SocNav format for foresight.

        Returns:
            dict[str, Any]: Structured observation block compatible with the foresight encoder.
        """
        if "robot" in obs and "goal" in obs and "pedestrians" in obs:
            return obs

        def _arr(key: str, default: list[float] | list[list[float]], *, dtype=float) -> np.ndarray:
            """Read one observation field as a NumPy array with a fallback default.

            Returns:
                np.ndarray: Array view of the observation value or default.
            """
            return np.asarray(obs.get(key, default), dtype=dtype)

        return {
            "robot": {
                "position": _arr("robot_position", [0.0, 0.0], dtype=np.float32).reshape(-1)[:2],
                "heading": _arr("robot_heading", [0.0], dtype=np.float32).reshape(-1)[:1],
                "speed": _arr("robot_speed", [0.0, 0.0], dtype=np.float32).reshape(-1)[:2],
                "radius": _arr("robot_radius", [0.3], dtype=np.float32).reshape(-1)[:1],
            },
            "goal": {
                "current": _arr("goal_current", [0.0, 0.0], dtype=np.float32).reshape(-1)[:2],
                "next": _arr("goal_next", [0.0, 0.0], dtype=np.float32).reshape(-1)[:2],
            },
            "pedestrians": {
                "positions": _arr(
                    "pedestrians_positions", np.zeros((0, 2), dtype=np.float32), dtype=np.float32
                ),
                "velocities": _arr(
                    "pedestrians_velocities", np.zeros((0, 2), dtype=np.float32), dtype=np.float32
                ),
                "count": _arr("pedestrians_count", [0.0], dtype=np.float32).reshape(-1)[:1],
                "radius": _arr("pedestrians_radius", [0.3], dtype=np.float32).reshape(-1)[:1],
            },
            "map": {
                "size": _arr("map_size", [20.0, 20.0], dtype=np.float32).reshape(-1)[:2],
            },
            "sim": {
                "timestep": _arr("sim_timestep", [0.1], dtype=np.float32).reshape(-1)[:1],
            },
        }

    def _build_model_obs(self, obs: Observation) -> np.ndarray:
        """Convert the benchmark observation into model input format.

        Args:
            obs: Benchmark observation.

        Returns:
            Model-ready observation array.
        """
        if self.config.obs_mode == "image":
            img = obs.robot.get("image") if isinstance(obs.robot, dict) else None
            if img is None:
                raise ValueError("Image observation requested but obs.robot['image'] is missing")
            return np.asarray(img)
        # Default: vector mode
        return self._vectorize(obs)

    def _vectorize(self, obs: Observation) -> np.ndarray:
        """Build a compact vector observation from the structured input.

        Args:
            obs: Benchmark observation.

        Returns:
            Flattened vector observation.
        """
        rp = np.asarray(obs.robot["position"], dtype=float)
        rv = np.asarray(obs.robot["velocity"], dtype=float)
        rg = np.asarray(obs.robot["goal"], dtype=float)
        rel_goal = rg - rp
        # Nearest-K pedestrian relative positions
        ped_rel: list[np.ndarray] = []
        for a in obs.agents:
            ap = np.asarray(a.get("position", [0.0, 0.0]), dtype=float)
            ped_rel.append(ap - rp)
        if ped_rel:
            dists = np.linalg.norm(np.stack(ped_rel), axis=1)
            idx = np.argsort(dists)[: max(0, int(self.config.nearest_k))]
            ped_rel_sorted = [ped_rel[i] for i in idx]
        else:
            ped_rel_sorted = []
        # Pad to K
        K = int(self.config.nearest_k)
        while len(ped_rel_sorted) < K:
            ped_rel_sorted.append(np.zeros(2, dtype=float))
        ped_flat = np.concatenate(ped_rel_sorted[:K]) if K > 0 else np.zeros(0, dtype=float)
        vec = np.concatenate([rel_goal, rv, ped_flat]).astype(float)
        return vec

    def _action_vec_to_dict_from_array(self, act: np.ndarray) -> dict[str, float]:
        """Convert a raw action vector to the configured action dictionary.

        Returns:
            Action dict in either velocity or unicycle format.
        """
        if self.config.action_space == "unicycle":
            # Expect [v, omega]
            v = float(act[0]) if act.size >= 1 else 0.0
            w = float(act[1]) if act.size >= 2 else 0.0
            v = max(0.0, min(v, self.config.v_max))
            w = max(-self.config.omega_max, min(w, self.config.omega_max))
            return {"v": v, "omega": w}
        # Default velocity space: expect [vx, vy]
        vx = float(act[0]) if act.size >= 1 else 0.0
        vy = float(act[1]) if act.size >= 2 else 0.0
        # Optional clamp to v_max
        spd = float(np.hypot(vx, vy))
        if spd > self.config.v_max and spd > self.EPS:
            scale = self.config.v_max / (spd + self.EPS)
            vx *= scale
            vy *= scale
        return {"vx": vx, "vy": vy}

    def _action_vec_to_dict(self, act: np.ndarray, _obs: Observation) -> dict[str, float]:
        """Convert raw action vector to configured action dict for Observation mode.

        Returns:
            Action dict in configured output space.
        """
        return self._action_vec_to_dict_from_array(act)

    def _fallback_action(self, obs: Observation) -> dict[str, float]:
        """Return a simple goal-seeking action when PPO is unavailable.

        Args:
            obs: Benchmark observation.

        Returns:
            Goal-directed action dict.
        """
        rp = np.asarray(obs.robot["position"], dtype=float)
        rg = np.asarray(obs.robot["goal"], dtype=float)
        vec = rg - rp
        dist = float(np.linalg.norm(vec))
        if dist < self.EPS:
            if self.config.action_space == "unicycle":
                return {"v": 0.0, "omega": 0.0}
            return {"vx": 0.0, "vy": 0.0}
        dir_unit = vec / dist
        if self.config.action_space == "unicycle":
            return {"v": min(self.config.v_max, dist), "omega": 0.0}
        return {
            "vx": float(dir_unit[0] * min(self.config.v_max, dist)),
            "vy": float(dir_unit[1] * min(self.config.v_max, dist)),
        }

    def _fallback_action_dict(self, obs: dict[str, Any]) -> dict[str, float]:
        """Goal-seeking fallback for flattened dict observations.

        Returns:
            Goal-directed fallback action dict.
        """
        source_obs = self._flatten_nested_observation(obs)
        robot_pos = np.asarray(source_obs.get("robot_position", [0.0, 0.0]), dtype=float).reshape(
            -1
        )
        goal_pos = np.asarray(source_obs.get("goal_current", [0.0, 0.0]), dtype=float).reshape(-1)
        if robot_pos.size < 2 or goal_pos.size < 2:
            if self.config.action_space == "unicycle":
                return {"v": 0.0, "omega": 0.0}
            return {"vx": 0.0, "vy": 0.0}
        delta = goal_pos[:2] - robot_pos[:2]
        dist = float(np.linalg.norm(delta))
        if dist < self.EPS:
            if self.config.action_space == "unicycle":
                return {"v": 0.0, "omega": 0.0}
            return {"vx": 0.0, "vy": 0.0}
        unit = delta / dist
        if self.config.action_space == "unicycle":
            return {"v": min(self.config.v_max, dist), "omega": 0.0}
        return {
            "vx": float(unit[0] * min(self.config.v_max, dist)),
            "vy": float(unit[1] * min(self.config.v_max, dist)),
        }

    # --- Metadata ------------------------------------------------------
    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the planner and load status.


        Returns:
            Metadata dict including status/fallback info.
        """
        cfg = asdict(self.config)
        # Avoid leaking full paths in metadata
        cfg["model_path"] = Path(self.config.model_path).name
        checkpoint_path = cfg.get("predictive_foresight_checkpoint_path")
        if isinstance(checkpoint_path, str) and checkpoint_path:
            cfg["predictive_foresight_checkpoint_path"] = Path(checkpoint_path).name
        meta = {"algorithm": "ppo", "config": cfg, "status": self._status}
        if self._fallback_reason:
            meta["fallback_reason"] = self._fallback_reason
        return meta


__all__ = ["PPOPlanner", "PPOPlannerConfig"]
