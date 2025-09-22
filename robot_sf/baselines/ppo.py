"""PPO baseline adapter for the Social Navigation Benchmark.

This adapter wraps a Stable-Baselines3 PPO policy (.zip) and exposes the
same simple interface as other baselines:

- init(config, seed)
- step(Observation|dict) -> action dict
- reset(), close(), get_metadata()

Observation comes from `robot_sf.baselines.social_force.Observation` and is
converted to the model's expected form. We support two modes:

- vector: derive a compact vector from the Observation (relative goal, robot
          velocity, nearest-K pedestrian relative positions). If the loaded
          model expects a different shape, we catch errors and optionally
          fallback to a simple goal-seeking action.
- image: pass-through of an image found under obs.robot["image"]; if missing,
         we raise unless fallback is enabled.

The adapter aims to be robust: if prediction fails (shape mismatch, device
issues), we return a goal-seeking fallback action when `fallback_to_goal` is
enabled (default True) so benchmarks can still run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:  # Lazy import; not required for type-check only
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - envs without SB3 installed
    PPO = None  # type: ignore

from robot_sf.baselines.social_force import Observation


@dataclass
class PPOPlannerConfig:
    # Required
    model_path: str = "model/ppo_model_retrained_10m_2025-02-01.zip"

    # Device handling: "auto" | "cpu" | "cuda" | "cuda:0" etc.
    device: str = "auto"
    deterministic: bool = True

    # Observation handling
    obs_mode: str = "vector"  # "vector" | "image"
    nearest_k: int = 5  # K for nearest pedestrian features

    # Action space formatting for benchmark
    action_space: str = "velocity"  # "velocity" | "unicycle"
    v_max: float = 2.0
    omega_max: float = 1.0

    # Robustness
    fallback_to_goal: bool = True


class PPOPlanner:
    """Baseline wrapper for SB3 PPO policies.

    Contract:
    - Inputs: benchmark Observation (robot pos/vel/goal, ped positions, dt)
    - Output: dict with either {"vx","vy"} or {"v","omega"}
    - Errors: On predict failure, returns fallback action when enabled
    """

    EPS: float = 1e-9

    def __init__(
        self, config: Union[PPOPlannerConfig, Dict[str, Any]], *, seed: Optional[int] = None
    ):
        self.config = self._parse_config(config)
        self._seed = seed
        self._model = None
        self._load_model()

    # --- Lifecycle -----------------------------------------------------
    def _parse_config(self, cfg: Union[PPOPlannerConfig, Dict[str, Any]]) -> PPOPlannerConfig:
        if isinstance(cfg, PPOPlannerConfig):
            return cfg
        if isinstance(cfg, dict):
            return PPOPlannerConfig(**cfg)
        raise TypeError(f"Invalid config type: {type(cfg)}")

    def _load_model(self) -> None:
        if PPO is None:  # pragma: no cover - missing sb3 at runtime
            self._model = None
            return
        mp = Path(self.config.model_path)
        if not mp.exists():
            # Delay error until predict; allow smoke tests to run with fallback
            self._model = None
            return
        try:
            # Avoid printing system info in CI/test logs
            self._model = PPO.load(str(mp), device=self.config.device, print_system_info=False)
        except Exception:
            # Defer to fallback
            self._model = None

    def reset(self, *, seed: Optional[int] = None) -> None:
        # No RNN state; just update seed and keep model
        if seed is not None:
            self._seed = seed

    def close(self) -> None:
        self._model = None

    def configure(self, config: Union[PPOPlannerConfig, Dict[str, Any]]) -> None:
        """Update the planner's configuration."""
        self.config = self._parse_config(config)
        # Need to reload the model if model_path changed
        self._load_model()

    # --- API -----------------------------------------------------------
    def step(self, obs: Union[Observation, Dict[str, Any]]) -> Dict[str, float]:
        if isinstance(obs, dict):
            obs = Observation(**obs)  # type: ignore[arg-type]
        assert isinstance(obs, Observation)

        # Try model predict
        try:
            action_vec = self._predict_action(obs)
            if action_vec is None:
                raise RuntimeError("PPO model unavailable or prediction failed")
            return self._action_vec_to_dict(action_vec, obs)
        except Exception:
            # Fallback for robustness
            if self.config.fallback_to_goal:
                return self._fallback_action(obs)
            raise

    # --- Helpers -------------------------------------------------------
    def _predict_action(self, obs: Observation) -> Optional[np.ndarray]:
        if self._model is None:
            return None

        model_obs = self._build_model_obs(obs)
        # SB3 supports batch and single obs; ensure correct shape for vector
        if isinstance(model_obs, np.ndarray) and model_obs.ndim == 1:
            model_obs_in = model_obs  # SB3 accepts 1D for single obs
        else:
            model_obs_in = model_obs
        act, _ = self._model.predict(model_obs_in, deterministic=self.config.deterministic)
        act = np.asarray(act, dtype=float).squeeze()
        return act

    def _build_model_obs(self, obs: Observation) -> np.ndarray:
        if self.config.obs_mode == "image":
            img = obs.robot.get("image") if isinstance(obs.robot, dict) else None
            if img is None:
                raise ValueError("Image observation requested but obs.robot['image'] is missing")
            return np.asarray(img)
        # Default: vector mode
        return self._vectorize(obs)

    def _vectorize(self, obs: Observation) -> np.ndarray:
        rp = np.asarray(obs.robot["position"], dtype=float)
        rv = np.asarray(obs.robot["velocity"], dtype=float)
        rg = np.asarray(obs.robot["goal"], dtype=float)
        rel_goal = rg - rp
        # Nearest-K pedestrian relative positions
        ped_rel: List[np.ndarray] = []
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

    def _action_vec_to_dict(self, act: np.ndarray, _obs: Observation) -> Dict[str, float]:
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

    def _fallback_action(self, obs: Observation) -> Dict[str, float]:
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

    # --- Metadata ------------------------------------------------------
    def get_metadata(self) -> Dict[str, Any]:
        cfg = asdict(self.config)
        # Avoid leaking full paths in metadata
        cfg["model_path"] = str(Path(self.config.model_path))
        return {"algorithm": "ppo", "config": cfg}


__all__ = ["PPOPlanner", "PPOPlannerConfig"]
