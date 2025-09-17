"""Random baseline planner for the Social Navigation Benchmark.

Produces simple random actions in either velocity or unicycle action spaces.
Values are clamped to configured limits. Deterministic when seeded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import numpy as np


@dataclass
class RandomPlannerConfig:
    mode: str = "velocity"  # "velocity" or "unicycle"
    v_max: float = 2.0
    omega_max: float = 1.0
    dt: float = 0.1
    safety_clamp: bool = True
    noise_std: float = 0.0  # Optional Gaussian jitter applied to sampled actions


@dataclass
class Observation:
    dt: float
    robot: Dict[str, Any]
    agents: list[Dict[str, Any]]
    obstacles: list[Any]


class RandomPlanner:
    def __init__(
        self, config: Union[Dict[str, Any], RandomPlannerConfig], *, seed: Optional[int] = None
    ):
        self.config = self._parse_config(config)
        self._rng = np.random.default_rng(seed)

    def _parse_config(
        self, config: Union[Dict[str, Any], RandomPlannerConfig]
    ) -> RandomPlannerConfig:
        if isinstance(config, dict):
            return RandomPlannerConfig(**config)
        if isinstance(config, RandomPlannerConfig):
            return config
        raise TypeError(f"Invalid config type: {type(config)}")

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def configure(self, config: Union[Dict[str, Any], RandomPlannerConfig]) -> None:
        self.config = self._parse_config(config)

    def step(self, obs: Union[Observation, Dict[str, Any]]) -> Dict[str, float]:
        # Support dict-style Observation
        if isinstance(obs, dict):
            obs = Observation(**obs)  # type: ignore[arg-type]

        if self.config.mode == "velocity":
            # Sample vx, vy uniformly in a disk scaled to v_max
            angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
            speed = float(self._rng.uniform(0.0, self.config.v_max))
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            if self.config.noise_std > 0:
                vx += float(self._rng.normal(0.0, self.config.noise_std))
                vy += float(self._rng.normal(0.0, self.config.noise_std))
            if self.config.safety_clamp:
                s = float(np.hypot(vx, vy))
                if s > self.config.v_max and s > 1e-9:
                    scale = self.config.v_max / s
                    vx *= scale
                    vy *= scale
            return {"vx": vx, "vy": vy}

        if self.config.mode == "unicycle":
            # Sample forward speed and angular rate independently
            v = float(self._rng.uniform(0.0, self.config.v_max))
            omega = float(self._rng.uniform(-self.config.omega_max, self.config.omega_max))
            if self.config.noise_std > 0:
                v = max(0.0, v + float(self._rng.normal(0.0, self.config.noise_std)))
                omega += float(self._rng.normal(0.0, self.config.noise_std))
            if self.config.safety_clamp:
                v = max(0.0, min(v, self.config.v_max))
                omega = max(-self.config.omega_max, min(omega, self.config.omega_max))
            return {"v": v, "omega": omega}

        raise ValueError(f"Unknown mode: {self.config.mode}")

    def close(self) -> None:  # For API symmetry
        pass

    def get_metadata(self) -> Dict[str, Any]:
        cfg = asdict(self.config)
        cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        return {"algorithm": "random", "config": cfg, "config_hash": cfg_hash}


__all__ = ["RandomPlanner", "RandomPlannerConfig", "Observation"]
