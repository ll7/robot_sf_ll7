"""DR-MPC baseline wrapper for the Social Navigation Benchmark.

This module provides a dependency-aware wrapper around an external DR-MPC implementation.
The wrapper is designed to be importable without the external package and to fail with a
clear diagnostic when the required runtime dependency is missing.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DRMPCPlannerConfig:
    """Configuration for the DR-MPC baseline wrapper."""

    checkpoint_path: str | None = None
    repo_root: str = "third_party/external_mpc_repos/dr_mpc"
    upstream_repo_url: str = "https://github.com/James-R-Han/DR-MPC"
    device: str = "cpu"
    mode: str = "unicycle"  # "velocity" or "unicycle"
    v_max: float = 2.0
    omega_max: float = 1.0
    safety_clamp: bool = True
    action_space: str = "unicycle"
    fallback_on_error: bool = False
    allow_testing_algorithms: bool = True
    include_in_paper: bool = False


@dataclass
class Observation:
    """Observation payload for the DR-MPC baseline wrapper."""

    dt: float
    robot: dict[str, Any]
    agents: list[dict[str, Any]]
    obstacles: list[Any]


class DRMPCPlanner:
    """Baseline adapter for external DR-MPC implementations."""

    def __init__(self, config: dict[str, Any] | DRMPCPlannerConfig, *, seed: int | None = None):
        """Initialize the DR-MPC wrapper with config and optional seed."""
        self.config = self._parse_config(config)
        self._policy: Any | None = None
        self._module: Any | None = None
        self._seed = seed

    def _parse_config(self, config: dict[str, Any] | DRMPCPlannerConfig) -> DRMPCPlannerConfig:
        if isinstance(config, dict):
            return build_dr_mpc_config(config)
        if isinstance(config, DRMPCPlannerConfig):
            return config
        raise TypeError(f"Invalid config type: {type(config)}")

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the DR-MPC wrapper state and optionally reseed the RNG."""
        if seed is not None:
            self._seed = seed
        self._policy = None
        self._module = None

    def configure(self, config: dict[str, Any] | DRMPCPlannerConfig) -> None:
        """Update the DR-MPC wrapper configuration."""
        self.config = self._parse_config(config)
        self._policy = None
        self._module = None

    def _import_dr_mpc_module(self) -> Any:
        if self._module is not None:
            return self._module

        for module_name in ("dr_mpc", "drmpc", "DR_MPC"):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            self._module = module
            return module

        raise ImportError(
            "DR-MPC dependency is missing. Install the DR-MPC package or make it available "
            "on PYTHONPATH to use the `dr_mpc` baseline."
        )

    def _build_policy(self) -> Any:
        try:
            module = self._import_dr_mpc_module()
        except ImportError as exc:
            raise RuntimeError(str(exc)) from exc

        if hasattr(module, "DRMPCPolicy"):
            policy_cls = module.DRMPCPolicy
            return policy_cls(
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device,
            )

        if hasattr(module, "load_policy"):
            return module.load_policy(
                self.config.checkpoint_path,
                device=self.config.device,
            )

        raise RuntimeError(
            "Imported DR-MPC package does not expose a supported policy constructor. "
            "Expected `DRMPCPolicy` or `load_policy`."
        )

    def step(self, obs: Observation | dict[str, Any]) -> dict[str, float]:
        """Compute a DR-MPC action for the current observation.

        Args:
            obs: The current observation payload.

        Returns:
            A dictionary action in either velocity or unicycle format.
        """
        if isinstance(obs, dict):
            obs = Observation(**obs)  # type: ignore[arg-type]

        if self._policy is None:
            self._policy = self._build_policy()

        action = self._policy.select_action(obs)
        if not isinstance(action, dict):
            raise ValueError("DR-MPC policy returned an invalid action payload")

        self._clamp_action(action)
        return action

    def _clamp_action(self, action: dict[str, float]) -> None:
        if self.config.safety_clamp:
            if "vx" in action and "vy" in action:
                speed = float(np.hypot(action["vx"], action["vy"]))
                if speed > self.config.v_max and speed > 1e-9:
                    scale = self.config.v_max / speed
                    action["vx"] *= scale
                    action["vy"] *= scale
            if "v" in action:
                action["v"] = max(0.0, min(float(action["v"]), self.config.v_max))
            if "omega" in action:
                action["omega"] = max(
                    -self.config.omega_max,
                    min(float(action["omega"]), self.config.omega_max),
                )

    def close(self) -> None:
        """Release any DR-MPC wrapper resources."""
        self._policy = None
        self._module = None

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the DR-MPC planner."""
        cfg = asdict(self.config)
        config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        status = "ok"

        try:
            self._import_dr_mpc_module()
        except (ImportError, RuntimeError):
            status = "missing_dependency"

        return {
            "algorithm": "dr_mpc",
            "config": cfg,
            "config_hash": config_hash,
            "status": status,
        }


def build_dr_mpc_config(data: dict[str, Any] | None) -> DRMPCPlannerConfig:
    """Build a DR-MPC config from a loose mapping while preserving explicit provenance.

    Returns:
        DRMPCPlannerConfig: Normalized config with repo-root provenance preserved.
    """
    payload = data or {}
    allowed = {field.name for field in fields(DRMPCPlannerConfig)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    if "repo_root" in filtered:
        filtered["repo_root"] = str(Path(str(filtered["repo_root"])).expanduser())
    return DRMPCPlannerConfig(**filtered)


__all__ = ["DRMPCPlanner", "DRMPCPlannerConfig", "Observation", "build_dr_mpc_config"]
