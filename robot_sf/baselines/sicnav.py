"""SICNav baseline wrapper for the Social Navigation Benchmark.

This module provides a dependency-aware wrapper around an external SICNav implementation.
The wrapper can be imported even when the external package is not installed, but
it will fail at execution time with a clear diagnostic if the runtime dependency is missing.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class SICNavPlannerConfig:
    """Configuration for the SICNav baseline wrapper."""

    checkpoint_path: str | None = None
    repo_root: str = "third_party/external_mpc_repos/sicnav"
    upstream_repo_url: str = "https://github.com/sepsamavi/safe-interactive-crowdnav"
    solver: str = "ipopt"
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
    """Observation payload for the SICNav baseline wrapper."""

    dt: float
    robot: dict[str, Any]
    agents: list[dict[str, Any]]
    obstacles: list[Any]


class SICNavPlanner:
    """Baseline adapter for external SICNav/MPC implementations."""

    def __init__(self, config: dict[str, Any] | SICNavPlannerConfig, *, seed: int | None = None):
        """Initialize the SICNav wrapper with config and optional seed."""
        self.config = self._parse_config(config)
        self._policy: Any | None = None
        self._module: Any | None = None
        self._seed = seed

    def _parse_config(self, config: dict[str, Any] | SICNavPlannerConfig) -> SICNavPlannerConfig:
        if isinstance(config, dict):
            return build_sicnav_config(config)
        if isinstance(config, SICNavPlannerConfig):
            return config
        raise TypeError(f"Invalid config type: {type(config)}")

    @staticmethod
    def _is_sicnav_module(name: str) -> bool:
        """Return True for SICNav-related modules that need import-state cleanup."""
        return (
            name == "sicnav"
            or name.startswith("sicnav.")
            or name == "sicnav_diffusion"
            or name.startswith("sicnav_diffusion.")
        )

    @contextmanager
    def _upstream_import_context(self) -> Iterator[set[str]]:
        """Temporarily add the vendored SICNav repo root to `sys.path`."""
        repo_root = Path(self.config.repo_root).expanduser()
        repo_str = str(repo_root)
        original_path = list(sys.path)
        original_modules = {
            name: module for name, module in sys.modules.items() if self._is_sicnav_module(name)
        }
        preserved_modules: set[str] = set()
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            for name in list(sys.modules):
                if self._is_sicnav_module(name):
                    sys.modules.pop(name, None)
            yield preserved_modules
        finally:
            for name in list(sys.modules):
                if self._is_sicnav_module(name) and name not in preserved_modules:
                    sys.modules.pop(name, None)
            for name, module in original_modules.items():
                if name not in preserved_modules:
                    sys.modules[name] = module
            sys.path[:] = original_path

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the SICNav wrapper state and optionally reseed the RNG."""
        if seed is not None:
            self._seed = seed
        self._policy = None
        self._module = None

    def configure(self, config: dict[str, Any] | SICNavPlannerConfig) -> None:
        """Update the SICNav wrapper configuration."""
        self.config = self._parse_config(config)
        self._policy = None
        self._module = None

    def _import_sicnav_module(self) -> Any:
        if self._module is not None:
            return self._module

        with self._upstream_import_context() as preserved_modules:
            for module_name in ("sicnav_diffusion", "sicnav"):
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    continue
                self._module = module
                preserved_modules.update(
                    name
                    for name in sys.modules
                    if self._is_sicnav_module(name)
                    and (name == module_name or name.startswith(f"{module_name}."))
                )
                return module

        raise ImportError(
            "SICNav dependency is missing. Install `sicnav_diffusion` or `sicnav`, "
            "or point `repo_root` at a checked-out upstream repo to use the "
            "`sicnav` baseline."
        )

    def _build_policy(self) -> Any:
        try:
            module = self._import_sicnav_module()
        except ImportError as exc:
            raise RuntimeError(str(exc)) from exc

        if hasattr(module, "SICNavPolicy"):
            policy_cls = module.SICNavPolicy
            return policy_cls(
                checkpoint_path=self.config.checkpoint_path,
                solver=self.config.solver,
                device=self.config.device,
            )

        if hasattr(module, "load_policy"):
            return module.load_policy(
                self.config.checkpoint_path,
                device=self.config.device,
                solver=self.config.solver,
            )

        raise RuntimeError(
            "Imported SICNav package does not expose a supported policy constructor. "
            "Expected `SICNavPolicy` or `load_policy`."
        )

    def step(self, obs: Observation | dict[str, Any]) -> dict[str, float]:
        """Compute a SICNav action for the current observation.

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
            raise ValueError("SICNav policy returned an invalid action payload")

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
        """Release any SICNav wrapper resources."""
        self._policy = None
        self._module = None

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the SICNav planner."""
        cfg = asdict(self.config)
        config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        status = "ok"

        try:
            self._import_sicnav_module()
        except ImportError:
            status = "missing_dependency"

        return {
            "algorithm": "sicnav",
            "config": cfg,
            "config_hash": config_hash,
            "status": status,
        }


def build_sicnav_config(data: dict[str, Any] | None) -> SICNavPlannerConfig:
    """Build a SICNav config from a loose mapping while preserving explicit provenance.

    Returns:
        SICNavPlannerConfig: Normalized config with repo-root provenance preserved.
    """
    payload = data or {}
    allowed = {field.name for field in fields(SICNavPlannerConfig)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    if "repo_root" in filtered:
        filtered["repo_root"] = str(Path(str(filtered["repo_root"])).expanduser())
    return SICNavPlannerConfig(**filtered)


__all__ = [
    "Observation",
    "SICNavPlanner",
    "SICNavPlannerConfig",
    "build_sicnav_config",
]
