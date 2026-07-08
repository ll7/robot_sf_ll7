"""SICNav baseline wrapper for the Social Navigation Benchmark.

This module provides a dependency-aware wrapper around an external SICNav implementation.
The wrapper can be imported even when the external package is not installed, but
it will fail at execution time with a clear diagnostic if the runtime dependency is missing.

Open-source dependency path (the one exercised at smoke time):

    The wrapper drives ``sicnav.policy.campc.CollisionAvoidMPC`` -- the CasADi + IPOPT
    MPC (SICNav-p / SICNav-np variant depending on ``priviledged_info``).  This is the
    *redistributable* path: it needs only CasADi (which bundles IPOPT) and python-RVO2.
    ``sicnav_diffusion.policy.sicnav_acados.SICNavAcados`` (the class the upstream metadata
    names) requires Acados + HSL, which are intentionally out of the redistributable
    footprint; that path is not driven here.
"""

from __future__ import annotations

import configparser
import hashlib
import importlib
import json
import math
import random
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.baselines.interface import (
    Observation,
    is_observation_mapping,
    observation_from_mapping,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SICNAV_IMPORT_LOCK = threading.RLock()
# Relative to ``repo_root``; these ship with the pinned upstream checkout.
_DEFAULT_POLICY_CONFIG = "sicnav/configs/policy.config"
_DEFAULT_ENV_CONFIG = "sicnav/configs/env.config"


class _CampcEnvAdapter:
    """Minimal upstream env surface touched by ``CollisionAvoidMPC.set_env``/``predict``.

    The real ``crowd_sim_plus`` env is a full gymnasium simulator; the wrapper only needs
    the few attributes the policy reads during ``set_env`` (``time_step``/``time_limit``/
    ``config``/``set_human_observability``) and ``predict`` (``global_time``/``sim_env``).
    """

    def __init__(self, env_config: configparser.RawConfigParser) -> None:
        self.config = env_config
        self.time_step = env_config.getfloat("env", "time_step")
        self.time_limit = env_config.getfloat("env", "time_limit")
        self.global_time = 0.0
        # A non-hallway value keeps the point-stabilization reference trajectory stable.
        self.sim_env = "circle_crossing"
        self.human_observability = False

    def set_human_observability(self, human_observability: bool) -> None:
        """Mirror the upstream env hook consumed by ``set_env``."""
        self.human_observability = bool(human_observability)


class _CampcPolicyRunner:
    """Adapter that drives the upstream CasADi/IPOPT ``CollisionAvoidMPC`` policy.

    Maps Robot SF :class:`Observation` payloads into the upstream ``FullState`` contract,
    calls ``CollisionAvoidMPC.predict`` (which returns an ``ActionRot(v, r)``), and projects
    the result back to a Robot SF unicycle ``{"v", "omega"}`` action. The runner holds a
    single policy instance across steps so the upstream MPC warmstart persists.
    """

    def __init__(
        self,
        policy: Any,
        env: _CampcEnvAdapter,
        full_state_cls: Any,
        joint_state_cls: Any,
        *,
        omega_max: float,
    ) -> None:
        self._policy = policy
        self._env = env
        self._full_state_cls = full_state_cls
        self._joint_state_cls = joint_state_cls
        self._omega_max = omega_max
        self._time_step = env.time_step

    def seed(self, seed: int | None) -> None:
        """Best-effort seed hook for the upstream MPC policy."""
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)

    def _to_full_state(self, agent: dict[str, Any], *, v_pref: float) -> Any:
        """Convert a Robot SF agent dict into an upstream ``FullState``.

        Returns:
            Any: An upstream ``FullState`` describing one robot or pedestrian.
        """
        pos = agent.get("position") or (0.0, 0.0)
        vel = agent.get("velocity") or (0.0, 0.0)
        px, py = float(pos[0]), float(pos[1])
        vx, vy = float(vel[0]), float(vel[1])
        radius = float(agent.get("radius") if agent.get("radius") is not None else 0.3)
        goal = agent.get("goal")
        if goal is not None:
            gx, gy = float(goal[0]), float(goal[1])
        else:
            # Constant-velocity goal projection mirrors the non-priviledged predict() path.
            gx, gy = px + vx * 2.0, py + vy * 2.0
        theta = float(np.arctan2(vy, vx)) if (vx or vy) else 0.0
        return self._full_state_cls(
            px=px,
            py=py,
            vx=vx,
            vy=vy,
            radius=radius,
            gx=gx,
            gy=gy,
            v_pref=float(v_pref),
            theta=theta,
        )

    def select_action(self, obs: Observation | dict[str, Any]) -> dict[str, float]:
        """Compute a unicycle ``{"v", "omega"}`` action for the current observation.

        Returns:
            dict[str, float]: Clamped unicycle action with ``v`` (m/s) and ``omega`` (rad/s).
        """
        if is_observation_mapping(obs):
            obs = observation_from_mapping(obs)
        robot_state = self._to_full_state(obs.robot, v_pref=float(obs.robot.get("v_pref", 1.0)))
        human_states = [self._to_full_state(a, v_pref=0.8) for a in obs.agents]
        joint_state = self._joint_state_cls(
            self_state=robot_state,
            human_states=human_states,
            static_obs=[],
        )
        action = self._policy.predict(joint_state)
        # Upstream ActionRot.r is the angular-velocity control integrated over one step
        # (``mpc_omega * time_step``); convert back to a rad/s unicycle omega.
        velocity = float(action.v)
        raw_omega = float(action.r)
        if not math.isfinite(velocity) or not math.isfinite(raw_omega):
            raise RuntimeError("SICNav CAMPc returned non-finite action")
        omega = raw_omega / self._time_step if self._time_step > 0.0 else 0.0
        omega = max(-self._omega_max, min(omega, self._omega_max))
        self._env.global_time += self._time_step
        return {"v": velocity, "omega": omega}


@dataclass
class SICNavPlannerConfig:
    """Configuration for the SICNav baseline wrapper."""

    checkpoint_path: str | None = None
    repo_root: str = "third_party/external_repos/sicnav"
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
    # Open-source CasADi/IPOPT campc driver overrides (default to upstream config files
    # under ``repo_root``). When ``None`` the driver loads
    # ``sicnav/configs/policy.config`` and ``sicnav/configs/env.config`` from the staged repo.
    policy_config_path: str | None = None
    env_config_path: str | None = None
    use_upstream_campc: bool = True


class SICNavPlanner:
    """Baseline adapter for external SICNav/MPC implementations."""

    def __init__(self, config: dict[str, Any] | SICNavPlannerConfig, *, seed: int | None = None):
        """Initialize the SICNav wrapper with config and optional seed."""
        self.config = self._parse_config(config)
        self._policy: Any | None = None
        self._module: Any | None = None
        self._seed = seed

    def _parse_config(self, config: dict[str, Any] | SICNavPlannerConfig) -> SICNavPlannerConfig:
        """Normalize accepted config payloads into the dataclass contract.

        Returns:
            SICNavPlannerConfig: Planner configuration used by the wrapper.
        """
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
            or name == "crowd_sim_plus"
            or name.startswith("crowd_sim_plus.")
        )

    @staticmethod
    def _has_supported_policy_constructor(module: Any) -> bool:
        """Return whether the imported module exposes a supported policy factory."""
        return callable(getattr(module, "SICNavPolicy", None)) or callable(
            getattr(module, "load_policy", None)
        )

    @staticmethod
    def _resolve_repo_root(repo_root: str) -> Path:
        """Resolve a configured repo root against the Robot SF checkout root.

        Returns:
            Path: Absolute path to the configured upstream checkout.
        """
        root = Path(repo_root).expanduser()
        if not root.is_absolute():
            root = _REPO_ROOT / root
        return root.resolve()

    @contextmanager
    def _upstream_import_context(self) -> Iterator[set[str]]:
        """Temporarily add the vendored SICNav repo root to `sys.path`."""
        with _SICNAV_IMPORT_LOCK:
            repo_root = self._resolve_repo_root(self.config.repo_root)
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
        """Import and cache the configured SICNav upstream module.

        The import context isolates transient upstream modules so Robot SF can
        probe different dependency states without leaking incompatible module
        objects into later imports.

        Returns:
            Any: Imported upstream module exposing a supported policy API.
        """
        with _SICNAV_IMPORT_LOCK:
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

    def _apply_seed(self, policy: Any) -> None:
        """Apply the configured seed to common upstream policy RNG hooks."""
        if self._seed is None:
            return
        random.seed(self._seed)
        np.random.seed(self._seed)
        for method_name in ("seed", "set_seed"):
            seed_method = getattr(policy, method_name, None)
            if callable(seed_method):
                seed_method(self._seed)
                return

    def _build_campc_runner(self) -> _CampcPolicyRunner | None:
        """Build a runner over the upstream CasADi/IPOPT ``CollisionAvoidMPC`` policy.

        Returns ``None`` (so the caller falls back to the generic constructor path) when
        the staged upstream repo is absent, the campc module cannot be imported, or the
        open-source dependency stack (CasADi/IPOPT/python-RVO2) is not installed. The
        upstream imports and the configure/set_env wiring all run inside the isolated
        import context so the upstream tree is resolved against the pinned checkout.

        Returns:
            _CampcPolicyRunner | None: A runner over the real upstream policy, or ``None``
            when the open-source campc path is unavailable on this machine.
        """
        # Probe the staged upstream checkout through the shared import hook so the
        # missing-dependency contract (and the unit-test monkeypatch of that hook) is honored.
        try:
            self._import_sicnav_module()
        except (ImportError, RuntimeError):
            return None
        repo_root = self._resolve_repo_root(self.config.repo_root)
        if not repo_root.is_dir():
            return None
        if self.config.policy_config_path:
            policy_config_path = Path(self.config.policy_config_path)
            if not policy_config_path.is_file():
                raise FileNotFoundError(f"Policy config file not found: {policy_config_path}")
        else:
            policy_config_path = repo_root / _DEFAULT_POLICY_CONFIG
        if self.config.env_config_path:
            env_config_path = Path(self.config.env_config_path)
            if not env_config_path.is_file():
                raise FileNotFoundError(f"Env config file not found: {env_config_path}")
        else:
            env_config_path = repo_root / _DEFAULT_ENV_CONFIG
        if not policy_config_path.is_file() or not env_config_path.is_file():
            return None

        policy_config = configparser.RawConfigParser()
        policy_config.read(policy_config_path)
        env_config = configparser.RawConfigParser()
        env_config.read(env_config_path)

        with self._upstream_import_context() as preserved_modules:
            try:
                campc_module = importlib.import_module("sicnav.policy.campc")
                state_module = importlib.import_module("crowd_sim_plus.envs.utils.state_plus")
                collision_avoid_mpc_cls = campc_module.CollisionAvoidMPC
                full_state_cls = state_module.FullState
                joint_state_cls = state_module.FullyObservableJointState
            except (ImportError, AttributeError):  # campc path unavailable on this machine
                return None
            try:
                policy = collision_avoid_mpc_cls()
                policy.configure(policy_config)
                env = _CampcEnvAdapter(env_config)
                policy.set_env(env)
                policy.time_step = env.time_step
            except Exception:  # noqa: BLE001 - upstream init failure means path unavailable
                return None
            # Keep the upstream package objects alive after the import context exits so the
            # live policy instance keeps resolving against the pinned checkout on later steps.
            preserved_modules.update(name for name in sys.modules if self._is_sicnav_module(name))

        return _CampcPolicyRunner(
            policy=policy,
            env=env,
            full_state_cls=full_state_cls,
            joint_state_cls=joint_state_cls,
            omega_max=self.config.omega_max,
        )

    def _build_policy(self) -> Any:
        """Construct the upstream SICNav policy through a supported factory hook.

        Returns:
            Any: Policy object exposing ``select_action``.
        """
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        # Preferred: drive the real open-source CasADi/IPOPT CollisionAvoidMPC policy
        # against the staged upstream checkout. Returns None when the upstream campc
        # path is unavailable (missing repo / dependencies), falling back to the legacy
        # generic constructor path below.
        if self.config.use_upstream_campc:
            runner = self._build_campc_runner()
            if runner is not None:
                self._apply_seed(runner)
                return runner

        try:
            module = self._import_sicnav_module()
        except ImportError as exc:
            raise RuntimeError(str(exc)) from exc

        if callable(getattr(module, "SICNavPolicy", None)):
            policy_cls = module.SICNavPolicy
            policy = policy_cls(
                checkpoint_path=self.config.checkpoint_path,
                solver=self.config.solver,
                device=self.config.device,
            )
            self._apply_seed(policy)
            return policy

        if callable(getattr(module, "load_policy", None)):
            policy = module.load_policy(
                self.config.checkpoint_path,
                device=self.config.device,
                solver=self.config.solver,
            )
            self._apply_seed(policy)
            return policy

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
        if is_observation_mapping(obs):
            obs = observation_from_mapping(obs)

        if self._policy is None:
            self._policy = self._build_policy()

        action = self._policy.select_action(obs)
        if not isinstance(action, dict):
            raise ValueError("SICNav policy returned an invalid action payload")

        self._clamp_action(action)
        return action

    def _clamp_action(self, action: dict[str, float]) -> None:
        """Clamp mutable action payloads to the configured velocity limits.

        This safety clamp keeps external policies within the benchmark action
        envelope without changing the action representation selected by the
        upstream implementation.
        """
        if self.config.safety_clamp:
            if "vx" in action and "vy" in action:
                speed = float(np.hypot(action["vx"], action["vy"]))
                if not math.isfinite(speed):
                    raise RuntimeError("SICNav policy returned non-finite velocity action")
                if speed > self.config.v_max and speed > 1e-9:
                    scale = self.config.v_max / speed
                    action["vx"] *= scale
                    action["vy"] *= scale
            if "v" in action:
                velocity = float(action["v"])
                if not math.isfinite(velocity):
                    raise RuntimeError("SICNav policy returned non-finite velocity action")
                action["v"] = max(0.0, min(velocity, self.config.v_max))
            if "omega" in action:
                omega = float(action["omega"])
                if not math.isfinite(omega):
                    raise RuntimeError("SICNav policy returned non-finite angular action")
                action["omega"] = max(
                    -self.config.omega_max,
                    min(omega, self.config.omega_max),
                )

    def close(self) -> None:
        """Release any SICNav wrapper resources."""
        self._policy = None
        self._module = None

    def _has_campc_capability(self) -> bool:
        """Return whether the staged upstream exposes the CasADi/IPOPT CollisionAvoidMPC path.

        Probes the pinned checkout for ``sicnav.policy.campc`` (and its transitive
        ``crowd_sim_plus``/gym import chain) inside the isolated import context.
        """
        with self._upstream_import_context():
            try:
                importlib.import_module("sicnav.policy.campc")
            except Exception:  # noqa: BLE001 - report any import failure as unavailable
                return False
            return True

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the SICNav planner."""
        cfg = asdict(self.config)
        config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        status = "ok"

        try:
            module = self._import_sicnav_module()
            if not (
                self._has_supported_policy_constructor(module)
                or (self.config.use_upstream_campc and self._has_campc_capability())
            ):
                status = "missing_dependency"
        except (ImportError, RuntimeError):
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
