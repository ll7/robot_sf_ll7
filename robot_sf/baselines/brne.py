"""BRNE baseline wrapper for the Social Navigation Benchmark.

This module provides a dependency-aware wrapper around the upstream BRNE
(Bayesian Recursive Nash Equilibrium) implementation.  The wrapper can be
imported even when the staged external clone is absent, but it will fail at
execution time with a clear diagnostic if the staged source is missing.

Bounded integration scope (issue #5318):

- Corridor-class scenarios only (``corridor_y_min/max`` bounds).
- Native unicycle ``(v, omega)`` output — no projection required.
- Fail-closed budget enforcement: zero motion on budget overrun.
- GPL-3.0 upstream: local-only staging, never vendored.

Upstream: ``MurpheyLab/brne`` @ ``633a5cd`` (IJRR 2024, GPL-3.0).
Core module: ``brne_nav/brne_py/brne_py/brne.py`` (pure-numpy/numba).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
import threading
import time
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
    from types import ModuleType

_LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRNE_IMPORT_LOCK = threading.RLock()
_BRNE_MODULE_NAME = "brne_upstream_planner"

# Upstream defaults (matching brne_nav/brne_py/brne_py/brne_nav.py).
_DEFAULT_NUM_SAMPLES = 196
_DEFAULT_PLAN_STEPS = 25
_DEFAULT_DT = 0.1
_DEFAULT_MAX_AGENTS = 8
_DEFAULT_KERNEL_A1 = 0.2
_DEFAULT_KERNEL_A2 = 0.2
_DEFAULT_COST_A1 = 4.0
_DEFAULT_COST_A2 = 1.0
_DEFAULT_COST_A3 = 80.0
_DEFAULT_PED_SAMPLE_SCALE = 0.1
_DEFAULT_CORRIDOR_Y_MIN = -0.65
_DEFAULT_CORRIDOR_Y_MAX = 0.65
_DEFAULT_STEP_BUDGET_S = 0.1


@dataclass
class BRNEPlannerConfig:
    """Configuration for the BRNE baseline wrapper."""

    stage_path: str = "third_party/external_repos/brne"
    num_samples: int = _DEFAULT_NUM_SAMPLES
    plan_steps: int = _DEFAULT_PLAN_STEPS
    dt: float = _DEFAULT_DT
    maximum_agents: int = _DEFAULT_MAX_AGENTS
    kernel_a1: float = _DEFAULT_KERNEL_A1
    kernel_a2: float = _DEFAULT_KERNEL_A2
    cost_a1: float = _DEFAULT_COST_A1
    cost_a2: float = _DEFAULT_COST_A2
    cost_a3: float = _DEFAULT_COST_A3
    ped_sample_scale: float = _DEFAULT_PED_SAMPLE_SCALE
    corridor_y_min: float = _DEFAULT_CORRIDOR_Y_MIN
    corridor_y_max: float = _DEFAULT_CORRIDOR_Y_MAX
    step_budget_s: float = _DEFAULT_STEP_BUDGET_S
    v_max: float = 2.0
    omega_max: float = 1.0
    safety_clamp: bool = True
    action_space: str = "unicycle"
    fallback_on_error: bool = False
    allow_testing_algorithms: bool = True
    include_in_paper: bool = False


def _load_brne_module(stage_path: Path) -> ModuleType:
    """Import the upstream brne.py core from the staged clone (GPL-3.0 local-only).

    Returns:
        The imported upstream BRNE core module.
    """
    core_rel = "brne_nav/brne_py/brne_py/brne.py"
    core_file = stage_path / core_rel
    if not core_file.is_file():
        raise FileNotFoundError(
            f"BRNE core algorithm not found at staged path: {core_file}. "
            "Run `uv run python scripts/tools/manage_external_repos.py stage brne`."
        )
    with _BRNE_IMPORT_LOCK:
        existing = sys.modules.get(_BRNE_MODULE_NAME)
        if existing is not None:
            return existing
        spec = importlib.util.spec_from_file_location(_BRNE_MODULE_NAME, core_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build import spec for {core_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_BRNE_MODULE_NAME] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module


class BRNEPlanner:
    """Baseline adapter for upstream BRNE (Bayesian Recursive Nash Equilibrium).

    Bounded integration tier (issue #5318): corridor-class scenarios only,
    fail-closed budget enforcement, native unicycle output.
    """

    def __init__(
        self, config: dict[str, Any] | BRNEPlannerConfig, *, seed: int | None = None
    ) -> None:
        """Initialize the BRNE wrapper with config and optional seed."""
        self.config = self._parse_config(config)
        self._seed = seed
        self._brne: ModuleType | None = None
        self._lmat: np.ndarray | None = None
        self._jit_warmup_done = False

    def _parse_config(self, config: dict[str, Any] | BRNEPlannerConfig) -> BRNEPlannerConfig:
        if isinstance(config, dict):
            return build_brne_config(config)
        if isinstance(config, BRNEPlannerConfig):
            return config
        raise TypeError(f"Invalid config type: {type(config)}")

    def _resolve_stage_path(self) -> Path:
        root = Path(self.config.stage_path).expanduser()
        if not root.is_absolute():
            root = _REPO_ROOT / root
        return root.resolve()

    def _ensure_brne_loaded(self) -> ModuleType:
        if self._brne is not None:
            return self._brne
        stage = self._resolve_stage_path()
        self._brne = _load_brne_module(stage)
        return self._brne

    def _ensure_cov(self, brne: ModuleType) -> np.ndarray:
        if self._lmat is not None:
            return self._lmat
        cfg = self.config
        tlist = np.arange(cfg.plan_steps) * cfg.dt
        train_ts = np.array([tlist[0]])
        train_noise = np.array([1e-04])
        lmat, _ = brne.get_Lmat_nb(train_ts, tlist, train_noise, cfg.kernel_a1, cfg.kernel_a2)
        self._lmat = lmat
        return lmat

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the BRNE wrapper state and optionally reseed the RNG."""
        if seed is not None:
            self._seed = seed
        self._lmat = None
        self._jit_warmup_done = False

    def configure(self, config: dict[str, Any] | BRNEPlannerConfig) -> None:
        """Update the BRNE wrapper configuration."""
        self.config = self._parse_config(config)
        self._lmat = None

    def step(self, obs: Observation | dict[str, Any]) -> dict[str, float]:
        """Compute a BRNE action for the current observation.

        Returns:
            A dict with ``v`` (forward speed) and ``omega`` (yaw rate).
        """
        if is_observation_mapping(obs):
            obs = observation_from_mapping(obs)

        try:
            return self._solve(obs)
        except FileNotFoundError:
            raise
        except Exception as exc:
            if self.config.fallback_on_error:
                _LOGGER.warning("BRNE solve failed, returning zero motion: %s", exc)
                return {"v": 0.0, "omega": 0.0}
            raise RuntimeError(f"BRNE solve failed: {exc}") from exc

    def _solve(self, obs: Observation) -> dict[str, float]:
        cfg = self.config
        brne = self._ensure_brne_loaded()
        lmat = self._ensure_cov(brne)

        robot = obs.robot
        r_pos = np.asarray(robot["position"], dtype=np.float64)
        r_vel = np.asarray(robot.get("velocity", [0.0, 0.0]), dtype=np.float64)
        r_goal = np.asarray(robot.get("goal", [r_pos[0] + 1.0, r_pos[1]]), dtype=np.float64)
        robot_pose = self._infer_robot_pose(r_pos, r_vel, r_goal)
        selected = self._select_agents(obs.agents, r_pos)
        num_peds = len(selected)
        num_agents = num_peds + 1
        num_samples = cfg.num_samples
        plan_steps = cfg.plan_steps
        dt = cfg.dt

        xtraj, ytraj, ulist = self._build_trajectories(
            brne,
            lmat,
            robot_pose,
            r_pos,
            r_vel,
            r_goal,
            obs.agents,
            selected,
            num_agents,
            num_samples,
            plan_steps,
            dt,
        )

        if not self._jit_warmup_done:
            self._brne_solve(brne, xtraj, ytraj, num_agents, plan_steps, num_samples)
            self._jit_warmup_done = True

        t0 = time.perf_counter()
        weights = self._brne_solve(brne, xtraj, ytraj, num_agents, plan_steps, num_samples)
        elapsed_s = time.perf_counter() - t0

        if weights is None or not np.all(np.isfinite(weights)):
            _LOGGER.debug(
                "BRNE returned out-of-bounds or non-finite weights; returning zero motion"
            )
            return {"v": 0.0, "omega": 0.0}

        if elapsed_s > cfg.step_budget_s:
            _LOGGER.debug(
                "BRNE solve exceeded budget (%.1f ms > %.1f ms); returning zero motion",
                elapsed_s * 1000.0,
                cfg.step_budget_s * 1000.0,
            )
            return {"v": 0.0, "omega": 0.0}

        robot_weights = weights[0]
        cmd = np.sum(ulist * robot_weights[:, np.newaxis, np.newaxis], axis=0)
        if not np.all(np.isfinite(cmd)):
            _LOGGER.debug("BRNE produced a non-finite control command; returning zero motion")
            return {"v": 0.0, "omega": 0.0}
        action = {"v": float(cmd[0, 0]), "omega": float(cmd[0, 1])}
        self._clamp_action(action)
        return action

    def _brne_solve(
        self,
        brne: ModuleType,
        xtraj: np.ndarray,
        ytraj: np.ndarray,
        num_agents: int,
        plan_steps: int,
        num_samples: int,
    ) -> np.ndarray | None:
        cfg = self.config
        return brne.brne_nav(
            xtraj,
            ytraj,
            num_agents,
            plan_steps,
            num_samples,
            cfg.cost_a1,
            cfg.cost_a2,
            cfg.cost_a3,
            cfg.ped_sample_scale,
            cfg.corridor_y_min,
            cfg.corridor_y_max,
        )

    @staticmethod
    def _infer_robot_pose(
        r_pos: np.ndarray,
        r_vel: np.ndarray,
        r_goal: np.ndarray,
    ) -> np.ndarray:
        if np.linalg.norm(r_vel) > 1e-6:
            theta = float(np.arctan2(r_vel[1], r_vel[0]))
        elif np.linalg.norm(r_goal - r_pos) > 1e-6:
            theta = float(np.arctan2(r_goal[1] - r_pos[1], r_goal[0] - r_pos[0]))
        else:
            theta = 0.0
        return np.array([r_pos[0], r_pos[1], theta])

    def _select_agents(
        self,
        agents: list[dict[str, Any]],
        r_pos: np.ndarray,
    ) -> list[tuple[float, int]]:
        cfg = self.config
        agent_dists: list[tuple[float, int]] = []
        for idx, agent in enumerate(agents):
            a_pos = np.asarray(agent["position"], dtype=np.float64)
            dist = float(np.linalg.norm(a_pos - r_pos))
            agent_dists.append((dist, idx))
        agent_dists.sort(key=lambda x: x[0])
        return agent_dists[: max(0, cfg.maximum_agents - 1)]

    def _build_trajectories(  # noqa: PLR0913
        self,
        brne: ModuleType,
        lmat: np.ndarray,
        robot_pose: np.ndarray,
        r_pos: np.ndarray,
        r_vel: np.ndarray,
        r_goal: np.ndarray,
        agents: list[dict[str, Any]],
        selected: list[tuple[float, int]],
        num_agents: int,
        num_samples: int,
        plan_steps: int,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        direction = r_goal - r_pos
        speed = (
            min(float(np.linalg.norm(r_vel)) or 0.4, cfg.v_max)
            if np.linalg.norm(direction) > 1e-6
            else 0.0
        )
        nominal_cmds = np.full((plan_steps, 2), [speed, 0.0])
        ulist = brne.get_ulist_essemble(nominal_cmds, 0.6, 1.0, num_samples)
        traj = brne.traj_sim_essemble(
            np.tile(robot_pose, reps=(num_samples, 1)).T,
            ulist,
            dt,
        )
        rx = traj[:, 0, :].T
        ry = traj[:, 1, :].T
        xtraj = np.zeros((num_agents * num_samples, plan_steps))
        ytraj = np.zeros((num_agents * num_samples, plan_steps))
        xtraj[:num_samples] = rx
        ytraj[:num_samples] = ry
        for ped_local_idx, (_, agent_idx) in enumerate(selected):
            agent = agents[agent_idx]
            a_pos = np.asarray(agent["position"], dtype=np.float64)
            a_vel = np.asarray(agent.get("velocity", [0.0, 0.0]), dtype=np.float64)
            speed_factor = float(np.linalg.norm(a_vel))
            xp = brne.mvn_sample_normal(num_samples, plan_steps, lmat)
            yp = brne.mvn_sample_normal(num_samples, plan_steps, lmat)
            xmean = a_pos[0] + np.arange(plan_steps) * dt * a_vel[0]
            ymean = a_pos[1] + np.arange(plan_steps) * dt * a_vel[1]
            scale = speed_factor + cfg.ped_sample_scale
            row_start = (ped_local_idx + 1) * num_samples
            xtraj[row_start : row_start + num_samples] = xp * scale + xmean
            ytraj[row_start : row_start + num_samples] = yp * scale + ymean
        return xtraj, ytraj, ulist

    def _clamp_action(self, action: dict[str, float]) -> None:
        if self.config.safety_clamp:
            action["v"] = max(0.0, min(float(action["v"]), self.config.v_max))
            action["omega"] = max(
                -self.config.omega_max, min(float(action["omega"]), self.config.omega_max)
            )

    def close(self) -> None:
        """Release BRNE wrapper resources."""
        self._brne = None
        self._lmat = None

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the BRNE planner."""
        cfg = asdict(self.config)

        config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        stage = self._resolve_stage_path()
        core_rel = "brne_nav/brne_py/brne_py/brne.py"
        status = "ok" if (stage / core_rel).is_file() else "missing_dependency"
        return {
            "algorithm": "brne",
            "config": cfg,
            "config_hash": config_hash,
            "status": status,
            "license": "GPL-3.0 (local-only staging; not vendored/redistributed)",
        }


def build_brne_config(data: dict[str, Any] | None) -> BRNEPlannerConfig:
    """Build a BRNE config from a loose mapping while preserving explicit provenance.

    Returns:
        Normalized BRNE planner configuration.
    """
    payload = data or {}
    allowed = {f.name for f in fields(BRNEPlannerConfig)}
    filtered = {k: v for k, v in payload.items() if k in allowed}
    if "stage_path" in filtered:
        filtered["stage_path"] = str(Path(str(filtered["stage_path"])).expanduser())
    return BRNEPlannerConfig(**filtered)


__all__ = ["BRNEPlanner", "BRNEPlannerConfig", "build_brne_config"]
