"""BRNE baseline wrapper for the Social Navigation Benchmark.

This module provides a dependency-aware wrapper around the upstream BRNE
(Bayesian Recursive Nash Equilibrium) implementation.  The wrapper can be
imported even when the staged external clone is absent, but it will fail at
execution time with a clear diagnostic if the staged source is missing.

Bounded integration scope (issue #5318):

- Corridor-class scenarios only (``corridor_y_min/max`` bounds).
- Native unicycle ``(v, omega)`` output — no projection required.
- Fail-closed budget enforcement: zero motion on budget overrun, with
  runtime failure provenance so diagnostic rows are excluded.
- GPL-3.0 upstream: local-only staging, never vendored.

Upstream: ``MurpheyLab/brne`` @ ``633a5cd`` (IJRR 2024, GPL-3.0).
Core module: ``brne_nav/brne_py/brne_py/brne.py`` (pure-numpy/numba).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
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
BRNE_PINNED_SHA = "633a5cdcb39ab27f18b596cb8cb1968644f82391"
BRNE_CORE_REL = "brne_nav/brne_py/brne_py/brne.py"

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
_UPSTREAM_BRNE_ACTIVATION_RADIUS_M = 3.5


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
    core_file = _validate_stage_provenance(stage_path)
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


def _validate_stage_provenance(stage_path: Path) -> Path:
    """Validate the staged clone and return its pinned BRNE core path.

    The external source is local-only, so the loader must verify both the git
    commit and the checked-out core file before importing it. A matching
    ``HEAD`` with a locally modified core is not an acceptable provenance
    boundary for diagnostic evidence.

    Raises:
        FileNotFoundError: If the staged clone or core file is unavailable.
        RuntimeError: If the clone is not the pinned, clean source checkout.

    Returns:
        Path: The validated upstream BRNE core module.
    """
    core_file = stage_path / BRNE_CORE_REL
    if not core_file.is_file():
        raise FileNotFoundError(
            f"BRNE core algorithm not found at staged path: {core_file}. "
            "Run `uv run python scripts/tools/manage_external_repos.py stage brne`."
        )
    if not (stage_path / ".git").exists():
        raise RuntimeError(f"BRNE staged path is not a git clone: {stage_path}")

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(stage_path), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    commit_result = _git("rev-parse", "--verify", "HEAD^{commit}")
    staged_commit = commit_result.stdout.strip() if commit_result.returncode == 0 else ""
    if staged_commit != BRNE_PINNED_SHA:
        raise RuntimeError(
            "BRNE staged source commit mismatch: "
            f"expected {BRNE_PINNED_SHA}, got {staged_commit or 'unavailable'}"
        )
    tracked_result = _git("ls-tree", "-r", "--name-only", "HEAD", "--", BRNE_CORE_REL)
    if tracked_result.returncode != 0 or tracked_result.stdout.strip() != BRNE_CORE_REL:
        raise RuntimeError(f"BRNE pinned commit does not track {BRNE_CORE_REL}")
    for diff_args in (
        ("diff", "--quiet", "--", BRNE_CORE_REL),
        ("diff", "--cached", "--quiet", "--", BRNE_CORE_REL),
    ):
        if _git(*diff_args).returncode != 0:
            raise RuntimeError(f"BRNE staged core is locally modified: {core_file}")
    return core_file


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
        self._upstream_rng_seeded = False
        self._last_effective_num_samples: int | None = None
        self._step_count = 0
        self._failure_count = 0
        self._failure_reasons: list[str] = []
        self._last_failure_reason: str | None = None
        self._last_step_status = "not_started"
        self._mechanism_trace_steps: list[dict[str, Any]] = []
        self._previous_action: dict[str, float] | None = None
        self._last_nominal_command: dict[str, Any] | None = None

    def _parse_config(self, config: dict[str, Any] | BRNEPlannerConfig) -> BRNEPlannerConfig:
        """Normalize ``config`` into a BRNEPlannerConfig, building it from a dict when needed.

        Returns:
            The normalized planner configuration.
        """
        if isinstance(config, dict):
            return build_brne_config(config)
        if isinstance(config, BRNEPlannerConfig):
            return config
        raise TypeError(f"Invalid config type: {type(config)}")

    def _resolve_stage_path(self) -> Path:
        """Resolve the configured BRNE staging directory, expanding ``~`` and repo-relative paths.

        Returns:
            The resolved absolute staging path.
        """
        root = Path(self.config.stage_path).expanduser()
        if not root.is_absolute():
            root = _REPO_ROOT / root
        return root.resolve()

    def _ensure_brne_loaded(self) -> ModuleType:
        """Lazily import and cache the staged BRNE core module on first use.

        Returns:
            The cached BRNE core module.
        """
        if self._brne is None:
            stage = self._resolve_stage_path()
            self._brne = _load_brne_module(stage)
        if not self._upstream_rng_seeded and self._seed is not None:
            upstream_rng = getattr(self._brne, "rng", None)
            if upstream_rng is not None:
                self._brne.rng = np.random.default_rng(self._seed)
            self._upstream_rng_seeded = True
        return self._brne

    def _ensure_cov(self, brne: ModuleType) -> np.ndarray:
        """Build and cache the kernel L-matrix used to sample trajectory noise.

        Returns:
            The cached kernel L-matrix.
        """
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
        self._upstream_rng_seeded = False
        self._last_effective_num_samples = None
        self._reset_runtime_diagnostics()

    def configure(self, config: dict[str, Any] | BRNEPlannerConfig) -> None:
        """Update the BRNE wrapper configuration."""
        self.config = self._parse_config(config)
        self._lmat = None
        self._upstream_rng_seeded = False
        self._last_effective_num_samples = None
        self._reset_runtime_diagnostics()

    def _reset_runtime_diagnostics(self) -> None:
        """Clear per-episode runtime diagnostics."""
        self._step_count = 0
        self._failure_count = 0
        self._failure_reasons = []
        self._last_failure_reason = None
        self._last_step_status = "not_started"
        self._mechanism_trace_steps = []
        self._previous_action = None
        self._last_nominal_command = None

    def _record_failure(self, reason: str) -> None:
        """Record a fail-closed step without hiding it behind zero motion."""
        normalized_reason = str(reason).strip() or "unknown_failure"
        self._step_count += 1
        self._failure_count += 1
        self._last_failure_reason = normalized_reason
        self._last_step_status = "failed"
        if normalized_reason not in self._failure_reasons:
            self._failure_reasons.append(normalized_reason)

    def _record_success(self) -> None:
        """Record one finite control step."""
        self._step_count += 1
        self._last_step_status = "ok"

    def _zero_action(self, reason: str) -> dict[str, float]:
        """Return the bounded stop action and retain its failure reason."""
        self._record_failure(reason)
        return {"v": 0.0, "omega": 0.0}

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to the principal ``[-pi, pi)`` interval.

        Returns:
            float: The wrapped angle in radians.
        """
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _summarize_control_candidates(
        ulist: np.ndarray,
        robot_weights: np.ndarray,
        *,
        aggregation_mode: str,
    ) -> dict[str, Any]:
        """Summarize candidate controls and weights without retaining raw tensors.

        Returns:
            A finite, bounded candidate/weight distribution summary.
        """
        candidates = np.asarray(ulist, dtype=float)
        weights = np.asarray(robot_weights, dtype=float)
        if aggregation_mode == "samples_first":
            candidates = np.transpose(candidates, (1, 0, 2))
        if (
            aggregation_mode not in {"plan_step_first", "samples_first"}
            or candidates.ndim != 3
            or candidates.shape[2] != 2
            or weights.ndim != 1
            or candidates.shape[1] != weights.size
            or not np.all(np.isfinite(candidates))
            or not np.all(np.isfinite(weights))
        ):
            return {
                "status": "unavailable",
                "reason": "invalid_candidate_or_weight_tensor",
            }

        def stats(values: np.ndarray) -> dict[str, float]:
            quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
            return {
                "min": float(quantiles[0]),
                "q25": float(quantiles[1]),
                "median": float(quantiles[2]),
                "mean": float(np.mean(values)),
                "q75": float(quantiles[3]),
                "max": float(quantiles[4]),
                "std": float(np.std(values)),
            }

        def step_summary(step_index: int) -> dict[str, Any]:
            controls = candidates[step_index]
            weighted_mean = np.mean(controls * weights[:, np.newaxis], axis=0)
            return {
                "candidate_controls": {
                    "v_m_s": stats(controls[:, 0]),
                    "omega_rad_s": stats(controls[:, 1]),
                },
                "weights": stats(weights),
                "weighted_mean": {
                    "v_m_s": float(weighted_mean[0]),
                    "omega_rad_s": float(weighted_mean[1]),
                },
            }

        first = step_summary(0)
        second = step_summary(1) if candidates.shape[0] > 1 else None
        first_to_second = None
        if second is not None:
            first_to_second = {
                "candidate_mean_delta_v_m_s": float(
                    second["candidate_controls"]["v_m_s"]["mean"]
                    - first["candidate_controls"]["v_m_s"]["mean"]
                ),
                "weighted_mean_delta_v_m_s": float(
                    second["weighted_mean"]["v_m_s"] - first["weighted_mean"]["v_m_s"]
                ),
                "candidate_mean_delta_omega_rad_s": float(
                    second["candidate_controls"]["omega_rad_s"]["mean"]
                    - first["candidate_controls"]["omega_rad_s"]["mean"]
                ),
                "weighted_mean_delta_omega_rad_s": float(
                    second["weighted_mean"]["omega_rad_s"] - first["weighted_mean"]["omega_rad_s"]
                ),
            }
        return {
            "status": "available",
            "schema_version": "brne-candidate-distribution.v1",
            "sample_count": int(candidates.shape[1]),
            "plan_step_count": int(candidates.shape[0]),
            "first": first,
            "second": second,
            "first_to_second": first_to_second,
        }

    def _record_mechanism_step(  # noqa: PLR0913
        self,
        *,
        robot_position: np.ndarray,
        robot_velocity: np.ndarray,
        declared_heading: float | None,
        goal_position: np.ndarray,
        selected_agents: list[tuple[float, int]],
        pedestrian_selection: dict[str, Any],
        agents: list[dict[str, Any]],
        action: dict[str, float],
        runtime_status: str,
        failure_reason: str | None,
        elapsed_s: float | None,
        ulist_shape: tuple[int, ...],
        weights_shape: tuple[int, ...] | None,
        aggregation_mode: str,
        effective_num_samples: int,
        pre_clamp_action: dict[str, float] | None,
        candidate_distribution: dict[str, Any] | None,
        nominal_command: dict[str, Any] | None,
    ) -> None:
        """Record compact BRNE mechanism telemetry for one planner observation.

        The trace is deliberately adapter-facing and bounded: it records pose,
        goal/frame geometry, the selected command, ensemble shapes, and runtime
        status without retaining raw trajectories, weights, or sampled control
        tensors. The environment trace remains the source of truth for applied
        motion and terminal outcome.
        """
        position = np.asarray(robot_position, dtype=float)
        velocity = np.asarray(robot_velocity, dtype=float)
        goal = np.asarray(goal_position, dtype=float)
        goal_delta = goal - position
        goal_distance = float(np.linalg.norm(goal_delta))
        goal_bearing = (
            float(np.arctan2(goal_delta[1], goal_delta[0])) if goal_distance > 1.0e-9 else None
        )
        velocity_norm = float(np.linalg.norm(velocity))
        velocity_heading = (
            float(np.arctan2(velocity[1], velocity[0])) if velocity_norm > 1.0e-9 else None
        )
        heading_reference = declared_heading
        if heading_reference is None:
            heading_reference = velocity_heading
        angular_difference = (
            self._wrap_angle(goal_bearing - heading_reference)
            if goal_bearing is not None and heading_reference is not None
            else None
        )
        previous_action = self._previous_action
        action_delta = (
            {
                "v": float(action["v"] - previous_action["v"]),
                "omega": float(action["omega"] - previous_action["omega"]),
                "changed": bool(
                    not np.isclose(action["v"], previous_action["v"])
                    or not np.isclose(action["omega"], previous_action["omega"])
                ),
            }
            if previous_action is not None
            else None
        )
        pre_clamp_payload = (
            {
                "v_m_s": float(pre_clamp_action["v"]),
                "omega_rad_s": float(pre_clamp_action["omega"]),
            }
            if pre_clamp_action is not None
            else None
        )
        action_clipping = (
            {
                "v_clipped": bool(
                    not np.isclose(action["v"], pre_clamp_action["v"], rtol=0.0, atol=1.0e-12)
                ),
                "omega_clipped": bool(
                    not np.isclose(
                        action["omega"], pre_clamp_action["omega"], rtol=0.0, atol=1.0e-12
                    )
                ),
            }
            if pre_clamp_action is not None
            else None
        )
        if action_clipping is not None:
            action_clipping["any_clipped"] = bool(
                action_clipping["v_clipped"] or action_clipping["omega_clipped"]
            )
        selected_pedestrians = []
        for distance, agent_idx in selected_agents:
            agent = agents[agent_idx]
            agent_position = np.asarray(agent.get("position", [np.nan, np.nan]), dtype=float)
            agent_velocity = np.asarray(agent.get("velocity", [np.nan, np.nan]), dtype=float)
            selected_pedestrians.append(
                {
                    "agent_index": int(agent_idx),
                    "distance_m": float(distance),
                    "position_world_m": [float(agent_position[0]), float(agent_position[1])],
                    "velocity_world_m_s": [float(agent_velocity[0]), float(agent_velocity[1])],
                }
            )
        self._mechanism_trace_steps.append(
            {
                "step": int(self._step_count - 1),
                "observation": {
                    "robot_position_world_m": [float(position[0]), float(position[1])],
                    "robot_velocity_world_m_s": [float(velocity[0]), float(velocity[1])],
                    "declared_heading_rad": declared_heading,
                    "velocity_derived_heading_rad": velocity_heading,
                    "goal_position_world_m": [float(goal[0]), float(goal[1])],
                    "goal_bearing_rad": goal_bearing,
                    "heading_goal_angular_difference_rad": angular_difference,
                    "heading_reference": (
                        "declared_heading"
                        if declared_heading is not None
                        else "velocity_derived_heading"
                        if velocity_heading is not None
                        else "unavailable"
                    ),
                },
                "selected_pedestrians": selected_pedestrians,
                "pedestrian_selection": dict(pedestrian_selection),
                "nominal_command": (
                    dict(nominal_command)
                    if nominal_command is not None
                    else {"status": "unavailable", "reason": "not_recorded"}
                ),
                "adapter_input_frame": "world",
                "pre_clamp_action": pre_clamp_payload,
                "selected_action": {
                    "v_m_s": float(action["v"]),
                    "omega_rad_s": float(action["omega"]),
                },
                "action_clipping": action_clipping,
                "action_delta": action_delta,
                "ensemble": {
                    "requested_num_samples": int(self.config.num_samples),
                    "effective_num_samples": int(effective_num_samples),
                    "control_ensemble_shape": list(ulist_shape),
                    "weight_shape": list(weights_shape) if weights_shape is not None else None,
                    "aggregation_mode": aggregation_mode,
                    "aggregation_formula": (
                        "mean_plan_step_first_over_samples"
                        if aggregation_mode == "plan_step_first"
                        else "mean_samples_first_over_samples"
                        if aggregation_mode == "samples_first"
                        else "not_applied"
                    ),
                    "candidate_distribution": candidate_distribution
                    if candidate_distribution is not None
                    else {"status": "unavailable", "reason": "not_recorded"},
                },
                "runtime": {
                    "status": runtime_status,
                    "failure_reason": failure_reason,
                    "failure_count": int(self._failure_count),
                    "failure_reasons": list(self._failure_reasons),
                    "elapsed_s": float(elapsed_s) if elapsed_s is not None else None,
                    "budget_s": float(self.config.step_budget_s),
                    "budget_exceeded": bool(
                        elapsed_s is not None and elapsed_s > self.config.step_budget_s
                    ),
                },
            }
        )
        self._previous_action = {
            "v": float(action["v"]),
            "omega": float(action["omega"]),
        }

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
            self._record_failure("missing_dependency")
            raise
        except Exception as exc:  # broad catch: solver fallback boundary
            self._record_failure("solver_exception")
            if self.config.fallback_on_error:
                _LOGGER.warning("BRNE solve failed, returning zero motion: %s", exc)
                return {"v": 0.0, "omega": 0.0}
            raise RuntimeError(f"BRNE solve failed: {exc}") from exc

    def _solve(self, obs: Observation) -> dict[str, float]:  # noqa: C901, PLR0915
        """Run the BRNE solver for one observation.

        Returns:
            The ``{v, omega}`` action, falling back to zero motion when the
            solver exceeds its budget or yields non-finite output.
        """
        cfg = self.config
        brne = self._ensure_brne_loaded()
        lmat = self._ensure_cov(brne)

        robot = obs.robot
        r_pos = np.asarray(robot["position"], dtype=np.float64)
        r_vel = np.asarray(robot.get("velocity", [0.0, 0.0]), dtype=np.float64)
        r_goal = np.asarray(robot.get("goal", [r_pos[0] + 1.0, r_pos[1]]), dtype=np.float64)
        heading_value = robot.get("heading")
        try:
            robot_heading = float(heading_value) if heading_value is not None else None
        except (TypeError, ValueError):
            robot_heading = None
        if robot_heading is not None and not np.isfinite(robot_heading):
            robot_heading = None
        robot_pose = self._infer_robot_pose(r_pos, r_vel, r_goal, r_heading=robot_heading)
        selected = self._select_agents(obs.agents, r_pos)
        pedestrian_selection = self._summarize_pedestrian_selection(
            obs.agents,
            r_pos,
            selected,
        )
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
        effective_num_samples = int(ulist.shape[1])
        self._last_effective_num_samples = effective_num_samples

        def finish(
            action: dict[str, float],
            *,
            failure_reason: str | None = None,
            elapsed_s: float | None = None,
            weights_shape: tuple[int, ...] | None = None,
            aggregation_mode: str = "not_applied",
            pre_clamp_action: dict[str, float] | None = None,
            candidate_distribution: dict[str, Any] | None = None,
        ) -> dict[str, float]:
            """Record runtime state and compact telemetry before returning an action.

            Returns:
                dict[str, float]: The action passed to the environment.
            """
            if failure_reason is None:
                self._record_success()
                runtime_status = "ok"
            else:
                self._record_failure(failure_reason)
                runtime_status = "failed"
            self._record_mechanism_step(
                robot_position=r_pos,
                robot_velocity=r_vel,
                declared_heading=robot_heading,
                goal_position=r_goal,
                selected_agents=selected,
                pedestrian_selection=pedestrian_selection,
                agents=obs.agents,
                action=action,
                runtime_status=runtime_status,
                failure_reason=failure_reason,
                elapsed_s=elapsed_s,
                ulist_shape=tuple(int(value) for value in ulist.shape),
                weights_shape=weights_shape,
                aggregation_mode=aggregation_mode,
                effective_num_samples=effective_num_samples,
                pre_clamp_action=pre_clamp_action,
                candidate_distribution=candidate_distribution,
                nominal_command=self._last_nominal_command,
            )
            return action

        if not self._jit_warmup_done:
            self._brne_solve(
                brne,
                xtraj,
                ytraj,
                num_agents,
                plan_steps,
                effective_num_samples,
            )
            self._jit_warmup_done = True

        t0 = time.perf_counter()
        weights = self._brne_solve(
            brne,
            xtraj,
            ytraj,
            num_agents,
            plan_steps,
            effective_num_samples,
        )
        elapsed_s = time.perf_counter() - t0

        if weights is None or not np.all(np.isfinite(weights)):
            _LOGGER.debug(
                "BRNE returned out-of-bounds or non-finite weights; returning zero motion"
            )
            return finish({"v": 0.0, "omega": 0.0}, failure_reason="nonfinite_weights")

        if elapsed_s > cfg.step_budget_s:
            _LOGGER.debug(
                "BRNE solve exceeded budget (%.1f ms > %.1f ms); returning zero motion",
                elapsed_s * 1000.0,
                cfg.step_budget_s * 1000.0,
            )
            return finish(
                {"v": 0.0, "omega": 0.0},
                failure_reason="step_budget_exceeded",
                elapsed_s=elapsed_s,
                weights_shape=tuple(int(value) for value in weights.shape),
            )

        robot_weights = weights[0]
        if ulist.ndim != 3 or robot_weights.ndim != 1:
            _LOGGER.debug("BRNE returned an invalid control ensemble shape; returning zero motion")
            return finish(
                {"v": 0.0, "omega": 0.0},
                failure_reason="invalid_control_ensemble_shape",
                elapsed_s=elapsed_s,
                weights_shape=tuple(int(value) for value in weights.shape),
            )
        aggregation_mode: str
        if ulist.shape[1] == robot_weights.size:
            # Match the pinned upstream ROS controller, which takes the sample
            # mean after mean-normalizing the BRNE weights.
            cmd = np.mean(ulist * robot_weights[np.newaxis, :, np.newaxis], axis=1)
            aggregation_mode = "plan_step_first"
        elif ulist.shape[0] == robot_weights.size:
            # Preserve compatibility with isolated adapters that expose samples first;
            # the pinned upstream helper uses the plan-step-first layout above.
            cmd = np.mean(ulist * robot_weights[:, np.newaxis, np.newaxis], axis=0)
            aggregation_mode = "samples_first"
        else:
            _LOGGER.debug(
                "BRNE weights and control ensemble shapes disagree; returning zero motion"
            )
            return finish(
                {"v": 0.0, "omega": 0.0},
                failure_reason="sample_weight_shape_mismatch",
                elapsed_s=elapsed_s,
                weights_shape=tuple(int(value) for value in weights.shape),
            )
        if not np.all(np.isfinite(cmd)):
            _LOGGER.debug("BRNE produced a non-finite control command; returning zero motion")
            return finish(
                {"v": 0.0, "omega": 0.0},
                failure_reason="nonfinite_control_command",
                elapsed_s=elapsed_s,
                weights_shape=tuple(int(value) for value in weights.shape),
                aggregation_mode=aggregation_mode,
            )
        pre_clamp_action = {"v": float(cmd[0, 0]), "omega": float(cmd[0, 1])}
        candidate_distribution = self._summarize_control_candidates(
            ulist,
            robot_weights,
            aggregation_mode=aggregation_mode,
        )
        action = dict(pre_clamp_action)
        self._clamp_action(action)
        return finish(
            action,
            elapsed_s=elapsed_s,
            weights_shape=tuple(int(value) for value in weights.shape),
            aggregation_mode=aggregation_mode,
            pre_clamp_action=pre_clamp_action,
            candidate_distribution=candidate_distribution,
        )

    def _brne_solve(
        self,
        brne: ModuleType,
        xtraj: np.ndarray,
        ytraj: np.ndarray,
        num_agents: int,
        plan_steps: int,
        num_samples: int,
    ) -> np.ndarray | None:
        """Invoke the upstream ``brne_nav`` kernel with the configured costs and corridor bounds.

        Returns:
            Per-agent sample weights from the upstream BRNE solve.
        """
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
        *,
        r_heading: float | None = None,
    ) -> np.ndarray:
        """Return the robot ``[x, y, theta]`` pose.

        A declared observation heading takes precedence. Velocity/goal bearing
        remains a compatibility fallback for callers using the older baseline
        mapping that did not carry heading explicitly.

        Returns:
            The ``[x, y, theta]`` robot pose.
        """
        if r_heading is not None and np.isfinite(r_heading):
            theta = float(r_heading)
        elif np.linalg.norm(r_vel) > 1e-6:
            theta = float(np.arctan2(r_vel[1], r_vel[0]))
        elif np.linalg.norm(r_goal - r_pos) > 1e-6:
            theta = float(np.arctan2(r_goal[1] - r_pos[1], r_goal[0] - r_pos[0]))
        else:
            theta = 0.0
        return np.array([r_pos[0], r_pos[1], theta])

    @staticmethod
    def _normalize_control_ensemble(ulist: Any, *, plan_steps: int) -> np.ndarray:
        """Normalize upstream control ensembles to ``(plan_steps, samples, 2)``.

        The pinned upstream helper returns plan-step-first arrays, while older
        isolated adapters used samples-first arrays. Accept both only when the
        plan-step axis is explicit and reject all malformed layouts.

        Returns:
            np.ndarray: Plan-step-first finite control ensemble.

        Raises:
            ValueError: If the ensemble shape or values are invalid.
        """
        array = np.asarray(ulist, dtype=float)
        if array.ndim != 3 or array.shape[2] != 2:
            raise ValueError("invalid_control_ensemble_shape")
        if array.shape[0] == plan_steps:
            normalized = array
        elif array.shape[1] == plan_steps:
            normalized = np.transpose(array, (1, 0, 2))
        else:
            raise ValueError("invalid_control_ensemble_plan_axis")
        if normalized.shape[0] != plan_steps or normalized.shape[1] < 1:
            raise ValueError("invalid_control_ensemble_sample_axis")
        if not np.all(np.isfinite(normalized)):
            raise ValueError("nonfinite_control_ensemble")
        return normalized

    def _select_agents(
        self,
        agents: list[dict[str, Any]],
        r_pos: np.ndarray,
    ) -> list[tuple[float, int]]:
        """Select the closest agents to the robot, capped at ``maximum_agents - 1`` pedestrians.

        Returns:
            ``(distance, index)`` tuples of the closest agents, sorted by distance.
        """
        cfg = self.config
        agent_dists: list[tuple[float, int]] = []
        for idx, agent in enumerate(agents):
            a_pos = np.asarray(agent["position"], dtype=np.float64)
            dist = float(np.linalg.norm(a_pos - r_pos))
            agent_dists.append((dist, idx))
        agent_dists.sort(key=lambda x: x[0])
        return agent_dists[: max(0, cfg.maximum_agents - 1)]

    @staticmethod
    def _summarize_pedestrian_selection(
        agents: list[dict[str, Any]],
        r_pos: np.ndarray,
        selected: list[tuple[float, int]],
    ) -> dict[str, Any]:
        """Record adapter selection counts against the pinned upstream gate.

        The adapter intentionally keeps its current nearest-agent selection semantics. This
        summary exposes how those semantics differ from the upstream controller's 3.5 m
        activation threshold without applying that threshold to runtime behavior.

        Returns:
            Compact counts and provenance for the observed and selected pedestrians.
        """
        distances = [
            float(np.linalg.norm(np.asarray(agent["position"], dtype=np.float64) - r_pos))
            for agent in agents
        ]
        return {
            "observed_count": len(agents),
            "within_upstream_activation_radius_count": sum(
                distance < _UPSTREAM_BRNE_ACTIVATION_RADIUS_M for distance in distances
            ),
            "passed_to_brne_count": len(selected),
            "upstream_activation_radius_m": _UPSTREAM_BRNE_ACTIVATION_RADIUS_M,
            "activation_gate_applied": False,
            "selection_mode": "nearest_up_to_maximum_agents",
        }

    def _build_nominal_commands(
        self,
        r_pos: np.ndarray,
        r_vel: np.ndarray,
        r_goal: np.ndarray,
        plan_steps: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Build the adapter's constant nominal command and its trace metadata.

        Returns:
            The plan-step-first nominal commands and a compact trace payload.
        """
        cfg = self.config
        direction = r_goal - r_pos
        speed = (
            min(float(np.linalg.norm(r_vel)) or 0.4, cfg.v_max)
            if np.linalg.norm(direction) > 1e-6
            else 0.0
        )
        nominal_cmds = np.full((plan_steps, 2), [speed, 0.0])
        return nominal_cmds, {
            "v_m_s": float(speed),
            "omega_rad_s": 0.0,
            "construction_mode": "straight_constant",
        }

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
        """Sample robot and pedestrian trajectory ensembles for BRNE.

        Returns:
            The stacked ``xtraj``/``ytraj`` arrays and the control ensemble ``ulist``.
        """
        cfg = self.config
        nominal_cmds, nominal_command = self._build_nominal_commands(
            r_pos,
            r_vel,
            r_goal,
            plan_steps,
        )
        self._last_nominal_command = nominal_command
        ulist = self._normalize_control_ensemble(
            brne.get_ulist_essemble(nominal_cmds, 0.6, 1.0, num_samples),
            plan_steps=plan_steps,
        )
        effective_num_samples = int(ulist.shape[1])
        traj = brne.traj_sim_essemble(
            np.tile(robot_pose, reps=(effective_num_samples, 1)).T,
            ulist,
            dt,
        )
        rx = traj[:, 0, :].T
        ry = traj[:, 1, :].T
        xtraj = np.zeros((num_agents * effective_num_samples, plan_steps))
        ytraj = np.zeros((num_agents * effective_num_samples, plan_steps))
        xtraj[:effective_num_samples] = rx
        ytraj[:effective_num_samples] = ry
        for ped_local_idx, (_, agent_idx) in enumerate(selected):
            agent = agents[agent_idx]
            a_pos = np.asarray(agent["position"], dtype=np.float64)
            a_vel = np.asarray(agent.get("velocity", [0.0, 0.0]), dtype=np.float64)
            speed_factor = float(np.linalg.norm(a_vel))
            xp = brne.mvn_sample_normal(effective_num_samples, plan_steps, lmat)
            yp = brne.mvn_sample_normal(effective_num_samples, plan_steps, lmat)
            xmean = a_pos[0] + np.arange(plan_steps) * dt * a_vel[0]
            ymean = a_pos[1] + np.arange(plan_steps) * dt * a_vel[1]
            scale = speed_factor + cfg.ped_sample_scale
            row_start = (ped_local_idx + 1) * effective_num_samples
            row_end = row_start + effective_num_samples
            xtraj[row_start:row_end] = xp * scale + xmean
            ytraj[row_start:row_end] = yp * scale + ymean
        return xtraj, ytraj, ulist

    def _clamp_action(self, action: dict[str, float]) -> None:
        """Clamp the action's speed and yaw rate to configured limits in place when enabled."""
        if self.config.safety_clamp:
            action["v"] = max(0.0, min(float(action["v"]), self.config.v_max))
            action["omega"] = max(
                -self.config.omega_max, min(float(action["omega"]), self.config.omega_max)
            )

    def close(self) -> None:
        """Release BRNE wrapper resources."""
        self._brne = None
        self._lmat = None
        self._upstream_rng_seeded = False
        self._last_effective_num_samples = None

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the BRNE planner."""
        cfg = asdict(self.config)

        config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        stage = self._resolve_stage_path()
        source_commit: str | None = None
        source_integrity = "missing"
        try:
            _validate_stage_provenance(stage)
        except FileNotFoundError:
            status = "missing_dependency"
        except RuntimeError:
            status = "invalid_provenance"
            source_integrity = "invalid"
            commit_result = subprocess.run(
                ["git", "-C", str(stage), "rev-parse", "--verify", "HEAD^{commit}"],
                check=False,
                capture_output=True,
                text=True,
            )
            source_commit = commit_result.stdout.strip() or None
        else:
            status = "ok"
            source_commit = BRNE_PINNED_SHA
            source_integrity = "clean_pinned_worktree"
        runtime_status = (
            "failed" if self._failure_count > 0 else "ok" if self._step_count > 0 else "not_started"
        )
        return {
            "algorithm": "brne",
            "config": cfg,
            "config_hash": config_hash,
            "seed": self._seed,
            "status": status,
            "source_commit": source_commit,
            "source_integrity": source_integrity,
            "source_pin": BRNE_PINNED_SHA,
            "effective_num_samples": self._last_effective_num_samples,
            "runtime_status": runtime_status,
            "step_count": self._step_count,
            "failure_count": self._failure_count,
            "failure_reasons": list(self._failure_reasons),
            "last_failure_reason": self._last_failure_reason,
            "last_step_status": self._last_step_status,
            "mechanism_trace": {
                "schema_version": "brne-mechanism-trace.v1",
                "status": "available" if self._mechanism_trace_steps else "unavailable",
                "claim_boundary": (
                    "Adapter-facing native BRNE telemetry for mechanism diagnosis only; "
                    "not benchmark, safety, realism, or paper evidence."
                ),
                "steps": list(self._mechanism_trace_steps),
            },
            "sample_count_note": (
                "The pinned upstream grid helper may return fewer samples than the requested "
                "num_samples; all trajectory and weight tensors use that effective count."
            ),
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
