"""Shared base/config classes for the SocNav planner family.

Extracted from the `robot_sf.planner.socnav` facade so the shared configuration,
reference/sampling adapters, and policy wrappers live in a focused module. The
facade re-exports every public name from here; object identity is preserved for
all five re-exported classes.
"""

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, pi
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pysocialforce.config import DEFAULT_OBSTACLE_FORCE_LAW, resolve_obstacle_force_law

from robot_sf.common.math_utils import wrap_angle_pi_closed
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin

_SOCNAV_ROOT_ENV = "ROBOT_SF_SOCNAV_ROOT"
_SOCNAV_ALLOW_UNTRUSTED_ENV = "ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT"
_SOCNAV_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "socnavbench"
_SOCNAV_REQUIRED_MODULES = (
    "control_pipelines.control_pipeline_v0",
    "objectives.goal_distance",
    "params.central_params",
    "planners.sampling_planner",
)
_SOCNAV_ASSET_SETUP_DOC = "docs/socnav_assets_setup.md"
_SOCNAV_ASSET_SETUP_CMD = "uv run python scripts/tools/prepare_socnav_assets.py"
_SACADRL_MODEL_ID = "ga3c_cadrl_iros18"
_PREDICTIVE_MODEL_ID = "predictive_proxy_selected_v1"
_SOCNAV_IMPORT_LOCK = threading.Lock()


@dataclass
class SocNavPlannerConfig:
    """Simple config for SocNav-like planner adapters."""

    max_linear_speed: float = 3.0
    max_angular_speed: float = 1.0
    angular_gain: float = 1.0
    goal_tolerance: float = 0.25
    sacadrl_neighbors: int = 3
    sacadrl_bias_weight: float = 0.6
    orca_avoidance_weight: float = 1.2
    orca_neighbor_dist: float = 10.0
    orca_max_neighbors: int = 10
    orca_time_horizon: float = 6.0
    orca_time_horizon_obst: float = 3.0
    orca_obstacle_threshold: float = 0.5
    orca_obstacle_range: float = 6.0
    orca_obstacle_max_points: int = 80
    orca_obstacle_radius_scale: float = 1.0
    orca_heading_slowdown: float = 0.2
    # Optional diagnostic-only capture of the holonomic-to-unicycle ORCA adapter.
    # Keep disabled by default so normal benchmark episodes do not retain per-step traces.
    orca_adapter_trace_enabled: bool = False
    orca_symmetry_bias: float = 0.22
    orca_head_on_bias: float = 0.32
    orca_stall_speed_threshold: float = 0.12
    orca_stall_progress_epsilon: float = 0.03
    orca_stall_cycles_before_commit: int = 3
    orca_commit_persistence_steps: int = 10
    orca_commit_distance: float = 1.6
    orca_commit_lateral_gain: float = 0.45
    orca_forward_probe_distance: float = 1.1
    orca_side_probe_offset: float = 0.45
    orca_corner_probe_forward_scale: float = 1.5
    orca_corner_probe_side_scale: float = 1.5
    orca_head_on_probe_side_scale: float = 1.5
    orca_stall_nudge_factor: float = 0.15
    orca_obstacle_margin: float = 0.12
    orca_corner_clearance_scale: float = 1.35
    hrvo_neighbor_dist: float = 8.0
    hrvo_max_neighbors: int = 8
    hrvo_time_horizon: float = 4.0
    hrvo_uncertainty_offset: float = 0.05
    social_force_repulsion_weight: float = 0.8
    social_force_desired_speed: float = 1.0
    social_force_tau: float = 0.5
    social_force_factor: float = 5.1
    social_force_lambda_importance: float = 2.0
    social_force_gamma: float = 0.35
    social_force_n: int = 2
    social_force_n_prime: int = 3
    social_force_obstacle_factor: float = 10.0
    social_force_obstacle_threshold: float = 0.5
    social_force_obstacle_range: float = 6.0
    social_force_obstacle_max_points: int = 80
    social_force_obstacle_radius_scale: float = 1.0
    social_force_clip_force: bool = True
    social_force_max_force: float = 100.0
    occupancy_lookahead: float = 2.5
    occupancy_heading_sweep: float = pi * 2 / 3
    occupancy_candidates: int = 7
    occupancy_weight: float = 2.0
    occupancy_angle_weight: float = 0.3
    sacadrl_model_id: str = _SACADRL_MODEL_ID
    sacadrl_checkpoint_path: str | None = None
    sacadrl_pref_speed: float = 1.0
    sacadrl_max_other_agents: int = 3
    sacadrl_sorting_method: str = "closest_first"
    predictive_model_id: str = _PREDICTIVE_MODEL_ID
    predictive_checkpoint_path: str | None = None
    predictive_device: str = "cpu"
    predictive_feature_schema_name: str = "predictive_legacy_v1"
    predictive_max_agents: int = 16
    predictive_horizon_steps: int = 8
    predictive_ego_conditioning: bool = False
    predictive_rollout_dt: float = 0.2
    predictive_goal_weight: float = 1.0
    predictive_collision_weight: float = 6.0
    predictive_near_miss_weight: float = 1.5
    predictive_velocity_weight: float = 0.05
    predictive_turn_weight: float = 0.15
    predictive_ttc_weight: float = 0.0
    predictive_ttc_distance: float = 0.8
    predictive_safe_distance: float = 0.6
    predictive_near_distance: float = 1.0
    predictive_robot_radius: float = 0.3
    predictive_pedestrian_radius: float = 0.3
    predictive_speed_clearance_gain: float = 0.0
    predictive_progress_risk_weight: float = 1.0
    predictive_progress_risk_distance: float = 1.2
    predictive_hard_clearance_distance: float = 0.75
    predictive_hard_clearance_weight: float = 2.5
    predictive_adaptive_horizon_enabled: bool = True
    predictive_horizon_boost_steps: int = 4
    predictive_near_field_distance: float = 2.4
    predictive_near_field_speed_cap: float = 0.75
    predictive_near_field_speed_samples: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5)
    predictive_near_field_heading_deltas: tuple[float, ...] = (
        -pi / 2,
        -pi / 3,
        -pi / 4,
        -pi / 6,
        0.0,
        pi / 6,
        pi / 4,
        pi / 3,
        pi / 2,
    )
    predictive_candidate_speeds: tuple[float, ...] = (0.0, 0.5, 1.0)
    predictive_candidate_heading_deltas: tuple[float, ...] = (
        -pi / 4,
        -pi / 8,
        0.0,
        pi / 8,
        pi / 4,
    )
    predictive_allow_reverse_candidates: bool = False
    predictive_reverse_candidate_speeds: tuple[float, ...] = (-0.15, -0.3)
    predictive_reverse_near_field_only: bool = True
    predictive_progress_escape_enabled: bool = False
    predictive_progress_escape_distance: float = 1.2
    predictive_progress_escape_min_speed_ratio: float = 0.35
    predictive_progress_escape_heading_gain: float = 1.4
    predictive_progress_escape_clearance_margin: float = 0.2
    predictive_sequence_search_enabled: bool = False
    predictive_sequence_segments: int = 3
    predictive_sequence_branch_factor: int = 5
    predictive_sequence_beam_width: int = 8
    predictive_uncertainty_mode: str = "deterministic"
    predictive_uncertainty_base_std: float = 0.05
    predictive_uncertainty_growth_per_step: float = 0.03
    predictive_uncertainty_speed_scale: float = 0.10
    predictive_uncertainty_density_scale: float = 0.02
    predictive_risk_objective: str = "mean"
    predictive_risk_sample_count: int = 1
    predictive_risk_cvar_alpha: float = 0.25
    predictive_risk_seed: int = 7
    predictive_mcts_enabled: bool = False
    predictive_mcts_iterations: int = 48
    predictive_mcts_branch_factor: int = 4
    predictive_mcts_rollout_count: int = 2
    predictive_mcts_exploration_weight: float = 0.8
    predictive_phase_logic_enabled: bool = True
    predictive_phase_commit_clearance: float = 1.4
    predictive_phase_yield_clearance: float = 0.95
    predictive_phase_recover_clearance: float = 1.8
    predictive_phase_recover_progress: float = 0.25
    predictive_phase_commit_weight: float = 1.4
    predictive_phase_yield_weight: float = 2.0
    predictive_phase_align_weight: float = 0.4
    predictive_phase_recover_weight: float = 1.5
    # Forecast variant selection for planner-consumed baseline probabilistic prediction.
    # "none" keeps the planner on its default prediction source.
    forecast_variant: str = "none"
    forecast_variant_horizons_s: tuple[float, ...] = (0.5, 1.0, 2.0)
    forecast_variant_dt_s: float = 0.1
    forecast_variant_risk_distance_m: float = 3.0
    social_force_obstacle_law: str = DEFAULT_OBSTACLE_FORCE_LAW

    def __post_init__(self) -> None:
        """Resolve the obstacle law while retaining legacy defaults for old configs."""
        self.social_force_obstacle_law = resolve_obstacle_force_law(self.social_force_obstacle_law)

    @property
    def social_force_obstacle_law_version(self) -> str:
        """Return the obstacle law through the explicit versioned alias."""
        return self.social_force_obstacle_law

    @social_force_obstacle_law_version.setter
    def social_force_obstacle_law_version(self, value: Any) -> None:
        """Set the obstacle law through the explicit versioned alias."""
        self.social_force_obstacle_law = resolve_obstacle_force_law(value)


class TrivialReferencePlannerAdapter(OccupancyAwarePlannerMixin):
    """Minimal deterministic adapter for contributor templates and smoke tests.

    This adapter exists to document the real Robot SF local-planner adapter
    contract: accept a SocNav structured observation and return a bounded
    ``(linear_velocity, angular_velocity)`` command. It is diagnostic only and
    must not be cited as benchmark planner evidence.
    """

    def __init__(self, config: SocNavPlannerConfig | None = None) -> None:
        """Initialize the reference adapter with normal SocNav speed limits."""
        self.config = config or SocNavPlannerConfig()
        self._steps = 0

    def reset(self, *, seed: int | None = None) -> None:
        """Reset deterministic runtime counters.

        The optional seed is accepted for compatibility with stateful adapters.
        """
        del seed
        self._steps = 0

    def configure(self, config: SocNavPlannerConfig | None = None) -> None:
        """Replace the adapter configuration."""
        self.config = config or SocNavPlannerConfig()

    def close(self) -> None:
        """Release adapter resources.

        The reference adapter holds no external resources, but real adapters
        should use this hook for model sessions, files, or simulator handles.
        """

    def diagnostics(self) -> dict[str, Any]:
        """Return lightweight runtime diagnostics for episode metadata."""
        return {
            "adapter": "TrivialReferencePlannerAdapter",
            "steps": self._steps,
            "contract": "diagnostic_reference_only",
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to the ``[-pi, pi]`` interval.

        Returns:
            float: Wrapped angle in radians.
        """
        return wrap_angle_pi_closed(angle)

    def plan(self, observation: dict) -> tuple[float, float]:
        """Return a bounded goal-facing command from a SocNav observation.

        Args:
            observation: SocNav structured observation with ``robot`` and
                ``goal`` fields, or the flattened map-runner equivalent.

        Returns:
            tuple[float, float]: Bounded ``(v, omega)`` command.
        """
        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        goal = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]

        goal_delta = goal - robot_pos
        distance = float(np.linalg.norm(goal_delta))
        self._steps += 1
        if distance <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        desired_heading = float(atan2(goal_delta[1], goal_delta[0]))
        heading_error = self._wrap_angle(desired_heading - heading)
        angular = float(
            np.clip(
                float(self.config.angular_gain) * heading_error,
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        alignment = max(0.0, 1.0 - abs(heading_error) / pi)
        linear = float(
            np.clip(
                distance * alignment,
                0.0,
                float(self.config.max_linear_speed),
            )
        )
        return linear, angular


class SamplingPlannerAdapter(OccupancyAwarePlannerMixin):
    """
    Minimal waypoint-to-velocity adapter inspired by the SocNavBench sampling planner.

    ``use_upstream=False`` selects the explicit in-repository heuristic baseline used by
    ``algo=socnav_sampling``.  It is an experimental planner in its own right, not an
    implicit fallback for SocNavBench.  ``use_upstream=True`` delegates to the upstream
    SocNavBench planner and may use the heuristic only when ``allow_fallback=True``.
    """

    class _GoalDistanceObjective:
        """Minimal goal-distance objective for the upstream sampling planner."""

        def __init__(self, goal_pos: np.ndarray | None = None) -> None:
            """Initialize the goal-distance objective, defaulting the goal to the origin."""
            self._goal_pos = (
                np.zeros(2, dtype=float) if goal_pos is None else np.asarray(goal_pos, dtype=float)
            )

        def set_goal(self, goal_pos: np.ndarray) -> None:
            """Update the target goal position used for distance costs."""
            self._goal_pos = np.asarray(goal_pos, dtype=float)

        def evaluate_function(
            self, trajectory: Any, sim_state_hist: Any | None = None
        ) -> np.ndarray:
            """Return per-trajectory goal distance costs (lower is better)."""
            positions = trajectory.position_nk2()
            if positions.size == 0:
                return np.array([])
            valid_horizons = getattr(trajectory, "valid_horizons_n1", None)
            if valid_horizons is None:
                final_pos = positions[:, -1, :]
            else:
                idx = np.asarray(valid_horizons, dtype=int).reshape(-1) - 1
                idx = np.clip(idx, 0, positions.shape[1] - 1)
                final_pos = positions[np.arange(positions.shape[0]), idx, :]
            goal = self._goal_pos.reshape(1, 2)
            return np.linalg.norm(final_pos - goal, axis=1)

    def __init__(
        self,
        config: SocNavPlannerConfig | None = None,
        socnav_root: Path | None = None,
        planner_factory: Callable[[], Any] | None = None,
        *,
        use_upstream: bool = False,
        allow_fallback: bool = True,
    ):
        """Initialize the adapter with optional planner configuration."""

        self.config = config or SocNavPlannerConfig()
        self._planner = None
        self._goal_objective: SamplingPlannerAdapter._GoalDistanceObjective | None = None
        self._use_upstream = bool(use_upstream)
        self._allow_fallback = bool(allow_fallback)
        self._fallback_count = 0
        self._fallback_reason: str | None = None

        if self._use_upstream:
            if planner_factory is not None:
                self._planner = self._safe_call_factory(planner_factory)
            else:
                self._planner = self._load_upstream_planner(socnav_root)
            if self._planner is None and self._allow_fallback:
                self._record_fallback("upstream planner was unavailable during initialization")
                logger.warning(
                    "SamplingPlannerAdapter is running in fallback heuristic mode and "
                    "is not benchmark-ready."
                )
            if self._planner is None and not self._allow_fallback:
                raise RuntimeError(
                    "SamplingPlannerAdapter could not load the upstream planner. "
                    "Set allow_fallback=True to use the heuristic fallback."
                )
        else:
            logger.info(
                "SamplingPlannerAdapter is using the explicit in-repository heuristic baseline."
            )

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute a (v, w) command from the structured observation.

        Args:
            observation: SocNav structured observation Dict (robot, goal, pedestrians, map, sim).

        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        if self._planner is not None:
            return self._plan_upstream(observation)
        return self._heuristic_plan(observation)

    def _heuristic_plan(self, observation: dict) -> tuple[float, float]:
        """Compute a heuristic (v, w) command from the structured observation.

        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state["position"], pad=2)[:2]
        robot_heading = float(self._as_1d_float(robot_state["heading"], pad=1)[0])
        goal = self._as_1d_float(goal_state["current"], pad=2)[:2]

        to_goal = goal - robot_pos
        distance = float(np.linalg.norm(to_goal))
        if distance < self.config.goal_tolerance:
            return 0.0, 0.0

        # Light pedestrian repulsion to keep base planner pedestrian-aware
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        ped_count = (
            int(self._as_1d_float(ped_state.get("count", [0]), pad=1)[0]) if ped_state else 0
        )
        ped_positions = ped_positions[:ped_count]
        repulse = np.zeros(2, dtype=float)
        for ped in ped_positions:
            delta = robot_pos - ped
            dist = np.linalg.norm(delta) + 1e-6
            repulse += delta / dist**2

        base_vec = to_goal / (np.linalg.norm(to_goal) + 1e-6)
        if np.linalg.norm(repulse) > 1e-6:
            base_vec = base_vec + self.config.social_force_repulsion_weight * repulse
            if np.linalg.norm(base_vec) > 1e-6:
                base_vec = base_vec / np.linalg.norm(base_vec)

        # Adjust heading to favor obstacle-free paths when grid is available
        adjusted_vec, occ_penalty = self._get_safe_heading(robot_pos, base_vec, observation)

        desired_heading = atan2(adjusted_vec[1], adjusted_vec[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)

        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )

        # Slow down when sharply turning or when path shows occupancy
        linear_scale = max(0.0, 1.0 - abs(heading_error) / pi)
        linear_scale *= max(0.0, 1.0 - occ_penalty)
        linear = float(
            np.clip(distance * linear_scale, 0.0, self.config.max_linear_speed),
        )
        return linear, angular

    def _plan_upstream(self, observation: dict) -> tuple[float, float]:
        """Compute a (v, w) command using the upstream SocNavBench planner.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        if self._planner is None:
            return self._heuristic_plan(observation)

        try:
            robot_state, goal_state, _ = self._socnav_fields(observation)
            pos = robot_state["position"]
            robot_pos = np.asarray(pos, dtype=float)
            heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
            if self._goal_objective is not None:
                self._goal_objective.set_goal(goal_state["current"])
            start_config = self._planner.opt_waypt.__class__.from_pos3([pos[0], pos[1], heading])
            goal = goal_state["current"]
            goal_config = self._planner.opt_waypt.__class__.from_pos3([goal[0], goal[1], 0.0])
            data = self._planner.optimize(start_config=start_config, goal_config=goal_config)
            traj = data.get("trajectory")
            if traj is None:
                if not self._allow_fallback:
                    raise RuntimeError("SocNavBench planner returned no trajectory")
                self._record_fallback("upstream planner returned no trajectory")
                return self._heuristic_plan(observation)
            # NOTE: upstream returns a trajectory and controller matrices; for now we
            # consume only the immediate waypoint to preserve the (v, w) interface and
            # avoid binding to controller specifics. This keeps the adapter lightweight
            # while still aligning heading toward the planned path.
            next_pos = traj.position_nk2()[0, 0]
            to_next = next_pos - pos
            direction = to_next / (np.linalg.norm(to_next) + 1e-9)
            direction, occ_penalty = self._get_safe_heading(robot_pos, direction, observation)
            desired_heading = atan2(direction[1], direction[0])
            heading_error = self._wrap_angle(desired_heading - heading)
            angular = float(
                np.clip(
                    self.config.angular_gain * heading_error,
                    -self.config.max_angular_speed,
                    self.config.max_angular_speed,
                ),
            )
            linear = float(
                np.clip(
                    np.linalg.norm(to_next),
                    0.0,
                    self.config.max_linear_speed * max(0.0, 1.0 - occ_penalty),
                ),
            )
            return linear, angular
        # Upstream planner failures use the heuristic fallback when enabled.
        except Exception as exc:  # pragma: no cover - broad catch: planner surface unknown; heuristic fallback or re-raise
            if self._allow_fallback:
                self._record_fallback(f"upstream runtime failure: {type(exc).__name__}")
                return self._heuristic_plan(observation)
            raise RuntimeError("SocNavBench planner failed during _plan_upstream.") from exc

    def _record_fallback(self, reason: str) -> None:
        """Record a sticky upstream-to-heuristic fallback event."""
        self._fallback_count += 1
        self._fallback_reason = reason

    def _safe_call_factory(self, factory: Callable[[], Any]) -> Any | None:
        """Invoke a user-provided factory defensively.

        Returns:
            Planner instance from the factory or ``None`` on failure.
        """
        try:
            return factory()
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover
            return self._handle_socnav_failure(
                f"SocNavBench planner factory failed: {exc}", exc=exc
            )

    def _handle_socnav_failure(
        self, message: str, *, exc: Exception | None = None, not_found: bool = False
    ) -> Any | None:
        """Handle SocNavBench initialization failures with optional fallback.

        Returns:
            None when fallback is allowed; otherwise raises a descriptive error.
        """
        if self._allow_fallback:
            logger.warning("{}", message)
            return None
        if not_found:
            raise FileNotFoundError(message) from exc
        raise RuntimeError(message) from exc

    @staticmethod
    def _resolve_socnav_root(socnav_root: Path | None) -> Path:
        """Resolve the SocNavBench root directory.

        Returns:
            Path: Resolved SocNavBench root path.
        """
        if socnav_root is not None:
            return Path(socnav_root).expanduser()
        env_root = os.getenv(_SOCNAV_ROOT_ENV)
        if env_root:
            return Path(env_root).expanduser()
        return _SOCNAV_DEFAULT_ROOT

    @staticmethod
    def _allow_untrusted_socnav_root() -> bool:
        """Determine whether the environment explicitly allows untrusted roots.

        Returns:
            bool: True when the environment variable enables untrusted roots.
        """
        value = os.getenv(_SOCNAV_ALLOW_UNTRUSTED_ENV, "")
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _is_trusted_socnav_root(root: Path) -> bool:
        """Check whether the SocNavBench root lives inside the repository.

        Returns:
            bool: True when the root resolves under the repository directory.
        """
        repo_root = Path(__file__).resolve().parents[2]
        try:
            root.resolve().relative_to(repo_root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _validate_socnav_root(root: Path) -> list[Path]:
        """Validate that the SocNavBench root contains expected modules.

        Returns:
            list[Path]: Missing module paths.
        """
        missing: list[Path] = []
        for module in _SOCNAV_REQUIRED_MODULES:
            rel_path = Path(*module.split(".")).with_suffix(".py")
            if not (root / rel_path).exists():
                missing.append(root / rel_path)
        return missing

    @staticmethod
    def _import_socnav_modules() -> tuple[tuple[Any, Any, Any] | None, str | None]:
        """Import upstream SocNavBench modules from the provided root.

        Returns:
            tuple[tuple[Any, Any, Any] | None, str | None]:
            ``((central_params, sampling_planner, DotMap), None)`` on success;
            otherwise ``(None, error_message)`` on failure.
        """
        try:
            import params.central_params as central  # type: ignore  # noqa: PLC0415
            import planners.sampling_planner as sp  # type: ignore  # noqa: PLC0415
            from dotmap import DotMap  # type: ignore  # noqa: PLC0415

            return (central, sp, DotMap), None
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover
            hint = ""
            if isinstance(exc, ModuleNotFoundError) and "skfmm" in str(exc):
                hint = (
                    " Missing dependency `skfmm` detected. "
                    "Install SocNav prerequisites (for example `uv sync --extra socnav`)."
                )
            return None, f"{type(exc).__name__}: {exc}.{hint}"

    def _resolve_robot_dt(self, socnav_params: Any) -> float:
        """Resolve the robot dynamics timestep from SocNavBench params.

        Returns:
            float: Robot timestep.
        """
        dyn_params = getattr(socnav_params, "robot_dynamics_params", None)
        if dyn_params is None:
            return 0.1
        return float(getattr(dyn_params, "dt", 0.1))

    def _resolve_camera_dt(self, socnav_params: Any) -> float:
        """Resolve the camera timestep from SocNavBench params.

        Returns:
            float: Camera timestep.
        """
        camera_params = getattr(socnav_params, "camera_params", None)
        if camera_params is None:
            return 0.1
        return float(getattr(camera_params, "dt", 0.1))

    def _build_sampling_params(self, central: Any, sp: Any, DotMap: Any) -> Any | None:
        """Build sampling planner parameters for the upstream planner.

        Returns:
            Params object or ``None`` on failure.
        """
        params = DotMap()
        params.planner = sp.SamplingPlanner
        try:
            params.control_pipeline_params = central.create_control_pipeline_params()
        except SystemExit as exc:
            return self._handle_socnav_failure(
                "SocNavBench control pipeline parameters failed to load. "
                "Ensure the SocNavBench data directories exist. "
                f"See `{_SOCNAV_ASSET_SETUP_DOC}` and run `{_SOCNAV_ASSET_SETUP_CMD}`.",
                exc=exc,
            )
        return params

    def _load_upstream_planner(self, socnav_root: Path | None) -> Any | None:  # noqa: C901, PLR0912
        """Best-effort import of SocNavBench SamplingPlanner with defaults.

        Returns:
            Planner instance or ``None`` on failure.
        """
        env_root = os.getenv(_SOCNAV_ROOT_ENV)
        root_source = "argument" if socnav_root is not None else ("env" if env_root else "default")
        root_candidate = self._resolve_socnav_root(socnav_root)
        if root_source != "default" and ".." in root_candidate.parts:
            message = (
                "SocNavBench root contains parent-directory traversal segments ('..'). "
                f"Refusing to load from '{root_candidate}'. Provide a canonical trusted path."
            )
            return self._handle_socnav_failure(message)
        root = root_candidate.resolve()
        if not root.exists():
            message = (
                "SocNavBench root not found at "
                f"'{root}'. Set {_SOCNAV_ROOT_ENV} or pass socnav_root."
            )
            return self._handle_socnav_failure(message, not_found=True)

        if root_source != "default" and not self._is_trusted_socnav_root(root):
            if not self._allow_untrusted_socnav_root():
                message = (
                    "SocNavBench root is outside the repository root. "
                    f"Refusing to load from '{root}'. Set {_SOCNAV_ALLOW_UNTRUSTED_ENV}=1 "
                    "to explicitly allow untrusted SocNavBench roots."
                )
                return self._handle_socnav_failure(message)
            logger.warning(
                "Using SocNavBench root outside the repository: '{}'. Ensure this path is trusted.",
                root,
            )

        missing = self._validate_socnav_root(root)
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            message = (
                "SocNavBench root is missing required modules: "
                f"{missing_str}. Ensure the SocNavBench repo is complete."
            )
            return self._handle_socnav_failure(message, not_found=True)

        with _SOCNAV_IMPORT_LOCK:
            prev_cwd = Path.cwd()
            root_str = str(root)
            sys_path_inserted = False
            try:
                # Upstream SocNavBench params resolve INI paths relative to cwd at import time.
                os.chdir(root)
                if root_str not in sys.path:
                    sys.path.insert(0, root_str)
                    sys_path_inserted = True
                modules, import_error = self._import_socnav_modules()
                if modules is None:
                    message = "Failed to import SocNavBench modules."
                    if import_error:
                        message = f"{message} {import_error}"
                    return self._handle_socnav_failure(message)
                central, sp, DotMap = modules
                params = self._build_sampling_params(central, sp, DotMap)
                if params is None:
                    return None
                try:
                    obj_fn = self._GoalDistanceObjective()
                    self._goal_objective = obj_fn
                    return sp.SamplingPlanner(obj_fn=obj_fn, params=params)
                except AssertionError:
                    # Retry once after resetting the upstream singleton cache only when
                    # the upstream assertion indicates a stale/mismatched cached pipeline.
                    try:
                        import control_pipelines.control_pipeline_v0 as cp_v0  # type: ignore  # noqa: PLC0415

                        cp_v0.ControlPipelineV0.pipeline = None
                    except (ImportError, AttributeError) as exc:
                        return self._handle_socnav_failure(
                            "Failed to reset SocNavBench control pipeline singleton before retry.",
                            exc=exc,
                        )
                    obj_fn = self._GoalDistanceObjective()
                    self._goal_objective = obj_fn
                    try:
                        return sp.SamplingPlanner(obj_fn=obj_fn, params=params)
                    except (
                        AssertionError,
                        AttributeError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as exc:  # pragma: no cover
                        return self._handle_socnav_failure(
                            "Failed to initialize SocNavBench SamplingPlanner after singleton reset: "
                            f"{exc}. If this is an asset/data issue, see `{_SOCNAV_ASSET_SETUP_DOC}` "
                            f"and run `{_SOCNAV_ASSET_SETUP_CMD}`.",
                            exc=exc,
                        )
                except (
                    AttributeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:  # pragma: no cover
                    return self._handle_socnav_failure(
                        "Failed to initialize SocNavBench SamplingPlanner: "
                        f"{exc}. If this is an asset/data issue, see `{_SOCNAV_ASSET_SETUP_DOC}` "
                        f"and run `{_SOCNAV_ASSET_SETUP_CMD}`.",
                        exc=exc,
                    )
            finally:
                if sys_path_inserted:
                    try:
                        sys.path.remove(root_str)
                    except ValueError:
                        pass
                os.chdir(prev_cwd)

    def diagnostics(self) -> dict[str, Any]:
        """Return explicit implementation and fallback diagnostics."""
        upstream_requested = bool(getattr(self, "_use_upstream", False))
        upstream_loaded = getattr(self, "_planner", None) is not None
        fallback_count = int(getattr(self, "_fallback_count", 0) or 0)
        fallback_reason = getattr(self, "_fallback_reason", None)
        fallback_triggered = fallback_count > 0
        if fallback_triggered:
            implementation_mode = "heuristic_fallback"
        elif upstream_loaded:
            implementation_mode = "upstream_socnavbench"
        else:
            implementation_mode = "in_repo_heuristic_baseline"
        return {
            "planner_type": "SamplingPlannerAdapter",
            "implementation_mode": implementation_mode,
            "upstream_requested": upstream_requested,
            "upstream_loaded": upstream_loaded,
            "fallback_triggered": fallback_triggered,
            "fallback_count": fallback_count,
            "fallback_reason": fallback_reason,
            "readiness_status": "fallback" if fallback_triggered else "experimental",
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """
        Wrap angle to [-pi, pi].

        Returns:
            float: Wrapped angle in radians.
        """
        return wrap_angle_pi_closed(angle)


class SocNavPlannerPolicy:
    """Thin policy wrapper to plug planner adapters into Gym loops."""

    def __init__(self, adapter: SamplingPlannerAdapter | None = None):
        """Initialize the policy with a planner adapter."""

        self.adapter = adapter or SamplingPlannerAdapter()

    def act(self, observation: dict) -> tuple[float, float]:
        """Return (v, w) action for a SocNav structured observation."""
        return self.adapter.plan(observation)


class SocNavBenchComplexPolicy(SocNavPlannerPolicy):
    """
    Policy that prefers the upstream SocNavBench SamplingPlanner when available.

    By default this policy requires the upstream SocNavBench planner. Set
    ``allow_fallback=True`` to use the lightweight adapter when dependencies are missing.
    """

    def __init__(
        self,
        socnav_root: Path | None = None,
        adapter_config: SocNavPlannerConfig | None = None,
        *,
        allow_fallback: bool = False,
    ):
        """Initialize the policy, preferring the upstream SocNavBench planner when present."""
        # Deferred to call time: ``SocNavBenchSamplingAdapter`` lives in the socnav
        # facade, which imports this module eagerly, so a top-level import here would
        # create an import cycle.
        from robot_sf.planner.socnav import SocNavBenchSamplingAdapter  # noqa: PLC0415

        adapter = SocNavBenchSamplingAdapter(
            config=adapter_config,
            socnav_root=socnav_root,
            allow_fallback=allow_fallback,
        )
        super().__init__(adapter=adapter)


__all__ = [
    "SamplingPlannerAdapter",
    "SocNavBenchComplexPolicy",
    "SocNavPlannerConfig",
    "SocNavPlannerPolicy",
    "TrivialReferencePlannerAdapter",
]
