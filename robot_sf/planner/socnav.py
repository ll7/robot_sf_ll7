"""
Compatibility facade for SocNav-family planner adapters.

The original SocNav planner module remains import-compatible while planner-family
implementations are split into focused modules. Shared occupancy-grid helpers now
live in `robot_sf.planner.socnav_occupancy`; this module re-exports
`OccupancyAwarePlannerMixin` for existing imports.
"""

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from math import atan2, pi
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robot_sf.common.forecast_variants import FORECAST_VARIANT_CHOICES
from robot_sf.common.math_utils import wrap_angle_pi, wrap_angle_pi_closed

# Convention: optional-import guards catch ImportError only (ModuleNotFoundError is a
# subclass); bind the exception as `exc` for consistency across the codebase.
try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import tensorflow.compat.v1 as tf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import rvo2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    rvo2 = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pysocialforce import forces as sf_forces  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sf_forces = None  # type: ignore[assignment]

from robot_sf.models import resolve_model_path
from robot_sf.nav.occupancy_grid_utils import world_to_ego
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    LocalObstacleFeatureExtractor,
    infer_predictive_feature_schema,
    normalize_obstacle_lines,
    obstacle_lines_from_map,
    obstacle_lines_from_observation,
    validate_predictive_runtime_feature_schema,
)

try:  # pragma: no cover - exercised in minimal environments without torch
    from robot_sf.planner.predictive_model import (
        PredictiveTrajectoryModel,
        load_predictive_checkpoint,
    )
except ImportError:  # pragma: no cover - optional dependency
    PredictiveTrajectoryModel = Any  # type: ignore[misc,assignment]
    load_predictive_checkpoint = None  # type: ignore[assignment]
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin

if TYPE_CHECKING:
    from robot_sf.planner.socnav_prediction import (
        PredictionPlannerAdapter,
        SocNavBenchSamplingAdapter,
        make_prediction_policy,
    )
    from robot_sf.planner.socnav_sacadrl import SACADRLPlannerAdapter, make_sacadrl_policy
    from robot_sf.planner.socnav_social_force import (
        SocialForcePlannerAdapter,
        make_social_force_policy,
    )

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

_SACADRL_LAZY_EXPORTS = {
    "_SACADRLModel",
    "_SACADRL_STATE_ORDER",
    "_sacadrl_actions",
    "_sacadrl_session_config",
    "SACADRLPlannerAdapter",
    "make_sacadrl_policy",
}

_SOCIAL_FORCE_LAZY_EXPORTS = {
    "SocialForcePlannerAdapter",
    "make_social_force_policy",
}

_PREDICTION_LAZY_EXPORTS = {
    "PredictionPlannerAdapter",
    "SocNavBenchSamplingAdapter",
    "make_prediction_policy",
}


def __getattr__(name: str) -> Any:
    """Resolve extracted SACADRL, SocialForce, or Prediction symbols without importing eagerly.

    Returns:
        Any: Requested symbol from the extracted family module.
    """
    if name in _SACADRL_LAZY_EXPORTS:
        module = import_module("robot_sf.planner.socnav_sacadrl")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SOCIAL_FORCE_LAZY_EXPORTS:
        module = import_module("robot_sf.planner.socnav_social_force")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _PREDICTION_LAZY_EXPORTS:
        module = import_module("robot_sf.planner.socnav_prediction")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy public names in module introspection and wildcard imports.

    Returns:
        list[str]: Sorted names exposed by the facade and its lazy exports.
    """
    return sorted(
        set(globals())
        | {name for name in _SACADRL_LAZY_EXPORTS if not name.startswith("_")}
        | {name for name in _SOCIAL_FORCE_LAZY_EXPORTS if not name.startswith("_")}
        | {name for name in _PREDICTION_LAZY_EXPORTS if not name.startswith("_")}
    )


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

    Warning:
        By default this adapter uses a lightweight heuristic placeholder. Set
        ``use_upstream=True`` to delegate to the upstream SocNavBench sampling planner
        (benchmark-ready), and optionally allow fallback when dependencies are missing.
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

        if self._use_upstream:
            if planner_factory is not None:
                self._planner = self._safe_call_factory(planner_factory)
            else:
                self._planner = self._load_upstream_planner(socnav_root)
            if self._planner is None and self._allow_fallback:
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
            logger.warning(
                "SamplingPlannerAdapter is a heuristic fallback and is not benchmark-ready."
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
        except Exception as exc:  # pragma: no cover - safety net
            if self._allow_fallback:
                return self._heuristic_plan(observation)
            raise RuntimeError("SocNavBench planner failed during _plan_upstream.") from exc

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

        from robot_sf.planner.socnav_prediction import (  # noqa: PLC0415
            SocNavBenchSamplingAdapter,
        )

        adapter = SocNavBenchSamplingAdapter(
            config=adapter_config,
            socnav_root=socnav_root,
            allow_fallback=allow_fallback,
        )
        super().__init__(adapter=adapter)


class ORCAPlannerAdapter(SamplingPlannerAdapter):
    """ORCA planner adapter using rvo2 when available.

    Set ``allow_fallback=True`` to use the heuristic implementation when rvo2 is unavailable.
    """

    @dataclass
    class _OrcaLine:
        """ORCA half-plane constraint line."""

        point: np.ndarray
        direction: np.ndarray

    @dataclass
    class _Rvo2Scene:
        """Immutable simulator parameters plus mutable agent state for one ORCA step."""

        time_step: float
        neighbor_dist: float
        max_neighbors: int
        time_horizon: float
        time_horizon_obst: float
        robot_radius: float
        max_speed: float
        robot_pos: np.ndarray
        robot_velocity_world: np.ndarray
        ped_positions: np.ndarray
        ped_vel_world: np.ndarray
        ped_radius: float
        ped_max_speeds: tuple[float, ...]
        obstacle_vertices: tuple[tuple[tuple[float, float], ...], ...]

    _EPS = 1e-6

    def __init__(self, config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False):
        """Initialize the ORCA adapter with optional rvo2 fallback."""
        self.config = config or SocNavPlannerConfig()
        self._allow_fallback = allow_fallback
        self._fallback_warned = False
        self._bound_static_obstacle_points = np.zeros((0, 2), dtype=float)
        self._bound_static_obstacle_spacing = 0.0
        self._rvo2_sim: Any | None = None
        self._rvo2_signature: tuple[Any, ...] | None = None
        self._rvo2_robot_id: int | None = None
        self._rvo2_ped_ids: list[int] = []
        self.reset()

    def reset(self) -> None:
        """Clear per-episode commitment and stall tracking state."""
        self._stall_cycles = 0
        self._last_goal_distance: float | None = None
        self._commit_side = 0
        self._commit_side_ttl = 0
        self._clear_rvo2_simulator()

    def _clear_rvo2_simulator(self) -> None:
        """Discard cached rvo2 state at an explicit lifecycle boundary."""
        self._rvo2_sim = None
        self._rvo2_signature = None
        self._rvo2_robot_id = None
        self._rvo2_ped_ids = []

    def bind_static_obstacle_points(self, points: Any, *, spacing: float) -> None:
        """Bind sampled exact static obstacle points for use during planning."""
        previous_points = self._bound_static_obstacle_points
        previous_spacing = self._bound_static_obstacle_spacing
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            self._bound_static_obstacle_points = np.zeros((0, 2), dtype=float)
            self._bound_static_obstacle_spacing = 0.0
            if previous_points.size or previous_spacing != 0.0:
                self._clear_rvo2_simulator()
            return
        if arr.ndim == 1:
            if arr.size % 2 != 0:
                raise ValueError("Static obstacle points must have an even number of coordinates.")
            arr = arr.reshape(-1, 2)
        elif arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Static obstacle points must be convertible to an (N, 2) array.")
        self._bound_static_obstacle_points = np.asarray(arr[:, :2], dtype=float)
        self._bound_static_obstacle_spacing = float(max(spacing, self._EPS))
        if (
            not np.array_equal(previous_points, self._bound_static_obstacle_points)
            or previous_spacing != self._bound_static_obstacle_spacing
        ):
            self._clear_rvo2_simulator()

    def _extract_bound_static_obstacle_points(
        self,
        robot_pos: np.ndarray,
        robot_heading: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract nearby obstacle points from bound exact map geometry.

        Returns:
            tuple[np.ndarray, np.ndarray]: World-space obstacle centers and per-point radii.
        """
        points = self._bound_static_obstacle_points
        if points.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        centers, _dist_sq = self._select_nearby_points(
            points,
            robot_pos,
            float(self.config.orca_obstacle_range),
            max(int(self.config.orca_obstacle_max_points), 0),
        )
        if centers.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        spacing = float(max(self._bound_static_obstacle_spacing, self._EPS))
        base_radius = 0.5 * spacing * float(self.config.orca_obstacle_radius_scale)
        radii = np.full(
            (centers.shape[0],),
            base_radius + float(self.config.orca_obstacle_margin),
            dtype=float,
        )
        corner_mask = self._orca_corner_obstacle_mask(
            centers=centers,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
        )
        if np.any(corner_mask):
            radii[corner_mask] *= float(self.config.orca_corner_clearance_scale)
        return self._coalesce_static_obstacle_points(
            centers=centers,
            radii=radii,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            resolution=spacing,
        )

    def _ensure_rvo2(self) -> bool:
        """Return True when rvo2 is available, else handle fallback/error behavior.

        Returns:
            bool: True when rvo2 is available, False when falling back.
        """
        if rvo2 is not None:
            return True
        if self._allow_fallback:
            if not self._fallback_warned:
                logger.warning(
                    "rvo2 not available; falling back to heuristic ORCA behavior. "
                    "Install the 'orca' extra for the benchmark-ready implementation.",
                )
                self._fallback_warned = True
            return False
        raise RuntimeError(
            "rvo2 is required for the benchmark-ready ORCA planner. "
            "Install via `uv sync --extra orca` or set allow_fallback=True."
        )

    @staticmethod
    def _det(a: np.ndarray, b: np.ndarray) -> float:
        """2D determinant (cross product z-component).

        Returns:
            float: Determinant value for the 2D vectors.
        """
        return float(a[0] * b[1] - a[1] * b[0])

    @classmethod
    def _normalize(cls, vec: np.ndarray) -> np.ndarray:
        """Return a unit vector (or zeros when norm is too small).

        Returns:
            np.ndarray: Normalized 2D vector or zeros when norm is near zero.
        """
        norm = np.linalg.norm(vec)
        if norm < cls._EPS:
            return np.zeros(2, dtype=float)
        return vec / norm

    @classmethod
    def _linear_program_interval(
        cls,
        lines: list[_OrcaLine],
        line_no: int,
        radius: float,
    ) -> tuple[bool, float, float]:
        """Compute the feasible interval on a constraint line.

        Returns:
            tuple[bool, float, float]: (success, t_left, t_right).
        """
        line = lines[line_no]
        dot = float(np.dot(line.point, line.direction))
        discriminant = dot * dot + radius * radius - float(np.dot(line.point, line.point))
        if discriminant < 0.0:
            return False, 0.0, 0.0
        sqrt_discriminant = float(np.sqrt(discriminant))
        t_left = -dot - sqrt_discriminant
        t_right = -dot + sqrt_discriminant

        for i in range(line_no):
            denom = cls._det(line.direction, lines[i].direction)
            numer = cls._det(lines[i].direction, line.point - lines[i].point)
            if abs(denom) <= cls._EPS:
                if numer < 0.0:
                    return False, 0.0, 0.0
                continue
            t = numer / denom
            if denom >= 0.0:
                t_right = min(t_right, t)
            else:
                t_left = max(t_left, t)
            if t_left > t_right:
                return False, 0.0, 0.0
        return True, t_left, t_right

    @classmethod
    def _linear_program1(
        cls,
        lines: list[_OrcaLine],
        line_no: int,
        radius: float,
        opt_velocity: np.ndarray,
        direction_opt: bool,
    ) -> tuple[bool, np.ndarray]:
        """Solve a 1D linear program on a single constraint line.

        Returns:
            tuple[bool, np.ndarray]: (success, resulting velocity) tuple.
        """
        success, t_left, t_right = cls._linear_program_interval(lines, line_no, radius)
        if not success:
            return False, opt_velocity
        line = lines[line_no]

        if direction_opt:
            if np.dot(opt_velocity, line.direction) > 0.0:
                result = line.point + t_right * line.direction
            else:
                result = line.point + t_left * line.direction
        else:
            t = float(np.dot(line.direction, opt_velocity - line.point))
            if t < t_left:
                result = line.point + t_left * line.direction
            elif t > t_right:
                result = line.point + t_right * line.direction
            else:
                result = line.point + t * line.direction
        return True, result

    @classmethod
    def _linear_program2(
        cls,
        lines: list[_OrcaLine],
        radius: float,
        opt_velocity: np.ndarray,
        direction_opt: bool,
    ) -> tuple[int, np.ndarray]:
        """Solve a 2D linear program with circular bound and half-plane constraints.

        Returns:
            tuple[int, np.ndarray]: (violating line index, resulting velocity).
        """
        if direction_opt:
            result = cls._normalize(opt_velocity) * radius
        elif np.linalg.norm(opt_velocity) > radius:
            result = cls._normalize(opt_velocity) * radius
        else:
            result = opt_velocity.copy()

        for i, line in enumerate(lines):
            if cls._det(line.direction, line.point - result) > 0.0:
                temp = result.copy()
                success, result = cls._linear_program1(
                    lines, i, radius, opt_velocity, direction_opt
                )
                if not success:
                    return i, temp
        return len(lines), result

    @classmethod
    def _linear_program3(
        cls,
        lines: list[_OrcaLine],
        num_obst_lines: int,
        begin_line: int,
        radius: float,
        result: np.ndarray,
    ) -> np.ndarray:
        """Resolve infeasible constraints via projection (ORCA fallback).

        Returns:
            np.ndarray: Adjusted velocity satisfying constraints when possible.
        """
        distance = 0.0
        for i in range(begin_line, len(lines)):
            if cls._det(lines[i].direction, lines[i].point - result) > distance:
                proj_lines = list(lines[:num_obst_lines])
                for j in range(num_obst_lines, i):
                    determinant = cls._det(lines[i].direction, lines[j].direction)
                    if abs(determinant) <= cls._EPS:
                        if np.dot(lines[i].direction, lines[j].direction) > 0.0:
                            continue
                        point = 0.5 * (lines[i].point + lines[j].point)
                    else:
                        point = (
                            lines[i].point
                            + (
                                cls._det(lines[j].direction, lines[i].point - lines[j].point)
                                / determinant
                            )
                            * lines[i].direction
                        )
                    direction = cls._normalize(lines[j].direction - lines[i].direction)
                    proj_lines.append(cls._OrcaLine(point=point, direction=direction))
                temp_result = result.copy()
                perp_direction = np.array([-lines[i].direction[1], lines[i].direction[0]])
                _idx, result = cls._linear_program2(proj_lines, radius, perp_direction, True)
                if cls._det(lines[i].direction, lines[i].point - result) > distance:
                    result = temp_result
                distance = cls._det(lines[i].direction, lines[i].point - result)
        return result

    @staticmethod
    def _ego_to_world(vec: np.ndarray, heading: float) -> np.ndarray:
        """Rotate an ego-frame vector into world coordinates.

        Returns:
            np.ndarray: Vector expressed in world coordinates.
        """
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        return np.array([cos_h * vec[0] - sin_h * vec[1], sin_h * vec[0] + cos_h * vec[1]])

    @staticmethod
    def _world_to_ego_vec(vec: np.ndarray, heading: float) -> np.ndarray:
        """Rotate a world-frame vector into ego coordinates.

        Returns:
            np.ndarray: Vector expressed in ego coordinates.
        """
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        return np.array([cos_h * vec[0] + sin_h * vec[1], -sin_h * vec[0] + cos_h * vec[1]])

    @staticmethod
    def _side_sign(value: float) -> int:
        """Return deterministic sign for a scalar."""
        if value > 0.0:
            return 1
        if value < 0.0:
            return -1
        return 1

    @classmethod
    def _preferred_velocity(
        cls, goal: np.ndarray, robot_pos: np.ndarray, robot_heading: float, max_speed: float
    ) -> np.ndarray:
        """Compute the preferred velocity toward the goal in ego coordinates.

        Returns:
            np.ndarray: Preferred velocity in ego coordinates.
        """
        goal_ego = np.asarray(
            world_to_ego(float(goal[0]), float(goal[1]), (robot_pos, robot_heading)),
            dtype=float,
        )
        return cls._normalize(goal_ego) * max_speed

    @classmethod
    def _extract_pedestrians(cls, ped_state: dict) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Extract pedestrian positions/velocities and metadata.

        Returns:
            tuple[np.ndarray, np.ndarray, int, float]: Positions, velocities, count, radius.
        """
        raw_positions = ped_state.get("positions")
        if raw_positions is None:
            ped_positions = np.zeros((0, 2), dtype=float)
        else:
            ped_positions = np.asarray(raw_positions, dtype=float)
        raw_velocities = ped_state.get("velocities")
        if raw_velocities is None:
            ped_velocities = np.zeros_like(ped_positions, dtype=float)
        else:
            ped_velocities = np.asarray(raw_velocities, dtype=float)
        ped_count = int(np.asarray(ped_state.get("count", [0]), dtype=float)[0])
        ped_positions = ped_positions[:ped_count]
        if ped_velocities.size == 0:
            ped_velocities = np.zeros_like(ped_positions, dtype=float)
        ped_velocities = ped_velocities[:ped_count]
        ped_radius_arr = np.asarray(ped_state.get("radius", [0.3]), dtype=float)
        ped_radius = float(ped_radius_arr[0] if ped_radius_arr.ndim > 0 else ped_radius_arr)
        return ped_positions, ped_velocities, ped_count, ped_radius

    def _build_orca_lines(  # noqa: PLR0913
        self,
        *,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_velocity: np.ndarray,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        robot_radius: float,
        ped_radius: float | np.ndarray,
        time_step: float,
        time_horizon: float | None = None,
        neighbor_dist: float | None = None,
    ) -> list[_OrcaLine]:
        """Build ORCA half-plane constraints for nearby pedestrians/obstacles.

        Returns:
            list[_OrcaLine]: ORCA half-plane constraints.
        """
        lines: list[self._OrcaLine] = []
        effective_time_horizon = max(
            time_horizon if time_horizon is not None else self.config.orca_time_horizon,
            self._EPS,
        )
        effective_neighbor_dist = max(
            neighbor_dist if neighbor_dist is not None else self.config.orca_neighbor_dist,
            0.0,
        )
        neighbor_dist_sq = effective_neighbor_dist**2
        inv_time_horizon = 1.0 / effective_time_horizon
        inv_time_step = 1.0 / max(time_step, self._EPS)

        use_radius_array = isinstance(ped_radius, np.ndarray)
        for index, (ped_pos_world, ped_vel) in enumerate(
            zip(ped_positions, ped_velocities, strict=False)
        ):
            ped_pos_ego = np.asarray(
                world_to_ego(
                    float(ped_pos_world[0]),
                    float(ped_pos_world[1]),
                    (robot_pos, robot_heading),
                ),
                dtype=float,
            )
            if np.dot(ped_pos_ego, ped_pos_ego) > neighbor_dist_sq:
                continue

            rel_pos = ped_pos_ego
            rel_vel = robot_velocity - ped_vel
            dist_sq = float(np.dot(rel_pos, rel_pos))
            ped_radius_value = float(ped_radius[index]) if use_radius_array else float(ped_radius)
            combined_radius = robot_radius + ped_radius_value
            combined_radius_sq = combined_radius**2

            if dist_sq > combined_radius_sq:
                w = rel_vel - inv_time_horizon * rel_pos
                w_length_sq = float(np.dot(w, w))
                dot = float(np.dot(w, rel_pos))
                if dot < 0.0 and dot * dot > combined_radius_sq * w_length_sq:
                    w_length = float(np.sqrt(w_length_sq))
                    unit_w = w / max(w_length, self._EPS)
                    direction = np.array([unit_w[1], -unit_w[0]])
                    u = (combined_radius * inv_time_horizon - w_length) * unit_w
                else:
                    leg = float(np.sqrt(max(dist_sq - combined_radius_sq, 0.0)))
                    if self._det(rel_pos, w) > 0.0:
                        direction = np.array(
                            [
                                rel_pos[0] * leg - rel_pos[1] * combined_radius,
                                rel_pos[1] * leg + rel_pos[0] * combined_radius,
                            ]
                        ) / max(dist_sq, self._EPS)
                    else:
                        direction = -np.array(
                            [
                                rel_pos[0] * leg + rel_pos[1] * combined_radius,
                                rel_pos[1] * leg - rel_pos[0] * combined_radius,
                            ]
                        ) / max(dist_sq, self._EPS)
                    u = float(np.dot(rel_vel, direction)) * direction - rel_vel
            else:
                w = rel_vel - inv_time_step * rel_pos
                w_length = float(np.linalg.norm(w))
                unit_w = w / max(w_length, self._EPS)
                direction = np.array([unit_w[1], -unit_w[0]])
                u = (combined_radius * inv_time_step - w_length) * unit_w

            lines.append(self._OrcaLine(point=robot_velocity + 0.5 * u, direction=direction))
        return lines

    @staticmethod
    def _grid_cell_centers(
        indices: np.ndarray, origin: np.ndarray, resolution: float
    ) -> np.ndarray:
        """Convert grid indices to grid-frame centers.

        Returns:
            np.ndarray: Grid-frame centers for the provided indices.
        """
        rows = indices[:, 0].astype(float)
        cols = indices[:, 1].astype(float)
        x = origin[0] + (cols + 0.5) * resolution
        y = origin[1] + (rows + 0.5) * resolution
        return np.stack([x, y], axis=1)

    @staticmethod
    def _ego_centers_to_world(
        centers: np.ndarray, robot_pos: np.ndarray, robot_heading: float
    ) -> np.ndarray:
        """Rotate/translate ego-frame centers into world coordinates.

        Returns:
            np.ndarray: World-space centers.
        """
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        x_world = cos_h * centers[:, 0] - sin_h * centers[:, 1]
        y_world = sin_h * centers[:, 0] + cos_h * centers[:, 1]
        return np.stack([x_world, y_world], axis=1) + np.asarray(robot_pos, dtype=float)

    @staticmethod
    def _select_nearby_points(
        centers: np.ndarray,
        robot_pos: np.ndarray,
        max_range: float,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter centers by range and cap to the closest points.

        Returns:
            tuple[np.ndarray, np.ndarray]: Filtered centers and squared distances.
        """
        offsets = centers - np.asarray(robot_pos, dtype=float)
        dist_sq = np.einsum("ij,ij->i", offsets, offsets)
        keep = dist_sq <= max_range**2
        if not np.any(keep):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
        centers = centers[keep]
        dist_sq = dist_sq[keep]
        if max_points > 0 and centers.shape[0] > max_points:
            order = np.argsort(dist_sq)[:max_points]
            centers = centers[order]
            dist_sq = dist_sq[order]
        return centers, dist_sq

    @staticmethod
    def _forward_lateral_components(
        centers: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project world-space obstacle centers onto robot-forward and lateral axes.

        Returns:
            tuple[np.ndarray, np.ndarray]: Forward and lateral distances.
        """
        forward = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        offsets = centers - robot_pos[None, :]
        return offsets @ forward, offsets @ lateral

    def _coalesce_static_obstacle_points(
        self,
        *,
        centers: np.ndarray,
        radii: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
        resolution: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce dense occupied-cell clouds into a smaller static obstacle set.

        Returns:
            tuple[np.ndarray, np.ndarray]: Coalesced obstacle centers and radii.
        """
        if centers.shape[0] <= 1:
            return centers, radii

        forward_dist, lateral_dist = self._forward_lateral_components(
            centers,
            robot_pos,
            robot_heading,
        )
        ahead_mask = forward_dist >= -resolution
        if np.any(ahead_mask):
            centers = centers[ahead_mask]
            radii = radii[ahead_mask]
            forward_dist = forward_dist[ahead_mask]
            lateral_dist = lateral_dist[ahead_mask]
        if centers.shape[0] <= 1:
            return centers, radii

        forward_bin = max(resolution * 2.0, float(self.config.orca_forward_probe_distance) * 0.5)
        lateral_bin = max(resolution * 2.0, float(self.config.orca_side_probe_offset) * 1.5)
        clusters: dict[tuple[int, int], list[int]] = {}
        for index, (forward_value, lateral_value) in enumerate(
            zip(forward_dist, lateral_dist, strict=False)
        ):
            key = (
                int(np.floor(forward_value / max(forward_bin, self._EPS))),
                int(np.floor(lateral_value / max(lateral_bin, self._EPS))),
            )
            clusters.setdefault(key, []).append(index)

        coalesced_centers: list[np.ndarray] = []
        coalesced_radii: list[float] = []
        for member_indices in clusters.values():
            cluster_centers = centers[member_indices]
            cluster_radii = radii[member_indices]
            center = np.mean(cluster_centers, axis=0)
            spread = (
                float(np.max(np.linalg.norm(cluster_centers - center[None, :], axis=1)))
                if cluster_centers.shape[0] > 1
                else 0.0
            )
            radius = float(np.max(cluster_radii) + spread)
            coalesced_centers.append(center)
            coalesced_radii.append(radius)

        result_centers = np.asarray(coalesced_centers, dtype=float)
        result_radii = np.asarray(coalesced_radii, dtype=float)
        if result_centers.shape[0] <= 1:
            return result_centers, result_radii

        offsets = result_centers - robot_pos[None, :]
        dist_sq = np.einsum("ij,ij->i", offsets, offsets)
        max_points = max(int(self.config.orca_obstacle_max_points), 0)
        if max_points > 0 and result_centers.shape[0] > max_points:
            order = np.argsort(dist_sq)[:max_points]
            result_centers = result_centers[order]
            result_radii = result_radii[order]
        return result_centers, result_radii

    def _extract_obstacles_from_grid(
        self, observation: dict, robot_pos: np.ndarray, robot_heading: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract nearby obstacle points from the occupancy grid.

        Returns:
            tuple[np.ndarray, np.ndarray]: World-space obstacle centers and per-point radii.
        """
        bound_centers, bound_radii = self._extract_bound_static_obstacle_points(
            robot_pos,
            robot_heading,
        )
        if bound_centers.size:
            return bound_centers, bound_radii

        payload = self._obstacle_grid_payload(observation)
        if payload is None:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
        grid, meta, channel_idx, resolution = payload

        obstacle_mask = grid[channel_idx] >= float(self.config.orca_obstacle_threshold)
        if not np.any(obstacle_mask):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        indices = np.argwhere(obstacle_mask)
        if indices.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        origin = self._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)
        centers = self._grid_cell_centers(indices, origin, resolution)
        use_ego = bool(self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5)
        if use_ego:
            centers = self._ego_centers_to_world(centers, robot_pos, robot_heading)

        centers, _dist_sq = self._select_nearby_points(
            centers,
            robot_pos,
            float(self.config.orca_obstacle_range),
            max(int(self.config.orca_obstacle_max_points), 0),
        )
        if centers.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        base_radius = (
            0.5 * np.sqrt(2.0) * resolution * float(self.config.orca_obstacle_radius_scale)
        )
        radii = np.full(
            (centers.shape[0],),
            base_radius + float(self.config.orca_obstacle_margin),
            dtype=float,
        )
        corner_mask = self._orca_corner_obstacle_mask(
            centers=centers,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
        )
        if np.any(corner_mask):
            radii[corner_mask] *= float(self.config.orca_corner_clearance_scale)
        return self._coalesce_static_obstacle_points(
            centers=centers,
            radii=radii,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            resolution=resolution,
        )

    def _orca_corner_obstacle_mask(
        self,
        *,
        centers: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
    ) -> np.ndarray:
        """Return a mask for obstacle points that should receive corner-clearance inflation.

        Returns:
            np.ndarray: Boolean mask aligned with ``centers``.
        """
        if centers.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        forward = self._normalize(np.array([np.cos(robot_heading), np.sin(robot_heading)]))
        offsets = centers - robot_pos[None, :]
        forward_dist = offsets @ forward
        lateral = np.abs(offsets[:, 0] * forward[1] - offsets[:, 1] * forward[0])
        return (
            (forward_dist > 0.0)
            & (
                forward_dist
                <= float(self.config.orca_forward_probe_distance)
                * float(self.config.orca_corner_probe_forward_scale)
            )
            & (
                lateral
                <= float(self.config.orca_side_probe_offset)
                * float(self.config.orca_corner_probe_side_scale)
            )
        )

    def _direct_path_blocked(
        self,
        *,
        robot_pos: np.ndarray,
        robot_heading: float,
        goal_direction_world: np.ndarray,
        observation: dict,
    ) -> bool:
        """Check whether the immediate forward corridor looks blocked.

        Returns:
            bool: True when the occupancy probe ahead of the robot is blocked.
        """
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return False
        grid, meta = payload
        channel = self._preferred_channel(meta)
        if channel < 0 or channel >= grid.shape[0]:
            return False
        forward = self._normalize(goal_direction_world)
        if np.linalg.norm(forward) < self._EPS:
            forward = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        probe_distance = float(self.config.orca_forward_probe_distance)
        side_offset = float(self.config.orca_side_probe_offset)
        probe_points = (
            robot_pos + forward * probe_distance,
            robot_pos + forward * probe_distance + lateral * side_offset,
            robot_pos + forward * probe_distance - lateral * side_offset,
        )
        values = [self._grid_value(point, grid, meta, channel) for point in probe_points]
        return max(values, default=0.0) >= float(self.config.orca_obstacle_threshold)

    def _select_commit_side(
        self,
        *,
        robot_pos: np.ndarray,
        robot_heading: float,
        ped_positions: np.ndarray,
        goal_direction_world: np.ndarray,
        observation: dict,
    ) -> int:
        """Choose a deterministic lateral side for bypassing stalls or symmetry.

        Returns:
            int: ``1`` for left-bias, ``-1`` for right-bias.
        """
        if self._commit_side_ttl > 0 and self._commit_side != 0:
            return self._commit_side

        forward = self._normalize(goal_direction_world)
        if np.linalg.norm(forward) < self._EPS:
            forward = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        score = float(self.config.orca_symmetry_bias)

        if ped_positions.size:
            offsets = ped_positions - robot_pos[None, :]
            forward_dist = offsets @ forward
            near = (forward_dist > 0.0) & (forward_dist <= float(self.config.orca_commit_distance))
            if np.any(near):
                lateral_offsets = offsets[near] @ lateral
                score += -float(np.sum(lateral_offsets))

        payload = self._extract_grid_payload(observation)
        if payload is not None:
            grid, meta = payload
            channel = self._preferred_channel(meta)
            if channel >= 0:
                probe_distance = float(self.config.orca_forward_probe_distance)
                side_offset = float(self.config.orca_side_probe_offset)
                left_point = robot_pos + forward * probe_distance + lateral * side_offset
                right_point = robot_pos + forward * probe_distance - lateral * side_offset
                left_occ = self._grid_value(left_point, grid, meta, channel)
                right_occ = self._grid_value(right_point, grid, meta, channel)
                score += right_occ - left_occ

        side = self._side_sign(score)
        self._commit_side = side
        self._commit_side_ttl = max(int(self.config.orca_commit_persistence_steps), 1)
        return side

    def _update_stall_state(
        self, *, goal_distance: float, current_speed: float, blocked: bool
    ) -> bool:
        """Track repeated low-progress cycles and return whether commit mode should activate.

        Returns:
            bool: True when the stall counter has crossed the commit threshold.
        """
        progress = 0.0
        if self._last_goal_distance is not None:
            progress = self._last_goal_distance - goal_distance
        stalled = current_speed <= float(
            self.config.orca_stall_speed_threshold
        ) and progress <= float(self.config.orca_stall_progress_epsilon)
        if stalled:
            self._stall_cycles += 1
        else:
            self._stall_cycles = 0
            if self._commit_side_ttl > 0:
                self._commit_side_ttl -= 1
            if self._commit_side_ttl <= 0:
                self._commit_side = 0
        self._last_goal_distance = goal_distance
        return self._stall_cycles >= int(self.config.orca_stall_cycles_before_commit)

    def _apply_commit_bias(
        self,
        *,
        preferred_velocity_world: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
        ped_positions: np.ndarray,
        observation: dict,
        current_speed: float,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Adjust preferred world velocity for head-on, symmetry, and stall cases.

        Returns:
            np.ndarray: Bias-adjusted preferred world velocity.
        """
        goal_direction_world = self._normalize(goal - robot_pos)
        if np.linalg.norm(goal_direction_world) < self._EPS:
            return preferred_velocity_world

        blocked = self._direct_path_blocked(
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            goal_direction_world=goal_direction_world,
            observation=observation,
        )
        goal_distance = float(np.linalg.norm(goal - robot_pos))
        commit_active = self._update_stall_state(
            goal_distance=goal_distance,
            current_speed=current_speed,
            blocked=blocked,
        )
        lateral = np.array([-goal_direction_world[1], goal_direction_world[0]], dtype=float)
        side_sign = 0

        if ped_positions.size:
            offsets = ped_positions - robot_pos[None, :]
            forward_dist = offsets @ goal_direction_world
            lateral_dist = np.abs(offsets @ lateral)
            head_on = (
                (forward_dist > 0.0)
                & (forward_dist <= float(self.config.orca_commit_distance))
                & (
                    lateral_dist
                    <= float(self.config.orca_side_probe_offset)
                    * float(self.config.orca_head_on_probe_side_scale)
                )
            )
            if np.any(head_on):
                side_sign = self._select_commit_side(
                    robot_pos=robot_pos,
                    robot_heading=robot_heading,
                    ped_positions=ped_positions,
                    goal_direction_world=goal_direction_world,
                    observation=observation,
                )
                preferred_velocity_world = preferred_velocity_world + (
                    lateral
                    * side_sign
                    * float(self.config.orca_head_on_bias)
                    * float(self.config.max_linear_speed)
                )

        if blocked or commit_active:
            side_sign = side_sign or self._select_commit_side(
                robot_pos=robot_pos,
                robot_heading=robot_heading,
                ped_positions=ped_positions,
                goal_direction_world=goal_direction_world,
                observation=observation,
            )
            preferred_velocity_world = preferred_velocity_world + (
                lateral
                * side_sign
                * float(self.config.orca_commit_lateral_gain)
                * float(self.config.max_linear_speed)
            )
            if current_speed <= float(self.config.orca_stall_speed_threshold):
                preferred_velocity_world = preferred_velocity_world + (
                    goal_direction_world
                    * float(self.config.orca_stall_nudge_factor)
                    * float(self.config.max_linear_speed)
                )

        speed = np.linalg.norm(preferred_velocity_world)
        max_speed = float(self.config.max_linear_speed)
        if speed > max_speed:
            preferred_velocity_world = preferred_velocity_world / max(speed, self._EPS) * max_speed
        return preferred_velocity_world

    def _solve_orca_velocity(
        self, lines: list[_OrcaLine], preferred_velocity: np.ndarray
    ) -> np.ndarray:
        """Solve ORCA constraints for the new velocity.

        Returns:
            np.ndarray: Resulting velocity that satisfies the constraints.
        """
        if not lines:
            return preferred_velocity
        line_fail, new_velocity = self._linear_program2(
            lines,
            self.config.max_linear_speed,
            preferred_velocity,
            False,
        )
        if line_fail < len(lines):
            new_velocity = self._linear_program3(
                lines,
                num_obst_lines=0,
                begin_line=line_fail,
                radius=self.config.max_linear_speed,
                result=new_velocity,
            )
        return new_velocity

    def _velocity_world_to_command(
        self,
        *,
        velocity_world: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
        observation: dict,
    ) -> tuple[float, float]:
        """Convert a world-frame velocity vector into ``(v, w)`` with occupancy penalty.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        speed = float(np.linalg.norm(velocity_world))
        if speed < self._EPS:
            return 0.0, 0.0
        world_dir = self._normalize(np.asarray(velocity_world, dtype=float))
        world_dir, occ_penalty = self._get_safe_heading(robot_pos, world_dir, observation)
        desired_heading = atan2(world_dir[1], world_dir[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        angular = float(
            np.clip(
                1.5 * self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )
        heading_scale = 1.0 - min(1.0, abs(heading_error) / (pi / 2)) * float(
            self.config.orca_heading_slowdown
        )
        linear = float(
            np.clip(
                speed,
                0.0,
                self.config.max_linear_speed
                * max(0.0, 1.0 - occ_penalty)
                * max(0.0, heading_scale),
            )
        )
        return linear, angular

    def plan_velocity_world(self, observation: dict) -> np.ndarray:
        """Compute a world-frame translational velocity using ORCA or the heuristic fallback.

        Returns:
            np.ndarray: World-frame ``[vx, vy]`` translational velocity.
        """
        if not self._ensure_rvo2():
            return self._heuristic_velocity_world(observation)
        return self._rvo2_velocity_world(observation)

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute ``(v, w)`` using the ORCA world-velocity plan and unicycle projection.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, _goal_state, _ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        velocity_world = self.plan_velocity_world(observation)
        return self._velocity_world_to_command(
            velocity_world=velocity_world,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            observation=observation,
        )

    def _rvo2_simulator_for(self, scene: _Rvo2Scene) -> tuple[Any, int, list[int]]:
        """Return a simulator reused only for an identical immutable scene signature."""
        signature = (
            scene.time_step,
            scene.neighbor_dist,
            scene.max_neighbors,
            scene.time_horizon,
            scene.time_horizon_obst,
            scene.robot_radius,
            scene.max_speed,
            len(scene.ped_max_speeds),
            scene.ped_radius,
            scene.ped_max_speeds,
            scene.obstacle_vertices,
        )
        if self._rvo2_sim is not None and self._rvo2_signature == signature:
            assert self._rvo2_robot_id is not None
            return self._rvo2_sim, self._rvo2_robot_id, self._rvo2_ped_ids

        sim = rvo2.PyRVOSimulator(
            scene.time_step,
            scene.neighbor_dist,
            scene.max_neighbors,
            scene.time_horizon,
            scene.time_horizon_obst,
            scene.robot_radius,
            scene.max_speed,
        )
        robot_id = sim.addAgent(
            tuple(scene.robot_pos),
            scene.neighbor_dist,
            scene.max_neighbors,
            scene.time_horizon,
            scene.time_horizon_obst,
            scene.robot_radius,
            scene.max_speed,
            tuple(scene.robot_velocity_world),
        )
        ped_ids = []
        for idx, ped_max_speed in enumerate(scene.ped_max_speeds):
            ped_ids.append(
                sim.addAgent(
                    tuple(scene.ped_positions[idx]),
                    scene.neighbor_dist,
                    scene.max_neighbors,
                    scene.time_horizon,
                    scene.time_horizon_obst,
                    scene.ped_radius,
                    ped_max_speed,
                    tuple(scene.ped_vel_world[idx]),
                )
            )
        for vertices in scene.obstacle_vertices:
            sim.addObstacle(list(vertices))
        if scene.obstacle_vertices:
            sim.processObstacles()

        self._rvo2_sim = sim
        self._rvo2_signature = signature
        self._rvo2_robot_id = robot_id
        self._rvo2_ped_ids = ped_ids
        return sim, robot_id, ped_ids

    def _rvo2_velocity_world(self, observation: dict) -> np.ndarray:
        """Compute a world-frame velocity using the rvo2 ORCA solver.

        Returns:
            np.ndarray: World-frame ``[vx, vy]`` translational velocity.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        preferred_velocity_ego = self._preferred_velocity(
            goal, robot_pos, robot_heading, self.config.max_linear_speed
        )
        if np.linalg.norm(preferred_velocity_ego) < self._EPS:
            return 0.0, 0.0

        preferred_velocity_world = self._ego_to_world(preferred_velocity_ego, robot_heading)

        time_step = float(
            np.asarray(observation.get("sim", {}).get("timestep", [0.1]), dtype=float)[0]
        )
        if time_step <= self._EPS:
            logger.warning(
                "Invalid timestep ({}) for ORCA planner; defaulting to 0.1s.",
                time_step,
            )
            time_step = 0.1

        robot_radius = float(np.asarray(robot_state.get("radius", [0.3]), dtype=float)[0])
        robot_speed = float(np.asarray(robot_state.get("speed", [0.0]), dtype=float)[0])
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        robot_velocity_world = np.array(
            [robot_speed * cos_h, robot_speed * sin_h],
            dtype=float,
        )

        ped_positions, ped_velocities, ped_count, ped_radius = self._extract_pedestrians(ped_state)
        preferred_velocity_world = self._apply_commit_bias(
            preferred_velocity_world=preferred_velocity_world,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            ped_positions=ped_positions,
            observation=observation,
            current_speed=robot_speed,
            goal=goal,
        )
        if ped_count > 0:
            ped_vel_world = np.zeros_like(ped_velocities, dtype=float)
            ped_vel_world[:, 0] = cos_h * ped_velocities[:, 0] - sin_h * ped_velocities[:, 1]
            ped_vel_world[:, 1] = sin_h * ped_velocities[:, 0] + cos_h * ped_velocities[:, 1]
        else:
            ped_vel_world = np.zeros_like(ped_velocities, dtype=float)

        max_neighbors = int(self.config.orca_max_neighbors)
        if max_neighbors <= 0:
            max_neighbors = max(1, ped_count)

        neighbor_dist = float(self.config.orca_neighbor_dist)
        time_horizon = float(self.config.orca_time_horizon)
        time_horizon_obst = float(self.config.orca_time_horizon_obst)
        max_speed = float(self.config.max_linear_speed)
        obstacle_positions, obstacle_radii = self._extract_obstacles_from_grid(
            observation, robot_pos, robot_heading
        )
        obstacle_vertices: tuple[tuple[tuple[float, float], ...], ...] = tuple(
            (
                (float(center[0] - radius), float(center[1] - radius)),
                (float(center[0] + radius), float(center[1] - radius)),
                (float(center[0] + radius), float(center[1] + radius)),
                (float(center[0] - radius), float(center[1] + radius)),
            )
            for center, radius in zip(obstacle_positions, obstacle_radii, strict=True)
        )
        ped_max_speeds = tuple(
            max(float(np.linalg.norm(ped_vel_world[idx])), max_speed) for idx in range(ped_count)
        )
        scene = self._Rvo2Scene(
            time_step=time_step,
            neighbor_dist=neighbor_dist,
            max_neighbors=max_neighbors,
            time_horizon=time_horizon,
            time_horizon_obst=time_horizon_obst,
            robot_radius=robot_radius,
            max_speed=max_speed,
            robot_pos=robot_pos,
            robot_velocity_world=robot_velocity_world,
            ped_positions=ped_positions,
            ped_vel_world=ped_vel_world,
            ped_radius=ped_radius,
            ped_max_speeds=ped_max_speeds,
            obstacle_vertices=obstacle_vertices,
        )
        sim, robot_id, ped_ids = self._rvo2_simulator_for(scene)

        # The cached simulator carries state across doStep(), so reset every mutable agent field.
        sim.setAgentPosition(robot_id, tuple(robot_pos))
        sim.setAgentVelocity(robot_id, tuple(robot_velocity_world))
        sim.setAgentPrefVelocity(robot_id, tuple(preferred_velocity_world))
        for ped_id, position, velocity in zip(ped_ids, ped_positions, ped_vel_world, strict=True):
            sim.setAgentPosition(ped_id, tuple(position))
            sim.setAgentVelocity(ped_id, tuple(velocity))
            sim.setAgentPrefVelocity(ped_id, tuple(velocity))

        sim.doStep()
        new_velocity_world = np.asarray(sim.getAgentVelocity(robot_id), dtype=float)
        return new_velocity_world

    def _heuristic_velocity_world(self, observation: dict) -> np.ndarray:
        """Compute a world-frame velocity using the legacy ORCA-inspired heuristic.

        Returns:
            np.ndarray: World-frame ``[vx, vy]`` translational velocity.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        preferred_velocity = self._preferred_velocity(
            goal, robot_pos, robot_heading, self.config.max_linear_speed
        )
        if np.linalg.norm(preferred_velocity) < self._EPS:
            return 0.0, 0.0

        ped_positions, ped_velocities, _ped_count, ped_radius = self._extract_pedestrians(ped_state)
        robot_speed = float(np.asarray(robot_state.get("speed", [0.0]), dtype=float)[0])
        robot_velocity = np.array([robot_speed, 0.0], dtype=float)
        preferred_velocity_world = self._apply_commit_bias(
            preferred_velocity_world=self._ego_to_world(preferred_velocity, robot_heading),
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            ped_positions=ped_positions,
            observation=observation,
            current_speed=robot_speed,
            goal=goal,
        )
        preferred_velocity = self._world_to_ego_vec(preferred_velocity_world, robot_heading)

        time_step = float(
            np.asarray(observation.get("sim", {}).get("timestep", [0.1]), dtype=float)[0]
        )
        robot_radius = float(np.asarray(robot_state.get("radius", [0.3]), dtype=float)[0])

        lines = self._build_orca_lines(
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            robot_velocity=robot_velocity,
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
            time_step=time_step,
        )
        obstacle_positions, obstacle_radii = self._extract_obstacles_from_grid(
            observation, robot_pos, robot_heading
        )
        if obstacle_positions.size:
            obstacle_velocities = np.zeros_like(obstacle_positions, dtype=float)
            lines.extend(
                self._build_orca_lines(
                    robot_pos=robot_pos,
                    robot_heading=robot_heading,
                    robot_velocity=robot_velocity,
                    ped_positions=obstacle_positions,
                    ped_velocities=obstacle_velocities,
                    robot_radius=robot_radius,
                    ped_radius=obstacle_radii,
                    time_step=time_step,
                    time_horizon=self.config.orca_time_horizon_obst,
                    neighbor_dist=self.config.orca_obstacle_range,
                )
            )
        new_velocity = self._solve_orca_velocity(lines, preferred_velocity)
        return self._ego_to_world(new_velocity, robot_heading)

    def _heuristic_plan(self, observation: dict) -> tuple[float, float]:
        """Compute ``(v, w)`` using the legacy ORCA-inspired heuristic.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, _goal_state, _ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        velocity_world = self._heuristic_velocity_world(observation)
        return self._velocity_world_to_command(
            velocity_world=velocity_world,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            observation=observation,
        )


class HRVOPlannerAdapter(ORCAPlannerAdapter):
    """Hybrid Reciprocal Velocity Obstacles local planner.

    This is a local, benchmark-facing HRVO-inspired implementation informed by the
    upstream `snape/HRVO` geometry and the lightweight VO reference linked in
    issue `#726`. It remains an in-repo implementation rather than an upstream
    wrapper and should therefore be treated conservatively in benchmark claims.
    """

    @dataclass
    class _VelocityObstacle:
        """Hybrid reciprocal velocity obstacle with apex and side rays."""

        apex: np.ndarray
        side1: np.ndarray
        side2: np.ndarray

    @dataclass
    class _Candidate:
        """Candidate velocity and the VO boundaries that generated it."""

        position: np.ndarray
        obstacle1: int
        obstacle2: int

    @staticmethod
    def _clip_arcsin_ratio(numerator: float, denominator: float) -> float:
        """Return a numerically safe ratio for ``asin``."""
        if denominator <= 0.0:
            return 1.0
        return float(np.clip(numerator / denominator, -1.0, 1.0))

    @classmethod
    def _tangent_direction(cls, angle: float) -> np.ndarray:
        """Return a unit ray direction for a tangent boundary angle."""
        return cls._normalize(np.array([np.cos(angle), np.sin(angle)], dtype=float))

    @classmethod
    def _project_onto_side(
        cls, apex: np.ndarray, side: np.ndarray, point: np.ndarray
    ) -> np.ndarray:
        """Project a point onto the forward ray starting at ``apex`` along ``side``.

        Returns:
            np.ndarray: Closest point on the forward ray.
        """
        t = max(float(np.dot(point - apex, side)), 0.0)
        return apex + t * side

    @classmethod
    def _circle_side_intersections(
        cls,
        apex: np.ndarray,
        side: np.ndarray,
        radius: float,
    ) -> list[np.ndarray]:
        """Return intersections between a forward ray and the speed circle."""
        d = cls._det(apex, side)
        discriminant = radius * radius - d * d
        if discriminant <= 0.0:
            return []
        offset = float(np.dot(apex, side))
        root = float(np.sqrt(discriminant))
        intersections: list[np.ndarray] = []
        for t in (-offset - root, -offset + root):
            if t >= 0.0:
                intersections.append(apex + t * side)
        return intersections

    @classmethod
    def _ray_intersection(
        cls,
        apex1: np.ndarray,
        side1: np.ndarray,
        apex2: np.ndarray,
        side2: np.ndarray,
    ) -> np.ndarray | None:
        """Return the intersection of two forward rays when it lies on both rays."""
        determinant = cls._det(side1, side2)
        if abs(determinant) <= cls._EPS:
            return None
        delta = apex2 - apex1
        s = cls._det(delta, side2) / determinant
        t = cls._det(delta, side1) / determinant
        if s < 0.0 or t < 0.0:
            return None
        return apex1 + s * side1

    @classmethod
    def _inside_velocity_obstacle(
        cls,
        obstacle: _VelocityObstacle,
        velocity: np.ndarray,
    ) -> bool:
        """Return whether a velocity lies inside the HRVO forbidden cone."""
        rel = velocity - obstacle.apex
        return (
            cls._det(obstacle.side2, rel) <= cls._EPS and cls._det(obstacle.side1, rel) >= -cls._EPS
        )

    def _build_hrvo_obstacles(
        self,
        *,
        robot_velocity_world: np.ndarray,
        preferred_velocity_world: np.ndarray,
        other_positions: np.ndarray,
        other_velocities_world: np.ndarray,
        other_pref_velocities_world: np.ndarray,
        robot_radius: float,
        other_radii: np.ndarray,
        time_step: float,
    ) -> list[_VelocityObstacle]:
        """Construct HRVO cones for nearby dynamic neighbors.

        Returns:
            list[_VelocityObstacle]: HRVO cones in world-velocity space.
        """
        neighbor_dist = max(float(self.config.hrvo_neighbor_dist), 0.0)
        max_neighbors = max(int(self.config.hrvo_max_neighbors), 0)
        effective_time_horizon = max(float(self.config.hrvo_time_horizon), self._EPS)
        uncertainty_offset = max(float(self.config.hrvo_uncertainty_offset), 0.0)
        if other_positions.size == 0:
            return []

        offsets = other_positions
        dist_sq = np.einsum("ij,ij->i", offsets, offsets)
        if neighbor_dist > 0.0:
            keep = dist_sq <= neighbor_dist * neighbor_dist
        else:
            keep = np.ones((other_positions.shape[0],), dtype=bool)
        if not np.any(keep):
            return []

        order = np.argsort(dist_sq[keep])
        if max_neighbors > 0:
            order = order[:max_neighbors]
        kept_positions = other_positions[keep][order]
        kept_velocities = other_velocities_world[keep][order]
        kept_pref_velocities = other_pref_velocities_world[keep][order]
        kept_radii = other_radii[keep][order]

        velocity_obstacles: list[self._VelocityObstacle] = []
        for other_position, other_velocity, other_pref_velocity, other_radius in zip(
            kept_positions,
            kept_velocities,
            kept_pref_velocities,
            kept_radii,
            strict=True,
        ):
            relative_position = np.asarray(other_position, dtype=float)
            relative_velocity = robot_velocity_world - np.asarray(other_velocity, dtype=float)
            combined_radius = robot_radius + float(other_radius)
            distance = float(np.linalg.norm(relative_position))
            if distance < self._EPS:
                continue
            speed_horizon = (
                np.linalg.norm(robot_velocity_world)
                + np.linalg.norm(other_velocity)
                + np.linalg.norm(other_pref_velocity)
                + float(self.config.max_linear_speed)
            ) * effective_time_horizon
            if distance > combined_radius + speed_horizon:
                continue

            if distance > combined_radius:
                angle = float(np.arctan2(relative_position[1], relative_position[0]))
                opening = float(np.arcsin(self._clip_arcsin_ratio(combined_radius, distance)))
                side1 = self._tangent_direction(angle - opening)
                side2 = self._tangent_direction(angle + opening)
                side_det = max(2.0 * np.sin(opening) * np.cos(opening), self._EPS)
                pref_delta = preferred_velocity_world - np.asarray(other_pref_velocity, dtype=float)
                if self._det(relative_position, pref_delta) > 0.0:
                    scale = 0.5 * self._det(relative_velocity, side2) / side_det
                    apex = other_velocity + scale * side1
                else:
                    scale = 0.5 * self._det(relative_velocity, side1) / side_det
                    apex = other_velocity + scale * side2
                apex = apex - (
                    uncertainty_offset * distance / max(combined_radius, self._EPS)
                ) * self._normalize(relative_position)
            else:
                apex = 0.5 * (np.asarray(other_velocity, dtype=float) + robot_velocity_world) - (
                    uncertainty_offset
                    + 0.5 * (combined_radius - distance) / max(time_step, self._EPS)
                ) * self._normalize(relative_position)
                normal = self._normalize(np.array([-relative_position[1], relative_position[0]]))
                side1 = normal
                side2 = -normal

            velocity_obstacles.append(
                self._VelocityObstacle(
                    apex=np.asarray(apex, dtype=float),
                    side1=np.asarray(side1, dtype=float),
                    side2=np.asarray(side2, dtype=float),
                )
            )
        return velocity_obstacles

    def _seed_hrvo_candidate(self, preferred_velocity_world: np.ndarray) -> np.ndarray:
        """Return the bounded preferred velocity used to seed HRVO candidate search.

        Returns:
            np.ndarray: Preferred velocity clipped to the planner speed limit.
        """
        max_speed = float(self.config.max_linear_speed)
        preferred_speed = float(np.linalg.norm(preferred_velocity_world))
        if preferred_speed <= max_speed:
            return preferred_velocity_world.copy()
        return preferred_velocity_world / max(preferred_speed, self._EPS) * max_speed

    def _hrvo_boundary_candidates(
        self,
        velocity_obstacles: list[_VelocityObstacle],
        preferred_velocity_world: np.ndarray,
    ) -> list[_Candidate]:
        """Enumerate candidates from individual obstacle boundaries and the speed circle.

        Returns:
            list[_Candidate]: Candidate velocities sourced from one obstacle at a time.
        """
        max_speed = float(self.config.max_linear_speed)
        candidates: list[self._Candidate] = []
        for index, obstacle in enumerate(velocity_obstacles):
            for side in (obstacle.side1, obstacle.side2):
                projected = self._project_onto_side(obstacle.apex, side, preferred_velocity_world)
                if np.linalg.norm(projected) <= max_speed + self._EPS:
                    candidates.append(
                        self._Candidate(position=projected, obstacle1=index, obstacle2=index)
                    )
                for point in self._circle_side_intersections(obstacle.apex, side, max_speed):
                    candidates.append(
                        self._Candidate(position=point, obstacle1=-1, obstacle2=index)
                    )
        return candidates

    def _hrvo_intersection_candidates(
        self,
        velocity_obstacles: list[_VelocityObstacle],
    ) -> list[_Candidate]:
        """Enumerate candidates from intersections between obstacle boundary rays.

        Returns:
            list[_Candidate]: Candidate velocities defined by paired obstacle boundaries.
        """
        max_speed = float(self.config.max_linear_speed)
        candidates: list[self._Candidate] = []
        for left_index, left_obstacle in enumerate(velocity_obstacles[:-1]):
            for right_index, right_obstacle in enumerate(
                velocity_obstacles[left_index + 1 :],
                start=left_index + 1,
            ):
                for left_side in (left_obstacle.side1, left_obstacle.side2):
                    for right_side in (right_obstacle.side1, right_obstacle.side2):
                        point = self._ray_intersection(
                            left_obstacle.apex,
                            left_side,
                            right_obstacle.apex,
                            right_side,
                        )
                        if point is None:
                            continue
                        if np.linalg.norm(point) <= max_speed + self._EPS:
                            candidates.append(
                                self._Candidate(
                                    position=point,
                                    obstacle1=left_index,
                                    obstacle2=right_index,
                                )
                            )
        return candidates

    def _preferred_lateral_axis(self, preferred_velocity_world: np.ndarray) -> np.ndarray:
        """Return a rotation-invariant lateral axis relative to the preferred velocity.

        Returns:
            np.ndarray: Unit vector perpendicular to the preferred velocity.
        """
        norm = float(np.linalg.norm(preferred_velocity_world))
        if norm < self._EPS:
            return np.array([0.0, 1.0], dtype=float)
        pref_unit = preferred_velocity_world / norm
        return np.array([-pref_unit[1], pref_unit[0]], dtype=float)

    def _solve_hrvo_velocity(
        self,
        velocity_obstacles: list[_VelocityObstacle],
        preferred_velocity_world: np.ndarray,
    ) -> np.ndarray:
        """Select the feasible candidate nearest to the preferred velocity.

        Returns:
            np.ndarray: Chosen HRVO world-frame velocity.
        """
        seed = self._seed_hrvo_candidate(preferred_velocity_world)
        candidates = [self._Candidate(position=seed, obstacle1=-1, obstacle2=-1)]
        candidates.extend(
            self._hrvo_boundary_candidates(velocity_obstacles, preferred_velocity_world)
        )
        candidates.extend(self._hrvo_intersection_candidates(velocity_obstacles))
        lateral_axis = self._preferred_lateral_axis(preferred_velocity_world)

        candidates.sort(
            key=lambda candidate: (
                float(np.linalg.norm(candidate.position - preferred_velocity_world)),
                float(abs(float(np.dot(candidate.position, lateral_axis))) < 1e-6),
                -float(abs(float(np.dot(candidate.position, lateral_axis)))),
            )
        )
        best_invalid: np.ndarray | None = None
        best_invalid_cover = -1
        for candidate in candidates:
            valid = True
            for obstacle_index, obstacle in enumerate(velocity_obstacles):
                if obstacle_index in {candidate.obstacle1, candidate.obstacle2}:
                    continue
                if self._inside_velocity_obstacle(obstacle, candidate.position):
                    valid = False
                    if obstacle_index > best_invalid_cover:
                        best_invalid = candidate.position
                        best_invalid_cover = obstacle_index
                    break
            if valid:
                return candidate.position
        return best_invalid if best_invalid is not None else seed

    def _break_hrvo_symmetry(
        self,
        velocity_obstacles: list[_VelocityObstacle],
        preferred_velocity_world: np.ndarray,
        solved_velocity_world: np.ndarray,
    ) -> np.ndarray:
        """Resolve exact symmetric ties by committing to one obstacle boundary.

        Returns:
            np.ndarray: Symmetry-broken world-frame velocity.
        """
        lateral_axis = self._preferred_lateral_axis(preferred_velocity_world)
        solved_lateral = float(np.dot(solved_velocity_world, lateral_axis))
        if abs(solved_lateral) > 1e-6 or not velocity_obstacles:
            return solved_velocity_world
        if not any(
            self._inside_velocity_obstacle(obstacle, preferred_velocity_world)
            or self._inside_velocity_obstacle(obstacle, solved_velocity_world)
            for obstacle in velocity_obstacles
        ):
            return solved_velocity_world

        choices: list[np.ndarray] = []
        max_speed = float(self.config.max_linear_speed)
        for obstacle in velocity_obstacles:
            for side in (obstacle.side1, obstacle.side2):
                candidate = self._project_onto_side(obstacle.apex, side, preferred_velocity_world)
                candidate_lateral = float(np.dot(candidate, lateral_axis))
                if (
                    np.linalg.norm(candidate) <= max_speed + self._EPS
                    and abs(candidate_lateral) > 1e-6
                ):
                    choices.append(candidate)
        if not choices:
            return solved_velocity_world
        choices.sort(
            key=lambda candidate: (
                float(np.linalg.norm(candidate - preferred_velocity_world)),
                -float(abs(float(np.dot(candidate, lateral_axis)))),
                -float(np.dot(candidate, lateral_axis)),
            )
        )
        return choices[0]

    def plan_velocity_world(self, observation: dict) -> np.ndarray:
        """Compute a world-frame translational velocity with the local HRVO solver.

        Returns:
            np.ndarray: World-frame ``[vx, vy]`` translational velocity.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state["position"], dtype=float)
        robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
        goal = np.asarray(goal_state["current"], dtype=float)

        preferred_velocity_ego = self._preferred_velocity(
            goal,
            robot_pos,
            robot_heading,
            float(self.config.max_linear_speed),
        )
        if np.linalg.norm(preferred_velocity_ego) < self._EPS:
            return np.zeros(2, dtype=float)
        preferred_velocity_world = self._ego_to_world(preferred_velocity_ego, robot_heading)

        robot_speed = float(np.asarray(robot_state.get("speed", [0.0]), dtype=float)[0])
        robot_velocity_world = np.array(
            [
                robot_speed * float(np.cos(robot_heading)),
                robot_speed * float(np.sin(robot_heading)),
            ],
            dtype=float,
        )
        robot_radius = float(np.asarray(robot_state.get("radius", [0.3]), dtype=float)[0])
        time_step = float(
            np.asarray(observation.get("sim", {}).get("timestep", [0.1]), dtype=float)[0]
        )
        if time_step <= self._EPS:
            time_step = 0.1

        ped_positions, ped_velocities, ped_count, ped_radius = self._extract_pedestrians(ped_state)
        if ped_count > 0:
            cos_h = float(np.cos(robot_heading))
            sin_h = float(np.sin(robot_heading))
            ped_vel_world = np.zeros_like(ped_velocities, dtype=float)
            ped_vel_world[:, 0] = cos_h * ped_velocities[:, 0] - sin_h * ped_velocities[:, 1]
            ped_vel_world[:, 1] = sin_h * ped_velocities[:, 0] + cos_h * ped_velocities[:, 1]
            ped_pref_vel_world = ped_vel_world.copy()
            ped_radii = np.full((ped_positions.shape[0],), float(ped_radius), dtype=float)
        else:
            ped_positions = np.zeros((0, 2), dtype=float)
            ped_vel_world = np.zeros((0, 2), dtype=float)
            ped_pref_vel_world = np.zeros((0, 2), dtype=float)
            ped_radii = np.zeros((0,), dtype=float)

        obstacle_positions, obstacle_radii = self._extract_obstacles_from_grid(
            observation,
            robot_pos,
            robot_heading,
        )
        if obstacle_positions.size:
            obstacle_offsets = obstacle_positions - robot_pos[None, :]
            obstacle_vel_world = np.zeros_like(obstacle_offsets, dtype=float)
            obstacle_pref_vel_world = np.zeros_like(obstacle_offsets, dtype=float)
        else:
            obstacle_offsets = np.zeros((0, 2), dtype=float)
            obstacle_vel_world = np.zeros((0, 2), dtype=float)
            obstacle_pref_vel_world = np.zeros((0, 2), dtype=float)
            obstacle_radii = np.zeros((0,), dtype=float)

        if ped_positions.size or obstacle_offsets.size:
            other_positions = np.concatenate(
                [ped_positions - robot_pos[None, :], obstacle_offsets],
                axis=0,
            )
            other_velocities_world = np.concatenate([ped_vel_world, obstacle_vel_world], axis=0)
            other_pref_velocities_world = np.concatenate(
                [ped_pref_vel_world, obstacle_pref_vel_world],
                axis=0,
            )
            other_radii = np.concatenate([ped_radii, obstacle_radii], axis=0)
        else:
            return preferred_velocity_world

        velocity_obstacles = self._build_hrvo_obstacles(
            robot_velocity_world=robot_velocity_world,
            preferred_velocity_world=preferred_velocity_world,
            other_positions=other_positions,
            other_velocities_world=other_velocities_world,
            other_pref_velocities_world=other_pref_velocities_world,
            robot_radius=robot_radius,
            other_radii=other_radii,
            time_step=time_step,
        )
        if not velocity_obstacles:
            return preferred_velocity_world
        solved = self._solve_hrvo_velocity(velocity_obstacles, preferred_velocity_world)
        return self._break_hrvo_symmetry(
            velocity_obstacles,
            preferred_velocity_world,
            solved,
        )


def make_orca_policy(
    config: SocNavPlannerConfig | None = None, *, allow_fallback: bool = False
) -> SocNavPlannerPolicy:
    """
    Convenience constructor for ORCA-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping ORCAPlannerAdapter.
    """

    return SocNavPlannerPolicy(
        adapter=ORCAPlannerAdapter(config=config, allow_fallback=allow_fallback)
    )


def make_hrvo_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for the local HRVO planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping HRVOPlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=HRVOPlannerAdapter(config=config))


__all__ = [
    "FORECAST_VARIANT_CHOICES",
    "PREDICTIVE_OBSTACLE_FEATURE_SCHEMA",
    "Any",
    "Callable",
    "HRVOPlannerAdapter",
    "LocalObstacleFeatureExtractor",
    "ORCAPlannerAdapter",
    "OccupancyAwarePlannerMixin",
    "Path",
    "PredictionPlannerAdapter",
    "PredictiveTrajectoryModel",
    "SACADRLPlannerAdapter",
    "SamplingPlannerAdapter",
    "SocNavBenchComplexPolicy",
    "SocNavBenchSamplingAdapter",
    "SocNavPlannerConfig",
    "SocNavPlannerPolicy",
    "SocialForcePlannerAdapter",
    "TrivialReferencePlannerAdapter",
    "atan2",
    "dataclass",
    "import_module",
    "infer_predictive_feature_schema",
    "load_predictive_checkpoint",
    "logger",
    "make_hrvo_policy",
    "make_orca_policy",
    "make_prediction_policy",
    "make_sacadrl_policy",
    "make_social_force_policy",
    "normalize_obstacle_lines",
    "np",
    "obstacle_lines_from_map",
    "obstacle_lines_from_observation",
    "os",
    "pi",
    "resolve_model_path",
    "rvo2",
    "sf_forces",
    "sys",
    "tf",
    "threading",
    "torch",
    "validate_predictive_runtime_feature_schema",
    "world_to_ego",
    "wrap_angle_pi",
    "wrap_angle_pi_closed",
]
