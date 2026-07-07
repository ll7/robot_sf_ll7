"""Optional OMPL kinodynamic-route diagnostic for Robot SF.

This module uses OMPL (Open Motion Planning Library) control-based planners
to assess whether a route is feasible under a simple differential-drive
motion model. It is designed as a lightweight, optional diagnostic that
never blocks the main planning pipeline.

Key differences from existing planners:
    - A*/Theta* (classic_global_planner): grid-based holonomic planners. They
      ignore the robot's turning-radius and speed constraints.
    - Local social-navigation planners (ORCA, Social Force): reactive,
      pedestrian-aware controllers that assume a valid global waypoint exists.
    - This OMPL diagnostic: checks kinodynamic feasibility of a route segment
      using sampling-based planners (RRT, SST, KPIECE) that honour differential
      constraints. No pedestrian dynamics are modelled.

OMPL is imported lazily so the module fails closed when the package is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Optional OMPL import gate
# ---------------------------------------------------------------------------

_OMPL_AVAILABLE: bool = False
_OMPL_IMPORT_ERROR: str | None = None

try:
    # OMPL Python bindings are installed as `ompl` on PyPI (version 2.0.x).
    import ompl.base as ompl_base  # noqa: TC002
    import ompl.control as ompl_control  # noqa: TC002

    _OMPL_AVAILABLE = True
except ImportError as exc:
    _OMPL_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Configuration and diagnostics types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OmplSmokeConfig:
    """Configuration for the OMPL kinodynamic smoke diagnostic.

    Attributes:
        state_bounds: ``(min_x, max_x, min_y, max_y, min_theta, max_theta)``
            Planning space bounds. Theta is the heading in radians.
        control_bounds: ``(min_v, max_v, min_omega, max_omega)``
            Differential-drive command limits (linear speed, angular speed).
        robot_radius: Physical robot radius for validity checks (collision
            clearance from obstacles).
        dt: Integration timestep for the differential-drive propagator (s).
        max_planning_time_sec: Maximum time budget for OMPL planning (s).
        state_tolerance: Goal-region tolerance in the state space (m).
        seed: Optional random seed for reproducible planning. ``None`` means
            OMPL uses its internal random source.
        min_control_duration: Minimum number of propagation steps per control.
        max_control_duration: Maximum number of propagation steps per control.
    """

    state_bounds: tuple[float, float, float, float, float, float] = (
        0.0, 50.0, 0.0, 50.0, -3.1416, 3.1416
    )
    control_bounds: tuple[float, float, float, float] = (0.0, 1.5, -2.0, 2.0)
    robot_radius: float = 0.25
    dt: float = 0.1
    max_planning_time_sec: float = 5.0
    state_tolerance: float = 0.5
    seed: int | None = None
    min_control_duration: int = 1
    max_control_duration: int = 10


@dataclass
class OmplSmokeResult:
    """Result of an OMPL kinodynamic smoke diagnostic.

    Attributes:
        success: Whether a kinodynamically feasible path was found.
        path_length: Number of states in the returned path (0 if not found).
        path_states: List of ``(x, y, theta)`` states along the path.
        planning_time_sec: Wall-clock time used for planning.
        error: Error message if planning failed or OMPL is unavailable.
    """

    success: bool
    path_length: int
    path_states: list[tuple[float, float, float]]
    planning_time_sec: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Internal kinematics helper
# ---------------------------------------------------------------------------


def _differential_drive_propagate(
    state: np.ndarray,
    control: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Propagate a differential-drive state by one timestep.

    Args:
        state: ``[x, y, theta]`` current state.
        control: ``[v, omega]`` linear and angular velocity command.
        duration: Integration interval (s).

    Returns:
        ``[x_new, y_new, theta_new]`` next state.
    """
    x, y, theta = state
    v, omega = control
    x_new = x + v * np.cos(theta) * duration
    y_new = y + v * np.sin(theta) * duration
    theta_new = theta + omega * duration
    return np.array([x_new, y_new, theta_new], dtype=np.float64)


# ---------------------------------------------------------------------------
# Public diagnostic API
# ---------------------------------------------------------------------------


def check_ompl_available() -> tuple[bool, str | None]:
    """Return whether OMPL is importable and ready.

    Returns:
        ``(available, error_msg)`` where ``error_msg`` is ``None`` when
        OMPL is available.
    """
    return _OMPL_AVAILABLE, _OMPL_IMPORT_ERROR


def smoke_plan(  # noqa: C901, PLR0915
    start: tuple[float, float],
    goal: tuple[float, float],
    config: OmplSmokeConfig | None = None,
    obstacle_polygons: list[Any] | None = None,
) -> OmplSmokeResult:
    """Run an OMPL kinodynamic smoke diagnostic from ``start`` to ``goal``.

    This is a diagnostic-only function. It uses OMPL's control-based planners
    (RRT by default) to check if a kinodynamically feasible route exists
    between two 2D points under a differential-drive model.

    Args:
        start: ``(x, y)`` world coordinates.
        goal: ``(x, y)`` world coordinates.
        config: Planning configuration. Defaults to conservative values.
        obstacle_polygons: Optional list of Shapely Polygons to check
            against for collision validity.

    Returns:
        OmplSmokeResult with planning outcome.
    """
    cfg = config or OmplSmokeConfig()

    if not _OMPL_AVAILABLE:
        return OmplSmokeResult(
            success=False,
            path_length=0,
            path_states=[],
            planning_time_sec=0.0,
            error=(
                "OMPL not available"
                + (f": {_OMPL_IMPORT_ERROR}" if _OMPL_IMPORT_ERROR else "")
            ),
        )

    import ompl.base as ompl_base  # noqa: PLC0415
    import ompl.control as ompl_control  # noqa: PLC0415

    # Build the state space: RealVectorStateSpace(3) for (x, y, theta)
    state_space = ompl_base.RealVectorStateSpace(3)
    bounds = ompl_base.RealVectorBounds(3)
    bounds.setLow(0, cfg.state_bounds[0])
    bounds.setHigh(0, cfg.state_bounds[1])
    bounds.setLow(1, cfg.state_bounds[2])
    bounds.setHigh(1, cfg.state_bounds[3])
    bounds.setLow(2, cfg.state_bounds[4])
    bounds.setHigh(2, cfg.state_bounds[5])
    state_space.setBounds(bounds)

    # Build the control space: (v, omega)
    control_space = ompl_control.RealVectorControlSpace(state_space, 2)
    control_bounds = ompl_base.RealVectorBounds(2)
    control_bounds.setLow(0, cfg.control_bounds[0])
    control_bounds.setHigh(0, cfg.control_bounds[1])
    control_bounds.setLow(1, cfg.control_bounds[2])
    control_bounds.setHigh(1, cfg.control_bounds[3])
    control_space.setBounds(control_bounds)

    # Create SimpleSetup with the control space
    ss = ompl_control.SimpleSetup(control_space)

    # SpaceInformation for propagator and validity checker
    si = ss.getSpaceInformation()

    # Propagator: (state_in, ctrl, duration, state_out) — four arguments.
    def propagator(
        state: ompl_base.State,
        ctrl: ompl_control.Control,
        duration: float,
        result: ompl_base.State,
    ) -> None:
        x = state[0]
        y = state[1]
        theta = state[2]
        v = ctrl[0]
        omega = ctrl[1]
        result[0] = x + v * np.cos(theta) * duration
        result[1] = y + v * np.sin(theta) * duration
        result[2] = theta + omega * duration

    si.setStatePropagator(propagator)
    si.setMinMaxControlDuration(cfg.min_control_duration, cfg.max_control_duration)
    si.setPropagationStepSize(cfg.dt)

    # State validity checker: obstacle collision
    if obstacle_polygons:
        try:
            import shapely.geometry as sg  # noqa: PLC0415

            def is_state_valid(state: ompl_base.State) -> bool:
                point = sg.Point(state[0], state[1])
                for poly in obstacle_polygons:
                    if poly.buffer(cfg.robot_radius).contains(point):
                        return False
                return True

            si.setStateValidityChecker(is_state_valid)
        except ImportError:
            logger.warning("shapely not available; skipping obstacle collision checks")

    # Set start and goal states by allocating State objects
    start_state = state_space.allocState()
    start_state[0] = float(start[0])
    start_state[1] = float(start[1])
    start_state[2] = 0.0  # neutral heading

    goal_state = state_space.allocState()
    goal_state[0] = float(goal[0])
    goal_state[1] = float(goal[1])
    goal_state[2] = 0.0  # neutral heading

    ss.setStartAndGoalStates(start_state, goal_state, cfg.state_tolerance)

    # Setup and solve
    time_start = float(np.datetime64("now", "us").astype(np.int64)) / 1e6

    if cfg.seed is not None:
        ompl_base.RNG().setSeed(cfg.seed)

    planner = ompl_control.RRT(si)
    ss.setPlanner(planner)
    ss.setup()

    solved = ss.solve(cfg.max_planning_time_sec)
    time_end = float(np.datetime64("now", "us").astype(np.int64)) / 1e6
    planning_time = time_end - time_start

    if not solved:
        return OmplSmokeResult(
            success=False,
            path_length=0,
            path_states=[],
            planning_time_sec=planning_time,
            error="OMPL did not find a solution within the time budget",
        )

    # Extract path states using index-based access
    path = ss.getSolutionPath()
    states = path.getStates()
    path_states: list[tuple[float, float, float]] = []
    for s in states:
        path_states.append((float(s[0]), float(s[1]), float(s[2])))

    return OmplSmokeResult(
        success=True,
        path_length=len(path_states),
        path_states=path_states,
        planning_time_sec=planning_time,
    )


def compare_with_classic_route(
    ompl_result: OmplSmokeResult,
    classic_path: list[tuple[float, float]],
) -> dict[str, Any]:
    """Compare an OMPL result against a classic grid-based route.

    This computes basic diagnostics: path length difference, max lateral
    deviation, and whether the OMPL path is smoother.

    Args:
        ompl_result: Result from :func:`smoke_plan`.
        classic_path: List of ``(x, y)`` waypoints from a classic planner.

    Returns:
        Dict with comparison diagnostics.
    """
    if not ompl_result.success or not ompl_result.path_states:
        return {
            "comparison_possible": False,
            "reason": "OMPL did not produce a valid path",
        }

    if not classic_path:
        return {
            "comparison_possible": False,
            "reason": "Classic path is empty",
        }

    # Compute path lengths
    ompl_xy = np.array([s[:2] for s in ompl_result.path_states])
    classic_xy = np.array(classic_path)

    ompl_deltas = np.diff(ompl_xy, axis=0)
    classic_deltas = np.diff(classic_xy, axis=0)
    ompl_length = float(np.sum(np.linalg.norm(ompl_deltas, axis=1)))
    classic_length = float(np.sum(np.linalg.norm(classic_deltas, axis=1)))

    # Compute max lateral deviation of classic path from OMPL path
    # (simplified: project classic points onto OMPL segments)
    max_deviation = 0.0
    if len(ompl_xy) > 1:
        for point in classic_xy:
            min_dist = float("inf")
            for i in range(len(ompl_xy) - 1):
                seg_start = ompl_xy[i]
                seg_end = ompl_xy[i + 1]
                seg_vec = seg_end - seg_start
                seg_len = np.linalg.norm(seg_vec)
                if seg_len > 1e-9:
                    t = np.dot(point - seg_start, seg_vec) / (seg_len**2)
                    t = np.clip(t, 0.0, 1.0)
                    projection = seg_start + t * seg_vec
                    dist = np.linalg.norm(point - projection)
                    min_dist = min(min_dist, float(dist))
            max_deviation = max(max_deviation, min_dist)

    return {
        "comparison_possible": True,
        "ompl_path_steps": len(ompl_result.path_states),
        "classic_path_steps": len(classic_path),
        "ompl_length_m": round(ompl_length, 3),
        "classic_length_m": round(classic_length, 3),
        "max_lateral_deviation_m": round(max_deviation, 4),
    }
