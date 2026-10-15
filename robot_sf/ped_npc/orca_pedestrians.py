"""Native ORCA crowd stepping, independent of Social Force equations."""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_sf.common.optional_import import try_import


def require_native_orca() -> Any:
    """Return the existing rvo2 backend, failing explicitly without a fallback."""
    backend = try_import("rvo2")
    if backend is None:
        raise RuntimeError(
            "orca_pedestrians_v1 requires the native rvo2 backend; "
            "install the repository's third_party/python-rvo2 dependency"
        )
    return backend


def _validate_orca_inputs(state, speeds, walls, robots, *, dt, radius) -> None:
    """Reject malformed arrays before invoking the native extension."""
    if state.ndim != 2 or state.shape[1] < 6:
        raise ValueError("ORCA state must have shape (N, >=6)")
    if speeds.shape != (len(state),) or np.any(speeds < 0):
        raise ValueError("ORCA max_speeds must have shape (N,) and be non-negative")
    if walls.ndim != 2 or walls.shape[1] != 4:
        raise ValueError("ORCA obstacle_segments must have shape (M, 4)")
    if robots.ndim != 2 or robots.shape[1] != 5 or np.any(robots[:, 4] <= 0):
        raise ValueError("ORCA robot_discs must have shape (R, 5) and positive radii")
    if not all(np.isfinite(array).all() for array in (state, speeds, walls, robots)):
        raise ValueError("ORCA inputs must be finite")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("ORCA dt and radius must be finite and positive")


def step_orca_pedestrians(
    state: np.ndarray,
    max_speeds: np.ndarray,
    *,
    dt: float,
    radius: float,
    obstacle_segments: np.ndarray,
    robot_discs: np.ndarray,
) -> np.ndarray:
    """Advance positions/velocities using native ORCA and geometric goal preferences.

    Input rows retain the existing ``x,y,vx,vy,gx,gy,...`` storage format. Wall
    rows are ``x1,y1,x2,y2``; robot rows are ``x,y,vx,vy,radius``. Robot solver
    outputs are discarded, so ORCA never changes the robot's actual dynamics.
    Neighbor range is 10 m and both prediction horizons are 2.5 s. All actors
    within range can be neighbors. Each call rebuilds from current state, making
    resets, controller teleports and changing crowd counts explicit inputs.

    Returns:
        A state copy with ORCA positions/velocities and untouched goal/extra columns.
    """
    next_state = np.array(state, dtype=float, copy=True)
    speeds = np.asarray(max_speeds, dtype=float)
    walls = np.asarray(obstacle_segments, dtype=float)
    robots = np.asarray(robot_discs, dtype=float)
    _validate_orca_inputs(next_state, speeds, walls, robots, dt=dt, radius=radius)
    count = len(next_state)
    backend = require_native_orca()
    if count == 0:
        return next_state

    solver = backend.PyRVOSimulator(
        dt, 10.0, max(count + len(robots) - 1, 1), 2.5, 2.5, radius, 1.0
    )
    for wall in walls:
        if np.linalg.norm(wall[:2] - wall[2:]) > 1e-9:
            solver.addObstacle([tuple(wall[:2]), tuple(wall[2:])])
    solver.processObstacles()
    pedestrian_ids = []
    for row, speed in zip(next_state, speeds, strict=True):
        actor = solver.addAgent(tuple(row[:2]))
        pedestrian_ids.append(actor)
        solver.setAgentVelocity(actor, tuple(row[2:4]))
        solver.setAgentMaxSpeed(actor, float(speed))
        to_goal = row[4:6] - row[:2]
        distance = float(np.linalg.norm(to_goal))
        preference = (
            to_goal * (min(float(speed), distance / dt) / distance) if distance else to_goal
        )
        solver.setAgentPrefVelocity(actor, tuple(preference))
    for robot in robots:
        actor = solver.addAgent(tuple(robot[:2]))
        solver.setAgentRadius(actor, float(robot[4]))
        solver.setAgentMaxSpeed(actor, float(np.linalg.norm(robot[2:4])))
        solver.setAgentVelocity(actor, tuple(robot[2:4]))
        solver.setAgentPrefVelocity(actor, tuple(robot[2:4]))
        solver.setAgentCollabCoeff(actor, 0.0)
    solver.doStep()
    for row, actor in zip(next_state, pedestrian_ids, strict=True):
        row[:2] = solver.getAgentPosition(actor)
        row[2:4] = solver.getAgentVelocity(actor)
    if not np.isfinite(next_state).all():
        raise RuntimeError("Native ORCA returned non-finite pedestrian state")
    return next_state
