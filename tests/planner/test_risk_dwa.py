"""Contract and numeric-parity tests for the RiskDWA rollout vectorization.

These tests gate the behavior-changing vectorization required by issue #5412:
the vectorized rollout must reproduce the legacy step-by-step scalar loop
within an explicitly documented tolerance on a fixed seed/observation set.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, RiskDWAPlannerConfig


def _observation(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: float = 0.0,
    goal: tuple[float, float] = (3.0, 0.0),
    pedestrians: list[tuple[float, float]] | None = None,
    pedestrian_velocities: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a minimal structured observation accepted by RiskDWA."""
    positions = [] if pedestrians is None else pedestrians
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "angular_velocity": np.asarray([0.0], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(positions, dtype=float),
            "velocities": np.asarray(
                [] if pedestrian_velocities is None else pedestrian_velocities, dtype=float
            ),
            "count": np.asarray([len(positions)], dtype=float),
        },
    }


def _scalar_rollout_score(  # noqa: PLR0913
    planner: RiskDWAPlannerAdapter,
    *,
    robot_pos: np.ndarray,
    heading: float,
    goal: np.ndarray,
    command: tuple[float, float],
    ped_pos: np.ndarray,
    ped_vel: np.ndarray,
    observation: dict[str, object],
    current_speed: float,
    config: RiskDWAPlannerConfig,
) -> float:
    """Faithful scalar reimplementation of the legacy RiskDWA rollout loop.

    Used only as a numeric-parity reference; it mirrors the pre-vectorization
    step-by-step pose integration (accumulated orientation ``o += w*dt`` and
    per-step ``min`` over pedestrian and obstacle clearance) so the vectorized
    path can be gated against it.
    """
    from robot_sf.common.math_utils import wrap_angle_pi as _wa

    dt = float(config.rollout_dt)
    x = np.array(robot_pos, dtype=float)
    theta = float(heading)
    start_dist = float(np.linalg.norm(goal - x))
    min_ped_clear = float("inf")
    min_obs_clear = float("inf")
    steps = int(config.rollout_steps)
    for step in range(max(steps, 1)):
        t = (step + 1) * dt
        x = x + np.array(
            [
                float(command[0]) * np.cos(theta) * dt,
                float(command[0]) * np.sin(theta) * dt,
            ]
        )
        theta = _wa(theta + float(command[1]) * dt)
        if ped_pos.size > 0:
            ped_t = ped_pos + ped_vel * t
            ped_dist = np.linalg.norm(ped_t - x[None, :], axis=1)
            min_ped_clear = min(min_ped_clear, float(np.min(ped_dist)))
        min_obs_clear = min(min_obs_clear, planner._min_obstacle_clearance(x, observation))

    end_dist = float(np.linalg.norm(goal - x))
    progress = start_dist - end_dist
    goal_heading = float(np.arctan2(goal[1] - x[1], goal[0] - x[0]))
    heading_score = float(np.cos(_wa(goal_heading - theta)))
    smooth_penalty = abs(float(command[0]) - float(current_speed)) + 0.2 * abs(float(command[1]))

    ped_risk = max(0.0, float(config.near_distance) - min_ped_clear)
    obs_risk = max(0.0, float(config.near_distance) - min_obs_clear)
    ttc = planner._ttc_proxy(robot_pos, command, ped_pos, ped_vel, heading)
    ttc_term = 0.0 if not np.isfinite(ttc) else 1.0 / max(ttc, 1e-3)

    return (
        float(config.goal_progress_weight) * progress
        + float(config.heading_weight) * heading_score
        + float(config.ped_clearance_weight) * min(min_ped_clear, 2.0)
        + float(config.obstacle_clearance_weight) * min(min_obs_clear, 2.0)
        - 2.0 * ped_risk
        - 1.5 * obs_risk
        - float(config.ttc_weight) * ttc_term
        - float(config.smoothness_weight) * smooth_penalty
    )


def test_risk_dwa_vectorized_rollout_matches_scalar_reference() -> None:
    """The vectorized rollout must reproduce finite scalar scores for a fixed seed set.

    This is the numeric-parity gate required by issue #5412 for the
    behavior-changing RiskDWA vectorization: outputs must stay within 1e-9 of
    the legacy step-by-step loop on a fixed observation/command grid.
    """
    config = RiskDWAPlannerConfig(rollout_steps=20)
    rng = np.random.default_rng(5412)
    obs = _observation(
        robot=(0.0, 0.0),
        heading=0.0,
        goal=(5.0, 2.0),
        pedestrians=[tuple(p) for p in rng.uniform(-2.0, 2.0, size=(6, 2))],
        pedestrian_velocities=[tuple(v) for v in rng.uniform(-0.5, 0.5, size=(6, 2))],
    )
    planner = RiskDWAPlannerAdapter(config)
    robot_pos, heading, goal, ped_pos, ped_vel = planner._extract_robot_goal_ped(obs)
    current_speed = 0.0

    worst_finite_drift = 0.0
    for v in np.linspace(0.0, 1.2, 7):
        for w in np.linspace(-1.2, 1.2, 11):
            cmd = (float(v), float(w))
            vec = planner._rollout_score(
                robot_pos=robot_pos,
                heading=heading,
                goal=goal,
                command=cmd,
                ped_pos=ped_pos,
                ped_vel=ped_vel,
                observation=obs,
                current_speed=current_speed,
            )
            scalar = _scalar_rollout_score(
                planner,
                robot_pos=robot_pos,
                heading=heading,
                goal=goal,
                command=cmd,
                ped_pos=ped_pos,
                ped_vel=ped_vel,
                observation=obs,
                current_speed=current_speed,
                config=config,
            )
            if not np.isfinite(vec) and not np.isfinite(scalar):
                continue
            worst_finite_drift = max(worst_finite_drift, abs(vec - scalar))
            assert abs(vec - scalar) <= 1e-9, (
                f"score parity violated for {cmd}: vec={vec}, scalar={scalar}"
            )
    assert worst_finite_drift <= 1e-9


def test_risk_dwa_vectorized_rollout_parity_fixture() -> None:
    """Vectorized rollout preserves the scalar parity point for the issue fixture."""
    config = RiskDWAPlannerConfig(rollout_steps=15)
    obs = _observation(
        goal=(3.0, 0.0),
        pedestrians=[(0.6, 0.8)],
        pedestrian_velocities=[(0.0, -1.0)],
    )
    planner = RiskDWAPlannerAdapter(config)
    robot_pos, heading, goal, ped_pos, ped_vel = planner._extract_robot_goal_ped(obs)
    command = (0.5, 0.2)
    vec = planner._rollout_score(
        robot_pos=robot_pos,
        heading=heading,
        goal=goal,
        command=command,
        ped_pos=ped_pos,
        ped_vel=ped_vel,
        observation=obs,
        current_speed=0.0,
    )
    scalar = _scalar_rollout_score(
        planner,
        robot_pos=robot_pos,
        heading=heading,
        goal=goal,
        command=command,
        ped_pos=ped_pos,
        ped_vel=ped_vel,
        observation=obs,
        current_speed=0.0,
        config=config,
    )
    assert vec == pytest.approx(scalar, rel=0.0, abs=1e-9)


def test_risk_dwa_vectorized_rollout_prediction_parity_multiple_pedestrians() -> None:
    """Pedestrian clearance broadcasts over every active pedestrian."""
    config = RiskDWAPlannerConfig(rollout_steps=15)
    obs = _observation(
        goal=(3.0, 0.0),
        pedestrians=[(1.5, 1.5), (-1.5, 1.5), (1.5, -1.5)],
        pedestrian_velocities=[(0.0, -0.2), (0.2, 0.0), (-0.1, 0.1)],
    )
    planner = RiskDWAPlannerAdapter(config)
    robot_pos, heading, goal, ped_pos, ped_vel = planner._extract_robot_goal_ped(obs)
    command = (0.5, 0.2)
    vectorized = planner._rollout_score(
        robot_pos=robot_pos,
        heading=heading,
        goal=goal,
        command=command,
        ped_pos=ped_pos,
        ped_vel=ped_vel,
        observation=obs,
        current_speed=0.0,
    )
    scalar = _scalar_rollout_score(
        planner,
        robot_pos=robot_pos,
        heading=heading,
        goal=goal,
        command=command,
        ped_pos=ped_pos,
        ped_vel=ped_vel,
        observation=obs,
        current_speed=0.0,
        config=config,
    )
    assert vectorized == pytest.approx(scalar, rel=0.0, abs=1e-9)


def test_risk_dwa_vectorized_rollout_trajectory_matches_scalar() -> None:
    """The rollout trajectory matches the step-by-step scalar integration.

    The helper intentionally retains the legacy sequential heading and position
    recurrence; the planner score is gated separately by the parity tests above.
    """
    config = RiskDWAPlannerConfig()
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(10):
        robot_pos = rng.uniform(-5.0, 5.0, size=2)
        heading = float(rng.uniform(-np.pi, np.pi))
        command = (float(rng.uniform(0.0, 1.2)), float(rng.uniform(-1.2, 1.2)))
        steps = int(rng.integers(1, 30))
        traj, _ = RiskDWAPlannerAdapter(config)._rollout_trajectory(
            robot_pos=robot_pos, heading=heading, command=command, steps=steps
        )
        dt = float(config.rollout_dt)
        pos = np.array(robot_pos, dtype=float)
        theta = float(heading)
        for _ in range(steps):
            pos = pos + np.array([command[0] * np.cos(theta) * dt, command[0] * np.sin(theta) * dt])
            theta = wrap_angle_pi(theta + command[1] * dt)
        worst = max(worst, float(np.max(np.abs(traj[-1] - pos))))
    assert worst <= 1e-9


def test_risk_dwa_plan_picks_feasible_command() -> None:
    """Planning produces a goal-directed, dynamically bounded command."""
    config = RiskDWAPlannerConfig()
    obs = _observation(robot=(0.0, 0.0), heading=0.0, goal=(4.0, 0.0))
    planner = RiskDWAPlannerAdapter(config)
    command = planner.plan(obs)
    assert abs(command[0]) <= config.max_linear_speed + 1e-9
    assert abs(command[1]) <= config.max_angular_speed + 1e-9
    assert command[0] > 0.0


def test_risk_dwa_plan_stops_at_goal() -> None:
    """At the goal the planner returns the zero command."""
    config = RiskDWAPlannerConfig()
    obs = _observation(robot=(0.0, 0.0), heading=0.0, goal=(0.1, 0.0))
    planner = RiskDWAPlannerAdapter(config)
    assert planner.plan(obs) == (0.0, 0.0)
