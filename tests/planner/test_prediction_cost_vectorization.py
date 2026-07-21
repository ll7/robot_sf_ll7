"""Contract and numeric-parity tests for PredictionPlannerAdapter per-timestep cost vectorization.

These tests gate the behavior-changing vectorization required by issue #5412:
the vectorized per-timestep cost loops must reproduce the legacy step-by-step
scalar loop within an explicitly documented tolerance on a fixed seed/observation set.
"""

from __future__ import annotations

import numpy as np

from robot_sf.planner.socnav import PredictionPlannerAdapter, SocNavPlannerConfig


def _build_observation(
    *,
    robot_pos: tuple[float, float] = (0.0, 0.0),
    robot_heading: float = 0.0,
    goal: tuple[float, float] = (3.0, 0.0),
    pedestrians: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a minimal structured observation for PredictionPlannerAdapter."""
    positions = [] if pedestrians is None else pedestrians
    vels = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot_pos, dtype=float),
            "heading": np.asarray([robot_heading], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(positions, dtype=float),
            "velocities": np.asarray(vels, dtype=float),
            "count": np.asarray([len(positions)], dtype=float),
        },
    }


def _scalar_collision_cost(
    planner: PredictionPlannerAdapter,
    *,
    future_peds: np.ndarray,
    mask: np.ndarray,
    v: float,
    w: float,
    steps: int,
    valid_dists: np.ndarray | None,
) -> tuple[float, float]:
    """Faithful scalar reference for _collision_cost before vectorization."""
    steps_val = steps
    radius_margin = float(planner.config.predictive_robot_radius) + float(
        planner.config.predictive_pedestrian_radius
    )
    speed_margin = float(planner.config.predictive_speed_clearance_gain) * abs(v)
    safe_dist = float(planner.config.predictive_safe_distance) + radius_margin + speed_margin
    near_dist = max(
        float(planner.config.predictive_near_distance) + radius_margin + speed_margin, safe_dist
    )
    collisions = 0.0
    near_misses = 0.0

    if valid_dists is not None:
        limit = min(steps_val, future_peds.shape[1], valid_dists.shape[1])
        for t in range(limit):
            valid_dist = valid_dists[:, t]
            if valid_dist.size == 0:
                continue
            collisions += float(np.sum(np.maximum(0.0, safe_dist - valid_dist)))
            near_misses += float(np.sum(np.maximum(0.0, near_dist - valid_dist)))
    else:
        dt = max(float(planner.config.predictive_rollout_dt), 1e-3)
        robot_traj = planner._rollout_robot(v=v, w=w, dt=dt, steps=steps_val)
        for t in range(min(steps_val, future_peds.shape[1])):
            delta = future_peds[:, t, :] - robot_traj[t].reshape(1, 2)
            dist = np.linalg.norm(delta, axis=1)
            valid_dist = dist[mask > 0.5]
            if valid_dist.size == 0:
                continue
            collisions += float(np.sum(np.maximum(0.0, safe_dist - valid_dist)))
            near_misses += float(np.sum(np.maximum(0.0, near_dist - valid_dist)))
    return collisions, near_misses


def _scalar_ttc_penalty(
    planner: PredictionPlannerAdapter,
    *,
    future_peds: np.ndarray,
    mask: np.ndarray,
    v: float,
    w: float,
    steps: int,
    valid_dists: np.ndarray | None,
) -> float:
    """Faithful scalar reference for _ttc_penalty before vectorization."""
    radius_margin = float(planner.config.predictive_robot_radius) + float(
        planner.config.predictive_pedestrian_radius
    )
    speed_margin = float(planner.config.predictive_speed_clearance_gain) * abs(v)
    threshold = float(planner.config.predictive_ttc_distance) + radius_margin + speed_margin
    if threshold <= 0.0:
        return 0.0
    dt = max(float(planner.config.predictive_rollout_dt), 1e-3)
    steps_val = steps
    penalty = 0.0

    if valid_dists is not None:
        limit = min(steps_val, future_peds.shape[1], valid_dists.shape[1])
        for t in range(limit):
            valid_dist = valid_dists[:, t]
            if valid_dist.size == 0:
                continue
            shortfall = np.maximum(0.0, threshold - valid_dist)
            time_weight = 1.0 / (float(t + 1) * dt + 1e-6)
            penalty += float(np.sum(shortfall * time_weight))
    else:
        robot_traj = planner._rollout_robot(v=v, w=w, dt=dt, steps=steps_val)
        for t in range(min(steps_val, future_peds.shape[1])):
            delta = future_peds[:, t, :] - robot_traj[t].reshape(1, 2)
            dist = np.linalg.norm(delta, axis=1)
            valid_dist = dist[mask > 0.5]
            if valid_dist.size == 0:
                continue
            shortfall = np.maximum(0.0, threshold - valid_dist)
            time_weight = 1.0 / (float(t + 1) * dt + 1e-6)
            penalty += float(np.sum(shortfall * time_weight))
    return penalty


def test_collision_cost_vectorized_matches_scalar() -> None:
    """Vectorized _collision_cost must reproduce scalar loop scores within 1e-9."""
    config = SocNavPlannerConfig(
        predictive_horizon_steps=20,
        predictive_rollout_dt=0.2,
        predictive_robot_radius=0.3,
        predictive_pedestrian_radius=0.2,
        predictive_safe_distance=0.5,
        predictive_near_distance=0.8,
        predictive_speed_clearance_gain=0.1,
    )
    planner = PredictionPlannerAdapter(config=config)

    rng = np.random.default_rng(5412)
    future_peds = rng.uniform(-2.0, 2.0, size=(6, 20, 2))
    mask = np.ones((6,), dtype=float)

    worst_drift = 0.0
    for v in np.linspace(0.0, 1.2, 5):
        for w in np.linspace(-1.0, 1.0, 5):
            vec_coll, vec_near = planner._collision_cost(
                future_peds=future_peds,
                mask=mask,
                v=float(v),
                w=float(w),
                steps=20,
                valid_dists=None,
            )
            scalar_coll, scalar_near = _scalar_collision_cost(
                planner,
                future_peds=future_peds,
                mask=mask,
                v=float(v),
                w=float(w),
                steps=20,
                valid_dists=None,
            )
            drift_coll = abs(vec_coll - scalar_coll)
            drift_near = abs(vec_near - scalar_near)
            worst_drift = max(worst_drift, drift_coll, drift_near)
            assert drift_coll <= 1e-9, f"collision drift: vec={vec_coll}, scalar={scalar_coll}"
            assert drift_near <= 1e-9, f"near_miss drift: vec={vec_near}, scalar={scalar_near}"
    assert worst_drift <= 1e-9


def test_collision_cost_with_valid_dists_parity() -> None:
    """Vectorized _collision_cost with valid_dists matches scalar reference."""
    config = SocNavPlannerConfig(
        predictive_horizon_steps=15,
        predictive_robot_radius=0.3,
        predictive_pedestrian_radius=0.2,
        predictive_safe_distance=0.5,
        predictive_near_distance=0.8,
        predictive_speed_clearance_gain=0.1,
    )
    planner = PredictionPlannerAdapter(config=config)

    rng = np.random.default_rng(7)
    future_peds = rng.uniform(-2.0, 2.0, size=(5, 15, 2))
    valid_dists = rng.uniform(0.1, 1.5, size=(5, 15))
    mask = np.ones((5,), dtype=float)

    vec_coll, vec_near = planner._collision_cost(
        future_peds=future_peds,
        mask=mask,
        v=0.5,
        w=0.2,
        steps=15,
        valid_dists=valid_dists,
    )
    scalar_coll, scalar_near = _scalar_collision_cost(
        planner,
        future_peds=future_peds,
        mask=mask,
        v=0.5,
        w=0.2,
        steps=15,
        valid_dists=valid_dists,
    )
    assert abs(vec_coll - scalar_coll) <= 1e-9
    assert abs(vec_near - scalar_near) <= 1e-9


def test_ttc_penalty_vectorized_matches_scalar() -> None:
    """Vectorized _ttc_penalty must reproduce scalar loop scores within 1e-9."""
    config = SocNavPlannerConfig(
        predictive_horizon_steps=20,
        predictive_rollout_dt=0.2,
        predictive_robot_radius=0.3,
        predictive_pedestrian_radius=0.2,
        predictive_ttc_distance=0.6,
        predictive_speed_clearance_gain=0.1,
    )
    planner = PredictionPlannerAdapter(config=config)

    rng = np.random.default_rng(5413)
    future_peds = rng.uniform(-2.0, 2.0, size=(6, 20, 2))
    mask = np.ones((6,), dtype=float)

    worst_drift = 0.0
    for v in np.linspace(0.0, 1.2, 5):
        for w in np.linspace(-1.0, 1.0, 5):
            vec = planner._ttc_penalty(
                future_peds=future_peds,
                mask=mask,
                v=float(v),
                w=float(w),
                steps=20,
                valid_dists=None,
            )
            scalar = _scalar_ttc_penalty(
                planner,
                future_peds=future_peds,
                mask=mask,
                v=float(v),
                w=float(w),
                steps=20,
                valid_dists=None,
            )
            drift = abs(vec - scalar)
            worst_drift = max(worst_drift, drift)
            assert drift <= 1e-9, f"TTC drift: vec={vec}, scalar={scalar}"
    assert worst_drift <= 1e-9


def test_ttc_penalty_with_valid_dists_parity() -> None:
    """Vectorized _ttc_penalty with valid_dists matches scalar reference."""
    config = SocNavPlannerConfig(
        predictive_horizon_steps=15,
        predictive_robot_radius=0.3,
        predictive_pedestrian_radius=0.2,
        predictive_ttc_distance=0.6,
        predictive_speed_clearance_gain=0.1,
    )
    planner = PredictionPlannerAdapter(config=config)

    rng = np.random.default_rng(8)
    future_peds = rng.uniform(-2.0, 2.0, size=(5, 15, 2))
    valid_dists = rng.uniform(0.1, 1.5, size=(5, 15))
    mask = np.ones((5,), dtype=float)

    vec = planner._ttc_penalty(
        future_peds=future_peds,
        mask=mask,
        v=0.5,
        w=0.2,
        steps=15,
        valid_dists=valid_dists,
    )
    scalar = _scalar_ttc_penalty(
        planner,
        future_peds=future_peds,
        mask=mask,
        v=0.5,
        w=0.2,
        steps=15,
        valid_dists=valid_dists,
    )
    assert abs(vec - scalar) <= 1e-9


def test_score_action_sequence_vectorized_parity() -> None:
    """Vectorized _score_action_sequence per-timestep loop matches scalar."""
    config = SocNavPlannerConfig(
        predictive_horizon_steps=12,
        predictive_rollout_dt=0.2,
        predictive_robot_radius=0.3,
        predictive_pedestrian_radius=0.2,
        predictive_safe_distance=0.5,
        predictive_near_distance=0.8,
        predictive_ttc_distance=0.6,
        predictive_speed_clearance_gain=0.1,
        predictive_goal_weight=1.0,
        predictive_collision_weight=2.0,
        predictive_near_miss_weight=1.0,
        predictive_ttc_weight=0.5,
        predictive_velocity_weight=0.1,
        predictive_turn_weight=0.1,
        predictive_progress_risk_weight=0.0,
        predictive_hard_clearance_weight=0.0,
        occupancy_weight=0.0,
        predictive_phase_logic_enabled=False,
    )
    planner = PredictionPlannerAdapter(config=config)

    rng = np.random.default_rng(5414)
    future_peds = rng.uniform(-2.0, 2.0, size=(4, 12, 2))
    mask = np.ones((4,), dtype=float)
    obs = _build_observation(
        robot_pos=(0.0, 0.0),
        robot_heading=0.0,
        goal=(3.0, 0.0),
        pedestrians=[(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5)],
        ped_velocities=[(0.1, 0.0), (-0.1, 0.0), (0.0, 0.1)],
    )

    sequence = [(0.5, 0.2), (0.5, 0.1), (0.4, 0.0)]

    vec = planner._score_action_sequence(
        observation=obs,
        future_peds=future_peds,
        mask=mask,
        sequence=sequence,
        steps=12,
    )

    def _scalar_score_action_sequence(
        p: PredictionPlannerAdapter,
        *,
        observation: dict,
        future_peds: np.ndarray,
        mask: np.ndarray,
        sequence: list[tuple[float, float]],
        steps: int,
    ) -> float:
        """Scalar reference for _score_action_sequence per-timestep loop."""
        robot_state, goal_state, _ = p._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(p._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2]
        dt = max(float(p.config.predictive_rollout_dt), 1e-3)
        segments = max(1, len(sequence))
        segment_steps = max(1, int(np.ceil(max(1, steps) / segments)))
        local_traj, _local_headings = p._rollout_robot_sequence(
            sequence=sequence,
            segment_steps=segment_steps,
            dt=dt,
        )
        horizon = min(local_traj.shape[0], int(steps), int(future_peds.shape[1]))
        local_traj = local_traj[:horizon]

        radius_margin = float(p.config.predictive_robot_radius) + float(
            p.config.predictive_pedestrian_radius
        )
        min_clearance = float("inf")
        collision_pen = 0.0
        near_pen = 0.0
        ttc_pen = 0.0
        safe_dist = float(p.config.predictive_safe_distance) + radius_margin
        near_dist = max(float(p.config.predictive_near_distance) + radius_margin, safe_dist)
        ttc_threshold = float(p.config.predictive_ttc_distance) + radius_margin

        for t in range(horizon):
            delta = future_peds[:, t, :] - local_traj[t].reshape(1, 2)
            dist = np.linalg.norm(delta, axis=1)
            valid_dist = dist[mask > 0.5]
            if valid_dist.size == 0:
                continue
            min_clearance = min(min_clearance, float(np.min(valid_dist)))
            collision_pen += float(np.sum(np.maximum(0.0, safe_dist - valid_dist)))
            near_pen += float(np.sum(np.maximum(0.0, near_dist - valid_dist)))
            shortfall = np.maximum(0.0, ttc_threshold - valid_dist)
            ttc_pen += float(np.sum(shortfall / (float(t + 1) * dt + p._EPS)))

        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        final_local = local_traj[-1] if horizon > 0 else np.zeros(2, dtype=float)
        final_world = robot_pos + np.array(
            [
                cos_h * final_local[0] - sin_h * final_local[1],
                sin_h * final_local[0] + cos_h * final_local[1],
            ],
            dtype=float,
        )
        initial_dist = float(np.linalg.norm(goal - robot_pos))
        final_dist = float(np.linalg.norm(goal - final_world))
        goal_progress = initial_dist - final_dist

        direction = final_world - robot_pos
        if np.linalg.norm(direction) <= p._EPS:
            direction = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)

        velocity_pen = float(sum(abs(v) for v, _ in sequence))
        turn_pen = float(sum(abs(w) for _, w in sequence))
        progress_risk_shortfall = max(
            0.0, float(p.config.predictive_progress_risk_distance) - float(min_clearance)
        )
        progress_risk_pen = max(0.0, goal_progress) * progress_risk_shortfall
        hard_clearance_shortfall = max(
            0.0, float(p.config.predictive_hard_clearance_distance) - float(min_clearance)
        )

        return (
            -float(p.config.predictive_goal_weight) * goal_progress
            + float(p.config.predictive_collision_weight) * collision_pen
            + float(p.config.predictive_near_miss_weight) * near_pen
            + float(p.config.predictive_progress_risk_weight) * progress_risk_pen
            + float(p.config.predictive_hard_clearance_weight) * hard_clearance_shortfall
            + float(p.config.predictive_velocity_weight) * velocity_pen
            + float(p.config.predictive_turn_weight) * turn_pen
            + float(p.config.predictive_ttc_weight) * ttc_pen
            + float(p.config.occupancy_weight) * 0.0
        )

    scalar = _scalar_score_action_sequence(
        planner,
        observation=obs,
        future_peds=future_peds,
        mask=mask,
        sequence=sequence,
        steps=12,
    )
    assert abs(vec - scalar) <= 1e-9
