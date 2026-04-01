"""Tests for environment metric helpers."""

import math

import pytest

from robot_sf.eval import EnvMetrics, EnvOutcome, PedEnvMetrics, PedVecEnvMetrics, VecEnvMetrics


def test_total_routes():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    assert metrics.total_routes == 1
    metrics.route_outcomes.append(EnvOutcome.REACHED_GOAL)
    assert metrics.total_routes == 1


def test_total_intermediate_goals():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    assert metrics.total_intermediate_goals == 1
    metrics.intermediate_goal_outcomes.append(EnvOutcome.REACHED_GOAL)
    assert metrics.total_intermediate_goals == 1


def test_pedestrian_collisions():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    assert metrics.pedestrian_collisions == 0
    metrics.route_outcomes.append(EnvOutcome.PEDESTRIAN_COLLISION)
    assert metrics.pedestrian_collisions == 1


def test_obstacle_collisions():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    assert metrics.obstacle_collisions == 0
    metrics.route_outcomes.append(EnvOutcome.OBSTACLE_COLLISION)
    assert metrics.obstacle_collisions == 1


def test_exceeded_timesteps():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    assert metrics.exceeded_timesteps == 0
    metrics.route_outcomes.append(EnvOutcome.TIMEOUT)
    assert metrics.exceeded_timesteps == 1


def test_completed_routes():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    assert metrics.completed_routes == 0
    metrics.route_outcomes.append(EnvOutcome.REACHED_GOAL)
    assert metrics.completed_routes == 1


def test_reached_intermediate_goals():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    assert metrics.reached_intermediate_goals == 0
    metrics.intermediate_goal_outcomes.append(EnvOutcome.REACHED_GOAL)
    assert metrics.reached_intermediate_goals == 1


def test_robot_collisions():
    """TODO docstring. Document this function."""
    metrics = PedEnvMetrics()
    assert metrics.robot_collisions == 0
    metrics.route_outcomes.append(EnvOutcome.ROBOT_COLLISION)
    assert metrics.robot_collisions == 1


def test_update():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    meta = {
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_waypoint_complete": True,
        "is_timesteps_exceeded": False,
        "is_route_complete": True,
    }
    metrics.update(meta)
    assert metrics.completed_routes == 1
    assert metrics.reached_intermediate_goals == 1


def test_on_next_intermediate_outcome():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    meta = {
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_waypoint_complete": True,
        "is_timesteps_exceeded": False,
    }
    metrics._on_next_intermediate_outcome(meta)
    assert metrics.reached_intermediate_goals == 1


def test_on_next_route_outcome():
    """TODO docstring. Document this function."""
    metrics = EnvMetrics()
    meta = {
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_route_complete": True,
        "is_timesteps_exceeded": False,
    }
    metrics._on_next_route_outcome(meta)
    assert metrics.completed_routes == 1


def test_ped_update():
    """TODO docstring. Document this function."""
    metrics = PedEnvMetrics()
    meta = {
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_timesteps_exceeded": True,
        "is_robot_collision": False,
        "distance_to_robot": 45.0,
        "ego_ped_speed": 0.4,
        "collision_impact_angle_rad": 0.0,
    }
    metrics.update(meta)
    assert metrics.exceeded_timesteps == 1
    assert metrics.route_end_distance == 45.0
    assert metrics.avg_ego_ped_speed == 0.0
    assert metrics.avg_collision_impact_angle_rad_at_collision == 0.0
    assert metrics.avg_collision_impact_angle_rad == 0.0


def test_env_rate_properties_and_vector_aggregation():
    """Vector aggregation should expose consistent route/intermediate rates."""
    m1 = EnvMetrics()
    m1.route_outcomes.extend([EnvOutcome.REACHED_GOAL, EnvOutcome.TIMEOUT])
    m1.intermediate_goal_outcomes.extend([EnvOutcome.REACHED_GOAL, EnvOutcome.OBSTACLE_COLLISION])
    m2 = EnvMetrics()
    m2.route_outcomes.extend([EnvOutcome.PEDESTRIAN_COLLISION, EnvOutcome.REACHED_GOAL])
    m2.intermediate_goal_outcomes.extend([EnvOutcome.REACHED_GOAL, EnvOutcome.REACHED_GOAL])

    vec = VecEnvMetrics(metrics=[m1, m2])
    assert 0.0 <= m1.route_completion_rate <= 1.0
    assert 0.0 <= m1.interm_goal_completion_rate <= 1.0
    assert 0.0 <= m1.timeout_rate <= 1.0
    assert 0.0 <= m1.obstacle_collision_rate <= 1.0
    assert 0.0 <= m1.pedestrian_collision_rate <= 1.0
    assert 0.0 <= vec.route_completion_rate <= 1.0
    assert 0.0 <= vec.interm_goal_completion_rate <= 1.0
    assert 0.0 <= vec.timeout_rate <= 1.0
    assert 0.0 <= vec.obstacle_collision_rate <= 1.0
    assert 0.0 <= vec.pedestrian_collision_rate <= 1.0

    before_m1 = len(m1.route_outcomes)
    before_m2 = len(m2.route_outcomes)
    vec.update(
        [
            {
                "is_pedestrian_collision": False,
                "is_obstacle_collision": False,
                "is_waypoint_complete": False,
                "is_timesteps_exceeded": True,
                "is_route_complete": False,
            },
            {
                "is_pedestrian_collision": True,
                "is_obstacle_collision": False,
                "is_waypoint_complete": False,
                "is_timesteps_exceeded": False,
                "is_route_complete": False,
            },
        ]
    )
    assert len(m1.route_outcomes) == before_m1 + 1
    assert len(m2.route_outcomes) == before_m2 + 1
    assert m1.route_outcomes[-1] == EnvOutcome.TIMEOUT
    assert m2.route_outcomes[-1] == EnvOutcome.PEDESTRIAN_COLLISION


def test_ped_outcome_priority_and_vector_rates():
    """Pedestrian metrics should prefer collision outcomes over completion flags."""
    ped = PedEnvMetrics()
    ped.update(
        {
            "distance_to_robot": 1.0,
            "ego_ped_speed": 1.25,
            "is_robot_collision": True,
            "is_route_complete": True,
            "is_timesteps_exceeded": False,
            "is_obstacle_collision": False,
            "is_pedestrian_collision": False,
            "is_robot_obstacle_collision": False,
            "is_robot_pedestrian_collision": False,
            "collision_impact_angle_rad": 0.2,
        }
    )
    ped.update(
        {
            "distance_to_robot": 2.0,
            "ego_ped_speed": 0.75,
            "is_robot_collision": False,
            "is_route_complete": True,
            "is_timesteps_exceeded": False,
            "is_obstacle_collision": False,
            "is_pedestrian_collision": False,
            "is_robot_obstacle_collision": False,
            "is_robot_pedestrian_collision": False,
            "collision_impact_angle_rad": 0.0,
        }
    )
    ped.update(
        {
            "distance_to_robot": 3.0,
            "ego_ped_speed": 0.25,
            "is_robot_collision": False,
            "is_route_complete": False,
            "is_timesteps_exceeded": False,
            "is_obstacle_collision": False,
            "is_pedestrian_collision": False,
            "is_robot_obstacle_collision": True,
            "is_robot_pedestrian_collision": False,
            "collision_impact_angle_rad": 0.4,
        }
    )
    ped.update(
        {
            "distance_to_robot": 0.5,
            "ego_ped_speed": 1.1,
            "is_robot_collision": True,
            "is_route_complete": False,
            "is_timesteps_exceeded": False,
            "is_obstacle_collision": False,
            "is_pedestrian_collision": False,
            "is_robot_obstacle_collision": False,
            "is_robot_pedestrian_collision": True,
            "collision_impact_angle_rad": 0.1,
        }
    )

    assert ped.robot_collisions == 1
    assert ped.robot_at_goal == 1
    assert ped.robot_obstacle_collisions == 1
    assert ped.robot_pedestrian_collisions == 1
    assert ped.route_outcomes[-1] == EnvOutcome.ROBOT_PEDESTRIAN_COLLISION
    assert list(ped.ego_ped_speed_at_collision) == [1.1]
    assert ped.avg_ego_ped_speed == 1.1
    assert ped.avg_collision_impact_angle_rad_at_collision == pytest.approx(0.1)
    assert ped.avg_collision_impact_angle_rad == pytest.approx(0.1)
    assert 0.0 <= ped.robot_collision_rate <= 1.0
    assert 0.0 <= ped.robot_at_goal_rate <= 1.0
    assert 0.0 <= ped.robot_obstacle_collision_rate <= 1.0
    assert 0.0 <= ped.robot_pedestrian_collision_rate <= 1.0
    assert ped.route_end_distance >= 0.0

    ped2 = PedEnvMetrics()
    vec = PedVecEnvMetrics(metrics=[ped, ped2])
    assert 0.0 <= vec.timeout_rate <= 1.0
    assert 0.0 <= vec.obstacle_collision_rate <= 1.0
    assert 0.0 <= vec.pedestrian_collision_rate <= 1.0
    assert 0.0 <= vec.robot_collision_rate <= 1.0
    assert 0.0 <= vec.robot_at_goal_rate <= 1.0
    assert 0.0 <= vec.robot_obstacle_collision_rate <= 1.0
    assert 0.0 <= vec.robot_pedestrian_collision_rate <= 1.0
    assert vec.route_end_distance >= 0.0
    assert 0.0 <= vec.avg_ego_ped_speed_at_collision
    assert vec.avg_ego_ped_speed == vec.avg_ego_ped_speed_at_collision
    before_p1 = len(ped.route_outcomes)
    before_p2 = len(ped2.route_outcomes)
    vec.update(
        [
            {
                "distance_to_robot": 0.5,
                "ego_ped_speed": 0.3,
                "is_robot_collision": False,
                "is_route_complete": False,
                "is_timesteps_exceeded": True,
                "is_obstacle_collision": False,
                "is_pedestrian_collision": False,
                "is_robot_obstacle_collision": False,
                "is_robot_pedestrian_collision": False,
                "collision_impact_angle_rad": 0.0,
            },
            {
                "distance_to_robot": 0.8,
                "ego_ped_speed": 0.1,
                "is_robot_collision": False,
                "is_route_complete": False,
                "is_timesteps_exceeded": True,
                "is_obstacle_collision": False,
                "is_pedestrian_collision": False,
                "is_robot_obstacle_collision": False,
                "is_robot_pedestrian_collision": False,
                "collision_impact_angle_rad": 0.0,
            },
        ]
    )
    assert len(ped.route_outcomes) == before_p1 + 1
    assert len(ped2.route_outcomes) == before_p2 + 1
    assert ped.route_outcomes[-1] == EnvOutcome.TIMEOUT
    assert ped2.route_outcomes[-1] == EnvOutcome.TIMEOUT


def test_ped_vec_env_metrics_weight_collision_sample_aggregates() -> None:
    """Collision-speed and impact-angle aggregates should weight actual samples, not env means."""
    ped1 = PedEnvMetrics()
    ped1.ego_ped_speed_at_collision.extend([1.0, 3.0])
    ped1.collision_impact_angle_rad_at_collision.extend([0.2, 0.4])

    ped2 = PedEnvMetrics()
    ped2.ego_ped_speed_at_collision.extend([10.0])
    ped2.collision_impact_angle_rad_at_collision.extend([1.0])

    vec = PedVecEnvMetrics(metrics=[ped1, ped2])

    assert math.isclose(vec.avg_ego_ped_speed_at_collision, 14.0 / 3.0)
    assert math.isclose(vec.avg_collision_impact_angle_rad_at_collision, 1.6 / 3.0)
