"""Tests for environment metric helpers."""

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
    }
    metrics.update(meta)
    assert metrics.exceeded_timesteps == 1
    assert metrics.route_end_distance == 45.0


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


def test_ped_outcome_priority_and_vector_rates():
    """Pedestrian metrics should prefer collision outcomes over completion flags."""
    ped = PedEnvMetrics()
    ped.update(
        {
            "distance_to_robot": 1.0,
            "is_robot_collision": True,
            "is_route_complete": True,
            "is_timesteps_exceeded": False,
            "is_obstacle_collision": False,
            "is_pedestrian_collision": False,
            "is_robot_obstacle_collision": False,
            "is_robot_pedestrian_collision": False,
        }
    )
    ped.update(
        {
            "distance_to_robot": 2.0,
            "is_robot_collision": False,
            "is_route_complete": True,
            "is_timesteps_exceeded": False,
            "is_obstacle_collision": False,
            "is_pedestrian_collision": False,
            "is_robot_obstacle_collision": False,
            "is_robot_pedestrian_collision": False,
        }
    )
    ped.update(
        {
            "distance_to_robot": 3.0,
            "is_robot_collision": False,
            "is_route_complete": False,
            "is_timesteps_exceeded": False,
            "is_obstacle_collision": False,
            "is_pedestrian_collision": False,
            "is_robot_obstacle_collision": True,
            "is_robot_pedestrian_collision": False,
        }
    )

    assert ped.robot_collisions == 1
    assert ped.robot_at_goal == 1
    assert ped.robot_obstacle_collisions == 1
    assert 0.0 <= ped.robot_collision_rate <= 1.0
    assert 0.0 <= ped.robot_at_goal_rate <= 1.0
    assert 0.0 <= ped.robot_obstacle_collision_rate <= 1.0
    assert 0.0 <= ped.robot_pedestrian_collision_rate <= 1.0
    assert ped.route_end_distance >= 0.0

    vec = PedVecEnvMetrics(metrics=[ped, PedEnvMetrics()])
    assert 0.0 <= vec.timeout_rate <= 1.0
    assert 0.0 <= vec.obstacle_collision_rate <= 1.0
    assert 0.0 <= vec.pedestrian_collision_rate <= 1.0
    assert 0.0 <= vec.robot_collision_rate <= 1.0
    assert 0.0 <= vec.robot_at_goal_rate <= 1.0
    assert 0.0 <= vec.robot_obstacle_collision_rate <= 1.0
    assert 0.0 <= vec.robot_pedestrian_collision_rate <= 1.0
    assert vec.route_end_distance >= 0.0
    vec.update(
        [
            {
                "distance_to_robot": 0.5,
                "is_robot_collision": False,
                "is_route_complete": False,
                "is_timesteps_exceeded": True,
                "is_obstacle_collision": False,
                "is_pedestrian_collision": False,
                "is_robot_obstacle_collision": False,
                "is_robot_pedestrian_collision": False,
            },
            {
                "distance_to_robot": 0.8,
                "is_robot_collision": False,
                "is_route_complete": False,
                "is_timesteps_exceeded": True,
                "is_obstacle_collision": False,
                "is_pedestrian_collision": False,
                "is_robot_obstacle_collision": False,
                "is_robot_pedestrian_collision": False,
            },
        ]
    )
