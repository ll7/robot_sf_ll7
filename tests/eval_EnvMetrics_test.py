"""
test robot_sf.eval.EnvMetrics
"""
from robot_sf.eval import EnvMetrics, EnvOutcome

def test_total_routes():
    metrics = EnvMetrics()
    assert metrics.total_routes == 1
    metrics.route_outcomes.append(EnvOutcome.REACHED_GOAL)
    assert metrics.total_routes == 1

def test_total_intermediate_goals():
    metrics = EnvMetrics()
    assert metrics.total_intermediate_goals == 1
    metrics.intermediate_goal_outcomes.append(EnvOutcome.REACHED_GOAL)
    assert metrics.total_intermediate_goals == 1

def test_pedestrian_collisions():
    metrics = EnvMetrics()
    assert metrics.pedestrian_collisions == 0
    metrics.route_outcomes.append(EnvOutcome.PEDESTRIAN_COLLISION)
    assert metrics.pedestrian_collisions == 1

def test_obstacle_collisions():
    metrics = EnvMetrics()
    assert metrics.obstacle_collisions == 0
    metrics.route_outcomes.append(EnvOutcome.OBSTACLE_COLLISION)
    assert metrics.obstacle_collisions == 1

def test_exceeded_timesteps():
    metrics = EnvMetrics()
    assert metrics.exceeded_timesteps == 0
    metrics.route_outcomes.append(EnvOutcome.TIMEOUT)
    assert metrics.exceeded_timesteps == 1

def test_completed_routes():
    metrics = EnvMetrics()
    assert metrics.completed_routes == 0
    metrics.route_outcomes.append(EnvOutcome.REACHED_GOAL)
    assert metrics.completed_routes == 1

def test_reached_intermediate_goals():
    metrics = EnvMetrics()
    assert metrics.reached_intermediate_goals == 0
    metrics.intermediate_goal_outcomes.append(EnvOutcome.REACHED_GOAL)
    assert metrics.reached_intermediate_goals == 1

def test_update():
    metrics = EnvMetrics()
    meta = {
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_robot_at_goal": True,
        "is_timesteps_exceeded": False,
        "is_route_complete": True
    }
    metrics.update(meta)
    assert metrics.completed_routes == 1
    assert metrics.reached_intermediate_goals == 1

def test_on_next_intermediate_outcome():
    metrics = EnvMetrics()
    meta = {
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_robot_at_goal": True,
        "is_timesteps_exceeded": False
    }
    metrics._on_next_intermediate_outcome(meta)
    assert metrics.reached_intermediate_goals == 1

def test_on_next_route_outcome():
    metrics = EnvMetrics()
    meta = {
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_route_complete": True,
        "is_timesteps_exceeded": False
    }
    metrics._on_next_route_outcome(meta)
    assert metrics.completed_routes == 1