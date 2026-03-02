"""TODO docstring. Document this module."""

from robot_sf.gym_env.reward import simple_reward


def test_simple_reward():
    """TODO docstring. Document this function."""
    meta = {
        "step": 0,
        "episode": 0,
        "step_of_episode": 0,
        "is_pedestrian_collision": False,
        "is_robot_collision": False,
        "is_obstacle_collision": False,
        "is_waypoint_complete": True,
        "is_route_complete": True,
        "is_timesteps_exceeded": False,
        "max_sim_steps": 1000,
    }

    reward = simple_reward(meta)
    assert reward == 0.9999
