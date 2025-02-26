from robot_sf.gym_env.robot_env import simple_reward


def test_simple_reward():
    meta = {
        "step": 0,
        "episode": 0,
        "step_of_episode": 0,
        "is_pedestrian_collision": False,
        "is_robot_collision": False,
        "is_obstacle_collision": False,
        "is_robot_at_goal": True,
        "is_route_complete": False,
        "is_timesteps_exceeded": False,
        "max_sim_steps": 1000,
    }

    reward = simple_reward(meta)
    assert reward == 0.9999
