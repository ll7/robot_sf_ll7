def simple_reward(
        meta: dict,
        max_episode_step_discount: float=-0.1,
        ped_coll_penalty: float=-5,
        obst_coll_penalty: float=-2,
        reach_waypoint_reward: float=1) -> float:
    """
    Calculate the reward for the robot's current state.

    Parameters:
    meta (dict): Metadata containing information about the robot's current state.
    max_episode_step_discount (float): Discount factor for each step in the episode.
    ped_coll_penalty (float): Penalty for colliding with a pedestrian.
    obst_coll_penalty (float): Penalty for colliding with an obstacle.
    reach_waypoint_reward (float): Reward for reaching a waypoint.

    Returns:
    float: The calculated reward.
    """

    # Initialize reward with a discount based on the maximum simulation steps
    reward = max_episode_step_discount / meta["max_sim_steps"]

    # If there's a collision with a pedestrian or another robot, apply penalty
    if meta["is_pedestrian_collision"] or meta["is_robot_collision"]:
        reward += ped_coll_penalty

    # If there's a collision with an obstacle, apply penalty
    if meta["is_obstacle_collision"]:
        reward += obst_coll_penalty

    # If the robot has reached its goal, apply reward
    if meta["is_robot_at_goal"]:
        reward += reach_waypoint_reward

    return reward


def simple_ped_reward(meta: dict, max_episode_step_discount: float=-0.1,
                      ped_coll_penalty: float=-5,
                      obst_coll_penalty: float=-2,
                      robot_coll_reward: float=1,
                      robot_at_goal_penalty: float= -1) -> float:
    """
    Calculate the reward for the pedestrian's current state.

    Parameters:
    meta (dict): Metadata containing information about the pedestrian's current state.
    max_episode_step_discount (float): Discount factor for each step in the episode.

    Returns:
    float: The calculated reward.
    """

    # Initialize reward with a discount based on the maximum simulation steps

    reward = max_episode_step_discount / meta["max_sim_steps"]

    distance = meta["distance_to_robot"]
    reward += distance * -0.001

    # If there's a collision with a pedestrian or another robot, apply penalty
    if meta["is_pedestrian_collision"]:
        reward += ped_coll_penalty

    # If there's a collision with an obstacle, apply penalty
    if meta["is_obstacle_collision"]:
        reward += obst_coll_penalty

    # there's a collision with a robot, apply reward
    if meta["is_robot_collision"]:
        reward += robot_coll_reward

    if meta["is_robot_at_goal"]:
        reward += robot_at_goal_penalty

    return reward
