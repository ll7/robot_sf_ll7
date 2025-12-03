"""
This module defines the reward function for the robot environment.
"""

import numpy as np


def simple_reward(
    meta: dict,
    max_episode_step_discount: float = -0.1,
    ped_coll_penalty: float = -5,
    obst_coll_penalty: float = -2,
    reach_waypoint_reward: float = 1,
) -> float:
    """Calculate the reward for the robot's current state.

    Args:
        meta: Metadata dictionary (see ``RobotState.meta_dict``).
        max_episode_step_discount: Per-step discount divided by ``max_sim_steps``.
        ped_coll_penalty: Penalty applied when colliding with pedestrians/robots.
        obst_coll_penalty: Penalty applied when colliding with obstacles.
        reach_waypoint_reward: Reward granted when the robot reaches its goal.

    Returns:
        float: Scalar reward for the timestep.
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

    return float(reward)


def simple_ped_reward(
    meta: dict,
    max_episode_step_discount: float = -0.1,
    ped_coll_penalty: float = -5,
    obst_coll_penalty: float = -5,
    robot_coll_reward: float = 5,
    robot_at_goal_penalty: float = -1,
) -> float:
    """Calculate the reward for the pedestrian's current state.

    Args:
        meta: Metadata dictionary describing collisions, goal status, and distances.
        max_episode_step_discount: Per-step discount divided by ``max_sim_steps``.
        ped_coll_penalty: Penalty applied when the ego pedestrian collides with others.
        obst_coll_penalty: Penalty applied when colliding with obstacles.
        robot_coll_reward: Bonus granted when colliding with the robot.
        robot_at_goal_penalty: Penalty applied if the robot reaches its goal.

    Returns:
        float: Scalar reward for the timestep.
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

    # If the robot has reached its goal, apply penalty
    if meta["is_robot_at_goal"]:
        reward += robot_at_goal_penalty

    return float(reward)


def punish_action_reward(
    meta: dict,
    max_episode_step_discount: float = -0.1,
    ped_coll_penalty: float = -5,
    obst_coll_penalty: float = -2,
    reach_waypoint_reward: float = 1,
    punish_action: bool = True,
    punish_action_penalty: float = -0.1,
) -> float:
    """Robot reward variant that penalizes action changes.

    Args:
        meta: Metadata dictionary describing the current state/action.
        max_episode_step_discount: Per-step discount divided by ``max_sim_steps``.
        ped_coll_penalty: Penalty applied when colliding with pedestrians/robots.
        obst_coll_penalty: Penalty applied when colliding with obstacles.
        reach_waypoint_reward: Reward granted when the robot reaches its goal.
        punish_action: Whether to penalize deviations from the previous action.
        punish_action_penalty: Scaling factor for the action difference penalty.

    Returns:
        float: Scalar reward for the timestep.
    """

    # Initialize reward with a discount based on the maximum simulation steps
    reward = simple_reward(
        meta,
        max_episode_step_discount,
        ped_coll_penalty,
        obst_coll_penalty,
        reach_waypoint_reward,
    )

    # punish the robot taking a different action from the last action
    if punish_action and meta["last_action"] is not None:
        action_diff = np.linalg.norm(np.array(meta["action"]) - np.array(meta["last_action"]))
        if action_diff > 0:
            reward += punish_action_penalty * action_diff

    return float(reward)
