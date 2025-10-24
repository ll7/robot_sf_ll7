import numpy as np

from robot_sf.benchmark.metrics import (
    EpisodeData,
    agent_collisions,
    human_collisions,
    success_rate,
    wall_collisions,
)


def make_base_episode(T=5, with_peds=True):
    robot_pos = np.zeros((T, 2))
    robot_vel = np.zeros((T, 2))
    robot_acc = np.zeros((T, 2))
    if with_peds:
        # place a single pedestrian far away so it doesn't collide in default tests
        peds_pos = np.zeros((T, 1, 2)) + np.array([100.0, 100.0])
    else:
        peds_pos = np.zeros((T, 0, 2))
    ped_forces = np.zeros((T, max(1, peds_pos.shape[1]), 2))
    goal = np.array([10.0, 0.0])
    dt = 0.1
    return robot_pos, robot_vel, robot_acc, peds_pos, ped_forces, goal, dt


def test_success_without_collisions_reaches_goal():
    T = 10
    robot_pos, robot_vel, robot_acc, peds_pos, ped_forces, goal, dt = make_base_episode(T)
    # place robot at goal at step 3
    robot_pos[:4, 0] = np.linspace(0.0, 10.0, 4)
    reached_step = 3
    data = EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=goal,
        dt=dt,
        reached_goal_step=reached_step,
        obstacles=None,
        other_agents_pos=None,
    )
    assert success_rate(data, horizon=T) == 1.0


def test_failure_on_wall_collision():
    T = 10
    robot_pos, robot_vel, robot_acc, peds_pos, ped_forces, goal, dt = make_base_episode(
        T, with_peds=False
    )
    robot_pos[:4, 0] = np.linspace(0.0, 10.0, 4)
    reached_step = 3
    # single obstacle very close to robot path
    obstacles = np.array([[0.0, 0.0]])
    data = EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=goal,
        dt=dt,
        reached_goal_step=reached_step,
        obstacles=obstacles,
        other_agents_pos=None,
    )
    # sanity: wall_collisions should be > 0
    assert wall_collisions(data) > 0
    assert success_rate(data, horizon=T) == 0.0


def test_failure_on_agent_collision():
    T = 10
    robot_pos, robot_vel, robot_acc, peds_pos, ped_forces, goal, dt = make_base_episode(
        T, with_peds=False
    )
    robot_pos[:4, 0] = np.linspace(0.0, 10.0, 4)
    reached_step = 3
    # other agent occupying same position as robot at step 2
    other_agents_pos = np.zeros((T, 1, 2))
    other_agents_pos[2] = robot_pos[2]
    data = EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=goal,
        dt=dt,
        reached_goal_step=reached_step,
        obstacles=None,
        other_agents_pos=other_agents_pos,
    )
    assert agent_collisions(data) > 0
    assert success_rate(data, horizon=T) == 0.0


def test_failure_on_human_collision():
    T = 10
    robot_pos, robot_vel, robot_acc, peds_pos, ped_forces, goal, dt = make_base_episode(T)
    robot_pos[:4, 0] = np.linspace(0.0, 10.0, 4)
    reached_step = 3
    # place ped at robot position at step 2
    peds_pos[2, 0] = robot_pos[2]
    data = EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=goal,
        dt=dt,
        reached_goal_step=reached_step,
        obstacles=None,
        other_agents_pos=None,
    )
    assert human_collisions(data) > 0
    assert success_rate(data, horizon=T) == 0.0
