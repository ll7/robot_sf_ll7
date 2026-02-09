
import numpy as np
from metrics.cost_utils import *
from trajectory.trajectory import SystemConfig, Trajectory


def asym_gauss_from_vel(
    x: float,
    y: float,
    velx: float,
    vely: float,
    xc: float | None = 0,
    yc: float | None = 0,
) -> np.ndarray:
    """
    computation of the value of an arbitrarily rotated (by theta)
    centered at (xc, yc)
    Asymmetric Gaussian at some point (x, y)
    Obviously, the velocities are for the peds
    around whom the gaussian is centered
    Variances are:
    sig_theta: in direction of motion
    sig_r: opp direction of motion (rear)
    sig_s: variance to the sides

    can calculate sig_theta = max(2*velocity, 0.5) [Rachel Kirby thesis 2005?]
    """
    speed = np.sqrt(velx ** 2 + vely ** 2)
    theta = np.arctan2(vely, velx)
    sig_theta = vel2sig(speed)
    return asym_gauss(x, y, theta, sig_theta, xc=xc, yc=yc)


def _coerce_traj_with_heading(trajectory: Trajectory | np.ndarray) -> np.ndarray:
    if isinstance(trajectory, Trajectory):
        traj = trajectory.position_and_heading_nk3()
        if traj.ndim == 3:
            if traj.shape[0] != 1:
                raise ValueError("path metrics expect a single-trajectory batch.")
            traj = traj[0]
        return traj
    traj = np.asarray(trajectory)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    if traj.ndim == 3:
        if traj.shape[0] != 1:
            raise ValueError("path metrics expect a single-trajectory batch.")
        traj = traj[0]
    if traj.shape[-1] == 2:
        if traj.shape[0] < 2:
            headings = np.zeros((traj.shape[0],), dtype=float)
        else:
            diffs = traj[1:] - traj[:-1]
            headings = np.arctan2(diffs[:, 1], diffs[:, 0])
            headings = np.concatenate([headings, headings[-1:]], axis=0)
        traj = np.column_stack([traj, headings])
    return traj


def _coerce_traj_xy(trajectory: Trajectory | np.ndarray) -> np.ndarray:
    traj = _coerce_traj_with_heading(trajectory)
    return traj[:, :2]


def asym_gauss(
    x: float,
    y: float,
    theta: float | None = 0,
    sig_theta: float | None = 2,
    xc: float | None = 0,
    yc: float | None = 0,
) -> np.ndarray:
    """
    computation of the value of an arbitrarily rotated (by theta)
    centered at (xc, yc)
    Asymmetric Gaussian at some point (x, y)
    Variances are:
    sig_theta: in direction of motion
    sig_r: opp direction of motion (rear)
    sig_s: variance to the sides

    can calculate sig_theta = max(2*velocity, 0.5) [Rachel Kirby thesis 2005?]
    """
    alpha = np.arctan2(y - yc, x - xc) - theta + np.pi / 2
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    # print(alpha[np.where(alpha>np.pi)])
    # sigma = np.zeros_like(x)
    # sigma = sig_r if alpha <= 0 else sig_h

    sig_s = sig_theta / 4
    sig_r = sig_theta / 3
    sigma = np.where(alpha <= 0, sig_r, sig_theta)

    a = ((np.cos(theta) / sigma) ** 2 + (np.sin(theta) / sig_s) ** 2) / 2
    b = np.sin(2 * theta) * (1 / (sigma ** 2) - 1 / (sig_s ** 2)) / 4
    c = ((np.sin(theta) / sigma) ** 2 + (np.cos(theta) / sig_s) ** 2) / 2

    # gaussian
    agxy = np.exp(
        -(a * (x - xc) ** 2 + 2 * b * (x - xc) * (y - yc) + c * (y - yc) ** 2)
    )

    return agxy


def path_length(trajectory: Trajectory | np.ndarray) -> float:
    traj_xy = _coerce_traj_xy(trajectory)
    distance_sq = np.sum(np.power(np.diff(traj_xy, axis=0), 2), axis=1)
    distance = np.sum(np.sqrt(distance_sq))
    return distance


def path_length_ratio(
    trajectory: Trajectory | np.ndarray, goal_config: SystemConfig | np.ndarray | None = None
) -> float:
    """
    Returns displacement/distance -- displacement may be zero
    (Distance is an approximation based on the time resolution of stored trajectory)
    Higher should be better but also depends on your exact scenario
    """
    # TODO: make this run in batch mode

    traj_xy = _coerce_traj_xy(trajectory)

    # for incomplete trajectories you want to pass in the aspirational goal
    if goal_config is None:
        goal_xy = traj_xy[-1, :]
    else:
        goal_arr = _coerce_traj_xy(goal_config)
        goal_xy = goal_arr[-1, :]

    start_config = traj_xy[0, :]
    epsilon = 0.00001  # for numerical stability
    distance = path_length(traj_xy) + epsilon
    displacement = np.linalg.norm(goal_xy - start_config)
    return displacement / distance


def path_irregularity(
    trajectory: Trajectory | np.ndarray, goal_config: SystemConfig | np.ndarray | None = None
) -> float:
    """
    defined as the amount of unnecessary turning per unit path length performed by a robot,
    where unnecessary turning corresponds to the
    total amount of robot rotation minus the minimum amount of rotation
    which would be needed to reach the same targets
    with the most direct path. Path irregularity is measured in rad/m
    :return:
    """
    traj = _coerce_traj_with_heading(trajectory)
    if goal_config is None:
        goal_arr = traj[-1, :]
    else:
        goal_arr = _coerce_traj_with_heading(goal_config)[-1, :]
    assert traj.shape[-1] == 3 and goal_arr.shape[-1] == 3

    # To compute the per step angle away from straight line to goal
    # compute the ray to goal from each traj step
    traj_xy = traj[:, :-1]
    heading_vectors = np.column_stack([np.cos(traj[:, 2]), np.sin(traj[:, 2])])
    point_to_goal_traj = np.squeeze(goal_arr)[:-1] - traj_xy
    denom = np.linalg.norm(point_to_goal_traj, axis=1) + 1e-10
    cos_theta = np.sum(point_to_goal_traj * heading_vectors, axis=1) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_to_goal_traj = np.arccos(cos_theta)
    path_length_m = path_length(traj_xy) + 1e-10
    path_irr = np.sum(np.abs(theta_to_goal_traj)) / path_length_m

    return path_irr


# def time_to_collision(sim_state):
#     """
#
#     :param sim_state:
#     :return: least time to collision of ego agent to any agent
#     """
#     ttc=0
#     return ttc
