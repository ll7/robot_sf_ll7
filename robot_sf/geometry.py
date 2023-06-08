# pylint: disable=too-many-locals
from typing import Tuple

import numpy as np
import numba


Vec2D = Tuple[float, float]
Line2D = Tuple[Vec2D, Vec2D]
Circle2D = Tuple[Vec2D, float]


@numba.njit(fastmath=True)
def euclid_dist(vec_1: Vec2D, vec_2: Vec2D) -> float:
    return ((vec_1[0] - vec_2[0])**2 + (vec_1[1] - vec_2[1])**2)**0.5


@numba.njit(fastmath=True)
def euclid_dist_sq(vec_1: Vec2D, vec_2: Vec2D) -> float:
    return (vec_1[0] - vec_2[0])**2 + (vec_1[1] - vec_2[1])**2


@numba.njit(fastmath=True)
def cos_sim(vec_1: Vec2D, vec_2: Vec2D) -> float:
    return (vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]) \
        / (euclid_dist(vec_1, (0, 0)) * euclid_dist(vec_2, (0, 0)))


@numba.njit(fastmath=True)
def lineseg_line_intersection_distance(segment: Line2D, sensor_pos: Vec2D, ray_vec: Vec2D) -> float:
    (x_1, y_1), (x_2, y_2) = segment
    x_sensor, y_sensor = sensor_pos
    x_diff, y_diff = x_1 - x_sensor, y_1 - y_sensor
    x_seg, y_seg = x_2 - x_1, y_2 - y_1
    x_ray, y_ray = ray_vec

    num = x_ray * y_diff - y_ray * x_diff
    den = x_seg * y_ray - x_ray * y_seg

    # edge case: line segment has same orientation as ray vector
    if den == 0:
        return np.inf

    mu = num / den
    tau = (mu * x_seg + x_diff) / x_ray

    if 0 <= mu <= 1 and tau >= 0:
        cross_x, cross_y = x_1 + mu * (x_2 - x_1), y_1 + mu * (y_2 - y_1)
        return euclid_dist(sensor_pos, (cross_x, cross_y))
    else:
        return np.inf

@numba.njit(fastmath=True)
def circle_line_intersection_distance(circle: Circle2D, origin: Vec2D, ray_vec: Vec2D) -> float:
    (c_x, c_y), r = circle
    ray_x, ray_y = ray_vec

    # shift circle's center to the origin (0, 0)
    (p1_x, p1_y) = origin[0] - c_x, origin[1] - c_y

    r_sq = r**2
    norm_p1 = p1_x**2 + p1_y**2

    # cofficients a, b, c of the quadratic solution formula
    s_x, s_y = ray_x, ray_y
    t_x, t_y = p1_x, p1_y
    a = s_x**2 + s_y**2
    b = 2 * (s_x * t_x + s_y * t_y)
    c = norm_p1 - r_sq

    # abort when ray doesn't collide
    disc = b**2 - 4 * a * c
    if disc < 0 or (b > 0 and b**2 > disc):
        return np.inf

    # compute quadratic solutions
    disc_root = disc**0.5
    mu_1 = (-b - disc_root) / 2 * a
    mu_2 = (-b + disc_root) / 2 * a

    # compute cross points S1, S2 and distances to the origin
    s1_x, s1_y = mu_1 * s_x + t_x, mu_1 * s_y  + t_y
    s2_x, s2_y = mu_2 * s_x + t_x, mu_2 * s_y  + t_y
    dist_1 = euclid_dist((p1_x, p1_y), (s1_x, s1_y))
    dist_2 = euclid_dist((p1_x, p1_y), (s2_x, s2_y))

    if mu_1 >= 0 and mu_2 >= 0:
        return min(dist_1, dist_2)
    elif mu_1 >= 0:
        return dist_1
    else: # if mu_2 >= 0:
        return dist_2


@numba.njit(fastmath=True)
def is_circle_circle_intersection(c_1: Circle2D, c_2: Circle2D) -> bool:
    center_1, radius_1 = c_1
    center_2, radius_2 = c_2
    dist_sq = euclid_dist_sq(center_1, center_2)
    rad_sum_sq = (radius_1 + radius_2)**2
    return dist_sq <= rad_sum_sq


@numba.njit(fastmath=True)
def is_circle_line_intersection(circle: Circle2D, segment: Line2D) -> bool:
    """Simple vector math implementation using quadratic solution formula."""
    (c_x, c_y), r = circle
    (p1_x, p1_y), (p2_x, p2_y) = segment

    # shift circle's center to the origin (0, 0)
    (p1_x, p1_y), (p2_x, p2_y) = (p1_x - c_x, p1_y - c_y), (p2_x - c_x, p2_y - c_y)

    r_sq = r**2
    norm_p1, norm_p2 = p1_x**2 + p1_y**2, p2_x**2 + p2_y**2

    # edge case: line segment ends inside the circle -> collision!
    if norm_p1 <= r_sq or norm_p2 <= r_sq:
        return True

    # cofficients a, b, c of the quadratic solution formula
    s_x, s_y = p2_x - p1_x, p2_y - p1_y
    t_x, t_y = p1_x, p1_y
    a = s_x**2 + s_y**2
    b = 2 * (s_x * t_x + s_y * t_y)
    c = norm_p1 - r_sq

    # discard cases where infinite line doesn't collide
    disc = b**2 - 4 * a * c
    if disc < 0:
        return False

    # check if collision is actually within the line segment
    disc_root = disc**0.5
    return 0 <= -b - disc_root <= 2 * a or 0 <= -b + disc_root <= 2 * a
