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
def circle_line_intersection_distance_2(circle: Circle2D, origin: Vec2D, ray_vec: Vec2D) -> float:
    # source: https://mathworld.wolfram.com/Circle-LineIntersection.html
    (circle_x, circle_y), radius = circle
    (x_1, y_1) = origin[0] - circle_x, origin[1] - circle_y
    (x_2, y_2) = x_1 + ray_vec[0], y_1 + ray_vec[1]

    det = x_1 * y_2 - x_2 * y_1
    d_x, d_y = x_2 - x_1, y_2 - y_1
    d_r_sq = euclid_dist_sq((d_x, d_y), (0, 0))
    disc = radius**2 * d_r_sq - det**2

    if disc < 0:
        return np.inf

    disc_root = disc**0.5
    sign_dy = 1 if d_y >= 0 else -1
    cross_x1 = (det * d_y + sign_dy * d_x * disc_root) / d_r_sq
    cross_y1 = (-det * d_x + abs(d_y) * disc_root) / d_r_sq
    cross_x2 = (det * d_y - sign_dy * d_x * disc_root) / d_r_sq
    cross_y2 = (-det * d_x - abs(d_y) * disc_root) / d_r_sq

    dist_cross1 = euclid_dist((x_1, y_1), (cross_x1, cross_y1))
    dist_cross2 = euclid_dist((x_1, y_1), (cross_x2, cross_y2))

    vec_cross1 = cross_x1 - x_1, cross_y1 - y_1 # vector |origin -> cross_1|
    vec_cross2 = cross_x2 - x_1, cross_y2 - y_1 # vector |origin -> cross_2|
    sim1, sim2 = cos_sim(ray_vec, vec_cross1), cos_sim(ray_vec, vec_cross2)

    cross1_aligned = sim1 > 0.999
    cross2_aligned = sim2 > 0.999
    if cross1_aligned and cross2_aligned:
        return min(dist_cross1, dist_cross2)
    elif cross1_aligned:
        return dist_cross1
    elif cross2_aligned:
        return dist_cross2
    else:
        return np.inf


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


@numba.njit(fastmath=True)
def is_circle_line_intersection_2(circle: Circle2D, segment: Line2D) -> bool:
    """Alternative implementation using determinant vector math"""
    (circle_x, circle_y), radius = circle
    p_1, p_2 = segment
    (x_1, y_1) = p_1[0] - circle_x, p_1[1] - circle_y
    (x_2, y_2) = p_2[0] - circle_x, p_2[1] - circle_y
    r_sq = radius**2

    # edge case: line segment's end point(s) are inside the circle -> collision!
    if x_1**2 + y_1**2 <= r_sq or x_2**2 + y_2**2 <= r_sq:
        return True

    det = x_1 * y_2 - x_2 * y_1
    d_x, d_y = x_2 - x_1, y_2 - y_1
    d_r_sq = d_x**2 + d_y**2
    disc = r_sq * d_r_sq - det**2

    if disc < 0:
        return False

    disc_root = disc**0.5
    sign_dy = 1 if d_y >= 0 else -1
    cross_x1 = (det * d_y + sign_dy * d_x * disc_root)
    cross_y1 = (-det * d_x + abs(d_y) * disc_root)
    cross_x2 = (det * d_y - sign_dy * d_x * disc_root)
    cross_y2 = (-det * d_x - abs(d_y) * disc_root)
    # info: don't divide by d_r_sq, rather multiply the comparison (-> faster)

    min_x, max_x = min(x_1 * d_r_sq, x_2 * d_r_sq), max(x_1 * d_r_sq, x_2 * d_r_sq)
    min_y, max_y = min(y_1 * d_r_sq, y_2 * d_r_sq), max(y_1 * d_r_sq, y_2 * d_r_sq)
    is_coll1 = min_x <= cross_x1 <= max_x and min_y <= cross_y1 <= max_y
    is_coll2 = min_x <= cross_x2 <= max_x and min_y <= cross_y2 <= max_y
    return is_coll1 or is_coll2


# def is_lineseg_line_intersection(segment: Line2D, origin: Vec2D, ray_vec: Vec2D) -> bool:
#     # source: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
#     (x_1, y_1), (x_2, y_2) = segment
#     (x_3, y_3), (x_4, y_4) = origin, (origin[0] + ray_vec[0], origin[1] + ray_vec[1])

#     num = (x_1 - x_3) * (y_3 - y_4) - (y_1 - y_3) * (x_3 - x_4)
#     den = (x_1 - x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3 - x_4)

#     if den == 0:
#         return False

#     t = num / den
#     return 0 <= t <= 1
