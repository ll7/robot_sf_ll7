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
def lineseg_line_intersection_distance(segment: Line2D, origin: Vec2D, ray_vec: Vec2D) -> float:
    # source: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    (x_1, y_1), (x_2, y_2) = segment
    (x_3, y_3), (x_4, y_4) = origin, (origin[0] + ray_vec[0], origin[1] + ray_vec[1])

    num = (x_1 - x_3) * (y_3 - y_4) - (y_1 - y_3) * (x_3 - x_4)
    den = (x_1 - x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3 - x_4)

    # edge case: line segment has same orientation as ray vector
    if den == 0:

        # check if parallel lines are aligned
        v_1 = (x_1 - x_3, y_1 - y_3)
        v_2 = (x_2 - x_4, y_2 - y_4)
        v_3 = (v_1[0] * -1, v_1[1] * -1)
        if v_1 == (0, 0) or v_2 == (0, 0) or \
                cos_sim(v_1, v_2) > 0.999 or cos_sim(v_2, v_3) > 0.999:
            min_x, max_x = min(x_1, x_2), max(x_1, x_2)
            min_y, max_y = min(y_1, y_2), max(y_1, y_2)

            if min_x <= origin[0] <= max_x and min_y <= origin[1] <= max_y:
                return 0.0
            else:
                dist1 = euclid_dist(origin, (x_1, y_1))
                dist2 = euclid_dist(origin, (x_2, y_2))
                return min(dist1, dist2)
        else:
            # parallel lines cannot intersect
            return np.inf

    hit_scale = num / den # called 't' in formula
    if 0 <= hit_scale <= 1:
        cross_x, cross_y = x_1 + hit_scale * (x_2 - x_1), y_1 + hit_scale * (y_2 - y_1)
        return euclid_dist(origin, (cross_x, cross_y))
    else:
        return np.inf


@numba.njit(fastmath=True)
def circle_line_intersection_distance(circle: Circle2D, origin: Vec2D, ray_vec: Vec2D) -> float:
    # source: https://mathworld.wolfram.com/Circle-LineIntersection.html
    (circle_x, circle_y), radius = circle
    (x_1, y_1) = origin[0] - circle_x, origin[1] - circle_y
    (x_2, y_2) = x_1 - ray_vec[0], y_1 - ray_vec[1]

    det = x_1 * y_2 - x_2 * y_1
    d_x, d_y = x_2 - x_1, y_2 - y_1
    d_r_sq = euclid_dist_sq((d_x, d_y), (0, 0))
    disc = radius**2 * d_r_sq - det**2

    if not disc >= 0:
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
    (circle_x, circle_y), radius = circle
    p_1, p_2 = segment
    (x_1, y_1) = p_1[0] - circle_x, p_1[1] - circle_y
    (x_2, y_2) = p_2[0] - circle_x, p_2[1] - circle_y
    r_sq = radius**2

    # edge case: line segment's end point(s) inside circle
    if euclid_dist_sq((x_1, y_1), (0, 0)) <= r_sq \
            or euclid_dist_sq((x_2, y_2), (0, 0)) <= r_sq:
        return True

    det = x_1 * y_2 - x_2 * y_1
    d_x, d_y = x_2 - x_1, y_2 - y_1
    d_r_sq = euclid_dist_sq((d_x, d_y), (0, 0))
    disc = r_sq * d_r_sq - det**2

    if not disc >= 0:
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
