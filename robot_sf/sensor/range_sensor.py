"""TODO docstring. Document this module."""

from dataclasses import dataclass, field
from math import cos, sin

import numba
import numpy as np
from gymnasium import spaces

from robot_sf.common.geometry import euclid_dist
from robot_sf.common.types import Circle2D, Line2D, Range, RobotPose, Vec2D
from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy


@numba.njit(fastmath=True)
def lineseg_line_intersection_distance(segment: Line2D, sensor_pos: Vec2D, ray_vec: Vec2D) -> float:
    """Distance from sensor position to the intersection with a ray and segment.

    Args:
        segment: Line segment represented by two endpoints.
        sensor_pos: Origin of the ray.
        ray_vec: Ray direction vector.

    Returns:
        float: Distance to the intersection point, or ``np.inf`` when no intersection exists.
    """
    # Unpack segment endpoints, sensor position, and ray vector
    # Line2D is ((x1, y1), (x2, y2))
    (x_1, y_1), (x_2, y_2) = segment
    x_sensor, y_sensor = sensor_pos
    x_ray, y_ray = ray_vec

    # Calculate differences for intersection formula
    x_diff, y_diff = x_1 - x_sensor, y_1 - y_sensor
    x_seg, y_seg = x_2 - x_1, y_2 - y_1

    # Calculate numerator and denominator of intersection formula
    num = x_ray * y_diff - y_ray * x_diff
    den = x_seg * y_ray - x_ray * y_seg

    # Edge case: line segment has same orientation as ray vector
    if den == 0:
        return np.inf

    # Calculate intersection parameters
    mu = num / den
    tau = (mu * x_seg + x_diff) / x_ray

    # If intersection is within segment and in ray direction, calculate distance
    if 0 <= mu <= 1 and tau >= 0:
        cross_x, cross_y = x_1 + mu * (x_2 - x_1), y_1 + mu * (y_2 - y_1)
        return euclid_dist(sensor_pos, (cross_x, cross_y))
    else:
        return np.inf


@numba.njit(fastmath=True)
def circle_line_intersection_distance(circle: Circle2D, origin: Vec2D, ray_vec: Vec2D) -> float:
    """Distance from origin to circle/ray intersection.

    Args:
        circle: Circle defined by ``(center, radius)``.
        origin: Origin of the ray.
        ray_vec: Ray direction vector.

    Returns:
        float: Distance to the nearest valid intersection, or ``np.inf`` if the ray misses.
    """
    # Unpack circle center and radius, and ray vector
    (circle_x, circle_y), radius = circle
    ray_x, ray_y = ray_vec

    # Shift circle's center to the origin (0, 0)
    p1_x = origin[0] - circle_x
    p1_y = origin[1] - circle_y

    # Calculate squared radius and norm of p1
    r_sq = radius**2
    norm_p1 = p1_x**2 + p1_y**2

    # Coefficients a, b, c of the quadratic solution formula
    # ax^2+bx+c=0
    s_x, s_y = ray_x, ray_y
    t_x, t_y = p1_x, p1_y
    a = s_x**2 + s_y**2
    b = 2 * (s_x * t_x + s_y * t_y)
    c = norm_p1 - r_sq

    # Abort when ray doesn't collide with circle
    disc = b**2 - 4 * a * c
    if disc < 0 or (b > 0 and b**2 > disc):
        return np.inf

    # Compute quadratic solutions
    disc_root = disc**0.5
    mu_1 = (-b - disc_root) / (2 * a)
    mu_2 = (-b + disc_root) / (2 * a)

    # Compute cross points S1, S2 and distances to the origin
    s1_x, s1_y = mu_1 * s_x + t_x, mu_1 * s_y + t_y
    s2_x, s2_y = mu_2 * s_x + t_x, mu_2 * s_y + t_y
    dist_1 = euclid_dist((p1_x, p1_y), (s1_x, s1_y))
    dist_2 = euclid_dist((p1_x, p1_y), (s2_x, s2_y))

    # Return the distance to the nearest intersection point
    if mu_1 >= 0 and mu_2 >= 0:
        return min(dist_1, dist_2)
    elif mu_1 >= 0:
        return dist_1
    else:  # if mu_2 >= 0:
        return dist_2


@dataclass
class LidarScannerSettings:
    """Representing LiDAR sensor configuration settings."""

    max_scan_dist: float = 10.0
    visual_angle_portion: float = 1.0
    num_rays: int = 272
    scan_noise: list[float] = field(default_factory=lambda: [0.005, 0.002])
    angle_opening: Range = field(init=False)

    def __post_init__(self):
        """TODO docstring. Document this function."""
        if not 0 < self.visual_angle_portion <= 1:
            raise ValueError("Scan angle portion needs to be within (0, 1]!")
        if self.max_scan_dist <= 0:
            raise ValueError("Max. scan distance mustn't be negative or zero!")
        if self.num_rays <= 0:
            raise ValueError("Amount of LiDAR rays mustn't be negative or zero!")
        if any(not 0 <= prob <= 1 for prob in self.scan_noise):
            raise ValueError("Scan noise probabilities must be within [0, 1]!")

        self.angle_opening = (-np.pi * self.visual_angle_portion, np.pi * self.visual_angle_portion)


@numba.njit(fastmath=True)
def raycast_pedestrians(
    out_ranges: np.ndarray,
    scanner_pos: Vec2D,
    max_scan_range: float,
    ped_positions: np.ndarray,
    ped_radius: float,
    ray_angles: np.ndarray,
):
    """Perform raycasts to detect pedestrians within the scanner's range.

    Args:
        out_ranges: Output array modified in-place with the detected range per ray.
        scanner_pos: Position of the LiDAR sensor in world coordinates.
        max_scan_range: Maximum detection distance for each ray.
        ped_positions: Pedestrian positions to test against the rays.
        ped_radius: Radius used to approximate pedestrians as discs.
        ray_angles: Ray directions in radians.

    Notes:
        ``out_ranges`` is modified in place and no value is returned.
    """

    # Check if pedestrian positions array is empty or not 2D
    if len(ped_positions.shape) != 2:
        return
    if ped_positions.shape[0] == 0 or ped_positions.shape[1] != 2:
        return

    # Convert scanner position to numpy array
    scanner_pos_np = np.array([scanner_pos[0], scanner_pos[1]])

    # Calculate square of maximum scan range
    threshold_sq = max_scan_range**2

    # Calculate relative positions of pedestrians and their squared distances
    relative_ped_pos = ped_positions - scanner_pos_np
    dist_sq = np.sum(relative_ped_pos**2, axis=1)

    # Find pedestrians within scanner's range
    ped_dist_mask = np.where(dist_sq <= threshold_sq)[0]
    close_ped_pos = relative_ped_pos[ped_dist_mask]

    # If no pedestrians are within range, return
    if len(ped_dist_mask) == 0:
        return

    # For each ray angle, calculate cosine similarities and find pedestrians
    # in the direction of the ray
    for i, angle in enumerate(ray_angles):
        unit_vec = cos(angle), sin(angle)
        cos_sims = close_ped_pos[:, 0] * unit_vec[0] + close_ped_pos[:, 1] * unit_vec[1]

        # Find pedestrians in the direction of the ray
        ped_dir_mask = np.where(cos_sims >= 0)[0]
        joined_mask = ped_dist_mask[ped_dir_mask]
        relevant_ped_pos = relative_ped_pos[joined_mask]

        # For each pedestrian in the direction of the ray, calculate the
        # distance to the pedestrian's edge and update the output range
        for pos in relevant_ped_pos:
            ped_circle = ((pos[0], pos[1]), ped_radius)
            coll_dist = circle_line_intersection_distance(ped_circle, (0.0, 0.0), unit_vec)
            out_ranges[i] = min(coll_dist, out_ranges[i])


@numba.njit(fastmath=True)
def raycast_obstacles(
    out_ranges: np.ndarray,
    scanner_pos: Vec2D,
    obstacles: np.ndarray,
    ray_angles: np.ndarray,
):
    """TODO docstring. Document this function.

    Args:
        out_ranges: TODO docstring.
        scanner_pos: TODO docstring.
        obstacles: TODO docstring.
        ray_angles: TODO docstring.
    """
    if len(obstacles.shape) != 2 or obstacles.shape[0] == 0 or obstacles.shape[1] != 4:
        return

    for i, angle in enumerate(ray_angles):
        unit_vec = cos(angle), sin(angle)
        for s_x, s_y, e_x, e_y in obstacles:
            obst_lineseg = ((s_x, s_y), (e_x, e_y))
            coll_dist = lineseg_line_intersection_distance(obst_lineseg, scanner_pos, unit_vec)
            out_ranges[i] = min(coll_dist, out_ranges[i])


@numba.njit()
def raycast(
    scanner_pos: Vec2D,
    obstacles: np.ndarray,
    max_scan_range: float,
    ped_pos: np.ndarray,
    ped_radius: float,
    ray_angles: np.ndarray,
    enemy_pos: np.ndarray | None = None,
    enemy_radius: float = 0.0,
) -> np.ndarray:
    """Cast rays to compute minimal collision distances along given angles.

    The scan originates from the scanner's position and considers pedestrians,
    obstacles, and optionally an enemy agent. When no collision occurs, the
    maximum scan range is returned for that ray.

    Returns:
        numpy.ndarray: Per-ray distances with shape ``(num_rays,)``.
    """
    out_ranges = np.full((ray_angles.shape[0]), np.inf)
    raycast_pedestrians(out_ranges, scanner_pos, max_scan_range, ped_pos, ped_radius, ray_angles)
    raycast_obstacles(out_ranges, scanner_pos, obstacles, ray_angles)

    # As Pedestrian detect the Robot
    if enemy_pos is not None:
        raycast_pedestrians(
            out_ranges,
            scanner_pos,
            max_scan_range,
            enemy_pos,
            enemy_radius,
            ray_angles,
        )
    # TODO(#251): add raycast for other robots
    # See: https://github.com/ll7/robot_sf_ll7/issues/251
    return out_ranges


@numba.njit(fastmath=True)
def range_postprocessing(out_ranges: np.ndarray, scan_noise: np.ndarray, max_scan_dist: float):
    """Postprocess the raycast results to simulate a noisy scan result."""
    prob_scan_loss, prob_scan_corruption = scan_noise
    for i in range(out_ranges.shape[0]):
        out_ranges[i] = min(out_ranges[i], max_scan_dist)
        if np.random.random() < prob_scan_loss:
            out_ranges[i] = max_scan_dist
        elif np.random.random() < prob_scan_corruption:
            out_ranges[i] = out_ranges[i] * np.random.random()


def lidar_ray_scan(
    pose: RobotPose,
    occ: ContinuousOccupancy,
    settings: LidarScannerSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a radial LiDAR scan on a continuous occupancy with objects.

    The occupancy contains the robot (as circle), a set of pedestrians
    (as circles) and a set of static obstacles (as 2D lines).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A pair ``(ranges, ray_angles)`` where
        ``ranges`` are per-ray distances and ``ray_angles`` are absolute ray angles.
    """

    (pos_x, pos_y), robot_orient = pose
    scan_noise = np.array(settings.scan_noise)
    scan_dist = settings.max_scan_dist

    ped_pos = occ.pedestrian_coords
    obstacles = occ.obstacle_coords

    lower = robot_orient + settings.angle_opening[0]
    upper = robot_orient + settings.angle_opening[1]
    ray_angles = np.linspace(lower, upper, settings.num_rays + 1)[:-1]
    ray_angles = np.array([(angle + np.pi * 2) % (np.pi * 2) for angle in ray_angles])

    if isinstance(occ, EgoPedContinuousOccupancy):
        enemy_pos = np.array([occ.enemy_coords])
        ranges = raycast(
            (pos_x, pos_y),
            obstacles,
            scan_dist,
            ped_pos,
            occ.ped_radius,
            ray_angles,
            enemy_pos=enemy_pos,
            enemy_radius=occ.enemy_radius,
        )
    else:
        ranges = raycast((pos_x, pos_y), obstacles, scan_dist, ped_pos, occ.ped_radius, ray_angles)
    range_postprocessing(ranges, scan_noise, scan_dist)
    return ranges, ray_angles


def lidar_sensor_space(num_rays: int, max_scan_dist: float) -> spaces.Box:
    """TODO docstring. Document this function.

    Args:
        num_rays: TODO docstring.
        max_scan_dist: TODO docstring.

    Returns:
        TODO docstring.
    """
    high = np.full((num_rays), max_scan_dist, dtype=np.float32)
    low = np.zeros((num_rays), dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)
