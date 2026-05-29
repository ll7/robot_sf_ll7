"""LiDAR-style range scanning utilities for continuous occupancy maps.

The helpers in this module cast radial rays from a robot or ego pedestrian pose
against line-segment obstacles and circular dynamic objects. Distances are
reported in world units and clipped/noised to match the configured scanner
settings before they are exposed as Gymnasium observation spaces.
"""

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
    """Configuration for the range scanner used by ``lidar_ray_scan``.

    Distances use the same world-coordinate units as the occupancy object
    passed to ``lidar_ray_scan``. Angles are radians. ``num_rays`` controls the
    one-dimensional observation length, and ``scan_noise`` contains two
    probabilities: scan loss and scan corruption.
    """

    max_scan_dist: float = 10.0
    visual_angle_portion: float = 1.0
    num_rays: int = 272
    scan_noise: list[float] = field(default_factory=lambda: [0.005, 0.002])
    detect_other_robots: bool = True
    angle_opening: Range = field(init=False)

    def __post_init__(self):
        """Validate scanner settings and derive the symmetric angular opening.

        ``visual_angle_portion`` is a fraction of a full circle, so ``1.0``
        scans 360 degrees and ``1 / 3`` scans 120 degrees. ``scan_noise`` stores
        scan-loss and corruption probabilities, both constrained to ``[0, 1]``.
        """
        if not 0 < self.visual_angle_portion <= 1:
            raise ValueError("Scan angle portion needs to be within (0, 1]!")
        if self.max_scan_dist <= 0:
            raise ValueError("Max. scan distance mustn't be negative or zero!")
        if self.num_rays <= 0:
            raise ValueError("Amount of LiDAR rays mustn't be negative or zero!")
        if any(not 0 <= prob <= 1 for prob in self.scan_noise):
            raise ValueError("Scan noise probabilities must be within [0, 1]!")

        self.angle_opening = (-np.pi * self.visual_angle_portion, np.pi * self.visual_angle_portion)

    @classmethod
    def ego_pedestrian_lidar(cls) -> "LidarScannerSettings":
        """Create a lidar configuration for ego pedestrian with 120 degree view and extended range.

        Returns:
            LidarScannerSettings: Lidar configuration with 120° field of view and 30m range.
        """
        return cls(
            max_scan_dist=30.0,  # Extended range
            visual_angle_portion=1.0 / 3.0,  # 120 degrees (1/3 of 360)
            num_rays=272,  # Same granularity
        )

    @classmethod
    def default(cls) -> "LidarScannerSettings":
        """Create default lidar configuration (360 degree view, 10m range).

        Returns:
            LidarScannerSettings: Default lidar configuration.
        """
        return cls()


@numba.njit(fastmath=True)
def raycast_pedestrians(
    out_ranges: np.ndarray,
    scanner_pos: Vec2D,
    max_scan_range: float,
    ped_positions: np.ndarray,
    ped_radius: float,
    ray_angles: np.ndarray,
):
    """Update ray ranges with pedestrian-circle intersections.

    Args:
        out_ranges: Output array modified in-place with the detected range per ray.
        scanner_pos: Position of the LiDAR sensor in world coordinates.
        max_scan_range: Maximum detection distance for each ray.
        ped_positions: Pedestrian centers as an ``(N, 2)`` array of ``x, y``
            world coordinates. Empty arrays are valid. Arrays with any other
            shape are treated as no detections.
        ped_radius: Radius, in world units, used to approximate pedestrians as discs.
        ray_angles: Absolute ray directions in radians, one per ``out_ranges``
            entry.

    Notes:
        ``out_ranges`` is modified in place and no value is returned.
    """

    if len(ped_positions.shape) != 2 or ped_positions.shape[1] != 2:
        return

    circles = np.empty((ped_positions.shape[0], 3), dtype=np.float64)
    circles[:, :2] = ped_positions
    circles[:, 2] = ped_radius
    raycast_circles(out_ranges, scanner_pos, max_scan_range, circles, ray_angles)


@numba.njit(fastmath=True)
def raycast_circles(
    out_ranges: np.ndarray,
    scanner_pos: Vec2D,
    max_scan_range: float,
    circles: np.ndarray,
    ray_angles: np.ndarray,
):
    """Update ray ranges with generic circle intersections.

    Args:
        out_ranges: Mutable per-ray distances, updated in place with nearer
            circle hits.
        scanner_pos: Scanner origin in world coordinates.
        max_scan_range: Maximum distance used to prefilter candidate circles.
        circles: Circle rows as an ``(N, 3)`` array of ``x, y, radius`` values
            in world units. Empty arrays, malformed arrays, and ``None`` callers
            handled by ``raycast`` are treated as no detections.
        ray_angles: Absolute ray directions in radians, one per ``out_ranges``
            entry.
    """
    if len(circles.shape) != 2 or circles.shape[0] == 0 or circles.shape[1] != 3:
        return

    scanner_pos_np = np.array([scanner_pos[0], scanner_pos[1]])
    threshold_sq = max_scan_range**2
    relative_circle_pos = circles[:, :2] - scanner_pos_np
    circle_radii = circles[:, 2]
    dist_sq = np.sum(relative_circle_pos**2, axis=1)
    near_mask = np.where(dist_sq <= threshold_sq)[0]
    if len(near_mask) == 0:
        return
    close_circle_pos = relative_circle_pos[near_mask]
    close_circle_radii = circle_radii[near_mask]

    for i, angle in enumerate(ray_angles):
        unit_vec = cos(angle), sin(angle)
        cos_sims = close_circle_pos[:, 0] * unit_vec[0] + close_circle_pos[:, 1] * unit_vec[1]
        circle_dir_mask = np.where(cos_sims >= 0)[0]
        for idx in circle_dir_mask:
            pos = close_circle_pos[idx]
            radius = close_circle_radii[idx]
            circle = ((pos[0], pos[1]), radius)
            coll_dist = circle_line_intersection_distance(circle, (0.0, 0.0), unit_vec)
            out_ranges[i] = min(coll_dist, out_ranges[i])


@numba.njit(fastmath=True)
def raycast_obstacles(
    out_ranges: np.ndarray,
    scanner_pos: Vec2D,
    obstacles: np.ndarray,
    ray_angles: np.ndarray,
):
    """Update ray ranges with intersections against static obstacle segments.

    Args:
        out_ranges: Mutable per-ray distances. Entries are replaced in place
            when a nearer obstacle intersection is found.
        scanner_pos: Scanner origin in world coordinates.
        obstacles: Obstacle segments as an ``(N, 4)`` array of
            ``start_x, start_y, end_x, end_y`` rows.
        ray_angles: Absolute ray directions in radians, one per ``out_ranges``
            entry.

    Notes:
        Malformed or empty obstacle arrays are treated as no detections. The
        function mutates ``out_ranges`` and returns ``None``.
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
def raycast(  # noqa: PLR0913
    scanner_pos: Vec2D,
    obstacles: np.ndarray,
    max_scan_range: float,
    ped_pos: np.ndarray,
    ped_radius: float,
    ray_angles: np.ndarray,
    enemy_pos: np.ndarray | None = None,
    enemy_radius: float = 0.0,
    other_robot_circles: np.ndarray | None = None,
) -> np.ndarray:
    """Cast rays to compute minimal collision distances along given angles.

    The scan originates from ``scanner_pos`` and considers static obstacle
    segments, pedestrian discs, an optional enemy disc, and optional dynamic
    robot discs. ``scanner_pos`` is ``(x, y)`` in world coordinates, obstacles
    are ``(N, 4)`` rows of ``start_x, start_y, end_x, end_y``, pedestrian and
    enemy positions are ``(N, 2)`` arrays, and ``other_robot_circles`` is an
    ``(N, 3)`` array of ``x, y, radius`` rows.

    ``max_scan_range`` is used to prefilter circular objects. No-hit rays remain
    ``np.inf`` here; ``lidar_ray_scan`` clamps them to ``max_scan_dist`` during
    postprocessing.

    Returns:
        numpy.ndarray: Per-ray distances with shape ``(len(ray_angles),)`` in
        world units. The returned array is newly allocated.
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
    if other_robot_circles is not None:
        raycast_circles(
            out_ranges,
            scanner_pos,
            max_scan_range,
            other_robot_circles,
            ray_angles,
        )
    return out_ranges


def _dynamic_objects_to_circle_array(occ: ContinuousOccupancy) -> np.ndarray | None:
    """Convert occupancy dynamic objects into an ``(N, 3)`` raycast circle array.

    Returns:
        np.ndarray | None: Array columns are ``x``, ``y``, ``radius``.
        Returns ``None`` when no usable dynamic circles are available.
    """
    get_dynamic_objects = getattr(occ, "get_dynamic_objects", None)
    if get_dynamic_objects is None:
        return None

    objects = get_dynamic_objects()
    if not objects:
        return None

    rows: list[tuple[float, float, float]] = []
    for circle in objects:
        try:
            (x, y), radius = circle
        except (TypeError, ValueError):
            continue
        rows.append((float(x), float(y), float(radius)))

    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


@numba.njit(fastmath=True)
def range_postprocessing(out_ranges: np.ndarray, scan_noise: np.ndarray, max_scan_dist: float):
    """Clamp and noise raycast results in place.

    Args:
        out_ranges: Mutable per-ray distances. Values above ``max_scan_dist``
            and ``np.inf`` no-hit sentinels are clipped to ``max_scan_dist``.
        scan_noise: Two probabilities ``[scan_loss, scan_corruption]``. Loss
            replaces the ray with ``max_scan_dist``; corruption scales the
            current distance by a random factor in ``[0, 1)``.
        max_scan_dist: Inclusive maximum scanner range in world units.

    Notes:
        This function mutates ``out_ranges`` and returns ``None``. Settings
        validation is performed by ``LidarScannerSettings`` before normal use;
        direct callers are expected to pass two probabilities.
    """
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

    Args:
        pose: Robot pose as ``((x, y), heading)`` in world coordinates and
            radians.
        occ: Occupancy provider exposing ``pedestrian_coords``,
            ``obstacle_coords``, and ``ped_radius``. ``obstacle_coords`` must be
            an ``(N, 4)`` array of segment endpoints, and ``pedestrian_coords``
            must be an ``(N, 2)`` array. ``EgoPedContinuousOccupancy`` also
            contributes its enemy circle. When ``settings.detect_other_robots``
            is true, an optional ``get_dynamic_objects()`` callback may return
            ``[((x, y), radius), ...]`` circles.
        settings: Scanner configuration. Distances use world units and angles
            use radians.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A pair ``(ranges, ray_angles)`` where
        ``ranges`` are per-ray distances with shape ``(settings.num_rays,)`` and
        bounds ``[0, settings.max_scan_dist]`` after postprocessing.
        ``ray_angles`` has the same shape and contains absolute angles in
        ``[0, 2*pi)`` radians.

    Notes:
        The occupancy object is read but not mutated. Malformed dynamic object
        entries are ignored by the conversion helper; malformed pedestrian or
        obstacle arrays are treated as empty detections by the raycast helpers.
    """

    (pos_x, pos_y), robot_orient = pose
    scan_noise = np.array(settings.scan_noise)
    scan_dist = settings.max_scan_dist

    ped_pos = occ.pedestrian_coords
    obstacles = occ.obstacle_coords
    dynamic_robot_circles = (
        _dynamic_objects_to_circle_array(occ) if settings.detect_other_robots else None
    )

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
            other_robot_circles=dynamic_robot_circles,
        )
    else:
        ranges = raycast(
            (pos_x, pos_y),
            obstacles,
            scan_dist,
            ped_pos,
            occ.ped_radius,
            ray_angles,
            other_robot_circles=dynamic_robot_circles,
        )
    range_postprocessing(ranges, scan_noise, scan_dist)
    return ranges, ray_angles


def lidar_sensor_space(num_rays: int, max_scan_dist: float) -> spaces.Box:
    """Build the Gymnasium observation space for LiDAR range readings.

    Args:
        num_rays: Number of scalar range readings in each scan.
        max_scan_dist: Inclusive upper bound for each range reading, in world
            distance units.

    Returns:
        A float32 ``Box`` with shape ``(num_rays,)`` and bounds
        ``[0, max_scan_dist]`` for every ray.
    """
    high = np.full((num_rays), max_scan_dist, dtype=np.float32)
    low = np.zeros((num_rays), dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)
