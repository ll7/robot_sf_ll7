"""LiDAR-style range scanning utilities for continuous occupancy maps.

The helpers in this module cast radial rays from a robot or ego pedestrian pose
against line-segment obstacles and circular dynamic objects. Distances are
reported in world units and clipped/noised to match the configured scanner
settings before they are exposed as Gymnasium observation spaces.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from math import cos, sin
from threading import Barrier, BrokenBarrierError, Lock, local

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
    """Configuration for a radial LiDAR scan in world-coordinate units.

    Attributes:
        max_scan_dist: Maximum distance returned by any ray, in map/world units.
        visual_angle_portion: Fraction of a full circle scanned around the
            sensor heading. ``1.0`` covers 360 degrees and ``1 / 3`` covers a
            120-degree field of view centered on the heading.
        num_rays: Number of equally spaced scalar range readings per scan.
        scan_noise: Two probabilities ``[loss, corruption]``. Loss replaces a
            ray with ``max_scan_dist``; corruption scales the clipped range by a
            random factor in ``[0, 1)``. Set this to ``[0.0, 0.0]`` when running
            Gymnasium's deterministic environment checker.
        scan_noise_array: Cached ndarray derived from ``scan_noise``, allocated
            once at settings init to avoid per-scan array creation.
        detect_other_robots: When true, dynamic objects exposed by the
            occupancy source are treated as circular obstacles.
        angle_opening: Derived symmetric angular interval in radians, populated
            during initialization.
    """

    max_scan_dist: float = 10.0
    visual_angle_portion: float = 1.0
    num_rays: int = 272
    scan_noise: list[float] = field(default_factory=lambda: [0.005, 0.002])
    detect_other_robots: bool = True
    angle_opening: Range = field(init=False)
    scan_noise_array: np.ndarray = field(init=False)
    ray_offsets: np.ndarray = field(init=False)

    def __post_init__(self):
        """Validate scanner settings and derive derived fields.

        ``visual_angle_portion`` is a fraction of a full circle, so ``1.0``
        scans 360 degrees and ``1 / 3`` scans 120 degrees. ``scan_noise`` stores
        scan-loss and corruption probabilities, both constrained to ``[0, 1]``.
        ``ray_offsets`` is a cached heading-independent linspace of ray angles
        relative to the sensor heading. ``scan_noise_array`` is a read-only
        float64 ndarray pre-converted from ``scan_noise``.
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
        self.ray_offsets = lidar_ray_offsets(self.num_rays, self.visual_angle_portion)
        scan_noise_arr = np.array(self.scan_noise, dtype=np.float64)
        scan_noise_arr.flags.writeable = False
        self.scan_noise_array = scan_noise_arr

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


@numba.njit(fastmath=True, nogil=True)
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

    if len(ped_positions.shape) != 2 or ped_positions.shape[1] != 2:
        return

    circles = np.empty((ped_positions.shape[0], 3), dtype=np.float64)
    circles[:, :2] = ped_positions
    circles[:, 2] = ped_radius
    raycast_circles(out_ranges, scanner_pos, max_scan_range, circles, ray_angles)


@numba.njit(fastmath=True, nogil=True)
def raycast_circles(
    out_ranges: np.ndarray,
    scanner_pos: Vec2D,
    max_scan_range: float,
    circles: np.ndarray,
    ray_angles: np.ndarray,
):
    """Update ray ranges with intersections against circular obstacles.

    Args:
        out_ranges: Mutable per-ray distances. Values are lowered in place when
            a circle intersects a ray closer than the current entry.
        scanner_pos: Scanner origin in world coordinates.
        max_scan_range: Distance threshold used to skip circles too far from
            the scanner to affect the scan.
        circles: ``(N, 3)`` float array of ``x, y, radius`` rows.
        ray_angles: Absolute ray directions in radians, one per output entry.

    Notes:
        Invalid or empty circle arrays are ignored. The function mutates
        ``out_ranges`` and returns ``None``.
    """
    if len(circles.shape) != 2 or circles.shape[0] == 0 or circles.shape[1] != 3:
        return

    scanner_x, scanner_y = scanner_pos
    threshold_sq = max_scan_range**2

    for i, angle in enumerate(ray_angles):
        unit_vec = cos(angle), sin(angle)
        best_range = out_ranges[i]
        for circle_idx in range(circles.shape[0]):
            rel_x = circles[circle_idx, 0] - scanner_x
            rel_y = circles[circle_idx, 1] - scanner_y
            if rel_x * rel_x + rel_y * rel_y > threshold_sq:
                continue
            if rel_x * unit_vec[0] + rel_y * unit_vec[1] < 0.0:
                continue

            radius = circles[circle_idx, 2]
            circle = ((rel_x, rel_y), radius)
            coll_dist = circle_line_intersection_distance(circle, (0.0, 0.0), unit_vec)
            best_range = min(coll_dist, best_range)
        out_ranges[i] = best_range


@numba.njit(fastmath=True, nogil=True)
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
    """
    if len(obstacles.shape) != 2 or obstacles.shape[0] == 0 or obstacles.shape[1] != 4:
        return

    for i, angle in enumerate(ray_angles):
        unit_vec = cos(angle), sin(angle)
        for s_x, s_y, e_x, e_y in obstacles:
            obst_lineseg = ((s_x, s_y), (e_x, e_y))
            coll_dist = lineseg_line_intersection_distance(obst_lineseg, scanner_pos, unit_vec)
            out_ranges[i] = min(coll_dist, out_ranges[i])


@numba.njit(  # pragma: no cover - exercised via caller; not line-traceable.
    fastmath=True,
    nogil=True,
)
def _raycast_obstacles_batch_kernel(
    out_ranges: np.ndarray,
    scanner_positions: np.ndarray,
    obstacles: np.ndarray,
    obstacle_counts: np.ndarray,
    ray_angles: np.ndarray,
):
    """Run the scalar obstacle raycast for each row in one compiled dispatch."""
    for env_idx in range(out_ranges.shape[0]):
        raycast_obstacles(
            out_ranges[env_idx],
            scanner_positions[env_idx],
            obstacles[env_idx, : obstacle_counts[env_idx]],
            ray_angles[env_idx],
        )


def raycast_obstacles_batch(
    out_ranges: np.ndarray,
    scanner_positions: np.ndarray,
    obstacles: np.ndarray,
    obstacle_counts: np.ndarray,
    ray_angles: np.ndarray,
) -> None:
    """Update several environments' light detection and ranging data in one CPU dispatch.

    This opt-in adapter batches the existing :func:`raycast_obstacles` kernel without
    changing its arithmetic or any environment default. Obstacle rows are padded to a
    common length; ``obstacle_counts`` marks how many rows belong to each environment.
    A one-environment batch calls the established scalar kernel directly so its output
    remains bit-identical to the existing path.

    Args:
        out_ranges: Mutable floating-point array with shape ``(B, R)``.
        scanner_positions: Scanner origins with shape ``(B, 2)``.
        obstacles: Padded obstacle segments with shape ``(B, M, 4)``.
        obstacle_counts: Integer array with shape ``(B,)`` and values in ``[0, M]``.
        ray_angles: Absolute ray directions with shape ``(B, R)``.

    Raises:
        TypeError: If ``out_ranges`` is not a writable floating-point NumPy array or
            ``obstacle_counts`` does not contain integers.
        ValueError: If an input shape or obstacle count violates the batch contract.
    """
    if not isinstance(out_ranges, np.ndarray) or not np.issubdtype(out_ranges.dtype, np.floating):
        raise TypeError("out_ranges must be a floating-point NumPy array")
    if not out_ranges.flags.writeable:
        raise TypeError("out_ranges must be writable")
    if out_ranges.ndim != 2 or out_ranges.shape[0] == 0:
        raise ValueError("out_ranges must have shape (B, R) with B >= 1")

    batch_size, num_rays = out_ranges.shape
    scanner_positions_array = np.asarray(scanner_positions)
    obstacles_array = np.asarray(obstacles)
    obstacle_counts_array = np.asarray(obstacle_counts)
    ray_angles_array = np.asarray(ray_angles)

    if scanner_positions_array.shape != (batch_size, 2):
        raise ValueError("scanner_positions must have shape (B, 2)")
    if (
        obstacles_array.ndim != 3
        or obstacles_array.shape[0] != batch_size
        or obstacles_array.shape[2] != 4
    ):
        raise ValueError("obstacles must have shape (B, M, 4)")
    if obstacle_counts_array.shape != (batch_size,) or not np.issubdtype(
        obstacle_counts_array.dtype, np.integer
    ):
        raise TypeError("obstacle_counts must be an integer array with shape (B,)")
    if np.any(obstacle_counts_array < 0) or np.any(
        obstacle_counts_array > obstacles_array.shape[1]
    ):
        raise ValueError("obstacle_counts values must be within [0, M]")
    if ray_angles_array.shape != (batch_size, num_rays):
        raise ValueError("ray_angles must have shape (B, R) matching out_ranges")

    if batch_size == 1:
        raycast_obstacles(
            out_ranges[0],
            scanner_positions_array[0],
            obstacles_array[0, : obstacle_counts_array[0]],
            ray_angles_array[0],
        )
        return

    _raycast_obstacles_batch_kernel(
        out_ranges,
        scanner_positions_array,
        obstacles_array,
        obstacle_counts_array,
        ray_angles_array,
    )


@dataclass(slots=True)
class _LidarBatchEntry:
    """One environment's static-obstacle raycast inputs for a coordinated step."""

    out_ranges: np.ndarray
    scanner_position: np.ndarray
    obstacles: np.ndarray
    ray_angles: np.ndarray


class LidarBatchCoordinator:
    """Collect one homogeneous LiDAR request per environment and dispatch it together.

    The coordinator is intended for a fixed group of in-process rollout workers. Each
    worker owns one ``env_index`` and submits exactly one request per coordination cycle.
    A timeout or incompatible row aborts the coordinator so partial batches cannot hang
    or silently fall back to scalar execution.
    """

    def __init__(self, batch_size: int, *, timeout_seconds: float = 30.0) -> None:
        """Initialize a reusable fixed-size synchronization barrier."""
        if batch_size < 2:
            raise ValueError("LiDAR batch coordination requires at least two environments")
        if timeout_seconds <= 0:
            raise ValueError("LiDAR batch coordination timeout must be positive")
        self.batch_size = int(batch_size)
        self.timeout_seconds = float(timeout_seconds)
        self._entries: list[_LidarBatchEntry | None] = [None] * self.batch_size
        self._error: BaseException | None = None
        self._error_lock = Lock()
        self._barrier = Barrier(self.batch_size, action=self._dispatch)

    def submit(
        self,
        env_index: int,
        out_ranges: np.ndarray,
        scanner_position: np.ndarray,
        obstacles: np.ndarray,
        ray_angles: np.ndarray,
    ) -> None:
        """Submit one row and wait until the complete batch has been dispatched."""
        if not 0 <= env_index < self.batch_size:
            raise ValueError(f"env_index must be within [0, {self.batch_size})")
        self._raise_if_failed()
        self._entries[env_index] = _LidarBatchEntry(
            out_ranges=out_ranges,
            scanner_position=np.asarray(scanner_position),
            obstacles=np.asarray(obstacles),
            ray_angles=np.asarray(ray_angles),
        )
        try:
            self._barrier.wait(timeout=self.timeout_seconds)
        except BrokenBarrierError as exc:
            self._raise_if_failed()
            raise RuntimeError("coordinated LiDAR batch did not receive every environment") from exc
        self._raise_if_failed()

    def abort(self, error: BaseException) -> None:
        """Abort pending and future batches after an environment-step failure."""
        self._record_error(error)
        self._barrier.abort()

    def _dispatch(self) -> None:
        """Validate, pad, and dispatch the complete barrier generation."""
        try:
            entries = list(self._entries)
            if any(entry is None for entry in entries):
                raise RuntimeError("coordinated LiDAR batch is missing an environment row")
            rows = [entry for entry in entries if entry is not None]
            first = rows[0]
            num_rays = first.out_ranges.shape
            out_dtype = first.out_ranges.dtype
            ray_dtype = first.ray_angles.dtype
            scanner_dtype = first.scanner_position.dtype
            obstacle_dtype = first.obstacles.dtype
            for row in rows:
                if row.out_ranges.shape != num_rays or row.ray_angles.shape != num_rays:
                    raise ValueError("coordinated LiDAR rows must use one common ray count")
                if row.scanner_position.shape != (2,):
                    raise ValueError("coordinated LiDAR scanner positions must have shape (2,)")
                if row.obstacles.ndim != 2 or row.obstacles.shape[1] != 4:
                    raise ValueError("coordinated LiDAR obstacles must have shape (N, 4)")
                if (
                    row.out_ranges.dtype != out_dtype
                    or row.ray_angles.dtype != ray_dtype
                    or row.scanner_position.dtype != scanner_dtype
                    or row.obstacles.dtype != obstacle_dtype
                ):
                    raise TypeError("coordinated LiDAR rows must use homogeneous dtypes")

            out_ranges = np.stack([row.out_ranges for row in rows])
            scanner_positions = np.stack([row.scanner_position for row in rows])
            ray_angles = np.stack([row.ray_angles for row in rows])
            obstacle_counts = np.asarray(
                [row.obstacles.shape[0] for row in rows],
                dtype=np.int64,
            )
            max_obstacles = int(obstacle_counts.max(initial=0))
            padded_obstacles = np.zeros(
                (self.batch_size, max_obstacles, 4),
                dtype=obstacle_dtype,
            )
            for env_index, row in enumerate(rows):
                padded_obstacles[env_index, : row.obstacles.shape[0]] = row.obstacles

            raycast_obstacles_batch(
                out_ranges,
                scanner_positions,
                padded_obstacles,
                obstacle_counts,
                ray_angles,
            )
            for env_index, row in enumerate(rows):
                row.out_ranges[:] = out_ranges[env_index]
        except BaseException as exc:  # noqa: BLE001 - propagate one batch failure to every worker.
            self._record_error(exc)

    def _record_error(self, error: BaseException) -> None:
        """Retain the first failure as the canonical batch error."""
        with self._error_lock:
            if self._error is None:
                self._error = error

    def _raise_if_failed(self) -> None:
        """Raise the canonical failure for all participating workers."""
        if self._error is not None:
            raise RuntimeError("coordinated LiDAR batch failed") from self._error


@dataclass(frozen=True, slots=True)
class _LidarBatchBinding:
    """Thread-local coordinator binding for one vector-environment worker."""

    coordinator: LidarBatchCoordinator
    env_index: int


_LIDAR_BATCH_CONTEXT = local()
_LIDAR_RANGES_ONLY_CONTEXT = local()


@contextmanager
def lidar_batch_context(
    coordinator: LidarBatchCoordinator,
    env_index: int,
) -> Iterator[None]:
    """Bind a rollout worker's LiDAR calls to a shared batch coordinator."""
    previous = getattr(_LIDAR_BATCH_CONTEXT, "binding", None)
    _LIDAR_BATCH_CONTEXT.binding = _LidarBatchBinding(coordinator, env_index)
    try:
        yield
    finally:
        if previous is None:
            del _LIDAR_BATCH_CONTEXT.binding
        else:
            _LIDAR_BATCH_CONTEXT.binding = previous


@numba.njit(nogil=True)
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

    The scan originates from the scanner's position and considers pedestrians,
    obstacles, and optionally an enemy agent. Rays that miss every object remain
    ``np.inf`` here; ``range_postprocessing`` later clips them to the configured
    maximum scan distance.

    Args:
        scanner_pos: Scanner origin in world coordinates.
        obstacles: Static obstacle segments as an ``(N, 4)`` array of
            ``start_x, start_y, end_x, end_y`` rows.
        max_scan_range: Distance threshold for dynamic circular objects.
        ped_pos: Pedestrian centers as an ``(N, 2)`` array.
        ped_radius: Radius used to approximate pedestrians as circles.
        ray_angles: Absolute ray directions in radians.
        enemy_pos: Optional ``(N, 2)`` array for the robot seen by an
            ego-pedestrian scanner.
        enemy_radius: Radius used for ``enemy_pos`` circles.
        other_robot_circles: Optional dynamic robot circles as ``(N, 3)`` rows
            of ``x, y, radius``.

    Returns:
        numpy.ndarray: Raw per-ray distances with shape ``(num_rays,)``.
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


@numba.njit(fastmath=True, nogil=True)
def range_postprocessing(out_ranges: np.ndarray, scan_noise: np.ndarray, max_scan_dist: float):
    """Clip and optionally corrupt ray ranges in place.

    Args:
        out_ranges: Mutable per-ray distances from ``raycast``. Values larger
            than ``max_scan_dist`` are clipped before noise is applied.
        scan_noise: Two probabilities ``[loss, corruption]``. Loss replaces the
            reading with ``max_scan_dist``; corruption scales the reading by a
            fresh random factor.
        max_scan_dist: Maximum sensor range in world units.

    Notes:
        The input array is modified in place. Callers that need deterministic
        output should pass ``[0.0, 0.0]`` for ``scan_noise``.
    """
    prob_scan_loss, prob_scan_corruption = scan_noise
    for i in range(out_ranges.shape[0]):
        out_ranges[i] = min(out_ranges[i], max_scan_dist)
        if np.random.random() < prob_scan_loss:
            out_ranges[i] = max_scan_dist
        elif np.random.random() < prob_scan_corruption:
            out_ranges[i] = out_ranges[i] * np.random.random()


@lru_cache(maxsize=32)
def lidar_ray_offsets(num_rays: int, visual_angle_portion: float) -> np.ndarray:
    """Return cached heading-relative ray offsets.

    Args:
        num_rays: Number of equally spaced rays.
        visual_angle_portion: Fraction of full circle covered.

    Returns:
        Read-only array of relative ray offsets in the same endpoint convention
        used by :func:`lidar_ray_scan`.
    """
    half_span = np.pi * visual_angle_portion
    angles = np.linspace(-half_span, half_span, num_rays + 1)[:-1]
    angles.flags.writeable = False
    return angles


def _lidar_ray_scan_impl(
    pose: RobotPose,
    occ: ContinuousOccupancy,
    settings: LidarScannerSettings,
    ray_angles_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared scan body for public lidar_ray_scan and lidar_ray_scan_ranges_only.

    Computes ray angles into the provided ``ray_angles_out`` buffer, performs
    raycasting, and applies range postprocessing.

    Args:
        pose: ``((x, y), heading)`` scanner pose in world coordinates and radians.
        occ: Continuous occupancy provider.
        settings: Scanner settings.
        ray_angles_out: Pre-allocated output buffer for ray angles. Must have
            shape ``(settings.num_rays,)`` and dtype float64.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(ranges, ray_angles)`` where ranges are
        per-ray distances and ray_angles are the absolute ray directions.
    """
    (pos_x, pos_y), robot_orient = pose
    scan_noise = settings.scan_noise_array
    scan_dist = settings.max_scan_dist

    ped_pos = occ.pedestrian_coords
    obstacles = occ.obstacle_coords
    dynamic_robot_circles = (
        _dynamic_objects_to_circle_array(occ) if settings.detect_other_robots else None
    )

    np.add(robot_orient, settings.ray_offsets, out=ray_angles_out)
    np.mod(ray_angles_out, 2.0 * np.pi, out=ray_angles_out)

    binding: _LidarBatchBinding | None = getattr(_LIDAR_BATCH_CONTEXT, "binding", None)
    if binding is None:
        enemy_pos = (
            np.array([occ.enemy_coords]) if isinstance(occ, EgoPedContinuousOccupancy) else None
        )
        enemy_radius = occ.enemy_radius if isinstance(occ, EgoPedContinuousOccupancy) else 0.0
        ranges = raycast(
            (pos_x, pos_y),
            obstacles,
            scan_dist,
            ped_pos,
            occ.ped_radius,
            ray_angles_out,
            enemy_pos=enemy_pos,
            enemy_radius=enemy_radius,
            other_robot_circles=dynamic_robot_circles,
        )
    else:
        ranges = np.full(ray_angles_out.shape[0], np.inf)
        raycast_pedestrians(
            ranges,
            (pos_x, pos_y),
            scan_dist,
            ped_pos,
            occ.ped_radius,
            ray_angles_out,
        )
        binding.coordinator.submit(
            binding.env_index,
            ranges,
            np.asarray((pos_x, pos_y)),
            obstacles,
            ray_angles_out,
        )
        if isinstance(occ, EgoPedContinuousOccupancy):
            raycast_pedestrians(
                ranges,
                (pos_x, pos_y),
                scan_dist,
                np.array([occ.enemy_coords]),
                occ.enemy_radius,
                ray_angles_out,
            )
        if dynamic_robot_circles is not None:
            raycast_circles(
                ranges,
                (pos_x, pos_y),
                scan_dist,
                dynamic_robot_circles,
                ray_angles_out,
            )
    range_postprocessing(ranges, scan_noise, scan_dist)
    return ranges, ray_angles_out


def lidar_ray_scan(
    pose: RobotPose,
    occ: ContinuousOccupancy,
    settings: LidarScannerSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a radial LiDAR scan on a continuous occupancy with objects.

    The occupancy contains the robot (as circle), a set of pedestrians
    (as circles) and a set of static obstacles (as 2D lines).

    Args:
        pose: ``((x, y), heading)`` scanner pose in world coordinates and
            radians.
        occ: Continuous occupancy provider exposing pedestrian coordinates,
            obstacle segments, and optionally dynamic robot circles.
        settings: Scanner settings controlling range, field of view, ray count,
            noise, and dynamic-object detection.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A pair ``(ranges, ray_angles)`` where
        ``ranges`` are per-ray distances with shape ``(settings.num_rays,)`` and
        ``ray_angles`` are absolute ray angles in radians with the same shape.
    """
    ray_angles = np.empty(settings.num_rays, dtype=np.float64)
    ranges, ray_angles = _lidar_ray_scan_impl(pose, occ, settings, ray_angles)
    return ranges, ray_angles


def lidar_ray_scan_ranges_only(
    pose: RobotPose,
    occ: ContinuousOccupancy,
    settings: LidarScannerSettings,
) -> np.ndarray:
    """Simulate a radial LiDAR scan returning only ranges, no ray angles.

    This lightweight variant of :func:`lidar_ray_scan` reuses a thread-local
    work buffer to avoid allocating a new ray_angles array on every call.
    The buffer is isolated per thread so parallel environment observations
    cannot overwrite each other's angles.

    Args:
        pose: ``((x, y), heading)`` scanner pose in world coordinates and radians.
        occ: Continuous occupancy provider.
        settings: Scanner settings controlling range, field of view, ray count,
            noise, and dynamic-object detection.

    Returns:
        numpy.ndarray: Per-ray distances with shape ``(settings.num_rays,)``.
    """
    ray_angles = getattr(_LIDAR_RANGES_ONLY_CONTEXT, "ray_angles", None)
    if ray_angles is None or ray_angles.shape != (settings.num_rays,):
        ray_angles = np.empty(settings.num_rays, dtype=np.float64)
        _LIDAR_RANGES_ONLY_CONTEXT.ray_angles = ray_angles
    ranges, _ray_angles = _lidar_ray_scan_impl(pose, occ, settings, ray_angles)
    return ranges


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
