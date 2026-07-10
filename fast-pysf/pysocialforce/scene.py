"""This module tracks the state of scene and scene elements
like pedestrians, groups and obstacles"""

from dataclasses import dataclass, field
from math import atan2, cos, pi, sin

import numpy as np

from pysocialforce.config import SceneConfig

Line2D = tuple[float, float, float, float]
Point2D = tuple[float, float]
Group = list[int]

EXPLICIT_EULER = "explicit_euler"
SEMI_IMPLICIT_EULER = "semi_implicit_euler"
SUPPORTED_INTEGRATION_SCHEMES = frozenset({EXPLICIT_EULER, SEMI_IMPLICIT_EULER})

#: Default spread (m/s) for the desired-speed distribution when a mean is
#: configured without an explicit standard deviation.
DEFAULT_DESIRED_SPEED_STD = 0.2


def sample_truncated_normal_speeds(
    num_peds: int,
    mean: float,
    std: float,
    high: float,
    seed: int | None = None,
) -> np.ndarray:
    """Sample non-negative desired walking speeds from a truncated normal.

    Speeds are drawn from ``N(mean, std)`` and truncated to ``[0, high]``. This
    decouples the preferred walking speed from the spawn velocity (issue #4972).
    Negative samples are clipped to ``0`` and samples above ``high`` are clipped
    to ``high``; for literature-calibrated means (~1.3 m/s) with a 0.2 m/s spread
    the truncation is negligible in practice.

    Args:
        num_peds: Number of pedestrian desired speeds to sample.
        mean: Desired-speed distribution mean in m/s.
        std: Desired-speed distribution standard deviation in m/s.
        high: Inclusive upper bound for the truncated distribution (m/s).
        seed: Optional RNG seed for deterministic sampling.

    Returns:
        np.ndarray: Non-negative desired speeds with shape ``(num_peds,)``.
    """
    if not np.isfinite(mean) or mean < 0:
        raise ValueError("desired speed mean must be finite and non-negative")
    if not np.isfinite(std) or std < 0:
        raise ValueError("desired speed std must be finite and non-negative")
    if not np.isfinite(high) or high < 0:
        raise ValueError("desired speed high bound must be finite and non-negative")
    if num_peds <= 0:
        return np.zeros((0,), dtype=float)
    rng = np.random.default_rng(seed)
    std_eff = std if std is not None and std > 0 else 0.0
    if std_eff > 0.0:
        speeds = rng.normal(loc=mean, scale=std_eff, size=num_peds)
    else:
        speeds = np.full(num_peds, float(mean), dtype=float)
    return np.clip(speeds, 0.0, float(high))


def normalize_integration_scheme(value: str | None) -> str:
    """Return a supported pedestrian integration scheme or raise a clear error."""
    if value is None:
        value = SEMI_IMPLICIT_EULER
    normalized = str(value).strip()
    if normalized in SUPPORTED_INTEGRATION_SCHEMES:
        return normalized
    supported = ", ".join(sorted(SUPPORTED_INTEGRATION_SCHEMES))
    raise ValueError(f"Unsupported integration_scheme {value!r}. Supported values: {supported}.")


class PedState:
    """Track pedestrian kinematic state and optional social groups."""

    def __init__(self, state: np.ndarray, groups: list[Group], config: SceneConfig):
        """Initialize pedestrian state from raw simulator arrays.

        Args:
            state: Pedestrian state matrix ``(N, 6|7)`` with position/velocity/goal.
            groups: Pedestrian grouping indices.
            config: Scene configuration with integration and speed settings.
        """
        self.default_tau = config.tau
        self.d_t = config.dt_secs
        self.integration_scheme = normalize_integration_scheme(config.integration_scheme)
        self.agent_radius = config.agent_radius
        self.max_speed_multiplier = config.max_speed_multiplier
        self.desired_speed_mean = config.desired_speed_mean
        self.desired_speed_std = config.desired_speed_std
        self.desired_speed_high = config.desired_speed_high
        self.desired_speed_seed = config.desired_speed_seed

        self.max_speeds: np.ndarray | None = None
        self.initial_speeds: np.ndarray | None = None
        # Explicitly assigned desired speeds decouple the goal-driving speed
        # (``max_speeds``) from the spawn speed (issue #4972). When ``None`` the
        # legacy ``max_speed_multiplier * initial_speed`` derivation is used.
        self._explicit_desired_speeds: np.ndarray | None = None
        self.update(state, groups)
        # After the initial state is cached, optionally sample a decoupled
        # desired-speed distribution from the scene configuration.
        if self.desired_speed_mean is not None:
            std = (
                self.desired_speed_std
                if self.desired_speed_std is not None
                else DEFAULT_DESIRED_SPEED_STD
            )
            self.assign_desired_speeds(
                sample_truncated_normal_speeds(
                    self.size(),
                    float(self.desired_speed_mean),
                    float(std),
                    float(self.desired_speed_high),
                    seed=self.desired_speed_seed,
                )
            )

    def update(self, state: np.ndarray, groups: list[list[int]]) -> None:
        """Update pedestrian state and group memberships.

        Args:
            state: Updated pedestrian state matrix.
            groups: Updated pedestrian groups.
        """
        self.state = state
        self.groups = groups

    def assign_desired_speeds(self, desired_speeds: np.ndarray | None) -> None:
        """Assign per-pedestrian desired speeds decoupled from the spawn speed.

        When set, these speeds drive the goal-directed force (``max_speeds``)
        instead of the legacy ``max_speed_multiplier * initial_speed`` derivation.
        Pass ``None`` (or call :meth:`clear_desired_speeds`) to restore the
        legacy behavior. The assignment is length-checked against the current
        pedestrian count and re-applied on every state refresh so it survives
        repeated integration steps (issue #4972).

        Args:
            desired_speeds: Non-negative desired speeds in m/s, one per pedestrian.
        """
        if desired_speeds is None:
            self.clear_desired_speeds()
            return
        speeds = np.asarray(desired_speeds, dtype=float)
        if speeds.ndim != 1 or speeds.shape[0] != self.size():
            raise ValueError(
                "desired_speeds must be a 1D array with one entry per pedestrian "
                f"(got shape {speeds.shape} for {self.size()} pedestrians)"
            )
        if not np.all(np.isfinite(speeds)) or np.any(speeds < 0):
            raise ValueError("desired_speeds must be finite and non-negative")
        self._explicit_desired_speeds = speeds
        self._refresh_max_speeds()

    def clear_desired_speeds(self) -> None:
        """Restore the legacy spawn-coupled desired-speed derivation.

        Clears any explicit desired speeds and recomputes ``max_speeds`` from
        ``max_speed_multiplier * initial_speed`` (issue #4972).
        """
        self._explicit_desired_speeds = None
        self._refresh_max_speeds()

    def _refresh_max_speeds(self) -> None:
        """Refresh the goal-driving cap from explicit or legacy desired speeds."""
        if self._explicit_desired_speeds is not None:
            self.max_speeds = self._explicit_desired_speeds.copy()
        else:
            self.max_speeds = self.max_speed_multiplier * self.initial_speeds

    def _update_state(self, state: np.ndarray) -> None:
        """Normalize and cache state-derived fields.

        When explicit desired speeds are assigned they drive ``max_speeds``
        directly (decoupled from spawn speed, issue #4972); otherwise the legacy
        ``max_speed_multiplier * initial_speed`` derivation is preserved.

        Args:
            state: Pedestrian state matrix with or without explicit ``tau`` column.
        """
        tau = np.full((state.shape[0]), self.default_tau)
        if state.shape[1] < 7:
            self._state = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1)
        else:
            self._state = state
        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        self._refresh_max_speeds()

    @property
    def state(self) -> np.ndarray:
        """Return the current pedestrian state matrix."""
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """Set pedestrian state and refresh derived caches.

        Args:
            state: Pedestrian state matrix.
        """
        self._update_state(state)

    def get_states(self) -> tuple[np.ndarray, list[list[list[int]]]]:
        """Return a batched snapshot of state and group assignments."""
        return np.array([self.state]), [self.groups]

    def size(self) -> int:
        """Return the number of pedestrians in the scene."""
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        """Return pedestrian positions as ``(N, 2)`` array."""
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        """Return pedestrian velocities as ``(N, 2)`` array."""
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        """Return pedestrian goal positions as ``(N, 2)`` array."""
        return self.state[:, 4:6]

    def tau(self) -> np.ndarray:
        """Return relaxation-time values as ``(N, 1)`` column vector."""
        return self.state[:, 6:7]

    def speeds(self) -> np.ndarray:
        """Return the speeds corresponding to a given state."""
        return np.linalg.norm(self.vel(), axis=1)

    def step(self, force, groups=None):
        """Advance pedestrians by one integration step.

        Args:
            force: Per-pedestrian force vectors.
            groups: Optional updated group assignments.
        """
        previous_velocity = self.vel().copy()
        desired_velocity = previous_velocity + self.d_t * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        # stop when arrived
        # desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]

        # update state
        next_state = self.state
        position_velocity = (
            previous_velocity if self.integration_scheme == EXPLICIT_EULER else desired_velocity
        )
        next_state[:, 0:2] += position_velocity * self.d_t
        next_state[:, 2:4] = desired_velocity
        next_groups = groups if groups is not None else self.groups
        self.update(next_state, next_groups)

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down desired velocities to their capped speeds.

        Args:
            desired_velocity: Array of desired velocity vectors.
            max_velocity: Maximum allowed speed per pedestrian.

        Returns:
            np.ndarray: Capped velocity vectors.
        """
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.zeros_like(desired_speeds, dtype=float)
        np.divide(max_velocity, desired_speeds, out=factor, where=desired_speeds > 0.0)
        np.minimum(factor, 1.0, out=factor)
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> list[list]:
        """Return grouped pedestrian indices."""
        return self._groups

    @groups.setter
    def groups(self, groups: list[list]):
        """Set grouped pedestrian indices.

        Args:
            groups: Group definitions; ``None`` maps to empty groups.
        """
        if groups is None:
            self._groups = []
        else:
            self._groups = groups

    def has_group(self) -> bool:
        """Return whether group metadata is present."""
        return self.groups is not None

    def which_group(self, index: int) -> int:
        """Return the group index for a pedestrian id.

        Args:
            index: Pedestrian index.

        Returns:
            int: Group index, or -1 if not found.
        """
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1


@dataclass
class EnvState:
    """State of the environment obstacles"""

    _orig_obstacles: list[Line2D]
    _resolution: float = 10
    _obstacles_linspace: list[np.ndarray] = field(init=False)
    _obstacles_raw: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize cached obstacle representations."""
        self._obstacles_raw = self._update_obstacles_raw(self._orig_obstacles)
        self._obstacles_linspace = self._update_obstacles_linspace(self._orig_obstacles)

    @property
    def obstacles_raw(self) -> np.ndarray:
        """a 2D numpy array representing a list of 2D lines
        as (start_x, start_y, end_x, end_y) for array indices 0-3.
        Additionally, the array contains the orthogonal unit vector
        for each 2D line at indices 4-5."""
        return self._obstacles_raw

    @property
    def obstacles(self) -> list[np.ndarray]:
        """a list of np.ndarrays, each representing a uniform
        linspace of 0.1 steps between |p_start, p_end|"""
        return self._obstacles_linspace

    @obstacles.setter
    def obstacles(self, obstacles: list[Line2D]):
        """Input an list of (start_x, end_x, start_y, end_y) as start and end of a line"""
        self._orig_obstacles = obstacles
        self._obstacles_raw = self._update_obstacles_raw(obstacles)
        self._obstacles_linspace = self._update_obstacles_linspace(obstacles)

    def _update_obstacles_linspace(self, obs_lines: list[Line2D]) -> list[np.ndarray]:
        """Convert obstacle segments into sampled polyline points.

        Args:
            obs_lines: Obstacle line segments.

        Returns:
            list[np.ndarray]: Sampled points for each obstacle segment.
        """
        if obs_lines is None:
            obstacles = []
        else:
            obstacles = []
            for start_x, end_x, start_y, end_y in obs_lines:
                samples = int(np.linalg.norm((start_x - end_x, start_y - end_y)) * self._resolution)
                line = np.array(
                    list(
                        zip(
                            np.linspace(start_x, end_x, samples),
                            np.linspace(start_y, end_y, samples),
                            strict=False,
                        )
                    )
                )
                obstacles.append(line)
        return obstacles

    def _update_obstacles_raw(self, obs_lines: list[Line2D]) -> np.ndarray:
        """Convert obstacle line segments into the raw obstacle array.

        Args:
            obs_lines: Line segments describing obstacles.

        Returns:
            np.ndarray: Array of line segments with orthogonal unit vectors.
        """

        def orient(line):
            """Compute the orientation of a line segment.

            Args:
                line: Line segment as (start_x, end_x, start_y, end_y).

            Returns:
                float: Orientation angle in radians.
            """
            start_x, end_x, start_y, end_y = line
            vec_x, vec_y = end_x - start_x, end_y - start_y
            return (atan2(vec_y, vec_x) + 2 * pi) % (2 * pi)

        def unit_vec(orient):
            """Convert an orientation angle to a unit vector.

            Args:
                orient: Orientation angle in radians.

            Returns:
                tuple[float, float]: Unit vector components.
            """
            return cos(orient), sin(orient)

        if obs_lines is None or len(obs_lines) == 0:
            return np.array([])

        line_orients = np.array([orient(line) for line in obs_lines])
        ortho_orients = (line_orients + pi / 2) % (2 * pi)
        ortho_vecs = np.array([unit_vec(orient) for orient in ortho_orients])

        obstacles = np.zeros((len(obs_lines), 6))
        obstacles[:, :4] = [
            [start_x, start_y, end_x, end_y] for start_x, end_x, start_y, end_y in obs_lines
        ]
        obstacles[:, 4:] = ortho_vecs

        return obstacles
