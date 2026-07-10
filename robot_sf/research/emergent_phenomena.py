"""Reproducible emergent-phenomena demonstration harness for the released
pedestrian substrate (the bundled ``fast-pysf`` / PySocialForce model).

This module builds small, self-contained pedestrian scenarios directly on the
released simulator (``pysocialforce.Simulator``), runs them at the *released*
default speed calibration and at a *literature-typical* speed calibration, and
computes simple, interpretable order parameters for the three canonical
emergent phenomena of crowd dynamics:

- **Lane formation** in bidirectional corridor flow.
- **Oscillation** (alternating direction) at a narrow doorway bottleneck.
- **Arching / clogging** at a high-density exit.

Design notes
------------
The harness is intentionally substrate-faithful: it does **not** patch the
force stack or the speed-derivation logic. It reproduces the released default
speed regime the way the released code derives it -- ``max_speeds =
max_speed_multiplier * initial_speeds`` (see ``fast-pysf`` ``scene.py`` and
issue robot_sf_ll7#4972) -- by setting each pedestrian's initial velocity
magnitude to ``desired_speed / max_speed_multiplier`` along its goal direction.
The released default uses ``initial_speed = 0.5`` m/s with multiplier ``1.3``,
i.e. a desired speed of ≈0.65 m/s; the literature-typical calibration targets
≈1.3 m/s (Moussaïd et al. 2010, doi:10.1371/journal.pone.0010047).

Everything is deterministic given a seed: the only randomness is pedestrian
placement, drawn from a seeded ``numpy.random.default_rng``.

The order parameters are deliberately simple and documented so they can double
as a regression anchor for force-model changes (e.g. the anticipatory variant
tracked in robot_sf_ll7#4973) and as a precursor to the empirical validation
harness in robot_sf_ll7#4975. They are diagnostic behavioral-validity signals,
not paper-grade validation against real trajectory data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pysocialforce as pysf
from pysocialforce.config import (
    ObstacleForceConfig,
    SceneConfig,
    SimulatorConfig,
    SocialForceConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_LITERATURE_DESIRED_SPEED",
    "DEFAULT_RELEASED_DESIRED_SPEED",
    "INITIAL_SPEED_RELEASED",
    "MAX_SPEED_MULTIPLIER",
    "EmergentPhenomenaReport",
    "ScenarioConfig",
    "ScenarioResult",
    "SpeedCalibration",
    "TrajectoryRecord",
    "build_bidirectional_corridor",
    "build_high_density_exit",
    "build_narrow_doorway",
    "doorway_oscillation",
    "exit_arching",
    "lane_purity",
    "lane_segregation_index",
    "released_default_config",
    "run_emergent_phenomena_demo",
    "run_scenario",
]

# Released-default speed derivation (fast-pysf config.py / scene.py).
# initial_speed = 0.5 m/s, max_speed_multiplier = 1.3  -> desired ~= 0.65 m/s.
INITIAL_SPEED_RELEASED: float = 0.5
MAX_SPEED_MULTIPLIER: float = 1.3
DEFAULT_RELEASED_DESIRED_SPEED: float = INITIAL_SPEED_RELEASED * MAX_SPEED_MULTIPLIER
# Literature-typical unimpeded adult walking speed (Moussaid et al. 2010).
DEFAULT_LITERATURE_DESIRED_SPEED: float = 1.3

# Released-default force factors (fast-pysf config.py SimulatorConfig defaults) are
# materialized explicitly by ``released_default_config()`` below so the demo's
# parameterization is self-describing and pinned even if upstream defaults drift.


def released_default_config() -> SimulatorConfig:
    """Return a ``SimulatorConfig`` matching the released fast-pysf defaults.

    The defaults are the dataclass defaults of ``pysocialforce.config``; this
    helper materializes them explicitly so the demonstration's parameterization
    is self-describing and pinned even if upstream defaults drift.

    Returns:
        SimulatorConfig: the released-default configuration (slow speed regime,
        default force factors, groups disabled for the substrate-only demo).
    """
    return SimulatorConfig(
        scene_config=SceneConfig(
            enable_group=False,
            agent_radius=0.35,
            dt_secs=0.1,
            max_speed_multiplier=MAX_SPEED_MULTIPLIER,
            tau=0.5,
            resolution=10,
        ),
        social_force_config=SocialForceConfig(
            factor=5.1,
            lambda_importance=2.0,
            gamma=0.35,
            n=2,
            n_prime=3,
            activation_threshold=20.0,
        ),
        obstacle_force_config=ObstacleForceConfig(factor=10.0, sigma=0.0, threshold=-0.57),
    )


@dataclass(frozen=True)
class SpeedCalibration:
    """A named desired-speed calibration for a demonstration run.

    Attributes:
        name: Human-readable label, e.g. ``"released_default"``.
        desired_speed_mean: Target mean desired walking speed (m/s).
        desired_speed_std: Per-pedestrian standard deviation (m/s); a small
            spread avoids lattice artifacts while keeping the population
            centered on the calibration.
    """

    name: str
    desired_speed_mean: float
    desired_speed_std: float = 0.0


# Canonical calibrations used by the demonstration.
RELEASED_DEFAULT_CALIBRATION = SpeedCalibration(
    name="released_default",
    desired_speed_mean=DEFAULT_RELEASED_DESIRED_SPEED,
    desired_speed_std=0.0,
)
LITERATURE_CALIBRATION = SpeedCalibration(
    name="literature_typical",
    desired_speed_mean=DEFAULT_LITERATURE_DESIRED_SPEED,
    desired_speed_std=0.2,
)


@dataclass(frozen=True)
class ScenarioConfig:
    """Geometry + population parameters for one emergent-phenomenon scenario.

    Attributes:
        name: Scenario identifier (``"bidirectional_corridor"`` etc.).
        length: Corridor/room extent along the primary travel axis (m).
        half_width: Half-width of the walkable corridor/room (m).
        n_pedestrians: Number of pedestrians to spawn.
        seed: RNG seed for deterministic placement.
        n_steps: Number of simulator steps to run.
        extra: Scenario-specific parameters (door position, exit width, ...).
    """

    name: str
    length: float
    half_width: float
    n_pedestrians: int
    seed: int
    n_steps: int
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
    """Recorded per-step pedestrian state for one scenario run.

    Attributes:
        positions: Array of shape ``(T, N, 2)`` of pedestrian positions.
        velocities: Array of shape ``(T, N, 2)`` of pedestrian velocities.
        desired_directions: Array of shape ``(N, 2)`` unit desired directions
            (sign of the primary-axis goal), captured at spawn time.
        times: Array of shape ``(T,)`` simulation times in seconds.
        dt: Integration step in seconds.
    """

    positions: np.ndarray
    velocities: np.ndarray
    desired_directions: np.ndarray
    times: np.ndarray
    dt: float


@dataclass
class ScenarioResult:
    """Result of running one scenario under one speed calibration.

    Attributes:
        scenario: The scenario configuration that was run.
        calibration: The speed calibration that was used.
        trajectory: Recorded trajectory.
        order_parameters: Computed order parameters for the phenomenon.
        max_speeds: Per-pedestrian desired (capped) speeds actually used.
    """

    scenario: ScenarioConfig
    calibration: SpeedCalibration
    trajectory: TrajectoryRecord
    order_parameters: dict[str, float]
    max_speeds: np.ndarray


@dataclass
class EmergentPhenomenaReport:
    """Aggregate report across all scenarios and calibrations.

    Attributes:
        results: Flat list of per-scenario/per-calibration results.
        substrate_version: ``pysocialforce.__version__`` for provenance.
        config_json: JSON-serializable snapshot of released-default config.
    """

    results: list[ScenarioResult]
    substrate_version: str
    config_json: dict


# --------------------------------------------------------------------------- #
# Geometry / scenario builders
# --------------------------------------------------------------------------- #


def _sample_desired_speeds(
    rng: np.random.Generator, n: int, calibration: SpeedCalibration
) -> np.ndarray:
    """Draw per-pedestrian desired speeds from a (truncated) normal distribution.

    Speeds are clipped to a positive, physical range ``[0.3, 2.0]`` m/s so a
    large draw never produces an unstable or negative target.

    Returns:
        Array of shape ``(n,)`` desired speeds in m/s.
    """
    speeds = rng.normal(
        loc=calibration.desired_speed_mean,
        scale=calibration.desired_speed_std,
        size=n,
    )
    return np.clip(speeds, 0.3, 2.0)


def _desired_to_initial(v_desired: np.ndarray) -> np.ndarray:
    """Convert desired speeds to initial-velocity magnitudes for the substrate.

    The released substrate derives ``max_speeds = max_speed_multiplier *
    initial_speeds`` (``scene.py``), so to realize a target desired speed we set
    the spawn velocity magnitude to ``desired / max_speed_multiplier``. This
    faithfully exercises the released speed-derivation logic rather than
    bypassing it.

    Returns:
        Array of initial-velocity magnitudes (m/s).
    """
    return v_desired / MAX_SPEED_MULTIPLIER


def _state_row(
    x: float,
    y: float,
    vx: float,
    vy: float,
    gx: float,
    gy: float,
    tau: float = 0.5,
) -> list[float]:
    """Build one pedestrian state row ``[x, y, vx, vy, gx, gy, tau]``.

    Returns:
        A 7-element state row for the PySocialForce state matrix.
    """
    return [x, y, vx, vy, gx, gy, tau]


def build_bidirectional_corridor(
    config: ScenarioConfig, calibration: SpeedCalibration
) -> tuple[np.ndarray, list[tuple[float, float, float, float]], np.ndarray]:
    """Build a bidirectional corridor: two opposing flows that must pass each other.

    The corridor runs along ``x`` with walls at ``y = +/- half_width``. Half the
    pedestrians walk toward ``+x`` and half toward ``-x``, randomly placed in
    the first and last third of the corridor respectively. Lane formation is the
    tendency for same-direction pedestrians to cluster into lateral (``y``)
    bands so the two flows separate rather than interpenetrate.

    Args:
        config: Scenario configuration (uses ``length``, ``half_width``,
            ``n_pedestrians``, ``seed``).
        calibration: Desired-speed calibration.

    Returns:
        Tuple of ``(state, obstacles, desired_directions)`` where ``state`` has
        shape ``(N, 7)``, ``obstacles`` is the wall line segments, and
        ``desired_directions`` has shape ``(N, 2)`` unit vectors.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_pedestrians
    n_per_dir = n // 2
    n_extra = n - n_per_dir
    speeds = _sample_desired_speeds(rng, n, calibration)
    init_mags = _desired_to_initial(speeds)
    length = config.length
    hw = config.half_width
    # Keep peds off the walls.
    y_span = hw * 0.75

    rows: list[list[float]] = []
    dirs: list[tuple[float, float]] = []

    # Rightward flow: spawn in the first third, goal at +x end.
    xs = rng.uniform(0.5, length / 3.0, n_per_dir)
    ys = rng.uniform(-y_span, y_span, n_per_dir)
    for i in range(n_per_dir):
        rows.append(_state_row(xs[i], ys[i], init_mags[i], 0.0, length, ys[i]))
        dirs.append((1.0, 0.0))

    # Leftward flow: spawn in the last third, goal at -x end (x=0).
    xs = rng.uniform((2.0 / 3.0) * length, length - 0.5, n_extra)
    ys = rng.uniform(-y_span, y_span, n_extra)
    for i in range(n_extra):
        rows.append(_state_row(xs[i], ys[i], -init_mags[n_per_dir + i], 0.0, 0.0, ys[i]))
        dirs.append((-1.0, 0.0))

    state = np.array(rows, dtype=float)
    obstacles = [
        (-1.0, hw, length + 1.0, hw),  # top wall
        (-1.0, -hw, length + 1.0, -hw),  # bottom wall
    ]
    desired_directions = np.array(dirs, dtype=float)
    return state, obstacles, desired_directions


def build_narrow_doorway(
    config: ScenarioConfig, calibration: SpeedCalibration
) -> tuple[np.ndarray, list[tuple[float, float, float, float]], np.ndarray]:
    """Build a corridor with a narrow doorway bottleneck at ``x = door_x``.

    A bidirectional crowd must squeeze through a gap of width
    ``2 * door_half_width``. At bottlenecks, bidirectional pedestrian flow is
    known to oscillate: one direction dominates for a burst, then the other.

    Args:
        config: Scenario configuration; ``extra`` may carry ``door_x`` (default
            ``length/2``) and ``door_half_width`` (default ``0.6`` m).
        calibration: Desired-speed calibration.

    Returns:
        Tuple of ``(state, obstacles, desired_directions)``.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_pedestrians
    n_per_dir = n // 2
    n_extra = n - n_per_dir
    speeds = _sample_desired_speeds(rng, n, calibration)
    init_mags = _desired_to_initial(speeds)
    length = config.length
    hw = config.half_width
    door_x = float(config.extra.get("door_x", length / 2.0))
    door_half = float(config.extra.get("door_half_width", 0.6))
    y_span = hw * 0.7

    rows: list[list[float]] = []
    dirs: list[tuple[float, float]] = []

    xs = rng.uniform(0.5, door_x - 1.5, n_per_dir)
    ys = rng.uniform(-y_span, y_span, n_per_dir)
    for i in range(n_per_dir):
        rows.append(_state_row(xs[i], ys[i], init_mags[i], 0.0, length, ys[i]))
        dirs.append((1.0, 0.0))

    xs = rng.uniform(door_x + 1.5, length - 0.5, n_extra)
    ys = rng.uniform(-y_span, y_span, n_extra)
    for i in range(n_extra):
        rows.append(_state_row(xs[i], ys[i], -init_mags[n_per_dir + i], 0.0, 0.0, ys[i]))
        dirs.append((-1.0, 0.0))

    state = np.array(rows, dtype=float)
    # Corridor walls + the two door jambs (wall segments closing the gap edges).
    obstacles = [
        (-1.0, hw, length + 1.0, hw),
        (-1.0, -hw, length + 1.0, -hw),
        (door_x, door_half, door_x, hw + 1.0),  # upper jamb
        (door_x, -door_half, door_x, -hw - 1.0),  # lower jamb
    ]
    desired_directions = np.array(dirs, dtype=float)
    return state, obstacles, desired_directions


def build_high_density_exit(
    config: ScenarioConfig, calibration: SpeedCalibration
) -> tuple[np.ndarray, list[tuple[float, float, float, float]], np.ndarray]:
    """Build a high-density room with a single narrow exit on the ``+x`` wall.

    Pedestrians fill a square room and all head for one narrow exit; at high
    density this is the classic arching/clogging setup (a semicircular arch
    forms at the exit and throughput drops below free flow).

    Args:
        config: Scenario configuration; ``extra`` may carry ``exit_half_width``
            (default ``0.6`` m).
        calibration: Desired-speed calibration.

    Returns:
        Tuple of ``(state, obstacles, desired_directions)``.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_pedestrians
    speeds = _sample_desired_speeds(rng, n, calibration)
    init_mags = _desired_to_initial(speeds)
    length = config.length
    hw = config.half_width
    exit_half = float(config.extra.get("exit_half_width", 0.6))
    # Pack pedestrians into the left ~70% of the room, away from the exit.
    pack_x = length * 0.7
    y_span = hw * 0.85

    rows: list[list[float]] = []
    dirs: list[tuple[float, float]] = []
    # Use a jittered-grid placement to reach high density without overlaps
    # landing exactly on top of each other (which the SFM handles poorly).
    cols = int(np.ceil(np.sqrt(n * pack_x / (2 * y_span)))) or 1
    xs_grid = np.linspace(0.8, pack_x, cols)
    ys_grid = np.linspace(-y_span, y_span, max(1, int(np.ceil(n / cols))))
    gx, gy = np.meshgrid(xs_grid, ys_grid)
    flat_x = gx.ravel()[:n] + rng.uniform(-0.15, 0.15, n)
    flat_y = gy.ravel()[:n] + rng.uniform(-0.15, 0.15, n)
    # Each pedestrian aims at the exit center (x=length, y=0).
    for i in range(n):
        rows.append(_state_row(flat_x[i], flat_y[i], init_mags[i], 0.0, length, 0.0))
        dirs.append((1.0, 0.0))

    state = np.array(rows, dtype=float)
    # Room walls with a gap on the +x wall for the exit.
    obstacles = [
        (-1.0, hw, length + 2.0, hw),  # top
        (-1.0, -hw, length + 2.0, -hw),  # bottom
        (-1.0, -hw, -1.0, hw),  # left (closed back wall)
        (length, exit_half, length, hw + 1.0),  # upper exit jamb
        (length, -exit_half, length, -hw - 1.0),  # lower exit jamb
    ]
    desired_directions = np.array(dirs, dtype=float)
    return state, obstacles, desired_directions


_BUILDERS = {
    "bidirectional_corridor": build_bidirectional_corridor,
    "narrow_doorway": build_narrow_doorway,
    "high_density_exit": build_high_density_exit,
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_scenario(
    config: ScenarioConfig,
    calibration: SpeedCalibration,
    sim_config: SimulatorConfig | None = None,
) -> ScenarioResult:
    """Build, run, and measure one scenario under one speed calibration.

    Args:
        config: Scenario configuration.
        calibration: Speed calibration.
        sim_config: Optional simulator config; defaults to the released defaults
            via :func:`released_default_config`.

    Returns:
        ScenarioResult with the recorded trajectory and order parameters.
    """
    if sim_config is None:
        sim_config = released_default_config()
    builder = _BUILDERS.get(config.name)
    if builder is None:
        raise ValueError(
            f"Unknown scenario {config.name!r}; expected one of {sorted(_BUILDERS)}",
        )

    state, obstacles, desired_directions = builder(config, calibration)
    sim = pysf.Simulator(state=state.copy(), obstacles=obstacles, config=sim_config)
    dt = sim_config.scene_config.dt_secs
    max_speeds = np.asarray(sim.peds.max_speeds, dtype=float).copy()

    n = state.shape[0]
    positions = np.empty((config.n_steps + 1, n, 2), dtype=float)
    velocities = np.empty((config.n_steps + 1, n, 2), dtype=float)
    positions[0] = sim.peds.pos()
    velocities[0] = sim.peds.vel()
    for t in range(1, config.n_steps + 1):
        sim.step()
        positions[t] = sim.peds.pos()
        velocities[t] = sim.peds.vel()

    times = np.arange(config.n_steps + 1) * dt
    trajectory = TrajectoryRecord(
        positions=positions,
        velocities=velocities,
        desired_directions=desired_directions,
        times=times,
        dt=dt,
    )
    order_parameters = _compute_order_parameters(config.name, trajectory, config)
    return ScenarioResult(
        scenario=config,
        calibration=calibration,
        trajectory=trajectory,
        order_parameters=order_parameters,
        max_speeds=max_speeds,
    )


# --------------------------------------------------------------------------- #
# Order parameters
# --------------------------------------------------------------------------- #


def lane_segregation_index(trajectory: TrajectoryRecord) -> float:
    """Mean |Pearson correlation| between travel direction and lateral position.

    For each timestep, compute the correlation across pedestrians between the
    sign of their desired primary-axis direction (``+1`` for ``+x``, ``-1`` for
    ``-x``) and their lateral (``y``) position. Strong lane formation -> same
    direction clusters at one ``y`` band -> ``|corr|`` close to 1. Well-mixed
    flow -> ``|corr|`` near 0.

    The returned value is the mean of ``|corr|`` over a steady-state window
    (the second half of the run, after transients settle), in ``[0, 1]``.

    Returns:
        Lane segregation index in ``[0, 1]`` (higher = stronger lane formation).
    """
    pos = trajectory.positions
    dirs = trajectory.desired_directions[:, 0]  # +/-1 along x
    n_steps = pos.shape[0]
    start = n_steps // 2
    corrs: list[float] = []
    for t in range(start, n_steps):
        y = pos[t, :, 1]
        if y.std() < 1e-9:
            continue
        c = float(np.corrcoef(dirs, y)[0, 1])
        if np.isfinite(c):
            corrs.append(abs(c))
    return float(np.mean(corrs)) if corrs else 0.0


def lane_purity(trajectory: TrajectoryRecord, n_bins: int = 8) -> float:
    """Binned lane purity: how strongly each lateral band is directionally dominated.

    Bin the corridor width into ``n_bins`` lateral (``y``) bands; for each band
    compute the fraction of pedestrians whose desired direction is ``+x``. Lane
    purity per band is ``|2*frac - 1|`` (0 = perfectly mixed, 1 = single
    direction). The index is the occupancy-weighted mean purity over the
    steady-state window, in ``[0, 1]``.

    Returns:
        Lane purity in ``[0, 1]`` (higher = stronger separation).
    """
    pos = trajectory.positions
    dirs = trajectory.desired_directions[:, 0]
    n_steps = pos.shape[0]
    start = n_steps // 2
    ys = pos[start:, :, 1]
    y_min, y_max = ys.min(), ys.max()
    if y_max - y_min < 1e-9:
        return 0.0
    edges = np.linspace(y_min, y_max + 1e-6, n_bins + 1)
    purities: list[float] = []
    weights: list[float] = []
    for t in range(start, n_steps):
        y = pos[t, :, 1]
        idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            mask = idx == b
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            frac_plus = float((dirs[mask] > 0).mean())
            purities.append(abs(2 * frac_plus - 1))
            weights.append(cnt)
    if not purities:
        return 0.0
    w = np.asarray(weights, dtype=float)
    p = np.asarray(purities, dtype=float)
    return float(np.average(p, weights=w))


def doorway_oscillation(
    trajectory: TrajectoryRecord, door_x: float, window_steps: int = 10
) -> dict[str, float]:
    """Measure bidirectional oscillation at a doorway.

    For each time window, compute the net number of pedestrians that crossed the
    door in each direction (``+x`` vs ``-x``); the dominant-direction sign of
    the window is the sign of net crossings. Oscillation is the number of sign
    flips of this dominant-direction series. A purely alternating flow has many
    flips; a one-directional or frozen flow has ~0.

    Also reports throughput (pedestrians per second) and the mean burst length
    (windows per dominant-direction run).

    Args:
        trajectory: Recorded trajectory.
        door_x: Door x-coordinate.
        window_steps: Window size in steps for the dominant-direction series.

    Returns:
        Dict with ``oscillation_flips``, ``throughput_peds_per_sec``,
        ``mean_burst_windows``.
    """
    pos = trajectory.positions  # (T, N, 2)
    n_steps = pos.shape[0]
    dt = trajectory.dt
    # crossing events: a ped crosses when its x passes door_x in either direction.
    series: list[int] = []
    w = max(1, int(window_steps))
    net_in_window = 0
    windows = 0
    total_crossings = 0
    for t in range(1, n_steps):
        prev_x = pos[t - 1, :, 0]
        cur_x = pos[t, :, 0]
        # crossed +x this step
        plus = int(np.sum((prev_x < door_x) & (cur_x >= door_x)))
        minus = int(np.sum((prev_x >= door_x) & (cur_x < door_x)))
        net_in_window += plus - minus
        total_crossings += plus + minus
        if t % w == 0:
            sign = 1 if net_in_window > 0 else (-1 if net_in_window < 0 else 0)
            series.append(sign)
            net_in_window = 0
            windows += 1
    # count flips over nonzero signs
    nonzero = [s for s in series if s != 0]
    flips = sum(1 for a, b in pairwise(nonzero) if a != b)
    # burst lengths: run-length of consecutive equal nonzero signs
    bursts: list[int] = []
    if nonzero:
        run = 1
        for a, b in pairwise(nonzero):
            if a == b:
                run += 1
            else:
                bursts.append(run)
                run = 1
        bursts.append(run)
    mean_burst = float(np.mean(bursts)) if bursts else 0.0
    duration = n_steps * dt
    throughput = total_crossings / duration if duration > 0 else 0.0
    return {
        "oscillation_flips": float(flips),
        "throughput_peds_per_sec": float(throughput),
        "mean_burst_windows": float(mean_burst),
    }


def exit_arching(
    trajectory: TrajectoryRecord, exit_x: float, exit_radius: float = 1.5
) -> dict[str, float]:
    """Measure arching/clogging near a high-density exit.

    Two signals:

    - ``exit_density_ratio``: ratio of mean local density within ``exit_radius``
      of the exit to the mean bulk density (over the steady-state window). A
      clogging arch produces a pronounced density cusp at the exit, so this
      ratio is ``> 1`` (often substantially) when arching occurs.
    - ``arch_lateral_spread``: standard deviation of the lateral (``y``)
      position of pedestrians within ``exit_radius`` of the exit, normalized by
      ``exit_radius``. An arch spreads pedestrians angularly around the door
      (larger spread) rather than queuing in a single file (small spread).

    Args:
        trajectory: Recorded trajectory.
        exit_x: Exit x-coordinate.
        exit_radius: Radius (m) defining the "near exit" region.

    Returns:
        Dict with ``exit_density_ratio`` and ``arch_lateral_spread``.
    """
    pos = trajectory.positions
    n_steps = pos.shape[0]
    start = n_steps // 2
    n = pos.shape[1]
    # Bulk density: pedestrians per m^2 over the room footprint, estimated from
    # the observed position envelope over the steady-state window.
    xs_all = pos[start:, :, 0]
    ys_all = pos[start:, :, 1]
    x_min, x_max = float(xs_all.min()), float(xs_all.max())
    y_min, y_max = float(ys_all.min()), float(ys_all.max())
    room_area = max(1e-6, (x_max - x_min) * (y_max - y_min))
    bulk_density = n / room_area

    near_counts: list[int] = []
    near_y_spread: list[float] = []
    exit_area = max(1e-6, np.pi * exit_radius**2)
    for t in range(start, n_steps):
        p = pos[t]
        dist = np.abs(p[:, 0] - exit_x)
        near = dist <= exit_radius
        near_counts.append(int(near.sum()))
        if near.sum() >= 2:
            near_y_spread.append(float(p[near, 1].std()))
    mean_near = float(np.mean(near_counts)) if near_counts else 0.0
    exit_density = mean_near / exit_area
    ratio = exit_density / bulk_density if bulk_density > 0 else 0.0
    spread = float(np.mean(near_y_spread)) if near_y_spread else 0.0
    spread_norm = spread / exit_radius
    return {
        "exit_density_ratio": float(ratio),
        "arch_lateral_spread": float(spread_norm),
    }


def _compute_order_parameters(
    scenario_name: str, trajectory: TrajectoryRecord, config: ScenarioConfig
) -> dict[str, float]:
    """Dispatch to the phenomenon-specific order-parameter computation.

    Returns:
        Dict of order-parameter name to value for the given scenario.
    """
    if scenario_name == "bidirectional_corridor":
        return {
            "lane_segregation_index": lane_segregation_index(trajectory),
            "lane_purity": lane_purity(trajectory),
        }
    if scenario_name == "narrow_doorway":
        door_x = float(config.extra.get("door_x", config.length / 2.0))
        osc = doorway_oscillation(trajectory, door_x=door_x)
        return osc
    if scenario_name == "high_density_exit":
        return exit_arching(trajectory, exit_x=config.length)
    raise ValueError(f"Unknown scenario {scenario_name!r}")


# --------------------------------------------------------------------------- #
# Top-level demonstration entry point
# --------------------------------------------------------------------------- #


def _default_scenario_set() -> list[ScenarioConfig]:
    """Return the canonical demonstration scenario set (small CPU compute)."""
    return [
        ScenarioConfig(
            name="bidirectional_corridor",
            length=24.0,
            half_width=2.5,
            n_pedestrians=24,
            seed=5149,
            n_steps=400,
        ),
        ScenarioConfig(
            name="narrow_doorway",
            length=16.0,
            half_width=2.0,
            n_pedestrians=20,
            seed=5149,
            n_steps=500,
            extra={"door_x": 8.0, "door_half_width": 0.6},
        ),
        ScenarioConfig(
            name="high_density_exit",
            length=10.0,
            half_width=4.0,
            n_pedestrians=40,
            seed=5149,
            n_steps=500,
            extra={"exit_half_width": 0.6},
        ),
    ]


def run_emergent_phenomena_demo(
    scenarios: Sequence[ScenarioConfig] | None = None,
    calibrations: Sequence[SpeedCalibration] | None = None,
    sim_config: SimulatorConfig | None = None,
) -> EmergentPhenomenaReport:
    """Run the full emergent-phenomena demonstration and return a report.

    This is the canonical entry point invoked by the generation script
    ``scripts/validation/build_issue_5149_emergent_phenomena_demo.py``. It runs
    each scenario under each speed calibration (released default and
    literature-typical) and aggregates results with provenance metadata.

    Args:
        scenarios: Optional scenario list; defaults to the canonical set.
        calibrations: Optional calibrations; defaults to released + literature.
        sim_config: Optional simulator config; defaults to released defaults.

    Returns:
        EmergentPhenomenaReport with all results and provenance.
    """
    if scenarios is None:
        scenarios = _default_scenario_set()
    if calibrations is None:
        calibrations = [RELEASED_DEFAULT_CALIBRATION, LITERATURE_CALIBRATION]
    if sim_config is None:
        sim_config = released_default_config()

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        for cal in calibrations:
            results.append(run_scenario(scenario, cal, sim_config=sim_config))

    config_json = {
        "scene": {
            "enable_group": sim_config.scene_config.enable_group,
            "agent_radius": sim_config.scene_config.agent_radius,
            "dt_secs": sim_config.scene_config.dt_secs,
            "max_speed_multiplier": sim_config.scene_config.max_speed_multiplier,
            "tau": sim_config.scene_config.tau,
        },
        "social_force": {
            "factor": sim_config.social_force_config.factor,
            "lambda_importance": sim_config.social_force_config.lambda_importance,
            "gamma": sim_config.social_force_config.gamma,
            "n": sim_config.social_force_config.n,
            "n_prime": sim_config.social_force_config.n_prime,
            "activation_threshold": sim_config.social_force_config.activation_threshold,
        },
        "obstacle_force": {
            "factor": sim_config.obstacle_force_config.factor,
            "sigma": sim_config.obstacle_force_config.sigma,
            "threshold": sim_config.obstacle_force_config.threshold,
        },
    }
    return EmergentPhenomenaReport(
        results=results,
        substrate_version=getattr(pysf, "__version__", "unknown"),
        config_json=config_json,
    )
