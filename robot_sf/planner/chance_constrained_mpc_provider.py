"""Concrete multimodal K-mode predictor provider for issue #5307.

The chance-constrained MPC arm (``chance_constrained_mpc.py``, delivered by PR
#5322) defines the ``GaussianMixturePedestrianPredictor`` boundary but has no
concrete provider because the learned K-mode/GMM predictor (#2844) is still
open. This module delivers the maintainer-sanctioned "populate the interface"
slice: a *surrogate* constant-velocity K-mode provider so the control law is
actually runnable and CPU-validatable today, plus the issue's primary
measure -- realized collision-risk CALIBRATION (claimed vs. observed).

The surrogate is explicitly a diagnostic stand-in, NOT a calibrated forecast.
The real ``#2844`` provider plugs into the same ``GaussianMixturePedestrianPredictor``
protocol and replaces the surrogate without any change to the planner.

The calibration harness is a **closed-loop** diagnostic (issue #5307 measure
#1): it rolls the planner through many short episodes in which the realized
pedestrian dynamics match the surrogate's own constant-velocity forecast, then
pairs the planner's *claimed* per-horizon risk against the *observed*
collision frequency and the other named measures (infeasibility rate,
freezing, completion, tail clearance, compute time). Under matched dynamics
this is an API/self-consistency diagnostic of the GMM risk control law, not a
matched-arm benchmark result and not a real-world risk claim; replacing the
surrogate with the learned ``#2844`` predictor is required before any
benchmark-facing calibration claim.

The :class:`~robot_sf.planner.chance_constrained_mpc.GaussianMixturePedestrianForecast`
container is reused for every forecast this module emits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.planner.chance_constrained_mpc import GaussianMixturePedestrianForecast

if TYPE_CHECKING:
    from robot_sf.planner.chance_constrained_mpc import ChanceConstrainedMPCPlannerAdapter


def build_constant_velocity_gmm_forecast(
    observation: dict[str, object],
    *,
    horizon_steps: int,
    dt: float,
    mode_count: int = 1,
    velocity_noise_std_mps: float = 0.15,
    heading_spread_rad: float = 0.35,
) -> GaussianMixturePedestrianForecast:
    """Build a contract-valid GMM forecast from a SocNav observation.

    A constant-velocity mean is integrated in world frame from the current
    pedestrian state. When ``mode_count > 1`` the surrogate spreads
    additional modes symmetrically around the heading to approximate a
    multi-modal intent distribution; all modes share one covariance derived
    from ``velocity_noise_std_mps``. This is a deterministic stand-in for the
    learned K-mode forecast, not a calibrated predictor.

    Args:
        observation: SocNav-structured observation (robot/goal/pedestrians).
        horizon_steps: Number of forecast steps (matched to the MPC horizon).
        dt: Forecast timestep in seconds (must match the MPC ``rollout_dt``).
        mode_count: Number of Gaussian modes per pedestrian (>= 1).
        velocity_noise_std_mps: Per-axis velocity uncertainty (m/s).
        heading_spread_rad: Symmetric heading angular spread for extra modes.

    Returns:
        A ``GaussianMixturePedestrianForecast`` validated by its own contract.
    """

    if mode_count < 1:
        raise ValueError("mode_count must be >= 1")

    peds = observation.get("pedestrians", {}) if isinstance(observation, dict) else {}
    positions = np.asarray(peds.get("positions", []), dtype=float)
    velocities = np.asarray(peds.get("velocities", []), dtype=float)
    if positions.ndim == 1 and positions.size % 2 == 0:
        positions = positions.reshape(-1, 2)
    if velocities.ndim == 1 and velocities.size % 2 == 0:
        velocities = velocities.reshape(-1, 2)
    if positions.ndim != 2 or positions.shape[-1] != 2:
        positions = np.zeros((0, 2), dtype=float)
    if velocities.ndim != 2 or velocities.shape[-1] != 2:
        velocities = np.zeros_like(positions)
    count = (
        int(np.asarray(peds.get("count", [positions.shape[0]]), dtype=float).reshape(-1)[0])
        if positions.size
        else 0
    )
    count = max(0, min(count, positions.shape[0]))
    positions = positions[:count]
    velocities = velocities[:count] if velocities.shape[0] >= count else np.zeros_like(positions)

    steps = max(int(horizon_steps), 1)
    k = int(mode_count)
    if count == 0:
        return GaussianMixturePedestrianForecast(
            means_world=np.zeros((0, k, steps, 2), dtype=float),
            covariances_world=np.tile(np.eye(2), (0, k, steps, 1, 1)),
            mode_weights=np.zeros((0, k), dtype=float),
            dt=float(dt),
            source="constant_velocity_gmm_surrogate",
        )

    means = np.zeros((count, k, steps, 2), dtype=float)
    covariances = np.tile(
        np.diag([velocity_noise_std_mps**2, velocity_noise_std_mps**2]).astype(float),
        (count, k, steps, 1, 1),
    )
    speed = np.linalg.norm(velocities, axis=1)
    base_heading = np.arctan2(velocities[:, 1], velocities[:, 0] + 1e-9)
    zero_speed = speed < 1e-6
    weights = np.full((count, k), 1.0 / float(k), dtype=float)

    for step in range(steps):
        tau = float(step + 1) * float(dt)
        for mode in range(k):
            if mode == 0:
                heading = base_heading.copy()
            else:
                offset = heading_spread_rad * (1.0 if mode % 2 == 1 else -1.0)
                offset *= float((mode + 1) // 2)
                heading = base_heading + offset
            direction = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
            scale = np.where(zero_speed, 0.0, speed)
            means[:, mode, step, :] = positions + direction * (scale[:, None] * tau)

    return GaussianMixturePedestrianForecast(
        means_world=means,
        covariances_world=covariances,
        mode_weights=weights,
        dt=float(dt),
        source="constant_velocity_gmm_surrogate",
    )


class ConstantVelocityGmmPredictor:
    """Surrogate ``GaussianMixturePedestrianPredictor`` for CPU validation.

    Implements the provider protocol consumed by ``ChanceConstrainedMPCPlannerAdapter``
    using a deterministic constant-velocity GMM. This is a diagnostic stand-in
    for the #2844 learned K-mode/GMM predictor, enabling the control law to be
    exercised end-to-end before #2844 lands.
    """

    def __init__(
        self,
        *,
        mode_count: int = 1,
        velocity_noise_std_mps: float = 0.15,
        heading_spread_rad: float = 0.35,
    ) -> None:
        """Store surrogate spread parameters; default is a single-mode forecast."""

        self.mode_count = int(mode_count)
        self.velocity_noise_std_mps = float(velocity_noise_std_mps)
        self.heading_spread_rad = float(heading_spread_rad)

    def predict(
        self,
        observation: dict[str, object],
        *,
        horizon_steps: int,
        dt: float,
    ) -> GaussianMixturePedestrianForecast:
        """Return a constant-velocity GMM forecast aligned with the MPC horizon."""

        return build_constant_velocity_gmm_forecast(
            observation,
            horizon_steps=horizon_steps,
            dt=dt,
            mode_count=self.mode_count,
            velocity_noise_std_mps=self.velocity_noise_std_mps,
            heading_spread_rad=self.heading_spread_rad,
        )

    def reset(self) -> None:
        """No per-episode state to clear for the deterministic surrogate."""


def _spawn_scenario(
    rng: np.random.Generator,
    *,
    num_pedestrians: int,
    robot_goal_distance_m: float,
    spawn_radius_m: float,
    max_ped_speed_mps: float,
) -> dict[str, object]:
    """Sample one planar crossing scenario with a goal-ahead robot.

    The robot starts at the origin facing +x with a goal straight ahead;
    pedestrians are placed in a disc ahead of it and given roughly inbound
    constant-velocity headings. This scenario family is the *matched* dynamics
    of the constant-velocity surrogate forecast: the realized motion is exactly
    the model the surrogate assumes, so a well-built risk control law should
    observe a collision frequency close to its claimed risk.

    Returns:
        A SocNav-structured observation with one goal-ahead robot and the
        sampled pedestrian states.
    """

    ped_positions: list[list[float]] = []
    ped_velocities: list[list[float]] = []
    for _ in range(int(num_pedestrians)):
        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(0.8, float(spawn_radius_m))
        px = np.cos(angle) * radius + 1.0
        py = np.sin(angle) * radius
        speed = rng.uniform(0.0, float(max_ped_speed_mps))
        # Inbound heading (biased toward the robot lane at the origin).
        heading = np.arctan2(-py, -px) + rng.uniform(-0.6, 0.6)
        ped_positions.append([float(px), float(py)])
        ped_velocities.append([float(speed * np.cos(heading)), float(speed * np.sin(heading))])
    return {
        "robot": {
            "position": np.asarray([0.0, 0.0]),
            "heading": np.asarray([0.0]),
            "speed": np.asarray([0.0]),
            "radius": np.asarray([0.25]),
        },
        "goal": {
            "current": np.asarray([float(robot_goal_distance_m), 0.0]),
            "next": np.asarray([float(robot_goal_distance_m), 0.0]),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([float(num_pedestrians)]),
            "radius": np.asarray([0.25]),
        },
    }


def _step_pedestrians(observation: dict[str, object], *, dt: float) -> None:
    """Advance the pedestrian positions in-place by one constant-velocity step."""

    peds = observation["pedestrians"]
    positions = np.asarray(peds["positions"], dtype=float)
    velocities = np.asarray(peds["velocities"], dtype=float)
    positions = positions + velocities * float(dt)
    peds["positions"] = positions


def _step_robot(observation: dict[str, object], command: tuple[float, float], *, dt: float) -> None:
    """Apply one unicycle command to the mutable calibration observation."""

    robot = observation["robot"]
    position = np.asarray(robot["position"], dtype=float)
    heading = float(np.asarray(robot["heading"], dtype=float).reshape(-1)[0])
    speed = max(float(command[0]), 0.0)
    angular_speed = float(command[1])
    next_heading = heading + angular_speed * float(dt)
    robot["heading"] = np.asarray([next_heading], dtype=float)
    robot["position"] = position + speed * float(dt) * np.asarray(
        [np.cos(next_heading), np.sin(next_heading)], dtype=float
    )
    robot["speed"] = np.asarray([speed], dtype=float)


def _min_robot_pedestrian_clearance(
    observation: dict[str, object], *, collision_radius_m: float
) -> float:
    """Return current clearance using the scenario's declared contact radius."""

    contact_radius = float(collision_radius_m)
    if not np.isfinite(contact_radius) or contact_radius <= 0.0:
        raise ValueError("collision_radius_m must be finite and > 0")
    robot_pos = np.asarray(observation["robot"]["position"], dtype=float)
    peds = observation["pedestrians"]
    ped_positions = np.asarray(peds["positions"], dtype=float)
    if ped_positions.size == 0:
        return float("inf")
    dists = np.linalg.norm(ped_positions - robot_pos[None, :], axis=1)
    return float(np.min(dists)) - contact_radius


@dataclass(frozen=True)
class CalibrationScenario:
    """Scenario family for the closed-loop realized-risk calibration harness.

    The pedestrian dynamics are sampled so they *match* the surrogate's own
    constant-velocity forecast; see :func:`realized_collision_risk_calibration`
    and :func:`_spawn_scenario`.
    """

    num_episodes: int = 60
    steps_per_episode: int = 24
    dt: float = 0.25
    num_pedestrians: int = 6
    robot_goal_distance_m: float = 6.0
    spawn_radius_m: float = 4.0
    max_ped_speed_mps: float = 0.7
    collision_radius_m: float = 0.85


def realized_collision_risk_calibration(
    planner: ChanceConstrainedMPCPlannerAdapter,
    scenario: CalibrationScenario | None = None,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Closed-loop realized-risk calibration harness for issue #5307's primary measure.

    Rolls the chance-constrained MPC planner through many short episodes whose
    pedestrian dynamics *match* the surrogate's own constant-velocity forecast.
    For every control step it records the planner's **claimed** per-horizon risk
    (``planner.claimed_risk``) and, after integrating the realized motion, the
    **observed** collision occurrence. The harness then reports the issue's named
    measures: realized collision-risk calibration (claimed vs. observed),
    infeasibility rate, freezing (zero-progress) rate, completion rate, tail
    clearance, and per-step compute time.

    Under matched dynamics this is an API/self-consistency diagnostic of the GMM
    risk control law: it shows whether the planner's claimed risk bound is
    realized in closed loop. It is **not** a matched-arm benchmark result and
    makes **no** planner-superiority or real-world risk claim; replacing the
    surrogate with the learned ``#2844`` predictor is required before any
    benchmark-facing calibration claim. The routine is CPU-only.

    Args:
        planner: A configured ``ChanceConstrainedMPCPlannerAdapter`` (typically
            built over the ``constant_velocity_gmm`` surrogate backend).
        scenario: Scenario family controlling episode/step counts, dynamics, and
            the contact radius. Defaults to :class:`CalibrationScenario`.
        seed: RNG seed for reproducibility.

    Returns:
        Mapping with ``claimed_risk_per_horizon``, ``observed_collision_rate``,
        ``calibration_error`` (observed - claimed, the issue's primary measure),
        ``infeasible_rate``, ``freeze_rate``, ``completion_rate``,
        ``mean_tail_clearance_m``, ``mean_compute_time_ms``, and a
        ``claim_boundary`` marking the diagnostic scope.
    """

    scenario = scenario or CalibrationScenario()
    planner_dt = float(getattr(getattr(planner, "config", None), "rollout_dt", scenario.dt))
    if not np.isclose(planner_dt, float(scenario.dt)):
        raise ValueError(
            "CalibrationScenario.dt must match the planner rollout_dt "
            f"({scenario.dt!r} != {planner_dt!r})"
        )
    rng = np.random.default_rng(int(seed))
    compute_times_ms: list[float] = []
    tail_clearances: list[float] = []
    collisions = 0
    infeasible = 0
    freezes = 0
    completions = 0
    claimed = float(getattr(planner, "claimed_risk", float("nan")))
    for _ in range(int(scenario.num_episodes)):
        planner.reset()
        observation = _spawn_scenario(
            rng,
            num_pedestrians=scenario.num_pedestrians,
            robot_goal_distance_m=scenario.robot_goal_distance_m,
            spawn_radius_m=scenario.spawn_radius_m,
            max_ped_speed_mps=scenario.max_ped_speed_mps,
        )
        episode_collision = False
        episode_moved = False
        for step in range(int(scenario.steps_per_episode)):
            start_ns = time.perf_counter_ns()
            raw_command = planner.plan(observation)
            command = (float(raw_command[0]), float(raw_command[1]))
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            compute_times_ms.append(float(elapsed_ms))
            speed = float(command[0])
            episode_moved = episode_moved or speed > 1e-3
            tail_clearances.append(
                _min_robot_pedestrian_clearance(
                    observation, collision_radius_m=scenario.collision_radius_m
                )
            )
            if tail_clearances[-1] < 0.0:
                episode_collision = True
                break
            _step_robot(observation, command, dt=float(scenario.dt))
            _step_pedestrians(observation, dt=float(scenario.dt))
            tail_clearances.append(
                _min_robot_pedestrian_clearance(
                    observation, collision_radius_m=scenario.collision_radius_m
                )
            )
            if tail_clearances[-1] < 0.0:
                episode_collision = True
                break
            goal = np.asarray(observation["goal"]["current"], dtype=float)
            robot_pos = np.asarray(observation["robot"]["position"], dtype=float)
            if float(np.linalg.norm(goal - robot_pos)) <= 0.25:
                completions += 1
                break
        if episode_collision:
            collisions += 1
        elif not episode_moved:
            # A finished episode counts as moved; only a non-colliding, non-finished
            # episode that never issued forward velocity is a freeze.
            freezes += 1
        if planner.diagnostics().get("solver_failures", 0) > 0:
            infeasible += 1
    n = max(int(scenario.num_episodes), 1)
    observed = collisions / float(n)
    return {
        "claimed_risk_per_horizon": float(claimed),
        "observed_collision_rate": float(observed),
        "calibration_error": float(observed - claimed),
        "infeasible_rate": float(infeasible / n),
        "freeze_rate": float(freezes / n),
        "completion_rate": float(completions / n),
        "mean_tail_clearance_m": float(np.mean(tail_clearances))
        if tail_clearances
        else float("nan"),
        "mean_compute_time_ms": float(np.mean(compute_times_ms))
        if compute_times_ms
        else float("nan"),
        "sample_count": float(n),
        "claim_boundary": (
            "closed-loop self-consistency diagnostic under matched constant-velocity "
            "dynamics; not a matched-arm benchmark calibration result and makes no "
            "planner-value or real-world risk claim; replace the surrogate with the "
            "learned #2844 predictor before any benchmark-facing claim"
        ),
    }


__all__ = [
    "ConstantVelocityGmmPredictor",
    "build_constant_velocity_gmm_forecast",
    "realized_collision_risk_calibration",
]
