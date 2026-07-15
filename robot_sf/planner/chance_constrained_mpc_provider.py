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
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.planner.chance_constrained_mpc import GaussianMixturePedestrianForecast

if TYPE_CHECKING:
    from collections.abc import Callable

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

    def __post_init__(self) -> None:
        """Reject empty or physically invalid diagnostic scenarios."""
        if int(self.num_episodes) <= 0:
            raise ValueError("CalibrationScenario.num_episodes must be positive")
        if int(self.steps_per_episode) <= 0:
            raise ValueError("CalibrationScenario.steps_per_episode must be positive")
        if not np.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("CalibrationScenario.dt must be finite and positive")
        if int(self.num_pedestrians) < 0:
            raise ValueError("CalibrationScenario.num_pedestrians must be non-negative")
        for name in (
            "robot_goal_distance_m",
            "spawn_radius_m",
            "max_ped_speed_mps",
            "collision_radius_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"CalibrationScenario.{name} must be finite and non-negative")
        if float(self.spawn_radius_m) < 0.8:
            raise ValueError("CalibrationScenario.spawn_radius_m must be at least 0.8 m")
        if float(self.collision_radius_m) <= 0.0:
            raise ValueError("CalibrationScenario.collision_radius_m must be positive")


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
        ``mean_tail_clearance_m``, ``mean_compute_time_ms``,
        ``max_compute_time_ms``, and a
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
    max_compute_time_ms = 0.0
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
            max_compute_time_ms = max(max_compute_time_ms, float(elapsed_ms))
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
        "max_compute_time_ms": float(max_compute_time_ms) if compute_times_ms else float("nan"),
        "sample_count": float(n),
        "claim_boundary": (
            "closed-loop self-consistency diagnostic under matched constant-velocity "
            "dynamics; not a matched-arm benchmark calibration result and makes no "
            "planner-value or real-world risk claim; replace the surrogate with the "
            "learned #2844 predictor before any benchmark-facing claim"
        ),
    }


@dataclass(frozen=True)
class CalibrationSweep:
    """Risk-budget grid for the calibration-sweep reliability curve.

    Issue #5307's primary measure is realized collision-risk CALIBRATION
    (claimed vs. observed). Calibration is inherently a *curve* over the risk
    budget the planner is allowed to claim, not a single point: a well-built
    chance-constraint control law should observe a collision frequency that
    tracks the claimed ``max_collision_risk`` as that budget is swept, and the
    curve should be monotone non-decreasing (loosening the bound admits more
    risk). This dataclass holds the configuration of that sweep.

    The closed-loop scenarios at every budget point come from the same RNG seed,
    so each point replays identical pedestrian spawn/motion sequences
    (apples-to-apples across risk budgets and across chance-constraint
    formulations).
    """

    risk_budgets: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    formulations: tuple[str, ...] = ("marginal",)
    scenario: CalibrationScenario = field(default_factory=CalibrationScenario)


def _is_nondecreasing(values: list[float], *, atol: float = 1e-9) -> bool:
    """True when ``values`` is monotone non-decreasing within ``atol``.

    Returns:
        ``True`` when every successive pair is non-decreasing within ``atol``.
    """

    for a, b in pairwise(values):
        if b < a - atol:
            return False
    return True


def realized_collision_risk_calibration_sweep(
    base_algo_config: dict[str, Any] | None,
    sweep: CalibrationSweep | None = None,
    *,
    adapter_builder: Callable[[dict[str, Any]], ChanceConstrainedMPCPlannerAdapter] | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Issue #5307 primary measure as a calibration *curve* over the risk budget.

    Sweeps ``max_collision_risk`` across a configurable budget grid and, for
    each budget, across the requested chance-constraint formulations
    (``marginal``, ``joint_horizon`` Boole-union bound, or ``cvar_tail``
    Conditional Value-at-Risk tail-risk). At every grid point the planner is
    rebuilt with the patched ``max_collision_risk`` / formulation and rolled
    out through :func:`realized_collision_risk_calibration` over the *same*
    matched-constant-velocity scenarios (same ``seed``), so the
    claimed-vs-observed pairs are directly comparable across the budget and
    across formulations.

    The routine assembles the claimed-vs-observed reliability curve plus a
    per-formulation reliability summary. The summary surfaces the issue's
    pre-registered stop-rule *observables*: the per-step compute time against
    the MPC control period (``stop if the solver cannot meet the control
    period``) and the freeze rate at the tightest risk budget (``stop if
    safety gains come with the zero-progress behavior already observed for
    stream_gap``). These are exposed as observables for inspection, not as a
    gate verdict: the cheap-lane slice does not make a campaign-routing
    decision.

    This slice is CPU-runnable today over the surrogate
    ``constant_velocity_gmm`` provider. It is an API/self-consistency
    diagnostic under matched surrogate dynamics and is **not** a benchmark
    calibration result: replacing the surrogate with the learned ``#2844``
    predictor is required before any matched-arm benchmark calibration claim.

    Args:
        base_algo_config: Base YAML-style config mapping for the chance
            constrained MPC adapter. ``max_collision_risk`` and
            ``chance_constraint_formulation`` are overridden per grid point.
        sweep: :class:`CalibrationSweep` controlling the risk-budget grid, the
            formulations to compare, and the closed-loop scenario. Defaults to a
            single-formulation four-point grid over :class:`CalibrationScenario`.
        adapter_builder: Callable that builds the planner from a config mapping;
            defaults to lazily-imported
            :func:`build_chance_constrained_mpc_adapter`. Injected for testability.
        seed: RNG seed shared across every grid point so the same scenario
            sequence replays at every budget and formulation.

    Returns:
        Mapping with ``sweep`` (grid + seed + scenario provenance), ``points``
        (flat list of per-budget-per-formulation calibration records), ``points_by_formulation``
        (grouped), and ``reliability_summary`` (per-formulation curve statistics and
        stop-rule observables). The output records the current claim/observation
        units explicitly; unit alignment is tracked by follow-up issue #5737.
        A ``claim_boundary`` marks the diagnostic scope.

    Raises:
        ValueError: If the budget grid is empty, any budget is outside ``(0, 1)``,
            the formulation list is empty, or a formulation is not one of
            ``marginal``/``joint_horizon``/``cvar_tail``.
    """

    sweep = sweep or CalibrationSweep()
    if adapter_builder is None:
        from robot_sf.planner.chance_constrained_mpc import (  # noqa: PLC0415
            build_chance_constrained_mpc_adapter,
        )

        adapter_builder = build_chance_constrained_mpc_adapter

    budgets: tuple[float, ...] = tuple(float(b) for b in sweep.risk_budgets)
    if not budgets:
        raise ValueError("CalibrationSweep.risk_budgets must be non-empty")
    if not all(0.0 < b < 1.0 for b in budgets):
        raise ValueError("CalibrationSweep.risk_budgets must lie in (0, 1)")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("CalibrationSweep.risk_budgets must be strictly increasing")

    valid_formulations = {"marginal", "joint_horizon", "cvar_tail"}
    formulations = tuple(str(f) for f in sweep.formulations)
    if not formulations:
        raise ValueError("CalibrationSweep.formulations must be non-empty")
    unsupported = [f for f in formulations if f not in valid_formulations]
    if unsupported:
        raise ValueError(
            f"Unsupported formulation(s) {unsupported!r}; "
            f"expected one of {sorted(valid_formulations)}"
        )

    base = dict(base_algo_config or {})
    base.setdefault("predictor_backend", "constant_velocity_gmm")

    scenario = sweep.scenario
    control_period_ms = float(scenario.dt) * 1000.0

    points_by_formulation: dict[str, list[dict[str, Any]]] = {}
    for formulation in formulations:
        points: list[dict[str, Any]] = []
        for budget in budgets:
            cfg = dict(base)
            cfg["max_collision_risk"] = float(budget)
            cfg["chance_constraint_formulation"] = formulation
            planner = adapter_builder(cfg)
            result = realized_collision_risk_calibration(planner, scenario, seed=seed)
            point = {
                "formulation": formulation,
                "requested_risk_budget": float(budget),
                "claimed_risk": float(result["claimed_risk_per_horizon"]),
                "claim_unit": "planner_risk_budget_per_horizon",
                "observed_unit": "episode_any_collision_rate",
                "calibration_comparability": "pending_issue_5737",
                "observed_collision_rate": float(result["observed_collision_rate"]),
                "calibration_error": float(result["calibration_error"]),
                "freeze_rate": float(result["freeze_rate"]),
                "infeasible_rate": float(result["infeasible_rate"]),
                "completion_rate": float(result["completion_rate"]),
                "mean_tail_clearance_m": float(result["mean_tail_clearance_m"]),
                "mean_compute_time_ms": float(result["mean_compute_time_ms"]),
                "sample_count": float(result["sample_count"]),
                "max_compute_time_ms": float(result["max_compute_time_ms"]),
            }
            points.append(point)
        points_by_formulation[formulation] = points

    reliability_summary: dict[str, dict[str, Any]] = {}
    for formulation, points in points_by_formulation.items():
        abs_errors = [abs(float(p["calibration_error"])) for p in points]
        observed = [float(p["observed_collision_rate"]) for p in points]
        compute_times = [float(p["max_compute_time_ms"]) for p in points]
        freeze_rates = [float(p["freeze_rate"]) for p in points]
        # The tightest (lowest) budget sits first because budgets sweep small->large.
        freeze_at_tightest = freeze_rates[0] if freeze_rates else 0.0
        reliability_summary[formulation] = {
            "mean_abs_calibration_error": float(np.mean(abs_errors))
            if abs_errors
            else float("nan"),
            "max_abs_calibration_error": float(np.max(abs_errors)) if abs_errors else float("nan"),
            # A well-calibrated risk bound should make observed collision rate
            # rise with the claimed budget; a non-monotone curve, beyond
            # sparse-sample noise, signals that the bound is not translating to
            # realized risk (a calibration defect the issue asks to surface).
            "observed_monotone_non_decreasing_in_claimed": bool(_is_nondecreasing(observed)),
            # Issue #5307 pre-registered stop rules, surfaced as observables
            # (not a routing decision): compute time vs the MPC control period,
            # and zero-progress freeze at the tightest budget.
            "control_period_ms": float(control_period_ms),
            "max_per_step_compute_time_ms": float(np.max(compute_times))
            if compute_times
            else float("nan"),
            "compute_exceeds_control_period": bool(
                bool(compute_times) and float(np.max(compute_times)) > control_period_ms
            ),
            "freeze_rate_at_tightest_budget": float(freeze_at_tightest),
        }

    all_points = [p for pts in points_by_formulation.values() for p in pts]
    return {
        "sweep": {
            "base_algo_config": dict(base),
            "risk_budgets": list(budgets),
            "formulations": list(formulations),
            "seed": int(seed),
            "scenario": {
                "num_episodes": int(scenario.num_episodes),
                "steps_per_episode": int(scenario.steps_per_episode),
                "dt": float(scenario.dt),
                "num_pedestrians": int(scenario.num_pedestrians),
                "robot_goal_distance_m": float(scenario.robot_goal_distance_m),
                "spawn_radius_m": float(scenario.spawn_radius_m),
                "max_ped_speed_mps": float(scenario.max_ped_speed_mps),
                "collision_radius_m": float(scenario.collision_radius_m),
            },
        },
        "points": all_points,
        "points_by_formulation": points_by_formulation,
        "reliability_summary": reliability_summary,
        "claim_boundary": (
            "closed-loop self-consistency calibration curve under matched "
            "constant-velocity surrogate dynamics; not a matched-arm benchmark "
            "calibration result; the current risk/observation units are recorded "
            "but require alignment under follow-up issue #5737. This makes no "
            "planner-value or real-world risk claim; replace the surrogate with "
            "the learned #2844 predictor and run the ops-queue campaign before "
            "any benchmark-facing calibration or planner-superiority claim"
        ),
    }


def build_calibration_sweep_config(
    raw: dict[str, Any] | None,
) -> tuple[dict[str, Any], CalibrationSweep]:
    """Split a sweep YAML into the base adapter config and the sweep grid.

    The config-first reproducibility contract (``configs/`` carries stable
    experiments) is kept by letting a single YAML describe both the base
    chance-constrained MPC settings and the calibration-sweep grid. The
    optional ``calibration_sweep`` mapping carries ``risk_budgets`` and
    ``formulations``; everything else is forwarded as the base ``algo`` config.

    Args:
        raw: Parsed YAML mapping. If ``calibration_sweep`` is absent the default
            :class:`CalibrationSweep` is used and the whole mapping is treated
            as the base config.

    Returns:
        ``(base_algo_config, sweep)``: the base adapter config (with the
        ``calibration_sweep`` sub-key removed) and the parsed sweep grid.
    """

    src = dict(raw or {})
    sweep_raw = src.pop("calibration_sweep", None) or {}
    budgets_raw = sweep_raw.get("risk_budgets")
    if budgets_raw is None:
        risk_budgets: tuple[float, ...] = CalibrationSweep().risk_budgets
    else:
        risk_budgets = tuple(float(b) for b in budgets_raw)
    formulations_raw = sweep_raw.get("formulations")
    if formulations_raw is None:
        formulations: tuple[str, ...] = CalibrationSweep().formulations
    else:
        formulations = tuple(str(f) for f in formulations_raw)
    scenario_raw = sweep_raw.get("scenario") or {}
    scenario_kwargs: dict[str, Any] = {}
    for key in (
        "num_episodes",
        "steps_per_episode",
        "dt",
        "num_pedestrians",
        "robot_goal_distance_m",
        "spawn_radius_m",
        "max_ped_speed_mps",
        "collision_radius_m",
    ):
        if key in scenario_raw:
            scenario_kwargs[key] = scenario_raw[key]
    scenario = CalibrationScenario(**scenario_kwargs)  # type: ignore[arg-type]
    sweep = CalibrationSweep(
        risk_budgets=risk_budgets,
        formulations=formulations,
        scenario=scenario,
    )
    return src, sweep


__all__ = [
    "CalibrationSweep",
    "ConstantVelocityGmmPredictor",
    "build_calibration_sweep_config",
    "build_constant_velocity_gmm_forecast",
    "realized_collision_risk_calibration",
    "realized_collision_risk_calibration_sweep",
]
