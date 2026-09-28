"""Operational quantities for transparent external total-cost models — Issue #9350.

This module separates what the Robot SF simulator can honestly provide from
what an external total-cost model must assume. Simulated distance and waiting
time are **not** validated energy consumption, maintenance cost, teleoperation
demand, or currency per kilometer.

Three typed input layers (see :data:`FIELD_AVAILABILITY`) keep physics, service
semantics, and external cost assumptions separate:

- :class:`SimulatorObservables`: measured or derived from one episode
  trajectory (meters, seconds, counts, IDs).
- :class:`ServiceInputs`: service-defined denominators (productive vs empty
  distance, orders, passenger load, operating hours). ``success=True`` does not
  prove a paid trip; unknown categories stay ``None``.
- :class:`ExternalCostInputs`: prices, lifetimes, and rates with currency,
  price basis, period, source/date, provenance (``measured`` / ``literature`` /
  ``assumed``), and uncertainty. Missing prices are visible blockers, never
  implicit zeros.

:func:`compute_cost_breakdown` applies one transparent schematic form::

    cost per productive km =
        (annualized capital + fixed annual + variable trip costs
         + time-dependent costs) / productive annual kilometers

It also reports cost per all vehicle kilometers, and cost per order or per
passenger-kilometer only when those denominators are evidenced. Nothing here
enters a planner ranking; any such consumer needs a separate authorizing issue.

Scope and sensitivity note: results are a sensitivity surface over utilization,
lifetime, and supervision assumptions — not a market-price forecast, financial
advice, real accident-cost estimate, or service guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from robot_sf.benchmark.metrics import EpisodeData

Availability = Literal["present", "derivable", "external", "unknown"]
Provenance = Literal["measured", "literature", "assumed"]

#: Field-availability matrix: which operational inputs Robot SF can provide.
#: ``present`` = stored directly; ``derivable`` = computed from stored fields;
#: ``external`` = must be supplied by service/cost owners; ``unknown`` = not
#: available without new instrumentation or semantics.
FIELD_AVAILABILITY: dict[str, dict[str, str]] = {
    # Simulator-observable fields.
    "distance_m": {"status": "derivable", "from": "robot_pos diffs", "units": "m"},
    "sim_time_s": {"status": "derivable", "from": "trajectory length * dt", "units": "s"},
    "active_time_s": {
        "status": "derivable",
        "from": "robot_vel norms above threshold",
        "units": "s",
    },
    "idle_time_s": {
        "status": "derivable",
        "from": "sim_time_s - active_time_s",
        "units": "s",
    },
    "goal_reached": {
        "status": "derivable",
        "from": "reached_goal_step is not None",
        "units": "bool",
    },
    "exposure_duration_s": {
        "status": "derivable",
        "from": "sim_time_s (simulator exposure only)",
        "units": "s",
    },
    "queue_time_s": {
        "status": "unknown",
        "from": "no explicit queue timer in Simulator.step_once",
        "units": "s",
    },
    "world_id": {"status": "external", "from": "caller-supplied run identity", "units": "id"},
    "episode_id": {"status": "external", "from": "caller-supplied run identity", "units": "id"},
    "agent_id": {"status": "external", "from": "caller-supplied run identity", "units": "id"},
    "result_status": {
        "status": "external",
        "from": "caller-supplied outcome label",
        "units": "label",
    },
    # Service-semantics fields (unknown until a service definition supplies them).
    "productive_distance_m": {"status": "external", "from": "service trip log", "units": "m"},
    "empty_distance_m": {"status": "external", "from": "service trip log", "units": "m"},
    "completed_orders": {"status": "external", "from": "service order log", "units": "count"},
    "passenger_km": {"status": "external", "from": "service load log", "units": "passenger*km"},
    "operating_hours_s": {"status": "external", "from": "service availability log", "units": "s"},
    # External cost assumptions (never inferred from trajectories).
    "capital_cost": {"status": "external", "from": "procurement / finance", "units": "currency"},
    "energy_price": {
        "status": "external",
        "from": "tariff + calibrated consumption",
        "units": "currency/kWh",
    },
    "maintenance_cost": {
        "status": "external",
        "from": "service records",
        "units": "currency/period",
    },
    "supervision_cost": {
        "status": "external",
        "from": "staffing model",
        "units": "currency/period",
    },
}

#: Default speed norm (m/s) at or below which a timestep counts as idle.
IDLE_SPEED_THRESHOLD_M_S = 0.05

#: Cost basis periods supported by :class:`ExternalCostInputs`.
CostPeriod = Literal["annual", "lifecycle"]


class OperationalCostBlockedError(ValueError):
    """Raised when a cost total cannot be produced without hidden assumptions.

    The error message names every missing input so a missing price,
    calibration, or service definition stays a visible blocker instead of an
    implicit zero.
    """


@dataclass(frozen=True)
class SimulatorObservables:
    """Measured or trajectory-derived quantities for one episode (SI units)."""

    distance_m: float
    sim_time_s: float
    active_time_s: float
    idle_time_s: float
    goal_reached: bool
    reached_goal_step: int | None
    exposure_duration_s: float
    dt_s: float
    queue_time_s: float | None = None
    world_id: str | None = None
    episode_id: str | None = None
    agent_id: str | None = None
    result_status: str = "unknown"


@dataclass(frozen=True)
class ServiceInputs:
    """Service-defined denominators. Unknown categories stay ``None``."""

    productive_distance_m: float | None = None
    empty_distance_m: float | None = None
    completed_orders: int | None = None
    passenger_km: float | None = None
    operating_hours_s: float | None = None


@dataclass(frozen=True)
class ExternalCostInputs:
    """External price/rate assumptions with provenance.

    Set exactly one of ``annualized_capital`` (already amortized) or
    ``full_investment`` (amortized here over ``lifetime_periods``). Supplying
    both is a double-counting error.
    """

    currency: str = "EUR"
    price_basis_year: int | None = None
    period: CostPeriod = "annual"
    annualized_capital: float | None = None
    full_investment: float | None = None
    lifetime_periods: float | None = None
    residual_value: float = 0.0
    fixed_per_period: float = 0.0
    variable_per_km: float = 0.0
    time_per_hour: float = 0.0
    source: str = "unknown"
    source_date: str = "unknown"
    provenance: dict[str, Provenance] = field(default_factory=dict)
    uncertainty: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CostBreakdown:
    """Transparent per-unit cost result with fixed/variable split."""

    currency: str
    period: CostPeriod
    total_per_period: float
    fixed_per_period: float
    variable_total: float
    cost_per_productive_km: float
    fixed_component_per_productive_km: float
    cost_per_all_km: float | None
    cost_per_order: float | None
    cost_per_passenger_km: float | None
    productive_distance_m: float
    provenance: dict[str, Provenance]
    source: str
    source_date: str


def extract_operational_observables(
    data: EpisodeData,
    *,
    world_id: str | None = None,
    episode_id: str | None = None,
    agent_id: str | None = None,
    result_status: str = "unknown",
    queue_time_s: float | None = None,
    idle_speed_threshold_m_s: float = IDLE_SPEED_THRESHOLD_M_S,
) -> SimulatorObservables:
    """Derive simulator observables from one episode trajectory.

    Distance comes from successive ``robot_pos`` diffs, simulated time from
    ``T * dt``, and active/idle split from ``robot_vel`` norms. Simulated time
    is kept separate from computer wall-clock time. Queue time has no simulator
    timer and stays ``None`` (unknown) unless an explicitly measured value is
    passed.

    Returns
    -------
    SimulatorObservables
        Trajectory-derived quantities in SI units with caller-supplied IDs.
    """
    positions = np.asarray(data.robot_pos, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise OperationalCostBlockedError(
            f"robot_pos must have shape (T, 2), got {positions.shape}"
        )
    steps = max(int(positions.shape[0]) - 1, 0)
    distance_m = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if steps else 0.0
    dt = float(data.dt)
    if not np.isfinite(dt) or dt <= 0:
        raise OperationalCostBlockedError(f"dt must be a positive timestep, got {data.dt!r}")
    sim_time_s = float(positions.shape[0] * dt)
    speeds = np.linalg.norm(np.asarray(data.robot_vel, dtype=float), axis=1)
    active_steps = int(np.count_nonzero(speeds > idle_speed_threshold_m_s))
    active_time_s = float(active_steps * dt)
    idle_time_s = float(max(sim_time_s - active_time_s, 0.0))
    reached = data.reached_goal_step is not None
    return SimulatorObservables(
        distance_m=distance_m,
        sim_time_s=sim_time_s,
        active_time_s=active_time_s,
        idle_time_s=idle_time_s,
        goal_reached=reached,
        reached_goal_step=data.reached_goal_step,
        exposure_duration_s=sim_time_s,
        dt_s=dt,
        queue_time_s=queue_time_s,
        world_id=world_id,
        episode_id=episode_id,
        agent_id=agent_id,
        result_status=result_status,
    )


def trajectory_only_value() -> tuple[bool, str]:
    """Report whether trajectory-only data can produce a total economic value.

    It cannot: a numeric total requires service denominators and external cost
    parameters.

    Returns
    -------
    tuple[bool, str]
        ``(False, reason)`` so callers surface the block.
    """
    return False, (
        "trajectory-only data carries no service denominators (productive "
        "distance, orders, passenger load) and no cost parameters (capital, "
        "prices, supervision); a total economic value is blocked until "
        "ServiceInputs and ExternalCostInputs are supplied."
    )


def _require_positive(name: str, value: float | None, missing: list[str]) -> float | None:
    if value is None:
        missing.append(name)
        return None
    if not np.isfinite(value) or value < 0:
        raise OperationalCostBlockedError(
            f"{name} must be a finite non-negative number, got {value!r}"
        )
    return value


def _resolve_capital_per_period(costs: ExternalCostInputs, missing: list[str]) -> float | None:
    """Resolve amortized capital per period or record why it is blocked.

    Returns
    -------
    float | None
        Capital cost per period, or ``None`` when inputs are missing (names are
        appended to ``missing``).
    """
    if costs.annualized_capital is not None:
        return _require_positive("annualized_capital", costs.annualized_capital, missing)
    if costs.full_investment is None:
        missing.append("annualized_capital or (full_investment + lifetime_periods)")
        return None
    investment = _require_positive("full_investment", costs.full_investment, missing)
    lifetime = _require_positive("lifetime_periods", costs.lifetime_periods, missing)
    if investment is None or lifetime is None:
        return None
    if lifetime == 0:
        raise OperationalCostBlockedError("lifetime_periods must be non-zero")
    return (investment - costs.residual_value) / lifetime


def _per_unit_total(name: str, denominator: float | int | None, total: float) -> float | None:
    """Divide ``total`` by an optional evidenced denominator.

    Returns
    -------
    float | None
        Per-unit cost, or ``None`` when the denominator was not supplied.
    """
    if denominator is None:
        return None
    if denominator <= 0:
        raise OperationalCostBlockedError(f"{name} must be positive to report a per-unit cost")
    return total / denominator


def compute_cost_breakdown(
    *,
    observables: SimulatorObservables,
    service: ServiceInputs,
    costs: ExternalCostInputs,
) -> CostBreakdown:
    """Compute a transparent per-productive-km cost from typed inputs.

    Raises :class:`OperationalCostBlockedError` naming every missing input
    instead of substituting zeros. Raises on double counting (both
    ``annualized_capital`` and ``full_investment``) and on zero productive
    distance. Units, period, provenance, and source/date round-trip into the
    result; see the module docstring for the schematic form.

    Returns
    -------
    CostBreakdown
        Transparent per-unit costs with the fixed/variable split preserved.
    """
    missing: list[str] = []
    productive_m = _require_positive(
        "productive_distance_m", service.productive_distance_m, missing
    )
    if costs.annualized_capital is not None and costs.full_investment is not None:
        raise OperationalCostBlockedError(
            "double counting: supply either annualized_capital or "
            "(full_investment + lifetime_periods), not both"
        )
    capital_per_period = _resolve_capital_per_period(costs, missing)
    if missing:
        raise OperationalCostBlockedError(
            "cannot produce a total economic value; missing: " + ", ".join(missing)
        )
    assert productive_m is not None and capital_per_period is not None
    if productive_m == 0:
        raise OperationalCostBlockedError(
            "productive_distance_m is zero: per-productive-km costs are undefined "
            "(empty trips alone cannot carry a productive-km denominator)"
        )
    productive_km = productive_m / 1000.0
    empty_m = service.empty_distance_m or 0.0
    if empty_m < 0 or not np.isfinite(empty_m):
        raise OperationalCostBlockedError(f"empty_distance_m must be finite >= 0, got {empty_m!r}")
    all_km = (productive_m + empty_m) / 1000.0
    operating_hours = (service.operating_hours_s or observables.sim_time_s) / 3600.0
    fixed_per_period = capital_per_period + costs.fixed_per_period
    variable_total = costs.variable_per_km * all_km + costs.time_per_hour * operating_hours
    total = fixed_per_period + variable_total
    return CostBreakdown(
        currency=costs.currency,
        period=costs.period,
        total_per_period=total,
        fixed_per_period=fixed_per_period,
        variable_total=variable_total,
        cost_per_productive_km=total / productive_km,
        fixed_component_per_productive_km=fixed_per_period / productive_km,
        cost_per_all_km=(total / all_km) if all_km > 0 else None,
        cost_per_order=_per_unit_total("completed_orders", service.completed_orders, total),
        cost_per_passenger_km=_per_unit_total("passenger_km", service.passenger_km, total),
        productive_distance_m=productive_m,
        provenance=dict(costs.provenance),
        source=costs.source,
        source_date=costs.source_date,
    )


def availability_matrix() -> dict[str, dict[str, str]]:
    """Return a copy of the field-availability matrix for docs and exports.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of field name to status/source/units info.
    """
    return {name: dict(info) for name, info in FIELD_AVAILABILITY.items()}


def to_record(
    observables: SimulatorObservables,
    breakdown: CostBreakdown | None = None,
) -> dict[str, Any]:
    """Serialize observables (and an optional breakdown) to a JSON-safe record.

    Returns
    -------
    dict[str, Any]
        ``operational_quantities.v1`` record with simulator, availability, and
        optional cost sections.
    """
    record: dict[str, Any] = {
        "schema": "operational_quantities.v1",
        "simulator": {
            "distance_m": observables.distance_m,
            "sim_time_s": observables.sim_time_s,
            "active_time_s": observables.active_time_s,
            "idle_time_s": observables.idle_time_s,
            "goal_reached": observables.goal_reached,
            "reached_goal_step": observables.reached_goal_step,
            "exposure_duration_s": observables.exposure_duration_s,
            "dt_s": observables.dt_s,
            "queue_time_s": observables.queue_time_s,
            "world_id": observables.world_id,
            "episode_id": observables.episode_id,
            "agent_id": observables.agent_id,
            "result_status": observables.result_status,
        },
        "field_availability": availability_matrix(),
    }
    if breakdown is not None:
        record["cost"] = {
            "currency": breakdown.currency,
            "period": breakdown.period,
            "total_per_period": breakdown.total_per_period,
            "fixed_per_period": breakdown.fixed_per_period,
            "variable_total": breakdown.variable_total,
            "cost_per_productive_km": breakdown.cost_per_productive_km,
            "fixed_component_per_productive_km": breakdown.fixed_component_per_productive_km,
            "cost_per_all_km": breakdown.cost_per_all_km,
            "cost_per_order": breakdown.cost_per_order,
            "cost_per_passenger_km": breakdown.cost_per_passenger_km,
            "productive_distance_m": breakdown.productive_distance_m,
            "provenance": breakdown.provenance,
            "source": breakdown.source,
            "source_date": breakdown.source_date,
        }
    return record
