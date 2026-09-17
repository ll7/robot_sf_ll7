"""Synthetic tests for the operational-quantities boundary — Issue #9350.

All fixtures are synthetic arithmetic checks: no market prices, no planner
rankings, no simulator campaigns. Missing inputs must block with named
reasons, never implicit zeros.
"""

from dataclasses import replace

import numpy as np
import pytest

from robot_sf.benchmark.metrics import EpisodeData
from robot_sf.benchmark.operational_quantities import (
    ExternalCostInputs,
    OperationalCostBlockedError,
    ServiceInputs,
    availability_matrix,
    compute_cost_breakdown,
    extract_operational_observables,
    to_record,
    trajectory_only_value,
)


def _episode(*, steps: int = 10, speed: float = 1.0, dt: float = 0.1) -> EpisodeData:
    """Straight-line synthetic episode: distance == speed * sim_time."""
    robot_pos = np.array([[speed * dt * t, 0.0] for t in range(steps)], dtype=float)
    robot_vel = np.array([[speed, 0.0] for _ in range(steps)], dtype=float)
    robot_acc = np.zeros((steps, 2), dtype=float)
    peds_pos = np.zeros((steps, 0, 2), dtype=float)
    ped_forces = np.zeros((steps, 0, 2), dtype=float)
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=np.array([speed * dt * steps, 0.0]),
        dt=dt,
        reached_goal_step=steps - 1,
    )


def _costs(**overrides) -> ExternalCostInputs:
    base = {
        "currency": "EUR",
        "price_basis_year": 2026,
        "period": "annual",
        "annualized_capital": 1000.0,
        "fixed_per_period": 500.0,
        "variable_per_km": 0.5,
        "time_per_hour": 10.0,
        "source": "synthetic-test",
        "source_date": "2026-09-17",
        "provenance": {"annualized_capital": "assumed"},
    }
    base.update(overrides)
    return ExternalCostInputs(**base)


def _service(**overrides) -> ServiceInputs:
    base = {
        "productive_distance_m": 1000.0,
        "empty_distance_m": 0.0,
        "operating_hours_s": 0.0,
    }
    base.update(overrides)
    return ServiceInputs(**base)


@pytest.mark.parametrize("field", ["fixed_per_period", "variable_per_km", "time_per_hour"])
def test_absent_cost_rate_blocks_total(field: str) -> None:
    obs = extract_operational_observables(_episode())
    with pytest.raises(OperationalCostBlockedError, match=field):
        compute_cost_breakdown(
            observables=obs, service=_service(), costs=replace(_costs(), **{field: None})
        )


def test_default_rates_remain_unknown() -> None:
    costs = ExternalCostInputs(annualized_capital=1000.0)
    with pytest.raises(OperationalCostBlockedError, match="fixed_per_period"):
        compute_cost_breakdown(
            observables=extract_operational_observables(_episode()),
            service=_service(),
            costs=costs,
        )


@pytest.mark.parametrize("field", ["empty_distance_m", "operating_hours_s"])
def test_absent_service_quantity_blocks_total(field: str) -> None:
    with pytest.raises(OperationalCostBlockedError, match=field):
        compute_cost_breakdown(
            observables=extract_operational_observables(_episode()),
            service=_service(**{field: None}),
            costs=_costs(),
        )


def test_explicit_zero_service_hours_do_not_use_episode_time() -> None:
    result = compute_cost_breakdown(
        observables=extract_operational_observables(_episode(dt=3600)),
        service=_service(operating_hours_s=0.0),
        costs=_costs(variable_per_km=0.0),
    )
    assert result.variable_total == 0.0


@pytest.mark.parametrize("field", ["fixed_per_period", "variable_per_km", "time_per_hour"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_cost_rate_blocks_total(field: str, value: float) -> None:
    with pytest.raises(OperationalCostBlockedError, match=field):
        compute_cost_breakdown(
            observables=extract_operational_observables(_episode()),
            service=_service(),
            costs=_costs(**{field: value}),
        )


def test_extract_observables_units_and_split() -> None:
    obs = extract_operational_observables(_episode(), result_status="goal_reached")
    assert obs.distance_m == pytest.approx(0.9)  # 9 steps * 0.1 m
    assert obs.sim_time_s == pytest.approx(1.0)  # 10 samples * 0.1 s
    assert obs.active_time_s == pytest.approx(1.0)
    assert obs.idle_time_s == pytest.approx(0.0)
    assert obs.goal_reached is True
    assert obs.queue_time_s is None  # no simulator queue timer
    assert obs.result_status == "goal_reached"


def test_extract_idle_split_and_sim_vs_wall_clock() -> None:
    data = _episode(steps=10, speed=1.0)
    data.robot_vel[5:] = 0.0  # robot stops halfway: wall clock keeps running
    obs = extract_operational_observables(data)
    assert obs.active_time_s == pytest.approx(0.5)
    assert obs.idle_time_s == pytest.approx(0.5)
    assert obs.sim_time_s == pytest.approx(1.0)
    assert obs.exposure_duration_s == pytest.approx(obs.sim_time_s)


def test_availability_matrix_marks_queue_unknown() -> None:
    matrix = availability_matrix()
    assert matrix["distance_m"]["status"] == "derivable"
    assert matrix["queue_time_s"]["status"] == "unknown"
    assert matrix["productive_distance_m"]["status"] == "external"
    assert matrix["capital_cost"]["status"] == "external"


def test_trajectory_only_cannot_produce_total() -> None:
    ok, reason = trajectory_only_value()
    assert ok is False
    assert "ServiceInputs" in reason and "ExternalCostInputs" in reason


def test_missing_cost_inputs_block_with_names() -> None:
    obs = extract_operational_observables(_episode())
    with pytest.raises(OperationalCostBlockedError, match="annualized_capital"):
        compute_cost_breakdown(
            observables=obs,
            service=_service(productive_distance_m=1000.0),
            costs=ExternalCostInputs(),
        )


def test_missing_service_denominator_blocks() -> None:
    obs = extract_operational_observables(_episode())
    with pytest.raises(OperationalCostBlockedError, match="productive_distance_m"):
        compute_cost_breakdown(observables=obs, service=ServiceInputs(), costs=_costs())


def test_zero_productive_distance_blocks() -> None:
    obs = extract_operational_observables(_episode())
    with pytest.raises(OperationalCostBlockedError, match="zero"):
        compute_cost_breakdown(
            observables=obs,
            service=_service(productive_distance_m=0.0, empty_distance_m=500.0),
            costs=_costs(),
        )


def test_zero_denominators_for_optional_units_block() -> None:
    obs = extract_operational_observables(_episode())
    with pytest.raises(OperationalCostBlockedError, match="completed_orders"):
        compute_cost_breakdown(
            observables=obs,
            service=_service(productive_distance_m=1000.0, completed_orders=0),
            costs=_costs(),
        )
    with pytest.raises(OperationalCostBlockedError, match="passenger_km"):
        compute_cost_breakdown(
            observables=obs,
            service=_service(productive_distance_m=1000.0, passenger_km=0.0),
            costs=_costs(),
        )


def test_double_counting_capital_blocked() -> None:
    obs = extract_operational_observables(_episode())
    with pytest.raises(OperationalCostBlockedError, match="double counting"):
        compute_cost_breakdown(
            observables=obs,
            service=_service(productive_distance_m=1000.0),
            costs=_costs(annualized_capital=1000.0, full_investment=5000.0, lifetime_periods=5.0),
        )


def test_fixed_cost_dilution_property() -> None:
    """Holding annual costs fixed, doubling productive distance halves the fixed component."""
    obs = extract_operational_observables(_episode())
    near = compute_cost_breakdown(
        observables=obs,
        service=_service(productive_distance_m=1000.0),
        costs=_costs(variable_per_km=0.0, time_per_hour=0.0),
    )
    far = compute_cost_breakdown(
        observables=obs,
        service=_service(productive_distance_m=2000.0),
        costs=_costs(variable_per_km=0.0, time_per_hour=0.0),
    )
    assert far.fixed_component_per_productive_km == pytest.approx(
        near.fixed_component_per_productive_km / 2.0
    )
    assert far.cost_per_productive_km == pytest.approx(near.cost_per_productive_km / 2.0)


def test_empty_trips_carry_all_km_but_not_productive() -> None:
    obs = extract_operational_observables(_episode())
    result = compute_cost_breakdown(
        observables=obs,
        service=_service(productive_distance_m=1000.0, empty_distance_m=1000.0),
        costs=_costs(variable_per_km=0.0, time_per_hour=0.0),
    )
    assert result.cost_per_all_km == pytest.approx(result.cost_per_productive_km / 2.0)
    assert result.cost_per_order is None
    assert result.cost_per_passenger_km is None


def test_lifecycle_period_and_provenance_round_trip() -> None:
    obs = extract_operational_observables(_episode())
    result = compute_cost_breakdown(
        observables=obs,
        service=_service(productive_distance_m=10000.0, completed_orders=50, passenger_km=200.0),
        costs=_costs(
            annualized_capital=None,
            full_investment=5000.0,
            lifetime_periods=5.0,
            period="lifecycle",
        ),
    )
    assert result.period == "lifecycle"
    assert result.provenance == {"annualized_capital": "assumed"}
    assert result.cost_per_order == pytest.approx(result.total_per_period / 50)
    assert result.cost_per_passenger_km == pytest.approx(result.total_per_period / 200.0)
    record = to_record(obs, result)
    assert record["schema"] == "operational_quantities.v1"
    assert record["cost"]["currency"] == "EUR"
    assert record["cost"]["source"] == "synthetic-test"
    assert record["simulator"]["queue_time_s"] is None
