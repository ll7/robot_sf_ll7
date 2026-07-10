"""Tests for the realized-distribution audit (issue #3574 DoD item 5).

The audit summarizes configured target vs. realized per-step distributions from a
per-pedestrian control trace, so the interaction/truncation shift the #3206 smoke
could not compute is made explicit while failing closed on missing trace metadata.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.heterogeneous_population_metrics import (
    HETEROGENEOUS_POPULATION_METRICS_SCHEMA,
    RealizedDistributionSpec,
    realized_distribution_audit,
    summarize_distribution,
)


def _pedestrian(
    *,
    simulator_index: int,
    archetype: str,
    speeds: list[float],
    clearances: list[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    steps = [
        {"step": step, "speed_m_s": speed, "clearance_m": clearance}
        for step, (speed, clearance) in enumerate(zip(speeds, clearances, strict=True))
    ]
    payload: dict[str, Any] = {
        "id": f"pedestrian_{simulator_index}",
        "simulator_index": simulator_index,
        "archetype": archetype,
        "steps": steps,
    }
    if extra:
        payload.update(extra)
    return payload


def _control_trace() -> dict[str, Any]:
    return {
        "schema_version": "pedestrian-control-trace.v1",
        "pedestrians": [
            _pedestrian(
                simulator_index=0,
                archetype="cooperative",
                speeds=[1.0, 1.0],
                clearances=[2.0, 3.0],
                extra={"desired_speed_factor": 1.0, "target_speed_m_s": 1.0},
            ),
            _pedestrian(
                simulator_index=1,
                archetype="cooperative",
                speeds=[1.0, 1.0],
                clearances=[1.0, 1.0],
                extra={"desired_speed_factor": 1.0, "target_speed_m_s": 1.0},
            ),
            _pedestrian(
                simulator_index=2,
                archetype="rushed",
                speeds=[2.0, 2.0],
                clearances=[0.5, 0.5],
                extra={"desired_speed_factor": 1.5, "target_speed_m_s": 2.5},
            ),
        ],
    }


def test_summarize_distribution_reports_expected_stats() -> None:
    """The summary must report the sample size, mean, and median."""
    summary = summarize_distribution("sample", [1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(4.0)


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_summarize_distribution_fails_closed(values: list[float]) -> None:
    """Empty or non-finite samples must fail closed rather than degrade."""
    with pytest.raises(ValueError):
        summarize_distribution("sample", values)


def test_realized_audit_reports_per_archetype_distributions() -> None:
    """A ready trace yields realized + configured distributions per archetype."""
    audit = realized_distribution_audit(
        _control_trace(),
        metric_specs=[
            RealizedDistributionSpec(
                name="speed",
                realized_step_key="speed_m_s",
                configured_label_key="desired_speed_factor",
            ),
            RealizedDistributionSpec(name="clearance", realized_step_key="clearance_m"),
        ],
    )

    assert audit["schema_version"] == HETEROGENEOUS_POPULATION_METRICS_SCHEMA
    assert audit["evidence_kind"] == "realized_distribution_audit"
    assert audit["status"] == "ready"
    assert audit["pedestrian_count"] == 3

    speed = audit["metrics"]["speed"]
    assert speed["ready"] is True
    assert speed["realized"]["overall"]["mean"] == pytest.approx(8.0 / 6.0)
    assert speed["realized"]["per_archetype"]["cooperative"]["mean"] == pytest.approx(1.0)
    assert speed["realized"]["per_archetype"]["rushed"]["mean"] == pytest.approx(2.0)
    # Configured target distribution is drawn one-per-pedestrian from trace labels.
    assert speed["configured"]["overall"]["n"] == 3
    assert speed["configured"]["overall"]["mean"] == pytest.approx(3.5 / 3.0)
    # desired_speed_factor is a unitless multiplier -> no numeric shift by default.
    assert speed["configured_to_realized_shift"]["status"] == "not_computed"

    clearance = audit["metrics"]["clearance"]
    assert clearance["configured"] is None
    assert clearance["configured_to_realized_shift"] is None
    assert clearance["realized"]["per_archetype"]["rushed"]["mean"] == pytest.approx(0.5)


def test_realized_audit_computes_shift_when_comparable() -> None:
    """When the caller declares comparability, the mean shift is computed per archetype."""
    audit = realized_distribution_audit(
        _control_trace(),
        metric_specs=[
            RealizedDistributionSpec(
                name="speed",
                realized_step_key="speed_m_s",
                configured_label_key="target_speed_m_s",
                configured_is_comparable=True,
            )
        ],
    )
    shift = audit["metrics"]["speed"]["configured_to_realized_shift"]
    assert shift["status"] == "computed"
    # realized overall mean 1.3333 - configured overall mean 1.5.
    assert shift["overall_mean_delta"] == pytest.approx(8.0 / 6.0 - 1.5)
    assert shift["per_archetype_mean_delta"]["cooperative"] == pytest.approx(0.0)
    assert shift["per_archetype_mean_delta"]["rushed"] == pytest.approx(-0.5)


def test_realized_audit_blocks_on_missing_realized_field() -> None:
    """A missing realized step field fails closed for that metric and the audit."""
    audit = realized_distribution_audit(
        _control_trace(),
        metric_specs=[
            RealizedDistributionSpec(name="force", realized_step_key="force_norm"),
        ],
    )
    force = audit["metrics"]["force"]
    assert audit["status"] == "blocked"
    assert force["ready"] is False
    assert force["realized"]["status"] == "blocked"
    assert force["realized"]["overall"] is None
    assert any("force_norm" in blocker for blocker in force["blockers"])


def test_realized_audit_blocks_configured_but_keeps_realized_ready() -> None:
    """A missing configured label blocks only the configured side; realized stays ready."""
    trace = _control_trace()
    del trace["pedestrians"][2]["desired_speed_factor"]
    audit = realized_distribution_audit(
        trace,
        metric_specs=[
            RealizedDistributionSpec(
                name="speed",
                realized_step_key="speed_m_s",
                configured_label_key="desired_speed_factor",
            )
        ],
    )
    speed = audit["metrics"]["speed"]
    assert audit["status"] == "blocked"
    assert speed["ready"] is False
    assert speed["realized"]["ready"] is True
    assert speed["configured"]["ready"] is False
    assert any("desired_speed_factor" in blocker for blocker in speed["configured"]["blockers"])


def test_realized_audit_rejects_empty_pedestrians() -> None:
    """A control trace with no pedestrians must fail closed."""
    with pytest.raises(ValueError):
        realized_distribution_audit(
            {"pedestrians": []},
            metric_specs=[RealizedDistributionSpec(name="speed", realized_step_key="speed_m_s")],
        )


def test_realized_audit_rejects_duplicate_spec_names() -> None:
    """Duplicate metric names must fail closed to avoid silent overwrite."""
    with pytest.raises(ValueError):
        realized_distribution_audit(
            _control_trace(),
            metric_specs=[
                RealizedDistributionSpec(name="speed", realized_step_key="speed_m_s"),
                RealizedDistributionSpec(name="speed", realized_step_key="clearance_m"),
            ],
        )


def test_realized_audit_accepts_mapping_specs() -> None:
    """Specs may be plain mappings so callers can drive the audit from config."""
    audit = realized_distribution_audit(
        _control_trace(),
        metric_specs=[{"name": "clearance", "realized_step_key": "clearance_m"}],
    )
    assert audit["metrics"]["clearance"]["ready"] is True
