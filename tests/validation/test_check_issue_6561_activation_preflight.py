"""Tests for the issue #6561 pedestrian-speed activation preflight checker.

All fixtures are synthetic and use only the reserved disjoint seed block; no
registered seed 111-140 is ever executed by these tests.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.validation import check_issue_6561_activation_preflight as preflight

PREFLIGHT_SEEDS = (311, 312, 313, 314)
TREATED_REGIMES = {"slow_distributed": 0.65, "typical_distributed": 1.3}


def _diagnostics(
    protocol: dict[str, Any],
    *,
    fraction: float = 0.9,
    time_to_target: float = 0.8,
    transient_steps: int = 5,
    spawn_mean: float = 0.5,
    spawn_peak: float = 0.52,
    mean_offset: float = 0.01,
    row_status: str = "native",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scenario in protocol["scenario_contract"]["selected_scenarios"]:
        for regime_id, configured_mean in TREATED_REGIMES.items():
            for planner in protocol["planner_contract"]["roster"]:
                for seed in PREFLIGHT_SEEDS:
                    rows.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "regime_id": regime_id,
                            "planner_id": planner["planner_id"],
                            "seed": seed,
                            "row_status": row_status,
                            "diagnostics": {
                                "configured_desired_speed_mean_m_s": configured_mean,
                                "configured_desired_speed_std_m_s": 0.2,
                                "realized_desired_speed_mean_m_s": configured_mean + mean_offset,
                                "realized_desired_speed_std_m_s": 0.21,
                                "initial_spawn_speed_mean_m_s": spawn_mean,
                                "initial_spawn_speed_peak_m_s": spawn_peak,
                                "time_to_desired_speed_target_seconds": time_to_target,
                                "acceleration_transient_steps": transient_steps,
                                "desired_speed_activation_fraction": fraction,
                            },
                        }
                    )
    return {
        "schema_version": ("robot_sf.issue_6561_pedestrian_speed_activation_diagnostics.v1"),
        "seeds": list(PREFLIGHT_SEEDS),
        "rows": rows,
    }


def test_check_only_compiles_disjoint_192_cells() -> None:
    """The check-only manifest must be deterministic, unique, and disjoint."""
    payload, protocol = preflight.load_preflight()
    first = preflight.compile_preflight_manifest(payload, protocol)
    second = preflight.compile_preflight_manifest(payload, protocol)

    assert first["identity_count"] == 192
    assert first["unique_identity_count"] == 192
    assert first["preflight_manifest_hash"] == second["preflight_manifest_hash"]
    assert first["banner"] == preflight.NOT_EVIDENCE_BANNER
    assert {row["seed"] for row in first["identities"]} == set(PREFLIGHT_SEEDS)
    assert all(
        row["registered"] is False and row["preflight"] is True for row in first["identities"]
    )
    assert all(
        {row["regime_id"] for row in first["identities"] if row["seed"] == seed}
        == set(TREATED_REGIMES)
        for seed in PREFLIGHT_SEEDS
    )


def test_validate_rejects_registered_seed_overlap() -> None:
    """A preflight seed inside 111-140 must fail closed."""
    payload, protocol = preflight.load_preflight()
    mutated = copy.deepcopy(payload)
    mutated["preflight_seed_block"]["seeds"] = [140, 311, 312, 313]

    with pytest.raises(preflight.PreflightError, match="overlap"):
        preflight.validate_preflight(mutated, protocol)


def test_validate_rejects_threshold_drift() -> None:
    """Activation thresholds must match the frozen protocol."""
    payload, protocol = preflight.load_preflight()
    mutated = copy.deepcopy(payload)
    mutated["activation_rule"]["target_tolerance_m_s"] = 0.5

    with pytest.raises(preflight.PreflightError, match="tolerance"):
        preflight.validate_preflight(mutated, protocol)


def test_validate_rejects_production_enablement() -> None:
    """The preflight packet must remain check-only."""
    payload, protocol = preflight.load_preflight()
    mutated = copy.deepcopy(payload)
    mutated["validation_contract"]["registered_rows_allowed"] = True

    with pytest.raises(preflight.PreflightError, match="registered rows"):
        preflight.validate_preflight(mutated, protocol)


def test_classify_activation_passes_synthetic_native_rows() -> None:
    """Synthetic native diagnostics within every threshold classify as activated."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(payload, protocol, _diagnostics(protocol))

    assert result["ok"] is True
    assert result["registered_seed_overlap"] is False
    assert set(result["per_regime"]) == set(TREATED_REGIMES)
    assert all(
        entry["classification"] == "intervention_activated"
        for entry in result["per_regime"].values()
    )


def test_classify_activation_flags_inactive_intervention() -> None:
    """An activation fraction below the frozen minimum must be inactive."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(payload, protocol, _diagnostics(protocol, fraction=0.5))

    assert result["ok"] is False
    assert all(
        entry["classification"] == "intervention_inactive"
        for entry in result["per_regime"].values()
    )


def test_classify_activation_flags_invalid_transient() -> None:
    """A spawn transient beyond the frozen window must be invalid."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(
        payload, protocol, _diagnostics(protocol, transient_steps=25)
    )

    assert result["ok"] is False
    assert all(
        entry["classification"] == "invalid_transient" for entry in result["per_regime"].values()
    )


def test_classify_activation_flags_invalid_initial_speed() -> None:
    """A perturbed initial spawn speed must be invalid, not inactive."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(
        payload, protocol, _diagnostics(protocol, spawn_mean=0.7, spawn_peak=0.72)
    )

    assert result["ok"] is False
    assert all(
        entry["classification"] == "invalid_initial_speed"
        for entry in result["per_regime"].values()
    )


def test_classify_fails_closed_on_forbidden_row_status() -> None:
    """Fallback or degraded rows cannot become activation evidence."""
    payload, protocol = preflight.load_preflight()
    result = preflight.classify_activation(
        payload, protocol, _diagnostics(protocol, row_status="degraded")
    )

    assert result["ok"] is False
    assert all(
        entry["classification"] == "not_evaluable" for entry in result["per_regime"].values()
    )


def test_classify_rejects_non_preflight_seed() -> None:
    """Diagnostics from registered seeds must be rejected outright."""
    payload, protocol = preflight.load_preflight()
    diagnostics = _diagnostics(protocol)
    diagnostics["rows"][0]["seed"] = 120

    with pytest.raises(preflight.PreflightError, match="non-preflight seed"):
        preflight.classify_activation(payload, protocol, diagnostics)
