"""Tests for the issue #5578 robot speed-tier evidence synthesis.

These tests prove the frozen #5557 / #6100 decision rule is implemented correctly and
fails closed: paired-delta estimand, Holm-Bonferroni correction, harm-threshold
classification, cap-inactive intervention classification, and visible exclusion of non-native / fallback / degraded /
failed rows. No campaign is run; synthetic per-cell summaries drive the checks.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.issue_5578_speed_tier_synthesis import (
    CONFIDENCE_LEVEL,
    NOMINAL_TIER_ID,
    NON_NOMINAL_TIERS,
    PRIMARY_METRICS,
    _classify_interval,
    _holm_adjust,
    _margin_aligned_one_sided_p_value,
    _z_critical,
    classify_excluded,
    parse_cell,
    synthesize_speed_tier_sweep,
)

SCENARIOS = (
    "classic_head_on_corridor_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_merging_medium",
    "classic_overtaking_medium",
    "classic_station_platform_medium",
)
PLANNERS = (
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "ppo",
    "orca",
    "prediction_planner",
)
SEEDS = list(range(111, 141))


def _cell(
    scenario_id: str,
    tier_id: str,
    cap: float,
    planner_id: str,
    seed: int,
    metrics: dict[str, float],
    execution_mode: str = "native",
    **kwargs: Any,
) -> dict[str, object]:
    fraction_above_2_0_mps = kwargs.get("fraction_above_2_0_mps")
    realized_speed_peak_m_s = kwargs.get("realized_speed_peak_m_s")
    frac = (
        fraction_above_2_0_mps
        if fraction_above_2_0_mps is not None
        else (0.5 if cap > 2.0 else 0.0)
    )
    peak = realized_speed_peak_m_s if realized_speed_peak_m_s is not None else cap
    return {
        "scenario_id": scenario_id,
        "speed_tier_id": tier_id,
        "speed_cap_m_s": cap,
        "planner_id": planner_id,
        "seed": seed,
        "horizon_steps": 600,
        "dt_seconds": 0.1,
        "execution_mode": execution_mode,
        "success_rate": metrics.get("success_rate", 0.0),
        "collision_rate": metrics.get("collision_rate", 0.0),
        "near_miss_rate": metrics.get("near_miss_rate", 0.0),
        "ped_collision_rate": 0.0,
        "obstacle_collision_rate": 0.0,
        "agent_collision_rate": 0.0,
        "unclassified_collision_rate": 0.0,
        "commanded_speed_mean_m_s": cap * 0.9,
        "realized_speed_mean_m_s": cap * 0.85,
        "realized_speed_peak_m_s": peak,
        "fraction_above_2_0_mps": frac,
        "cap_saturation_fraction": 0.3,
        "resolved_actuation_envelope": {
            "drive_model": "bicycle_drive",
            "max_velocity": cap,
            "max_accel": cap * 0.5,
            "max_decel": cap,
            "stopping_distance": cap * 0.5,
        },
        "time_to_goal_norm": 0.5,
        "total_exposure_seconds": 30.0,
        "travel_distance_m": 60.0,
        "mean_clearance_m": 1.2,
        "min_clearance_m": 0.4,
    }


def _full_native_grid(
    *,
    nominal_metrics: dict[str, float],
    tier_metrics: dict[str, float],
    fraction_above_2_0_mps: float | None = None,
    realized_speed_peak_m_s: float | None = None,
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for planner in PLANNERS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                cells.append(
                    _cell(
                        scenario,
                        NOMINAL_TIER_ID,
                        2.0,
                        planner,
                        seed,
                        metrics=nominal_metrics,
                    )
                )
                for t_id, t_cap in (("cap_3_0", 3.0), ("cap_4_2", 4.2)):
                    cells.append(
                        _cell(
                            scenario,
                            t_id,
                            t_cap,
                            planner,
                            seed,
                            metrics=tier_metrics,
                            fraction_above_2_0_mps=fraction_above_2_0_mps,
                            realized_speed_peak_m_s=realized_speed_peak_m_s,
                        )
                    )
    return cells


def test_parse_cell_validates_required_keys() -> None:
    """A well-formed cell parses; a missing key fails closed."""
    cell = parse_cell(
        _cell(
            "classic_doorway_medium",
            NOMINAL_TIER_ID,
            2.0,
            "orca",
            111,
            metrics={"success_rate": 0.9},
        )
    )
    assert cell.planner_id == "orca"
    assert cell.speed_cap_m_s == pytest.approx(2.0)
    bad = dict(_cell("classic_doorway_medium", NOMINAL_TIER_ID, 2.0, "orca", 111, metrics={}))
    bad.pop("scenario_id")
    with pytest.raises(ValueError, match="scenario_id"):
        parse_cell(bad)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scenario_id", None, "non-empty string"),
        ("seed", True, "must be an integer"),
        ("seed", 111.5, "must be an integer"),
        ("horizon_steps", "600", "must be an integer"),
        ("dt_seconds", 0.0, "must be positive"),
        ("speed_cap_m_s", -1.0, "must be positive"),
        ("success_rate", 1.1, "must be in.*0, 1"),
    ],
)
def test_parse_cell_rejects_coercive_or_out_of_range_values(
    field: str, value: object, message: str
) -> None:
    """Malformed identifiers, integral fields, durations, and rates fail closed."""
    row = _cell(
        "classic_doorway_medium",
        NOMINAL_TIER_ID,
        2.0,
        "orca",
        111,
        metrics={"success_rate": 0.9},
    )
    row[field] = value
    with pytest.raises(ValueError, match=message):
        parse_cell(row)


def test_parse_cell_requires_typed_collision_breakdown() -> None:
    """Typed collision rates are mandatory rather than silently omitted."""
    row = _cell("classic_doorway_medium", NOMINAL_TIER_ID, 2.0, "orca", 111, metrics={})
    row.pop("ped_collision_rate")
    with pytest.raises(ValueError, match="typed collision metric"):
        parse_cell(row)


def test_classify_excluded_marks_non_native_rows() -> None:
    """Only native rows pass; fallback/degraded/failed are visible exclusions."""
    native = parse_cell(_cell("s", "t", 2.0, "orca", 1, metrics={}, execution_mode="native"))
    assert classify_excluded(native) is None
    for mode in ("fallback", "degraded", "failed"):
        cell = parse_cell(_cell("s", "t", 2.0, "orca", 1, metrics={}, execution_mode=mode))
        assert classify_excluded(cell) is not None
        assert (
            "fallback" in classify_excluded(cell)
            or "degraded" in classify_excluded(cell)
            or "failed" in classify_excluded(cell)
        )


def test_holm_adjust_is_monotone_and_stepwise() -> None:
    """Holm adjustment must be monotone non-decreasing in the sorted order."""
    p_values = [("a", 0.01), ("b", 0.02), ("c", 0.04)]
    adjusted = _holm_adjust(p_values)
    vals = [adjusted[k] for k, _ in p_values]
    assert vals == sorted(vals)
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert adjusted["a"] <= adjusted["c"]


def test_two_sided_critical_value_matches_declared_confidence() -> None:
    """The 95% two-sided normal critical value is approximately 1.96."""
    assert _z_critical(0.95) == pytest.approx(1.96, abs=0.005)
    with pytest.raises(ValueError, match="between 0 and 1"):
        _z_critical(1.0)


def test_classify_interval_detects_harmful_collision_increase() -> None:
    """A collision-rate CI entirely above the +0.02 harm threshold is harmful."""
    ci_low, ci_high = 0.05, 0.10
    assert _classify_interval("collision_rate", ci_low, ci_high, n_pairs=6) == "materially_harmful"
    assert _classify_interval("collision_rate", 0.0, 0.01, n_pairs=6) == "no_material_shift"
    assert _classify_interval("collision_rate", -0.01, 0.05, n_pairs=6) == "inconclusive"
    assert _classify_interval("collision_rate", 0.05, 0.10, n_pairs=1) == "inconclusive"


def test_classify_interval_detects_harmful_success_decrease() -> None:
    """A success-rate CI entirely below the -0.05 harm threshold is harmful."""
    assert _classify_interval("success_rate", -0.20, -0.08, n_pairs=6) == "materially_harmful"
    assert _classify_interval("success_rate", 0.0, 0.10, n_pairs=6) == "no_material_shift"


def test_classify_interval_handles_cap_inactive_case() -> None:
    """When intervention is not activated, classification must be intervention_not_activated."""
    assert (
        _classify_interval("success_rate", 0.0, 0.10, n_pairs=6, intervention_activated=False)
        == "intervention_not_activated"
    )


def test_synthesis_full_native_grid_reports_all_cells() -> None:
    """A complete native grid synthesizes paired deltas and a decision table."""
    cells = _full_native_grid(
        nominal_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
        tier_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
    )
    scenario_effects = {scenario: 0.01 * (index + 1) for index, scenario in enumerate(SCENARIOS)}
    for cell in cells:
        if cell["speed_tier_id"] != NOMINAL_TIER_ID:
            cell["success_rate"] = 0.8 + scenario_effects[str(cell["scenario_id"])]
    result = synthesize_speed_tier_sweep(cells)
    assert result.native_cell_count == len(cells)
    assert result.excluded_cell_count == 0
    assert result.all_native is True
    assert len(result.decision_table) == len(PLANNERS) * len(NON_NOMINAL_TIERS) * len(
        PRIMARY_METRICS
    )
    for row in result.decision_table:
        assert row["p_value_holm"] <= 1.0
        assert row["n_scenarios"] == len(SCENARIOS)
        assert row["classification"] == "no_material_shift"
        assert row["adjusted_confidence_level"] >= CONFIDENCE_LEVEL
        assert set(row["typed_collision_breakdown"]) == {
            "ped_collision_rate",
            "obstacle_collision_rate",
            "agent_collision_rate",
            "unclassified_collision_rate",
        }
        assert "activation_diagnostics_summary" in row
        assert "exposure_summary" in row


def test_synthesis_flags_fallback_rows_as_exclusions() -> None:
    """Fallback/degraded rows must appear in the exclusion table and shrink native."""
    cells = _full_native_grid(
        nominal_metrics={"success_rate": 0.8},
        tier_metrics={"success_rate": 0.8},
    )
    for c in cells:
        if c["planner_id"] == "orca" and c["speed_tier_id"] != NOMINAL_TIER_ID:
            c["execution_mode"] = "fallback"
    result = synthesize_speed_tier_sweep(cells)
    assert result.all_native is False
    assert result.excluded_cell_count == len(SCENARIOS) * len(SEEDS) * len(NON_NOMINAL_TIERS)
    reasons = {e["exclusion_reason"] for e in result.exclusions}
    assert {"non_native_execution:fallback"} <= reasons
    orca_tier_rows = [
        r
        for r in result.decision_table
        if r["planner_id"] == "orca" and r["speed_tier_id"] == "cap_3_0"
    ]
    assert orca_tier_rows == []


def test_synthesis_cap_inactive_cells_classified_as_intervention_not_activated() -> None:
    """Cells failing the minimum activation rule are classified as intervention_not_activated."""
    cells = _full_native_grid(
        nominal_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
        tier_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
        fraction_above_2_0_mps=0.01,
        realized_speed_peak_m_s=1.9,
    )
    result = synthesize_speed_tier_sweep(cells)
    for row in result.decision_table:
        assert row["intervention_activated"] is False
        assert row["classification"] == "intervention_not_activated"


def test_descriptive_ranking_stability_computation() -> None:
    """Descriptive ranking stability computes rankings and is explicitly marked descriptive_only."""
    cells = _full_native_grid(
        nominal_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
        tier_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
    )
    result = synthesize_speed_tier_sweep(cells)
    ranking = result.descriptive_ranking_stability
    assert ranking["scope"] == "descriptive_only"
    assert "nominal_ranking" in ranking
    assert "tier_rankings" in ranking


def test_margin_aligned_one_sided_p_value() -> None:
    """Margin-aligned one-sided p-value measures proportion of bootstrap mass in harm region."""
    sorted_deltas = [-0.10, -0.08, -0.07, -0.06, -0.05, 0.0, 0.01, 0.02, 0.03, 0.04]
    p_val = _margin_aligned_one_sided_p_value(sorted_deltas, "success_rate")
    assert p_val == pytest.approx(6 / 11)
