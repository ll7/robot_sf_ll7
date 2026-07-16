"""Tests for the issue #5578 robot speed-tier evidence synthesis.

These tests prove the frozen #5557 decision rule is implemented correctly and
fails closed: paired-delta estimand, Holm-Bonferroni correction, harm-threshold
classification, and visible exclusion of non-native / fallback / degraded /
failed rows. No campaign is run; synthetic per-cell summaries drive the checks.
"""

from __future__ import annotations

import math

import pytest

from robot_sf.benchmark.issue_5578_speed_tier_synthesis import (
    CONFIDENCE_LEVEL,
    HARM_THRESHOLDS,
    NOMINAL_TIER_ID,
    NON_NOMINAL_TIERS,
    PRIMARY_METRICS,
    _classify_interval,
    _holm_adjust,
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
    *,
    metrics: dict[str, float],
    execution_mode: str = "native",
) -> dict[str, object]:
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
    }


def _full_native_grid(
    *,
    nominal_metrics: dict[str, float],
    tier_metrics: dict[str, float],
    tier_id: str = "cap_3_0",
    cap: float = 3.0,
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
    # Smallest raw p gets the largest multiplier (m - rank + 1 = 3), but the
    # stepwise max keeps larger adjusted values non-decreasing; the first raw p
    # yields 0.01*3=0.03 which is <= 0.04*1=0.04 for the largest raw p.
    assert adjusted["a"] <= adjusted["c"]


def test_classify_interval_detects_harmful_collision_increase() -> None:
    """A collision-rate CI entirely above the +0.02 harm threshold is harmful."""
    ci_low, ci_high = 0.05, 0.10
    assert _classify_interval("collision_rate", ci_low, ci_high, n_pairs=6) == "materially_harmful"
    # A CI entirely below the threshold is a no-material-shift.
    assert _classify_interval("collision_rate", 0.0, 0.01, n_pairs=6) == "no_material_shift"
    # Overlapping the threshold is inconclusive.
    assert _classify_interval("collision_rate", -0.01, 0.05, n_pairs=6) == "inconclusive"
    # Too few pairs is inconclusive (never evidence).
    assert _classify_interval("collision_rate", 0.05, 0.10, n_pairs=1) == "inconclusive"


def test_classify_interval_detects_harmful_success_decrease() -> None:
    """A success-rate CI entirely below the -0.05 harm threshold is harmful."""
    assert _classify_interval("success_rate", -0.20, -0.08, n_pairs=6) == "materially_harmful"
    assert _classify_interval("success_rate", 0.0, 0.10, n_pairs=6) == "no_material_shift"


def test_synthesis_full_native_grid_reports_all_cells() -> None:
    """A complete native grid synthesizes paired deltas and a decision table."""
    cells = _full_native_grid(
        nominal_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
        tier_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
    )
    result = synthesize_speed_tier_sweep(cells)
    assert result.native_cell_count == len(cells)
    assert result.excluded_cell_count == 0
    assert result.all_native is True
    # 4 planners x 2 non-nominal tiers x 3 metrics = 24 decision rows.
    assert len(result.decision_table) == len(PLANNERS) * len(NON_NOMINAL_TIERS) * len(
        PRIMARY_METRICS
    )
    for row in result.decision_table:
        assert row["p_value_holm"] <= 1.0
        # Identical metrics -> zero pooled delta -> no material shift.
        assert row["classification"] == "no_material_shift"
        assert row["pooled_delta_mean"] == pytest.approx(0.0)


def test_synthesis_flags_fallback_rows_as_exclusions() -> None:
    """Fallback/degraded rows must appear in the exclusion table and shrink native."""
    cells = _full_native_grid(
        nominal_metrics={"success_rate": 0.8},
        tier_metrics={"success_rate": 0.8},
    )
    # Corrupt one planner's tier cells with fallback execution.
    for c in cells:
        if c["planner_id"] == "orca" and c["speed_tier_id"] != NOMINAL_TIER_ID:
            c["execution_mode"] = "fallback"
    result = synthesize_speed_tier_sweep(cells)
    assert result.all_native is False
    assert result.excluded_cell_count == len(SCENARIOS) * len(SEEDS) * len(NON_NOMINAL_TIERS)
    reasons = {e["exclusion_reason"] for e in result.exclusions}
    assert {"non_native_execution:fallback"} <= reasons
    # The orca decision rows must be absent when its tier rows are excluded.
    orca_tier_rows = [
        r
        for r in result.decision_table
        if r["planner_id"] == "orca" and r["speed_tier_id"] == "cap_3_0"
    ]
    assert orca_tier_rows == []


def test_synthesis_paired_delta_is_tier_minus_nominal() -> None:
    """The pooled delta must equal tier minus nominal mean across scenarios."""
    cells = _full_native_grid(
        nominal_metrics={"collision_rate": 0.10},
        tier_metrics={"collision_rate": 0.18},
    )
    result = synthesize_speed_tier_sweep(cells)
    row = next(
        r
        for r in result.decision_table
        if r["planner_id"] == "orca"
        and r["speed_tier_id"] == "cap_3_0"
        and r["metric"] == "collision_rate"
    )
    assert row["pooled_delta_mean"] == pytest.approx(0.08)
    assert row["classification"] == "materially_harmful"


def test_synthesis_fails_closed_on_missing_metric() -> None:
    """A cell missing a primary metric fails closed rather than guessing."""
    bad = _cell("classic_doorway_medium", NOMINAL_TIER_ID, 2.0, "orca", 111, metrics={})
    bad.pop("success_rate")
    with pytest.raises(ValueError, match="success_rate"):
        synthesize_speed_tier_sweep([bad])


def test_synthesis_holm_family_is_six_tests_per_planner() -> None:
    """The multiplicity family must be six tests per planner (2 tiers x 3 metrics)."""
    cells = _full_native_grid(
        nominal_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
        tier_metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
    )
    result = synthesize_speed_tier_sweep(cells)
    for planner in PLANNERS:
        planner_rows = [r for r in result.decision_table if r["planner_id"] == planner]
        assert len(planner_rows) == 6
    # Harm thresholds and confidence are the frozen preregistration values.
    assert HARM_THRESHOLDS["success_rate"] == -0.05
    assert HARM_THRESHOLDS["collision_rate"] == 0.02
    assert HARM_THRESHOLDS["near_miss_rate"] == 0.05
    assert math.isclose(CONFIDENCE_LEVEL, 0.95)
