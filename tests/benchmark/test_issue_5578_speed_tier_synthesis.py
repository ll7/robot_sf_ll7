"""Tests for the issue #5578 robot speed-tier evidence synthesis.

These tests prove the frozen #5557 decision rule is implemented correctly and
fails closed: paired-delta estimand, Holm-Bonferroni correction, harm-threshold
classification, and visible exclusion of non-native / fallback / degraded /
failed rows. No campaign is run; synthetic per-cell summaries drive the checks.
"""

from __future__ import annotations

import math
import statistics

import pytest

from robot_sf.benchmark.issue_5578_speed_tier_synthesis import (
    CONFIDENCE_LEVEL,
    HARM_THRESHOLDS,
    NOMINAL_TIER_ID,
    NON_NOMINAL_TIERS,
    PRIMARY_METRICS,
    _build_decision_table,
    _classify_interval,
    _holm_adjust,
    _holm_adjust_by_planner,
    _z_critical,
    classify_excluded,
    main,
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
    # Smallest raw p gets the largest multiplier (m - rank + 1 = 3), but the
    # stepwise max keeps larger adjusted values non-decreasing; the first raw p
    # yields 0.01*3=0.03 which is <= 0.04*1=0.04 for the largest raw p.
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
    scenario_effects = {scenario: 0.01 * (index + 1) for index, scenario in enumerate(SCENARIOS)}
    for cell in cells:
        if cell["speed_tier_id"] != NOMINAL_TIER_ID:
            cell["success_rate"] = 0.8 + scenario_effects[str(cell["scenario_id"])]
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
        assert row["n_scenarios"] == len(SCENARIOS)
        assert row["classification"] == "no_material_shift"
        assert row["adjusted_confidence_level"] >= CONFIDENCE_LEVEL
        assert set(row["typed_collision_breakdown"]) == {
            "ped_collision_rate",
            "obstacle_collision_rate",
            "agent_collision_rate",
            "unclassified_collision_rate",
        }
    success_row = next(
        row
        for row in result.decision_table
        if row["planner_id"] == "orca"
        and row["speed_tier_id"] == "cap_3_0"
        and row["metric"] == "success_rate"
    )
    expected_se = statistics.stdev(scenario_effects.values()) / math.sqrt(len(SCENARIOS))
    assert success_row["pooled_delta_mean"] == pytest.approx(0.035)
    assert success_row["pooled_delta_se"] == pytest.approx(expected_se)


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
        expected = _holm_adjust([(row["test_id"], row["p_value_raw"]) for row in planner_rows])
        assert {row["test_id"]: row["p_value_holm"] for row in planner_rows} == expected
    # Harm thresholds and confidence are the frozen preregistration values.
    assert HARM_THRESHOLDS["success_rate"] == -0.05
    assert HARM_THRESHOLDS["collision_rate"] == 0.02
    assert HARM_THRESHOLDS["near_miss_rate"] == 0.05
    assert math.isclose(CONFIDENCE_LEVEL, 0.95)


def test_holm_families_are_independent_per_planner() -> None:
    """Distinct planner inputs receive independent six-test Holm adjustments."""
    values = [
        *(("a", f"a{i}", p) for i, p in enumerate((0.01, 0.02, 0.03, 0.04, 0.05, 0.06))),
        *(("b", f"b{i}", p) for i, p in enumerate((0.001, 0.2, 0.3, 0.4, 0.5, 0.6))),
    ]
    adjusted, confidence = _holm_adjust_by_planner(values)
    assert {key: adjusted[key] for key in adjusted if key.startswith("a")} == _holm_adjust(
        [(test_id, p_value) for planner, test_id, p_value in values if planner == "a"]
    )
    assert {key: adjusted[key] for key in adjusted if key.startswith("b")} == _holm_adjust(
        [(test_id, p_value) for planner, test_id, p_value in values if planner == "b"]
    )
    assert confidence["a0"] == pytest.approx(1.0 - 0.05 / 6.0)
    assert confidence["b0"] == pytest.approx(1.0 - 0.05 / 6.0)


def test_holm_step_down_blocks_decisive_later_classifications() -> None:
    """A failed ordered comparison makes every later family row inconclusive."""
    raw_p_values = (0.001, 0.011, 0.012, 0.5, 0.6, 0.7)
    bootstrap_distributions = (
        [0.1] * 100,
        [-0.01] * 50 + [0.05] * 50,
        *([0.1] * 100 for _ in range(4)),
    )
    summaries = [
        {
            "test_id": f"orca__test_{index}",
            "planner_id": "orca",
            "speed_tier_id": "cap_3_0",
            "metric": "collision_rate",
            "n_scenarios": len(SCENARIOS),
            "pooled_delta_mean": 0.1,
            "pooled_delta_se": 0.01,
            "ci_low_unadjusted": 0.08,
            "ci_high_unadjusted": 0.12,
            "p_value_raw": p_value,
            "typed_collision_breakdown": {},
            "_bootstrap_distribution": bootstrap_distributions[index],
        }
        for index, p_value in enumerate(raw_p_values)
    ]
    decision_table, _ = _build_decision_table(summaries)
    assert decision_table[0]["holm_step_down_eligible"] is True
    assert decision_table[0]["classification"] == "materially_harmful"
    assert decision_table[1]["holm_step_down_eligible"] is True
    assert decision_table[1]["classification"] == "inconclusive"
    assert all(row["holm_step_down_eligible"] is False for row in decision_table[2:])
    assert all(row["classification"] == "inconclusive" for row in decision_table[2:])


def test_synthesis_rejects_incomplete_duplicate_and_drifted_grids() -> None:
    """Incomplete, duplicate, and dimension-drifted inputs cannot become evidence."""
    cells = _full_native_grid(nominal_metrics={}, tier_metrics={})
    with pytest.raises(ValueError, match="grid incomplete"):
        synthesize_speed_tier_sweep(cells[:-1])
    with pytest.raises(ValueError, match="duplicate declared cell identity"):
        synthesize_speed_tier_sweep([*cells, dict(cells[0])])
    drifted = [dict(cell) for cell in cells]
    drifted[0]["speed_cap_m_s"] = 2.1
    with pytest.raises(ValueError, match="speed cap drift"):
        synthesize_speed_tier_sweep(drifted)


def test_custom_declared_dimensions_remain_smoke_not_benchmark_evidence() -> None:
    """A complete caller-defined mini-grid cannot impersonate the frozen suite."""
    cells = [
        _cell(
            "classic_doorway_medium",
            tier_id,
            cap,
            "orca",
            111,
            metrics={"success_rate": 0.8, "collision_rate": 0.1, "near_miss_rate": 0.2},
        )
        for tier_id, cap in ((NOMINAL_TIER_ID, 2.0), ("cap_3_0", 3.0), ("cap_4_2", 4.2))
    ]
    result = synthesize_speed_tier_sweep(
        cells,
        declared_scenarios={"classic_doorway_medium"},
        declared_planners={"orca"},
        declared_seeds={111},
    )
    assert result.grid_complete is True
    assert result.evidence_status == "smoke_or_incomplete_not_benchmark_evidence"


def test_cli_demo_is_explicit_smoke_not_benchmark_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The built-in partial demo never prints a benchmark-success PASS label."""
    assert main([]) == 0
    output = capsys.readouterr().out
    assert output.startswith("SMOKE (not benchmark evidence):")
