"""Tests for the multi-seed emergent-phenomena campaign (issue #5149).

Covers the canonical verdict mapping, the multi-seed campaign grid and its
determinism, run-record serialization, and the aggregation statistics with
conservative (weaker-verdict) tie-breaking.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.research.emergent_phenomena import (
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    default_scenario_set,
    derive_phenomenon_verdict,
)
from robot_sf.research.emergent_phenomena_campaign import (
    DEFAULT_CAMPAIGN_SEEDS,
    aggregate_run_records,
    result_to_run_record,
    run_multiseed_campaign,
)


def _small_corridor(seed: int = 1) -> ScenarioConfig:
    """Return a tiny corridor scenario for fast tests."""
    return ScenarioConfig(
        name="bidirectional_corridor",
        length=10.0,
        half_width=2.0,
        n_pedestrians=6,
        seed=seed,
        n_steps=30,
    )


@pytest.mark.parametrize(
    ("scenario", "ops", "expected"),
    [
        ("bidirectional_corridor", {"lane_segregation_index": 0.6}, "clearly_present"),
        ("bidirectional_corridor", {"lane_segregation_index": 0.2}, "weak_partial"),
        ("bidirectional_corridor", {"lane_segregation_index": 0.05}, "absent_or_negligible"),
        ("narrow_doorway", {"oscillation_flips": 3.0}, "clearly_present"),
        ("narrow_doorway", {"oscillation_flips": 1.0}, "absent_or_negligible"),
        ("high_density_exit", {"exit_density_ratio": 5.0}, "clearly_present"),
        ("high_density_exit", {"exit_density_ratio": 1.2}, "absent_or_negligible"),
        ("unknown_scenario", {}, "unknown"),
    ],
)
def test_derive_phenomenon_verdict_thresholds(scenario, ops, expected):
    assert derive_phenomenon_verdict(scenario, ops) == expected


def test_default_campaign_seeds_pinned_and_unique():
    assert len(DEFAULT_CAMPAIGN_SEEDS) == 10
    assert len(set(DEFAULT_CAMPAIGN_SEEDS)) == 10
    # First seed stays the pinned 2026-07 exhibit seed for continuity.
    assert DEFAULT_CAMPAIGN_SEEDS[0] == 5149


def test_default_scenario_set_is_public_and_canonical():
    scenarios = default_scenario_set()
    assert [s.name for s in scenarios] == [
        "bidirectional_corridor",
        "narrow_doorway",
        "high_density_exit",
    ]
    assert all(s.seed == 5149 for s in scenarios)


def test_run_multiseed_campaign_covers_grid_and_is_deterministic():
    seeds = [11, 12]
    scenarios = [_small_corridor()]
    calibrations = [RELEASED_DEFAULT_CALIBRATION]
    results = run_multiseed_campaign(seeds=seeds, scenarios=scenarios, calibrations=calibrations)
    assert len(results) == len(scenarios) * len(calibrations) * len(seeds)
    assert [r.scenario.seed for r in results] == seeds

    repeat = run_multiseed_campaign(seeds=seeds, scenarios=scenarios, calibrations=calibrations)
    for first, second in zip(results, repeat, strict=True):
        assert first.order_parameters == second.order_parameters


def test_result_to_run_record_fields():
    [result] = run_multiseed_campaign(
        seeds=[7],
        scenarios=[_small_corridor()],
        calibrations=[RELEASED_DEFAULT_CALIBRATION],
    )
    record = result_to_run_record(result)
    assert record["record_type"] == "emergent_phenomena_run.v1"
    assert record["scenario"] == "bidirectional_corridor"
    assert record["calibration"] == "released_default"
    assert record["seed"] == 7
    assert record["n_pedestrians"] == 6
    assert record["phenomenon_verdict"] in {
        "clearly_present",
        "weak_partial",
        "absent_or_negligible",
    }
    assert set(record["order_parameters"]) == {"lane_segregation_index", "lane_purity"}


def _synthetic_record(scenario: str, calibration: str, seed: int, value: float, verdict: str):
    return {
        "record_type": "emergent_phenomena_run.v1",
        "scenario": scenario,
        "calibration": calibration,
        "seed": seed,
        "order_parameters": {"lane_segregation_index": value},
        "phenomenon_verdict": verdict,
    }


def test_aggregate_run_records_statistics():
    records = [
        _synthetic_record("bidirectional_corridor", "released_default", 1, 0.2, "weak_partial"),
        _synthetic_record("bidirectional_corridor", "released_default", 2, 0.4, "weak_partial"),
        _synthetic_record("bidirectional_corridor", "released_default", 3, 0.6, "clearly_present"),
    ]
    [agg] = aggregate_run_records(records)
    assert agg["scenario"] == "bidirectional_corridor"
    assert agg["n_seeds"] == 3
    assert agg["seeds"] == [1, 2, 3]
    stats = agg["order_parameter_stats"]["lane_segregation_index"]
    assert stats["mean"] == pytest.approx(0.4)
    assert stats["min"] == pytest.approx(0.2)
    assert stats["max"] == pytest.approx(0.6)
    assert stats["median"] == pytest.approx(0.4)
    assert stats["std"] == pytest.approx(0.2)
    assert agg["verdict_counts"] == {"clearly_present": 1, "weak_partial": 2}
    assert agg["majority_verdict"] == "weak_partial"


def test_aggregate_run_records_tie_breaks_toward_weaker_verdict():
    records = [
        _synthetic_record("bidirectional_corridor", "released_default", 1, 0.6, "clearly_present"),
        _synthetic_record("bidirectional_corridor", "released_default", 2, 0.2, "weak_partial"),
    ]
    [agg] = aggregate_run_records(records)
    assert agg["majority_verdict"] == "weak_partial"


def test_aggregate_run_records_groups_by_scenario_and_calibration():
    records = [
        _synthetic_record("bidirectional_corridor", "released_default", 1, 0.2, "weak_partial"),
        _synthetic_record("bidirectional_corridor", "literature_typical", 1, 0.3, "weak_partial"),
        _synthetic_record("narrow_doorway", "released_default", 1, 3.0, "clearly_present"),
    ]
    aggregates = aggregate_run_records(records)
    keys = [(a["scenario"], a["calibration"]) for a in aggregates]
    assert keys == sorted(keys)
    assert len(aggregates) == 3


# Aggregation now shares its majority-verdict rule with the replay-video
# selector, so the archived bundle doubles as a regression fixture: re-deriving
# its reported verdicts from its own runs.jsonl must reproduce them exactly.
_BUNDLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs/context/evidence/issue_5149_emergent_phenomena_multiseed_2026-08"
)


def test_aggregate_run_records_reproduces_archived_bundle_verdicts():
    runs_path = _BUNDLE_DIR / "runs.jsonl"
    summary_path = _BUNDLE_DIR / "summary.json"
    if not (runs_path.is_file() and summary_path.is_file()):
        pytest.skip(f"campaign bundle not present at {_BUNDLE_DIR}")
    records = [
        json.loads(line)
        for line in runs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    archived = {
        (agg["scenario"], agg["calibration"]): agg["majority_verdict"]
        for agg in json.loads(summary_path.read_text(encoding="utf-8"))["aggregates"]
    }
    recomputed = {
        (agg["scenario"], agg["calibration"]): agg["majority_verdict"]
        for agg in aggregate_run_records(records)
    }
    assert recomputed == archived
