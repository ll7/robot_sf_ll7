"""Tests for the reactive-vs-replay pedestrian-reactivity ablation campaign (#3573)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robot_sf.benchmark.reactivity_ablation import REACTIVITY_ABLATION_SCHEMA, ReactivityContrast

# scripts/benchmark/ has no package __init__; load the campaign module by path (repo convention).
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "run_reactivity_ablation_campaign_issue_3573.py"
)
_SPEC = importlib.util.spec_from_file_location("_issue_3573_reactivity_campaign", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
aggregate = _MODULE.aggregate
build_contrast = _MODULE.build_contrast
run_campaign = _MODULE.run_campaign


def _record(collisions: float, near_misses: float, clearance: float) -> dict:
    """Build a minimal episode record with the metrics the aggregator reads."""
    return {
        "metrics": {
            "total_collision_count": collisions,
            "near_misses": near_misses,
            "min_clearance": clearance,
        }
    }


def test_aggregate_computes_rates_and_mean_clearance() -> None:
    """Aggregation must report collision/near-miss rates and mean clearance."""
    records = [_record(0, 0, 2.0), _record(1, 0, 0.5), _record(0, 2, 1.0), _record(0, 0, 1.5)]
    agg = aggregate(records)

    assert agg["collision_rate"] == pytest.approx(0.25)  # 1 of 4
    assert agg["near_miss_rate"] == pytest.approx(0.25)  # 1 of 4
    assert agg["min_separation_m"] == pytest.approx(1.25)  # mean(2,0.5,1,1.5)
    assert agg["episodes"] == 4


def test_aggregate_rejects_empty() -> None:
    """A condition with no episodes cannot be aggregated."""
    with pytest.raises(ValueError):
        aggregate([])


def test_build_contrast_maps_conditions() -> None:
    """The per-planner contrast must carry both conditions' aggregates."""
    reactive = {"collision_rate": 0.1, "near_miss_rate": 0.2, "min_separation_m": 0.6}
    replay = {"collision_rate": 0.3, "near_miss_rate": 0.4, "min_separation_m": 0.4}
    contrast = build_contrast("orca", reactive, replay)

    assert isinstance(contrast, ReactivityContrast)
    assert contrast.planner == "orca"
    assert contrast.reactive_collision_rate == 0.1
    assert contrast.replay_min_separation_m == 0.4


def test_campaign_runs_real_runner_and_feeds_quantifier(tmp_path: Path) -> None:
    """End-to-end smoke: the campaign runs the real runner and assembles the quantifier report.

    Deliberately tiny (1 planner, 1 seed, short horizon). It asserts wiring, not a reactivity
    effect — at this short horizon the robot has not reached the crossing (see the horizon caveat
    in the script's claim boundary).
    """
    report = run_campaign(
        Path("configs/scenarios/sets/classic_crossing_subset.yaml"),
        seeds=[101],
        planners=("goal",),
        out_dir=tmp_path,
        horizon=60,
        dt=0.1,
        workers=1,
    )

    assert report["issue"] == 3573
    assert report["evidence_tier"] == "diagnostic"
    assert "goal" in report["per_planner"]
    for condition in ("reactive", "replay"):
        block = report["per_planner"]["goal"][condition]
        assert {"collision_rate", "near_miss_rate", "min_separation_m"} <= set(block)
    assert report["assessment"]["schema_version"] == REACTIVITY_ABLATION_SCHEMA
    assert report["assessment"]["n_planners"] == 1
