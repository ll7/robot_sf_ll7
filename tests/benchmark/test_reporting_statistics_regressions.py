"""External-review counterexamples through real reporting functions; no rollouts."""

import json
import math
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.aggregate import _paired_metric_differences, compute_aggregates
from robot_sf.benchmark.camera_ready._reporting import (
    _build_breakdown_rows,
    _resolve_planner_metrics,
)
from robot_sf.benchmark.full_classic.aggregation import _bootstrap_params
from robot_sf.benchmark.rank_metrics import spearman_from_rank_maps
from robot_sf.benchmark.seed_variance import build_seed_variability_rows


def test_f1_table_recomputation_preserves_aggregate_evidence_cohort():
    """Ineligible rows cannot re-enter table means, clearance minima, or CIs."""
    rows = [
        {
            "algo": "A",
            "metrics": {"success": False, "snqi": 0, "collisions": 1, "min_clearance": 0.4},
            "algorithm_metadata": {},
        },
        {
            "algo": "A",
            "metrics": {"success": True, "snqi": 1, "collisions": 0, "min_clearance": 0.1},
            "algorithm_metadata": {"foresight_prediction": {"evidence_eligible": False}},
        },
    ]
    aggregate = compute_aggregates(rows)["A"]
    assert aggregate["success"]["mean"] == 0.0
    resolved, success_ci, collision_ci, snqi_ci = _resolve_planner_metrics(
        aggregate, rows, (9, 9), (9, 9), (9, 9)
    )
    assert resolved["success_mean"] == 0.0
    assert resolved["snqi_mean"] == 0.0
    assert resolved["min_clearance_m"] == 0.4
    # Recomputed table metrics deliberately invalidate stale aggregate CIs.
    for interval in (success_ci, collision_ci, snqi_ci):
        assert len(interval) == 2
        assert all(math.isnan(bound) for bound in interval)


def test_breakdown_excludes_reviewers_ineligible_success(tmp_path):
    """Scenario and family breakdowns use only the eligible failure."""
    episodes_path = tmp_path / "episodes.jsonl"
    rows = [
        {"algo": "A", "scenario_id": "s", "seed": 1001, "metrics": {"success": 0.0}},
        {
            "algo": "A",
            "scenario_id": "s",
            "seed": 1002,
            "metrics": {"success": 1.0},
            "algorithm_metadata": {"foresight_prediction": {"evidence_eligible": False}},
        },
    ]
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    scenario_rows, family_rows = _build_breakdown_rows(
        [{"planner": {"key": "A", "algo": "A"}, "episodes_path": str(episodes_path)}]
    )
    assert len(scenario_rows) == len(family_rows) == 1
    assert float(scenario_rows[0]["success_mean"]) == 0.0
    assert float(family_rows[0]["success_mean"]) == 0.0
    assert scenario_rows[0]["episodes"] == family_rows[0]["episodes"] == 1


def test_seed_variability_excludes_reviewers_ineligible_success():
    """The seed report averages only the eligible failure, with one retained seed."""
    rows = [
        {"algo": "A", "scenario_id": "s", "seed": 1001, "metrics": {"success": 0.0}},
        {
            "algo": "A",
            "scenario_id": "s",
            "seed": 1002,
            "metrics": {"success": 1.0},
            "algorithm_metadata": {"foresight_prediction": {"evidence_eligible": False}},
        },
    ]
    report_rows = build_seed_variability_rows(
        rows,
        metrics=("success",),
        campaign_id="synthetic",
        config_hash="synthetic",
        git_hash="synthetic",
        confidence_settings={"bootstrap_samples": 0},
    )
    assert len(report_rows) == 1
    report = report_rows[0]
    assert report["summary"]["success"]["mean"] == 0.0
    assert report["seed_count"] == report["episode_count"] == 1
    assert report["seed_list"] == [1001]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ({"A": 1, "B": 2, "C": 3}, {"B": 1, "C": 2}, None),
        ({"B": 3, "C": 4}, {"B": 1, "C": 2}, None),
        ({"A": 1.5, "B": 1.5, "C": 3}, {"A": 1, "B": 2, "C": 3}, 0.8660254037844387),
        ({"A": 1, "B": 2, "C": 3, "D": 4}, {"B": 1, "C": 2, "D": 3}, 1.0),
    ],
)
def test_f2_spearman_reranks_common_cohort_and_handles_ties(left, right, expected):
    """Two-item reviewer examples are unavailable; tied rho is sqrt(3)/2."""
    actual = spearman_from_rank_maps(left, right)
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_f3_duplicate_paired_keys_are_rejected_independently_of_order(reverse):
    """Repeated cells require an explicit upstream reduction policy, never last-wins."""
    left = [{"scenario_id": "s", "seed": 1001, "success": v} for v in (0, 1)]
    right = [{"scenario_id": "s", "seed": 1001, "success": v} for v in (1, 0)]
    if reverse:
        left.reverse()
        right.reverse()
    with pytest.raises(ValueError, match="duplicate.*scenario_id.*seed"):
        _paired_metric_differences(left, right)


def _seed_report(**settings):
    rows = [
        {"scenario_id": "s", "algo": "A", "seed": 1001 + i, "metrics": {"near_misses": i}}
        for i in range(30)
    ]
    return build_seed_variability_rows(
        rows,
        metrics=("near_misses",),
        campaign_id="synthetic",
        config_hash="synthetic",
        git_hash="synthetic",
        confidence_settings={"bootstrap_samples": 1000, **settings},
    )[0]


def test_f5_zero_bootstrap_seed_matches_advertised_seed():
    """PCG64 seed zero has the reviewer's independently computed percentile bounds."""
    report = _seed_report(bootstrap_seed=0)
    assert report["provenance"]["confidence"]["bootstrap_seed"] == 0
    summary = report["summary"]["near_misses"]
    assert [summary["ci_low"], summary["ci_high"]] == pytest.approx(
        [11.432500000000001, 17.466666666666665]
    )


def test_null_bootstrap_seed_resolves_to_effective_default_in_provenance():
    """Null uses the historical default and reports the seed actually used."""
    report = _seed_report(bootstrap_seed=None)
    assert report["provenance"]["confidence"]["bootstrap_seed"] == 123
    summary = report["summary"]["near_misses"]
    assert [summary["ci_low"], summary["ci_high"]] == pytest.approx(
        [11.499166666666667, 17.500833333333333]
    )


def test_zero_confidence_is_preserved():
    """Zero confidence selects the bootstrap median instead of the 95% interval."""
    report = _seed_report(bootstrap_seed=0, confidence=0)
    summary = report["summary"]["near_misses"]
    assert summary["ci_low"] == summary["ci_high"]


def test_full_classic_zero_bootstrap_samples_disable_resampling():
    """The full-classic config resolver must preserve an explicit disabled bootstrap."""
    assert _bootstrap_params(SimpleNamespace(bootstrap_samples=0))[0] == 0


@pytest.mark.parametrize(
    ("group_by", "expected"),
    [("seed", {"1001", "1002"}), ("scenario_params.ped_density", {"0.05", "0.2"})],
)
def test_f6_numeric_group_fields_do_not_fall_back_to_algorithm(group_by, expected):
    """Present numeric fields are distinct groups even when algo is also present."""
    rows = [
        {
            "algo": "A",
            "seed": 1001,
            "scenario_params": {"ped_density": 0.05},
            "metrics": {"success": 0},
        },
        {
            "algo": "A",
            "seed": 1002,
            "scenario_params": {"ped_density": 0.2},
            "metrics": {"success": 1},
        },
    ]
    groups = compute_aggregates(rows, group_by=group_by)
    assert set(groups) - {"_meta"} == expected
