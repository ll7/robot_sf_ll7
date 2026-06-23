"""Tests for the issue #1554 S20/S30 seed-budget bundle builder.

These exercise the small-synthetic-rows contract: per-planner-by-seed summary
shape, bootstrap-uncertainty wiring through the canonical functions, the
seed-resampling rank-flip detection, fail-closed classification on degraded /
missing rows, and the blocked_until_run path when S20/S30 rows are absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/benchmark/build_s20_s30_seed_budget_bundle_issue_1554.py"


def _load_module():
    """Import the bundle builder script as a module."""
    spec = importlib.util.spec_from_file_location("s20_s30_bundle", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["s20_s30_bundle"] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _make_rows(*, planners, seeds, status="native", success_by_planner=None):
    """Build small synthetic episode rows for the given planners/seeds."""
    success_by_planner = success_by_planner or dict.fromkeys(planners, 0.5)
    rows = []
    for planner in planners:
        base_success = success_by_planner[planner]
        for seed in seeds:
            # Deterministic per-seed variation so rank-flip can be exercised.
            local = base_success + ((seed % 3) - 1) * 0.15
            rows.append(
                {
                    "run_id": "r1",
                    "episode_id": f"{planner}-{seed}",
                    "planner": planner,
                    "planner_key": planner,
                    "algo": planner,
                    "scenario_id": "s1",
                    "scenario_family": "crossing",
                    "seed": seed,
                    "row_status": status,
                    "artifact_uri": "x",
                    "artifact_sha256": "y",
                    # Canonical episode-record shape: outcome metrics under metrics.*
                    # so build_seed_variability_rows / flatten_metrics consume them.
                    "metrics": {
                        "success": 1.0 if local > 0.5 else 0.0,
                        "collisions": 0.0,
                        "near_misses": float(seed % 4),
                        "time_to_goal_norm": 0.4 + 0.01 * (seed % 5),
                        "snqi": base_success - 0.01 * (seed % 3),
                    },
                }
            )
    return rows


def _store(rows, source="synthetic", kind="json"):
    """Wrap rows in a StoreRows instance."""
    return mod.StoreRows(rows=list(rows), source=source, source_kind=kind)


def test_blocked_when_no_rows():
    """Empty store yields blocked_until_run naming the missing tier."""
    bundle = mod.build_bundle(_store([], source="missing", kind="missing"), git_head="abc")
    assert bundle["status"] == "blocked_until_run"
    assert bundle["missing_seed_tier"] == "s20_and_s30"
    assert "S10" in bundle["blocked_reason"]


def test_blocked_when_below_s20_tier():
    """A 10-seed (S10) store blocks because it is below the S20 paper tier."""
    rows = _make_rows(planners=["goal", "orca"], seeds=range(111, 121))
    bundle = mod.build_bundle(_store(rows), git_head="abc")
    assert bundle["status"] == "blocked_until_run"
    assert bundle["achieved_seed_tier"] == "s10"
    assert bundle["min_seeds_per_planner"] == 10


def test_blocked_when_all_rows_fail_closed():
    """A store of only degraded rows blocks (no native/adapter rows)."""
    rows = _make_rows(planners=["goal", "orca"], seeds=range(111, 131), status="degraded")
    bundle = mod.build_bundle(_store(rows), git_head="abc")
    assert bundle["status"] == "blocked_until_run"
    assert "fail-closed" in bundle["blocked_reason"]


def test_fail_closed_classification_reasons():
    """Degraded/missing rows are partitioned out with explicit reasons."""
    rows = _make_rows(planners=["goal"], seeds=range(111, 131), status="native")
    rows += _make_rows(planners=["orca"], seeds=range(111, 131), status="fallback")
    rows.append({"planner": "broken", "seed": 111, "success": 1.0})  # missing row_status
    classification = mod.classify_rows(rows)
    assert all(r["row_status"] == "native" for r in classification["valid_rows"])
    reasons = {r["planner"]: r for r in classification["fail_closed_reasons"]}
    assert "orca" in reasons
    assert reasons["orca"]["statuses"].get("fallback") == 20
    # Missing row_status defaults to a fail-closed 'unavailable' classification.
    assert "broken" in reasons
    assert reasons["broken"]["statuses"].get("unavailable") == 1


def test_ok_bundle_shape_and_bootstrap_wiring():
    """S20 native rows for >=2 planners produce a real bundle with CI fields."""
    rows = _make_rows(
        planners=["goal", "social_force", "orca"],
        seeds=range(111, 131),
        success_by_planner={"goal": 0.4, "social_force": 0.8, "orca": 0.6},
    )
    bundle = mod.build_bundle(_store(rows), git_head="abc")
    assert bundle["status"] == "ok"
    assert bundle["achieved_seed_tier"] == "s20"
    # Per-planner-by-seed summary shape: one variability row per planner.
    var_rows = bundle["per_planner_seed_variability"]
    assert len(var_rows) == 3
    sample = var_rows[0]
    assert "per_seed" in sample and "summary" in sample
    assert sample["seed_count"] == 20
    # Bootstrap CI fields come from the canonical _stats_for_vals wiring.
    success_summary = sample["summary"]["success"]
    for field in ("mean", "std", "ci_low", "ci_high", "ci_half_width"):
        assert field in success_summary
    # The confidence settings drove a real (non-zero) bootstrap sample count.
    assert bundle["confidence_settings"]["bootstrap_samples"] == 1000


def test_seed_resampling_rank_flip_present():
    """Rank-flip analysis runs via rank_metrics and reports the flip fraction."""
    rows = _make_rows(
        planners=["goal", "social_force", "orca"],
        seeds=range(111, 131),
        success_by_planner={"goal": 0.5, "social_force": 0.52, "orca": 0.48},
    )
    bundle = mod.build_bundle(_store(rows), git_head="abc")
    flip = bundle["seed_resampling_rank_flip"]["success"]
    assert flip["status"] == "ok"
    assert flip["method"] == "seed_resampling_kendall_tau"
    assert 0.0 <= flip["rank_flip_fraction"] <= 1.0
    assert "baseline_order" in flip
    # Aggregated conclusion key is present and boolean.
    assert isinstance(bundle["rank_conclusion_flips_under_resampling"], bool)


def test_snqi_ranking_stability_uses_canonical_bootstrap():
    """SNQI stability is the canonical bootstrap_stability payload when present."""
    rows = _make_rows(
        planners=["goal", "social_force"],
        seeds=range(111, 131),
        success_by_planner={"goal": 0.4, "social_force": 0.8},
    )
    bundle = mod.build_bundle(_store(rows), git_head="abc")
    snqi = bundle["snqi_ranking_stability"]
    assert snqi["status"] == "ok"
    assert snqi["method"] == "bootstrap_spearman"
    assert 0.0 <= snqi["stability"] <= 1.0


def test_reused_canonical_functions_recorded():
    """The bundle records the canonical functions it reuses (no reinvention)."""
    bundle = mod.build_bundle(_store([], kind="missing"), git_head="abc")
    reused = set(bundle["reused_canonical_functions"])
    assert "robot_sf.benchmark.seed_variance.build_seed_variability_rows" in reused
    assert "robot_sf.benchmark.snqi.bootstrap.bootstrap_stability" in reused
    assert "robot_sf.benchmark.rank_metrics.kendall_tau" in reused


def test_s30_tier_detection():
    """A 30-seed store is classified at the s30 tier."""
    rows = _make_rows(
        planners=["goal", "orca"],
        seeds=range(111, 141),
        success_by_planner={"goal": 0.4, "orca": 0.7},
    )
    bundle = mod.build_bundle(_store(rows), git_head="abc")
    assert bundle["status"] == "ok"
    assert bundle["achieved_seed_tier"] == "s30"
    assert bundle["min_seeds_per_planner"] == 30


@pytest.mark.parametrize("status", sorted(mod.FAIL_CLOSED_STATUSES))
def test_each_fail_closed_status_excluded(status):
    """Every fail-closed status is excluded from the valid set."""
    rows = _make_rows(planners=["goal", "orca"], seeds=range(111, 131), status=status)
    classification = mod.classify_rows(rows)
    assert classification["valid_rows"] == []
