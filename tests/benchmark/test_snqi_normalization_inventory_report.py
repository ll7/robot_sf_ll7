"""Tests for the standalone SNQI normalization inventory report."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from robot_sf.benchmark.snqi.compute import compute_snqi

REPO_ROOT = pathlib.Path(__file__).parents[2]


_BASELINE_STATS = {
    "collisions": {"med": 1.0, "p95": 5.0},
    "near_misses": {"med": 2.0, "p95": 8.0},
    "force_exceed_events": {"med": 0.0, "p95": 4.0},
    "jerk_mean": {"med": 0.1, "p95": 1.0},
}

_WEIGHTS = {
    "w_success": 1.0,
    "w_time": 0.8,
    "w_collisions": 2.0,
    "w_near": 1.0,
    "w_comfort": 0.5,
    "w_force_exceed": 1.5,
    "w_jerk": 0.3,
}

_METRICS = {
    "success": 1.0,
    "time_to_goal_norm": 1.25,
    "collisions": 3.0,
    "near_misses": 4.0,
    "comfort_exposure": 7.0,
    "force_exceed_events": 2.0,
    "jerk_mean": 0.7,
}


def test_standalone_report_fails_closed_without_changing_snqi(tmp_path) -> None:
    """The issue #3699 report surfaces mixed inputs but never mutates scoring."""

    baseline_path = tmp_path / "baseline_stats.json"
    json_out = tmp_path / "inventory.json"
    baseline_path.write_text(json.dumps(_BASELINE_STATS), encoding="utf-8")

    before = compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/benchmark/snqi_normalization_inventory_report.py"),
            "--baseline-stats",
            str(baseline_path),
            "--json-out",
            str(json_out),
            "--fail-on-mixed-scale",
            "--fail-on-missing-baseline",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    after = compute_snqi(_METRICS, _WEIGHTS, _BASELINE_STATS)

    assert result.returncode == 1
    assert "FAIL: SNQI penalty terms mix raw" in result.stderr
    assert after == before

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    inventory = payload["inventory"]
    assert inventory["mixed_scale"] is True
    assert inventory["is_consistent"] is False
    assert inventory["raw_penalty_terms"] == ["time", "comfort"]
    assert inventory["normalized_penalty_terms"] == [
        "collisions",
        "near",
        "force_exceed",
        "jerk",
    ]
    assert inventory["missing_baseline_coverage"] == []

    status_by_term = {term["term"]: term["normalization_status"] for term in inventory["terms"]}
    assert status_by_term["time"] == "raw_unbounded"
    assert status_by_term["comfort"] == "raw_unbounded"
    assert status_by_term["collisions"] == "baseline_normalized_bounded"
