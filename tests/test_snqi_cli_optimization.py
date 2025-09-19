"""CLI test for SNQI weight optimization script.

Creates a tiny synthetic episodes JSONL and baseline stats JSON, invokes the
optimization script via subprocess, and validates core fields of the produced
output JSON (metadata + summary + recommended weights). Uses minimal grid
resolution and low iteration counts for speed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(30)
def test_snqi_weight_optimization_cli(tmp_path: Path):
    # Prepare synthetic episodes (two episodes w/ minimal metrics)
    episodes_path = tmp_path / "episodes.jsonl"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "weights.json"

    episodes = [
        {
            "scenario_id": "s1",
            "metrics": {
                "success": 1.0,
                "time_normalized": 0.6,
                "collisions": 0,
                "near_misses": 1,
                "comfort_penalty": 0.2,
                "force_exceed_events": 0,
                "jerk_mean": 0.1,
            },
        },
        {
            "scenario_id": "s2",
            "metrics": {
                "success": 0.0,
                "time_normalized": 0.8,
                "collisions": 1,
                "near_misses": 2,
                "comfort_penalty": 0.4,
                "force_exceed_events": 1,
                "jerk_mean": 0.3,
            },
        },
    ]
    with episodes_path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    baseline = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.5, "p95": 2.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.05, "p95": 0.5},
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/snqi_weight_optimization.py",
        "--episodes",
        str(episodes_path),
        "--baseline",
        str(baseline_path),
        "--output",
        str(output_path),
        "--method",
        "grid",  # simple/fast
        "--grid-resolution",
        "2",  # keep combinations tiny
        "--seed",
        "123",
        "--validate",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert output_path.exists(), "Expected output JSON file not created"

    data = json.loads(output_path.read_text(encoding="utf-8"))

    # Core structural assertions
    assert "_metadata" in data
    assert "summary" in data
    assert "recommended" in data
    meta = data["_metadata"]
    summary = data["summary"]
    reco = data["recommended"]

    for field in ["schema_version", "generated_at", "runtime_seconds", "start_time", "end_time"]:
        assert field in meta
    assert isinstance(meta["runtime_seconds"], (int, float)) and meta["runtime_seconds"] >= 0

    for field in ["method", "weights", "runtime_seconds"]:
        assert field in summary
    assert summary["method"] in ("grid_search", "differential_evolution")

    weights = reco.get("weights")
    assert isinstance(weights, dict) and len(weights) > 0
    # Weight bounds sanity (0.05..5 as broad safety net)
    for v in weights.values():
        assert 0.05 <= float(v) <= 5.0
