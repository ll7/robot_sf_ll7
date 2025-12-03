"""CLI test for SNQI weight recomputation script with normalization comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(30)
def test_snqi_weight_recompute_cli_with_normalization(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    episodes_path = tmp_path / "episodes.jsonl"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "recompute.json"

    # Minimal two-episode dataset
    episodes = [
        {
            "scenario_id": "s1",
            "metrics": {
                "success": 1.0,
                "time_normalized": 0.5,
                "collisions": 0,
                "near_misses": 1,
                "comfort_penalty": 0.3,
                "force_exceed_events": 0,
                "jerk_mean": 0.15,
            },
        },
        {
            "scenario_id": "s2",
            "metrics": {
                "success": 0.0,
                "time_normalized": 0.9,
                "collisions": 1,
                "near_misses": 3,
                "comfort_penalty": 0.5,
                "force_exceed_events": 1,
                "jerk_mean": 0.35,
            },
        },
    ]
    with episodes_path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    baseline = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 1.0, "p95": 3.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.1, "p95": 0.6},
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/recompute_snqi_weights.py",
        "--episodes",
        str(episodes_path),
        "--baseline",
        str(baseline_path),
        "--strategy",
        "balanced",
        "--output",
        str(output_path),
        "--compare-normalization",
        "--seed",
        "42",
        "--validate",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert output_path.exists(), "Expected recompute output JSON not created"

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "_metadata" in data and "summary" in data
    meta = data["_metadata"]
    summary = data["summary"]

    for field in ["schema_version", "generated_at", "runtime_seconds", "start_time", "end_time"]:
        assert field in meta
    assert isinstance(meta["runtime_seconds"], int | float) and meta["runtime_seconds"] >= 0

    # Normalization comparison block
    norm = data.get("normalization_comparison")
    assert norm is not None and len(norm) >= 1
    # Expect base strategy always present
    assert "median_p95" in norm
    # If alternative strategies produced, ensure correlations bounded
    for name, entry in norm.items():
        if name != "median_p95":
            corr = entry.get("correlation_with_base")
            if corr is not None:
                assert -1.0 <= corr <= 1.0

    # Summary integrity
    for field in ["method", "weights", "runtime_seconds"]:
        assert field in summary
    assert summary["method"] in (
        "balanced",
        "default",
        "safety_focused",
        "efficiency_focused",
        "pareto",
    )
