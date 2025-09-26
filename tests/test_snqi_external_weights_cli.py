"""Tests for new CLI arguments handling external / initial weights with validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

VALID_WEIGHTS = {
    "w_success": 2.0,
    "w_time": 1.0,
    "w_collisions": 2.5,
    "w_near": 1.2,
    "w_comfort": 1.3,
    "w_force_exceed": 1.4,
    "w_jerk": 0.7,
}

INVALID_WEIGHTS_MISSING = {
    "w_success": 2.0,
    # missing w_time
    "w_collisions": 2.5,
    "w_near": 1.2,
    "w_comfort": 1.3,
    "w_force_exceed": 1.4,
    "w_jerk": 0.7,
}

INVALID_WEIGHTS_NON_NUMERIC = {
    "w_success": 2.0,
    "w_time": "abc",  # non-numeric
    "w_collisions": 2.5,
    "w_near": 1.2,
    "w_comfort": 1.3,
    "w_force_exceed": 1.4,
    "w_jerk": 0.7,
}


@pytest.fixture
def snqi_small_dataset(tmp_path: Path):
    episodes_path = tmp_path / "episodes.jsonl"
    baseline_path = tmp_path / "baseline.json"
    episodes = [
        {
            "scenario_id": "s1",
            "metrics": {
                "success": 1.0,
                "time_to_goal_norm": 0.5,
                "collisions": 0,
                "near_misses": 1,
                "comfort_exposure": 0.3,
                "force_exceed_events": 0,
                "jerk_mean": 0.15,
            },
        },
        {
            "scenario_id": "s2",
            "metrics": {
                "success": 0.0,
                "time_to_goal_norm": 0.9,
                "collisions": 1,
                "near_misses": 3,
                "comfort_exposure": 0.5,
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
    return episodes_path, baseline_path


def _write_weights(path: Path, weights: dict):
    path.write_text(json.dumps(weights), encoding="utf-8")


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "weights, expect_success",
    [
        (VALID_WEIGHTS, True),
        (INVALID_WEIGHTS_MISSING, False),
        (INVALID_WEIGHTS_NON_NUMERIC, False),
    ],
)
@pytest.mark.parametrize("script_kind", ["optimization", "recompute"])
def test_cli_external_initial_weights_validation(
    tmp_path: Path,
    snqi_small_dataset,
    weights,
    expect_success,
    script_kind,
):
    episodes_path, baseline_path = snqi_small_dataset
    weights_file = tmp_path / "weights.json"
    _write_weights(weights_file, weights)
    output_file = tmp_path / f"out_{script_kind}.json"

    if script_kind == "optimization":
        cmd = [
            sys.executable,
            "scripts/snqi_weight_optimization.py",
            "--episodes",
            str(episodes_path),
            "--baseline",
            str(baseline_path),
            "--output",
            str(output_file),
            "--method",
            "grid",
            "--grid-resolution",
            "2",  # make fast
            "--initial-weights-file",
            str(weights_file),
        ]
    else:
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
            str(output_file),
            "--external-weights-file",
            str(weights_file),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), check=False)
    if expect_success:
        assert proc.returncode == 0, proc.stderr or proc.stdout
        data = json.loads(output_file.read_text(encoding="utf-8"))
        if script_kind == "optimization":
            assert "initial_weights" in data
        else:
            assert "external_weights" in data
    else:
        assert proc.returncode != 0, "Expected failure for invalid weights file"
