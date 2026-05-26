"""Contract tests for issue #1556 synthetic AMV actuation stress slice config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a benchmark YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_issue_1556_config_declares_synthetic_differential_drive_slice() -> None:
    """The config should stay synthetic-only, non-paper-facing, and differential-drive-only."""
    payload = _load_yaml(CONFIG_PATH)

    assert payload["paper_facing"] is False
    assert payload["kinematics_matrix"] == ["differential_drive"]
    assert payload["seed_policy"]["mode"] == "seed-set"
    assert payload["seed_policy"]["seed_set"] == "eval"
    assert payload["scenario_candidates"] == [
        "classic_overtaking_medium",
        "classic_bottleneck_high",
        "classic_cross_trap_high",
        "francis2023_blind_corner",
        "francis2023_intersection_wait",
    ]

    profile = payload["synthetic_actuation_profile"]
    assert profile == {
        "name": "amv-actuation-stress-v0",
        "profile_version": "v0",
        "claim_scope": "synthetic-only",
        "max_linear_accel_m_s2": 2.0,
        "max_linear_decel_m_s2": 2.5,
        "max_yaw_rate_rad_s": 1.2,
        "max_angular_accel_rad_s2": 4.0,
        "latency_mode": "one-step-delay",
        "update_mode": "5hz-hold",
    }


def test_issue_1556_config_reuses_compact_primary_planner_set() -> None:
    """The slice should stay compact and avoid broad planner expansion."""
    payload = _load_yaml(CONFIG_PATH)

    assert payload["paper_interpretation_profile"] == "issue-1556-amv-actuation-diagnostic"
    assert payload["planners"] == [
        {
            "key": "goal",
            "algo": "goal",
            "planner_group": "core",
            "benchmark_profile": "baseline-safe",
        },
        {
            "key": "social_force",
            "algo": "social_force",
            "planner_group": "core",
            "benchmark_profile": "baseline-safe",
        },
        {
            "key": "orca",
            "algo": "orca",
            "planner_group": "core",
            "benchmark_profile": "baseline-safe",
            "socnav_missing_prereq_policy": "fallback",
        },
    ]
