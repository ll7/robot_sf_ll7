"""Tests for holonomic adapter error analysis tooling."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from scripts.tools.analyze_holonomic_adapter_error import analyze, analyze_command, plt

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_analyze_command_matches_exact_projection_math() -> None:
    """The midpoint projection should match the closed-form formula used in analysis."""
    sample = analyze_command(v=1.2, omega=0.8, heading=0.3, dt=0.1)
    phase = 0.8 * 0.1 * 0.5
    scale = math.sin(phase) / phase
    mid_heading = 0.3 + phase
    expected_vx = 1.2 * scale * math.cos(mid_heading)
    expected_vy = 1.2 * scale * math.sin(mid_heading)
    assert sample.exact_vx == pytest.approx(expected_vx)
    assert sample.exact_vy == pytest.approx(expected_vy)
    assert sample.approx_vx == pytest.approx(1.2 * math.cos(mid_heading))
    assert sample.approx_vy == pytest.approx(1.2 * math.sin(mid_heading))
    assert sample.error_norm_d > 0.0
    assert sample.relative_speed_error > 0.0


def test_analyze_writes_campaign_and_grid_artifacts(tmp_path: Path) -> None:
    """Analyzer should write a report and summarize adapter projection versus baseline."""
    candidate_root = tmp_path / "candidate"
    baseline_root = tmp_path / "baseline"
    candidate_summary = {
        "campaign": {"campaign_id": "candidate_campaign"},
        "planner_rows": [
            {
                "planner_key": "ppo",
                "success_mean": "0.2400",
                "collision_mean": "0.1000",
                "snqi_mean": "-1.3000",
            },
            {
                "planner_key": "orca",
                "success_mean": "0.0100",
                "collision_mean": "0.7000",
                "snqi_mean": "-2.8000",
            },
        ],
        "runs": [
            {
                "planner": {"key": "ppo", "algo": "ppo"},
                "summary": {
                    "algorithm_metadata_contract": {
                        "planner_kinematics": {
                            "execution_mode": "native",
                            "adapter_name": "ppo_action_to_unicycle",
                        },
                        "adapter_impact": {
                            "status": "complete",
                            "adapter_fraction": 0.0,
                        },
                        "kinematics_feasibility": {
                            "projection_rate": 0.0,
                            "mean_abs_delta_linear": 0.0,
                            "mean_abs_delta_angular": 0.0,
                            "max_abs_delta_linear": 0.0,
                            "max_abs_delta_angular": 0.0,
                        },
                    }
                },
            },
            {
                "planner": {"key": "orca", "algo": "orca"},
                "summary": {
                    "algorithm_metadata_contract": {
                        "planner_kinematics": {
                            "execution_mode": "adapter",
                            "adapter_name": "ORCAPlannerAdapter",
                        },
                        "adapter_impact": {
                            "status": "disabled",
                            "adapter_fraction": 0.0,
                        },
                        "kinematics_feasibility": {
                            "projection_rate": 0.125,
                            "mean_abs_delta_linear": 0.05,
                            "mean_abs_delta_angular": 0.03,
                            "max_abs_delta_linear": 0.20,
                            "max_abs_delta_angular": 0.40,
                        },
                    }
                },
            },
        ],
    }
    baseline_summary = {
        "campaign": {"campaign_id": "baseline_campaign"},
        "planner_rows": [
            {
                "planner_key": "ppo",
                "success_mean": "0.3000",
                "collision_mean": "0.1500",
                "snqi_mean": "-1.1000",
            },
            {
                "planner_key": "orca",
                "success_mean": "0.2000",
                "collision_mean": "0.0400",
                "snqi_mean": "-1.0000",
            },
        ],
        "runs": [],
    }
    _write_json(candidate_root / "reports" / "campaign_summary.json", candidate_summary)
    _write_json(baseline_root / "reports" / "campaign_summary.json", baseline_summary)

    out_dir = tmp_path / "analysis"
    payload = analyze(
        v=1.0,
        omega=2.0,
        heading=0.25,
        dt=0.1,
        theta_min=-math.pi,
        theta_max=math.pi,
        theta_samples=33,
        omega_min=-4.0,
        omega_max=4.0,
        omega_samples=41,
        campaign_root=candidate_root,
        baseline_campaign_root=baseline_root,
        output_dir=out_dir,
    )

    assert (out_dir / "analysis.json").exists()
    assert (out_dir / "analysis.md").exists()
    assert (out_dir / "grid_samples.csv").exists()
    if plt is not None:
        assert (out_dir / "error_heatmap.png").exists()
    assert payload["grid_summary"]["max_relative_speed_error"] > 0.0
    assert payload["campaign"]["candidate_campaign_id"] == "candidate_campaign"
    assert payload["campaign"]["baseline_campaign_id"] == "baseline_campaign"
    interpretation = "\n".join(payload["campaign"]["interpretation"])
    assert "nonzero adapter projection" in interpretation
    assert "regressions" in interpretation

    analysis_md = (out_dir / "analysis.md").read_text(encoding="utf-8")
    assert "Holonomic Adapter Error Analysis" in analysis_md
    assert "Baseline Comparison" in analysis_md
    assert "orca" in analysis_md

    oracle = payload["campaign"]["comparison"]
    assert len(oracle) == 2
    orca_row = next(item for item in oracle if item["planner_key"] == "orca")
    assert orca_row["projection_rate"] == pytest.approx(0.125)
    assert orca_row["metrics"]["success_mean"]["delta"] == pytest.approx(-0.19)
