"""Tests for paper-matrix seed-schedule comparison helper."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from scripts.tools.compare_seed_schedule_campaigns import (
    _build_markdown,
    compare_seed_schedules,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _campaign_summary(campaign_id: str, rows: list[dict]) -> dict:
    return {
        "campaign": {
            "campaign_id": campaign_id,
            "name": campaign_id,
            "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
            "scenario_matrix_hash": "matrix-hash",
            "paper_profile_version": "paper-matrix-v1",
            "paper_interpretation_profile": "baseline-ready-core",
            "total_episodes": 10,
            "total_runs": 2,
            "successful_runs": 2,
            "benchmark_success": True,
            "runtime_sec": 12.0,
            "git_hash": "abc123",
        },
        "planner_rows": rows,
    }


def _matrix_summary(seed_set: str, seeds: list[int]) -> dict:
    return {
        "rows": [
            {
                "resolved_seeds": seeds,
                "repeats": len(seeds),
                "seed_policy.mode": "seed-set",
                "seed_policy.seed_set": seed_set,
            }
        ]
    }


def _seed_row(
    scenario_id: str,
    planner_key: str,
    *,
    seed_count: int,
    success: float,
    snqi: float,
    snqi_ci_half_width: float,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "planner_key": planner_key,
        "algo": planner_key,
        "planner_group": "core",
        "kinematics": "differential_drive",
        "benchmark_profile": "baseline-safe",
        "seed_count": seed_count,
        "summary": {
            "success": {
                "mean": success,
                "ci_low": success - 0.10,
                "ci_high": success + 0.10,
                "ci_half_width": 0.10,
            },
            "snqi": {
                "mean": snqi,
                "ci_low": snqi - snqi_ci_half_width,
                "ci_high": snqi + snqi_ci_half_width,
                "ci_half_width": snqi_ci_half_width,
            },
        },
    }


def _seed_payload(rows: list[dict]) -> dict:
    return {
        "schema_version": "benchmark-seed-variability-by-scenario.v1",
        "metrics": ["success", "snqi"],
        "rows": rows,
    }


def _write_campaign(
    root: Path,
    *,
    campaign_id: str,
    seed_set: str,
    seeds: list[int],
    planner_rows: list[dict],
    seed_rows: list[dict],
) -> None:
    _write_json(
        root / "reports" / "campaign_summary.json", _campaign_summary(campaign_id, planner_rows)
    )
    _write_json(root / "reports" / "matrix_summary.json", _matrix_summary(seed_set, seeds))
    _write_json(root / "reports" / "seed_variability_by_scenario.json", _seed_payload(seed_rows))


def test_compare_seed_schedules_reports_stable_extension(tmp_path: Path) -> None:
    """Narrower CIs with unchanged ranking and winners should produce a stable verdict."""
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_campaign(
        base,
        campaign_id="base-s3",
        seed_set="eval",
        seeds=[111, 112, 113],
        planner_rows=[
            {"planner_key": "goal", "kinematics": "differential_drive", "snqi_mean": "0.4"},
            {"planner_key": "orca", "kinematics": "differential_drive", "snqi_mean": "0.2"},
        ],
        seed_rows=[
            _seed_row("s1", "goal", seed_count=3, success=0.8, snqi=0.4, snqi_ci_half_width=0.20),
            _seed_row("s1", "orca", seed_count=3, success=0.7, snqi=0.2, snqi_ci_half_width=0.20),
            _seed_row("s2", "goal", seed_count=3, success=0.9, snqi=0.5, snqi_ci_half_width=0.20),
            _seed_row("s2", "orca", seed_count=3, success=0.6, snqi=0.1, snqi_ci_half_width=0.20),
        ],
    )
    _write_campaign(
        candidate,
        campaign_id="candidate-s5",
        seed_set="paper_eval_s5",
        seeds=[111, 112, 113, 114, 115],
        planner_rows=[
            {"planner_key": "goal", "kinematics": "differential_drive", "snqi_mean": "0.42"},
            {"planner_key": "orca", "kinematics": "differential_drive", "snqi_mean": "0.22"},
        ],
        seed_rows=[
            _seed_row("s1", "goal", seed_count=5, success=0.8, snqi=0.41, snqi_ci_half_width=0.10),
            _seed_row("s1", "orca", seed_count=5, success=0.7, snqi=0.2, snqi_ci_half_width=0.10),
            _seed_row("s2", "goal", seed_count=5, success=0.9, snqi=0.51, snqi_ci_half_width=0.10),
            _seed_row("s2", "orca", seed_count=5, success=0.6, snqi=0.12, snqi_ci_half_width=0.10),
        ],
    )

    payload = compare_seed_schedules(base, candidate)

    assert payload["candidate_campaign"]["resolved_seeds"] == [111, 112, 113, 114, 115]
    assert payload["interval_width"]["aggregate"]["snqi"]["target_met"] is True
    assert payload["ranking_stability"]["status"] == "stable"
    assert payload["scenario_winner_stability"]["status"] == "stable"
    assert payload["interpretation"]["status"] == "stable"
    markdown = _build_markdown(payload)
    assert "Seed Schedule Comparison" in markdown
    assert "paper_eval_s5" in json.dumps(payload)


def test_compare_seed_schedules_flags_ranking_and_winner_drift(tmp_path: Path) -> None:
    """Ranking and scenario-winner changes should be surfaced as review triggers."""
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_campaign(
        base,
        campaign_id="base-s3",
        seed_set="eval",
        seeds=[111, 112, 113],
        planner_rows=[
            {"planner_key": "goal", "kinematics": "differential_drive", "snqi_mean": "0.4"},
            {"planner_key": "orca", "kinematics": "differential_drive", "snqi_mean": "0.2"},
        ],
        seed_rows=[
            _seed_row("s1", "goal", seed_count=3, success=0.8, snqi=0.4, snqi_ci_half_width=0.2),
            _seed_row("s1", "orca", seed_count=3, success=0.7, snqi=0.2, snqi_ci_half_width=0.2),
        ],
    )
    _write_campaign(
        candidate,
        campaign_id="candidate-s5",
        seed_set="paper_eval_s5",
        seeds=[111, 112, 113, 114, 115],
        planner_rows=[
            {"planner_key": "goal", "kinematics": "differential_drive", "snqi_mean": "0.1"},
            {"planner_key": "orca", "kinematics": "differential_drive", "snqi_mean": "0.5"},
        ],
        seed_rows=[
            _seed_row("s1", "goal", seed_count=5, success=0.8, snqi=0.1, snqi_ci_half_width=0.1),
            _seed_row("s1", "orca", seed_count=5, success=0.7, snqi=0.5, snqi_ci_half_width=0.1),
        ],
    )

    payload = compare_seed_schedules(base, candidate)

    assert payload["ranking_stability"]["status"] == "unstable"
    assert payload["scenario_winner_stability"]["changed_count"] == 1
    assert payload["interpretation"]["headline_interpretation_changes"] is True


def test_main_writes_comparison_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should write JSON and Markdown artifacts under the current working directory."""
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_campaign(
        base,
        campaign_id="base-s3",
        seed_set="eval",
        seeds=[111, 112, 113],
        planner_rows=[
            {"planner_key": "goal", "kinematics": "differential_drive", "snqi_mean": "0.4"},
            {"planner_key": "orca", "kinematics": "differential_drive", "snqi_mean": "0.2"},
        ],
        seed_rows=[
            _seed_row("s1", "goal", seed_count=3, success=0.8, snqi=0.4, snqi_ci_half_width=0.2),
            _seed_row("s1", "orca", seed_count=3, success=0.7, snqi=0.2, snqi_ci_half_width=0.2),
        ],
    )
    _write_campaign(
        candidate,
        campaign_id="candidate-s5",
        seed_set="paper_eval_s5",
        seeds=[111, 112, 113, 114, 115],
        planner_rows=[
            {"planner_key": "goal", "kinematics": "differential_drive", "snqi_mean": "0.4"},
            {"planner_key": "orca", "kinematics": "differential_drive", "snqi_mean": "0.2"},
        ],
        seed_rows=[
            _seed_row("s1", "goal", seed_count=5, success=0.8, snqi=0.4, snqi_ci_half_width=0.1),
            _seed_row("s1", "orca", seed_count=5, success=0.7, snqi=0.2, snqi_ci_half_width=0.1),
        ],
    )
    out_json = tmp_path / "reports" / "comparison.json"
    out_md = tmp_path / "reports" / "comparison.md"
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_seed_schedule_campaigns.py",
            "--base-campaign-root",
            str(base),
            "--candidate-campaign-root",
            str(candidate),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
    )

    assert main() == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["schema_version"] == (
        "benchmark-seed-schedule-comparison.v1"
    )
    assert "Seed Schedule Comparison" in out_md.read_text(encoding="utf-8")
