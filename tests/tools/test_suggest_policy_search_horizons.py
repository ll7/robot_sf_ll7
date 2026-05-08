"""Tests for policy-search horizon recommendation generation."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import yaml

from scripts.tools import suggest_policy_search_horizons

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write policy-search episode records as JSONL."""
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_horizon_recommendations_mark_no_success_scenarios_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Scenario horizons should derive from p95 success steps and flag no-success blockers."""
    jsonl_path = tmp_path / "episodes.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "scenario_id": "scenario_a",
                "steps": 10,
                "metrics": {"success": True},
                "outcome": {"timeout_event": False},
                "termination_reason": "success",
            },
            {
                "scenario_id": "scenario_a",
                "steps": 30,
                "metrics": {"success": True},
                "outcome": {"timeout_event": False},
                "termination_reason": "success",
            },
            {
                "scenario_id": "scenario_a",
                "steps": 50,
                "metrics": {"success": False},
                "outcome": {"timeout_event": True},
                "termination_reason": "max_steps",
            },
            {
                "scenario_id": "scenario_b",
                "steps": 50,
                "metrics": {"success": False},
                "outcome": {"timeout_event": True},
                "termination_reason": "max_steps",
            },
        ],
    )
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "candidate": "candidate_a",
                "stage": "full_matrix_h500",
                "jsonl_path": str(jsonl_path),
                "summary": {
                    "success_rate": 0.90,
                    "collision_rate": 0.01,
                    "near_miss_rate": 0.20,
                },
            }
        ),
        encoding="utf-8",
    )
    output_yaml = tmp_path / "horizons.yaml"
    output_md = tmp_path / "horizons.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "suggest_policy_search_horizons.py",
            str(summary_json),
            "--buffer-steps",
            "5",
            "--floor-steps",
            "20",
            "--cap-steps",
            "50",
            "--near-cap-margin",
            "5",
            "--output-yaml",
            str(output_yaml),
            "--output-md",
            str(output_md),
        ],
    )

    assert suggest_policy_search_horizons.main() == 0

    payload = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    scenario_a = payload["scenarios"]["scenario_a"]
    scenario_b = payload["scenarios"]["scenario_b"]
    assert payload["selection"]["p95_multiplier"] == 1.2
    assert scenario_a["recommended_horizon_steps"] == 40
    assert scenario_a["status"] == "recommended"
    assert scenario_a["timeout_failure_count"] == 1
    assert scenario_b["recommended_horizon_steps"] == 50
    assert scenario_b["status"] == "planner_blocked"
    assert output_md.exists()
