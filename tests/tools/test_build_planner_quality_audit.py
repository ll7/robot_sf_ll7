"""Tests for planner quality audit generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

from scripts.tools.build_planner_quality_audit import _build_markdown, build_audit

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_audit_combines_hard_and_sanity_metrics(tmp_path: Path) -> None:
    """Audit should combine hard-matrix metrics, sanity checks, and raw failure modes."""
    hard_root = tmp_path / "hard"
    sanity_root = tmp_path / "sanity"
    parity = tmp_path / "planner_quality_audit.yaml"
    _write_json(
        hard_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "hard_campaign"},
            "planner_rows": [
                {
                    "planner_key": "ppo",
                    "success_mean": "0.2500",
                    "collisions_mean": "0.1000",
                    "snqi_mean": "-0.3000",
                    "runtime_sec": "20.0",
                    "episodes_per_second": "2.0",
                }
            ],
            "runs": [
                {
                    "planner": {"key": "ppo"},
                    "episodes_path": str(hard_root / "runs" / "ppo" / "episodes.jsonl"),
                }
            ],
        },
    )
    _write_jsonl(
        hard_root / "runs" / "ppo" / "episodes.jsonl",
        [
            {"termination_reason": "success"},
            {"termination_reason": "max_steps"},
            {"termination_reason": "collision"},
            {"termination_reason": "max_steps"},
        ],
    )
    _write_json(
        sanity_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "sanity_campaign"},
            "planner_rows": [
                {
                    "planner_key": "ppo",
                    "success_mean": "1.0000",
                    "collisions_mean": "0.0000",
                    "runtime_sec": "5.0",
                    "status": "ok",
                }
            ],
        },
    )
    parity.write_text(
        """
version: planner-quality-audit-v1
reproduction_priority: []
planners:
  ppo:
    classification: credible benchmark baseline
    headline_recommendation: keep
    paper_reference_family: End-to-end learned navigation baseline
    paper_evaluates: source harness
    current_implementation: local ppo
    missing_for_fair_comparison: better performance needed
    interpret_result_as: implementation-level local baseline evidence
""",
        encoding="utf-8",
    )

    payload = build_audit(hard_root, sanity_root, parity)
    row = payload["planner_audit_rows"][0]
    assert row["planner_key"] == "ppo"
    assert row["hard_matrix"]["max_steps_rate"] == 0.5
    assert row["hard_matrix"]["primary_failure_mode"] == "max_steps"
    assert row["sanity_matrix"]["success_mean"] == 1.0
    assert payload["headline_suite"]["headline_suite"] == ["ppo"]


def test_build_audit_resolves_relative_episode_paths_from_campaign_root(tmp_path: Path) -> None:
    """Relative episodes paths should resolve against the campaign root, not the process cwd."""
    hard_root = tmp_path / "hard"
    sanity_root = tmp_path / "sanity"
    parity = tmp_path / "planner_quality_audit.yaml"
    _write_json(
        hard_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "hard_campaign"},
            "planner_rows": [{"planner_key": "ppo", "success_mean": "0.0"}],
            "runs": [{"planner": {"key": "ppo"}, "episodes_path": "runs/ppo/episodes.jsonl"}],
        },
    )
    _write_jsonl(
        hard_root / "runs" / "ppo" / "episodes.jsonl", [{"termination_reason": "max_steps"}]
    )
    _write_json(
        sanity_root / "reports" / "campaign_summary.json",
        {"campaign": {"campaign_id": "sanity_campaign"}, "planner_rows": []},
    )
    parity.write_text(
        """
version: planner-quality-audit-v1
reproduction_priority: []
planners: {}
""",
        encoding="utf-8",
    )

    payload = build_audit(hard_root, sanity_root, parity)
    row = payload["planner_audit_rows"][0]
    assert row["hard_matrix"]["termination_reason_counts"] == {"max_steps": 1}


def test_build_audit_falls_back_to_repo_relative_episode_paths(tmp_path: Path) -> None:
    """Repo-root-relative episode paths should still resolve when not nested under campaign_root."""
    repo_root = tmp_path / "repo"
    hard_root = repo_root / "hard"
    sanity_root = repo_root / "sanity"
    parity = tmp_path / "planner_quality_audit.yaml"
    _write_json(
        hard_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "hard_campaign"},
            "planner_rows": [{"planner_key": "ppo", "success_mean": "0.0"}],
            "runs": [
                {
                    "planner": {"key": "ppo"},
                    "episodes_path": "output/benchmarks/example/runs/ppo/episodes.jsonl",
                }
            ],
        },
    )
    _write_jsonl(
        repo_root / "output" / "benchmarks" / "example" / "runs" / "ppo" / "episodes.jsonl",
        [{"termination_reason": "collision"}],
    )
    _write_json(
        sanity_root / "reports" / "campaign_summary.json",
        {"campaign": {"campaign_id": "sanity_campaign"}, "planner_rows": []},
    )
    parity.write_text(
        """
version: planner-quality-audit-v1
reproduction_priority: []
planners: {}
""",
        encoding="utf-8",
    )

    with patch(
        "scripts.tools.build_planner_quality_audit._repository_root", return_value=repo_root
    ):
        payload = build_audit(hard_root, sanity_root, parity)
    row = payload["planner_audit_rows"][0]
    assert row["hard_matrix"]["termination_reason_counts"] == {"collision": 1}
    assert row["hard_matrix"]["primary_failure_mode"] == "collision"


def test_build_markdown_renders_decision_table_and_priority() -> None:
    """Markdown output should include the decision table and reproduction priority sections."""
    payload = {
        "hard_matrix_campaign_id": "hard",
        "sanity_campaign_id": "sanity",
        "policy_version": "planner-quality-audit-v1",
        "planner_audit_rows": [
            {
                "planner_key": "orca",
                "classification": "credible benchmark baseline",
                "headline_recommendation": "keep",
                "paper_reference_family": "Reciprocal-velocity-obstacle baseline",
                "paper_evaluates": "source orca",
                "current_implementation": "local orca",
                "missing_for_fair_comparison": "simulator differs",
                "interpret_result_as": "family-level local baseline evidence",
                "hard_matrix": {
                    "success_mean": 0.23,
                    "collisions_mean": 0.04,
                    "max_steps_rate": 0.70,
                    "snqi_mean": -0.23,
                    "runtime_sec": 98.2,
                    "primary_failure_mode": "max_steps",
                },
                "sanity_matrix": {"success_mean": 1.0},
            }
        ],
        "headline_suite": {
            "headline_suite": ["orca"],
            "control_only": [],
            "non_headline": [],
        },
        "external_candidates": [
            {
                "label": "CrowdNav / SoNIC family",
                "closest_local_proxies": ["ppo", "sacadrl"],
                "observation_contract_gap": "different observation stack",
                "action_contract_gap": "different action stack",
                "scenario_assumption_gap": "different scenarios",
                "evaluation_harness_gap": "different harness",
                "interpretation": "not family-level evidence",
            }
        ],
        "reproduction_priority": [
            {
                "label": "CrowdNav / SoNIC family",
                "rationale": "highest value",
                "exact_policy_or_config_source": [
                    "output/repos/SoNIC-Social-Nav",
                    "output/repos/CrowdNav",
                ],
                "expected_observation_action_contract": "obs/action",
                "expected_scenario_and_eval_protocol": "source benchmark",
                "wrapper_strategy": "adapter",
                "acceptance_threshold": "match source behavior",
            }
        ],
    }
    markdown = _build_markdown(payload)
    assert "# Planner Quality Audit" in markdown
    assert "| orca | credible benchmark baseline | keep |" in markdown
    assert "## External Candidate Parity Gaps" in markdown
    assert "CrowdNav / SoNIC family" in markdown


def test_build_markdown_keeps_priority_header_without_external_candidates() -> None:
    """Priority section header should render even when parity-gap rows are absent."""
    payload = {
        "hard_matrix_campaign_id": "hard",
        "sanity_campaign_id": "sanity",
        "policy_version": "planner-quality-audit-v1",
        "planner_audit_rows": [],
        "headline_suite": {
            "headline_suite": [],
            "control_only": [],
            "non_headline": [],
        },
        "external_candidates": [],
        "reproduction_priority": [
            {
                "label": "CrowdNav / SoNIC family",
                "rationale": "highest value",
                "exact_policy_or_config_source": ["output/repos/SoNIC-Social-Nav"],
                "expected_observation_action_contract": "obs/action",
                "expected_scenario_and_eval_protocol": "source benchmark",
                "wrapper_strategy": "adapter",
                "acceptance_threshold": "match source behavior",
            }
        ],
    }

    markdown = _build_markdown(payload)
    assert "## External Reproduction Priority" in markdown
    assert "### CrowdNav / SoNIC family" in markdown
