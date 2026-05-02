"""Tests for the policy-search failure report CLI."""

from __future__ import annotations

import json

from scripts.tools import build_policy_search_failure_report


def test_failure_report_separates_evidence_backed_scenario_exclusions(
    tmp_path,
    monkeypatch,
) -> None:
    """Reports should show raw metrics and a separate evidence-backed exclusion table."""
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "scenario_id": "classic_cross_trap_high",
                        "seed": 112,
                        "termination_reason": "collision",
                        "scenario_params": {"humans": [{"id": "p1"}]},
                        "scenario_exclusion": {
                            "status": "invalid",
                            "reason": "zero_action_first_step_collision",
                            "evidence": [
                                "zero-action first step terminates before policy control matters"
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "scenario_id": "classic_cross_trap_high",
                        "seed": 113,
                        "termination_reason": "success",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_policy_search_failure_report.py",
            "--jsonl",
            str(jsonl_path),
            "--output",
            str(output_dir),
        ],
    )

    assert build_policy_search_failure_report.main() == 0

    payload = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "failure_report.md").read_text(encoding="utf-8")
    assert payload["summary"]["success_rate"] == 0.5
    assert payload["summary"]["evidence_adjusted"]["success_rate"] == 1.0
    assert payload["summary"]["scenario_exclusions"]["count"] == 1
    assert payload["scenario_failure_counts"] == []
    assert "zero_action_first_step_collision" in markdown
    assert "| raw_success_rate | 0.5000 |" in markdown
    assert "| evidence_adjusted_success_rate | 1.0000 |" in markdown
