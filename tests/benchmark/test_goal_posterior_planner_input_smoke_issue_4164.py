"""Tests for issue #4164 goal-posterior planner-input smoke report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT_PATH = Path("scripts/benchmark/run_goal_posterior_planner_input_smoke_issue_4164.py")
_SPEC = importlib.util.spec_from_file_location("issue_4164_goal_posterior_smoke", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_report = _MODULE.build_report
main = _MODULE.main


def test_goal_posterior_smoke_report_has_paired_enabled_rows() -> None:
    """Smoke report includes paired disabled/enabled rows."""

    report = build_report(Path("configs/benchmarks/issue_4164_goal_intention_smoke.yaml"))

    assert report["schema_version"] == "issue_4164_goal_posterior_planner_input_smoke.v1"
    assert "no full benchmark campaign" in report["claim_boundary"]
    assert len(report["scenarios"]) == 2
    for scenario in report["scenarios"]:
        assert scenario["without_goal_posterior"]["channel_present"] is False
        assert scenario["with_goal_posterior"]["channel_present"] is True
        assert scenario["with_goal_posterior"]["top_goal_ids"]
        assert scenario["with_goal_posterior"]["blockers"] == {}


def test_goal_posterior_smoke_main_writes_json(tmp_path: Path) -> None:
    """Smoke CLI writes compact JSON evidence."""

    output_path = tmp_path / "smoke.json"

    exit_code = main(
        [
            "--config",
            "configs/benchmarks/issue_4164_goal_intention_smoke.yaml",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["posterior_config"]["config_hash"]
    assert len(payload["scenarios"]) == 2
