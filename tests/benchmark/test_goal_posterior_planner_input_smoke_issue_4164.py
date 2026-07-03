"""Tests for issue #4164 goal-posterior planner-consumption smoke report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = (
    _REPO_ROOT / "scripts" / "benchmark" / "run_goal_posterior_planner_input_smoke_issue_4164.py"
)
_CONFIG_PATH = _REPO_ROOT / "configs" / "benchmarks" / "issue_4164_goal_intention_smoke.yaml"
_SPEC = importlib.util.spec_from_file_location("issue_4164_goal_posterior_smoke", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_report = _MODULE.build_report
main = _MODULE.main


def test_goal_posterior_smoke_report_has_planner_consumption_rows() -> None:
    """Smoke report includes paired planner-consumption effects."""
    report = build_report(_CONFIG_PATH)

    assert report["schema_version"] == "issue_4164_goal_posterior_planner_consumption_smoke.v1"
    assert "no full benchmark campaign" in report["claim_boundary"]
    assert len(report["scenarios"]) == 2
    for scenario in report["scenarios"]:
        without = scenario["without_goal_posterior"]
        with_posterior = scenario["with_goal_posterior"]

        assert scenario["planner_path"] == "hybrid_rule_local_planner.goal_posterior_avoidance"
        assert without["planner_consumed_channel"] is False
        assert with_posterior["planner_consumed_channel"] is True
        assert with_posterior["posterior_active"] is True
        assert scenario["command_source_changed"] or scenario["trajectory_changed"]
        assert isinstance(scenario["route_progress_delta"], float)
        assert scenario["fallback_or_degraded_exclusions"]["with"]["fallback_or_degraded"] is False
        assert with_posterior["selected_sources"]
        assert any(
            source.startswith("goal_posterior_yield_")
            for source in with_posterior["selected_sources"]
        )


def test_goal_posterior_smoke_main_writes_json(tmp_path: Path) -> None:
    """Smoke CLI writes compact JSON evidence."""
    output_path = tmp_path / "smoke.json"
    exit_code = main(
        [
            "--config",
            str(_CONFIG_PATH),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["posterior_config"]["config_hash"]
    assert payload["scenarios"][0]["with_goal_posterior"]["planner_consumed_channel"] is True
