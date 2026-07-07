"""Tests for the issue #3465 topology-gate paired decision builder CLI script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmark"))

import build_issue3465_topology_gate_paired_decision as decision_builder  # noqa: E402

from robot_sf.benchmark.near_parity_promotion_gate import (  # noqa: E402
    DIAGNOSTIC,
    ELIGIBLE_FOR_PROMOTION,
    REVISE,
    STOP,
)

CONFIG_PATH = _REPO_ROOT / "configs" / "benchmarks" / "issue_3465_topology_gate_paired.yaml"
_DECISION_SCRIPT = (
    _REPO_ROOT / "scripts" / "benchmark" / "build_issue3465_topology_gate_paired_decision.py"
)


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Fixture to create a temporary config file with custom values."""
    config_data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    temp_path = tmp_path / "issue_3465_topology_gate_paired_temp.yaml"
    temp_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return temp_path


def test_decision_blocked_when_corrective_not_complete(
    temp_config_file: Path, tmp_path: Path
) -> None:
    """If corrective_complete is False in config, the decision remains blocked and diagnostic."""
    with open(temp_config_file, "r+", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config["readiness"]["corrective_complete"] = False
        f.seek(0)
        yaml.safe_dump(config, f)
        f.truncate()

    report = decision_builder.build_decision_report(temp_config_file, tmp_path)
    assert report["status"] == "blocked"
    assert report["reason"] == "corrective_incomplete"
    assert report["verdict"]["verdict"] == DIAGNOSTIC
    assert report["verdict"]["promote"] is False
    # The corrective-gate criterion must reflect the incomplete state, not a static "met".
    corrective = next(
        c for c in report["acceptance_evidence"] if c["criterion"].startswith("Corrective #3463")
    )
    assert corrective["status"] == "not_met"


def test_decision_blocked_when_campaign_summary_missing(
    temp_config_file: Path, tmp_path: Path
) -> None:
    """If campaign_summary.json is missing, decision report status is blocked."""
    report = decision_builder.build_decision_report(CONFIG_PATH, tmp_path)
    assert report["status"] == "blocked"
    assert report["reason"] == "campaign_summary_missing"
    assert "Campaign summary JSON is missing" in report["blocked_reasons"][0]
    assert str(_REPO_ROOT) not in report["blocked_reasons"][0]
    assert report["config_path"] == "configs/benchmarks/issue_3465_topology_gate_paired.yaml"
    assert report["acceptance_evidence"]
    assert report["acceptance_evidence"][-1]["status"] == "met"


def test_decision_blocked_when_campaign_summary_invalid(
    temp_config_file: Path, tmp_path: Path
) -> None:
    """If campaign_summary.json contains invalid JSON, it fails closed with blocked status."""
    summary_path = tmp_path / "campaign_summary.json"
    summary_path.write_text("invalid json", encoding="utf-8")

    report = decision_builder.build_decision_report(temp_config_file, tmp_path)
    assert report["status"] == "blocked"
    assert report["reason"] == "campaign_summary_invalid"
    assert "Failed to read/parse" in report["blocked_reasons"][0]


def test_decision_blocked_when_arms_not_found(temp_config_file: Path, tmp_path: Path) -> None:
    """If one or both required arms are missing in campaign_summary.json, report is blocked."""
    summary_path = tmp_path / "campaign_summary.json"
    summary_path.write_text(
        json.dumps({"planner_rows": [{"planner_key": "some_other_arm"}]}), encoding="utf-8"
    )

    report = decision_builder.build_decision_report(temp_config_file, tmp_path)
    assert report["status"] == "blocked"
    assert report["reason"] == "arms_not_found"
    assert "Campaign summary missing planner keys" in report["blocked_reasons"][0]


def test_decision_blocked_when_metrics_missing(temp_config_file: Path, tmp_path: Path) -> None:
    """If required metrics are missing in planner rows, report is blocked."""
    summary_path = tmp_path / "campaign_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "planner_rows": [
                    {"planner_key": "topology_gate_disabled", "status": "ok"},
                    {"planner_key": "topology_gate_enabled", "status": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = decision_builder.build_decision_report(temp_config_file, tmp_path)
    assert report["status"] == "blocked"
    assert report["reason"] == "metrics_missing"
    assert any("collision" in r for r in report["blocked_reasons"])


def test_decision_ready_with_mock_promotion_verdict(temp_config_file: Path, tmp_path: Path) -> None:
    """Mock-enabled build producing significant native improvement reports eligible_for_promotion."""
    report = decision_builder.build_decision_report(
        temp_config_file,
        tmp_path,
        mock=True,
        mock_safety_imp=0.05,
        mock_efficiency_imp=0.05,
        mock_paired_significant=True,
        mock_relies_on_fallback=False,
    )
    assert report["status"] == "ready"
    assert report["verdict"]["verdict"] == ELIGIBLE_FOR_PROMOTION
    assert report["verdict"]["promote"] is True
    assert report["deltas"]["safety_improvement"] == pytest.approx(0.05)
    assert report["deltas"]["efficiency_improvement"] == pytest.approx(0.05)
    # With a loaded, classified campaign the conservative-classification criterion is met.
    classification = next(
        c
        for c in report["acceptance_evidence"]
        if c["criterion"].startswith("Result classification is recorded")
    )
    assert classification["status"] == "met"


def test_decision_ready_with_mock_regression_verdict(
    temp_config_file: Path, tmp_path: Path
) -> None:
    """Mock-enabled build producing regression reports stop verdict."""
    report = decision_builder.build_decision_report(
        temp_config_file,
        tmp_path,
        mock=True,
        mock_safety_imp=-0.05,
        mock_efficiency_imp=0.0,
        mock_paired_significant=True,
    )
    assert report["status"] == "ready"
    assert report["verdict"]["verdict"] == STOP
    assert report["verdict"]["promote"] is False


def test_decision_ready_with_mock_fallback_verdict(temp_config_file: Path, tmp_path: Path) -> None:
    """Mock-enabled build relying on fallback reports revise verdict."""
    report = decision_builder.build_decision_report(
        temp_config_file,
        tmp_path,
        mock=True,
        mock_safety_imp=0.05,
        mock_efficiency_imp=0.05,
        mock_relies_on_fallback=True,
    )
    assert report["status"] == "ready"
    assert report["verdict"]["verdict"] == REVISE
    assert report["verdict"]["promote"] is False


def test_decision_ready_with_mock_not_significant_verdict(
    temp_config_file: Path, tmp_path: Path
) -> None:
    """Mock-enabled build with non-significant difference reports revise verdict."""
    report = decision_builder.build_decision_report(
        temp_config_file,
        tmp_path,
        mock=True,
        mock_safety_imp=0.05,
        mock_efficiency_imp=0.05,
        mock_paired_significant=False,
    )
    assert report["status"] == "ready"
    assert report["verdict"]["verdict"] == REVISE
    assert report["verdict"]["promote"] is False


def test_write_decision_artifacts(temp_config_file: Path, tmp_path: Path) -> None:
    """The artifact writer correctly emits summary.json, paired_deltas.csv, and README.md."""
    report = decision_builder.build_decision_report(
        temp_config_file,
        tmp_path,
        mock=True,
        mock_safety_imp=0.04,
        mock_efficiency_imp=0.03,
    )
    out_dir = tmp_path / "evidence"
    decision_builder.write_decision_artifacts(report, out_dir)

    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "paired_deltas.csv").is_file()
    assert (out_dir / "README.md").is_file()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ready"
    assert summary["verdict"]["verdict"] == ELIGIBLE_FOR_PROMOTION
    assert summary["acceptance_evidence"][0]["criterion"].startswith("Paired enabled/disabled")

    csv_lines = (out_dir / "paired_deltas.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "metric,disabled_val,enabled_val,delta"
    assert "collisions_mean" in csv_lines[1]
    assert "-0.04" in csv_lines[1]

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "# Issue #3465" in readme
    assert "**Decision Status:** `ready`" in readme
    assert "eligible_for_promotion" in readme
    assert "## Acceptance Evidence" in readme
    assert "PR #4487" in readme


def test_cli_execution_with_mock_flag(tmp_path: Path) -> None:
    """The CLI script can be run with --mock flag to generate artifacts."""
    out_dir = tmp_path / "cli_out"
    completed = subprocess.run(
        [
            sys.executable,
            str(_DECISION_SCRIPT),
            "--config",
            str(CONFIG_PATH),
            "--out",
            str(out_dir),
            "--mock",
            "--mock-safety-imp",
            "0.06",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Decision artifacts written to" in completed.stdout
    assert "Status: ready" in completed.stdout
    assert "Verdict: eligible_for_promotion" in completed.stdout

    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "README.md").is_file()
    assert (out_dir / "paired_deltas.csv").is_file()


def _write_valid_campaign_summary(tmp_path: Path, *, paired_significant: bool | None) -> None:
    """Write a real (non-mock) campaign summary with a measurable native improvement.

    ``paired_significant=None`` omits the field entirely, exercising the fail-closed
    default in real mode.
    """
    campaign: dict = {}
    if paired_significant is not None:
        campaign["paired_significant"] = paired_significant
    summary = {
        "campaign": campaign,
        "planner_rows": [
            {
                "planner_key": "topology_gate_disabled",
                "collisions_mean": 0.10,
                "path_efficiency_mean": 0.90,
                "status": "ok",
            },
            {
                "planner_key": "topology_gate_enabled",
                "collisions_mean": 0.05,
                "path_efficiency_mean": 0.95,
                "status": "ok",
            },
        ],
    }
    (tmp_path / "campaign_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_real_mode_missing_paired_significant_is_fail_closed(
    temp_config_file: Path, tmp_path: Path
) -> None:
    """A real campaign summary that omits paired_significant must not promote (fail-closed)."""
    _write_valid_campaign_summary(tmp_path, paired_significant=None)

    report = decision_builder.build_decision_report(temp_config_file, tmp_path)
    assert report["status"] == "ready"
    assert report["deltas"]["paired_significant"] is False
    assert report["verdict"]["verdict"] == REVISE
    assert report["verdict"]["promote"] is False


def test_real_mode_explicit_paired_significant_promotes(
    temp_config_file: Path, tmp_path: Path
) -> None:
    """An explicit paired_significant=true with a measurable improvement is promotable."""
    _write_valid_campaign_summary(tmp_path, paired_significant=True)

    report = decision_builder.build_decision_report(temp_config_file, tmp_path)
    assert report["status"] == "ready"
    assert report["deltas"]["paired_significant"] is True
    assert report["verdict"]["verdict"] == ELIGIBLE_FOR_PROMOTION
    assert report["verdict"]["promote"] is True
