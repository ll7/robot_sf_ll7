"""Tests for the forecast-lane capability inventory / preflight (issue #2835).

These are synthetic introspection tests: they assert that the read-only
inventory correctly reports present vs missing forecast-lane capabilities and
that it fails closed when a required capability is broken. No predictors,
training, or benchmark runs are involved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark import forecast_lane_inventory as fli

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_inventory_passes_on_current_checkout():
    """Every required forecast-lane capability is present on the real repo."""
    report = fli.build_forecast_lane_inventory()
    assert report["schema"] == "forecast_lane_inventory.v1"
    assert report["ok"] is True, report["summary"]["missing_required_ids"]
    summary = report["summary"]
    assert summary["required_missing"] == 0
    assert summary["required_present"] == summary["required"]
    assert summary["required"] >= 1


def test_every_capability_declares_an_owner_and_note():
    """The registry stays self-documenting: owner + note for each capability."""
    for spec in fli._FORECAST_CAPABILITIES:
        assert spec.owner, spec.capability_id
        assert spec.note, spec.capability_id
        assert spec.symbols, spec.capability_id


def test_capability_ids_are_unique():
    """Capability ids must be unique so report rows are unambiguous."""
    ids = [spec.capability_id for spec in fli._FORECAST_CAPABILITIES]
    assert len(ids) == len(set(ids))


def test_missing_file_is_reported_as_blocker(tmp_path):
    """A declared companion file that is absent surfaces as an explicit blocker."""
    spec = fli.ForecastCapabilitySpec(
        capability_id="probe_missing_file",
        sublane="contract",
        module="robot_sf.benchmark.forecast_batch",
        symbols=("ForecastBatch",),
        required=True,
        files=("does/not/exist.json",),
        owner="n/a",
        note="probe",
    )
    result = fli._probe_capability(spec, repo_root=tmp_path)
    assert not result.present
    assert result.status == "missing_files"
    assert any("does/not/exist.json" in b for b in result.blockers)


def test_missing_module_is_reported_as_blocker():
    """An unimportable module is reported as a missing_module blocker, not raised."""
    spec = fli.ForecastCapabilitySpec(
        capability_id="probe_missing_module",
        sublane="contract",
        module="robot_sf.benchmark._definitely_not_a_real_forecast_module",
        symbols=("Whatever",),
        required=True,
        owner="n/a",
        note="probe",
    )
    result = fli._probe_capability(spec, repo_root=_REPO_ROOT)
    assert result.status == "missing_module"
    assert not result.present
    assert any("import failed" in b for b in result.blockers)


def test_missing_symbol_is_reported_as_blocker():
    """A real module missing a declared symbol surfaces as missing_symbols."""
    spec = fli.ForecastCapabilitySpec(
        capability_id="probe_missing_symbol",
        sublane="contract",
        module="robot_sf.benchmark.forecast_batch",
        symbols=("ForecastBatch", "ThisSymbolDoesNotExist"),
        required=True,
        owner="n/a",
        note="probe",
    )
    result = fli._probe_capability(spec, repo_root=_REPO_ROOT)
    assert result.status == "missing_symbols"
    assert not result.present
    assert any("ThisSymbolDoesNotExist" in b for b in result.blockers)


def test_optional_missing_does_not_fail_overall(monkeypatch):
    """Optional capabilities never flip the overall ok flag to False."""
    broken_optional = fli.ForecastCapabilitySpec(
        capability_id="broken_optional",
        sublane="risk",
        module="robot_sf.benchmark._missing_optional_module",
        symbols=("X",),
        required=False,
        owner="n/a",
        note="probe",
    )
    good_required = fli._FORECAST_CAPABILITIES[0]
    monkeypatch.setattr(fli, "_FORECAST_CAPABILITIES", (good_required, broken_optional))
    report = fli.build_forecast_lane_inventory()
    assert report["ok"] is True
    assert report["summary"]["optional_missing"] == 1
    assert "broken_optional" in report["summary"]["missing_optional_ids"]


def test_missing_required_fails_overall(monkeypatch):
    """A missing required capability flips ok to False and lists the id."""
    broken_required = fli.ForecastCapabilitySpec(
        capability_id="broken_required",
        sublane="contract",
        module="robot_sf.benchmark._missing_required_module",
        symbols=("X",),
        required=True,
        owner="n/a",
        note="probe",
    )
    monkeypatch.setattr(fli, "_FORECAST_CAPABILITIES", (broken_required,))
    report = fli.build_forecast_lane_inventory()
    assert report["ok"] is False
    assert "broken_required" in report["summary"]["missing_required_ids"]


def test_markdown_render_contains_overall_and_blockers(monkeypatch):
    """The markdown view shows the verdict and lists blockers when present."""
    broken_required = fli.ForecastCapabilitySpec(
        capability_id="broken_required",
        sublane="contract",
        module="robot_sf.benchmark._missing_required_module",
        symbols=("X",),
        required=True,
        owner="owner.py",
        note="probe",
    )
    monkeypatch.setattr(fli, "_FORECAST_CAPABILITIES", (broken_required,))
    report = fli.build_forecast_lane_inventory()
    text = fli.format_inventory_markdown(report)
    assert "FAIL" in text
    assert "Blockers" in text
    assert "broken_required" in text


def test_cli_runs_and_emits_json():
    """The runnable script exits 0 and emits a valid JSON report on the real repo."""
    proc = subprocess.run(
        [sys.executable, "scripts/benchmark/forecast_lane_preflight.py", "--json"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["schema"] == "forecast_lane_inventory.v1"


def test_module_invocation_matches_script(monkeypatch):
    """`python -m` entry returns 0 when all required capabilities are present."""
    assert fli.main([]) == 0
    assert fli.main(["--json"]) == 0


def test_status_report_preserves_learned_predictor_gate():
    """Status ledger keeps learned predictors blocked while gate rows remain unresolved."""
    report = fli.build_forecast_lane_status()

    assert report["schema"] == "forecast_lane_status.v1"
    assert report["ok"] is True
    assert report["issue"] == 2835
    assert report["learned_predictor_unblocked"] is False
    assert "closed_loop_gate" in report["summary"]["learned_predictor_blocker_ids"]
    assert "learned_predictor" in report["summary"]["learned_predictor_blocker_ids"]


def test_status_rows_are_unique_and_valid():
    """Ledger rows have stable ids, valid statuses, and actionable blockers."""
    report = fli.build_forecast_lane_status()
    rows = report["requirements"]
    ids = [row["requirement_id"] for row in rows]

    assert len(ids) == len(set(ids))
    assert report["summary"]["invalid_status_ids"] == []
    for row in rows:
        assert row["status"] in fli._VALID_LANE_STATUSES
        assert row["current_artifacts"]
        assert row["remaining_blocker"]
        assert row["next_action"]


def test_status_markdown_lists_blockers():
    """Markdown status view exposes learned-predictor blockers."""
    report = fli.build_forecast_lane_status()
    text = fli.format_status_markdown(report)

    assert "learned predictor BLOCKED" in text
    assert "Learned-predictor blockers" in text
    assert "closed_loop_gate" in text


def test_cli_status_json_reports_expected_blocker():
    """The runnable script emits status JSON without running forecast workloads."""
    proc = subprocess.run(
        [sys.executable, "scripts/benchmark/forecast_lane_preflight.py", "--status", "--json"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "forecast_lane_status.v1"
    assert payload["learned_predictor_unblocked"] is False
    assert "closed_loop_gate" in payload["summary"]["learned_predictor_blocker_ids"]


def test_closure_audit_maps_criteria_to_evidence_and_keeps_issue_open():
    """Closure audit records criterion evidence and remaining unmet work."""
    report = fli.build_forecast_lane_closure_audit()

    assert report["schema"] == "forecast_lane_closure_audit.v1"
    assert report["ok"] is True
    assert report["issue"] == 2835
    assert report["closable"] is False
    assert report["recommendation"] == "keep_open"
    assert "closed_loop_same_seed_gate" in report["summary"]["unmet_criterion_ids"]
    assert "closed_loop_same_seed_gate" in report["summary"]["next_empirical_actions"]

    criteria = {row["criterion_id"]: row for row in report["criteria"]}
    assert criteria["forecast_batch_artifact_contract"]["status"] == "met"
    assert "#2849" in criteria["forecast_batch_artifact_contract"]["evidence"]
    assert criteria["closed_loop_same_seed_gate"]["status"] == "unresolved"
    assert "#2916" in criteria["closed_loop_same_seed_gate"]["evidence"]
    assert (
        "same-seed replay slice" in criteria["closed_loop_same_seed_gate"]["next_empirical_action"]
    )
    assert "fallback/degraded" in criteria["closed_loop_same_seed_gate"]["claim_boundary"]


def test_closure_audit_markdown_lists_remaining_criteria():
    """Markdown closure audit exposes keep-open verdict and unmet ids."""
    report = fli.build_forecast_lane_closure_audit()
    text = fli.format_closure_audit_markdown(report)

    assert "KEEP OPEN" in text
    assert "Remaining criteria" in text
    assert "Next empirical actions" in text
    assert "same-seed replay slice" in text
    assert "closed_loop_same_seed_gate" in text
    assert "#2849" in text


def test_cli_closure_audit_json_reports_keep_open():
    """The runnable script emits closure-audit JSON without forecast workloads."""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/forecast_lane_preflight.py",
            "--closure-audit",
            "--json",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "forecast_lane_closure_audit.v1"
    assert payload["recommendation"] == "keep_open"
    assert payload["closable"] is False
    assert "closed_loop_same_seed_gate" in payload["summary"]["next_empirical_actions"]


def test_module_status_invocation_succeeds():
    """Status mode succeeds because blockers are issue state, not CLI failure."""
    assert fli.main(["--status"]) == 0
    assert fli.main(["--status", "--json"]) == 0
    assert fli.main(["--closure-audit"]) == 0
    assert fli.main(["--closure-audit", "--json"]) == 0
    assert fli.main(["--integration-report"]) == 0
    assert fli.main(["--integration-report", "--json"]) == 0
    assert fli.main(["--final-synthesis"]) == 0
    assert fli.main(["--final-synthesis", "--json"]) == 0


def test_integration_report_consolidates_remaining_blockers_and_gates():
    """Integration report combines closure blockers and intentional learned gates."""
    report = fli.build_forecast_lane_integration_report()

    assert report["schema"] == "forecast_lane_integration_report.v1"
    assert report["ok"] is True
    assert report["issue"] == 2835
    assert report["recommendation"] == "keep_open"
    assert report["closable"] is False
    assert report["learned_predictor_unblocked"] is False
    assert "closed_loop_same_seed_gate" in report["summary"]["unmet_criterion_ids"]
    assert "learned_predictor" in report["summary"]["intentional_gate_ids"]
    blocker_ids = {row["criterion_id"] for row in report["blockers_remaining"]}
    assert "closed_loop_same_seed_gate" in blocker_ids
    gate_ids = {row["requirement_id"] for row in report["intentional_gates"]}
    assert "closed_loop_gate" in gate_ids


def test_integration_report_markdown_lists_remaining_blockers_and_gates():
    """Markdown integration report is a compact human-readable closure surface."""
    report = fli.build_forecast_lane_integration_report()
    text = fli.format_integration_report_markdown(report)

    assert "KEEP OPEN" in text
    assert "Intentional gates" in text
    assert "closed_loop_same_seed_gate" in text
    assert "learned_predictor" in text
    assert "same-seed replay slice" in text


def test_cli_integration_report_json_reports_keep_open():
    """The runnable script emits integration-report JSON without forecast workloads."""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/forecast_lane_preflight.py",
            "--integration-report",
            "--json",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "forecast_lane_integration_report.v1"
    assert payload["recommendation"] == "keep_open"
    assert payload["closable"] is False
    assert "learned_predictor" in payload["summary"]["intentional_gate_ids"]


def test_final_synthesis_records_continue_revise_stop_decisions():
    """Final synthesis makes learned-prediction decision explicit."""
    report = fli.build_forecast_lane_final_synthesis()

    assert report["schema"] == "forecast_lane_final_synthesis.v1"
    assert report["ok"] is True
    assert report["issue"] == 2835
    assert report["recommendation"] == "revise"
    assert report["closable"] is False
    recommendations = report["summary"]["recommendations"]
    assert recommendations["infrastructure"] == "continue"
    assert recommendations["learned_predictor_expansion"] == "revise"
    assert recommendations["paper_facing_claims"] == "stop"
    assert "closed_loop_same_seed_gate" in report["summary"]["unmet_criterion_ids"]


def test_final_synthesis_markdown_lists_remaining_empirical_blockers():
    """Markdown synthesis names decision and remaining empirical blockers."""
    report = fli.build_forecast_lane_final_synthesis()
    text = fli.format_final_synthesis_markdown(report)

    assert "Forecast lane final synthesis: revise" in text
    assert "Keep typed forecast-lane infrastructure" in text
    assert "Remaining empirical blockers" in text
    assert "closed_loop_same_seed_gate" in text


def test_cli_final_synthesis_json_reports_revise_decision():
    """The runnable script emits final-synthesis JSON without forecast workloads."""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/forecast_lane_preflight.py",
            "--final-synthesis",
            "--json",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "forecast_lane_final_synthesis.v1"
    assert payload["recommendation"] == "revise"
    assert payload["summary"]["recommendations"]["paper_facing_claims"] == "stop"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
