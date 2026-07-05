"""Tests for the issue #2845 heavy-predictor revival decision packet."""

from __future__ import annotations

import json

from robot_sf.research import forecast_heavy_model_inventory as inv
from robot_sf.research.forecast_heavy_model_inventory import (
    ENTRY_POINT_SURFACES,
    EXPERIMENT_PREREQUISITES,
    HeavyModelRevivalDecisionPacket,
    MinimumExperimentStatus,
    build_inventory_report,
    build_revival_decision_packet,
    render_revival_decision_packet_markdown,
    repo_root,
)
from scripts.research.check_forecast_heavy_model_inventory import main as cli_main


def _write(root, relpath: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# test fixture\n", encoding="utf-8")


def test_revival_decision_packet_is_fail_closed_on_real_checkout():
    """The live checkout remains deferred; packet names blockers and allowed scope."""
    packet = build_revival_decision_packet(build_inventory_report(root=repo_root()))

    assert isinstance(packet, HeavyModelRevivalDecisionPacket)
    assert packet.issue == inv.ISSUE
    assert packet.evidence_status == "analysis_only"
    assert packet.revival_status == "deferred_blocked"
    assert "do not implement or train" in packet.recommendation
    assert packet.blockers
    assert any("staged_holdout_dataset" in blocker for blocker in packet.blockers)
    assert "Slurm/GPU submission" in packet.forbidden_actions
    assert "heavy model training" in packet.forbidden_actions
    payload = packet.to_dict()
    assert payload["inventory"]["minimum_experiment_status"] == "blocked"


def test_revival_decision_packet_reports_surface_contract_blocker(tmp_path):
    """Broken required surfaces outrank prerequisite planning blockers."""
    report = build_inventory_report(root=tmp_path)
    packet = build_revival_decision_packet(report)

    assert packet.revival_status == "blocked_surface_contract"
    assert "repair offline-evaluation surfaces" in packet.recommendation
    assert any("forecast_metrics" in blocker for blocker in packet.blockers)


def test_revival_decision_packet_marks_local_ready_external_blocked(tmp_path):
    """When local prerequisites are present, external decisions still gate runs."""
    for surface in ENTRY_POINT_SURFACES:
        _write(tmp_path, surface.file_path)
    for prereq in EXPERIMENT_PREREQUISITES:
        if prereq.external or not prereq.probe_paths:
            continue
        concrete = prereq.probe_paths[0].replace("*", "match").replace("?", "x")
        _write(tmp_path, concrete)

    report = build_inventory_report(root=tmp_path)
    assert report.minimum_experiment_status is MinimumExperimentStatus.READY
    packet = build_revival_decision_packet(report)

    assert packet.revival_status == "local_ready_external_blocked"
    assert "dependency/checkpoint decision" in packet.recommendation
    assert packet.required_before_offline_run


def test_render_revival_decision_packet_markdown_contains_guardrails():
    """Markdown packet exposes status, blockers, and forbidden actions."""
    packet = build_revival_decision_packet(build_inventory_report(root=repo_root()))
    text = render_revival_decision_packet_markdown(packet)

    assert f"Issue #{inv.ISSUE}" in text
    assert packet.revival_status in text
    assert "## Blockers" in text
    assert "## Forbidden actions" in text
    assert "full benchmark campaign" in text


def test_cli_decision_packet_modes_emit_guardrails(capsys):
    """CLI packet modes emit Markdown and JSON guardrail views."""
    code = cli_main(["--decision-packet"])
    out = capsys.readouterr().out
    assert code == 0
    assert "revival decision packet" in out
    assert "Slurm/GPU submission" in out

    code = cli_main(["--decision-packet", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["issue"] == inv.ISSUE
    assert payload["revival_status"] == "deferred_blocked"
    assert "full benchmark campaign" in payload["forbidden_actions"]
