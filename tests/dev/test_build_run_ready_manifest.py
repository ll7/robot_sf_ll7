"""Tests run-ready launch-packet manifest builder."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

import yaml

from scripts.dev.build_run_ready_manifest import build_run_ready_manifest, main


def _write_packet(path: Path, *, issue: int, command: str = "echo ok") -> None:
    """Write a minimal launch packet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "example.v1",
                "issue": issue,
                "claim_gate": {"target_claim": "diagnostic only"},
                "seed_budget": {"primary": {"seed_set": "tiny"}},
                "campaign_command": command,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_build_run_ready_manifest_summarizes_packets(tmp_path: Path) -> None:
    """Manifest should include compact readiness entries for each packet."""

    _write_packet(tmp_path / "configs" / "benchmarks" / "issue_1_launch_packet.yaml", issue=1)
    _write_packet(
        tmp_path / "configs" / "training" / "issue_2_launch_packet.yaml",
        issue=2,
        command="echo <placeholder>",
    )

    manifest = build_run_ready_manifest(repo_root=tmp_path)

    assert manifest["schema_version"] == "run-ready-launch-packet-manifest.v1"
    assert manifest["packet_count"] == 2
    assert manifest["ready_count"] == 1
    by_issue = {entry["issue"]: entry for entry in manifest["entries"]}
    assert by_issue[1]["ready"] is True
    assert by_issue[1]["kind"] == "benchmark"
    assert by_issue[1]["queue_hint"]["public_issue"] == "ll7/robot_sf_ll7#1"
    assert by_issue[1]["queue_hint"]["job_class"] == "policy_search_sweep"
    assert by_issue[1]["queue_hint"]["state"] == "proposed"
    assert by_issue[2]["ready"] is False
    assert by_issue[2]["kind"] == "training"


def test_build_run_ready_manifest_cli_writes_yaml_or_json(tmp_path: Path, capsys) -> None:
    """CLI should write output and print the same payload."""

    _write_packet(tmp_path / "configs" / "benchmarks" / "issue_3_launch_packet.yaml", issue=3)
    output = tmp_path / "experiments" / "run_ready_manifest.yaml"

    rc = main(["--repo-root", str(tmp_path), "--output", str(output)])
    printed = yaml.safe_load(capsys.readouterr().out)
    written = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert printed == written
    assert written["ready_count"] == 1

    json_output = tmp_path / "run_ready_manifest.json"
    rc = main(["--repo-root", str(tmp_path), "--output", str(json_output), "--json"])
    printed_json = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert printed_json["packet_count"] == 1
    assert json.loads(json_output.read_text(encoding="utf-8"))["ready_count"] == 1
