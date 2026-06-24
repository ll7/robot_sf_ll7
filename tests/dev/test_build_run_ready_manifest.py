"""Tests for the run-ready launch-packet manifest builder (#3549)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from scripts.dev.build_run_ready_manifest import build_manifest, discover_packets
from scripts.dev.preflight_launch_packet import sha256_of

if TYPE_CHECKING:
    from pathlib import Path


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts").mkdir()
    return tmp_path


def _ready_packet(repo: Path, issue: int) -> None:
    cfg = repo / "configs" / f"cfg_{issue}.yaml"
    cfg.write_text("a: 1\n")
    packet = {
        "runnable_config": {"config": f"configs/cfg_{issue}.yaml", "config_sha256": sha256_of(cfg)},
        "campaign_command": "run",
        "claim_gate": "no_claim_until_run",
    }
    (repo / "configs" / f"r_issue_{issue}_launch_packet.yaml").write_text(yaml.safe_dump(packet))


def _broken_packet(repo: Path, issue: int) -> None:
    # references a missing config -> not ready
    packet = {
        "runnable_config": {"config": "configs/missing.yaml", "config_sha256": "a" * 64},
        "campaign_command": "run",
        "claim_gate": "x",
    }
    (repo / "configs" / f"b_issue_{issue}_launch_packet.yaml").write_text(yaml.safe_dump(packet))


def test_discover_packets_globs_launch_packets(tmp_path: Path):
    """Only files matching the launch-packet glob are discovered."""
    repo = _make_repo(tmp_path)
    _ready_packet(repo, 10)
    (repo / "configs" / "not_a_packet.yaml").write_text("x: 1\n")
    found = discover_packets(repo)
    assert len(found) == 1
    assert found[0].name == "r_issue_10_launch_packet.yaml"


def test_manifest_summary_counts_and_sorting(tmp_path: Path):
    """Summary tallies ready/total and entries sort by issue number."""
    repo = _make_repo(tmp_path)
    _ready_packet(repo, 30)
    _ready_packet(repo, 10)
    _broken_packet(repo, 20)
    manifest = build_manifest(repo)

    assert manifest["schema_version"] == "run-ready-launch-packet-manifest.v1"
    assert manifest["summary"] == {"total": 3, "ready": 2, "drift_guarded": 2}
    # sorted by issue number ascending
    assert [e["issue"] for e in manifest["packets"]] == [10, 20, 30]
    # the broken one is reported not-ready with a reason
    broken = next(e for e in manifest["packets"] if e["issue"] == 20)
    assert broken["ready"] is False
    assert broken["reasons"]


def test_manifest_entries_have_relative_paths(tmp_path: Path):
    """Packet paths in the manifest are repo-relative, not absolute."""
    repo = _make_repo(tmp_path)
    _ready_packet(repo, 42)
    manifest = build_manifest(repo)
    entry = manifest["packets"][0]
    assert entry["packet_path"] == "configs/r_issue_42_launch_packet.yaml"
    assert not entry["packet_path"].startswith("/")
