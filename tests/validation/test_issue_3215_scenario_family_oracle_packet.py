"""Tests for the issue #3215 scenario-family oracle packet builder."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.validation.build_issue_3215_scenario_family_oracle_packet import (
    DEFAULT_PACKET,
    build_manifest,
    render_markdown,
    validate_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_default_packet() -> dict:
    return yaml.safe_load((REPO_ROOT / DEFAULT_PACKET).read_text(encoding="utf-8"))


def test_default_packet_is_ready_and_keeps_diagnostic_boundary() -> None:
    """Default packet validates but remains diagnostic-only launch metadata."""
    packet = _load_default_packet()
    manifest = build_manifest(DEFAULT_PACKET, packet)

    assert manifest["status"] == "ready"
    assert manifest["evidence_tier"] == "diagnostic-only"
    assert "does not run paired seeds" in manifest["claim_boundary"]
    assert "full benchmark campaign run" in manifest["out_of_scope"]
    assert not manifest["blocking_issues"]


def test_default_packet_has_oracle_arm_and_paired_seed_family() -> None:
    """Packet includes the oracle upper-bound arm and paired seed schedule."""
    packet = _load_default_packet()
    manifest = build_manifest(DEFAULT_PACKET, packet)

    assert {arm["id"] for arm in manifest["forecast_arms"]} == {
        "none",
        "constant_velocity",
        "interaction_aware",
        "oracle_future",
    }
    oracle = next(arm for arm in manifest["forecast_arms"] if arm["id"] == "oracle_future")
    assert oracle["oracle_state_allowed"] is True
    assert manifest["scenario_family"]["paired_seed_count"] == 30
    assert manifest["scenario_family"]["planned_paired_rows"] > 30


def test_validation_fails_closed_without_oracle_arm() -> None:
    """Removing the oracle arm blocks the launch packet."""
    packet = _load_default_packet()
    packet["forecast_arms"] = [
        arm for arm in packet["forecast_arms"] if arm["id"] != "oracle_future"
    ]

    checks = validate_packet(packet)

    assert any(check.name == "forecast_arms" and not check.passed for check in checks)
    assert build_manifest(DEFAULT_PACKET, packet)["status"] == "blocked"


def test_markdown_summary_preserves_boundary_and_checks() -> None:
    """Generated Markdown exposes checks and no-run claim boundary."""
    packet = _load_default_packet()
    markdown = render_markdown(build_manifest(DEFAULT_PACKET, packet))

    assert "Issue #3215 Scenario-Family Oracle Packet" in markdown
    assert "`oracle_future`" in markdown
    assert "does not run paired seeds" in markdown
    assert "| `schema_version` | PASS |" in markdown


def test_issue_3215_write_path_emits_review_markers(tmp_path: Path) -> None:
    """The migrated shared-writer path adds the required AI-GENERATED markers."""
    from scripts.validation import build_issue_3215_scenario_family_oracle_packet as builder

    out_json = tmp_path / "packet.json"
    out_md = tmp_path / "README.md"
    rc = builder.main(["--output-json", str(out_json), "--output-md", str(out_md)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert out_md.read_text(encoding="utf-8").splitlines()[0] == (
        "<!-- AI-GENERATED (robot_sf#3215) - NEEDS-REVIEW -->"
    )


def test_issue_3215_write_path_is_byte_deterministic(tmp_path: Path) -> None:
    """Re-running the deterministic packet writer yields identical evidence bytes."""
    from scripts.validation import build_issue_3215_scenario_family_oracle_packet as builder

    first = tmp_path / "first"
    second = tmp_path / "second"
    for out_dir in (first, second):
        rc = builder.main(
            [
                "--output-json",
                str(out_dir / "packet.json"),
                "--output-md",
                str(out_dir / "README.md"),
            ]
        )
        assert rc == 0
    assert (first / "packet.json").read_bytes() == (second / "packet.json").read_bytes()
    assert (first / "README.md").read_bytes() == (second / "README.md").read_bytes()
