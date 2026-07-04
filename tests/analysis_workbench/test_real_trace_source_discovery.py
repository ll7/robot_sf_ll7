"""Tests for issue #3278 public-source discovery ledger checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.analysis_workbench.real_trace_source_discovery import (
    REAL_TRACE_SOURCE_DISCOVERY_SCHEMA_VERSION,
    SOURCE_DISCOVERY_EVIDENCE_BOUNDARY,
    SOURCE_DISCOVERY_STATUS_BLOCKED,
    SOURCE_DISCOVERY_STATUS_READY,
    RealTraceSourceDiscoveryError,
    check_real_trace_source_discovery,
    load_real_trace_source_discovery,
)

EXAMPLE_LEDGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_3278_real_trace_source_discovery_example.yaml"
)


def _ready_ledger() -> dict:
    """Return a complete discovery ledger with usable sources for both targets."""
    return {
        "schema_version": REAL_TRACE_SOURCE_DISCOVERY_SCHEMA_VERSION,
        "issue": 3278,
        "discovery_status": "complete",
        "required_targets": ["late_evasive_reaction", "oscillatory_local_control"],
        "candidate_sources": [
            {
                "source_id": "usable_late",
                "title": "Usable late-evasive source",
                "source_type": "public_dataset",
                "url_or_reference": "https://example.invalid/late",
                "license_status": "accepted",
                "access_status": "available",
                "coverage_status": "direct",
                "covered_targets": ["late_evasive_reaction"],
            },
            {
                "source_id": "usable_oscillatory",
                "title": "Usable oscillatory source",
                "source_type": "paper_supplement",
                "url_or_reference": "https://example.invalid/oscillatory",
                "license_status": "permissive",
                "access_status": "available",
                "coverage_status": "direct",
                "covered_targets": ["oscillatory_local_control"],
            },
        ],
    }


def test_ready_ledger_requires_usable_source_for_each_target() -> None:
    """Accepted, available, direct sources make every required target ready."""
    report = check_real_trace_source_discovery(_ready_ledger())

    assert report.discovery_status == SOURCE_DISCOVERY_STATUS_READY
    assert report.ready_targets == ("late_evasive_reaction", "oscillatory_local_control")
    assert report.blocked_targets == ()
    assert report.blockers == ()
    assert all(source.status == SOURCE_DISCOVERY_STATUS_READY for source in report.source_reports)
    assert report.evidence_boundary == SOURCE_DISCOVERY_EVIDENCE_BOUNDARY


def test_proxy_or_blocked_sources_fail_closed() -> None:
    """Proxy-only or inaccessible candidates keep the issue externally blocked."""
    ledger = _ready_ledger()
    ledger["candidate_sources"][0]["access_status"] = "blocked"
    ledger["candidate_sources"][0]["coverage_status"] = "proxy_only"

    report = check_real_trace_source_discovery(ledger)

    assert report.discovery_status == SOURCE_DISCOVERY_STATUS_BLOCKED
    assert report.ready_targets == ("oscillatory_local_control",)
    assert report.blocked_targets == ("late_evasive_reaction",)
    assert any("late_evasive_reaction" in blocker for blocker in report.blockers)
    blocked_source = report.source_reports[0]
    assert blocked_source.status == SOURCE_DISCOVERY_STATUS_BLOCKED
    assert any("access_status" in blocker for blocker in blocked_source.blockers)
    assert any("coverage_status" in blocker for blocker in blocked_source.blockers)


def test_incomplete_discovery_status_blocks_even_with_usable_source() -> None:
    """Ledger-level discovery status must be complete before readiness is allowed."""
    ledger = _ready_ledger()
    ledger["discovery_status"] = "in_progress"

    report = check_real_trace_source_discovery(ledger)

    assert report.discovery_status == SOURCE_DISCOVERY_STATUS_BLOCKED
    assert any("discovery_status" in blocker for blocker in report.blockers)


def test_schema_rejects_wrong_issue() -> None:
    """Discovery ledgers are issue-specific and cannot be reused silently."""
    ledger = _ready_ledger()
    ledger["issue"] = 3277

    with pytest.raises(RealTraceSourceDiscoveryError, match="/issue"):
        check_real_trace_source_discovery(ledger)


def test_example_ledger_loads_and_reports_external_block() -> None:
    """The committed example remains a metadata-only blocked ledger."""
    ledger = load_real_trace_source_discovery(EXAMPLE_LEDGER_PATH)

    report = check_real_trace_source_discovery(ledger, source=EXAMPLE_LEDGER_PATH)

    assert report.discovery_status == SOURCE_DISCOVERY_STATUS_BLOCKED
    assert set(report.blocked_targets) == {
        "late_evasive_reaction",
        "oscillatory_local_control",
    }
    assert report.source_reports[0].status == SOURCE_DISCOVERY_STATUS_BLOCKED


def test_load_missing_file_raises() -> None:
    """Loading a missing ledger path raises a clear error."""
    with pytest.raises(RealTraceSourceDiscoveryError, match="not found"):
        load_real_trace_source_discovery(EXAMPLE_LEDGER_PATH.with_suffix(".missing.yaml"))
