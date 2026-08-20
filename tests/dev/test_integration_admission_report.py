"""Focused offline tests for the integration admission report contract."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from scripts.dev.integration_admission_report import (
    SCHEMA_VERSION,
    VALID_STATES,
    build_report,
    main,
)

FIXTURE = Path(__file__).parent / "fixtures" / "integration_admission_report" / "states.json"
SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "contracts"
    / "integration_admission_report.v1.schema.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_offline_fixture_covers_every_required_state() -> None:
    """The offline corpus keeps all admission states executable and visible."""
    payload = _fixture()
    states = {
        build_report(payload, pr_number=row.get("number"), as_of=payload["captured_at"])["pr"][
            "state"
        ]
        if isinstance(row, dict) and "number" in row
        else build_report({"prs": [row]}, as_of=payload["captured_at"])["pr"]["state"]
        for row in payload["prs"]
    }
    assert states == set(VALID_STATES)


def test_merge_candidate_is_fail_closed_and_exposes_dimensions() -> None:
    """Only exact-head policy evidence reaches merge-candidate."""
    report = build_report(_fixture(), pr_number=76474, as_of="2026-08-20T12:00:00Z")
    assert report["schema"] == SCHEMA_VERSION
    assert report["report_only"] is True
    assert report["pr"]["state"] == "merge_candidate"
    assert set(report["pr"]["dimensions"]) == {
        "change_class",
        "shared_surface",
        "ci_cost",
        "review_requirement",
        "external_action",
        "base_sensitivity",
        "ownership",
    }
    assert report["pr"]["dimensions"] == {
        "change_class": "tooling",
        "shared_surface": "repository_control_plane",
        "ci_cost": "standard",
        "review_requirement": "ordinary",
        "external_action": "none",
        "base_sensitivity": "ordinary",
        "ownership": "unassigned",
    }
    assert report["pr"]["age_freshness"]["freshness"] == "fresh"
    assert report["pr"]["blockers"]["dependency"]["blocked"] is False


def test_explicit_blocker_is_reported_as_invalidation() -> None:
    """Policy blockers remain visible as exact admission invalidation codes."""
    report = build_report(_fixture(), pr_number=76472, as_of="2026-08-20T12:00:00Z")
    assert report["pr"]["state"] == "integration_blocked"
    assert "explicit_blocked:state:blocked" in report["pr"]["invalidation_codes"]


def test_missing_current_main_is_unavailable_not_invalid() -> None:
    """Unavailable baseline provenance remains distinct from malformed input."""
    report = build_report(_fixture(), pr_number=76476, as_of="2026-08-20T12:00:00Z")
    assert report["pr"]["state"] == "unavailable"
    assert "current_main_sha_unavailable" in report["pr"]["reason_codes"]
    assert report["pr"]["invalidation_codes"] == []


def test_inconsistent_baseline_is_invalid() -> None:
    """A contradictory baseline verdict cannot become an admitted candidate."""
    payload = _fixture()
    row = next(row for row in payload["prs"] if row.get("number") == 76471)
    row["base_freshness"] = {
        "base_sha": "different-base",
        "current_main_sha": "main-sha",
        "verdict": "fresh",
    }
    report = build_report(payload, pr_number=76471, as_of="2026-08-20T12:00:00Z")
    assert report["pr"]["state"] == "invalid"
    assert "inconsistent_base_freshness" in report["pr"]["invalidation_codes"]


def test_domain_and_external_lanes_remain_distinct() -> None:
    """Domain review and external-operation demand retain separate blockers."""
    payload = _fixture()
    domain_row = next(row for row in payload["prs"] if row.get("number") == 76471)
    domain_row.update(
        {
            "review_requirement": "domain",
            "external_action": "artifact",
        }
    )
    report = build_report(payload, pr_number=76471, as_of="2026-08-20T12:00:00Z")
    assert report["pr"]["state"] == "integration_blocked"
    assert "domain_review_required" in report["pr"]["invalidation_codes"]
    assert "external_action_required:artifact" in report["pr"]["invalidation_codes"]


def test_required_blocker_types_remain_distinct() -> None:
    """The pilot fixture keeps shared, decision, domain, age, and base blockers separate."""
    payload = _fixture()
    cases = {
        76478: ("integration_blocked", "shared_main_failure", "dependency"),
        76479: ("review_active", "author_decision_required", "decision"),
        76480: ("integration_blocked", "domain_review_required", "review"),
        76481: ("preparation_pr", None, None),
        76482: ("integration_blocked", "base_sha_stale", "dependency"),
    }
    for number, (expected_state, blocker_code, blocker_lane) in cases.items():
        report = build_report(payload, pr_number=number, as_of="2026-08-20T12:00:00Z")
        classification = report["pr"]
        assert classification["state"] == expected_state
        if number == 76481:
            assert classification["age_freshness"]["freshness"] == "stale"
            assert "pr_is_draft" in classification["reason_codes"]
            continue
        assert blocker_code in classification["reason_codes"]
        assert blocker_code in classification["blockers"][blocker_lane]["codes"]


def test_requested_as_of_drives_queue_age_and_lane_demand() -> None:
    """The fixed evaluation instant applies to queue rows as well as the selected PR."""
    payload = _fixture()
    report = build_report(payload, pr_number=76471, as_of="2026-08-27T12:00:00Z", max_queue_items=9)
    candidate = next(row for row in payload["prs"] if row.get("number") == 76471)
    assert candidate["updated_at"] == "2026-08-20T11:00:00Z"
    assert report["queue"]["lane_demand"]["ci"]["candidates"] == 3
    assert report["queue"]["lane_demand"]["review"]["demand"] == 3
    assert report["queue"]["lane_demand"]["external"]["demand"] == 0
    assert report["pr"]["age_freshness"]["freshness"] == "stale"
    assert report["queue"]["state_counts"]["integration_candidate"] == 2


def test_malformed_row_is_invalid_and_queue_demand_is_bounded() -> None:
    """Malformed input and queue truncation carry exact machine-readable codes."""
    payload = _fixture()
    payload["truncated"] = True
    report = build_report(payload, pr_number=76477, as_of="2026-08-20T12:00:00Z", max_queue_items=4)
    assert report["pr"]["state"] == "invalid"
    assert "invalid_labels" in report["pr"]["invalidation_codes"]
    assert report["queue"]["truncated"] is True
    assert "queue_snapshot_truncated" in report["queue"]["invalidation_codes"]
    assert report["queue"]["capacity_demand"]["pressure"] == "unknown"


def test_report_is_deterministic_and_schema_valid() -> None:
    """Same snapshot and evaluation instant produce byte-equivalent JSON data."""
    payload = _fixture()
    first = build_report(payload, pr_number=76471, as_of="2026-08-20T12:00:00Z", max_queue_items=5)
    second = build_report(payload, pr_number=76471, as_of="2026-08-20T12:00:00Z", max_queue_items=5)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(first)


def test_cli_reads_snapshot_only(tmp_path: Path, capsys) -> None:
    """The CLI emits a report from a fixture and requires no live route."""
    output = tmp_path / "report.json"
    assert (
        main(["--snapshot", str(FIXTURE), "--pr", "76471", "--json", "--output", str(output)]) == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pr"]["state"] == "integration_candidate"
    assert json.loads(capsys.readouterr().out)["report_only"] is True
