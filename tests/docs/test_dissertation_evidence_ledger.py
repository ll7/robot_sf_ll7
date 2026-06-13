"""Validation tests for the dissertation evidence ledger JSON schema."""

# ruff: noqa: D102

from __future__ import annotations

import json
from pathlib import Path

LEDGER_PATH = Path("docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json")
REQUIRED_ROW_FIELDS = {
    "area",
    "claim",
    "artifact_status",
    "evidence_tier",
    "allowed_wording",
    "caveat",
    "source_issues",
    "dissertation_chapter",
    "claim_gap",
}
VALID_ARTIFACT_STATUSES = {"current", "stale", "blocked"}
VALID_EVIDENCE_TIERS = {"release-backed", "diagnostic", "proposal", "non-claimable"}
BLOCKED_WORDING = {"do-not-use"}
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "created_at_utc",
    "issue",
    "purpose",
    "rows",
    "stale_artifact_summary",
    "reuse_recommendations",
    "claim_boundaries",
}


def _load_ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


class TestDissertationEvidenceLedger:
    """Schema and invariant checks for the dissertation evidence ledger."""

    def test_ledger_file_exists(self) -> None:
        assert LEDGER_PATH.exists(), f"Ledger not found at {LEDGER_PATH}"

    def test_ledger_is_valid_json(self) -> None:
        ledger = _load_ledger()
        assert isinstance(ledger, dict), "Ledger root must be a JSON object"

    def test_top_level_fields_present(self) -> None:
        ledger = _load_ledger()
        missing = REQUIRED_TOP_LEVEL_FIELDS - set(ledger.keys())
        assert not missing, f"Missing top-level fields: {missing}"

    def test_schema_version_matches(self) -> None:
        ledger = _load_ledger()
        assert ledger["schema_version"] == "dissertation_evidence_ledger.v1"

    def test_purpose_declares_synthesis_not_evidence(self) -> None:
        ledger = _load_ledger()
        purpose = ledger["purpose"].lower()
        assert "synthesis" in purpose, "Purpose must declare synthesis/planning aid"
        assert "not new benchmark" in purpose, "Purpose must exclude new benchmark evidence"

    def test_rows_are_non_empty_list(self) -> None:
        ledger = _load_ledger()
        assert isinstance(ledger["rows"], list), "rows must be a list"
        assert len(ledger["rows"]) > 0, "rows must not be empty"

    def test_all_row_fields_present(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            missing = REQUIRED_ROW_FIELDS - set(row.keys())
            assert not missing, f"Row {i} ({row.get('area', '?')}) missing fields: {missing}"

    def test_all_rows_have_allowed_wording(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            assert row["allowed_wording"], (
                f"Row {i} ({row.get('area', '?')}) must have non-empty allowed_wording"
            )

    def test_all_rows_have_caveat(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            assert row["caveat"], f"Row {i} ({row.get('area', '?')}) must have non-empty caveat"

    def test_blocked_wording_for_stale_rows(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            if row["artifact_status"] in ("stale", "blocked"):
                assert row["allowed_wording"].lower() in BLOCKED_WORDING, (
                    f"Row {i} ({row.get('area', '?')}) with status '{row['artifact_status']}' "
                    f"must use blocked wording, got '{row['allowed_wording']}'"
                )

    def test_valid_artifact_statuses(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            assert row["artifact_status"] in VALID_ARTIFACT_STATUSES, (
                f"Row {i} ({row.get('area', '?')}) has invalid artifact_status: "
                f"'{row['artifact_status']}'"
            )

    def test_valid_evidence_tiers(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            assert row["evidence_tier"] in VALID_EVIDENCE_TIERS, (
                f"Row {i} ({row.get('area', '?')}) has invalid evidence_tier: "
                f"'{row['evidence_tier']}'"
            )

    def test_source_issues_are_lists(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            assert isinstance(row["source_issues"], list), (
                f"Row {i} ({row.get('area', '?')}) source_issues must be a list"
            )
            assert len(row["source_issues"]) > 0, (
                f"Row {i} ({row.get('area', '?')}) source_issues must not be empty"
            )

    def test_no_diagnostic_rows_promoted_to_results(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            if row["evidence_tier"] == "diagnostic":
                allowed_lower = row["allowed_wording"].lower()
                assert (
                    "results" not in row["dissertation_chapter"].lower()
                    or "diagnostic" in allowed_lower
                ), (
                    f"Row {i} ({row.get('area', '?')}) is diagnostic but lists Results "
                    f"chapter without diagnostic qualifier in allowed_wording"
                )

    def test_stale_artifact_summary_present(self) -> None:
        ledger = _load_ledger()
        assert isinstance(ledger["stale_artifact_summary"], list)
        assert len(ledger["stale_artifact_summary"]) > 0

    def test_reuse_recommendations_present(self) -> None:
        ledger = _load_ledger()
        recs = ledger["reuse_recommendations"]
        assert isinstance(recs, dict)
        assert len(recs) > 0

    def test_claim_boundaries_present(self) -> None:
        ledger = _load_ledger()
        boundaries = ledger["claim_boundaries"]
        assert isinstance(boundaries, list)
        assert len(boundaries) > 0
        # Must mention synthesis/planning aid
        text = " ".join(boundaries).lower()
        assert "synthesis" in text or "planning" in text
