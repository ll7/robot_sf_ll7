"""Validation tests for the dissertation evidence ledger JSON schema."""


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
    "evidence_promotion_path",
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
        assert ledger["schema_version"] == "dissertation_evidence_ledger.v2"

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

    def test_all_rows_have_evidence_promotion_path(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            assert "evidence_promotion_path" in row, (
                f"Row {i} ({row.get('area', '?')}) missing evidence_promotion_path"
            )

    def test_evidence_promotion_path_types(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            val = row["evidence_promotion_path"]
            assert val is None or isinstance(val, str), (
                f"Row {i} ({row.get('area', '?')}) evidence_promotion_path must be "
                f"null or string, got {type(val).__name__}"
            )

    def test_live_replay_promotion_rows(self) -> None:
        ledger = _load_ledger()
        live_replay_areas = {"topology_guidance", "signalized_behavior"}
        for i, row in enumerate(ledger["rows"]):
            if row["area"] in live_replay_areas:
                assert row["evidence_promotion_path"] is not None, (
                    f"Row {i} ({row['area']}) should have a live replay promotion path"
                )
                assert "live replay" in row["evidence_promotion_path"].lower(), (
                    f"Row {i} ({row['area']}) promotion path must mention 'live replay'"
                )

    def test_exported_tables_preserve_invalid_campaign_caveat(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            if row["area"] == "exported_tables":
                assert row["evidence_promotion_path"] is not None, (
                    f"Row {i} ({row['area']}) should keep a promotion path for invalid evidence"
                )
                assert "snqi contract" in row["evidence_promotion_path"].lower(), (
                    f"Row {i} ({row['area']}) promotion path must mention SNQI repair"
                )
                assert "diagnostic_only" in row["caveat"].lower(), (
                    f"Row {i} ({row['area']}) caveat must preserve the PPO failure"
                )
                assert row["evidence_tier"] == "diagnostic", (
                    f"Row {i} ({row['area']}) must not be treated as benchmark evidence"
                )

    def test_forecast_support_row_present(self) -> None:
        ledger = _load_ledger()
        supported = [row for row in ledger["rows"] if row["area"] == "prediction_supported"]
        unsupported = [row for row in ledger["rows"] if row["area"] == "prediction_unsupported"]
        assert len(supported) == 1, "Exactly one supported forecast-lane row is required"
        assert len(unsupported) == 1, "Exactly one unsupported forecast-lane row is required"

    def test_forecast_rows_reference_anchor_issues(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            if row["area"].startswith("prediction_"):
                assert 2761 in row["source_issues"], (
                    f"Row {i} ({row.get('area', '?')}) should reference #2761"
                )
                assert 2835 in row["source_issues"], (
                    f"Row {i} ({row.get('area', '?')}) should reference #2835"
                )

    def test_forecast_supported_row_claim_boundary(self) -> None:
        ledger = _load_ledger()
        supported_rows = [row for row in ledger["rows"] if row["area"] == "prediction_supported"]
        assert supported_rows, "Prediction-supported row missing from ledger"
        row = supported_rows[0]
        assert row["artifact_status"] == "current"
        assert row["evidence_tier"] == "diagnostic"
        assert row["evidence_promotion_path"] is not None, (
            "Supported forecast rows need a promotion path"
        )
        claim_lower = row["claim"].lower()
        caveat_lower = row["caveat"].lower()
        for keyword in [
            "forecastbatch.v1",
            "observation-tier",
            "probabilistic",
            "baseline",
            "transferability",
            "conformal",
            "closed-loop",
        ]:
            assert keyword in claim_lower, (
                f"Supported forecast claim {row.get('area', '?')} missing '{keyword}'"
            )
        assert "not yet benchmark- or paper-grade claim support" in caveat_lower

    def test_forecast_unsupported_row_claim_boundary(self) -> None:
        ledger = _load_ledger()
        unsupported_rows = [
            row for row in ledger["rows"] if row["area"] == "prediction_unsupported"
        ]
        assert unsupported_rows, "Prediction-unsupported row missing from ledger"
        row = unsupported_rows[0]
        claim_lower = row["claim"].lower()
        allowed_lower = row["allowed_wording"].lower()
        caveat_lower = row["caveat"].lower()
        assert "not support" in claim_lower or "no durable evidence" in claim_lower
        assert "safety" in allowed_lower, (
            "Unsupported forecast wording should reference safety boundary"
        )
        assert "progress" in allowed_lower, (
            "Unsupported forecast wording should reference progress boundary"
        )
        assert "transfer" in caveat_lower, (
            "Unsupported forecast caveat should include transfer boundary"
        )
        assert row["evidence_promotion_path"] is not None, (
            "Unsupported forecast row should include a concrete promotion path before claims can be upgraded"
        )

    def test_no_promotion_path_rows(self) -> None:
        ledger = _load_ledger()
        no_path_areas = {"observation_robustness", "pedestrian_density_stress"}
        for i, row in enumerate(ledger["rows"]):
            if row["area"] in no_path_areas:
                assert row["evidence_promotion_path"] is None, (
                    f"Row {i} ({row['area']}) should have null promotion path "
                    f"(no credible path), got '{row['evidence_promotion_path']}'"
                )

    def test_promotion_path_does_not_mark_rows_release_backed(self) -> None:
        ledger = _load_ledger()
        for i, row in enumerate(ledger["rows"]):
            if row["evidence_promotion_path"] is not None:
                assert row["evidence_tier"] != "release-backed", (
                    f"Row {i} ({row.get('area', '?')}) has a promotion path but is already "
                    f"release-backed; promotion paths must remain pending work, not evidence "
                    f"upgrades"
                )

    def test_claim_boundaries_include_promotion_path_rule(self) -> None:
        ledger = _load_ledger()
        boundaries = " ".join(ledger["claim_boundaries"]).lower()
        assert "evidence_promotion_path" in boundaries, (
            "claim_boundaries must mention the evidence_promotion_path fail-closed rule"
        )
        assert "does not upgrade" in boundaries
        assert "benchmark or paper evidence" in boundaries
        assert "reclassifying the evidence tier" in boundaries
