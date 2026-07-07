"""Validation tests for the dissertation gap report JSON and source accounting."""


from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LEDGER_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json"
)
REGISTER_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_2762_negative_result_register/register.json"
)
GAP_REPORT_JSON = (
    REPO_ROOT / "docs/context/evidence/issue_2784_dissertation_gap_report/gap_report.json"
)
CONTEXT_NOTE = REPO_ROOT / "docs/context/dissertation_gap_report.md"

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "generated_at",
    "purpose",
    "source_ledger",
    "source_register",
    "gaps",
    "claim_boundaries",
}
REQUIRED_GAP_FIELDS = {
    "bucket",
    "promotion_step_or_reason",
    "source",
    "source_issue",
    "allowed_wording_or_boundary",
    "caveat",
    "claim_gap_or_reason",
}
VALID_BUCKETS = {"supported", "blocked", "negative_revise_only", "remove_weaken"}
LEDGER_ROW_COUNT = 7
REGISTER_ENTRY_COUNT = 5


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestGapReportExists:
    """Verify output artifacts exist and are valid JSON."""

    def test_gap_report_json_exists(self) -> None:
        assert GAP_REPORT_JSON.exists(), f"Gap report not found at {GAP_REPORT_JSON}"

    def test_gap_report_json_is_valid(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        assert isinstance(report, dict), "Gap report root must be a JSON object"


class TestGapReportSchema:
    """Schema checks for the gap report JSON."""

    def test_top_level_fields_present(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        missing = REQUIRED_TOP_LEVEL_FIELDS - set(report.keys())
        assert not missing, f"Missing top-level fields: {missing}"

    def test_schema_version_matches(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        assert report["schema_version"] == "dissertation_gap_report.v1"

    def test_purpose_declares_synthesis_not_evidence(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        purpose = report["purpose"].lower()
        assert "synthesis" in purpose, "Purpose must declare synthesis/planning aid"
        assert "not new benchmark" in purpose, "Purpose must exclude new benchmark evidence"

    def test_all_gaps_have_required_fields(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        for i, gap in enumerate(report["gaps"]):
            missing = REQUIRED_GAP_FIELDS - set(gap.keys())
            assert not missing, f"Gap {i} missing fields: {missing}"
            # Must have either area or entry_id
            assert gap.get("area") or gap.get("entry_id"), (
                f"Gap {i} must have either 'area' or 'entry_id'"
            )

    def test_valid_bucket_values(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        for i, gap in enumerate(report["gaps"]):
            assert gap["bucket"] in VALID_BUCKETS, f"Gap {i} has invalid bucket: '{gap['bucket']}'"

    def test_no_claim_creation(self) -> None:
        """Gaps must not introduce claim-creation patterns that weren't in source."""
        report = _load_json(GAP_REPORT_JSON)
        claim_creation_patterns = [
            "is benchmark evidence",
            "is paper evidence",
            "qualifies as results",
            "promoted to results",
            "constitutes benchmark",
            "constitutes paper",
            "establishes benchmark",
            "establishes results",
            "paper-grade evidence",
        ]
        for i, gap in enumerate(report["gaps"]):
            val = (gap.get("allowed_wording_or_boundary") or "").lower()
            for pattern in claim_creation_patterns:
                assert pattern not in val, (
                    f"Gap {i} allowed_wording_or_boundary contains "
                    f"claim-creation pattern '{pattern}'"
                )

    def test_supported_bucket_only_release_backed(self) -> None:
        """Every gap in the supported bucket must come from a release-backed source."""
        report = _load_json(GAP_REPORT_JSON)
        for i, gap in enumerate(report["gaps"]):
            if gap["bucket"] == "supported":
                assert gap.get("evidence_tier") == "release-backed", (
                    f"Gap {i} is 'supported' but evidence_tier is "
                    f"'{gap.get('evidence_tier')}' (expected 'release-backed')"
                )

    def test_context_note_exists(self) -> None:
        assert CONTEXT_NOTE.exists(), f"Context note not found at {CONTEXT_NOTE}"


class TestSourceAccounting:
    """Verify all ledger rows and register entries appear in the gap report."""

    def test_all_ledger_rows_accounted_for(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        ledger = _load_json(LEDGER_PATH)
        ledger_gaps = [g for g in report["gaps"] if g["source"] == "ledger"]
        assert len(ledger_gaps) == len(ledger["rows"]), (
            f"Expected {len(ledger['rows'])} ledger gaps, got {len(ledger_gaps)}"
        )

    def test_all_register_entries_accounted_for(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        register = _load_json(REGISTER_PATH)
        register_gaps = [g for g in report["gaps"] if g["source"] == "register"]
        assert len(register_gaps) == len(register["entries"]), (
            f"Expected {len(register['entries'])} register gaps, got {len(register_gaps)}"
        )

    def test_total_gap_count(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        assert len(report["gaps"]) == LEDGER_ROW_COUNT + REGISTER_ENTRY_COUNT, (
            f"Expected {LEDGER_ROW_COUNT + REGISTER_ENTRY_COUNT} total gaps, "
            f"got {len(report['gaps'])}"
        )

    def test_bucket_distribution(self) -> None:
        """Verify expected bucket counts from known source data."""
        report = _load_json(GAP_REPORT_JSON)
        buckets = {}
        for gap in report["gaps"]:
            buckets[gap["bucket"]] = buckets.get(gap["bucket"], 0) + 1
        # supported: 1 (observation_robustness)
        assert buckets.get("supported", 0) == 1, (
            f"Expected 1 supported gap, got {buckets.get('supported', 0)}"
        )
        # blocked: 5 (topology_guidance, signalized_behavior, forecast-supported,
        # forecast-unsupported, exported_tables invalid-campaign caveat)
        assert buckets.get("blocked", 0) == 5, (
            f"Expected 5 blocked gaps, got {buckets.get('blocked', 0)}"
        )
        # negative_revise_only: 6 (pedestrian_density_stress + 5 register diagnostics)
        assert buckets.get("negative_revise_only", 0) == 6, (
            f"Expected 6 negative_revise_only gaps, got {buckets.get('negative_revise_only', 0)}"
        )
        # remove_weaken: 0 (exported_tables now has payloads but remains blocked/diagnostic)
        assert buckets.get("remove_weaken", 0) == 0, (
            f"Expected 0 remove_weaken gaps, got {buckets.get('remove_weaken', 0)}"
        )


class TestFailClosedWording:
    """Verify fail-closed discipline is preserved."""

    def test_claim_boundaries_present(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        boundaries = report["claim_boundaries"]
        assert isinstance(boundaries, list)
        assert len(boundaries) > 0

    def test_purpose_mentions_not_evidence(self) -> None:
        report = _load_json(GAP_REPORT_JSON)
        purpose = report["purpose"].lower()
        assert "not new benchmark" in purpose
        assert "not" in purpose

    def test_no_allowed_wording_upgraded(self) -> None:
        """Verbatim copy from source; no new wording introduced."""
        report = _load_json(GAP_REPORT_JSON)
        ledger = _load_json(LEDGER_PATH)
        ledger_allowed = {row["area"]: row["allowed_wording"] for row in ledger["rows"]}
        for gap in report["gaps"]:
            if gap["source"] == "ledger" and gap["area"]:
                assert gap["allowed_wording_or_boundary"] == ledger_allowed[gap["area"]], (
                    f"Gap for '{gap['area']}' has modified allowed_wording; "
                    f"must be verbatim from source"
                )

    def test_no_caveat_upgraded(self) -> None:
        """Verbatim copy from source; no new caveat introduced."""
        report = _load_json(GAP_REPORT_JSON)
        ledger = _load_json(LEDGER_PATH)
        ledger_caveats = {row["area"]: row["caveat"] for row in ledger["rows"]}
        for gap in report["gaps"]:
            if gap["source"] == "ledger" and gap["area"]:
                assert gap["caveat"] == ledger_caveats[gap["area"]], (
                    f"Gap for '{gap['area']}' has modified caveat; must be verbatim from source"
                )

    def test_register_boundaries_copied_verbatim(self) -> None:
        """Register entries reuse claim_boundary verbatim for boundary and caveat fields."""
        report = _load_json(GAP_REPORT_JSON)
        register = _load_json(REGISTER_PATH)
        register_boundaries = {
            entry["id"]: entry["claim_boundary"] for entry in register["entries"]
        }
        for gap in report["gaps"]:
            if gap["source"] == "register" and gap["entry_id"]:
                assert gap["allowed_wording_or_boundary"] == register_boundaries[gap["entry_id"]]
                assert gap["caveat"] == register_boundaries[gap["entry_id"]]
