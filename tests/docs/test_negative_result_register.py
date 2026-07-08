"""Validation tests for the negative result register schema and seeded entries."""

from __future__ import annotations

import json
from pathlib import Path

REGISTER_MD = Path("docs/context/negative_result_register.md")
REGISTER_JSON = Path("docs/context/evidence/issue_2762_negative_result_register/register.json")
EVIDENCE_DIR = Path("docs/context/evidence/issue_2762_negative_result_register")

REQUIRED_ENTRY_FIELDS = {
    "id",
    "hypothesis",
    "tested_artifact",
    "scenario",
    "comparator",
    "result_classification",
    "failure_mode",
    "why_failed_or_inconclusive",
    "evidence_pointer",
    "recommended_next_action",
    "linked_issues",
    "claim_boundary",
    "created_at",
}
VALID_RESULT_CLASSIFICATIONS = {"revise", "diagnostic_only", "failed", "inconclusive"}
VALID_FAILURE_MODES = {
    "mechanism_failed",
    "scenario_too_weak",
    "evidence_diagnostic_only",
    "infrastructure_only",
    "stale",
    "blocked",
}
BLOCKED_PROMOTION_WORDS = {"promote", "benchmark evidence", "paper evidence", "results tier"}
PROMOTION_PATTERNS = [
    "is benchmark evidence",
    "is paper evidence",
    "qualifies as results",
    "promoted to results",
    "constitutes benchmark",
    "constitutes paper",
    "establishes benchmark",
    "proves benchmark",
    "paper-grade",
    "paper-grade evidence",
]
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "created_at_utc",
    "issue",
    "purpose",
    "entries",
    "claim_boundaries",
    "classification_definitions",
    "failure_mode_definitions",
}
SEeded_IDS = {
    "issue-2716-topology-reselection-cross-slice",
    "issue-2749-observation-noise-distant-pedestrian",
    "issue-2760-dissertation-evidence-ledger-diagnostic-rows",
}


def _load_register() -> dict:
    return json.loads(REGISTER_JSON.read_text(encoding="utf-8"))


class TestNegativeResultRegisterSchema:
    """Schema and invariant checks for the negative result register JSON."""

    def test_register_json_exists(self) -> None:
        assert REGISTER_JSON.exists(), f"Register not found at {REGISTER_JSON}"

    def test_register_md_exists(self) -> None:
        assert REGISTER_MD.exists(), f"Markdown register not found at {REGISTER_MD}"

    def test_register_is_valid_json(self) -> None:
        register = _load_register()
        assert isinstance(register, dict), "Register root must be a JSON object"

    def test_top_level_fields_present(self) -> None:
        register = _load_register()
        missing = REQUIRED_TOP_LEVEL_FIELDS - set(register.keys())
        assert not missing, f"Missing top-level fields: {missing}"

    def test_schema_version_matches(self) -> None:
        register = _load_register()
        assert register["schema_version"] == "negative_result_register.v1"

    def test_purpose_declares_synthesis_not_evidence(self) -> None:
        register = _load_register()
        purpose = register["purpose"].lower()
        assert "synthesis" in purpose or "planning" in purpose, (
            "Purpose must declare synthesis/planning aid"
        )
        assert "not new benchmark" in purpose or "not" in purpose, (
            "Purpose must exclude new benchmark evidence"
        )

    def test_entries_are_non_empty_list(self) -> None:
        register = _load_register()
        assert isinstance(register["entries"], list), "entries must be a list"
        assert len(register["entries"]) > 0, "entries must not be empty"

    def test_all_entry_fields_present(self) -> None:
        register = _load_register()
        for i, entry in enumerate(register["entries"]):
            missing = REQUIRED_ENTRY_FIELDS - set(entry.keys())
            assert not missing, f"Entry {i} ({entry.get('id', '?')}) missing fields: {missing}"

    def test_valid_result_classifications(self) -> None:
        register = _load_register()
        for i, entry in enumerate(register["entries"]):
            assert entry["result_classification"] in VALID_RESULT_CLASSIFICATIONS, (
                f"Entry {i} ({entry.get('id', '?')}) has invalid result_classification: "
                f"'{entry['result_classification']}'"
            )

    def test_valid_failure_modes(self) -> None:
        register = _load_register()
        for i, entry in enumerate(register["entries"]):
            assert entry["failure_mode"] in VALID_FAILURE_MODES, (
                f"Entry {i} ({entry.get('id', '?')}) has invalid failure_mode: "
                f"'{entry['failure_mode']}'"
            )

    def test_evidence_pointer_is_non_empty_list(self) -> None:
        register = _load_register()
        for i, entry in enumerate(register["entries"]):
            assert isinstance(entry["evidence_pointer"], list), (
                f"Entry {i} ({entry.get('id', '?')}) evidence_pointer must be a list"
            )
            assert len(entry["evidence_pointer"]) > 0, (
                f"Entry {i} ({entry.get('id', '?')}) evidence_pointer must not be empty"
            )

    def test_linked_issues_is_non_empty_list(self) -> None:
        register = _load_register()
        for i, entry in enumerate(register["entries"]):
            assert isinstance(entry["linked_issues"], list), (
                f"Entry {i} ({entry.get('id', '?')}) linked_issues must be a list"
            )
            assert len(entry["linked_issues"]) > 0, (
                f"Entry {i} ({entry.get('id', '?')}) linked_issues must not be empty"
            )

    def test_no_result_promotion_in_claim_boundary(self) -> None:
        register = _load_register()
        for i, entry in enumerate(register["entries"]):
            boundary_lower = entry["claim_boundary"].lower()
            for pattern in PROMOTION_PATTERNS:
                assert pattern not in boundary_lower, (
                    f"Entry {i} ({entry.get('id', '?')}) claim_boundary contains "
                    f"promotion pattern '{pattern}': {entry['claim_boundary']}"
                )

    def test_no_diagnostic_entries_promoted_to_results(self) -> None:
        register = _load_register()
        results_promotion_patterns = [
            "constitutes results",
            "qualifies as results",
            "promoted to results",
            "establishes results",
            "is results evidence",
        ]
        for i, entry in enumerate(register["entries"]):
            if entry["result_classification"] in ("diagnostic_only", "inconclusive"):
                boundary_lower = entry["claim_boundary"].lower()
                for pattern in results_promotion_patterns:
                    assert pattern not in boundary_lower, (
                        f"Entry {i} ({entry.get('id', '?')}) is diagnostic but "
                        f"claim_boundary contains results promotion pattern: '{pattern}'"
                    )

    def test_claim_boundaries_present(self) -> None:
        register = _load_register()
        boundaries = register["claim_boundaries"]
        assert isinstance(boundaries, list)
        assert len(boundaries) > 0
        text = " ".join(boundaries).lower()
        assert "synthesis" in text or "planning" in text


class TestSeededEntries:
    """Verify that the three seeded entries are present and reference durable evidence."""

    def test_seeded_ids_present(self) -> None:
        register = _load_register()
        actual_ids = {e["id"] for e in register["entries"]}
        missing = SEeded_IDS - actual_ids
        assert not missing, f"Missing seeded entry IDs: {missing}"

    def test_topology_entry_evidence_files_exist(self) -> None:
        evidence_paths = [
            Path("docs/context/evidence/issue_2716_topology_reselection_cross_slice/summary.json"),
            Path("docs/context/evidence/issue_2716_topology_reselection_cross_slice/report.md"),
        ]
        for p in evidence_paths:
            assert p.exists(), f"Seeded evidence file missing: {p}"

    def test_observation_noise_entry_evidence_files_exist(self) -> None:
        evidence_paths = [
            Path("docs/context/evidence/issue_2749_observation_noise_diagnostics/summary.json"),
            Path("docs/context/evidence/issue_2749_observation_noise_diagnostics/RESULT.md"),
        ]
        for p in evidence_paths:
            assert p.exists(), f"Seeded evidence file missing: {p}"

    def test_dissertation_ledger_entry_evidence_files_exist(self) -> None:
        evidence_paths = [
            Path("docs/context/dissertation_evidence_ledger.md"),
            Path("docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json"),
        ]
        for p in evidence_paths:
            assert p.exists(), f"Seeded evidence file missing: {p}"

    def test_topology_entry_classification_is_revise(self) -> None:
        register = _load_register()
        entry = next(
            e
            for e in register["entries"]
            if e["id"] == "issue-2716-topology-reselection-cross-slice"
        )
        assert entry["result_classification"] == "revise"
        assert entry["failure_mode"] == "mechanism_failed"

    def test_observation_noise_entry_classification_is_diagnostic_only(self) -> None:
        register = _load_register()
        entry = next(
            e
            for e in register["entries"]
            if e["id"] == "issue-2749-observation-noise-distant-pedestrian"
        )
        assert entry["result_classification"] == "diagnostic_only"
        assert entry["failure_mode"] == "scenario_too_weak"

    def test_dissertation_ledger_entry_classification_is_diagnostic_only(self) -> None:
        register = _load_register()
        entry = next(
            e
            for e in register["entries"]
            if e["id"] == "issue-2760-dissertation-evidence-ledger-diagnostic-rows"
        )
        assert entry["result_classification"] == "diagnostic_only"
        assert entry["failure_mode"] == "evidence_diagnostic_only"


class TestMarkdownRegisterConsistency:
    """Cross-check that Markdown and JSON registers are consistent."""

    def test_md_contains_all_seeded_ids(self) -> None:
        md_text = REGISTER_MD.read_text(encoding="utf-8")
        for entry_id in SEeded_IDS:
            assert entry_id in md_text, f"Markdown register does not contain seeded ID: {entry_id}"

    def test_md_contains_classification_definitions(self) -> None:
        md_text = REGISTER_MD.read_text(encoding="utf-8").lower()
        for classification in VALID_RESULT_CLASSIFICATIONS:
            assert classification in md_text, (
                f"Markdown register does not mention classification: {classification}"
            )

    def test_md_contains_failure_mode_definitions(self) -> None:
        md_text = REGISTER_MD.read_text(encoding="utf-8").lower()
        for mode in VALID_FAILURE_MODES:
            assert mode in md_text, f"Markdown register does not mention failure mode: {mode}"

    def test_md_no_result_promotion(self) -> None:
        md_text = REGISTER_MD.read_text(encoding="utf-8").lower()
        for pattern in PROMOTION_PATTERNS:
            assert pattern not in md_text, (
                f"Markdown register contains promotion pattern: {pattern}"
            )
