"""Validation tests for the issue #2724 adversarial candidate register."""


from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from scripts.validation.validate_generated_scenario_candidate import validate_candidate

CANDIDATE_JSON = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "context"
    / "evidence"
    / "issue_2724_adversarial_candidate"
    / "candidate.json"
)
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "generated_scenario_candidate.v1.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

SCOUT_CANDIDATE_ID = "issue-2724-nr001-bottleneck-route-offset-001"
NEGATIVE_RESULT_SOURCE = "issue-2716-topology-reselection-cross-slice"
BLOCKED_PROMOTION_WORDS = {"promote", "benchmark evidence", "paper evidence", "results tier"}


def _load_candidate() -> dict:
    return json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))


class TestCandidateFile:
    """Candidate file must exist and be valid JSON."""

    def test_candidate_json_exists(self) -> None:
        assert CANDIDATE_JSON.exists(), f"Candidate not found at {CANDIDATE_JSON}"

    def test_candidate_is_valid_json(self) -> None:
        candidate = _load_candidate()
        assert isinstance(candidate, dict), "Candidate root must be a JSON object"


class TestSchemaValidation:
    """Candidate must pass the generated_scenario_candidate.v1 schema."""

    def test_candidate_validates_against_schema(self) -> None:
        candidate = _load_candidate()
        jsonschema.validate(instance=candidate, schema=SCHEMA)


class TestScoutCandidateContract:
    """Scout candidate must satisfy the minimal diagnostic-only contract."""

    def test_scout_candidate_id(self) -> None:
        candidate = _load_candidate()
        assert candidate["candidate_id"] == SCOUT_CANDIDATE_ID

    def test_generator_family_is_heuristic(self) -> None:
        candidate = _load_candidate()
        assert candidate["generator_family"] == "heuristic_perturbation"

    def test_promotion_status_not_promoted(self) -> None:
        candidate = _load_candidate()
        assert candidate["promotion_status"] == "not_promoted"

    def test_scenario_not_certified(self) -> None:
        candidate = _load_candidate()
        assert candidate["validity"]["scenario_certified"] is False

    def test_preflight_passed(self) -> None:
        candidate = _load_candidate()
        assert candidate["validity"]["preflight_passed"] is True

    def test_map_exists(self) -> None:
        candidate = _load_candidate()
        assert candidate["validity"]["map_exists"] is True

    def test_has_single_perturbation(self) -> None:
        candidate = _load_candidate()
        assert len(candidate["perturbations"]) == 1

    def test_perturbation_is_route_offset(self) -> None:
        candidate = _load_candidate()
        pert = candidate["perturbations"][0]
        assert pert["family"] == "robot_route_offset"
        assert "dx_m" in pert["parameters"]
        assert "dy_m" in pert["parameters"]
        assert "max_magnitude_m" in pert["parameters"]

    def test_candidate_is_grounded_in_negative_result_register(self) -> None:
        candidate = _load_candidate()
        haystack = " ".join(
            [
                candidate["candidate_id"],
                candidate["generator_run_id"],
                candidate["source_scenario_ref"]["scenario_name"],
                candidate["perturbations"][0].get("rationale", ""),
                candidate.get("notes", ""),
            ]
        )
        assert "nr-001" in haystack.lower()
        assert NEGATIVE_RESULT_SOURCE in haystack

    def test_severity_metrics_are_null(self) -> None:
        candidate = _load_candidate()
        severity = candidate["metrics_summary"]["severity"]
        for key in severity:
            assert severity[key] is None, f"Severity metric {key} should be null (unevaluated)"

    def test_diversity_metrics_are_null(self) -> None:
        candidate = _load_candidate()
        diversity = candidate["metrics_summary"]["diversity"]
        for key in diversity:
            assert diversity[key] is None, f"Diversity metric {key} should be null (unevaluated)"

    def test_trace_lineage_has_source_issue(self) -> None:
        candidate = _load_candidate()
        assert candidate["trace_lineage"]["provenance"]["source_issue"] == "#2724"

    def test_notes_declares_scout_scope(self) -> None:
        candidate = _load_candidate()
        notes = candidate.get("notes", "").lower()
        assert "scout" in notes, "Notes should declare scout diagnostic scope"
        assert "not_promoted" in notes, "Notes should reference not_promoted status"


class TestNoClaimOverreach:
    """Scout candidate must not contain promotion or benchmark-strength language."""

    def test_notes_no_benchmark_claim(self) -> None:
        candidate = _load_candidate()
        notes = candidate.get("notes", "").lower()
        # Remove the schema field value "not_promoted" before checking,
        # since "promote" is a substring of the allowed status string.
        notes_safe = notes.replace("not_promoted", "")
        for word in BLOCKED_PROMOTION_WORDS:
            assert word not in notes_safe, (
                f"Notes must not contain promotion word '{word}': {candidate.get('notes')}"
            )

    def test_rejection_reasons_explain_not_certified(self) -> None:
        candidate = _load_candidate()
        reasons = candidate["validity"].get("rejection_reasons", [])
        assert len(reasons) > 0, "Should have rejection reasons explaining non-certified status"
        reason_text = " ".join(reasons).lower()
        assert "not certified" in reason_text or "certification" in reason_text, (
            "Rejection reasons should explain certification status"
        )


class TestValidatorCli:
    """Validator helper should fail closed with readable errors."""

    def test_malformed_candidate_json_returns_error(self, tmp_path: Path) -> None:
        bad_candidate = tmp_path / "bad_candidate.json"
        bad_candidate.write_text("{not valid json", encoding="utf-8")

        errors = validate_candidate(str(bad_candidate))

        assert len(errors) == 1
        assert "Invalid JSON syntax" in errors[0]
