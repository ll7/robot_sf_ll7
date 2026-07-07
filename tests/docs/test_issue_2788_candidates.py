"""Validation tests for the issue #2788 negative-result candidate directory."""


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest

EVIDENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "context"
    / "evidence"
    / "issue_2788_negative_result_scenario_candidates"
)
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "generated_scenario_candidate.v1.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

BLOCKED_PROMOTION_WORDS = {"promote", "benchmark evidence", "paper evidence", "results tier"}
EXPECTED_CANDIDATE_COUNT = 3
EXPECTED_NEGATIVE_RESULT_IDS = {"nr-001", "nr-002"}
EXPECTED_SEVERITY_KEYS = {
    "collision_count",
    "comfort_force_max_N",
    "min_clearance_m",
    "near_miss_count",
    "objective_value",
    "timeout_risk",
    "ttc_min_s",
}
EXPECTED_DIVERSITY_KEYS = {
    "coverage_fraction",
    "dedup_rate",
    "param_distance_mean_m",
    "param_distance_min_m",
    "unique_scenario_families",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def get_candidates() -> list[Path]:
    """Retrieve all JSON candidate files from the evidence directory."""
    return sorted(EVIDENCE_DIR.glob("*.json"))


def _load_candidate(candidate_path: Path) -> dict[str, Any]:
    return json.loads(candidate_path.read_text(encoding="utf-8"))


def _candidate_text(candidate: dict[str, Any]) -> str:
    perturbations = candidate["perturbations"]
    rationale = " ".join(pert.get("rationale", "") for pert in perturbations)
    return " ".join(
        [
            candidate["candidate_id"],
            candidate["generator_run_id"],
            candidate["source_scenario_ref"]["scenario_name"],
            rationale,
            candidate.get("notes", ""),
        ]
    ).lower()


def test_candidate_count_matches_issue_contract() -> None:
    """Issue #2788 requires at least three candidate manifests."""
    assert len(get_candidates()) >= EXPECTED_CANDIDATE_COUNT


@pytest.mark.parametrize("candidate_path", get_candidates(), ids=lambda p: p.name)
class TestIssue2788Candidates:
    """Validate all candidates in the issue #2788 directory."""

    def test_candidate_validates_against_schema(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        jsonschema.validate(instance=candidate, schema=SCHEMA)

    def test_promotion_status_not_promoted(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        assert candidate["promotion_status"] == "not_promoted"

    def test_scenario_not_certified(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        assert candidate["validity"]["scenario_certified"] is False

    def test_trace_lineage_has_source_issue(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        assert candidate["trace_lineage"]["provenance"]["source_issue"] == "#2788"

    def test_notes_declares_scout_scope(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        notes = candidate.get("notes", "").lower()
        assert "scout" in notes
        assert "not_promoted" in notes

    def test_no_claim_overreach(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        notes = candidate.get("notes", "").lower()
        notes_safe = notes.replace("not_promoted", "")
        for word in BLOCKED_PROMOTION_WORDS:
            assert word not in notes_safe, f"Notes must not contain promotion word '{word}'"

    def test_grounded_in_negative_result_register(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        haystack = _candidate_text(candidate).replace("-", "")
        normalized_ids = {
            negative_id.replace("-", "") for negative_id in EXPECTED_NEGATIVE_RESULT_IDS
        }
        assert any(negative_id in haystack for negative_id in normalized_ids)

    def test_config_paths_exist_and_hashes_are_not_placeholders(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        for ref in (candidate["generator_config_ref"], candidate["source_scenario_ref"]):
            config_path = Path(ref["config_path"])
            assert (Path(__file__).resolve().parents[2] / config_path).exists()
            assert SHA256_RE.fullmatch(ref["config_hash"])

    def test_candidate_states_how_it_addresses_failure_mode(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        haystack = _candidate_text(candidate)
        assert "addresses the prior" in haystack
        assert "rather than repeating" in haystack

    def test_severity_and_diversity_are_unevaluated(self, candidate_path: Path) -> None:
        candidate = _load_candidate(candidate_path)
        severity = candidate["metrics_summary"]["severity"]
        diversity = candidate["metrics_summary"]["diversity"]
        assert set(severity) == EXPECTED_SEVERITY_KEYS
        assert set(diversity) == EXPECTED_DIVERSITY_KEYS
        assert severity == dict.fromkeys(EXPECTED_SEVERITY_KEYS)
        assert diversity == dict.fromkeys(EXPECTED_DIVERSITY_KEYS)
