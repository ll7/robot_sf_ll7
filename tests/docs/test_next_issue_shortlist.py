"""Validation tests for the next-issue shortlist generator.

Covers deterministic ranking, missing-source graceful degradation, and
JSON/Markdown shape contracts.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

from pytest import MonkeyPatch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_sf.benchmark.algorithm_readiness import _ALGORITHMS
from scripts.tools.generate_next_issue_shortlist import (
    _candidates_from_gap_report,
    _candidates_from_ledger,
    _candidates_from_open_issues,
    _candidates_from_register,
    _deduplicate,
    _load_algorithm_readiness,
    _rank_score,
    generate_shortlist,
)

# -- Minimal fixture data ---------------------------------------------------

MINIMAL_REGISTER = {
    "schema_version": "negative_result_register.v1",
    "entries": [
        {
            "id": "nr-001",
            "hypothesis": "Test hypothesis",
            "tested_artifact": "test_artifact",
            "scenario": "test_scenario",
            "comparator": "baseline",
            "result_classification": "revise",
            "failure_mode": "mechanism_failed",
            "why_failed_or_inconclusive": "Mechanism did not clear hard slices.",
            "recommended_next_action": "Design a successor mechanism.",
            "linked_issues": [100],
            "claim_boundary": "Diagnostic-only.",
            "created_at": "2026-06-13",
        },
        {
            "id": "nr-002",
            "hypothesis": "Test hypothesis 2",
            "tested_artifact": "test_artifact_2",
            "scenario": "test_scenario_2",
            "comparator": "baseline",
            "result_classification": "diagnostic_only",
            "failure_mode": "scenario_too_weak",
            "why_failed_or_inconclusive": "Scenario too weak.",
            "recommended_next_action": None,
            "linked_issues": [101],
            "claim_boundary": "Diagnostic-only.",
            "created_at": "2026-06-13",
        },
    ],
}

MINIMAL_LEDGER = {
    "schema_version": "dissertation_evidence_ledger.v2",
    "rows": [
        {
            "area": "topology_guidance",
            "claim": "Test claim",
            "artifact_status": "current",
            "evidence_tier": "diagnostic",
            "allowed_wording": "Near-parity.",
            "caveat": "Do not claim improvement.",
            "source_issues": [200],
            "dissertation_chapter": "Discussion",
            "claim_gap": "New hypothesis needed.",
            "evidence_promotion_path": "Fresh campaign required.",
        },
        {
            "area": "observation_robustness",
            "claim": "Test claim 2",
            "artifact_status": "current",
            "evidence_tier": "release-backed",
            "allowed_wording": "Defines boundaries.",
            "caveat": "Not sim-to-real.",
            "source_issues": [201],
            "dissertation_chapter": "Methods",
            "claim_gap": None,
            "evidence_promotion_path": None,
        },
    ],
}

MINIMAL_GAP_REPORT = {
    "schema_version": "dissertation_gap_report.v1",
    "gaps": [
        {
            "area": "topology_guidance",
            "entry_id": None,
            "bucket": "blocked",
            "promotion_step_or_reason": "Fresh campaign required.",
            "source": "ledger",
            "source_issue": 200,
            "evidence_tier": "diagnostic",
            "result_classification": None,
            "allowed_wording_or_boundary": "Near-parity.",
            "caveat": "Do not claim improvement.",
            "claim_gap_or_reason": "New hypothesis needed.",
        },
        {
            "area": "observation_robustness",
            "entry_id": None,
            "bucket": "supported",
            "promotion_step_or_reason": None,
            "source": "ledger",
            "source_issue": 201,
            "evidence_tier": "release-backed",
            "result_classification": None,
            "allowed_wording_or_boundary": "Defines boundaries.",
            "caveat": "Not sim-to-real.",
            "claim_gap_or_reason": "Requires perception pipeline.",
        },
        {
            "area": None,
            "entry_id": "issue-2716-test",
            "bucket": "negative_revise_only",
            "promotion_step_or_reason": "Design successor.",
            "source": "register",
            "source_issue": 100,
            "evidence_tier": None,
            "result_classification": "revise",
            "allowed_wording_or_boundary": "Diagnostic-only.",
            "caveat": "Diagnostic-only.",
            "claim_gap_or_reason": "Mechanism did not clear.",
        },
    ],
}

MINIMAL_OPEN_ISSUES = {
    "issues": [
        {
            "number": 2792,
            "title": "planning: generate next-issue shortlist",
            "state": "OPEN",
            "labels": [{"name": "research"}, {"name": "evidence:analysis-only"}],
            "body_excerpt": "Generate a deterministic planning shortlist.",
        },
        {
            "number": 9999,
            "title": "closed issue ignored",
            "state": "CLOSED",
            "labels": [],
        },
    ]
}


# -- Shape validation -------------------------------------------------------


class TestJsonShape:
    """Verify JSON output has required top-level structure."""

    def test_top_level_fields(self) -> None:
        report, _ = generate_shortlist()
        required = {
            "schema_version",
            "generated_at",
            "purpose",
            "sources_status",
            "degradation_notes",
            "candidates",
            "candidate_count",
            "claim_boundaries",
        }
        assert required.issubset(set(report.keys())), (
            f"Missing fields: {required - set(report.keys())}"
        )

    def test_schema_version(self) -> None:
        report, _ = generate_shortlist()
        assert report["schema_version"] == "next_issue_shortlist.v1"

    def test_purpose_declares_synthesis_not_evidence(self) -> None:
        report, _ = generate_shortlist()
        purpose = report["purpose"].lower()
        assert "synthesis" in purpose
        assert "not new benchmark" in purpose

    def test_candidate_count_matches_list(self) -> None:
        report, _ = generate_shortlist()
        assert report["candidate_count"] == len(report["candidates"])

    def test_candidate_fields(self) -> None:
        report, _ = generate_shortlist()
        required = {
            "id",
            "title",
            "source",
            "bucket",
            "reason",
            "blockers",
            "data_source_status",
            "degradation_notes",
            "rank",
            "ranking_score",
        }
        for c in report["candidates"]:
            missing = required - set(c.keys())
            assert not missing, f"Candidate {c.get('id')} missing: {missing}"

    def test_missing_optional_snapshots_are_degradation_notes(self) -> None:
        report, _ = generate_shortlist()
        notes = "\n".join(report["degradation_notes"])
        assert "open_issues_snapshot" in notes
        assert "recent_prs_snapshot" in notes


class TestMarkdownShape:
    """Verify Markdown output has expected sections."""

    def test_has_heading(self) -> None:
        _, md = generate_shortlist()
        assert "# Next-Issue Shortlist" in md

    def test_has_data_sources_section(self) -> None:
        _, md = generate_shortlist()
        assert "## Data Sources" in md

    def test_has_candidates_section(self) -> None:
        _, md = generate_shortlist()
        assert "## Candidates" in md

    def test_has_claim_boundaries(self) -> None:
        _, md = generate_shortlist()
        assert "## Claim Boundaries" in md


# -- Deterministic ranking --------------------------------------------------


class TestDeterministicRanking:
    """Same inputs always produce the same ranking order."""

    def test_same_output_on_repeated_calls(self) -> None:
        report1, _ = generate_shortlist()
        report2, _ = generate_shortlist()
        ids1 = [c["id"] for c in report1["candidates"]]
        ids2 = [c["id"] for c in report2["candidates"]]
        assert ids1 == ids2, "Ranking order differs between identical runs"

    def test_rank_numbers_are_sequential(self) -> None:
        report, _ = generate_shortlist()
        ranks = [c["rank"] for c in report["candidates"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scores_are_non_increasing(self) -> None:
        report, _ = generate_shortlist()
        scores = [c["ranking_score"] for c in report["candidates"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at rank {i + 1} ({scores[i]}) < score at rank {i + 2} ({scores[i + 1]})"
            )


class TestRankScore:
    """Verify ranking score computation."""

    def test_negative_revise_beats_blocked(self) -> None:
        c_neg = {"bucket": "negative_revise_only", "evidence_tier": "diagnostic"}
        c_blk = {"bucket": "blocked", "evidence_tier": "diagnostic"}
        assert _rank_score(c_neg) > _rank_score(c_blk)

    def test_blocked_beats_remove_weaken(self) -> None:
        c_blk = {"bucket": "blocked", "evidence_tier": "diagnostic"}
        c_rm = {"bucket": "remove_weaken", "evidence_tier": "non-claimable"}
        assert _rank_score(c_blk) > _rank_score(c_rm)

    def test_supported_is_lowest(self) -> None:
        c_sup = {"bucket": "supported", "evidence_tier": "release-backed"}
        c_any = {"bucket": "negative_revise_only", "evidence_tier": "diagnostic"}
        assert _rank_score(c_sup) < _rank_score(c_any)

    def test_tie_breaking_by_id(self) -> None:
        report, _ = generate_shortlist()
        # Within same score, ids should be ascending
        scores_ids = [(c["ranking_score"], c["id"]) for c in report["candidates"]]
        for i in range(len(scores_ids) - 1):
            if scores_ids[i][0] == scores_ids[i + 1][0]:
                assert scores_ids[i][1] <= scores_ids[i + 1][1], (
                    f"Tie-break violated: {scores_ids[i][1]} > {scores_ids[i + 1][1]}"
                )


# -- Missing-source degradation ---------------------------------------------


class TestMissingSourceDegradation:
    """Graceful handling when sources are absent."""

    def test_missing_route_efficiency_records_degradation(self) -> None:
        report, _ = generate_shortlist()
        route_status = report["sources_status"]["route_efficiency"]
        assert route_status["available"] is False
        route_notes = [n for n in report["degradation_notes"] if "route_efficiency" in n]
        assert len(route_notes) > 0, "Missing route-efficiency should add degradation note"

    def test_all_sources_recorded_in_status(self) -> None:
        report, _ = generate_shortlist()
        expected_keys = {
            "negative_result_register",
            "dissertation_evidence_ledger",
            "dissertation_gap_report",
            "route_efficiency",
            "algorithm_readiness",
            "open_issues_snapshot",
            "recent_prs_snapshot",
        }
        assert expected_keys.issubset(set(report["sources_status"].keys()))

    def test_algorithm_readiness_loader_matches_registry(self) -> None:
        algorithms = _load_algorithm_readiness()
        assert len(algorithms) == len(_ALGORITHMS), (
            f"Expected {len(_ALGORITHMS)} algorithms, got {len(algorithms)}"
        )
        assert any(algo["canonical_name"] == "drl_vo" for algo in algorithms)

    def test_degradation_notes_are_strings(self) -> None:
        report, _ = generate_shortlist()
        for note in report["degradation_notes"]:
            assert isinstance(note, str)


class TestMissingSourceWithFixtures:
    """Test with explicit missing sources using tmp fixtures."""

    def test_empty_register_produces_candidates_from_other_sources(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Even with empty register, gap report and ledger still produce candidates."""
        import scripts.tools.generate_next_issue_shortlist as mod

        monkeypatch.setattr(mod, "REGISTER_PATH", tmp_path / "missing_register.json")
        report, _ = generate_shortlist()
        assert report["candidate_count"] > 0, (
            f"Expected candidate_count > 0, got {report['candidate_count']}"
        )
        register_notes = [n for n in report["degradation_notes"] if "negative_result_register" in n]
        assert register_notes, "Expected non-empty register_notes list"

    def test_empty_gap_report_still_has_register_candidates(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Missing gap report still yields register + ledger candidates."""
        import scripts.tools.generate_next_issue_shortlist as mod

        monkeypatch.setattr(mod, "GAP_REPORT_PATH", tmp_path / "missing_gap.json")
        report, _ = generate_shortlist()
        assert report["candidate_count"] > 0, (
            f"Expected candidate_count > 0, got {report['candidate_count']}"
        )
        gap_notes = [n for n in report["degradation_notes"] if "dissertation_gap_report" in n]
        assert gap_notes, "Expected non-empty gap_notes list"


# -- Candidate builder unit tests -------------------------------------------


class TestCandidateBuilders:
    """Unit tests for individual candidate builder functions."""

    def test_register_candidates_only_with_next_action(self) -> None:
        candidates = _candidates_from_register(MINIMAL_REGISTER)
        # nr-002 has recommended_next_action=None, should be excluded
        assert len(candidates) == 1
        assert candidates[0]["id"] == "nr-001"

    def test_gap_report_candidates_only_with_promotion(self) -> None:
        candidates = _candidates_from_gap_report(MINIMAL_GAP_REPORT)
        # observation_robustness has promotion_step_or_reason=None, excluded
        assert len(candidates) == 2

    def test_ledger_candidates_from_claim_gap(self) -> None:
        candidates = _candidates_from_ledger(MINIMAL_LEDGER)
        # topology_guidance has claim_gap + promotion_path
        assert len(candidates) >= 1
        topo = [c for c in candidates if "topology_guidance" in c["id"]]
        assert len(topo) == 1

    def test_deduplicate_removes_duplicates(self) -> None:
        c1 = {"id": "a", "source": "x", "source_issue": 1}
        c2 = {"id": "b", "source": "x", "source_issue": 1}
        c3 = {"id": "c", "source": "x", "source_issue": 2}
        result = _deduplicate([c1, c2, c3])
        assert len(result) == 2
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "c"

    def test_deduplicate_keeps_distinct_source_local_ids_without_issue(self) -> None:
        c1 = {"id": "algo-a", "source": "algorithm_readiness", "source_issue": None}
        c2 = {"id": "algo-b", "source": "algorithm_readiness", "source_issue": None}
        result = _deduplicate([c1, c2])
        assert [c["id"] for c in result] == ["algo-a", "algo-b"]


class TestOpenIssueSnapshots:
    """Verify optional open-issue snapshots produce bounded candidates."""

    def test_open_issue_candidate_shape(self) -> None:
        candidates = _candidates_from_open_issues(MINIMAL_OPEN_ISSUES)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["id"] == "open-issue-2792"
        assert candidate["source"] == "open_issues_snapshot"
        assert candidate["source_issue"] == 2792
        assert candidate["evidence_tier"] == "analysis-only"

    def test_generate_shortlist_uses_open_issue_snapshot(self, tmp_path: Path) -> None:
        import json

        issue_snapshot = tmp_path / "issues.json"
        issue_snapshot.write_text(json.dumps(MINIMAL_OPEN_ISSUES), encoding="utf-8")
        pr_snapshot = tmp_path / "prs.json"
        pr_snapshot.write_text(
            json.dumps({"prs": [{"number": 2810, "title": "merged PR"}]}),
            encoding="utf-8",
        )

        report, _ = generate_shortlist(
            open_issues_snapshot_path=issue_snapshot,
            recent_prs_snapshot_path=pr_snapshot,
        )

        assert report["sources_status"]["open_issues_snapshot"]["available"] is True
        assert report["sources_status"]["recent_prs_snapshot"]["available"] is True
        assert any(c["id"] == "open-issue-2792" for c in report["candidates"])
