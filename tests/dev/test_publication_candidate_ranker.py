"""Tests for publication candidate ranker helpers and fixture-backed rankings."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.dev import publication_candidate_ranker as ranker

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests/fixtures/publication_candidate_ranker"


def _fixture(name: str) -> object:
    """Load one JSON fixture from test resources."""
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_rank_candidates_high_evidence_readiness() -> None:
    """High evidence-readiness candidate should rank with a strong total score."""
    candidates_raw = _fixture("candidates_high_evidence.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)

    assert len(ranked) == 1
    top = ranked[0]
    assert top.rank == 1
    assert top.issue_number == 2700
    assert top.total_score > 0.7
    assert top.scores["evidence_readiness"] == 0.95
    assert any("linter passes" in r for r in top.reasons)


def test_rank_candidates_duplicate_risk_penalizes() -> None:
    """High duplication risk should reduce total score relative to low-risk peers."""
    candidates_raw = _fixture("candidates_duplicate_risk.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)

    assert len(ranked) == 2
    high_dup = next(r for r in ranked if r.issue_number == 2710)
    low_dup = next(r for r in ranked if r.issue_number == 2711)

    assert low_dup.total_score > high_dup.total_score
    assert high_dup.duplication_notes
    assert any("overlaps" in n for n in high_dup.duplication_notes)


def test_rank_candidates_blocked_by_preserved() -> None:
    """Blocked-by relationships must appear in the ranked output."""
    candidates_raw = _fixture("candidates_blocked.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)

    blocked = next(r for r in ranked if r.issue_number == 2720)
    unblocked = next(r for r in ranked if r.issue_number == 2721)

    assert blocked.blocked_by == (2700, 2701)
    assert unblocked.blocked_by == ()


def test_rank_candidates_ordering_by_score() -> None:
    """Candidates must be sorted by total score descending, then issue number."""
    candidates_raw = _fixture("candidates_mixed.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)

    assert len(ranked) == 3
    scores_desc = [r.total_score for r in ranked]
    assert scores_desc == sorted(scores_desc, reverse=True)

    assert ranked[0].issue_number == 2731
    assert ranked[1].issue_number == 2732
    assert ranked[2].issue_number == 2730


def test_rank_candidates_empty_list() -> None:
    """Empty input should produce an empty ranking."""
    candidates_raw = _fixture("candidates_empty.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)

    assert ranked == []


def test_format_markdown_report_empty() -> None:
    """Markdown report for empty candidates should state no candidates."""
    report = ranker.format_markdown_report([], total_count=0)
    assert "No candidates to rank" in report


def test_format_markdown_report_includes_details() -> None:
    """Markdown report must include reasons, caveats, and duplication notes."""
    candidates_raw = _fixture("candidates_mixed.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)
    report = ranker.format_markdown_report(ranked, total_count=len(candidates))

    assert "# Publication Candidate Ranking" in report
    assert "## Candidate Details" in report
    assert "Reasons:" in report
    assert "Evidence caveats:" in report
    assert "Duplication notes:" in report
    assert "Blocked by:" in report


def test_compute_total_score_weights() -> None:
    """Verify weighted scoring formula with known inputs."""
    scores = {
        "dissertation_relevance": 1.0,
        "evidence_readiness": 1.0,
        "implementation_boundedness": 1.0,
        "artifact_dependency": 0.0,
        "expected_research_value": 1.0,
        "duplication_risk": 0.0,
    }
    total = ranker._compute_total_score(scores)
    assert abs(total - 1.0) < 1e-6

    scores_low = {
        "dissertation_relevance": 0.0,
        "evidence_readiness": 0.0,
        "implementation_boundedness": 0.0,
        "artifact_dependency": 1.0,
        "expected_research_value": 0.0,
        "duplication_risk": 1.0,
    }
    total_low = ranker._compute_total_score(scores_low)
    assert abs(total_low - 0.0) < 1e-6


def test_parse_candidate_clamps_scores() -> None:
    """Out-of-range scores should be clamped to [0, 1]."""
    raw = {
        "issue_number": 9999,
        "issue_url": "https://github.com/ll7/robot_sf_ll7/issues/9999",
        "title": "clamped",
        "linter_ok": True,
        "linter_findings": [],
        "scores": {
            "dissertation_relevance": 1.5,
            "evidence_readiness": -0.3,
            "implementation_boundedness": 2.0,
            "artifact_dependency": -1.0,
            "expected_research_value": 0.5,
            "duplication_risk": 0.5,
        },
        "reasons": [],
        "evidence_caveats": [],
        "duplication_notes": [],
        "blocked_by": [],
    }
    candidate = ranker._parse_candidate(raw)
    assert candidate.scores["dissertation_relevance"] == 1.0
    assert candidate.scores["evidence_readiness"] == 0.0
    assert candidate.scores["implementation_boundedness"] == 1.0
    assert candidate.scores["artifact_dependency"] == 0.0


def test_main_json_output(capsys) -> None:
    """CLI with --format json should produce valid machine-readable output."""
    candidates_path = FIXTURE_DIR / "candidates_mixed.json"
    exit_code = ranker.main(
        [
            "--candidates-json",
            str(candidates_path),
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "publication_candidate_ranker.v1"
    assert payload["ok"] is True
    assert payload["read_only"] is True
    assert payload["project_writes"] is False
    assert payload["ranked_count"] == 3
    assert len(payload["ranked_candidates"]) == 3


def test_main_markdown_output(capsys) -> None:
    """CLI with default markdown format should produce a Markdown report."""
    candidates_path = FIXTURE_DIR / "candidates_high_evidence.json"
    exit_code = ranker.main(
        [
            "--candidates-json",
            str(candidates_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "# Publication Candidate Ranking" in output
    assert "Ranked 1 candidate(s)" in output


def test_rank_candidates_linter_fields_preserved() -> None:
    """linter_ok and linter_findings must flow from CandidateInput to RankedCandidate."""
    candidates_raw = _fixture("candidates_linter_fail.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)

    failing = next(r for r in ranked if r.issue_number == 2800)
    passing = next(r for r in ranked if r.issue_number == 2801)

    assert failing.linter_ok is False
    assert len(failing.linter_findings) == 2
    assert failing.linter_findings[0]["code"] == "MISSING_DOCSTRING"
    assert failing.linter_findings[1]["code"] == "UNUSED_IMPORT"

    assert passing.linter_ok is True
    assert passing.linter_findings == ()


def test_markdown_report_includes_linter_findings() -> None:
    """Markdown candidate details must show linter status and finding codes."""
    candidates_raw = _fixture("candidates_linter_fail.json")
    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)
    report = ranker.format_markdown_report(ranked, total_count=len(candidates))

    assert "**Linter:** fail" in report
    assert "**Linter:** pass" in report
    assert "`MISSING_DOCSTRING`" in report
    assert "public function lacks docstring" in report
    assert "`UNUSED_IMPORT`" in report
    assert "unused import os" in report


def test_json_report_includes_linter_findings() -> None:
    """JSON ranked_candidates must carry linter_ok and linter_findings."""
    candidates_raw = _fixture("candidates_linter_fail.json")
    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(candidates)
    report = ranker._build_report(ranked, total_count=len(candidates))

    failing = next(c for c in report["ranked_candidates"] if c["issue_number"] == 2800)
    passing = next(c for c in report["ranked_candidates"] if c["issue_number"] == 2801)

    assert failing["linter_ok"] is False
    assert len(failing["linter_findings"]) == 2
    assert failing["linter_findings"][0]["code"] == "MISSING_DOCSTRING"

    assert passing["linter_ok"] is True
    assert passing["linter_findings"] == []


def test_cli_json_output_includes_linter(capsys) -> None:
    """CLI --format json must include linter fields in ranked candidates."""
    candidates_path = FIXTURE_DIR / "candidates_linter_fail.json"
    exit_code = ranker.main(["--candidates-json", str(candidates_path), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    failing = next(c for c in payload["ranked_candidates"] if c["issue_number"] == 2800)
    assert failing["linter_ok"] is False
    assert any(f["code"] == "MISSING_DOCSTRING" for f in failing["linter_findings"])


def test_cli_markdown_output_includes_linter(capsys) -> None:
    """CLI default markdown must include linter status and findings."""
    candidates_path = FIXTURE_DIR / "candidates_linter_fail.json"
    exit_code = ranker.main(["--candidates-json", str(candidates_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "**Linter:** fail" in output
    assert "`MISSING_DOCSTRING`" in output
    assert "`UNUSED_IMPORT`" in output
