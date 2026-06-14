"""Tests for publication candidate ranker helpers and fixture-backed rankings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert payload["schema"] == "publication_candidate_ranker.v2"
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


def test_ledger_boosts_candidate_score() -> None:
    """Candidate in ledger with claim_gap_closed should receive a score boost."""
    candidates_raw = _fixture("candidates_ledger_register.json")
    ledger_raw = _fixture("ledger_evidence.json")
    register_raw = _fixture("register_negative_results.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ledger = ranker._parse_ledger(ledger_raw)
    register = ranker._parse_register(register_raw)
    ranked = ranker.rank_candidates(candidates, ledger=ledger, register=register)

    ledger_candidate = next(r for r in ranked if r.issue_number == 2750)
    nonclaimable_candidate = next(r for r in ranked if r.issue_number == 2751)
    no_entry_candidate = next(r for r in ranked if r.issue_number == 2754)

    assert ledger_candidate.ledger_relevance == 1.0
    assert ledger_candidate.negative_result_awareness == 0.0
    assert any("open claim gap" in n for n in ledger_candidate.ledger_notes)
    assert any("promotion path" in n for n in ledger_candidate.ledger_notes)
    assert any("evidence tier diagnostic" in n for n in ledger_candidate.ledger_notes)
    assert ledger_candidate.negative_result_caveats == ()

    assert nonclaimable_candidate.ledger_relevance == 0.0
    assert any("evidence tier non-claimable" in n for n in nonclaimable_candidate.ledger_notes)

    assert no_entry_candidate.ledger_relevance == 0.0
    assert no_entry_candidate.negative_result_awareness == 0.0


def test_negative_result_penalizes_candidate() -> None:
    """Candidate in register with diagnostic_only status should be penalized."""
    candidates_raw = _fixture("candidates_ledger_register.json")
    ledger_raw = _fixture("ledger_evidence.json")
    register_raw = _fixture("register_negative_results.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ledger = ranker._parse_ledger(ledger_raw)
    register = ranker._parse_register(register_raw)
    ranked = ranker.rank_candidates(candidates, ledger=ledger, register=register)

    diag_candidate = next(r for r in ranked if r.issue_number == 2752)
    failed_candidate = next(r for r in ranked if r.issue_number == 2753)

    assert diag_candidate.negative_result_awareness == 1.0
    assert any("diagnostic_only" in c for c in diag_candidate.negative_result_caveats)
    assert any("why-this-is-different" in c for c in diag_candidate.negative_result_caveats)

    assert failed_candidate.negative_result_awareness == 1.0
    assert any("failed" in c for c in failed_candidate.negative_result_caveats)
    assert not any("why-this-is-different" in c for c in failed_candidate.negative_result_caveats)


def test_ranking_changes_with_ledger_register() -> None:
    """Ranking order should change when ledger/register signals are present."""
    candidates_raw = _fixture("candidates_ledger_register.json")
    assert isinstance(candidates_raw, list)

    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked_without = ranker.rank_candidates(candidates)
    ranked_with = ranker.rank_candidates(
        candidates,
        ledger=ranker._parse_ledger(_fixture("ledger_evidence.json")),
        register=ranker._parse_register(_fixture("register_negative_results.json")),
    )

    without_order = [r.issue_number for r in ranked_without]
    with_order = [r.issue_number for r in ranked_with]

    assert without_order != with_order

    with_scores = {r.issue_number: r.total_score for r in ranked_with}
    without_scores = {r.issue_number: r.total_score for r in ranked_without}

    assert with_scores[2750] > without_scores[2750]
    assert with_scores[2752] < without_scores[2752]
    assert with_scores[2753] < without_scores[2753]


def test_cli_with_ledger_and_register_json(capsys) -> None:
    """CLI with --ledger-json and --register-json should produce valid output."""
    candidates_path = FIXTURE_DIR / "candidates_ledger_register.json"
    ledger_path = FIXTURE_DIR / "ledger_evidence.json"
    register_path = FIXTURE_DIR / "register_negative_results.json"
    exit_code = ranker.main(
        [
            "--candidates-json",
            str(candidates_path),
            "--ledger-json",
            str(ledger_path),
            "--register-json",
            str(register_path),
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "publication_candidate_ranker.v2"
    assert payload["ranked_count"] == 5

    ledger_candidate = next(c for c in payload["ranked_candidates"] if c["issue_number"] == 2750)
    assert ledger_candidate["ledger_relevance"] == 1.0
    assert ledger_candidate["negative_result_awareness"] == 0.0
    assert any("claim gap" in n for n in ledger_candidate["ledger_notes"])

    diag_candidate = next(c for c in payload["ranked_candidates"] if c["issue_number"] == 2752)
    assert diag_candidate["negative_result_awareness"] == 1.0
    assert any("why-this-is-different" in c for c in diag_candidate["negative_result_caveats"])


def test_cli_rejects_wrong_ledger_schema(capsys, tmp_path) -> None:
    """Explicit ledger/register files should fail closed when schema versions drift."""
    candidates_path = FIXTURE_DIR / "candidates_ledger_register.json"
    ledger_path = tmp_path / "bad-ledger.json"
    ledger_path.write_text(
        json.dumps({"schema_version": "wrong", "rows": []}),
        encoding="utf-8",
    )

    exit_code = ranker.main(
        [
            "--candidates-json",
            str(candidates_path),
            "--ledger-json",
            str(ledger_path),
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["ok"] is False
    assert "schema_version" in payload["error"]


def test_cli_rejects_wrong_register_shape(capsys, tmp_path) -> None:
    """Explicit register files should fail closed when the entries shape is wrong."""
    candidates_path = FIXTURE_DIR / "candidates_ledger_register.json"
    register_path = tmp_path / "bad-register.json"
    register_path.write_text(
        json.dumps({"schema_version": "negative_result_register.v1", "entries": {}}),
        encoding="utf-8",
    )

    exit_code = ranker.main(
        [
            "--candidates-json",
            str(candidates_path),
            "--register-json",
            str(register_path),
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["ok"] is False
    assert "entries must be a list" in payload["error"]


def test_cli_without_ledger_register_backward_compat(capsys) -> None:
    """CLI without --ledger-json/--register-json should preserve v1-like behavior."""
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
    assert payload["schema"] == "publication_candidate_ranker.v2"
    for item in payload["ranked_candidates"]:
        assert item["ledger_relevance"] == 0.0
        assert item["negative_result_awareness"] == 0.0
        assert item["ledger_notes"] == []
        assert item["negative_result_caveats"] == []


def test_markdown_report_includes_ledger_and_negative_result() -> None:
    """Markdown report must include ledger notes and negative-result caveats."""
    candidates_raw = _fixture("candidates_ledger_register.json")
    candidates = [ranker._parse_candidate(item) for item in candidates_raw]
    ranked = ranker.rank_candidates(
        candidates,
        ledger=ranker._parse_ledger(_fixture("ledger_evidence.json")),
        register=ranker._parse_register(_fixture("register_negative_results.json")),
    )
    report = ranker.format_markdown_report(ranked, total_count=len(candidates))

    assert "**Ledger notes:**" in report
    assert "**Ledger relevance:**" in report
    assert "**Negative-result caveats:**" in report
    assert "**Negative-result awareness:**" in report
    assert "open claim gap" in report
    assert "why-this-is-different" in report


def test_parse_ledger_empty_inputs() -> None:
    """_parse_ledger should reject bad top-level schema and skip bad source_issues."""
    with pytest.raises(ranker.RankerInputError):
        ranker._parse_ledger({})
    with pytest.raises(ranker.RankerInputError):
        ranker._parse_ledger("not a dict")
    with pytest.raises(ranker.RankerInputError):
        ranker._parse_ledger(
            {"schema_version": "dissertation_evidence_ledger.v2", "rows": "not a list"}
        )
    assert (
        ranker._parse_ledger(
            {
                "schema_version": "dissertation_evidence_ledger.v2",
                "rows": [{"source_issues": "bad"}],
            }
        )
        == {}
    )


def test_parse_ledger_accepts_real_rows_schema() -> None:
    """_parse_ledger should match the current dissertation ledger rows schema."""
    payload = {
        "schema_version": "dissertation_evidence_ledger.v2",
        "rows": [
            {
                "area": "prediction",
                "source_issues": [2475],
                "claim_gap": "Needs denominator repair.",
                "evidence_promotion_path": "executed planner campaign",
                "evidence_tier": "diagnostic",
            }
        ],
    }
    parsed = ranker._parse_ledger(payload)
    assert 2475 in parsed
    assert parsed[2475][0]["area"] == "prediction"


def test_parse_ledger_skips_non_strict_issue_numbers() -> None:
    """Ledger parsing should skip bools and float issue identifiers."""
    row = {"source_issues": [True, False, 2475.5, float("nan"), float("inf"), 2476.0, 2477]}

    parsed = ranker._parse_ledger(
        {"schema_version": "dissertation_evidence_ledger.v2", "rows": [row]}
    )

    assert parsed == {2477: [row]}


def test_ledger_notes_ignore_null_promotion_path() -> None:
    """Null promotion paths should not render as notes or crash."""
    relevance, notes = ranker._ledger_notes_for_rows(
        [
            {
                "area": "observation_robustness",
                "claim_gap": "Needs perception proof.",
                "evidence_promotion_path": None,
                "evidence_tier": "release-backed",
            }
        ]
    )

    assert relevance == 1.0
    assert any("Needs perception proof" in note for note in notes)
    assert not any("promotion path" in note for note in notes)


def test_parse_register_empty_inputs() -> None:
    """_parse_register should reject bad top-level schema and skip bad linked_issues."""
    with pytest.raises(ranker.RankerInputError):
        ranker._parse_register({})
    with pytest.raises(ranker.RankerInputError):
        ranker._parse_register("not a dict")
    with pytest.raises(ranker.RankerInputError):
        ranker._parse_register(
            {"schema_version": "negative_result_register.v1", "entries": "not a list"}
        )
    assert (
        ranker._parse_register(
            {"schema_version": "negative_result_register.v1", "entries": [{"linked_issues": "bad"}]}
        )
        == {}
    )


def test_parse_register_accepts_real_result_classification_schema() -> None:
    """_parse_register should preserve current negative-result register entries."""
    payload = {
        "schema_version": "negative_result_register.v1",
        "entries": [
            {
                "linked_issues": [2716],
                "result_classification": "revise",
                "why_failed_or_inconclusive": "No hard slice cleared.",
                "recommended_next_action": "Change mechanism.",
                "claim_boundary": "Diagnostic-only.",
            }
        ],
    }
    parsed = ranker._parse_register(payload)
    assert 2716 in parsed
    assert parsed[2716][0]["result_classification"] == "revise"


def test_parse_register_skips_non_strict_issue_numbers() -> None:
    """Register parsing should skip bools and float issue identifiers."""
    entry = {"linked_issues": [True, False, 2716.2, float("nan"), float("inf"), 2717.0, 2718]}

    parsed = ranker._parse_register(
        {"schema_version": "negative_result_register.v1", "entries": [entry]}
    )

    assert parsed == {2718: [entry]}
