"""Tests for the superseded-draft scanner (issue #5393).

All tests use mocked GitHub CLI payloads — no network access required.
"""

# ruff: noqa: D101 — test classes do not need class-level docstrings

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from scripts.dev import scan_superseded_drafts as scanner

# ---------------------------------------------------------------------------
# Helpers to build fixtures
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ago(hours: int) -> str:
    dt = datetime.now(UTC) - timedelta(hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _draft(
    number: int = 1,
    body: str = "",
    files: list[str] | None = None,
    created_at: str | None = None,
) -> scanner.DraftPr:
    return scanner.DraftPr(
        number=number,
        title=f"Draft PR #{number}",
        body=body,
        url=f"https://github.com/ll7/robot_sf_ll7/pull/{number}",
        created_at=created_at or _now_iso(),
        updated_at=_now_iso(),
        files=files or [],
    )


# ---------------------------------------------------------------------------
# DraftPr unit tests
# ---------------------------------------------------------------------------


class TestDraftPr:
    def test_linked_issue_numbers_picks_closes(self) -> None:
        pr = _draft(body="This fixes the bug. Closes #42")
        assert pr.linked_issue_numbers() == [42]

    def test_linked_issue_numbers_picks_refs(self) -> None:
        pr = _draft(body="WIP progress on Refs #7, Refs #8")
        assert pr.linked_issue_numbers() == [7, 8]

    def test_linked_issue_numbers_picks_fixes(self) -> None:
        pr = _draft(body="Fixes #99")
        assert pr.linked_issue_numbers() == [99]

    def test_linked_issue_numbers_picks_raw_hash(self) -> None:
        pr = _draft(body="Working on #999")
        assert pr.linked_issue_numbers() == [999]

    def test_linked_issue_numbers_deduplicates(self) -> None:
        pr = _draft(body="Refs #1 Closes #1 Fixes #1")
        assert pr.linked_issue_numbers() == [1]

    def test_linked_issue_numbers_empty_when_no_refs(self) -> None:
        pr = _draft(body="No references here at all.")
        assert pr.linked_issue_numbers() == []

    def test_linked_issue_numbers_empty_body(self) -> None:
        pr = _draft(body="")
        assert pr.linked_issue_numbers() == []

    def test_age_returns_positive_timedelta(self) -> None:
        pr = _draft(created_at=_ago(5))
        assert pr.age > timedelta(hours=4)
        assert pr.age < timedelta(hours=6)

    def test_to_payload_is_json_serializable(self) -> None:
        pr = _draft(body="Closes #12", files=["a.py"])
        payload = pr.to_payload()
        # Must round-trip without error
        json.dumps(payload)
        assert payload["number"] == 1
        assert payload["linked_issues"] == [12]

    def test_to_payload_file_count(self) -> None:
        pr = _draft(files=["a.py", "b.py"])
        assert pr.to_payload()["file_count"] == 2


# ---------------------------------------------------------------------------
# Rule 1: linked issue closed
# ---------------------------------------------------------------------------


class TestRule1LinkedIssueClosed:
    def test_triggers_when_linked_issue_is_closed(self) -> None:
        pr = _draft(body="Closes #42")
        rules, evidence = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "CLOSED" if number == 42 else "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert "linked_issue_closed" in rules
        assert any("42" in e for e in evidence)

    def test_does_not_trigger_when_linked_issue_open(self) -> None:
        pr = _draft(body="Closes #42")
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert "linked_issue_closed" not in rules

    def test_checks_all_linked_issues(self) -> None:
        pr = _draft(body="Refs #1 Closes #2")
        closed_set: set[int] = {2}
        rules, evidence = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "CLOSED" if number in closed_set else "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert rules.count("linked_issue_closed") == 1
        assert any("2" in e for e in evidence)

    def test_does_not_fire_on_no_linked_issue(self) -> None:
        pr = _draft(body="No issue references")
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "CLOSED",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert "linked_issue_closed" not in rules


# ---------------------------------------------------------------------------
# Rule 2: superseded by merged PR
# ---------------------------------------------------------------------------


class TestRule2SupersededByMergedPr:
    def test_triggers_when_merged_pr_closes_same_issue(self) -> None:
        pr = _draft(body="Closes #42")
        rules, evidence = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: (
                [
                    {
                        "number": 99,
                        "title": "Fixed #42",
                        "url": "https://github.com/ll7/robot_sf_ll7/pull/99",
                    }
                ]
                if issue_number == 42
                else []
            ),
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert "superseded_by_merged_pr" in rules
        assert any("99" in e for e in evidence)

    def test_does_not_trigger_when_no_merged_pr(self) -> None:
        pr = _draft(body="Closes #42")
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert "superseded_by_merged_pr" not in rules

    def test_merged_pr_for_different_issue_ignored(self) -> None:
        pr = _draft(body="Closes #42")
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            # Merged PR exists for #99, not #42
            get_merged_prs=lambda *, repo, issue_number, limit=30: (
                [{"number": 99, "title": "X", "url": "X"}] if issue_number == 99 else []
            ),
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert "superseded_by_merged_pr" not in rules


# ---------------------------------------------------------------------------
# Rule 3: stale + superseded files (weak)
# ---------------------------------------------------------------------------


class TestRule3StaleFiles:
    def test_triggers_when_old_and_all_files_modified(self) -> None:
        pr = _draft(
            body="Closes #42",
            files=["a.py", "b.py"],
            created_at=_ago(72),  # 72h old > 48h
        )
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: ["a.py", "b.py", "c.py"],
        )
        assert "stale_all_files_modified_on_main" in rules

    def test_does_not_trigger_when_young(self) -> None:
        pr = _draft(
            body="Closes #42",
            files=["a.py"],
            created_at=_ago(1),  # 1h old
        )
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: ["a.py"],
        )
        assert "stale_all_files_modified_on_main" not in rules

    def test_does_not_trigger_when_not_all_files_modified(self) -> None:
        pr = _draft(
            body="Closes #42",
            files=["a.py", "b.py"],
            created_at=_ago(72),
        )
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: ["a.py"],  # only a.py
        )
        assert "stale_all_files_modified_on_main" not in rules

    def test_no_files_in_draft_means_no_rule3(self) -> None:
        pr = _draft(body="Closes #42", files=[], created_at=_ago(100))
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: ["a.py"],
        )
        assert "stale_all_files_modified_on_main" not in rules

    def test_empty_modified_files_means_no_rule3(self) -> None:
        pr = _draft(body="Closes #42", files=["a.py"], created_at=_ago(100))
        rules, _ = scanner.evaluate_rules(
            pr,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert "stale_all_files_modified_on_main" not in rules


# ---------------------------------------------------------------------------
# SupersededCandidate
# ---------------------------------------------------------------------------


class TestSupersededCandidate:
    def test_hard_rule_classification(self) -> None:
        c = scanner.SupersededCandidate(pr=_draft(), rules=["linked_issue_closed"], evidence=["e"])
        assert c.has_hard_rule

    def test_hard_rule_false_for_weak_only(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(),
            rules=["stale_all_files_modified_on_main"],
            evidence=["e"],
        )
        assert not c.has_hard_rule

    def test_to_payload_contains_evidence(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(number=5),
            rules=["linked_issue_closed"],
            evidence=["ev1", "ev2"],
        )
        p = c.to_payload()
        assert p["pr"]["number"] == 5
        assert p["rules"] == ["linked_issue_closed"]
        assert p["evidence"] == ["ev1", "ev2"]
        assert p["hard"] is True

    def test_json_roundtrip(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(number=1, body="Refs #9"),
            rules=["r"],
            evidence=["e"],
        )
        json.dumps(c.to_payload())


# ---------------------------------------------------------------------------
# scan_drafts integration
# ---------------------------------------------------------------------------


class TestScanDrafts:
    def test_returns_candidates_for_superseded_draft(self) -> None:
        prs = [_draft(number=1, body="Closes #42")]
        candidates = scanner.scan_drafts(
            prs,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "CLOSED" if number == 42 else "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert len(candidates) == 1
        assert candidates[0].pr.number == 1
        assert candidates[0].has_hard_rule

    def test_returns_no_candidates_when_none_superseded(self) -> None:
        prs = [_draft(number=1, body="Closes #42")]
        candidates = scanner.scan_drafts(
            prs,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert len(candidates) == 0

    def test_scans_multiple_drafts_independently(self) -> None:
        prs = [
            _draft(number=1, body="Closes #42"),
            _draft(number=2, body="Closes #43"),
        ]
        candidates = scanner.scan_drafts(
            prs,
            repo="ll7/robot_sf_ll7",
            get_issue_state=lambda *, repo, number: "CLOSED" if number == 42 else "OPEN",
            get_merged_prs=lambda *, repo, issue_number, limit=30: [],
            get_modified_files=lambda *, repo, pr_number: [],
        )
        assert len(candidates) == 1
        assert candidates[0].pr.number == 1


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_ok_when_no_hard_candidates(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(),
            rules=["stale_all_files_modified_on_main"],
            evidence=["e"],
        )
        report = scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[c],
            scanned_count=5,
        )
        assert report["ok"] is True  # weak rule only
        assert report["hard_candidate_count"] == 0

    def test_not_ok_when_hard_candidate(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(number=1, body="Closes #1"),
            rules=["linked_issue_closed"],
            evidence=["Rule 1: linked issue #1 is CLOSED"],
        )
        report = scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[c],
            scanned_count=3,
        )
        assert report["ok"] is False
        assert report["hard_candidate_count"] == 1
        assert report["failure_summary"] is not None

    def test_schema_and_read_only(self) -> None:
        report = scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[],
            scanned_count=0,
        )
        assert report["schema"] == "superseded_draft_scanner.v1"
        assert report["read_only"] is True

    def test_json_roundtrip(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(number=1, body="Refs #5"),
            rules=["r"],
            evidence=["e"],
        )
        report = scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[c],
            scanned_count=1,
        )
        json.dumps(report)


class TestBuildMarkdown:
    def test_includes_candidate_details(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(number=42, body="Closes #9"),
            rules=["linked_issue_closed"],
            evidence=["Rule 1: linked issue #9 is CLOSED"],
        )
        report = scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[c],
            scanned_count=42,
        )
        md = scanner.build_markdown(report)
        assert "#42" in md
        assert "ll7/robot_sf_ll7" in md
        assert "Rule 1: linked issue #9 is CLOSED" in md

    def test_no_candidates_message(self) -> None:
        report = scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[],
            scanned_count=5,
        )
        md = scanner.build_markdown(report)
        assert "No superseded draft candidates found" in md

    def test_hard_tag_present(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(number=1, body="Closes #1"),
            rules=["linked_issue_closed"],
            evidence=["e"],
        )
        report = scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[c],
            scanned_count=1,
        )
        md = scanner.build_markdown(report)
        assert "[HARD]" in md

    def test_weak_candidate_no_hard_tag(self) -> None:
        c = scanner.SupersededCandidate(
            pr=_draft(number=1, body="refs #1"),
            rules=["stale_all_files_modified_on_main"],
            evidence=["e"],
        )
        report = scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[c],
            scanned_count=1,
        )
        md = scanner.build_markdown(report)
        assert "[HARD]" not in md


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_check_exits_1_on_hard_candidate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--check must return 1 when hard candidates exist."""
        pr = _draft(number=1, body="Closes #42")

        def fake_fetch_drafts(*, repo: str, limit: int) -> tuple[list[scanner.DraftPr], bool]:
            return [pr], False

        def fake_pr_files(*, repo: str, pr_number: int) -> list[str]:
            return []

        monkeypatch.setattr(scanner, "fetch_draft_prs", fake_fetch_drafts)
        monkeypatch.setattr(scanner, "fetch_pr_files", fake_pr_files)
        monkeypatch.setattr(
            scanner,
            "fetch_issue_state",
            lambda *, repo, number: "CLOSED" if number == 42 else "OPEN",
        )
        monkeypatch.setattr(
            scanner,
            "fetch_merged_prs_for_issue",
            lambda *, repo, issue_number, limit=30: [],
        )
        monkeypatch.setattr(
            scanner,
            "get_modified_files_on_main_since",
            lambda *, repo, pr_number, branch="main": [],
        )

        code = scanner.main(["--check", "--repo", "ll7/robot_sf_ll7"])
        assert code == 1
        out = capsys.readouterr()
        assert "FAIL" in out.err

    def test_check_returns_0_when_clean(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--check returns 0 when no hard candidates."""
        pr = _draft(number=1, body="Closes #42")

        def fake_fetch_drafts(*, repo: str, limit: int) -> tuple[list[scanner.DraftPr], bool]:
            return [pr], False

        monkeypatch.setattr(scanner, "fetch_draft_prs", fake_fetch_drafts)
        monkeypatch.setattr(scanner, "fetch_pr_files", lambda *, repo, pr_number: [])
        monkeypatch.setattr(scanner, "fetch_issue_state", lambda *, repo, number: "OPEN")
        monkeypatch.setattr(
            scanner, "fetch_merged_prs_for_issue", lambda *, repo, issue_number, limit=30: []
        )
        monkeypatch.setattr(
            scanner,
            "get_modified_files_on_main_since",
            lambda *, repo, pr_number, branch="main": [],
        )

        code = scanner.main(["--check", "--repo", "ll7/robot_sf_ll7"])
        assert code == 0

    def test_no_check_returns_0_even_with_weak_candidate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without --check, weak candidates should not change exit from 0."""
        pr = _draft(
            number=1,
            body="Closes #42",
            files=["a.py"],
            created_at=_ago(72),
        )

        def fake_fetch_drafts(*, repo: str, limit: int) -> tuple[list[scanner.DraftPr], bool]:
            return [pr], False

        monkeypatch.setattr(scanner, "fetch_draft_prs", fake_fetch_drafts)
        monkeypatch.setattr(scanner, "fetch_pr_files", lambda *, repo, pr_number: ["a.py"])
        monkeypatch.setattr(scanner, "fetch_issue_state", lambda *, repo, number: "OPEN")
        monkeypatch.setattr(
            scanner, "fetch_merged_prs_for_issue", lambda *, repo, issue_number, limit=30: []
        )
        monkeypatch.setattr(
            scanner,
            "get_modified_files_on_main_since",
            lambda *, repo, pr_number, branch="main": ["a.py"],
        )

        code = scanner.main(["--repo", "ll7/robot_sf_ll7"])
        assert code == 0

    def test_emits_json_on_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Default mode emits JSON to stdout."""

        def fake_fetch_drafts(*, repo: str, limit: int) -> tuple[list[scanner.DraftPr], bool]:
            return [], False

        monkeypatch.setattr(scanner, "fetch_draft_prs", fake_fetch_drafts)

        code = scanner.main(["--repo", "ll7/robot_sf_ll7"])
        assert code == 0
        out = capsys.readouterr()
        payload = json.loads(out.out)
        assert payload["schema"] == "superseded_draft_scanner.v1"

    def test_error_returns_exit_2_with_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Network errors should emit JSON with error field and exit 2."""

        def fake_fetch(*, repo: str, limit: int) -> tuple[list[scanner.DraftPr], bool]:
            raise RuntimeError("gh not available")

        monkeypatch.setattr(scanner, "fetch_draft_prs", fake_fetch)

        code = scanner.main(["--repo", "ll7/robot_sf_ll7"])
        assert code == 2
        out = capsys.readouterr()
        payload = json.loads(out.out)
        assert payload.get("error")

    def test_markdown_flag_emits_to_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--markdown should emit markdown to stderr alongside JSON."""

        def fake_fetch_drafts(*, repo: str, limit: int) -> tuple[list[scanner.DraftPr], bool]:
            return [], False

        monkeypatch.setattr(scanner, "fetch_draft_prs", fake_fetch_drafts)

        scanner.main(["--repo", "ll7/robot_sf_ll7", "--markdown"])
        out, err = capsys.readouterr()
        json.loads(out)  # stdout is valid JSON
        assert "Superseded Draft PR Scan" in err


# ---------------------------------------------------------------------------
# fetch_merged_prs_for_issue pattern matching
# ---------------------------------------------------------------------------


def test_fetch_merged_prs_pattern_matches_correctly() -> None:
    """The pattern matching finds 'Closes #N' in PR body (case-insensitive, word boundary)."""
    raw = [
        {
            "number": 10,
            "title": "PR A",
            "body": "Closes #42. Some text.",
            "url": "https://example.com/pull/10",
        },
        {
            "number": 11,
            "title": "PR B",
            "body": "Refs #42 no close.",
            "url": "https://example.com/pull/11",
        },
        {
            "number": 12,
            "title": "PR C",
            "body": "closes #42 and more.",
            "url": "https://example.com/pull/12",
        },
        {
            "number": 13,
            "title": "PR D",
            "body": "Closes #4200 not our issue.",
            "url": "https://example.com/pull/13",
        },
    ]
    import re

    pattern = re.compile(r"Closes\s+#42\b", re.IGNORECASE)
    matches = [r for r in raw if pattern.search(str(r.get("body", "")))]
    # PR 10 and PR 12 match; PR 11 (Refs) and PR 13 (#4200) do not
    assert len(matches) == 2
    assert matches[0]["number"] == 10
    assert matches[1]["number"] == 12


# ---------------------------------------------------------------------------
# fetch_draft_prs
# ---------------------------------------------------------------------------


class TestFetchDraftPrs:
    def test_parses_valid_draft_pr_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fetch_draft_prs parses gh JSON output into DraftPr objects."""
        mock_data = [
            {
                "number": 1,
                "title": "Draft PR",
                "body": "Closes #42",
                "url": "https://github.com/ll7/robot_sf_ll7/pull/1",
                "createdAt": _ago(10),
                "updatedAt": _now_iso(),
            }
        ]

        def fake_run_json(cmd: list[str], *, default: Any = None) -> Any:
            return mock_data

        monkeypatch.setattr(scanner, "_run_json", fake_run_json)

        prs, truncated = scanner.fetch_draft_prs(repo="ll7/robot_sf_ll7", limit=100)
        assert len(prs) == 1
        assert prs[0].number == 1
        assert not truncated

    def test_detects_truncation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fetch_draft_prs should detect when result hits the limit."""
        mock_data = [
            {
                "number": i,
                "title": f"Draft {i}",
                "body": "",
                "url": f"https://github.com/ll7/robot_sf_ll7/pull/{i}",
                "createdAt": _ago(i),
                "updatedAt": _now_iso(),
            }
            for i in range(50)
        ]

        def fake_run_json(cmd: list[str], *, default: Any = None) -> Any:
            return mock_data

        monkeypatch.setattr(scanner, "_run_json", fake_run_json)

        _, truncated = scanner.fetch_draft_prs(
            repo="ll7/robot_sf_ll7",
            limit=50,
        )
        assert truncated is True


# ---------------------------------------------------------------------------
# _run_json error handling
# ---------------------------------------------------------------------------


class TestRunJson:
    def test_missing_gh_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("gh")

        monkeypatch.setattr(scanner.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="GitHub CLI 'gh' was not found"):
            scanner._run_json(["gh", "pr", "list"])

    def test_captured_process_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: object, **kwargs: object) -> None:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=("gh", "pr", "list"),
                stderr="authentication failed",
            )

        import subprocess

        monkeypatch.setattr(scanner.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="authentication failed"):
            scanner._run_json(["gh", "pr", "list"])


# ---------------------------------------------------------------------------
# build_report truncated marking
# ---------------------------------------------------------------------------


def test_report_truncated_marker() -> None:
    """The report should carry the truncation warning from fetch."""
    report = scanner.build_report(
        repo="ll7/robot_sf_ll7",
        candidates=[],
        scanned_count=100,
        truncated=True,
    )
    assert report["truncated"] is True


def test_markdown_includes_truncation_warning() -> None:
    """Markdown report should mention truncation when gh search was capped."""
    md = scanner.build_markdown(
        scanner.build_report(
            repo="ll7/robot_sf_ll7",
            candidates=[],
            scanned_count=100,
            truncated=True,
        )
    )
    assert "truncated" in md.lower()
