"""Tests for the superseded-draft scanner (issue #5393).

All tests use mocked GitHub CLI payloads — no network access required.
"""

# ruff: noqa: D101 — test classes do not need class-level docstrings

from __future__ import annotations

import json
import subprocess
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

    def test_invalid_timestamp_is_rejected_during_construction(self) -> None:
        with pytest.raises(ValueError, match="created_at"):
            _draft(created_at="not-a-timestamp")

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
            "fetch_merged_pr_inventory",
            lambda *, repo, max_pages=scanner.DEFAULT_MAX_PR_PAGES: (
                [],
                {"mode": "rest", "truncated": False},
            ),
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
            scanner,
            "fetch_merged_pr_inventory",
            lambda *, repo, max_pages=scanner.DEFAULT_MAX_PR_PAGES: (
                [],
                {"mode": "rest", "truncated": False},
            ),
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
            scanner,
            "fetch_merged_pr_inventory",
            lambda *, repo, max_pages=scanner.DEFAULT_MAX_PR_PAGES: (
                [],
                {"mode": "rest", "truncated": False},
            ),
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
        tmp_path,
    ) -> None:
        """Network errors should emit JSON with error field and exit 2."""

        def fake_fetch(*, repo: str, limit: int) -> tuple[list[scanner.DraftPr], bool]:
            raise RuntimeError("gh not available")

        monkeypatch.setattr(scanner, "fetch_draft_prs", fake_fetch)

        output_path = tmp_path / "nested" / "error-report.json"
        code = scanner.main(
            ["--repo", "ll7/robot_sf_ll7", "--output", str(output_path), "--markdown"]
        )
        assert code == 2
        out = capsys.readouterr()
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload.get("error")
        assert "Scanner failed" in out.err

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
# Search-throttle regression (issue #8927)
# ---------------------------------------------------------------------------


def _draft_api_row(number: int, body: str) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Draft PR #{number}",
        "body": body,
        "url": f"https://github.com/ll7/robot_sf_ll7/pull/{number}",
        "createdAt": _now_iso(),
        "updatedAt": _now_iso(),
    }


def _throttling_scanner_subprocess(
    draft_rows: list[dict[str, Any]],
    *,
    search_calls: list[list[str]],
) -> Any:
    """A scanner subprocess stand-in whose search endpoint always returns HTTP 403."""

    def fake_run(
        command: list[str],
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        **_: Any,
    ) -> subprocess.CompletedProcess[str]:
        if "search" in command:
            search_calls.append(command)
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=command,
                stderr="HTTP 403: API rate limit exceeded",
            )
        if command[:2] == ["gh", "pr"] and "list" in command:
            stdout = json.dumps(draft_rows)
        elif command[:2] == ["gh", "pr"] and "view" in command:
            stdout = json.dumps({"commits": []})
        elif command[:2] == ["gh", "issue"]:
            stdout = json.dumps({"state": "OPEN"})
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    return fake_run


def test_main_completes_from_rest_inventory_when_search_is_throttled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A 403 search endpoint cannot block rule 2 once it is REST-backed."""
    search_calls: list[list[str]] = []
    monkeypatch.setattr(
        scanner.subprocess,
        "run",
        _throttling_scanner_subprocess(
            [_draft_api_row(7, "Closes #42")],
            search_calls=search_calls,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "_gh_api_get",
        lambda path, **_: subprocess.CompletedProcess(
            ["gh", "api", path],
            0,
            stdout=json.dumps([_rest_pr_row(9000, body="Closes #42")]),
        ),
    )
    output_path = tmp_path / "search-throttled.json"

    code = scanner.main(["--repo", "ll7/robot_sf_ll7", "--output", str(output_path)])

    assert search_calls == []
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["quota_degraded"] is False
    assert payload["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["pr"]["number"] == 7
    assert "superseded_by_merged_pr" in candidate["rules"]
    assert any("9000" in line for line in candidate["evidence"])
    # The hard candidate still fails the run, but for the candidate reason, not quota.
    assert payload["ok"] is False
    assert payload["failure_summary"]["reason"] == "superseded_draft_candidates_found"
    assert code == 1


def test_main_reports_quota_degraded_when_rest_inventory_is_forbidden(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """A 403 REST inventory reports quota_degraded and exits nonzero, even with --check."""
    search_calls: list[list[str]] = []
    monkeypatch.setattr(
        scanner.subprocess,
        "run",
        _throttling_scanner_subprocess(
            [_draft_api_row(7, "Closes #42")],
            search_calls=search_calls,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "_gh_api_get",
        lambda path, **_: subprocess.CompletedProcess(
            ["gh", "api", path],
            1,
            stdout="",
            stderr="HTTP 403: API rate limit exceeded",
        ),
    )
    output_path = tmp_path / "degraded.json"

    code = scanner.main(["--check", "--repo", "ll7/robot_sf_ll7", "--output", str(output_path)])

    assert code == 1
    assert search_calls == []
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["quota_degraded"] is True
    assert "HTTP 403" in payload["degraded_reason"]
    assert payload["merged_pr_inventory"]["error"]
    assert payload["failure_summary"]["reason"] == "merged_pr_inventory_degraded"
    assert "cannot verify" in capsys.readouterr().err


def test_main_reports_quota_degraded_when_inventory_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Hitting the page budget marks coverage degraded instead of empty-clean."""
    full_page = [_rest_pr_row(number) for number in range(100)]
    monkeypatch.setattr(
        scanner,
        "fetch_draft_prs",
        lambda *, repo, limit: ([_draft(number=7, body="Closes #42")], False),
    )
    monkeypatch.setattr(scanner, "fetch_pr_files", lambda *, repo, pr_number: [])
    monkeypatch.setattr(scanner, "fetch_issue_state", lambda *, repo, number: "OPEN")
    monkeypatch.setattr(
        scanner,
        "get_modified_files_on_main_since",
        lambda *, repo, pr_number, branch="main": [],
    )
    monkeypatch.setattr(
        scanner,
        "_gh_api_get",
        lambda path, **_: subprocess.CompletedProcess(
            ["gh", "api", path], 0, stdout=json.dumps(full_page)
        ),
    )
    output_path = tmp_path / "truncated.json"

    code = scanner.main(
        ["--repo", "ll7/robot_sf_ll7", "--max-pr-pages", "1", "--output", str(output_path)]
    )

    assert code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["quota_degraded"] is True
    assert payload["merged_pr_inventory"]["truncated"] is True
    assert "truncated" in payload["degraded_reason"]


def test_report_and_markdown_expose_quota_degraded() -> None:
    """The degraded status is machine-readable and visible in the markdown summary."""
    report = scanner.build_report(
        repo="ll7/robot_sf_ll7",
        candidates=[],
        scanned_count=3,
        quota_degraded=True,
        degraded_reason="merged-PR REST inventory unavailable",
    )

    assert report["ok"] is False
    assert report["quota_degraded"] is True
    assert report["failure_summary"]["reason"] == "merged_pr_inventory_degraded"
    assert "degraded" in scanner.build_markdown(report).lower()


# ---------------------------------------------------------------------------
# REST-first merged-PR inventory and index
# ---------------------------------------------------------------------------


def _rest_pr_row(
    number: int,
    *,
    body: str = "",
    title: str = "",
    merged_at: str | None = "2026-01-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title or f"PR #{number}",
        "body": body,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/pull/{number}",
        "merged_at": merged_at,
    }


def test_build_merged_pr_index_matches_closes_patterns() -> None:
    """The in-process index finds 'Closes #N' matches (case-insensitive, word boundary)."""
    rows = [
        _rest_pr_row(10, body="Closes #42. Some text."),
        _rest_pr_row(11, body="Refs #42 no close."),
        _rest_pr_row(12, body="closes #42 and more."),
        _rest_pr_row(13, body="Closes #4200 not our issue."),
    ]

    index = scanner.build_merged_pr_index(rows, [42, 4200])

    assert [row["number"] for row in index[42]] == [10, 12]
    assert [row["number"] for row in index[4200]] == [13]
    assert 99 not in index


def test_build_merged_pr_index_rejects_malformed_closes_suffix() -> None:
    """A word suffix is not part of a valid ``Closes #N`` reference."""
    rows = [
        _rest_pr_row(10, body="Closes #42foo"),
        _rest_pr_row(11, body="Closes #42."),
        _rest_pr_row(12, body="closes #42"),
    ]

    index = scanner.build_merged_pr_index(rows, [42])

    assert [row["number"] for row in index[42]] == [11, 12]


def test_build_merged_pr_index_is_deterministically_ordered() -> None:
    """Index rows are ordered by PR number regardless of inventory order."""
    rows = [
        _rest_pr_row(30, body="Closes #7"),
        _rest_pr_row(10, body="Closes #7"),
        _rest_pr_row(20, body="Closes #7"),
    ]

    index = scanner.build_merged_pr_index(rows, [7])

    assert [row["number"] for row in index[7]] == [10, 20, 30]


def test_fetch_merged_pr_inventory_reads_rest_endpoint_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inventory is one bounded ``pulls?state=closed`` REST pass, never search."""
    paths: list[str] = []
    pages = [
        [
            _rest_pr_row(1, body="Closes #7"),
            _rest_pr_row(2, body="Closes #7", merged_at=None),
        ],
        [_rest_pr_row(3, body="Closes #7")],
    ]

    def fake_api_get(path: str, **_: Any) -> subprocess.CompletedProcess[str]:
        paths.append(path)
        payload = pages[len(paths) - 1] if len(paths) <= len(pages) else []
        return subprocess.CompletedProcess(["gh", "api", path], 0, stdout=json.dumps(payload))

    monkeypatch.setattr(scanner, "_gh_api_get", fake_api_get)

    merged_rows, metadata = scanner.fetch_merged_pr_inventory(
        repo="ll7/robot_sf_ll7",
        max_pages=5,
        per_page=2,
    )

    assert len(paths) == 2
    assert all("pulls?state=closed" in path for path in paths)
    assert all("search" not in path for path in paths)
    assert [row["number"] for row in merged_rows] == [1, 3]
    assert metadata["mode"] == "rest"
    assert metadata["closed_rows"] == 3
    assert metadata["merged_count"] == 2
    assert metadata["truncated"] is False


def test_fetch_merged_pr_inventory_flags_page_budget_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full page at the page budget marks the inventory as potentially truncated."""
    full_page = [_rest_pr_row(number) for number in range(100)]

    monkeypatch.setattr(
        scanner,
        "_gh_api_get",
        lambda path, **_: subprocess.CompletedProcess(
            ["gh", "api", path], 0, stdout=json.dumps(full_page)
        ),
    )

    merged_rows, metadata = scanner.fetch_merged_pr_inventory(
        repo="ll7/robot_sf_ll7",
        max_pages=2,
        per_page=100,
    )

    assert len(merged_rows) == 200
    assert metadata["truncated"] is True
    assert metadata["pages_read"] == 2


def test_fetch_merged_pr_inventory_fails_closed_on_rest_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REST 403 surfaces as a RuntimeError, never as an empty inventory."""
    monkeypatch.setattr(
        scanner,
        "_gh_api_get",
        lambda path, **_: subprocess.CompletedProcess(
            ["gh", "api", path], 1, stderr="HTTP 403: API rate limit exceeded"
        ),
    )

    with pytest.raises(RuntimeError, match="HTTP 403"):
        scanner.fetch_merged_pr_inventory(repo="ll7/robot_sf_ll7", max_pages=2)


def test_fetch_issue_state_does_not_hide_api_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An issue-state API failure must reach the scanner error report."""

    def failed_run_json(command: list[str], *, default: Any = None) -> Any:
        raise RuntimeError("GitHub unavailable")

    monkeypatch.setattr(scanner, "_run_json", failed_run_json)
    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        scanner.fetch_issue_state(repo="ll7/robot_sf_ll7", number=42)


def test_fetch_issue_state_accepts_merged_pull_request_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A merged pull-request reference is a valid supersession state."""
    monkeypatch.setattr(scanner, "_run_json", lambda *_args, **_kwargs: {"state": "MERGED"})

    assert scanner.fetch_issue_state(repo="ll7/robot_sf_ll7", number=8686) == "MERGED"


def test_fetch_issue_state_classifies_unknown_reference_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing issue/PR number is skippable, unlike an API failure."""

    def failed_run_json(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("GraphQL: Could not resolve to an Issue with the number of 999")

    monkeypatch.setattr(scanner, "_run_json", failed_run_json)
    with pytest.raises(scanner.ReferenceLookupError, match="#999"):
        scanner.fetch_issue_state(repo="ll7/robot_sf_ll7", number=999)


def test_unresolvable_reference_is_reported_without_stopping_other_drafts() -> None:
    """One malformed reference must not suppress candidates from later drafts."""
    prs = [
        _draft(number=1, body="Refs #999"),
        _draft(number=2, body="Closes #42"),
    ]
    warnings: list[dict[str, Any]] = []

    def get_issue_state(*, repo: str, number: int) -> str:
        if number == 999:
            raise scanner.ReferenceLookupError("reference does not resolve")
        return "CLOSED"

    candidates = scanner.scan_drafts(
        prs,
        repo="ll7/robot_sf_ll7",
        get_issue_state=get_issue_state,
        get_merged_prs=lambda *, repo, issue_number, limit=30: [],
        get_modified_files=lambda *, repo, pr_number: [],
        warnings=warnings,
    )

    assert [candidate.pr.number for candidate in candidates] == [2]
    assert warnings == [
        {
            "type": "skipped_reference",
            "draft_pr": 1,
            "reference": 999,
            "reason": "reference does not resolve",
        }
    ]


def test_report_and_markdown_include_reference_warnings() -> None:
    """Expected reference skips are visible without making the report fail."""
    warnings = [
        {
            "type": "skipped_reference",
            "draft_pr": 1,
            "reference": 999,
            "reason": "reference does not resolve",
        }
    ]
    report = scanner.build_report(
        repo="ll7/robot_sf_ll7",
        candidates=[],
        scanned_count=1,
        warnings=warnings,
    )

    assert report["ok"] is True
    assert report["warnings"] == warnings
    assert "reference #999 skipped" in scanner.build_markdown(report)


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

        commands: list[list[str]] = []

        def fake_run_json(cmd: list[str], *, default: Any = None) -> Any:
            commands.append(cmd)
            return mock_data

        monkeypatch.setattr(scanner, "_run_json", fake_run_json)

        prs, truncated = scanner.fetch_draft_prs(repo="ll7/robot_sf_ll7", limit=100)
        assert len(prs) == 1
        assert prs[0].number == 1
        assert not truncated
        assert commands[0][commands[0].index("--state") + 1] == "open"
        assert "--draft" in commands[0]

    def test_skips_rows_with_invalid_timestamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_data = [
            {
                "number": 1,
                "title": "Malformed draft",
                "body": "",
                "url": "https://example.com/pull/1",
                "createdAt": "bad",
                "updatedAt": _now_iso(),
            },
            {
                "number": 2,
                "title": "Valid draft",
                "body": "",
                "url": "https://example.com/pull/2",
                "createdAt": _ago(1),
                "updatedAt": _now_iso(),
            },
        ]

        monkeypatch.setattr(scanner, "_run_json", lambda *_args, **_kwargs: mock_data)
        prs, _truncated = scanner.fetch_draft_prs(repo="ll7/robot_sf_ll7", limit=100)

        assert [pr.number for pr in prs] == [2]

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
