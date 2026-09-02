"""Tests for closed-issue state-label hygiene helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.dev import closed_state_label_hygiene


def _mock_current_rest_issue(
    monkeypatch: pytest.MonkeyPatch,
    *,
    labels: list[str],
    title: str = "current issue",
    number: int = 12,
) -> None:
    """Install a current closed-issue REST response for reconciliation tests."""

    monkeypatch.setattr(
        "scripts.dev.gh_issue_rest.fetch_issue",
        lambda requested_number, **kwargs: {
            "number": requested_number,
            "status": "ok",
            "title": title,
            "url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
            "state": "CLOSED",
            "is_pull_request": False,
            "labels": labels,
        },
    )


def test_collect_stale_issues_aggregates_closed_issue_state_labels() -> None:
    """Closed issues should be reported once with all stale live state labels."""
    rows_by_label = {
        "state:ready": [
            {
                "number": 12,
                "title": "done but still queued",
                "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                "state": "closed",
                "is_pull_request": False,
                "labels": [{"name": "state:ready"}, {"name": "workflow"}],
            },
            {
                "number": 13,
                "title": "open issue should not count",
                "url": "https://github.com/ll7/robot_sf_ll7/issues/13",
                "state": "open",
                "is_pull_request": False,
                "labels": [{"name": "state:ready"}],
            },
        ],
        "state:blocked": [
            {
                "number": 12,
                "title": "done but still queued",
                "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                "state": "closed",
                "is_pull_request": False,
                "labels": [{"name": "state:ready"}, {"name": "state:blocked"}],
            }
        ],
    }

    stale = closed_state_label_hygiene.collect_stale_issues(rows_by_label)

    assert [issue.number for issue in stale] == [12]
    assert stale[0].stale_labels == ("state:blocked", "state:ready")


def test_collect_stale_issues_ignores_pull_request_rows() -> None:
    """The guard is issue-specific even if a caller supplies PR-shaped search rows."""
    rows_by_label = {
        "state:ready": [
            {
                "number": 12,
                "title": "closed PR with a state label",
                "url": "https://github.com/ll7/robot_sf_ll7/pull/12",
                "state": "closed",
                "is_pull_request": True,
                "labels": [{"name": "state:ready"}],
            }
        ],
    }

    assert closed_state_label_hygiene.collect_stale_issues(rows_by_label) == []


def test_reconcile_stale_issues_suppresses_search_index_lag() -> None:
    """A removed REST label should override the stale label still returned by search."""
    candidates = [_stale(12, ("state:ready",))]

    stale = closed_state_label_hygiene.reconcile_stale_issues(
        repo="ll7/robot_sf_ll7",
        candidates=candidates,
        fetch_issue=lambda number, **kwargs: {
            "number": number,
            "status": "ok",
            "title": "done and cleaned up",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
            "state": "CLOSED",
            "is_pull_request": False,
            "labels": ["priority:4", "technical-debt"],
        },
    )

    assert stale == []


def test_reconcile_stale_issues_preserves_current_rest_live_labels() -> None:
    """A genuinely stale live label in the current REST record remains a failure."""
    candidates = [_stale(12, ("state:ready", "state:running"))]

    stale = closed_state_label_hygiene.reconcile_stale_issues(
        repo="ll7/robot_sf_ll7",
        candidates=candidates,
        fetch_issue=lambda number, **kwargs: {
            "number": number,
            "status": "ok",
            "title": "current REST title",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
            "state": "CLOSED",
            "is_pull_request": False,
            "labels": ["priority:4", "state:running"],
        },
    )

    assert len(stale) == 1
    assert stale[0].title == "current REST title"
    assert stale[0].stale_labels == ("state:running",)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/ll7/robot_sf_ll7/pull/12", True),
        ("https://github.com/ll7/robot_sf_ll7/issues/12?next=/pull/12", False),
        ("https://github.com/ll7/robot_sf_ll7/issues/pull", False),
        ("https://github.com/ll7/robot_sf_ll7/pull/not-a-number", False),
        ("https://github.com/ll7/robot_sf_ll7/pulls/12", False),
        ("http://github.com/ll7/robot_sf_ll7/pull/12", False),
        ("https://evil.example/ll7/robot_sf_ll7/pull/12", False),
        ("https://github.com:443/ll7/robot_sf_ll7/pull/12", False),
        ("https://user@github.com/ll7/robot_sf_ll7/pull/12", False),
        ("https://user:pass@github.com/ll7/robot_sf_ll7/pull/12", False),
        (None, False),
    ],
)
def test_is_pull_request_url_requires_canonical_pull_path(url: object, expected: bool) -> None:
    """PR detection should not treat arbitrary '/pull/' substrings as pull requests."""
    assert closed_state_label_hygiene._is_pull_request_url(url) is expected


def test_build_report_emits_machine_readable_failure_summary() -> None:
    """Reports should expose a stable failure summary when stale labels exist."""
    stale = [
        closed_state_label_hygiene.StaleIssue(
            number=12,
            title="done but still queued",
            url="https://github.com/ll7/robot_sf_ll7/issues/12",
            state="closed",
            stale_labels=("state:ready",),
        )
    ]

    report = closed_state_label_hygiene.build_report(
        repo="ll7/robot_sf_ll7",
        checked_labels=("state:ready", "state:running", "state:blocked"),
        stale_issues=stale,
    )

    assert report["schema"] == "closed_state_label_hygiene.v1"
    assert report["ok"] is False
    assert report["read_only"] is True
    assert report["project_writes"] is False
    assert report["stale_count"] == 1
    assert report["issues"][0]["stale_labels"] == ["state:ready"]


def test_build_search_command_uses_read_only_closed_issue_search() -> None:
    """The GitHub command should only search issues and avoid Project writes."""
    command = closed_state_label_hygiene.build_search_command(
        repo="ll7/robot_sf_ll7",
        label="state:ready",
        limit=200,
    )

    assert command[:3] == ["gh", "search", "issues"]
    assert "--state" in command
    assert command[command.index("--state") + 1] == "closed"
    assert "--label" in command
    assert command[command.index("--label") + 1] == "state:ready"
    assert "url" in command[command.index("--json") + 1].split(",")
    assert "isPullRequest" in command[command.index("--json") + 1].split(",")
    assert "--project" not in command
    assert "edit" not in command


def test_closure_workflow_routes_state_label_io_through_rest_helpers() -> None:
    """The closure Action must not use native issue label commands."""
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "strip-closed-state-labels.yml"
    ).read_text(encoding="utf-8")
    code = "\n".join(line for line in workflow.splitlines() if not line.lstrip().startswith("#"))

    assert "python -m scripts.dev.gh_issue_rest view" in code
    assert "python -m scripts.dev.gh_pr_label_rest list" in code
    assert "python -m scripts.dev.gh_pr_label_rest remove" in code
    assert "if ! live_labels_output=$(" in code
    assert "mapfile -t LIVE_LABELS < <(" not in code
    assert "gh issue view" not in code
    assert "gh issue edit" not in code


def test_closure_workflow_allowlist_failure_stops_before_cleanup() -> None:
    """The exact workflow guard must propagate a failing allowlist producer."""
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "strip-closed-state-labels.yml"
    ).read_text(encoding="utf-8")
    start = workflow.index("          if ! live_labels_output=$(")
    end = workflow.index('          echo "Live state labels:', start)
    guard = textwrap.dedent(workflow[start:end])
    producer_start = guard.index("python -c '")
    producer_end = guard.index("\n'", producer_start) + len("\n'")
    failing_guard = (
        guard[:producer_start] + "python -c 'raise SystemExit(7)'" + guard[producer_end:]
    )

    probe = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{failing_guard}\necho sentinel"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 1
    assert "sentinel" not in probe.stdout


def _closure_workflow_script() -> str:
    """Extract the checked-in Action shell block for an offline execution test."""
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "strip-closed-state-labels.yml"
    ).read_text(encoding="utf-8")
    start = workflow.index("        run: |\n") + len("        run: |\n")
    return textwrap.dedent(workflow[start:])


def _run_closure_workflow(
    tmp_path: Path,
    *,
    issue_payload: dict[str, object],
    list_stdout: str,
    remove_stdout: str,
    remove_exit: int = 0,
    environment_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    """Run the real Action shell block with deterministic helper subprocesses."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-c" ]; then
  case "${2:-}" in
    *LIVE_STATE_LABELS*)
      printf '%s\\n' "${FAKE_ALLOWLIST_OUTPUT}"
      exit "${FAKE_ALLOWLIST_EXIT}"
      ;;
  esac
fi

if [ "${1:-}" = "-m" ] && [ "${2:-}" = "scripts.dev.gh_issue_rest" ] && [ "${3:-}" = "view" ]; then
  : > "${FAKE_VIEW_MARKER}"
  cat "${FAKE_ISSUE_PAYLOAD}"
  exit "${FAKE_VIEW_EXIT}"
fi

if [ "${1:-}" = "-m" ] && [ "${2:-}" = "scripts.dev.gh_pr_label_rest" ] && [ "${3:-}" = "list" ]; then
  : > "${FAKE_LIST_MARKER}"
  cat "${FAKE_LIST_PAYLOAD}"
  exit "${FAKE_LIST_EXIT}"
fi

if [ "${1:-}" = "-m" ] && [ "${2:-}" = "scripts.dev.gh_pr_label_rest" ] && [ "${3:-}" = "remove" ]; then
  printf '%s\\n' "$*" >> "${FAKE_REMOVE_LOG}"
  cat "${FAKE_REMOVE_PAYLOAD}"
  exit "${FAKE_REMOVE_EXIT}"
fi

exec "${REAL_PYTHON}" "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    issue_path = tmp_path / "issue.json"
    list_path = tmp_path / "list.json"
    remove_path = tmp_path / "remove.json"
    issue_path.write_text(json.dumps(issue_payload), encoding="utf-8")
    list_path.write_text(list_stdout, encoding="utf-8")
    remove_path.write_text(remove_stdout, encoding="utf-8")
    view_marker = tmp_path / "view-called"
    list_marker = tmp_path / "list-called"
    remove_log = tmp_path / "remove.log"
    remove_log.write_text("", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment.get('PATH', '')}",
            "PYTHONPATH": f"{repo_root}{os.pathsep}{environment.get('PYTHONPATH', '')}",
            "REAL_PYTHON": sys.executable,
            "REPO": "ll7/robot_sf_ll7",
            "ISSUE_NUMBER": "12",
            "FAKE_ALLOWLIST_OUTPUT": "state:ready\nstate:running\nstate:blocked",
            "FAKE_ALLOWLIST_EXIT": "0",
            "FAKE_VIEW_EXIT": "0",
            "FAKE_LIST_EXIT": "0",
            "FAKE_REMOVE_EXIT": str(remove_exit),
            "FAKE_ISSUE_PAYLOAD": str(issue_path),
            "FAKE_LIST_PAYLOAD": str(list_path),
            "FAKE_REMOVE_PAYLOAD": str(remove_path),
            "FAKE_VIEW_MARKER": str(view_marker),
            "FAKE_LIST_MARKER": str(list_marker),
            "FAKE_REMOVE_LOG": str(remove_log),
        }
    )
    if environment_overrides:
        environment.update(environment_overrides)
    probe = subprocess.run(
        ["bash", "-c", _closure_workflow_script()],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return probe, view_marker, list_marker, remove_log


def _workflow_issue(
    *,
    status: str = "ok",
    state: str = "CLOSED",
    url: str = "https://github.com/ll7/robot_sf_ll7/issues/12",
    is_pull_request: bool = False,
) -> dict[str, object]:
    """Build the normalized issue payload emitted by the REST view helper."""
    return {
        "status": status,
        "number": 12,
        "state": state,
        "url": url,
        "is_pull_request": is_pull_request,
    }


def _workflow_list(*labels: str) -> str:
    """Build a successful label-list envelope for the Action subprocess stub."""
    return json.dumps(
        {
            "status": "ok",
            "action": "list",
            "number": 12,
            "repo": "ll7/robot_sf_ll7",
            "labels": list(labels),
        }
    )


def _workflow_remove(label: str) -> str:
    """Build a successful label-removal envelope for the Action subprocess stub."""
    return json.dumps(
        {
            "status": "ok",
            "action": "remove",
            "number": 12,
            "repo": "ll7/robot_sf_ll7",
            "label": label,
        }
    )


def test_closure_workflow_removes_only_present_label_after_validating_results(
    tmp_path: Path,
) -> None:
    """A valid closed issue uses the checked helper envelopes and removes one label."""
    probe, view_marker, list_marker, remove_log = _run_closure_workflow(
        tmp_path,
        issue_payload=_workflow_issue(),
        list_stdout=_workflow_list("state:ready", "bug"),
        remove_stdout=_workflow_remove("state:ready"),
    )

    assert probe.returncode == 0, probe.stderr
    assert view_marker.exists()
    assert list_marker.exists()
    assert remove_log.read_text(encoding="utf-8").count("state:ready") == 1


def test_closure_workflow_closed_issue_without_live_labels_is_a_no_op(tmp_path: Path) -> None:
    """A valid closed issue with no live labels must not invoke the remove helper."""
    probe, _, list_marker, remove_log = _run_closure_workflow(
        tmp_path,
        issue_payload=_workflow_issue(),
        list_stdout=_workflow_list("bug"),
        remove_stdout="",
    )

    assert probe.returncode == 0, probe.stderr
    assert list_marker.exists()
    assert remove_log.read_text(encoding="utf-8") == ""


def test_closure_workflow_rejects_malformed_allowlist_output(tmp_path: Path) -> None:
    """A malformed source-of-truth allowlist cannot turn cleanup into a no-op."""
    probe, _, _, remove_log = _run_closure_workflow(
        tmp_path,
        issue_payload=_workflow_issue(),
        list_stdout=_workflow_list("state:ready"),
        remove_stdout=_workflow_remove("state:ready"),
        environment_overrides={"FAKE_ALLOWLIST_OUTPUT": "state:ready\nstate:ready"},
    )

    assert probe.returncode != 0
    assert remove_log.read_text(encoding="utf-8") == ""


def test_closure_workflow_open_issue_is_a_no_op(tmp_path: Path) -> None:
    """A valid open issue is skipped before label inventory or writes."""
    probe, _, list_marker, remove_log = _run_closure_workflow(
        tmp_path,
        issue_payload=_workflow_issue(state="OPEN"),
        list_stdout=_workflow_list("state:ready"),
        remove_stdout=_workflow_remove("state:ready"),
    )

    assert probe.returncode == 0, probe.stderr
    assert not list_marker.exists()
    assert remove_log.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "issue_payload",
    [
        _workflow_issue(status="error"),
        _workflow_issue(state="BANANA"),
        _workflow_issue(url="not-a-url"),
        _workflow_issue(is_pull_request=True),
    ],
)
def test_closure_workflow_rejects_unknown_or_inconsistent_issue_identity(
    tmp_path: Path,
    issue_payload: dict[str, object],
) -> None:
    """Malformed state, URL, or PR marker must stop before label discovery or writes."""
    probe, _, list_marker, remove_log = _run_closure_workflow(
        tmp_path,
        issue_payload=issue_payload,
        list_stdout=_workflow_list("state:ready"),
        remove_stdout=_workflow_remove("state:ready"),
    )

    assert probe.returncode != 0
    assert not list_marker.exists()
    assert remove_log.read_text(encoding="utf-8") == ""


def test_closure_workflow_skips_a_valid_pull_request_identity(tmp_path: Path) -> None:
    """A canonical PR response is an intentional no-op, not a malformed issue."""
    probe, _, list_marker, remove_log = _run_closure_workflow(
        tmp_path,
        issue_payload=_workflow_issue(
            url="https://github.com/ll7/robot_sf_ll7/pull/12",
            is_pull_request=True,
        ),
        list_stdout=_workflow_list("state:ready"),
        remove_stdout=_workflow_remove("state:ready"),
    )

    assert probe.returncode == 0, probe.stderr
    assert not list_marker.exists()
    assert remove_log.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "list_stdout",
    [
        "",
        json.dumps([]),
        json.dumps({"status": "ok"}),
        _workflow_list("state:ready", "state:ready"),
    ],
)
def test_closure_workflow_rejects_empty_or_malformed_label_inventory(
    tmp_path: Path,
    list_stdout: str,
) -> None:
    """A zero-exit list helper with an invalid envelope must not become a clean no-op."""
    probe, _, _, remove_log = _run_closure_workflow(
        tmp_path,
        issue_payload=_workflow_issue(),
        list_stdout=list_stdout,
        remove_stdout=_workflow_remove("state:ready"),
    )

    assert probe.returncode != 0
    assert remove_log.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("remove_stdout", "remove_exit"),
    [
        ("", 0),
        (json.dumps({"status": "ok", "action": "remove"}), 0),
        (_workflow_remove("state:ready"), 1),
    ],
)
def test_closure_workflow_rejects_remove_failure_or_wrong_success_envelope(
    tmp_path: Path,
    remove_stdout: str,
    remove_exit: int,
) -> None:
    """Removal failures and mismatched zero-exit results must fail the Action."""
    probe, _, _, remove_log = _run_closure_workflow(
        tmp_path,
        issue_payload=_workflow_issue(),
        list_stdout=_workflow_list("state:ready"),
        remove_stdout=remove_stdout,
        remove_exit=remove_exit,
    )

    assert probe.returncode != 0
    assert remove_log.read_text(encoding="utf-8").count("state:ready") == 1


def test_main_returns_nonzero_json_summary_without_live_github(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should be testable with an injected fetcher and emit JSON on failure."""

    def fake_fetch(
        *,
        repo: str,
        labels: tuple[str, ...],
        limit: int,
    ) -> dict[str, list[dict[str, object]]]:
        assert repo == "ll7/robot_sf_ll7"
        assert labels == ("state:ready", "state:running", "state:blocked")
        assert limit == 1000
        return {
            "state:ready": [
                {
                    "number": 12,
                    "title": "done but still queued",
                    "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                    "state": "closed",
                    "is_pull_request": False,
                    "labels": [{"name": "state:ready"}],
                }
            ]
        }

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fake_fetch)
    _mock_current_rest_issue(
        monkeypatch,
        labels=["state:ready"],
        title="done but still queued",
    )

    exit_code = closed_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["stale_count"] == 1
    assert payload["issues"][0]["number"] == 12


def test_main_suppresses_search_label_absent_from_current_rest_issue(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search-index lag should preserve the successful read-only JSON and exit contracts."""
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label",
        lambda **kwargs: {
            "state:ready": [
                {
                    "number": 12,
                    "title": "stale search row",
                    "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                    "state": "closed",
                    "is_pull_request": False,
                    "labels": [{"name": "state:ready"}],
                }
            ]
        },
    )
    _mock_current_rest_issue(monkeypatch, labels=["priority:4", "technical-debt"])

    exit_code = closed_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema"] == "closed_state_label_hygiene.v1"
    assert payload["ok"] is True
    assert payload["read_only"] is True
    assert payload["stale_count"] == 0
    assert payload["issues"] == []


def test_run_search_command_reports_missing_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing gh should produce an actionable runtime error."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GitHub CLI 'gh' was not found"):
        closed_state_label_hygiene._run_search_command(["gh", "search", "issues"])


def test_run_search_command_preserves_captured_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured gh stderr should appear in the machine-readable error path."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=("gh", "search", "issues"),
            stderr="authentication required",
        )

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="authentication required"):
        closed_state_label_hygiene._run_search_command(["gh", "search", "issues"])


def test_run_search_command_reports_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed gh output should be diagnosed before row filtering."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=("gh",), returncode=0, stdout="not-json")

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Failed to parse GitHub CLI JSON output"):
        closed_state_label_hygiene._run_search_command(["gh", "search", "issues"])


def _native_issue(
    *,
    number: int = 12,
    state: str = "closed",
    url: str = "https://github.com/ll7/robot_sf_ll7/issues/12",
    is_pull_request: object = False,
) -> dict[str, object]:
    """Build a normalized GitHub CLI search row."""
    return {
        "number": number,
        "title": "candidate issue",
        "url": url,
        "state": state,
        "labels": [{"name": "state:ready"}],
        "isPullRequest": is_pull_request,
    }


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (None, "expected an object"),
        ({**_native_issue(), "state": "BANANA"}, "OPEN or CLOSED"),
        ({**_native_issue(), "isPullRequest": "false"}, "isPullRequest"),
        ({**_native_issue(), "url": "not-a-url"}, "canonical"),
        ({**_native_issue(), "number": 0}, "positive integer"),
    ],
)
def test_run_search_command_rejects_malformed_candidate_rows(
    monkeypatch: pytest.MonkeyPatch,
    row: object,
    message: str,
) -> None:
    """Every malformed native search row must make discovery indeterminate."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=("gh",), returncode=0, stdout=json.dumps([row]))

    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fake_run)

    with pytest.raises((RuntimeError, ValueError), match=message):
        closed_state_label_hygiene._run_search_command(
            ["gh", "search", "issues"],
            repo="ll7/robot_sf_ll7",
        )


def test_run_search_command_rejects_empty_success_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty successful search response is malformed, not an empty inventory."""

    monkeypatch.setattr(
        closed_state_label_hygiene.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=("gh",), returncode=0, stdout=""),
    )

    with pytest.raises(RuntimeError, match="Failed to parse GitHub CLI JSON output"):
        closed_state_label_hygiene._run_search_command(["gh", "search", "issues"])


def _rest_page(rows: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
    """Build a successful mocked ``gh api`` page."""
    return subprocess.CompletedProcess(args=("gh", "api"), returncode=0, stdout=json.dumps(rows))


def _rest_issue(number: int, title: str, labels: list[str]) -> dict[str, object]:
    """Build a raw REST issue row."""
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "state": "closed",
        "labels": [{"name": label} for label in labels],
    }


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (None, "list of objects"),
        ({**_rest_issue(12, "candidate", ["state:ready"]), "pull_request": {}}, "resource kind"),
        ({**_rest_issue(12, "candidate", ["state:ready"]), "pull_request": None}, "pull_request"),
        ({**_rest_issue(12, "candidate", ["state:ready"]), "html_url": "not-a-url"}, "canonical"),
        ({**_rest_issue(12, "candidate", ["state:ready"]), "state": "banana"}, "OPEN or CLOSED"),
    ],
)
def test_fetch_closed_issues_by_label_rest_rejects_malformed_candidate_rows(
    row: object,
    message: str,
) -> None:
    """REST discovery must preserve malformed rows as an observable failure."""

    def fake_gh_api(path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=("gh", "api"), returncode=0, stdout=json.dumps([row])
        )

    with pytest.raises(ValueError, match=message):
        closed_state_label_hygiene.fetch_closed_issues_by_label_rest(
            repo="ll7/robot_sf_ll7",
            labels=("state:ready",),
            max_pages=1,
            per_page=100,
            gh_api=fake_gh_api,
        )


def test_fetch_closed_issues_by_label_rest_rejects_empty_success_output() -> None:
    """An empty successful REST response is malformed, not an empty page."""

    def fake_gh_api(path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=("gh", "api"), returncode=0, stdout="")

    with pytest.raises(ValueError, match="Invalid JSON"):
        closed_state_label_hygiene.fetch_closed_issues_by_label_rest(
            repo="ll7/robot_sf_ll7",
            labels=("state:ready",),
            max_pages=1,
            gh_api=fake_gh_api,
        )


def test_fetch_closed_issues_by_label_rest_paginates_until_partial_page() -> None:
    """REST discovery reads bounded pages and emits existing candidate row shape."""
    seen_paths: list[str] = []

    def fake_gh_api(path: str) -> subprocess.CompletedProcess[str]:
        seen_paths.append(path)
        if "page=1" in path:
            return _rest_page(
                [
                    _rest_issue(12, "done but still queued", ["state:ready"]),
                    _rest_issue(13, "also queued", ["state:ready"]),
                ]
            )
        if "page=2" in path:
            return _rest_page([_rest_issue(14, "last queued", ["state:ready"])])
        raise AssertionError(f"unexpected REST path: {path}")

    result = closed_state_label_hygiene.fetch_closed_issues_by_label_rest(
        repo="ll7/robot_sf_ll7",
        labels=("state:ready",),
        max_pages=3,
        per_page=2,
        gh_api=fake_gh_api,
    )

    assert [row["number"] for row in result.rows_by_label["state:ready"]] == [12, 13, 14]
    assert result.source == "rest"
    assert result.truncations == [
        {
            "label": "state:ready",
            "truncated": False,
            "row_count": 3,
            "limit": 6,
            "pages_read": 2,
            "per_page": 2,
            "page_budget": 3,
            "source": "rest",
            "note": "",
        }
    ]
    assert seen_paths == [
        "repos/ll7/robot_sf_ll7/issues?state=closed&labels=state%3Aready&per_page=2&page=1",
        "repos/ll7/robot_sf_ll7/issues?state=closed&labels=state%3Aready&per_page=2&page=2",
    ]


def test_fetch_closed_issues_by_label_rest_marks_page_budget_exhaustion() -> None:
    """A full final REST page is marked as partial inventory."""

    def fake_gh_api(path: str) -> subprocess.CompletedProcess[str]:
        page = 1 if "page=1" in path else 2
        offset = 10 * page
        return _rest_page(
            [
                _rest_issue(offset + 1, "queued a", ["state:ready"]),
                _rest_issue(offset + 2, "queued b", ["state:ready"]),
            ]
        )

    result = closed_state_label_hygiene.fetch_closed_issues_by_label_rest(
        repo="ll7/robot_sf_ll7",
        labels=("state:ready",),
        max_pages=2,
        per_page=2,
        gh_api=fake_gh_api,
    )

    marker = result.truncations[0]
    assert marker["truncated"] is True
    assert marker["row_count"] == 4
    assert marker["pages_read"] == 2
    assert "raise --max-rest-pages" in marker["note"]


def test_discover_closed_issues_falls_back_to_rest_when_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphQL/search exhaustion should not block REST candidate discovery."""

    def fail_search(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        raise RuntimeError("GitHub CLI command failed (gh search issues): GraphQL: API rate limit")

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fail_search)
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label_rest",
        lambda **kwargs: closed_state_label_hygiene.CandidateDiscoveryResult(
            rows_by_label={
                "state:ready": [
                    {
                        "number": 12,
                        "title": "done but still queued",
                        "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                        "state": "closed",
                        "is_pull_request": False,
                        "labels": [{"name": "state:ready"}],
                    }
                ]
            },
            truncations=[],
            source="rest",
        ),
    )

    result = closed_state_label_hygiene.discover_closed_issues_by_label(
        repo="ll7/robot_sf_ll7",
        labels=("state:ready",),
        limit=1000,
        max_rest_pages=2,
    )

    assert result.source == "rest"
    assert result.rows_by_label["state:ready"][0]["number"] == 12


def test_discover_closed_issues_keeps_non_graphql_failures_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authentication and repository errors must not be hidden by REST fallback."""
    search_error = RuntimeError("GitHub CLI command failed: authentication required")

    def fail_search(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        raise search_error

    def fail_rest(**kwargs: object) -> closed_state_label_hygiene.CandidateDiscoveryResult:
        raise AssertionError("REST fallback must not mask non-GraphQL failures")

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fail_search)
    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label_rest", fail_rest)

    with pytest.raises(RuntimeError, match="authentication required"):
        closed_state_label_hygiene.discover_closed_issues_by_label(
            repo="ll7/robot_sf_ll7",
            labels=("state:ready",),
            limit=1000,
        )


def test_main_rest_fallback_truncation_exits_nonzero_without_silent_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A partial REST fallback inventory must not be treated as a complete clean audit."""

    def fail_search(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        raise RuntimeError("GitHub CLI command failed (gh search issues): GraphQL: API rate limit")

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fail_search)
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label_rest",
        lambda **kwargs: closed_state_label_hygiene.CandidateDiscoveryResult(
            rows_by_label={"state:ready": []},
            truncations=[
                {
                    "label": "state:ready",
                    "truncated": True,
                    "row_count": 100,
                    "limit": 100,
                    "pages_read": 1,
                    "per_page": 100,
                    "page_budget": 1,
                    "source": "rest",
                    "note": "closed-issue REST label inventory may be partial",
                }
            ],
            source="rest",
        ),
    )

    exit_code = closed_state_label_hygiene.main(
        ["--repo", "ll7/robot_sf_ll7", "--label", "state:ready", "--max-rest-pages", "1"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is True
    assert payload["stale_count"] == 0
    assert payload["candidate_discovery_source"] == "rest"
    assert payload["truncated_any"] is True


def test_main_skips_fix_when_candidate_discovery_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A partial inventory must not mutate any issue through ``--fix``."""
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("GraphQL: rate limit")),
    )
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label_rest",
        lambda **kwargs: closed_state_label_hygiene.CandidateDiscoveryResult(
            rows_by_label={
                "state:ready": [
                    {
                        "number": 12,
                        "title": "stale candidate",
                        "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                        "state": "closed",
                        "is_pull_request": False,
                        "labels": [{"name": "state:ready"}],
                    }
                ]
            },
            truncations=[{"label": "state:ready", "truncated": True}],
            source="rest",
        ),
    )
    monkeypatch.setattr(
        closed_state_label_hygiene,
        "reconcile_stale_issues",
        lambda **kwargs: [_stale(12, ("state:ready",))],
    )

    def fail_if_mutated(**kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("partial discovery must not enter fix mode")

    monkeypatch.setattr(closed_state_label_hygiene, "fix_stale_issues", fail_if_mutated)

    exit_code = closed_state_label_hygiene.main(
        ["--repo", "ll7/robot_sf_ll7", "--label", "state:ready", "--fix"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["fix_applied"] is False
    assert payload["fix_skipped"] == "candidate discovery was truncated"
    assert payload["truncated_any"] is True


def test_main_rejects_nonpositive_rest_page_budget(capsys: pytest.CaptureFixture[str]) -> None:
    """REST fallback bounds should be validated before candidate discovery."""
    exit_code = closed_state_label_hygiene.main(["--max-rest-pages", "0"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert "--max-rest-pages must be >= 1" in payload["error"]


def _stale(number: int, labels: tuple[str, ...]) -> closed_state_label_hygiene.StaleIssue:
    """Build a StaleIssue fixture for fix-mode tests."""
    return closed_state_label_hygiene.StaleIssue(
        number=number,
        title=f"issue {number}",
        url=f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        state="closed",
        stale_labels=labels,
    )


def _remove_result(number: int, label: str) -> dict[str, object]:
    """Build a successful REST label-removal envelope for injected callbacks."""
    return {
        "status": "ok",
        "number": number,
        "label": label,
        "action": "remove",
        "repo": "ll7/robot_sf_ll7",
    }


def test_fix_stale_issues_removes_only_live_state_labels_from_closed_issues() -> None:
    """Fix mode strips the live state labels after re-confirming each issue is closed."""
    removed: list[tuple[int, str]] = []
    confirmed: list[int] = []

    def fake_confirm(*, repo: str, number: int) -> bool:
        confirmed.append(number)
        return True

    def fake_remove(number: int, label: str) -> dict[str, object]:
        removed.append((number, label))
        return _remove_result(number, label)

    actions = closed_state_label_hygiene.fix_stale_issues(
        repo="ll7/robot_sf_ll7",
        stale_issues=[_stale(12, ("state:blocked", "state:ready"))],
        confirm_closed=fake_confirm,
        remove_label=fake_remove,
    )

    assert confirmed == [12]
    assert removed == [(12, "state:blocked"), (12, "state:ready")]
    assert actions == [
        {"number": 12, "skipped": False, "removed_labels": ["state:blocked", "state:ready"]}
    ]


def test_fix_stale_issues_does_not_touch_issue_that_is_not_closed() -> None:
    """Read-then-write guard: an issue reported as not-closed must not be edited."""
    removed: list[tuple[int, str]] = []

    def fake_confirm(*, repo: str, number: int) -> bool:
        return False

    def fake_remove(number: int, label: str) -> dict[str, object]:
        removed.append((number, label))
        return _remove_result(number, label)

    actions = closed_state_label_hygiene.fix_stale_issues(
        repo="ll7/robot_sf_ll7",
        stale_issues=[_stale(12, ("state:ready",))],
        confirm_closed=fake_confirm,
        remove_label=fake_remove,
    )

    assert removed == []
    assert actions == [
        {"number": 12, "skipped": True, "reason": "not_closed", "removed_labels": []}
    ]


def test_fix_stale_issues_is_a_no_op_when_there_are_no_stale_issues() -> None:
    """Fix mode performs no reads or writes when nothing is stale."""

    def fail_confirm(*, repo: str, number: int) -> bool:  # pragma: no cover - must not run
        raise AssertionError("confirm should not be called when there are no stale issues")

    actions = closed_state_label_hygiene.fix_stale_issues(
        repo="ll7/robot_sf_ll7",
        stale_issues=[],
        confirm_closed=fail_confirm,
        remove_label=_remove_result,
    )

    assert actions == []


def test_fix_stale_issues_only_removes_documented_label_set() -> None:
    """Only labels in LIVE_STATE_LABELS are removed even if other labels slip in."""
    removed: list[tuple[int, str]] = []

    actions = closed_state_label_hygiene.fix_stale_issues(
        repo="ll7/robot_sf_ll7",
        stale_issues=[_stale(12, ("state:ready", "workflow", "priority:high"))],
        confirm_closed=lambda *, repo, number: True,
        remove_label=lambda number, label: (
            removed.append((number, label)) or _remove_result(number, label)
        ),
    )

    assert removed == [(12, "state:ready")]
    assert actions[0]["removed_labels"] == ["state:ready"]
    # The fix set is exactly the single-source-of-truth live label tuple.
    assert all(label in closed_state_label_hygiene.LIVE_STATE_LABELS for _, label in removed)


def test_fix_stale_issues_rejects_unsupported_library_allowlist() -> None:
    """Direct callers cannot bypass the canonical live-label allowlist."""
    with pytest.raises(ValueError, match="unsupported"):
        closed_state_label_hygiene.fix_stale_issues(
            repo="ll7/robot_sf_ll7",
            stale_issues=[_stale(12, ("workflow",))],
            watched_labels=("workflow",),
            confirm_closed=lambda *, repo, number: True,
            remove_label=_remove_result,
        )


def test_fix_stale_issues_rejects_zero_exit_malformed_remove_result() -> None:
    """A successful subprocess/library exit is insufficient without its envelope."""
    with pytest.raises(RuntimeError, match="invalid result"):
        closed_state_label_hygiene.fix_stale_issues(
            repo="ll7/robot_sf_ll7",
            stale_issues=[_stale(12, ("state:ready",))],
            confirm_closed=lambda *, repo, number: True,
            remove_label=lambda number, label: {},
        )


def test_confirm_issue_closed_reads_state_via_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    """confirm_issue_closed returns True only for closed, non-PR issues."""

    def fake_fetch_issue(number: int, **kwargs: object) -> dict:
        return {
            "number": number,
            "status": "ok",
            "state": "CLOSED",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
            "is_pull_request": False,
        }

    monkeypatch.setattr("scripts.dev.gh_issue_rest.fetch_issue", fake_fetch_issue)

    assert closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)


def test_confirm_issue_closed_is_false_for_open_or_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open issues and pull requests must fail the read-then-write guard."""

    def fake_fetch_issue_open(number: int, **kwargs: object) -> dict:
        return {
            "number": number,
            "status": "ok",
            "state": "OPEN",
            "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
            "is_pull_request": False,
        }

    monkeypatch.setattr("scripts.dev.gh_issue_rest.fetch_issue", fake_fetch_issue_open)
    assert not closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)

    def fake_fetch_issue_pr(number: int, **kwargs: object) -> dict:
        return {
            "number": number,
            "status": "ok",
            "state": "CLOSED",
            "url": "https://github.com/ll7/robot_sf_ll7/pull/12",
            "is_pull_request": True,
        }

    monkeypatch.setattr("scripts.dev.gh_issue_rest.fetch_issue", fake_fetch_issue_pr)
    assert not closed_state_label_hygiene.confirm_issue_closed(repo="ll7/robot_sf_ll7", number=12)


def test_main_fix_mode_strips_labels_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI --fix removes labels via mocked gh and reports success without network access."""

    def fake_fetch(
        *, repo: str, labels: tuple[str, ...], limit: int
    ) -> dict[str, list[dict[str, object]]]:
        return {
            "state:ready": [
                {
                    "number": 12,
                    "title": "done but still queued",
                    "url": "https://github.com/ll7/robot_sf_ll7/issues/12",
                    "state": "closed",
                    "is_pull_request": False,
                    "labels": [{"name": "state:ready"}],
                }
            ]
        }

    edits: list[tuple[int, str]] = []

    def fake_remove(number: int, label: str, *, repo: str) -> dict[str, object]:
        assert repo == "ll7/robot_sf_ll7"
        edits.append((number, label))
        return _remove_result(number, label)

    monkeypatch.setattr(closed_state_label_hygiene, "fetch_closed_issues_by_label", fake_fetch)
    _mock_current_rest_issue(monkeypatch, labels=["state:ready"])
    monkeypatch.setattr(closed_state_label_hygiene.gh_pr_label_rest, "remove_label", fake_remove)

    exit_code = closed_state_label_hygiene.main(["--repo", "ll7/robot_sf_ll7", "--fix"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["fix_applied"] is True
    assert payload["read_only"] is False
    assert payload["fix_actions"][0]["removed_labels"] == ["state:ready"]
    assert edits == [(12, "state:ready")]


def test_fix_stale_issues_fails_closed_when_rest_helper_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified REST helper error must stop fix mode instead of reporting success."""

    monkeypatch.setattr(
        closed_state_label_hygiene.gh_pr_label_rest,
        "remove_label",
        lambda number, label, *, repo: {
            "status": "error",
            "error": "HTTP 403: forbidden",
        },
    )

    with pytest.raises(RuntimeError, match="status must be ok"):
        closed_state_label_hygiene.fix_stale_issues(
            repo="ll7/robot_sf_ll7",
            stale_issues=[_stale(12, ("state:ready",))],
            confirm_closed=lambda *, repo, number: True,
        )


def test_main_fix_mode_rejects_labels_outside_live_state_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fix mode must fail closed before searching or editing an arbitrary label."""

    def fail_fetch(**kwargs: object) -> dict[str, list[dict[str, object]]]:
        raise AssertionError("GitHub search must not run for an unsupported fix label")

    def fail_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("GitHub edit must not run for an unsupported fix label")

    monkeypatch.setattr(
        closed_state_label_hygiene,
        "fetch_closed_issues_by_label",
        fail_fetch,
    )
    monkeypatch.setattr(closed_state_label_hygiene.subprocess, "run", fail_run)

    exit_code = closed_state_label_hygiene.main(["--label", "workflow", "--fix"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["read_only"] is False
    assert "--fix only supports live state labels" in payload["error"]
