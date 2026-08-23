"""Tests for fail-closed autonomous issue implementation admission."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.dev import issue_dependency_packet
from scripts.dev.issue_implementability import evaluate_issue, inspect_contract, live_issue_report

REPOSITORY = "ll7/robot_sf_ll7"

COMPLETE_BODY = """## Objective
Repair one bounded workflow defect.

## Scope
Change one helper. Do not change scientific semantics.

## Candidate paths
- `scripts/dev/example.py`

## Acceptance criteria
- [ ] The regression passes.

## Validation
```bash
uv run pytest -q tests/dev/test_example.py
```
"""


def _issue(*, labels: list[str] | None = None, body: str = COMPLETE_BODY) -> dict[str, object]:
    return {
        "number": 7611,
        "title": "fix: bounded workflow repair",
        "body": body,
        "state": "OPEN",
        "url": "https://github.test/issues/7611",
        "labels": labels if labels is not None else ["state:ready"],
        "assignees": [],
    }


def _claim(*, claimed: bool = False, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "claimed": claimed if ok else None,
        "claim_ref": "agent-claims/issue-7611" if claimed else None,
        "sha": "abc" if claimed else None,
    }


def test_complete_ready_issue_is_admitted() -> None:
    report = evaluate_issue(_issue(), _claim())

    assert report["classification"] == "ready"
    assert report["ready"] is True
    assert report["write_allowed"] is True
    assert report["contract"]["missing_fields"] == []


def test_goal_problem_template_heading_is_admitted() -> None:
    """The canonical `Goal / Problem` template heading must satisfy the objective field."""
    body = COMPLETE_BODY.replace("## Objective", "## Goal / Problem")
    report = evaluate_issue(_issue(body=body), _claim())

    assert report["classification"] == "ready"
    assert report["contract"]["missing_fields"] == []
    assert "goal problem" in report["contract"]["fields"]["objective"]["matched_headings"]


def test_goal_problem_prefixed_heading_still_rejected() -> None:
    """A retrospective `Goal / Problem history` heading must not satisfy the objective field."""
    body = COMPLETE_BODY.replace("## Objective", "## Goal / Problem history")
    report = evaluate_issue(_issue(body=body), _claim())

    assert report["classification"] == "needs_spec"
    assert "objective" in report["contract"]["missing_fields"]


def test_state_ready_is_required() -> None:
    report = evaluate_issue(_issue(labels=["type:workflow"]), _claim())

    assert report["classification"] == "needs_ready_label"
    assert report["ready"] is False


def test_missing_contract_field_returns_needs_spec() -> None:
    body = COMPLETE_BODY.replace("## Validation", "## Notes")
    report = evaluate_issue(_issue(body=body), _claim())

    assert report["classification"] == "needs_spec"
    assert report["contract"]["missing_fields"] == ["verification"]


@pytest.mark.parametrize(
    ("exact_heading", "informational_heading", "missing_field"),
    [
        ("Objective", "Objective history", "objective"),
        ("Scope", "Scope discussion", "scope"),
        ("Candidate paths", "Candidate paths received", "inputs"),
        ("Acceptance criteria", "Acceptance criteria discussion", "acceptance"),
        ("Validation", "Validation results", "verification"),
    ],
)
def test_alias_prefixed_headings_do_not_satisfy_contract(
    exact_heading: str,
    informational_heading: str,
    missing_field: str,
) -> None:
    """Retrospective or informational headings must not authorize a claim write."""
    body = COMPLETE_BODY.replace(f"## {exact_heading}", f"## {informational_heading}")

    report = evaluate_issue(_issue(body=body), _claim())

    assert report["classification"] == "needs_spec"
    assert missing_field in report["contract"]["missing_fields"]
    assert report["write_allowed"] is False


@pytest.mark.parametrize(
    ("labels", "classification"),
    [
        (["state:ready", "state:blocked"], "blocked"),
        (["state:ready", "decision-required"], "human_decision"),
        (["state:ready", "parent"], "parent"),
        (["state:ready", "resource:slurm"], "needs_compute"),
        (["state:ready", "state:working"], "working"),
        (["state:ready", "needs-review"], "review"),
    ],
)
def test_stop_labels_fail_closed(labels: list[str], classification: str) -> None:
    report = evaluate_issue(_issue(labels=labels), _claim())

    assert report["classification"] == classification
    assert report["write_allowed"] is False


def test_parent_title_is_not_a_leaf() -> None:
    issue = _issue()
    issue["title"] = "[Epic] broad workflow programme"

    report = evaluate_issue(issue, _claim())

    assert report["classification"] == "parent"


def test_assigned_and_claimed_states_are_distinct() -> None:
    assigned = _issue()
    assigned["assignees"] = ["worker"]

    assert evaluate_issue(assigned, _claim())["classification"] == "assigned"
    assert evaluate_issue(_issue(), _claim(claimed=True))["classification"] == "already_claimed"


def test_unknown_claim_state_fails_as_error() -> None:
    report = evaluate_issue(_issue(), _claim(ok=False))

    assert report["classification"] == "error"
    assert report["write_allowed"] is False


def test_live_issue_report_applies_explicit_dependency_packet_gate(tmp_path: Path) -> None:
    """A mandatory packet failure reaches the live preflight before claim acquisition."""
    issue = _issue()
    issue["body"] = issue["body"] + (
        "\n```json\n" + json.dumps({"schema": issue_dependency_packet.SCHEMA}) + "\n```\n"
    )
    dependency_evaluation = {
        "schema": issue_dependency_packet.EVALUATION_SCHEMA,
        "ok": False,
        "verdict": "blocked",
        "packet_digest": "d" * 64,
        "mandatory_failures": [
            {
                "id": "required-pr",
                "reason": "required PR is not merged",
                "unblock_condition": "merge PR #42",
            }
        ],
        "advisory_failures": [],
    }

    with (
        patch("scripts.dev.issue_implementability.fetch_live_issue", return_value=issue),
        patch("scripts.dev.issue_implementability.issue_claim.status_issue", return_value=_claim()),
        patch(
            "scripts.dev.issue_implementability.issue_dependency_packet.resolve_packet",
            return_value=dependency_evaluation,
        ) as resolve_packet,
    ):
        report = live_issue_report(7611, repo=REPOSITORY, remote="origin", repo_root=tmp_path)

    assert report["classification"] == "needs_dependency"
    assert report["ready"] is False
    assert report["write_allowed"] is False
    assert report["dependency_gate"]["mandatory_failures"][0]["id"] == "required-pr"
    resolve_packet.assert_called_once_with(
        {"schema": issue_dependency_packet.SCHEMA},
        repo_root=tmp_path,
        expected_repository=REPOSITORY,
        expected_issue=7611,
    )


def test_decision_heading_is_ignored_after_ruling() -> None:
    body = COMPLETE_BODY + "\n## Decision required\nHistorical choice is resolved.\n"

    pending = evaluate_issue(_issue(body=body), _claim())
    ruled = evaluate_issue(_issue(body=body, labels=["state:ready", "ruled"]), _claim())

    assert pending["classification"] == "human_decision"
    assert ruled["classification"] == "ready"


def test_contract_and_report_are_deterministic() -> None:
    issue = _issue(labels=["type:workflow", "state:ready", "state:ready"])

    first = evaluate_issue(issue, _claim())
    second = evaluate_issue(copy.deepcopy(issue), copy.deepcopy(_claim()))

    assert first == second
    assert inspect_contract(COMPLETE_BODY) == inspect_contract(COMPLETE_BODY)


def test_offline_cli_exercises_real_process_boundary(tmp_path: Path) -> None:
    """The documented offline command must preserve JSON and exit-code semantics."""
    body_file = tmp_path / "issue.md"
    body_file.write_text(COMPLETE_BODY, encoding="utf-8")
    root = Path(__file__).resolve().parents[2]

    ready = subprocess.run(
        [
            sys.executable,
            "scripts/dev/issue_implementability.py",
            "1",
            "--body-file",
            str(body_file),
            "--label",
            "state:ready",
            "--title",
            "fixture issue",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    not_ready = subprocess.run(
        [
            sys.executable,
            "scripts/dev/issue_implementability.py",
            "1",
            "--body-file",
            str(body_file),
            "--title",
            "fixture issue",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert ready.returncode == 0
    assert json.loads(ready.stdout)["classification"] == "ready"
    assert not_ready.returncode == 2
    assert json.loads(not_ready.stdout)["classification"] == "needs_ready_label"


@pytest.mark.parametrize(
    "heading",
    ["Verification gates", "Validation gates"],
)
def test_gate_heading_aliases_satisfy_verification(heading: str) -> None:
    """Issue #7701: 'Verification Gates' style headings satisfy the verification field."""
    body = COMPLETE_BODY.replace("## Validation", f"## {heading}")

    report = evaluate_issue(_issue(body=body), _claim())

    assert report["contract"]["fields"]["verification"]["present"] is True
    assert "verification" not in report["contract"]["missing_fields"]


def test_unknown_verification_variant_still_rejected() -> None:
    """Unlisted variants such as 'Verification gate checklist' must keep failing closed."""
    body = COMPLETE_BODY.replace("## Validation", "## Verification gate checklist")

    report = evaluate_issue(_issue(body=body), _claim())

    assert report["contract"]["missing_fields"] == ["verification"]
