"""Tests for fail-closed autonomous issue implementation admission."""

from __future__ import annotations

import copy
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.dev import issue_dependency_packet, issue_implementability
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

ISSUE_8453_PRE_REPAIR_BODY = """## Goal
Make agents choose a bounded task route.

## Scope and ownership
Keep the implementation packet scoped to one route and preserve existing gates.

## Required changes
- Update the canonical workflow entrypoint.

## Acceptance and validation
- [ ] The route is explicit and the focused checks pass.
- `uv run python scripts/tools/sync_ai_config.py --check`
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


def _execution_body(
    *,
    owning_repo: str = REPOSITORY,
    mutation_repos: list[str] | None = None,
    route_required: str = "local",
    external_inputs: list[object] | None = None,
) -> str:
    """Return a complete issue body with an explicit execution declaration."""
    repos = mutation_repos if mutation_repos is not None else [REPOSITORY]
    inputs = external_inputs if external_inputs is not None else []
    repo_lines = "".join(f"    - {repo}\n" for repo in repos)
    return (
        COMPLETE_BODY
        + f"""
## Execution
```yaml
execution:
  owning_repo: {owning_repo}
  mutation_repos:
{repo_lines}  route_required: {route_required}
  external_inputs: {json.dumps(inputs)}
```
"""
    )


def test_explicit_local_execution_contract_is_admitted() -> None:
    report = evaluate_issue(_issue(body=_execution_body()), _claim())

    assert report["classification"] == "ready"
    assert report["admission_reason"] == "claimable"
    assert report["execution_contract"]["present"] is True
    assert report["execution_contract"]["route_preflight"]["status"] == "not_required"


def test_empty_execution_section_is_not_an_implicit_local_contract() -> None:
    report = evaluate_issue(_issue(body=COMPLETE_BODY + "\n## Execution\n"), _claim())

    assert report["classification"] == "needs_spec"
    assert report["admission_reason"] == "needs_spec"
    assert report["execution_contract"]["valid"] is False


def test_wrong_owner_repo_fails_closed_without_route_preflight() -> None:
    body = _execution_body(
        owning_repo="ll7/codex-orchestrator",
        mutation_repos=["ll7/codex-orchestrator"],
        route_required="multi_repository",
    )

    report = evaluate_issue(_issue(body=body), _claim(), repository=REPOSITORY)

    assert report["classification"] == "wrong_owner_repo"
    assert report["admission_reason"] == "wrong_owner_repo"
    assert report["execution_contract"]["route_preflight"]["status"] == "missing"
    assert report["write_allowed"] is False


def test_fresh_multi_repository_route_preflight_can_admit_owner_route() -> None:
    now = dt.datetime(2026, 8, 28, 19, 10, tzinfo=dt.UTC)
    route = {
        "selected_route": {"provider": "codex", "model": "gpt-5.6-luna"},
        "config_digest": "a" * 64,
        "created_at": "2026-08-28T19:00:00Z",
        "ttl_seconds": 1800,
    }
    body = _execution_body(
        owning_repo="ll7/codex-orchestrator",
        mutation_repos=["ll7/codex-orchestrator"],
        route_required="multi_repository",
    )

    report = evaluate_issue(
        _issue(body=body),
        _claim(),
        repository=REPOSITORY,
        route_preflight=route,
        now=now,
    )

    assert report["classification"] == "ready"
    assert report["admission_reason"] == "claimable"
    assert report["execution_contract"]["route_preflight"]["status"] == "fresh"


def test_expired_multi_repository_route_preflight_fails_closed() -> None:
    route = {
        "selected_route": {"provider": "codex", "model": "gpt-5.6-luna"},
        "config_digest": "a" * 64,
        "created_at": "2026-08-28T18:00:00Z",
        "ttl_seconds": 1800,
    }
    body = _execution_body(
        owning_repo="ll7/codex-orchestrator",
        mutation_repos=["ll7/codex-orchestrator"],
        route_required="multi_repository",
    )

    report = evaluate_issue(
        _issue(body=body),
        _claim(),
        repository=REPOSITORY,
        route_preflight=route,
        now=dt.datetime(2026, 8, 28, 19, 0, tzinfo=dt.UTC),
    )

    assert report["classification"] == "wrong_owner_repo"
    assert report["admission_reason"] == "stale_route_state"
    assert report["execution_contract"]["route_preflight"]["status"] == "stale"


def test_non_executable_selected_route_fails_closed() -> None:
    body = _execution_body(
        owning_repo="ll7/codex-orchestrator",
        mutation_repos=["ll7/codex-orchestrator"],
        route_required="multi_repository",
    )
    report = evaluate_issue(
        _issue(body=body),
        _claim(),
        route_preflight={
            "selected_route": {"is_worker_executable": False},
            "config_digest": "a" * 64,
            "created_at": "2026-08-28T19:00:00Z",
        },
        now=dt.datetime(2026, 8, 28, 19, 10, tzinfo=dt.UTC),
    )

    assert report["classification"] == "wrong_owner_repo"
    assert report["admission_reason"] == "stale_route_state"
    assert report["execution_contract"]["route_preflight"]["status"] == "invalid"


def test_declared_external_input_is_not_local_claimable_work() -> None:
    report = evaluate_issue(
        _issue(body=_execution_body(external_inputs=["source-bytes"])),
        _claim(),
    )

    assert report["classification"] == "blocked"
    assert report["admission_reason"] == "external_input_missing"


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

    assert report["classification"] == "state_conflict"
    assert report["admission_reason"] == "state_label_conflict"
    assert report["ready"] is False


def test_two_execution_state_labels_fail_closed() -> None:
    report = evaluate_issue(
        _issue(labels=["type:workflow", "state:ready", "state:blocked"]), _claim()
    )

    assert report["classification"] == "state_conflict"
    assert report["admission_reason"] == "state_label_conflict"


@pytest.mark.parametrize(
    ("labels", "classification"),
    [
        (["state:ready", "state:review"], "review"),
        (["state:ready", "state:needs-artifact-promotion"], "ready"),
        (["state:ready", "state:blocked-no-code-slice"], "blocked"),
        (["state:ready", "state:parked"], "blocked"),
        (["state:ready", "state:deferred"], "blocked"),
        (["state:ready", "state:working"], "working"),
        (["state:blocked", "state:blocked-no-code-slice"], "blocked"),
        (["state:blocked", "state:parked"], "blocked"),
        (["state:running", "state:blocked-no-code-slice"], "stale_running"),
        (["state:running", "state:parked"], "stale_running"),
    ],
)
def test_state_qualifiers_do_not_create_execution_state_conflicts(
    labels: list[str], classification: str
) -> None:
    report = evaluate_issue(_issue(labels=labels), _claim())

    assert report["classification"] == classification
    assert report["admission_reason"] != "state_label_conflict"


def test_unknown_state_label_fails_closed_until_classified() -> None:
    report = evaluate_issue(_issue(labels=["state:ready", "state:surprise"]), _claim())

    assert report["classification"] == "state_conflict"
    assert report["admission_reason"] == "state_label_conflict"
    assert "unknown state:* label" in report["reasons"][0]


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
        (["state:ready", "state:blocked"], "state_conflict"),
        (["state:blocked-no-code-slice"], "blocked"),
        (["state:ready", "deferred"], "blocked"),
        (["state:ready", "state:deferred"], "blocked"),
        (["state:ready", "state:parked"], "blocked"),
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


def test_running_state_without_claim_is_stale() -> None:
    report = evaluate_issue(_issue(labels=["state:running"]), _claim())

    assert report["classification"] == "stale_running"
    assert report["admission_reason"] == "stale_running_state"


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


def test_live_issue_report_can_evaluate_prospective_ready_without_mutating_issue() -> None:
    """Prospective readiness is isolated to evaluation and leaves the REST payload unchanged."""
    issue = _issue(labels=["type:workflow"])

    with (
        patch("scripts.dev.issue_implementability.fetch_live_issue", return_value=issue),
        patch("scripts.dev.issue_implementability.issue_claim.status_issue", return_value=_claim()),
    ):
        report = live_issue_report(
            7611,
            repo=REPOSITORY,
            remote="origin",
            prospective_ready=True,
        )

    assert report["classification"] == "ready"
    assert report["issue"]["labels"] == ["state:ready", "type:workflow"]
    assert issue["labels"] == ["type:workflow"]


def test_live_issue_report_does_not_override_existing_state_with_prospective_ready() -> None:
    """An existing execution state remains authoritative during prospective evaluation."""
    issue = _issue(labels=["state:blocked"])

    with (
        patch("scripts.dev.issue_implementability.fetch_live_issue", return_value=issue),
        patch("scripts.dev.issue_implementability.issue_claim.status_issue", return_value=_claim()),
    ):
        report = live_issue_report(
            7611,
            repo=REPOSITORY,
            remote="origin",
            prospective_ready=True,
        )

    assert report["classification"] == "state_conflict"
    assert report["admission_reason"] == "state_label_conflict"
    assert issue["labels"] == ["state:blocked"]


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
    assert json.loads(not_ready.stdout)["classification"] == "state_conflict"


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


@pytest.mark.parametrize(
    "template_path",
    sorted(Path(".github/ISSUE_TEMPLATE").glob("*.md")),
    ids=lambda p: p.name,
)
def test_all_repository_markdown_issue_templates_satisfy_contract(template_path: Path) -> None:
    """Issue #7793: every canonical markdown issue template must satisfy the contract."""
    body = template_path.read_text(encoding="utf-8")
    inspection = inspect_contract(body)
    assert inspection["complete"] is True
    assert inspection["missing_fields"] == []

    report = evaluate_issue(_issue(body=body), _claim())
    assert report["contract"]["missing_fields"] == []


def test_preflight_body_text_accepts_complete_body() -> None:
    """A body with all five canonical fields passes the zero-write preflight."""
    body = (
        "## Goal / Problem\n\nFix the thing.\n\n"
        "## Scope Boundary\n\nOnly this file.\n\n"
        "## Inputs\n\n- one file\n\n"
        "## Acceptance Criteria\n\n- checker green\n\n"
        "## Verification\n\n- run the checker\n"
    )
    payload = issue_implementability.preflight_body_text(body)

    assert payload["schema"] == "issue_body_preflight.v1"
    assert payload["ready"] is True
    assert payload["missing_fields"] == []
    assert payload["body_sha256"] == issue_implementability.preflight_body_text(body)["body_sha256"]


def test_preflight_body_text_accepts_issue_8453_pre_repair_shape() -> None:
    """Issue #8453's compound headings satisfy their combined contract fields."""
    inspection = inspect_contract(ISSUE_8453_PRE_REPAIR_BODY)
    payload = issue_implementability.preflight_body_text(ISSUE_8453_PRE_REPAIR_BODY)

    assert inspection["fields"]["scope"]["matched_headings"] == ["scope and ownership"]
    assert inspection["fields"]["acceptance"]["matched_headings"] == ["acceptance and validation"]
    assert inspection["fields"]["verification"]["matched_headings"] == ["acceptance and validation"]
    assert payload["ready"] is True
    assert payload["missing_fields"] == []


@pytest.mark.parametrize(
    ("missing", "headings"),
    [
        (
            "objective",
            [
                "## Scope Boundary\n\ntext\n",
                "## Inputs\n\ntext\n",
                "## Acceptance Criteria\n\ntext\n",
                "## Verification\n\ntext\n",
            ],
        ),
        (
            "scope",
            [
                "## Goal / Problem\n\ntext\n",
                "## Inputs\n\ntext\n",
                "## Acceptance Criteria\n\ntext\n",
                "## Verification\n\ntext\n",
            ],
        ),
        (
            "inputs",
            [
                "## Goal / Problem\n\ntext\n",
                "## Scope Boundary\n\ntext\n",
                "## Acceptance Criteria\n\ntext\n",
                "## Verification\n\ntext\n",
            ],
        ),
        (
            "acceptance",
            [
                "## Goal / Problem\n\ntext\n",
                "## Scope Boundary\n\ntext\n",
                "## Inputs\n\ntext\n",
                "## Verification\n\ntext\n",
            ],
        ),
        (
            "verification",
            [
                "## Goal / Problem\n\ntext\n",
                "## Scope Boundary\n\ntext\n",
                "## Inputs\n\ntext\n",
                "## Acceptance Criteria\n\ntext\n",
            ],
        ),
    ],
)
def test_preflight_body_text_rejects_each_missing_field(missing: str, headings: list[str]) -> None:
    """Each observed incomplete-body shape is rejected with the exact field name."""
    body = "".join(headings)
    payload = issue_implementability.preflight_body_text(body)

    assert payload["ready"] is False
    assert payload["missing_fields"] == [missing]


def test_preflight_body_text_rejects_empty_body() -> None:
    """An empty body is missing all five canonical fields."""
    payload = issue_implementability.preflight_body_text("")

    assert payload["ready"] is False
    assert payload["missing_fields"] == [
        "objective",
        "scope",
        "inputs",
        "acceptance",
        "verification",
    ]


def test_preflight_body_file_reads_disk_without_network(tmp_path: Path) -> None:
    """The file preflight reads one local file and returns the same verdict shape."""
    body_path = tmp_path / "body.md"
    body_path.write_text(
        "## Goal / Problem\n\nx\n\n## Scope\n\nx\n\n## Inputs\n\nx\n\n"
        "## Acceptance Criteria\n\nx\n\n## Verification\n\nx\n",
        encoding="utf-8",
    )
    payload = issue_implementability.preflight_body_file(body_path)

    assert payload["ready"] is True
    assert payload["missing_fields"] == []


def test_main_preflight_body_mode_is_zero_write(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The CLI preflight mode reports ready/incomplete without any issue argument."""
    complete = tmp_path / "complete.md"
    complete.write_text(
        "## Goal / Problem\n\nx\n\n## Scope\n\nx\n\n## Inputs\n\nx\n\n"
        "## Acceptance Criteria\n\nx\n\n## Verification\n\nx\n",
        encoding="utf-8",
    )
    incomplete = tmp_path / "incomplete.md"
    incomplete.write_text("## Goal / Problem\n\nx\n", encoding="utf-8")

    assert issue_implementability.main(["--preflight-body", str(complete)]) == 0
    ready_payload = json.loads(capsys.readouterr().out)
    assert ready_payload["schema"] == "issue_body_preflight.v1"
    assert ready_payload["ready"] is True

    assert issue_implementability.main(["--preflight-body", str(incomplete)]) == 2
    incomplete_payload = json.loads(capsys.readouterr().out)
    assert incomplete_payload["ready"] is False
    assert incomplete_payload["missing_fields"] == [
        "scope",
        "inputs",
        "acceptance",
        "verification",
    ]
