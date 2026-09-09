"""Tests for fail-closed autonomous issue implementation admission."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import yaml

from scripts.dev import issue_dependency_packet, issue_implementability
from scripts.dev.issue_implementability import evaluate_issue, inspect_contract, live_issue_report

if TYPE_CHECKING:
    from collections.abc import Iterator

REPOSITORY = "ll7/robot_sf_ll7"


@pytest.fixture
def cli_checkout(tmp_path: Path) -> Iterator[Path]:
    """Stage the real offline import closure in a checkout whose path has spaces."""
    root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "active checkout"
    paths = ["scripts/__init__.py"] + [
        f"scripts/dev/{name}.py"
        for name in (
            "issue_implementability",
            "gh_issue_rest",
            "issue_claim",
            "issue_dependency_packet",
            "issue_state_taxonomy",
            "_gh_rest",
            "github_transport_policy",
        )
    ]
    for path in paths:
        target = checkout / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, target)
    original = {
        path.relative_to(checkout): path.read_bytes()
        for path in checkout.rglob("*")
        if path.is_file()
    }
    yield checkout
    assert {
        path.relative_to(checkout): path.read_bytes()
        for path in checkout.rglob("*")
        if path.is_file()
    } == original


@pytest.fixture
def isolated_cli_env(tmp_path: Path) -> Iterator[dict[str, str]]:
    """Keep real PyYAML, but exclude editable hooks, credentials and source roots."""
    dependencies = tmp_path / "dependencies"
    shutil.copytree(
        Path(yaml.__file__).parent,
        dependencies / "yaml",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    audit = tmp_path / "external-calls"
    for name in ("gh", "git", "uv", "pip"):
        command = commands / name
        command.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$0" >> "$CALL_AUDIT"\nexit 97\n', encoding="utf-8"
        )
        command.chmod(0o755)
    env = {"PATH": str(commands), "PYTHONPATH": str(dependencies), "CALL_AUDIT": str(audit)}
    dependency = subprocess.run(
        [sys.executable, "-S", "-B", "-c", "import yaml; assert yaml.safe_load('ok: true')['ok']"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert dependency.returncode == 0, dependency.stderr
    yield env
    assert not audit.exists(), "Offline CLI invoked an external command"
    assert not list(tmp_path.rglob("*.pyc"))


@pytest.mark.parametrize("foreign_cwd", [False, True], ids=["checkout", "foreign"])
def test_direct_help_without_ambient_source_path(
    tmp_path: Path,
    isolated_cli_env: dict[str, str],
    foreign_cwd: bool,
) -> None:
    """The actual direct entrypoint works without inherited repository imports."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-S", "-B", str(root / "scripts/dev/issue_implementability.py"), "--help"],
        cwd=tmp_path if foreign_cwd else root,
        env=isolated_cli_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--preflight-body" in result.stdout
    assert result.stderr == ""


def test_direct_help_from_space_checkout(
    cli_checkout: Path,
    tmp_path: Path,
    isolated_cli_env: dict[str, str],
) -> None:
    """An absolute entrypoint in a space-containing checkout needs no cwd assistance."""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            str(cli_checkout / "scripts/dev/issue_implementability.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=isolated_cli_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--preflight-body" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("mode", "complete", "labels", "claimed", "classification", "exit_code"),
    [
        ("preflight", True, [], "false", "ready", 0),
        ("preflight", False, [], "false", "needs_spec", 2),
        ("body", True, ["state:ready"], "false", "ready", 0),
        ("body", True, ["state:ready", "state:blocked"], "false", "state_conflict", 2),
        ("body", True, [], "false", "state_conflict", 2),
        ("body", True, ["state:ready"], "true", "already_claimed", 2),
        ("body", True, ["state:ready"], "unknown", "error", 1),
    ],
)
def test_isolated_offline_cli_direct_module_parity(
    cli_checkout: Path,
    isolated_cli_env: dict[str, str],
    mode: str,
    complete: bool,
    labels: list[str],
    claimed: str,
    classification: str,
    exit_code: int,
) -> None:
    """Direct execution preserves the real offline decisions and exact JSON bytes."""
    tmp_path = cli_checkout.parent
    body = COMPLETE_BODY if complete else "## Objective\nRepair one defect.\n"
    body_file = tmp_path / "body with spaces.md"
    body_file.write_text(body, encoding="utf-8")
    if mode == "preflight":
        args = ["--preflight-body", str(body_file)]
    else:
        args = ["1", "--body-file", str(body_file), "--claimed", claimed]
        for label in labels:
            args.extend(["--label", label])
    results = []
    for entrypoint, cwd in (
        ([str(cli_checkout / "scripts/dev/issue_implementability.py")], tmp_path),
        (["-m", "scripts.dev.issue_implementability"], cli_checkout),
    ):
        result = subprocess.run(
            [sys.executable, "-S", "-B", *entrypoint, *args],
            cwd=cwd,
            env=isolated_cli_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        assert result.returncode == exit_code, result.stderr or result.stdout
        assert result.stderr == ""
        results.append(result.stdout)
    assert results[0] == results[1]
    report = json.loads(results[0])
    assert report["ready"] is (exit_code == 0)
    if mode == "preflight":
        assert report["schema"] == "issue_body_preflight.v1"
        assert report["body_sha256"] == hashlib.sha256(body.encode()).hexdigest()
        assert report["missing_fields"] == (
            [] if complete else ["scope", "inputs", "acceptance", "verification"]
        )
    else:
        assert report["schema"] == "issue_implementability.v1"
        assert report["classification"] == classification
    assert body_file.read_text(encoding="utf-8") == body


def test_safe_path_direct_cli_prefers_active_checkout(
    cli_checkout: Path,
    tmp_path: Path,
    isolated_cli_env: dict[str, str],
) -> None:
    """An active root already later in sys.path must outrank a competing checkout."""
    hostile = tmp_path / "competing checkout"
    package = hostile / "scripts"
    package.mkdir(parents=True)
    poison = package / "__init__.py"
    poison.write_text("raise AssertionError('competing checkout imported')\n", encoding="utf-8")
    env = dict(isolated_cli_env)
    env["PYTHONPATH"] = os.pathsep.join([str(hostile), str(cli_checkout), env["PYTHONPATH"]])
    script = cli_checkout / "scripts/dev/issue_implementability.py"
    result = subprocess.run(
        [sys.executable, "-P", "-S", "-B", str(script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--preflight-body" in result.stdout
    assert result.stderr == ""
    # Supplemental readback verifies the actual helper files after public CLI execution.
    readback = subprocess.run(
        [
            sys.executable,
            "-P",
            "-S",
            "-B",
            "-c",
            "import runpy,sys; from pathlib import Path; script=sys.argv[1]; "
            "sys.argv=[script,'--help'];\n"
            "try: runpy.run_path(script,run_name='__main__')\n"
            "except SystemExit as exc: assert exc.code == 0\n"
            "for name in ('gh_issue_rest','issue_claim','issue_dependency_packet','issue_state_taxonomy'):\n"
            " assert Path(sys.modules['scripts.dev.'+name].__file__).resolve().parent == Path(script).parent\n",
            str(script),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert readback.returncode == 0, readback.stderr
    assert (
        poison.read_text(encoding="utf-8")
        == "raise AssertionError('competing checkout imported')\n"
    )


def test_ordinary_module_import_preserves_search_path(
    cli_checkout: Path,
    isolated_cli_env: dict[str, str],
) -> None:
    """Importing as a package keeps the caller's module search order unchanged."""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-c",
            "import sys; from pathlib import Path; before=list(sys.path); "
            "from scripts.dev import issue_implementability; "
            "assert sys.path == before; "
            "assert Path(issue_implementability.__file__).resolve() == "
            "Path('scripts/dev/issue_implementability.py').resolve()",
        ],
        cwd=cli_checkout,
        env=isolated_cli_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""


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
        (["state:running", "state:needs-interpretation"], "stale_running"),
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


def test_empty_decision_heading_is_admission_neutral() -> None:
    """An empty decision heading is an offline hint candidate, not an admission blocker."""
    body = COMPLETE_BODY + "\n## Decision required\n"

    report = evaluate_issue(_issue(body=body), _claim())

    assert report["classification"] == "ready"
    assert "decision required" not in report["contract"]["headings"]


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
    assert payload["heading_suggestions"] == {}


def test_preflight_body_text_suggests_alias_for_parenthesized_heading() -> None:
    """Issue #8694: a near-miss heading maps to its closest missing-field alias."""
    body = (
        "## Goal / Problem\n\nFix the thing.\n\n"
        "## Scope Boundary\n\nOnly this file.\n\n"
        "## Inputs (formalization appended 2026-09-08)\n\n- one file\n\n"
        "## Acceptance Criteria\n\n- checker green\n\n"
        "## Verification\n\n- run the checker\n"
    )
    payload = issue_implementability.preflight_body_text(body)

    assert payload["ready"] is False
    assert payload["missing_fields"] == ["inputs"]
    suggestion = payload["heading_suggestions"]["inputs (formalization appended 2026 09 08)"]
    assert suggestion["field"] == "inputs"
    assert suggestion["alias"] == "inputs"


def test_preflight_body_text_suggests_alias_for_input_contract_heading() -> None:
    """A near-miss `Input contract` heading suggests the `inputs` field."""
    body = (
        "## Goal / Problem\n\nFix the thing.\n\n"
        "## Scope Boundary\n\nOnly this file.\n\n"
        "## Input contract\n\n- one file\n\n"
        "## Acceptance Criteria\n\n- checker green\n\n"
        "## Verification\n\n- run the checker\n"
    )
    payload = issue_implementability.preflight_body_text(body)

    assert payload["ready"] is False
    assert payload["missing_fields"] == ["inputs"]
    suggestion = payload["heading_suggestions"]["input contract"]
    assert suggestion["field"] == "inputs"
    assert suggestion["alias"] == "inputs"


def test_preflight_body_text_suggests_alias_for_empty_input_contract_heading() -> None:
    """Empty near-miss headings remain visible to the suggestion helper."""
    body = (
        "## Goal / Problem\n\nFix the thing.\n\n"
        "## Scope Boundary\n\nOnly this file.\n\n"
        "## Input contract\n\n"
        "## Acceptance Criteria\n\n- checker green\n\n"
        "## Verification\n\n- run the checker\n"
    )
    payload = issue_implementability.preflight_body_text(body)

    assert payload["ready"] is False
    assert payload["missing_fields"] == ["inputs"]
    suggestion = payload["heading_suggestions"]["input contract"]
    assert suggestion["field"] == "inputs"
    assert suggestion["alias"] == "inputs"


@pytest.mark.parametrize("heading", ["Outputs", "Implementation"])
def test_preflight_body_text_omits_misleading_heading_suggestions(heading: str) -> None:
    """Sequence-only similarity must not suggest unrelated missing-field aliases."""
    body = (
        "## Goal / Problem\n\nFix the thing.\n\n"
        "## Scope Boundary\n\nOnly this file.\n\n"
        f"## {heading}\n\n- described content\n\n"
        "## Acceptance Criteria\n\n- checker green\n\n"
        "## Verification\n\n- run the checker\n"
    )

    payload = issue_implementability.preflight_body_text(body)

    assert payload["ready"] is False
    assert payload["missing_fields"] == ["inputs"]
    assert payload["heading_suggestions"] == {}


def test_preflight_body_text_suggests_nothing_for_exact_body() -> None:
    """Exact-heading bodies produce an empty suggestion map."""
    body = (
        "## Goal / Problem\n\nFix the thing.\n\n"
        "## Scope Boundary\n\nOnly this file.\n\n"
        "## Inputs\n\n- one file\n\n"
        "## Acceptance Criteria\n\n- checker green\n\n"
        "## Verification\n\n- run the checker\n"
    )
    payload = issue_implementability.preflight_body_text(body)

    assert payload["ready"] is True
    assert payload["heading_suggestions"] == {}


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
