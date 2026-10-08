"""Behavioral proof for commit-range admission and workflow hosted fallback."""

from __future__ import annotations

import copy
import os
import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from tests.ci.test_self_hosted_job_hook import (
    BASE,
    COMPARE,
    HEAD,
    OTHER,
    _commit,
    _comparison,
    _event,
    _run_hook,
)
from tests.ci.test_self_hosted_routing import (
    CI_WORKFLOW,
    ROOT,
    ROUTED_JOBS,
    _github_context,
    _resolve_runs_on,
)


def _assert_decision(tmp_path: Path, *, expected: bool, event: str = "pull_request", **kwargs):
    """Both actual hook execution and scheduling must enforce the same policy."""
    hook = _run_hook(tmp_path, event_name=event, **kwargs)
    # Execute the workflow's admission step, including its fallback for script errors.
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    gate = workflow["jobs"].get("self-hosted-admission")
    if gate is None:
        # Baseline has no provenance gate; evaluate its real runs-on expression.
        provenance = "true"
    else:
        assert gate["runs-on"] == "ubuntu-latest"
        step = next(step for step in gate["steps"] if step.get("id") == "provenance")
        output = tmp_path / "github-output"
        env = {"GITHUB_OUTPUT": str(output)}
        # _run_hook's mock curl writes its fixture and payload; retain that exact
        # offline transport while executing the production workflow step.
        env.update(
            PATH=f"{tmp_path}:{os.environ['PATH']}",
            API_FIXTURES=str(tmp_path / "responses.json"),
            API_CALLS=str(tmp_path / "workflow-api-calls.txt"),
            GITHUB_REPOSITORY="ll7/robot_sf_ll7",
            GITHUB_ACTOR="ll7",
            GITHUB_TRIGGERING_ACTOR="ll7",
            GITHUB_EVENT_NAME=event,
            GITHUB_EVENT_PATH=str(tmp_path / "event.json"),
            GITHUB_SHA=kwargs.get("changes", {}).get("sha", HEAD),
        )
        completed = subprocess.run(
            ["bash", "-e", "-c", step["run"]],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        provenance = output.read_text().strip().removeprefix("self_hosted=")
    for name in ROUTED_JOBS:
        actual = _resolve_runs_on(
            workflow["jobs"][name]["runs-on"], _github_context(event), provenance=provenance
        )
        assert actual == ("robot-sf-ci-ephemeral" if expected else "ubuntu-latest"), name
    assert hook.returncode == (0 if expected else 1), hook.stderr


@pytest.mark.parametrize("event", ["pull_request", "push"])
@pytest.mark.parametrize("identity", ["author", "committer"])
def test_third_party_range_never_reaches_self_hosted(tmp_path: Path, event: str, identity: str):
    """Owner ready_for_review/push cannot authorize a collaborator's earlier commit."""
    payload = _event()
    payload["action"] = "ready_for_review"
    payload["pull_request"]["draft"] = False
    earlier = _commit(OTHER, **{identity: "collaborator"})
    _assert_decision(
        tmp_path,
        expected=False,
        event=event,
        payload=payload,
        responses={COMPARE: _comparison([earlier, _commit()])},
    )


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_owner_range_reaches_self_hosted(tmp_path: Path, event: str):
    """Complete owner author AND committer provenance retains the fast route."""
    _assert_decision(
        tmp_path,
        expected=True,
        event=event,
        responses={COMPARE: _comparison([_commit(OTHER), _commit()])},
    )


@pytest.mark.parametrize(
    "fault",
    [
        "api",
        "empty",
        "json",
        "multiple-json",
        "missing-author",
        "missing-committer",
        "truncated",
        "duplicate",
        "no-head",
        "wrong-base",
        "missing-total",
        "bot",
        "web-flow-committer",
    ],
)
def test_unknown_or_non_owner_evidence_stays_hosted(tmp_path: Path, fault: str):
    """Missing data, malformed ranges, bots and transport errors never grant access."""
    payload = _event()
    comparison = _comparison()
    responses = {COMPARE: comparison}
    if fault == "api":
        responses = {}
    elif fault in {"empty", "json", "multiple-json"}:
        responses = {
            COMPARE: {"_raw": {"empty": "", "json": "not json", "multiple-json": "{} {}"}[fault]}
        }
    elif fault in {"missing-author", "missing-committer"}:
        comparison["commits"][0][fault.removeprefix("missing-")] = None
    elif fault == "truncated":
        comparison["total_commits"] = 2
    elif fault == "duplicate":
        comparison["commits"].append(_commit())
        comparison["total_commits"] = 2
    elif fault == "no-head":
        comparison["commits"][0]["sha"] = OTHER
    elif fault == "wrong-base":
        comparison["base_commit"]["sha"] = OTHER
    elif fault == "missing-total":
        del comparison["total_commits"]
    elif fault in {"bot", "web-flow-committer"}:
        comparison["commits"][0] = (
            _commit(author="dependabot[bot]") if fault == "bot" else _commit(committer="web-flow")
        )
    # Push does not consult the optional PR approval path.
    _assert_decision(
        tmp_path,
        expected=False,
        event="push",
        payload=payload,
        responses=responses,
    )


@pytest.mark.parametrize("outsider", [False, True])
def test_every_page_is_checked(tmp_path: Path, outsider: bool):
    """A later-page outsider cannot hide behind the first 100 owner commits."""
    commits = [_commit(f"{i:040x}") for i in range(100, 200)]
    responses = {
        COMPARE: _comparison(commits, total=101),
        f"compare/{BASE}...{HEAD}?per_page=100&page=2": _comparison(
            [_commit(author="collaborator" if outsider else "ll7")], total=101
        ),
    }
    _assert_decision(tmp_path, expected=not outsider, event="push", responses=responses)


def _approval(*, actor: str = "ll7", head: str = HEAD, present: bool = True):
    """Live SHA label and historical authority are separate API evidence."""
    label = f"ci-owner:{head}"
    pr = copy.deepcopy(_event()["pull_request"])
    pr.update(state="open", labels=[{"name": label}] if present else [])
    return {
        COMPARE: _comparison([_commit(OTHER, author="collaborator"), _commit()]),
        "pulls/42": pr,
        "issues/42/events?per_page=100&page=1": [
            {
                "event": "labeled",
                "label": {"name": label},
                "actor": {"login": actor, "type": "User"},
            },
        ],
    }


@pytest.mark.parametrize(
    "case",
    [
        "owner",
        "collaborator",
        "stale",
        "removed",
        "missing-event",
        "reapplied",
        "history-api",
        "history-json",
        "changed-head",
    ],
)
def test_only_live_owner_approval_of_exact_head_grants_exception(tmp_path: Path, case: str):
    """Presence alone, third-party relabeling and old-head approval are insufficient."""
    responses = _approval(
        actor="collaborator" if case == "collaborator" else "ll7",
        head=OTHER if case == "stale" else HEAD,
        present=case != "removed",
    )
    history = "issues/42/events?per_page=100&page=1"
    if case == "missing-event":
        responses[history] = []
    elif case == "reapplied":
        responses[history].append(_approval(actor="collaborator")[history][0])
    elif case == "history-api":
        del responses[history]
    elif case == "history-json":
        responses[history] = {"_raw": ""}
    elif case == "changed-head":
        responses["pulls/42"]["head"]["sha"] = OTHER
    _assert_decision(tmp_path, expected=case == "owner", responses=responses)


@pytest.mark.parametrize("value", ["false", "", "true "])
def test_gate_failure_or_missing_output_routes_hosted(tmp_path: Path, value: str):
    """Every routed job requires an exact true from hosted admission."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]
    for name in ROUTED_JOBS:
        assert (
            _resolve_runs_on(jobs[name]["runs-on"], _github_context("push"), provenance=value)
            == "ubuntu-latest"
        )
        assert "self-hosted-admission" in jobs[name]["needs"]
        # Failed admission jobs must not cause bot/other-author CI to be skipped.
        assert jobs[name]["if"].startswith(
            "always() && needs.dispatch-ownership.result == 'success'"
        )


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_start_hook_independently_blocks_collaborator_range(tmp_path: Path, event: str):
    """A workflow selecting the private label directly cannot bypass the hook."""
    result = _run_hook(
        tmp_path,
        event_name=event,
        responses={COMPARE: _comparison([_commit(OTHER, author="collaborator"), _commit()])},
    )
    assert result.returncode == 1, result.stderr


@pytest.mark.parametrize("before", [None, "0" * 40, HEAD])
def test_unknown_push_range_stays_hosted(tmp_path: Path, before: str | None):
    """Missing, new-branch and empty ranges have no complete provenance proof."""
    payload = _event()
    payload["before"] = before
    _assert_decision(tmp_path, expected=False, event="push", payload=payload)


def test_mismatched_push_head_stays_hosted(tmp_path: Path):
    """The event range must bind the actual push run SHA."""
    _assert_decision(tmp_path, expected=False, event="push", changes={"sha": OTHER})


@pytest.mark.parametrize("fault", ["removed", "head", "api"])
def test_approval_rechecked_after_history_read(tmp_path: Path, fault: str):
    """A grant revoked or changed while history is paginated cannot admit steps."""
    responses = _approval()
    changed = copy.deepcopy(responses["pulls/42"])
    if fault == "removed":
        changed["labels"] = []
    elif fault == "head":
        changed["head"]["sha"] = OTHER
    else:
        changed = {"_raw": ""}
    responses["pulls/42"] = {"_sequence": [responses["pulls/42"], changed]}
    _assert_decision(tmp_path, expected=False, responses=responses)


@pytest.mark.parametrize("latest_actor", ["ll7", "collaborator"])
def test_approval_history_checked_through_last_page(tmp_path: Path, latest_actor: str):
    """Later label events determine authority, even after 100 earlier events."""
    responses = _approval()
    responses["issues/42/events?per_page=100&page=1"] += [{"event": "commented"}] * 99
    responses["issues/42/events?per_page=100&page=2"] = _approval(actor=latest_actor)[
        "issues/42/events?per_page=100&page=1"
    ]
    _assert_decision(tmp_path, expected=latest_actor == "ll7", responses=responses)


@pytest.mark.parametrize("fault", ["missing-page", "wrong-total", "repeated-page"])
def test_incomplete_pagination_never_grants_access(tmp_path: Path, fault: str):
    """Missing, inconsistent or repeated API pages cannot masquerade as a full range."""
    first = [_commit(f"{i:040x}") for i in range(100, 200)]
    responses = {COMPARE: _comparison(first, total=101)}
    if fault != "missing-page":
        responses[f"compare/{BASE}...{HEAD}?per_page=100&page=2"] = (
            _comparison([_commit()], total=102)
            if fault == "wrong-total"
            else _comparison(first, total=101)
        )
    _assert_decision(tmp_path, expected=False, event="push", responses=responses)


@pytest.mark.parametrize("fault", ["missing-base", "missing-head", "missing-author", "api"])
def test_exact_head_approval_cannot_override_missing_provenance(tmp_path: Path, fault: str):
    """Explicit approval is an identity exception, not a missing-evidence exception."""
    responses = _approval()
    payload = _event()
    if fault in {"missing-base", "missing-head"}:
        del payload["pull_request"][fault.removeprefix("missing-")]["sha"]
    elif fault == "missing-author":
        responses[COMPARE]["commits"][0]["author"] = None
    else:
        del responses[COMPARE]
    _assert_decision(tmp_path, expected=False, payload=payload, responses=responses)
