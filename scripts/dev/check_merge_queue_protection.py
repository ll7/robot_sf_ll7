#!/usr/bin/env python3
"""Read-only merge-queue activation checker for issue #6404 (parent #6274).

This is a **SUPPORTING TOOL ONLY**. It lets a maintainer confirm whether the
issue #6404 activation has been performed and lets reviewers detect when the
native merge queue is enabled. It does **not** modify GitHub branch protection
or rulesets (a maintainer repository-admin operation that is not doable from a
worktree and not authorized here), it does **not** produce a real ``merge_group``
run, and it does **not** close #6404 or justify any fail-closed enforcement
claim for the parent #6274. #6274's enforcement stays explicitly unproven until
a maintainer activates the queue, records a real ``merge_group`` run, and runs a
fail-closed negative queue probe.

The #6404 activation dimensions verified by this checker are:

  1. ``merge_queue_required``: an active branch ruleset on the default branch
     enforces a ``merge_queue`` rule.
  2. ``gate_required_status_check``: the ``Merge Queue Gate / merge-queue-gate``
     context appears in a ``required_status_checks`` rule.
  3. ``strategy_allgreen``: the live merge-queue merging strategy is ``ALLGREEN``,
     read through ``scripts.dev.merge_queue_gate.fetch_merge_queue_strategy``
     (reused by import; that file is not modified). This is only verifiable when a
     PR is enqueued, so pass ``--pr`` once a maintainer queues a candidate.
  4. ``conversation_resolution_required``: a ``pull_request`` rule requires
     conversation resolution before merging.
  5. ``bypass_prohibited``: no bypass actor is configured on the inspected
     rulesets (admins / direct maintainers cannot bypass the rules).
  6. ``merge_group_run_recorded``: at least one ``merge_group`` workflow run has
     ever executed repo-wide.

``--check`` (live) queries these dimensions through read-only GitHub API calls
and exits non-zero when any dimension is unsatisfied or unverifiable, so a
maintainer sees a clear fail-closed signal while the activation is incomplete.
``--self-test`` (offline) runs deterministic assertions over fixed fixtures,
mirroring ``scripts/dev/merge_queue_gate.py --self-test``.

The pure evaluator ``evaluate_protection`` is deterministic and exercised by
both ``--self-test`` and the live ``--check`` path so the contract is identical.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# Make the sibling ``scripts.dev`` package importable when this file is run as a
# standalone script (``python scripts/dev/check_merge_queue_protection.py``).
# Under pytest or ``uv run`` the project root is already on ``sys.path``; this
# insert is a no-op there and only matters for direct script invocation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dev.merge_queue_gate import (  # noqa: E402
    GATE_JOB_NAME,
    GATE_WORKFLOW_NAME,
    fetch_merge_queue_strategy,
)

AUDIT_SCHEMA = "check_merge_queue_protection.v1"
# Reuse the gate's own workflow/job names so the required-check context tracks
# the workflow file in ``.github/workflows/merge-queue-gate.yml`` without that
# file being modified here.
GATE_CONTEXT = f"{GATE_WORKFLOW_NAME} / {GATE_JOB_NAME}"
DEFAULT_BRANCH_REF = "~DEFAULT_BRANCH"
DIM_MERGE_QUEUE = "merge_queue_required"
DIM_GATE = "gate_required_status_check"
DIM_STRATEGY = "strategy_allgreen"
DIM_CONVERSATION = "conversation_resolution_required"
DIM_BYPASS = "bypass_prohibited"
DIM_RUN = "merge_group_run_recorded"
ALLGREEN = "ALLGREEN"
HEADGREEN = "HEADGREEN"


@dataclass(frozen=True, slots=True)
class Dimension:
    """One #6404 activation dimension and its verified state.

    ``status`` is one of ``satisfied``, ``not_satisfied``, or
    ``not_verifiable``. The authoritative gate signal is ``satisfied``: an
    unverifiable dimension is treated as not satisfied (fail closed), and
    ``reason`` explains why the dimension could not be confirmed.
    """

    key: str
    satisfied: bool
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class ProtectionAudit:
    """Inspectable, reproducible #6404 activation verdict for one check."""

    schema: str
    dimensions: list[Dimension]
    passed: bool
    reasons: list[str]
    gate_context: str = GATE_CONTEXT
    repo: str = ""
    default_branch: str = ""
    pr: int | None = None
    strategy_value: str | None = None
    strategy_error: str | None = None
    strategy_probed: bool = False
    ruleset_count: int = 0
    bypass_actor_count: int = 0
    merge_group_runs_total: int = 0
    fetch_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit as a plain JSON-able dict."""
        return asdict(self)


def _iter_rules(rulesets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the flat list of rule dicts across all rulesets."""
    rules: list[dict[str, Any]] = []
    for ruleset in rulesets:
        raw_rules = ruleset.get("rules") if isinstance(ruleset, dict) else None
        if not isinstance(raw_rules, list):
            continue
        for rule in raw_rules:
            if isinstance(rule, dict):
                rules.append(rule)
    return rules


def _has_rule_type(rulesets: list[dict[str, Any]], rule_type: str) -> bool:
    """Return whether any inspected ruleset enforces ``rule_type``."""
    return any(rule.get("type") == rule_type for rule in _iter_rules(rulesets))


def _required_check_contexts(rulesets: list[dict[str, Any]]) -> list[str]:
    """Return every required status-check context across inspected rulesets."""
    contexts: list[str] = []
    for rule in _iter_rules(rulesets):
        if rule.get("type") != "required_status_checks":
            continue
        parameters = rule.get("parameters")
        if not isinstance(parameters, dict):
            continue
        entries = parameters.get("required_status_checks")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                context = entry.get("context")
                if isinstance(context, str) and context:
                    contexts.append(context)
    return contexts


def _conversation_resolution_required(rulesets: list[dict[str, Any]]) -> bool:
    """Return whether a pull_request rule requires conversation resolution.

    GitHub rulesets expose this as ``required_review_thread_resolution``; the
    legacy branch-protection surface used ``required_conversation_resolution``.
    Accept either name so the checker stays correct across API-surface drift.
    """
    for rule in _iter_rules(rulesets):
        if rule.get("type") != "pull_request":
            continue
        parameters = rule.get("parameters")
        if not isinstance(parameters, dict):
            continue
        if parameters.get("required_review_thread_resolution") is True:
            return True
        if parameters.get("required_conversation_resolution") is True:
            return True
    return False


def _bypass_actor_count(rulesets: list[dict[str, Any]]) -> int:
    """Return the total number of bypass actors across inspected rulesets."""
    total = 0
    for ruleset in rulesets:
        if not isinstance(ruleset, dict):
            continue
        actors = ruleset.get("bypass_actors")
        if isinstance(actors, list):
            total += len(actors)
    return total


def _unsatisfied_dimension_keys(dimensions: list[Dimension]) -> list[str]:
    """Return the keys of dimensions that are not satisfied (fail-closed set)."""
    reasons: list[str] = []
    for dimension in dimensions:
        if not dimension.satisfied:
            reasons.append(dimension.key)
    return reasons


def _merge_queue_dimension(ruleset_fetch_error: str | None, merge_queue_present: bool) -> Dimension:
    """Build the ``merge_queue_required`` dimension from parsed ruleset state."""
    if ruleset_fetch_error:
        return Dimension(
            DIM_MERGE_QUEUE, False, "not_verifiable", f"ruleset_fetch_failed:{ruleset_fetch_error}"
        )
    if merge_queue_present:
        return Dimension(DIM_MERGE_QUEUE, True, "satisfied", "merge_queue_rule_present")
    return Dimension(DIM_MERGE_QUEUE, False, "not_satisfied", "merge_queue_rule_absent")


def _gate_dimension(ruleset_fetch_error: str | None, gate_present: bool) -> Dimension:
    """Build the ``gate_required_status_check`` dimension from parsed state."""
    if ruleset_fetch_error:
        return Dimension(
            DIM_GATE, False, "not_verifiable", f"ruleset_fetch_failed:{ruleset_fetch_error}"
        )
    if gate_present:
        return Dimension(DIM_GATE, True, "satisfied", "gate_context_required")
    return Dimension(DIM_GATE, False, "not_satisfied", "gate_context_not_required")


def _strategy_dimension(
    *,
    strategy_value: str | None,
    strategy_error: str | None,
    merge_queue_present: bool,
    ruleset_fetch_error: str | None,
) -> Dimension:
    """Build the ``strategy_allgreen`` dimension from the live strategy probe."""
    if strategy_error:
        return Dimension(
            DIM_STRATEGY, False, "not_verifiable", f"strategy_query_failed:{strategy_error}"
        )
    if strategy_value == ALLGREEN:
        return Dimension(DIM_STRATEGY, True, "satisfied", "allgreen")
    if strategy_value == HEADGREEN:
        return Dimension(DIM_STRATEGY, False, "not_satisfied", "unsafe_strategy:HEADGREEN")
    if ruleset_fetch_error:
        return Dimension(
            DIM_STRATEGY, False, "not_verifiable", f"ruleset_fetch_failed:{ruleset_fetch_error}"
        )
    if merge_queue_present:
        return Dimension(
            DIM_STRATEGY, False, "not_verifiable", "strategy_not_probed_no_enqueued_pr"
        )
    return Dimension(DIM_STRATEGY, False, "not_satisfied", "merge_queue_not_required")


def _conversation_dimension(
    ruleset_fetch_error: str | None, conversation_required: bool
) -> Dimension:
    """Build the ``conversation_resolution_required`` dimension from parsed state."""
    if ruleset_fetch_error:
        return Dimension(
            DIM_CONVERSATION,
            False,
            "not_verifiable",
            f"ruleset_fetch_failed:{ruleset_fetch_error}",
        )
    if conversation_required:
        return Dimension(DIM_CONVERSATION, True, "satisfied", "conversation_resolution_required")
    return Dimension(
        DIM_CONVERSATION, False, "not_satisfied", "conversation_resolution_not_required"
    )


def _bypass_dimension(ruleset_fetch_error: str | None, bypass_count: int) -> Dimension:
    """Build the ``bypass_prohibited`` dimension from parsed bypass actors."""
    if ruleset_fetch_error:
        return Dimension(
            DIM_BYPASS, False, "not_verifiable", f"ruleset_fetch_failed:{ruleset_fetch_error}"
        )
    if bypass_count == 0:
        return Dimension(DIM_BYPASS, True, "satisfied", "no_bypass_actors")
    return Dimension(DIM_BYPASS, False, "not_satisfied", f"bypass_actors_present:{bypass_count}")


def _run_dimension(runs_total: int, runs_error: str | None) -> Dimension:
    """Build the ``merge_group_run_recorded`` dimension from the run count."""
    if runs_error:
        return Dimension(DIM_RUN, False, "not_verifiable", f"runs_query_failed:{runs_error}")
    if runs_total > 0:
        return Dimension(DIM_RUN, True, "satisfied", f"merge_group_runs:{runs_total}")
    return Dimension(DIM_RUN, False, "not_satisfied", "zero_merge_group_runs")


def evaluate_protection(
    *,
    rulesets: list[dict[str, Any]] | None,
    strategy: tuple[str | None, str | None],
    merge_group_runs: int = 0,
    merge_group_runs_error: str | None = None,
    ruleset_fetch_error: str | None = None,
    fetch_errors: list[str] | None = None,
    gate_context: str = GATE_CONTEXT,
) -> ProtectionAudit:
    """Evaluate the #6404 activation dimensions over parsed GitHub state.

    Pure function: no side effects, no GitHub calls. Fail-closed by design: any
    unsatisfied or unverifiable dimension makes ``passed`` False.

    Inputs:
      rulesets: active branch rulesets applying to the default branch, each with
        ``rules`` and ``bypass_actors``. ``None``/empty is treated as "no
        protection read" (every config dimension fails closed).
      strategy: ``(value, error)`` from
        ``scripts.dev.merge_queue_gate.fetch_merge_queue_strategy``. Pass
        ``(None, None)`` when no enqueued PR was probed; the strategy dimension
        then reports ``not_verifiable`` when a merge_queue rule exists, or
        ``not_satisfied`` when the queue is not required at all.
      merge_group_runs: total repo-wide ``merge_group`` workflow run count.
      merge_group_runs_error: error string when the run-count query failed
        (dimension becomes ``not_verifiable``).
      ruleset_fetch_error: error string when the ruleset list query failed
        (every config dimension becomes ``not_verifiable``).
      fetch_errors: additional partial errors to record for diagnostics.

    Returns a ``ProtectionAudit`` with every dimension, ``passed``, and the
    unsatisfied dimension keys as ``reasons``.
    """
    rulesets = rulesets or []
    strategy_value, strategy_error = strategy
    runs_total = int(merge_group_runs or 0)
    merge_queue_present = _has_rule_type(rulesets, "merge_queue")
    gate_present = gate_context in _required_check_contexts(rulesets)
    conversation_required = _conversation_resolution_required(rulesets)
    bypass_count = _bypass_actor_count(rulesets)
    strategy_probed = strategy_value is not None or strategy_error is not None

    dimensions = [
        _merge_queue_dimension(ruleset_fetch_error, merge_queue_present),
        _gate_dimension(ruleset_fetch_error, gate_present),
        _strategy_dimension(
            strategy_value=strategy_value,
            strategy_error=strategy_error,
            merge_queue_present=merge_queue_present,
            ruleset_fetch_error=ruleset_fetch_error,
        ),
        _conversation_dimension(ruleset_fetch_error, conversation_required),
        _bypass_dimension(ruleset_fetch_error, bypass_count),
        _run_dimension(runs_total, merge_group_runs_error),
    ]

    reasons = _unsatisfied_dimension_keys(dimensions)
    all_fetch_errors = [ruleset_fetch_error, merge_group_runs_error]
    all_fetch_errors = [error for error in all_fetch_errors if error] + list(fetch_errors or [])

    return ProtectionAudit(
        schema=AUDIT_SCHEMA,
        dimensions=dimensions,
        passed=not reasons,
        reasons=reasons,
        gate_context=gate_context,
        strategy_value=strategy_value,
        strategy_error=strategy_error,
        strategy_probed=strategy_probed,
        ruleset_count=len(rulesets),
        bypass_actor_count=bypass_count,
        merge_group_runs_total=runs_total,
        fetch_errors=all_fetch_errors,
    )


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run a read-only ``gh`` command and return the completed process."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_json(stdout: str) -> tuple[Any, str | None]:
    """Parse JSON stdout into a Python object or return an error string."""
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse JSON: {exc}"


def _safe_int(value: Any) -> int | None:
    """Coerce a PR-number-like value to int, returning None when not parseable."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_owner_repo(explicit: str) -> str | None:
    """Resolve the ``owner/repo`` identifier, auto-detecting from gh when empty."""
    if explicit:
        return explicit
    result = _gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    if result.returncode != 0:
        return None
    repo = result.stdout.strip()
    return repo if repo else None


def _ruleset_applies_to_branch(ruleset: dict[str, Any], branch: str) -> bool:
    """Return whether an active branch ruleset's conditions match ``branch``."""
    if ruleset.get("target") != "branch":
        return False
    conditions = ruleset.get("conditions")
    if not isinstance(conditions, dict):
        # No conditions means the ruleset applies to every branch.
        return True
    ref_name = conditions.get("ref_name")
    if not isinstance(ref_name, dict):
        return True
    include = ref_name.get("include")
    exclude = ref_name.get("exclude")
    exclude_list = exclude if isinstance(exclude, list) else []
    if branch in exclude_list or "*" in exclude_list:
        return False
    include_list = include if isinstance(include, list) else []
    if not include_list:
        return True
    return branch in include_list or DEFAULT_BRANCH_REF in include_list or "*" in include_list


def fetch_default_branch(repo: str) -> tuple[str, str | None]:
    """Return the repository default branch name, or an error."""
    result = _gh(["api", f"repos/{repo}", "--jq", ".default_branch"])
    if result.returncode != 0:
        return "", result.stderr.strip() or "default branch query failed"
    branch = result.stdout.strip()
    return (branch or "main"), None


def _load_ruleset_detail(repo: str, ruleset_id: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch one ruleset's full detail, returning ``(ruleset, error)``."""
    detail_result = _gh(["api", f"repos/{repo}/rulesets/{ruleset_id}"])
    if detail_result.returncode != 0:
        stderr = detail_result.stderr.strip() or detail_result.returncode
        return None, f"ruleset {ruleset_id} detail fetch failed: {stderr}"
    ruleset, detail_error = _parse_json(detail_result.stdout)
    if detail_error or not isinstance(ruleset, dict):
        return None, f"ruleset {ruleset_id} detail is not JSON: {detail_error}"
    return ruleset, None


def _active_main_ruleset(ruleset: dict[str, Any], branch: str) -> bool:
    """Return whether a parsed ruleset is active and applies to ``branch``."""
    if str(ruleset.get("enforcement", "")).lower() != "active":
        return False
    return _ruleset_applies_to_branch(ruleset, branch)


def fetch_active_branch_rulesets(
    repo: str, branch: str
) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    """Return active branch rulesets applying to ``branch`` plus any errors.

    Returns ``(rulesets, list_error, partial_errors)``. ``list_error`` is set
    when the ruleset listing itself failed (drives fail-closed dimensions).
    ``partial_errors`` records individual ruleset-detail fetch failures. The
    caller must treat a non-empty list as an incomplete ruleset inventory and
    fail closed: an unread active ruleset could carry a bypass actor or a
    required protection rule that changes the activation verdict.
    """
    list_result = _gh(["api", f"repos/{repo}/rulesets"])
    if list_result.returncode != 0:
        return [], list_result.stderr.strip() or "rulesets list query failed", []
    summaries, error = _parse_json(list_result.stdout)
    if error or not isinstance(summaries, list):
        return [], error or "rulesets list response is not a JSON array", []

    active_main: list[dict[str, Any]] = []
    partial_errors: list[str] = []
    for summary in summaries:
        ruleset = _summarized_ruleset(summary)
        if ruleset is None:
            continue
        detail, detail_error = _load_ruleset_detail(repo, ruleset["id"])
        if detail is None:
            partial_errors.append(detail_error or "ruleset detail unavailable")
            continue
        if _active_main_ruleset(detail, branch):
            active_main.append(detail)
    return active_main, None, partial_errors


def _summarized_ruleset(summary: Any) -> dict[str, Any] | None:
    """Return an active branch-ruleset summary ``{"id": ...}`` or ``None``."""
    if not isinstance(summary, dict):
        return None
    if str(summary.get("enforcement", "")).lower() != "active":
        return None
    if summary.get("target") != "branch":
        return None
    ruleset_id = summary.get("id")
    if ruleset_id is None:
        return None
    return {"id": ruleset_id}


def fetch_merge_group_runs_total(repo: str) -> tuple[int, str | None]:
    """Return the total repo-wide ``merge_group`` workflow run count."""
    result = _gh(["api", f"repos/{repo}/actions/runs?event=merge_group"])
    if result.returncode != 0:
        return 0, result.stderr.strip() or "merge_group runs query failed"
    payload, error = _parse_json(result.stdout)
    if error or not isinstance(payload, dict):
        return 0, error or "merge_group runs response is not a JSON object"
    total = payload.get("total_count")
    if not isinstance(total, int):
        return 0, "merge_group runs total_count is missing or not an integer"
    return total, None


def _dimension_emoji(dimension: Dimension) -> str:
    """Return a compact status marker for the summary block."""
    if dimension.status == "satisfied":
        return "PASS"
    if dimension.status == "not_verifiable":
        return "UNVERIFIED"
    return "FAIL"


def _format_summary(audit: ProtectionAudit) -> str:
    """Format the audit as a compact GitHub step-summary block."""
    verdict = "ACTIVE" if audit.passed else "INCOMPLETE"
    pr_display = f"#{audit.pr}" if audit.pr else "n/a"
    lines = [
        f"### Merge-queue protection (#6404): {verdict}",
        "",
        f"- repository: `{audit.repo or '?'}`",
        f"- default branch: `{audit.default_branch or '?'}`",
        f"- required-check context: `{audit.gate_context}`",
        f"- inspected active rulesets: `{audit.ruleset_count}`",
        f"- bypass actors: `{audit.bypass_actor_count}`",
        f"- merge_group runs total: `{audit.merge_group_runs_total}`",
        f"- strategy probed: `{audit.strategy_probed}` "
        f"(value=`{audit.strategy_value}`, pr=`{pr_display}`)",
        "",
        "| dimension | status | reason |",
        "| --- | --- | --- |",
    ]
    for dimension in audit.dimensions:
        lines.append(
            f"| `{dimension.key}` | {_dimension_emoji(dimension)} | `{dimension.reason}` |"
        )
    if audit.fetch_errors:
        lines.append("")
        lines.append(f"- fetch errors: `{'; '.join(audit.fetch_errors)}`")
    lines.append("")
    lines.append(
        "Read-only supporting tool for issue #6404. It does not modify branch "
        "protection, does not enable the merge queue, and does not prove #6274 "
        "fail-closed enforcement. A maintainer must activate the queue, require "
        "`Merge Queue Gate / merge-queue-gate`, set ALLGREEN, require conversation "
        "resolution, prohibit bypass, and record a real merge_group run plus a "
        "fail-closed negative probe before #6404 can close."
    )
    return "\n".join(lines)


def _append_step_summary(text: str) -> None:
    """Append the summary block to ``GITHUB_STEP_SUMMARY`` when running in CI."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    except OSError:
        # Summary is best-effort; never fail the checker on a summary write error.
        pass


def _fixture_ruleset(
    *,
    rules: list[dict[str, Any]],
    bypass_actors: list[dict[str, Any]] | None = None,
    include: tuple[str, ...] = (DEFAULT_BRANCH_REF,),
) -> dict[str, Any]:
    return {
        "id": 18917814,
        "name": "main-protection",
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": list(include), "exclude": []}},
        "rules": rules,
        "bypass_actors": list(bypass_actors or []),
    }


def _fixture_merge_queue_rule() -> dict[str, Any]:
    return {"type": "merge_queue", "parameters": {}}


def _fixture_status_checks_rule(contexts: list[str]) -> dict[str, Any]:
    return {
        "type": "required_status_checks",
        "parameters": {
            "strict": True,
            "required_status_checks": [{"context": context} for context in contexts],
        },
    }


def _fixture_pull_request_rule(*, conversation_resolution: bool) -> dict[str, Any]:
    return {
        "type": "pull_request",
        "parameters": {"required_review_thread_resolution": conversation_resolution},
    }


def _run_self_test_scenarios() -> list[str]:
    """Evaluate the deterministic #6404 fixtures and return failure messages."""
    deletion = {"type": "deletion"}
    non_fast_forward = {"type": "non_fast_forward"}
    admin_bypass = [{"actor_id": 1, "actor_type": "Admin"}]
    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def dimension(audit: ProtectionAudit, key: str) -> Dimension:
        for entry in audit.dimensions:
            if entry.key == key:
                return entry
        failures.append(f"dimension {key} missing from audit")
        return Dimension(key, False, "not_satisfied", "missing")

    # Fully activated: every dimension satisfied -> pass.
    audit = evaluate_protection(
        rulesets=[
            _fixture_ruleset(
                rules=[
                    _fixture_merge_queue_rule(),
                    _fixture_status_checks_rule([GATE_CONTEXT]),
                    _fixture_pull_request_rule(conversation_resolution=True),
                ]
            )
        ],
        strategy=(ALLGREEN, None),
        merge_group_runs=1,
    )
    expect(audit.passed, "fully-activated: all dimensions satisfied must pass")
    expect(audit.reasons == [], "fully-activated: no fail-closed reasons when active")

    # Current verified state: queue not required, gate not required,
    # conversation resolution off, zero merge_group runs; bypass already empty.
    audit = evaluate_protection(
        rulesets=[
            _fixture_ruleset(
                rules=[
                    deletion,
                    non_fast_forward,
                    _fixture_pull_request_rule(conversation_resolution=False),
                ]
            )
        ],
        strategy=(None, None),
        merge_group_runs=0,
    )
    expect(not audit.passed, "current-state: incomplete activation must fail closed")
    expect(DIM_MERGE_QUEUE in audit.reasons, "current-state: merge_queue_required unsatisfied")
    expect(DIM_GATE in audit.reasons, "current-state: gate_required_status_check unsatisfied")
    expect(DIM_STRATEGY in audit.reasons, "current-state: strategy_allgreen unsatisfied")
    expect(
        DIM_CONVERSATION in audit.reasons,
        "current-state: conversation_resolution_required unsatisfied",
    )
    expect(DIM_RUN in audit.reasons, "current-state: merge_group_run_recorded unsatisfied")
    expect(
        DIM_BYPASS not in audit.reasons,
        "current-state: bypass already prohibited reports satisfied",
    )
    expect(
        dimension(audit, DIM_STRATEGY).reason == "merge_queue_not_required",
        "current-state: strategy reason is merge_queue_not_required",
    )
    expect(
        dimension(audit, DIM_BYPASS).status == "satisfied",
        "current-state: bypass_prohibited satisfied when no bypass actors",
    )

    # Gate absent from required_status_checks.
    audit = evaluate_protection(
        rulesets=[
            _fixture_ruleset(
                rules=[
                    _fixture_merge_queue_rule(),
                    _fixture_status_checks_rule(["other-check"]),
                    _fixture_pull_request_rule(conversation_resolution=True),
                ]
            )
        ],
        strategy=(ALLGREEN, None),
        merge_group_runs=1,
    )
    expect(not audit.passed, "gate-absent: must fail closed")
    expect(
        dimension(audit, DIM_GATE).reason == "gate_context_not_required",
        "gate-absent: gate reason is gate_context_not_required",
    )

    # HEADGREEN strategy is unsafe.
    audit = evaluate_protection(
        rulesets=[
            _fixture_ruleset(
                rules=[
                    _fixture_merge_queue_rule(),
                    _fixture_status_checks_rule([GATE_CONTEXT]),
                    _fixture_pull_request_rule(conversation_resolution=True),
                ]
            )
        ],
        strategy=(HEADGREEN, None),
        merge_group_runs=1,
    )
    expect(not audit.passed, "headgreen: HEADGREEN must fail closed")
    expect(
        dimension(audit, DIM_STRATEGY).reason == "unsafe_strategy:HEADGREEN",
        "headgreen: strategy reason names HEADGREEN",
    )

    # Conversation resolution off.
    audit = evaluate_protection(
        rulesets=[
            _fixture_ruleset(
                rules=[
                    _fixture_merge_queue_rule(),
                    _fixture_status_checks_rule([GATE_CONTEXT]),
                    _fixture_pull_request_rule(conversation_resolution=False),
                ]
            )
        ],
        strategy=(ALLGREEN, None),
        merge_group_runs=1,
    )
    expect(not audit.passed, "conversation-off: must fail closed")
    expect(
        DIM_CONVERSATION in audit.reasons, "conversation-off: conversation dimension unsatisfied"
    )

    # Admin/direct-maintainer bypass allowed.
    audit = evaluate_protection(
        rulesets=[
            _fixture_ruleset(
                rules=[
                    _fixture_merge_queue_rule(),
                    _fixture_status_checks_rule([GATE_CONTEXT]),
                    _fixture_pull_request_rule(conversation_resolution=True),
                ],
                bypass_actors=admin_bypass,
            )
        ],
        strategy=(ALLGREEN, None),
        merge_group_runs=1,
    )
    expect(not audit.passed, "bypass-allowed: must fail closed")
    expect(
        dimension(audit, DIM_BYPASS).reason == "bypass_actors_present:1",
        "bypass-allowed: bypass reason counts the configured actor",
    )

    # Strategy query failed -> not verifiable (fail closed).
    audit = evaluate_protection(
        rulesets=[
            _fixture_ruleset(
                rules=[
                    _fixture_merge_queue_rule(),
                    _fixture_status_checks_rule([GATE_CONTEXT]),
                    _fixture_pull_request_rule(conversation_resolution=True),
                ]
            )
        ],
        strategy=(None, "merge queue strategy missing or unsupported"),
        merge_group_runs=1,
    )
    expect(not audit.passed, "strategy-error: must fail closed")
    expect(
        dimension(audit, DIM_STRATEGY).status == "not_verifiable",
        "strategy-error: status is not_verifiable",
    )

    # Queue required but no enqueued PR probed -> strategy not verifiable.
    audit = evaluate_protection(
        rulesets=[
            _fixture_ruleset(
                rules=[
                    _fixture_merge_queue_rule(),
                    _fixture_status_checks_rule([GATE_CONTEXT]),
                    _fixture_pull_request_rule(conversation_resolution=True),
                ]
            )
        ],
        strategy=(None, None),
        merge_group_runs=1,
    )
    expect(
        dimension(audit, DIM_STRATEGY).reason == "strategy_not_probed_no_enqueued_pr",
        "no-pr: strategy reason is strategy_not_probed_no_enqueued_pr",
    )
    expect(audit.strategy_probed is False, "no-pr: strategy_probed is False")

    # Ruleset list fetch failed -> config dimensions not verifiable.
    audit = evaluate_protection(
        rulesets=[],
        strategy=(None, None),
        merge_group_runs=0,
        ruleset_fetch_error="rulesets list query failed",
    )
    expect(not audit.passed, "ruleset-fetch-error: must fail closed")
    expect(
        {DIM_MERGE_QUEUE, DIM_GATE, DIM_STRATEGY, DIM_CONVERSATION, DIM_BYPASS}.issubset(
            audit.reasons
        ),
        "ruleset-fetch-error: config dimensions are unsatisfied (not_verifiable)",
    )
    expect(
        "rulesets list query failed" in audit.fetch_errors,
        "ruleset-fetch-error: error recorded in fetch_errors",
    )

    # Legacy conversation-resolution field name is also recognized.
    audit = evaluate_protection(
        rulesets=[
            {
                "id": 1,
                "target": "branch",
                "enforcement": "active",
                "rules": [
                    {"type": "merge_queue"},
                    {
                        "type": "required_status_checks",
                        "parameters": {"required_status_checks": []},
                    },
                    {
                        "type": "pull_request",
                        "parameters": {"required_conversation_resolution": True},
                    },
                ],
                "bypass_actors": [],
            }
        ],
        strategy=(ALLGREEN, None),
        merge_group_runs=2,
    )
    expect(
        dimension(audit, DIM_CONVERSATION).status == "satisfied",
        "legacy-field: required_conversation_resolution=True satisfies the dimension",
    )

    # Audit carries the required inspectable context fields.
    audit = evaluate_protection(
        rulesets=[_fixture_ruleset(rules=[_fixture_merge_queue_rule()])],
        strategy=(ALLGREEN, None),
        merge_group_runs=3,
    )
    expect(audit.schema == AUDIT_SCHEMA, "audit: schema tag is check_merge_queue_protection.v1")
    expect(audit.merge_group_runs_total == 3, "audit: merge_group_runs_total recorded")
    expect(audit.ruleset_count == 1, "audit: inspected ruleset count recorded")
    expect(audit.strategy_value == ALLGREEN, "audit: strategy value recorded")
    expect(len(audit.dimensions) == 6, "audit: all six dimensions present")

    return failures


def _self_test() -> int:
    """Run deterministic assertions covering the issue #6404 contract.

    Exercises every required dimension pair (required vs not, present vs absent,
    ALLGREEN vs HEADGREEN, on vs off, allowed vs prohibited) plus the
    fail-closed not-verifiable paths. Exits 0 when all assertions hold and 1
    otherwise, mirroring ``merge_queue_gate.py --self-test``.
    """
    failures = _run_self_test_scenarios()
    if failures:
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1
    print("check_merge_queue_protection self-test: all assertions passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: evaluate #6404 activation and print the audit JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="query live GitHub protection state and report #6404 activation dimensions",
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic offline assertions over fixed fixtures and exit",
    )
    parser.add_argument("--repo", default="", help="owner/repo (default: detect from gh)")
    parser.add_argument(
        "--pr",
        default="",
        help=(
            "optional PR number enqueued in the merge queue, used to probe the "
            "live ALLGREEN strategy via fetch_merge_queue_strategy"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help="write only the step summary; suppress stdout audit JSON",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    repo = _resolve_owner_repo(args.repo)
    if not repo:
        print("Failed to detect repository. Pass --repo owner/repo.", file=sys.stderr)
        return 1

    branch, branch_error = fetch_default_branch(repo)
    if branch_error:
        print(
            f"warning: default branch query failed ({branch_error}); assuming 'main'",
            file=sys.stderr,
        )
        branch = "main"

    rulesets, list_error, partial_errors = fetch_active_branch_rulesets(repo, branch)
    runs_total, runs_error = fetch_merge_group_runs_total(repo)

    pr_number = _safe_int(args.pr)
    if pr_number is not None:
        strategy_value, strategy_error = fetch_merge_queue_strategy(pr_number, repo=repo)
        strategy: tuple[str | None, str | None] = (strategy_value, strategy_error)
    else:
        strategy = (None, None)

    # The activation contract applies to the repository's *actual* default
    # branch.  Falling back to ``main`` lets diagnostics continue after a
    # failed metadata lookup, but it cannot establish that the inspected
    # rulesets protect the real default branch.  Carry that failure into the
    # ruleset inventory verdict so this path cannot report activation as
    # complete merely because a branch named ``main`` happens to be protected.
    ruleset_inventory_errors = [error for error in (branch_error, list_error) if error]
    ruleset_inventory_errors.extend(partial_errors)
    ruleset_inventory_error = "; ".join(ruleset_inventory_errors) or None

    audit = evaluate_protection(
        rulesets=rulesets,
        strategy=strategy,
        merge_group_runs=runs_total,
        merge_group_runs_error=runs_error,
        ruleset_fetch_error=ruleset_inventory_error,
        fetch_errors=partial_errors,
    )
    audit = replace(audit, repo=repo, default_branch=branch, pr=pr_number)

    _append_step_summary(_format_summary(audit))
    if not args.summary_only:
        print(json.dumps(audit.to_dict()))
    if audit.fetch_errors:
        print(
            "protection check completed with fetch warnings: " + "; ".join(audit.fetch_errors),
            file=sys.stderr,
        )
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
