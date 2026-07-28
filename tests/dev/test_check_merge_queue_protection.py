"""Regression tests for the read-only merge-queue protection checker (issue #6404)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.check_merge_queue_protection import (
    DIM_BYPASS,
    DIM_CONVERSATION,
    DIM_GATE,
    DIM_MERGE_QUEUE,
    DIM_RUN,
    DIM_STRATEGY,
    GATE_CONTEXT,
    _ruleset_applies_to_branch,
    evaluate_protection,
    fetch_active_branch_rulesets,
    fetch_default_branch,
    fetch_merge_group_runs_total,
    main,
)
from scripts.dev.merge_queue_gate import GATE_JOB_NAME, GATE_WORKFLOW_NAME


def _gh_response(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Create a mock ``subprocess.CompletedProcess`` for GitHub CLI calls."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def _ruleset(
    *,
    rules: list[dict[str, Any]],
    bypass_actors: list[dict[str, Any]] | None = None,
    enforcement: str = "active",
    include: tuple[str, ...] = ("~DEFAULT_BRANCH",),
    target: str = "branch",
) -> dict[str, Any]:
    """Build a ruleset fixture in the shape returned by the GitHub rulesets API."""
    return {
        "id": 18917814,
        "name": "main-protection",
        "target": target,
        "enforcement": enforcement,
        "conditions": {"ref_name": {"include": list(include), "exclude": []}},
        "rules": rules,
        "bypass_actors": list(bypass_actors or []),
    }


def _merge_queue_rule() -> dict[str, Any]:
    return {"type": "merge_queue", "parameters": {}}


def _status_checks_rule(contexts: list[str]) -> dict[str, Any]:
    return {
        "type": "required_status_checks",
        "parameters": {
            "strict": True,
            "required_status_checks": [{"context": context} for context in contexts],
        },
    }


def _pull_request_rule(*, conversation_resolution: bool) -> dict[str, Any]:
    return {
        "type": "pull_request",
        "parameters": {"required_review_thread_resolution": conversation_resolution},
    }


def _configured_ruleset(
    *, gate_contexts: list[str] | None = None, bypass_actors=None, conversation=True
) -> dict[str, Any]:
    """Build a fully-configured main ruleset (queue required + gate + resolution)."""
    return _ruleset(
        rules=[
            _merge_queue_rule(),
            _status_checks_rule(gate_contexts if gate_contexts is not None else [GATE_CONTEXT]),
            _pull_request_rule(conversation_resolution=conversation),
        ],
        bypass_actors=bypass_actors,
    )


def _dimension(audit, key: str):
    """Return one dimension from an audit by key."""
    for entry in audit.dimensions:
        if entry.key == key:
            return entry
    raise AssertionError(f"dimension {key} missing from audit")


def test_gate_context_reuses_merge_queue_gate_constants() -> None:
    """The required-check context tracks the gate workflow without editing its file."""
    assert GATE_CONTEXT == f"{GATE_WORKFLOW_NAME} / {GATE_JOB_NAME}"


def test_evaluate_protection_passes_only_when_all_dimensions_satisfied() -> None:
    """Every activation dimension green is required for a fail-closed pass."""
    audit = evaluate_protection(
        rulesets=[_configured_ruleset()],
        strategy=("ALLGREEN", None),
        merge_group_runs=1,
    )
    assert audit.passed is True
    assert audit.reasons == []
    assert {d.key for d in audit.dimensions} == {
        DIM_MERGE_QUEUE,
        DIM_GATE,
        DIM_STRATEGY,
        DIM_CONVERSATION,
        DIM_BYPASS,
        DIM_RUN,
    }


def test_evaluate_protection_reports_current_unconfigured_state() -> None:
    """The current repo state (no queue, no gate, zero runs) fails closed honestly."""
    audit = evaluate_protection(
        rulesets=[
            _ruleset(
                rules=[
                    {"type": "deletion"},
                    {"type": "non_fast_forward"},
                    _pull_request_rule(conversation_resolution=False),
                ]
            )
        ],
        strategy=(None, None),
        merge_group_runs=0,
    )
    assert audit.passed is False
    assert DIM_MERGE_QUEUE in audit.reasons
    assert DIM_GATE in audit.reasons
    assert DIM_STRATEGY in audit.reasons
    assert DIM_CONVERSATION in audit.reasons
    assert DIM_RUN in audit.reasons
    # Bypass is already prohibited today; report it honestly as satisfied.
    assert DIM_BYPASS not in audit.reasons
    assert _dimension(audit, DIM_BYPASS).status == "satisfied"
    assert _dimension(audit, DIM_STRATEGY).reason == "merge_queue_not_required"
    assert audit.merge_group_runs_total == 0
    assert audit.ruleset_count == 1


def test_evaluate_protection_gate_absent_fails_closed() -> None:
    """A required_status_checks rule without the gate context fails the dimension."""
    audit = evaluate_protection(
        rulesets=[_configured_ruleset(gate_contexts=["unrelated-check"])],
        strategy=("ALLGREEN", None),
        merge_group_runs=1,
    )
    assert audit.passed is False
    assert _dimension(audit, DIM_GATE).reason == "gate_context_not_required"


def test_evaluate_protection_headgreen_strategy_fails_closed() -> None:
    """HEADGREEN lets an earlier failing entry hitchhike; fail closed."""
    audit = evaluate_protection(
        rulesets=[_configured_ruleset()],
        strategy=("HEADGREEN", None),
        merge_group_runs=1,
    )
    assert audit.passed is False
    assert _dimension(audit, DIM_STRATEGY).reason == "unsafe_strategy:HEADGREEN"


def test_evaluate_protection_conversation_resolution_off_fails_closed() -> None:
    """Conversation resolution must be required for the dimension to be satisfied."""
    audit = evaluate_protection(
        rulesets=[_configured_ruleset(conversation=False)],
        strategy=("ALLGREEN", None),
        merge_group_runs=1,
    )
    assert audit.passed is False
    assert DIM_CONVERSATION in audit.reasons


def test_evaluate_protection_bypass_allowed_fails_closed() -> None:
    """A configured bypass actor fails the bypass-prohibited dimension."""
    bypass = [{"actor_id": 1, "actor_type": "Admin"}]
    audit = evaluate_protection(
        rulesets=[_configured_ruleset(bypass_actors=bypass)],
        strategy=("ALLGREEN", None),
        merge_group_runs=1,
    )
    assert audit.passed is False
    assert _dimension(audit, DIM_BYPASS).reason == "bypass_actors_present:1"


def test_evaluate_protection_strategy_query_error_is_not_verifiable() -> None:
    """A failed live strategy probe fails closed as not_verifiable."""
    audit = evaluate_protection(
        rulesets=[_configured_ruleset()],
        strategy=(None, "merge queue strategy missing or unsupported"),
        merge_group_runs=1,
    )
    assert audit.passed is False
    assert _dimension(audit, DIM_STRATEGY).status == "not_verifiable"
    assert audit.strategy_probed is True


def test_evaluate_protection_strategy_not_probed_when_queue_required_is_not_verifiable() -> None:
    """A required queue without an enqueued PR cannot prove ALLGREEN."""
    audit = evaluate_protection(
        rulesets=[_configured_ruleset()],
        strategy=(None, None),
        merge_group_runs=1,
    )
    assert _dimension(audit, DIM_STRATEGY).reason == "strategy_not_probed_no_enqueued_pr"
    assert audit.strategy_probed is False


def test_evaluate_protection_ruleset_fetch_error_makes_config_dims_not_verifiable() -> None:
    """An unreadable ruleset list fails every config dimension closed."""
    audit = evaluate_protection(
        rulesets=[],
        strategy=(None, None),
        merge_group_runs=0,
        ruleset_fetch_error="rulesets list query failed",
    )
    assert audit.passed is False
    config_keys = {DIM_MERGE_QUEUE, DIM_GATE, DIM_STRATEGY, DIM_CONVERSATION, DIM_BYPASS}
    assert config_keys.issubset(audit.reasons)
    assert "rulesets list query failed" in audit.fetch_errors


def test_evaluate_protection_merge_group_runs_error_is_not_verifiable() -> None:
    """A failed run-count query fails the run-recorded dimension as not_verifiable."""
    audit = evaluate_protection(
        rulesets=[_configured_ruleset()],
        strategy=("ALLGREEN", None),
        merge_group_runs=0,
        merge_group_runs_error="merge_group runs query failed",
    )
    assert _dimension(audit, DIM_RUN).status == "not_verifiable"
    assert "merge_group runs query failed" in audit.fetch_errors


def test_evaluate_protection_records_partial_fetch_errors() -> None:
    """Partial fetch errors are surfaced for diagnostics without changing the verdict."""
    audit = evaluate_protection(
        rulesets=[_configured_ruleset()],
        strategy=("ALLGREEN", None),
        merge_group_runs=1,
        fetch_errors=["ruleset 99 detail fetch failed"],
    )
    assert audit.passed is True
    assert "ruleset 99 detail fetch failed" in audit.fetch_errors


def test_evaluate_protection_legacy_conversation_field_name_is_recognized() -> None:
    """The legacy ``required_conversation_resolution`` name also satisfies the dimension."""
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
        strategy=("ALLGREEN", None),
        merge_group_runs=2,
    )
    assert _dimension(audit, DIM_CONVERSATION).status == "satisfied"


@pytest.mark.parametrize(
    ("ruleset", "applies"),
    [
        (_ruleset(rules=[{"type": "deletion"}]), True),
        (_ruleset(rules=[{"type": "deletion"}], include=("main",)), True),
        (_ruleset(rules=[{"type": "deletion"}], include=("*",)), True),
        (_ruleset(rules=[{"type": "deletion"}], include=("feature",)), False),
        (_ruleset(rules=[{"type": "deletion"}], target="tag"), False),
    ],
)
def test_ruleset_applies_to_branch_respects_conditions(
    ruleset: dict[str, Any], applies: bool
) -> None:
    """Only active branch rulesets matching the default branch are inspected."""
    assert _ruleset_applies_to_branch(ruleset, "main") is applies


def test_fetch_default_branch_detects_branch() -> None:
    """The default branch is read from the repository metadata."""
    with patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout="main")
        branch, error = fetch_default_branch("owner/repo")
    assert (branch, error) == ("main", None)
    assert mock_gh.call_args.args[0] == ["api", "repos/owner/repo", "--jq", ".default_branch"]


def test_fetch_active_branch_rulesets_filters_active_main_rulesets() -> None:
    """Only active branch rulesets applying to the default branch are returned."""
    summary_active = {"id": 1, "target": "branch", "enforcement": "active"}
    summary_disabled = {"id": 2, "target": "branch", "enforcement": "disabled"}
    summary_tag = {"id": 3, "target": "tag", "enforcement": "active"}
    full = _ruleset(rules=[_merge_queue_rule()])
    with patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps([summary_active, summary_disabled, summary_tag])),
            _gh_response(stdout=json.dumps(full)),
        ]
        rulesets, list_error, partial = fetch_active_branch_rulesets("owner/repo", "main")
    assert list_error is None
    assert partial == []
    assert rulesets == [full]


def test_fetch_active_branch_rulesets_returns_list_error_on_failure() -> None:
    """A failed ruleset listing surfaces a list_error that drives fail-closed dims."""
    with patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="forbidden")
        rulesets, list_error, partial = fetch_active_branch_rulesets("owner/repo", "main")
    assert rulesets == []
    assert list_error is not None
    assert "forbidden" in list_error
    assert partial == []


def test_fetch_active_branch_rulesets_records_partial_detail_errors() -> None:
    """A ruleset whose detail fetch fails is skipped with a recorded partial error."""
    summary = {"id": 1, "target": "branch", "enforcement": "active"}
    with patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=json.dumps([summary])),
            _gh_response(returncode=1, stderr="server error"),
        ]
        rulesets, list_error, partial = fetch_active_branch_rulesets("owner/repo", "main")
    assert rulesets == []
    assert list_error is None
    assert len(partial) == 1
    assert "ruleset 1" in partial[0]


def test_fetch_merge_group_runs_total_reads_total_count() -> None:
    """The repo-wide merge_group run count is parsed from the actions API."""
    with patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=json.dumps({"total_count": 7}))
        total, error = fetch_merge_group_runs_total("owner/repo")
    assert (total, error) == (7, None)


def test_fetch_merge_group_runs_total_returns_error_on_failure() -> None:
    """A failed run-count query returns a not_verifiable-driving error."""
    with patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="unavailable")
        total, error = fetch_merge_group_runs_total("owner/repo")
    assert (total, error) == (0, "unavailable")


def _current_state_gh_side_effect() -> list[MagicMock]:
    """Return ordered _gh responses reproducing the current repo state."""
    summary = {"id": 18917814, "target": "branch", "enforcement": "active"}
    full = _ruleset(
        rules=[
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            _pull_request_rule(conversation_resolution=False),
        ]
    )
    return [
        _gh_response(stdout="main"),  # default branch
        _gh_response(stdout=json.dumps([summary])),  # rulesets list
        _gh_response(stdout=json.dumps(full)),  # ruleset detail
        _gh_response(stdout=json.dumps({"total_count": 0})),  # merge_group runs
    ]


def test_main_check_reports_current_state_and_exits_nonzero(capsys) -> None:
    """``--check`` reports the verified inactive state and fails closed."""
    with (
        patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh,
        patch(
            "scripts.dev.check_merge_queue_protection.fetch_merge_queue_strategy"
        ) as mock_strategy,
    ):
        mock_gh.side_effect = _current_state_gh_side_effect()
        exit_code = main(["--check", "--repo", "owner/repo"])
    audit = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert audit["passed"] is False
    assert audit["merge_group_runs_total"] == 0
    assert "merge_queue_required" in audit["reasons"]
    assert "gate_required_status_check" in audit["reasons"]
    assert "merge_group_run_recorded" in audit["reasons"]
    assert "bypass_prohibited" not in audit["reasons"]
    assert mock_strategy.call_count == 0


def test_main_check_does_not_probe_strategy_without_pr() -> None:
    """Without ``--pr`` the live strategy helper is not invoked."""
    with (
        patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh,
        patch(
            "scripts.dev.check_merge_queue_protection.fetch_merge_queue_strategy"
        ) as mock_strategy,
    ):
        mock_gh.side_effect = _current_state_gh_side_effect()
        main(["--check", "--repo", "owner/repo"])
    assert mock_strategy.call_count == 0


def test_main_check_probes_strategy_with_pr() -> None:
    """``--check --pr`` probes the live ALLGREEN strategy via the reused helper."""
    full = _configured_ruleset()
    summary = {"id": 18917814, "target": "branch", "enforcement": "active"}
    with (
        patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh,
        patch(
            "scripts.dev.check_merge_queue_protection.fetch_merge_queue_strategy"
        ) as mock_strategy,
    ):
        mock_gh.side_effect = [
            _gh_response(stdout="main"),
            _gh_response(stdout=json.dumps([summary])),
            _gh_response(stdout=json.dumps(full)),
            _gh_response(stdout=json.dumps({"total_count": 1})),
        ]
        mock_strategy.return_value = ("ALLGREEN", None)
        exit_code = main(["--check", "--repo", "owner/repo", "--pr", "42"])
    assert exit_code == 0
    mock_strategy.assert_called_once_with(42, repo="owner/repo")


def test_main_self_test_exits_zero() -> None:
    """``--self-test`` runs the offline assertions and exits 0."""
    with patch("scripts.dev.check_merge_queue_protection._gh") as mock_gh:
        exit_code = main(["--self-test"])
    assert exit_code == 0
    assert mock_gh.call_count == 0


def test_main_check_missing_repo_exits_nonzero(capsys) -> None:
    """A missing repository identifier fails before any ruleset query."""
    with patch("scripts.dev.check_merge_queue_protection._resolve_owner_repo", return_value=None):
        exit_code = main(["--check", "--repo", ""])
    assert exit_code == 1
    assert "Failed to detect repository" in capsys.readouterr().err
