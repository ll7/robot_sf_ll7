"""PR claim and residual-owner regressions from issue #9852."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import pr_contract_check

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_9785_BODY = ROOT / "tests/fixtures/issue_9852_pr_9785_body.md"
CORE_ONLY_RECEIPT = """Readiness lane coverage summary (issue #9754)
  core lane:      ran
  optional lane:  skipped (no optional-extra changed files)
  extended lane:  NOT RUN
  NOT COVERED by this readiness run: tests/validation tests/maps
"""
ALL_READINESS_LANES_RECEIPT = """Readiness lane coverage summary (issue #9754)
  core lane:      ran
  optional lane:  ran
  extended lane:  ran (tests/validation tests/maps)
  uncovered roots: none remaining
"""


@pytest.mark.parametrize(
    "claim", ("The full suite passed from pr_ready_check.sh.", "PR readiness: all tests pass.")
)
def test_core_only_receipt_rejects_global_pass_claim(claim: str) -> None:
    blockers = pr_contract_check.check_readiness_claims(claim, CORE_ONLY_RECEIPT)
    assert any("core" in blocker.lower() and "full" in blocker.lower() for blocker in blockers)


@pytest.mark.parametrize(
    "claim",
    (
        "The core readiness lane passed; optional and extended lanes did not run.",
        "The full suite was not run; only core readiness passed.",
        "The full suite passed with ROBOT_SF_TEST_LANE=all scripts/dev/run_tests_parallel.sh --lane all; core readiness also passed.",
    ),
)
def test_core_only_receipt_accepts_accurately_scoped_claim(claim: str) -> None:
    assert pr_contract_check.check_readiness_claims(claim, CORE_ONLY_RECEIPT) == []


def test_last_complete_receipt_controls_the_claim() -> None:
    receipt = ALL_READINESS_LANES_RECEIPT + CORE_ONLY_RECEIPT
    blockers = pr_contract_check.check_readiness_claims(
        "Full suite passed from readiness.", receipt
    )
    assert blockers


def test_explicit_readiness_full_suite_claim_needs_a_receipt() -> None:
    blockers = pr_contract_check.check_readiness_claims("pr_ready_check.sh: all tests pass.", None)
    assert blockers


def test_hosted_checker_requires_evidence_for_global_claim_without_local_receipt() -> None:
    assert pr_contract_check.check_readiness_claims("Full suite passed.", None)
    assert (
        pr_contract_check.check_readiness_claims(
            "Full suite passed with ROBOT_SF_TEST_LANE=all scripts/dev/run_tests_parallel.sh --lane all.",
            None,
        )
        == []
    )


def test_unrelated_all_lane_mention_does_not_exempt_a_readiness_claim() -> None:
    body = "I did not run scripts/dev/run_tests_parallel.sh --lane all.\nFull suite passed from pr_ready_check.sh."
    assert pr_contract_check.check_readiness_claims(body, CORE_ONLY_RECEIPT)


def test_comment_reports_both_new_contract_failures() -> None:
    blockers = [
        "BLOCKER: [readiness-claim] core-only receipt cannot support full suite",
        "BLOCKER: [close-while-deferred] #42 still owns residual work",
    ]
    comment = pr_contract_check.build_comment_body(blockers, [], [], "🔴 FAILED")
    assert "| 11. Readiness claims | ❌" in comment
    assert "| 12. Close while deferred | ❌" in comment


def _v2_body(*, linked: str, deferred: str, owner: int | None) -> str:
    owner_list = f"[{owner}]" if owner is not None else "[]"
    return f"""## Linked Issues

- {linked} #42

## Follow-Up / Residual Scope

- Remaining validation work is pending.

<!-- pr-contract:v2
change_class: tooling
linked_issues:
  closes: {[42] if linked == "Closes" else []}
  relates: []
deferred_work:
  status: {deferred}
  issues: {owner_list}
  reason: "Remaining validation work is pending."
-->
"""


def test_historical_9785_body_flags_its_own_closed_residual_owner() -> None:
    body = HISTORICAL_9785_BODY.read_text()
    blockers = pr_contract_check.check_close_while_deferred(body, "ll7/robot_sf_ll7")
    assert any("9754" in blocker for blocker in blockers)
    # The historical body quotes an unfinished full-suite criterion; it does
    # not itself claim all tests passed. A positive claim added to that body
    # must trigger the second guard against its actual core-only receipt.
    assert pr_contract_check.check_readiness_claims(body, CORE_ONLY_RECEIPT) == []
    assert pr_contract_check.check_readiness_claims(
        body + "\nFull suite passed from pr_ready_check.sh.\n", CORE_ONLY_RECEIPT
    )


def test_closing_with_same_or_no_residual_owner_is_flagged() -> None:
    same = _v2_body(linked="Closes", deferred="open", owner=42)
    ownerless = _v2_body(linked="Closes", deferred="open", owner=None)
    assert pr_contract_check.check_close_while_deferred(same, "ll7/robot_sf_ll7")
    assert pr_contract_check.check_close_while_deferred(ownerless, "ll7/robot_sf_ll7")


def test_refs_or_different_open_owner_is_accepted() -> None:
    refs = _v2_body(linked="Refs", deferred="open", owner=42)
    different = _v2_body(linked="Closes", deferred="open", owner=43)
    assert pr_contract_check.check_close_while_deferred(refs, "ll7/robot_sf_ll7") == []
    assert (
        pr_contract_check.check_close_while_deferred(
            different,
            "ll7/robot_sf_ll7",
            issue_is_open=lambda issue, repo: issue == 43 and repo == "ll7/robot_sf_ll7",
        )
        == []
    )


def test_different_closed_owner_is_flagged() -> None:
    body = _v2_body(linked="Closes", deferred="open", owner=43)
    blockers = pr_contract_check.check_close_while_deferred(
        body, "ll7/robot_sf_ll7", issue_is_open=lambda _issue, _repo: False
    )
    assert any("43" in blocker for blocker in blockers)


def test_completed_remaining_work_is_not_deferred() -> None:
    body = """## Linked Issues

- Closes #42

## Follow-Up / Residual Scope

No further code scope remains. This completes the remaining queue-recovery behavior.
"""
    assert pr_contract_check.check_close_while_deferred(body, "ll7/robot_sf_ll7") == []


def test_narrow_prose_owner_is_accepted_when_v2_owner_list_is_empty() -> None:
    body = _v2_body(linked="Closes", deferred="open", owner=None).replace(
        "Remaining validation work is pending.",
        "Remaining validation work remains under #43.",
        1,
    )
    assert (
        pr_contract_check.check_close_while_deferred(
            body,
            "ll7/robot_sf_ll7",
            issue_is_open=lambda issue, _repo: issue == 43,
        )
        == []
    )


def test_no_deferred_implementation_work_remains_is_not_a_residual() -> None:
    body = """## Linked Issues

- Closes #42

## Follow-Up / Residual Scope

No deferred implementation work remains. Manual recovery is outside this preflight.
"""
    assert pr_contract_check.check_close_while_deferred(body, "ll7/robot_sf_ll7") == []
