"""Focused tests for the versioned issue-completion receipt contract."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.dev.issue_audit_core import build_audit_plan, classify_issue
from scripts.dev.issue_completion_receipt import (
    VERIFICATION_SCHEMA,
    admit_completion_receipt,
    build_receipt,
    compute_receipt_digest,
    main,
    sha256_bytes,
    sha256_text,
    validate_receipt,
    verify_receipt_against_git,
)

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
NEW_HEAD_SHA = "c" * 40
CONTRACT = "Completion condition: merged PR #9000\n"
BRANCH = "issue-7614-completion-receipt"


def _payload(*, artifact_root: Path | None = None) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    validation_inputs: list[dict[str, Any]] = []
    if artifact_root is not None:
        report = artifact_root / "reports" / "completion.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"status": "complete"}\n', encoding="utf-8")
        artifacts.append(
            {
                "path": "reports/completion.json",
                "schema": "completion_report.v1",
                "digest": sha256_bytes(report.read_bytes()),
                "captured_head_sha": HEAD_SHA,
            }
        )
        source = artifact_root / "inputs" / "contract.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(CONTRACT, encoding="utf-8")
        validation_inputs.append(
            {
                "path": "inputs/contract.txt",
                "schema": "issue_contract_text.v1",
                "digest": sha256_bytes(source.read_bytes()),
                "captured_head_sha": HEAD_SHA,
            }
        )
    return {
        "repository": "ll7/robot_sf_ll7",
        "issue": 7614,
        "contract": {"source": "issue body", "digest": sha256_text(CONTRACT)},
        "delivery": {
            "base_ref": "origin/main",
            "base_sha": BASE_SHA,
            "branch": BRANCH,
            "head_sha": HEAD_SHA,
        },
        "covering_pr": {
            "number": 9000,
            "state": "OPEN",
            "head_sha": HEAD_SHA,
            "base_sha": BASE_SHA,
            "head_ref": BRANCH,
        },
        "diff": {
            "changed_paths": ["scripts/dev/issue_completion_receipt.py"],
            "stat": {"files": 1, "additions": 4, "deletions": 1},
        },
        "validation": [
            {
                "command": "pytest -q tests/dev -k completion_receipt",
                "status": "passed",
                "exit_code": 0,
                "summary": "focused receipt tests passed",
                "head_sha": HEAD_SHA,
            }
        ],
        "validation_inputs": validation_inputs,
        "artifacts": artifacts,
        "acceptance_criteria": [
            {
                "id": "exact-head",
                "disposition": "met",
                "evidence": ["Git-backed verifier matched base/head and diff"],
            },
            {
                "id": "independent-review",
                "disposition": "met",
                "evidence": ["independent verifier status is verified"],
            },
        ],
        "residuals": {
            "risks": ["Domain and scientific validity remain out of scope."],
            "deferred": [],
            "forbidden_claims": ["No benchmark result is claimed."],
        },
        "producer": {"identity": "agent-7614", "head_sha": HEAD_SHA},
        "independent_verifier": {
            "identity": "reviewer-7614",
            "status": "verified",
            "head_sha": HEAD_SHA,
        },
        "drift_policy": {
            "invalidate_on": ["head", "contract", "artifacts", "validation_inputs", "review"],
            "post_review": "invalidate_on_change",
        },
        "receipt_digest": None,
    }


def _receipt(*, artifact_root: Path | None = None) -> dict[str, Any]:
    """Build a valid receipt fixture."""
    return build_receipt(_payload(artifact_root=artifact_root))


def _verification(receipt: dict[str, Any]) -> dict[str, Any]:
    """Build the minimal exact-head verification result consumed by admission."""
    delivery = receipt["delivery"]
    return {
        "schema": VERIFICATION_SCHEMA,
        "ok": True,
        "receipt_digest": receipt["receipt_digest"],
        "base_sha": delivery["base_sha"],
        "head_sha": delivery["head_sha"],
        "branch": delivery["branch"],
        "git": {"diff": receipt["diff"]},
    }


def _git_runner(*, branch_sha: str = HEAD_SHA, head_exists: bool = True):
    """Return a deterministic runner for the Git-backed verifier."""

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        argument = command[-1]
        if "rev-parse" in command:
            if BASE_SHA in argument:
                return subprocess.CompletedProcess(command, 0, BASE_SHA, "")
            if HEAD_SHA in argument:
                if head_exists:
                    return subprocess.CompletedProcess(command, 0, HEAD_SHA, "")
                return subprocess.CompletedProcess(command, 128, "", "missing head")
            if "refs/heads/" in argument:
                return subprocess.CompletedProcess(command, 0, branch_sha, "")
            return subprocess.CompletedProcess(command, 128, "", "missing reference")
        if "--name-only" in command:
            return subprocess.CompletedProcess(
                command, 0, "scripts/dev/issue_completion_receipt.py\n", ""
            )
        if "--numstat" in command:
            return subprocess.CompletedProcess(
                command, 0, "4\t1\tscripts/dev/issue_completion_receipt.py\n", ""
            )
        raise AssertionError(f"unexpected Git command: {command}")

    return run


def test_build_and_validate_self_digested_receipt() -> None:
    """The offline builder emits the version and canonical self-digest."""
    receipt = _receipt()

    assert receipt["schema"] == "issue_completion_receipt.v1"
    assert receipt["receipt_digest"] == compute_receipt_digest(receipt)
    result = validate_receipt(receipt, expected_repository="ll7/robot_sf_ll7", expected_issue=7614)
    assert result["ok"] is True


def test_validation_states_are_explicit_and_admission_stays_fail_closed() -> None:
    """Skipped, unavailable, and failed checks do not authorize completion."""
    for status, exit_code in (("skipped", None), ("unavailable", None), ("failed", 7)):
        payload = _payload()
        payload["validation"][0]["status"] = status
        payload["validation"][0]["exit_code"] = exit_code
        receipt = build_receipt(payload)
        assert validate_receipt(receipt)["ok"] is True
        admission = admit_completion_receipt(
            {"receipt": receipt, "verification": _verification(receipt)},
            expected_repository="ll7/robot_sf_ll7",
            expected_issue=7614,
            issue_contract=CONTRACT,
        )
        assert admission["eligible"] is False
        assert status in admission["reason"] or status in " ".join(admission["errors"])


def test_duplicate_criterion_disposition_is_rejected() -> None:
    """Every criterion identifier must occur exactly once."""
    receipt = _receipt()
    invalid = copy.deepcopy(receipt)
    invalid["acceptance_criteria"].append(copy.deepcopy(invalid["acceptance_criteria"][0]))
    invalid["receipt_digest"] = compute_receipt_digest(invalid)

    result = validate_receipt(invalid)

    assert result["ok"] is False
    assert any("more than one disposition" in error for error in result["errors"])


def test_artifact_digest_drift_is_rejected(tmp_path: Path) -> None:
    """Changing a declared artifact after capture invalidates the receipt."""
    receipt = _receipt(artifact_root=tmp_path)
    report = tmp_path / "reports" / "completion.json"
    report.write_text('{"status": "changed"}\n', encoding="utf-8")

    result = validate_receipt(receipt, artifact_root=tmp_path)

    assert result["ok"] is False
    assert any("digest drift" in error for error in result["errors"])


def test_git_verifier_checks_exact_diff_and_pull_request_snapshot() -> None:
    """Git and PR snapshots must agree with the receipt's exact delivery."""
    receipt = _receipt()
    pr_snapshot = {
        "state": "open",
        "merged_at": None,
        "head": {"sha": HEAD_SHA, "ref": BRANCH},
        "base": {"sha": BASE_SHA},
    }

    result = verify_receipt_against_git(
        receipt,
        repo_root=Path("/tmp/receipt-test-repo"),
        repository="ll7/robot_sf_ll7",
        issue_contract=CONTRACT,
        pr_snapshot=pr_snapshot,
        git_runner=_git_runner(),
    )

    assert result["ok"] is True
    assert result["schema"] == VERIFICATION_SCHEMA
    assert result["git"]["diff"] == receipt["diff"]


def test_git_verifier_rejects_a_later_branch_head() -> None:
    """A branch that moved after review cannot reuse the earlier receipt."""
    receipt = _receipt()

    result = verify_receipt_against_git(
        receipt,
        repo_root=Path("/tmp/receipt-test-repo"),
        repository="ll7/robot_sf_ll7",
        issue_contract=CONTRACT,
        pr_snapshot={
            "state": "open",
            "head": {"sha": HEAD_SHA, "ref": BRANCH},
            "base": {"sha": BASE_SHA},
        },
        git_runner=_git_runner(branch_sha=NEW_HEAD_SHA),
    )

    assert result["ok"] is False
    assert any("moved" in error for error in result["errors"])


def test_git_verifier_rejects_missing_named_head() -> None:
    """A receipt cannot prove delivery when its named head no longer exists."""
    receipt = _receipt()

    result = verify_receipt_against_git(
        receipt,
        repo_root=Path("/tmp/receipt-test-repo"),
        repository="ll7/robot_sf_ll7",
        issue_contract=CONTRACT,
        pr_snapshot={
            "state": "open",
            "head": {"sha": HEAD_SHA, "ref": BRANCH},
            "base": {"sha": BASE_SHA},
        },
        git_runner=_git_runner(head_exists=False),
    )

    assert result["ok"] is False
    assert any("head commit does not exist" in error for error in result["errors"])


def test_git_verifier_rejects_covering_pr_state_drift() -> None:
    """A changed PR lifecycle state invalidates the receipt snapshot."""
    receipt = _receipt()

    result = verify_receipt_against_git(
        receipt,
        repo_root=Path("/tmp/receipt-test-repo"),
        repository="ll7/robot_sf_ll7",
        issue_contract=CONTRACT,
        pr_snapshot={
            "state": "closed",
            "head": {"sha": HEAD_SHA, "ref": BRANCH},
            "base": {"sha": BASE_SHA},
        },
        git_runner=_git_runner(),
    )

    assert result["ok"] is False
    assert any("PR OPEN" in error for error in result["errors"])


def test_admission_requires_independent_git_verification() -> None:
    """A producer-authored receipt alone is never enough for closure."""
    receipt = _receipt()

    pending = admit_completion_receipt(
        {"receipt": receipt},
        expected_repository="ll7/robot_sf_ll7",
        expected_issue=7614,
        issue_contract=CONTRACT,
    )
    accepted = admit_completion_receipt(
        {"receipt": receipt, "verification": _verification(receipt)},
        expected_repository="ll7/robot_sf_ll7",
        expected_issue=7614,
        issue_contract=CONTRACT,
    )

    assert pending["eligible"] is False
    assert "exact-head Git verification" in pending["reason"]
    assert accepted["eligible"] is True


def test_admission_rejects_untrusted_or_mismatched_verification_evidence() -> None:
    """A producer-shaped verification map cannot authorize completion by itself."""
    receipt = _receipt()
    verification = _verification(receipt)

    missing_schema = copy.deepcopy(verification)
    missing_schema.pop("schema")
    rejected_schema = admit_completion_receipt(
        {"receipt": receipt, "verification": missing_schema},
        expected_repository="ll7/robot_sf_ll7",
        expected_issue=7614,
        issue_contract=CONTRACT,
    )
    assert rejected_schema["eligible"] is False
    assert "verification.schema" in rejected_schema["reason"]

    mismatched_diff = copy.deepcopy(verification)
    mismatched_diff["git"]["diff"] = {"changed_paths": [], "stat": {}}
    rejected_diff = admit_completion_receipt(
        {"receipt": receipt, "verification": mismatched_diff},
        expected_repository="ll7/robot_sf_ll7",
        expected_issue=7614,
        issue_contract=CONTRACT,
    )
    assert rejected_diff["eligible"] is False
    assert "Git diff" in rejected_diff["reason"]


def test_issue_audit_withholds_close_without_receipt_and_accepts_verified_entry() -> None:
    """The canonical issue-close planner consumes the verified receipt gate."""
    issue = {
        "number": 7614,
        "title": "Require completion receipt",
        "state": "open",
        "body": CONTRACT,
        "labels": [],
        "comments": [],
    }
    merged_prs = [
        {
            "number": 9000,
            "title": "Fixes #7614",
            "merged_at": "2026-08-20T00:00:00Z",
            "linked_issue_numbers": [7614],
        }
    ]
    pending = classify_issue(issue, merged_prs=merged_prs, open_issue_numbers={7614})
    receipt = _receipt()
    accepted = classify_issue(
        issue,
        merged_prs=merged_prs,
        open_issue_numbers={7614},
        completion_receipt={"receipt": receipt, "verification": _verification(receipt)},
    )

    assert not any(mutation["operation"] == "close_issue" for mutation in pending.mutations)
    assert any("completion receipt" in finding for finding in pending.findings)
    close_mutations = [
        mutation for mutation in accepted.mutations if mutation["operation"] == "close_issue"
    ]
    assert len(close_mutations) == 1
    assert receipt["receipt_digest"] in " ".join(close_mutations[0]["evidence"])


def test_state_working_never_promotes_without_a_receipt() -> None:
    """The working qualifier blocks downstream ready promotion until receipt proof exists."""
    issue = {
        "number": 7614,
        "title": "Require completion receipt",
        "state": "open",
        "body": "## Definition of Done\n- [x] verify the change\n",
        "labels": ["state:ready", "state:working"],
        "comments": [],
    }

    classification = classify_issue(issue, available_labels={"state:ready", "state:working"})

    assert any(
        "state:working downstream promotion withheld" in finding
        for finding in classification.findings
    )
    assert not any(
        mutation["operation"] == "add_label" and mutation["value"] == "state:ready"
        for mutation in classification.mutations
    )
    assert any(
        mutation["operation"] == "remove_label" and mutation["value"] == "state:ready"
        for mutation in classification.mutations
    )


def test_audit_plan_accepts_issue_number_keyed_receipt_inventory() -> None:
    """Batch planning forwards receipt entries without inventing a new closure owner."""
    receipt = _receipt()
    plan = build_audit_plan(
        {
            "repo": "ll7/robot_sf_ll7",
            "issues": [
                {
                    "number": 7614,
                    "title": "Require completion receipt",
                    "state": "open",
                    "body": CONTRACT,
                    "labels": [],
                    "comments": [],
                }
            ],
            "merged_prs": [
                {
                    "number": 9000,
                    "title": "Fixes #7614",
                    "merged_at": "2026-08-20T00:00:00Z",
                    "linked_issue_numbers": [7614],
                }
            ],
            "open_prs": [],
            "claims": {},
            "worktrees": [],
            "jobs": [],
            "labels": [],
            "completion_receipts": {
                "7614": {"receipt": receipt, "verification": _verification(receipt)}
            },
            "inventory": {},
        }
    )

    assert plan["issues"][0]["closure_evidence"]["completion_receipt"]["eligible"] is True
    assert any(mutation["operation"] == "close_issue" for mutation in plan["mutations"])


def test_cli_build_and_offline_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The versioned helper is usable from the canonical script entry point."""
    input_path = tmp_path / "payload.json"
    output_path = tmp_path / "receipt.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")

    assert main(["build", "--input", str(input_path), "--output", str(output_path)]) == 0
    assert main(["verify", "--offline", "--receipt", str(output_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
