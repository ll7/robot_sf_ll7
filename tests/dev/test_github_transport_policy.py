"""Focused tests for the canonical GitHub helper transport policy."""

from __future__ import annotations

from pathlib import Path

from scripts.dev.github_transport_policy import (
    FAIL_CLOSED_ERROR_MARKERS,
    FALLBACK_ELIGIBLE_MARKERS,
    POLICY_SCHEMA,
    TRANSPORT_CONTRACTS,
    audit_helpers,
    check_helper,
    classify_error,
    get_transport_contract,
    is_fallback_eligible,
    main,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_registry_covers_current_github_helpers() -> None:
    """Every checked-in issue, PR, and comment helper has a complete contract."""
    result = audit_helpers(root=REPOSITORY_ROOT)

    assert result["schema"] == POLICY_SCHEMA
    assert result["status"] == "ok", result["findings"]
    assert result["discovered_helpers"] == sorted(TRANSPORT_CONTRACTS)
    assert not result["findings"]


def test_unregistered_helper_is_rejected(tmp_path: Path) -> None:
    """Adding a new gh helper without registration fails the audit."""
    scripts_dir = tmp_path / "scripts" / "dev"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "gh_new_helper.py").write_text("print('new')\n", encoding="utf-8")

    result = audit_helpers(root=tmp_path)

    assert result["status"] == "error"
    assert {
        finding["kind"] for finding in result["findings"] if finding["helper"] == "gh_new_helper.py"
    } == {"missing_registration"}


def test_registered_helper_requires_policy_reference(tmp_path: Path) -> None:
    """A registry entry is insufficient when the helper does not use the policy."""
    helper = tmp_path / "scripts" / "dev" / "gh_comment.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text("#!/usr/bin/env bash\necho comment\n", encoding="utf-8")

    result = check_helper("gh_comment.sh", root=tmp_path)
    kinds = {finding["kind"] for finding in result["findings"]}

    assert result["status"] == "error"
    assert "missing_policy_reference" in kinds
    assert "missing_smoke_test" in kinds


def test_label_helper_contract_requires_its_result_validator(tmp_path: Path) -> None:
    """The label transport contract must name an executable result validator."""
    helper = tmp_path / "scripts" / "dev" / "gh_pr_label_rest.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "# github_transport_policy\n# deliberately missing the response checker\n",
        encoding="utf-8",
    )
    smoke_test = tmp_path / "tests" / "dev" / "test_gh_pr_label_rest.py"
    smoke_test.parent.mkdir(parents=True)
    smoke_test.write_text("", encoding="utf-8")

    result = check_helper("gh_pr_label_rest.py", root=tmp_path)

    assert result["status"] == "error"
    assert {finding["kind"] for finding in result["findings"]} == {"missing_result_validator"}


def test_graphql_fallback_and_fail_closed_markers_are_disjoint() -> None:
    """Generic GraphQL fallback never masks an authentication or permission error."""
    assert not set(FALLBACK_ELIGIBLE_MARKERS) & set(FAIL_CLOSED_ERROR_MARKERS)
    assert is_fallback_eligible("GraphQL: repository.issue.projectCards")
    assert not is_fallback_eligible("GraphQL: forbidden")
    assert not is_fallback_eligible("connection reset by peer")


def test_helper_specific_fallback_policy_is_centralized() -> None:
    """The merge wrapper declares receipt-owner delegation and no fallback writer."""
    contract = get_transport_contract("gh_pr_merge.sh")
    decision = classify_error("gh_pr_merge.sh", "fatal: already used by worktree")
    quota = classify_error("gh_pr_merge.sh", "GraphQL: API rate limit exceeded")
    denied = classify_error("gh_pr_merge.sh", "fatal: permission denied")

    assert contract.allowed_transports == ("receipt_owner",)
    assert contract.fallback_markers == ()
    assert decision["decision"] == "fail_closed"
    assert decision["matched_fallback_markers"] == []
    assert quota["decision"] == "fail_closed"
    assert quota["matched_fallback_markers"] == []
    assert denied["decision"] == "fail_closed"
    assert denied["matched_fail_closed_markers"] == ["permission denied"]


def test_issue_view_wrapper_matches_delegated_issue_reader() -> None:
    """The compatibility wrapper exposes the route implemented by its reader."""
    wrapper = get_transport_contract("gh_issue_view.sh")
    reader = get_transport_contract("gh_issue_rest.py")

    assert wrapper.allowed_transports == reader.allowed_transports
    assert wrapper.fallback_markers == reader.fallback_markers
    assert wrapper.fail_closed_markers == reader.fail_closed_markers


def test_label_helper_contract_covers_reads_and_verified_writes() -> None:
    """The shared label helper contract covers strict reads and verified writes."""
    contract = get_transport_contract("gh_pr_label_rest.py")

    assert contract.allowed_transports == ("rest",)
    assert "read" in contract.purpose
    assert "verify" in contract.purpose
    assert contract.result_validator == "validate_result_envelope"
    check = check_helper(contract.helper, root=REPOSITORY_ROOT)
    assert check["status"] == "ok", check["findings"]


def test_all_contracts_declare_help_and_smoke_paths() -> None:
    """The registry keeps discoverability and focused proof mandatory."""
    for name in TRANSPORT_CONTRACTS:
        contract = get_transport_contract(name)
        assert contract.help_command == "--help"
        assert contract.smoke_test
        assert contract.allowed_transports


def test_cli_show_is_discoverable(capsys) -> None:
    """The policy has one machine-readable, help-friendly discovery command."""
    assert main(["show", "--json"]) == 0
    output = capsys.readouterr().out

    assert POLICY_SCHEMA in output
    assert "gh_comment.sh" in output
