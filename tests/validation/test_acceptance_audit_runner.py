"""Tests for strict declarative acceptance-audit contract execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import yaml

from scripts.validation.acceptance_audit_runner import (
    ContractValidationError,
    CriterionAudit,
    check_contract,
    run_contract,
    validate_contract,
)


def _write_contract(tmp_path: Path, **changes: Any) -> Path:
    """Write a minimal valid contract, applying test-specific changes."""

    contract: dict[str, Any] = {
        "schema_version": "acceptance-audit-contract.v1",
        "issue": 9999,
        "context_builder": "context",
        "defaults": {"evidence_path": "evidence.txt", "state_path": "state.yaml"},
        "evidence_paths": ["{evidence_path}", "{state_path}"],
        "status_mapping": {
            "allowed": ["met", "not_met"],
            "blocked": ["not_met"],
            "complete_when": "all_criteria_met",
        },
        "state_source": {
            "path_input": "state_path",
            "issue": 9999,
            "selector": "last",
            "acceptance_evidence_path": "acceptance_evidence",
            "closure_call_path": "closure_boundary.closure_call_for_this_pr",
            "integration_report_path": None,
            "fallback_acceptance_evidence_path": None,
            "include_latest_recorded_at_utc": False,
            "include_entry_status": False,
            "include_integration_report_status": False,
        },
        "criteria": [
            {
                "id": "criterion_one",
                "criterion": "Criterion one.",
                "evaluator": "evaluator",
                "evidence_paths": ["{evidence_path}"],
                "blocking_on": ["not_met"],
            }
        ],
        "report": {
            "schema_version": "issue-9999-audit.v1",
            "claim_boundary": "test only",
            "checked_paths": ["{evidence_path}"],
            "static_fields": {},
            "computed_fields": {},
        },
    }
    contract.update(changes)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return path


def _registries() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return explicit registries accepted by the minimal contract."""

    def build_context(repo_root: Path, inputs: dict[str, Path]) -> dict[str, Any]:
        del repo_root, inputs
        return {}

    def evaluate(definition: Any, context: dict[str, Any]) -> CriterionAudit:
        del context
        return CriterionAudit(definition.criterion, "met", "test evidence")

    return ({"context": build_context}, {"evaluator": evaluate}, {})


def test_validate_contract_rejects_unknown_root_field(tmp_path: Path) -> None:
    """Unknown contract fields must fail before execution."""

    path = _write_contract(tmp_path, unexpected_field=True)
    with pytest.raises(ContractValidationError, match="unknown fields"):
        validate_contract(path, evaluator_names={"evaluator"}, context_builder_names={"context"})


def test_validate_contract_rejects_duplicate_criterion_ids(tmp_path: Path) -> None:
    """Criterion identifiers must be unique and preserve declared ordering."""

    path = _write_contract(
        tmp_path,
        criteria=[
            {
                "id": "criterion_one",
                "criterion": "Criterion one.",
                "evaluator": "evaluator",
                "evidence_paths": ["{evidence_path}"],
                "blocking_on": ["not_met"],
            },
            {
                "id": "criterion_one",
                "criterion": "Criterion two.",
                "evaluator": "evaluator",
                "evidence_paths": ["{evidence_path}"],
                "blocking_on": ["not_met"],
            },
        ],
    )
    with pytest.raises(ContractValidationError, match="duplicate criterion id"):
        validate_contract(path, evaluator_names={"evaluator"}, context_builder_names={"context"})


def test_validate_contract_rejects_unresolved_evaluator(tmp_path: Path) -> None:
    """Contracts may reference only explicitly registered evaluators."""

    path = _write_contract(
        tmp_path,
        criteria=[
            {
                "id": "criterion_one",
                "criterion": "Criterion one.",
                "evaluator": "missing_evaluator",
                "evidence_paths": ["{evidence_path}"],
                "blocking_on": ["not_met"],
            }
        ],
    )
    with pytest.raises(ContractValidationError, match="unresolved evaluator"):
        validate_contract(path, evaluator_names={"evaluator"}, context_builder_names={"context"})


def test_validate_contract_rejects_unknown_evidence_reference(tmp_path: Path) -> None:
    """Evidence references must resolve to declared contract inputs."""

    path = _write_contract(tmp_path, evidence_paths=["{missing_input}"])
    with pytest.raises(ContractValidationError, match="unknown input path"):
        validate_contract(path, evaluator_names={"evaluator"}, context_builder_names={"context"})


def test_validate_contract_rejects_invalid_status_mapping(tmp_path: Path) -> None:
    """Blocking statuses must be declared in the allowed status vocabulary."""

    path = _write_contract(
        tmp_path,
        status_mapping={
            "allowed": ["met", "not_met"],
            "blocked": ["unknown"],
            "complete_when": "all_criteria_met",
        },
    )
    with pytest.raises(ContractValidationError, match="unknown status"):
        validate_contract(path, evaluator_names={"evaluator"}, context_builder_names={"context"})


def test_check_contract_does_not_read_missing_evidence(tmp_path: Path) -> None:
    """Contract-only checks validate registries without requiring evidence files."""

    path = _write_contract(tmp_path)
    check_contract(
        path,
        evaluator_names={"evaluator"},
        context_builder_names={"context"},
        report_resolver_names=set(),
    )


def test_run_contract_fails_closed_on_missing_evidence(tmp_path: Path) -> None:
    """Normal execution must reject missing evidence references."""

    path = _write_contract(tmp_path)
    context_builders, evaluators, resolvers = _registries()
    with pytest.raises(FileNotFoundError, match="contract evidence reference"):
        run_contract(
            contract_path=path,
            repo_root=tmp_path,
            input_paths={},
            context_builders=context_builders,
            evaluators=evaluators,
            report_resolvers=resolvers,
        )
