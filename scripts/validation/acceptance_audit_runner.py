"""Strict declarative runner shared by the #1358 and #1475 acceptance audits.

The runner owns contract loading, input/evidence path checks, criterion ordering,
status aggregation, canonical state-surface validation, and report assembly.
Issue-specific context builders and evaluators are registered explicitly by the
catalog module; contracts never import or execute arbitrary Python.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONTRACT_SCHEMA_VERSION = "acceptance-audit-contract.v1"
_MISSING = object()


class ContractValidationError(ValueError):
    """Raised when a declarative acceptance-audit contract is not executable."""


@dataclass(frozen=True)
class CriterionDefinition:
    """Validated declarative definition for one acceptance criterion."""

    identifier: str
    criterion: str
    evaluator: str
    evidence_paths: tuple[str, ...]
    blocking_on: tuple[str, ...]


@dataclass(frozen=True)
class CriterionAudit:
    """One evaluated acceptance criterion."""

    criterion: str
    status: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON representation used by both audits."""
        return {
            "criterion": self.criterion,
            "status": self.status,
            "evidence": self.evidence,
        }


ContextBuilder = Callable[[Path, Mapping[str, Path]], Mapping[str, Any]]
CriterionEvaluator = Callable[[CriterionDefinition, Mapping[str, Any]], CriterionAudit]
ReportResolver = Callable[[Mapping[str, Any], Sequence[CriterionAudit]], Any]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{label} must be a mapping")
    return value


def _list_of_strings(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ContractValidationError(f"{label} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ContractValidationError(f"{label} must not be empty")
    return value


def _check_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractValidationError(f"{label} has unknown fields: {', '.join(unknown)}")


def _validate_path_tokens(tokens: Iterable[str], defaults: Mapping[str, str], label: str) -> None:
    for token in tokens:
        if token.startswith("{") or token.endswith("}"):
            if not (token.startswith("{") and token.endswith("}") and token.count("{") == 1):
                raise ContractValidationError(f"{label} has malformed path token {token!r}")
            name = token[1:-1]
            if name not in defaults:
                raise ContractValidationError(f"{label} references unknown input path {name!r}")


def load_contract(path: Path) -> dict[str, Any]:
    """Load one YAML contract without reading any empirical evidence."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractValidationError(f"failed to read contract {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ContractValidationError(f"failed to parse contract {path}: {exc}") from exc
    return _mapping(raw, f"contract {path}")


def validate_contract(  # noqa: C901, PLR0912, PLR0915
    path: Path,
    *,
    evaluator_names: Iterable[str] = (),
    context_builder_names: Iterable[str] = (),
    report_resolver_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate contract shape and explicit registry references.

    This function reads only the contract YAML. It deliberately does not check
    whether evidence paths exist, so ``--check-contract`` can validate a new
    contract without reading empirical evidence or writing a report.
    """
    contract = load_contract(path)
    _check_keys(
        contract,
        {
            "schema_version",
            "issue",
            "context_builder",
            "defaults",
            "evidence_paths",
            "status_mapping",
            "state_source",
            "criteria",
            "report",
        },
        "contract",
    )
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractValidationError(
            f"contract schema_version must be {CONTRACT_SCHEMA_VERSION!r}"
        )
    issue = contract.get("issue")
    if isinstance(issue, bool) or not isinstance(issue, int) or issue < 1:
        raise ContractValidationError("contract issue must be a positive integer")

    context_builder = contract.get("context_builder")
    if not isinstance(context_builder, str) or not context_builder:
        raise ContractValidationError("contract context_builder must be a non-empty string")
    known_context_builders = set(context_builder_names)
    if known_context_builders and context_builder not in known_context_builders:
        raise ContractValidationError(f"unresolved context builder {context_builder!r}")

    defaults = _mapping(contract.get("defaults"), "contract defaults")
    if any(not isinstance(key, str) or not key for key in defaults):
        raise ContractValidationError("contract defaults keys must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in defaults.values()):
        raise ContractValidationError("contract defaults values must be non-empty strings")

    evidence_paths = _list_of_strings(contract.get("evidence_paths"), "contract evidence_paths")
    _validate_path_tokens(evidence_paths, defaults, "contract evidence_paths")

    status_mapping = _mapping(contract.get("status_mapping"), "contract status_mapping")
    _check_keys(status_mapping, {"allowed", "blocked", "complete_when"}, "status_mapping")
    allowed_statuses = _list_of_strings(status_mapping.get("allowed"), "status_mapping.allowed")
    blocked_statuses = _list_of_strings(status_mapping.get("blocked"), "status_mapping.blocked")
    if any(status not in allowed_statuses for status in blocked_statuses):
        raise ContractValidationError("status_mapping.blocked contains an unknown status")
    if status_mapping.get("complete_when") != "all_criteria_met":
        raise ContractValidationError("status_mapping.complete_when must be 'all_criteria_met'")
    if "met" not in allowed_statuses or "not_met" not in blocked_statuses:
        raise ContractValidationError(
            "status_mapping must include met as allowed and not_met as blocked"
        )

    state_source = _mapping(contract.get("state_source"), "contract state_source")
    _check_keys(
        state_source,
        {
            "path_input",
            "issue",
            "selector",
            "acceptance_evidence_path",
            "closure_call_path",
            "integration_report_path",
            "fallback_acceptance_evidence_path",
            "include_latest_recorded_at_utc",
            "include_entry_status",
            "include_integration_report_status",
        },
        "state_source",
    )
    path_input = state_source.get("path_input")
    if not isinstance(path_input, str) or path_input not in defaults:
        raise ContractValidationError("state_source.path_input must name a contract input")
    if state_source.get("issue") != issue:
        raise ContractValidationError("state_source.issue must match contract issue")
    if state_source.get("selector") not in {"last", "latest_recorded_at_utc"}:
        raise ContractValidationError("state_source.selector is invalid")
    for key in (
        "acceptance_evidence_path",
        "closure_call_path",
        "fallback_acceptance_evidence_path",
    ):
        value = state_source.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ContractValidationError(f"state_source.{key} must be a path string or null")
    integration_path = state_source.get("integration_report_path")
    if integration_path is not None and (
        not isinstance(integration_path, str) or not integration_path
    ):
        raise ContractValidationError(
            "state_source.integration_report_path must be a path string or null"
        )
    for key in (
        "include_latest_recorded_at_utc",
        "include_entry_status",
        "include_integration_report_status",
    ):
        if not isinstance(state_source.get(key), bool):
            raise ContractValidationError(f"state_source.{key} must be boolean")

    criteria = contract.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ContractValidationError("contract criteria must be a non-empty list")
    seen_ids: set[str] = set()
    known_evaluators = set(evaluator_names)
    validated_criteria: list[CriterionDefinition] = []
    for index, raw_criterion in enumerate(criteria):
        definition = _mapping(raw_criterion, f"criteria[{index}]")
        _check_keys(
            definition,
            {"id", "criterion", "evaluator", "evidence_paths", "blocking_on"},
            f"criteria[{index}]",
        )
        identifier = definition.get("id")
        criterion = definition.get("criterion")
        evaluator = definition.get("evaluator")
        if not isinstance(identifier, str) or not identifier:
            raise ContractValidationError(f"criteria[{index}].id must be non-empty")
        if identifier in seen_ids:
            raise ContractValidationError(f"duplicate criterion id {identifier!r}")
        seen_ids.add(identifier)
        if not isinstance(criterion, str) or not criterion:
            raise ContractValidationError(f"criteria[{index}].criterion must be non-empty")
        if not isinstance(evaluator, str) or not evaluator:
            raise ContractValidationError(f"criteria[{index}].evaluator must be non-empty")
        if known_evaluators and evaluator not in known_evaluators:
            raise ContractValidationError(f"unresolved evaluator {evaluator!r}")
        criterion_paths = _list_of_strings(
            definition.get("evidence_paths"),
            f"criteria[{index}].evidence_paths",
        )
        _validate_path_tokens(criterion_paths, defaults, f"criteria[{index}].evidence_paths")
        blocking_on = _list_of_strings(
            definition.get("blocking_on"),
            f"criteria[{index}].blocking_on",
        )
        if any(status not in allowed_statuses for status in blocking_on):
            raise ContractValidationError(
                f"criteria[{index}].blocking_on contains an unknown status"
            )
        validated_criteria.append(
            CriterionDefinition(
                identifier=identifier,
                criterion=criterion,
                evaluator=evaluator,
                evidence_paths=tuple(criterion_paths),
                blocking_on=tuple(blocking_on),
            )
        )

    report = _mapping(contract.get("report"), "contract report")
    _check_keys(
        report,
        {"schema_version", "claim_boundary", "checked_paths", "static_fields", "computed_fields"},
        "report",
    )
    for key in ("schema_version", "claim_boundary"):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ContractValidationError(f"report.{key} must be a non-empty string")
    checked_paths = _list_of_strings(report.get("checked_paths"), "report.checked_paths")
    _validate_path_tokens(checked_paths, defaults, "report.checked_paths")
    _mapping(report.get("static_fields"), "report.static_fields")
    computed_fields = _mapping(report.get("computed_fields", {}), "report.computed_fields")
    if any(not isinstance(key, str) or not key for key in computed_fields):
        raise ContractValidationError("report.computed_fields keys must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in computed_fields.values()):
        raise ContractValidationError("report.computed_fields values must be non-empty strings")
    known_resolvers = set(report_resolver_names)
    if known_resolvers:
        for resolver in computed_fields.values():
            if resolver not in known_resolvers:
                raise ContractValidationError(f"unresolved report resolver {resolver!r}")

    # Store normalized definitions for callers while retaining the declarative values.
    normalized = dict(contract)
    normalized["_criteria"] = tuple(validated_criteria)
    normalized["_allowed_statuses"] = tuple(allowed_statuses)
    normalized["_blocked_statuses"] = tuple(blocked_statuses)
    return normalized


def _resolve_input_paths(
    contract: Mapping[str, Any], repo_root: Path, overrides: Mapping[str, Path]
) -> dict[str, Path]:
    defaults = contract["defaults"]
    paths: dict[str, Path] = {}
    for name, default in defaults.items():
        value = overrides.get(name)
        if value is None:
            value = default
        paths[name] = Path(value)
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ContractValidationError(
            f"input path overrides name unknown contract inputs: {', '.join(unknown)}"
        )
    return paths


def _resolve_path_token(token: str, inputs: Mapping[str, Path], repo_root: Path) -> Path:
    if token.startswith("{") and token.endswith("}"):
        path = inputs[token[1:-1]]
    else:
        path = Path(token)
    return path if path.is_absolute() else repo_root / path


def _render_path_tokens(
    tokens: Sequence[str], inputs: Mapping[str, Path], repo_root: Path
) -> list[str]:
    del repo_root
    return [str(inputs[token[1:-1]]) if token.startswith("{") else token for token in tokens]


def _check_evidence_paths(
    contract: Mapping[str, Any], inputs: Mapping[str, Path], repo_root: Path
) -> None:
    paths = list(contract["evidence_paths"])
    for definition in contract["_criteria"]:
        paths.extend(definition.evidence_paths)
    for token in paths:
        path = _resolve_path_token(token, inputs, repo_root)
        if not path.exists():
            raise FileNotFoundError(f"contract evidence reference is missing: {path}")


def _nested_get(mapping: Mapping[str, Any], dotted_path: str | None) -> Any:
    if dotted_path is None:
        return _MISSING
    current: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


def _load_state_surface(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"failed to parse YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"expected YAML mapping in {path}")
    return data


def _state_surface_check(  # noqa: C901
    *,
    contract: Mapping[str, Any],
    repo_root: Path,
    state_path: Path,
    criteria: Sequence[CriterionAudit],
    closure_call: str,
) -> dict[str, Any]:
    config = contract["state_source"]
    state_file = state_path if state_path.is_absolute() else repo_root / state_path
    state_surface = _load_state_surface(state_file)
    entries = state_surface.get("entries")
    if not isinstance(entries, list) or not entries:
        return {
            "path": str(state_path),
            "status": "invalid",
            "errors": ["state surface has no entries"],
        }
    entry_dicts = [entry for entry in entries if isinstance(entry, dict)]
    if config["selector"] == "last":
        latest = entries[-1]
    else:
        latest = max(
            entry_dicts,
            key=lambda entry: str(entry.get("recorded_at_utc", "")),
            default={},
        )
    latest = latest if isinstance(latest, Mapping) else {}
    errors: list[str] = []
    if state_surface.get("issue") != contract["issue"]:
        errors.append(f"issue must be {contract['issue']}, got {state_surface.get('issue')!r}")
    recorded_closure = _nested_get(latest, config["closure_call_path"])
    if recorded_closure != closure_call:
        errors.append(
            f"latest entry closure_call_for_this_pr {recorded_closure!r} != {closure_call!r}"
        )
    state_evidence = _nested_get(latest, config["acceptance_evidence_path"])
    if state_evidence is _MISSING:
        state_evidence = _nested_get(latest, config["fallback_acceptance_evidence_path"])
    state_by_criterion = (
        {
            item.get("criterion"): item.get("status")
            for item in state_evidence
            if isinstance(item, dict)
        }
        if isinstance(state_evidence, list)
        else {}
    )
    for item in criteria:
        if state_by_criterion.get(item.criterion) != item.status:
            errors.append(
                f"{item.criterion!r} status "
                f"{state_by_criterion.get(item.criterion)!r} != {item.status!r}"
            )
    result: dict[str, Any] = {
        "path": str(state_path),
        "status": "valid" if not errors else "invalid",
    }
    if config["include_latest_recorded_at_utc"]:
        result["latest_recorded_at_utc"] = latest.get("recorded_at_utc")
    if config["include_entry_status"]:
        result["entry_status"] = latest.get("status")
    if config["include_integration_report_status"]:
        integration_status = _nested_get(latest, config["integration_report_path"])
        result["integration_report_status"] = (
            None if integration_status is _MISSING else integration_status
        )
    result["errors"] = errors
    return result


def run_contract(
    *,
    contract_path: Path,
    repo_root: Path,
    input_paths: Mapping[str, Path],
    context_builders: Mapping[str, ContextBuilder],
    evaluators: Mapping[str, CriterionEvaluator],
    report_resolvers: Mapping[str, ReportResolver],
) -> dict[str, Any]:
    """Evaluate one validated contract using explicit Python registries."""
    contract = validate_contract(
        contract_path,
        evaluator_names=evaluators,
        context_builder_names=context_builders,
        report_resolver_names=report_resolvers,
    )
    inputs = _resolve_input_paths(contract, repo_root, input_paths)
    _check_evidence_paths(contract, inputs, repo_root)
    context = dict(context_builders[contract["context_builder"]](repo_root, inputs))
    criteria: list[CriterionAudit] = []
    for definition in contract["_criteria"]:
        result = evaluators[definition.evaluator](definition, context)
        if result.criterion != definition.criterion:
            raise ContractValidationError(
                f"evaluator {definition.evaluator!r} returned criterion {result.criterion!r}; "
                f"expected {definition.criterion!r}"
            )
        if result.status not in contract["_allowed_statuses"]:
            raise ContractValidationError(
                f"evaluator {definition.evaluator!r} returned invalid status {result.status!r}"
            )
        criteria.append(result)
    status = "complete" if all(item.status == "met" for item in criteria) else "blocked"
    closure_call = "close" if status == "complete" else "keep_open"
    report_spec = contract["report"]
    report: dict[str, Any] = {
        "schema_version": report_spec["schema_version"],
        "issue": contract["issue"],
        "status": status,
        "closure_call": closure_call,
        "claim_boundary": report_spec["claim_boundary"],
        "checked_paths": _render_path_tokens(report_spec["checked_paths"], inputs, repo_root),
    }
    report.update(report_spec["static_fields"])
    for field, resolver_name in report_spec.get("computed_fields", {}).items():
        report[field] = report_resolvers[resolver_name](context, criteria)
    report["acceptance_evidence"] = [item.to_dict() for item in criteria]
    report["remaining_criteria"] = [
        item.to_dict()
        for definition, item in zip(contract["_criteria"], criteria, strict=True)
        if item.status in definition.blocking_on
    ]
    report["state_surface"] = _state_surface_check(
        contract=contract,
        repo_root=repo_root,
        state_path=inputs[contract["state_source"]["path_input"]],
        criteria=criteria,
        closure_call=closure_call,
    )
    return report


def check_contract(
    contract_path: Path,
    *,
    evaluator_names: Iterable[str],
    context_builder_names: Iterable[str],
    report_resolver_names: Iterable[str],
) -> None:
    """Validate one contract without reading empirical inputs or writing output."""
    validate_contract(
        contract_path,
        evaluator_names=evaluator_names,
        context_builder_names=context_builder_names,
        report_resolver_names=report_resolver_names,
    )
