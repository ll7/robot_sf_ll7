"""Parse and validate the versioned machine-readable PR contract.

The v2 block is intentionally small and strict.  It complements the human PR
narrative while preserving the existing Markdown parser as the compatibility
path for bodies that do not contain a ``pr-contract:v2`` marker.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import yaml

CONTRACT_MARKER = "pr-contract:v2"
CHANGE_CLASSES = frozenset(
    {
        "docs",
        "tooling",
        "runtime",
        "benchmark_or_metric",
        "paper_or_claim",
        "performance",
    }
)
EVIDENCE_APPLICABILITIES = frozenset({"evidence-bearing", "docs-only", "na"})
EVIDENCE_TIERS = frozenset(
    {
        "idea",
        "launch_packet",
        "preflight_valid",
        "smoke",
        "nominal",
        "stress",
        "full_matrix",
        "analysis_only",
        "synthesis",
        "paper_grade",
        "blocked",
    }
)
RESULT_CLASSIFICATIONS = frozenset(
    {"positive", "negative", "inconclusive", "diagnostic-only", "blocker-resolution", "na"}
)
DEFERRED_STATUSES = frozenset({"none", "deferred", "blocked", "open"})
DOMAIN_STATUSES = frozenset({"approved", "waived", "pending", "blocked", "not_required"})
PERFORMANCE_FIELDS = (
    "baseline",
    "changed",
    "representative_command",
    "hot_path",
    "cache",
    "rollback",
)
VALIDITY_CHECKLIST_FIELDS = (
    "target_claim",
    "comparator_validity",
    "fallback_exclusions",
    "claim_boundary",
    "implementation_integrity",
)

_MARKER_RE = re.compile(
    r"<!--[ \t]*pr-contract:v2[ \t]*\r?\n(?P<payload>.*?)\r?\n[ \t]*-->",
    re.DOTALL,
)
_MARKER_PREFIX_RE = re.compile(r"<!--[ \t]*pr-contract:v2")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
):
    if not isinstance(node, yaml.nodes.MappingNode):
        raise yaml.constructor.ConstructorError(None, None, "expected a mapping", node.start_mark)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class PrContractV2:
    """Validated machine-readable PR contract values."""

    change_class: str
    closes: tuple[int, ...]
    relates: tuple[int, ...]
    deferred_status: str
    deferred_issues: tuple[int, ...]
    deferred_reason: str
    evidence_applicability: str
    evidence_tier: str | None
    evidence_result: str
    domain_required: bool
    domain_status: str
    domain_domains: tuple[str, ...]
    domain_note: str
    domain_checklist: tuple[tuple[str, str], ...]
    performance_claimed: bool
    performance_measurements: tuple[tuple[str, str], ...]
    exact_head: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for CLI reports."""
        return asdict(self)


@dataclass(frozen=True)
class PrContractV2Result:
    """Result of locating and validating a v2 marker."""

    status: str
    source: str
    contract: PrContractV2 | None
    errors: tuple[str, ...]
    message: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "status": self.status,
            "source": self.source,
            "errors": list(self.errors),
            "message": self.message,
            "contract": self.contract.as_dict() if self.contract is not None else None,
        }


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        errors.append(f"{path} must be a mapping with string keys")
        return {}
    return value


def _required_keys(
    value: Mapping[str, Any], path: str, required: set[str], allowed: set[str], errors: list[str]
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        errors.append(f"{path} is missing required field(s): {', '.join(missing)}")
    if unknown:
        errors.append(f"{path} contains unknown field(s): {', '.join(unknown)}")


def _string(value: Any, path: str, errors: list[str], *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        errors.append(f"{path} must be a non-empty string")
        return ""
    return value.strip()


def _optional_string(value: Any, path: str, errors: list[str]) -> str | None:
    if value is None:
        return None
    return _string(value, path, errors)


def _boolean(value: Any, path: str, errors: list[str]) -> bool:
    if type(value) is not bool:
        errors.append(f"{path} must be a boolean")
        return False
    return value


def _enum(value: Any, path: str, choices: frozenset[str], errors: list[str]) -> str:
    result = _string(value, path, errors)
    if result and result not in choices:
        errors.append(f"{path} must be one of: {', '.join(sorted(choices))}")
    return result


def _issues(value: Any, path: str, errors: list[str]) -> tuple[int, ...]:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list of positive issue numbers")
        return ()
    parsed: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or item <= 0:
            errors.append(f"{path}[{index}] must be a positive integer")
            continue
        parsed.append(item)
    if len(set(parsed)) != len(parsed):
        errors.append(f"{path} contains duplicate issue references")
    return tuple(parsed)


def _string_list(value: Any, path: str, errors: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list of strings")
        return ()
    values: list[str] = []
    for index, item in enumerate(value):
        values.append(_string(item, f"{path}[{index}]", errors))
    if len(set(values)) != len(values):
        errors.append(f"{path} contains duplicate values")
    return tuple(value for value in values if value)


def _validate_evidence_declaration(
    applicability: str,
    tier: str | None,
    result: str,
    errors: list[str],
) -> None:
    """Reject evidence fields that contradict their applicability."""
    if applicability == "evidence-bearing":
        if tier is None:
            errors.append("evidence-bearing contracts require evidence.tier")
        if result == "na":
            errors.append("evidence-bearing contracts require a non-na evidence.result")
        return
    if tier is not None:
        errors.append("non-evidence contracts must set evidence.tier=null")
    if result != "na":
        errors.append("non-evidence contracts must set evidence.result=na")


def _validate_domain_declaration(
    applicability: str,
    required: bool,
    status: str,
    domains: tuple[str, ...],
    note: str,
    checklist: Mapping[str, str],
    errors: list[str],
) -> None:
    """Reject approval metadata that contradicts evidence applicability."""
    if applicability == "evidence-bearing":
        if not required:
            errors.append("evidence-bearing contracts require domain_approval.required=true")
        if status not in {"approved", "waived"}:
            errors.append(
                "evidence-bearing contracts require domain_approval.status=approved or waived"
            )
        if not domains:
            errors.append("evidence-bearing contracts require domain_approval.domains")
        if not note:
            errors.append("evidence-bearing contracts require domain_approval.note")
        missing = [key for key in VALIDITY_CHECKLIST_FIELDS if key not in checklist]
        if missing:
            errors.append(
                "evidence-bearing contracts require validity checklist field(s): "
                + ", ".join(missing)
            )
        return
    if required:
        errors.append("non-evidence contracts require domain_approval.required=false")
    if status != "not_required":
        errors.append("non-evidence contracts require domain_approval.status=not_required")


def _validate_linked_and_deferred(data: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    """Validate linked issue and deferred-work declarations."""
    linked = _mapping(data.get("linked_issues"), "linked_issues", errors)
    _required_keys(linked, "linked_issues", {"closes", "relates"}, {"closes", "relates"}, errors)
    closes = _issues(linked.get("closes"), "linked_issues.closes", errors)
    relates = _issues(linked.get("relates"), "linked_issues.relates", errors)
    deferred = _mapping(data.get("deferred_work"), "deferred_work", errors)
    _required_keys(
        deferred,
        "deferred_work",
        {"status", "issues"},
        {"status", "issues", "reason"},
        errors,
    )
    deferred_status = _enum(
        deferred.get("status"), "deferred_work.status", DEFERRED_STATUSES, errors
    )
    deferred_issues = _issues(deferred.get("issues"), "deferred_work.issues", errors)
    deferred_reason = _string(
        deferred.get("reason", ""), "deferred_work.reason", errors, allow_empty=True
    )
    all_issue_refs = (*closes, *relates, *deferred_issues)
    if len(set(all_issue_refs)) != len(all_issue_refs):
        errors.append("linked_issues and deferred_work contain duplicate issue references")
    if deferred_status == "none" and deferred_issues:
        errors.append("deferred_work.status=none cannot list follow-up issues")
    if deferred_status != "none" and not deferred_issues and not deferred_reason:
        errors.append(
            "deferred_work needs at least one follow-up issue or an explicit reason when status is "
            f"{deferred_status!r}"
        )
    return {
        "closes": closes,
        "relates": relates,
        "deferred_status": deferred_status,
        "deferred_issues": deferred_issues,
        "deferred_reason": deferred_reason,
    }


def _validate_evidence_and_domain(data: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    """Validate evidence applicability and domain-aware approval values."""
    evidence = _mapping(data.get("evidence"), "evidence", errors)
    _required_keys(
        evidence,
        "evidence",
        {"applicability", "tier", "result"},
        {"applicability", "tier", "result"},
        errors,
    )
    evidence_applicability = _enum(
        evidence.get("applicability"),
        "evidence.applicability",
        EVIDENCE_APPLICABILITIES,
        errors,
    )
    evidence_tier = _optional_string(evidence.get("tier"), "evidence.tier", errors)
    if evidence_tier is not None and evidence_tier not in EVIDENCE_TIERS:
        errors.append(f"evidence.tier must be one of: {', '.join(sorted(EVIDENCE_TIERS))}")
    evidence_result = _enum(
        evidence.get("result"), "evidence.result", RESULT_CLASSIFICATIONS, errors
    )

    domain = _mapping(data.get("domain_approval"), "domain_approval", errors)
    _required_keys(
        domain,
        "domain_approval",
        {"required", "status"},
        {"required", "status", "domains", "note", "validity_checklist"},
        errors,
    )
    domain_required = _boolean(domain.get("required"), "domain_approval.required", errors)
    domain_status = _enum(domain.get("status"), "domain_approval.status", DOMAIN_STATUSES, errors)
    domain_domains = _string_list(domain.get("domains", []), "domain_approval.domains", errors)
    domain_note = _string(domain.get("note", ""), "domain_approval.note", errors, allow_empty=True)
    checklist = _mapping(
        domain.get("validity_checklist", {}), "domain_approval.validity_checklist", errors
    )
    unknown_checklist = sorted(set(checklist) - set(VALIDITY_CHECKLIST_FIELDS))
    if unknown_checklist:
        errors.append(
            "domain_approval.validity_checklist contains unknown field(s): "
            + ", ".join(unknown_checklist)
        )
    domain_checklist_values: dict[str, str] = {}
    for key in VALIDITY_CHECKLIST_FIELDS:
        if key in checklist:
            domain_checklist_values[key] = _string(
                checklist[key], f"domain_approval.validity_checklist.{key}", errors
            )

    _validate_evidence_declaration(evidence_applicability, evidence_tier, evidence_result, errors)
    _validate_domain_declaration(
        evidence_applicability,
        domain_required,
        domain_status,
        domain_domains,
        domain_note,
        domain_checklist_values,
        errors,
    )
    return {
        "evidence_applicability": evidence_applicability,
        "evidence_tier": evidence_tier,
        "evidence_result": evidence_result,
        "domain_required": domain_required,
        "domain_status": domain_status,
        "domain_domains": domain_domains,
        "domain_note": domain_note,
        "domain_checklist": tuple(
            (key, domain_checklist_values[key])
            for key in VALIDITY_CHECKLIST_FIELDS
            if key in domain_checklist_values
        ),
    }


def _validate_performance_and_head(data: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    """Validate performance evidence and the optional exact-head carrier."""
    performance = _mapping(data.get("performance"), "performance", errors)
    _required_keys(
        performance,
        "performance",
        {"claimed"},
        {"claimed", *PERFORMANCE_FIELDS},
        errors,
    )
    performance_claimed = _boolean(performance.get("claimed"), "performance.claimed", errors)
    performance_measurements: dict[str, str] = {}
    for key in PERFORMANCE_FIELDS:
        if key in performance:
            performance_measurements[key] = _string(performance[key], f"performance.{key}", errors)
    if performance_claimed:
        missing_measurements = [
            key for key in PERFORMANCE_FIELDS if key not in performance_measurements
        ]
        if missing_measurements:
            errors.append(
                "performance claims require measurement field(s): "
                + ", ".join(missing_measurements)
            )
    elif performance_measurements:
        errors.append("performance measurements cannot be declared when claimed=false")

    exact_head_value = data.get("exact_head")
    exact_head: str | None = None
    if exact_head_value is not None:
        exact_head = _string(exact_head_value, "exact_head", errors).lower()
        if not re.fullmatch(r"[0-9a-f]{40}", exact_head):
            errors.append("exact_head must be a full 40-character lowercase hexadecimal SHA")
    return {
        "performance_claimed": performance_claimed,
        "performance_measurements": tuple(performance_measurements.items()),
        "exact_head": exact_head,
    }


def _validate_contract(data: Mapping[str, Any]) -> tuple[PrContractV2 | None, tuple[str, ...]]:
    """Validate a decoded v2 mapping and return a typed contract."""
    errors: list[str] = []
    allowed = {
        "change_class",
        "linked_issues",
        "deferred_work",
        "evidence",
        "domain_approval",
        "performance",
        "exact_head",
    }
    _required_keys(data, "contract", allowed - {"exact_head"}, allowed, errors)
    change_class = _enum(data.get("change_class"), "change_class", CHANGE_CLASSES, errors)
    linked_values = _validate_linked_and_deferred(data, errors)
    evidence_values = _validate_evidence_and_domain(data, errors)
    performance_values = _validate_performance_and_head(data, errors)

    if (
        change_class in {"benchmark_or_metric", "paper_or_claim"}
        and evidence_values["evidence_applicability"] != "evidence-bearing"
    ):
        errors.append(
            f"change_class={change_class} requires evidence.applicability=evidence-bearing"
        )
    if change_class == "docs" and evidence_values["evidence_applicability"] == "evidence-bearing":
        errors.append("change_class=docs cannot declare evidence-bearing applicability")
    if change_class == "performance" and not performance_values["performance_claimed"]:
        errors.append("change_class=performance requires performance.claimed=true")
    if performance_values["performance_claimed"] and change_class != "performance":
        errors.append("performance.claimed=true requires change_class=performance")

    if errors:
        return None, tuple(dict.fromkeys(errors))
    return (
        PrContractV2(
            change_class=change_class,
            closes=linked_values["closes"],
            relates=linked_values["relates"],
            deferred_status=linked_values["deferred_status"],
            deferred_issues=linked_values["deferred_issues"],
            deferred_reason=linked_values["deferred_reason"],
            evidence_applicability=evidence_values["evidence_applicability"],
            evidence_tier=evidence_values["evidence_tier"],
            evidence_result=evidence_values["evidence_result"],
            domain_required=evidence_values["domain_required"],
            domain_status=evidence_values["domain_status"],
            domain_domains=evidence_values["domain_domains"],
            domain_note=evidence_values["domain_note"],
            domain_checklist=evidence_values["domain_checklist"],
            performance_claimed=performance_values["performance_claimed"],
            performance_measurements=performance_values["performance_measurements"],
            exact_head=performance_values["exact_head"],
        ),
        (),
    )


def parse_pr_contract_v2(body: str, *, source: str = "body") -> PrContractV2Result:
    """Locate and validate one v2 block without falling back after a marker error."""
    prefixes = list(_MARKER_PREFIX_RE.finditer(body))
    if not prefixes:
        return PrContractV2Result(
            status="absent",
            source=source,
            contract=None,
            errors=(),
            message="No pr-contract:v2 marker present; use the v1 Markdown compatibility parser.",
        )

    matches = list(_MARKER_RE.finditer(body))
    if len(prefixes) != 1 or len(matches) != 1:
        errors = ("expected exactly one complete pr-contract:v2 HTML comment",)
        return PrContractV2Result(
            status="malformed",
            source=source,
            contract=None,
            errors=errors,
            message="Malformed pr-contract:v2: " + "; ".join(errors),
        )

    try:
        raw = yaml.load(
            matches[0].group("payload"),
            Loader=_UniqueKeyLoader,  # noqa: S506
        )
    except yaml.YAMLError as exc:
        detail = " ".join(str(exc).split()) or "invalid YAML"
        errors = (f"invalid YAML ({detail})",)
        return PrContractV2Result(
            status="malformed",
            source=source,
            contract=None,
            errors=errors,
            message="Malformed pr-contract:v2: " + "; ".join(errors),
        )

    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        errors = ("top-level value must be a mapping with string keys",)
        return PrContractV2Result(
            status="malformed",
            source=source,
            contract=None,
            errors=errors,
            message="Malformed pr-contract:v2: " + "; ".join(errors),
        )

    contract, errors = _validate_contract(raw)
    if errors:
        return PrContractV2Result(
            status="malformed",
            source=source,
            contract=None,
            errors=errors,
            message="Malformed pr-contract:v2: " + "; ".join(errors),
        )
    return PrContractV2Result(
        status="ok",
        source=source,
        contract=contract,
        errors=(),
        message="pr-contract:v2 is valid; v2 fields are authoritative for machine checks.",
    )
