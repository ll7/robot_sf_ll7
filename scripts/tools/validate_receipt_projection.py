#!/usr/bin/env python3
"""Validate private-to-public receipt projections against schema-driven policies (#8844).

Derives the sanitized public projection of an exact private receipt from explicit
per-field-path policies (``keep``, ``omit``, ``reject``, ``digest``, ``constant``,
``stable_alias``) and checks a proposed public receipt against it. Required public
identity (source/config digests, environment class, row counts, terminal status,
artifact checksums, claim boundary) must survive; credentials, user/home paths,
hostnames, IPs, signed URLs, accounts, and queue topology must not. Private IDs and
artifact roots become deterministic path-scoped aliases; the public receipt binds
the full private receipt by canonical SHA-256 without publishing private bytes;
unsupported classes are reported and repeated projection is byte-stable.

Check-only: never writes receipts or state. CLI: ``--check --private <json>
--public <json> [--format json|text]``; exit codes 0 valid, 2 invalid or
unsupported, 3 malformed input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from robot_sf.adversarial.public_projection import find_offending_paths
from scripts.tools.classify_scheduler_failure import sanitize_text
from scripts.tools.scheduler_allocation_receipt import validate_receipt as validate_public_receipt

REPORT_SCHEMA = "robot_sf.receipt_projection_check.v1"
POLICY_SCHEMA = "robot_sf.receipt_projection_policy.v1"
POLICY_PATH = Path(__file__).with_name("receipt_projection_policies.json")
EXPLICIT_TOKENS = frozenset(
    ("", "redacted", "unavailable", "not_observed", "not_applicable", "none", "null")
)
ALLOWED_PUBLIC_URL_RE = re.compile(r"https://github\.com/ll7/robot_sf_ll7/(?:issues|pull)/\d+")
# Public resource classes (``cpu_8``, ``mem_64gb``, ``gpu_1``) resemble the bare node
# labels the shared sanitizer redacts; the contract declares these values public, so
# the scanner exempts exactly this shape and nothing wider.
ALLOWED_RESOURCE_CLASS_RE = re.compile(r"^(?:cpu|mem|memory|gpu|accel|accelerator)_[a-z0-9]+$")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_RE = re.compile(r"(?i)(?:\b(?:[0-9a-f]{1,4}:){2,}[0-9a-f]{0,4}\b|\b[0-9a-f:]*::[0-9a-f:]*)")
_MAX_MISMATCHES = 24
_OMIT = object()
_MISSING = object()
CLAIM_BOUNDARY = (
    "Receipt projection hygiene only. A valid projection does not establish artifact custody, "
    "benchmark eligibility, or scientific success, and this check never publishes private bytes."
)
_PUBLIC_VALIDATORS = {"robot_sf.scheduler_allocation_receipt.v1": validate_public_receipt}


class Issue(NamedTuple):
    """One sanitized fail-closed issue, free of private values."""

    code: str
    location: str
    message: str


class FieldPolicy(NamedTuple):
    """One resolved projection policy: kind plus an optional constant/alias prefix."""

    kind: str
    value: Any = None


class Contract(NamedTuple):
    """One private/public receipt projection contract."""

    private_schema: str
    public_schema: str
    binding_path: str
    fields: Mapping[tuple[str, ...], FieldPolicy]
    wildcards: tuple[tuple[tuple[str, ...], FieldPolicy], ...]
    rejected_names: frozenset[str]
    identity: tuple[str, ...]
    patterns: Mapping[str, str]
    enums: Mapping[str, tuple[str, ...]]


class Projection(NamedTuple):
    """Derived public receipt plus the private binding digest and policy issues."""

    public_receipt: dict[str, Any] | None
    private_receipt_sha256: str
    issues: tuple[Issue, ...]


def _issue(code: str, location: str, message: str) -> Issue:
    return Issue(code=code, location=location, message=message)


def _canonical_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _location(path: Sequence[str]) -> str:
    return "/" + "/".join(path)


def detect_receipt_class(payload: Mapping[str, Any]) -> str | None:
    """Return the declared private receipt class, or ``None`` when undeclared."""
    for key in ("schema", "schema_version"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_path(dotted: str) -> tuple[str, ...]:
    """Expand a dotted policy path, turning ``name[*]`` into two array segments."""
    segments: list[str] = []
    for part in dotted.split("."):
        if part.endswith("[*]"):
            segments.extend((part[:-3], "*"))
        else:
            segments.append(part)
    return tuple(segments)


def _parse_policy(raw: Any) -> FieldPolicy:
    if isinstance(raw, str):
        return FieldPolicy(kind=raw)
    if isinstance(raw, list) and len(raw) == 2:
        return FieldPolicy(kind=str(raw[0]), value=raw[1])
    raise ValueError("policy entries must be strings or [kind, value] pairs")


@lru_cache(maxsize=1)
def load_contracts(policy_path: str = str(POLICY_PATH)) -> dict[str, Contract]:
    """Load and index the versioned projection policy table by private schema."""
    payload = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if payload.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy table must declare {POLICY_SCHEMA}")
    rejected = frozenset(str(name).lower() for name in payload.get("rejected_field_names", ()))
    contracts: dict[str, Contract] = {}
    for entry in payload.get("classes", ()):
        fields: dict[tuple[str, ...], FieldPolicy] = {}
        wildcards: list[tuple[tuple[str, ...], FieldPolicy]] = []
        entries = [(path, _parse_policy(raw)) for path, raw in entry["fields"].items()]
        entries += [(path, FieldPolicy(kind="keep")) for path in entry.get("keep", ())]
        for path, policy in entries:
            parsed = _parse_path(path)
            if "*" in path:
                wildcards.append((parsed, policy))
            else:
                fields.setdefault(parsed, policy)
        wildcards.sort(
            key=lambda item: (item[0].count("**"), item[0].count("*"), -len(item[0]), item[0])
        )
        required = entry.get("required", {})
        contracts[entry["private_schema"]] = Contract(
            private_schema=entry["private_schema"],
            public_schema=entry["public_schema"],
            binding_path=entry.get("binding_path", "source_binding"),
            fields=fields,
            wildcards=tuple(wildcards),
            rejected_names=rejected,
            identity=tuple(required.get("identity", ())),
            patterns={key: str(value) for key, value in required.get("patterns", {}).items()},
            enums={key: tuple(value) for key, value in required.get("enums", {}).items()},
        )
    return contracts


def _pattern_matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if not pattern or len(pattern) > len(path):
        return False
    if pattern[-1] == "**":
        prefix = pattern[:-1]
        return all(part in ("*", segment) for part, segment in zip(prefix, path, strict=False))
    return len(pattern) == len(path) and all(
        part in ("*", segment) for part, segment in zip(pattern, path, strict=True)
    )


def _policy_for(contract: Contract, path: tuple[str, ...]) -> FieldPolicy | None:
    exact = contract.fields.get(path)
    if exact is not None:
        return exact
    for pattern, policy in contract.wildcards:
        if _pattern_matches(pattern, path):
            return policy
    return None


def _project_node(
    value: Any, path: tuple[str, ...], contract: Contract, issues: list[Issue]
) -> Any:
    policy = _policy_for(contract, path)
    if policy is None:
        issues.append(_issue("unsupported_field", _location(path), "no projection policy"))
        return _OMIT
    if policy.kind == "omit":
        return _OMIT
    if policy.kind == "reject":
        issues.append(_issue("rejected_private_field", _location(path), "rejected by policy"))
        return _OMIT
    if policy.kind == "digest":
        return "sha256:" + _canonical_digest(value)
    if policy.kind == "constant":
        return policy.value
    if policy.kind == "stable_alias":
        if not isinstance(value, str) or not value:
            issues.append(
                _issue("invalid_alias_source", _location(path), "stable alias needs a string")
            )
            return _OMIT
        seed = f"{policy.value}\x00{_location(path)}\x00{value}".encode()
        return f"{policy.value}-{hashlib.sha256(seed).hexdigest()[:16]}"
    if policy.kind != "keep":
        issues.append(_issue("unknown_policy", _location(path), policy.kind))
        return _OMIT
    return (
        _project_children(value, path, contract, issues)
        if isinstance(value, (Mapping, list))
        else value
    )


def _project_children(
    value: Any, path: tuple[str, ...], contract: Contract, issues: list[Issue]
) -> Any:
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key in sorted(value, key=str):
            child = (*path, str(key))
            if str(key).lower() in contract.rejected_names:
                issues.append(
                    _issue("rejected_private_field", _location(child), "rejected field name")
                )
                continue
            item = _project_node(value[key], child, contract, issues)
            if item is not _OMIT:
                projected[str(key)] = item
        return projected
    projected_list: list[Any] = []
    for item in value:
        result = (
            _project_children(item, (*path, "*"), contract, issues)
            if isinstance(item, Mapping)
            else _project_node(item, (*path, "*"), contract, issues)
        )
        if result is not _OMIT:
            projected_list.append(result)
    return projected_list


def _value_at(receipt: Mapping[str, Any], dotted: str) -> Any:
    node: Any = receipt
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return _MISSING
    return node


def project_receipt(
    private: Mapping[str, Any],
    *,
    contract: Contract | None = None,
    contracts: Mapping[str, Contract] | None = None,
) -> Projection:
    """Derive the public projection binding the full private receipt by digest."""
    receipt_class = detect_receipt_class(private)
    if receipt_class is None:
        return Projection(None, "", (_issue("missing_receipt_class", "/", "schema undeclared"),))
    contract = contract or (contracts or load_contracts()).get(receipt_class)
    if contract is None:
        return Projection(
            None, "", (_issue("unsupported_receipt_class", "/schema", receipt_class),)
        )
    issues: list[Issue] = []
    public = _project_children(private, (), contract, issues)
    # Contract constants the private schema omits (for example the claim boundary) are
    # materialized only when the private receipt does not already carry the path.
    for path, policy in contract.fields.items():
        if policy.kind == "constant" and _value_at(public, ".".join(path)) is _MISSING:
            node: dict[str, Any] = public
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = policy.value
    digest = _canonical_digest(private)
    public[contract.binding_path] = {"schema": contract.private_schema, "receipt_sha256": digest}
    return Projection(public, digest, tuple(issues))


def _explicit(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in EXPLICIT_TOKENS


def _check_identity(
    contract: Contract, receipt: Mapping[str, Any], issues: list[Issue], *, origin: str
) -> None:
    for path in contract.identity:
        value = _value_at(receipt, path)
        location = f"{origin}:{path}"
        empty = value is _MISSING or value is None or value == ""
        if empty or (isinstance(value, (list, dict)) and not value):
            issues.append(_issue("missing_required_identity", location, "required identity absent"))
        elif _explicit(value):
            issues.append(_issue("over_redacted_identity", location, "required identity redacted"))
    for path, pattern in contract.patterns.items():
        value = _value_at(receipt, path)
        observed = value if isinstance(value, str) else ""
        if value is _MISSING or re.fullmatch(pattern, observed) is not None:
            continue
        code = "over_redacted_identity" if _explicit(value) else "invalid_required_identity"
        issues.append(_issue(code, f"{origin}:{path}", "does not match the required format"))
    for path, options in contract.enums.items():
        value = _value_at(receipt, path)
        if value is _MISSING or value in options:
            continue
        code = "over_redacted_identity" if _explicit(value) else "invalid_required_status"
        issues.append(_issue(code, f"{origin}:{path}", "not an allowed status value"))


def _private_content_reason(text: str) -> str:
    if text and (
        ALLOWED_PUBLIC_URL_RE.fullmatch(text) or ALLOWED_RESOURCE_CLASS_RE.fullmatch(text)
    ):
        return ""
    if find_offending_paths([text]):
        return "private filesystem path"
    if _IPV4_RE.search(text) or _IPV6_RE.search(text):
        return "network address"
    if sanitize_text(text, limit=max(len(text), 1)) != text:
        return "credential, host, identity, or signed URL content"
    return ""


def private_content_reason(text: str) -> str:
    """Return why ``text`` is not public-safe, or ``""`` when it is public-safe."""
    return _private_content_reason(text)


def _scan_content(
    contract: Contract, value: Any, issues: list[Issue], *, origin: str, location: str = ""
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{location}/{key}"
            if str(key).lower() in contract.rejected_names:
                issues.append(
                    _issue("forbidden_field", f"{origin}:{child}", "private topology field name")
                )
            _scan_content(contract, item, issues, origin=origin, location=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_content(contract, item, issues, origin=origin, location=f"{location}[{index}]")
    elif isinstance(value, str) and (reason := _private_content_reason(value)):
        issues.append(_issue("under_redacted_value", f"{origin}:{location}", reason))


def _differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        found: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}/{key}"
            if key not in expected or key not in actual:
                found.append(child)
            else:
                found.extend(_differences(expected[key], actual[key], child))
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}[*]"]
        found = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            found.extend(_differences(left, right, f"{path}[{index}]"))
        return found
    return [] if expected == actual else [path]


def _check_binding(
    contract: Contract, public: Mapping[str, Any], digest: str, issues: list[Issue]
) -> None:
    binding = _value_at(public, contract.binding_path)
    path = f"/{contract.binding_path}"
    if not isinstance(binding, Mapping):
        issues.append(_issue("missing_source_binding", path, "required"))
    elif binding.get("schema") != contract.private_schema:
        issues.append(_issue("source_binding_schema_mismatch", path, "differs"))
    elif binding.get("receipt_sha256") != digest:
        issues.append(_issue("source_binding_mismatch", path, "digest differs"))


def _check_public_semantics(
    contract: Contract, public: Mapping[str, Any], issues: list[Issue]
) -> None:
    validator = _PUBLIC_VALIDATORS.get(contract.public_schema)
    if validator is None:
        return
    view = {key: value for key, value in public.items() if key != contract.binding_path}
    for issue in validator(view):
        issues.append(_issue(f"public_schema_{issue.code}", issue.location, issue.message))


def check_receipt_projection(
    private: Mapping[str, Any],
    public: Mapping[str, Any],
    *,
    contracts: Mapping[str, Contract] | None = None,
) -> dict[str, Any]:
    """Validate one proposed public receipt against the derived private projection."""
    contracts = contracts if contracts is not None else load_contracts()
    receipt_class = detect_receipt_class(private) or ""
    contract = contracts.get(receipt_class)
    if contract is None:
        return _build_report(
            receipt_class,
            None,
            None,
            public,
            (_issue("unsupported_receipt_class", "/schema", receipt_class),),
            "unsupported_receipt_class",
        )
    projection = project_receipt(private, contract=contract)
    expected = projection.public_receipt
    issues = list(projection.issues)
    if expected is not None:
        for view, origin in ((expected, "projected"), (public, "public")):
            _check_identity(contract, view, issues, origin=origin)
            _scan_content(contract, view, issues, origin=origin)
        for path in _differences(expected, public)[:_MAX_MISMATCHES]:
            issues.append(
                _issue("projection_mismatch", path or "/", "differs from derived projection")
            )
        _check_binding(contract, public, projection.private_receipt_sha256, issues)
        _check_public_semantics(contract, public, issues)
    return _build_report(
        receipt_class,
        contract,
        projection,
        public,
        issues,
        "projection_valid" if not issues else "projection_invalid",
    )


def _build_report(
    receipt_class: str,
    contract: Contract | None,
    projection: Projection | None,
    public: Mapping[str, Any],
    issues: Sequence[Issue],
    status: str,
) -> dict[str, Any]:
    unique = sorted({(item.code, item.location, item.message) for item in issues})
    expected = projection.public_receipt if projection is not None else None
    return {
        "schema": REPORT_SCHEMA,
        "check_only": True,
        "status": status,
        "receipt_class": receipt_class,
        "public_schema": contract.public_schema if contract is not None else None,
        "private_receipt_sha256": projection.private_receipt_sha256 or None
        if projection is not None
        else None,
        "projected_receipt_sha256": _canonical_digest(expected) if expected is not None else None,
        "public_receipt_sha256": _canonical_digest(public),
        "issue_count": len(unique),
        "issues": [
            {"code": code, "location": location, "message": message}
            for code, location, message in unique
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def render_json(payload: Mapping[str, Any]) -> str:
    """Return byte-stable sorted-key JSON with a trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the projection-check argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate only; never writes")
    parser.add_argument("--private", type=Path, required=True, help="exact private receipt JSON")
    parser.add_argument("--public", type=Path, required=True, help="proposed public receipt JSON")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def _load_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("receipt must be a JSON object")
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check-only projection validation and return a shell-friendly exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("only --check is supported; this tool never writes receipts or state")
    try:
        private, public = _load_receipt(args.private), _load_receipt(args.public)
    except (OSError, ValueError):
        sys.stderr.write("FAIL malformed_input: receipt is unreadable or not a JSON object\n")
        return 3
    if detect_receipt_class(private) is None:
        sys.stderr.write("FAIL missing_receipt_class: private receipt declares no schema\n")
        return 3
    report = check_receipt_projection(private, public)
    if args.format == "json":
        sys.stdout.write(render_json(report))
    else:
        codes = ", ".join(item["code"] for item in report["issues"][:6])
        sys.stdout.write(f"{report['status']}: {report['issue_count']} issue(s) {codes}\n")
    return 0 if report["status"] == "projection_valid" else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
