"""Fail-closed, stage-1 intervention specifications for issue #9308.

This module records a bounded intervention design without applying it.  It is
deliberately separate from the existing counterfactual-pair evaluator and
replay contracts: those consumers operate on results, while this contract
binds one proposed factor, its controls, and the source identity needed before
any future execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from robot_sf.errors import RobotSfError

INTERVENTION_SPEC_SCHEMA_VERSION = "intervention_spec.v1"
SCHEMA_VERSION = INTERVENTION_SPEC_SCHEMA_VERSION
INTERVENTION_SPEC_ISSUE = 9308
INTERVENTION_SPEC_SCHEMA_PATH = Path(__file__).with_name("schemas") / "intervention_spec.v1.json"
SCHEMA_PATH = INTERVENTION_SPEC_SCHEMA_PATH
CLAIM_BOUNDARY = "diagnostic_only_no_execution"
EVIDENCE_TIER = "analysis_only"
STATUS = "specification_only"
COMPARISON_CLASSIFICATIONS = frozenset({"matched_start_replay", "genuine_shared_prefix"})
FACTOR_NAMES = frozenset({"visibility", "delay", "control_clipping", "planner_response"})
MATCH_BASIS_FIELDS = frozenset(
    {"scenario_id", "seed", "planner_id", "initial_state", "map_id", "config_digest"}
)
REQUIRED_HELD_FIXED_FIELDS = frozenset({"scenario_id", "seed", "planner_id", "initial_state"})
STOP_CONDITIONS = (
    "comparison_contract_failure",
    "factor_not_activated",
    "held_fixed_violation",
    "missing_or_ambiguous_field",
    "negative_control_activation",
    "source_identity_mismatch",
)
SOURCE_REF_ROLES = frozenset(
    {
        "case_dossier",
        "comparison_receipt",
        "mechanism_trace",
        "seed_sensitivity_replay",
        "trace_predicate",
    }
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\[\]-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FALLBACK_IDENTITIES = frozenset({"", "0", "none", "null", "unknown", "unavailable"})
_LOCAL_ONLY_PATH_PARTS = frozenset({".git", ".venv", "output", "results"})


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys and recursive aliases."""


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently keeping the last value.

    Returns:
        The object mapping when every key is unique.
    """

    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        mapping[key] = value
    return mapping


def _load_strict_json(text: str) -> Any:
    """Parse JSON while preserving the contract's fail-closed key semantics.

    Returns:
        The parsed JSON value.
    """

    return json.loads(text, object_pairs_hook=_reject_duplicate_json_object)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, _deep: bool = False
) -> dict[Any, Any]:
    """Construct a YAML mapping without duplicate keys or deferred cycles.

    Returns:
        The mapping with each key constructed exactly once.
    """

    if not isinstance(node, yaml.nodes.MappingNode):
        raise yaml.constructor.ConstructorError(None, None, "expected a mapping", node.start_mark)
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class InterventionSpecValidationError(RobotSfError, ValueError):
    """Raised when an intervention specification is missing or ambiguous."""

    def __init__(self, message: str, *, source: str | Path | None = None) -> None:
        """Initialize the error with an optional source-path prefix."""

        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + message)


def _canonical_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible values with the contract's stable profile.

    Returns:
        Canonical UTF-8 JSON bytes without a trailing newline.
    """

    _assert_json_value(value, "$")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _assert_json_value(value: Any, field: str, *, _active_ids: set[int] | None = None) -> None:
    """Reject values that are not finite, unambiguous JSON values."""

    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InterventionSpecValidationError(f"{field} must contain finite JSON numbers")
        return
    if type(value) is list or isinstance(value, Mapping):
        active_ids = _active_ids if _active_ids is not None else set()
        value_id = id(value)
        if value_id in active_ids:
            raise InterventionSpecValidationError(
                f"{field} must not contain recursive mappings or sequences"
            )
        active_ids.add(value_id)
        try:
            if type(value) is list:
                for index, child in enumerate(value):
                    _assert_json_value(child, f"{field}[{index}]", _active_ids=active_ids)
                return
            for key, child in value.items():
                if not isinstance(key, str):
                    raise InterventionSpecValidationError(f"{field} object keys must be strings")
                _assert_json_value(child, f"{field}.{key}", _active_ids=active_ids)
        finally:
            active_ids.remove(value_id)
        return
    raise InterventionSpecValidationError(f"{field} must contain only JSON-compatible values")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    """Require a mapping at a named contract field.

    Returns:
        The validated mapping.
    """

    if not isinstance(value, Mapping):
        raise InterventionSpecValidationError(f"{field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    """Require non-empty text and return its stripped form.

    Returns:
        The stripped text.
    """

    if not isinstance(value, str) or not value.strip():
        raise InterventionSpecValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, field: str) -> str:
    """Require a stable identifier rather than a free-form or fallback value.

    Returns:
        The validated identifier.
    """

    text = _text(value, field)
    if _ID_RE.fullmatch(text) is None:
        raise InterventionSpecValidationError(f"{field} must be a stable identifier")
    return text


def _field_path(value: Any, field: str) -> str:
    """Require a dotted logical field path without whitespace or coercion.

    Returns:
        The validated field path.
    """

    text = _text(value, field)
    if _PATH_RE.fullmatch(text) is None:
        raise InterventionSpecValidationError(f"{field} must be an unambiguous field path")
    return text


def _sha256(value: Any, field: str) -> str:
    """Require a lowercase SHA-256 digest.

    Returns:
        The validated digest.
    """

    digest = _text(value, field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise InterventionSpecValidationError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _commit(value: Any, field: str) -> str:
    """Require a full lowercase Git commit object id.

    Returns:
        The validated commit id.
    """

    commit = _text(value, field)
    if _COMMIT_RE.fullmatch(commit) is None:
        raise InterventionSpecValidationError(f"{field} must be a 40-character lowercase SHA-1")
    return commit


def _normalise_set(value: Any, field: str, *, allow_empty: bool) -> list[str]:
    """Validate a declared set and return a deterministic sorted list.

    Returns:
        Sorted unique field paths.
    """

    if type(value) is not list:
        raise InterventionSpecValidationError(f"{field} must be a list")
    if not allow_empty and not value:
        raise InterventionSpecValidationError(f"{field} must not be empty")
    values = [_field_path(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(set(values)) != len(values):
        raise InterventionSpecValidationError(f"{field} must not contain duplicate fields")
    return sorted(values)


def _validate_json_distinct(left: Any, right: Any, field: str) -> None:
    """Require two JSON values to be observably different under canonical JSON."""

    _assert_json_value(left, f"{field}.left")
    _assert_json_value(right, f"{field}.right")
    if _canonical_bytes(left) == _canonical_bytes(right):
        raise InterventionSpecValidationError(f"{field} must declare a changed value")


def _normalise_file_path(value: Any, field: str) -> str:
    """Validate a repository-relative durable source path.

    Returns:
        The normalized repository-relative path.
    """

    text = _text(value, field)
    path = Path(text)
    if (
        path.is_absolute()
        or "\\" in text
        or "." in path.parts
        or ".." in path.parts
        or not path.parts
        or path.parts[0] in _LOCAL_ONLY_PATH_PARTS
        or path.as_posix() != text
    ):
        raise InterventionSpecValidationError(
            f"{field} must be a normalized repository-relative durable path"
        )
    return text


def _validate_source_file(
    reference: Mapping[str, Any],
    *,
    repo_root: Path,
    field: str,
) -> None:
    """Verify one declared source file when an explicit repository root is supplied."""

    relative = _normalise_file_path(reference["path"], f"{field}.path")
    root = repo_root.resolve()
    unresolved = root / relative
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise InterventionSpecValidationError(
            f"{field}.path resolves outside the repository"
        ) from exc
    if unresolved.is_symlink() or not resolved.is_file():
        raise InterventionSpecValidationError(f"{field}.path is not a regular file: {relative}")
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual != reference["sha256"]:
        raise InterventionSpecValidationError(f"{field}.sha256 does not match source bytes")


@lru_cache(maxsize=1)
def load_intervention_spec_schema() -> dict[str, Any]:
    """Load and validate the checked-in intervention specification schema.

    Returns:
        Parsed JSON Schema mapping.
    """

    try:
        schema = _load_strict_json(INTERVENTION_SPEC_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise InterventionSpecValidationError(
            f"cannot load intervention specification schema: {INTERVENTION_SPEC_SCHEMA_PATH}"
        ) from exc
    if not isinstance(schema, dict):
        raise InterventionSpecValidationError("intervention specification schema must be a mapping")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:  # pragma: no cover - schema maintenance failure
        raise InterventionSpecValidationError(
            "intervention specification schema is invalid"
        ) from exc
    return schema


def _validate_schema(payload: Mapping[str, Any]) -> None:
    """Apply the machine-readable schema before semantic checks."""

    errors = sorted(
        Draft202012Validator(load_intervention_spec_schema()).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "payload"
        raise InterventionSpecValidationError(
            f"schema validation failed at {location}: {error.message}"
        )


def _validate_semantics(payload: dict[str, Any]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Validate relationships that JSON Schema cannot express safely."""

    _assert_json_value(payload, "payload")
    spec_id = _identifier(payload["spec_id"], "spec_id")
    payload["spec_id"] = spec_id

    hypothesis = _mapping(payload["hypothesis"], "hypothesis")
    hypothesis_copy = dict(hypothesis)
    hypothesis_copy["mechanism"] = _text(hypothesis["mechanism"], "hypothesis.mechanism")
    hypothesis_copy["statement"] = _text(hypothesis["statement"], "hypothesis.statement")
    payload["hypothesis"] = hypothesis_copy

    factor = _mapping(payload["factor"], "factor")
    factor_copy = dict(factor)
    factor_copy["name"] = _text(factor["name"], "factor.name")
    factor_copy["path"] = _field_path(factor["path"], "factor.path")
    factor_copy["unit"] = _text(factor["unit"], "factor.unit")
    _assert_json_value(factor_copy["baseline"], "factor.baseline")
    _assert_json_value(factor_copy["intervention"], "factor.intervention")
    _validate_json_distinct(factor_copy["baseline"], factor_copy["intervention"], "factor")
    payload["factor"] = factor_copy

    held_fixed = _normalise_set(payload["held_fixed"], "held_fixed", allow_empty=False)
    known_unfixable = _normalise_set(
        payload["known_unfixable"], "known_unfixable", allow_empty=True
    )
    if set(held_fixed) & set(known_unfixable):
        raise InterventionSpecValidationError(
            "held_fixed and known_unfixable must be disjoint declared sets"
        )
    if not REQUIRED_HELD_FIXED_FIELDS.issubset(held_fixed):
        raise InterventionSpecValidationError(
            "held_fixed must include scenario_id, seed, planner_id, and initial_state"
        )
    if factor_copy["path"] in set(held_fixed) | set(known_unfixable):
        raise InterventionSpecValidationError("factor.path cannot be held_fixed or known_unfixable")
    payload["held_fixed"] = held_fixed
    payload["known_unfixable"] = known_unfixable

    comparison = _mapping(payload["comparison"], "comparison")
    comparison_copy = dict(comparison)
    classification = _text(comparison["classification"], "comparison.classification")
    if classification not in COMPARISON_CLASSIFICATIONS:
        raise InterventionSpecValidationError(
            "comparison.classification must be matched_start_replay or genuine_shared_prefix"
        )
    match_basis = _normalise_set(
        comparison["match_basis"], "comparison.match_basis", allow_empty=False
    )
    if not set(match_basis).issubset(MATCH_BASIS_FIELDS):
        raise InterventionSpecValidationError(
            f"comparison.match_basis must use only {sorted(MATCH_BASIS_FIELDS)!r}"
        )
    if not {"scenario_id", "seed", "initial_state"}.issubset(match_basis):
        raise InterventionSpecValidationError(
            "comparison.match_basis must include scenario_id, seed, and initial_state"
        )
    required_prefix_steps = comparison["required_shared_prefix_steps"]
    if type(required_prefix_steps) is not int or required_prefix_steps < 0:
        raise InterventionSpecValidationError(
            "comparison.required_shared_prefix_steps must be a non-negative integer"
        )
    if classification == "matched_start_replay" and required_prefix_steps != 0:
        raise InterventionSpecValidationError(
            "matched_start_replay must require zero shared-prefix steps"
        )
    if classification == "genuine_shared_prefix" and required_prefix_steps < 1:
        raise InterventionSpecValidationError(
            "genuine_shared_prefix must require at least one shared-prefix step"
        )
    comparison_copy["classification"] = classification
    comparison_copy["match_basis"] = match_basis
    payload["comparison"] = comparison_copy

    negative_control = _mapping(payload["negative_control"], "negative_control")
    negative_control_copy = dict(negative_control)
    negative_control_copy["id"] = _identifier(negative_control["id"], "negative_control.id")
    negative_control_copy["factor_path"] = _field_path(
        negative_control["factor_path"], "negative_control.factor_path"
    )
    negative_control_copy["expected"] = _text(
        negative_control["expected"], "negative_control.expected"
    )
    negative_control_copy["rationale"] = _text(
        negative_control["rationale"], "negative_control.rationale"
    )
    _assert_json_value(negative_control_copy["value"], "negative_control.value")
    if negative_control_copy["factor_path"] != factor_copy["path"]:
        raise InterventionSpecValidationError(
            "negative_control.factor_path must name the declared factor for the no-op control"
        )
    if _canonical_bytes(negative_control_copy["value"]) != _canonical_bytes(
        factor_copy["baseline"]
    ):
        raise InterventionSpecValidationError(
            "negative_control.value must equal factor.baseline for the no-op control"
        )
    payload["negative_control"] = negative_control_copy

    stop_rule = _mapping(payload["stop_rule"], "stop_rule")
    stop_rule_copy = dict(stop_rule)
    conditions = _normalise_set(stop_rule["conditions"], "stop_rule.conditions", allow_empty=False)
    if set(conditions) != set(STOP_CONDITIONS):
        raise InterventionSpecValidationError(
            f"stop_rule.conditions must be exactly {list(STOP_CONDITIONS)!r}"
        )
    stop_rule_copy["conditions"] = conditions
    payload["stop_rule"] = stop_rule_copy

    provenance = _mapping(payload["provenance"], "provenance")
    provenance_copy = dict(provenance)
    source_identity = _mapping(provenance["source_identity"], "provenance.source_identity")
    source_identity_copy = dict(source_identity)
    for field in ("scenario_id", "planner_id", "episode_id"):
        identity = _text(source_identity[field], f"provenance.source_identity.{field}")
        if identity.lower() in _FALLBACK_IDENTITIES:
            raise InterventionSpecValidationError(
                f"provenance.source_identity.{field} must not be a fallback identity"
            )
        source_identity_copy[field] = identity
    seed = source_identity["seed"]
    if type(seed) is not int or seed < 0:
        raise InterventionSpecValidationError(
            "provenance.source_identity.seed must be a non-negative integer"
        )
    source_identity_copy["seed"] = seed
    source_identity_copy["source_kind"] = _text(
        source_identity["source_kind"], "provenance.source_identity.source_kind"
    )
    if source_identity_copy["source_kind"] != "existing_diagnostic_trace_or_dossier":
        raise InterventionSpecValidationError(
            "provenance.source_identity.source_kind must be existing_diagnostic_trace_or_dossier"
        )
    source_refs: list[dict[str, Any]] = []
    for index, raw_reference in enumerate(source_identity["source_refs"]):
        reference = _mapping(raw_reference, f"provenance.source_identity.source_refs[{index}]")
        reference_copy = dict(reference)
        reference_copy["path"] = _normalise_file_path(
            reference["path"], f"provenance.source_identity.source_refs[{index}].path"
        )
        reference_copy["sha256"] = _sha256(
            reference["sha256"], f"provenance.source_identity.source_refs[{index}].sha256"
        )
        reference_copy["role"] = _text(
            reference["role"], f"provenance.source_identity.source_refs[{index}].role"
        )
        if reference_copy["role"] not in SOURCE_REF_ROLES:
            raise InterventionSpecValidationError(
                f"provenance.source_identity.source_refs[{index}].role is unsupported"
            )
        source_refs.append(reference_copy)
    if len({reference["path"] for reference in source_refs}) != len(source_refs):
        raise InterventionSpecValidationError("provenance source references must use unique paths")
    source_refs.sort(key=lambda reference: reference["path"])
    source_identity_copy["source_refs"] = source_refs
    provenance_copy["source_identity"] = source_identity_copy

    config_identity = _mapping(provenance["config_identity"], "provenance.config_identity")
    config_identity_copy = dict(config_identity)
    config_identity_copy["config_id"] = _identifier(
        config_identity["config_id"], "provenance.config_identity.config_id"
    )
    config_identity_copy["path"] = _normalise_file_path(
        config_identity["path"], "provenance.config_identity.path"
    )
    config_identity_copy["sha256"] = _sha256(
        config_identity["sha256"], "provenance.config_identity.sha256"
    )
    provenance_copy["config_identity"] = config_identity_copy

    contract_identity = _mapping(provenance["contract_identity"], "provenance.contract_identity")
    contract_identity_copy = dict(contract_identity)
    if contract_identity["owner"] != "robot_sf.benchmark.intervention_spec":
        raise InterventionSpecValidationError(
            "provenance.contract_identity.owner must name the canonical intervention-spec owner"
        )
    if contract_identity["schema_version"] != INTERVENTION_SPEC_SCHEMA_VERSION:
        raise InterventionSpecValidationError(
            "provenance.contract_identity.schema_version must match intervention_spec.v1"
        )
    contract_identity_copy["base_commit"] = _commit(
        contract_identity["base_commit"], "provenance.contract_identity.base_commit"
    )
    provenance_copy["contract_identity"] = contract_identity_copy
    payload["provenance"] = provenance_copy


def _validate_bound_files(payload: Mapping[str, Any], repo_root: str | Path) -> None:
    """Verify declared config/source bytes when the caller supplies a checkout root."""

    root = Path(repo_root)
    if not root.is_dir():
        raise InterventionSpecValidationError(f"repo_root is not a directory: {root}")
    provenance = payload["provenance"]
    source_refs = provenance["source_identity"]["source_refs"]
    for index, reference in enumerate(source_refs):
        _validate_source_file(
            reference,
            repo_root=root,
            field=f"provenance.source_identity.source_refs[{index}]",
        )
    _validate_source_file(
        provenance["config_identity"],
        repo_root=root,
        field="provenance.config_identity",
    )


def validate_intervention_spec(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and normalize one stage-1 intervention specification.

    ``repo_root`` is optional because source artifacts may be held in an
    external durable store.  When supplied, every declared repository-relative
    source/config digest is checked against bytes in that checkout.  Omitting
    required identities or hashes always fails; the optional argument only
    controls whether local bytes are additionally verified.

    Returns:
        A deep-copied, normalized specification mapping.
    """

    if not isinstance(payload, Mapping):
        raise InterventionSpecValidationError(
            "intervention specification must be a mapping", source=source
        )
    try:
        normalized = copy.deepcopy(dict(payload))
        _assert_json_value(normalized, "payload")
        _validate_schema(normalized)
        _validate_semantics(normalized)
        if repo_root is not None:
            _validate_bound_files(normalized, repo_root)
    except InterventionSpecValidationError as exc:
        if source is not None and exc.source is None:
            raise InterventionSpecValidationError(str(exc), source=source) from exc
        raise
    except RecursionError as exc:
        raise InterventionSpecValidationError(
            "intervention specification contains a recursive mapping or sequence",
            source=source,
        ) from exc
    return normalized


def load_intervention_spec(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load YAML/JSON and validate it without executing an intervention.

    Returns:
        A normalized specification mapping.
    """

    spec_path = Path(path)
    try:
        text = spec_path.read_text(encoding="utf-8")
        payload = (
            _load_strict_json(text)
            if spec_path.suffix.lower() == ".json"
            else yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506
        )
    except RecursionError as exc:
        raise InterventionSpecValidationError(
            "cannot read intervention specification: recursive YAML alias",
            source=spec_path,
        ) from exc
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise InterventionSpecValidationError(
            f"cannot read intervention specification: {exc}", source=spec_path
        ) from exc
    if not isinstance(payload, Mapping):
        raise InterventionSpecValidationError(
            "intervention specification must be a mapping", source=spec_path
        )
    return validate_intervention_spec(payload, repo_root=repo_root, source=spec_path)


def compute_intervention_spec_digest(payload: Mapping[str, Any]) -> str:
    """Return the digest of a validated canonical specification payload.

    Returns:
        Lowercase SHA-256 digest of the canonical specification bytes.
    """

    normalized = validate_intervention_spec(payload)
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


__all__ = [
    "CLAIM_BOUNDARY",
    "COMPARISON_CLASSIFICATIONS",
    "EVIDENCE_TIER",
    "FACTOR_NAMES",
    "INTERVENTION_SPEC_ISSUE",
    "INTERVENTION_SPEC_SCHEMA_PATH",
    "INTERVENTION_SPEC_SCHEMA_VERSION",
    "MATCH_BASIS_FIELDS",
    "REQUIRED_HELD_FIXED_FIELDS",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "STATUS",
    "STOP_CONDITIONS",
    "InterventionSpecValidationError",
    "compute_intervention_spec_digest",
    "load_intervention_spec",
    "load_intervention_spec_schema",
    "validate_intervention_spec",
]
