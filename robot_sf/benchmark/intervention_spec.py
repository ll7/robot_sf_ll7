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
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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
# Keep canonical JSON serialization below CPython's decimal conversion limit.
_MAX_JSON_INTEGER_BITS = 4096
_GIT_PROBE_TIMEOUT_SECONDS = 5
# Exact case-insensitive tokens only; meaningful identifiers are not rejected by substring.
_FALLBACK_IDENTITIES = frozenset(
    {
        "",
        "0",
        "degraded",
        "failed",
        "fallback",
        "n/a",
        "na",
        "none",
        "not-available",
        "not_available",
        "null",
        "partial-failure",
        "partial_failure",
        "unknown",
        "unknown_id",
        "unknown_planner",
        "unknown_scenario",
        "unavailable",
    }
)
_LOCAL_ONLY_PATH_PARTS = frozenset({".git", ".venv", "output", "results"})


@dataclass(frozen=True)
class _GitCheckout:
    """Resolved Git paths for a sanitized provenance probe context."""

    root: Path
    git_dir: Path
    common_dir: Path
    object_dir: Path


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys and recursive aliases."""


class _LexicalFloat(float):
    """A parsed YAML or JSON float that retains its source number spelling."""

    lexeme: str

    def __new__(cls, lexeme: str) -> _LexicalFloat:
        """Create the normal float value while retaining its decimal lexeme."""

        value = super().__new__(cls, lexeme)
        value.lexeme = lexeme
        return value


def _parse_json_float(lexeme: str) -> _LexicalFloat:
    """Parse a JSON float without discarding its exact decimal spelling.

    Returns:
        The parsed float with its source decimal spelling attached.
    """

    return _LexicalFloat(lexeme)


def _construct_yaml_float(loader: _UniqueKeyLoader, node: yaml.nodes.ScalarNode) -> float:
    """Construct YAML floats while retaining exact decimal spellings.

    Returns:
        The parsed YAML float, with a decimal lexeme attached when supported.
    """

    parsed = yaml.constructor.SafeConstructor.construct_yaml_float(loader, node)
    if not math.isfinite(parsed):
        return parsed
    lexeme = loader.construct_scalar(node).replace("_", "")
    try:
        Decimal(lexeme)
    except InvalidOperation:
        return parsed
    return _LexicalFloat(lexeme)


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

    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_json_object,
        parse_float=_parse_json_float,
    )


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
_UniqueKeyLoader.add_constructor("tag:yaml.org,2002:float", _construct_yaml_float)


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


def _assert_json_integer(value: int, field: str) -> None:
    """Reject integers too large for bounded canonical JSON serialization."""

    if value.bit_length() > _MAX_JSON_INTEGER_BITS:
        raise InterventionSpecValidationError(
            f"{field} must contain a bounded JSON integer (at most {_MAX_JSON_INTEGER_BITS} bits)"
        )


def _assert_json_value(  # noqa: C901
    value: Any, field: str, *, _active_ids: set[int] | None = None
) -> None:
    """Reject values that are not finite, unambiguous JSON values."""

    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int):
        _assert_json_integer(value, field)
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


def _reject_fallback_identity(value: str, field: str) -> str:
    """Reject stable placeholder identities used by fallback or unavailable paths.

    Returns:
        The unchanged identity when it is not a reserved fallback sentinel.
    """

    if value.casefold() in _FALLBACK_IDENTITIES:
        raise InterventionSpecValidationError(f"{field} must not be a fallback identity")
    return value


def _identity_text(value: Any, field: str) -> str:
    """Require non-empty text that identifies a real source or experiment entity.

    Returns:
        The stripped, non-sentinel identity.
    """

    return _reject_fallback_identity(_text(value, field), field)


def _identifier(value: Any, field: str) -> str:
    """Require a stable identifier rather than a free-form or fallback value.

    Returns:
        The validated, non-sentinel identifier.
    """

    text = _text(value, field)
    if _ID_RE.fullmatch(text) is None:
        raise InterventionSpecValidationError(f"{field} must be a stable identifier")
    return _reject_fallback_identity(text, field)


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
    if commit == "0" * 40:
        raise InterventionSpecValidationError(f"{field} must not be the all-zero commit")
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


def _is_json_number(value: Any) -> bool:
    """Return whether ``value`` is a JSON number rather than a boolean."""

    return isinstance(value, int | float) and not isinstance(value, bool)


def _json_number_decimal(value: int | float) -> Decimal:
    """Return the exact decimal represented by one validated JSON number."""

    if isinstance(value, _LexicalFloat):
        return Decimal(value.lexeme)
    if isinstance(value, float):
        return Decimal(repr(value))
    return Decimal(value)


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare validated JSON values using JSON number semantics.

    JSON has one number type, so integral and floating-point spellings compare
    by numeric value. JSON strings, booleans, arrays, and objects retain their
    type distinctions for the no-op and changed-factor checks.

    Returns:
        Whether the values are equal under the contract's JSON semantics.
    """

    if _is_json_number(left) and _is_json_number(right):
        return _json_number_decimal(left) == _json_number_decimal(right)
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    return left == right


def _strip_numeric_lexemes(value: Any) -> Any:
    """Return a validated value without exposing loader-only float subclasses.

    Returns:
        The value with loader-only float subclasses converted to plain numbers.
    """

    if isinstance(value, _LexicalFloat):
        exact = Decimal(value.lexeme)
        if exact != Decimal(repr(value)) and exact == exact.to_integral_value():
            integer = int(exact)
            _assert_json_integer(integer, "numeric value")
            return integer
        return float(value)
    if type(value) is list:
        return [_strip_numeric_lexemes(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _strip_numeric_lexemes(item) for key, item in value.items()}
    return value


def _contains_numeric_lexemes(value: Any) -> bool:
    """Return whether a value still contains a loader-only numeric lexeme.

    Returns:
        Whether a lexical float marker occurs in the value tree.
    """

    if isinstance(value, _LexicalFloat):
        return True
    if type(value) is list:
        return any(_contains_numeric_lexemes(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_numeric_lexemes(item) for item in value.values())
    return False


def _validate_json_distinct(left: Any, right: Any, field: str) -> None:
    """Require two JSON values to differ under semantic JSON equality."""

    _assert_json_value(left, f"{field}.left")
    _assert_json_value(right, f"{field}.right")
    if _json_values_equal(left, right):
        raise InterventionSpecValidationError(f"{field} must declare a changed value")


def _normalise_file_path(value: Any, field: str) -> str:
    """Validate a repository-relative durable source path.

    Returns:
        The normalized repository-relative path.
    """

    text = _text(value, field)
    if "\x00" in text:
        raise InterventionSpecValidationError(f"{field} must not contain an embedded NUL")
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


def _reject_symlink_components(root: Path, relative: str, field: str) -> None:
    """Reject symlinks in every lexical component of a bound repository path."""

    candidate = root
    for component in Path(relative).parts:
        candidate /= component
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise InterventionSpecValidationError(
                f"{field}.path components cannot be inspected: {relative}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise InterventionSpecValidationError(
                f"{field}.path contains a symlink component: {relative}"
            )


def _validate_source_file(
    reference: Mapping[str, Any],
    *,
    checkout: _GitCheckout,
    base_commit: str,
    field: str,
) -> None:
    """Verify one declared source file against its immutable base-commit blob.

    The working-tree file is checked as a regular file and against its declared
    SHA-256. Git then proves that the same repository-relative path is a tracked
    regular-file blob at ``base_commit`` and that the current bytes have the
    exact Git blob identity recorded by that historical tree entry. Git probes
    use the sanitized, replacement-ref-free checkout context.
    """

    relative = _normalise_file_path(reference["path"], f"{field}.path")
    root = checkout.root
    unresolved = root / relative
    _reject_symlink_components(root, relative, field)
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise InterventionSpecValidationError(
            f"{field}.path resolves outside the repository"
        ) from exc
    if not resolved.is_file():
        raise InterventionSpecValidationError(f"{field}.path is not a regular file: {relative}")
    try:
        current_bytes = resolved.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise InterventionSpecValidationError(
            f"{field}.path current bytes cannot be read: {relative}"
        ) from exc
    actual = hashlib.sha256(current_bytes).hexdigest()
    if actual != reference["sha256"]:
        raise InterventionSpecValidationError(
            f"{field}.sha256 does not match source bytes in the current checkout"
        )

    historical_object_id = _historical_blob_identity(
        checkout, base_commit=base_commit, relative=relative, field=field
    )
    historical_blob = _git_command_bytes(
        checkout.root,
        ("cat-file", "blob", historical_object_id.decode("ascii")),
        checkout=checkout,
    )
    if historical_blob.returncode != 0 or historical_blob.stdout != current_bytes:
        raise InterventionSpecValidationError(
            f"{field}.path current bytes do not match the base_commit {base_commit} blob: "
            f"{relative}"
        )


def _historical_blob_identity(
    checkout: _GitCheckout,
    *,
    base_commit: str,
    relative: str,
    field: str,
) -> bytes:
    """Return the Git blob identity for a tracked regular file at ``base_commit``."""

    historical = _git_command_bytes(
        checkout.root,
        (
            "--literal-pathspecs",
            "ls-tree",
            "-z",
            "--full-tree",
            base_commit,
            "--",
            relative,
        ),
        checkout=checkout,
    )
    if historical.returncode != 0:
        raise InterventionSpecValidationError(
            f"{field}.path cannot be inspected at base_commit {base_commit}"
        )
    records = [record for record in historical.stdout.split(b"\0") if record]
    if not records:
        raise InterventionSpecValidationError(
            f"{field}.path is not tracked at base_commit {base_commit}: {relative}"
        )
    if len(records) != 1:
        raise InterventionSpecValidationError(
            f"{field}.path resolves ambiguously at base_commit {base_commit}: {relative}"
        )
    try:
        metadata, historical_path = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.split()
    except ValueError as exc:
        raise InterventionSpecValidationError(
            f"{field}.path has an invalid Git tree entry at base_commit {base_commit}"
        ) from exc
    try:
        encoded_path = os.fsencode(relative)
    except UnicodeError as exc:
        raise InterventionSpecValidationError(
            f"{field}.path cannot be encoded for Git at base_commit {base_commit}"
        ) from exc
    if historical_path != encoded_path:
        raise InterventionSpecValidationError(
            f"{field}.path is not the declared Git tree path at base_commit {base_commit}"
        )
    if mode not in {b"100644", b"100755"} or object_type != b"blob":
        raise InterventionSpecValidationError(
            f"{field}.path is not a tracked regular file blob at base_commit {base_commit}: "
            f"{relative}"
        )
    if re.fullmatch(rb"[0-9a-f]{40}", object_id) is None:
        raise InterventionSpecValidationError(
            f"{field}.path has an invalid Git blob identity at base_commit {base_commit}"
        )
    return object_id


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

    try:
        errors = sorted(
            Draft202012Validator(load_intervention_spec_schema()).iter_errors(payload),
            key=lambda error: list(error.absolute_path),
        )
    except InterventionSpecValidationError:
        raise
    except (OverflowError, ValueError) as exc:
        raise InterventionSpecValidationError(
            "schema validation failed for a bounded JSON value"
        ) from exc
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
    if not _json_values_equal(negative_control_copy["value"], factor_copy["baseline"]):
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
        source_identity_copy[field] = _identity_text(
            source_identity[field], f"provenance.source_identity.{field}"
        )
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


def _git_environment(checkout: _GitCheckout | None) -> dict[str, str]:
    """Build a Git environment with inherited repository controls removed.

    Returns:
        Environment variables safe for a provenance-only Git probe.
    """

    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    if checkout is not None:
        environment.update(
            {
                "GIT_COMMON_DIR": str(checkout.common_dir),
                "GIT_DIR": str(checkout.git_dir),
                "GIT_OBJECT_DIRECTORY": str(checkout.object_dir),
                "GIT_WORK_TREE": str(checkout.root),
            }
        )
    return environment


def _git_arguments(
    repo_root: Path, args: tuple[str, ...], checkout: _GitCheckout | None
) -> list[str]:
    """Build Git arguments bound to the discovered checkout when available.

    Returns:
        A no-replacement Git command argument vector.
    """

    command = ["git", "--no-replace-objects"]
    if checkout is not None:
        command.extend(
            [
                "--git-dir",
                str(checkout.git_dir),
                "--work-tree",
                str(checkout.root),
            ]
        )
    command.extend(["-C", str(repo_root), *args])
    return command


def _git_command(
    repo_root: Path,
    args: tuple[str, ...],
    *,
    checkout: _GitCheckout | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded, replacement-free Git probe without invoking a shell.

    Returns:
        The completed Git process result.
    """

    try:
        return subprocess.run(
            _git_arguments(repo_root, args, checkout),
            capture_output=True,
            check=False,
            shell=False,
            text=True,
            timeout=_GIT_PROBE_TIMEOUT_SECONDS,
            env=_git_environment(checkout),
        )
    except subprocess.TimeoutExpired as exc:
        raise InterventionSpecValidationError(
            "Git probe timed out while inspecting repo_root"
        ) from exc
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as exc:
        raise InterventionSpecValidationError("repo_root is not a usable Git checkout") from exc


def _git_command_bytes(
    repo_root: Path,
    args: tuple[str, ...],
    *,
    checkout: _GitCheckout | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a bounded, replacement-free Git probe while preserving file bytes.

    Returns:
        The completed Git process result.
    """

    try:
        return subprocess.run(
            _git_arguments(repo_root, args, checkout),
            capture_output=True,
            check=False,
            shell=False,
            timeout=_GIT_PROBE_TIMEOUT_SECONDS,
            env=_git_environment(checkout),
        )
    except subprocess.TimeoutExpired as exc:
        raise InterventionSpecValidationError(
            "Git probe timed out while inspecting repo_root"
        ) from exc
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as exc:
        raise InterventionSpecValidationError("repo_root is not a usable Git checkout") from exc


def _resolve_git_path(root: Path, args: tuple[str, ...], label: str) -> Path:
    """Resolve one absolute path reported by Git for a validated checkout.

    Returns:
        The resolved Git path.
    """

    probe = _git_command(root, ("rev-parse", "--path-format=absolute", *args))
    values = probe.stdout.splitlines()
    if probe.returncode != 0 or len(values) != 1 or not values[0].strip():
        raise InterventionSpecValidationError(f"repo_root Git {label} path cannot be resolved")
    value = values[0].strip()
    path = Path(value)
    if not path.is_absolute():
        raise InterventionSpecValidationError(f"repo_root Git {label} path is not absolute")
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise InterventionSpecValidationError(f"repo_root Git {label} path is unusable") from exc
    if not resolved.is_dir():
        raise InterventionSpecValidationError(f"repo_root Git {label} path is not a directory")
    return resolved


def _validate_git_object_store(object_dir: Path) -> None:
    """Reject configured alternate object databases during provenance probes."""

    alternates = object_dir / "info" / "alternates"
    try:
        try:
            mode = alternates.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise InterventionSpecValidationError(
                "repo_root Git object database has an unusable alternates file"
            )
        if alternates.read_bytes().strip():
            raise InterventionSpecValidationError(
                "repo_root Git object database uses alternate object storage"
            )
    except InterventionSpecValidationError:
        raise
    except OSError as exc:
        raise InterventionSpecValidationError(
            "repo_root Git object database alternates cannot be inspected"
        ) from exc


def _validate_git_checkout(repo_root: Path, commit: str) -> _GitCheckout:  # noqa: C901
    """Require a worktree and return its pinned, replacement-free Git context.

    Returns:
        The resolved checkout paths used for all subsequent Git probes.
    """

    try:
        root = repo_root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise InterventionSpecValidationError("repo_root is not a usable Git checkout") from exc
    if not root.is_dir():
        raise InterventionSpecValidationError(f"repo_root is not a directory: {repo_root}")

    worktree = _git_command(root, ("rev-parse", "--is-inside-work-tree"))
    if worktree.returncode != 0 or worktree.stdout.strip() != "true":
        raise InterventionSpecValidationError("repo_root must be a Git worktree")

    top_level = _git_command(root, ("rev-parse", "--show-toplevel"))
    top_level_text = top_level.stdout.strip()
    if top_level.returncode != 0 or not top_level_text:
        raise InterventionSpecValidationError("repo_root must be a Git worktree")
    try:
        top_level_root = Path(top_level_text).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise InterventionSpecValidationError("repo_root must be the Git checkout root") from exc
    if top_level_root != root:
        raise InterventionSpecValidationError("repo_root must be the Git checkout root")

    git_dir = _resolve_git_path(root, ("--absolute-git-dir",), "directory")
    common_dir = _resolve_git_path(root, ("--git-common-dir",), "common directory")
    object_dir = _resolve_git_path(root, ("--git-path", "objects"), "object directory")
    if object_dir != common_dir / "objects":
        raise InterventionSpecValidationError(
            "repo_root Git object directory is not bound to its common directory"
        )
    _validate_git_object_store(object_dir)
    checkout = _GitCheckout(
        root=root,
        git_dir=git_dir,
        common_dir=common_dir,
        object_dir=object_dir,
    )

    bound_top_level = _git_command(root, ("rev-parse", "--show-toplevel"), checkout=checkout)
    bound_top_level_text = bound_top_level.stdout.strip()
    if bound_top_level.returncode != 0 or not bound_top_level_text:
        raise InterventionSpecValidationError("repo_root Git context is not a worktree")
    try:
        bound_top_level_root = Path(bound_top_level_text).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise InterventionSpecValidationError(
            "repo_root Git context has an invalid worktree"
        ) from exc
    if bound_top_level_root != root:
        raise InterventionSpecValidationError("repo_root Git context is not bound to repo_root")

    commit_probe = _git_command(
        root,
        ("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"),
        checkout=checkout,
    )
    if commit_probe.returncode != 0 or commit_probe.stdout.strip() != commit:
        raise InterventionSpecValidationError(
            "provenance.contract_identity.base_commit is not a commit in repo_root"
        )
    return checkout


def _validate_bound_files(payload: Mapping[str, Any], repo_root: str | Path) -> None:
    """Verify source/config blobs against a sanitized checkout-root Git context."""

    if "\x00" in str(repo_root):
        raise InterventionSpecValidationError("repo_root must not contain an embedded NUL")
    root = Path(repo_root)
    base_commit = payload["provenance"]["contract_identity"]["base_commit"]
    checkout = _validate_git_checkout(root, base_commit)
    provenance = payload["provenance"]
    source_refs = provenance["source_identity"]["source_refs"]
    for index, reference in enumerate(source_refs):
        _validate_source_file(
            reference,
            checkout=checkout,
            base_commit=base_commit,
            field=f"provenance.source_identity.source_refs[{index}]",
        )
    _validate_source_file(
        provenance["config_identity"],
        checkout=checkout,
        base_commit=base_commit,
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
    external durable store. When supplied, it must be the top-level Git
    worktree containing the declared contract commit. Every declared
    repository-relative source/config path must be a tracked regular-file blob
    at that commit, and the current checkout bytes must match that historical
    blob and the declared SHA-256. Git directory/object environment overrides
    and replacement refs are ignored; lexical symlink components are rejected.
    Omitting ``repo_root`` leaves the commit as
    explicitly opaque metadata: its syntax is checked, but no local checkout or
    source bytes are claimed. Omitting required identities or hashes always
    fails.

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

    When ``repo_root`` is supplied, the specification's declared
    ``provenance.contract_identity.base_commit`` is the authority for every
    repository-relative source/config path. Each path must resolve to a
    tracked regular-file blob at that commit, and the current bytes must match
    both that blob and the declared SHA-256; inherited Git repository/object
    settings and replacement refs are ignored, and lexical symlink components
    are rejected.

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
    normalized = validate_intervention_spec(payload, repo_root=repo_root, source=spec_path)
    if not _contains_numeric_lexemes(normalized):
        return normalized
    normalized = _strip_numeric_lexemes(normalized)
    try:
        validate_intervention_spec(normalized)
    except InterventionSpecValidationError as exc:
        raise InterventionSpecValidationError(
            "numeric value cannot survive normalized loader round-trip without precision loss",
            source=spec_path,
        ) from exc
    return normalized


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
