"""Fail-closed retained-row contract for paired safety-wrapper effects.

The issue #4598 report builder consumes ``metric_values`` rows.  This module owns
the versioned field manifest and the producer-side validation used by the #6970
camera-ready runner gate.  It deliberately does not derive values from similarly
named legacy metrics: a missing or non-finite field is a hard contract failure.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONTRACT_SCHEMA_VERSION = "paired_effect_metric_contract.v1"
CLARIFICATION_SCHEMA_VERSION = "paired_effect_metric_clarification.v1"
PRODUCER_SCHEMA_VERSION = "paired_effect_metric_producer.v1"
REPORT_BUILDER_ISSUE = 4598
CLARIFICATION_ISSUE = 8567
REQUIRED_METRIC_NAMES: tuple[str, ...] = (
    "exact_collision_probability",
    "near_miss_probability",
    "min_predicted_separation_m",
    "completion_probability",
    "progress_at_timeout",
    "false_positive_stop_rate",
    "stop_yield_latency_s",
    "wrapper_intervention_rate",
)
CLARIFICATION_METRIC_NAMES: tuple[str, ...] = (
    "false_positive_stop_rate",
    "stop_yield_latency_s",
    "progress_at_timeout",
)
PAIRING_KEY_FIELDS: tuple[str, ...] = ("planner", "scenario_id", "seed")
WRAPPER_ARM_KEYS: tuple[str, ...] = ("wrapper_off", "wrapper_on")
_STOP_YIELD_INTERVENTIONS = frozenset({"hard_stop", "yield"})
_TIMEOUT_TERMINATION_REASONS = frozenset({"max_steps", "truncated"})
_REQUIRED_NATIVE_STATE = (
    "wrapper arm and eligible step state",
    "intervention label and declared final forward-progress command state",
    "post-step canonical collision and near-miss flags",
    "declared termination, horizon, start-to-goal denominator, and final robot position",
)
_DEFAULT_CLARIFICATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/benchmarks/paired_effect_metric_contract_v1_clarification.yaml"
)
_COUNTERFACTUAL_WINDOW_S = 2.0
REQUIRED_RETAINED_ROW_PATH = "metric_values"
MAX_INVALID_ROW_SAMPLES = 10
_REQUIRED_FIELD_KEYS = {
    "name",
    "path",
    "unit",
    "definition",
    "emitting_component",
    "representation",
    "value_type",
}
_REPRESENTATIONS = {"raw", "normalized"}
_VALUE_TYPES = {"finite_scalar"}
_PATH_PREFIX = f"{REQUIRED_RETAINED_ROW_PATH}."
_BOUNDED_METRIC_NAMES = {
    "exact_collision_probability",
    "near_miss_probability",
    "completion_probability",
    "progress_at_timeout",
    "false_positive_stop_rate",
    "wrapper_intervention_rate",
}


class PairedEffectMetricContractError(ValueError):
    """Raised when a retained-row contract or its episode records is invalid."""


def load_paired_effect_metric_clarification(path: str | Path) -> dict[str, Any]:
    """Load and validate the versioned #8567 clarification companion.

    Returns:
        Normalized clarification mapping.
    """

    clarification_path = Path(path)
    try:
        payload = yaml.safe_load(clarification_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PairedEffectMetricContractError(
            f"paired-effect clarification cannot be read: {clarification_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PairedEffectMetricContractError(
            f"paired-effect clarification must be a mapping: {clarification_path}"
        )
    return validate_paired_effect_metric_clarification(payload, source=clarification_path)


def validate_paired_effect_metric_clarification(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the native paired-effect definitions without changing v1 bytes.

    The companion is intentionally a separate schema.  It describes how the three
    deferred fields may be derived, while :func:`validate_paired_effect_metric_contract`
    continues to validate the frozen #6970 retained-row manifest unchanged.

    Returns:
        Normalized clarification mapping.
    """

    location = f" in {source}" if source is not None else ""
    normalized = dict(payload)
    expected = {
        "schema_version": CLARIFICATION_SCHEMA_VERSION,
        "issue": CLARIFICATION_ISSUE,
        "parent_contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "parent_issue": 6970,
    }
    for key, value in expected.items():
        if normalized.get(key) != value:
            raise PairedEffectMetricContractError(f"{key} must be {value!r}{location}")
    claim_boundary = normalized.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise PairedEffectMetricContractError(f"claim_boundary must be non-empty{location}")

    pairing = normalized.get("pairing")
    if not isinstance(pairing, Mapping):
        raise PairedEffectMetricContractError(f"pairing must be a mapping{location}")
    if tuple(pairing.get("key_fields") or ()) != PAIRING_KEY_FIELDS:
        raise PairedEffectMetricContractError(
            f"pairing.key_fields must be {list(PAIRING_KEY_FIELDS)!r}{location}"
        )
    if tuple(pairing.get("arm_keys") or ()) != WRAPPER_ARM_KEYS:
        raise PairedEffectMetricContractError(
            f"pairing.arm_keys must be {list(WRAPPER_ARM_KEYS)!r}{location}"
        )
    for key in ("arm_field", "config_identity", "source_identity"):
        value = pairing.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PairedEffectMetricContractError(f"pairing.{key} must be non-empty{location}")

    native_trace = normalized.get("native_trace")
    if not isinstance(native_trace, Mapping):
        raise PairedEffectMetricContractError(f"native_trace must be a mapping{location}")
    if native_trace.get("schema_version") != "paired_effect_native_trace.v1":
        raise PairedEffectMetricContractError(
            f"native_trace.schema_version must be 'paired_effect_native_trace.v1'{location}"
        )
    for key in ("wrapper_trace_path", "simulation_trace_path", "time_basis"):
        value = native_trace.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PairedEffectMetricContractError(f"native_trace.{key} must be non-empty{location}")
    if tuple(native_trace.get("required_state") or ()) != _REQUIRED_NATIVE_STATE:
        raise PairedEffectMetricContractError(
            f"native_trace.required_state must declare the exact native producer state{location}"
        )
    if normalized.get("unavailable_status") != "unavailable":
        raise PairedEffectMetricContractError(f"unavailable_status must be 'unavailable'{location}")
    if normalized.get("invalid_status") != "invalid":
        raise PairedEffectMetricContractError(f"invalid_status must be 'invalid'{location}")
    if normalized.get("no_campaign_compute") is not True:
        raise PairedEffectMetricContractError(f"no_campaign_compute must be true{location}")
    raw_window_s = normalized.get("counterfactual_window_s")
    if isinstance(raw_window_s, bool):
        raise PairedEffectMetricContractError(
            f"counterfactual_window_s must be {_COUNTERFACTUAL_WINDOW_S!r}{location}"
        )
    try:
        window_s = float(raw_window_s)
    except (TypeError, ValueError):
        raise PairedEffectMetricContractError(
            f"counterfactual_window_s must be {_COUNTERFACTUAL_WINDOW_S!r}{location}"
        ) from None
    if not math.isfinite(window_s) or not math.isclose(
        window_s, _COUNTERFACTUAL_WINDOW_S, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise PairedEffectMetricContractError(
            f"counterfactual_window_s must be {_COUNTERFACTUAL_WINDOW_S!r}{location}"
        )
    normalized["counterfactual_window_s"] = window_s

    raw_fields = normalized.get("fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise PairedEffectMetricContractError(f"fields must be a list{location}")
    if len(raw_fields) != len(CLARIFICATION_METRIC_NAMES):
        raise PairedEffectMetricContractError(
            f"fields must contain exactly {len(CLARIFICATION_METRIC_NAMES)} entries{location}"
        )
    required_field_keys = {
        "name",
        "source",
        "time_alignment",
        "window_or_denominator",
        "aggregation",
        "no_opportunity",
        "unavailable_reasons",
        "invalid_reasons",
    }
    fields: list[dict[str, Any]] = []
    for index, raw_field in enumerate(raw_fields):
        if not isinstance(raw_field, Mapping):
            raise PairedEffectMetricContractError(f"fields[{index}] must be a mapping{location}")
        missing = sorted(required_field_keys - set(raw_field))
        if missing:
            raise PairedEffectMetricContractError(
                f"fields[{index}] missing required keys {missing}{location}"
            )
        field = dict(raw_field)
        name = field.get("name")
        if not isinstance(name, str) or not name.strip():
            raise PairedEffectMetricContractError(
                f"fields[{index}].name must be non-empty{location}"
            )
        field["name"] = name.strip()
        for key in required_field_keys - {"name", "unavailable_reasons", "invalid_reasons"}:
            value = field.get(key)
            if not isinstance(value, str) or not value.strip():
                raise PairedEffectMetricContractError(
                    f"fields[{index}].{key} must be non-empty{location}"
                )
            field[key] = value.strip()
        for key in ("unavailable_reasons", "invalid_reasons"):
            values = field.get(key)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                raise PairedEffectMetricContractError(
                    f"fields[{index}].{key} must be a non-empty list{location}"
                )
            field[key] = [str(value).strip() for value in values if str(value).strip()]
            if len(field[key]) != len(values):
                raise PairedEffectMetricContractError(
                    f"fields[{index}].{key} must contain non-empty strings{location}"
                )
        fields.append(field)
    if tuple(field["name"] for field in fields) != CLARIFICATION_METRIC_NAMES:
        raise PairedEffectMetricContractError(
            "fields must use the exact deferred-field order "
            f"{list(CLARIFICATION_METRIC_NAMES)!r}{location}"
        )
    normalized["pairing"] = dict(pairing)
    normalized["native_trace"] = dict(native_trace)
    normalized["fields"] = fields
    normalized["required_metric_names"] = list(CLARIFICATION_METRIC_NAMES)
    return normalized


@lru_cache(maxsize=1)
def _load_default_paired_effect_metric_clarification() -> dict[str, Any]:
    """Load the repository-owned #8567 companion once per interpreter.

    Returns:
        Normalized clarification mapping.
    """

    return load_paired_effect_metric_clarification(_DEFAULT_CLARIFICATION_PATH)


def load_paired_effect_metric_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate one versioned paired-effect metric contract.

    Returns:
        Normalized contract mapping.
    """

    contract_path = Path(path)
    try:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PairedEffectMetricContractError(
            f"paired-effect metric contract cannot be read: {contract_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PairedEffectMetricContractError(
            f"paired-effect metric contract must be a mapping: {contract_path}"
        )
    return validate_paired_effect_metric_contract(payload, source=contract_path)


def validate_paired_effect_metric_contract(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and return a normalized retained-row field manifest.

    Validation is intentionally strict: the report-builder outcome roster, field
    paths, representations, and producer metadata are all fixed before a run.

    Returns:
        Normalized contract mapping.
    """

    location = f" in {source}" if source is not None else ""
    normalized = dict(payload)
    _validate_contract_header(normalized, location=location)

    raw_fields = normalized.get("fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise PairedEffectMetricContractError(f"fields must be a list{location}")
    if len(raw_fields) != len(REQUIRED_METRIC_NAMES):
        raise PairedEffectMetricContractError(
            f"fields must contain exactly {len(REQUIRED_METRIC_NAMES)} entries{location}"
        )

    fields = [
        _validate_contract_field(field, index=index, location=location)
        for index, field in enumerate(raw_fields)
    ]

    if tuple(field["name"] for field in fields) != REQUIRED_METRIC_NAMES:
        raise PairedEffectMetricContractError(
            "fields must use the exact #4598 outcome order "
            f"{list(REQUIRED_METRIC_NAMES)!r}{location}"
        )
    normalized["fields"] = fields
    normalized["required_metric_names"] = list(REQUIRED_METRIC_NAMES)
    return normalized


def _validate_contract_header(payload: Mapping[str, Any], *, location: str) -> None:
    """Validate fixed contract-level identifiers."""
    expected_values = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "issue": 6970,
        "report_builder_issue": REPORT_BUILDER_ISSUE,
        "retained_row_path": REQUIRED_RETAINED_ROW_PATH,
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise PairedEffectMetricContractError(f"{key} must be {expected!r}{location}")
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise PairedEffectMetricContractError(f"claim_boundary must be non-empty{location}")


def _validate_contract_field(
    raw_field: Any,
    *,
    index: int,
    location: str,
) -> dict[str, Any]:
    """Validate and normalize one retained metric field declaration.

    Returns:
        Normalized field mapping.
    """
    if not isinstance(raw_field, Mapping):
        raise PairedEffectMetricContractError(f"fields[{index}] must be a mapping{location}")
    missing = sorted(_REQUIRED_FIELD_KEYS - set(raw_field))
    if missing:
        raise PairedEffectMetricContractError(
            f"fields[{index}] missing required keys {missing}{location}"
        )
    field = dict(raw_field)
    name = field.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PairedEffectMetricContractError(f"fields[{index}].name must be non-empty{location}")
    name = name.strip()
    path = field.get("path")
    if not isinstance(path, str) or path != f"{_PATH_PREFIX}{name}":
        raise PairedEffectMetricContractError(
            f"fields[{index}].path must be {_PATH_PREFIX}{name}{location}"
        )
    for key in ("unit", "definition", "emitting_component"):
        value = field.get(key)
        if not isinstance(value, str) or not value.strip() or "<" in value or ">" in value:
            raise PairedEffectMetricContractError(
                f"fields[{index}].{key} must be a concrete non-placeholder string{location}"
            )
        field[key] = value.strip()
    if field.get("representation") not in _REPRESENTATIONS:
        raise PairedEffectMetricContractError(
            f"fields[{index}].representation must be raw or normalized{location}"
        )
    if field.get("value_type") not in _VALUE_TYPES:
        raise PairedEffectMetricContractError(
            f"fields[{index}].value_type must be finite_scalar{location}"
        )
    if name in _BOUNDED_METRIC_NAMES and field.get("bounds") != {
        "lower": 0.0,
        "upper": 1.0,
    }:
        raise PairedEffectMetricContractError(
            f"fields[{index}].bounds must be {{'lower': 0.0, 'upper': 1.0}}{location}"
        )
    field["name"] = name
    return field


def validate_paired_effect_metric_record(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    row_index: int | None = None,
) -> dict[str, Any]:
    """Validate one retained episode row against a validated contract.

    ``None``, booleans, non-numeric values, non-finite values, and out-of-range
    normalized values are all rejected.  Similar legacy fields are never used as
    aliases or fallbacks.

    Returns:
        JSON-safe row validation report.
    """

    validated = validate_paired_effect_metric_contract(contract)
    prefix = f"row {row_index}: " if row_index is not None else ""
    missing: list[str] = []
    invalid: list[dict[str, Any]] = []
    metric_values = record.get(REQUIRED_RETAINED_ROW_PATH)
    if not isinstance(metric_values, Mapping):
        missing.extend(REQUIRED_METRIC_NAMES)
        return {
            "status": "blocked",
            "row_index": row_index,
            "missing_fields": missing,
            "invalid_fields": invalid,
            "message": f"{prefix}missing mapping {REQUIRED_RETAINED_ROW_PATH!r}",
        }

    for field in validated["fields"]:
        name = str(field["name"])
        value = metric_values.get(name)
        if value is None:
            missing.append(str(field["path"]))
            continue
        if isinstance(value, bool):
            invalid.append({"field": str(field["path"]), "reason": "boolean_is_not_scalar"})
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid.append({"field": str(field["path"]), "reason": "not_numeric"})
            continue
        if not math.isfinite(numeric):
            invalid.append({"field": str(field["path"]), "reason": "non_finite"})
            continue
        bounds = field.get("bounds")
        if isinstance(bounds, Mapping):
            lower = float(bounds["lower"])
            upper = float(bounds["upper"])
            if numeric < lower or numeric > upper:
                invalid.append(
                    {
                        "field": str(field["path"]),
                        "reason": "out_of_bounds",
                        "value": numeric,
                        "bounds": {"lower": lower, "upper": upper},
                    }
                )

    status = "ok" if not missing and not invalid else "blocked"
    return {
        "status": status,
        "row_index": row_index,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "message": None if status == "ok" else f"{prefix}retained metric contract failed",
    }


def _finite_metric_scalar(value: Any) -> float | None:
    """Return a finite non-boolean scalar, preserving unavailable values."""

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _status_payload(
    status: str,
    reason: str,
    *,
    value: float | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Build one machine-readable field status payload.

    Returns:
        Status payload with optional value and details.
    """

    payload: dict[str, Any] = {"status": status, "reason": reason}
    if value is not None:
        payload["value"] = float(value)
    if details:
        payload["details"] = details
    return payload


def _canonical_digest(value: Any) -> str | None:
    """Hash a JSON-safe mapping without permitting non-finite values.

    Returns:
        SHA-256 hex digest, or ``None`` when the value is not JSON-safe.
    """

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _record_pair_identity(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the canonical ``[planner, scenario_id, seed]`` identity.

    Returns:
        Pair identity and an unavailable reason, respectively.
    """

    scenario_params = record.get("scenario_params")
    scenario_params = scenario_params if isinstance(scenario_params, Mapping) else {}
    planner_raw = record.get("algo")
    if planner_raw is None:
        planner_raw = record.get("planner")
    if planner_raw is None:
        planner_raw = scenario_params.get("algo")
    scenario_raw = record.get("scenario_id")
    if scenario_raw is None:
        scenario_raw = scenario_params.get("id") or scenario_params.get("name")
    seed = record.get("seed")
    if not isinstance(planner_raw, str) or not planner_raw.strip():
        return None, "pair_identity_missing_planner"
    if not isinstance(scenario_raw, str) or not scenario_raw.strip():
        return None, "pair_identity_missing_scenario_id"
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None, "pair_identity_missing_seed"
    return {
        "planner": planner_raw.strip(),
        "scenario_id": scenario_raw.strip(),
        "seed": int(seed),
    }, None


def _record_source_commit(record: Mapping[str, Any]) -> str | None:
    """Return only the declared native source commit identity.

    Pair-level causal fields must not fall back to generic row provenance: a copied or
    hand-authored generic ``git_hash`` is not evidence that both arms share the native
    producer implementation.
    """

    metadata = record.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        native = metadata.get("paired_effect_native_trace")
        if isinstance(native, Mapping):
            value = native.get("source_commit")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _native_pair_config_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the native arm-independent config hash against record contents.

    Returns:
        Machine-readable identity status and, when valid, the recomputed hash.
    """

    native = _record_native_trace_metadata(record)
    if native is None:
        return _status_payload("unavailable", "pair_config_identity_unavailable")
    value = native.get("pair_config_hash")
    if not isinstance(value, str) or not value.strip():
        return _status_payload("unavailable", "pair_config_identity_unavailable")
    scenario_params = record.get("scenario_params")
    if not isinstance(scenario_params, Mapping):
        return _status_payload("unavailable", "pair_config_identity_unavailable")
    arm_independent = dict(scenario_params)
    arm_independent.pop("safety_wrapper", None)
    digest = _canonical_digest(arm_independent)
    expected = digest[:16] if digest is not None else None
    if expected is None:
        return _status_payload("unavailable", "pair_config_identity_unavailable")
    if value.strip() != expected:
        return _status_payload(
            "invalid",
            "pair_config_identity_mismatch",
            declared=value.strip(),
            recomputed=expected,
        )
    return {"status": "available", "reason": None, "value": value.strip()}


def _record_pair_config_hash(record: Mapping[str, Any]) -> str | None:
    """Return a recomputed-and-validated native config identity for metadata only."""

    identity = _native_pair_config_identity(record)
    return identity.get("value") if identity.get("status") == "available" else None


def _record_safety_wrapper_summary(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the canonical wrapper summary from one native episode record."""

    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        return None
    summary = metadata.get("safety_wrapper")
    return summary if isinstance(summary, Mapping) else None


def _record_native_trace_metadata(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the #8567 native trace metadata block, when present."""

    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        return None
    native = metadata.get("paired_effect_native_trace")
    return native if isinstance(native, Mapping) else None


def _declared_trace_time_step(
    record: Mapping[str, Any],
    summary: Mapping[str, Any],
    native: Mapping[str, Any],
) -> dict[str, Any]:
    """Read one declared trace time step without repairing invalid declarations.

    Returns:
        Time-step status and finite value when available.
    """

    candidates = (
        (native, "dt_s"),
        (summary, "time_per_step_s"),
        (
            record.get("scenario_params")
            if isinstance(record.get("scenario_params"), Mapping)
            else {},
            "run_dt",
        ),
    )
    observed: list[float] = []
    for owner, key in candidates:
        if key not in owner:
            continue
        numeric = _finite_metric_scalar(owner.get(key))
        if numeric is None:
            return _status_payload("invalid", "nonfinite_trace_time_step")
        if numeric <= 0.0:
            return _status_payload("invalid", "invalid_trace_time_step")
        observed.append(float(numeric))
    if not observed:
        return _status_payload("unavailable", "missing_trace_time_step")
    reference = observed[0]
    if any(
        not math.isclose(value, reference, rel_tol=0.0, abs_tol=1.0e-12) for value in observed[1:]
    ):
        return _status_payload("invalid", "trace_time_step_mismatch")
    return {"status": "available", "reason": None, "value": reference}
    return _status_payload("unavailable", "missing_trace_time_step")


def _trace_view(record: Mapping[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912
    """Validate and index one native wrapper trace without repairing it.

    Returns:
        Trace status, indexed steps, and trace provenance when available.
    """

    summary = _record_safety_wrapper_summary(record)
    if summary is None:
        return _status_payload("unavailable", "missing_safety_wrapper_summary")
    if summary.get("schema_version") != "safety_wrapper_episode_summary.v1":
        return _status_payload("invalid", "invalid_safety_wrapper_summary_schema")
    arm = summary.get("arm_key")
    if arm not in WRAPPER_ARM_KEYS:
        return _status_payload("invalid", "invalid_wrapper_arm")
    raw_trace = summary.get("step_trace")
    if not isinstance(raw_trace, Sequence) or isinstance(raw_trace, (str, bytes)):
        return _status_payload("unavailable", "missing_safety_wrapper_trace")
    if not raw_trace:
        return _status_payload("unavailable", "empty_safety_wrapper_trace")

    native = _record_native_trace_metadata(record)
    if native is None:
        return _status_payload("unavailable", "missing_paired_effect_native_trace")
    if native.get("schema_version") != "paired_effect_native_trace.v1":
        return _status_payload("invalid", "invalid_native_trace_schema")
    native_arm = native.get("arm_key")
    if native_arm is not None and native_arm != arm:
        return _status_payload("invalid", "native_trace_arm_mismatch")
    time_step = _declared_trace_time_step(record, summary, native)
    if time_step["status"] != "available":
        return time_step
    dt_s = float(time_step["value"])

    trace: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    previous_step: int | None = None
    for index, raw_step in enumerate(raw_trace):
        if not isinstance(raw_step, Mapping):
            return _status_payload("invalid", "invalid_trace_step", index=index)
        step = raw_step.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            return _status_payload("invalid", "invalid_trace_step_index", index=index)
        if step in seen_steps:
            return _status_payload("invalid", "duplicate_trace_step", step=step)
        if previous_step is not None and step != previous_step + 1:
            return _status_payload("unavailable", "missing_trace_step", step=previous_step + 1)
        if previous_step is None and step != 0:
            return _status_payload("unavailable", "missing_trace_step", step=0)
        if raw_step.get("schema_version") != "safety_wrapper_runtime_step.v1":
            return _status_payload("invalid", "invalid_safety_wrapper_step_schema", step=step)
        if raw_step.get("arm_key") != arm:
            return _status_payload("invalid", "trace_step_arm_mismatch", step=step)
        eligible = raw_step.get("eligible_for_wrapper")
        if not isinstance(eligible, bool):
            return _status_payload("unavailable", "missing_eligible_wrapper_state", step=step)
        intervention = raw_step.get("intervention")
        if not isinstance(intervention, str) or not intervention.strip():
            return _status_payload("unavailable", "missing_intervention_state", step=step)
        if intervention not in {"disabled", "none", "speed_cap", "hard_stop", "yield"}:
            return _status_payload("invalid", "invalid_intervention_state", step=step)
        intervened = raw_step.get("intervened")
        if not isinstance(intervened, bool):
            return _status_payload("unavailable", "missing_intervention_flag", step=step)
        expected_intervened = intervention in {"speed_cap", "hard_stop", "yield"}
        if intervened != expected_intervened:
            return _status_payload("invalid", "intervention_state_mismatch", step=step)
        time_s = raw_step.get("time_s")
        numeric_time = _finite_metric_scalar(time_s)
        expected_time = (step + 1) * dt_s
        if numeric_time is None:
            return _status_payload("invalid", "nonfinite_trace_time", step=step)
        if not math.isclose(numeric_time, expected_time, rel_tol=0.0, abs_tol=1.0e-9):
            return _status_payload("invalid", "trace_time_mismatch", step=step)
        trace.append(dict(raw_step))
        seen_steps.add(step)
        previous_step = step

    # A declared timeout is only complete when its wrapper trace reaches the
    # declared horizon. Without this check, a truncated timeout trace could be
    # misclassified as a valid no-opportunity or recovered intervention.
    if native.get("declared_timeout") is True:
        horizon = native.get("horizon_steps")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            return _status_payload("invalid", "invalid_declared_horizon")
        if int(trace[-1]["step"]) != horizon - 1:
            return _status_payload("unavailable", "timeout_trace_truncated")

    return {
        "status": "available",
        "reason": None,
        "arm_key": str(arm),
        "dt_s": float(dt_s),
        "trace": trace,
        "by_step": {int(step["step"]): step for step in trace},
        "max_step": int(trace[-1]["step"]),
        "trace_digest": _canonical_digest(trace),
    }


def _validate_pair(  # noqa: C901
    wrapper_on_record: Mapping[str, Any],
    wrapper_off_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the explicit wrapper-on/off native counterpart handoff.

    Returns:
        Pair status, identity, provenance, and indexed arm traces.
    """

    if wrapper_off_record is None:
        return _status_payload("unavailable", "unpaired_counterfactual")
    on_identity, on_identity_reason = _record_pair_identity(wrapper_on_record)
    off_identity, off_identity_reason = _record_pair_identity(wrapper_off_record)
    if on_identity is None or off_identity is None:
        return _status_payload(
            "unavailable",
            on_identity_reason or off_identity_reason or "pair_identity_unavailable",
        )
    if on_identity != off_identity:
        return {
            **_status_payload("invalid", "pair_identity_mismatch"),
            "key": {"wrapper_on": on_identity, "wrapper_off": off_identity},
        }

    on_trace = _trace_view(wrapper_on_record)
    off_trace = _trace_view(wrapper_off_record)
    if on_trace["status"] != "available":
        return _status_payload(
            str(on_trace["status"]),
            str(on_trace["reason"]),
            side="wrapper_on",
        )
    if off_trace["status"] != "available":
        return _status_payload(
            str(off_trace["status"]),
            str(off_trace["reason"]),
            side="wrapper_off",
        )
    if on_trace["arm_key"] != "wrapper_on" or off_trace["arm_key"] != "wrapper_off":
        return _status_payload("invalid", "wrapper_arm_mismatch")
    off_interventions = {
        str(step.get("intervention"))
        for step in off_trace["trace"]
        if str(step.get("intervention")) in {"speed_cap", "hard_stop", "yield"}
    }
    if off_interventions:
        return _status_payload(
            "invalid",
            "wrapper_off_intervention_present",
            interventions=sorted(off_interventions),
        )

    on_config_state = _native_pair_config_identity(wrapper_on_record)
    off_config_state = _native_pair_config_identity(wrapper_off_record)
    for config_state in (on_config_state, off_config_state):
        if config_state["status"] != "available":
            return _status_payload(
                str(config_state["status"]),
                str(config_state["reason"]),
                side="wrapper_on" if config_state is on_config_state else "wrapper_off",
                **(
                    config_state.get("details", {})
                    if isinstance(config_state.get("details"), Mapping)
                    else {}
                ),
            )
    on_config = str(on_config_state["value"])
    off_config = str(off_config_state["value"])
    if on_config != off_config:
        return _status_payload("invalid", "pair_config_mismatch")
    on_source = _record_source_commit(wrapper_on_record)
    off_source = _record_source_commit(wrapper_off_record)
    if on_source is None or off_source is None:
        return _status_payload(
            "unavailable",
            "pair_source_identity_unavailable",
            side="wrapper_on" if on_source is None else "wrapper_off",
        )
    if on_source != off_source:
        return _status_payload("invalid", "pair_source_mismatch")
    if not math.isclose(on_trace["dt_s"], off_trace["dt_s"], rel_tol=0.0, abs_tol=1.0e-12):
        return _status_payload("invalid", "pair_time_step_mismatch")
    return {
        "status": "available",
        "reason": None,
        "key": dict(on_identity),
        "arm_keys": {"wrapper_on": "wrapper_on", "wrapper_off": "wrapper_off"},
        "pair_config_hash": on_config,
        "source_commit": on_source,
        "time_per_step_s": float(on_trace["dt_s"]),
        "trace_digests": {
            "wrapper_on": on_trace.get("trace_digest"),
            "wrapper_off": off_trace.get("trace_digest"),
        },
        "wrapper_on_trace": on_trace,
        "wrapper_off_trace": off_trace,
    }


def _stop_yield_events(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    """Group consecutive hard-stop/yield steps into intervention events.

    Returns:
        Inclusive onset/end step pairs in trace order.
    """

    events: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    for raw_step in trace:
        step = int(raw_step["step"])
        is_stop = raw_step.get("intervention") in _STOP_YIELD_INTERVENTIONS
        if not is_stop:
            current = None
            continue
        if current is not None and step == current["end_step"] + 1:
            current["end_step"] = step
            continue
        current = {"onset_step": step, "end_step": step}
        events.append(current)
    return events


def _forward_progress_state(step: Mapping[str, Any]) -> dict[str, Any]:
    """Read the declared final forward-progress command state.

    Returns:
        Availability, validity, and numeric command state.
    """

    if not bool(step.get("eligible_for_wrapper", True)):
        return _status_payload("unavailable", "ineligible_wrapper_step")
    command = step.get("forward_progress_command")
    if not isinstance(command, Mapping):
        return _status_payload("unavailable", "missing_forward_progress_command")
    status = command.get("status")
    linear = _finite_metric_scalar(command.get("linear_velocity_m_s"))
    if status == "valid":
        if linear is None:
            return _status_payload("invalid", "nonfinite_forward_progress_command")
        if linear <= 0.0:
            return _status_payload("invalid", "invalid_forward_progress_command")
        return {"status": "available", "reason": None, "valid": True, "value": linear}
    if status == "not_forward_progress":
        if linear is None:
            return _status_payload("invalid", "nonfinite_forward_progress_command")
        return {"status": "available", "reason": None, "valid": False, "value": linear}
    if status == "unavailable":
        return _status_payload(
            "unavailable",
            str(command.get("reason") or "forward_progress_command_unavailable"),
        )
    return _status_payload("invalid", "invalid_forward_progress_status")


def _post_step_outcome_state(step: Mapping[str, Any]) -> dict[str, Any]:
    """Read the canonical post-step collision/near-miss state.

    Returns:
        Availability and canonical outcome flags for one trace step.
    """

    outcome = step.get("post_step_outcome")
    if not isinstance(outcome, Mapping):
        return _status_payload("unavailable", "missing_post_step_outcome")
    status = outcome.get("status")
    if status == "unavailable":
        return _status_payload(
            "unavailable",
            str(outcome.get("reason") or "post_step_outcome_unavailable"),
        )
    if status != "available":
        return _status_payload("invalid", "invalid_post_step_outcome_status")
    collision = outcome.get("collision")
    near_miss = outcome.get("near_miss")
    if not isinstance(collision, bool) or not isinstance(near_miss, bool):
        return _status_payload("invalid", "invalid_post_step_outcome_flags")
    return {
        "status": "available",
        "reason": None,
        "collision": collision,
        "near_miss": near_miss,
    }


def _evaluate_false_positive_stop_rate(  # noqa: C901
    wrapper_on_record: Mapping[str, Any],
    wrapper_off_record: Mapping[str, Any] | None,
    *,
    window_s: float,
) -> dict[str, Any]:
    """Evaluate false positives only from a paired off-arm complete window.

    Returns:
        Machine-readable field result with a finite rate only when the window is complete.
    """

    on_trace = _trace_view(wrapper_on_record)
    if on_trace["status"] != "available":
        return _status_payload(str(on_trace["status"]), str(on_trace["reason"]))
    if on_trace["arm_key"] != "wrapper_on":
        return _status_payload("unavailable", "wrapper_on_arm_required")
    events = _stop_yield_events(on_trace["trace"])
    if not events:
        return _status_payload("unavailable", "no_stop_yield_opportunity")
    pair = _validate_pair(wrapper_on_record, wrapper_off_record)
    if pair["status"] != "available":
        return pair
    if not math.isfinite(window_s) or window_s <= 0.0:
        return _status_payload("invalid", "invalid_counterfactual_window")
    dt_s = float(pair["time_per_step_s"])
    ratio = window_s / dt_s
    window_steps = round(ratio)
    if window_steps <= 0 or not math.isclose(ratio, window_steps, rel_tol=0.0, abs_tol=1.0e-9):
        return _status_payload("invalid", "counterfactual_window_not_step_aligned")

    off_trace = pair["wrapper_off_trace"]
    false_positive_count = 0
    windows: list[dict[str, Any]] = []
    for event in events:
        onset = int(event["onset_step"])
        window_end = onset + window_steps
        if int(off_trace["max_step"]) < window_end:
            return _status_payload(
                "unavailable",
                "counterfactual_window_truncated",
                event_onset_step=onset,
                window_end_step=window_end,
            )
        window_steps_payload: list[Mapping[str, Any]] = []
        for step in range(onset, window_end + 1):
            candidate = off_trace["by_step"].get(step)
            if candidate is None:
                return _status_payload(
                    "unavailable",
                    "counterfactual_window_missing_step",
                    event_onset_step=onset,
                    missing_step=step,
                )
            window_steps_payload.append(candidate)
        for candidate in window_steps_payload:
            outcome = _post_step_outcome_state(candidate)
            if outcome["status"] != "available":
                return _status_payload(
                    str(outcome["status"]),
                    str(outcome["reason"]),
                    event_onset_step=onset,
                    step=int(candidate["step"]),
                )
        final_command = _forward_progress_state(window_steps_payload[-1])
        if final_command["status"] != "available":
            return _status_payload(
                str(final_command["status"]),
                str(final_command["reason"]),
                event_onset_step=onset,
                step=window_end,
            )
        collision = any(
            bool(_post_step_outcome_state(candidate)["collision"])
            for candidate in window_steps_payload
        )
        near_miss = any(
            bool(_post_step_outcome_state(candidate)["near_miss"])
            for candidate in window_steps_payload
        )
        false_positive = not collision and not near_miss and bool(final_command["valid"])
        false_positive_count += int(false_positive)
        windows.append(
            {
                "event_onset_step": onset,
                "window_end_step": window_end,
                "collision": collision,
                "near_miss": near_miss,
                "final_forward_progress_valid": bool(final_command["valid"]),
                "false_positive": false_positive,
            }
        )
    rate = float(false_positive_count / len(events))
    return {
        "status": "available",
        "reason": None,
        "value": rate,
        "details": {
            "event_count": len(events),
            "false_positive_event_count": false_positive_count,
            "window_s": float(window_s),
            "window_steps": window_steps,
            "windows": windows,
        },
    }


def _evaluate_stop_yield_latency(  # noqa: C901
    wrapper_on_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate mean intervention-to-recovery latency on the on-arm trace.

    Returns:
        Machine-readable field result with mean event latency when all recoveries exist.
    """

    trace_view = _trace_view(wrapper_on_record)
    if trace_view["status"] != "available":
        return _status_payload(str(trace_view["status"]), str(trace_view["reason"]))
    if trace_view["arm_key"] != "wrapper_on":
        return _status_payload("unavailable", "wrapper_on_arm_required")
    events = _stop_yield_events(trace_view["trace"])
    if not events:
        return _status_payload("unavailable", "no_stop_yield_opportunity")
    dt_s = float(trace_view["dt_s"])
    latencies: list[float] = []
    for event in events:
        onset = int(event["onset_step"])
        end_step = int(event["end_step"])
        recovery_step: int | None = None
        for step in range(end_step + 1, int(trace_view["max_step"]) + 1):
            candidate = trace_view["by_step"].get(step)
            if candidate is None:
                return _status_payload("unavailable", "missing_trace_step", step=step)
            command = _forward_progress_state(candidate)
            if command["status"] == "invalid":
                return _status_payload(str(command["status"]), str(command["reason"]), step=step)
            if command["status"] == "unavailable":
                return _status_payload(str(command["status"]), str(command["reason"]), step=step)
            if bool(command["valid"]):
                recovery_step = step
                break
        if recovery_step is None:
            return _status_payload(
                "unavailable", "intervention_recovery_missing", event_onset_step=onset
            )
        latencies.append(float((recovery_step - onset) * dt_s))
    return {
        "status": "available",
        "reason": None,
        "value": float(sum(latencies) / len(latencies)),
        "details": {
            "event_count": len(events),
            "event_latencies_s": latencies,
            "aggregation": "mean_over_stop_yield_events",
        },
    }


def _simulation_trace_view(record: Mapping[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912
    """Validate the simulation trace used by timeout progress.

    Returns:
        Trace status, time step, and contiguous simulation frames.
    """

    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        return _status_payload("unavailable", "missing_algorithm_metadata")
    raw_trace = metadata.get("simulation_step_trace")
    if not isinstance(raw_trace, Mapping):
        return _status_payload("unavailable", "missing_simulation_step_trace")
    if raw_trace.get("schema_version") != "simulation-step-trace.v1":
        return _status_payload("invalid", "invalid_simulation_trace_schema")
    raw_steps = raw_trace.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)) or not raw_steps:
        return _status_payload("unavailable", "missing_simulation_step_trace_steps")
    native = _record_native_trace_metadata(record)
    declared_time_steps: list[float] = []
    if "dt" in raw_trace:
        raw_dt = _finite_metric_scalar(raw_trace.get("dt"))
        if raw_dt is None:
            return _status_payload("invalid", "nonfinite_simulation_trace_time_step")
        if raw_dt <= 0.0:
            return _status_payload("invalid", "invalid_simulation_trace_time_step")
        declared_time_steps.append(float(raw_dt))
    if isinstance(native, Mapping) and "dt_s" in native:
        native_dt = _finite_metric_scalar(native.get("dt_s"))
        if native_dt is None:
            return _status_payload("invalid", "nonfinite_simulation_trace_time_step")
        if native_dt <= 0.0:
            return _status_payload("invalid", "invalid_simulation_trace_time_step")
        declared_time_steps.append(float(native_dt))
    if not declared_time_steps:
        return _status_payload("unavailable", "missing_simulation_trace_time_step")
    dt_s = declared_time_steps[0]
    if any(
        not math.isclose(value, dt_s, rel_tol=0.0, abs_tol=1.0e-12)
        for value in declared_time_steps[1:]
    ):
        return _status_payload("invalid", "simulation_trace_time_step_mismatch")
    steps: list[dict[str, Any]] = []
    previous: int | None = None
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            return _status_payload("invalid", "invalid_simulation_trace_step")
        step = raw_step.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            return _status_payload("invalid", "invalid_simulation_trace_step_index")
        if previous is None and step != 0:
            return _status_payload("unavailable", "missing_simulation_trace_step", step=0)
        if previous is not None and step != previous + 1:
            return _status_payload(
                "unavailable", "missing_simulation_trace_step", step=previous + 1
            )
        time_s = _finite_metric_scalar(raw_step.get("time_s"))
        if time_s is None:
            return _status_payload("invalid", "nonfinite_simulation_trace_time", step=step)
        if not math.isclose(time_s, (step + 1) * dt_s, rel_tol=0.0, abs_tol=1.0e-9):
            return _status_payload("invalid", "simulation_trace_time_mismatch", step=step)
        steps.append(dict(raw_step))
        previous = step
    return {
        "status": "available",
        "reason": None,
        "dt_s": dt_s,
        "steps": steps,
        "max_step": int(steps[-1]["step"]),
    }


def _evaluate_progress_at_timeout(  # noqa: C901, PLR0912
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate timeout-only progress using the declared start-goal distance.

    Returns:
        Machine-readable field result with normalized timeout progress when valid.
    """

    native = _record_native_trace_metadata(record)
    if native is None:
        return _status_payload("unavailable", "missing_paired_effect_native_trace")
    if "declared_timeout" not in native:
        return _status_payload("unavailable", "missing_declared_timeout")
    declared_timeout = native.get("declared_timeout")
    outcome = record.get("outcome")
    outcome_timeout = outcome.get("timeout_event") if isinstance(outcome, Mapping) else None
    if declared_timeout is not None and not isinstance(declared_timeout, bool):
        return _status_payload("invalid", "invalid_declared_timeout")
    if outcome_timeout is not None and not isinstance(outcome_timeout, bool):
        return _status_payload("invalid", "invalid_timeout_outcome_flag")
    if declared_timeout is not None and outcome_timeout is not None:
        if declared_timeout != outcome_timeout:
            return _status_payload("invalid", "timeout_state_mismatch")
    timeout = declared_timeout
    if timeout is not True:
        return _status_payload("unavailable", "non_timeout_termination")
    native_termination_reason = native.get("termination_reason")
    record_termination_reason = record.get("termination_reason")
    if native_termination_reason is None and record_termination_reason is None:
        return _status_payload("unavailable", "missing_declared_termination_reason")
    if (
        native_termination_reason is not None
        and record_termination_reason is not None
        and native_termination_reason != record_termination_reason
    ):
        return _status_payload("invalid", "termination_reason_mismatch")
    termination_reason = (
        native_termination_reason
        if native_termination_reason is not None
        else record_termination_reason
    )
    if not isinstance(termination_reason, str) or not termination_reason.strip():
        return _status_payload("invalid", "invalid_timeout_termination_reason")
    if termination_reason not in _TIMEOUT_TERMINATION_REASONS:
        return _status_payload("invalid", "invalid_timeout_termination_reason")

    initial_distance = _finite_metric_scalar(native.get("initial_goal_distance_m"))
    if initial_distance is None:
        return _status_payload("invalid", "nonfinite_start_to_goal_denominator")
    if initial_distance <= 0.0:
        return _status_payload("invalid", "invalid_start_to_goal_denominator")
    goal_raw = native.get("goal_position")
    if (
        not isinstance(goal_raw, Sequence)
        or isinstance(goal_raw, (str, bytes))
        or len(goal_raw) < 2
    ):
        return _status_payload("unavailable", "missing_declared_goal_position")
    goal = [_finite_metric_scalar(goal_raw[0]), _finite_metric_scalar(goal_raw[1])]
    if goal[0] is None or goal[1] is None:
        return _status_payload("invalid", "nonfinite_declared_goal_position")
    simulation = _simulation_trace_view(record)
    if simulation["status"] != "available":
        return _status_payload(str(simulation["status"]), str(simulation["reason"]))
    if "horizon_steps" not in native:
        return _status_payload("unavailable", "missing_declared_horizon")
    horizon = native.get("horizon_steps")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        return _status_payload("invalid", "invalid_declared_horizon")
    if int(simulation["max_step"]) != horizon - 1:
        return _status_payload("unavailable", "timeout_trace_truncated")
    final_robot = simulation["steps"][-1].get("robot")
    final_position = final_robot.get("position") if isinstance(final_robot, Mapping) else None
    if (
        not isinstance(final_position, Sequence)
        or isinstance(final_position, (str, bytes))
        or len(final_position) < 2
    ):
        return _status_payload("unavailable", "missing_timeout_robot_position")
    final_x = _finite_metric_scalar(final_position[0])
    final_y = _finite_metric_scalar(final_position[1])
    if final_x is None or final_y is None:
        return _status_payload("invalid", "nonfinite_timeout_robot_position")
    final_distance = math.hypot(final_x - goal[0], final_y - goal[1])
    progress = (initial_distance - final_distance) / initial_distance
    if not math.isfinite(progress):
        return _status_payload("invalid", "nonfinite_timeout_progress")
    if progress < 0.0 or progress > 1.0:
        return _status_payload("invalid", "timeout_progress_out_of_bounds", value=progress)
    return {
        "status": "available",
        "reason": None,
        "value": float(progress),
        "details": {
            "initial_goal_distance_m": float(initial_distance),
            "final_goal_distance_m": float(final_distance),
            "horizon_steps": int(horizon),
            "termination_reason": str(termination_reason),
        },
    }


def evaluate_paired_effect_metric_fields(
    record: Mapping[str, Any],
    *,
    paired_wrapper_off_record: Mapping[str, Any] | None = None,
    window_s: float = 2.0,
) -> dict[str, Any]:
    """Evaluate the three #8567 fields from native episode records.

    This function is deliberately pair-aware but does not mutate either record.  A
    caller may pass the validated wrapper-off counterpart after both native arms
    have been acquired.  Missing counterpart data leaves only the causal false-stop
    field unavailable; invalid or truncated evidence is never repaired or imputed.

    Returns:
        Producer status, field-level outcomes, provenance, and valid metric values.
    """

    if not isinstance(record, Mapping):
        return {
            "schema_version": PRODUCER_SCHEMA_VERSION,
            "status": "invalid",
            "reason": "native_record_not_mapping",
            "metric_values": {},
            "fields": {},
        }
    clarification = _load_default_paired_effect_metric_clarification()
    configured_window_s = float(clarification["counterfactual_window_s"])
    window_value = _finite_metric_scalar(window_s)
    if window_value is None or not math.isclose(
        window_value, configured_window_s, rel_tol=0.0, abs_tol=1.0e-12
    ):
        window_value = None
    fields = {
        "false_positive_stop_rate": _evaluate_false_positive_stop_rate(
            record,
            paired_wrapper_off_record,
            window_s=float("nan") if window_value is None else window_value,
        ),
        "stop_yield_latency_s": _evaluate_stop_yield_latency(record),
        "progress_at_timeout": _evaluate_progress_at_timeout(record),
    }
    metric_values: dict[str, float] = {}
    for name, result in fields.items():
        if (
            result.get("status") == "available"
            and _finite_metric_scalar(result.get("value")) is not None
        ):
            metric_values[name] = float(result["value"])
    statuses = {name: dict(result) for name, result in fields.items()}
    field_states = {str(result.get("status")) for result in fields.values()}
    if field_states == {"available"}:
        status = "available"
        reason = None
    elif "invalid" in field_states:
        status = "invalid"
        reason = "one_or_more_fields_invalid"
    else:
        status = "unavailable"
        reason = "one_or_more_fields_unavailable"
    identity, identity_reason = _record_pair_identity(record)
    native = _record_native_trace_metadata(record)
    return {
        "schema_version": PRODUCER_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "pairing": {
            "key_fields": list(PAIRING_KEY_FIELDS),
            "key": identity,
            "identity_status": "available" if identity is not None else "unavailable",
            "identity_reason": identity_reason,
            "counterpart_present": paired_wrapper_off_record is not None,
        },
        "source": {
            "clarification_schema_version": CLARIFICATION_SCHEMA_VERSION,
            "source_commit": _record_source_commit(record),
            "pair_config_hash": _record_pair_config_hash(record),
            "native_trace_schema_version": (
                native.get("schema_version") if isinstance(native, Mapping) else None
            ),
        },
        "fields": statuses,
        "metric_values": metric_values,
    }


def validate_paired_effect_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    include_row_reports: bool = False,
) -> dict[str, Any]:
    """Validate all retained episode rows and return a JSON-safe gate report.

    Returns:
        Aggregate validation report.
    """

    validated = validate_paired_effect_metric_contract(contract)
    missing_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    row_reports: list[dict[str, Any]] = []
    invalid_row_samples: list[dict[str, Any]] = []
    valid_row_count = 0
    for index, row in enumerate(rows):
        report = validate_paired_effect_metric_record(row, validated, row_index=index)
        missing_counts.update(report["missing_fields"])
        invalid_counts.update(item["field"] for item in report["invalid_fields"])
        if report["status"] == "ok":
            valid_row_count += 1
        elif len(invalid_row_samples) < MAX_INVALID_ROW_SAMPLES:
            invalid_row_samples.append(report)
        if include_row_reports:
            row_reports.append(report)
    complete = bool(rows) and valid_row_count == len(rows)
    result: dict[str, Any] = {
        "schema_version": "paired_effect_metric_validation.v1",
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "ok" if complete else "blocked",
        "complete": complete,
        "row_count": len(rows),
        "valid_row_count": valid_row_count,
        "required_metric_names": list(REQUIRED_METRIC_NAMES),
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "invalid_field_counts": dict(sorted(invalid_counts.items())),
        "invalid_row_samples": invalid_row_samples,
        "claim_boundary": (
            "Instrumentation contract gate only. A passing retained-row check does not make a "
            "campaign result benchmark or paper evidence; it only establishes that the declared "
            "paired metrics were retained without alias substitution."
        ),
    }
    if include_row_reports:
        result["row_reports"] = row_reports
    return result


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON-list or JSONL episode rows for the validation CLI.

    Returns:
        Parsed object rows.
    """

    row_path = Path(path)
    if row_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            row_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise PairedEffectMetricContractError(
                    f"{row_path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
        return rows
    value = json.loads(row_path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise PairedEffectMetricContractError(
            f"{row_path} must contain a JSON list or JSONL object rows"
        )
    return value


def enforce_paired_effect_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    include_row_reports: bool = False,
) -> dict[str, Any]:
    """Raise on a retained-row mismatch and return the successful gate report.

    Returns:
        Successful aggregate validation report.
    """

    report = validate_paired_effect_metric_rows(
        rows,
        contract,
        include_row_reports=include_row_reports,
    )
    if not report["complete"]:
        raise PairedEffectMetricContractError(
            f"paired-effect retained-row contract failed: {json.dumps(report, sort_keys=True)}"
        )
    return report


__all__ = [
    "CLARIFICATION_METRIC_NAMES",
    "CLARIFICATION_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "PAIRING_KEY_FIELDS",
    "PRODUCER_SCHEMA_VERSION",
    "REQUIRED_METRIC_NAMES",
    "WRAPPER_ARM_KEYS",
    "PairedEffectMetricContractError",
    "enforce_paired_effect_metric_rows",
    "evaluate_paired_effect_metric_fields",
    "load_json_rows",
    "load_paired_effect_metric_clarification",
    "load_paired_effect_metric_contract",
    "validate_paired_effect_metric_clarification",
    "validate_paired_effect_metric_contract",
    "validate_paired_effect_metric_record",
    "validate_paired_effect_metric_rows",
]
