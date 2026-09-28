#!/usr/bin/env python3
"""Validate, render, and account for the frozen #8872 campaign.

This is a public admission and row-accounting boundary with one fixed native
map-runner execution path, not a scheduler launcher.  The production identities
are always compiled by ``check_issue_6561_pedestrian_speed_protocol.compile_manifest``.
A production packet is accepted only when the current preserved #8871 activation
receipt, the verified #6102 integrity receipt, native planner/checkpoint preflight,
all six private admission predicates, and the authorization receipt are bound to
the same packet.  ``run-production`` is intentionally disabled: it fails closed
before reading a token, opening execution paths, or invoking a runner until
trusted private-ops authenticated-signer verification is configured.  It never
imports an arbitrary executor or submits Slurm work.

The command modes are intentionally separate:

``validate``
    Compile and inspect the frozen protocol, or validate an existing packet.
``smoke``
    Build three disjoint canary identities for diagnostic plumbing only.  It
    never touches a registered identity and never runs an episode.
``render-production``
    Validate every admission receipt and write one immutable packet.  It does
    not execute rows or submit work.
``run-production``
    Always fail closed after packet validation.  The fixed native runner,
    execution token, and execution paths remain unreachable until trusted
    private-ops authenticated-signer verification is configured.

The private operations layer is responsible for translating its host-specific
checks into the normalized receipt shapes documented in the context note and
for providing the future authenticated authorization.  Preparation status remains
``PREPARATION_INCOMPLETE`` while the scientific #8871 activation gate is not
an ``activation_pass``; the fixed public executor does not override that gate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from scripts.validation.check_issue_6561_pedestrian_speed_protocol import (
    DEFAULT_CONFIG as DEFAULT_PROTOCOL_CONFIG,
)
from scripts.validation.check_issue_6561_pedestrian_speed_protocol import (
    EXPECTED_PLANNERS,
    EXPECTED_PROTOCOL_SEMANTIC_HASH,
    compile_manifest,
    load_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANARY_CONFIG = REPO_ROOT / "configs/benchmarks/issue_8871_pedestrian_speed_canary_v1.yaml"
PRODUCTION_MANIFEST_HASH = "371f1a0160ec7faf1ade531691f104e2a1c92f7c34857e887ba1ba539e1b5238"
ROBOT_SPEED_MANIFEST_HASH = "e32ce197149af62bf366f5ca95abbb42215b379fe7916d916ccdd544dce8666f"
PACKET_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_campaign_packet.v1"
ROW_ACCOUNTING_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_row_accounting.v1"
SMOKE_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_smoke.v1"
ACTIVATION_RECEIPT_SCHEMA_VERSION = "robot_sf.issue_8871_activation_receipt.v1"
SPEED_INTEGRITY_RECEIPT_SCHEMA_VERSION = "robot_sf.issue_6102_integrity_receipt.v1"
NATIVE_PREFLIGHT_RECEIPT_SCHEMA_VERSION = "robot_sf.issue_8872_native_preflight_receipt.v1"
PRIVATE_ADMISSION_RECEIPT_SCHEMA_VERSION = "robot_sf.issue_8872_private_admission_receipt.v1"
AUTHORIZATION_RECEIPT_SCHEMA_VERSION = "robot_sf.issue_8872_production_authorization.v1"
EXPECTED_ROWS = 2160
PREPARATION_STATUS = "PREPARATION_INCOMPLETE"
EXPECTED_PROTOCOL_CONFIG = "configs/benchmarks/issue_6561_pedestrian_speed_protocol.yaml"
REQUIRED_PRIVATE_PREDICATES = (
    "production_wrapper_rehearsal",
    "duplicate_guard",
    "queue_admission",
    "route_capacity",
    "storage_reservation",
    "preservation_capacity",
)
SUCCESS_STATUS = "native_complete"
TERMINAL_STATUSES = frozenset(
    {
        SUCCESS_STATUS,
        "failed",
        "missing",
        "unavailable",
        "fallback",
        "degraded",
        "non_native",
        "provenance_invalid",
        "intervention_not_activated",
        "duplicate",
    }
)
FORBIDDEN_SUCCESS_STATUSES = frozenset(
    {
        "fallback",
        "degraded",
        "non_native",
        "provenance_invalid",
        "intervention_not_activated",
    }
)
FORBIDDEN_SECRET_KEYS = frozenset(
    {
        "access_token",
        "credential",
        "private_key",
        "raw_token",
        "secret",
        "token",
    }
)
FORBIDDEN_TRANSIENT_KEYS = (
    frozenset(
        {
            "host",
            "hostname",
            "job_id",
            "queue",
            "scheduler",
            "scratch_path",
            "target_host",
            "worktree",
        }
    )
    | FORBIDDEN_SECRET_KEYS
)
SAFE_ARTIFACT_REFERENCE = re.compile(r"^(?:artifact|wandb)://[A-Za-z0-9._/-]+(?::v[0-9]+)?$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
JOURNAL_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_execution_journal.v1"
RECEIPT_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_execution_receipt.v1"
JOURNAL_HEADER_FIELDS = frozenset(
    {
        "event",
        "schema_version",
        "issue",
        "packet_sha256",
        "packet_binding_hash",
        "source_commit",
        "manifest_hash",
        "expected_rows",
    }
)
PRODUCTION_TOKEN_ENV = "ROBOT_SF_8872_EXECUTION_TOKEN"
ROBOT_SPEED_CAP_M_S = 2.0
PRODUCTION_EXECUTION_DISABLED_REASON = (
    "run-production is disabled until private-ops authenticated authorization verification "
    "is configured"
)
AUTHORIZATION_VERIFICATION_CONTRACT = "private_ops_authenticated_signer_required"
PROTOCOL_METRIC_SECTIONS = (
    "primary_metrics",
    "exposure_metrics",
    "typed_collision_metrics",
)
AUTHORIZATION_KEYS = frozenset(
    {
        "decision_id",
        "issuer",
        "issue",
        "packet_binding_hash",
        "schema_version",
        "scope",
        "source_commit",
        "token_binding_sha256",
        "token_sha256",
    }
)
PUBLIC_RECEIPT_SCHEMAS = {
    "activation_receipt": {
        "schema_version": None,
        "issue": None,
        "verdict": None,
        "preserved": None,
        "current": None,
        "registered_rows_executed": None,
        "seed_disjoint": None,
        "protocol_semantic_hash": None,
        "production_manifest_hash": None,
        "source_commit": None,
        "packet_binding_hash": None,
        "preservation": {
            "status": None,
            "artifact_reference": None,
            "receipt_sha256": None,
        },
    },
    "speed_integrity_receipt": {
        "schema_version": None,
        "issue": None,
        "status": None,
        "preserved": None,
        "manifest_hash": None,
        "expected_rows": None,
        "native_rows": None,
        "excluded_rows": None,
        "fallback_rows": None,
        "degraded_rows": None,
        "missing_rows": None,
        "duplicate_rows": None,
        "provenance_invalid_rows": None,
        "execution_mode": None,
        "artifact_reference": None,
        "artifact_digest": None,
        "source_commit": None,
        "packet_binding_hash": None,
    },
    "native_preflight": {
        "schema_version": None,
        "issue": None,
        "status": None,
        "native": None,
        "fallback": None,
        "degraded": None,
        "planner_ids": None,
        "checkpoint_provenance_complete": None,
        "protocol_semantic_hash": None,
        "manifest_hash": None,
        "source_commit": None,
        "packet_binding_hash": None,
    },
    "private_admission": {
        "schema_version": None,
        "issue": None,
        "wrapper_status": None,
        "predicates": dict.fromkeys(REQUIRED_PRIVATE_PREDICATES),
        "private_details_excluded": None,
        "source_commit": None,
        "packet_binding_hash": None,
    },
    "production_authorization": dict.fromkeys(AUTHORIZATION_KEYS),
}
SAFE_JOURNAL_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_IDENTITY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
SAFE_REASON_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SAFE_DECISION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
JOURNAL_PROVENANCE_FIELDS = frozenset(
    {
        "provenance_status",
        "identity_key",
        "scenario_id",
        "scenario_source_sha256",
        "regime_id",
        "planner_id",
        "planner_algorithm",
        "planner_config_sha256",
        "seed",
        "horizon_steps",
        "dt_seconds",
        "robot_speed_cap_m_s",
        "runtime_controls",
        "protocol_semantic_hash",
        "manifest_hash",
        "source_commit",
        "execution_mode",
        "native",
        "fallback",
        "degraded",
        "intervention_status",
        "diagnostics",
        "trace_sha256",
        "checkpoint_provenance",
        "metrics",
        "terminal_status",
    }
)
JOURNAL_SUCCESS_PROVENANCE_FIELDS = frozenset(
    {
        "identity_key",
        "scenario_id",
        "scenario_source_sha256",
        "regime_id",
        "planner_id",
        "planner_algorithm",
        "planner_config_sha256",
        "seed",
        "horizon_steps",
        "dt_seconds",
        "robot_speed_cap_m_s",
        "runtime_controls",
        "protocol_semantic_hash",
        "manifest_hash",
        "source_commit",
        "execution_mode",
        "native",
        "fallback",
        "degraded",
        "intervention_status",
        "diagnostics",
        "trace_sha256",
        "checkpoint_provenance",
        "metrics",
    }
)
JOURNAL_DIAGNOSTIC_SCALARS = frozenset(
    {
        "configured_desired_speed_mean_m_s",
        "configured_desired_speed_std_m_s",
        "realized_desired_speed_mean_m_s",
        "realized_desired_speed_std_m_s",
        "initial_spawn_speed_mean_m_s",
        "initial_spawn_speed_peak_m_s",
        "time_to_desired_speed_target_seconds",
        "acceleration_transient_steps",
        "desired_speed_activation_fraction",
    }
)
JOURNAL_DIAGNOSTIC_VECTOR_MAPS = frozenset(
    {
        "initial_spawn_velocity_xy_by_pedestrian",
        "final_post_integration_velocity_xy_by_pedestrian",
    }
)
JOURNAL_DIAGNOSTIC_SPEED_MAP = "runtime_max_speed_m_s_by_pedestrian"
JOURNAL_CHECKPOINT_FIELDS = frozenset(
    {"model_id", "path_label", "sha256", "size_bytes", "expected_sha256"}
)
SENSITIVE_REASON_TOKENS = frozenset(
    {
        "access_token",
        "cluster",
        "credential",
        "host",
        "hostname",
        "path",
        "private",
        "scratch",
        "secret",
        "ssh",
        "token",
        "worktree",
    }
)


class CampaignAdapterError(ValueError):
    """Raised when a packet, receipt, or row outcome is unsafe to admit."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignAdapterError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be a mapping")
    return dict(value)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_digest(value: Any, field: str, length: int = 64) -> str:
    _require(
        _is_digest(value, length),
        f"{field} must be a lowercase {length}-character digest",
    )
    return str(value)


def _require_nonempty_string(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{field} must be a non-empty string",
    )
    return value


def _sanitize_reason(value: Any, *, fallback: str = "reason:unspecified") -> str:
    """Keep terminal diagnostics stable without copying arbitrary exception text."""
    if not isinstance(value, str) or not value.strip():
        return fallback
    raw = value.strip()
    reason_tokens = set(re.findall(r"[A-Za-z0-9_]+", raw.lower()))
    if (
        "\x00" in raw
        or "/" in raw
        or "\\" in raw
        or ".." in raw
        or raw.startswith(("~", "file:", "ssh:"))
        or reason_tokens.intersection(SENSITIVE_REASON_TOKENS)
    ):
        return "reason:unsafe_detail_redacted"
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_")
    if not token or not SAFE_REASON_TOKEN.fullmatch(token):
        return "reason:unstructured_detail_redacted"
    return f"reason:{token}"


def _exception_reason(exc: BaseException) -> str:
    """Return exception class only; never expose its message or traceback details."""
    class_name = type(exc).__qualname__
    if not SAFE_REASON_TOKEN.fullmatch(class_name):
        class_name = "UnknownException"
    return f"exception:{class_name}"


@lru_cache(maxsize=1)
def _protocol_metric_names() -> tuple[str, ...]:
    """Resolve the exact metric names declared by the frozen protocol."""
    contract = _mapping(
        load_protocol(DEFAULT_PROTOCOL_CONFIG).get("metric_contract"), "metric_contract"
    )
    names: list[str] = []
    for section in PROTOCOL_METRIC_SECTIONS:
        values = contract.get(section)
        _require(isinstance(values, list), f"metric_contract.{section} must be a list")
        _require(
            all(isinstance(name, str) and bool(name.strip()) for name in values),
            f"metric_contract.{section} contains an invalid name",
        )
        names.extend(str(name) for name in values)
    _require(len(names) == len(set(names)), "metric_contract contains duplicate metric names")
    return tuple(names)


def _finite_metric(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be a finite numeric value",
    )
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CampaignAdapterError(f"{field} must be a finite numeric value") from exc
    _require(math.isfinite(number), f"{field} must be a finite numeric value")
    return number


def _finite_protocol_metrics(value: Any, *, field: str = "metrics") -> dict[str, float]:
    """Project a mapping onto protocol metrics, retaining only finite values."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for name in _protocol_metric_names():
        if name not in value:
            continue
        try:
            result[name] = _finite_metric(value[name], f"{field}.{name}")
        except CampaignAdapterError:
            continue
    return result


def _validated_protocol_metrics(
    value: Any, *, field: str = "metrics", require_complete: bool
) -> dict[str, float]:
    """Validate one complete protocol metric mapping without accepting extras."""
    _require(isinstance(value, Mapping), f"{field} must be a mapping")
    names = _protocol_metric_names()
    _require(
        set(value).issubset(names),
        f"{field} contains an unsupported metric",
    )
    result = {
        str(name): _finite_metric(metric, f"{field}.{name}") for name, metric in value.items()
    }
    _require(
        set(result) == set(names) if require_complete else True,
        f"{field} must contain every frozen protocol metric",
    )
    return result


def _safe_journal_reference(path: str | Path) -> str:
    """Expose only a safe basename, never a private host path, in public output."""
    reference = Path(path).name
    _require(
        bool(SAFE_JOURNAL_REFERENCE.fullmatch(reference)) and ".." not in reference,
        "journal reference is not safe",
    )
    return reference


def _validate_artifact_reference(value: Any, path: str) -> None:
    _require(isinstance(value, str), f"{path} must be a string reference")
    _require(
        bool(SAFE_ARTIFACT_REFERENCE.fullmatch(value)) and ".." not in value,
        f"{path} is not a safe durable artifact reference",
    )


def _assert_safe_public_value(value: Any, path: str = "receipt", key: str = "") -> None:
    """Reject private host/path material before it can enter a public packet."""
    if isinstance(value, str):
        _require("\x00" not in value, f"{path} contains a NUL byte")
        lower_key = key.lower()
        if lower_key in {"artifact_reference", "artifact_ref"}:
            _validate_artifact_reference(value, path)
            return
        _require(
            not value.startswith(("/", "~", "file:", "ssh:")),
            f"{path} is absolute/private",
        )
        _require("\\" not in value, f"{path} contains an unsafe path separator")
        if "path" in lower_key:
            path_value = Path(value)
            _require(
                not path_value.is_absolute() and ".." not in path_value.parts,
                f"{path} leaves repository",
            )
        return
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            child_name = str(child_key)
            lower_child_name = child_name.lower()
            secret_key = lower_child_name in FORBIDDEN_SECRET_KEYS or (
                "token" in lower_child_name and not lower_child_name.endswith("_sha256")
            )
            _require(
                lower_child_name not in FORBIDDEN_TRANSIENT_KEYS and not secret_key,
                (
                    f"{path}.{child_name} contains secret material"
                    if secret_key
                    else f"{path}.{child_name} contains private transient state"
                ),
            )
            _assert_safe_public_value(child, f"{path}.{child_name}", child_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_safe_public_value(child, f"{path}[{index}]")


def _copy_public_leaf(value: Any, path: str) -> Any:
    """Copy JSON-shaped receipt leaves without permitting hidden mappings."""
    if isinstance(value, Mapping):
        raise CampaignAdapterError(f"{path} contains an unsupported nested mapping")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_public_leaf(item, f"{path}[{index}]") for index, item in enumerate(value)]
    _require(
        value is None or isinstance(value, (bool, int, float, str)),
        f"{path} contains an unsupported value",
    )
    return value


def _project_public_mapping(value: Any, *, path: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project one receipt mapping through an explicit nested public schema."""
    mapping = _mapping(value, path)
    _require(
        set(mapping).issubset(schema),
        f"{path} contains an unsupported public field",
    )
    projected: dict[str, Any] = {}
    for key, item in mapping.items():
        child_schema = schema[key]
        child_path = f"{path}.{key}"
        if child_schema is None:
            projected[key] = _copy_public_leaf(item, child_path)
        else:
            projected[key] = _project_public_mapping(
                item,
                path=child_path,
                schema=child_schema,
            )
    return projected


def _project_public_receipt(value: Any, field: str) -> dict[str, Any]:
    """Project a private-ops receipt before it is copied into a public packet."""
    schema = PUBLIC_RECEIPT_SCHEMAS.get(field)
    _require(schema is not None, f"receipt field {field} has no public schema")
    projected = _project_public_mapping(value, path=field, schema=schema)
    _assert_no_transient_state(projected, field)
    return projected


def _assert_no_transient_state(value: Any, path: str = "receipt") -> None:
    """Reject scheduler details and unsafe paths from a public receipt."""
    _assert_safe_public_value(value, path)


def _git_clean() -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CampaignAdapterError(
            f"cannot inspect source checkout: {exc.__class__.__name__}"
        ) from exc
    return not result.stdout.strip()


def _validate_source_checkout(source_commit: str) -> None:
    """Require the packet SHA to resolve to the clean checkout used for execution."""
    _require(HEX_COMMIT.fullmatch(source_commit) is not None, "source commit is not a SHA-1")
    try:
        resolved = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "--verify",
                f"{source_commit}^{{commit}}",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CampaignAdapterError("source commit does not resolve in the checkout") from exc
    _require(resolved == source_commit, "source commit does not resolve exactly")
    _require(_git_head() == source_commit, "source commit differs from checked-out HEAD")
    _require(_git_clean(), "source checkout is not clean; refuse execution")


def _token_binding_digest(token: str, packet_binding_hash: str) -> str:
    _require(isinstance(token, str) and len(token) >= 32, "execution token is too short")
    return hashlib.sha256(f"{token}:{packet_binding_hash}".encode()).hexdigest()


def _validate_token_binding(
    authorization: Mapping[str, Any], packet_binding_hash: str, token: str
) -> None:
    """Check a private-ops token without storing or printing its value."""
    _require(
        authorization.get("token_sha256") == hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "execution token digest does not match private authorization",
    )
    _require(
        authorization.get("token_binding_sha256")
        == _token_binding_digest(token, packet_binding_hash),
        "execution token is not bound to this exact packet",
    )


def _git_head() -> str:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CampaignAdapterError(
            f"cannot determine source commit: {exc.__class__.__name__}"
        ) from exc
    _require_digest(head, "source_commit", length=40)
    return head


def _load_json(path: str | Path, field: str) -> dict[str, Any]:
    file_path = Path(path)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignAdapterError(f"cannot read {field}: {exc.__class__.__name__}") from exc
    return _mapping(value, field)


def _compiled_manifest(
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    protocol = load_protocol(config_path)
    manifest = compile_manifest(protocol)
    _require(
        manifest["expected_cell_count"] == EXPECTED_ROWS,
        "compiled identity count drifted",
    )
    _require(
        manifest["identity_count"] == EXPECTED_ROWS,
        "compiled identity count is not 2160",
    )
    _require(
        manifest["unique_identity_count"] == EXPECTED_ROWS,
        "compiled identity uniqueness is not 2160",
    )
    _require(
        manifest["manifest_hash"] == PRODUCTION_MANIFEST_HASH,
        "compiled production manifest hash drifted",
    )
    return manifest


@lru_cache(maxsize=1)
def _compiled_identity_rows() -> dict[str, dict[str, Any]]:
    """Return the compiled production identity contract keyed by identity."""
    return {
        str(identity["identity_key"]): dict(identity)
        for identity in _compiled_manifest()["identities"]
    }


@lru_cache(maxsize=1)
def _compiled_identity_keys() -> frozenset[str]:
    """Return the only identity keys a production journal may expose."""
    return frozenset(_compiled_identity_rows())


def _journal_identity_key(value: Any) -> str:
    """Validate a journal identity before copying it into a public summary."""
    _require(
        isinstance(value, str)
        and bool(SAFE_IDENTITY_KEY.fullmatch(value))
        and ".." not in value
        and value in _compiled_identity_keys(),
        "journal row identity is invalid",
    )
    return value


def _journal_safe_token(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and bool(SAFE_IDENTITY_KEY.fullmatch(value)) and ".." not in value,
        f"journal provenance {field} is invalid",
    )
    return value


def _journal_relative_path(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "journal checkpoint path is invalid")
    path = Path(value)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
        and all(bool(SAFE_JOURNAL_REFERENCE.fullmatch(part)) for part in path.parts),
        "journal checkpoint path is invalid",
    )
    return value


def _journal_diagnostic_map(value: Any, *, vectors: bool) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "journal diagnostic map is invalid")
    result: dict[str, Any] = {}
    for key, item in value.items():
        safe_key = _journal_safe_token(key, "diagnostic pedestrian key")
        if vectors:
            _require(
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
                and len(item) == 2,
                "journal diagnostic vector is invalid",
            )
            result[safe_key] = [
                _finite_metric(component, "journal diagnostic vector component")
                for component in item
            ]
        else:
            result[safe_key] = _finite_metric(item, "journal diagnostic speed")
    return result


def _journal_diagnostics(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "journal diagnostics are invalid")
    allowed = (
        JOURNAL_DIAGNOSTIC_SCALARS | JOURNAL_DIAGNOSTIC_VECTOR_MAPS | {JOURNAL_DIAGNOSTIC_SPEED_MAP}
    )
    _require(set(value).issubset(allowed), "journal diagnostics contain an unsupported field")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in JOURNAL_DIAGNOSTIC_SCALARS:
            _require(
                item is None or (isinstance(item, (int, float)) and not isinstance(item, bool)),
                "journal diagnostic scalar is invalid",
            )
            result[key] = None if item is None else _finite_metric(item, "journal diagnostic")
        elif key == JOURNAL_DIAGNOSTIC_SPEED_MAP:
            result[key] = _journal_diagnostic_map(item, vectors=False)
        else:
            result[key] = _journal_diagnostic_map(item, vectors=True)
    return result


def _journal_checkpoint_provenance(value: Any) -> list[dict[str, Any]]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)),
        "journal checkpoint provenance is invalid",
    )
    result: list[dict[str, Any]] = []
    for item in value:
        _require(isinstance(item, Mapping), "journal checkpoint provenance item is invalid")
        _require(
            set(item).issubset(JOURNAL_CHECKPOINT_FIELDS),
            "journal checkpoint provenance contains an unsupported field",
        )
        normalized: dict[str, Any] = {}
        if "model_id" in item:
            normalized["model_id"] = _journal_safe_token(item["model_id"], "model id")
        if "path_label" in item:
            normalized["path_label"] = _journal_relative_path(item["path_label"])
        for field in ("sha256", "expected_sha256"):
            if field in item:
                _require_digest(item[field], f"journal checkpoint {field}")
                normalized[field] = item[field]
        if "size_bytes" in item:
            _require(
                isinstance(item["size_bytes"], int)
                and not isinstance(item["size_bytes"], bool)
                and item["size_bytes"] >= 0,
                "journal checkpoint size is invalid",
            )
            normalized["size_bytes"] = item["size_bytes"]
        result.append(normalized)
    return result


def _journal_provenance(  # noqa: C901, PLR0912
    value: Any, identity_key: str
) -> dict[str, Any]:
    """Validate and project protocol provenance before it enters journal output."""
    provenance = _mapping(value, "journal.row_finished.provenance")
    _require(
        set(provenance).issubset(JOURNAL_PROVENANCE_FIELDS),
        "journal provenance contains an unsupported field",
    )
    if "identity_key" in provenance:
        nested_identity = _journal_identity_key(provenance["identity_key"])
        _require(
            nested_identity == identity_key,
            "journal provenance identity does not match row identity",
        )
    expected = _compiled_identity_rows()[identity_key]
    normalized = dict(provenance)
    for field in (
        "identity_key",
        "scenario_id",
        "scenario_source_sha256",
        "regime_id",
        "planner_id",
        "planner_config_sha256",
        "seed",
        "horizon_steps",
        "dt_seconds",
        "robot_speed_cap_m_s",
        "runtime_controls",
    ):
        if field in normalized:
            _require(
                normalized[field] == expected[field],
                "journal provenance identity contract is invalid",
            )
    for field in ("scenario_source_sha256", "planner_config_sha256"):
        if field in normalized:
            _require_digest(normalized[field], f"journal provenance {field}")
    for field in ("protocol_semantic_hash", "manifest_hash", "trace_sha256"):
        if field in normalized:
            _require_digest(normalized[field], f"journal provenance {field}")
    if "source_commit" in normalized:
        _require(
            isinstance(normalized["source_commit"], str)
            and HEX_COMMIT.fullmatch(normalized["source_commit"]) is not None,
            "journal provenance source commit is invalid",
        )
    for field in ("planner_algorithm",):
        if field in normalized:
            normalized[field] = _journal_safe_token(normalized[field], field)
    if "provenance_status" in normalized:
        _require(
            normalized["provenance_status"] == "expected_identity_only",
            "journal provenance status is invalid",
        )
    if "execution_mode" in normalized:
        _require(
            normalized["execution_mode"] == expected["execution_mode"],
            "journal provenance mode is invalid",
        )
    if "terminal_status" in normalized:
        _require(
            isinstance(normalized["terminal_status"], str)
            and normalized["terminal_status"] in TERMINAL_STATUSES,
            "journal provenance terminal status is invalid",
        )
    if "intervention_status" in normalized:
        _require(
            normalized["intervention_status"] in {"not_applicable", "activated", "not_activated"},
            "journal provenance intervention status is invalid",
        )
    for field in ("native", "fallback", "degraded"):
        if field in normalized:
            _require(isinstance(normalized[field], bool), "journal provenance flag is invalid")
    if "diagnostics" in normalized:
        normalized["diagnostics"] = _journal_diagnostics(normalized["diagnostics"])
    if "checkpoint_provenance" in normalized:
        normalized["checkpoint_provenance"] = _journal_checkpoint_provenance(
            normalized["checkpoint_provenance"]
        )
    if "metrics" in normalized:
        normalized["metrics"] = _validated_protocol_metrics(
            normalized["metrics"],
            field="journal.row_finished.provenance.metrics",
            require_complete=False,
        )
    _assert_no_transient_state(normalized, "journal.row_finished.provenance")
    return normalized


def _validate_journal_success_provenance(
    provenance: Mapping[str, Any],
    identity_key: str,
    *,
    packet_context: Mapping[str, Any] | None = None,
) -> None:
    """Require a successful row to carry complete packet-bound identity evidence."""
    _require(
        JOURNAL_SUCCESS_PROVENANCE_FIELDS.issubset(provenance),
        "journal native terminal row provenance is incomplete",
    )
    _require(
        set(provenance) == JOURNAL_SUCCESS_PROVENANCE_FIELDS,
        "journal native terminal row provenance contains contradictory semantic fields",
    )
    expected = _compiled_identity_rows()[identity_key]
    for field in (
        "identity_key",
        "scenario_id",
        "scenario_source_sha256",
        "regime_id",
        "planner_id",
        "planner_config_sha256",
        "seed",
        "horizon_steps",
        "dt_seconds",
        "robot_speed_cap_m_s",
        "runtime_controls",
    ):
        _require(
            provenance.get(field) == expected[field],
            "journal native terminal row identity does not match the packet",
        )
    _require(
        provenance.get("protocol_semantic_hash") == EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "journal native terminal row protocol hash does not match the packet",
    )
    _require(
        provenance.get("manifest_hash") == PRODUCTION_MANIFEST_HASH,
        "journal native terminal row manifest does not match the packet",
    )
    _require(
        provenance.get("source_commit") == packet_context.get("source_commit")
        if packet_context is not None
        else isinstance(provenance.get("source_commit"), str),
        "journal native terminal row source does not match the packet",
    )
    if packet_context is not None:
        _require(
            provenance.get("manifest_hash") == packet_context.get("manifest_hash")
            and provenance.get("protocol_semantic_hash")
            == packet_context.get("protocol_semantic_hash"),
            "journal native terminal row packet context does not match",
        )
    _require(
        provenance.get("execution_mode") == expected["execution_mode"]
        and provenance.get("native") is True
        and provenance.get("fallback") is False
        and provenance.get("degraded") is False,
        "journal native terminal row execution mode is invalid",
    )
    expected_intervention_status = (
        "not_applicable" if expected["regime_id"] == "legacy_default" else "activated"
    )
    _require(
        provenance.get("intervention_status") == expected_intervention_status,
        "journal native terminal row activation status is invalid",
    )
    diagnostics = provenance.get("diagnostics")
    _require(
        isinstance(diagnostics, Mapping) and bool(diagnostics),
        "journal native terminal row diagnostics are incomplete",
    )
    _require(
        JOURNAL_DIAGNOSTIC_SCALARS.issubset(diagnostics),
        "journal native terminal row activation diagnostics are incomplete",
    )
    checkpoint_provenance = provenance.get("checkpoint_provenance")
    _require(
        isinstance(checkpoint_provenance, list) and bool(checkpoint_provenance),
        "journal native terminal row checkpoint provenance is incomplete",
    )
    _require(
        all(
            isinstance(item, Mapping) and set(item) == JOURNAL_CHECKPOINT_FIELDS
            for item in checkpoint_provenance
        ),
        "journal native terminal row checkpoint provenance is incomplete",
    )
    _validated_protocol_metrics(
        provenance.get("metrics"),
        field="journal.row_finished.provenance.metrics",
        require_complete=True,
    )


def _journal_missingness(value: Any) -> str | None:
    _require(
        value is None or (isinstance(value, str) and value in TERMINAL_STATUSES),
        "journal row missingness is invalid",
    )
    return value


def _journal_reason(value: Any) -> str | None:
    _require(value is None or isinstance(value, str), "journal row reason is invalid")
    return None if value is None else _sanitize_reason(value)


def _packet_binding_hash(manifest: Mapping[str, Any], source_commit: str) -> str:
    return _canonical_hash(
        {
            "schema_version": PACKET_SCHEMA_VERSION,
            "issue": 8872,
            "source_commit": source_commit,
            "protocol_semantic_hash": EXPECTED_PROTOCOL_SEMANTIC_HASH,
            "manifest_hash": manifest["manifest_hash"],
            "identities": manifest["identities"],
        }
    )


def _validate_binding(receipt: Mapping[str, Any], binding_hash: str, field: str) -> None:
    _require(
        receipt.get("packet_binding_hash") == binding_hash,
        f"{field}.packet_binding_hash does not bind the compiled packet",
    )


def _validate_receipt_source(
    value: Mapping[str, Any], field: str, source_commit: str, binding_hash: str
) -> None:
    _require_digest(value.get("source_commit"), f"{field}.source_commit", length=40)
    _require(
        value.get("source_commit") == source_commit,
        f"{field}.source_commit does not match production packet",
    )
    _validate_binding(value, binding_hash, field)


def _validate_activation_receipt(
    receipt: Mapping[str, Any], binding_hash: str, source_commit: str
) -> None:
    value = _mapping(receipt, "activation_receipt")
    _require(
        value.get("schema_version") == ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "activation receipt schema drifted",
    )
    _require(value.get("issue") == 8871, "activation receipt must be for issue 8871")
    _require(
        value.get("verdict") == "activation_pass",
        "#8871 receipt is not activation_pass",
    )
    _require(value.get("preserved") is True, "#8871 activation receipt is not preserved")
    _require(value.get("current") is True, "#8871 activation receipt is not current")
    _require(
        value.get("registered_rows_executed") is False,
        "activation receipt must prove registered rows were not executed",
    )
    _require(
        value.get("seed_disjoint") is True,
        "activation receipt must prove disjoint seeds",
    )
    _require(
        value.get("protocol_semantic_hash") == EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "activation receipt protocol hash drifted",
    )
    _require(
        value.get("production_manifest_hash") == PRODUCTION_MANIFEST_HASH,
        "activation receipt production manifest hash drifted",
    )
    preservation = _mapping(value.get("preservation"), "activation_receipt.preservation")
    _require(
        preservation.get("status") == "preserved",
        "activation preservation status is not preserved",
    )
    _require_nonempty_string(
        preservation.get("artifact_reference"),
        "activation_receipt.preservation.artifact_reference",
    )
    _require_digest(
        preservation.get("receipt_sha256"),
        "activation_receipt.preservation.receipt_sha256",
    )
    _validate_receipt_source(value, "activation_receipt", source_commit, binding_hash)


def _validate_speed_integrity_receipt(
    receipt: Mapping[str, Any], binding_hash: str, source_commit: str
) -> None:
    value = _mapping(receipt, "speed_integrity_receipt")
    _require(
        value.get("schema_version") == SPEED_INTEGRITY_RECEIPT_SCHEMA_VERSION,
        "#6102 integrity receipt schema drifted",
    )
    _require(value.get("issue") == 6102, "speed integrity receipt must be for issue 6102")
    _require(
        value.get("status") == "integrity_validated",
        "#6102 integrity receipt is not integrity_validated",
    )
    _require(value.get("preserved") is True, "#6102 integrity receipt is not preserved")
    _require(
        value.get("manifest_hash") == ROBOT_SPEED_MANIFEST_HASH,
        "#6102 manifest hash drifted",
    )
    for field in (
        "expected_rows",
        "native_rows",
        "excluded_rows",
        "fallback_rows",
        "degraded_rows",
        "missing_rows",
        "duplicate_rows",
        "provenance_invalid_rows",
    ):
        expected = EXPECTED_ROWS if field in {"expected_rows", "native_rows"} else 0
        _require(
            value.get(field) == expected,
            f"#6102 integrity field {field} is not {expected}",
        )
    _require(
        value.get("execution_mode") == "native_only",
        "#6102 execution was not native-only",
    )
    _require_nonempty_string(
        value.get("artifact_reference"), "speed_integrity_receipt.artifact_reference"
    )
    _require_digest(value.get("artifact_digest"), "speed_integrity_receipt.artifact_digest")
    _validate_receipt_source(value, "speed_integrity_receipt", source_commit, binding_hash)


def _validate_native_preflight(
    receipt: Mapping[str, Any], binding_hash: str, source_commit: str
) -> None:
    value = _mapping(receipt, "native_preflight")
    _require(
        value.get("schema_version") == NATIVE_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "native preflight receipt schema drifted",
    )
    _require(value.get("issue") == 8872, "native preflight receipt must be for issue 8872")
    _require(
        value.get("status") == "pass",
        "native planner/checkpoint preflight did not pass",
    )
    _require(value.get("native") is True, "native preflight must declare native=true")
    _require(value.get("fallback") is False, "native preflight must reject fallback")
    _require(
        value.get("degraded") is False,
        "native preflight must reject degraded execution",
    )
    _require(
        tuple(value.get("planner_ids", ())) == tuple(EXPECTED_PLANNERS),
        "native preflight planner roster drifted",
    )
    _require(
        value.get("checkpoint_provenance_complete") is True,
        "native checkpoint provenance is incomplete",
    )
    _require(
        value.get("protocol_semantic_hash") == EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "native preflight protocol hash drifted",
    )
    _require(
        value.get("manifest_hash") == PRODUCTION_MANIFEST_HASH,
        "native preflight manifest hash drifted",
    )
    _validate_receipt_source(value, "native_preflight", source_commit, binding_hash)


def _validate_private_admission(
    receipt: Mapping[str, Any], binding_hash: str, source_commit: str
) -> None:
    value = _mapping(receipt, "private_admission")
    _require(
        value.get("schema_version") == PRIVATE_ADMISSION_RECEIPT_SCHEMA_VERSION,
        "private admission receipt schema drifted",
    )
    _require(value.get("issue") == 8872, "private admission receipt must be for issue 8872")
    _require(value.get("wrapper_status") == "pass", "private wrapper admission did not pass")
    predicates = _mapping(value.get("predicates"), "private_admission.predicates")
    for predicate in REQUIRED_PRIVATE_PREDICATES:
        _require(
            predicates.get(predicate) is True,
            f"private admission predicate failed: {predicate}",
        )
    _require(
        value.get("private_details_excluded") is True,
        "private details may not enter the public packet",
    )
    _validate_receipt_source(value, "private_admission", source_commit, binding_hash)


def _validate_authorization(
    receipt: Mapping[str, Any], binding_hash: str, source_commit: str
) -> None:
    value = _mapping(receipt, "production_authorization")
    _require(
        value.get("schema_version") == AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
        "production authorization schema drifted",
    )
    _require(
        set(value) == AUTHORIZATION_KEYS,
        "production authorization contains unknown or secret-bearing keys",
    )
    _require(value.get("issue") == 8872, "production authorization must be for issue 8872")
    _require("authorized" not in value, "self-authenticated authorization flag is forbidden")
    _require(value.get("scope") == "run-production", "production authorization scope drifted")
    _require(value.get("issuer") == "private-ops", "production authorization issuer drifted")
    decision_id = _require_nonempty_string(
        value.get("decision_id"), "production_authorization.decision_id"
    )
    decision_id_lower = decision_id.lower()
    _require(
        bool(SAFE_DECISION_ID.fullmatch(decision_id))
        and ".." not in decision_id
        and not any(
            marker in decision_id_lower
            for marker in ("token", "secret", "credential", "bearer", "private")
        ),
        "production_authorization.decision_id is not a safe public identifier",
    )
    _require_digest(value.get("token_sha256"), "production_authorization.token_sha256")
    _require_digest(
        value.get("token_binding_sha256"),
        "production_authorization.token_binding_sha256",
    )
    _validate_receipt_source(value, "production_authorization", source_commit, binding_hash)


def _packet_core(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in packet.items() if key != "packet_sha256"}


def _validate_packet_structure(
    packet: Mapping[str, Any],
    manifest: Mapping[str, Any],
    source_commit: str,
) -> str:
    value = _mapping(packet, "production packet")
    _require(
        value.get("schema_version") == PACKET_SCHEMA_VERSION,
        "production packet schema drifted",
    )
    _require(value.get("issue") == 8872, "production packet issue must be 8872")
    _require(value.get("mode") == "production", "production packet mode drifted")
    _require(
        value.get("protocol_config") == EXPECTED_PROTOCOL_CONFIG,
        "production protocol path drifted",
    )
    _require(
        value.get("protocol_semantic_hash") == EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "production protocol hash drifted",
    )
    _require(
        value.get("manifest_hash") == PRODUCTION_MANIFEST_HASH,
        "production manifest hash drifted",
    )
    _require(
        value.get("source_commit") == source_commit,
        "production packet source commit drifted",
    )
    _require(
        value.get("expected_rows") == EXPECTED_ROWS,
        "production packet expected row count drifted",
    )
    identities = value.get("identities")
    _require(isinstance(identities, list), "production packet identities must be a list")
    _require(
        identities == manifest["identities"],
        "production packet identities drifted from compile_manifest",
    )
    _require(
        value.get("identity_count") == EXPECTED_ROWS,
        "production packet identity count drifted",
    )
    _require(
        value.get("unique_identity_count") == EXPECTED_ROWS,
        "production packet uniqueness drifted",
    )
    binding_hash = _packet_binding_hash(manifest, source_commit)
    _require(
        value.get("packet_binding_hash") == binding_hash,
        "production packet binding hash drifted",
    )
    _require(
        value.get("packet_sha256") == _canonical_hash(_packet_core(value)),
        "production packet digest drifted",
    )
    boundary = _mapping(value.get("execution_boundary"), "execution_boundary")
    _require(
        boundary.get("scheduler_submission") == "private_executor_only",
        "scheduler boundary drifted",
    )
    _require(boundary.get("default_mode") == "validate", "default mode must remain validate")
    _require(
        boundary.get("production_token_required") is True,
        "production token gate is missing",
    )
    _require(
        boundary.get("production_execution_disabled") is True,
        "production execution must remain disabled until authenticated authorization exists",
    )
    _require(
        boundary.get("authorization_verification") == AUTHORIZATION_VERIFICATION_CONTRACT,
        "authorization verification contract drifted",
    )
    accounting = _mapping(value.get("row_accounting"), "row_accounting")
    _require(
        accounting.get("one_terminal_status_per_identity") is True,
        "row accounting must require one terminal status per identity",
    )
    return binding_hash


def validate_production_packet(
    packet: Mapping[str, Any],
    *,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
    _require_source_checkout: bool = True,
) -> dict[str, Any]:
    """Validate a complete packet and all admission receipts before execution."""
    manifest = _compiled_manifest(config_path)
    source_commit = _require_digest(
        packet.get("source_commit"), "production packet source_commit", 40
    )
    if _require_source_checkout:
        _validate_source_checkout(source_commit)
    binding_hash = _validate_packet_structure(packet, manifest, source_commit)
    receipt_fields = (
        "activation_receipt",
        "speed_integrity_receipt",
        "native_preflight",
        "private_admission",
        "production_authorization",
    )
    for field in receipt_fields:
        _require(field in packet, f"production packet is missing {field}")
        projected = _project_public_receipt(packet.get(field), field)
        _require(
            projected == packet.get(field),
            f"{field} is not the canonical public receipt projection",
        )
        _assert_no_transient_state(packet.get(field), field)
    _validate_activation_receipt(packet["activation_receipt"], binding_hash, source_commit)
    _validate_speed_integrity_receipt(
        packet["speed_integrity_receipt"], binding_hash, source_commit
    )
    _validate_native_preflight(packet["native_preflight"], binding_hash, source_commit)
    _validate_private_admission(packet["private_admission"], binding_hash, source_commit)
    _validate_authorization(packet["production_authorization"], binding_hash, source_commit)
    return manifest


def inspect_packet(
    packet: Mapping[str, Any],
    *,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """Return a non-executing readiness result without weakening strict execution gates."""
    try:
        manifest = validate_production_packet(packet, config_path=config_path)
    except Exception:  # noqa: BLE001 - diagnostics must not echo untrusted validation details
        return {
            "ready": False,
            "manifest_hash": None,
            "expected_rows": EXPECTED_ROWS,
            "reason": "production packet rejected",
        }
    return {
        "ready": False,
        "manifest_hash": manifest["manifest_hash"],
        "expected_rows": EXPECTED_ROWS,
        "reason": PRODUCTION_EXECUTION_DISABLED_REASON,
    }


def build_production_packet(
    *,
    source_commit: str,
    activation_receipt: Mapping[str, Any],
    speed_integrity_receipt: Mapping[str, Any],
    native_preflight: Mapping[str, Any],
    private_admission: Mapping[str, Any],
    production_authorization: Mapping[str, Any],
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """Build one digest-bound production packet after all receipts pass."""
    _require_digest(source_commit, "source_commit", length=40)
    manifest = _compiled_manifest(config_path)
    binding_hash = _packet_binding_hash(manifest, source_commit)
    receipts = {
        "activation_receipt": _project_public_receipt(activation_receipt, "activation_receipt"),
        "speed_integrity_receipt": _project_public_receipt(
            speed_integrity_receipt, "speed_integrity_receipt"
        ),
        "native_preflight": _project_public_receipt(native_preflight, "native_preflight"),
        "private_admission": _project_public_receipt(private_admission, "private_admission"),
        "production_authorization": _project_public_receipt(
            production_authorization, "production_authorization"
        ),
    }
    _validate_source_checkout(source_commit)
    _validate_activation_receipt(receipts["activation_receipt"], binding_hash, source_commit)
    _validate_speed_integrity_receipt(
        receipts["speed_integrity_receipt"], binding_hash, source_commit
    )
    _validate_native_preflight(receipts["native_preflight"], binding_hash, source_commit)
    _validate_private_admission(receipts["private_admission"], binding_hash, source_commit)
    _validate_authorization(receipts["production_authorization"], binding_hash, source_commit)
    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "issue": 8872,
        "mode": "production",
        "protocol_config": EXPECTED_PROTOCOL_CONFIG,
        "protocol_semantic_hash": EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "source_commit": source_commit,
        "manifest_hash": manifest["manifest_hash"],
        "packet_binding_hash": binding_hash,
        "expected_rows": EXPECTED_ROWS,
        "identity_count": manifest["identity_count"],
        "unique_identity_count": manifest["unique_identity_count"],
        "identities": manifest["identities"],
        "execution_boundary": {
            "scheduler_submission": "private_executor_only",
            "public_repo_submits": False,
            "default_mode": "validate",
            "production_token_required": True,
            "production_token_source": "private-ops-ephemeral-environment",
            "production_execution_disabled": True,
            "authorization_verification": AUTHORIZATION_VERIFICATION_CONTRACT,
        },
        "row_accounting": {
            "one_terminal_status_per_identity": True,
            "forbidden_success_statuses": sorted(FORBIDDEN_SUCCESS_STATUSES),
            "identity_source": "scripts.validation.check_issue_6561_pedestrian_speed_protocol.compile_manifest",
        },
        **receipts,
    }
    packet["packet_sha256"] = _canonical_hash(packet)
    validate_production_packet(packet, config_path=config_path)
    return packet


def _native_outcome_status(
    expected: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    source_commit: str,
    manifest_hash: str,
) -> str:
    if provenance.get("fallback") is True:
        return "fallback"
    if provenance.get("degraded") is True:
        return "degraded"
    if provenance.get("execution_mode") != "native" or provenance.get("native") is not True:
        return "non_native"
    if expected["regime_id"] == "legacy_default":
        if provenance.get("intervention_status") != "not_applicable":
            return "intervention_not_activated"
    elif provenance.get("intervention_status") != "activated":
        return "intervention_not_activated"
    expected_fields = {
        "identity_key": expected["identity_key"],
        "scenario_id": expected["scenario_id"],
        "scenario_source_sha256": expected["scenario_source_sha256"],
        "regime_id": expected["regime_id"],
        "planner_id": expected["planner_id"],
        "planner_config_sha256": expected["planner_config_sha256"],
        "seed": expected["seed"],
        "horizon_steps": expected["horizon_steps"],
        "dt_seconds": expected["dt_seconds"],
        "robot_speed_cap_m_s": expected["robot_speed_cap_m_s"],
        "execution_mode": "native",
        "native": True,
        "fallback": False,
        "degraded": False,
        "protocol_semantic_hash": EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "manifest_hash": manifest_hash,
        "source_commit": source_commit,
        "runtime_controls": expected["runtime_controls"],
    }
    if any(
        provenance.get(field) != expected_value for field, expected_value in expected_fields.items()
    ):
        return "provenance_invalid"
    try:
        _validated_protocol_metrics(
            provenance.get("metrics"),
            field="row.provenance.metrics",
            require_complete=True,
        )
    except CampaignAdapterError:
        return "provenance_invalid"
    _assert_no_transient_state(provenance, "row.provenance")
    return SUCCESS_STATUS


def _expected_row_provenance(
    expected: Mapping[str, Any], *, source_commit: str, manifest_hash: str, status: str
) -> dict[str, Any]:
    """Preserve the validated identity contract even when execution is incomplete."""
    return {
        "provenance_status": "expected_identity_only",
        "identity_key": expected["identity_key"],
        "scenario_id": expected["scenario_id"],
        "scenario_source_sha256": expected["scenario_source_sha256"],
        "regime_id": expected["regime_id"],
        "planner_id": expected["planner_id"],
        "planner_config_sha256": expected["planner_config_sha256"],
        "seed": expected["seed"],
        "horizon_steps": expected["horizon_steps"],
        "dt_seconds": expected["dt_seconds"],
        "robot_speed_cap_m_s": expected["robot_speed_cap_m_s"],
        "runtime_controls": dict(expected["runtime_controls"]),
        "execution_mode": expected["execution_mode"],
        "protocol_semantic_hash": EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "manifest_hash": manifest_hash,
        "source_commit": source_commit,
        "terminal_status": status,
    }


def _normalize_outcome(  # noqa: C901
    expected: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *,
    source_commit: str,
    manifest_hash: str,
) -> dict[str, Any]:
    identity_key = str(expected["identity_key"])
    status = outcome.get("terminal_status")
    supplied_provenance = outcome.get("provenance")

    def _row_base(terminal_status: str, reason: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "identity_key": identity_key,
            "terminal_status": terminal_status,
            "missingness": terminal_status,
            "reason": _sanitize_reason(reason),
        }
        supplied_metrics = outcome.get("metrics")
        if isinstance(supplied_provenance, Mapping):
            provenance = dict(supplied_provenance)
            if supplied_metrics is None:
                supplied_metrics = provenance.get("metrics")
            if supplied_metrics is not None:
                safe_metrics = _finite_protocol_metrics(
                    supplied_metrics,
                    field="row.provenance.metrics",
                )
                provenance["metrics"] = safe_metrics
                if safe_metrics:
                    row["metrics"] = safe_metrics
            _assert_no_transient_state(provenance, "row.provenance")
            row["provenance"] = provenance
        else:
            row["provenance"] = _expected_row_provenance(
                expected,
                source_commit=source_commit,
                manifest_hash=manifest_hash,
                status=terminal_status,
            )
            if supplied_metrics is not None:
                safe_metrics = _finite_protocol_metrics(
                    supplied_metrics,
                    field="row.metrics",
                )
                if safe_metrics:
                    row["metrics"] = safe_metrics
        return row

    if not isinstance(status, str) or status not in TERMINAL_STATUSES:
        return _row_base("provenance_invalid", "executor returned an unknown terminal status")
    if status != SUCCESS_STATUS:
        return _row_base(status, str(outcome.get("reason") or status))
    provenance = supplied_provenance
    if not isinstance(provenance, Mapping):
        status = "provenance_invalid"
    else:
        status = _native_outcome_status(
            expected,
            provenance,
            source_commit=source_commit,
            manifest_hash=manifest_hash,
        )
    if status != SUCCESS_STATUS:
        return _row_base(status, status)
    metrics = _validated_protocol_metrics(
        provenance.get("metrics"),
        field="row.provenance.metrics",
        require_complete=True,
    )
    return {
        "identity_key": identity_key,
        "terminal_status": SUCCESS_STATUS,
        "missingness": None,
        "reason": None,
        "provenance": dict(provenance),
        "metrics": metrics,
    }


def account_production_rows(
    manifest: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Account outcomes into exactly one terminal row per compiled identity."""
    expected_rows = list(manifest["identities"])
    expected_by_key = {str(row["identity_key"]): row for row in expected_rows}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unexpected_count = 0
    for outcome in outcomes:
        key = outcome.get("identity_key") if isinstance(outcome, Mapping) else None
        if not isinstance(key, str) or key not in expected_by_key:
            unexpected_count += 1
            continue
        grouped[key].append(outcome)
    rows: list[dict[str, Any]] = []
    for expected in expected_rows:
        key = str(expected["identity_key"])
        observed = grouped.get(key, [])
        if not observed:
            rows.append(
                {
                    "identity_key": key,
                    "terminal_status": "missing",
                    "missingness": "missing",
                    "reason": "executor emitted no outcome for this identity",
                    "provenance": _expected_row_provenance(
                        expected,
                        source_commit=source_commit,
                        manifest_hash=str(manifest["manifest_hash"]),
                        status="missing",
                    ),
                }
            )
        elif len(observed) != 1:
            rows.append(
                {
                    "identity_key": key,
                    "terminal_status": "duplicate",
                    "missingness": "duplicate",
                    "reason": "executor emitted multiple outcomes for this identity",
                    "provenance": _expected_row_provenance(
                        expected,
                        source_commit=source_commit,
                        manifest_hash=str(manifest["manifest_hash"]),
                        status="duplicate",
                    ),
                }
            )
        else:
            rows.append(
                _normalize_outcome(
                    expected,
                    observed[0],
                    source_commit=source_commit,
                    manifest_hash=str(manifest["manifest_hash"]),
                )
            )
    statuses = Counter(row["terminal_status"] for row in rows)
    complete_native = (
        len(rows) == EXPECTED_ROWS
        and len({row["identity_key"] for row in rows}) == EXPECTED_ROWS
        and unexpected_count == 0
        and statuses == Counter({SUCCESS_STATUS: EXPECTED_ROWS})
    )
    return {
        "schema_version": ROW_ACCOUNTING_SCHEMA_VERSION,
        "issue": 8872,
        "protocol_semantic_hash": EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "manifest_hash": str(manifest["manifest_hash"]),
        "source_commit": source_commit,
        "expected_rows": EXPECTED_ROWS,
        "observed_outcomes": len(outcomes),
        "accounted_rows": len(rows),
        "unique_accounted_identities": len({row["identity_key"] for row in rows}),
        "unexpected_outcome_count": unexpected_count,
        "terminal_status_counts": dict(sorted(statuses.items())),
        "complete_native": complete_native,
        "admissible": complete_native,
        "claim_boundary": "row accounting only; no synthesis or scientific claim",
        "rows": rows,
    }


def _validate_identity_contract(
    identity: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve one row against the frozen scenario, planner, and regime bytes."""
    scenario_specs = protocol["scenario_contract"]["selected_scenarios"]
    scenario_spec = next(
        (
            dict(item)
            for item in scenario_specs
            if item.get("scenario_id") == identity.get("scenario_id")
        ),
        None,
    )
    _require(scenario_spec is not None, "identity scenario is not in the frozen protocol")
    planner_spec = next(
        (
            dict(item)
            for item in protocol["planner_contract"]["roster"]
            if item.get("planner_id") == identity.get("planner_id")
        ),
        None,
    )
    _require(planner_spec is not None, "identity planner is not in the frozen protocol")
    regime_spec = next(
        (
            dict(item)
            for item in protocol["pedestrian_speed_contract"]["regimes"]
            if item.get("regime_id") == identity.get("regime_id")
        ),
        None,
    )
    _require(regime_spec is not None, "identity regime is not in the frozen protocol")
    _require(
        identity.get("scenario_source_sha256") == scenario_spec.get("source_sha256"),
        "identity scenario digest drifted",
    )
    _require(
        identity.get("planner_config_sha256") == planner_spec.get("config_sha256"),
        "identity planner/config digest drifted",
    )
    _require(
        identity.get("runtime_controls") == regime_spec.get("runtime_controls"),
        "identity runtime controls drifted",
    )
    baseline = protocol["baseline_protocol"]
    for field, expected in (
        ("horizon_steps", baseline["horizon_steps"]),
        ("dt_seconds", baseline["dt_seconds"]),
        ("robot_speed_cap_m_s", baseline["robot_speed_cap_m_s"]),
        ("execution_mode", baseline["execution_mode"]),
    ):
        _require(identity.get(field) == expected, f"identity {field} drifted")
    _require(isinstance(identity.get("seed"), int), "identity seed is not an integer")
    return scenario_spec, planner_spec, regime_spec


def _safe_checkpoint_provenance(
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: checkpoint[key]
            for key in (
                "model_id",
                "path_label",
                "sha256",
                "size_bytes",
                "expected_sha256",
            )
            if key in checkpoint
        }
        for checkpoint in sorted(checkpoints.values(), key=lambda item: str(item.get("model_id")))
    ]


def _record_metric_sources(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return metric envelopes in precedence order without copying arbitrary fields."""
    sources: list[Mapping[str, Any]] = []
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        sources.append(metrics)
        nested = metrics.get("metric_values")
        if isinstance(nested, Mapping):
            sources.append(nested)
    for key in ("metric_values",):
        value = record.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    sources.append(record)
    return sources


def _record_metric_value(
    metrics: Mapping[str, Any], *names: str, field: str
) -> tuple[float | None, str | None]:
    """Read one finite raw metric, returning a stable missingness code."""
    for name in names:
        if name in metrics:
            try:
                return _finite_metric(metrics[name], f"record.{field}"), None
            except CampaignAdapterError:
                return None, f"metric_contract_nonfinite:{field}"
    return None, f"metric_contract_missing:{field}"


def _record_binary_metric(
    metrics: Mapping[str, Any], *names: str, field: str
) -> tuple[float | None, str | None]:
    """Convert an established count/flag source to a per-episode binary metric."""
    for name in names:
        if name not in metrics:
            continue
        value = metrics[name]
        if isinstance(value, bool):
            return (1.0 if value else 0.0), None
        number, error = _record_metric_value(metrics, name, field=field)
        if error is not None:
            return None, error
        return (1.0 if number > 0.0 else 0.0), None
    return None, f"metric_contract_missing:{field}"


def _extract_protocol_metrics(  # noqa: C901, PLR0912, PLR0915
    record: Mapping[str, Any], identity: Mapping[str, Any]
) -> tuple[dict[str, float], str | None]:
    """Extract the frozen row metric contract from a native episode record.

    Direct protocol-named fields are preferred.  When the map runner emits its
    established raw episode fields, the projection mirrors the reviewed #5578
    adapter: binary event rates come from counts/flags and exposure seconds come
    from the recorded interaction-exposure denominator.  No default or synthetic
    value is introduced.
    """
    names = _protocol_metric_names()
    values: dict[str, float] = {}
    sources = _record_metric_sources(record)
    present: set[str] = set()
    for name in names:
        for source in sources:
            if name in source:
                present.add(name)
                try:
                    values[name] = _finite_metric(source[name], f"record.metrics.{name}")
                except CampaignAdapterError:
                    return values, f"metric_contract_nonfinite:{name}"
                break

    raw_metrics = record.get("metrics")
    raw_metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}

    def _set_if_missing(name: str, value: float | None, error: str | None) -> str | None:
        if name in present:
            return None
        if error is not None or value is None:
            return error or f"metric_contract_missing:{name}"
        values[name] = value
        return None

    success, error = _record_binary_metric(raw_metrics, "success", field="success_rate")
    failure = _set_if_missing("success_rate", success, error)
    if failure:
        return values, failure

    total_collision, total_error = _record_metric_value(
        raw_metrics,
        "total_collision_count",
        "collisions",
        field="collision_count",
    )
    if "total_collision_count" in raw_metrics and "collisions" in raw_metrics:
        alternate_collision, alternate_error = _record_metric_value(
            raw_metrics,
            "collisions",
            field="collision_count",
        )
        if total_error is None and alternate_error is None:
            total_collision = max(total_collision, alternate_collision)
        elif total_error is None:
            return values, alternate_error
        else:
            return values, total_error
    collision = None if total_collision is None else (1.0 if total_collision > 0.0 else 0.0)
    failure = _set_if_missing("collision_rate", collision, total_error)
    if failure:
        return values, failure

    near_miss, error = _record_binary_metric(raw_metrics, "near_misses", field="near_miss_rate")
    failure = _set_if_missing("near_miss_rate", near_miss, error)
    if failure:
        return values, failure

    typed_counts: dict[str, float] = {}
    for metric_name, raw_name in (
        ("ped_collision_rate", "ped_collision_count"),
        ("obstacle_collision_rate", "obstacle_collision_count"),
        ("agent_collision_rate", "agent_collision_count"),
    ):
        count, error = _record_metric_value(raw_metrics, raw_name, field=raw_name)
        if error is not None:
            if metric_name in present:
                continue
            return values, error
        typed_counts[metric_name] = count
        failure = _set_if_missing(
            metric_name,
            1.0 if count > 0.0 else 0.0,
            None,
        )
        if failure:
            return values, failure

    if "unclassified_collision_rate" not in present:
        if total_collision is None or len(typed_counts) != 3:
            return values, "metric_contract_missing:unclassified_collision_rate"
        typed_total = sum(typed_counts.values())
        values["unclassified_collision_rate"] = 1.0 if total_collision > typed_total + 1e-9 else 0.0

    time_to_goal, error = _record_metric_value(
        raw_metrics,
        "time_to_goal_norm",
        field="time_to_goal_norm",
    )
    failure = _set_if_missing("time_to_goal_norm", time_to_goal, error)
    if failure:
        return values, failure

    exposure = record.get("interaction_exposure")
    if "total_exposure_seconds" not in present:
        if not isinstance(exposure, Mapping):
            return values, "metric_contract_missing:total_exposure_seconds"
        exposure_share, error = _record_metric_value(
            exposure,
            "interaction_exposure_share",
            field="interaction_exposure_share",
        )
        if error is not None:
            return values, "metric_contract_missing:total_exposure_seconds"
        exposure_steps, error = _record_metric_value(
            exposure,
            "interaction_exposure_denominator_steps",
            field="interaction_exposure_denominator_steps",
        )
        if error is not None:
            return values, "metric_contract_missing:total_exposure_seconds"
        dt_seconds = _finite_metric(identity.get("dt_seconds"), "identity.dt_seconds")
        total_exposure = exposure_share * exposure_steps * dt_seconds
        if not math.isfinite(total_exposure):
            return values, "metric_contract_nonfinite:total_exposure_seconds"
        values["total_exposure_seconds"] = total_exposure

    travel_distance, error = _record_metric_value(
        raw_metrics,
        "socnavbench_path_length",
        field="travel_distance_m",
    )
    failure = _set_if_missing("travel_distance_m", travel_distance, error)
    if failure:
        return values, failure

    mean_clearance, error = _record_metric_value(
        raw_metrics,
        "mean_clearance",
        field="mean_clearance_m",
    )
    failure = _set_if_missing("mean_clearance_m", mean_clearance, error)
    if failure:
        return values, failure

    min_clearance, error = _record_metric_value(
        raw_metrics,
        "min_clearance",
        field="min_clearance_m",
    )
    failure = _set_if_missing("min_clearance_m", min_clearance, error)
    if failure:
        return values, failure

    return values, None


def _native_outcome_from_record(
    identity: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    source_commit: str,
    manifest_hash: str,
    planner_algorithm: str,
    robot_speed_cap_m_s: float,
    checkpoint_provenance: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Turn one native map-runner record into a fully bound terminal row."""
    from scripts.benchmark.run_issue_8871_pedestrian_speed_canary import (
        _execution_disposition,
    )

    disposition, reason = _execution_disposition(record)
    metrics, metric_error = _extract_protocol_metrics(record, identity)
    if disposition != "native":
        outcome = {
            "identity_key": identity["identity_key"],
            "terminal_status": "degraded" if disposition == "degraded" else "failed",
            "reason": _sanitize_reason(reason or disposition),
        }
        if metrics:
            outcome["metrics"] = metrics
        return outcome
    metadata = record.get("algorithm_metadata")
    kinematics = metadata.get("planner_kinematics") if isinstance(metadata, Mapping) else None
    if not isinstance(kinematics, Mapping) or kinematics.get("execution_mode") != "native":
        outcome = {
            "identity_key": identity["identity_key"],
            "terminal_status": "non_native",
            "reason": "map-runner did not report native planner execution",
        }
        if metrics:
            outcome["metrics"] = metrics
        return outcome
    if metric_error is not None:
        return {
            "identity_key": identity["identity_key"],
            "terminal_status": "provenance_invalid",
            "reason": metric_error,
            "metrics": metrics,
        }
    from scripts.benchmark.run_issue_8871_pedestrian_speed_canary import (
        extract_activation_diagnostics,
    )

    diagnostics = extract_activation_diagnostics(record, identity)
    treated = identity["regime_id"] != "legacy_default"
    activated = not treated or (
        float(diagnostics["desired_speed_activation_fraction"]) >= 0.8
        and float(diagnostics["time_to_desired_speed_target_seconds"]) <= 2.0
    )
    intervention_status = (
        "not_applicable" if not treated else ("activated" if activated else "not_activated")
    )
    provenance = {
        "identity_key": identity["identity_key"],
        "scenario_id": identity["scenario_id"],
        "scenario_source_sha256": identity["scenario_source_sha256"],
        "regime_id": identity["regime_id"],
        "planner_id": identity["planner_id"],
        "planner_algorithm": planner_algorithm,
        "planner_config_sha256": identity["planner_config_sha256"],
        "seed": identity["seed"],
        "horizon_steps": identity["horizon_steps"],
        "dt_seconds": identity["dt_seconds"],
        "robot_speed_cap_m_s": robot_speed_cap_m_s,
        "runtime_controls": dict(identity["runtime_controls"]),
        "protocol_semantic_hash": EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "manifest_hash": manifest_hash,
        "source_commit": source_commit,
        "execution_mode": "native",
        "native": True,
        "fallback": False,
        "degraded": False,
        "intervention_status": intervention_status,
        "diagnostics": diagnostics,
        "trace_sha256": _canonical_hash(
            record.get("algorithm_metadata", {}).get("simulation_step_trace")
        ),
        "checkpoint_provenance": [dict(item) for item in checkpoint_provenance],
        "metrics": metrics,
    }
    _assert_no_transient_state(provenance, "row.provenance")
    if not activated:
        return {
            "identity_key": identity["identity_key"],
            "terminal_status": "intervention_not_activated",
            "missingness": "intervention_not_activated",
            "reason": "native trace failed the frozen activation rule",
            "provenance": provenance,
            "metrics": metrics,
        }
    return {
        "identity_key": identity["identity_key"],
        "terminal_status": SUCCESS_STATUS,
        "provenance": provenance,
        "metrics": metrics,
    }


def _execute_native_identity(  # noqa: PLR0913
    identity: Mapping[str, Any],
    *,
    source_commit: str,
    manifest_hash: str,
    protocol: Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, Any]],
    planner_specs: Mapping[str, Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
    policy_builder: Callable[..., Any],
    checkpoint_root: Path,
    episode_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one exact identity through the fixed native map-runner path."""
    from robot_sf.benchmark.map_runner import map_runner_episode
    from robot_sf.benchmark.map_runner.map_runner import (
        _resolve_policy_search_candidate_runtime,
    )
    from scripts.benchmark.run_issue_8871_pedestrian_speed_canary import (
        _bind_checkpoint_paths,
        _repo_path,
        _runtime_binding_context,
        _runtime_controls,
        _sha256,
        build_execution_scenario,
    )

    scenario_spec, _planner_spec, _regime_spec = _validate_identity_contract(identity, protocol)
    scenario_id = str(identity["scenario_id"])
    planner_id = str(identity["planner_id"])
    base = scenarios.get(scenario_id)
    _require(base is not None, f"scenario {scenario_id} was not loaded")
    _require(planner_id in planner_specs, f"planner {planner_id} was not loaded")
    spec = planner_specs[planner_id]
    _require(
        spec.get("config_sha256") == identity.get("planner_config_sha256"),
        "planner config digest does not match identity",
    )
    source_path = _repo_path(scenario_spec["source_path"], "scenario source")
    _require(
        _sha256(source_path) == identity["scenario_source_sha256"],
        "scenario source changed",
    )
    raw_config = dict(spec.get("raw_config") or {})
    resolved_algo, effective_config = _resolve_policy_search_candidate_runtime(
        default_algo=str(spec["algorithm"]),
        algo_config_path=str(spec["config_path"]) if spec.get("config_path") else None,
        algo_config=raw_config,
        scenario=base,
    )
    effective_config = _bind_checkpoint_paths(resolved_algo, effective_config, checkpoints)
    scenario = build_execution_scenario(base, identity)
    controls = _runtime_controls(identity)
    runner = episode_runner or map_runner_episode.run_map_episode
    with _runtime_binding_context(controls, seed=int(identity["seed"])):
        runtime_config = map_runner_episode._build_env_config(scenario, scenario_path=source_path)
        observed_cap = float(getattr(runtime_config.robot_config, "max_linear_speed", 0.0))
        _require(
            observed_cap == float(identity["robot_speed_cap_m_s"]) == ROBOT_SPEED_CAP_M_S,
            "native robot speed cap does not match the frozen identity",
        )
        record = runner(
            scenario=scenario,
            seed=int(identity["seed"]),
            horizon=int(identity["horizon_steps"]),
            dt=float(identity["dt_seconds"]),
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo=resolved_algo,
            scenario_path=source_path,
            algo_config=effective_config,
            algo_config_path=None,
            benchmark_track="experimental",
            record_simulation_step_trace=True,
            close_policy=False,
            policy_builder=policy_builder,
        )
    _require(isinstance(record, Mapping), "native map runner returned a non-mapping record")
    return _native_outcome_from_record(
        identity,
        record,
        source_commit=source_commit,
        manifest_hash=manifest_hash,
        planner_algorithm=resolved_algo,
        robot_speed_cap_m_s=observed_cap,
        checkpoint_provenance=_safe_checkpoint_provenance(checkpoints),
    )


@contextmanager
def _campaign_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignAdapterError("another #8872 execution holds the campaign lock") from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_journal(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CampaignAdapterError("cannot read existing execution journal") from exc
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CampaignAdapterError(
                f"execution journal line {line_number} is malformed"
            ) from exc
        _require(
            isinstance(event, Mapping),
            f"execution journal line {line_number} is not a mapping",
        )
        events.append(dict(event))
    return events


def _validate_journal_packet(packet: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the preserved production packet used as journal context."""
    _require(packet is not None, "journal reconciliation requires the preserved production packet")
    value = _mapping(packet, "journal production packet")
    validate_production_packet(value, _require_source_checkout=False)
    _require(
        value.get("expected_rows") == EXPECTED_ROWS
        and value.get("identity_count") == EXPECTED_ROWS
        and value.get("unique_identity_count") == EXPECTED_ROWS,
        "journal packet cardinality is not the frozen production cardinality",
    )
    _require(
        value.get("manifest_hash") == PRODUCTION_MANIFEST_HASH,
        "journal packet manifest is not the frozen production manifest",
    )
    return value


def _validate_journal_header(
    header: Mapping[str, Any],
    *,
    packet_context: Mapping[str, Any],
    expected_rows: int | None,
) -> int:
    """Validate the immutable execution identity before reading terminal rows."""
    _require(
        set(header) == JOURNAL_HEADER_FIELDS,
        "execution journal header is incomplete or contains unsupported fields",
    )
    _require(
        header.get("event") == "header"
        and header.get("schema_version") == JOURNAL_SCHEMA_VERSION
        and header.get("issue") == 8872,
        "execution journal header is invalid",
    )
    header_packet_sha256 = _require_digest(
        header.get("packet_sha256"), "execution journal header packet_sha256"
    )
    header_binding_hash = _require_digest(
        header.get("packet_binding_hash"),
        "execution journal header packet_binding_hash",
    )
    header_source_commit = header.get("source_commit")
    _require(
        isinstance(header_source_commit, str)
        and HEX_COMMIT.fullmatch(header_source_commit) is not None,
        "execution journal header source_commit is invalid",
    )
    header_manifest_hash = _require_digest(
        header.get("manifest_hash"), "execution journal header manifest_hash"
    )
    _require(
        header_manifest_hash == PRODUCTION_MANIFEST_HASH,
        "execution journal header manifest_hash is not the frozen production manifest",
    )
    _require(
        header_binding_hash == _packet_binding_hash(_compiled_manifest(), header_source_commit),
        "execution journal header packet binding is invalid",
    )
    header_expected_rows = header.get("expected_rows")
    _require(
        isinstance(header_expected_rows, int)
        and not isinstance(header_expected_rows, bool)
        and header_expected_rows > 0,
        "execution journal header expected_rows is invalid",
    )
    if expected_rows is not None:
        _require(
            expected_rows == header_expected_rows == packet_context["expected_rows"],
            "execution journal header expected_rows does not match the requested count",
        )
    _require(
        header_expected_rows == packet_context["expected_rows"]
        and header_packet_sha256 == packet_context["packet_sha256"]
        and header_binding_hash == packet_context["packet_binding_hash"]
        and header_source_commit == packet_context["source_commit"]
        and header_manifest_hash == packet_context["manifest_hash"],
        "execution journal header does not match the preserved production packet",
    )
    return header_expected_rows


def reconcile_execution_journal(
    journal_path: str | Path,
    *,
    packet: Mapping[str, Any] | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Summarize a journal only against its exact preserved production packet."""
    path = Path(journal_path)
    _require(path.is_file(), "execution journal does not exist")
    events = _read_journal(path)
    packet_context = _validate_journal_packet(packet)
    _require(events and events[0].get("event") == "header", "execution journal header is missing")
    header_expected_rows = _validate_journal_header(
        events[0],
        packet_context=packet_context,
        expected_rows=expected_rows,
    )
    started: list[str] = []
    terminal_rows: list[dict[str, Any]] = []
    finished: list[str] = []
    started_set: set[str] = set()
    for event in events[1:]:
        event_name = event.get("event")
        _require(
            event_name in {"row_started", "row_finished"},
            "execution journal event ordering is invalid",
        )
        identity_key = _journal_identity_key(event.get("identity_key"))
        if event_name == "row_started":
            _require(
                identity_key not in started_set,
                "execution journal row was started more than once",
            )
            started.append(identity_key)
            started_set.add(identity_key)
            continue
        _require(
            identity_key in started_set,
            "execution journal row finished before it started",
        )
        terminal_status = event.get("terminal_status")
        _require(
            isinstance(terminal_status, str) and terminal_status in TERMINAL_STATUSES,
            "journal row terminal status is invalid",
        )
        missingness = _journal_missingness(event.get("missingness"))
        reason = _journal_reason(event.get("reason"))
        if terminal_status == SUCCESS_STATUS:
            _require(
                missingness is None and reason is None,
                "journal native terminal success has contradictory missingness or reason",
            )
        row: dict[str, Any] = {
            "identity_key": identity_key,
            "terminal_status": terminal_status,
            "missingness": missingness,
            "reason": reason,
        }
        provenance = None
        if event.get("provenance") is not None:
            provenance = _journal_provenance(event.get("provenance"), identity_key)
        metric_value = event.get("metrics")
        if metric_value is None and isinstance(provenance, Mapping):
            metric_value = provenance.get("metrics")
        if terminal_status == SUCCESS_STATUS:
            _require(
                isinstance(provenance, Mapping),
                "journal native terminal row is missing provenance",
            )
            _validate_journal_success_provenance(
                provenance,
                identity_key,
                packet_context=packet_context,
            )
            _validated_protocol_metrics(
                metric_value,
                field="journal.row_finished.metrics",
                require_complete=True,
            )
            _require(
                metric_value == provenance.get("metrics"),
                "journal row metrics do not match provenance",
            )
        if metric_value is not None:
            row["metrics"] = _validated_protocol_metrics(
                metric_value,
                field="journal.row_finished.metrics",
                require_complete=terminal_status == SUCCESS_STATUS,
            )
        if provenance is not None:
            _assert_no_transient_state(provenance, "journal.row_finished.provenance")
            row["provenance"] = provenance
        _assert_no_transient_state(row, "journal.row_finished")
        finished.append(identity_key)
        terminal_rows.append(row)
    finished_set = set(finished)
    journal_reference = _safe_journal_reference(path)
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "journal_path": journal_reference,
        "journal_reference": journal_reference,
        "started_rows": len(started),
        "finished_rows": len(finished),
        "rows": terminal_rows,
        "in_flight_identity_keys": sorted(started_set - finished_set),
        "duplicate_finished_identity_keys": sorted(
            key for key, count in Counter(finished).items() if count > 1
        ),
        "expected_rows": header_expected_rows,
        "complete": (
            len(finished) == header_expected_rows
            and len(finished_set) == header_expected_rows
            and not any(count > 1 for count in Counter(finished).values())
        ),
        "retry_allowed": False,
        "resolution": "preserve journal and issue a new campaign identity after review",
    }


def _prepare_execution_paths(
    output_path: Path,
    journal_path: Path,
    lock_path: Path,
    *,
    packet: Mapping[str, Any] | None = None,
) -> None:
    _require(
        output_path.resolve() not in {journal_path.resolve(), lock_path.resolve()},
        "execution paths collide",
    )
    _require(journal_path.resolve() != lock_path.resolve(), "journal and lock paths collide")
    _require(not output_path.exists(), "refusing duplicate execution: receipt already exists")
    if journal_path.exists():
        _require(
            packet is not None,
            "existing execution journal reconciliation requires the preserved packet",
        )
        summary = reconcile_execution_journal(
            journal_path,
            packet=packet,
            expected_rows=packet.get("expected_rows") if packet is not None else None,
        )
        raise CampaignAdapterError(
            "existing execution journal requires reconciliation; automatic retry is forbidden "
            f"(started={summary['started_rows']}, finished={summary['finished_rows']})"
        )


def _append_journal_event(handle: Any, event: str, **payload: Any) -> None:
    record = {"event": event, **payload}
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _journal_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a complete, already-normalized terminal row for the durable journal."""
    identity_key = _journal_identity_key(row.get("identity_key"))
    terminal_status = row.get("terminal_status")
    _require(
        isinstance(terminal_status, str) and terminal_status in TERMINAL_STATUSES,
        "journal row terminal status is invalid",
    )
    missingness = _journal_missingness(row.get("missingness"))
    reason = _journal_reason(row.get("reason"))
    if terminal_status == SUCCESS_STATUS:
        _require(
            missingness is None and reason is None,
            "journal native terminal success has contradictory missingness or reason",
        )
    provenance = row.get("provenance")
    metrics = row.get("metrics")
    if metrics is not None:
        metrics = _validated_protocol_metrics(
            metrics,
            field="journal.row_finished.metrics",
            require_complete=terminal_status == SUCCESS_STATUS,
        )
    if provenance is not None:
        provenance = _journal_provenance(provenance, identity_key)
        if terminal_status == SUCCESS_STATUS:
            _validate_journal_success_provenance(provenance, identity_key)
        if metrics is not None and "metrics" in provenance:
            _require(
                metrics == provenance["metrics"],
                "journal row metrics do not match provenance",
            )
    payload = {
        "identity_key": identity_key,
        "terminal_status": terminal_status,
        "missingness": missingness,
        "reason": reason,
        "metrics": metrics,
        "provenance": provenance,
    }
    _assert_no_transient_state(payload, "journal.row_finished")
    return payload


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_production(
    packet: Mapping[str, Any],
    production_token: str,
    *,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
    checkpoint_root: str | Path,
    output_path: str | Path,
    journal_path: str | Path,
    lock_path: str | Path,
    _episode_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail closed until trusted private-ops authenticated-signer verification exists.

    Packet validation is the only work performed.  Token access, execution-path
    preparation, journal writes, and the native runner remain unreachable while
    the authenticated authorization contract is not configured.
    """
    manifest = validate_production_packet(packet, config_path=config_path)
    raise CampaignAdapterError(PRODUCTION_EXECUTION_DISABLED_REASON)
    _validate_token_binding(
        packet["production_authorization"],
        str(packet["packet_binding_hash"]),
        production_token,
    )
    output_path = Path(output_path)
    journal_path = Path(journal_path)
    lock_path = Path(lock_path)
    _prepare_execution_paths(output_path, journal_path, lock_path, packet=packet)
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    from robot_sf.benchmark.map_runner.map_runner import build_map_policy
    from scripts.benchmark.run_issue_8871_pedestrian_speed_canary import (
        _load_planner_specs,
        _load_scenarios,
        _registry_checkpoint,
        _required_model_ids,
    )

    protocol = load_protocol(config_path)
    scenarios = _load_scenarios(protocol)
    loaded_specs = _load_planner_specs(protocol)
    planner_specs = {str(spec["planner_id"]): spec for spec in loaded_specs}
    checkpoints = {
        checkpoint["model_id"]: checkpoint
        for checkpoint in (
            _registry_checkpoint(model_id, checkpoint_root)
            for model_id in _required_model_ids(loaded_specs)
        )
    }
    policy_cache: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {}

    def policy_builder(
        algo: str,
        algo_config: dict[str, Any],
        *,
        robot_kinematics: str | None = None,
        robot_command_mode: str | None = None,
        adapter_impact_eval: bool = False,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        key = _canonical_hash({"algo": algo, "config": algo_config})
        if key not in policy_cache:
            policy_cache[key] = build_map_policy(
                algo,
                algo_config,
                robot_kinematics=robot_kinematics,
                robot_command_mode=robot_command_mode,
                adapter_impact_eval=adapter_impact_eval,
            )
        return policy_cache[key]

    outcomes: list[Mapping[str, Any]] = []
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with _campaign_lock(lock_path), journal_path.open("x", encoding="utf-8") as journal:
        _append_journal_event(
            journal,
            "header",
            schema_version=JOURNAL_SCHEMA_VERSION,
            issue=8872,
            packet_sha256=packet["packet_sha256"],
            packet_binding_hash=packet["packet_binding_hash"],
            source_commit=packet["source_commit"],
            manifest_hash=packet["manifest_hash"],
            expected_rows=EXPECTED_ROWS,
        )
        try:
            for row_index, identity in enumerate(manifest["identities"]):
                _append_journal_event(
                    journal,
                    "row_started",
                    row_index=row_index,
                    identity_key=identity["identity_key"],
                )
                try:
                    outcome = _execute_native_identity(
                        identity,
                        source_commit=str(packet["source_commit"]),
                        manifest_hash=str(packet["manifest_hash"]),
                        protocol=protocol,
                        scenarios=scenarios,
                        planner_specs=planner_specs,
                        checkpoints=checkpoints,
                        policy_builder=policy_builder,
                        checkpoint_root=checkpoint_root,
                        episode_runner=_episode_runner,
                    )
                except Exception as exc:  # noqa: BLE001 - terminal row must be preserved
                    outcome = {
                        "identity_key": identity["identity_key"],
                        "terminal_status": "failed",
                        "reason": _exception_reason(exc),
                    }
                normalized = _normalize_outcome(
                    identity,
                    outcome,
                    source_commit=str(packet["source_commit"]),
                    manifest_hash=str(packet["manifest_hash"]),
                )
                outcomes.append(normalized)
                _append_journal_event(
                    journal,
                    "row_finished",
                    **_journal_row_payload(normalized),
                )
        finally:
            for policy, _meta in policy_cache.values():
                close = getattr(policy, "_planner_close", None)
                if callable(close):
                    close()
    report = account_production_rows(
        manifest,
        outcomes,
        source_commit=str(packet["source_commit"]),
    )
    report["packet_binding_hash"] = packet["packet_binding_hash"]
    report["packet_sha256"] = packet["packet_sha256"]
    report["execution_boundary"] = "fixed_native_map_runner; no scheduler submission"
    report["preparation_status"] = PREPARATION_STATUS
    report["scientific_evidence"] = False
    journal_reference = _safe_journal_reference(journal_path)
    report["journal_path"] = journal_reference
    report["journal_reference"] = journal_reference
    report["receipt_schema_version"] = RECEIPT_SCHEMA_VERSION
    _atomic_write_json(output_path, report)
    return report


def compile_smoke_manifest(*, source_commit: str | None = None) -> dict[str, Any]:
    """Compile three disjoint canary identities for diagnostic plumbing only."""
    from scripts.validation.build_issue_8871_pedestrian_speed_canary import (
        build_manifest,
        load_canary_config,
    )

    source_commit = source_commit or _git_head()
    _require_digest(source_commit, "source_commit", length=40)
    canary = build_manifest(load_canary_config(DEFAULT_CANARY_CONFIG), source_commit=source_commit)
    first_scenario = canary["identities"][0]["scenario_id"]
    first_planner = canary["identities"][0]["planner_id"]
    identities = [
        row
        for row in canary["identities"]
        if row["scenario_id"] == first_scenario and row["planner_id"] == first_planner
    ]
    _require(len(identities) == 3, "smoke selection must contain one row per regime")
    _require(
        all(row.get("registered") is False for row in identities),
        "smoke selected a registered row",
    )
    packet = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "issue": 8872,
        "diagnostic_only": True,
        "execution_allowed": False,
        "scientific_evidence": False,
        "preparation_status": PREPARATION_STATUS,
        "claim_boundary": "three disjoint canary identities; no scientific or benchmark evidence",
        "source_commit": source_commit,
        "expected_rows": len(identities),
        "identities": identities,
    }
    packet["manifest_hash"] = _canonical_hash(packet)
    return packet


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_json(path, value)


def _summary(mode: str, value: Mapping[str, Any], *, output: Path | None = None) -> dict[str, Any]:
    result = {
        "mode": mode,
        "schema_version": value.get("schema_version"),
        "manifest_hash": value.get("manifest_hash"),
        "expected_rows": value.get(
            "expected_rows",
            value.get("identity_count", value.get("expected_cell_count")),
        ),
    }
    for field in (
        "ready",
        "admissible",
        "complete_native",
        "preparation_status",
        "terminal_status_counts",
        "reason",
    ):
        if field in value:
            result[field] = value[field]
    if output is not None:
        result["output"] = _safe_journal_reference(output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit check/render/accounting mode; no mode submits production."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    validate_parser = subparsers.add_parser("validate", help="compile or inspect without execution")
    validate_parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL_CONFIG)
    validate_parser.add_argument("--packet", type=Path)

    smoke_parser = subparsers.add_parser("smoke", help="build a tiny disjoint diagnostic packet")
    smoke_parser.add_argument("--source-commit")
    smoke_parser.add_argument("--output", type=Path)

    render_parser = subparsers.add_parser("render-production", help="write an admitted packet only")
    render_parser.add_argument("--source-commit", required=True)
    render_parser.add_argument("--activation-receipt", type=Path, required=True)
    render_parser.add_argument("--speed-integrity-receipt", type=Path, required=True)
    render_parser.add_argument("--native-preflight", type=Path, required=True)
    render_parser.add_argument("--private-admission", type=Path, required=True)
    render_parser.add_argument("--production-authorization", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run-production",
        help="disabled until trusted private-ops signer verification is configured",
    )
    run_parser.add_argument("--packet", type=Path, required=True)
    run_parser.add_argument("--checkpoint-root", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--journal", type=Path, required=True)
    run_parser.add_argument("--lock", type=Path, required=True)
    run_parser.add_argument(
        "--token-env",
        default=PRODUCTION_TOKEN_ENV,
        help="reserved compatibility option; the disabled route never reads a token",
    )

    reconcile_parser = subparsers.add_parser(
        "reconcile", help="inspect an interrupted journal bound to its preserved packet"
    )
    reconcile_parser.add_argument("--journal", type=Path, required=True)
    reconcile_parser.add_argument("--packet", type=Path, required=True)
    reconcile_parser.add_argument("--expected-rows", type=int)

    args = parser.parse_args(argv)
    try:
        if args.mode == "validate":
            if args.packet:
                packet = _load_json(args.packet, "production packet")
                result = inspect_packet(packet, config_path=args.config)
                print(json.dumps(_summary("validate", result), sort_keys=True))
                return 0 if result["ready"] else 2
            manifest = _compiled_manifest(args.config)
            print(json.dumps(_summary("validate", manifest), sort_keys=True))
            return 0
        if args.mode == "smoke":
            smoke = compile_smoke_manifest(source_commit=args.source_commit)
            if args.output:
                _write_json(args.output, smoke)
            print(json.dumps(_summary("smoke", smoke, output=args.output), sort_keys=True))
            return 0
        if args.mode == "render-production":
            packet = build_production_packet(
                source_commit=args.source_commit,
                activation_receipt=_load_json(args.activation_receipt, "activation receipt"),
                speed_integrity_receipt=_load_json(
                    args.speed_integrity_receipt, "speed integrity receipt"
                ),
                native_preflight=_load_json(args.native_preflight, "native preflight"),
                private_admission=_load_json(args.private_admission, "private admission"),
                production_authorization=_load_json(
                    args.production_authorization, "production authorization"
                ),
            )
            _write_json(args.output, packet)
            print(
                json.dumps(
                    _summary("render-production", packet, output=args.output),
                    sort_keys=True,
                )
            )
            return 0
        if args.mode == "run-production":
            raise CampaignAdapterError(PRODUCTION_EXECUTION_DISABLED_REASON)
        if args.mode == "reconcile":
            packet = _load_json(args.packet, "production packet")
            summary = reconcile_execution_journal(
                args.journal,
                packet=packet,
                expected_rows=(
                    args.expected_rows
                    if args.expected_rows is not None
                    else packet.get("expected_rows")
                ),
            )
            print(json.dumps(summary, sort_keys=True))
            return 0
    except CampaignAdapterError as exc:
        print(json.dumps({"mode": args.mode, "error": str(exc)}, sort_keys=True))
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
