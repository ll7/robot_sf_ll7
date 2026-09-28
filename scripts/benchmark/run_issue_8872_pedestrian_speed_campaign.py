#!/usr/bin/env python3
"""Validate, render, and account for the frozen #8872 campaign.

This is a public admission and row-accounting boundary with one fixed native
map-runner execution path, not a scheduler launcher.  The production identities
are always compiled by ``check_issue_6561_pedestrian_speed_protocol.compile_manifest``.
A production packet is accepted only when the current preserved #8871 activation
receipt, the verified #6102 integrity receipt, native planner/checkpoint preflight,
all six private admission predicates, and an ephemeral private-ops token digest
are bound to the same packet.  ``run-production`` never imports an arbitrary
executor or submits Slurm work.

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
    Require a validated packet and an ephemeral private-ops token, then invoke
    the fixed native ``map_runner_episode.run_map_episode`` path and record one
    terminal status/missingness value for every expected identity.  Any
    fallback, degraded, non-native, duplicate, missing, provenance-invalid,
    or intervention-not-activated row makes the report non-admissible.

The private operations layer is responsible for translating its host-specific
checks into the normalized receipt shapes documented in the context note and
for issuing the ephemeral token.  Preparation status remains
``PREPARATION_INCOMPLETE`` while the scientific #8871 activation gate is not
an ``activation_pass``; the fixed public executor does not override that gate.
"""

from __future__ import annotations

import argparse
import hashlib
import fcntl
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
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
DEFAULT_CANARY_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_8871_pedestrian_speed_canary_v1.yaml"
)
PRODUCTION_MANIFEST_HASH = (
    "371f1a0160ec7faf1ade531691f104e2a1c92f7c34857e887ba1ba539e1b5238"
)
ROBOT_SPEED_MANIFEST_HASH = (
    "e32ce197149af62bf366f5ca95abbb42215b379fe7916d916ccdd544dce8666f"
)
PACKET_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_campaign_packet.v1"
ROW_ACCOUNTING_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_row_accounting.v1"
SMOKE_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_smoke.v1"
ACTIVATION_RECEIPT_SCHEMA_VERSION = "robot_sf.issue_8871_activation_receipt.v1"
SPEED_INTEGRITY_RECEIPT_SCHEMA_VERSION = "robot_sf.issue_6102_integrity_receipt.v1"
NATIVE_PREFLIGHT_RECEIPT_SCHEMA_VERSION = (
    "robot_sf.issue_8872_native_preflight_receipt.v1"
)
PRIVATE_ADMISSION_RECEIPT_SCHEMA_VERSION = (
    "robot_sf.issue_8872_private_admission_receipt.v1"
)
AUTHORIZATION_RECEIPT_SCHEMA_VERSION = "robot_sf.issue_8872_production_authorization.v1"
EXPECTED_ROWS = 2160
PREPARATION_STATUS = "PREPARATION_INCOMPLETE"
EXPECTED_PROTOCOL_CONFIG = (
    "configs/benchmarks/issue_6561_pedestrian_speed_protocol.yaml"
)
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
FORBIDDEN_TRANSIENT_KEYS = frozenset(
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
SAFE_ARTIFACT_REFERENCE = re.compile(
    r"^(?:artifact|wandb)://[A-Za-z0-9._/-]+(?::v[0-9]+)?$"
)
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
JOURNAL_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_execution_journal.v1"
RECEIPT_SCHEMA_VERSION = "robot_sf.issue_8872_pedestrian_speed_execution_receipt.v1"
PRODUCTION_TOKEN_ENV = "ROBOT_SF_8872_EXECUTION_TOKEN"
ROBOT_SPEED_CAP_M_S = 2.0


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
            _require(
                child_name.lower() not in FORBIDDEN_TRANSIENT_KEYS,
                f"{path}.{child_name} contains private transient state",
            )
            _assert_safe_public_value(child, f"{path}.{child_name}", child_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_safe_public_value(child, f"{path}[{index}]")


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
    _require(
        HEX_COMMIT.fullmatch(source_commit) is not None, "source commit is not a SHA-1"
    )
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
        raise CampaignAdapterError(
            "source commit does not resolve in the checkout"
        ) from exc
    _require(resolved == source_commit, "source commit does not resolve exactly")
    _require(
        _git_head() == source_commit, "source commit differs from checked-out HEAD"
    )
    _require(_git_clean(), "source checkout is not clean; refuse execution")


def _token_binding_digest(token: str, packet_binding_hash: str) -> str:
    _require(
        isinstance(token, str) and len(token) >= 32, "execution token is too short"
    )
    return hashlib.sha256(f"{token}:{packet_binding_hash}".encode("utf-8")).hexdigest()


def _validate_token_binding(
    authorization: Mapping[str, Any], packet_binding_hash: str, token: str
) -> None:
    """Check a private-ops token without storing or printing its value."""
    _require(
        authorization.get("token_sha256")
        == hashlib.sha256(token.encode("utf-8")).hexdigest(),
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
        raise CampaignAdapterError(
            f"cannot read {field}: {exc.__class__.__name__}"
        ) from exc
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


def _validate_binding(
    receipt: Mapping[str, Any], binding_hash: str, field: str
) -> None:
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
    _require(
        value.get("preserved") is True, "#8871 activation receipt is not preserved"
    )
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
    preservation = _mapping(
        value.get("preservation"), "activation_receipt.preservation"
    )
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
    _require(
        value.get("issue") == 6102, "speed integrity receipt must be for issue 6102"
    )
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
    _require_digest(
        value.get("artifact_digest"), "speed_integrity_receipt.artifact_digest"
    )
    _validate_receipt_source(
        value, "speed_integrity_receipt", source_commit, binding_hash
    )


def _validate_native_preflight(
    receipt: Mapping[str, Any], binding_hash: str, source_commit: str
) -> None:
    value = _mapping(receipt, "native_preflight")
    _require(
        value.get("schema_version") == NATIVE_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "native preflight receipt schema drifted",
    )
    _require(
        value.get("issue") == 8872, "native preflight receipt must be for issue 8872"
    )
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
    _require(
        value.get("issue") == 8872, "private admission receipt must be for issue 8872"
    )
    _require(
        value.get("wrapper_status") == "pass", "private wrapper admission did not pass"
    )
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
        value.get("issue") == 8872, "production authorization must be for issue 8872"
    )
    _require(
        "authorized" not in value, "self-authenticated authorization flag is forbidden"
    )
    _require(
        value.get("scope") == "run-production", "production authorization scope drifted"
    )
    _require(
        value.get("issuer") == "private-ops", "production authorization issuer drifted"
    )
    _require_nonempty_string(
        value.get("decision_id"), "production_authorization.decision_id"
    )
    _require_digest(value.get("token_sha256"), "production_authorization.token_sha256")
    _require_digest(
        value.get("token_binding_sha256"),
        "production_authorization.token_binding_sha256",
    )
    _validate_receipt_source(
        value, "production_authorization", source_commit, binding_hash
    )


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
    _require(
        isinstance(identities, list), "production packet identities must be a list"
    )
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
    _require(
        boundary.get("default_mode") == "validate", "default mode must remain validate"
    )
    _require(
        boundary.get("production_token_required") is True,
        "production token gate is missing",
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
) -> dict[str, Any]:
    """Validate a complete packet and all admission receipts before execution."""
    manifest = _compiled_manifest(config_path)
    source_commit = _require_digest(
        packet.get("source_commit"), "production packet source_commit", 40
    )
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
        _assert_no_transient_state(packet.get(field), field)
    _validate_activation_receipt(
        packet["activation_receipt"], binding_hash, source_commit
    )
    _validate_speed_integrity_receipt(
        packet["speed_integrity_receipt"], binding_hash, source_commit
    )
    _validate_native_preflight(packet["native_preflight"], binding_hash, source_commit)
    _validate_private_admission(
        packet["private_admission"], binding_hash, source_commit
    )
    _validate_authorization(
        packet["production_authorization"], binding_hash, source_commit
    )
    return manifest


def inspect_packet(
    packet: Mapping[str, Any],
    *,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """Return a non-executing readiness result without weakening strict execution gates."""
    try:
        manifest = validate_production_packet(packet, config_path=config_path)
    except CampaignAdapterError as exc:
        return {
            "ready": False,
            "manifest_hash": packet.get("manifest_hash"),
            "expected_rows": EXPECTED_ROWS,
            "reason": str(exc),
        }
    return {
        "ready": True,
        "manifest_hash": manifest["manifest_hash"],
        "expected_rows": EXPECTED_ROWS,
        "reason": None,
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
        "activation_receipt": dict(activation_receipt),
        "speed_integrity_receipt": dict(speed_integrity_receipt),
        "native_preflight": dict(native_preflight),
        "private_admission": dict(private_admission),
        "production_authorization": dict(production_authorization),
    }
    for field, receipt in receipts.items():
        _assert_no_transient_state(receipt, field)
    _validate_source_checkout(source_commit)
    _validate_activation_receipt(
        receipts["activation_receipt"], binding_hash, source_commit
    )
    _validate_speed_integrity_receipt(
        receipts["speed_integrity_receipt"], binding_hash, source_commit
    )
    _validate_native_preflight(
        receipts["native_preflight"], binding_hash, source_commit
    )
    _validate_private_admission(
        receipts["private_admission"], binding_hash, source_commit
    )
    _validate_authorization(
        receipts["production_authorization"], binding_hash, source_commit
    )
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
    if (
        provenance.get("execution_mode") != "native"
        or provenance.get("native") is not True
    ):
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
        provenance.get(field) != expected_value
        for field, expected_value in expected_fields.items()
    ):
        return "provenance_invalid"
    _assert_no_transient_state(provenance, "row.provenance")
    return SUCCESS_STATUS


def _normalize_outcome(
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
            "reason": reason,
        }
        if isinstance(supplied_provenance, Mapping):
            _assert_no_transient_state(supplied_provenance, "row.provenance")
            row["provenance"] = dict(supplied_provenance)
        return row

    if not isinstance(status, str) or status not in TERMINAL_STATUSES:
        return _row_base(
            "provenance_invalid", "executor returned an unknown terminal status"
        )
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
    return {
        "identity_key": identity_key,
        "terminal_status": SUCCESS_STATUS,
        "missingness": None,
        "reason": None,
        "provenance": dict(provenance),
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
                }
            )
        elif len(observed) != 1:
            rows.append(
                {
                    "identity_key": key,
                    "terminal_status": "duplicate",
                    "missingness": "duplicate",
                    "reason": "executor emitted multiple outcomes for this identity",
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
    _require(
        scenario_spec is not None, "identity scenario is not in the frozen protocol"
    )
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
        for checkpoint in sorted(
            checkpoints.values(), key=lambda item: str(item.get("model_id"))
        )
    ]


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
    if disposition != "native":
        return {
            "identity_key": identity["identity_key"],
            "terminal_status": "degraded" if disposition == "degraded" else "failed",
            "reason": reason or disposition,
        }
    metadata = record.get("algorithm_metadata")
    kinematics = (
        metadata.get("planner_kinematics") if isinstance(metadata, Mapping) else None
    )
    if (
        not isinstance(kinematics, Mapping)
        or kinematics.get("execution_mode") != "native"
    ):
        return {
            "identity_key": identity["identity_key"],
            "terminal_status": "non_native",
            "reason": "map-runner did not report native planner execution",
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
        "not_applicable"
        if not treated
        else ("activated" if activated else "not_activated")
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
    }
    _assert_no_transient_state(provenance, "row.provenance")
    if not activated:
        return {
            "identity_key": identity["identity_key"],
            "terminal_status": "intervention_not_activated",
            "missingness": "intervention_not_activated",
            "reason": "native trace failed the frozen activation rule",
            "provenance": provenance,
        }
    return {
        "identity_key": identity["identity_key"],
        "terminal_status": SUCCESS_STATUS,
        "provenance": provenance,
    }


def _execute_native_identity(
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
        _sha256,
        _repo_path,
        _runtime_binding_context,
        build_execution_scenario,
        _runtime_controls,
    )

    scenario_spec, planner_spec, _regime_spec = _validate_identity_contract(
        identity, protocol
    )
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
    effective_config = _bind_checkpoint_paths(
        resolved_algo, effective_config, checkpoints
    )
    scenario = build_execution_scenario(base, identity)
    controls = _runtime_controls(identity)
    runner = episode_runner or map_runner_episode.run_map_episode
    with _runtime_binding_context(controls, seed=int(identity["seed"])):
        runtime_config = map_runner_episode._build_env_config(
            scenario, scenario_path=source_path
        )
        observed_cap = float(
            getattr(runtime_config.robot_config, "max_linear_speed", 0.0)
        )
        _require(
            observed_cap
            == float(identity["robot_speed_cap_m_s"])
            == ROBOT_SPEED_CAP_M_S,
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
    _require(
        isinstance(record, Mapping), "native map runner returned a non-mapping record"
    )
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
            raise CampaignAdapterError(
                "another #8872 execution holds the campaign lock"
            ) from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_journal(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CampaignAdapterError(
            f"cannot read existing execution journal: {exc}"
        ) from exc
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


def reconcile_execution_journal(
    journal_path: str | Path, *, expected_rows: int | None = None
) -> dict[str, Any]:
    """Summarize an interrupted journal; this function never authorizes a retry."""
    path = Path(journal_path)
    _require(path.is_file(), f"execution journal does not exist: {path}")
    events = _read_journal(path)
    started = [
        str(event.get("identity_key"))
        for event in events
        if event.get("event") == "row_started"
    ]
    finished = [
        str(event.get("identity_key"))
        for event in events
        if event.get("event") == "row_finished"
    ]
    started_set = set(started)
    finished_set = set(finished)
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "journal_path": str(path),
        "started_rows": len(started),
        "finished_rows": len(finished),
        "in_flight_identity_keys": sorted(started_set - finished_set),
        "duplicate_finished_identity_keys": sorted(
            key for key, count in Counter(finished).items() if count > 1
        ),
        "expected_rows": expected_rows,
        "complete": expected_rows is not None and len(finished) == expected_rows,
        "retry_allowed": False,
        "resolution": "preserve journal and issue a new campaign identity after review",
    }


def _prepare_execution_paths(
    output_path: Path, journal_path: Path, lock_path: Path
) -> None:
    _require(
        output_path.resolve() not in {journal_path.resolve(), lock_path.resolve()},
        "execution paths collide",
    )
    _require(
        journal_path.resolve() != lock_path.resolve(), "journal and lock paths collide"
    )
    _require(
        not output_path.exists(), "refusing duplicate execution: receipt already exists"
    )
    if journal_path.exists():
        summary = reconcile_execution_journal(journal_path)
        raise CampaignAdapterError(
            "existing execution journal requires reconciliation; automatic retry is forbidden "
            f"(started={summary['started_rows']}, finished={summary['finished_rows']})"
        )


def _append_journal_event(handle: Any, event: str, **payload: Any) -> None:
    record = {"event": event, **payload}
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
    """Run the fixed native executor with durable, no-retry accounting."""
    manifest = validate_production_packet(packet, config_path=config_path)
    _validate_token_binding(
        packet["production_authorization"],
        str(packet["packet_binding_hash"]),
        production_token,
    )
    output_path = Path(output_path)
    journal_path = Path(journal_path)
    lock_path = Path(lock_path)
    _prepare_execution_paths(output_path, journal_path, lock_path)
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    from scripts.benchmark.run_issue_8871_pedestrian_speed_canary import (
        _load_planner_specs,
        _load_scenarios,
        _registry_checkpoint,
        _required_model_ids,
    )
    from robot_sf.benchmark.map_runner.map_runner import build_map_policy

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
            packet_sha256=packet["packet_sha256"],
            packet_binding_hash=packet["packet_binding_hash"],
            source_commit=packet["source_commit"],
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
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                outcomes.append(outcome)
                _append_journal_event(
                    journal,
                    "row_finished",
                    identity_key=identity["identity_key"],
                    terminal_status=outcome.get("terminal_status"),
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
    report["journal_path"] = str(journal_path)
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
    canary = build_manifest(
        load_canary_config(DEFAULT_CANARY_CONFIG), source_commit=source_commit
    )
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


def _summary(
    mode: str, value: Mapping[str, Any], *, output: Path | None = None
) -> dict[str, Any]:
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
        result["output"] = str(output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit check/render/accounting mode; no mode submits production."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="compile or inspect without execution"
    )
    validate_parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL_CONFIG)
    validate_parser.add_argument("--packet", type=Path)

    smoke_parser = subparsers.add_parser(
        "smoke", help="build a tiny disjoint diagnostic packet"
    )
    smoke_parser.add_argument("--source-commit")
    smoke_parser.add_argument("--output", type=Path)

    render_parser = subparsers.add_parser(
        "render-production", help="write an admitted packet only"
    )
    render_parser.add_argument("--source-commit", required=True)
    render_parser.add_argument("--activation-receipt", type=Path, required=True)
    render_parser.add_argument("--speed-integrity-receipt", type=Path, required=True)
    render_parser.add_argument("--native-preflight", type=Path, required=True)
    render_parser.add_argument("--private-admission", type=Path, required=True)
    render_parser.add_argument("--production-authorization", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run-production", help="execute the fixed native runner"
    )
    run_parser.add_argument("--packet", type=Path, required=True)
    run_parser.add_argument("--checkpoint-root", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--journal", type=Path, required=True)
    run_parser.add_argument("--lock", type=Path, required=True)
    run_parser.add_argument(
        "--token-env",
        default=PRODUCTION_TOKEN_ENV,
        help="environment variable containing the ephemeral private-ops token",
    )

    reconcile_parser = subparsers.add_parser(
        "reconcile", help="inspect an interrupted journal without retrying it"
    )
    reconcile_parser.add_argument("--journal", type=Path, required=True)
    reconcile_parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)

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
            print(
                json.dumps(_summary("smoke", smoke, output=args.output), sort_keys=True)
            )
            return 0
        if args.mode == "render-production":
            packet = build_production_packet(
                source_commit=args.source_commit,
                activation_receipt=_load_json(
                    args.activation_receipt, "activation receipt"
                ),
                speed_integrity_receipt=_load_json(
                    args.speed_integrity_receipt, "speed integrity receipt"
                ),
                native_preflight=_load_json(args.native_preflight, "native preflight"),
                private_admission=_load_json(
                    args.private_admission, "private admission"
                ),
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
            packet = _load_json(args.packet, "production packet")
            token = os.environ.get(args.token_env)
            _require(
                token is not None,
                f"missing private-ops token environment {args.token_env}",
            )
            report = run_production(
                packet,
                token,
                checkpoint_root=args.checkpoint_root,
                output_path=args.output,
                journal_path=args.journal,
                lock_path=args.lock,
            )
            print(
                json.dumps(
                    _summary("run-production", report, output=args.output),
                    sort_keys=True,
                )
            )
            return 0 if report["admissible"] else 2
        if args.mode == "reconcile":
            summary = reconcile_execution_journal(
                args.journal, expected_rows=args.expected_rows
            )
            print(json.dumps(summary, sort_keys=True))
            return 0
    except CampaignAdapterError as exc:
        print(json.dumps({"mode": args.mode, "error": str(exc)}, sort_keys=True))
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
