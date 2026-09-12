#!/usr/bin/env python3
"""Read-only checkpoint preservation custody checker (issue #8831).

Builds a deterministic, sanitized preservation inventory for checkpoint/model artifacts
referenced by active compute-window workloads. Identity resolution is delegated to existing
owners: the model registry (:func:`robot_sf.models.registry.load_registry`), the local-artifact
classifier (:func:`robot_sf.benchmark.local_model_artifacts.is_local_output_model_path`), the
oracle trace-URI registry
(:func:`robot_sf.training.oracle_trace_uri_registry.load_trace_uri_registry`), and a checkpoint
compatibility audit receipt produced by ``scripts/models/audit_checkpoint_compatibility.py`` for
metadata-only loadability (no inference, no reimplemented loader). Byte digests use
:func:`robot_sf.evidence.writers.sha256_file`.

Check-only: it never copies, downloads, publishes, deletes, or mutates scheduler, GitHub,
credential, or scientific state. Every artifact ends in one stable state; loadability is reported
separately and is never a performance, benchmark, or redistribution claim. CLI::

    uv run python scripts/validation/check_checkpoint_preservation.py \
        --check --fixture <checkpoint-fixture> --format json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.local_model_artifacts import is_local_output_model_path
from robot_sf.evidence.writers import sha256_file
from robot_sf.models.registry import load_registry
from robot_sf.training.oracle_trace_uri_registry import load_trace_uri_registry
from scripts.models.audit_checkpoint_compatibility import ALIAS_VERSIONS, DURABLE_LOCATORS

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SCHEMA = "robot_sf.checkpoint_preservation_input.v1"
REPORT_SCHEMA = "robot_sf.checkpoint_preservation_report.v1"
AUDIT_SCHEMA = "robot_sf.checkpoint_compatibility_audit.v1"
LOADABILITY_OWNER = "scripts/models/audit_checkpoint_compatibility.py"
CLAIM_BOUNDARY = (
    "Read-only preservation inventory and fail-closed custody classification; it does not copy, "
    "download, publish, delete, or mutate scheduler, GitHub, credential, or scientific state. "
    "Loadability is metadata-only and never planner-performance, benchmark, or rights evidence."
)
EXIT_READY, EXIT_BLOCKED, EXIT_MALFORMED = 0, 1, 2

STATE_READY = "preservation_ready"
STATE_AMBIGUOUS = "blocked_ambiguous_identity"
STATE_LINEAGE = "blocked_missing_lineage"
STATE_INVENTORY = "blocked_incomplete_inventory"
STATE_NO_ARTIFACT = "blocked_missing_artifact"
STATE_NO_COMPANION = "blocked_missing_companion"
STATE_PARTIAL = "blocked_partial_copy"
STATE_DIGEST = "blocked_digest_mismatch"
STATE_LOAD = "blocked_loadability_failed"
STATE_CONTRACT = "blocked_contract_mismatch"
STATE_TRAINING = "blocked_incomplete_training"
STATE_DESTINATION = "blocked_unsafe_destination"
STATE_PUBLICATION = "blocked_uncleared_publication"
LOAD_VERIFIED = "verified_metadata"
LOAD_UNAVAILABLE = "loadability_unavailable"
LOAD_FAILED = "loadability_failed"
LOAD_NOT_CHECKED = "not_checked"

# Fail-closed state groups in precedence order; the first matching group wins.
_STATE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (state, tuple(codes.split()))
    for state, codes in (
        (
            STATE_AMBIGUOUS,
            "ambiguous_identity unknown_model_id identity_conflict digest_not_pinned invalid_artifact_record",
        ),
        (STATE_LINEAGE, "missing_lineage data_identity_unresolved missing_training_status"),
        (STATE_INVENTORY, "inventory_field_missing"),
        (STATE_NO_ARTIFACT, "artifact_missing unsafe_input_path unsafe_symlink"),
        (
            STATE_NO_COMPANION,
            "companion_missing normalizer_missing companion_unsafe normalizer_unsafe",
        ),
        (STATE_PARTIAL, "partial_copy companion_partial_copy"),
        (
            STATE_DIGEST,
            "digest_mismatch companion_digest_mismatch normalizer_digest_mismatch receipt_digest_mismatch",
        ),
        (
            STATE_LOAD,
            "loadability_failed invalid_artifact missing_data_member "
            "non_finite_parameters artifact_unreadable",
        ),
        (
            STATE_CONTRACT,
            "observation_contract_mismatch action_contract_mismatch loadability_contract_mismatch",
        ),
        (STATE_TRAINING, "incomplete_training"),
        (
            STATE_DESTINATION,
            "undeclared_destination unsafe_destination mutable_destination publication_destination_not_public",
        ),
        (STATE_PUBLICATION, "publication_uncleared publication_rights_blocked rights_blocked"),
    )
)
_LOAD_ENV_CODES = frozenset(
    {
        "dependency_unavailable",
        "dependency_group_unknown",
        "missing_custom_object",
        "loader_probe_timeout",
        "loader_probe_failed",
        "loader_probe_malformed",
    }
)
_LOAD_FATAL_CODES = frozenset({"artifact_unreadable", "invalid_artifact", "missing_data_member"})
_URI_SUFFIXES = (":latest", "/latest", ":head", "/head", ":main", "/main")
_URI_PREFIXES = (
    "artifact://",
    "private-artifact://",
    "wandb-artifact://",
    "s3://",
    "gs://",
    "https://",
)
_ROW_KEYS = (
    "artifact_id model_id artifact_version artifact_path artifact_sha256 sha256_observed "
    "byte_size byte_size_observed availability storage_class architecture observation_contract "
    "action_contract seed training_status downstream_consumers load_status loadability_owner"
).split()
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_CREDENTIAL_RE = re.compile(
    r"(?i)(?:^|[/_.?&=-])(?:api[_-]?key|secret|password|passwd|token|credential|bearer)(?:$|[/_.?&=-])"
)
_PRIVATE_TEXT_RE = re.compile(
    r"(?i)(?::/{2}|(?:^|[^a-z0-9])(?:api[_-]?key|secret|password|passwd|token|credential|bearer)(?:[^a-z0-9]|$))"
)
_SIGNED_URL_RE = re.compile(r"(?i)[?&](?:sig|signature|token|expires|x-amz-|x-goog-)")
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s\"'(=,])(?:~(?:[/\\]|$)|/[\w.-]+(?:/[\w.-]+)+|[A-Za-z]:[\\/])"
)


@dataclass(frozen=True)
class _Context:
    """Owners resolved once per fixture evaluation."""

    root: Path
    registry: Mapping[str, Mapping[str, Any]]
    traces: frozenset[str]
    audit: Mapping[str, Mapping[str, Any]]


def _find(code: str, artifact_id: str | None, detail: str) -> dict[str, str | None]:
    """Return a sanitized finding record."""
    return {"code": code, "artifact_id": artifact_id, "detail": detail}


def _label(path: Path) -> str:
    """Return a private-safe label for an intake path."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _text(value: Any, limit: int = 200) -> str | None:
    """Return a private-safe stripped string, or ``None`` when unsafe or empty.

    Free-text fields must never echo credential-like values or absolute private
    paths, so any absolute-path shape is rejected as unsafe.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > limit:
        return None
    return None if _PRIVATE_TEXT_RE.search(text) or _ABSOLUTE_PATH_RE.search(text) else text


def _rel(value: Any) -> str | None:
    """Return a private-safe relative POSIX path, or ``None`` when unsafe."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("\\", "/")
    if raw.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", raw) or "@" in raw:
        return None
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or ".." in parts or _CREDENTIAL_RE.search(raw):
        return None
    return "/".join(parts)


def _uri(value: Any) -> str | None:
    """Return a private-safe declared durable URI, or ``None`` when unsafe."""
    if not isinstance(value, str):
        return None
    uri = value.strip()
    if not uri.startswith(_URI_PREFIXES) or ".." in uri.split("/"):
        return None
    return None if _SIGNED_URL_RE.search(uri) or _CREDENTIAL_RE.search(uri) else uri


def _sha(value: Any) -> str | None:
    """Return a normalized 64-hex digest, or ``None`` when not pinned."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if _SHA_RE.fullmatch(normalized) else None


def _int(value: Any) -> int | None:
    """Return a non-negative integer field, or ``None`` when invalid."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _contract(value: Any) -> dict[str, Any] | None:
    """Return a validated observation/action contract, or ``None`` when incomplete."""
    if not isinstance(value, Mapping):
        return None
    schema, shape = _text(value.get("schema"), 128), value.get("shape")
    if schema is None or not isinstance(shape, list) or not shape:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in shape):
        return None
    return {"schema": schema, "shape": [int(item) for item in shape]}


def _pin(entry: Mapping[str, Any], name: str) -> str | None:
    """Return the registry-pinned SHA-256 for an artifact filename, if declared."""
    release = entry.get("github_release")
    if isinstance(release, Mapping):
        per_file = release.get("per_file_sha256")
        if name and isinstance(per_file, Mapping) and (pinned := _sha(per_file.get(name))):
            return pinned
        if pinned := _sha(release.get("sha256")):
            return pinned
    return _sha(entry.get("sha256"))


def _storage_class(entry: Mapping[str, Any] | None, rel: str | None) -> str:
    """Classify storage class from registry ownership and local path conventions."""
    if entry is not None:
        if isinstance(entry.get("github_release"), Mapping):
            return "public_release"
        if entry.get("wandb_artifact_path") or entry.get("wandb_run_path"):
            return "cloud_durable"
        if entry.get("local_only") is True:
            return "personal_durable"
    scratch = rel is not None and (is_local_output_model_path(rel) or rel.startswith("output/"))
    return "local_scratch" if scratch else "unknown"


def _read_declared(
    payload: Mapping[str, Any],
    key: str,
    root: Path,
    findings: list[dict[str, Any]],
    *,
    kind: str,
    reader: Callable[[Path], Any],
) -> Any:
    """Read one declared owner document, failing closed on error."""
    label = _rel(payload.get(key))
    path = root / label if label else None
    if path is None:
        findings.append(_find("unsafe_input_path", None, kind))
        return None
    if not path.is_file():
        findings.append(_find(f"{key}_missing", None, label))
        return None
    try:
        return reader(path)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        findings.append(_find(f"{key}_unreadable", None, label))
        return None


def _read_json(path: Path) -> Any:
    """Return parsed JSON from *path*, raising ``ValueError`` on malformed content."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("JSON document must be a mapping")
    return payload


def _trace_identities(source: Mapping[str, Any]) -> tuple[list[str], bool]:
    """Return canonical trace identities and validity for one registry entry.

    Only canonical identifiers are indexed. Explicit non-digest placeholders
    (``pending``/``unknown``/``unavailable``) stay valid but are never treated as
    resolvable identities, and malformed shapes fail closed.
    """
    identities: list[str] = []
    valid = True
    for key in ("dataset_id", "trace_id"):
        value = source.get(key)
        if value is None:
            continue
        canonical = _text(value, 128)
        if canonical is None or _SAFE_ID_RE.fullmatch(canonical) is None:
            valid = False
            continue
        identities.append(canonical)
    if source.get("uri") is not None:
        uri = _uri(source.get("uri"))
        if uri is None:
            valid = False
        else:
            identities.append(uri)
    if source.get("sha256") is not None:
        digest = _sha(source.get("sha256"))
        if digest is not None:
            identities.append(digest)
        elif str(source.get("sha256")).strip().lower() not in {"pending", "unknown", "unavailable"}:
            valid = False
    return identities, valid


def _load_context(
    payload: Mapping[str, Any], root: Path, findings: list[dict[str, Any]]
) -> _Context:
    """Resolve registry, trace, and audit owners through existing modules."""
    registry = _read_declared(
        payload, "registry", root, findings, kind="registry", reader=load_registry
    )
    trace_doc = _read_declared(
        payload,
        "trace_registry",
        root,
        findings,
        kind="trace registry",
        reader=load_trace_uri_registry,
    )
    audit_doc = _read_declared(
        payload, "compatibility_audit", root, findings, kind="audit receipt", reader=_read_json
    )
    traces: set[str] = set()
    if isinstance(trace_doc, Mapping):
        identities, valid = _trace_identities(trace_doc)
        if not valid or not identities:
            findings.append(_find("unsupported_receipt", None, "trace registry document"))
        traces.update(identities)
        raw_traces = trace_doc.get("traces")
        if not isinstance(raw_traces, list):
            findings.append(_find("unsupported_receipt", None, "trace registry traces list"))
            raw_traces = []
        for index, trace in enumerate(raw_traces):
            if not isinstance(trace, Mapping):
                findings.append(_find("unsupported_receipt", None, f"trace registry entry {index}"))
                continue
            entry_ids, entry_valid = _trace_identities(trace)
            if not entry_valid or not entry_ids:
                findings.append(_find("unsupported_receipt", None, f"trace registry entry {index}"))
                continue
            traces.update(entry_ids)
    elif trace_doc is not None:
        findings.append(_find("unsupported_receipt", None, "trace registry document"))
    rows = (
        audit_doc.get("models")
        if isinstance(audit_doc, Mapping) and audit_doc.get("schema") == AUDIT_SCHEMA
        else None
    )
    if rows is None and audit_doc is not None:
        findings.append(_find("unsupported_receipt", None, "audit receipt schema"))
    audit = {
        row["model_id"]: row
        for row in rows or []
        if isinstance(row, Mapping) and isinstance(row.get("model_id"), str)
    }
    return _Context(
        root=root,
        registry=registry if isinstance(registry, Mapping) else {},
        traces=frozenset(traces),
        audit=audit,
    )


def _blank_row(artifact_id: str) -> dict[str, Any]:
    """Return the complete sanitized inventory skeleton for one artifact."""
    row = dict.fromkeys(_ROW_KEYS)
    row.update(
        artifact_id=artifact_id,
        availability="missing",
        storage_class="unknown",
        framework={"name": None, "version": None},
        normalization={
            "required": False,
            "embedded": False,
            "path": None,
            "sha256": None,
            "byte_size": None,
        },
        companions=[],
        producer={"config": None, "commit": None, "data_identity": None},
        downstream_consumers=[],
        destination={"uri": None, "class": None},
        publication={"requested": False, "rights": None},
        load_status=LOAD_NOT_CHECKED,
        loadability_owner=None,
        state=STATE_READY,
        reason_codes=[],
    )
    return row


def _resolve(  # noqa: C901, PLR0912 - flat fail-closed check sequence
    record: Mapping[str, Any], index: int, ctx: _Context
) -> tuple[dict[str, Any], list[str], Mapping[str, Any] | None]:
    """Resolve identity, availability, digest, and inventory fields for one record."""
    codes: list[str] = []
    row = _blank_row(f"redacted-artifact-{index}")
    artifact_id = _text(record.get("artifact_id"), 128)
    if artifact_id is None or _SAFE_ID_RE.fullmatch(artifact_id) is None or ".." in artifact_id:
        codes.append("invalid_artifact_record")
    else:
        row["artifact_id"] = artifact_id
    declared = row["artifact_sha256"] = _sha(record.get("artifact_sha256"))
    if declared is None:
        codes.append("digest_not_pinned")
    row["artifact_version"] = version = _text(record.get("artifact_version"), 64)
    if version is not None and version.lower() in ALIAS_VERSIONS and declared is None:
        codes.append("ambiguous_identity")
    row["byte_size"] = _int(record.get("byte_size"))
    framework = record.get("framework")
    framework = framework if isinstance(framework, Mapping) else {}
    row["framework"] = {
        "name": _text(framework.get("name")),
        "version": _text(framework.get("version")),
    }
    row["architecture"] = _text(record.get("architecture"))
    row["observation_contract"] = _contract(record.get("observation_contract"))
    row["action_contract"] = _contract(record.get("action_contract"))
    row["seed"] = _int(record.get("seed"))
    if None in (
        row["byte_size"],
        *row["framework"].values(),
        row["architecture"],
        row["observation_contract"],
        row["action_contract"],
        row["seed"],
    ):
        codes.append("inventory_field_missing")
    model_id = _text(record.get("model_id"), 128)
    entry = ctx.registry.get(model_id) if model_id else None
    if model_id is not None:
        row["model_id"] = model_id
        if entry is None:
            codes.append("unknown_model_id")
    rel = _rel(record.get("artifact_path")) if record.get("artifact_path") is not None else None
    if record.get("artifact_path") is not None and rel is None:
        codes.append("unsafe_input_path")
    if rel is None and entry is not None and entry.get("local_path"):
        rel = _rel(entry.get("local_path"))
        if rel is None:
            codes.append("unsafe_input_path")
    if entry is not None and (pinned := _pin(entry, Path(rel).name if rel else "")):
        if declared is not None and declared != pinned:
            codes.append("identity_conflict")
        row["artifact_sha256"] = pinned
    local = ctx.root / rel if rel is not None else None
    if local is not None and local.is_symlink():
        codes.append("unsafe_symlink")
    elif local is not None and local.is_file():
        row.update(
            artifact_path=rel,
            availability="present_local",
            sha256_observed=sha256_file(local),
            byte_size_observed=local.stat().st_size,
        )
        if row["artifact_sha256"] is not None and row["sha256_observed"] != row["artifact_sha256"]:
            codes.append("digest_mismatch")
        if row["byte_size"] is not None and row["byte_size_observed"] != row["byte_size"]:
            codes.append("partial_copy")
    elif entry is not None and any(
        entry.get(key) for key in ("github_release", "wandb_run_path", "wandb_artifact_path")
    ):
        row["availability"] = "remote_only"
    else:
        codes.append("artifact_missing")
    row["storage_class"] = _text(record.get("storage_class"), 64) or _storage_class(entry, rel)
    return row, codes, entry


def _contained(candidate: Path, root: Path) -> bool:
    """Return True when *candidate* resolves inside *root* without symlinks."""
    try:
        candidate.resolve(strict=True).relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _companions(
    record: Mapping[str, Any], ctx: _Context
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Verify declared normalizer and companion files against their digests."""
    codes: list[str] = []
    raw_norm = record.get("normalization")
    raw_norm = raw_norm if isinstance(raw_norm, Mapping) else {}
    norm = {
        "required": bool(raw_norm.get("required")),
        "embedded": bool(raw_norm.get("embedded")),
        "path": None,
        "sha256": None,
        "byte_size": None,
    }
    raw: list[Any] = []
    if norm["required"] and not norm["embedded"]:
        norm.update(
            path=_rel(raw_norm.get("path")),
            sha256=_sha(raw_norm.get("sha256")),
            byte_size=_int(raw_norm.get("byte_size")),
        )
        raw.append({"role": "normalizer", **norm})
    raw_companions = record.get("companions")
    if raw_companions is not None and not isinstance(raw_companions, list):
        codes.append("inventory_field_missing")
        raw_companions = []
    companions: list[dict[str, Any]] = []
    for index, item in enumerate([*raw, *(raw_companions or [])]):
        item = item if isinstance(item, Mapping) else {}
        role = _text(item.get("role")) or f"companion-{index}"
        rel, digest, size = (
            _rel(item.get("path")),
            _sha(item.get("sha256")),
            _int(item.get("byte_size")),
        )
        companions.append({"role": role, "path": rel, "sha256": digest, "byte_size": size})
        prefix = "normalizer" if role == "normalizer" else "companion"
        if rel is None or digest is None or size is None:
            codes.append("inventory_field_missing")
        else:
            candidate = ctx.root / rel
            if not candidate.is_file() and not candidate.is_symlink():
                codes.append(f"{prefix}_missing")
            elif candidate.is_symlink() or not _contained(candidate, ctx.root):
                codes.append(f"{prefix}_unsafe")
            elif candidate.stat().st_size != size:
                codes.append(f"{prefix}_partial_copy")
            elif sha256_file(candidate) != digest:
                codes.append(f"{prefix}_digest_mismatch")
    return norm, companions, codes


def _lineage(
    record: Mapping[str, Any], entry: Mapping[str, Any] | None, ctx: _Context
) -> tuple[dict[str, Any], list[str]]:
    """Verify producer config, commit, data identity, and training status."""
    codes: list[str] = []
    producer = record.get("producer")
    producer = producer if isinstance(producer, Mapping) else {}
    config = _rel(producer.get("config")) if producer.get("config") is not None else None
    if config is None and entry is not None and entry.get("config_path"):
        config = _rel(entry.get("config_path"))
    if config is None or not (ctx.root / config).is_file():
        codes.append("missing_lineage")
    commit = producer.get("commit")
    if not isinstance(commit, str) and entry is not None:
        commit = entry.get("commit")
    commit = _text(commit, 40)
    if commit is None or _COMMIT_RE.fullmatch(commit.lower()) is None:
        codes.append("missing_lineage")
    data_identity = _uri(producer.get("data_identity")) or _text(producer.get("data_identity"))
    if data_identity is None:
        codes.append("missing_lineage")
    elif ctx.traces and data_identity not in ctx.traces:
        codes.append("data_identity_unresolved")
    status = _text(record.get("training_status"), 32)
    if status not in {"final", "partial"}:
        codes.append("missing_training_status")
    elif status == "partial":
        codes.append("incomplete_training")
    return (
        {
            "config": config,
            "commit": commit,
            "data_identity": data_identity,
            "training_status": status,
        },
        codes,
    )


def _receipt_binding_codes(
    receipt: Mapping[str, Any], observed: str | None, framework: str | None
) -> list[str]:
    """Return digest and framework binding codes for a loadability receipt."""
    codes: list[str] = []
    artifact = receipt.get("artifact")
    receipt_sha = _sha(artifact.get("sha256")) if isinstance(artifact, Mapping) else None
    verified = receipt.get("load_status") == "verified"
    if (verified and (receipt_sha is None or observed is None)) or (
        receipt_sha is not None and observed is not None and receipt_sha != observed
    ):
        codes.append("receipt_digest_mismatch")
    probe = receipt.get("probe") if isinstance(receipt.get("probe"), Mapping) else {}
    loader = _text(probe.get("loader"), 64)
    if verified and framework is not None and loader is not None and loader != framework:
        codes.append("loadability_contract_mismatch")
    return codes


def _probe_contract_codes(probe: Mapping[str, Any], contracts: Mapping[str, Any]) -> list[str]:
    """Return shape and finiteness contract codes from the loadability probe."""
    codes: list[str] = []
    for prefix in ("observation", "action"):
        declared = (contracts.get(prefix) or {}).get("shape")
        seen = probe.get(f"{prefix}_shape")
        if declared is not None and seen is not None and list(declared) != list(seen):
            codes.append(f"{prefix}_contract_mismatch")
    if probe.get("parameters_finite") is False:
        codes.append("non_finite_parameters")
    return codes


def _loadability(
    ctx: _Context,
    model_id: str | None,
    contracts: Mapping[str, Any],
    observed: str | None,
    framework: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Classify metadata-only loadability from the existing owner's receipt.

    A receipt that claims ``verified`` is trusted only when it is bound to the
    declared artifact digest and framework loader; every fatal outcome stays
    blocking even when its reason code is not part of the generic fatal set.
    """
    if not ctx.audit:
        return {"status": LOAD_NOT_CHECKED, "owner": None}, []
    receipt = ctx.audit.get(model_id) if model_id is not None else None
    if receipt is None:
        return (
            {"status": LOAD_UNAVAILABLE, "owner": LOADABILITY_OWNER},
            ["loadability_receipt_missing"],
        )
    probe = receipt.get("probe") if isinstance(receipt.get("probe"), Mapping) else {}
    codes = _receipt_binding_codes(receipt, observed, framework) + _probe_contract_codes(
        probe, contracts
    )
    if receipt.get("load_status") == "verified":
        return {"status": LOAD_VERIFIED, "owner": LOADABILITY_OWNER}, codes
    reasons = {str(code) for code in receipt.get("reason_codes") or []}
    fatal = reasons & _LOAD_FATAL_CODES
    if fatal:
        return {"status": LOAD_FAILED, "owner": LOADABILITY_OWNER}, [
            *codes,
            "loadability_failed",
            *sorted(fatal),
        ]
    status = str(receipt.get("load_status") or "").lower()
    if status in {"failed", "error", "blocked", "invalid"}:
        return {"status": LOAD_FAILED, "owner": LOADABILITY_OWNER}, [*codes, "loadability_failed"]
    detail = "framework_unavailable" if reasons & _LOAD_ENV_CODES else "loadability_unavailable"
    return {"status": LOAD_UNAVAILABLE, "owner": LOADABILITY_OWNER}, [*codes, detail]


def _destination(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Verify durable destination declaration and publication clearance."""
    codes: list[str] = []
    destination = record.get("destination")
    destination = destination if isinstance(destination, Mapping) else {}
    class_name, uri = _text(destination.get("class"), 64), _uri(destination.get("uri"))
    publication = record.get("publication")
    publication = publication if isinstance(publication, Mapping) else {}
    requested, rights = bool(publication.get("requested")), _text(publication.get("rights"), 32)
    if class_name not in DURABLE_LOCATORS or uri is None:
        codes.append("undeclared_destination")
    elif any(uri.endswith(suffix) for suffix in _URI_SUFFIXES):
        codes.append("mutable_destination")
    if requested:
        if rights == "blocked":
            codes.append("publication_rights_blocked")
        if rights != "cleared" or publication.get("cleared") is not True:
            codes.append("publication_uncleared")
        if class_name != "public_release":
            codes.append("publication_destination_not_public")
    return (
        {"uri": uri, "class": class_name},
        {"requested": requested, "rights": rights},
        codes,
    )


def _state(codes: list[str]) -> str:
    """Return the highest-precedence stable state for one artifact."""
    present = set(codes)
    for state, group in _STATE_GROUPS:
        if present & set(group):
            return state
    return STATE_READY


def _primary_code(row: Mapping[str, Any]) -> str:
    """Return the first reason code that determines a row's blocking state."""
    for state, group in _STATE_GROUPS:
        if row["state"] == state:
            return next((c for c in row["reason_codes"] if c in group), str(row["state"]))
    return str(row["state"])


def _evaluate(raw: Any, index: int, ctx: _Context) -> dict[str, Any]:
    """Evaluate one artifact record into a sanitized preservation inventory row."""
    if not isinstance(raw, Mapping):
        return {
            **_blank_row(f"redacted-artifact-{index}"),
            "state": STATE_AMBIGUOUS,
            "reason_codes": ["invalid_artifact_record"],
        }
    row, codes, entry = _resolve(raw, index, ctx)
    norm, companions, companion_codes = _companions(raw, ctx)
    producer, lineage_codes = _lineage(raw, entry, ctx)
    destination, publication, destination_codes = _destination(raw)
    load, load_codes = _loadability(
        ctx,
        row["model_id"],
        {"observation": row["observation_contract"], "action": row["action_contract"]},
        row["sha256_observed"],
        (row["framework"] or {}).get("name"),
    )
    consumers: set[str] = set()
    raw_consumers = raw.get("downstream_consumers")
    if raw_consumers is not None and not isinstance(raw_consumers, list):
        codes.append("inventory_field_missing")
        raw_consumers = []
    for consumer in raw_consumers or []:
        label = _text(consumer)
        if label is None:
            codes.append("inventory_field_missing")
        else:
            consumers.add(label)
    reasons = sorted({*codes, *companion_codes, *lineage_codes, *destination_codes, *load_codes})
    return {
        **row,
        "state": _state(reasons),
        "reason_codes": reasons,
        "normalization": norm,
        "companions": companions,
        "producer": producer,
        "downstream_consumers": sorted(consumers),
        "destination": destination,
        "publication": publication,
        "load_status": load["status"],
        "loadability_owner": load["owner"],
    }


def _workloads(
    payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], findings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind every active workload checkpoint reference to an inventory row."""
    known = {row["artifact_id"] for row in rows}
    raw_workloads = payload.get("workloads") or []
    if not isinstance(raw_workloads, list):
        findings.append(_find("invalid_fixture", None, "workloads must be a list"))
        return []
    workloads = []
    for index, item in enumerate(raw_workloads):
        if not isinstance(item, Mapping):
            findings.append(_find("invalid_fixture", None, f"workload[{index}]"))
            continue
        workload_id = _text(item.get("workload_id"), 128) or f"workload-{index}"
        raw_refs = item.get("checkpoint_refs") or []
        if not isinstance(raw_refs, list):
            findings.append(_find("invalid_fixture", None, f"workload[{index}] refs"))
            raw_refs = []
        refs, unresolved = [], []
        for ref in sorted({str(value) for value in raw_refs}):
            ref_id = _text(ref, 128)
            if ref_id is None or ref_id not in known:
                unresolved.append(ref_id or "unsafe-reference")
                findings.append(_find("workload_reference_uncovered", ref_id, workload_id))
            else:
                refs.append(ref_id)
        workloads.append(
            {
                "workload_id": workload_id,
                "checkpoint_refs": sorted(refs + unresolved),
                "resolved": not unresolved,
                "unresolved_refs": sorted(unresolved),
            }
        )
    return sorted(workloads, key=lambda item: item["workload_id"])


def _load_fixture(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load and validate the fixture header without echoing private values."""
    if not path.is_file():
        return None, [_find("fixture_unreadable", None, _label(path))]
    try:
        payload = _read_json(path)
    except (OSError, ValueError):
        return None, [_find("fixture_unreadable", None, _label(path))]
    if payload.get("schema") != FIXTURE_SCHEMA:
        return None, [_find("invalid_fixture", None, "fixture schema mismatch")]
    return dict(payload), []


def build_report(fixture_path: Path) -> dict[str, Any]:
    """Build the deterministic preservation custody report for one fixture.

    Args:
        fixture_path: Sanitized ``robot_sf.checkpoint_preservation_input.v1`` JSON fixture.

    Returns:
        The report mapping; ``status`` is ``ready``, ``blocked``, or ``unknown``.
    """
    fixture_path = fixture_path.resolve()
    label = _label(fixture_path)
    payload, findings = _load_fixture(fixture_path)
    if payload is None:
        return {
            "schema": REPORT_SCHEMA,
            "status": "unknown",
            "claim_boundary": CLAIM_BOUNDARY,
            "fixture": label,
            "summary": {},
            "workloads": [],
            "artifacts": [],
            "findings": findings,
        }
    root_label = _rel(payload.get("root"))
    root = (fixture_path.parent / root_label).resolve() if root_label else fixture_path.parent
    if not root.is_dir():
        findings.append(_find("invalid_root", None, root_label or "<fixture-dir>"))
    ctx = _load_context(payload, root, findings)
    defaults = payload.get("defaults")
    defaults = defaults if isinstance(defaults, Mapping) else {}
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        findings.append(_find("invalid_fixture", None, "artifacts must be a non-empty list"))
        raw_artifacts = []
    rows = sorted(
        (
            _evaluate({**defaults, **raw} if isinstance(raw, Mapping) else raw, index, ctx)
            for index, raw in enumerate(raw_artifacts)
        ),
        key=lambda item: item["artifact_id"],
    )
    workloads = _workloads(payload, rows, findings)
    blocked = [row for row in rows if row["state"] != STATE_READY]
    findings.extend(_find(_primary_code(row), row["artifact_id"], row["state"]) for row in blocked)
    findings.sort(key=lambda item: (item["code"], item["artifact_id"] or "", item["detail"]))
    state_counts = Counter(row["state"] for row in rows)
    load_counts = Counter(row["load_status"] for row in rows)
    return {
        "schema": REPORT_SCHEMA,
        "status": "ready" if rows and not blocked and not findings else "blocked",
        "claim_boundary": CLAIM_BOUNDARY,
        "fixture": label,
        "summary": {
            "artifact_count": len(rows),
            "ready_count": len(rows) - len(blocked),
            "blocked_count": len(blocked),
            "workload_count": len(workloads),
            "unresolved_workload_refs": sum(len(item["unresolved_refs"]) for item in workloads),
            "state_counts": dict(sorted(state_counts.items())),
            "load_status_counts": dict(sorted(load_counts.items())),
        },
        "workloads": workloads,
        "artifacts": rows,
        "findings": findings,
    }


def render_json(report: Mapping[str, Any]) -> str:
    """Return byte-stable report JSON with sorted keys and a trailing newline."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_text(report: Mapping[str, Any]) -> str:
    """Return a compact deterministic text projection of the report."""
    summary = report.get("summary") or {}
    lines = [
        f"Checkpoint preservation: {str(report['status']).upper()} artifacts="
        f"{summary.get('artifact_count', 0)} blocked={summary.get('blocked_count', 0)}"
    ]
    lines.extend(
        f"{row['artifact_id']}: {row['state']} load={row['load_status']}"
        for row in report.get("artifacts", [])
    )
    lines.extend(
        f"! {item['code']}: {item['artifact_id'] or '-'}" for item in report.get("findings", [])
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="check_checkpoint_preservation",
        description="Read-only checkpoint preservation custody checker (issue #8831).",
    )
    parser.add_argument("--fixture", type=Path, required=True, help="Checkpoint fixture JSON.")
    parser.add_argument("--check", action="store_true", help="Exit 1 on blocked preservation.")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preservation checker and return a shell-friendly exit code."""
    args = _build_parser().parse_args(argv)
    report = build_report(args.fixture)
    sys.stdout.write(render_text(report) if args.format == "text" else render_json(report))
    if report["status"] == "unknown":
        return EXIT_MALFORMED
    if report["status"] == "blocked" and args.check:
        return EXIT_BLOCKED
    return EXIT_READY


if __name__ == "__main__":
    raise SystemExit(main())
