"""Standalone, offline review-session surface (SREV-28, issue #9299).

This module is intentionally a thin controller around
``analysis_workbench.review_experiment_loop`` (SREV-24).  It owns the
component-facing preview, navigation, browser/control boundary, and CLI
shape; SREV-24 remains the only owner of execution, accounting, journaling,
locking, recovery, and terminal-outcome semantics.

The default operation is a read-only preview.  A caller must explicitly pass
``autonomous=True`` (or ``start=True``) before the delegated loop can dispatch
an executor.  Preview and navigation never call an executor and never modify a
source artifact.  All output is diagnostic-only and must not be promoted to
benchmark or scientific evidence.
"""

# The controller deliberately keeps its preview/dispatch/control boundary in
# one auditable module.  Splitting it into speculative adapters would make it
# easier to accidentally bypass SREV-24's journal owner.
# ruff: noqa: C901, D102, D107, DOC201, PLR0912, PLR0913, T201

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import secrets
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from robot_sf.analysis_workbench import review_execute
from robot_sf.analysis_workbench import review_experiment_loop as loop
from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    RESULT_STATUSES,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)

COMPONENT_ID = "srev28-review-sessions"
COMPONENT_VERSION = "1.0.0"
DESCRIPTOR_SCHEMA_VERSION = COMPONENT_DESCRIPTOR_SCHEMA_VERSION
SESSION_PREVIEW_SCHEMA_VERSION = "review-session-preview.v1"
SESSION_VIEW_SCHEMA_VERSION = "review-session.v1"
CONTROL_SCHEMA_VERSION = "review-session-control.v1"
PREVIEW_FILENAME = f"{SESSION_PREVIEW_SCHEMA_VERSION}.json"
VIEW_FILENAME = f"{SESSION_VIEW_SCHEMA_VERSION}.json"
HTML_FILENAME = f"{SESSION_VIEW_SCHEMA_VERSION}.html"
JOURNAL_FILENAME = loop.SESSION_JOURNAL_FILENAME
REPORT_FILENAME = loop.LOOP_REPORT_FILENAME
ASSET_FILENAME = "review_sessions.js"
EVIDENCE_BOUNDARY = "diagnostic_only"
DEPENDENT_FAMILY_STATUS = "standalone_fixture_only"

SUPPORTED_CAPABILITIES = frozenset(
    {
        "bounded-experiment-session",
        "bounded-execution",
        "offline-browser",
        "source-admission-preview",
        "session-resume",
        "session-navigation",
    }
)
OPTIONAL_CAPABILITIES = tuple(sorted(SUPPORTED_CAPABILITIES - {"bounded-experiment-session"}))

_WRAPPER_CONFIG_KEYS = frozenset(
    {
        "admission_config",
        "required_component_version",
        "min_component_version",
        "session_token",
        "origin",
        "preservation",
        "preview",
        "loop",
    }
)
_VERSION_KEYS = ("required_component_version", "min_component_version")
_MAX_JSON_BYTES = 256 * 1024
_MAX_TOKEN_CHARS = 256
_LOOP_CONFIG_KEYS = frozenset(loop._ALLOWED_CONFIG_KEYS)

_DESCRIPTOR_DOCUMENT = {
    "schema_version": DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": ["bounded-experiment-session"],
    "optional_capabilities": list(OPTIONAL_CAPABILITIES),
    "output_types": [
        SESSION_PREVIEW_SCHEMA_VERSION,
        SESSION_VIEW_SCHEMA_VERSION,
        "experiment-loop-report.v1",
        "experiment-loop-session.v1",
    ],
}
DESCRIPTOR: ComponentDescriptor = component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)


class ReviewSessionError(ValueError):
    """Raised for an invalid review-session request or control operation."""


class ControlAuthorizationError(ReviewSessionError):
    """Raised when a browser control does not satisfy the local boundary."""


@dataclass(frozen=True, slots=True)
class SessionControl:
    """Authenticated local control envelope.

    The browser component only produces this data.  A caller-owned local
    adapter must validate it before invoking :func:`run`; no server or remote
    scheduler is started by this module.
    """

    action: str
    origin: str
    session_token: str
    payload: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "action": self.action,
            "origin": self.origin,
            "session_token": self.session_token,
            "payload": dict(self.payload or {}),
        }


def descriptor() -> dict[str, Any]:
    """Return the validated v1 component descriptor."""

    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        **{
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(DESCRIPTOR).items()
        },
    }


descriptor_document = descriptor


def _strict_loads(raw: str | bytes) -> Any:
    """Parse bounded JSON while rejecting non-finite constants."""

    if isinstance(raw, bytes) and len(raw) > _MAX_JSON_BYTES:
        raise ReviewSessionError("input exceeds review-session JSON limit")
    if isinstance(raw, str) and len(raw.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ReviewSessionError("input exceeds review-session JSON limit")

    def reject_constant(value: str) -> Any:
        raise ReviewSessionError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(raw, parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ReviewSessionError("JSON cannot be parsed safely") from error


def _read_json(path: Path) -> Any:
    try:
        return _strict_loads(path.read_bytes())
    except OSError as error:
        raise ReviewSessionError(f"cannot read JSON: {path}") from error


def _json_bytes(payload: Any) -> bytes:
    try:
        return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ReviewSessionError(f"strict JSON required: {error}") from error


def _canonical_digest(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ReviewSessionError(f"strict JSON required: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _atomic_new_file(path: Path, payload: bytes, *, overwrite: bool = False) -> str:
    """Create or replace a regular file without following destination symlinks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not overwrite):
        raise ReviewSessionError(f"output_collision: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.is_symlink() or temporary.exists():
        raise ReviewSessionError(f"output_collision: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor_fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor_fd, "wb") as stream:
            descriptor_fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise ReviewSessionError(f"cannot write output: {error}") from error
    finally:
        if descriptor_fd >= 0:
            os.close(descriptor_fd)
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _unsafe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return True
    path = Path(value)
    windows = PureWindowsPath(value)
    return (
        path.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in path.parts
        or ".." in windows.parts
    )


def _resolve_output(base: Path, output_directory: str, *, create: bool) -> Path:
    if _unsafe_relative(output_directory):
        raise ReviewSessionError("invalid_output_path: relative contained path required")
    try:
        root = base.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReviewSessionError("invalid_output_path: base directory is unavailable") from error
    if not root.is_dir() or root.is_symlink():
        raise ReviewSessionError("invalid_output_path: base directory is not regular")
    output = root.joinpath(*Path(output_directory).parts)
    for index in range(1, len(Path(output_directory).parts) + 1):
        part = root.joinpath(*Path(output_directory).parts[:index])
        if part.is_symlink():
            raise ReviewSessionError("invalid_output_path: symlinked output component")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise ReviewSessionError("invalid_output_path: output is not a directory")
    if create:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _version_compatible(config: Mapping[str, Any]) -> tuple[bool, str]:
    for key in _VERSION_KEYS:
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            return False, f"incompatible_component_version: {key} must be a version string"
        requested_major = value.split(".", 1)[0].lstrip("v")
        current_major = COMPONENT_VERSION.split(".", 1)[0].lstrip("v")
        if requested_major != current_major:
            return (
                False,
                f"incompatible_component_version: requested {value}, implemented {COMPONENT_VERSION}",
            )
    return True, ""


def _normalise_request(request: ComponentRequest | Mapping[str, Any]) -> ComponentRequest:
    if isinstance(request, ComponentRequest):
        return request
    try:
        return component_request_from_dict(request)
    except (ReviewContractsValidationError, TypeError, ValueError) as error:
        raise ReviewSessionError(f"invalid_input: {error}") from error


def _loop_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the SREV-24 config while accepting a nested ``loop`` block."""

    nested = config.get("loop")
    if nested is not None and not isinstance(nested, Mapping):
        raise ReviewSessionError("invalid_config: loop must be a mapping")
    source = (
        dict(nested)
        if nested is not None
        else {key: value for key, value in config.items() if key not in _WRAPPER_CONFIG_KEYS}
    )
    unknown = sorted(str(key) for key in source if key not in _LOOP_CONFIG_KEYS)
    if unknown:
        raise ReviewSessionError("invalid_config: unknown loop keys: " + ", ".join(unknown))
    return source


def _loop_request(
    request: ComponentRequest, *, output_directory: str | None = None
) -> ComponentRequest:
    config = _loop_config(request.config)
    return replace(
        request,
        component_id=loop.COMPONENT_ID,
        output_directory=output_directory or request.output_directory,
        config=config,
    )


def _safe_request_identity(request: ComponentRequest) -> dict[str, Any]:
    return loop._request_identity(request)


def _source_proof_for_child(
    request: ComponentRequest,
    child_request: ComponentRequest,
    recipe: Mapping[str, Any],
    proof: Mapping[str, Any] | None,
    *,
    base: Path,
) -> dict[str, Any] | None:
    """Validate a caller proof before rebinding it to the delegated child.

    The rebind changes only the component identity digest.  Source bytes,
    source root, recipe digest, and diagnostic boundary are still checked by
    SREV-24's measured proof validator before dispatch.
    """

    if proof is None:
        return None
    validated = loop._validate_injected_source_proof(request, recipe, proof, base=base)
    if validated is None:
        return None
    rebound = dict(validated)
    rebound["request_digest"] = loop._canonical_digest(_safe_request_identity(child_request))
    return rebound


def _recipe_budget_preview(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        validated = loop._validate_input(
            _loop_request(
                ComponentRequest(
                    request_id="preview",
                    component_id=loop.COMPONENT_ID,
                    sources=(),
                    output_directory="preview",
                    config=dict(config),
                )
            ),
            autonomous=False,
            read_only=True,
            resume=False,
        )
        candidates = loop._candidate_order(validated.recipe)[: validated.budget.max_candidates]
        budget = validated.budget.to_dict()
        budget.update(
            {
                "executions_consumed": 0,
                "reserved_executions": 0,
                "remaining_executions": validated.budget.max_executions,
                "elapsed_s": 0.0,
                "remaining_elapsed_s": validated.budget.wall_timeout_s,
            }
        )
        return budget, [
            {
                "intervention_id": item["intervention_id"],
                "priority": item["priority"],
                "factor": item.get("factor", ""),
                "state": "pending",
            }
            for item in candidates
        ]
    except (ReviewSessionError, loop.ExperimentLoopError) as error:
        raise ReviewSessionError(str(error)) from error


def _preservation_preview(
    config: Mapping[str, Any], admission_config: Mapping[str, Any] | None
) -> dict[str, Any]:
    admission = dict(admission_config or {})
    configured = config.get("preservation")
    if configured is not None and not isinstance(configured, Mapping):
        return {"status": "invalid", "reason": "preservation must be a mapping"}
    if isinstance(configured, Mapping):
        admission = {**admission, **dict(configured)}
    destination = admission.get("preservation_destination") or admission.get("destination")
    receipt = admission.get("preservation_receipt_reference") or admission.get("receipt_reference")
    if destination and receipt:
        return {
            "status": "configured",
            "destination": str(destination),
            "receipt_reference": str(receipt),
        }
    return {
        "status": "required",
        "reason": "preservation destination and receipt are launcher-owned",
    }


def _admission_preview(
    request: ComponentRequest,
    *,
    config: Mapping[str, Any],
    admission_config: Mapping[str, Any] | review_execute.ExecutorAdmissionConfig | None,
    executor: Any | None,
    source_admission: Mapping[str, Any] | None,
    base: Path,
) -> dict[str, Any]:
    if executor is not None:
        proof = source_admission or config.get("source_admission")
        if isinstance(proof, Mapping):
            return {
                "status": "provided",
                "reason": "injected executor proof is checked again before dispatch",
                "evidence_boundary": EVIDENCE_BOUNDARY,
            }
        return {
            "status": "required",
            "reason": "injected executor requires measured source_admission proof",
        }
    if admission_config is None:
        return {
            "status": "required",
            "reason": "explicit launcher admission configuration is required for native start",
        }
    try:
        validated = loop._validate_input(
            _loop_request(request), autonomous=True, read_only=False, resume=False
        )
        child = _loop_request(request)
        document, _normalized, error = loop._preflight_native_admission(
            child,
            recipe=validated.recipe,
            executor_config=loop._native_executor_config(validated),
            admission_config=admission_config,
        )
        if error or document is None:
            return {"status": "unavailable", "reason": error or "source admission unavailable"}
        return {
            "status": "ready",
            "receipt_id": document.get("receipt_id", ""),
            "source": document.get("source", {}),
            "evidence_boundary": document.get("evidence_boundary", EVIDENCE_BOUNDARY),
        }
    except (ReviewSessionError, loop.ExperimentLoopError, OSError, ValueError, TypeError) as error:
        return {"status": "unavailable", "reason": f"source_admission: {error}"}


def _output_state(base: Path, request: ComponentRequest) -> dict[str, Any]:
    try:
        output = _resolve_output(base, request.output_directory, create=False)
    except ReviewSessionError as error:
        return {"status": "invalid", "reason": str(error), "path": request.output_directory}
    if not output.exists():
        return {"status": "available", "collision": False, "path": request.output_directory}
    journal = output / JOURNAL_FILENAME
    return {
        "status": "resume_available" if journal.is_file() else "collision",
        "collision": not journal.is_file(),
        "path": request.output_directory,
        "journal": journal.is_file(),
    }


def _read_journal(base: Path, request: ComponentRequest) -> dict[str, Any] | None:
    state = _output_state(base, request)
    if state.get("status") not in {"resume_available", "collision"}:
        return None
    try:
        output = _resolve_output(base, request.output_directory, create=False)
        payload = _read_json(output / JOURNAL_FILENAME)
    except (ReviewSessionError, OSError, ValueError, TypeError) as error:
        return {"_read_error": str(error)}
    return payload if isinstance(payload, dict) else {"_read_error": "journal is not an object"}


def _journal_progress(journal: Mapping[str, Any]) -> dict[str, Any]:
    budget = dict(journal.get("budget", {})) if isinstance(journal.get("budget"), Mapping) else {}
    consumed = int(journal.get("executions_consumed", budget.get("executions_consumed", 0)) or 0)
    reserved = int(journal.get("reserved_executions", budget.get("reserved_executions", 0)) or 0)
    maximum = int(budget.get("max_executions", 0) or 0)
    elapsed = float(journal.get("elapsed_s", budget.get("elapsed_s", 0.0)) or 0.0)
    wall = float(budget.get("wall_timeout_s", 0.0) or 0.0)
    candidate_rows = journal.get("candidates", {})
    candidates = list(candidate_rows.values()) if isinstance(candidate_rows, Mapping) else []
    outcomes = journal.get("outcomes", [])
    return {
        "status": journal.get("status", "running"),
        "stop_reason": journal.get("stop_reason", ""),
        "budget": {
            **budget,
            "executions_consumed": consumed,
            "reserved_executions": reserved,
            "remaining_executions": max(0, maximum - consumed - reserved),
            "elapsed_s": elapsed,
            "remaining_elapsed_s": max(0.0, wall - elapsed),
        },
        "candidates": [dict(item) for item in candidates if isinstance(item, Mapping)],
        "outcomes": [dict(item) for item in outcomes if isinstance(item, Mapping)],
        "operations": [
            dict(item) for item in journal.get("operations", []) if isinstance(item, Mapping)
        ],
        "source_admission": dict(journal.get("source_admission", {})),
        "evidence_boundary": journal.get("evidence_boundary", EVIDENCE_BOUNDARY),
        "scientific_claim_allowed": journal.get("scientific_claim_allowed", False),
        "dependent_family_status": journal.get("dependent_family_status", DEPENDENT_FAMILY_STATUS),
    }


def progress(
    request: ComponentRequest | Mapping[str, Any], *, base: Path | None = None
) -> dict[str, Any]:
    """Read durable progress without acquiring a session lock or dispatching."""

    normalized = _normalise_request(request)
    root = base if base is not None else Path.cwd()
    journal = _read_journal(root, normalized)
    if journal is None:
        budget, candidates = _recipe_budget_preview(normalized.config)
        return {
            "schema_version": SESSION_PREVIEW_SCHEMA_VERSION,
            "status": "not_started",
            "budget": budget,
            "candidates": candidates,
            "source_admission": {"status": "not_checked"},
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "scientific_claim_allowed": False,
        }
    if "_read_error" in journal:
        return {"status": "failed", "reason": journal["_read_error"]}
    return _journal_progress(journal)


def result_navigation(
    request: ComponentRequest | Mapping[str, Any],
    *,
    base: Path | None = None,
    index: int = 0,
) -> dict[str, Any]:
    """Return deterministic result navigation over retained candidate outcomes."""

    normalized = _normalise_request(request)
    root = base if base is not None else Path.cwd()
    try:
        output = _resolve_output(root, normalized.output_directory, create=False)
        report = _read_json(output / REPORT_FILENAME)
    except (ReviewSessionError, OSError, ValueError, TypeError) as error:
        return {"status": "unavailable", "reason": f"result_unavailable: {error}"}
    if not isinstance(report, Mapping):
        return {"status": "failed", "reason": "result report is not an object"}
    outcomes = [dict(item) for item in report.get("outcomes", []) if isinstance(item, Mapping)]
    if not outcomes:
        return {
            "status": report.get("status", "unavailable"),
            "index": None,
            "total": 0,
            "items": [],
            "current": None,
        }
    if not isinstance(index, int) or isinstance(index, bool):
        return {"status": "failed", "reason": "result index must be an integer"}
    bounded = max(0, min(index, len(outcomes) - 1))
    return {
        "status": report.get("status", "unavailable"),
        "index": bounded,
        "total": len(outcomes),
        "items": outcomes,
        "current": outcomes[bounded],
    }


def navigate_result(
    request: ComponentRequest | Mapping[str, Any],
    *,
    base: Path | None = None,
    index: int = 0,
    direction: int = 0,
) -> dict[str, Any]:
    """Navigate retained results with clamped previous/next semantics."""

    target = index + direction
    return result_navigation(request, base=base, index=target)


def preview(
    request: ComponentRequest | Mapping[str, Any],
    *,
    base: Path | None = None,
    admission_config: Mapping[str, Any] | review_execute.ExecutorAdmissionConfig | None = None,
    executor: Any | None = None,
    source_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded, no-dispatch session preview."""

    normalized = _normalise_request(request)
    if normalized.component_id != COMPONENT_ID:
        raise ReviewSessionError(f"unsupported component: {normalized.component_id}")
    config = dict(normalized.config)
    compatible, version_reason = _version_compatible(config)
    if not compatible:
        return {
            "schema_version": SESSION_PREVIEW_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": version_reason,
            "evidence_boundary": EVIDENCE_BOUNDARY,
        }
    missing = [cap for cap in normalized.required_capabilities if cap not in SUPPORTED_CAPABILITIES]
    if missing:
        return {
            "schema_version": SESSION_PREVIEW_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "missing_capabilities: " + ", ".join(sorted(missing)),
            "missing_capabilities": sorted(missing),
            "evidence_boundary": EVIDENCE_BOUNDARY,
        }
    effective_config = _loop_config(config)
    budget, candidates = _recipe_budget_preview(effective_config)
    root = base if base is not None else Path.cwd()
    journal = _read_journal(root, normalized)
    if journal is not None and "_read_error" not in journal:
        persisted = _journal_progress(journal)
        budget = dict(persisted["budget"])
        candidates = list(persisted["candidates"])
        state = str(persisted.get("status", "running"))
        source_state = dict(persisted.get("source_admission", {}))
        stop_reason = str(persisted.get("stop_reason", ""))
    else:
        state = "not_started"
        source_state = _admission_preview(
            normalized,
            config=effective_config,
            admission_config=admission_config,
            executor=executor,
            source_admission=source_admission,
            base=root,
        )
        stop_reason = ""
    preservation = _preservation_preview(
        config, admission_config if isinstance(admission_config, Mapping) else None
    )
    output_state = _output_state(root, normalized)
    return {
        "schema_version": SESSION_PREVIEW_SCHEMA_VERSION,
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": normalized.request_id,
        "session_id": str(effective_config.get("session_id", "")),
        "status": state,
        "stop_reason": stop_reason,
        "authorization": {
            "explicit_start_required": True,
            "granted": False,
            "read_only": True,
        },
        "budget": budget,
        "candidates": candidates,
        "source_admission": source_state,
        "preservation": preservation,
        "output": output_state,
        "navigation": result_navigation(normalized, base=root),
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
    }


def _provenance(request: ComponentRequest) -> dict[str, Any]:
    return {
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_digest": _canonical_digest(_safe_request_identity(request)),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "benchmark_success": False,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
    }


def _result(
    request: ComponentRequest,
    status: str,
    *,
    reason: str = "",
    artifacts: Sequence[Mapping[str, Any]] = (),
    diagnostics: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> ComponentResult:
    if status not in RESULT_STATUSES:
        status = "failed"
        reason = reason or "invalid result status"
    payload = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "component_id": COMPONENT_ID,
        "status": status,
        "reason": reason,
        "artifacts": [dict(item) for item in artifacts] if status == "complete" else [],
        "diagnostics": [dict(item) for item in diagnostics],
        "provenance": dict(provenance or _provenance(request)),
    }
    try:
        return component_result_from_dict(payload)
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=COMPONENT_ID,
            status="failed",
            reason=f"internal_result_invalid: {'; '.join(error.errors)}",
        )


def _preview_result(
    request: ComponentRequest,
    *,
    base: Path,
    admission_config: Mapping[str, Any] | review_execute.ExecutorAdmissionConfig | None,
    executor: Any | None,
    source_admission: Mapping[str, Any] | None,
    write: bool,
) -> ComponentResult:
    try:
        document = preview(
            request,
            base=base,
            admission_config=admission_config,
            executor=executor,
            source_admission=source_admission,
        )
    except (ReviewSessionError, loop.ExperimentLoopError, OSError, TypeError, ValueError) as error:
        return _result(request, "failed", reason=str(error))
    artifacts: list[dict[str, Any]] = []
    if write:
        try:
            output = _resolve_output(base, request.output_directory, create=True)
            digest = _atomic_new_file(output / PREVIEW_FILENAME, _json_bytes(document))
            artifacts.append(
                {
                    "artifact_id": PREVIEW_FILENAME,
                    "uri": str(Path(request.output_directory) / PREVIEW_FILENAME),
                    "sha256": digest,
                }
            )
        except (ReviewSessionError, OSError) as error:
            return _result(request, "failed", reason=str(error))
    status = "complete" if document.get("status") != "unavailable" else "unavailable"
    return _result(
        request,
        status,
        reason=str(document.get("reason", "")),
        artifacts=artifacts,
        diagnostics=(document,),
    )


def _write_browser_view(
    output: Path, request: ComponentRequest, document: Mapping[str, Any]
) -> list[dict[str, Any]]:
    asset_path = (
        Path(__file__).with_name("web_assets") / "components" / "review_sessions" / ASSET_FILENAME
    )
    try:
        javascript = asset_path.read_bytes()
    except OSError as error:
        raise ReviewSessionError("offline browser asset is unavailable") from error
    asset_output = output / "components" / "review_sessions" / ASSET_FILENAME
    asset_digest = _atomic_new_file(asset_output, javascript, overwrite=True)
    document_bytes = _json_bytes(document)
    view_digest = _atomic_new_file(output / VIEW_FILENAME, document_bytes, overwrite=True)
    escaped = html.escape(document_bytes.decode("utf-8"), quote=False)
    markup = (
        '<!doctype html>\n<html><head><meta charset="utf-8"><title>Review session</title>'
        '</head><body><main id="review-session-root" data-read-only="true"></main>'
        f'<script type="application/json" id="review-session-data">{escaped}</script>\n'
        f'<script type="module" src="components/review_sessions/{ASSET_FILENAME}"></script>\n'
        "</body></html>\n"
    ).encode()
    html_digest = _atomic_new_file(output / HTML_FILENAME, markup, overwrite=True)
    prefix = Path(request.output_directory)
    return [
        {"artifact_id": VIEW_FILENAME, "uri": str(prefix / VIEW_FILENAME), "sha256": view_digest},
        {"artifact_id": HTML_FILENAME, "uri": str(prefix / HTML_FILENAME), "sha256": html_digest},
        {
            "artifact_id": f"components/review_sessions/{ASSET_FILENAME}",
            "uri": str(prefix / "components" / "review_sessions" / ASSET_FILENAME),
            "sha256": asset_digest,
        },
    ]


def run(
    request: ComponentRequest | Mapping[str, Any],
    *,
    base: Path | None = None,
    autonomous: bool = False,
    start: bool = False,
    resume: bool = False,
    read_only: bool = False,
    admission_config: Mapping[str, Any] | review_execute.ExecutorAdmissionConfig | None = None,
    executor: Any | None = None,
    source_admission: Mapping[str, Any] | None = None,
    cancel: Callable[[], bool] | Any | None = None,
) -> ComponentResult:
    """Run one SREV-28 session, delegating execution and resume to SREV-24."""

    try:
        normalized = _normalise_request(request)
    except ReviewSessionError as error:
        return ComponentResult("unknown", COMPONENT_ID, "failed", reason=str(error))
    if normalized.component_id != COMPONENT_ID:
        return _result(
            normalized, "unavailable", reason=f"unsupported component: {normalized.component_id}"
        )
    compatible, version_reason = _version_compatible(normalized.config)
    if not compatible:
        return _result(normalized, "unavailable", reason=version_reason)
    missing = [cap for cap in normalized.required_capabilities if cap not in SUPPORTED_CAPABILITIES]
    if missing:
        return _result(
            normalized,
            "unavailable",
            reason="missing_capabilities: " + ", ".join(sorted(missing)),
        )
    root = base if base is not None else Path.cwd()
    if read_only or bool(normalized.config.get("read_only", False)):
        try:
            preview_document = preview(
                normalized,
                base=root,
                admission_config=admission_config,
                executor=executor,
                source_admission=source_admission,
            )
        except (
            ReviewSessionError,
            loop.ExperimentLoopError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            return _result(normalized, "failed", reason=str(error))
        return _result(
            normalized,
            "unavailable",
            reason="read_only_never_executes",
            diagnostics=(preview_document,),
        )
    if not (autonomous or start or bool(normalized.config.get("autonomous", False))):
        try:
            preview_document = preview(
                normalized,
                base=root,
                admission_config=admission_config,
                executor=executor,
                source_admission=source_admission,
            )
        except (
            ReviewSessionError,
            loop.ExperimentLoopError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            return _result(normalized, "failed", reason=str(error))
        return _result(
            normalized,
            "unavailable",
            reason="autonomous_start_authorization_required",
            diagnostics=(preview_document,),
        )
    try:
        child_request = _loop_request(normalized)
        validated = loop._validate_input(
            child_request, autonomous=True, read_only=False, resume=resume
        )
        child_proof = _source_proof_for_child(
            normalized,
            child_request,
            validated.recipe,
            source_admission or normalized.config.get("source_admission"),
            base=root,
        )
        child_result = loop.run(
            child_request,
            base=root,
            resume=resume,
            autonomous=True,
            admission_config=admission_config,
            executor=executor,
            source_admission=child_proof,
            cancel=cancel,
        )
    except (ReviewSessionError, loop.ExperimentLoopError, OSError, TypeError, ValueError) as error:
        return _result(normalized, "failed", reason=str(error))
    provenance = {
        **dict(child_result.provenance),
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "executor_component_id": loop.COMPONENT_ID,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "benchmark_success": False,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
    }
    if child_result.status != "complete":
        return _result(
            normalized,
            child_result.status,
            reason=child_result.reason,
            diagnostics=child_result.diagnostics,
            provenance=provenance,
        )
    try:
        output = _resolve_output(root, normalized.output_directory, create=False)
        document = {
            "schema_version": SESSION_VIEW_SCHEMA_VERSION,
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "request_id": normalized.request_id,
            "status": child_result.status,
            "progress": _journal_progress(_read_journal(root, normalized) or {}),
            "navigation": result_navigation(normalized, base=root),
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "scientific_claim_allowed": False,
            "provenance": provenance,
        }
        wrapper_artifacts = _write_browser_view(output, normalized, document)
    except (ReviewSessionError, OSError, TypeError, ValueError) as error:
        return _result(
            normalized, "failed", reason=f"browser_view_failed: {error}", provenance=provenance
        )
    return _result(
        normalized,
        child_result.status,
        reason=child_result.reason,
        artifacts=(*child_result.artifacts, *wrapper_artifacts),
        diagnostics=child_result.diagnostics,
        provenance=provenance,
    )


class ReviewSession:
    """Stateful convenience controller with explicit lifecycle operations."""

    def __init__(
        self,
        request: ComponentRequest | Mapping[str, Any],
        *,
        base: Path | None = None,
        admission_config: Mapping[str, Any] | review_execute.ExecutorAdmissionConfig | None = None,
        executor: Any | None = None,
        source_admission: Mapping[str, Any] | None = None,
        cancel: Callable[[], bool] | Any | None = None,
    ) -> None:
        self.request = _normalise_request(request)
        self.base = base if base is not None else Path.cwd()
        self.admission_config = admission_config
        self.executor = executor
        self.source_admission = source_admission
        self.cancel = cancel
        self._stop_requested = threading.Event()
        self._active_lock = threading.Lock()
        self._active = False
        self._active_done = threading.Event()
        self._active_done.set()
        self._last_result: ComponentResult | None = None

    def _cancel_callback(self) -> bool:
        if self._stop_requested.is_set():
            return True
        if callable(self.cancel):
            return bool(self.cancel())
        return bool(self.cancel) if self.cancel is not None else False

    def _execute(self, *, resume: bool) -> ComponentResult:
        with self._active_lock:
            self._active = True
            self._active_done.clear()
        try:
            result = run(
                self.request,
                base=self.base,
                autonomous=True,
                resume=resume,
                admission_config=self.admission_config,
                executor=self.executor,
                source_admission=self.source_admission,
                cancel=self._cancel_callback,
            )
            self._last_result = result
            return result
        finally:
            with self._active_lock:
                self._active = False
                self._active_done.set()

    def preview(self) -> dict[str, Any]:
        return preview(
            self.request,
            base=self.base,
            admission_config=self.admission_config,
            executor=self.executor,
            source_admission=self.source_admission,
        )

    def start(self) -> ComponentResult:
        self._stop_requested.clear()
        return self._execute(resume=False)

    def resume(self) -> ComponentResult:
        self._stop_requested.clear()
        return self._execute(resume=True)

    def stop(self) -> ComponentResult:
        """Request cancellation through the existing SREV-24 journal owner."""

        self._stop_requested.set()
        with self._active_lock:
            active = self._active
        if active:
            self._active_done.wait(timeout=5.0)
            if self._last_result is not None:
                return self._last_result
            return _result(self.request, "cancelled", reason="cancellation_requested")
        return run(
            self.request,
            base=self.base,
            autonomous=True,
            resume=True,
            admission_config=self.admission_config,
            executor=self.executor,
            source_admission=self.source_admission,
            cancel=self._cancel_callback,
        )

    def progress(self) -> dict[str, Any]:
        return progress(self.request, base=self.base)

    def result(self, index: int = 0) -> dict[str, Any]:
        return result_navigation(self.request, base=self.base, index=index)

    def navigate(self, index: int = 0, *, direction: int = 0) -> dict[str, Any]:
        return navigate_result(self.request, base=self.base, index=index, direction=direction)


ExperimentSession = ReviewSession
SessionController = ReviewSession


def validate_control(
    control: SessionControl | Mapping[str, Any],
    *,
    expected_origin: str,
    expected_session_token: str,
) -> SessionControl:
    """Validate a loopback, origin-bound, token-authenticated control envelope."""

    if isinstance(control, SessionControl):
        current = control
    elif isinstance(control, Mapping):
        if control.get("schema_version", CONTROL_SCHEMA_VERSION) != CONTROL_SCHEMA_VERSION:
            raise ControlAuthorizationError("unsupported control schema")
        current = SessionControl(
            action=str(control.get("action", "")),
            origin=str(control.get("origin", "")),
            session_token=str(control.get("session_token", "")),
            payload=control.get("payload") if isinstance(control.get("payload"), Mapping) else {},
        )
    else:
        raise ControlAuthorizationError("control must be a mapping")
    if current.action not in {"start", "stop", "resume", "progress", "result"}:
        raise ControlAuthorizationError("unsupported control action")
    if len(current.session_token) > _MAX_TOKEN_CHARS or not current.session_token:
        raise ControlAuthorizationError("session token is required")
    if not secrets.compare_digest(current.session_token, expected_session_token):
        raise ControlAuthorizationError("session token mismatch")
    if not is_loopback_origin(current.origin):
        raise ControlAuthorizationError("control origin must be loopback")
    if current.origin != expected_origin:
        raise ControlAuthorizationError("control origin mismatch")
    return current


def is_loopback_origin(origin: str) -> bool:
    """Return whether an origin is an explicit HTTP(S) loopback origin."""

    if not isinstance(origin, str) or len(origin) > 512:
        return False
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname
        in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and not parsed.username
        and not parsed.password
    )


def make_control_handler(
    session: ReviewSession,
    *,
    origin: str,
    session_token: str,
) -> Callable[[SessionControl | Mapping[str, Any]], Any]:
    """Build an authenticated callback for an already-running local adapter."""

    def handle(control: SessionControl | Mapping[str, Any]) -> Any:
        authorized = validate_control(
            control,
            expected_origin=origin,
            expected_session_token=session_token,
        )
        if authorized.action == "start":
            return session.start()
        if authorized.action == "stop":
            return session.stop()
        if authorized.action == "resume":
            return session.resume()
        if authorized.action == "progress":
            return session.progress()
        return session.result(int((authorized.payload or {}).get("index", 0)))

    return handle


def _result_document(result: ComponentResult) -> dict[str, Any]:
    return {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **asdict(result)}


def _cli_failure(reason: str, payload: Any = None) -> int:
    request_id = payload.get("request_id", "unknown") if isinstance(payload, Mapping) else "unknown"
    result = ComponentResult(
        request_id=str(request_id), component_id=COMPONENT_ID, status="failed", reason=reason
    )
    print(json.dumps(_result_document(result), indent=2, sort_keys=True))
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or explicitly start a bounded review session."
    )
    parser.add_argument("--input", required=True, help="component-request.v1 JSON")
    parser.add_argument("--config", default=None, help="optional config JSON merged into request")
    parser.add_argument("--output", required=True, help="relative output directory")
    parser.add_argument("--base", default=None, help="base directory for output")
    parser.add_argument("--admission-config", default=None, help="launcher-owned admission JSON")
    parser.add_argument("--autonomous", action="store_true", help="authorize execution")
    parser.add_argument("--start", action="store_true", help="explicitly start execution")
    parser.add_argument("--resume", action="store_true", help="resume an existing session")
    parser.add_argument("--stop", action="store_true", help="cancel an existing session")
    parser.add_argument("--read-only", action="store_true", help="inspect without execution")
    parser.add_argument("--result-index", type=int, default=0, help="result to show")
    parser.add_argument("--origin", default=None, help="loopback control origin")
    parser.add_argument("--session-token", default=None, help="local control session token")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the preview or explicit session CLI; return zero only on success."""

    args = _build_parser().parse_args(argv)
    try:
        payload = _read_json(Path(args.input))
    except ReviewSessionError as error:
        return _cli_failure(f"invalid_input: {error}")
    if not isinstance(payload, dict):
        return _cli_failure("invalid_input: request must be a JSON object", payload)
    if args.config is not None:
        try:
            override = _read_json(Path(args.config))
        except ReviewSessionError as error:
            return _cli_failure(f"invalid_input: config {error}", payload)
        if not isinstance(override, dict):
            return _cli_failure("invalid_input: config must be an object", payload)
        current = payload.get("config", {})
        if not isinstance(current, dict):
            return _cli_failure("invalid_input: request config must be an object", payload)
        payload = {**payload, "config": {**current, **override}}
    payload = {**payload, "component_id": COMPONENT_ID, "output_directory": args.output}
    try:
        request = component_request_from_dict(payload, source=args.input)
    except (ReviewContractsValidationError, TypeError, ValueError, RecursionError) as error:
        return _cli_failure(
            f"invalid_input: request does not satisfy component-request.v1: {error}", payload
        )
    root = Path(args.base) if args.base is not None else Path.cwd()
    admission: Any = None
    if args.admission_config:
        try:
            admission = _read_json(Path(args.admission_config))
        except ReviewSessionError as error:
            return _cli_failure(f"invalid_input: admission config {error}", payload)
    if args.stop and not (args.origin and args.session_token):
        return _cli_failure(
            "control_authorization_required: --stop requires --origin and --session-token", payload
        )
    if args.stop and (not is_loopback_origin(args.origin) or not args.session_token):
        return _cli_failure(
            "control_authorization_failed: loopback origin and session token required", payload
        )
    try:
        if args.stop:
            result = run(
                request,
                base=root,
                autonomous=True,
                resume=True,
                read_only=False,
                admission_config=admission,
                cancel=lambda: True,
            )
        elif args.autonomous or args.start or args.resume:
            result = run(
                request,
                base=root,
                autonomous=args.autonomous or args.start or args.resume,
                resume=args.resume,
                read_only=args.read_only,
                admission_config=admission,
            )
        elif args.read_only:
            result = run(request, base=root, read_only=True, admission_config=admission)
        else:
            result = _preview_result(
                request,
                base=root,
                admission_config=admission,
                executor=None,
                source_admission=None,
                write=True,
            )
    except (ReviewSessionError, OSError, TypeError, ValueError) as error:
        return _cli_failure(str(error), payload)
    print(json.dumps(_result_document(result), indent=2, sort_keys=True))
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
