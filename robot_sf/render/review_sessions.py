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
import math
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
        "campaign_id",
        "context_revision",
        "episode_id",
        "required_component_version",
        "min_component_version",
        "session_token",
        "origin",
        "preservation",
        "preview",
        "selection_revision",
        "session_context",
        "source_revision",
        "loop",
    }
)
_CONTEXT_KEYS = (
    "campaign_id",
    "episode_id",
    "source_revision",
    "selection_revision",
    "context_revision",
)
_JOURNAL_STATUSES = frozenset(
    {"running", "complete", "partial", "failed", "unavailable", "cancelled"}
)
_SHA256_LENGTH = 64
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
    session_id: str = ""
    context_revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "action": self.action,
            "origin": self.origin,
            "session_token": self.session_token,
            "payload": dict(self.payload or {}),
            "session_id": self.session_id,
            "context_revision": self.context_revision,
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


def _ensure_directory_chain(path: Path) -> None:
    """Create a directory chain without following pre-existing symlinks."""

    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    if current.is_symlink() or not current.is_dir():
        raise ReviewSessionError(f"output_collision: parent is not a regular directory: {current}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        if directory.is_symlink() or not directory.is_dir():
            raise ReviewSessionError(
                f"output_collision: parent is not a regular directory: {directory}"
            )


def _atomic_new_file(path: Path, payload: bytes, *, overwrite: bool = False) -> str:
    """Create or replace a regular file without following destination symlinks."""

    _ensure_directory_chain(path.parent)
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


def _context_identity(request: ComponentRequest) -> dict[str, Any]:
    """Return the caller-selected campaign/episode context identity.

    Context values are wrapper metadata rather than SREV-24 executor config.
    They are nevertheless part of the session identity so a browser or API
    reader cannot reuse durable state after changing its selected episode or
    source revision.
    """

    raw = request.config.get("session_context")
    if raw is not None and not isinstance(raw, Mapping):
        raise ReviewSessionError("invalid_config: session_context must be a mapping")
    context = dict(raw) if isinstance(raw, Mapping) else {}
    for key in _CONTEXT_KEYS:
        if key in request.config:
            if key in context and context[key] != request.config[key]:
                raise ReviewSessionError(f"invalid_config: conflicting context field {key}")
            context[key] = request.config[key]
    unknown = sorted(str(key) for key in context if key not in _CONTEXT_KEYS)
    if unknown:
        raise ReviewSessionError(
            "invalid_config: unknown session_context keys: " + ", ".join(unknown)
        )
    for key, value in context.items():
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise ReviewSessionError(f"invalid_config: context field {key} must be text or integer")
        if isinstance(value, str) and (not value.strip() or len(value) > 256 or "\x00" in value):
            raise ReviewSessionError(f"invalid_config: context field {key} is invalid")
    return {key: context[key] for key in _CONTEXT_KEYS if key in context}


def _context_document(request: ComponentRequest, validated: Any) -> dict[str, Any]:
    """Build the versioned selection identity exposed to API/browser readers."""

    child = _loop_request(request)
    context = _context_identity(request)
    if not context.get("source_revision") and len(request.sources) == 1:
        source_commit = request.sources[0].source_commit
        if source_commit:
            context["source_revision"] = source_commit
    return {
        **context,
        "session_id": str(validated.session_id),
        "request_id": request.request_id,
        "request_digest": loop._canonical_digest(loop._request_identity(child)),
        "recipe_id": str(validated.recipe.get("recipe_id", "")),
        "recipe_digest": loop.experiment_recipe_canonical_digest(validated.recipe),
    }


def _control_context(request: ComponentRequest) -> dict[str, Any]:
    try:
        validated = loop._validate_input(
            _loop_request(request), autonomous=True, read_only=False, resume=False
        )
    except (ReviewSessionError, loop.ExperimentLoopError, OSError, TypeError, ValueError) as error:
        raise ReviewSessionError(f"invalid session context: {error}") from error
    return _context_document(request, validated)


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
    context = _context_identity(request)
    binding: dict[str, Any] = {}
    if context:
        binding["context"] = context
    configured_token = request.config.get("session_token")
    if configured_token is not None:
        if (
            not isinstance(configured_token, str)
            or not configured_token
            or len(configured_token) > _MAX_TOKEN_CHARS
        ):
            raise ReviewSessionError("invalid_config: session_token is invalid")
        binding["control"] = {
            "origin": request.config.get("origin", ""),
            "session_token": configured_token,
        }
    if binding:
        session_prefix = str(config.get("session_id") or request.request_id)
        binding_digest = _canonical_digest(binding)[:16]
        config["session_id"] = f"{session_prefix}:{binding_digest}"
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
        try:
            child = _loop_request(request)
            validated = loop._validate_input(child, autonomous=True, read_only=False, resume=False)
            admitted = _source_proof_for_child(
                request,
                child,
                validated.recipe,
                proof if isinstance(proof, Mapping) else None,
                base=base,
            )
        except (ReviewSessionError, loop.ExperimentLoopError, OSError, TypeError, ValueError):
            admitted = None
        if admitted is not None:
            return {
                "status": "provided",
                "reason": "measured proof is valid and checked again before dispatch",
                "evidence_boundary": EVIDENCE_BOUNDARY,
            }
        return {
            "status": "unavailable",
            "reason": "injected executor proof is missing or failed source validation",
            "evidence_boundary": EVIDENCE_BOUNDARY,
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


def _safe_source_relative(uri: Any) -> Path | None:
    if not isinstance(uri, str) or not uri or "\x00" in uri or "\\" in uri:
        return None
    relative = Path(uri)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative == Path(".")
        or ".." in relative.parts
    ):
        return None
    return relative


def _source_digest(root: Path, relative: Path) -> str | None:
    try:
        if root.is_symlink():
            return None
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_dir():
            return None
        source = resolved_root.joinpath(*relative.parts)
        resolved_source = source.resolve(strict=True)
        resolved_source.relative_to(resolved_root)
        if source.is_symlink() or not resolved_source.is_file():
            return None
        return hashlib.sha256(resolved_source.read_bytes()).hexdigest()
    except (OSError, RuntimeError, ValueError):
        return None


def _source_integrity_error(
    base: Path,
    request: ComponentRequest,
    recipe: Mapping[str, Any],
    source_admission: Mapping[str, Any],
) -> str | None:
    """Re-check durable source identity before exposing a persisted session."""

    recipe_identity = recipe.get("source_identity")
    recipe_ref = recipe_identity.get("source_ref") if isinstance(recipe_identity, Mapping) else None
    if recipe_ref is not None and not isinstance(recipe_ref, Mapping):
        return "journal_source_identity_mismatch: recipe source_ref is malformed"
    admission_source = source_admission.get("source")
    if admission_source is not None and not isinstance(admission_source, Mapping):
        return "journal_source_identity_mismatch: admission source is malformed"
    roots: list[Path] = [base]
    source_root = source_admission.get("source_root")
    if source_root is not None:
        if not isinstance(source_root, str) or not source_root:
            return "journal_source_identity_mismatch: admission source_root is malformed"
        roots.insert(0, Path(source_root))
    checked = 0
    for source in request.sources:
        relative = _safe_source_relative(source.uri)
        if relative is None:
            return "journal_source_identity_mismatch: source URI is not a safe relative path"
        if recipe_ref is not None:
            for key in ("artifact_id", "uri", "format"):
                if recipe_ref.get(key) != getattr(source, key):
                    return "journal_source_identity_mismatch: recipe source reference differs"
        if isinstance(admission_source, Mapping):
            for key in ("artifact_id", "uri", "format"):
                if admission_source.get(key) != getattr(source, key):
                    return "journal_source_identity_mismatch: admission source differs"
        expected_digest = source.sha256
        if not isinstance(expected_digest, str) or len(expected_digest) != _SHA256_LENGTH:
            if isinstance(admission_source, Mapping):
                expected_digest = admission_source.get("sha256", "")
            if not isinstance(expected_digest, str) or len(expected_digest) != _SHA256_LENGTH:
                return "journal_source_integrity_unavailable: source digest is missing"
        expected_digest = expected_digest.lower()
        if any(character not in "0123456789abcdef" for character in expected_digest):
            return "journal_source_integrity_unavailable: source digest is malformed"
        checked += 1
        actual = None
        for root in roots:
            actual = _source_digest(root, relative)
            if actual is not None:
                break
        if actual is None:
            return "journal_source_integrity_unavailable: source bytes are unavailable"
        if actual != expected_digest:
            return "journal_source_mutated: source bytes no longer match the admitted digest"
    if checked == 0:
        return "journal_source_integrity_unavailable: no hashed source was declared"
    if isinstance(admission_source, Mapping):
        admission_digest = admission_source.get("sha256")
        request_digests = {source.sha256 for source in request.sources if source.sha256}
        if admission_digest not in request_digests:
            return "journal_source_identity_mismatch: admission digest differs"
    return None


def _contains_true_claim(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("scientific_claim_allowed") is True:
            return True
        return any(_contains_true_claim(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_true_claim(item) for item in value)
    return False


def _journal_integrity_error(
    base: Path, request: ComponentRequest, journal: Mapping[str, Any]
) -> str | None:
    """Validate the read-only journal boundary using SREV-24's identity rules."""

    try:
        child = _loop_request(request)
        validated = loop._validate_input(child, autonomous=True, read_only=False, resume=True)
    except (ReviewSessionError, loop.ExperimentLoopError, OSError, TypeError, ValueError) as error:
        return f"journal_request_invalid: {error}"
    if not isinstance(journal, Mapping):
        return "journal_malformed: journal is not an object"
    identity = {
        "schema_version": loop.SESSION_JOURNAL_SCHEMA_VERSION,
        "component_id": loop.COMPONENT_ID,
        "component_version": loop.COMPONENT_VERSION,
        "session_id": validated.session_id,
        "request_id": child.request_id,
        "request_digest": loop._canonical_digest(loop._request_identity(child)),
        "recipe_id": str(validated.recipe.get("recipe_id", "")),
        "recipe_digest": loop.experiment_recipe_canonical_digest(validated.recipe),
    }
    for key, expected in identity.items():
        if journal.get(key) != expected:
            return f"journal_identity_mismatch: {key}"
    if journal.get("evidence_boundary") != EVIDENCE_BOUNDARY:
        return "journal_boundary_tampered: evidence boundary mismatch"
    if journal.get("scientific_claim_allowed") is not False:
        return "journal_boundary_tampered: scientific claims are forbidden"
    if journal.get("dependent_family_status") != DEPENDENT_FAMILY_STATUS:
        return "journal_boundary_tampered: dependent family status mismatch"
    if _contains_true_claim(journal):
        return "journal_boundary_tampered: nested scientific claim flag is true"
    status = journal.get("status")
    if status not in _JOURNAL_STATUSES:
        return "journal_malformed: invalid status"
    if not isinstance(journal.get("stop_reason"), str):
        return "journal_malformed: stop reason is not text"
    source_admission = journal.get("source_admission")
    if not isinstance(source_admission, Mapping):
        return "journal_malformed: source admission is not an object"
    source_error = _source_integrity_error(base, request, validated.recipe, source_admission)
    if source_error is not None:
        return source_error
    budget = journal.get("budget")
    if not isinstance(budget, Mapping):
        return "journal_malformed: budget is not an object"
    required_budget_keys = {
        "max_candidates",
        "max_executions",
        "wall_timeout_s",
        "max_concurrent_local_cpu_processes",
        "max_retries",
    }
    if set(budget) != required_budget_keys:
        return "journal_malformed: budget identity is malformed"
    for key in required_budget_keys:
        value = budget.get(key)
        if key == "wall_timeout_s":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return f"journal_malformed: budget {key} is invalid"
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"journal_malformed: budget {key} is invalid"
    for key in ("executions_consumed", "reserved_executions"):
        value = journal.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"journal_malformed: {key} is invalid"
    elapsed = journal.get("elapsed_s")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        return "journal_malformed: elapsed_s is invalid"
    if not math.isfinite(float(elapsed)) or float(elapsed) < 0:
        return "journal_malformed: elapsed_s is invalid"
    if journal["executions_consumed"] + journal["reserved_executions"] > budget["max_executions"]:
        return "journal_malformed: execution accounting exceeds budget"
    candidate_catalog = journal.get("candidate_catalog")
    expected_catalog = [
        {
            "intervention_id": item["intervention_id"],
            "priority": item["priority"],
            "factor": item.get("factor", ""),
        }
        for item in loop._candidate_order(validated.recipe)
    ]
    if candidate_catalog != expected_catalog:
        return "journal_identity_mismatch: candidate catalog"
    candidate_order = journal.get("candidate_order")
    if (
        not isinstance(candidate_order, list)
        or candidate_order != expected_catalog[: len(candidate_order)]
    ):
        return "journal_malformed: candidate order"
    candidates = journal.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != {
        str(item["intervention_id"]) for item in candidate_order
    }:
        return "journal_malformed: candidate state"
    for candidate_id, candidate in candidates.items():
        if not isinstance(candidate, Mapping) or candidate.get("intervention_id") != candidate_id:
            return "journal_malformed: candidate identity"
        if candidate.get("state") not in {
            "pending",
            "reserved",
            *(_JOURNAL_STATUSES - {"running"}),
        }:
            return "journal_malformed: candidate state value"
    for key in ("outcomes", "operations"):
        value = journal.get(key)
        if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
            return f"journal_malformed: {key}"
    policy = journal.get("policy")
    expected_policy = {"autonomous": True, "read_only": False}
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != value for key, value in expected_policy.items()
    ):
        return "journal_identity_mismatch: policy"
    expected_config_identity = loop._canonical_digest(
        loop._config_identity_document(child.config, expected_policy)
    )
    if journal.get("config_identity_digest") != expected_config_identity:
        return "journal_identity_mismatch: config identity"
    if "answerability" not in journal:
        return "journal_malformed: answerability is missing"
    try:
        expected_answerability = loop._answerability_document(validated.answerability)
    except (loop.ExperimentLoopError, TypeError, ValueError) as error:
        return f"journal_request_invalid: answerability: {error}"
    if journal.get("answerability") != expected_answerability:
        return "journal_identity_mismatch: answerability"
    provenance = journal.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        return "journal_malformed: provenance"
    return None


def _report_integrity_error(
    request: ComponentRequest,
    report: Mapping[str, Any],
    journal: Mapping[str, Any],
) -> str | None:
    try:
        child = _loop_request(request)
        validated = loop._validate_input(child, autonomous=True, read_only=False, resume=True)
    except (ReviewSessionError, loop.ExperimentLoopError, OSError, TypeError, ValueError) as error:
        return f"report_request_invalid: {error}"
    if not isinstance(report, Mapping):
        return "report_malformed: report is not an object"
    expected = {
        "schema_version": loop.LOOP_REPORT_SCHEMA_VERSION,
        "component_id": loop.COMPONENT_ID,
        "component_version": loop.COMPONENT_VERSION,
        "session_id": validated.session_id,
        "request_id": child.request_id,
        "recipe_id": str(validated.recipe.get("recipe_id", "")),
        "recipe_digest": loop.experiment_recipe_canonical_digest(validated.recipe),
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
        "status": journal.get("status"),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            return f"report_identity_mismatch: {key}"
    if _contains_true_claim(report):
        return "report_boundary_tampered: nested scientific claim flag is true"
    expected_source_identity = validated.recipe.get("source_identity", {})
    if report.get("source_identity") != expected_source_identity:
        return "report_identity_mismatch: source identity"
    if report.get("source_admission") != journal.get("source_admission"):
        return "report_identity_mismatch: source admission"
    if report.get("candidate_order") != journal.get("candidate_order"):
        return "report_identity_mismatch: candidate order"
    if not isinstance(report.get("outcomes"), list) or any(
        not isinstance(item, Mapping) for item in report["outcomes"]
    ):
        return "report_malformed: outcomes"
    expected_outcomes = [
        loop.ExperimentLoop._canonical_report_outcome(item) for item in journal.get("outcomes", [])
    ]
    if report.get("outcomes") != expected_outcomes:
        return "report_identity_mismatch: outcomes"
    expected_negative = [item for item in expected_outcomes if item.get("negative") is True]
    if report.get("negative_outcomes") != expected_negative:
        return "report_identity_mismatch: negative outcomes"
    expected_budget = {
        **dict(journal["budget"]),
        "executions_consumed": journal["executions_consumed"],
        "reserved_executions": journal["reserved_executions"],
        "elapsed_s": journal["elapsed_s"],
    }
    if report.get("budget") != expected_budget:
        return "report_identity_mismatch: budget"
    if report.get("operations") != journal.get("operations"):
        return "report_identity_mismatch: operations"
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        return "report_malformed: provenance"
    if provenance.get("request_digest") != loop._canonical_digest(loop._request_identity(child)):
        return "report_identity_mismatch: provenance request digest"
    if provenance.get("recipe_digest") != expected["recipe_digest"]:
        return "report_identity_mismatch: provenance recipe digest"
    return None


def _read_journal(base: Path, request: ComponentRequest) -> dict[str, Any] | None:
    state = _output_state(base, request)
    if state.get("status") not in {"resume_available", "collision"}:
        return None
    try:
        output = _resolve_output(base, request.output_directory, create=False)
        payload = _read_json(output / JOURNAL_FILENAME)
    except (ReviewSessionError, OSError, ValueError, TypeError) as error:
        return {"_read_error": str(error)}
    if not isinstance(payload, dict):
        return {"_read_error": "journal is not an object"}
    integrity_error = _journal_integrity_error(base, request, payload)
    if integrity_error is not None:
        return {"_read_error": integrity_error}
    return payload


def _journal_progress(journal: Mapping[str, Any]) -> dict[str, Any]:
    safe_failure = {
        "status": "failed",
        "reason": "journal_malformed: progress fields are invalid",
        "budget": {},
        "candidates": [],
        "outcomes": [],
        "operations": [],
        "source_admission": {},
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
    }
    if not isinstance(journal, Mapping):
        return safe_failure
    budget = dict(journal.get("budget", {})) if isinstance(journal.get("budget"), Mapping) else {}
    try:
        consumed = int(
            journal.get("executions_consumed", budget.get("executions_consumed", 0)) or 0
        )
        reserved = int(
            journal.get("reserved_executions", budget.get("reserved_executions", 0)) or 0
        )
        maximum = int(budget.get("max_executions", 0) or 0)
        elapsed = float(journal.get("elapsed_s", budget.get("elapsed_s", 0.0)) or 0.0)
        wall = float(budget.get("wall_timeout_s", 0.0) or 0.0)
        if any(
            value < 0 or not math.isfinite(float(value))
            for value in (consumed, reserved, maximum, elapsed, wall)
        ):
            return safe_failure
    except (TypeError, ValueError, OverflowError):
        return safe_failure
    candidate_rows = journal.get("candidates", {})
    candidates = list(candidate_rows.values()) if isinstance(candidate_rows, Mapping) else []
    outcomes = journal.get("outcomes", [])
    return {
        "status": journal.get("status", "running")
        if journal.get("status", "running") in _JOURNAL_STATUSES
        else "failed",
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
        "source_admission": dict(journal.get("source_admission", {}))
        if isinstance(journal.get("source_admission", {}), Mapping)
        else {},
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
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
        return {
            "status": "failed",
            "reason": journal["_read_error"],
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "scientific_claim_allowed": False,
            "dependent_family_status": DEPENDENT_FAMILY_STATUS,
        }
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
    journal = _read_journal(root, normalized)
    if journal is None:
        return {"status": "unavailable", "reason": "result_unavailable: session journal is missing"}
    if "_read_error" in journal:
        return {"status": "unavailable", "reason": f"result_unavailable: {journal['_read_error']}"}
    try:
        output = _resolve_output(root, normalized.output_directory, create=False)
        report = _read_json(output / REPORT_FILENAME)
    except (ReviewSessionError, OSError, ValueError, TypeError) as error:
        return {"status": "unavailable", "reason": f"result_unavailable: {error}"}
    if not isinstance(report, Mapping):
        return {"status": "failed", "reason": "result report is not an object"}
    report_error = _report_integrity_error(normalized, report, journal)
    if report_error is not None:
        return {"status": "unavailable", "reason": f"result_unavailable: {report_error}"}
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
    child_request = _loop_request(normalized)
    try:
        validated = loop._validate_input(
            child_request, autonomous=False, read_only=True, resume=False
        )
    except (ReviewSessionError, loop.ExperimentLoopError, OSError, TypeError, ValueError) as error:
        raise ReviewSessionError(str(error)) from error
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
    candidates = [
        {
            "intervention_id": item["intervention_id"],
            "priority": item["priority"],
            "factor": item.get("factor", ""),
            "state": "pending",
        }
        for item in loop._candidate_order(validated.recipe)[: validated.budget.max_candidates]
    ]
    root = base if base is not None else Path.cwd()
    journal = _read_journal(root, normalized)
    journal_error = None
    if journal is not None and "_read_error" not in journal:
        persisted = _journal_progress(journal)
        budget = dict(persisted["budget"])
        candidates = list(persisted["candidates"])
        state = str(persisted.get("status", "running"))
        source_state = dict(persisted.get("source_admission", {}))
        stop_reason = str(persisted.get("stop_reason", ""))
    elif journal is not None:
        journal_error = str(journal.get("_read_error", "journal unavailable"))
        state = "failed"
        source_state = {
            "status": "unavailable",
            "reason": "durable session state failed integrity validation",
        }
        stop_reason = journal_error
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
        "session_id": str(validated.session_id),
        "context": _context_document(normalized, validated),
        "status": state,
        **({"reason": journal_error} if journal_error else {}),
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
        durable_journal = _read_journal(root, normalized)
        if durable_journal is None or "_read_error" in durable_journal:
            raise ReviewSessionError(
                "journal_integrity_failed: "
                + str((durable_journal or {}).get("_read_error", "journal is missing"))
            )
        validated_context = loop._validate_input(
            _loop_request(normalized), autonomous=True, read_only=False, resume=True
        )
        document = {
            "schema_version": SESSION_VIEW_SCHEMA_VERSION,
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "request_id": normalized.request_id,
            "status": child_result.status,
            "session_id": str(validated_context.session_id),
            "context": _context_document(normalized, validated_context),
            "progress": _journal_progress(durable_journal),
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
            self._last_result = None
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
            # A cancellation request is only terminal once SREV-24 has
            # persisted the settled journal.  Returning a synthetic cancelled
            # result while the owner thread is still dispatching would let a
            # caller race recovery or resume against live work.
            self._active_done.wait()
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
    expected_session_id: str | None = None,
    expected_context_revision: str | None = None,
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
            session_id=str(control.get("session_id", "")),
            context_revision=str(control.get("context_revision", "")),
        )
    else:
        raise ControlAuthorizationError("control must be a mapping")
    if current.action not in {"start", "stop", "resume", "progress", "result"}:
        raise ControlAuthorizationError("unsupported control action")
    if (
        not isinstance(current.session_token, str)
        or len(current.session_token) > _MAX_TOKEN_CHARS
        or not current.session_token
    ):
        raise ControlAuthorizationError("session token is required")
    if not secrets.compare_digest(current.session_token, expected_session_token):
        raise ControlAuthorizationError("session token mismatch")
    if not is_loopback_origin(current.origin):
        raise ControlAuthorizationError("control origin must be loopback")
    if current.origin != expected_origin:
        raise ControlAuthorizationError("control origin mismatch")
    if expected_session_id is not None:
        if not current.session_id or current.session_id != expected_session_id:
            raise ControlAuthorizationError("session context mismatch")
    if (
        expected_context_revision is not None
        and current.context_revision != expected_context_revision
    ):
        raise ControlAuthorizationError("context revision mismatch")
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

    context = _control_context(session.request)

    def handle(control: SessionControl | Mapping[str, Any]) -> Any:
        authorized = validate_control(
            control,
            expected_origin=origin,
            expected_session_token=session_token,
            expected_session_id=str(context["session_id"]),
            expected_context_revision=(
                str(context["context_revision"])
                if context.get("context_revision") is not None
                else None
            ),
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


def _configured_control_binding(request: ComponentRequest) -> tuple[str, str] | None:
    configured_origin = request.config.get("origin")
    configured_token = request.config.get("session_token")
    if not isinstance(configured_origin, str) or not isinstance(configured_token, str):
        return None
    if not configured_origin or not configured_token:
        return None
    return configured_origin, configured_token


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
    if args.stop:
        configured_binding = _configured_control_binding(request)
        if configured_binding is None:
            return _cli_failure(
                "control_authorization_failed: request has no session-owned control binding",
                payload,
            )
        expected_origin, expected_token = configured_binding
        try:
            context = _control_context(request)
            validate_control(
                SessionControl(
                    action="stop",
                    origin=args.origin,
                    session_token=args.session_token,
                    session_id=str(context["session_id"]),
                    context_revision=str(context.get("context_revision", "")),
                ),
                expected_origin=expected_origin,
                expected_session_token=expected_token,
                expected_session_id=str(context["session_id"]),
                expected_context_revision=(
                    str(context["context_revision"])
                    if context.get("context_revision") is not None
                    else None
                ),
            )
            persisted = _read_journal(root, request)
            if persisted is None or "_read_error" in persisted:
                raise ControlAuthorizationError(
                    "session journal is unavailable for control authentication"
                )
            if persisted.get("session_id") != context["session_id"]:
                raise ControlAuthorizationError("session token is not bound to this session")
        except (ControlAuthorizationError, ReviewSessionError, loop.ExperimentLoopError) as error:
            return _cli_failure(f"control_authorization_failed: {error}", payload)
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
