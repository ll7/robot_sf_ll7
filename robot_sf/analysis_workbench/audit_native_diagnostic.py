# ruff: noqa: C901, PLR0912, PLR0915, BLE001, DOC201

"""Bounded native-source diagnostic adapter for BA-05 (issue #9488).

This module owns the deliberately small native seam that the fixture-oriented
SREV-22 executor cannot provide.  A caller supplies an externally admitted
source bundle containing one historical canonical-runner episode and its exact
runner inputs.  The adapter then:

* resolves the source through the #9417 admitted-source resolver and a pinned,
  descriptor-relative trust root;
* runs a separately identified control and one goal intervention through
  :func:`robot_sf.benchmark.runner.run_episode` in spawned, killable children;
* compares the new control with the original before treating intervention
  activation as observed; and
* returns diagnostic-only, fail-closed evidence.

The source bundle is data, never executable instructions.  Only the closed
``simple_policy`` runner argument allowlist below is accepted.  Stateful,
checkpoint, model, and arbitrary native-command inputs are unavailable rather
than silently downgraded.  ``complete`` means that this *diagnostic pair* was
measured and its control was exact within the reported fingerprint scope; it is
not benchmark or scientific evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import stat
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from robot_sf.analysis_workbench.review_contracts import (
    AdmittedSourceResolution,
    ComponentRequest,
    ExperimentRecipe,
    ReviewContractsValidationError,
    component_request_canonical_digest,
    component_request_from_dict,
    experiment_recipe_canonical_digest,
    experiment_recipe_from_dict,
    resolve_admitted_source,
)
from robot_sf.benchmark.analysis_trace import trace_artifact_sha256
from robot_sf.benchmark.runner import run_episode

NATIVE_DIAGNOSTIC_SCHEMA_VERSION = "native-diagnostic.v1"
NATIVE_SOURCE_SCHEMA_VERSION = "native-diagnostic-source.v1"
NATIVE_SOURCE_FORMAT = "native-diagnostic-source.v1+json"
DIAGNOSTIC_EVIDENCE_BOUNDARY = "diagnostic_only"
DEPENDENT_FAMILY_STATUS = "standalone_fixture_only"
COMPONENT_ID = "ba05-native-diagnostic"
SUPPORTED_PLANNERS = ("simple_policy",)

STATUS_COMPLETE = "complete"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
RESULT_STATUSES = (STATUS_COMPLETE, STATUS_UNAVAILABLE, STATUS_FAILED, STATUS_CANCELLED)

MAX_TIMEOUT_S = 60.0
MAX_HORIZON_STEPS = 600
MAX_SOURCE_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_RECORD_STEPS = MAX_HORIZON_STEPS + 1
MAX_TEXT_CHARS = 512
MAX_JSON_NODES = 200_000
_SHA256_RE = frozenset("0123456789abcdef")
_SHA40_RE = frozenset("0123456789abcdef")
_UNSUPPORTED_INPUT_TOKENS = frozenset(
    {
        "checkpoint",
        "checkpoint_path",
        "model_path",
        "model_checkpoint",
        "policy_state",
        "resume",
        "stateful",
        "weights",
    }
)
_RUNNER_INPUT_KEYS = frozenset(
    {
        "scenario_params",
        "seed",
        "horizon",
        "dt",
        "robot_start",
        "robot_goal",
        "algo",
        "algo_config_path",
        "record_forces",
        "telemetry",
        "environment_identity",
    }
)
_ADMISSION_KEYS = frozenset(
    {
        "source_root",
        "receipt_reference",
        "receipt_sha256",
        "expected_source_commit",
        "expected_config_identity",
    }
)
_REQUEST_KEYS = frozenset(
    {"schema_version", "request_id", "admission", "request", "recipe", "intervention", "timeout_s"}
)
_SOURCE_KEYS = frozenset(
    {"schema_version", "source_id", "original_record", "runner_input", "identity"}
)
_SOURCE_IDENTITY_KEYS = frozenset(
    {
        "source_commit",
        "config_identity",
        "initial_state_sha256",
        "environment_identity",
    }
)
_RECORD_IDENTITY_KEYS = frozenset(
    {"config_identity", "initial_state_sha256", "environment_identity"}
)
_EXACT_INPUT_STATE_KEYS = frozenset(
    {
        "recording",
        "original_recording",
        "original_trace",
        "retained_trace",
        "retained_trace_source",
        "retained_trace_recording",
        "simulation_trace",
        "trace",
        "retained_state",
        "retained_states",
        "replay_steps",
    }
)
_NATIVE_CONFIG_IDENTITY_PREFIX = "simple_policy-config.v1:"
ROLE_HISTORICAL_ORIGINAL = "historical_original"
ROLE_DIAGNOSTIC_CONTROL = "new_diagnostic_control"
ROLE_DIAGNOSTIC_INTERVENTION = "new_diagnostic_intervention"
ROLE_EXACT_INPUT_REGENERATION = "new_exact_input_regeneration"
_NATIVE_RECORD_ROLES = frozenset(
    {
        ROLE_HISTORICAL_ORIGINAL,
        ROLE_DIAGNOSTIC_CONTROL,
        ROLE_DIAGNOSTIC_INTERVENTION,
        ROLE_EXACT_INPUT_REGENERATION,
    }
)
_RUNTIME_BOOLEAN_MARKERS = frozenset(
    {
        "fallback",
        "fallback_active",
        "fallback_triggered",
        "fallback_or_degraded",
        "fallback_used",
        "degraded",
        "degraded_active",
        "adapter_active",
    }
)
_RUNTIME_STATUS_FIELDS = frozenset(
    {"status", "row_status", "readiness_status", "availability_status"}
)
_FORBIDDEN_RUNTIME_STATUSES = frozenset({"fallback", "degraded", "not_available", "unavailable"})
_FORBIDDEN_RUNTIME_STATUS_PREFIXES = ("predictive_foresight_model_fallback",)
_RUNTIME_REASON_MARKERS = frozenset({"fallback_reason", "degraded_reason", "degradation_reason"})
_TRACE_LIMITATION_BLOCKS = frozenset({"analysis_trace_coverage", "analysis_trace_unavailable"})
_RUNTIME_MARKER_BLOCKS = frozenset(
    {
        "adapter_impact",
        "foresight_prediction",
        "force_diagnostics",
        "planner_diagnostics",
        "planner_runtime",
    }
)


class NativeDiagnosticError(ValueError):
    """Raised when a native diagnostic request or source is malformed."""

    def __init__(self, errors: Sequence[str], *, source: str | Path | None = None):
        """Build a bounded, actionable validation error."""
        self.errors = tuple(str(error) for error in errors)
        prefix = f"{source}: " if source is not None else ""
        super().__init__(prefix + "; ".join(self.errors))


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeDiagnosticError([f"{name} must be a finite number"])
    result = float(value)
    if not math.isfinite(result):
        raise NativeDiagnosticError([f"{name} must be a finite number"])
    return result


def _text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise NativeDiagnosticError([f"{name} must be a non-empty string"])
    if len(value) > MAX_TEXT_CHARS:
        raise NativeDiagnosticError([f"{name} exceeds {MAX_TEXT_CHARS} characters"])
    if "\x00" in value:
        raise NativeDiagnosticError([f"{name} contains a NUL"])
    return value


def _sha256(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if len(text) != 64 or any(char.lower() not in _SHA256_RE for char in text):
        raise NativeDiagnosticError([f"{name} must be a 64-hex SHA-256"])
    return text.lower()


def _sha40(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if len(text) != 40 or any(char.lower() not in _SHA40_RE for char in text):
        raise NativeDiagnosticError([f"{name} must be a 40-hex commit SHA"])
    return text.lower()


def _strict_json(
    value: Any, *, path: str = "$", depth: int = 0, nodes: list[int] | None = None
) -> None:
    """Reject non-finite, opaque, and excessively large source input values."""
    counter = nodes if nodes is not None else [0]
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise NativeDiagnosticError(["source contains too many JSON values"])
    if depth > 32:
        raise NativeDiagnosticError([f"{path} exceeds maximum nesting depth"])
    if isinstance(value, float) and not math.isfinite(value):
        raise NativeDiagnosticError([f"{path} contains a non-finite number"])
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise NativeDiagnosticError([f"{path} contains a non-string key"])
            _strict_json(child, path=f"{path}.{key}", depth=depth + 1, nodes=counter)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _strict_json(child, path=f"{path}[{index}]", depth=depth + 1, nodes=counter)


def _canonical_json(value: Any) -> str:
    _strict_json(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise NativeDiagnosticError([f"value is not strict JSON: {error}"]) from error


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _scenario_identity_document(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return the environment fields used by the canonical trace digest."""
    return {key: value for key, value in scenario.items() if key not in {"seed", "repeats"}}


def _environment_identity_for_runner(runner_input: Mapping[str, Any]) -> dict[str, str]:
    """Derive the non-opaque environment identity from actual runner inputs."""
    scenario = dict(runner_input["scenario_params"])
    scenario.setdefault("algo", runner_input.get("algo", "simple_policy"))
    return {
        "scenario_id": str(scenario["id"]),
        "scenario_digest": _digest(_scenario_identity_document(scenario)),
    }


def _initial_state_document(runner_input: Mapping[str, Any]) -> dict[str, list[float]]:
    """Return the immutable initial robot state bound by the native source."""
    return {
        "robot_start": [float(value) for value in runner_input["robot_start"]],
        "robot_goal": [float(value) for value in runner_input["robot_goal"]],
    }


def _initial_state_digest(runner_input: Mapping[str, Any]) -> str:
    return _digest(_initial_state_document(runner_input))


def _config_identity_for_runner(runner_input: Mapping[str, Any]) -> str:
    """Derive a stable planner-config identity instead of trusting an opaque label."""
    return _NATIVE_CONFIG_IDENTITY_PREFIX + _digest(
        {
            "algorithm": runner_input.get("algo", "simple_policy"),
            "algo_config_path": None,
            "planner_config": {},
        }
    )


def _record_identity_proof(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Select the identity fields that canonical records must carry in provenance."""
    return {key: identity[key] for key in _RECORD_IDENTITY_KEYS}


def _runner_input_identity(runner_input: Mapping[str, Any]) -> dict[str, Any]:
    """Return the identity derived from the exact inputs sent to ``run_episode``."""
    return {
        "environment_identity": runner_input["environment_identity"],
        "initial_state_sha256": _initial_state_digest(runner_input),
        "config_identity": _config_identity_for_runner(runner_input),
    }


def _canonical_metrics(value: Any) -> dict[str, Any] | None:
    """Return the complete JSON metric mapping, or ``None`` when unverifiable."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        return None
    candidate = dict(value)
    try:
        _canonical_json(candidate)
    except (NativeDiagnosticError, TypeError, ValueError, RecursionError):
        return None
    return candidate


def _metadata_ineligibility_markers(
    value: Any,
    *,
    path: str = "algorithm_metadata",
    diagnostic_context: bool = False,
) -> tuple[str, ...]:
    """Return typed native-ineligible markers nested in planner metadata.

    The native gate intentionally has a stricter contract than the generic
    benchmark status helpers: every declared execution mode must be exactly
    ``native`` and every evidence flag must be a real boolean.  Explicit
    fallback/degraded fields use a small typed truth table so values such as
    ``0`` or ``"false"`` cannot masquerade as a valid false marker.  The two
    analysis-trace limitation blocks are the only places where ``status`` may
    legitimately be ``unavailable``; they describe optional trace coverage and
    do not describe planner execution.
    """
    markers: list[str] = []

    def _append(message: str) -> None:
        if len(markers) < 16:
            markers.append(message)

    def _counter_marker(child: Any, child_path: str) -> None:
        if isinstance(child, bool) or not isinstance(child, (int, float)):
            _append(f"{child_path} has malformed fallback/degraded counter")
            return
        try:
            finite = math.isfinite(float(child))
        except (OverflowError, ValueError):
            finite = False
        if not finite or child < 0:
            _append(f"{child_path} has malformed fallback/degraded counter")
        elif child > 0:
            _append(f"{child_path} is set")

    def _neutral_reason(container: Mapping[str, Any], key: str) -> bool:
        def _zero_counter(candidate: str) -> bool:
            value = container.get(candidate)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            try:
                return math.isfinite(float(value)) and value == 0
            except (OverflowError, ValueError):
                return False

        if key.startswith("fallback"):
            return any(
                container.get(candidate) is False
                for candidate in (
                    "fallback",
                    "fallback_active",
                    "fallback_triggered",
                    "fallback_used",
                    "fallback_or_degraded",
                )
            ) or _zero_counter("fallback_count")
        if key.startswith("degrad"):
            return (
                container.get("degraded") is False
                or container.get("degraded_active") is False
                or _zero_counter("degraded_count")
            )
        return False

    def _trace_limitation_status(parent_path: str, normalized: str) -> bool:
        parent_parts = parent_path.split(".")
        return (
            normalized == STATUS_UNAVAILABLE
            and len(parent_parts) >= 2
            and parent_parts[-2] == "algorithm_metadata"
            and parent_parts[-1] in _TRACE_LIMITATION_BLOCKS
        )

    def _visit(item: Any, item_path: str, depth: int, node_count: list[int]) -> None:
        if len(markers) >= 16:
            return
        node_count[0] += 1
        if node_count[0] > MAX_JSON_NODES:
            _append(f"{item_path} exceeds marker traversal limit")
            return
        if depth > 32:
            _append(f"{item_path} exceeds marker nesting limit")
            return
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                if not isinstance(raw_key, str):
                    _append(f"{item_path} contains a non-string marker key")
                    continue
                key = raw_key
                lowered = key.lower()
                child_path = f"{item_path}.{key}"
                if lowered == "evidence_eligible":
                    if not isinstance(child, bool):
                        _append(f"{child_path} is not a boolean")
                    elif child is False:
                        _append(f"{child_path}=false")
                elif lowered.endswith("_execution_mode") or lowered == "execution_mode":
                    if child != "native" or not isinstance(child, str):
                        _append(f"{child_path} is not native")
                elif lowered in _RUNTIME_BOOLEAN_MARKERS:
                    if not isinstance(child, bool):
                        _append(f"{child_path} is not a boolean")
                    elif child is True:
                        _append(f"{child_path}=true")
                elif lowered in _RUNTIME_STATUS_FIELDS:
                    if not isinstance(child, str):
                        _append(f"{child_path} is not a status string")
                    else:
                        normalized = child.strip().lower().replace("-", "_")
                        if (
                            normalized in _FORBIDDEN_RUNTIME_STATUSES
                            or any(
                                normalized.startswith(prefix)
                                for prefix in _FORBIDDEN_RUNTIME_STATUS_PREFIXES
                            )
                        ) and not _trace_limitation_status(item_path, normalized):
                            _append(f"{child_path}={child}")
                elif lowered in _RUNTIME_REASON_MARKERS:
                    if child is not None and not isinstance(child, str):
                        _append(f"{child_path} is malformed")
                    elif child not in (None, "") or not _neutral_reason(item, lowered):
                        _append(f"{child_path} is set or malformed")
                elif lowered in {"fallback_reasons", "degraded_reasons"}:
                    if not isinstance(child, (Mapping, list, tuple)):
                        _append(f"{child_path} is malformed")
                    elif child:
                        _append(f"{child_path} is set")
                elif "fallback" in lowered or "degrad" in lowered:
                    _counter_marker(child, child_path)
                if len(markers) >= 16:
                    return
                _visit(child, child_path, depth + 1, node_count)
                if len(markers) >= 16:
                    return
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                _visit(child, f"{item_path}[{index}]", depth + 1, node_count)
                if len(markers) >= 16:
                    return

    _visit(value, path, 0, [0])
    return tuple(markers)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _safe_directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(nofollow, int) or not isinstance(directory, int):
        raise OSError("descriptor-relative no-follow directory support is unavailable")
    if os.open not in os.supports_dir_fd:
        raise OSError("descriptor-relative open support is unavailable")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _safe_file_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblocking = getattr(os, "O_NONBLOCK", None)
    if not isinstance(nofollow, int) or not isinstance(nonblocking, int):
        raise OSError("descriptor-relative no-follow file support is unavailable")
    if os.open not in os.supports_dir_fd:
        raise OSError("descriptor-relative open support is unavailable")
    return os.O_RDONLY | nofollow | nonblocking | getattr(os, "O_CLOEXEC", 0)


def _open_pinned_root(source_root: str | Path) -> tuple[Path, int]:
    """Open every trust-root component without following symlinks."""
    flags = _safe_directory_flags()
    configured = Path(source_root)
    if not configured.parts or ".." in configured.parts:
        raise OSError("source_root must be a non-empty path without traversal")
    if configured.is_absolute():
        current_fd = os.open(configured.anchor, flags)
        components = configured.parts[1:]
        display = configured
    else:
        current_fd = os.open(".", flags)
        components = configured.parts
        display = Path.cwd() / configured
    try:
        for component in components:
            if component in {"", "."}:
                continue
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise OSError("source_root is not a directory")
        return display, current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_root_reference(root_fd: int, reference: str) -> int:
    """Open one root-relative regular file through the pinned root descriptor."""
    path = Path(reference)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or "\\" in reference
        or urlsplit(reference).scheme
        or urlsplit(reference).netloc
        or urlsplit(reference).query
        or urlsplit(reference).fragment
    ):
        raise OSError("reference must be a safe relative path")
    directory_flags = _safe_directory_flags()
    file_flags = _safe_file_flags()
    parent_fd = os.dup(root_fd)
    opened: list[int] = [parent_fd]
    try:
        for component in path.parts[:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            opened.append(child_fd)
            os.close(parent_fd)
            opened.remove(parent_fd)
            parent_fd = child_fd
        file_fd = os.open(path.parts[-1], file_flags, dir_fd=parent_fd)
        opened.append(file_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise OSError("reference must be a regular file")
        os.close(parent_fd)
        opened.remove(parent_fd)
        return file_fd
    except BaseException:
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _read_fd(file_fd: int, *, maximum_bytes: int, label: str) -> bytes:
    content = bytearray()
    while True:
        chunk = os.read(file_fd, maximum_bytes + 1 - len(content))
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > maximum_bytes:
            raise OSError(f"{label} exceeds {maximum_bytes} bytes")
    return bytes(content)


def _read_root_reference(root_fd: int, reference: str, *, label: str) -> tuple[bytes, str]:
    file_fd: int | None = None
    try:
        file_fd = _open_root_reference(root_fd, reference)
        content = _read_fd(file_fd, maximum_bytes=MAX_SOURCE_DOCUMENT_BYTES, label=label)
    finally:
        if file_fd is not None:
            os.close(file_fd)
    return content, hashlib.sha256(content).hexdigest()


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise NativeDiagnosticError([f"{label} is malformed JSON: {error}"]) from error
    if not isinstance(payload, dict):
        raise NativeDiagnosticError([f"{label} must be a JSON object"])
    _strict_json(payload, path=label)
    return payload


@dataclass(frozen=True, slots=True)
class NativeDiagnosticAdmission:
    """Launcher-owned source trust configuration."""

    source_root: Path
    receipt_reference: str
    receipt_sha256: str
    expected_source_commit: str | None = None
    expected_config_identity: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> NativeDiagnosticAdmission:
        """Parse the external trust configuration and reject traversal."""
        if not isinstance(payload, Mapping):
            raise NativeDiagnosticError(["admission must be a mapping"])
        unknown = set(payload) - _ADMISSION_KEYS
        if unknown:
            raise NativeDiagnosticError([f"admission has unknown keys: {sorted(unknown)}"])
        root_value = _text(payload.get("source_root"), name="admission.source_root")
        root = Path(root_value)
        if ".." in root.parts or "\\" in root_value or urlsplit(root_value).scheme:
            raise NativeDiagnosticError(
                ["admission.source_root must be a local path without traversal"]
            )
        reference = _text(payload.get("receipt_reference"), name="admission.receipt_reference")
        reference_path = Path(reference)
        ref_uri = urlsplit(reference)
        if (
            reference_path.is_absolute()
            or not reference_path.parts
            or "." in reference_path.parts
            or ".." in reference_path.parts
            or "\\" in reference
            or ref_uri.scheme
            or ref_uri.netloc
            or ref_uri.query
            or ref_uri.fragment
        ):
            raise NativeDiagnosticError(
                ["admission.receipt_reference must be a safe relative path"]
            )
        expected_commit = payload.get("expected_source_commit")
        if expected_commit is not None:
            expected_commit = _sha40(expected_commit, name="admission.expected_source_commit")
        expected_config = payload.get("expected_config_identity")
        if expected_config is not None:
            expected_config = _text(expected_config, name="admission.expected_config_identity")
        return cls(
            source_root=root,
            receipt_reference=reference,
            receipt_sha256=_sha256(payload.get("receipt_sha256"), name="admission.receipt_sha256"),
            expected_source_commit=expected_commit,
            expected_config_identity=expected_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe trust configuration without source bytes."""
        return {
            "source_root": str(self.source_root),
            "receipt_reference": self.receipt_reference,
            "receipt_sha256": self.receipt_sha256,
            "expected_source_commit": self.expected_source_commit,
            "expected_config_identity": self.expected_config_identity,
        }


@dataclass(frozen=True, slots=True)
class NativeDiagnosticRequest:
    """Validated native diagnostic input envelope.

    ``request`` and ``recipe`` are the stable #9417 admission contexts.  The
    source root and receipt are intentionally carried in the separate
    ``admission`` object so a source artifact cannot nominate its own trust
    root.  ``intervention`` is limited to one changed robot goal.
    """

    request_id: str
    admission: NativeDiagnosticAdmission
    request: ComponentRequest
    recipe: ExperimentRecipe
    intervention: dict[str, Any]
    timeout_s: float = 30.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> NativeDiagnosticRequest:
        """Parse and validate one request envelope."""
        if not isinstance(payload, Mapping):
            raise NativeDiagnosticError(["native diagnostic request must be a mapping"])
        unknown = set(payload) - _REQUEST_KEYS
        if unknown:
            raise NativeDiagnosticError([f"request has unknown keys: {sorted(unknown)}"])
        if payload.get("schema_version") != NATIVE_DIAGNOSTIC_SCHEMA_VERSION:
            raise NativeDiagnosticError(
                [f"schema_version must be {NATIVE_DIAGNOSTIC_SCHEMA_VERSION}"]
            )
        request_id = _text(payload.get("request_id"), name="request_id")
        admission = NativeDiagnosticAdmission.from_mapping(payload.get("admission"))
        try:
            request = component_request_from_dict(payload.get("request"), source="request")
            recipe = experiment_recipe_from_dict(payload.get("recipe"), source="recipe")
        except (ReviewContractsValidationError, TypeError, ValueError) as error:
            raise NativeDiagnosticError([f"invalid admission context: {error}"]) from error
        if request.request_id != request_id:
            raise NativeDiagnosticError(["request_id must match request.request_id"])
        if request.component_id != COMPONENT_ID:
            raise NativeDiagnosticError([f"request.component_id must be {COMPONENT_ID}"])
        if len(request.sources) != 1:
            raise NativeDiagnosticError(["request must carry exactly one source reference"])
        source_ref = request.sources[0]
        if (
            source_ref.format != NATIVE_SOURCE_FORMAT
            or source_ref.schema != NATIVE_SOURCE_SCHEMA_VERSION
        ):
            raise NativeDiagnosticError(["request source must use native-diagnostic-source.v1"])
        if not isinstance(payload.get("intervention"), Mapping):
            raise NativeDiagnosticError(["intervention must be a mapping"])
        intervention = dict(payload["intervention"])
        allowed_intervention = {"intervention_id", "factor", "robot_goal", "activation_epsilon_m"}
        unknown_intervention = set(intervention) - allowed_intervention
        if unknown_intervention:
            raise NativeDiagnosticError(
                [f"intervention has unknown keys: {sorted(unknown_intervention)}"]
            )
        intervention_id = _text(intervention.get("intervention_id"), name="intervention_id")
        factor = intervention.get("factor", "robot_goal")
        if factor != "robot_goal":
            raise NativeDiagnosticError(["only factor='robot_goal' is supported"])
        goal = _vector2(intervention.get("robot_goal"), name="intervention.robot_goal")
        epsilon = float(intervention.get("activation_epsilon_m", 1e-9))
        if not math.isfinite(epsilon) or epsilon < 0.0 or epsilon > 1.0:
            raise NativeDiagnosticError(["intervention.activation_epsilon_m must be within 0..1"])
        intervention = {
            "intervention_id": intervention_id,
            "factor": "robot_goal",
            "robot_goal": goal,
            "activation_epsilon_m": epsilon,
        }
        timeout = _finite(payload.get("timeout_s", 30.0), name="timeout_s")
        if timeout <= 0.0 or timeout > MAX_TIMEOUT_S:
            raise NativeDiagnosticError([f"timeout_s must be within (0, {MAX_TIMEOUT_S:g}]"])
        return cls(
            request_id=request_id,
            admission=admission,
            request=request,
            recipe=recipe,
            intervention=intervention,
            timeout_s=timeout,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable request representation."""
        return {
            "schema_version": NATIVE_DIAGNOSTIC_SCHEMA_VERSION,
            "request_id": self.request_id,
            "admission": self.admission.to_dict(),
            "request": asdict(self.request),
            "recipe": asdict(self.recipe),
            "intervention": dict(self.intervention),
            "timeout_s": self.timeout_s,
        }


def _vector2(value: Any, *, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise NativeDiagnosticError([f"{name} must contain exactly two numbers"])
    return (_finite(value[0], name=f"{name}[0]"), _finite(value[1], name=f"{name}[1]"))


def _source_request_projection(
    request: ComponentRequest, admission: NativeDiagnosticAdmission
) -> ComponentRequest:
    """Bind only stable source identity to the #9417 receipt digest.

    Runtime timeout and intervention controls remain outside this projection;
    this mirrors the admitted-source executor boundary instead of making a
    receipt hash cycle through transient operation state.
    """
    return ComponentRequest(
        request_id=request.request_id,
        component_id=request.component_id,
        sources=request.sources,
        output_directory=request.output_directory,
        config={
            "admission": {
                "schema_version": "native-diagnostic-admission.v1",
                "config_identity": admission.expected_config_identity
                or request.sources[0].config_identity,
            }
        },
        required_capabilities=request.required_capabilities,
    )


def _open_and_resolve_source(
    request: NativeDiagnosticRequest,
) -> tuple[Path, int, dict[str, Any], str, AdmittedSourceResolution]:
    """Pin the trust root, verify receipt bytes, and resolve admitted source bytes."""
    root, root_fd = _open_pinned_root(request.admission.source_root)
    try:
        receipt_bytes, receipt_digest = _read_root_reference(
            root_fd, request.admission.receipt_reference, label="admitted-source receipt"
        )
        if receipt_digest != request.admission.receipt_sha256:
            raise NativeDiagnosticError(
                ["source_admission: receipt_stale: receipt digest mismatch"]
            )
        receipt = _json_object(receipt_bytes, label="admitted-source receipt")
        projection = _source_request_projection(request.request, request.admission)
        resolution = resolve_admitted_source(
            receipt,
            allowed_root=root,
            allowed_root_fd=root_fd,
            request=projection,
            recipe=request.recipe,
            expected_source_commit=request.admission.expected_source_commit,
            expected_config_identity=request.admission.expected_config_identity,
        )
        if resolution.status != "admitted" or resolution.receipt is None:
            raise NativeDiagnosticError(
                [f"source_admission: {resolution.reason}: {resolution.detail}".rstrip()]
            )
        if resolution.source_bytes is None:
            raise NativeDiagnosticError(["source_admission: source_mutated: no protected bytes"])
        if resolution.receipt.source_kind != "diagnostic":
            raise NativeDiagnosticError(
                ["source_admission: fixture sources cannot satisfy native path"]
            )
        if resolution.receipt.source.format != NATIVE_SOURCE_FORMAT:
            raise NativeDiagnosticError(["source_admission: unsupported native source format"])
        if resolution.receipt.source.schema != NATIVE_SOURCE_SCHEMA_VERSION:
            raise NativeDiagnosticError(["source_admission: unsupported native source schema"])
        return root, root_fd, receipt, receipt_digest, resolution
    except BaseException:
        os.close(root_fd)
        raise


@dataclass(frozen=True, slots=True)
class _SourceDocument:
    source_id: str
    original_record: dict[str, Any]
    runner_input: dict[str, Any]
    identity: dict[str, Any]


def _validate_runner_input(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise NativeDiagnosticError(["runner_input must be a mapping"])
    unknown = set(raw) - _RUNNER_INPUT_KEYS
    if unknown:
        raise NativeDiagnosticError([f"runner_input has unknown keys: {sorted(unknown)}"])
    _strict_json(raw, path="runner_input")
    runner = dict(raw)
    runner.setdefault("algo", "simple_policy")
    runner.setdefault("algo_config_path", None)
    runner.setdefault("record_forces", False)
    scenario = runner.get("scenario_params")
    if not isinstance(scenario, Mapping) or not scenario.get("id"):
        raise NativeDiagnosticError(["runner_input.scenario_params.id is required"])
    algo = runner["algo"]
    if algo not in SUPPORTED_PLANNERS:
        raise NativeDiagnosticError([f"unsupported native planner: {algo!r}"])
    if runner.get("algo_config_path") not in (None, ""):
        raise NativeDiagnosticError(["stateful/checkpoint planner configuration is unsupported"])
    _reject_unsupported_tokens(runner)
    seed = runner.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**32:
        raise NativeDiagnosticError(["runner_input.seed must be a non-negative 32-bit integer"])
    horizon = runner.get("horizon")
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= MAX_HORIZON_STEPS
    ):
        raise NativeDiagnosticError([f"runner_input.horizon must be within 1..{MAX_HORIZON_STEPS}"])
    dt = _finite(runner.get("dt", 0.1), name="runner_input.dt")
    if not 0.001 <= dt <= 1.0:
        raise NativeDiagnosticError(["runner_input.dt must be within 0.001..1.0"])
    _vector2(runner.get("robot_start"), name="runner_input.robot_start")
    _vector2(runner.get("robot_goal"), name="runner_input.robot_goal")
    telemetry = runner.get("telemetry")
    if not isinstance(telemetry, Mapping) or telemetry.get("analysis_trace") != "all":
        raise NativeDiagnosticError(["runner_input.telemetry.analysis_trace='all' is required"])
    if not isinstance(runner["record_forces"], bool):
        raise NativeDiagnosticError(["runner_input.record_forces must be boolean"])
    environment_identity = runner.get("environment_identity")
    if not isinstance(environment_identity, Mapping):
        raise NativeDiagnosticError(["runner_input.environment_identity is required"])
    _strict_json(environment_identity, path="runner_input.environment_identity")
    expected_environment = _environment_identity_for_runner(runner)
    if dict(environment_identity) != expected_environment:
        raise NativeDiagnosticError(
            ["runner_input.environment_identity does not match scenario_params"]
        )
    return runner


def _reject_unsupported_tokens(value: Any, *, path: str = "runner_input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _UNSUPPORTED_INPUT_TOKENS:
                if child not in (None, False, "", [], {}):
                    raise NativeDiagnosticError(
                        [f"{path}.{key}: stateful/checkpoint configuration is unsupported"]
                    )
            _reject_unsupported_tokens(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_unsupported_tokens(child, path=f"{path}[{index}]")


def _source_document(source_bytes: bytes, receipt_source: Any) -> _SourceDocument:
    payload = _json_object(source_bytes, label="native source")
    unknown = set(payload) - _SOURCE_KEYS
    if unknown:
        raise NativeDiagnosticError([f"native source has unknown keys: {sorted(unknown)}"])
    if payload.get("schema_version") != NATIVE_SOURCE_SCHEMA_VERSION:
        raise NativeDiagnosticError(
            [f"native source schema must be {NATIVE_SOURCE_SCHEMA_VERSION}"]
        )
    source_id = _text(payload.get("source_id"), name="native source.source_id")
    original = payload.get("original_record")
    if not isinstance(original, Mapping):
        raise NativeDiagnosticError(["native source.original_record must be a mapping"])
    runner = _validate_runner_input(payload.get("runner_input"))
    identity = payload.get("identity", {})
    if not isinstance(identity, Mapping):
        raise NativeDiagnosticError(["native source.identity must be a mapping"])
    identity = dict(identity)
    missing_identity = _SOURCE_IDENTITY_KEYS - set(identity)
    if missing_identity:
        raise NativeDiagnosticError(
            [f"native source.identity is missing: {sorted(missing_identity)}"]
        )
    unknown_identity = set(identity) - _SOURCE_IDENTITY_KEYS
    if unknown_identity:
        raise NativeDiagnosticError(
            [f"native source.identity has unknown keys: {sorted(unknown_identity)}"]
        )
    source_commit = _sha40(receipt_source.source_commit, name="receipt.source.source_commit")
    config_identity = _text(receipt_source.config_identity, name="receipt.source.config_identity")
    identity_commit = identity["source_commit"]
    if not isinstance(identity_commit, str) or identity_commit.lower() != source_commit:
        raise NativeDiagnosticError(["native source source_commit differs from admitted receipt"])
    identity_config = identity["config_identity"]
    expected_config = _config_identity_for_runner(runner)
    if not isinstance(identity_config, str) or identity_config != config_identity:
        raise NativeDiagnosticError(["native source config_identity differs from admitted receipt"])
    if identity_config != expected_config:
        raise NativeDiagnosticError(
            ["native source config_identity does not match canonical simple_policy config"]
        )
    initial_state_sha256 = identity["initial_state_sha256"]
    if (
        not isinstance(initial_state_sha256, str)
        or _sha256(initial_state_sha256, name="native source.initial_state_sha256")
        != initial_state_sha256.lower()
        or initial_state_sha256.lower() != _initial_state_digest(runner)
    ):
        raise NativeDiagnosticError(
            ["native source initial_state_sha256 does not match runner inputs"]
        )
    environment_identity = identity["environment_identity"]
    if not isinstance(environment_identity, Mapping) or dict(
        environment_identity
    ) != _environment_identity_for_runner(runner):
        raise NativeDiagnosticError(
            ["native source environment_identity does not match runner inputs"]
        )
    _strict_json(original, path="native source.original_record")
    return _SourceDocument(
        source_id=source_id,
        original_record=dict(original),
        runner_input=runner,
        identity=identity,
    )


def _trace(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        return None
    candidate = metadata.get("analysis_trace")
    return candidate if isinstance(candidate, Mapping) else None


def _record_execution_errors(
    record: Mapping[str, Any],
    runner_input: Mapping[str, Any],
    *,
    source_commit: str,
    source_identity: Mapping[str, Any],
    expected_role: str = ROLE_HISTORICAL_ORIGINAL,
    request_id: str | None = None,
    source_id: str | None = None,
) -> tuple[str, ...]:
    """Return errors that make a record unsafe for native comparison."""
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ("record is not a mapping",)
    if record.get("scenario_id") != runner_input["scenario_params"]["id"]:
        errors.append("scenario identity differs from admitted runner input")
    if record.get("seed") != runner_input["seed"]:
        errors.append("seed differs from admitted runner input")
    if record.get("horizon") != runner_input["horizon"]:
        errors.append("horizon differs from admitted runner input")
    try:
        if not math.isclose(
            float(record.get("dt_s")), float(runner_input["dt"]), rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("dt differs from admitted runner input")
    except (TypeError, ValueError):
        errors.append("record dt_s is missing")
    if record.get("algo") != runner_input.get("algo", "simple_policy"):
        errors.append("planner differs from admitted runner input")
    if record.get("git_hash") != source_commit:
        errors.append("source commit differs from admitted record")
    expected_scenario = dict(runner_input["scenario_params"])
    expected_scenario.setdefault("algo", runner_input.get("algo", "simple_policy"))
    if record.get("scenario_params") != expected_scenario:
        errors.append("environment scenario parameters differ from admitted identity")
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        errors.append("algorithm metadata is missing")
        return tuple(errors)
    if metadata.get("algorithm") != runner_input.get("algo", "simple_policy"):
        errors.append("algorithm metadata is not the intended planner")
    if metadata.get("status") != "ok":
        errors.append(f"planner execution status is {metadata.get('status')!r}")
    if metadata.get("config") != {} or metadata.get("config_hash") != "na":
        errors.append("planner config metadata differs from canonical simple_policy config")
    metadata_markers = _metadata_ineligibility_markers(metadata)
    record_marker_payload = {
        key: value
        for key, value in record.items()
        if isinstance(key, str)
        and (
            key.lower() in _RUNTIME_BOOLEAN_MARKERS
            or key.lower() in _RUNTIME_REASON_MARKERS
            or key.lower() in _RUNTIME_STATUS_FIELDS
            or key.lower() == "evidence_eligible"
            or key.lower() == "execution_mode"
            or key.lower().endswith("_execution_mode")
            or key.lower() in _RUNTIME_MARKER_BLOCKS
            or "fallback" in key.lower()
            or "degrad" in key.lower()
        )
    }
    metadata_markers += _metadata_ineligibility_markers(record_marker_payload, path="record")
    errors.extend(
        f"planner metadata is fallback/ineligible: {marker}" for marker in metadata_markers
    )
    kinematics = metadata.get("planner_kinematics")
    if not isinstance(kinematics, Mapping) or kinematics.get("execution_mode") != "native":
        errors.append("planner execution is not native")
    force_diagnostics = metadata.get("force_diagnostics")
    if isinstance(force_diagnostics, Mapping) and force_diagnostics.get("fallback") is True:
        errors.append("force diagnostics report fallback")
    trace = _trace(record)
    if trace is None:
        errors.append("analysis trace is missing")
        return tuple(errors)
    if trace.get("schema_version") != "analysis-trace.v1":
        errors.append("analysis trace schema is missing or unsupported")
    if trace.get("planner") != runner_input.get("algo", "simple_policy"):
        errors.append("analysis trace planner differs")
    if trace.get("scenario_id") != runner_input["scenario_params"]["id"]:
        errors.append("analysis trace scenario differs")
    expected_environment = source_identity["environment_identity"]
    if (
        trace.get("scenario_digest") != expected_environment["scenario_digest"]
        or trace.get("scenario_id") != expected_environment["scenario_id"]
    ):
        errors.append("analysis trace environment identity differs")
    if trace.get("git_hash") != source_commit:
        errors.append("analysis trace source commit differs")
    stored_digest = trace.get("artifact_sha256")
    if not isinstance(stored_digest, str) or len(stored_digest) != 64:
        errors.append("analysis trace artifact digest is missing")
    else:
        try:
            if trace_artifact_sha256(trace) != stored_digest:
                errors.append("analysis trace artifact digest does not match content")
        except (TypeError, ValueError, RecursionError):
            errors.append("analysis trace artifact digest cannot be recomputed")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not 2 <= len(steps) <= MAX_RECORD_STEPS:
        errors.append("analysis trace does not contain at least two bounded steps")
    else:
        initial_step = steps[0]
        initial_position = (
            initial_step.get("robot", {}).get("position")
            if isinstance(initial_step, Mapping) and isinstance(initial_step.get("robot"), Mapping)
            else None
        )
        try:
            expected_start = _vector2(runner_input["robot_start"], name="runner_input.robot_start")
            observed_start = _vector2(
                initial_position, name="analysis_trace.initial_robot_position"
            )
            if any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(expected_start, observed_start, strict=True)
            ):
                errors.append("analysis trace initial state differs from admitted runner input")
        except NativeDiagnosticError:
            errors.append("analysis trace initial state is missing")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                errors.append(f"analysis trace step {index} is not a mapping")
                continue
            robot = step.get("robot")
            if not isinstance(robot, Mapping):
                errors.append(f"analysis trace step {index} has no robot state")
                continue
            try:
                _vector2(
                    robot.get("position"), name=f"analysis_trace.steps[{index}].robot.position"
                )
            except NativeDiagnosticError:
                errors.append(f"analysis trace step {index} has no finite robot position")
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        errors.append("episode outcome is missing")
    if _canonical_metrics(record.get("metrics")) is None:
        errors.append("episode metrics are missing or not canonical JSON")
    provenance = record.get("provenance")
    native_provenance = (
        provenance.get("native_diagnostic") if isinstance(provenance, Mapping) else None
    )
    observed_identity = (
        native_provenance.get("source_identity") if isinstance(native_provenance, Mapping) else None
    )
    if not isinstance(native_provenance, Mapping):
        errors.append("native diagnostic role provenance is missing")
    else:
        observed_role = native_provenance.get("identity")
        if observed_role != expected_role:
            errors.append(
                f"record role differs: expected {expected_role!r}, observed {observed_role!r}"
            )
        if expected_role not in _NATIVE_RECORD_ROLES:
            errors.append(f"record role is unsupported: {expected_role!r}")
        if source_id is not None and native_provenance.get("source_id") != source_id:
            errors.append("record source_id provenance differs")
        if request_id is not None and native_provenance.get("request_id") != request_id:
            errors.append("record request_id provenance differs")
        if native_provenance.get("source_commit") != source_commit:
            errors.append("record source commit provenance differs")
        if native_provenance.get("config_identity") != _config_identity_for_runner(runner_input):
            errors.append("record config identity provenance differs")
    if not isinstance(observed_identity, Mapping):
        errors.append("record source identity provenance is missing")
    elif dict(observed_identity) != _record_identity_proof(source_identity):
        errors.append("record source identity provenance differs")
    observed_runner_identity = (
        native_provenance.get("runner_input_identity")
        if isinstance(native_provenance, Mapping)
        else None
    )
    if not isinstance(observed_runner_identity, Mapping):
        errors.append("record runner input identity provenance is missing")
    elif dict(observed_runner_identity) != _runner_input_identity(runner_input):
        errors.append("record runner input identity provenance differs")
    status = record.get("status")
    if status not in {"success", "failure"}:
        errors.append(f"episode status is not a completed runner outcome: {status!r}")
    return tuple(errors)


def _selected_identity_candidates(
    episode: Mapping[str, Any], aliases: Sequence[str]
) -> tuple[Any, ...]:
    """Return explicit selected-row identity claims from named envelopes."""

    containers: list[Mapping[str, Any]] = [episode]
    pending: list[Mapping[str, Any]] = [episode]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        for key in ("source", "source_identity", "provenance", "identity"):
            nested = current.get(key)
            if isinstance(nested, Mapping):
                containers.append(nested)
                pending.append(nested)
    values: list[Any] = []
    for container in containers:
        for alias in aliases:
            if alias in container and container[alias] not in (None, ""):
                values.append(container[alias])
    return tuple(values)


def _selected_claim_candidates(
    episode: Mapping[str, Any], aliases: Sequence[str]
) -> tuple[Any, ...]:
    """Return explicit state/planner claims from the selected row envelopes."""

    containers: list[Mapping[str, Any]] = []
    pending: list[Any] = [episode]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(current, Mapping):
            containers.append(current)
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    values: list[Any] = []
    for container in containers:
        for alias in aliases:
            if alias in container and container[alias] not in (None, ""):
                values.append(container[alias])
    return tuple(values)


def _validate_source_recipe_identity(
    source_document: _SourceDocument, request: NativeDiagnosticRequest
) -> None:
    """Bind every recipe source identity field to the admitted source bundle."""

    recipe_identity = request.recipe.document.get("source_identity")
    if not isinstance(recipe_identity, Mapping):
        raise NativeDiagnosticError(["source_admission: recipe source identity is missing"])
    for identity_key in _SOURCE_IDENTITY_KEYS:
        if source_document.identity[identity_key] != recipe_identity.get(identity_key):
            raise NativeDiagnosticError(
                [f"source_admission: {identity_key} differs between source and recipe"]
            )


def _selected_exact_input_errors(
    episode: Mapping[str, Any],
    source_document: _SourceDocument,
    *,
    source_digest: str,
) -> tuple[str, ...]:
    """Validate selected-row claims before trusted exact-input execution.

    The source document remains authoritative for executable inputs.  A row may
    omit optional identity fields, but every explicit claim must agree with the
    admitted source.  This prevents a selected campaign row from changing the
    scenario, planner, or state that the launcher admitted.
    """

    del source_digest  # Campaign-file digests are not native source identity.
    if not isinstance(episode, Mapping):
        return ("selected episode is not a mapping",)
    errors: list[str] = []
    runner = source_document.runner_input
    original = source_document.original_record
    selected_episode_id = episode.get("episode_id")
    expected_episode_id = original.get("episode_id")
    if selected_episode_id != expected_episode_id:
        errors.append("selected episode_id does not match admitted historical episode")

    def _check_text(
        label: str,
        aliases: Sequence[str],
        expected: str,
        *,
        accepted: Sequence[str] = (),
    ) -> None:
        claims = _selected_identity_candidates(episode, aliases)
        if not claims:
            return
        if any(not isinstance(value, str) for value in claims):
            errors.append(f"selected {label} identity is malformed")
            return
        accepted_values = {expected, *accepted}
        if any(value not in accepted_values for value in claims):
            errors.append(f"selected {label} identity does not match admitted source")

    _check_text(
        "source_commit",
        ("source_commit", "repo_commit", "commit_sha", "commit"),
        str(source_document.identity["source_commit"]),
    )
    historical_trace = _trace(original)
    historical_trace_config = (
        historical_trace.get("config_digest") if isinstance(historical_trace, Mapping) else None
    )
    accepted_config = (historical_trace_config,) if isinstance(historical_trace_config, str) else ()
    _check_text(
        "config",
        ("config_identity", "config_digest", "config_hash"),
        str(source_document.identity["config_identity"]),
        accepted=accepted_config,
    )
    _check_text(
        "initial_state",
        ("initial_state_sha256", "initial_state_digest", "initial_state_hash"),
        str(source_document.identity["initial_state_sha256"]),
    )

    environment_claims = _selected_identity_candidates(episode, ("environment_identity",))
    expected_environment = source_document.identity["environment_identity"]
    for claim in environment_claims:
        if not isinstance(claim, Mapping) or dict(claim) != dict(expected_environment):
            errors.append("selected environment identity does not match admitted source")
            break
    _check_text(
        "environment",
        ("environment_digest", "environment_hash"),
        str(expected_environment["scenario_digest"]),
    )

    scenario_claims = _selected_claim_candidates(episode, ("scenario_params",))
    expected_scenario = runner["scenario_params"]
    for claim in scenario_claims:
        if not isinstance(claim, Mapping) or dict(claim) != dict(expected_scenario):
            errors.append("selected scenario_params do not match admitted runner input")
            break
    initial_state_claims = _selected_claim_candidates(episode, ("initial_state",))
    expected_initial_state = {
        "robot_start": list(runner["robot_start"]),
        "robot_goal": list(runner["robot_goal"]),
    }
    for claim in initial_state_claims:
        if not isinstance(claim, Mapping) or dict(claim) != expected_initial_state:
            errors.append("selected initial_state does not match admitted runner input")
            break
    for field_name, expected in (
        ("robot_start", runner["robot_start"]),
        ("robot_goal", runner["robot_goal"]),
    ):
        for claim in _selected_claim_candidates(episode, (field_name,)):
            try:
                if list(_vector2(claim, name=f"selected.{field_name}")) != list(
                    _vector2(expected, name=f"runner_input.{field_name}")
                ):
                    errors.append(f"selected {field_name} differs from admitted runner input")
                    break
            except NativeDiagnosticError:
                errors.append(f"selected {field_name} is malformed")
                break
    for claim in _selected_claim_candidates(episode, ("dt", "dt_s")):
        try:
            if not math.isclose(float(claim), float(runner["dt"]), abs_tol=1e-12):
                errors.append("selected dt differs from admitted runner input")
                break
        except (TypeError, ValueError):
            errors.append("selected dt is malformed")
            break
    for claim in _selected_claim_candidates(episode, ("record_forces",)):
        if claim != runner["record_forces"]:
            errors.append("selected record_forces differs from admitted runner input")
            break
    for claim in _selected_claim_candidates(episode, ("telemetry",)):
        if not isinstance(claim, Mapping) or dict(claim) != dict(runner["telemetry"]):
            errors.append("selected telemetry differs from admitted runner input")
            break

    historical_provenance = original.get("provenance")
    historical_native = (
        historical_provenance.get("native_diagnostic")
        if isinstance(historical_provenance, Mapping)
        else None
    )
    historical_execution_id = (
        historical_native.get("execution_id") if isinstance(historical_native, Mapping) else None
    ) or expected_episode_id
    execution_claims = _selected_identity_candidates(episode, ("execution_id", "run_id"))
    for claim in execution_claims:
        if not isinstance(claim, str) or claim != historical_execution_id:
            errors.append("selected execution_id does not match historical execution")
            break

    forbidden_claim_aliases = (
        "planner_config",
        "algorithm_config",
        "algo_config",
        "algo_config_path",
        "config",
        "planner_state",
        "policy_state",
        "model_path",
        "model",
        "model_id",
        "model_checkpoint",
        "checkpoint",
        "checkpoint_path",
        "checkpoint_uri",
        "checkpoint_ref",
        "checkpoint_digest",
        "checkpoint_hash",
        "stateful",
        "state",
        "state_path",
        "runtime_state",
        "replay_state",
        "weights",
        "resume",
    )
    for alias in forbidden_claim_aliases:
        claims = _selected_claim_candidates(episode, (alias,))
        if any(claim not in (None, "", False, [], {}) for claim in claims):
            errors.append(f"selected {alias} claim is unsupported for stateless exact input")

    for label, actual, expected in (
        ("scenario", episode.get("scenario_id"), runner["scenario_params"]["id"]),
        ("planner", episode.get("algo", episode.get("planner_id")), runner.get("algo")),
    ):
        if actual not in (None, "") and actual != expected:
            errors.append(f"selected {label} differs from admitted runner input")
    if "seed" in episode and episode["seed"] != runner["seed"]:
        errors.append("selected seed differs from admitted runner input")
    if "horizon" in episode and episode["horizon"] != runner["horizon"]:
        errors.append("selected horizon differs from admitted runner input")
    if "dt_s" in episode:
        try:
            if not math.isclose(float(episode["dt_s"]), float(runner["dt"]), abs_tol=1e-12):
                errors.append("selected dt differs from admitted runner input")
        except (TypeError, ValueError):
            errors.append("selected dt is malformed")

    # The exact-input route is only an explicit repair for absent review
    # material.  It must never silently replace a retained trace or recording
    # that the source-first materializer could have rendered.
    for key in _EXACT_INPUT_STATE_KEYS:
        value = episode.get(key)
        if value not in (None, "", [], {}):
            errors.append(f"selected episode carries retained material: {key}")
    metadata = episode.get("algorithm_metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("analysis_trace"), Mapping):
        errors.append(
            "selected episode carries retained material: algorithm_metadata.analysis_trace"
        )
    return tuple(errors)


def _record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    trace = _trace(record)
    steps = trace.get("steps", []) if trace is not None else []
    final_position = None
    if steps and isinstance(steps[-1], Mapping) and isinstance(steps[-1].get("robot"), Mapping):
        final_position = list(steps[-1]["robot"].get("position", ()))
    metric_values = _canonical_metrics(record.get("metrics"))
    return {
        "episode_id": record.get("episode_id"),
        "status": record.get("status"),
        "outcome": dict(record.get("outcome", {}))
        if isinstance(record.get("outcome"), Mapping)
        else None,
        "termination_reason": record.get("termination_reason"),
        "trace_sha256": trace.get("artifact_sha256") if trace is not None else None,
        "trace_steps": len(steps) if isinstance(steps, list) else 0,
        "final_robot_position": final_position,
        "metrics": metric_values,
    }


def _record_fingerprint(record: Mapping[str, Any]) -> str:
    return _digest(_record_summary(record))


def _trajectory_positions(record: Mapping[str, Any]) -> list[tuple[float, float]]:
    trace = _trace(record)
    if trace is None or not isinstance(trace.get("steps"), list):
        return []
    positions: list[tuple[float, float]] = []
    for step in trace["steps"]:
        if not isinstance(step, Mapping) or not isinstance(step.get("robot"), Mapping):
            continue
        position = step["robot"].get("position")
        if isinstance(position, (list, tuple)) and len(position) == 2:
            positions.append((float(position[0]), float(position[1])))
    return positions


def _trajectory_delta(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, float | int | None]:
    left = _trajectory_positions(control)
    right = _trajectory_positions(treatment)
    count = min(len(left), len(right))
    if count == 0:
        return {"max_position_delta_m": None, "final_position_delta_m": None, "paired_steps": 0}
    deltas = [math.hypot(left[i][0] - right[i][0], left[i][1] - right[i][1]) for i in range(count)]
    return {
        "max_position_delta_m": max(deltas),
        "final_position_delta_m": deltas[-1],
        "paired_steps": count,
    }


def _compare_control_fidelity(
    original: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every canonical metric and trace field available to this adapter."""
    original_trace = _trace(original)
    control_trace = _trace(control)
    if original_trace is None or control_trace is None:
        return {"status": "unverifiable", "differences": ["analysis_trace_missing"]}
    original_metrics = _canonical_metrics(original.get("metrics"))
    control_metrics = _canonical_metrics(control.get("metrics"))
    if original_metrics is None or control_metrics is None:
        return {
            "status": "unverifiable",
            "differences": ["metrics_unavailable"],
            "checks": {"metrics": False},
        }
    checks = {
        "trace_sha256": original_trace.get("artifact_sha256")
        == control_trace.get("artifact_sha256"),
        "outcome": original.get("outcome") == control.get("outcome"),
        "termination_reason": original.get("termination_reason")
        == control.get("termination_reason"),
        "record_status": original.get("status") == control.get("status"),
        "metrics": original_metrics == control_metrics,
    }
    differences = [name for name, equal in checks.items() if not equal]
    return {
        "status": "verified" if not differences else "diverged",
        "checks": checks,
        "differences": differences,
        "original_fingerprint": _record_fingerprint(original),
        "control_fingerprint": _record_fingerprint(control),
        "original_trace_sha256": original_trace.get("artifact_sha256"),
        "control_trace_sha256": control_trace.get("artifact_sha256"),
    }


def _runner_kwargs(
    runner_input: Mapping[str, Any], *, robot_goal: Sequence[float], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Project the closed source runner input into canonical ``run_episode`` args."""
    runner_provenance = dict(provenance)
    native_provenance = runner_provenance.get("native_diagnostic")
    if isinstance(native_provenance, Mapping):
        runner_provenance["native_diagnostic"] = {
            **dict(native_provenance),
            "runner_input_identity": _runner_input_identity(runner_input),
        }
    return {
        "scenario_params": dict(runner_input["scenario_params"]),
        "seed": int(runner_input["seed"]),
        "horizon": int(runner_input["horizon"]),
        "dt": float(runner_input["dt"]),
        "robot_start": tuple(runner_input["robot_start"]),
        "robot_goal": tuple(robot_goal),
        "record_forces": bool(runner_input.get("record_forces", False)),
        "algo": str(runner_input.get("algo", "simple_policy")),
        "algo_config_path": None,
        "telemetry": dict(runner_input["telemetry"]),
        "provenance": runner_provenance,
    }


def _native_child(conn: Any, kwargs: Mapping[str, Any]) -> None:
    """Execute one canonical runner call in the owned child process."""
    try:
        record = run_episode(**dict(kwargs))
        conn.send({"status": "ok", "record": record})
    except BaseException as error:  # pragma: no cover - exercised through parent transport
        try:
            conn.send(
                {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {str(error)[:512]}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()


def _terminate_child(process: Any) -> bool:
    """Terminate one owned child and report whether it settled."""
    # ``Process.start`` may fail before assigning a pid.  ``is_alive`` asserts
    # in that state on some multiprocessing implementations, so cleanup must
    # remain safe even when the child never became runnable.
    if getattr(process, "pid", None) is None:
        return True
    if process.is_alive():
        process.terminate()
        process.join(0.5)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(0.5)
    return not process.is_alive()


def _run_bounded(
    runner_input: Mapping[str, Any],
    *,
    robot_goal: Sequence[float],
    provenance: Mapping[str, Any],
    timeout_s: float,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Run one canonical episode with a hard parent-owned deadline."""
    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_native_child,
        args=(
            child_conn,
            _runner_kwargs(runner_input, robot_goal=robot_goal, provenance=provenance),
        ),
        name="ba05-native-diagnostic",
    )
    started = time.monotonic()
    try:
        process.start()
        child_conn.close()
        deadline = started + timeout_s
        while True:
            if cancel_event is not None and cancel_event.is_set():
                settled = _terminate_child(process)
                return {
                    "status": STATUS_CANCELLED,
                    "reason": "cancelled_by_user: owned child terminated",
                    "settled": settled,
                    "elapsed_s": time.monotonic() - started,
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                settled = _terminate_child(process)
                return {
                    "status": STATUS_UNAVAILABLE if settled else STATUS_FAILED,
                    "reason": "per_execution_timeout: owned child terminated",
                    "settled": settled,
                    "elapsed_s": time.monotonic() - started,
                }
            if parent_conn.poll(min(0.05, remaining)):
                try:
                    outcome = parent_conn.recv()
                except (EOFError, OSError) as error:
                    return {
                        "status": STATUS_FAILED,
                        "reason": f"child_transport_failed: {type(error).__name__}",
                        "elapsed_s": time.monotonic() - started,
                    }
                process.join(0.5)
                if not isinstance(outcome, Mapping):
                    return {
                        "status": STATUS_FAILED,
                        "reason": "child_transport_failed: malformed result",
                        "elapsed_s": time.monotonic() - started,
                    }
                if outcome.get("status") != "ok":
                    return {
                        "status": STATUS_FAILED,
                        "reason": str(outcome.get("reason", "child execution failed")),
                        "elapsed_s": time.monotonic() - started,
                    }
                return {
                    "status": "ok",
                    "record": outcome.get("record"),
                    "elapsed_s": time.monotonic() - started,
                }
            if not process.is_alive():
                return {
                    "status": STATUS_FAILED,
                    "reason": "child_crashed_without_result",
                    "elapsed_s": time.monotonic() - started,
                }
    except (OSError, RuntimeError) as error:
        settled = _terminate_child(process)
        return {
            "status": STATUS_FAILED if settled else STATUS_UNAVAILABLE,
            "reason": f"child_start_failed: {type(error).__name__}: {str(error)[:256]}",
            "settled": settled,
            "elapsed_s": time.monotonic() - started,
        }
    finally:
        try:
            parent_conn.close()
        except OSError:
            pass
        if process.is_alive():
            _terminate_child(process)


@dataclass(frozen=True, slots=True)
class NativeDiagnosticResult:
    """Diagnostic-only result envelope with no raw episode payloads."""

    request_id: str
    status: str
    reason: str
    evidence_boundary: str = DIAGNOSTIC_EVIDENCE_BOUNDARY
    scientific_claim_allowed: bool = False
    original_identity: dict[str, Any] = field(default_factory=dict)
    control_identity: dict[str, Any] = field(default_factory=dict)
    intervention_identity: dict[str, Any] = field(default_factory=dict)
    fidelity: dict[str, Any] = field(default_factory=dict)
    activation: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result envelope."""
        payload = asdict(self)
        payload["schema_version"] = NATIVE_DIAGNOSTIC_SCHEMA_VERSION
        payload["limitations"] = list(self.limitations)
        return payload


def _result_failure(
    request_id: str,
    status: str,
    reason: str,
    *,
    provenance: Mapping[str, Any] | None = None,
    limitations: Sequence[str] = (),
    original_identity: Mapping[str, Any] | None = None,
) -> NativeDiagnosticResult:
    return NativeDiagnosticResult(
        request_id=request_id,
        status=status if status in RESULT_STATUSES else STATUS_FAILED,
        reason=reason,
        provenance=dict(provenance or {}),
        limitations=tuple(str(item) for item in limitations),
        original_identity=dict(original_identity or {}),
    )


def _validate_request(request: NativeDiagnosticRequest) -> None:
    """Validate cross-document source identity before filesystem access."""
    source_ref = request.request.sources[0]
    if source_ref.uri != request.recipe.document.get("source_identity", {}).get("source_uri"):
        raise NativeDiagnosticError(["request and recipe source URI identities differ"])
    source_identity = request.recipe.document.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise NativeDiagnosticError(["recipe.source_identity is required"])
    if source_identity.get("kind") != "diagnostic":
        raise NativeDiagnosticError(["recipe source kind must be diagnostic"])
    if source_identity.get("evidence_boundary") != DIAGNOSTIC_EVIDENCE_BOUNDARY:
        raise NativeDiagnosticError(["recipe must preserve diagnostic_only evidence boundary"])
    if source_identity.get("scientific_claim_allowed") is not False:
        raise NativeDiagnosticError(["recipe cannot authorize scientific claims"])
    if source_identity.get("dependent_family_status") != DEPENDENT_FAMILY_STATUS:
        raise NativeDiagnosticError(["recipe must preserve standalone_fixture_only boundary"])
    missing_identity = _SOURCE_IDENTITY_KEYS - set(source_identity)
    if missing_identity:
        raise NativeDiagnosticError(
            [f"recipe source_identity is missing: {sorted(missing_identity)}"]
        )
    if not isinstance(source_identity.get("environment_identity"), Mapping):
        raise NativeDiagnosticError(["recipe environment_identity must be a mapping"])
    _sha256(source_identity.get("initial_state_sha256"), name="recipe.initial_state_sha256")
    if (
        not isinstance(source_identity["source_commit"], str)
        or source_identity["source_commit"].lower() != source_ref.source_commit.lower()
    ):
        raise NativeDiagnosticError(["recipe source commit differs from request source"])
    if (
        not isinstance(source_identity["config_identity"], str)
        or source_identity["config_identity"] != source_ref.config_identity
    ):
        raise NativeDiagnosticError(["recipe config identity differs from request source"])
    if (
        request.admission.expected_config_identity is not None
        and source_ref.config_identity != request.admission.expected_config_identity
    ):
        raise NativeDiagnosticError(
            ["request source config identity differs from launcher admission"]
        )
    if (
        request.admission.expected_source_commit is not None
        and source_ref.source_commit != request.admission.expected_source_commit
    ):
        raise NativeDiagnosticError(["request source commit differs from launcher admission"])


def _source_admission_provenance(
    resolution: AdmittedSourceResolution, receipt_digest: str, request: NativeDiagnosticRequest
) -> dict[str, Any]:
    receipt = resolution.receipt
    source = receipt.source if receipt is not None else None
    return {
        "receipt_id": receipt.receipt_id if receipt is not None else None,
        "receipt_sha256": receipt_digest,
        "request_sha256": component_request_canonical_digest(
            _source_request_projection(request.request, request.admission)
        ),
        "recipe_sha256": experiment_recipe_canonical_digest(request.recipe),
        "source_sha256": source.sha256 if source is not None else None,
        "source_commit": source.source_commit if source is not None else None,
        "config_identity": source.config_identity if source is not None else None,
        "source_kind": receipt.source_kind if receipt is not None else None,
        "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
    }


def _final_source_check(
    root: Path,
    root_fd: int,
    request: NativeDiagnosticRequest,
    initial_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    receipt_bytes, receipt_digest = _read_root_reference(
        root_fd, request.admission.receipt_reference, label="admitted-source receipt"
    )
    if receipt_digest != request.admission.receipt_sha256:
        raise NativeDiagnosticError(
            ["source_admission: receipt_stale: final receipt digest mismatch"]
        )
    receipt = _json_object(receipt_bytes, label="admitted-source receipt")
    projection = _source_request_projection(request.request, request.admission)
    resolution = resolve_admitted_source(
        receipt,
        allowed_root=root,
        allowed_root_fd=root_fd,
        request=projection,
        recipe=request.recipe,
        expected_source_commit=request.admission.expected_source_commit,
        expected_config_identity=request.admission.expected_config_identity,
    )
    if (
        resolution.status != "admitted"
        or resolution.receipt is None
        or resolution.source_bytes is None
    ):
        raise NativeDiagnosticError(
            [f"source_admission: {resolution.reason}: {resolution.detail}".rstrip()]
        )
    if resolution.receipt.source.sha256 != initial_source_sha256:
        raise NativeDiagnosticError(["source_admission: source_mutated: source digest changed"])
    return _source_admission_provenance(
        resolution, receipt_digest, request
    ), resolution.receipt.source.sha256


def run_native_diagnostic(
    request: NativeDiagnosticRequest | Mapping[str, Any],
    *,
    cancel_event: Any = None,
) -> NativeDiagnosticResult:
    """Run one admitted native control/intervention diagnostic pair.

    The result is ``complete`` only after a valid original, an exact control
    replay, an observed intervention input/trajectory change, and a final
    source-integrity recheck.  A route or episode outcome of ``failure`` is
    retained as a simulator outcome; planner fallback/crash/timeout is an
    execution failure and never becomes activation evidence.
    """
    request_id = (
        request.request_id if isinstance(request, NativeDiagnosticRequest) else "invalid-request"
    )
    try:
        validated = (
            request
            if isinstance(request, NativeDiagnosticRequest)
            else NativeDiagnosticRequest.from_mapping(request)
        )
        request_id = validated.request_id
        _validate_request(validated)
    except (NativeDiagnosticError, TypeError, ValueError) as error:
        return _result_failure(request_id, STATUS_FAILED, f"invalid_request: {str(error)[:1024]}")
    if cancel_event is not None and cancel_event.is_set():
        return _result_failure(request_id, STATUS_CANCELLED, "cancelled_by_user: before admission")

    root_fd: int | None = None
    provenance: dict[str, Any] = {
        "component_id": COMPONENT_ID,
        "component_version": NATIVE_DIAGNOSTIC_SCHEMA_VERSION,
        "canonical_runner": "robot_sf.benchmark.runner.run_episode",
        "route": "native",
        "planner": "simple_policy",
        "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
        "child_process": "multiprocessing.spawn",
    }
    try:
        root, root_fd, _receipt_payload, receipt_digest, resolution = _open_and_resolve_source(
            validated
        )
        source_document = _source_document(
            resolution.source_bytes or b"", resolution.receipt.source
        )
        runner_input = source_document.runner_input
        source_commit = resolution.receipt.source.source_commit
        config_identity = resolution.receipt.source.config_identity
        recipe_identity = validated.recipe.document["source_identity"]
        for identity_key in _SOURCE_IDENTITY_KEYS:
            if source_document.identity[identity_key] != recipe_identity[identity_key]:
                raise NativeDiagnosticError(
                    [f"source_admission: {identity_key} differs between source and recipe"]
                )
        original_errors = _record_execution_errors(
            source_document.original_record,
            runner_input,
            source_commit=source_commit,
            source_identity=source_document.identity,
            expected_role=ROLE_HISTORICAL_ORIGINAL,
            source_id=source_document.source_id,
        )
        original_identity = {
            "kind": "historical_original",
            "source_id": source_document.source_id,
            "source_sha256": resolution.receipt.source.sha256,
            "record_fingerprint": _record_fingerprint(source_document.original_record),
            "episode_id": source_document.original_record.get("episode_id"),
        }
        provenance["source_admission"] = _source_admission_provenance(
            resolution, receipt_digest, validated
        )
        provenance["input_digests"] = {
            "source_bytes_sha256": resolution.receipt.source.sha256,
            "runner_input_sha256": _digest(runner_input),
            "intervention_sha256": _digest(validated.intervention),
        }
        if original_errors:
            missing_telemetry = any(
                "trace" in error or "metrics" in error for error in original_errors
            )
            return _result_failure(
                validated.request_id,
                STATUS_UNAVAILABLE if missing_telemetry else STATUS_FAILED,
                "original_telemetry_missing: " + "; ".join(original_errors)
                if missing_telemetry
                else "original_invalid: " + "; ".join(original_errors),
                provenance=provenance,
                original_identity=original_identity,
                limitations=(
                    "historical original did not satisfy the essential native telemetry contract",
                ),
            )
        original_goal = _vector2(runner_input["robot_goal"], name="runner_input.robot_goal")
        intervention_goal = tuple(validated.intervention["robot_goal"])
        if all(
            math.isclose(original_goal[i], intervention_goal[i], rel_tol=0.0, abs_tol=0.0)
            for i in range(2)
        ):
            return _result_failure(
                validated.request_id,
                STATUS_FAILED,
                "intervention_not_effective: intervention goal equals original goal",
                provenance=provenance,
                original_identity=original_identity,
            )
        control_provenance = {
            "native_diagnostic": {
                "request_id": validated.request_id,
                "identity": ROLE_DIAGNOSTIC_CONTROL,
                "source_id": source_document.source_id,
                "source_commit": source_commit,
                "config_identity": config_identity,
                "source_identity": _record_identity_proof(source_document.identity),
                "runner_input_sha256": _digest(runner_input),
            }
        }
        control = _run_bounded(
            runner_input,
            robot_goal=original_goal,
            provenance=control_provenance,
            timeout_s=validated.timeout_s,
            cancel_event=cancel_event,
        )
        if control.get("status") == STATUS_CANCELLED:
            return _result_failure(
                validated.request_id,
                STATUS_CANCELLED,
                str(control.get("reason", "cancelled_by_user")),
                provenance={**provenance, "control_elapsed_s": control.get("elapsed_s")},
                original_identity=original_identity,
            )
        if control.get("status") != "ok" or not isinstance(control.get("record"), Mapping):
            return _result_failure(
                validated.request_id,
                STATUS_FAILED if control.get("status") == STATUS_FAILED else STATUS_UNAVAILABLE,
                "control_failed: " + str(control.get("reason", "control did not complete")),
                provenance={**provenance, "control_elapsed_s": control.get("elapsed_s")},
                original_identity=original_identity,
            )
        control_record = dict(control["record"])
        control_errors = _record_execution_errors(
            control_record,
            runner_input,
            source_commit=source_commit,
            source_identity=source_document.identity,
            expected_role=ROLE_DIAGNOSTIC_CONTROL,
            request_id=validated.request_id,
            source_id=source_document.source_id,
        )
        control_identity = {
            "kind": "new_diagnostic_control",
            "execution_id": f"{validated.request_id}:control",
            "record_fingerprint": _record_fingerprint(control_record),
            "episode_id": control_record.get("episode_id"),
        }
        if control_errors:
            return _result_failure(
                validated.request_id,
                STATUS_FAILED,
                "control_failed: " + "; ".join(control_errors),
                provenance={**provenance, "control_elapsed_s": control.get("elapsed_s")},
                original_identity=original_identity,
            )
        fidelity = _compare_control_fidelity(source_document.original_record, control_record)
        if fidelity["status"] != "verified":
            status = STATUS_FAILED if fidelity["status"] == "diverged" else STATUS_UNAVAILABLE
            reason = (
                "control_fidelity_diverged"
                if status == STATUS_FAILED
                else "control_fidelity_unverifiable"
            )
            return _result_failure(
                validated.request_id,
                status,
                f"{reason}: {', '.join(fidelity.get('differences', []))}",
                provenance={**provenance, "control_elapsed_s": control.get("elapsed_s")},
                original_identity=original_identity,
            )
        if cancel_event is not None and cancel_event.is_set():
            return _result_failure(
                validated.request_id,
                STATUS_CANCELLED,
                "cancelled_by_user: before intervention",
                provenance={**provenance, "control_elapsed_s": control.get("elapsed_s")},
                original_identity=original_identity,
                limitations=("control was verified; intervention was intentionally not started",),
            )
        treatment_input = dict(runner_input)
        treatment_input["robot_goal"] = list(intervention_goal)
        treatment_provenance = {
            "native_diagnostic": {
                "request_id": validated.request_id,
                "identity": ROLE_DIAGNOSTIC_INTERVENTION,
                "source_id": source_document.source_id,
                "source_commit": source_commit,
                "config_identity": config_identity,
                "source_identity": _record_identity_proof(source_document.identity),
                "runner_input_sha256": _digest(treatment_input),
                "intervention_id": validated.intervention["intervention_id"],
            }
        }
        treatment = _run_bounded(
            treatment_input,
            robot_goal=intervention_goal,
            provenance=treatment_provenance,
            timeout_s=validated.timeout_s,
            cancel_event=cancel_event,
        )
        if treatment.get("status") == STATUS_CANCELLED:
            return _result_failure(
                validated.request_id,
                STATUS_CANCELLED,
                str(treatment.get("reason", "cancelled_by_user")),
                provenance={
                    **provenance,
                    "control_elapsed_s": control.get("elapsed_s"),
                    "intervention_elapsed_s": treatment.get("elapsed_s"),
                },
                original_identity=original_identity,
                limitations=("control was verified; intervention did not complete",),
            )
        if treatment.get("status") != "ok" or not isinstance(treatment.get("record"), Mapping):
            return _result_failure(
                validated.request_id,
                STATUS_FAILED,
                "intervention_failed: "
                + str(treatment.get("reason", "intervention did not complete")),
                provenance={
                    **provenance,
                    "control_elapsed_s": control.get("elapsed_s"),
                    "intervention_elapsed_s": treatment.get("elapsed_s"),
                },
                original_identity=original_identity,
            )
        treatment_record = dict(treatment["record"])
        treatment_errors = _record_execution_errors(
            treatment_record,
            treatment_input,
            source_commit=source_commit,
            source_identity=source_document.identity,
            expected_role=ROLE_DIAGNOSTIC_INTERVENTION,
            request_id=validated.request_id,
            source_id=source_document.source_id,
        )
        if treatment_errors:
            return _result_failure(
                validated.request_id,
                STATUS_FAILED,
                "intervention_failed: " + "; ".join(treatment_errors),
                provenance={
                    **provenance,
                    "control_elapsed_s": control.get("elapsed_s"),
                    "intervention_elapsed_s": treatment.get("elapsed_s"),
                },
                original_identity=original_identity,
                control_identity=control_identity,
            )
        trajectory = _trajectory_delta(control_record, treatment_record)
        observed_delta = trajectory.get("max_position_delta_m")
        input_changed = _digest(runner_input) != _digest(treatment_input)
        trace_changed = _trace(control_record).get("artifact_sha256") != _trace(
            treatment_record
        ).get("artifact_sha256")
        activation_status = (
            "verified"
            if input_changed
            and trace_changed
            and isinstance(observed_delta, (int, float))
            and observed_delta > float(validated.intervention["activation_epsilon_m"])
            else "not_observed"
        )
        intervention_identity = {
            "kind": "new_diagnostic_intervention",
            "execution_id": f"{validated.request_id}:intervention:{validated.intervention['intervention_id']}",
            "intervention_id": validated.intervention["intervention_id"],
            "record_fingerprint": _record_fingerprint(treatment_record),
            "episode_id": treatment_record.get("episode_id"),
        }
        activation = {
            "status": activation_status,
            "factor": "robot_goal",
            "input_changed": input_changed,
            "trace_changed": trace_changed,
            "original_goal": list(original_goal),
            "intervention_goal": list(intervention_goal),
            **trajectory,
            "control_trace_sha256": _trace(control_record).get("artifact_sha256"),
            "intervention_trace_sha256": _trace(treatment_record).get("artifact_sha256"),
            "epsilon_m": validated.intervention["activation_epsilon_m"],
        }
        final_admission, _ = _final_source_check(
            root, root_fd, validated, resolution.receipt.source.sha256
        )
        provenance.update(
            {
                "source_admission": final_admission,
                "control_elapsed_s": control.get("elapsed_s"),
                "intervention_elapsed_s": treatment.get("elapsed_s"),
            }
        )
        limitations = [
            "diagnostic_only: this result is not benchmark or scientific evidence",
            "fidelity is exact within native planner/config/seed/dt/horizon/outcome/trace fingerprint scope",
            "stateful and checkpoint planner configurations are unsupported and fail closed",
        ]
        original_coverage = source_document.original_record.get("algorithm_metadata", {}).get(
            "analysis_trace_coverage"
        )
        if isinstance(original_coverage, Mapping) and original_coverage.get("status") != "complete":
            limitations.append(
                "original analysis trace optional coverage is incomplete: "
                + ", ".join(str(item) for item in original_coverage.get("reasons", []))
            )
        if activation_status != "verified":
            return _result_failure(
                validated.request_id,
                STATUS_FAILED,
                "intervention_not_observed: requested goal change was not measured in telemetry",
                provenance=provenance,
                limitations=limitations,
                original_identity=original_identity,
            )
        return NativeDiagnosticResult(
            request_id=validated.request_id,
            status=STATUS_COMPLETE,
            reason="native control fidelity verified and intervention activation measured",
            original_identity=original_identity,
            control_identity=control_identity,
            intervention_identity=intervention_identity,
            fidelity=fidelity,
            activation=activation,
            provenance=provenance,
            limitations=tuple(limitations),
        )
    except NativeDiagnosticError as error:
        reason = str(error)
        status = (
            STATUS_UNAVAILABLE
            if any(
                token in reason
                for token in ("source_admission", "receipt_stale", "original_telemetry_missing")
            )
            else STATUS_FAILED
        )
        return _result_failure(validated.request_id, status, reason, provenance=provenance)
    except (OSError, RuntimeError, TypeError, ValueError, ReviewContractsValidationError) as error:
        return _result_failure(
            validated.request_id,
            STATUS_UNAVAILABLE if isinstance(error, OSError) else STATUS_FAILED,
            f"native_diagnostic_error: {type(error).__name__}: {str(error)[:1024]}",
            provenance=provenance,
        )
    finally:
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass


def _exact_materialization_failure(
    *,
    episode_id: str,
    status: str,
    reason: str,
    provenance: Mapping[str, Any] | None = None,
    diagnostics: Sequence[str] = (),
    simulation_executed: bool = False,
) -> Any:
    """Build a materialization envelope without importing the renderer early."""

    from robot_sf.analysis_workbench.audit_materialize import (  # noqa: PLC0415
        FIDELITY_UNAVAILABLE,
        UNAVAILABLE,
        MaterializationResult,
    )

    cache_key = hashlib.sha256(
        _canonical_json(
            {
                "native_exact_input": True,
                "episode_id": episode_id,
                "reason": reason,
            }
        ).encode("utf-8")
    ).hexdigest()
    return MaterializationResult(
        status=status,
        materialization_kind=UNAVAILABLE,
        fidelity=FIDELITY_UNAVAILABLE,
        episode_id=episode_id,
        cache_key=cache_key,
        diagnostics=tuple(str(item)[:MAX_TEXT_CHARS] for item in diagnostics),
        provenance=dict(provenance or {}),
        reason=str(reason)[:MAX_TEXT_CHARS],
        simulation_executed=simulation_executed,
    )


def _original_telemetry_errors(errors: Sequence[str]) -> tuple[str, ...]:
    """Select missing historical telemetry that makes comparison unverifiable."""

    telemetry_markers = (
        "analysis trace is missing",
        "analysis trace artifact digest is missing",
        "episode metrics are missing",
        "episode outcome is missing",
    )
    return tuple(
        error for error in errors if any(marker in error.lower() for marker in telemetry_markers)
    )


def materialize_exact_input(
    request: NativeDiagnosticRequest | Mapping[str, Any],
    episode: Mapping[str, Any],
    *,
    output_root: Any,
    output_directory: str | None = None,
    render_config: Mapping[str, Any] | None = None,
    timeout_s: float | None = None,
    execution_id: str | None = None,
    cancel_event: Any = None,
) -> Any:
    """Regenerate one admitted simple-policy input and render its new trace.

    This is the only native execution path owned by the materializer seam.  It
    is intentionally separate from :func:`run_native_diagnostic`: no goal
    intervention is run, and the selected row must have no recording or
    retained states.  The launcher-owned source document supplies the exact
    closed runner input; the selected row can only corroborate identity.

    The returned value is a :class:`MaterializationResult` whose derived
    artifact contains only the newly generated trace.  Historical telemetry is
    compared when present, but is never copied into the selected row or the
    artifact.  Missing historical telemetry therefore yields
    ``fidelity='unverifiable'`` rather than fabricated evidence.
    """

    from robot_sf.analysis_workbench.audit_materialize import (  # noqa: PLC0415
        FIDELITY_DIVERGED,
        FIDELITY_UNVERIFIABLE,
        FIDELITY_VERIFIED,
        MaterializationResult,
        materialize_native_record,
    )

    request_id = request.request_id if isinstance(request, NativeDiagnosticRequest) else "invalid"
    selected_episode_id = episode.get("episode_id") if isinstance(episode, Mapping) else ""
    selected_episode_id = selected_episode_id if isinstance(selected_episode_id, str) else ""
    validated: NativeDiagnosticRequest
    try:
        validated = (
            request
            if isinstance(request, NativeDiagnosticRequest)
            else NativeDiagnosticRequest.from_mapping(request)
        )
        request_id = validated.request_id
        _validate_request(validated)
        timeout = validated.timeout_s if timeout_s is None else _finite(timeout_s, name="timeout_s")
        if timeout <= 0.0 or timeout > MAX_TIMEOUT_S:
            raise NativeDiagnosticError([f"timeout_s must be within (0, {MAX_TIMEOUT_S:g}]"])
        if execution_id is None:
            execution_id = f"{request_id}:exact:{uuid.uuid4().hex}"
        execution_id = _text(execution_id, name="execution_id")
    except (NativeDiagnosticError, TypeError, ValueError) as error:
        return _exact_materialization_failure(
            episode_id=selected_episode_id,
            status=STATUS_FAILED,
            reason=f"invalid_request: {str(error)[:1024]}",
        )
    if cancel_event is not None and cancel_event.is_set():
        return _exact_materialization_failure(
            episode_id=selected_episode_id,
            status=STATUS_CANCELLED,
            reason="cancelled_by_user: before exact-input admission",
            diagnostics=("simulation_not_started",),
        )

    root_fd: int | None = None
    simulation_executed = False
    provenance: dict[str, Any] = {
        "component_id": COMPONENT_ID,
        "component_version": NATIVE_DIAGNOSTIC_SCHEMA_VERSION,
        "canonical_runner": "robot_sf.benchmark.runner.run_episode",
        "route": "native_exact_input",
        "planner": "simple_policy",
        "execution_id": execution_id,
        "execution_kind": "exact_input_regeneration",
        "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
        "diagnostic_only": True,
        "original_telemetry_fabricated": False,
        "checkpoint_identity": None,
        "child_process": "multiprocessing.spawn",
    }
    try:
        root, root_fd, _receipt_payload, receipt_digest, resolution = _open_and_resolve_source(
            validated
        )
        source_document = _source_document(
            resolution.source_bytes or b"", resolution.receipt.source
        )
        _validate_source_recipe_identity(source_document, validated)
        source_commit = resolution.receipt.source.source_commit
        source_config = resolution.receipt.source.config_identity
        selected_errors = _selected_exact_input_errors(
            episode,
            source_document,
            source_digest=resolution.receipt.source.sha256,
        )
        if selected_errors:
            return _exact_materialization_failure(
                episode_id=selected_episode_id,
                status=STATUS_UNAVAILABLE,
                reason="exact_input_identity_mismatch",
                provenance=provenance,
                diagnostics=selected_errors,
            )
        historical_id = source_document.original_record.get("episode_id")
        historical_provenance = source_document.original_record.get("provenance")
        historical_native = (
            historical_provenance.get("native_diagnostic")
            if isinstance(historical_provenance, Mapping)
            else None
        )
        historical_execution_id = (
            historical_native.get("execution_id")
            if isinstance(historical_native, Mapping)
            else None
        ) or historical_id
        provenance.update(
            {
                "historical_episode_id": historical_id,
                "historical_execution_id": historical_execution_id,
                "linked_historical_episode_id": historical_id,
                "source_commit": source_commit,
                "config_identity": source_config,
                "initial_state_sha256": source_document.identity["initial_state_sha256"],
                "environment_identity": source_document.identity["environment_identity"],
                "source_admission": _source_admission_provenance(
                    resolution, receipt_digest, validated
                ),
                "input_digests": {
                    "source_bytes_sha256": resolution.receipt.source.sha256,
                    "runner_input_sha256": _digest(source_document.runner_input),
                },
            }
        )
        original_errors = _record_execution_errors(
            source_document.original_record,
            source_document.runner_input,
            source_commit=source_commit,
            source_identity=source_document.identity,
            expected_role=ROLE_HISTORICAL_ORIGINAL,
            source_id=source_document.source_id,
        )
        telemetry_errors = _original_telemetry_errors(original_errors)
        hard_errors = tuple(error for error in original_errors if error not in telemetry_errors)
        if hard_errors:
            return _exact_materialization_failure(
                episode_id=selected_episode_id,
                status=STATUS_UNAVAILABLE,
                reason="historical_source_ineligible",
                provenance=provenance,
                diagnostics=hard_errors,
            )
        runner_input = source_document.runner_input
        native_provenance = {
            "native_diagnostic": {
                "request_id": request_id,
                "identity": ROLE_EXACT_INPUT_REGENERATION,
                "source_id": source_document.source_id,
                "source_commit": source_commit,
                "config_identity": source_config,
                "source_identity": _record_identity_proof(source_document.identity),
                "runner_input_identity": _runner_input_identity(runner_input),
                "execution_id": execution_id,
                "historical_episode_id": historical_id,
                "historical_execution_id": historical_execution_id,
            }
        }
        generated = _run_bounded(
            runner_input,
            robot_goal=runner_input["robot_goal"],
            provenance=native_provenance,
            timeout_s=timeout,
            cancel_event=cancel_event,
        )
        if generated.get("status") == STATUS_CANCELLED:
            return _exact_materialization_failure(
                episode_id=selected_episode_id,
                status=STATUS_CANCELLED,
                reason=str(generated.get("reason", "cancelled_by_user")),
                provenance=provenance,
                diagnostics=("simulation_started",),
                simulation_executed=True,
            )
        simulation_executed = True
        if cancel_event is not None and cancel_event.is_set():
            return _exact_materialization_failure(
                episode_id=selected_episode_id,
                status=STATUS_CANCELLED,
                reason="cancelled_by_user: after exact-input child",
                provenance=provenance,
                diagnostics=("simulation_started",),
                simulation_executed=True,
            )
        if generated.get("status") != "ok" or not isinstance(generated.get("record"), Mapping):
            return _exact_materialization_failure(
                episode_id=selected_episode_id,
                status=(
                    STATUS_UNAVAILABLE
                    if generated.get("status") == STATUS_UNAVAILABLE
                    else STATUS_FAILED
                ),
                reason="exact_input_execution_failed",
                provenance=provenance,
                diagnostics=(str(generated.get("reason", "canonical runner did not complete")),),
                simulation_executed=True,
            )
        generated_record = dict(generated["record"])
        generated_errors = _record_execution_errors(
            generated_record,
            runner_input,
            source_commit=source_commit,
            source_identity=source_document.identity,
            expected_role=ROLE_EXACT_INPUT_REGENERATION,
            request_id=request_id,
            source_id=source_document.source_id,
        )
        if generated_errors:
            return _exact_materialization_failure(
                episode_id=selected_episode_id,
                status=STATUS_FAILED,
                reason="exact_input_record_invalid",
                provenance=provenance,
                diagnostics=generated_errors,
                simulation_executed=True,
            )
        final_admission, _ = _final_source_check(
            root, root_fd, validated, resolution.receipt.source.sha256
        )
        provenance["source_admission"] = final_admission
        provenance["runner_elapsed_s"] = generated.get("elapsed_s")
        if telemetry_errors:
            fidelity = FIDELITY_UNVERIFIABLE
            fidelity_document = {
                "status": FIDELITY_UNVERIFIABLE,
                "differences": list(telemetry_errors),
                "historical_telemetry_available": False,
                "original_fingerprint": _record_fingerprint(source_document.original_record),
                "generated_fingerprint": _record_fingerprint(generated_record),
            }
        else:
            fidelity_document = _compare_control_fidelity(
                source_document.original_record, generated_record
            )
            fidelity = str(fidelity_document.get("status", FIDELITY_UNVERIFIABLE))
            if fidelity not in {FIDELITY_VERIFIED, FIDELITY_DIVERGED, FIDELITY_UNVERIFIABLE}:
                fidelity = FIDELITY_UNVERIFIABLE
        provenance["fidelity"] = fidelity_document
        provenance["simulation_advanced"] = True
        render_diagnostics = ["native_exact_input_regenerated", f"fidelity:{fidelity}"]
        if telemetry_errors:
            render_diagnostics.append("historical_telemetry_unavailable")
        render_result = materialize_native_record(
            episode,
            generated_record,
            output_root=output_root,
            output_directory=output_directory,
            render_config=render_config,
            fidelity=fidelity,
            diagnostics=render_diagnostics,
            provenance=provenance,
        )
        if cancel_event is not None and cancel_event.is_set():
            return _exact_materialization_failure(
                episode_id=selected_episode_id,
                status=STATUS_CANCELLED,
                reason="cancelled_by_user: after exact-input render",
                provenance=provenance,
                diagnostics=("simulation_started", "derived_artifact_published"),
                simulation_executed=True,
            )
        if not isinstance(render_result, MaterializationResult):
            return _exact_materialization_failure(
                episode_id=selected_episode_id,
                status=STATUS_FAILED,
                reason="native_renderer_returned_malformed_result",
                provenance=provenance,
                simulation_executed=True,
            )
        return render_result
    except NativeDiagnosticError as error:
        message = str(error)
        status = STATUS_UNAVAILABLE if "source_admission" in message else STATUS_FAILED
        return _exact_materialization_failure(
            episode_id=selected_episode_id,
            status=status,
            reason=message,
            provenance=provenance,
            simulation_executed=simulation_executed,
        )
    except (OSError, RuntimeError, TypeError, ValueError, ReviewContractsValidationError) as error:
        return _exact_materialization_failure(
            episode_id=selected_episode_id,
            status=STATUS_UNAVAILABLE if isinstance(error, OSError) else STATUS_FAILED,
            reason=f"native_exact_input_error: {type(error).__name__}: {str(error)[:1024]}",
            provenance=provenance,
            simulation_executed=simulation_executed,
        )
    finally:
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass


# Short aliases make the adapter straightforward to route from service/CLI/MCP
# integrations without creating another execution owner.
run = run_native_diagnostic
execute = run_native_diagnostic
regenerate_exact_input = materialize_exact_input
run_exact_input_materialization = materialize_exact_input


__all__ = [
    "COMPONENT_ID",
    "DEPENDENT_FAMILY_STATUS",
    "DIAGNOSTIC_EVIDENCE_BOUNDARY",
    "NATIVE_DIAGNOSTIC_SCHEMA_VERSION",
    "NATIVE_SOURCE_FORMAT",
    "NATIVE_SOURCE_SCHEMA_VERSION",
    "ROLE_DIAGNOSTIC_CONTROL",
    "ROLE_DIAGNOSTIC_INTERVENTION",
    "ROLE_EXACT_INPUT_REGENERATION",
    "ROLE_HISTORICAL_ORIGINAL",
    "STATUS_CANCELLED",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_UNAVAILABLE",
    "NativeDiagnosticAdmission",
    "NativeDiagnosticError",
    "NativeDiagnosticRequest",
    "NativeDiagnosticResult",
    "execute",
    "materialize_exact_input",
    "regenerate_exact_input",
    "run",
    "run_exact_input_materialization",
    "run_native_diagnostic",
]
