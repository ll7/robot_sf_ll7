"""Deterministic, redacted environment manifests for expiring platform classes (issue #8894).

A source commit and ``uv.lock`` do not fully describe platform-specific Robot SF execution:
GPU drivers, CUDA runtimes, host architecture, companion packages, thread controls, and scheduler
context can decide whether a preserved campaign or checkpoint stays executable. This module is the
canonical read-only owner for capturing that state for one explicitly selected platform class:

- every field carries a distinct status: ``observed``, ``unavailable``, ``redacted``, or
  ``declared``;
- captures are canonicalised so repeated captures with identical probes share one
  ``semantic_digest`` while volatile timestamps stay outside the digest;
- environment variables are only read from explicit allowlists (thread controls and scheduler
  context);
- home directories, hostnames, private cluster roots, credential-like values, signed-URL
  parameters, and scheduler account/partition identities are redacted before they can enter
  tracked output.

Capture and check paths live in ``scripts/tools/capture_environment_manifest.py``. Nothing in this
module installs packages, creates environments, mutates the repository, or runs a campaign.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf._numerical_thread_env import THREAD_ENV_VARS
from robot_sf.evidence.writers import sha256_file

ENVIRONMENT_MANIFEST_SCHEMA_VERSION = "environment_manifest.v1"

STATUS_OBSERVED = "observed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_REDACTED = "redacted"
STATUS_DECLARED = "declared"

REASON_FILE_MISSING = "file_missing"
REASON_GIT_UNAVAILABLE = "git_unavailable"
REASON_NOT_A_REPOSITORY = "not_a_repository"
REASON_NOT_SET = "not_set"
REASON_PROBE_DEPENDENCY_MISSING = "probe_dependency_missing"
REASON_CUDA_RUNTIME_MISSING = "cuda_runtime_missing"
REASON_CUDA_UNAVAILABLE = "cuda_unavailable"
REASON_DRIVER_UNAVAILABLE = "driver_unavailable"
REASON_DRIVER_PROBE_UNAVAILABLE = "driver_probe_unavailable"
REASON_COMPANION_PACKAGE_NOT_INSTALLED = "companion_package_not_installed"
REASON_SCHEDULER_CONTEXT_MISSING = "scheduler_context_missing"
REASON_SCHEDULER_IDENTITY_REDACTED = "scheduler_identity_redacted"
REASON_PRIVATE_OR_CREDENTIAL_CONTENT = "private_or_credential_content"
REASON_UNKNOWN_PLATFORM_CLASS = "unknown_platform_class"
REASON_PLATFORM_CLASS_MISMATCH = "platform_class_mismatch"
REASON_ACCELERATOR_UNAVAILABLE = "accelerator_unavailable"
REASON_CARLA_UNAVAILABLE = "carla_unavailable"
REASON_SCHEMA_VERSION_MISMATCH = "schema_version_mismatch"

THREAD_CONTROL_ENV_VARS = (*THREAD_ENV_VARS, "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS")

SCHEDULER_IDENTITY_VARS = {
    "SLURM_JOB_ID": "job_id",
    "SLURM_JOB_PARTITION": "partition",
    "SLURM_JOB_ACCOUNT": "account",
}
SCHEDULER_OBSERVED_VARS = {
    "SLURM_CPUS_PER_TASK": "cpus_per_task",
    "SLURM_JOB_NUM_NODES": "nodes",
}

COMPANION_DISTRIBUTIONS = {
    "pysocialforce": "PySocialForce",
    "carla": "carla",
}

GPU_ACCELERATOR_CLASSES = frozenset({"cuda", "rocm", "mps"})

_REPOSITORY_IDENTITY_FILES = ("uv.lock", "pyproject.toml")


@dataclass(frozen=True)
class PlatformContract:
    """Compatibility contract for one registered platform class."""

    name: str
    description: str
    requires_accelerator: bool = False
    requires_carla: bool = False


PLATFORM_CONTRACTS: dict[str, PlatformContract] = {
    "local_cpu": PlatformContract("local_cpu", "Local workstation or laptop without a GPU claim."),
    "local_gpu": PlatformContract(
        "local_gpu",
        "Local host with one or more usable GPU accelerators.",
        requires_accelerator=True,
    ),
    "slurm_cpu": PlatformContract("slurm_cpu", "SLURM batch allocation without a GPU claim."),
    "slurm_gpu": PlatformContract(
        "slurm_gpu",
        "SLURM batch allocation with one or more usable GPU accelerators.",
        requires_accelerator=True,
    ),
    "carla": PlatformContract(
        "carla",
        "CARLA simulation host with the companion CARLA package available.",
        requires_carla=True,
    ),
}


@dataclass(frozen=True)
class RedactionContext:
    """Inputs for strict private-data redaction."""

    home: str | None = None
    hostname: str | None = None
    private_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentProbes:
    """Injectable probe results so captures are testable and deterministic.

    ``None`` entries fall back to the real local probes. Every value must already be a manifest
    field mapping (``{"status": ..., "value": ...}``) so tests can construct edge cases such as a
    missing CUDA driver without importing the GPU stack.
    """

    accelerator: Mapping[str, Any] | None = None
    companions: Mapping[str, Mapping[str, Any]] | None = None
    env: Mapping[str, str] | None = None
    scheduler_env: Mapping[str, str] | None = None
    hostname: str | None = None
    home: str | None = None
    git_commit: Mapping[str, Any] | None = None
    clock: Callable[[], str] | None = None
    declared: Mapping[str, str] = field(default_factory=dict)


def observed(value: Any) -> dict[str, Any]:
    """Return an observed field."""
    return {"status": STATUS_OBSERVED, "value": value}


def unavailable(reason: str, *, detail: str | None = None) -> dict[str, Any]:
    """Return an unavailable field with a stable reason code."""
    field_value: dict[str, Any] = {"status": STATUS_UNAVAILABLE, "reason": reason}
    if detail is not None:
        field_value["detail"] = detail
    return field_value


def redacted(reason: str, *, digest: str | None = None, value: Any = None) -> dict[str, Any]:
    """Return a redacted field; ``value`` may carry the sanitized remainder only."""
    field_value: dict[str, Any] = {"status": STATUS_REDACTED, "reason": reason}
    if digest is not None:
        field_value["digest"] = digest
    if value is not None:
        field_value["value"] = value
    return field_value


def declared(value: Any) -> dict[str, Any]:
    """Return a caller-declared field."""
    return {"status": STATUS_DECLARED, "value": value}


def _default_clock() -> str:
    return datetime.now(UTC).isoformat()


_CREDENTIAL_URL_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+:[^/@\s]+@")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(token|secret|password|passwd|api[_-]?key|access[_-]?key)\b"
    r"(\s*[=:]\s*)([^\s&,;\"']+)"
)
_SIGNED_URL_PARAM_RE = re.compile(
    r"(?i)([?&](?:x-amz-signature|x-amz-credential|signature|sig|token|expires|credential)=)([^&\s]+)"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(token|secret|password|passwd|api[_-]?key|access[_-]?key|credential)"
)
_HOME_PREFIX_RE = re.compile(r"(?<![\w/.])/(?:home|Users)/[^/\s]+")
_PRIVATE_ROOT_RE = re.compile(
    r"(?<![\w.])/(?:net|scratch|work|global|gpfs|cluster|projects?)/[^\s\"']*"
)


def digest_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for a redacted identity value."""
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def sanitize_text(text: str, context: RedactionContext) -> tuple[str, bool]:
    """Redact private or credential-like content from ``text``.

    Returns:
        The sanitized text and whether any replacement changed it.
    """
    sanitized = text
    if context.home:
        sanitized = sanitized.replace(context.home, "<home>")
    if context.hostname and len(context.hostname) > 1:
        sanitized = sanitized.replace(context.hostname, "<host>")
    for root in context.private_roots:
        if root:
            sanitized = sanitized.replace(root, "<private-root>")
    sanitized = _CREDENTIAL_URL_RE.sub(r"\g<scheme><redacted>@", sanitized)
    sanitized = _SECRET_ASSIGNMENT_RE.sub(r"\1\2<redacted>", sanitized)
    sanitized = _SIGNED_URL_PARAM_RE.sub(r"\1<redacted>", sanitized)
    sanitized = _EMAIL_RE.sub("<email>", sanitized)
    sanitized = _HOME_PREFIX_RE.sub("<home>", sanitized)
    sanitized = _PRIVATE_ROOT_RE.sub("<private-root>", sanitized)
    return sanitized, sanitized != text


def sanitize_value(value: Any, context: RedactionContext) -> tuple[Any, bool]:
    """Recursively redact string content in a JSON-serialisable value.

    Returns:
        The sanitized value and whether any replacement changed it.
    """
    if isinstance(value, str):
        return sanitize_text(value, context)
    if isinstance(value, Mapping):
        changed = False
        sanitized_mapping: dict[Any, Any] = {}
        for key, item in value.items():
            sanitized_item, item_changed = sanitize_value(item, context)
            sanitized_mapping[key] = sanitized_item
            changed = changed or item_changed
        return sanitized_mapping, changed
    if isinstance(value, (list, tuple)):
        changed = False
        sanitized_items = []
        for item in value:
            sanitized_item, item_changed = sanitize_value(item, context)
            sanitized_items.append(sanitized_item)
            changed = changed or item_changed
        return sanitized_items, changed
    return value, False


def observed_sanitized(value: Any, context: RedactionContext) -> dict[str, Any]:
    """Return an observed field, downgraded to redacted when sanitization changed it."""
    sanitized, changed = sanitize_value(value, context)
    if changed:
        return redacted(REASON_PRIVATE_OR_CREDENTIAL_CONTENT, value=sanitized)
    return observed(sanitized)


def declared_sanitized(key: str, value: str, context: RedactionContext) -> dict[str, Any]:
    """Return a declared field, redacting sensitive key names or content."""
    if _SENSITIVE_KEY_RE.search(key):
        return redacted(REASON_PRIVATE_OR_CREDENTIAL_CONTENT, digest=digest_text(value))
    return observed_sanitized(value, context)


def probe_git_commit(repo_root: Path) -> dict[str, Any]:
    """Probe the repository commit without modifying the checkout.

    Returns:
        An observed commit field, or an unavailable field with a stable reason.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return unavailable(REASON_GIT_UNAVAILABLE)
    except subprocess.TimeoutExpired:
        return unavailable(REASON_GIT_UNAVAILABLE, detail="timeout")
    if result.returncode != 0:
        return unavailable(REASON_NOT_A_REPOSITORY)
    return observed(result.stdout.strip())


def collect_repository_identity(
    repo_root: Path,
    *,
    git_commit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect commit and lockfile/pyproject digests for the checkout.

    Returns:
        A section mapping commit and repository file-digest fields.
    """
    commit_field = dict(git_commit) if git_commit is not None else probe_git_commit(repo_root)
    section: dict[str, Any] = {"commit": commit_field}
    for name in _REPOSITORY_IDENTITY_FILES:
        path = repo_root / name
        if not path.is_file():
            section[f"{name}_sha256"] = unavailable(REASON_FILE_MISSING, detail=name)
            continue
        section[f"{name}_sha256"] = observed(f"sha256:{sha256_file(path)}")
    return section


def probe_accelerator() -> dict[str, Any]:
    """Probe the local accelerator stack through PyTorch, without importing it on CPU classes.

    Returns:
        A section mapping accelerator class, library, runtime, driver, and devices.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {
            "class": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
            "library": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
            "runtime_version": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
            "driver_version": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
            "devices": unavailable(REASON_PROBE_DEPENDENCY_MISSING),
            "reason": REASON_PROBE_DEPENDENCY_MISSING,
        }
    library = observed("torch")
    runtime_version = (
        observed(torch.version.cuda)
        if getattr(torch.version, "cuda", None)
        else unavailable(REASON_CUDA_RUNTIME_MISSING)
    )
    if torch.cuda.is_available():
        devices = [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(index)
                ),
                "total_memory_mb": torch.cuda.get_device_properties(index).total_memory
                // (1024 * 1024),
            }
            for index in range(torch.cuda.device_count())
        ]
        return {
            "class": observed("cuda"),
            "library": library,
            "runtime_version": runtime_version,
            "driver_version": _probe_cuda_driver_version(),
            "devices": observed(devices),
        }
    reason = (
        REASON_DRIVER_UNAVAILABLE
        if getattr(torch.version, "cuda", None)
        else REASON_CUDA_UNAVAILABLE
    )
    return {
        "class": unavailable(reason),
        "library": library,
        "runtime_version": runtime_version,
        "driver_version": unavailable(reason),
        "devices": unavailable(reason),
        "reason": reason,
    }


def _probe_cuda_driver_version() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return unavailable(REASON_DRIVER_PROBE_UNAVAILABLE)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return unavailable(REASON_DRIVER_PROBE_UNAVAILABLE)
    if result.returncode != 0 or not result.stdout.strip():
        return unavailable(REASON_DRIVER_PROBE_UNAVAILABLE)
    return observed(result.stdout.strip().splitlines()[0].strip())


def probe_companions() -> dict[str, dict[str, Any]]:
    """Probe companion packages without importing them.

    Returns:
        A mapping of companion module name to observed or unavailable field.
    """
    companions: dict[str, dict[str, Any]] = {}
    for module_name, distribution_name in COMPANION_DISTRIBUTIONS.items():
        try:
            found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            found = False
        if not found:
            companions[module_name] = unavailable(REASON_COMPANION_PACKAGE_NOT_INSTALLED)
            continue
        try:
            version: str | None = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        companions[module_name] = observed(
            {"module": module_name, "distribution": distribution_name, "version": version}
        )
    return companions


def collect_runtime_identity(context: RedactionContext) -> dict[str, Any]:
    """Collect OS, Python, and host identity fields with redaction applied.

    Returns:
        A section mapping runtime identity fields.
    """
    return {
        "python_implementation": observed(platform.python_implementation()),
        "python_version": observed(platform.python_version()),
        "platform_system": observed(platform.system()),
        "platform_machine": observed(platform.machine()),
        "platform_release": observed_sanitized(platform.release(), context),
        "hostname": redacted(
            REASON_PRIVATE_OR_CREDENTIAL_CONTENT,
            digest=digest_text(context.hostname) if context.hostname else None,
        ),
    }


def collect_package_identity(
    companions: Mapping[str, Mapping[str, Any]],
    context: RedactionContext,
) -> dict[str, Any]:
    """Collect distribution identity and companion-package availability.

    Returns:
        A section mapping the distribution version and companion fields.
    """
    try:
        version: str | None = importlib.metadata.version("robot_sf")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "distribution_version": observed(version) if version else unavailable("not_installed"),
        "companions": sanitize_value(dict(companions), context)[0],
    }


def collect_scheduler_identity(
    scheduler_env: Mapping[str, str],
    context: RedactionContext,
) -> dict[str, Any]:
    """Collect scheduler context while redacting account/partition identity.

    Returns:
        A section mapping scheduler detection and redacted identity fields.
    """
    detected = any(
        name in scheduler_env for name in (*SCHEDULER_IDENTITY_VARS, *SCHEDULER_OBSERVED_VARS)
    )
    section: dict[str, Any] = {"detected": observed(detected)}
    if not detected:
        section["reason"] = REASON_SCHEDULER_CONTEXT_MISSING
    for env_name, key in SCHEDULER_IDENTITY_VARS.items():
        raw = scheduler_env.get(env_name)
        if raw is None:
            section[key] = unavailable(REASON_NOT_SET)
        else:
            section[key] = redacted(REASON_SCHEDULER_IDENTITY_REDACTED, digest=digest_text(raw))
    for env_name, key in SCHEDULER_OBSERVED_VARS.items():
        raw = scheduler_env.get(env_name)
        section[key] = (
            observed_sanitized(raw, context) if raw is not None else unavailable(REASON_NOT_SET)
        )
    return section


def collect_thread_controls(env: Mapping[str, str], context: RedactionContext) -> dict[str, Any]:
    """Collect allowlisted numerical thread controls.

    Returns:
        A mapping of thread-control variable name to observed or unset field.
    """
    return {
        name: observed_sanitized(env[name], context) if name in env else unavailable(REASON_NOT_SET)
        for name in THREAD_CONTROL_ENV_VARS
    }


def build_environment_manifest(
    platform_class: str,
    *,
    repo_root: Path,
    declared_values: Mapping[str, str] | None = None,
    probes: EnvironmentProbes | None = None,
) -> dict[str, Any]:
    """Build one deterministic environment manifest for ``platform_class``.

    Returns:
        The manifest mapping including schema version, sections, and semantic digest.

    Raises:
        ValueError: When ``platform_class`` is not registered.
    """
    if platform_class not in PLATFORM_CONTRACTS:
        raise ValueError(
            f"{REASON_UNKNOWN_PLATFORM_CLASS}: {platform_class!r} is not registered; "
            f"known classes: {', '.join(sorted(PLATFORM_CONTRACTS))}"
        )
    probes = probes or EnvironmentProbes()
    env = probes.env if probes.env is not None else {}
    scheduler_env = probes.scheduler_env if probes.scheduler_env is not None else {}
    hostname = probes.hostname if probes.hostname is not None else platform.node()
    home = probes.home if probes.home is not None else str(Path.home())
    context = RedactionContext(home=home, hostname=hostname)
    clock = probes.clock or _default_clock

    accelerator = (
        dict(probes.accelerator) if probes.accelerator is not None else probe_accelerator()
    )
    companions = dict(probes.companions) if probes.companions is not None else probe_companions()
    declared_overrides = {
        key: declared_sanitized(key, value, context)
        for key, value in (declared_values or {}).items()
    }

    manifest: dict[str, Any] = {
        "schema_version": ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
        "platform_class": platform_class,
        "captured_at_utc": clock(),
        "declared": {
            "platform_class": declared(platform_class),
            "overrides": declared_overrides,
        },
        "repository": collect_repository_identity(repo_root, git_commit=probes.git_commit),
        "runtime": collect_runtime_identity(context),
        "packages": collect_package_identity(companions, context),
        "accelerator": accelerator,
        "scheduler": collect_scheduler_identity(scheduler_env, context),
        "thread_controls": collect_thread_controls(env, context),
    }
    manifest["semantic_digest"] = semantic_digest(manifest)
    return manifest


def semantic_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the digest-relevant payload, excluding volatile fields."""
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"captured_at_utc", "semantic_digest"}
    }


def semantic_digest(manifest: Mapping[str, Any]) -> str:
    """Return the canonical digest over the semantic payload."""
    canonical = json.dumps(semantic_payload(manifest), sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def evaluate_environment_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_platform_class: str,
) -> dict[str, Any]:
    """Evaluate a manifest against a registered platform class.

    Returns:
        Mapping with ``compatible``, ordered stable ``reasons``, and both platform classes.
    """
    reasons: list[str] = []
    contract = PLATFORM_CONTRACTS.get(expected_platform_class)
    if contract is None:
        reasons.append(REASON_UNKNOWN_PLATFORM_CLASS)
    if manifest.get("schema_version") != ENVIRONMENT_MANIFEST_SCHEMA_VERSION:
        reasons.append(REASON_SCHEMA_VERSION_MISMATCH)
    if manifest.get("platform_class") != expected_platform_class:
        reasons.append(REASON_PLATFORM_CLASS_MISMATCH)
    accelerator = manifest.get("accelerator")
    if contract is not None and contract.requires_accelerator and isinstance(accelerator, Mapping):
        accelerator_class = accelerator.get("class")
        observed_gpu = (
            isinstance(accelerator_class, Mapping)
            and accelerator_class.get("status") == STATUS_OBSERVED
            and accelerator_class.get("value") in GPU_ACCELERATOR_CLASSES
        )
        if not observed_gpu:
            reasons.append(REASON_ACCELERATOR_UNAVAILABLE)
            probe_reason = accelerator.get("reason")
            if isinstance(probe_reason, str) and probe_reason not in reasons:
                reasons.append(probe_reason)
    if contract is not None and contract.requires_carla:
        companions = manifest.get("packages", {}).get("companions", {})
        carla_field = companions.get("carla") if isinstance(companions, Mapping) else None
        if not (isinstance(carla_field, Mapping) and carla_field.get("status") == STATUS_OBSERVED):
            reasons.append(REASON_CARLA_UNAVAILABLE)
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "expected_platform_class": expected_platform_class,
        "manifest_platform_class": manifest.get("platform_class"),
        "semantic_digest": manifest.get("semantic_digest"),
    }


__all__ = [
    "ENVIRONMENT_MANIFEST_SCHEMA_VERSION",
    "PLATFORM_CONTRACTS",
    "STATUS_DECLARED",
    "STATUS_OBSERVED",
    "STATUS_REDACTED",
    "STATUS_UNAVAILABLE",
    "THREAD_CONTROL_ENV_VARS",
    "EnvironmentProbes",
    "PlatformContract",
    "RedactionContext",
    "build_environment_manifest",
    "collect_package_identity",
    "collect_repository_identity",
    "collect_runtime_identity",
    "collect_scheduler_identity",
    "collect_thread_controls",
    "declared",
    "declared_sanitized",
    "digest_text",
    "evaluate_environment_manifest",
    "observed",
    "observed_sanitized",
    "probe_accelerator",
    "probe_companions",
    "probe_git_commit",
    "redacted",
    "sanitize_text",
    "sanitize_value",
    "semantic_digest",
    "semantic_payload",
    "unavailable",
]
