"""Execute the canonical proof surfaces declared by a research manifest.

This module is a thin adapter for the existing campaign-manifest runner.  It
does not define a second admission contract: the answerability contract owns
the six proof surfaces, while this module invokes the available validators and
records unavailable ones explicitly.  All commands are argv lists and run
without a shell so a manifest cannot turn the preflight into an arbitrary
shell-expansion path.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.artifact_catalog import load_artifact_catalog, validate_artifact_catalog
from robot_sf.benchmark.research_answerability import PROOF_SURFACES
from scripts.validation.check_preregistration_inference_contract import (
    InferenceContractError,
    check_yaml_file,
)

_CHECK_KINDS = {
    "artifact_catalog",
    "command",
    "durable_path",
    "evidence_contract",
    "manifest_rows",
    "preregistration",
    "result_packet",
}
_MAX_OUTPUT_CHARS = 1200
_MAX_VALIDATOR_TIMEOUT_SECONDS = 120.0
_REGISTERED_VALIDATOR_IDS = {"pytest_contract"}
_PYTEST_FLAGS = {"-q", "-v", "--disable-warnings"}
_SHA256_RE = r"^[0-9a-f]{64}$"


class AnswerabilityProofError(ValueError):
    """Raised when a manifest declares an invalid executable proof check."""


def _repo_relative_path(repo_root: Path, value: Any, field: str) -> Path:
    """Resolve a safe repository-relative path declared by a manifest."""
    if not isinstance(value, str) or not value.strip():
        raise AnswerabilityProofError(f"{field} must be a non-empty repository-relative path")
    path = Path(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise AnswerabilityProofError(f"{field} must not be absolute or traverse '..'")
    return repo_root / path


def _argv(value: Any, field: str) -> list[str]:
    """Validate an argv-style command without accepting shell syntax."""
    if not isinstance(value, list) or not value:
        raise AnswerabilityProofError(f"{field} must be a non-empty argv list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise AnswerabilityProofError(f"{field} must contain only non-empty strings")
    return [item.strip() for item in value]


def _tail(value: str) -> str:
    """Keep command diagnostics compact and deterministic."""
    return value[-_MAX_OUTPUT_CHARS:]


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one proof input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_capable(manifest: Mapping[str, Any]) -> bool:
    """Return whether this collection is for decision-capable admission."""
    answerability = manifest.get("answerability")
    if not isinstance(answerability, Mapping):
        return False
    design = answerability.get("design")
    return isinstance(design, Mapping) and design.get("mode") == "decision_capable"


def _file_identity_failure(
    spec: Mapping[str, Any],
    *,
    path: Path,
    repo_root: Path,
    required: bool,
    kind: str,
    decision_capable: bool,
) -> dict[str, Any] | None:
    """Return a failed result when a required decision proof lacks a digest."""
    if not (decision_capable and required):
        return None
    expected = spec.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(_SHA256_RE, expected.lower()):
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason=f"{kind} proof requires an expected 64-hex sha256 digest",
            path=str(path.relative_to(repo_root)),
        )
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason=f"could not hash {kind} proof input: {exc}",
            path=str(path.relative_to(repo_root)),
        )
    if actual != expected.lower():
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason=f"{kind} proof input sha256 does not match its declared digest",
            path=str(path.relative_to(repo_root)),
            expected_sha256=expected.lower(),
            actual_sha256=actual,
        )
    return None


def _validate_registered_command(  # noqa: C901
    command: list[str], *, spec: Mapping[str, Any], field: str
) -> tuple[float, str | None]:
    """Validate one bounded command against the registered CPU-only test adapter."""
    validator_id = spec.get("validator_id")
    if validator_id not in _REGISTERED_VALIDATOR_IDS:
        raise AnswerabilityProofError(
            f"{field}.validator_id must be one of {sorted(_REGISTERED_VALIDATOR_IDS)}"
        )
    is_pytest = command[:3] == ["uv", "run", "pytest"] or (
        len(command) >= 3 and command[0] == sys.executable and command[1:3] == ["-m", "pytest"]
    )
    if not is_pytest:
        raise AnswerabilityProofError(
            f"{field}.command must use the registered pytest validator without shell execution"
        )
    paths: list[str] = []
    for item in command[3:]:
        if item.startswith("-"):
            if item not in _PYTEST_FLAGS:
                raise AnswerabilityProofError(
                    f"{field}.command contains an unsupported pytest flag: {item}"
                )
            continue
        path = Path(item)
        if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("tests",):
            raise AnswerabilityProofError(
                f"{field}.command test paths must stay under tests/: {item}"
            )
        if path.suffix != ".py":
            raise AnswerabilityProofError(
                f"{field}.command test paths must name Python files: {item}"
            )
        paths.append(item)
    if not paths:
        raise AnswerabilityProofError(f"{field}.command must name at least one tests/*.py path")
    timeout = spec.get("timeout_seconds", _MAX_VALIDATOR_TIMEOUT_SECONDS)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise AnswerabilityProofError(f"{field}.timeout_seconds must be a number")
    if not 0 < float(timeout) <= _MAX_VALIDATOR_TIMEOUT_SECONDS:
        raise AnswerabilityProofError(
            f"{field}.timeout_seconds must be in (0, {_MAX_VALIDATOR_TIMEOUT_SECONDS}]"
        )
    return float(timeout), str(validator_id)


def _result(
    *,
    status: str,
    required: bool,
    kind: str | None = None,
    reason: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Build one JSON-safe proof result."""
    payload: dict[str, Any] = {"status": status, "required": required}
    if kind is not None:
        payload["kind"] = kind
    if reason:
        payload["reason"] = reason
    payload.update(details)
    return payload


def _run_command(
    command: list[str],
    *,
    repo_root: Path,
    required: bool,
    kind: str,
    timeout_seconds: float,
    validator_id: str,
) -> dict[str, Any]:
    """Run one registered argv validator with a bounded wall-clock timeout."""
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason=f"registered validator exceeded {timeout_seconds:g}s timeout",
            command=command,
            validator_id=validator_id,
            timeout_seconds=timeout_seconds,
            stdout_excerpt=_tail(str(exc.stdout or "")),
            stderr_excerpt=_tail(str(exc.stderr or "")),
        )
    except OSError as exc:
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason=f"could not execute proof command: {exc}",
            command=command,
            validator_id=validator_id,
            timeout_seconds=timeout_seconds,
        )
    if completed.returncode == 0:
        return _result(
            status="passed",
            required=required,
            kind=kind,
            command=command,
            validator_id=validator_id,
            timeout_seconds=timeout_seconds,
            returncode=completed.returncode,
            stdout_excerpt=_tail(completed.stdout),
            stderr_excerpt=_tail(completed.stderr),
        )
    return _result(
        status="failed",
        required=required,
        kind=kind,
        reason=f"proof command exited with status {completed.returncode}",
        command=command,
        validator_id=validator_id,
        timeout_seconds=timeout_seconds,
        returncode=completed.returncode,
        stdout_excerpt=_tail(completed.stdout),
        stderr_excerpt=_tail(completed.stderr),
    )


def _run_preregistration(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    required: bool,
    decision_capable: bool,
) -> dict[str, Any]:
    """Invoke the public inference-contract checker for one YAML path."""
    path = _repo_relative_path(repo_root, spec.get("path"), "preregistration.path")
    identity_failure = _file_identity_failure(
        spec,
        path=path,
        repo_root=repo_root,
        required=required,
        kind="preregistration",
        decision_capable=decision_capable,
    )
    if identity_failure is not None:
        return identity_failure
    try:
        summary = check_yaml_file(path, repo_root=repo_root)
    except (OSError, InferenceContractError, ValueError, yaml.YAMLError) as exc:
        return _result(
            status="failed",
            required=required,
            kind="preregistration",
            reason=str(exc),
            path=str(path.relative_to(repo_root)),
        )
    return _result(
        status="passed",
        required=required,
        kind="preregistration",
        path=str(path.relative_to(repo_root)),
        summary=summary,
    )


def _run_artifact_catalog(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    required: bool,
    decision_capable: bool,
) -> dict[str, Any]:
    """Invoke the typed artifact-catalog validator."""
    path = _repo_relative_path(repo_root, spec.get("path"), "artifact_catalog.path")
    identity_failure = _file_identity_failure(
        spec,
        path=path,
        repo_root=repo_root,
        required=required,
        kind="artifact_catalog",
        decision_capable=decision_capable,
    )
    if identity_failure is not None:
        return identity_failure
    try:
        issues = validate_artifact_catalog(path)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        return _result(
            status="failed",
            required=required,
            kind="artifact_catalog",
            reason=str(exc),
            path=str(path.relative_to(repo_root)),
        )
    if issues:
        return _result(
            status="failed",
            required=required,
            kind="artifact_catalog",
            reason="artifact catalog validation reported issues",
            path=str(path.relative_to(repo_root)),
            issues=[{"path": issue.path, "message": issue.message} for issue in issues],
        )
    if decision_capable and required:
        try:
            catalog = load_artifact_catalog(path)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            return _result(
                status="failed",
                required=required,
                kind="artifact_catalog",
                reason=f"could not load artifact identity: {exc}",
                path=str(path.relative_to(repo_root)),
            )
        claim_boundaries = [entry.claim_boundary.lower() for entry in catalog.artifacts]
        diagnostic_tokens = ("fixture", "diagnostic", "not benchmark")
        if any(
            any(token in boundary for token in diagnostic_tokens) for boundary in claim_boundaries
        ):
            return _result(
                status="failed",
                required=required,
                kind="artifact_catalog",
                reason="diagnostic or fixture-only artifact catalog cannot authorize decision admission",
                path=str(path.relative_to(repo_root)),
                catalog_id=catalog.catalog_id,
            )
    return _result(
        status="passed",
        required=required,
        kind="artifact_catalog",
        path=str(path.relative_to(repo_root)),
        issues=[],
    )


def _run_result_packet(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    required: bool,
    decision_capable: bool,
) -> dict[str, Any]:
    """Use the merged result-packet validator when available, otherwise abstain."""
    path = _repo_relative_path(repo_root, spec.get("path"), "result_packet.path")
    identity_failure = _file_identity_failure(
        spec,
        path=path,
        repo_root=repo_root,
        required=required,
        kind="result_packet",
        decision_capable=decision_capable,
    )
    if identity_failure is not None:
        return identity_failure
    try:
        module = importlib.import_module("robot_sf.benchmark.result_interpretation_packet")
    except ModuleNotFoundError as exc:
        if exc.name != "robot_sf.benchmark.result_interpretation_packet":
            return _result(
                status="failed",
                required=required,
                kind="result_packet",
                reason=f"result-packet validator import failed: {exc}",
                path=str(path.relative_to(repo_root)),
            )
        return _result(
            status="unavailable",
            required=required,
            kind="result_packet",
            reason=(
                "no generic result-interpretation packet validator is available in this checkout"
            ),
            path=str(path.relative_to(repo_root)),
        )
    validator = getattr(module, "load_result_interpretation_packet", None)
    if not callable(validator):
        return _result(
            status="unavailable",
            required=required,
            kind="result_packet",
            reason="result-packet module has no public load_result_interpretation_packet validator",
            path=str(path.relative_to(repo_root)),
        )
    try:
        validator(path)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return _result(
            status="failed",
            required=required,
            kind="result_packet",
            reason=str(exc),
            path=str(path.relative_to(repo_root)),
        )
    return _result(
        status="passed",
        required=required,
        kind="result_packet",
        path=str(path.relative_to(repo_root)),
    )


def _run_durable_path(
    spec: Mapping[str, Any], *, repo_root: Path, required: bool
) -> dict[str, Any]:
    """Refuse path-existence checks as artifact admission proof.

    A path check cannot establish tracking, immutable retention, or checksum
    identity. The public artifact-catalog validator is the canonical owner for
    those properties, so a ``durable_path`` declaration is recorded as
    unavailable instead of being promoted to ``passed``.
    """
    path = _repo_relative_path(repo_root, spec.get("path"), "durable_path.path")
    relative = path.relative_to(repo_root)
    if relative.parts[:1] == ("output",):
        return _result(
            status="failed",
            required=required,
            kind="durable_path",
            reason="durable proof path points into disposable output/",
            path=str(relative),
        )
    if not path.exists():
        return _result(
            status="unavailable",
            required=required,
            kind="durable_path",
            reason="declared durable path does not exist yet",
            path=str(relative),
        )
    return _result(
        status="unavailable",
        required=required,
        kind="durable_path",
        reason=(
            "path existence does not prove tracked retention or checksum identity; "
            "configure an artifact_catalog proof instead"
        ),
        path=str(relative),
    )


def _run_surface(  # noqa: C901
    surface: str,
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    required: bool,
    manifest: Mapping[str, Any],
    build_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Dispatch one configured proof surface to its canonical validator."""
    kind = spec.get("kind")
    if kind not in _CHECK_KINDS:
        raise AnswerabilityProofError(
            f"validation.answerability_proof.{surface}.kind must be one of {sorted(_CHECK_KINDS)}"
        )
    decision_capable = _decision_capable(manifest)
    if (
        decision_capable
        and required
        and kind != "durable_path"
        and spec.get("proof_class") != "decision_capable"
    ):
        return _result(
            status="failed",
            required=required,
            kind=str(kind),
            reason=(
                "required decision-capable proof must declare proof_class=decision_capable; "
                "diagnostic-only or unclassified proof cannot authorize admission"
            ),
        )
    if kind == "command":
        command = _argv(spec.get("command"), f"validation.answerability_proof.{surface}.command")
        timeout_seconds, validator_id = _validate_registered_command(
            command,
            spec=spec,
            field=f"validation.answerability_proof.{surface}",
        )
        return _run_command(
            command,
            repo_root=repo_root,
            required=required,
            kind=kind,
            timeout_seconds=timeout_seconds,
            validator_id=validator_id,
        )
    if kind == "evidence_contract":
        contract_id = spec.get("contract_id")
        if not isinstance(contract_id, str) or not contract_id.strip():
            raise AnswerabilityProofError(
                f"validation.answerability_proof.{surface}.contract_id must be non-empty"
            )
        command = [
            sys.executable,
            str(repo_root / "scripts/validation/preflight_evidence_contract.py"),
            contract_id.strip(),
            "--json",
        ]
        return _run_command(
            command,
            repo_root=repo_root,
            required=required,
            kind=kind,
            timeout_seconds=_MAX_VALIDATOR_TIMEOUT_SECONDS,
            validator_id="evidence_contract",
        )
    if kind == "preregistration":
        return _run_preregistration(
            spec,
            repo_root=repo_root,
            required=required,
            decision_capable=decision_capable,
        )
    if kind == "artifact_catalog":
        return _run_artifact_catalog(
            spec,
            repo_root=repo_root,
            required=required,
            decision_capable=decision_capable,
        )
    if kind == "result_packet":
        return _run_result_packet(
            spec,
            repo_root=repo_root,
            required=required,
            decision_capable=decision_capable,
        )
    if kind == "durable_path":
        return _run_durable_path(spec, repo_root=repo_root, required=required)
    if build_rows is None:
        return _result(
            status="unavailable",
            required=required,
            kind=kind,
            reason="manifest-row producer callback is not available",
        )
    try:
        rows = build_rows(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason=f"manifest-row producer failed: {exc}",
        )
    if not rows:
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason="manifest-row producer emitted no rows",
        )
    if decision_capable:
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason="manifest-row proof is diagnostic_only and cannot authorize decision admission",
            row_count=len(rows),
        )
    return _result(status="passed", required=required, kind=kind, row_count=len(rows))


def _surface_declaration(contract: Mapping[str, Any], surface: str) -> Mapping[str, Any]:
    """Return one validated proof-surface declaration."""
    declarations = contract.get("proof_surfaces")
    if not isinstance(declarations, Mapping):
        raise AnswerabilityProofError(
            "answerability.proof_surfaces must be declared before executable admission proof"
        )
    declaration = declarations.get(surface)
    if not isinstance(declaration, Mapping):
        raise AnswerabilityProofError(f"answerability.proof_surfaces.{surface} must be a mapping")
    required = declaration.get("required")
    if not isinstance(required, bool):
        raise AnswerabilityProofError(
            f"answerability.proof_surfaces.{surface}.required must be a boolean"
        )
    return declaration


def collect_answerability_proof(  # noqa: C901
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    execute: bool,
    build_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None = None,
    proof_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect proof results for all six answerability surfaces.

    ``execute=False`` records configured checks as ``not_run``.  A required
    surface with no configured check is ``unavailable`` and therefore cannot
    pass a decision-capable admission gate.
    """
    answerability = manifest.get("answerability")
    if not isinstance(answerability, Mapping):
        return {
            "executed": execute,
            "surfaces": {},
            "status": "not_declared",
        }
    if not isinstance(answerability.get("proof_surfaces"), Mapping):
        return {
            "executed": execute,
            "surfaces": {},
            "status": "not_declared",
        }
    checks = manifest.get("validation", {})
    if not isinstance(checks, Mapping):
        raise AnswerabilityProofError("validation must be a mapping")
    configured = checks.get("answerability_proof", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise AnswerabilityProofError("validation.answerability_proof must be a mapping")

    surfaces: dict[str, dict[str, Any]] = {}
    for surface in PROOF_SURFACES:
        declaration = _surface_declaration(answerability, surface)
        required = bool(declaration["required"])
        spec = configured.get(surface)
        if spec is None:
            existing_status = declaration.get("status")
            existing_reason = declaration.get("unavailable_reason")
            if existing_status == "unavailable" and isinstance(existing_reason, str):
                surfaces[surface] = _result(
                    status="unavailable",
                    required=required,
                    reason=existing_reason,
                )
            else:
                surfaces[surface] = _result(
                    status="unavailable" if not required else "not_run",
                    required=required,
                    reason=(
                        "no executable proof check is configured"
                        if not required
                        else "required proof has no executable check configured"
                    ),
                )
            continue
        if not isinstance(spec, Mapping):
            raise AnswerabilityProofError(
                f"validation.answerability_proof.{surface} must be a mapping"
            )
        if not execute:
            surfaces[surface] = _result(
                status="not_run",
                required=required,
                kind=str(spec.get("kind")) if spec.get("kind") else None,
                reason="proof execution was not requested",
            )
            continue
        surfaces[surface] = _run_surface(
            surface,
            spec,
            repo_root=repo_root,
            required=required,
            manifest=manifest,
            build_rows=build_rows,
        )
    report: dict[str, Any] = {
        "executed": execute,
        "surfaces": surfaces,
        "status": "completed" if execute else "not_run",
    }
    if proof_binding is not None:
        report["binding"] = dict(proof_binding)
    return report


def apply_proof_results(
    contract: Mapping[str, Any], proof_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a contract copy whose proof statuses come only from the collector."""
    updated = copy.deepcopy(dict(contract))
    surfaces = proof_report.get("surfaces")
    if not isinstance(surfaces, Mapping):
        surfaces = {}
    declarations = updated.get("proof_surfaces")
    if not isinstance(declarations, Mapping):
        return updated
    updated_surfaces: dict[str, Any] = {}
    for surface in PROOF_SURFACES:
        declaration = declarations.get(surface)
        result = surfaces.get(surface)
        if not isinstance(declaration, Mapping):
            continue
        entry = dict(declaration)
        if not isinstance(result, Mapping):
            entry["status"] = "not_run"
            entry["unavailable_reason"] = "collector did not return a proof result"
            updated_surfaces[surface] = entry
            continue
        entry["status"] = result.get("status", "not_run")
        reason = result.get("reason") or result.get("unavailable_reason")
        if reason:
            entry["unavailable_reason"] = str(reason)
        else:
            entry.pop("unavailable_reason", None)
        updated_surfaces[surface] = entry
    if updated_surfaces:
        updated["proof_surfaces"] = updated_surfaces
    binding = proof_report.get("binding")
    if isinstance(binding, Mapping):
        updated["proof_binding"] = copy.deepcopy(dict(binding))
    return updated


__all__ = [
    "AnswerabilityProofError",
    "apply_proof_results",
    "collect_answerability_proof",
]
