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
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.artifact_catalog import (
    _resolve_catalog_path,
    load_artifact_catalog,
    validate_artifact_catalog,
)
from robot_sf.benchmark.research_answerability import (
    PROOF_SURFACES,
    strict_proof_input_provenance_error,
)
from scripts.validation.check_preregistration_inference_contract import (
    InferenceContractError,
    check_yaml_file,
)
from scripts.validation.preflight_evidence_contract import (
    _load_row,
    derive_claim_identity,
)

_CHECK_KINDS = {
    "artifact_catalog",
    "analysis_receipt",
    "command",
    "durable_path",
    "evidence_contract",
    "manifest_rows",
    "preregistration",
    "producer_receipt",
    "result_packet",
}
_MAX_OUTPUT_CHARS = 1200
_MAX_VALIDATOR_TIMEOUT_SECONDS = 120.0
_REGISTERED_VALIDATOR_IDS = {
    "pytest_contract",
}
_PYTEST_FLAGS = {"-q", "-v", "--disable-warnings"}
_SHA256_RE = r"^[0-9a-f]{64}$"
_DIAGNOSTIC_PACKET_TIERS = {"smoke_diagnostic", "visualization_fixture"}
_DIAGNOSTIC_PACKET_STATES = {"diagnostic_only", "unavailable_causal_inference"}
_DIAGNOSTIC_ARTIFACT_SOURCE_KINDS = {
    "fixture_construction",
    "diagnostic",
    "smoke_diagnostic",
    "visualization_fixture",
}
_STRICT_SURFACE_RULES: dict[str, dict[str, frozenset[str]]] = {
    "producer": {"kinds": frozenset({"producer_receipt"})},
    "preregistration": {"kinds": frozenset({"preregistration"})},
    "evidence_contract": {"kinds": frozenset({"evidence_contract"})},
    "analysis": {"kinds": frozenset({"analysis_receipt"})},
    "artifact": {"kinds": frozenset({"artifact_catalog", "durable_path"})},
    "result_packet": {"kinds": frozenset({"result_packet"})},
}


class _InputDriftError(OSError):
    """Raised when a proof input changes during one admission invocation."""


class AnswerabilityProofError(ValueError):
    """Raised when a manifest declares an invalid executable proof check."""


def _repo_relative_path(repo_root: Path, value: Any, field: str) -> Path:
    """Resolve a safe repository-relative path declared by a manifest."""
    if not isinstance(value, str) or not value.strip():
        raise AnswerabilityProofError(f"{field} must be a non-empty repository-relative path")
    path = Path(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise AnswerabilityProofError(f"{field} must not be absolute or traverse '..'")
    root = repo_root.resolve()
    candidate = root / path
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise AnswerabilityProofError(
            f"{field} cannot be resolved safely within the repository"
        ) from exc
    if resolved != root and root not in resolved.parents:
        raise AnswerabilityProofError(f"{field} must resolve within the repository root")
    return candidate


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


def _stable_file_bytes(path: Path) -> bytes:
    """Read one file twice and fail closed if its bytes drift during the read."""
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise _InputDriftError(f"proof input changed while being read: {path}")
    return first


def _stable_sha256_file(path: Path) -> str:
    """Return a digest for bytes that were stable across two consecutive reads."""
    return hashlib.sha256(_stable_file_bytes(path)).hexdigest()


def _input_drift_failure(
    path: Path,
    *,
    repo_root: Path,
    initial_sha256: str | None,
    required: bool,
    kind: str,
) -> dict[str, Any] | None:
    """Return a failed result when a validator observed mutable proof input."""
    if initial_sha256 is None:
        return None
    try:
        final_sha256 = _stable_sha256_file(path)
    except OSError as exc:
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason=f"could not verify {kind} proof input stability: {exc}",
            path=str(path.relative_to(repo_root)),
        )
    if final_sha256 == initial_sha256:
        return None
    return _result(
        status="failed",
        required=required,
        kind=kind,
        reason=f"{kind} proof input changed during validation; admission is blocked",
        path=str(path.relative_to(repo_root)),
        initial_sha256=initial_sha256,
        final_sha256=final_sha256,
    )


def _decision_identity(  # noqa: C901, PLR0912
    manifest: Mapping[str, Any], *, surface: str, spec: Mapping[str, Any]
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Validate the common identity carried by one strict proof declaration."""
    answerability = manifest.get("answerability")
    campaign = manifest.get("campaign")
    if not isinstance(answerability, Mapping) or not isinstance(campaign, Mapping):
        return None, f"{surface} proof cannot bind identity without campaign and answerability"
    question = answerability.get("question")
    estimand = answerability.get("estimand")
    if not isinstance(question, Mapping) or not isinstance(estimand, Mapping):
        return None, f"{surface} proof requires structured question and estimand identity"
    identity = spec.get("identity")
    if not isinstance(identity, Mapping):
        return None, (
            f"decision-capable {surface} proof requires an identity mapping containing "
            "campaign_id, question, and estimand"
        )
    expected = {
        "campaign_id": campaign.get("id"),
        "question": question.get("research_question"),
        "estimand": estimand.get("primary"),
    }
    for field, expected_value in expected.items():
        if not isinstance(expected_value, str) or not expected_value.strip():
            return None, f"answerability.{field} identity is missing for {surface} proof"
        if identity.get(field) != expected_value:
            return None, f"{surface} proof identity.{field} does not match the manifest"

    if surface == "producer":
        producers = answerability.get("producers")
        expected_fields = sorted(
            producer.get("field")
            for producer in producers or []
            if isinstance(producer, Mapping) and producer.get("required", True)
        )
        if identity.get("producer_fields") != expected_fields:
            return None, "producer proof identity.producer_fields does not match required producers"
    elif surface == "analysis":
        analysis = answerability.get("analysis")
        analysis_id = analysis.get("analysis_id") if isinstance(analysis, Mapping) else None
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            return None, "answerability.analysis.analysis_id is required for strict analysis proof"
        if identity.get("analysis_id") != analysis_id:
            return None, "analysis proof identity.analysis_id does not match the manifest"
    elif surface == "artifact":
        if not isinstance(identity.get("catalog_id"), str) or not identity["catalog_id"].strip():
            return None, "artifact proof identity.catalog_id is required"
        artifact_ids = identity.get("artifact_ids")
        if (
            not isinstance(artifact_ids, list)
            or not artifact_ids
            or not all(isinstance(item, str) and item.strip() for item in artifact_ids)
        ):
            return None, "artifact proof identity.artifact_ids must be a non-empty list"
        artifact_digests = identity.get("artifact_digests")
        if not isinstance(artifact_digests, Mapping):
            return None, "artifact proof identity.artifact_digests is required"
    elif surface == "result_packet":
        for field in (
            "packet_id",
            "evidence_id",
            "evidence_tier",
            "admission_state",
            "question_id",
            "estimand_id",
            "source_digests",
        ):
            if field not in identity:
                return None, f"result-packet proof identity.{field} is required"
    return identity, None


def _strict_surface_failure(
    surface: str,
    kind: Any,
    *,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    required: bool,
) -> dict[str, Any] | None:
    """Reject substitution of a generic validator across decision surfaces."""
    if not _decision_capable(manifest):
        return None
    rule = _STRICT_SURFACE_RULES[surface]
    if kind not in rule["kinds"]:
        return _result(
            status="failed",
            required=required,
            kind=str(kind) if kind is not None else None,
            reason=(
                f"decision-capable {surface} proof must use its canonical kind; "
                f"allowed={sorted(rule['kinds'])}"
            ),
        )
    _, identity_error = _decision_identity(manifest, surface=surface, spec=spec)
    if identity_error:
        return _result(
            status="failed",
            required=required,
            kind=str(kind),
            reason=identity_error,
        )
    return None


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
    provenance_error = strict_proof_input_provenance_error(
        path,
        repo_root=repo_root,
        field=f"{kind} proof input",
    )
    if provenance_error:
        return _result(
            status="failed",
            required=required,
            kind=kind,
            reason=provenance_error,
            path=str(path.relative_to(repo_root)),
        )
    try:
        actual = _stable_sha256_file(path)
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
    command: list[str], *, spec: Mapping[str, Any], field: str, repo_root: Path
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
        _repo_relative_path(repo_root, item, f"{field}.command test path")
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
    expected_json: Mapping[str, Any] | None = None,
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
        if expected_json is not None:
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                return _result(
                    status="failed",
                    required=required,
                    kind=kind,
                    reason=f"registered validator did not emit JSON identity: {exc}",
                    command=command,
                    validator_id=validator_id,
                    timeout_seconds=timeout_seconds,
                )
            if not isinstance(payload, Mapping) or any(
                payload.get(field) != expected_value
                for field, expected_value in expected_json.items()
            ):
                return _result(
                    status="failed",
                    required=required,
                    kind=kind,
                    reason="registered validator output identity does not match the proof declaration",
                    command=command,
                    validator_id=validator_id,
                    timeout_seconds=timeout_seconds,
                    output_identity={
                        field: payload.get(field) if isinstance(payload, Mapping) else None
                        for field in expected_json
                    },
                )
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
    initial_sha256 = None
    if decision_capable and required:
        try:
            initial_sha256 = _stable_sha256_file(path)
        except OSError as exc:
            return _result(
                status="failed",
                required=required,
                kind="preregistration",
                reason=f"could not read preregistration proof input: {exc}",
                path=str(path.relative_to(repo_root)),
            )
    try:
        payload = yaml.safe_load(_stable_file_bytes(path).decode("utf-8"))
        summary = check_yaml_file(path, repo_root=repo_root)
    except (OSError, UnicodeDecodeError, InferenceContractError, ValueError, yaml.YAMLError) as exc:
        return _result(
            status="failed",
            required=required,
            kind="preregistration",
            reason=str(exc),
            path=str(path.relative_to(repo_root)),
        )
    drift = _input_drift_failure(
        path,
        repo_root=repo_root,
        initial_sha256=initial_sha256,
        required=required,
        kind="preregistration",
    )
    if drift is not None:
        return drift
    if decision_capable and required:
        identity = spec["identity"]
        if not isinstance(payload, Mapping):
            return _result(
                status="failed",
                required=required,
                kind="preregistration",
                reason="preregistration proof input must be a mapping",
                path=str(path.relative_to(repo_root)),
            )
        identity_fields = {
            "study_id": payload.get("study_id"),
            "question": payload.get("research_question", payload.get("question")),
            "estimand": payload.get("estimand", payload.get("estimand_id")),
        }
        for field, value in identity_fields.items():
            if value != identity.get(field):
                return _result(
                    status="failed",
                    required=required,
                    kind="preregistration",
                    reason=f"preregistration identity.{field} is not bound to the declared proof",
                    path=str(path.relative_to(repo_root)),
                )
    return _result(
        status="passed",
        required=required,
        kind="preregistration",
        path=str(path.relative_to(repo_root)),
        summary=summary,
        study_id=(identity.get("study_id") if decision_capable and required else None),
    )


def _run_artifact_catalog(  # noqa: C901, PLR0912
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    required: bool,
    decision_capable: bool,
    approved_durable_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Invoke the typed artifact-catalog validator."""
    approved_roots = tuple(approved_durable_roots)
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
    initial_sha256 = None
    if decision_capable and required:
        try:
            initial_sha256 = _stable_sha256_file(path)
        except OSError as exc:
            return _result(
                status="failed",
                required=required,
                kind="artifact_catalog",
                reason=f"could not read artifact catalog proof input: {exc}",
                path=str(path.relative_to(repo_root)),
            )
    try:
        issues = validate_artifact_catalog(
            path,
            repository_root=repo_root,
            approved_durable_roots=approved_roots,
        )
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
            catalog = load_artifact_catalog(
                path,
                repository_root=repo_root,
                approved_durable_roots=approved_roots,
            )
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            return _result(
                status="failed",
                required=required,
                kind="artifact_catalog",
                reason=f"could not load artifact identity: {exc}",
                path=str(path.relative_to(repo_root)),
            )
        drift = _input_drift_failure(
            path,
            repo_root=repo_root,
            initial_sha256=initial_sha256,
            required=required,
            kind="artifact_catalog",
        )
        if drift is not None:
            return drift
        identity = spec["identity"]
        selected_ids = identity["artifact_ids"]
        selected = {entry.artifact_id: entry for entry in catalog.artifacts}
        if catalog.catalog_id != identity["catalog_id"]:
            return _result(
                status="failed",
                required=required,
                kind="artifact_catalog",
                reason="artifact catalog identity.catalog_id does not match the declared proof",
                path=str(path.relative_to(repo_root)),
                catalog_id=catalog.catalog_id,
            )
        expected_claim_identity = {
            field: identity[field] for field in ("campaign_id", "question", "estimand")
        }
        if catalog.claim_identity != expected_claim_identity:
            return _result(
                status="failed",
                required=required,
                kind="artifact_catalog",
                reason=(
                    "artifact catalog claim_identity is not bound to the declared "
                    "campaign/question/estimand"
                ),
                path=str(path.relative_to(repo_root)),
                claim_identity=catalog.claim_identity,
            )
        if set(selected_ids) != set(identity["artifact_digests"]):
            return _result(
                status="failed",
                required=required,
                kind="artifact_catalog",
                reason="artifact catalog selected artifact IDs do not match their digest set",
                path=str(path.relative_to(repo_root)),
                catalog_id=catalog.catalog_id,
            )
        for artifact_id in selected_ids:
            entry = selected.get(artifact_id)
            if entry is None:
                return _result(
                    status="failed",
                    required=required,
                    kind="artifact_catalog",
                    reason=f"artifact catalog does not contain selected artifact {artifact_id!r}",
                    path=str(path.relative_to(repo_root)),
                    catalog_id=catalog.catalog_id,
                )
            if entry.source_kind in _DIAGNOSTIC_ARTIFACT_SOURCE_KINDS:
                return _result(
                    status="failed",
                    required=required,
                    kind="artifact_catalog",
                    reason=(
                        "controlled diagnostic or fixture artifact source cannot authorize "
                        "decision admission"
                    ),
                    path=str(path.relative_to(repo_root)),
                    catalog_id=catalog.catalog_id,
                    artifact_id=artifact_id,
                )
            file_refs = [*entry.source_files, *entry.outputs.values()]
            if entry.caption_file is not None:
                file_refs.append(entry.caption_file)
            for file_ref in file_refs:
                provenance_error = strict_proof_input_provenance_error(
                    _resolve_catalog_path(
                        path,
                        file_ref.path,
                        repository_root=repo_root,
                    ),
                    repo_root=repo_root,
                    field=f"artifact catalog {artifact_id!r} file reference",
                )
                if provenance_error:
                    return _result(
                        status="failed",
                        required=required,
                        kind="artifact_catalog",
                        reason=provenance_error,
                        path=str(path.relative_to(repo_root)),
                        catalog_id=catalog.catalog_id,
                        artifact_id=artifact_id,
                    )
            actual_digests = sorted(
                [ref.sha256 for ref in entry.source_files]
                + [ref.sha256 for ref in entry.outputs.values()]
                + ([entry.caption_file.sha256] if entry.caption_file is not None else [])
            )
            expected_digests = identity["artifact_digests"].get(artifact_id)
            if not isinstance(expected_digests, list) or sorted(expected_digests) != actual_digests:
                return _result(
                    status="failed",
                    required=required,
                    kind="artifact_catalog",
                    reason=f"artifact {artifact_id!r} digest set is not bound to the catalog",
                    path=str(path.relative_to(repo_root)),
                    catalog_id=catalog.catalog_id,
                    artifact_id=artifact_id,
                    actual_digests=actual_digests,
                )
    return _result(
        status="passed",
        required=required,
        kind="artifact_catalog",
        path=str(path.relative_to(repo_root)),
        issues=[],
    )


def _run_result_packet(  # noqa: C901, PLR0912
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
    initial_sha256 = None
    if decision_capable and required:
        try:
            initial_sha256 = _stable_sha256_file(path)
        except OSError as exc:
            return _result(
                status="failed",
                required=required,
                kind="result_packet",
                reason=f"could not read result-packet proof input: {exc}",
                path=str(path.relative_to(repo_root)),
            )
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
        packet = validator(path)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return _result(
            status="failed",
            required=required,
            kind="result_packet",
            reason=str(exc),
            path=str(path.relative_to(repo_root)),
        )
    drift = _input_drift_failure(
        path,
        repo_root=repo_root,
        initial_sha256=initial_sha256,
        required=required,
        kind="result_packet",
    )
    if drift is not None:
        return drift
    if decision_capable and required:
        identity = spec["identity"]
        packet_evidence = getattr(packet, "evidence", None)
        packet_question = getattr(packet, "question", None)
        packet_estimand = getattr(packet, "estimand", None)
        packet_sources = getattr(packet, "sources", None)
        packet_claim_identity = getattr(packet, "claim_identity", None)
        observed = {
            "packet_id": getattr(packet, "packet_id", None),
            "evidence_id": getattr(packet_evidence, "evidence_id", None),
            "evidence_tier": getattr(packet_evidence, "tier", None),
            "admission_state": getattr(packet_evidence, "admission_state", None),
            "campaign_id": getattr(packet_claim_identity, "campaign_id", None),
            "question": getattr(packet_claim_identity, "question", None),
            "estimand": getattr(packet_claim_identity, "estimand", None),
            "question_id": getattr(packet_question, "question_id", None),
            "estimand_id": getattr(packet_estimand, "estimand_id", None),
            "source_digests": {
                getattr(source, "source_id", ""): getattr(source, "sha256", "")
                for source in packet_sources or []
            },
        }
        for field, expected in identity.items():
            if field not in observed:
                return _result(
                    status="failed",
                    required=required,
                    kind="result_packet",
                    reason=(
                        f"result-packet identity.{field} is not exposed by the canonical "
                        "packet validator"
                    ),
                    path=str(path.relative_to(repo_root)),
                )
            if observed[field] != expected:
                return _result(
                    status="failed",
                    required=required,
                    kind="result_packet",
                    reason=f"result-packet identity.{field} is not bound to the declared proof",
                    path=str(path.relative_to(repo_root)),
                    observed=observed[field],
                )
        for field, observed_value in (
            ("question", getattr(packet_question, "text", None)),
            ("estimand", getattr(packet_estimand, "description", None)),
        ):
            if observed_value != identity[field]:
                return _result(
                    status="failed",
                    required=required,
                    kind="result_packet",
                    reason=(f"result-packet {field} is not bound to the declared proof identity"),
                    path=str(path.relative_to(repo_root)),
                    observed=observed_value,
                )
        if (
            observed["evidence_tier"] in _DIAGNOSTIC_PACKET_TIERS
            or observed["admission_state"] in _DIAGNOSTIC_PACKET_STATES
        ):
            return _result(
                status="failed",
                required=required,
                kind="result_packet",
                reason=(
                    "controlled diagnostic or fixture result packet cannot authorize "
                    "decision admission"
                ),
                path=str(path.relative_to(repo_root)),
            )
        for source in packet_sources or []:
            try:
                source_path = _repo_relative_path(
                    repo_root,
                    getattr(source, "path", None),
                    f"result_packet.source[{getattr(source, 'source_id', '')!r}].path",
                )
            except AnswerabilityProofError as exc:
                return _result(
                    status="failed",
                    required=required,
                    kind="result_packet",
                    reason=str(exc),
                    path=str(path.relative_to(repo_root)),
                )
            provenance_error = strict_proof_input_provenance_error(
                source_path,
                repo_root=repo_root,
                field=f"result-packet source {getattr(source, 'source_id', '')!r}",
            )
            if provenance_error:
                return _result(
                    status="failed",
                    required=required,
                    kind="result_packet",
                    reason=provenance_error,
                    path=str(path.relative_to(repo_root)),
                )
    return _result(
        status="passed",
        required=required,
        kind="result_packet",
        path=str(path.relative_to(repo_root)),
        packet_id=(getattr(packet, "packet_id", None) if decision_capable else None),
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


def _run_receipt_verification(
    surface: str,
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    manifest: Mapping[str, Any],
    required: bool,
) -> dict[str, Any]:
    """Fail closed until a checked-in, receipt-aware validator is registered.

    The previous implementation treated a manifest-selected test file or an
    AST symbol in an answerability owner as independent proof. Neither path
    consumed the receipt, so a green unrelated test could authorize a strict
    producer or analysis surface. There is currently no canonical
    receipt-aware validator in this checkout; keeping the surface unavailable
    is safer than accepting self-attested or caller-selected proof.
    """
    verification = spec.get("verification")
    if not isinstance(verification, Mapping):
        return _result(
            status="failed",
            required=required,
            kind=f"{surface}_receipt",
            reason=(
                f"required {surface} receipt is self-attested; configure executable or "
                "canonical_owner verification"
            ),
        )
    verification_kind = verification.get("kind")
    if verification_kind not in {"command", "canonical_owner"}:
        return _result(
            status="failed",
            required=required,
            kind=f"{surface}_receipt",
            reason=(f"{surface} receipt verification.kind must be command or canonical_owner"),
        )
    return _result(
        status="failed",
        required=required,
        kind=f"{surface}_receipt",
        reason=(
            f"{surface} receipt verification is blocked: no checked-in, receipt-aware "
            "validator is registered; caller-selected commands and canonical_owner symbols "
            "cannot authorize admission"
        ),
        verification_kind=verification_kind,
    )


def _run_receipt(  # noqa: C901, PLR0912
    surface: str,
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    required: bool,
    manifest: Mapping[str, Any],
    decision_capable: bool,
) -> dict[str, Any]:
    """Validate a surface-specific, checksum-bound proof receipt."""
    path = _repo_relative_path(repo_root, spec.get("path"), f"{surface}_receipt.path")
    identity_failure = _file_identity_failure(
        spec,
        path=path,
        repo_root=repo_root,
        required=required,
        kind=f"{surface}_receipt",
        decision_capable=decision_capable,
    )
    if identity_failure is not None:
        return identity_failure
    initial_sha256 = None
    if decision_capable and required:
        try:
            initial_sha256 = _stable_sha256_file(path)
            payload = json.loads(_stable_file_bytes(path).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _result(
                status="failed",
                required=required,
                kind=f"{surface}_receipt",
                reason=f"could not load {surface} proof receipt: {exc}",
                path=str(path.relative_to(repo_root)),
            )
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _result(
                status="failed",
                required=required,
                kind=f"{surface}_receipt",
                reason=f"could not load {surface} proof receipt: {exc}",
                path=str(path.relative_to(repo_root)),
            )
    if not isinstance(payload, Mapping):
        return _result(
            status="failed",
            required=required,
            kind=f"{surface}_receipt",
            reason=f"{surface} proof receipt must be a JSON object",
            path=str(path.relative_to(repo_root)),
        )
    drift = _input_drift_failure(
        path,
        repo_root=repo_root,
        initial_sha256=initial_sha256,
        required=required,
        kind=f"{surface}_receipt",
    )
    if drift is not None:
        return drift
    identity = spec.get("identity")
    if not isinstance(identity, Mapping):
        return _result(
            status="failed",
            required=required,
            kind=f"{surface}_receipt",
            reason=f"{surface} receipt identity must be a mapping",
            path=str(path.relative_to(repo_root)),
        )
    expected_common = {
        "campaign_id": identity.get("campaign_id"),
        "question": identity.get("question"),
        "estimand": identity.get("estimand"),
    }
    for field, expected in expected_common.items():
        if payload.get(field) != expected:
            return _result(
                status="failed",
                required=required,
                kind=f"{surface}_receipt",
                reason=f"{surface} receipt {field} is not bound to the declared proof identity",
                path=str(path.relative_to(repo_root)),
            )
    if payload.get("status") != "passed":
        return _result(
            status="failed",
            required=required,
            kind=f"{surface}_receipt",
            reason=f"{surface} proof receipt status must be passed",
            path=str(path.relative_to(repo_root)),
        )
    if surface == "producer":
        if payload.get("schema_version") != "research_answerability_producer_receipt.v1":
            return _result(
                status="failed",
                required=required,
                kind="producer_receipt",
                reason="producer receipt schema_version is not canonical",
                path=str(path.relative_to(repo_root)),
            )
        if payload.get("producer_fields") != identity.get("producer_fields"):
            return _result(
                status="failed",
                required=required,
                kind="producer_receipt",
                reason="producer receipt fields are not bound to the required producer set",
                path=str(path.relative_to(repo_root)),
            )
    else:
        if payload.get("schema_version") != "research_answerability_analysis_receipt.v1":
            return _result(
                status="failed",
                required=required,
                kind="analysis_receipt",
                reason="analysis receipt schema_version is not canonical",
                path=str(path.relative_to(repo_root)),
            )
        analysis = manifest.get("answerability", {}).get("analysis", {})
        if payload.get("analysis_id") != identity.get("analysis_id") or payload.get(
            "command"
        ) != analysis.get("command"):
            return _result(
                status="failed",
                required=required,
                kind="analysis_receipt",
                reason="analysis receipt identity is not bound to answerability.analysis",
                path=str(path.relative_to(repo_root)),
            )
        if payload.get("dry_run_status") not in {"passed", "not_required"} or payload.get(
            "comparability_status"
        ) not in {"passed", "not_required"}:
            return _result(
                status="failed",
                required=required,
                kind="analysis_receipt",
                reason="analysis receipt must report passed or not_required analysis checks",
                path=str(path.relative_to(repo_root)),
            )
    verification_result = None
    if decision_capable and required:
        verification_result = _run_receipt_verification(
            surface,
            spec,
            repo_root=repo_root,
            manifest=manifest,
            required=required,
        )
        if verification_result["status"] != "passed":
            return verification_result
        drift = _input_drift_failure(
            path,
            repo_root=repo_root,
            initial_sha256=initial_sha256,
            required=required,
            kind=f"{surface}_receipt",
        )
        if drift is not None:
            return drift
    return _result(
        status="passed",
        required=required,
        kind=f"{surface}_receipt",
        path=str(path.relative_to(repo_root)),
        identity=dict(identity) if isinstance(identity, Mapping) else {},
        verification=verification_result,
    )


def _run_evidence_contract(  # noqa: C901
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    required: bool,
    decision_capable: bool,
) -> dict[str, Any]:
    """Invoke the public evidence-contract validator with canonical row identity.

    The ORCA smoke contract validates evidence fields, but its built-in
    representative row has no research-claim identity. A decision-capable
    proof therefore needs a checked-in canonical row carrying
    ``claim_identity``. Caller-supplied CLI identity flags are never used as
    evidence.
    """
    contract_id = spec.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id.strip():
        raise AnswerabilityProofError(
            "validation.answerability_proof.evidence_contract.contract_id must be non-empty"
        )
    identity = spec.get("identity")
    if (
        decision_capable
        and required
        and (
            not isinstance(identity, Mapping) or identity.get("contract_id") != contract_id.strip()
        )
    ):
        return _result(
            status="failed",
            required=required,
            kind="evidence_contract",
            reason="evidence-contract identity.contract_id does not match the declared contract",
        )
    expected_json: dict[str, Any] = {
        "contract_id": contract_id.strip(),
        "conforms": True,
    }
    row_path: Path | None = None
    initial_row_sha256: str | None = None
    if decision_capable and required:
        if not isinstance(identity, Mapping):  # pragma: no cover - guarded above
            return _result(
                status="failed",
                required=required,
                kind="evidence_contract",
                reason="evidence-contract proof identity must be a mapping",
            )
        row_value = spec.get("row")
        if not isinstance(row_value, str) or not row_value.strip():
            return _result(
                status="failed",
                required=required,
                kind="evidence_contract",
                reason=(
                    "decision-capable evidence-contract proof requires a canonical evidence row "
                    "with claim_identity"
                ),
            )
        row_path = _repo_relative_path(repo_root, row_value, "evidence_contract.row")
        identity_failure = _file_identity_failure(
            spec,
            path=row_path,
            repo_root=repo_root,
            required=required,
            kind="evidence_contract row",
            decision_capable=decision_capable,
        )
        if identity_failure is not None:
            return identity_failure
        try:
            initial_row_sha256 = _stable_sha256_file(row_path)
            canonical_identity = derive_claim_identity(_load_row(row_path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return _result(
                status="failed",
                required=required,
                kind="evidence_contract",
                reason=f"could not load canonical evidence row: {exc}",
                path=str(row_path.relative_to(repo_root)),
            )
        expected_identity = {
            field: identity.get(field) for field in ("campaign_id", "question", "estimand")
        }
        if canonical_identity is None:
            return _result(
                status="failed",
                required=required,
                kind="evidence_contract",
                reason="canonical evidence row has no complete claim_identity",
                path=str(row_path.relative_to(repo_root)),
            )
        if canonical_identity != expected_identity:
            return _result(
                status="failed",
                required=required,
                kind="evidence_contract",
                reason="canonical evidence row claim_identity does not match the declared proof",
                path=str(row_path.relative_to(repo_root)),
                claim_identity=canonical_identity,
            )
        expected_json["claim_identity"] = canonical_identity
    command = [
        sys.executable,
        str(repo_root / "scripts/validation/preflight_evidence_contract.py"),
        contract_id.strip(),
    ]
    if row_path is not None:
        command.extend(["--row", str(row_path.relative_to(repo_root))])
    command.append("--json")
    result = _run_command(
        command,
        repo_root=repo_root,
        required=required,
        kind="evidence_contract",
        timeout_seconds=_MAX_VALIDATOR_TIMEOUT_SECONDS,
        validator_id="evidence_contract",
        expected_json=expected_json,
    )
    if row_path is not None:
        drift = _input_drift_failure(
            row_path,
            repo_root=repo_root,
            initial_sha256=initial_row_sha256,
            required=required,
            kind="evidence-contract row",
        )
        if drift is not None:
            return drift
    return result


def _run_surface(  # noqa: C901, PLR0912
    surface: str,
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    required: bool,
    manifest: Mapping[str, Any],
    build_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None,
    approved_durable_roots: Iterable[Path],
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
    if not (surface == "artifact" and kind == "durable_path"):
        strict_failure = _strict_surface_failure(
            surface,
            kind,
            manifest=manifest,
            spec=spec,
            required=required,
        )
        if strict_failure is not None:
            return strict_failure
    if kind == "producer_receipt":
        return _run_receipt(
            "producer",
            spec,
            repo_root=repo_root,
            required=required,
            manifest=manifest,
            decision_capable=decision_capable,
        )
    if kind == "analysis_receipt":
        return _run_receipt(
            "analysis",
            spec,
            repo_root=repo_root,
            required=required,
            manifest=manifest,
            decision_capable=decision_capable,
        )
    if kind == "command":
        command = _argv(spec.get("command"), f"validation.answerability_proof.{surface}.command")
        timeout_seconds, validator_id = _validate_registered_command(
            command,
            spec=spec,
            field=f"validation.answerability_proof.{surface}",
            repo_root=repo_root,
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
        return _run_evidence_contract(
            spec,
            repo_root=repo_root,
            required=required,
            decision_capable=decision_capable,
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
            approved_durable_roots=approved_durable_roots,
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
    approved_durable_roots: Iterable[Path] = (),
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

    approved_roots = tuple(approved_durable_roots)
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
            approved_durable_roots=approved_roots,
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
