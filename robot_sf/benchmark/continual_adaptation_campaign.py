"""Campaign integration for continual-adaptation nominal/shift/forgetting evidence (issue #6657).

Connects continual-adaptation runs to the benchmark campaign machinery so an
adaptation run produces nominal/shift/forgetting evaluation result references
and a versioned evidence bundle under ``docs/context/evidence/`` naming the
validator-derived adapted-policy identifier distinct from baseline, satisfying
the promotion gate in
:func:`robot_sf.research.continual_adaptation_protocol.check_continual_adaptation_run`.

This module is metadata-only: it builds evidence-bundle structure and result
references but does not launch training, run evaluations, write checkpoints,
mutate the safety wrapper, or promote a policy. Only a positively identified
native record is accepted; fallback, degraded, failed, missing, duplicate,
provenance-invalid, or unknown records fail closed and are never presented as
benchmark evidence.

The evidence bundle is versioned via ``continual_adaptation_evidence.v1`` and
stamps the protocol evidence boundary so a passing promotion gate is never
mistaken for an executed adaptation or benchmark/paper evidence.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_sf.errors import RobotSfError
from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
    PROTOCOL_STATUS_VALID,
    check_continual_adaptation_run,
    derive_adapted_policy_identifier,
)

CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION = "continual_adaptation_evidence.v1"

#: Result reference names required by the promotion gate.
_RESULT_REF_NAMES = ("nominal_result", "shift_result", "forgetting_result")

#: This metadata-only integration accepts only a positively identified native record.
_ALLOWED_EXECUTION_MODES = frozenset({"native"})

_SHA256_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_METADATA_ONLY_PROMOTION_RATIONALE = (
    "Protocol-contract fixture only: reference completeness validates implementation wiring; "
    "no training, evaluation execution, checkpoint or policy promotion, benchmark ranking, "
    "or paper evidence occurred."
)


class ContinualAdaptationCampaignError(RobotSfError, ValueError):
    """Raised when campaign integration cannot produce valid promotion evidence."""

    def __init__(self, message: str, *, source: str | Path | None = None):
        """Build an actionable campaign error with an optional source path."""
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + message)


@dataclass(frozen=True, slots=True)
class ContinualAdaptationEvidenceBundle:
    """Versioned evidence bundle for a continual-adaptation promotion gate.

    The bundle names the validator-derived adapted-policy identifier distinct
    from the baseline, carries result references for nominal/shift/forgetting
    evaluations, and stamps the protocol evidence boundary.
    """

    schema_version: str
    run_id: str
    issue: int
    evidence_boundary: str
    claim_boundary: str
    baseline_policy_identifier: str
    derived_adapted_policy_identifier: str
    execution_mode: str
    nominal_result: dict[str, Any]
    shift_result: dict[str, Any]
    forgetting_result: dict[str, Any]
    evidence_bundle_ref: dict[str, Any]
    promotion_gate_ready: bool = True
    blockers: list[str] = field(default_factory=list)

    @property
    def is_promotion_ready(self) -> bool:
        """Return ``True`` when the bundle has no blockers."""
        return self.promotion_gate_ready and not self.blockers

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report including the external bundle reference."""
        payload = self.to_payload_dict()
        payload["evidence_bundle_ref"] = dict(self.evidence_bundle_ref)
        payload["promotion_ready"] = self.is_promotion_ready
        payload["blockers"] = list(self.blockers)
        return payload

    def to_payload_dict(self) -> dict[str, Any]:
        """Return the deterministic on-disk bundle payload.

        The payload intentionally excludes its own external reference and any
        wall-clock timestamp. Its SHA-256 can therefore bind the exact bytes
        written by :func:`write_evidence_bundle` without a circular checksum.
        """
        return _evidence_payload(
            run_id=self.run_id,
            issue=self.issue,
            evidence_boundary=self.evidence_boundary,
            claim_boundary=self.claim_boundary,
            baseline_policy_identifier=self.baseline_policy_identifier,
            derived_adapted_policy_identifier=self.derived_adapted_policy_identifier,
            execution_mode=self.execution_mode,
            nominal_result=self.nominal_result,
            shift_result=self.shift_result,
            forgetting_result=self.forgetting_result,
        )


def _sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _content_bytes(content: str | bytes) -> bytes:
    """Return exact bytes for checksum calculation."""
    return content if isinstance(content, bytes) else content.encode("utf-8")


def _checksum_for_content(content: str | bytes) -> dict[str, str]:
    """Build a ``{algorithm, digest}`` checksum mapping for *content*.

    Returns:
        A mapping with ``algorithm`` and ``digest`` keys.
    """
    return {"algorithm": "sha256", "digest": _sha256_hex(_content_bytes(content))}


def _validate_execution_mode(execution_mode: str) -> str:
    """Return normalized native mode, rejecting every other classification."""
    if not isinstance(execution_mode, str):
        raise ContinualAdaptationCampaignError(
            f"execution_mode={execution_mode!r} is not an allowed native record; "
            "execution mode must be a string"
        )
    normalized = execution_mode.strip().lower()
    if normalized not in _ALLOWED_EXECUTION_MODES:
        raise ContinualAdaptationCampaignError(
            f"execution_mode={execution_mode!r} is not an allowed native record; "
            "fallback, degraded, failed, missing, duplicate, provenance-invalid, "
            "and unknown records fail closed"
        )
    return normalized


def _validated_checksum(checksum: Mapping[str, Any]) -> dict[str, str]:
    """Return a normalized supported checksum or fail closed."""
    algorithm = checksum.get("algorithm")
    digest = checksum.get("digest")
    if algorithm != "sha256" or not isinstance(digest, str):
        raise ContinualAdaptationCampaignError(
            "checksum must declare algorithm='sha256' and a lowercase hexadecimal digest"
        )
    if _SHA256_DIGEST_PATTERN.fullmatch(digest) is None:
        raise ContinualAdaptationCampaignError(
            "sha256 checksum digest must be exactly 64 lowercase hexadecimal characters"
        )
    return {"algorithm": algorithm, "digest": digest}


def _canonical_yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a mapping to deterministic UTF-8 YAML bytes.

    Returns:
        Canonically ordered UTF-8 YAML bytes.
    """
    return yaml.safe_dump(
        dict(payload),
        default_flow_style=False,
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")


def _evidence_payload(  # noqa: PLR0913
    *,
    run_id: str,
    issue: int,
    evidence_boundary: str,
    claim_boundary: str,
    baseline_policy_identifier: str,
    derived_adapted_policy_identifier: str,
    execution_mode: str,
    nominal_result: Mapping[str, Any],
    shift_result: Mapping[str, Any],
    forgetting_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic, non-self-referential evidence payload.

    Returns:
        The exact mapping serialized into the evidence-bundle artifact.
    """
    return {
        "schema_version": CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION,
        "run_id": run_id,
        "issue": issue,
        "evidence_boundary": evidence_boundary,
        "claim_boundary": claim_boundary,
        "baseline_policy_identifier": baseline_policy_identifier,
        "derived_adapted_policy_identifier": derived_adapted_policy_identifier,
        "execution_mode": execution_mode,
        "nominal_result": dict(nominal_result),
        "shift_result": dict(shift_result),
        "forgetting_result": dict(forgetting_result),
    }


def build_result_reference(
    uri: str,
    *,
    content: str | bytes | None = None,
    checksum: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single result reference with a URI and a supported checksum.

    Exactly one of *content* or *checksum* must be provided.  When *content*
    is given the SHA-256 checksum is computed deterministically.

    Returns:
        A mapping with ``uri`` and ``checksum`` keys.

    Raises:
        ContinualAdaptationCampaignError: when neither or both are provided,
            or when the URI is empty.
    """
    if not uri or not uri.strip():
        raise ContinualAdaptationCampaignError("result reference uri must be non-empty")
    if (content is None) == (checksum is None):
        raise ContinualAdaptationCampaignError(
            "exactly one of content or checksum must be provided for a result reference"
        )
    if checksum is not None:
        resolved_checksum = _validated_checksum(checksum)
    elif content is not None:
        resolved_checksum = _checksum_for_content(content)
    else:  # Defensive narrowing; the exactly-one guard above already rejects this case.
        raise ContinualAdaptationCampaignError(
            "result reference content unexpectedly missing after validation"
        )
    return {"uri": uri, "checksum": resolved_checksum}


def build_evidence_bundle_ref(
    *,
    identifier: str,
    uri: str,
    policy_identifier: str,
    baseline_identifier: str,
    content: str | bytes | None = None,
    checksum: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the evidence-bundle reference required by the promotion gate.

    The reference carries a non-empty *identifier* distinct from
    *baseline_identifier*, a *uri*, a supported checksum, and a
    *policy_identifier* that must exactly match the validator-derived
    adapted-policy identifier.

    Returns:
        A mapping with ``identifier``, ``uri``, ``checksum``, and
        ``policy_identifier`` keys.

    Raises:
        ContinualAdaptationCampaignError: when the identifier collides with
            the baseline or the policy_identifier is empty.
    """
    if not identifier or not identifier.strip():
        raise ContinualAdaptationCampaignError("evidence bundle identifier must be non-empty")
    if identifier == baseline_identifier:
        raise ContinualAdaptationCampaignError(
            f"evidence bundle identifier {identifier!r} must not equal the baseline identifier"
        )
    if not policy_identifier or not policy_identifier.strip():
        raise ContinualAdaptationCampaignError(
            "evidence bundle policy_identifier must be non-empty"
        )
    ref = build_result_reference(uri, content=content, checksum=checksum)
    ref["identifier"] = identifier
    ref["policy_identifier"] = policy_identifier
    return ref


def build_continual_adaptation_evidence(  # noqa: PLR0913
    manifest: Mapping[str, Any],
    *,
    nominal_uri: str,
    shift_uri: str,
    forgetting_uri: str,
    evidence_bundle_uri: str,
    evidence_bundle_identifier: str,
    execution_mode: str = "native",
    nominal_content: str | bytes | None = None,
    shift_content: str | bytes | None = None,
    forgetting_content: str | bytes | None = None,
    source: str | Path | None = None,
) -> ContinualAdaptationEvidenceBundle:
    """Build a versioned evidence bundle for the continual-adaptation promotion gate.

    The bundle derives the adapted-policy identifier from the manifest using
    the merged validator, builds nominal/shift/forgetting result references,
    and constructs the evidence-bundle reference naming the derived identifier.
    Every *execution_mode* other than ``native`` fails closed.

    Args:
        manifest: A schema-valid ``continual_adaptation_run.v1`` mapping.
        nominal_uri: URI for the nominal evaluation result.
        shift_uri: URI for the shift evaluation result.
        forgetting_uri: URI for the forgetting evaluation result.
        evidence_bundle_uri: URI for the evidence bundle artifact.
        evidence_bundle_identifier: Non-empty identifier for the evidence
            bundle, distinct from the baseline policy identifier.
        execution_mode: Execution mode label; only ``native`` is accepted.
        nominal_content: Exact nominal-result bytes or text. Required.
        shift_content: Exact shift-result bytes or text. Required.
        forgetting_content: Exact forgetting-result bytes or text. Required.
        source: Optional source path for error messages.

    Returns:
        A :class:`ContinualAdaptationEvidenceBundle` with all result references
        and the evidence-bundle reference.

    Raises:
        ContinualAdaptationCampaignError: when execution mode is not native,
            the claim boundary is not metadata-only, or a reference cannot be built.
        ContinualAdaptationProtocolError: when the manifest is schema-invalid.
    """
    normalized_mode = _validate_execution_mode(execution_mode)

    report = check_continual_adaptation_run(manifest, source=source)
    if report.protocol_status != PROTOCOL_STATUS_VALID:
        raise ContinualAdaptationCampaignError(
            f"source manifest is not protocol-valid: {report.blockers}", source=source
        )

    claim_boundary = str(manifest.get("claim_boundary") or "")
    if "metadata-only" not in claim_boundary.lower():
        raise ContinualAdaptationCampaignError(
            "claim_boundary must explicitly identify this lane as metadata-only",
            source=source,
        )

    result_uris = [nominal_uri, shift_uri, forgetting_uri, evidence_bundle_uri]
    normalized_uris = [uri.strip() for uri in result_uris]
    if len(set(normalized_uris)) != len(normalized_uris):
        raise ContinualAdaptationCampaignError(
            "nominal, shift, forgetting, and evidence-bundle URIs must be distinct"
        )

    baseline_identifier = str(manifest["baseline_policy"]["identifier"])
    derived_identifier = derive_adapted_policy_identifier(manifest, source=source)

    nominal_ref = build_result_reference(
        nominal_uri,
        content=nominal_content,
    )
    shift_ref = build_result_reference(
        shift_uri,
        content=shift_content,
    )
    forgetting_ref = build_result_reference(
        forgetting_uri,
        content=forgetting_content,
    )
    payload = _evidence_payload(
        run_id=str(manifest["run_id"]),
        issue=int(manifest["issue"]),
        evidence_boundary=CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
        claim_boundary=claim_boundary,
        baseline_policy_identifier=baseline_identifier,
        derived_adapted_policy_identifier=derived_identifier,
        execution_mode=normalized_mode,
        nominal_result=nominal_ref,
        shift_result=shift_ref,
        forgetting_result=forgetting_ref,
    )
    evidence_ref = build_evidence_bundle_ref(
        identifier=evidence_bundle_identifier,
        uri=evidence_bundle_uri,
        policy_identifier=derived_identifier,
        baseline_identifier=baseline_identifier,
        content=_canonical_yaml_bytes(payload),
    )

    return ContinualAdaptationEvidenceBundle(
        schema_version=CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION,
        run_id=str(manifest["run_id"]),
        issue=int(manifest["issue"]),
        evidence_boundary=CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
        claim_boundary=claim_boundary,
        baseline_policy_identifier=baseline_identifier,
        derived_adapted_policy_identifier=derived_identifier,
        execution_mode=normalized_mode,
        nominal_result=nominal_ref,
        shift_result=shift_ref,
        forgetting_result=forgetting_ref,
        evidence_bundle_ref=evidence_ref,
    )


def build_promotion_results(
    evidence: ContinualAdaptationEvidenceBundle,
) -> dict[str, Any]:
    """Build the ``results`` block for a promotion-ready manifest.

    Returns:
        A mapping with ``nominal_result``, ``shift_result``,
        ``forgetting_result``, and ``evidence_bundle`` keys suitable for
        insertion into a ``continual_adaptation_run.v1`` manifest.
    """
    return {
        "nominal_result": dict(evidence.nominal_result),
        "shift_result": dict(evidence.shift_result),
        "forgetting_result": dict(evidence.forgetting_result),
        "evidence_bundle": dict(evidence.evidence_bundle_ref),
    }


def prepare_promotion_manifest(
    manifest: Mapping[str, Any],
    evidence: ContinualAdaptationEvidenceBundle,
    *,
    rationale: str = _METADATA_ONLY_PROMOTION_RATIONALE,
) -> dict[str, Any]:
    """Return a copy of *manifest* wired for promotion with *evidence*.

    The returned manifest has ``promotion_decision.decision`` set to
    ``'promote'`` and a complete ``results`` block built from *evidence*.
    The original manifest is not mutated.

    Returns:
        A new manifest mapping ready for
        :func:`check_continual_adaptation_run`.
    """
    if not evidence.is_promotion_ready:
        raise ContinualAdaptationCampaignError(
            f"cannot prepare a protocol fixture with blockers: {evidence.blockers}"
        )
    expected_bundle_digest = evidence.evidence_bundle_ref.get("checksum", {}).get("digest")
    actual_bundle_digest = _sha256_hex(_canonical_yaml_bytes(evidence.to_payload_dict()))
    if expected_bundle_digest != actual_bundle_digest:
        raise ContinualAdaptationCampaignError(
            "evidence bundle reference does not bind the deterministic payload"
        )

    promoted = dict(manifest)
    promoted["promotion_decision"] = {
        "decision": "promote",
        "rationale": rationale,
    }
    promoted["results"] = build_promotion_results(evidence)
    return promoted


def validate_promotion_readiness(
    manifest: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> ContinualAdaptationEvidenceBundle:
    """Validate that a promotion manifest satisfies the protocol gate.

    Runs :func:`check_continual_adaptation_run` on the manifest and returns
    an evidence bundle reflecting the result.  When the protocol check fails,
    the returned bundle carries the blockers and ``is_promotion_ready`` is
    ``False``.

    Returns:
        A :class:`ContinualAdaptationEvidenceBundle` reflecting the validation
        outcome.

    Raises:
        ContinualAdaptationProtocolError: when the manifest is schema-invalid.
    """
    report = check_continual_adaptation_run(manifest, source=source)
    results = manifest.get("results") or {}

    nominal_ref = dict(results.get("nominal_result") or {})
    shift_ref = dict(results.get("shift_result") or {})
    forgetting_ref = dict(results.get("forgetting_result") or {})
    evidence_ref = dict(results.get("evidence_bundle") or {})

    return ContinualAdaptationEvidenceBundle(
        schema_version=CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION,
        run_id=report.run_id,
        issue=report.issue,
        evidence_boundary=report.evidence_boundary,
        claim_boundary=str(manifest.get("claim_boundary") or ""),
        baseline_policy_identifier=report.baseline_policy_identifier,
        derived_adapted_policy_identifier=report.derived_adapted_policy_identifier,
        execution_mode="native",
        nominal_result=nominal_ref,
        shift_result=shift_ref,
        forgetting_result=forgetting_ref,
        evidence_bundle_ref=evidence_ref,
        promotion_gate_ready=report.promotion_ready,
        blockers=list(report.blockers),
    )


def write_evidence_bundle(
    evidence: ContinualAdaptationEvidenceBundle,
    out_path: Path,
    *,
    artifact_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the deterministic evidence bundle to *out_path*.

    Returns:
        The path to the written evidence bundle file.

    Raises:
        ContinualAdaptationCampaignError: when the file exists and
            *overwrite* is ``False``.
    """
    out_path = out_path.resolve()
    _validate_evidence_bundle_output_path(
        evidence.evidence_bundle_ref.get("uri"),
        out_path,
        artifact_root=artifact_root,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = evidence.to_payload_dict()
    payload_bytes = _canonical_yaml_bytes(payload)
    expected_digest = evidence.evidence_bundle_ref.get("checksum", {}).get("digest")
    actual_digest = _sha256_hex(payload_bytes)
    if expected_digest != actual_digest:
        raise ContinualAdaptationCampaignError(
            "evidence bundle reference does not bind the bytes being written",
            source=out_path,
        )
    _write_text_exclusively_unless_overwriting(
        out_path,
        payload_bytes.decode("utf-8"),
        overwrite=overwrite,
        artifact_name="evidence bundle",
    )
    return out_path


def _validate_evidence_bundle_output_path(
    uri: Any,
    out_path: Path,
    *,
    artifact_root: Path | None,
) -> None:
    """Ensure the emitted bundle path is the artifact named by its URI."""
    if not isinstance(uri, str) or not uri.strip():
        raise ContinualAdaptationCampaignError(
            "evidence bundle reference uri must be non-empty", source=out_path
        )
    relative = Path(uri)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContinualAdaptationCampaignError(
            f"evidence bundle uri must be a safe repository-relative path: {uri}",
            source=out_path,
        )
    resolved_out = out_path.resolve()
    if artifact_root is not None:
        expected = (artifact_root.resolve() / relative).resolve()
        if resolved_out != expected:
            raise ContinualAdaptationCampaignError(
                f"evidence bundle output must match its declared URI: expected {expected}, "
                f"got {resolved_out}",
                source=out_path,
            )
        return
    relative_parts = relative.parts
    output_suffix = resolved_out.parts[-len(relative_parts) :] if relative_parts else ()
    if len(relative_parts) > len(resolved_out.parts) or output_suffix != relative_parts:
        raise ContinualAdaptationCampaignError(
            f"evidence bundle output must end with its declared URI: {uri}", source=out_path
        )


def _resolve_local_artifact(
    root: Path,
    uri: str,
    *,
    name: str,
    source: str | Path | None,
) -> Path:
    """Resolve one safe repository-relative artifact URI.

    Returns:
        The resolved file path under *root*.
    """
    relative = Path(uri)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContinualAdaptationCampaignError(
            f"results.{name}.uri must be a safe repository-relative path: {uri}",
            source=source,
        )
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContinualAdaptationCampaignError(
            f"results.{name}.uri escapes artifact root: {uri}", source=source
        ) from exc
    if not path.is_file():
        raise ContinualAdaptationCampaignError(
            f"results.{name}.uri does not exist as a file: {uri}", source=source
        )
    return path


def verify_local_result_references(
    manifest: Mapping[str, Any],
    artifact_root: Path,
    *,
    include_evidence_bundle: bool = True,
    source: str | Path | None = None,
) -> dict[str, Path]:
    """Verify every promotion reference against exact local artifact bytes.

    This is deliberately stricter than the schema-level protocol validator,
    which validates checksum shape but cannot dereference arbitrary URIs. The
    metadata-only CLI uses repository-relative local paths, so it can and must
    prove that each declared digest matches the referenced file.

    Returns:
        A mapping from protocol reference name to verified local file path.
    """
    report = check_continual_adaptation_run(manifest, source=source)
    if report.protocol_status != PROTOCOL_STATUS_VALID or not report.promotion_ready:
        raise ContinualAdaptationCampaignError(
            f"manifest is not promotion-gate valid: {report.blockers}", source=source
        )

    root = artifact_root.resolve()
    results = manifest.get("results")
    if not isinstance(results, Mapping):
        raise ContinualAdaptationCampaignError("manifest results must be a mapping", source=source)

    resolved: dict[str, Path] = {}
    seen_uris: set[str] = set()
    reference_names = _RESULT_REF_NAMES + (("evidence_bundle",) if include_evidence_bundle else ())
    for name in reference_names:
        ref = results.get(name)
        if not isinstance(ref, Mapping):
            raise ContinualAdaptationCampaignError(
                f"results.{name} must be a mapping", source=source
            )
        uri = ref.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            raise ContinualAdaptationCampaignError(
                f"results.{name}.uri must be non-empty", source=source
            )
        if uri in seen_uris:
            raise ContinualAdaptationCampaignError(
                f"duplicate local result URI is not allowed: {uri}", source=source
            )
        seen_uris.add(uri)
        path = _resolve_local_artifact(root, uri, name=name, source=source)
        checksum = _validated_checksum(ref.get("checksum") or {})
        actual_digest = _sha256_hex(path.read_bytes())
        if checksum["digest"] != actual_digest:
            raise ContinualAdaptationCampaignError(
                f"results.{name} checksum mismatch for {uri}: "
                f"declared {checksum['digest']}, actual {actual_digest}",
                source=source,
            )
        resolved[name] = path
    return resolved


def write_promotion_manifest(
    manifest: Mapping[str, Any],
    out_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a promotion-ready manifest as YAML.

    Returns:
        The path to the written manifest file.

    Raises:
        ContinualAdaptationCampaignError: when the file exists and
            *overwrite* is ``False``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_exclusively_unless_overwriting(
        out_path,
        yaml.safe_dump(
            dict(manifest), default_flow_style=False, sort_keys=True, allow_unicode=True
        ),
        overwrite=overwrite,
        artifact_name="manifest",
    )
    return out_path


def _write_text_exclusively_unless_overwriting(
    path: Path,
    content: str,
    *,
    overwrite: bool,
    artifact_name: str,
) -> None:
    """Write text with atomic publication and race-safe no-overwrite semantics."""
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if overwrite:
                os.replace(temporary_path, path)
            else:
                # Hard-linking the fully written same-directory temporary file
                # publishes it atomically while preserving O_EXCL semantics.
                os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ContinualAdaptationCampaignError(
                f"{artifact_name} already exists: {path}", source=path
            ) from exc
    except FileExistsError as exc:
        raise ContinualAdaptationCampaignError(
            f"{artifact_name} already exists: {path}", source=path
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
