"""Campaign integration for continual-adaptation nominal/shift/forgetting evidence (issue #6657).

Connects continual-adaptation runs to the benchmark campaign machinery so an
adaptation run produces nominal/shift/forgetting evaluation result references
and a versioned evidence bundle under ``docs/context/evidence/`` naming the
validator-derived adapted-policy identifier distinct from baseline, satisfying
the promotion gate in
:func:`robot_sf.research.continual_adaptation_protocol.check_continual_adaptation_run`.

This module is metadata-only: it builds evidence-bundle structure and result
references but does not launch training, run evaluations, write checkpoints,
mutate the safety wrapper, or promote a policy.  Fallback or degraded execution
fails closed -- a result reference produced under fallback/degraded mode is
never presented as benchmark evidence.

The evidence bundle is versioned via ``continual_adaptation_evidence.v1`` and
stamps the protocol evidence boundary so a passing promotion gate is never
mistaken for an executed adaptation or benchmark/paper evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.errors import RobotSfError
from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
    check_continual_adaptation_run,
    derive_adapted_policy_identifier,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION = "continual_adaptation_evidence.v1"

#: Result reference names required by the promotion gate.
_RESULT_REF_NAMES = ("nominal_result", "shift_result", "forgetting_result")

#: Execution modes that are never benchmark evidence.
_FORBIDDEN_EXECUTION_MODES = frozenset({"fallback", "degraded"})


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
    baseline_policy_identifier: str
    derived_adapted_policy_identifier: str
    execution_mode: str
    nominal_result: dict[str, Any]
    shift_result: dict[str, Any]
    forgetting_result: dict[str, Any]
    evidence_bundle_ref: dict[str, Any]
    created_utc: str
    blockers: list[str] = field(default_factory=list)

    @property
    def is_promotion_ready(self) -> bool:
        """Return ``True`` when the bundle has no blockers."""
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


def _sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _checksum_for_content(content: str) -> dict[str, str]:
    """Build a ``{algorithm, digest}`` checksum mapping for *content*.

    Returns:
        A mapping with ``algorithm`` and ``digest`` keys.
    """
    return {"algorithm": "sha256", "digest": _sha256_hex(content.encode("utf-8"))}


def _validate_execution_mode(execution_mode: str) -> None:
    """Fail closed when the execution mode is fallback or degraded."""
    normalized = execution_mode.strip().lower()
    if normalized in _FORBIDDEN_EXECUTION_MODES:
        raise ContinualAdaptationCampaignError(
            f"execution_mode={execution_mode!r} is fallback/degraded; "
            "fallback or degraded execution is not benchmark evidence and cannot "
            "produce promotion-ready nominal/shift/forgetting results"
        )


def build_result_reference(
    uri: str,
    *,
    content: str | None = None,
    checksum: dict[str, str] | None = None,
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
    resolved_checksum = checksum if checksum is not None else _checksum_for_content(content)
    return {"uri": uri, "checksum": resolved_checksum}


def build_evidence_bundle_ref(
    *,
    identifier: str,
    uri: str,
    policy_identifier: str,
    baseline_identifier: str,
    content: str | None = None,
    checksum: dict[str, str] | None = None,
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
    nominal_content: str | None = None,
    shift_content: str | None = None,
    forgetting_content: str | None = None,
    evidence_bundle_content: str | None = None,
    source: str | Path | None = None,
) -> ContinualAdaptationEvidenceBundle:
    """Build a versioned evidence bundle for the continual-adaptation promotion gate.

    The bundle derives the adapted-policy identifier from the manifest using
    the merged validator, builds nominal/shift/forgetting result references,
    and constructs the evidence-bundle reference naming the derived identifier.
    Fallback or degraded *execution_mode* fails closed.

    Args:
        manifest: A schema-valid ``continual_adaptation_run.v1`` mapping.
        nominal_uri: URI for the nominal evaluation result.
        shift_uri: URI for the shift evaluation result.
        forgetting_uri: URI for the forgetting evaluation result.
        evidence_bundle_uri: URI for the evidence bundle artifact.
        evidence_bundle_identifier: Non-empty identifier for the evidence
            bundle, distinct from the baseline policy identifier.
        execution_mode: Execution mode label; ``fallback`` or ``degraded``
            fails closed.
        nominal_content: Optional content for computing the nominal checksum.
        shift_content: Optional content for computing the shift checksum.
        forgetting_content: Optional content for computing the forgetting checksum.
        evidence_bundle_content: Optional content for computing the evidence
            bundle checksum.
        source: Optional source path for error messages.

    Returns:
        A :class:`ContinualAdaptationEvidenceBundle` with all result references
        and the evidence-bundle reference.

    Raises:
        ContinualAdaptationCampaignError: when execution mode is
            fallback/degraded, or a reference cannot be built.
        ContinualAdaptationProtocolError: when the manifest is schema-invalid.
    """
    _validate_execution_mode(execution_mode)

    baseline_identifier = str(manifest["baseline_policy"]["identifier"])
    derived_identifier = derive_adapted_policy_identifier(manifest, source=source)

    nominal_ref = build_result_reference(
        nominal_uri,
        content=nominal_content if nominal_content is not None else f"nominal:{nominal_uri}",
    )
    shift_ref = build_result_reference(
        shift_uri,
        content=shift_content if shift_content is not None else f"shift:{shift_uri}",
    )
    forgetting_ref = build_result_reference(
        forgetting_uri,
        content=(
            forgetting_content if forgetting_content is not None else f"forgetting:{forgetting_uri}"
        ),
    )
    evidence_ref = build_evidence_bundle_ref(
        identifier=evidence_bundle_identifier,
        uri=evidence_bundle_uri,
        policy_identifier=derived_identifier,
        baseline_identifier=baseline_identifier,
        content=(
            evidence_bundle_content
            if evidence_bundle_content is not None
            else f"evidence:{evidence_bundle_uri}"
        ),
    )

    return ContinualAdaptationEvidenceBundle(
        schema_version=CONTINUAL_ADAPTATION_EVIDENCE_SCHEMA_VERSION,
        run_id=str(manifest["run_id"]),
        issue=int(manifest["issue"]),
        evidence_boundary=CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
        baseline_policy_identifier=baseline_identifier,
        derived_adapted_policy_identifier=derived_identifier,
        execution_mode=execution_mode,
        nominal_result=nominal_ref,
        shift_result=shift_ref,
        forgetting_result=forgetting_ref,
        evidence_bundle_ref=evidence_ref,
        created_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
    rationale: str = "All nominal/shift/forgetting gates passed; evidence bundle complete.",
) -> dict[str, Any]:
    """Return a copy of *manifest* wired for promotion with *evidence*.

    The returned manifest has ``promotion_decision.decision`` set to
    ``'promote'`` and a complete ``results`` block built from *evidence*.
    The original manifest is not mutated.

    Returns:
        A new manifest mapping ready for
        :func:`check_continual_adaptation_run`.
    """
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
        baseline_policy_identifier=report.baseline_policy_identifier,
        derived_adapted_policy_identifier=report.derived_adapted_policy_identifier,
        execution_mode="native",
        nominal_result=nominal_ref,
        shift_result=shift_ref,
        forgetting_result=forgetting_ref,
        evidence_bundle_ref=evidence_ref,
        created_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        blockers=list(report.blockers),
    )


def write_evidence_bundle(
    evidence: ContinualAdaptationEvidenceBundle,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the evidence bundle as a YAML file under *out_dir*.

    The file is named ``<run_id>_evidence.yaml`` and contains the full
    evidence bundle dictionary.

    Returns:
        The path to the written evidence bundle file.

    Raises:
        ContinualAdaptationCampaignError: when the file exists and
            *overwrite* is ``False``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{evidence.run_id}_evidence.yaml"
    out_path = out_dir / filename
    if out_path.exists() and not overwrite:
        raise ContinualAdaptationCampaignError(
            f"evidence bundle already exists: {out_path}", source=out_path
        )
    payload = evidence.to_dict()
    out_path.write_text(
        yaml.dump(payload, default_flow_style=False, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return out_path


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
    if out_path.exists() and not overwrite:
        raise ContinualAdaptationCampaignError(
            f"manifest already exists: {out_path}", source=out_path
        )
    out_path.write_text(
        yaml.dump(dict(manifest), default_flow_style=False, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return out_path
