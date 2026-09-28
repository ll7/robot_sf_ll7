"""Finding lifecycle helpers for durable, evidence-separated audit knowledge."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from robot_sf.analysis_workbench.audit_contracts import (
    Annotation,
    Finding,
    utc_now,
)
from robot_sf.analysis_workbench.audit_store import AuditStore, CommitResult, StoredRecord


class FindingError(ValueError):
    """Raised for an invalid finding transition or membership update."""


def _append_unique(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    if not value:
        raise FindingError("finding member IDs must be non-empty")
    return values if value in values else (*values, value)


def _remove(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    return tuple(item for item in values if item != value)


def new_finding(
    finding_id: str,
    title: str,
    *,
    status: str = "proposed",
    tags: tuple[str, ...] = (),
    source_revision: str = "",
) -> Finding:
    """Create an empty finding without claiming any manifestation is confirmed.

    Returns:
        A proposed finding with no members.
    """

    return Finding(
        finding_id=finding_id,
        title=title,
        status=status,
        tags=tuple(tags),
        source_revision=source_revision,
    )


def add_candidate(finding: Finding, episode_id: str) -> Finding:
    """Add one candidate manifestation while retaining separate counts.

    Returns:
        Finding with the candidate membership appended.
    """

    if episode_id in finding.negative_controls or episode_id in finding.confirmed_members:
        raise FindingError("a confirmed or negative-control member cannot be a candidate")
    return replace(
        finding,
        candidate_members=_append_unique(finding.candidate_members, episode_id),
        updated_at=utc_now(),
    )


def confirm_member(
    finding: Finding,
    episode_id: str,
    *,
    evidence: dict[str, Any] | None = None,
    retain_candidate: bool = False,
) -> Finding:
    """Promote a candidate to confirmed only with an explicit caller action.

    Similarity and agent suggestions never call this helper implicitly.  By
    default promotion removes the current candidate membership, while the
    optional ``retain_candidate`` mode preserves a provenance count for
    clients that model candidate and confirmed as independent sets.

    Returns:
        Finding with explicit confirmation evidence.
    """

    if episode_id in finding.negative_controls:
        raise FindingError("a negative-control member cannot be confirmed")
    candidate_members = finding.candidate_members
    if not retain_candidate:
        candidate_members = _remove(candidate_members, episode_id)
    updated = replace(
        finding,
        candidate_members=candidate_members,
        confirmed_members=_append_unique(finding.confirmed_members, episode_id),
        evidence=(*finding.evidence, evidence) if evidence is not None else finding.evidence,
        updated_at=utc_now(),
    )
    return updated


def add_negative_control(
    finding: Finding,
    episode_id: str,
    *,
    evidence: dict[str, Any] | None = None,
) -> Finding:
    """Record a detector-independent negative/control manifestation.

    Returns:
        Finding with the negative-control membership appended.
    """

    if episode_id in finding.confirmed_members:
        raise FindingError("a confirmed member cannot be a negative control")
    return replace(
        finding,
        candidate_members=_remove(finding.candidate_members, episode_id),
        negative_controls=_append_unique(finding.negative_controls, episode_id),
        negative_evidence=(
            (*finding.negative_evidence, evidence)
            if evidence is not None
            else finding.negative_evidence
        ),
        updated_at=utc_now(),
    )


def add_observation(finding: Finding, observation: str) -> Finding:
    """Append an observation without converting it into a hypothesis.

    Returns:
        Finding with the observation appended.
    """

    if not observation.strip():
        raise FindingError("observation must be non-empty")
    return replace(
        finding,
        observations=_append_unique(finding.observations, observation),
        updated_at=utc_now(),
    )


def add_hypothesis(finding: Finding, hypothesis: str) -> Finding:
    """Append a competing hypothesis; no status promotion occurs.

    Returns:
        Finding with the hypothesis appended.
    """

    if not hypothesis.strip():
        raise FindingError("hypothesis must be non-empty")
    return replace(
        finding, hypotheses=_append_unique(finding.hypotheses, hypothesis), updated_at=utc_now()
    )


def add_diagnostic_result(finding: Finding, result: dict[str, Any]) -> Finding:
    """Attach a diagnostic result while preserving its independent evidence status.

    Returns:
        Finding with the diagnostic result appended.
    """

    return replace(
        finding,
        diagnostic_results=(*finding.diagnostic_results, dict(result)),
        updated_at=utc_now(),
    )


def transition(finding: Finding, status: str) -> Finding:
    """Apply an explicit lifecycle transition.

    Returns:
        Finding with the requested status.
    """

    if status not in {"proposed", "under_investigation", "supported", "refuted", "resolved"}:
        raise FindingError(f"unknown finding status: {status}")
    return replace(finding, status=status, updated_at=utc_now())


def create_from_annotation(
    annotation: Annotation,
    *,
    finding_id: str,
    title: str | None = None,
) -> Finding:
    """Create a candidate finding from one annotation, never a confirmation.

    Returns:
        A proposed finding containing the annotation's episode as a candidate.
    """

    finding = new_finding(finding_id, title or annotation.classification)
    finding = add_candidate(finding, annotation.episode_id)
    return (
        add_observation(finding, annotation.observed_behavior)
        if annotation.observed_behavior
        else finding
    )


def finding_membership(finding: Finding, episode_id: str) -> str | None:
    """Return ``candidate``, ``confirmed``, ``negative_control``, or ``None``."""

    if episode_id in finding.confirmed_members:
        return "confirmed"
    if episode_id in finding.negative_controls:
        return "negative_control"
    if episode_id in finding.candidate_members:
        return "candidate"
    return None


def findings_for_episode(
    findings: list[Finding] | tuple[Finding, ...] | list[StoredRecord],
    episode_id: str,
) -> list[Finding]:
    """Return findings that mention an episode in any evidence membership."""

    result: list[Finding] = []
    for value in findings:
        finding = value.record if isinstance(value, StoredRecord) else value
        if not isinstance(finding, Finding):
            continue
        if finding_membership(finding, episode_id) is not None:
            result.append(finding)
    return sorted(result, key=lambda item: item.finding_id)


class FindingStore:
    """Small service adapter that keeps finding writes on the shared store."""

    def __init__(self, store: AuditStore):
        """Bind finding operations to an existing canonical store."""

        self.store = store

    def create(self, finding: Finding, *, operation_id: str, actor: str = "human") -> CommitResult:
        """Create a finding at revision zero.

        Returns:
            Durable commit receipt.
        """

        return self.store.save(finding, operation_id=operation_id, expected_revision=0, actor=actor)

    def get(self, finding_id: str) -> Finding | None:
        """Load one finding or return ``None`` when absent.

        Returns:
            Typed finding, or ``None``.
        """

        stored = self.store.get(finding_id)
        if stored is None:
            return None
        if not isinstance(stored.record, Finding):
            raise FindingError(f"record {finding_id!r} is not a finding")
        return stored.record

    def update(
        self,
        finding: Finding,
        *,
        operation_id: str,
        expected_revision: int,
        actor: str = "human",
    ) -> CommitResult:
        """Persist an explicitly revised finding.

        Returns:
            Durable commit receipt.
        """

        return self.store.save(
            finding,
            operation_id=operation_id,
            expected_revision=expected_revision,
            actor=actor,
        )

    def mutate(
        self,
        finding_id: str,
        mutation: Any,
        *,
        operation_id: str,
        expected_revision: int | None = None,
        actor: str = "human",
    ) -> CommitResult:
        """Apply a pure finding mutation with compare-and-swap semantics.

        Returns:
            Durable commit receipt.
        """

        stored = self.store.get(finding_id)
        if stored is None or not isinstance(stored.record, Finding):
            raise FindingError(f"finding {finding_id!r} does not exist")
        updated = mutation(stored.record)
        if not isinstance(updated, Finding):
            raise FindingError("finding mutation must return a Finding")
        return self.update(
            updated,
            operation_id=operation_id,
            expected_revision=stored.revision if expected_revision is None else expected_revision,
            actor=actor,
        )

    def add_candidate(
        self, finding_id: str, episode_id: str, *, operation_id: str, actor: str = "human"
    ) -> CommitResult:
        """Persist a candidate membership update.

        Returns:
            Durable commit receipt.
        """

        return self.mutate(
            finding_id,
            lambda item: add_candidate(item, episode_id),
            operation_id=operation_id,
            actor=actor,
        )

    def confirm_member(
        self,
        finding_id: str,
        episode_id: str,
        *,
        operation_id: str,
        evidence: dict[str, Any] | None = None,
        actor: str = "human",
    ) -> CommitResult:
        """Persist an explicit confirmation membership update.

        Returns:
            Durable commit receipt.
        """

        return self.mutate(
            finding_id,
            lambda item: confirm_member(item, episode_id, evidence=evidence),
            operation_id=operation_id,
            actor=actor,
        )

    def add_negative_control(
        self,
        finding_id: str,
        episode_id: str,
        *,
        operation_id: str,
        evidence: dict[str, Any] | None = None,
        actor: str = "human",
    ) -> CommitResult:
        """Persist a detector-independent negative-control update.

        Returns:
            Durable commit receipt.
        """

        return self.mutate(
            finding_id,
            lambda item: add_negative_control(item, episode_id, evidence=evidence),
            operation_id=operation_id,
            actor=actor,
        )


__all__ = [
    "FindingError",
    "FindingStore",
    "add_candidate",
    "add_diagnostic_result",
    "add_hypothesis",
    "add_negative_control",
    "add_observation",
    "confirm_member",
    "create_from_annotation",
    "finding_membership",
    "findings_for_episode",
    "new_finding",
    "transition",
]
