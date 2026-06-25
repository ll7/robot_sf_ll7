"""Annotation model for simulation-run analysis (issue #1646 child).

The analysis-workbench epic (#1646) needs a way for humans and agents to attach comments to
important moments in a run, anchored to durable evidence rather than loose screenshots, and to keep
**observation separate from hypothesis**. The renderer-neutral run/event timeline already exists
(``simulation_timeline.v1``); this module adds the matching pure, versioned **annotation model**:
comments anchored to frame ranges, event IDs, entities, and a source artifact.

This is schema/data only — no UI. Per the epic's queue note, browser/Three.js work stays blocked;
the sanctioned next step is trace/schema/report evidence, which this provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RUN_ANNOTATION_SCHEMA = "run_annotation.v1"

# Who authored the annotation.
AUTHOR_HUMAN = "human"
AUTHOR_AGENT = "agent"
_AUTHOR_KINDS = frozenset({AUTHOR_HUMAN, AUTHOR_AGENT})

# Observation vs hypothesis must stay separable (epic requirement).
STATEMENT_OBSERVATION = "observation"
STATEMENT_HYPOTHESIS = "hypothesis"
_STATEMENT_KINDS = frozenset({STATEMENT_OBSERVATION, STATEMENT_HYPOTHESIS})

# Canonical moment labels named by the epic.
CANONICAL_LABELS = frozenset(
    {
        "success",
        "near_miss",
        "collision_precursor",
        "deadlock",
        "social_force_artifact",
        "planner_issue",
        "policy_uncertainty",
    }
)


@dataclass(frozen=True, slots=True)
class RunAnnotation:
    """One annotation anchored to a frame range and durable evidence.

    Attributes:
        annotation_id: Stable identifier for the annotation.
        author: Author identifier (human handle or agent name).
        author_kind: ``human`` or ``agent``.
        frame_start: First anchored frame index (inclusive, >= 0).
        frame_end: Last anchored frame index (inclusive, >= frame_start).
        label: A canonical moment label (see ``CANONICAL_LABELS``).
        statement_kind: ``observation`` or ``hypothesis`` — kept separable by contract.
        text: The annotation text.
        source_artifact: Provenance — the timeline/trace artifact the annotation references.
        event_ids: Optional anchored event IDs.
        entity_ids: Optional anchored entity IDs (robot/pedestrian).
    """

    annotation_id: str
    author: str
    author_kind: str
    frame_start: int
    frame_end: int
    label: str
    statement_kind: str
    text: str
    source_artifact: str
    event_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the annotation contract."""
        if self.author_kind not in _AUTHOR_KINDS:
            raise ValueError(f"author_kind must be one of {sorted(_AUTHOR_KINDS)}")
        if self.statement_kind not in _STATEMENT_KINDS:
            raise ValueError(f"statement_kind must be one of {sorted(_STATEMENT_KINDS)}")
        if self.label not in CANONICAL_LABELS:
            raise ValueError(f"label must be one of {sorted(CANONICAL_LABELS)}, got {self.label!r}")
        if self.frame_start < 0:
            raise ValueError("frame_start must be >= 0")
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start")
        if not self.source_artifact:
            raise ValueError("source_artifact (provenance) is required")
        if not self.text.strip():
            raise ValueError("annotation text must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable annotation record.

        Returns:
            dict[str, Any]: Schema-tagged annotation payload.
        """
        return {
            "schema_version": RUN_ANNOTATION_SCHEMA,
            "annotation_id": self.annotation_id,
            "author": self.author,
            "author_kind": self.author_kind,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "label": self.label,
            "statement_kind": self.statement_kind,
            "text": self.text,
            "source_artifact": self.source_artifact,
            "event_ids": list(self.event_ids),
            "entity_ids": list(self.entity_ids),
        }


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    """A validated collection of annotations over a single run/timeline artifact."""

    source_artifact: str
    annotations: tuple[RunAnnotation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate annotation-id uniqueness and source consistency."""
        seen: set[str] = set()
        for annotation in self.annotations:
            if annotation.annotation_id in seen:
                raise ValueError(f"duplicate annotation_id: {annotation.annotation_id}")
            seen.add(annotation.annotation_id)
            if annotation.source_artifact != self.source_artifact:
                raise ValueError("all annotations must reference the set's source_artifact")

    def observations(self) -> tuple[RunAnnotation, ...]:
        """Return only the observation annotations.

        Returns:
            tuple[RunAnnotation, ...]: Annotations whose ``statement_kind`` is observation.
        """
        return tuple(a for a in self.annotations if a.statement_kind == STATEMENT_OBSERVATION)

    def hypotheses(self) -> tuple[RunAnnotation, ...]:
        """Return only the hypothesis annotations.

        Returns:
            tuple[RunAnnotation, ...]: Annotations whose ``statement_kind`` is hypothesis.
        """
        return tuple(a for a in self.annotations if a.statement_kind == STATEMENT_HYPOTHESIS)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable annotation-set record.

        Returns:
            dict[str, Any]: Schema-tagged annotation-set payload with observation/hypothesis split.
        """
        return {
            "schema_version": RUN_ANNOTATION_SCHEMA,
            "source_artifact": self.source_artifact,
            "n_annotations": len(self.annotations),
            "n_observations": len(self.observations()),
            "n_hypotheses": len(self.hypotheses()),
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }


__all__ = [
    "AUTHOR_AGENT",
    "AUTHOR_HUMAN",
    "CANONICAL_LABELS",
    "RUN_ANNOTATION_SCHEMA",
    "STATEMENT_HYPOTHESIS",
    "STATEMENT_OBSERVATION",
    "AnnotationSet",
    "RunAnnotation",
]
