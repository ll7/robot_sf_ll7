"""Tests for the run annotation model (issue #1646 child)."""

from __future__ import annotations

import pytest

from robot_sf.analysis.run_annotation import (
    RUN_ANNOTATION_SCHEMA,
    STATEMENT_HYPOTHESIS,
    STATEMENT_OBSERVATION,
    AnnotationSet,
    RunAnnotation,
)


def _annotation(**overrides) -> RunAnnotation:
    """Build a valid annotation with optional overrides."""
    kwargs = {
        "annotation_id": "a1",
        "author": "ll7",
        "author_kind": "human",
        "frame_start": 10,
        "frame_end": 20,
        "label": "near_miss",
        "statement_kind": STATEMENT_OBSERVATION,
        "text": "robot passed within 0.2 m of a pedestrian",
        "source_artifact": "docs/context/evidence/run/timeline.json",
        "event_ids": ("evt_3",),
        "entity_ids": ("ped_7",),
    }
    kwargs.update(overrides)
    return RunAnnotation(**kwargs)


def test_valid_annotation_serializes_with_anchors() -> None:
    """A valid annotation must serialize with its frame range, anchors, and provenance."""
    payload = _annotation().to_dict()

    assert payload["schema_version"] == RUN_ANNOTATION_SCHEMA
    assert payload["frame_start"] == 10
    assert payload["frame_end"] == 20
    assert payload["event_ids"] == ["evt_3"]
    assert payload["entity_ids"] == ["ped_7"]
    assert payload["source_artifact"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"author_kind": "robot"},  # not human/agent
        {"statement_kind": "guess"},  # not observation/hypothesis
        {"label": "vibes"},  # not canonical
        {"frame_start": -1},
        {"frame_start": 30, "frame_end": 20},  # end < start
        {"source_artifact": ""},  # missing provenance
        {"text": "   "},  # empty text
    ],
)
def test_invalid_annotations_fail_closed(overrides: dict[str, object]) -> None:
    """Contract violations must fail closed at construction."""
    with pytest.raises(ValueError):
        _annotation(**overrides)


def test_observation_and_hypothesis_stay_separable() -> None:
    """The set must split observation from hypothesis annotations."""
    src = "docs/context/evidence/run/timeline.json"
    annotations = (
        _annotation(annotation_id="a1", statement_kind=STATEMENT_OBSERVATION, source_artifact=src),
        _annotation(
            annotation_id="a2",
            statement_kind=STATEMENT_HYPOTHESIS,
            label="planner_issue",
            text="the planner likely under-weighted the crossing pedestrian",
            source_artifact=src,
        ),
    )
    annotation_set = AnnotationSet(source_artifact=src, annotations=annotations)

    assert len(annotation_set.observations()) == 1
    assert len(annotation_set.hypotheses()) == 1
    payload = annotation_set.to_dict()
    assert payload["n_observations"] == 1
    assert payload["n_hypotheses"] == 1
    assert payload["n_annotations"] == 2


def test_duplicate_annotation_id_is_rejected() -> None:
    """Duplicate annotation ids within a set must fail closed."""
    src = "x.json"
    with pytest.raises(ValueError):
        AnnotationSet(
            source_artifact=src,
            annotations=(
                _annotation(annotation_id="dup", source_artifact=src),
                _annotation(annotation_id="dup", source_artifact=src),
            ),
        )


def test_annotation_must_reference_set_source_artifact() -> None:
    """An annotation pointing at a different artifact than its set must fail closed."""
    with pytest.raises(ValueError):
        AnnotationSet(
            source_artifact="a.json",
            annotations=(_annotation(source_artifact="b.json"),),
        )
