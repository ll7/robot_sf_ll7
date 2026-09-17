"""Finding membership and compatible-case retrieval examples."""

from __future__ import annotations

import pytest

from robot_sf.analysis_workbench.audit_contracts import Annotation, EpisodeRef
from robot_sf.analysis_workbench.audit_findings import (
    FindingError,
    add_candidate,
    add_negative_control,
    confirm_member,
    create_from_annotation,
    finding_membership,
    new_finding,
)
from robot_sf.analysis_workbench.audit_similarity import (
    COMPATIBLE,
    INCOMPATIBLE,
    SIMILARITY_MODE_SAME_SCENARIO,
    UNKNOWN_COMPATIBILITY,
    compatible_case_similarity,
    find_similar_cases,
)


def _episode(
    execution_id: str, *, planner: str = "ppo", seed: int = 1, source: str = "b"
) -> EpisodeRef:
    return EpisodeRef(
        campaign_digest="a" * 64,
        source_digest=source * 64,
        execution_id=execution_id,
        planner_id=planner,
        scenario_id="corridor",
        seed=seed,
        config_digest="c" * 64,
    )


def test_finding_keeps_candidates_confirmations_and_controls_separate() -> None:
    finding = new_finding("f-1", "turning symptom")
    finding = add_candidate(finding, "episode-a")
    finding = add_candidate(finding, "episode-b")
    finding = confirm_member(finding, "episode-a", evidence={"kind": "human_review"})
    finding = add_negative_control(finding, "episode-control", evidence={"kind": "ordinary"})

    assert finding.candidate_members == ("episode-b",)
    assert finding.confirmed_members == ("episode-a",)
    assert finding.negative_controls == ("episode-control",)
    assert finding_membership(finding, "episode-a") == "confirmed"
    assert finding_membership(finding, "episode-control") == "negative_control"
    with pytest.raises(FindingError):
        add_candidate(finding, "episode-a")


def test_annotation_creates_candidate_not_causal_confirmation() -> None:
    annotation = Annotation(
        annotation_id="annotation-1",
        episode_id="episode-a",
        classification="planner_defect",
        mode="quick",
        observed_behavior="turns away from goal",
    )
    finding = create_from_annotation(annotation, finding_id="f-1")
    assert finding.status == "proposed"
    assert finding.candidate_members == ("episode-a",)
    assert finding.confirmed_members == ()


def test_similarity_explains_compatible_missing_and_incompatible_peers() -> None:
    query = _episode("query", planner="ppo", seed=1)
    compatible = _episode("other", planner="orca", seed=1)
    incompatible = _episode("changed", planner="orca", seed=1, source="d")
    missing = {"case_id": "missing", "scenario_id": "corridor", "planner_id": "orca"}

    result = compatible_case_similarity(query, compatible, mode=SIMILARITY_MODE_SAME_SCENARIO)
    assert result.compatibility == COMPATIBLE
    assert result.is_compatible
    assert "scenario_id matches" in result.reasons
    assert result.features["scenario_match"] is True

    bad = compatible_case_similarity(query, incompatible, mode=SIMILARITY_MODE_SAME_SCENARIO)
    assert bad.compatibility == INCOMPATIBLE
    assert bad.score == 0.0
    assert "mismatch:source_digest" in bad.missingness

    unknown = compatible_case_similarity(query, missing, mode=SIMILARITY_MODE_SAME_SCENARIO)
    assert unknown.compatibility == UNKNOWN_COMPATIBILITY
    assert "campaign_digest" in unknown.missingness

    ranked = find_similar_cases(
        query, [incompatible, compatible], mode=SIMILARITY_MODE_SAME_SCENARIO
    )
    assert [item.candidate_id for item in ranked] == [compatible.episode_id]
