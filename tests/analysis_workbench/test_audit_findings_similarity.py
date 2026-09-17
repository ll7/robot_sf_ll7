"""Finding membership and compatible-case retrieval examples."""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest

from robot_sf.analysis_workbench.audit_contracts import Annotation, EpisodeRef, Signal
from robot_sf.analysis_workbench.audit_findings import (
    FindingError,
    FindingStore,
    add_candidate,
    add_diagnostic_result,
    add_hypothesis,
    add_negative_control,
    add_observation,
    confirm_member,
    create_from_annotation,
    finding_membership,
    findings_for_episode,
    new_finding,
    transition,
)
from robot_sf.analysis_workbench.audit_similarity import (
    COMPATIBLE,
    INCOMPATIBLE,
    SIMILARITY_MODE_FINDING,
    SIMILARITY_MODE_GEOMETRY,
    SIMILARITY_MODE_OUTCOME,
    SIMILARITY_MODE_SAME_PLANNER,
    SIMILARITY_MODE_SAME_SCENARIO,
    SIMILARITY_MODE_SYMPTOM,
    UNKNOWN_COMPATIBILITY,
    SimilarityError,
    SimilarityResult,
    compatible_case_similarity,
    filter_compatible_cases,
    find_similar_cases,
    similar_cases,
)
from robot_sf.analysis_workbench.audit_store import AuditStore, StoredRecord


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
        environment_digest="e" * 64,
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


def test_finding_mutations_cover_guarded_evidence_and_lifecycle_paths() -> None:
    finding = new_finding("f-guards", "guarded transitions")
    with pytest.raises(FindingError, match="non-empty"):
        add_candidate(finding, "")
    finding = add_candidate(finding, "episode-a")
    assert add_candidate(finding, "episode-a").candidate_members == ("episode-a",)
    finding = confirm_member(
        finding,
        "episode-a",
        evidence={"kind": "human_review"},
        retain_candidate=True,
    )
    assert finding.candidate_members == ("episode-a",)
    assert finding.confirmed_members == ("episode-a",)
    with pytest.raises(FindingError, match="confirmed"):
        add_negative_control(finding, "episode-a")

    finding = add_negative_control(finding, "episode-control")
    with pytest.raises(FindingError, match="negative-control"):
        confirm_member(finding, "episode-control")
    with pytest.raises(FindingError, match="confirmed"):
        add_candidate(finding, "episode-a")
    with pytest.raises(FindingError, match="confirmed"):
        add_negative_control(finding, "episode-a")

    with pytest.raises(FindingError, match="observation"):
        add_observation(finding, " \t")
    finding = add_observation(finding, "observed turn")
    assert add_observation(finding, "observed turn").observations == ("observed turn",)
    with pytest.raises(FindingError, match="hypothesis"):
        add_hypothesis(finding, " \t")
    finding = add_hypothesis(finding, "controller instability")
    finding = add_diagnostic_result(finding, {"metric": "clearance", "value": 0.2})
    assert finding.diagnostic_results == ({"metric": "clearance", "value": 0.2},)
    with pytest.raises(FindingError, match="unknown"):
        transition(finding, "unclassified")
    finding = transition(finding, "under_investigation")
    assert finding.status == "under_investigation"


def test_findings_for_episode_handles_typed_stored_and_unrelated_values() -> None:
    candidate = add_candidate(new_finding("f-candidate", "candidate"), "episode-target")
    confirmed = confirm_member(
        add_candidate(new_finding("f-confirmed", "confirmed"), "episode-target"),
        "episode-target",
    )
    unrelated = new_finding("f-unrelated", "unrelated")
    stored = StoredRecord(
        record_id=confirmed.finding_id,
        record_type="finding",
        revision=1,
        deleted=False,
        record=confirmed,
        operation_id="op-confirmed",
        committed_at="2026-01-01T00:00:00+00:00",
        global_revision=1,
    )
    annotation = Annotation(
        annotation_id="not-a-finding",
        episode_id="episode-target",
        classification="unclear",
    )
    assert finding_membership(candidate, "episode-target") == "candidate"
    assert finding_membership(confirmed, "episode-target") == "confirmed"
    assert (
        finding_membership(add_negative_control(unrelated, "episode-target"), "episode-target")
        == "negative_control"
    )
    assert finding_membership(unrelated, "episode-target") is None
    matched = findings_for_episode([unrelated, annotation, stored, candidate], "episode-target")
    assert [item.finding_id for item in matched] == ["f-candidate", "f-confirmed"]
    assert findings_for_episode([unrelated, annotation, stored], "missing") == []


def test_finding_store_persists_mutations_and_rejects_invalid_callbacks(tmp_path) -> None:
    finding = new_finding("f-store", "stored finding")
    with AuditStore(tmp_path) as store:
        finding_store = FindingStore(store)
        created = finding_store.create(finding, operation_id="op-create")
        assert created.record_id == "f-store"
        assert finding_store.get("missing") is None
        assert finding_store.get("f-store") == finding

        with pytest.raises(FindingError, match="does not exist"):
            finding_store.mutate("missing", lambda item: item, operation_id="op-missing")
        with pytest.raises(FindingError, match="must return"):
            finding_store.mutate("f-store", lambda item: "not-a-finding", operation_id="op-bad")

        annotation = Annotation(
            annotation_id="stored-annotation",
            episode_id="episode-other",
            classification="unclear",
        )
        store.save(annotation, operation_id="op-annotation", expected_revision=0)
        with pytest.raises(FindingError, match="not a finding"):
            finding_store.get(annotation.annotation_id)

        finding_store.add_candidate("f-store", "episode-a", operation_id="op-candidate")
        finding_store.confirm_member("f-store", "episode-a", operation_id="op-confirm")
        finding_store.add_negative_control("f-store", "episode-control", operation_id="op-control")
        assert finding_store.get("f-store").confirmed_members == ("episode-a",)


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


def test_similarity_missing_mode_fields_and_absent_seeds_are_unknown() -> None:
    query = _episode("query", planner="ppo", seed=1)
    missing_scenario = {
        "case_id": "missing-scenario",
        "campaign_digest": "a" * 64,
        "source_digest": "b" * 64,
        "planner_id": "orca",
    }
    unknown_scenario = compatible_case_similarity(query, missing_scenario)
    assert unknown_scenario.compatibility == UNKNOWN_COMPATIBILITY
    assert "scenario_id" in unknown_scenario.missingness

    no_seed_left = _episode("no-seed-left", seed=None)
    no_seed_right = _episode("no-seed-right", seed=None)
    same_planner = compatible_case_similarity(
        no_seed_left,
        no_seed_right,
        mode="same_planner_across_seeds",
    )
    assert same_planner.compatibility == UNKNOWN_COMPATIBILITY
    assert "seed" in same_planner.missingness


def test_similarity_excludes_nonfinite_metrics_as_missing() -> None:
    query = _episode("query")
    query_payload = asdict(query)
    query_payload["case_id"] = query.episode_id
    candidate_payload = asdict(query)
    candidate_payload["case_id"] = "candidate"
    result = compatible_case_similarity(
        {**query_payload, "metrics": {"clearance": math.nan}},
        {**candidate_payload, "metrics": {"clearance": 1.0}},
        mode="metric_behaviour",
    )
    assert result.compatibility == UNKNOWN_COMPATIBILITY
    assert any("nonfinite" in field for field in result.missingness)


def test_similarity_result_validates_and_exposes_explanation_aliases() -> None:
    with pytest.raises(SimilarityError, match="compatibility"):
        SimilarityResult("query", "candidate", "mode", 0.5, "invalid")
    with pytest.raises(SimilarityError, match="score"):
        SimilarityResult("query", "candidate", "mode", 1.1, COMPATIBLE)
    result = SimilarityResult(
        "query",
        "candidate",
        SIMILARITY_MODE_SYMPTOM,
        0.5,
        COMPATIBLE,
        reasons=("shared features",),
        features={"shared": ["collision"]},
        missingness=("seed",),
        warnings=("retrieval aid",),
    )
    assert result.compatibility_status == COMPATIBLE
    assert result.candidate_episode_id == "candidate"
    assert result.matched_features == {"shared": ["collision"]}
    assert result.is_compatible
    assert result.explain() == result.to_dict()


def test_similarity_modes_cover_identity_mapping_and_missing_feature_paths() -> None:
    query = _episode("query", planner="ppo", seed=1)
    same_seed = _episode("same-seed", planner="ppo", seed=1)
    different_seed = _episode("different-seed", planner="ppo", seed=2)
    different_planner = _episode("different-planner", planner="orca", seed=2)
    assert compatible_case_similarity(query, same_seed, mode="same_planner").score == 0.6
    assert any(
        "seed matches" in reason
        for reason in compatible_case_similarity(
            query, same_seed, mode=SIMILARITY_MODE_SAME_PLANNER
        ).reasons
    )
    assert (
        compatible_case_similarity(query, different_seed, mode=SIMILARITY_MODE_SAME_PLANNER).score
        == 0.85
    )
    planner_mismatch = compatible_case_similarity(
        query, different_planner, mode=SIMILARITY_MODE_SAME_PLANNER
    )
    assert planner_mismatch.score == 0.0
    assert planner_mismatch.features["planner_match"] is False

    scenario_mismatch = asdict(query)
    scenario_mismatch.update({"case_id": "different-scenario", "scenario_id": "open_space"})
    scenario_result = compatible_case_similarity(query, scenario_mismatch)
    assert scenario_result.features["scenario_match"] is False
    missing_planner = asdict(query)
    missing_planner.update({"case_id": "missing-planner", "planner_id": ""})
    planner_missing_result = compatible_case_similarity(query, missing_planner)
    assert "planner_id" in planner_missing_result.missingness
    missing_scenario = asdict(query)
    missing_scenario.update({"case_id": "missing-scenario", "scenario_id": ""})
    assert "scenario_id" in compatible_case_similarity(query, missing_scenario).missingness

    with pytest.raises(SimilarityError, match="unsupported case"):
        compatible_case_similarity(query, object())
    with pytest.raises(SimilarityError, match="stable ID"):
        compatible_case_similarity(query, {"campaign_digest": "a" * 64})
    with pytest.raises(SimilarityError, match="similarity mode"):
        compatible_case_similarity(query, same_seed, mode="not-a-mode")


def test_similarity_symptom_geometry_outcome_and_nested_case_paths() -> None:
    query = asdict(_episode("query"))
    query["case_id"] = "query"
    query.update(
        {
            "symptoms": ["collision", "near_miss"],
            "tags": {"crowded": True, "ignored": None},
            "geometry_signature": ["map-a", "corridor"],
            "metrics": {"clearance": 1.0, "query_only": 3.0, "flag": True},
            "outcome": {"success": 1.0},
        }
    )
    candidate = dict(query)
    candidate.update(
        {
            "case_id": "candidate",
            "symptoms": ["collision"],
            "tags": {"different": True},
            "scenario_id": "different-scenario",
            "geometry_signature": ["map-b"],
            "metrics": {"clearance": 2.0, "candidate_only": 4.0, "text": "ignored"},
            "outcome": 5.0,
        }
    )
    symptom = compatible_case_similarity(query, candidate, mode=SIMILARITY_MODE_SYMPTOM)
    assert symptom.features["shared_features"] == ["collision"]
    assert symptom.score > 0.0
    anomaly = compatible_case_similarity(
        query, {**candidate, "symptoms": ["unrelated"]}, mode="anomaly"
    )
    assert "no shared symptom/anomaly features" in anomaly.reasons
    finding = compatible_case_similarity(query, candidate, mode=SIMILARITY_MODE_FINDING)
    assert finding.features["candidate_features"]
    geometry = compatible_case_similarity(query, candidate, mode=SIMILARITY_MODE_GEOMETRY)
    assert geometry.features["shared_geometry_features"] == []
    assert "geometry differs" in geometry.reasons
    missing_geometry = dict(candidate)
    for key in ("geometry_signature", "geometry", "map_id", "scenario_id"):
        missing_geometry.pop(key, None)
    assert (
        "geometry_signature"
        in compatible_case_similarity(
            query, missing_geometry, mode=SIMILARITY_MODE_GEOMETRY
        ).missingness
    )

    outcome = compatible_case_similarity(query, candidate, mode=SIMILARITY_MODE_OUTCOME)
    assert outcome.features["shared_metrics"] == ["clearance"]
    assert "metric:candidate_only" in outcome.missingness
    no_numeric = compatible_case_similarity(
        query, {**candidate, "metrics": {"label": "none"}, "outcome": "none"}, mode="metric"
    )
    assert "outcome/metrics" in no_numeric.missingness
    feature_unknown = compatible_case_similarity(
        query,
        {key: value for key, value in candidate.items() if key not in {"symptoms", "tags"}},
        mode=SIMILARITY_MODE_SYMPTOM,
    )
    assert feature_unknown.compatibility == UNKNOWN_COMPATIBILITY
    assert "symptom/tags" in feature_unknown.missingness

    nested_episode = _episode("nested")
    nested_ref = compatible_case_similarity(
        query,
        {"episode": nested_episode, "case_id": "outer", "scenario_id": "corridor"},
    )
    assert nested_ref.candidate_id == "outer"
    nested_mapping = compatible_case_similarity(
        query,
        {"episode_ref": {**asdict(nested_episode), "episode_id": "nested-map"}},
    )
    assert nested_mapping.candidate_id == "nested-map"
    signal = compatible_case_similarity(
        query,
        Signal(signal_id="signal-case", detector_id="detector"),
        mode=SIMILARITY_MODE_SYMPTOM,
    )
    assert signal.candidate_id == "signal-case"
    finding_case = compatible_case_similarity(query, new_finding("finding-case", "finding"))
    assert finding_case.candidate_id == "finding-case"


def test_similarity_ranking_alias_filters_unknown_and_limits() -> None:
    query = _episode("query")
    compatible = _episode("compatible", planner="orca")
    incompatible = _episode("incompatible", planner="orca", source="d")
    unknown = {"case_id": "unknown", "scenario_id": "corridor", "planner_id": "orca"}
    ranked = similar_cases(
        query,
        [incompatible, unknown, compatible],
        mode=SIMILARITY_MODE_SAME_SCENARIO,
        include_incompatible=True,
        include_unknown=True,
    )
    assert [item.candidate_id for item in ranked] == [
        compatible.episode_id,
        "unknown",
        incompatible.episode_id,
    ]
    assert (
        find_similar_cases(
            query,
            [unknown, compatible],
            mode=SIMILARITY_MODE_SAME_SCENARIO,
            include_unknown=False,
            limit=1,
        )[0].candidate_id
        == compatible.episode_id
    )
    assert (
        find_similar_cases(
            query,
            [compatible],
            mode=SIMILARITY_MODE_SAME_SCENARIO,
            limit=-1,
        )
        == []
    )
    assert filter_compatible_cases(
        query,
        [unknown, incompatible, compatible],
        mode=SIMILARITY_MODE_SAME_SCENARIO,
    ) == [compatible]
