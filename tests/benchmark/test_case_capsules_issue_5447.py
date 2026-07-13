"""Tests for the Chapter 7 case-capsule manifest builder (issue #5447).

These exercise the honest-selection contract: fail-closed on empty/wrong-schema
input, label unavailable archetypes rather than fabricate them, grade
descriptive-only unless a validated causal report is supplied, honour the
stop-rule floor, and validate the emitted manifest structurally while reporting
author-pending narrative fields.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.case_capsules import (
    AUTHOR_REQUIRED,
    CANDIDATE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    CaseCapsuleError,
    build_ch7_case_capsule_manifest,
    canonical_sha256,
    validate_ch7_case_capsule_manifest,
)
from scripts.analysis.build_ch7_case_capsules_issue_5447 import main as build_cli_main


def _seed_flip_candidate(cid: str, scenario: str, planner: str, **over: Any) -> dict[str, Any]:
    """Return a minimal seed-flip candidate record like the #5446 miner emits."""
    base = {
        "candidate_id": cid,
        "archetype": "seed_flip",
        "scenario_id": scenario,
        "planner": planner,
        "triage_only": False,
        "selected": True,
        "cross_planner_disagreement_entropy": 0.9,
        "seed_flip": {"entropy_bits": 1.0, "effective_denominator": 8},
        "upset_outcome": None,
        "reproducibility": {"n_seeds": 8, "raw_seed_outcomes": {"1": 1, "2": 0}},
    }
    base.update(over)
    return base


def _upset_candidate(cid: str, scenario: str, planner: str, **over: Any) -> dict[str, Any]:
    """Return a minimal planner-upset candidate record like the #5446 miner emits."""
    base = {
        "candidate_id": cid,
        "archetype": "planner_upset",
        "scenario_id": scenario,
        "planner": planner,
        "triage_only": False,
        "selected": True,
        "cross_planner_disagreement_entropy": 0.8,
        "seed_flip": None,
        "upset_outcome": {"underdog_planner": planner, "outcome_gap": 0.5},
        "reproducibility": {"underdog_n_seeds": 5, "favorite_n_seeds": 5},
    }
    base.update(over)
    return base


def _candidate_manifest(candidates: list[dict[str, Any]], **over: Any) -> dict[str, Any]:
    """Wrap candidates in a #5446-style candidate manifest."""
    manifest = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "issue": "#5446",
        "candidates": candidates,
        "archetype_availability": {"disagreement_recovery": {"available": True}},
    }
    manifest.update(over)
    return manifest


def test_case_capsule_builder_fails_closed_on_empty_manifest():
    """An empty / candidate-free manifest must fail closed, not emit an empty set."""
    with pytest.raises(CaseCapsuleError):
        build_ch7_case_capsule_manifest(_candidate_manifest([]))


def test_case_capsule_builder_fails_closed_on_wrong_schema():
    """A wrong-schema candidate manifest is rejected."""
    bad = _candidate_manifest([_seed_flip_candidate("c", "s", "p")], schema_version="other.v9")
    with pytest.raises(CaseCapsuleError):
        build_ch7_case_capsule_manifest(bad)


def test_case_capsule_builder_rejects_non_dict():
    """Non-dict input fails closed."""
    with pytest.raises(CaseCapsuleError):
        build_ch7_case_capsule_manifest([])  # type: ignore[arg-type]


def test_case_capsule_labels_unavailable_without_causal_or_risk_reports():
    """Causal/risk archetypes are unavailable (not fabricated) when reports absent."""
    candidates = [
        _seed_flip_candidate("sf1", "sA", "p1"),
        _upset_candidate("up1", "sB", "p2"),
        _seed_flip_candidate("sf2", "sC", "p3"),
        _upset_candidate("up2", "sD", "p4"),
    ]
    manifest = build_ch7_case_capsule_manifest(_candidate_manifest(candidates))

    by_arch = {c["archetype"]: c for c in manifest["capsules"]}
    # Causal-defined archetypes must be unavailable with a concrete reason.
    assert by_arch["paired_first_unsafe_action"]["status"] == "unavailable"
    assert by_arch["paired_first_unsafe_action"]["reason"] == "causal_report_unavailable"
    assert by_arch["ambiguous_abstention"]["reason"] == "causal_report_unavailable"
    # Risk archetype unavailable without a risk report.
    assert by_arch["near_miss_online_risk"]["reason"] == "risk_report_unavailable"
    # Data-sourced descriptive archetypes are admitted, descriptive-only.
    assert by_arch["hard_vs_easy_seed"]["status"] == "admitted"
    assert by_arch["hard_vs_easy_seed"]["evidence_grade"] == "descriptive-only"


def test_case_capsule_causal_report_upgrades_grade():
    """A validated causal report keyed by scenario upgrades the capsule to causal.

    Enough candidates are supplied that a planner-upset source remains for the
    causal ``paired_first_unsafe_action`` archetype after the two earlier
    archetypes consume theirs (candidates are never reused).
    """
    candidates = [
        _seed_flip_candidate("sf1", "sA", "p1"),
        _upset_candidate("up1", "sB", "p2"),
        _upset_candidate("up2", "sE", "p5"),
    ]
    causal = {
        "sB": {"report": "validated causal report for sB", "abstention": False},
        "sE": {"report": "validated causal report for sE", "abstention": False},
    }
    manifest = build_ch7_case_capsule_manifest(
        _candidate_manifest(candidates), causal_reports=causal
    )
    by_arch = {c["archetype"]: c for c in manifest["capsules"]}
    paired = by_arch["paired_first_unsafe_action"]
    assert paired["status"] == "admitted"
    assert paired["evidence_grade"] == "causal"
    assert paired["causal_label"]["status"] == "validated"


def test_case_capsule_insufficient_evidence_below_floor():
    """Fewer than min_capsules admitted archetypes -> insufficient_evidence status."""
    manifest = build_ch7_case_capsule_manifest(
        _candidate_manifest([_seed_flip_candidate("sf1", "sA", "p1")])
    )
    assert manifest["status"] == "insufficient_evidence"
    assert manifest["summary"]["meets_min_capsules"] is False


def test_case_capsule_triage_candidates_excluded_by_default():
    """Triage-only candidates never silently back a capsule unless explicitly allowed."""
    triage = _seed_flip_candidate("sf1", "sA", "p1", triage_only=True)
    manifest = build_ch7_case_capsule_manifest(_candidate_manifest([triage]))
    hard = next(c for c in manifest["capsules"] if c["archetype"] == "hard_vs_easy_seed")
    assert hard["status"] == "unavailable"

    allowed = build_ch7_case_capsule_manifest(_candidate_manifest([triage]), allow_triage=True)
    hard2 = next(c for c in allowed["capsules"] if c["archetype"] == "hard_vs_easy_seed")
    assert hard2["status"] == "admitted"


def test_case_capsule_candidates_not_reused_across_archetypes():
    """A single candidate must not source two different capsule archetypes."""
    candidates = [_upset_candidate("up1", "sB", "p2")]
    manifest = build_ch7_case_capsule_manifest(_candidate_manifest(candidates))
    sources = [
        c.get("source_candidate_id") for c in manifest["capsules"] if c["status"] == "admitted"
    ]
    assert len(sources) == len(set(sources))


def test_case_capsule_manifest_validates_structurally_with_author_pending():
    """A well-formed manifest is structurally valid but reports author-pending fields."""
    candidates = [
        _seed_flip_candidate("sf1", "sA", "p1"),
        _upset_candidate("up1", "sB", "p2"),
        _seed_flip_candidate("sf2", "sC", "p3"),
        _seed_flip_candidate("sf3", "sD", "p4"),
    ]
    manifest = build_ch7_case_capsule_manifest(_candidate_manifest(candidates))
    result = validate_ch7_case_capsule_manifest(manifest)
    assert result.ok, result.structural_violations
    # Narrative and figure sentinels are surfaced, not hidden.
    assert any("competing_explanation" in p for p in result.author_pending)
    assert any("marked_times" in p for p in result.author_pending)


def test_case_capsule_pair_requires_shared_axis_spec():
    """Pair capsules carry a shared-axis spec; corrupting it is a structural violation."""
    candidates = [_seed_flip_candidate("sf1", "sA", "p1")]
    manifest = build_ch7_case_capsule_manifest(_candidate_manifest(candidates), min_capsules=1)
    hard = next(c for c in manifest["capsules"] if c["archetype"] == "hard_vs_easy_seed")
    assert hard["is_pair"] is True
    assert set(hard["figure_spec"]["shared_axis_spec"]) >= {"axes_limits", "time_markers"}

    hard["figure_spec"].pop("shared_axis_spec")
    result = validate_ch7_case_capsule_manifest(manifest)
    assert not result.ok
    assert any("shared_axis_spec" in v for v in result.structural_violations)


def test_case_capsule_validator_flags_unknown_status():
    """A capsule with an unknown status is a structural violation."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "capsules": [{"capsule_id": "ch7::x", "status": "mystery"}],
    }
    result = validate_ch7_case_capsule_manifest(manifest)
    assert not result.ok
    assert any("unknown capsule status" in v for v in result.structural_violations)


def test_case_capsule_canonical_sha256_is_order_independent():
    """Input-hash provenance must be insertion-order independent."""
    a = {"schema_version": CANDIDATE_SCHEMA_VERSION, "x": 1, "y": 2}
    b = {"y": 2, "schema_version": CANDIDATE_SCHEMA_VERSION, "x": 1}
    assert canonical_sha256(a) == canonical_sha256(b)


def test_case_capsule_input_hash_recorded():
    """The emitted manifest records the candidate-manifest hash for reproducibility."""
    cm = _candidate_manifest([_seed_flip_candidate("sf1", "sA", "p1")])
    manifest = build_ch7_case_capsule_manifest(cm)
    assert manifest["inputs"]["candidate_manifest_sha256"] == canonical_sha256(cm)
    assert manifest["inputs"]["n_candidates"] == 1


def test_case_capsule_author_required_sentinel_never_prefilled():
    """The builder must not fabricate the subjective narrative fields."""
    candidates = [_upset_candidate("up1", "sB", "p2")]
    manifest = build_ch7_case_capsule_manifest(_candidate_manifest(candidates), min_capsules=1)
    admitted = next(c for c in manifest["capsules"] if c["status"] == "admitted")
    assert admitted["narrative"]["competing_explanation"] == AUTHOR_REQUIRED
    assert admitted["narrative"]["generalization_limits"] == AUTHOR_REQUIRED
    # But the mechanically-derivable rationale is filled from candidate data.
    assert admitted["narrative"]["selection_rationale"] != AUTHOR_REQUIRED
    assert "up1" in admitted["narrative"]["selection_rationale"]


def test_case_capsule_builder_rejects_non_dict_candidate_records():
    """Malformed candidate records fail closed before sort-key access."""
    with pytest.raises(CaseCapsuleError, match="non-dict candidate"):
        build_ch7_case_capsule_manifest(_candidate_manifest([None]))  # type: ignore[list-item]


def test_case_capsule_builder_normalizes_malformed_seed_flip():
    """A malformed optional seed_flip field cannot crash candidate ranking."""
    candidate = _seed_flip_candidate("sf1", "sA", "p1", seed_flip="malformed")
    manifest = build_ch7_case_capsule_manifest(_candidate_manifest([candidate]), min_capsules=1)
    admitted = next(c for c in manifest["capsules"] if c["status"] == "admitted")
    assert admitted["source_candidate_id"] == "sf1"


@pytest.mark.parametrize(
    ("parameter", "value", "expected"),
    [
        ("causal_reports", [], "causal_reports must be a dict"),
        ("risk_reports", [], "risk_reports must be a dict"),
    ],
)
def test_case_capsule_builder_rejects_invalid_report_types(parameter, value, expected):
    """Optional report bundles must be mappings when supplied."""
    with pytest.raises(CaseCapsuleError, match=expected):
        build_ch7_case_capsule_manifest(
            _candidate_manifest([_seed_flip_candidate("sf1", "sA", "p1")]),
            **{parameter: value},
        )


def test_case_capsule_builder_deep_copies_input_provenance_and_reports():
    """Mutating an emitted capsule cannot mutate caller-owned nested inputs."""
    candidate = _upset_candidate(
        "up1",
        "sB",
        "p2",
        reproducibility={"nested": {"value": 1}},
        seed_flip={"effective_denominator": 5, "nested": {"value": 2}},
    )
    causal_candidate = _upset_candidate("up2", "sC", "p3", reproducibility={"nested": {"value": 4}})
    causal = {"sC": {"nested": {"value": 3}}}
    manifest = build_ch7_case_capsule_manifest(
        _candidate_manifest([candidate, causal_candidate]), causal_reports=causal, min_capsules=1
    )
    admitted = next(
        c for c in manifest["capsules"] if c["archetype"] == "paired_first_unsafe_action"
    )
    admitted["source_provenance"]["reproducibility"]["nested"]["value"] = 10
    admitted["causal_label"]["report_ref"]["nested"]["value"] = 11
    assert causal_candidate["reproducibility"]["nested"]["value"] == 4
    assert causal["sC"]["nested"]["value"] == 3


def test_case_capsule_reports_are_scoped_to_required_archetypes():
    """Non-causal archetypes stay descriptive-only even when a report is available."""
    candidate = _seed_flip_candidate("sf1", "sA", "p1")
    manifest = build_ch7_case_capsule_manifest(
        _candidate_manifest([candidate]),
        causal_reports={"sA": {"report": "validated"}},
        min_capsules=1,
    )
    hard = next(c for c in manifest["capsules"] if c["archetype"] == "hard_vs_easy_seed")
    assert hard["causal_label"] == "descriptive-only"
    assert hard["risk_report_ref"] is None


def test_case_capsule_validator_rejects_non_dict_root_and_records():
    """Malformed manifest roots and capsule records are structural violations."""
    root_result = validate_ch7_case_capsule_manifest([])
    assert not root_result.ok
    assert any("manifest must be a dict" in v for v in root_result.structural_violations)

    record_result = validate_ch7_case_capsule_manifest(
        {"schema_version": SCHEMA_VERSION, "capsules": [None]}
    )
    assert not record_result.ok
    assert any("non-dict records" in v for v in record_result.structural_violations)


def test_case_capsule_validator_requires_all_figure_provenance_flags():
    """Figure validation protects trajectory and actor-footprint requirements too."""
    manifest = build_ch7_case_capsule_manifest(
        _candidate_manifest([_seed_flip_candidate("sf1", "sA", "p1")]), min_capsules=1
    )
    hard = next(c for c in manifest["capsules"] if c["status"] == "admitted")
    hard["figure_spec"]["trajectory_sources_required"] = False
    result = validate_ch7_case_capsule_manifest(manifest)
    assert not result.ok
    assert any("map/scale/trajectories/footprints" in v for v in result.structural_violations)


@pytest.mark.parametrize("payload", ["{", "[]"])
def test_case_capsule_cli_fails_closed_on_malformed_json(tmp_path, capsys, payload):
    """Missing or malformed JSON input uses the documented build-error exit code."""
    candidates = tmp_path / "candidates.json"
    candidates.write_text(payload, encoding="utf-8")
    assert build_cli_main(["--candidates", str(candidates)]) == 2
    assert "error:" in capsys.readouterr().err


def test_case_capsule_cli_fails_closed_on_missing_json(tmp_path, capsys):
    """A missing candidate file produces a concise fail-closed error."""
    missing = tmp_path / "missing.json"
    assert build_cli_main(["--candidates", str(missing)]) == 2
    assert "error:" in capsys.readouterr().err
