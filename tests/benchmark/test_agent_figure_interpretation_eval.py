# evidence-writer-exempt: tests write only temporary fixture mutations under pytest tmp_path
"""Tests for frozen agent-figure interpretation evaluation fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark import agent_figure_interpretation_eval as eval_mod
from robot_sf.benchmark.agent_figure_interpretation_eval import (
    CANDIDATE_SCHEMA_VERSION,
    CRITICAL_ERROR_KINDS,
    DIMENSIONS,
    AgentFigureEvalError,
    evaluate_manifest,
    evaluate_packet,
    list_fixture_mutations,
    load_verified_packets,
    replay_all_fixture_mutations,
    replay_fixture_mutation,
    validate_candidate_envelope,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "result_interpretation_packet" / "v1"
)
MANIFEST = FIXTURE_DIR.parent / "agent_figure_interpretation_eval_manifest.json"


def _case_by_id(result: dict[str, object]) -> dict[str, dict[str, object]]:
    cases = result["cases"]
    assert isinstance(cases, list)
    return {str(case["packet_id"]): case for case in cases}


def _copy_fixture_tree(tmp_path: Path) -> Path:
    root = tmp_path / "result_interpretation_packet"
    fixture_dir = root / "v1"
    fixture_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_DIR / "ch7_visualization_causal_abstention.json", fixture_dir)
    shutil.copy(MANIFEST, root)
    return root


def _clean_packet() -> dict[str, object]:
    return next(
        packet for _, packet in load_verified_packets(MANIFEST) if packet["packet_id"] == "clean"
    )


def _candidate(packet_id: str, mutation_id: str | None = None) -> dict[str, object]:
    packets = {packet["packet_id"]: packet for _, packet in load_verified_packets(MANIFEST)}
    packet = packets[packet_id]
    mutation_id = mutation_id or packet_id
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifact = manifest["packet"]
    figure = {
        "spec": {"fixture_id": packet["source"]["source_id"], "mutation_id": mutation_id},
        "caption": "Diagnostic fixture candidate; semantic review is unavailable.",
    }
    interpretation = packet["interpretation"]
    candidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "artifact_kind": "candidate_interpretation",
        "provider": "none",
        "fixture_id": packet["source"]["source_id"],
        "mutation_id": mutation_id,
        "workflow": {"id": "fixture-test-candidate", "revision": "test-revision-1"},
        "figure": figure,
        "limitations": ["frozen fixture only"],
        "confidence": {"status": "not_available", "value": None},
        "unresolved_questions": ["independent semantic review"],
        "claim_boundary": "fixture replay only; not benchmark evidence",
        "interpretation": interpretation,
        "mutation": {
            "id": mutation_id,
            "expected_detectors": [] if mutation_id == "clean" else [mutation_id],
        },
        "findings": {
            dimension: {
                "status": "not_available",
                "critical": dimension == eval_mod.CRITICAL_ERROR_DIMENSIONS.get(mutation_id),
            }
            for dimension in DIMENSIONS
        },
        "unavailable": ["independent semantic review"],
        "not_applicable": ["external provider execution"],
        "provenance": {
            "manifest_schema_version": "agent_figure_interpretation_eval_manifest.v1",
            "source_sha256": artifact["source_sha256"],
            "packet_sha256": artifact["sha256"],
            "reference_sha256": artifact["reference_sha256"],
            "candidate_sha256": "0" * 64,
            "figure_sha256": {
                "status": "available",
                "sha256": eval_mod._canonical_digest(figure["spec"]),
            },
            "caption_sha256": {
                "status": "available",
                "sha256": hashlib.sha256(figure["caption"].encode("utf-8")).hexdigest(),
            },
            "review_sha256": {"status": "not_available", "sha256": None},
        },
        "replay_provenance": {
            "mode": "fixture",
            "deterministic": True,
            "external_provider_called": False,
            "network_access": "none",
        },
        "verdict": "pending",
    }
    candidate["provenance"]["candidate_sha256"] = eval_mod._candidate_envelope_digest(candidate)
    return candidate


def _review_scores(value: float = 1.0) -> dict[str, float]:
    return dict.fromkeys(DIMENSIONS, value)


def _result_schema() -> dict[str, object]:
    return json.loads(
        Path("robot_sf/benchmark/schemas/agent_figure_interpretation_eval.v1.json").read_text(
            encoding="utf-8"
        )
    )


def _candidate_schema() -> dict[str, object]:
    return json.loads(
        Path("robot_sf/benchmark/schemas/agent_figure_interpretation_candidate.v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_clean_output_has_all_dimension_scores_and_no_critical_errors() -> None:
    result = evaluate_manifest(MANIFEST)
    cases = _case_by_id(result)
    clean = cases["clean"]

    assert result["status"] == "evaluation_artifacts_only"
    assert "no external model calls" in result["claim_boundary"]
    assert clean["status"] == "clean"
    assert clean["aggregate_score"] == 1.0
    assert [score["dimension"] for score in clean["scores"]] == list(DIMENSIONS)
    assert all(score["passed"] for score in clean["scores"])
    assert not any(clean["critical_errors"].values())


def test_manifest_summary_preserves_dimension_and_failure_evidence() -> None:
    summary = evaluate_manifest(MANIFEST)["aggregate_summary"]

    assert summary["case_status_counts"] == {"clean": 1, "failed": 11}
    assert summary["dimension_scores"]["source_denominator"] == {
        "case_count": 12,
        "passed_count": 11,
        "failed_count": 1,
        "pass_rate": pytest.approx(11 / 12),
    }
    assert summary["critical_failure_examples"]["causal_overclaim"] == ["causal_overclaim"]
    assert summary["critical_failure_examples"]["null_overclaim"] == ["null_overclaim"]
    assert summary["reviewer_accounting"]["status"] == "not_available"
    assert summary["workflow_variants"]["status"] == "not_available"


def test_manifest_summary_reports_unavailable_workflow_variants() -> None:
    variants = evaluate_manifest(MANIFEST)["aggregate_summary"]["workflow_variants"]

    assert variants["status"] == "not_available"
    assert variants["paired_case_count"] == 0


@pytest.mark.parametrize(
    ("packet_id", "critical_kind"),
    [
        ("unavailable_to_zero", "unavailable_to_zero"),
        ("denominator_loss", "denominator_loss"),
        ("wrong_pairing_resampling", "wrong_pairing_resampling"),
        ("fallback_degraded_promotion", "fallback_degraded_promotion"),
        ("causal_overclaim", "causal_overclaim"),
        ("unsupported_ranking", "unsupported_ranking"),
        ("null_overclaim", "null_overclaim"),
    ],
)
def test_each_critical_mutation_is_flagged(packet_id: str, critical_kind: str) -> None:
    cases = _case_by_id(evaluate_manifest(MANIFEST))
    case = cases[packet_id]

    assert case["status"] == "failed"
    assert case["aggregate_score"] < 1.0
    assert case["critical_errors"][critical_kind] is True
    for other_kind in set(CRITICAL_ERROR_KINDS) - {critical_kind}:
        assert case["critical_errors"][other_kind] is False


def test_replay_inventory_lists_source_fixtures_and_required_detectors() -> None:
    inventory = list_fixture_mutations(MANIFEST)
    assert inventory["status"] == "evaluation_artifacts_only"
    assert inventory["claim_boundary"] == eval_mod.EXPECTED_REPORT_CLAIM_BOUNDARY
    assert {record["mutation_id"] for record in inventory["mutations"]} == {
        "clean",
        *CRITICAL_ERROR_KINDS,
    }
    mutation_by_id = {record["mutation_id"]: record for record in inventory["mutations"]}
    assert mutation_by_id["clean"]["expected_detectors"] == []
    for mutation_id in CRITICAL_ERROR_KINDS:
        assert mutation_by_id[mutation_id]["expected_detectors"] == [mutation_id]
    assert {item["mutation_id"] for item in inventory["integrity_mutations"]} == {
        "digest_omission",
        "stale_post_review_bytes",
    }
    assert {fixture["fixture_id"] for fixture in inventory["fixtures"]} == {
        "ch7_visualization_causal_abstention_fixture"
    }


def test_candidate_envelope_is_provider_free_and_exact() -> None:
    candidate = _candidate("clean")
    validate_candidate_envelope(candidate)
    jsonschema.Draft202012Validator.check_schema(_candidate_schema())
    jsonschema.validate(candidate, _candidate_schema())

    semantic_review = copy.deepcopy(candidate)
    semantic_review["findings"]["caption_accuracy"]["status"] = "requires_semantic_review"
    validate_candidate_envelope(semantic_review)
    jsonschema.validate(semantic_review, _candidate_schema())

    invalid_provider = dict(candidate, provider="local-model")
    with pytest.raises(AgentFigureEvalError, match="provider must be 'none'"):
        validate_candidate_envelope(invalid_provider)

    invalid_boundary = dict(candidate, claim_boundary="benchmark evidence")
    with pytest.raises(AgentFigureEvalError, match="claim_boundary"):
        validate_candidate_envelope(invalid_boundary)

    with_reference = dict(candidate, reference=candidate["interpretation"])
    with pytest.raises(AgentFigureEvalError, match="unexpected reference"):
        validate_candidate_envelope(with_reference)


@pytest.mark.parametrize(
    ("field", "value"),
    [("evidence_tier", ["diagnostic"]), ("row_provenance", 1), ("rows_disclosed", "yes")],
)
def test_candidate_detector_fields_fail_closed_before_scoring(field: str, value: object) -> None:
    candidate = _candidate("clean")
    candidate["interpretation"]["evidence_tier_availability"][field] = value  # type: ignore[index]

    with pytest.raises(AgentFigureEvalError, match="candidate evidence_tier_availability"):
        validate_candidate_envelope(candidate)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_candidate_non_finite_nested_values_fail_closed(non_finite: float) -> None:
    candidate = _candidate("clean")
    candidate["figure"]["spec"]["score"] = non_finite  # type: ignore[index]

    with pytest.raises(AgentFigureEvalError, match="finite JSON numbers"):
        validate_candidate_envelope(candidate)


def test_manifest_non_finite_json_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected_metadata"] = {"score": float("nan")}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")

    with pytest.raises(AgentFigureEvalError, match="unreadable JSON"):
        load_verified_packets(manifest_path)


def test_replay_digest_omission_and_stale_bytes_fail_closed() -> None:
    omitted = _candidate("clean")
    del omitted["provenance"]["source_sha256"]  # type: ignore[index]
    with pytest.raises(AgentFigureEvalError, match="complete digest contract"):
        validate_candidate_envelope(omitted)

    for digest_name in ("figure_sha256", "caption_sha256", "review_sha256"):
        omitted = _candidate("clean")
        del omitted["provenance"][digest_name]  # type: ignore[index]
        with pytest.raises(AgentFigureEvalError, match="complete digest contract"):
            validate_candidate_envelope(omitted)

    stale = _candidate("clean")
    stale["provenance"]["packet_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(AgentFigureEvalError, match="does not match verified bytes"):
        replay_fixture_mutation(MANIFEST, stale)

    reviewed = _candidate("clean")
    reviewed["provenance"]["review_sha256"] = {  # type: ignore[index]
        "status": "available",
        "sha256": eval_mod._canonical_digest(eval_mod._review_digest_payload(reviewed)),
    }
    stale_review = copy.deepcopy(reviewed)
    stale_review["findings"]["caption_accuracy"]["critical"] = True
    stale_review["provenance"]["candidate_sha256"] = eval_mod._candidate_envelope_digest(
        stale_review
    )
    with pytest.raises(AgentFigureEvalError, match="post-review bytes"):
        replay_fixture_mutation(MANIFEST, stale_review)


@pytest.mark.parametrize("packet_id", ["clean", *CRITICAL_ERROR_KINDS])
def test_replay_one_pair_requires_the_named_detector(packet_id: str) -> None:
    report = replay_fixture_mutation(MANIFEST, _candidate(packet_id))

    expected = [] if packet_id == "clean" else [packet_id]
    assert report["mode"] == "single"
    assert report["expected_detectors"] == expected
    assert report["detected_detectors"] == expected
    assert report["detector_status"] == "pass"


def test_replay_rejects_candidate_critical_flags_that_disagree_with_detectors() -> None:
    candidate = _candidate("causal_overclaim")
    candidate["findings"]["claim_boundary"]["critical"] = False  # type: ignore[index]
    candidate["provenance"]["candidate_sha256"] = eval_mod._candidate_envelope_digest(candidate)

    with pytest.raises(AgentFigureEvalError, match="must match deterministic detector output"):
        replay_fixture_mutation(MANIFEST, candidate)


def test_replay_all_requires_exact_corpus_coverage_and_is_deterministic() -> None:
    packet_ids = ["clean", *CRITICAL_ERROR_KINDS]
    candidates = [_candidate(packet_id) for packet_id in reversed(packet_ids)]
    first = replay_all_fixture_mutations(MANIFEST, candidates)
    second = replay_all_fixture_mutations(MANIFEST, candidates)

    assert first == second
    assert first["mode"] == "all"
    assert first["case_count"] == 12
    assert first["passed_case_count"] == 12
    assert first["failed_case_count"] == 0
    assert first["detector_status"] == "pass"
    assert set(first["provenance"]) == {"code_sha256", "config_sha256", "fixture_sha256"}
    assert all(len(digest) == 64 for digest in first["provenance"].values())

    with pytest.raises(AgentFigureEvalError, match="coverage mismatch"):
        replay_all_fixture_mutations(MANIFEST, candidates[:-1])


@pytest.mark.parametrize(
    "evidence_tier",
    [
        "smoke",
        "smoke evidence",
        "benchmark",
        "nominal benchmark evidence",
        "paper_facing",
        "paper-grade",
        "paper-grade evidence",
    ],
)
def test_fallback_degraded_promotion_recognizes_supported_evidence_tiers(
    evidence_tier: str,
) -> None:
    packet = _clean_packet()
    packet["reference"]["evidence_tier_availability"]["execution_mode"] = "fallback"  # type: ignore[index]
    packet["interpretation"]["evidence_tier_availability"]["evidence_tier"] = evidence_tier  # type: ignore[index]

    result = evaluate_packet(packet).to_dict()

    assert result["critical_errors"]["fallback_degraded_promotion"] is True


@pytest.mark.parametrize("row_provenance", [["fallback"], ["degraded"]])
def test_fallback_degraded_row_provenance_cannot_be_promoted(
    row_provenance: list[str],
) -> None:
    packet = _clean_packet()
    evidence = packet["interpretation"]["evidence_tier_availability"]  # type: ignore[index]
    evidence["execution_mode"] = "native"
    evidence["row_provenance"] = row_provenance
    evidence["evidence_tier"] = "nominal benchmark evidence"

    result = evaluate_packet(packet).to_dict()

    assert result["critical_errors"]["fallback_degraded_promotion"] is True


def test_stale_bytes_digest_drift_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    packet_path = root / "v1" / "ch7_visualization_causal_abstention.json"
    packet_path.write_text(packet_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="sha256 mismatch"):
        load_verified_packets(root / "agent_figure_interpretation_eval_manifest.json")


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    """A missing manifest is reported as an evaluator error, not a traceback."""
    with pytest.raises(AgentFigureEvalError, match="unreadable JSON"):
        load_verified_packets(tmp_path / "missing.json")


def test_changed_fixture_bytes_fail_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    packet_path = root / "v1" / "ch7_visualization_causal_abstention.json"
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    payload["packet_id"] = "changed-clean"
    packet_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="sha256 mismatch"):
        load_verified_packets(root / "agent_figure_interpretation_eval_manifest.json")


def test_source_binding_digest_drift_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packet"]["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="source binding digest"):
        load_verified_packets(manifest_path)


def test_reference_digest_cannot_create_a_second_packet(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packet"]["reference_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="reference_sha256"):
        load_verified_packets(manifest_path)


def test_missing_packet_schema_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("expected_packet_schema")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="expected_packet_schema"):
        load_verified_packets(manifest_path)


@pytest.mark.parametrize("path", ["../clean.json", "/tmp/clean.json"])
def test_manifest_path_escape_fails_closed(tmp_path: Path, path: str) -> None:
    """Manifest entries cannot escape the canonical fixture root."""
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packet"]["path"] = path
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="repository-relative"):
        load_verified_packets(manifest_path)


def test_manifest_symlink_escape_fails_closed(tmp_path: Path) -> None:
    """Manifest entries cannot follow a symlink outside the fixture root."""
    root = _copy_fixture_tree(tmp_path)
    packet_path = root / "v1" / "ch7_visualization_causal_abstention.json"
    packet_path.unlink()
    packet_path.symlink_to(FIXTURE_DIR / "ch7_visualization_causal_abstention.json")

    with pytest.raises(AgentFigureEvalError, match="must not traverse a symlink"):
        load_verified_packets(root / "agent_figure_interpretation_eval_manifest.json")


def test_manifest_packet_identity_fails_closed(tmp_path: Path) -> None:
    """The manifest cannot redirect the evaluator to another canonical packet."""
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packet"]["id"] = "renamed-clean"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="canonical fixture ownership"):
        load_verified_packets(manifest_path)


def test_manifest_status_drift_fails_closed(tmp_path: Path) -> None:
    """A manifest cannot relabel fixture replay as a stronger evidence tier."""
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "benchmark_evidence"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="manifest status"):
        load_verified_packets(manifest_path)


def test_manifest_claim_boundary_drift_fails_closed(tmp_path: Path) -> None:
    """A manifest cannot relabel fixture replay through its claim boundary."""
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claim_boundary"] = "nominal benchmark evidence"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="claim_boundary"):
        load_verified_packets(manifest_path)


def test_packet_claim_boundary_drift_fails_closed() -> None:
    """A packet cannot relabel its evaluation artifact as benchmark evidence."""
    packet = _clean_packet()
    packet["claim_boundary"] = "nominal benchmark evidence"

    with pytest.raises(AgentFigureEvalError, match="claim_boundary"):
        evaluate_packet(packet)


def test_canonical_packet_identity_drift_fails_closed(tmp_path: Path) -> None:
    """A digest-valid packet must still belong to the canonical manifest identity."""
    root = _copy_fixture_tree(tmp_path)
    packet_path = root / "v1" / "ch7_visualization_causal_abstention.json"
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    payload["packet_id"] = "different_packet"
    packet_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packet"]["sha256"] = eval_mod.sha256_file(packet_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="canonical packet_id must match"):
        load_verified_packets(manifest_path)


def test_duplicate_manifest_mutations_fail_closed(tmp_path: Path) -> None:
    """A manifest cannot replay one mutation identity twice."""
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mutations"].append("clean")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="must not contain duplicates"):
        load_verified_packets(manifest_path)


@pytest.mark.parametrize("section", ["reference", "interpretation"])
def test_missing_scoring_dimension_fails_closed(section: str) -> None:
    """Missing dimensions cannot compare as equal null values."""
    packet = _clean_packet()
    del packet[section]["source_denominator"]  # type: ignore[index]

    with pytest.raises(AgentFigureEvalError, match="source_denominator"):
        evaluate_packet(packet)


def test_packet_schema_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    manifest_path = root / "agent_figure_interpretation_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_path = root / "v1" / "ch7_visualization_causal_abstention.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.pop("schema_version")
    packet_path.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
    manifest["packet"]["sha256"] = eval_mod.sha256_file(packet_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="canonical result packet validation failed"):
        load_verified_packets(manifest_path)


def test_variant_comparison_is_packet_aware_and_preserves_fidelity() -> None:
    packet = _clean_packet()
    reference = copy.deepcopy(packet["reference"])
    assert isinstance(reference, dict)
    baseline = copy.deepcopy(reference)
    baseline["claim_boundary"]["causal_claim_allowed"] = True
    constrained = copy.deepcopy(reference)
    packet["interpretation_variants"] = {
        "baseline": baseline,
        "packet_constrained": constrained,
    }

    result = evaluate_packet(packet).to_dict()
    comparison = result["interpretation_variant_comparison"]

    assert comparison["delta"]["critical_error_count"] == -1
    assert comparison["delta"]["packet_constrained_reduces_critical_errors"] is True
    assert comparison["delta"]["packet_constrained_preserves_source_fidelity"] is True


def test_variant_comparison_does_not_hide_constrained_source_loss() -> None:
    packet = _clean_packet()
    reference = copy.deepcopy(packet["reference"])
    assert isinstance(reference, dict)
    baseline = copy.deepcopy(reference)
    constrained = copy.deepcopy(reference)
    baseline["source_denominator"]["denominator_n"] = 23
    constrained["source_denominator"]["denominator_n"] = 23
    packet["interpretation_variants"] = {
        "baseline": baseline,
        "packet_constrained": constrained,
    }

    comparison = evaluate_packet(packet).to_dict()["interpretation_variant_comparison"]

    assert comparison["delta"]["packet_constrained_preserves_source_fidelity"] is False


def test_variant_metadata_fails_closed_when_workflows_are_incomplete() -> None:
    packet = _clean_packet()
    packet["interpretation_variants"] = {"baseline": packet["reference"]}

    with pytest.raises(AgentFigureEvalError, match="exactly baseline and packet_constrained"):
        evaluate_packet(packet)


def test_analysis_unit_mutation_is_a_critical_detector() -> None:
    packet = _clean_packet()
    packet["interpretation"]["estimand_unit"]["analysis_unit"] = "unpaired_episode"  # type: ignore[index]

    result = evaluate_packet(packet).to_dict()

    assert result["critical_errors"]["analysis_unit_mismatch"] is True
    assert sum(result["critical_errors"].values()) == 1


def test_blinded_reviewer_disagreement_and_adjudication_are_accounted_for() -> None:
    packet = _clean_packet()
    packet["reference_metadata"] = {
        "reviewed": True,
        "blinded": True,
        "reviewers": [
            {"reviewer_id": "reviewer-a", "scores": _review_scores()},
            {
                "reviewer_id": "reviewer-b",
                "scores": {**_review_scores(), "claim_boundary": 0.0},
            },
        ],
        "adjudication": {
            "adjudicator_id": "adjudicator-a",
            "resolved_scores": {"claim_boundary": 1.0},
        },
    }

    accounting = evaluate_packet(packet).to_dict()["reviewer_accounting"]

    assert accounting["reviewer_count"] == 2
    assert accounting["agreement_rate"] == pytest.approx(7 / 8)
    assert accounting["disagreement_count"] == 1
    assert accounting["adjudicated_dimensions"] == ["claim_boundary"]
    assert accounting["adjudication_complete"] is True


def test_reviewer_disagreement_requires_exact_adjudication() -> None:
    packet = _clean_packet()
    packet["reference_metadata"] = {
        "reviewed": True,
        "blinded": True,
        "reviewers": [
            {"reviewer_id": "reviewer-a", "scores": _review_scores()},
            {
                "reviewer_id": "reviewer-b",
                "scores": {**_review_scores(), "claim_boundary": 0.0},
            },
        ],
        "adjudication": {
            "adjudicator_id": "adjudicator-a",
            "resolved_scores": {
                "claim_boundary": 1.0,
                "visual_semantics": 1.0,
            },
        },
    }

    with pytest.raises(AgentFigureEvalError, match="exactly cover disagreements"):
        evaluate_packet(packet)


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_reviewer_scores_fail_closed(invalid_score: float) -> None:
    packet = _clean_packet()
    packet["reference_metadata"] = {
        "reviewed": True,
        "blinded": True,
        "reviewers": [
            {
                "reviewer_id": "reviewer-a",
                "scores": {**_review_scores(), "claim_boundary": invalid_score},
            },
            {"reviewer_id": "reviewer-b", "scores": _review_scores()},
        ],
    }

    with pytest.raises(AgentFigureEvalError, match=r"number in \[0, 1\]"):
        evaluate_packet(packet)


def test_correction_priority_ranking_is_deterministic_and_critical_first() -> None:
    packet = _clean_packet()
    observed = copy.deepcopy(packet["reference"])
    assert isinstance(observed, dict)
    observed["source_denominator"]["denominator_n"] = 23
    observed["claim_boundary"]["causal_claim_allowed"] = True
    packet["interpretation"] = observed
    packet["correction_candidates"] = [
        {"id": "caption", "dimension": "caption_accuracy", "severity": "major"},
        {"id": "claim", "dimension": "claim_boundary", "severity": "critical"},
        {"id": "denominator", "dimension": "source_denominator", "severity": "critical"},
    ]

    ranking = evaluate_packet(packet).to_dict()["correction_priority_ranking"]

    assert [item["id"] for item in ranking] == ["denominator", "claim", "caption"]
    assert ranking[0]["triggered_by_critical_error"] is True
    assert ranking[-1]["dimension_failed"] is False


def test_extended_evaluation_output_matches_schema() -> None:
    packet = _clean_packet()
    reference = copy.deepcopy(packet["reference"])
    assert isinstance(reference, dict)
    packet["interpretation_variants"] = {
        "baseline": copy.deepcopy(reference),
        "packet_constrained": copy.deepcopy(reference),
    }
    packet["reference_metadata"] = {
        "reviewed": True,
        "blinded": True,
        "reviewers": [
            {"reviewer_id": "reviewer-a", "scores": _review_scores()},
            {"reviewer_id": "reviewer-b", "scores": _review_scores()},
        ],
    }
    packet["correction_candidates"] = [
        {"id": "source", "dimension": "source_denominator", "severity": "critical"}
    ]
    case = evaluate_packet(packet).to_dict()
    report = {
        "schema_version": eval_mod.EVAL_SCHEMA_VERSION,
        "status": "evaluation_artifacts_only",
        "claim_boundary": "fixture replay only; no benchmark claims",
        "case_count": 1,
        "critical_error_counts": dict.fromkeys(CRITICAL_ERROR_KINDS, 0),
        "aggregate_summary": eval_mod._aggregate_summary(
            [case], dict.fromkeys(CRITICAL_ERROR_KINDS, 0)
        ),
        "cases": [case],
    }
    schema = _result_schema()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(report, schema)


def test_manifest_report_with_aggregate_summary_matches_schema() -> None:
    schema = _result_schema()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(evaluate_manifest(MANIFEST), schema)


def test_schema_requires_aggregate_summary() -> None:
    report = evaluate_manifest(MANIFEST)
    report.pop("aggregate_summary")

    jsonschema.Draft202012Validator.check_schema(_result_schema())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(report, _result_schema())


@pytest.mark.parametrize(
    ("duplicate_dimension", "missing_dimension"),
    [
        ("source_denominator", "claim_boundary"),
        ("evidence_tier_availability", "stats_multiplicity"),
    ],
)
def test_schema_rejects_duplicate_or_missing_score_dimensions(
    duplicate_dimension: str,
    missing_dimension: str,
) -> None:
    report = evaluate_manifest(MANIFEST)
    scores = report["cases"][0]["scores"]  # type: ignore[index]
    duplicate_score = next(score for score in scores if score["dimension"] == duplicate_dimension)
    for index, score in enumerate(scores):
        if score["dimension"] == missing_dimension:
            scores[index] = copy.deepcopy(duplicate_score)
            break

    jsonschema.Draft202012Validator.check_schema(_result_schema())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(report, _result_schema())


def test_cli_help_and_fixture_only_replay() -> None:
    script = Path("scripts/analysis/run_agent_figure_interpretation_eval.py")
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--manifest" in help_run.stdout

    replay = subprocess.run(
        [sys.executable, str(script), "--manifest", str(MANIFEST)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(replay.stdout)
    assert result["status"] == "evaluation_artifacts_only"
    assert result["case_count"] == 12
    assert result["aggregate_summary"]["workflow_variants"]["status"] == "not_available"


def test_cli_lists_and_replays_candidate_envelopes(tmp_path: Path) -> None:
    script = Path("scripts/analysis/run_agent_figure_interpretation_eval.py")
    listed = subprocess.run(
        [sys.executable, str(script), "--manifest", str(MANIFEST), "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = json.loads(listed.stdout)
    assert inventory["mutations"][-1]["mutation_id"] == "wrong_pairing_resampling"

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(_candidate("causal_overclaim")), encoding="utf-8")
    one = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(MANIFEST),
            "--candidate",
            str(candidate_path),
            "--fixture-id",
            "ch7_visualization_causal_abstention_fixture",
            "--mutation-id",
            "causal_overclaim",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(one.stdout)["detector_status"] == "pass"

    validated = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(MANIFEST),
            "--candidate",
            str(candidate_path),
            "--validate",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(validated.stdout)["verdict"] == "valid"

    all_candidates_path = tmp_path / "candidates.json"
    all_candidates_path.write_text(
        json.dumps([_candidate(packet_id) for packet_id in ["clean", *CRITICAL_ERROR_KINDS]]),
        encoding="utf-8",
    )
    all_replay = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(MANIFEST),
            "--candidate",
            str(all_candidates_path),
            "--replay-all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(all_replay.stdout)["detector_status"] == "pass"


def test_no_external_provider_path() -> None:
    source = Path(eval_mod.__file__).read_text(encoding="utf-8")
    cli_source = Path("scripts/analysis/run_agent_figure_interpretation_eval.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("openai", "anthropic", "api_key", "requests", "httpx", "--provider")
    haystack = f"{source}\n{cli_source}".lower()

    assert not any(term in haystack for term in forbidden)
