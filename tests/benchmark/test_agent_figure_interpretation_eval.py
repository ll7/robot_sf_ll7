# evidence-writer-exempt: tests write only temporary fixture mutations under pytest tmp_path
"""Tests for frozen agent-figure interpretation evaluation fixtures."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark import agent_figure_interpretation_eval as eval_mod
from robot_sf.benchmark.agent_figure_interpretation_eval import (
    CRITICAL_ERROR_KINDS,
    DIMENSIONS,
    EXPECTED_PACKET_SCHEMA,
    AgentFigureEvalError,
    evaluate_manifest,
    evaluate_packet,
    load_verified_packets,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "agent_figure_interpretation_eval" / "v1"
)
MANIFEST = FIXTURE_DIR / "manifest.json"


def _case_by_id(result: dict[str, object]) -> dict[str, dict[str, object]]:
    cases = result["cases"]
    assert isinstance(cases, list)
    return {str(case["packet_id"]): case for case in cases}


def _copy_fixture_tree(tmp_path: Path) -> Path:
    root = tmp_path / "v1"
    shutil.copytree(FIXTURE_DIR, root)
    return root


def _clean_packet() -> dict[str, object]:
    return json.loads((FIXTURE_DIR / "clean.json").read_text(encoding="utf-8"))


def _review_scores(value: float = 1.0) -> dict[str, float]:
    return dict.fromkeys(DIMENSIONS, value)


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


def test_stale_bytes_digest_drift_fails_closed() -> None:
    with pytest.raises(AgentFigureEvalError, match="sha256 mismatch"):
        load_verified_packets(FIXTURE_DIR / "stale_bytes_manifest.json")


def test_changed_fixture_bytes_fail_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    clean = root / "clean.json"
    payload = json.loads(clean.read_text(encoding="utf-8"))
    payload["packet_id"] = "changed-clean"
    clean.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="sha256 mismatch"):
        load_verified_packets(root / "manifest.json")


@pytest.mark.parametrize("sha_key", ["source_sha256", "reference_sha256"])
def test_source_and_reference_digest_drift_fail_closed(tmp_path: Path, sha_key: str) -> None:
    root = _copy_fixture_tree(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"][0][sha_key] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="sha256 mismatch"):
        load_verified_packets(root / "manifest.json")


def test_missing_packet_schema_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("expected_packet_schema")
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="expected_packet_schema"):
        load_verified_packets(root / "manifest.json")


def test_packet_schema_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture_tree(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["expected_packet_schema"] = EXPECTED_PACKET_SCHEMA
    packet = json.loads((root / "clean.json").read_text(encoding="utf-8"))
    packet.pop("schema_version")
    clean_path = root / "clean.json"
    clean_path.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
    manifest["artifacts"][0]["sha256"] = eval_mod.sha256_file(clean_path)
    manifest["artifacts"] = [manifest["artifacts"][0]]
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(AgentFigureEvalError, match="packet schema_version"):
        load_verified_packets(root / "manifest.json")


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
        "cases": [case],
    }
    schema = json.loads(
        (Path("robot_sf/benchmark/schemas/agent_figure_interpretation_eval.v1.json")).read_text(
            encoding="utf-8"
        )
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(report, schema)


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
    assert result["case_count"] == 8


def test_no_external_provider_path() -> None:
    source = Path(eval_mod.__file__).read_text(encoding="utf-8")
    cli_source = Path("scripts/analysis/run_agent_figure_interpretation_eval.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("openai", "anthropic", "api_key", "requests", "httpx", "--provider")
    haystack = f"{source}\n{cli_source}".lower()

    assert not any(term in haystack for term in forbidden)
