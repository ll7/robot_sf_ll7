"""Tests for the result interpretation packet contract (issue #7029)."""

# Test grouping classes are pytest organization surfaces, not public APIs.
# ruff: noqa: D101

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from robot_sf.benchmark.result_interpretation_packet import (
    SCHEMA_VERSION,
    build_and_validate_packet,
    compute_packet_digest,
    compute_post_review_digest,
    load_result_interpretation_packet,
    load_schema,
    validate_packet,
    validate_schema_is_valid,
    write_deterministic_json,
)

FIXTURES_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "result_interpretation_packet"
    / "v1"
)

_VALID_6474 = json.loads(
    (FIXTURES_DIR / "issue_6474_comfort_exposure_supported.json").read_text(encoding="utf-8")
)
_VALID_6944 = json.loads(
    (FIXTURES_DIR / "issue_6944_brne_candidate_transition_diagnostic.json").read_text(
        encoding="utf-8"
    )
)
_VALID_CH7 = json.loads(
    (FIXTURES_DIR / "ch7_visualization_causal_abstention.json").read_text(encoding="utf-8")
)


def _available_ch7_payload() -> dict:
    """Return the Chapter 7 packet with the committed catalog figure substituted."""
    payload = copy.deepcopy(_VALID_CH7)
    figure = payload["figure_links"][0]
    figure_path = Path("tests/fixtures/artifact_catalog/v1/fig_benchmark_outcome_matrix.svg")
    caption_path = Path("tests/fixtures/artifact_catalog/v1/captions.md")
    catalog_path = Path("tests/fixtures/artifact_catalog/v1/valid_catalog.yaml")
    figure.update(
        {
            "artifact_id": "fig_benchmark_outcome_matrix",
            "path": str(figure_path),
            "sha256": hashlib.sha256(figure_path.read_bytes()).hexdigest(),
            "encoding": "svg",
            "caption_file": {
                "path": str(caption_path),
                "sha256": hashlib.sha256(caption_path.read_bytes()).hexdigest(),
            },
            "artifact_catalog": {
                "catalog_id": "fixture_camera_ready_artifacts",
                "path": str(catalog_path),
                "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                "commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    check=True,
                    text=True,
                ).stdout.strip(),
            },
        }
    )
    assertion = payload["caption_assertions"][0]
    assertion["template_id"] = "observed_visualization.v1"
    assertion["assertion_text"] = (
        "Observed figure 'ch7_matched_start_process_contrast' for estimand "
        "'ch7_matched_planner_process_contrast' with direction 'not_applicable'."
    )
    assertion["status"] = "observed"
    return payload


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_is_valid_draft_2020_12(self) -> None:
        Draft202012Validator.check_schema(load_schema())

    def test_validate_schema_is_valid_runs(self) -> None:
        validate_schema_is_valid()

    def test_schema_version_is_const(self) -> None:
        schema = load_schema()
        assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixture validation (positive)
# ---------------------------------------------------------------------------


class TestFixturesValid:
    @pytest.mark.parametrize(
        ("fixture_name", "packet_dict"),
        [
            ("issue_6474", _VALID_6474),
            ("issue_6944", _VALID_6944),
            ("ch7_visualization", _VALID_CH7),
        ],
    )
    def test_fixture_passes_schema_validation(self, fixture_name: str, packet_dict: dict) -> None:
        errors = validate_packet(packet_dict)
        assert errors == [], f"{fixture_name}: {errors}"

    @pytest.mark.parametrize(
        ("fixture_name", "packet_dict"),
        [
            ("issue_6474", _VALID_6474),
            ("issue_6944", _VALID_6944),
            ("ch7_visualization", _VALID_CH7),
        ],
    )
    def test_fixture_loads_as_typed_packet(self, fixture_name: str, packet_dict: dict) -> None:
        packet = build_and_validate_packet(packet_dict)
        assert packet.schema_version == SCHEMA_VERSION
        assert packet.packet_id
        assert packet.evidence.evidence_id

    def test_fixture_execution_counts_reconcile_to_included_population(self) -> None:
        assert (
            sum(_VALID_6474["execution_mode"]["counts"].values())
            == _VALID_6474["population"]["included"]
        )
        assert (
            sum(_VALID_6944["execution_mode"]["counts"].values())
            == _VALID_6944["population"]["included"]
        )

    def test_source_refs_record_generation_provenance(self) -> None:
        for source in _VALID_6474["sources"] + _VALID_6944["sources"] + _VALID_CH7["sources"]:
            assert source["commit"]
            assert source["tracked_commit"]
            assert source["command"]

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "issue_6474_comfort_exposure_supported.json",
            "issue_6944_brne_candidate_transition_diagnostic.json",
            "ch7_visualization_causal_abstention.json",
        ],
    )
    def test_fixture_loads_from_disk(self, fixture_name: str) -> None:
        packet = load_result_interpretation_packet(FIXTURES_DIR / fixture_name)
        assert packet.schema_version == SCHEMA_VERSION

    def test_all_controlled_decision_outcomes_present(self) -> None:
        outcomes_seen: set[str] = set()
        for fixture in [_VALID_6474, _VALID_6944, _VALID_CH7]:
            for d in fixture["decisions"]:
                outcomes_seen.add(d["outcome"])
        assert outcomes_seen == {"supported", "not_supported", "unavailable"}

    def test_deterministic_json_roundtrip(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        write_deterministic_json(_VALID_6474, out)
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        assert reloaded == _VALID_6474

    def test_computed_digest_is_deterministic(self) -> None:
        packet = build_and_validate_packet(_VALID_6474)
        d1 = compute_packet_digest(packet)
        d2 = compute_packet_digest(packet)
        assert d1 == d2
        assert len(d1) == 64

    def test_post_review_digest_differs_when_reviewer_added(self) -> None:
        packet = build_and_validate_packet(_VALID_6474)
        pre_digest = compute_post_review_digest(packet)
        reviewed = build_and_validate_packet(
            {
                **_VALID_6474,
                "reviewer": {
                    "actor_id": "reviewer_1",
                    "commit": "abc1234",
                    "command": "review",
                    "status": "draft",
                },
            }
        )
        post_digest = compute_post_review_digest(reviewed)
        assert pre_digest != post_digest


# ---------------------------------------------------------------------------
# Fail-closed: missing denominator
# ---------------------------------------------------------------------------


class TestFailClosedMissingDenominator:
    def test_missing_denominator_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["denominator"] = 0
        errors = validate_packet(payload)
        assert any("denominator" in e for e in errors)

    def test_missing_analysis_unit_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["estimand"]["analysis_unit"] = ""
        errors = validate_packet(payload)
        assert any("analysis_unit" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: unsupported zero imputation (not_imputed missingness)
# ---------------------------------------------------------------------------


class TestFailClosedZeroImputation:
    def test_not_imputed_missingness_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["missingness"] = "not_imputed"
        errors = validate_packet(payload)
        assert any("not_imputed" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: undefined comparator direction
# ---------------------------------------------------------------------------


class TestFailClosedComparatorDirection:
    def test_missing_comparator_direction_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["estimand"]["comparator"].pop("direction")
        errors = validate_packet(payload)
        assert any("direction" in e and "required" in e for e in errors)

    def test_invalid_comparator_direction_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["estimand"]["comparator"]["direction"] = "bidirectional"
        errors = validate_packet(payload)
        assert any("direction" in e for e in errors)

    def test_valid_comparator_directions_accepted(self) -> None:
        for direction in (
            "comparison_minus_reference",
            "reference_minus_comparison",
            "not_applicable",
        ):
            payload = copy.deepcopy(_VALID_6474)
            payload["estimand"]["comparator"]["direction"] = direction
            errors = validate_packet(payload)
            assert errors == [], f"direction {direction!r} should be valid: {errors}"


# ---------------------------------------------------------------------------
# Fail-closed: inferential comparison without uncertainty
# ---------------------------------------------------------------------------


class TestFailClosedInferentialWithoutUncertainty:
    def test_declared_uncertainty_without_method_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["uncertainty"]["method"] = None
        errors = validate_packet(payload)
        assert any("uncertainty" in e and "method" in e for e in errors)

    def test_undeclared_uncertainty_allowed(self) -> None:
        payload = copy.deepcopy(_VALID_6944)
        payload["metrics"][0]["uncertainty"]["declared"] = False
        errors = validate_packet(payload)
        # Should still have other errors but not the uncertainty method one
        assert not any("uncertainty" in e and "method" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: unrecorded multiplicity
# ---------------------------------------------------------------------------


class TestFailClosedMultiplicity:
    def test_declared_multiplicity_without_method_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["multiplicity"]["method"] = None
        errors = validate_packet(payload)
        assert any("multiplicity" in e and "method" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: forbidden claim escalation (caption status)
# ---------------------------------------------------------------------------


class TestFailClosedCaptionAssertion:
    def test_inferred_caption_status_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["caption_assertions"][0]["status"] = "inferred"
        errors = validate_packet(payload)
        assert any("inferred" in e for e in errors)

    def test_observed_caption_status_accepted(self) -> None:
        payload = _available_ch7_payload()
        errors = validate_packet(payload)
        assert errors == [], f"observed should be valid: {errors}"

    def test_unavailable_caption_status_accepted(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["caption_assertions"][0]["status"] = "unavailable"
        errors = validate_packet(payload)
        assert errors == [], f"unavailable should be valid: {errors}"

    def test_unavailable_figure_cannot_have_observed_caption(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["caption_assertions"][0]["template_id"] = "observed_visualization.v1"
        payload["caption_assertions"][0]["assertion_text"] = (
            "Observed figure 'ch7_matched_start_process_contrast' for estimand "
            "'ch7_matched_planner_process_contrast' with direction 'not_applicable'."
        )
        payload["caption_assertions"][0]["status"] = "observed"
        errors = validate_packet(payload)
        assert any("unavailable figure" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: duplicate IDs
# ---------------------------------------------------------------------------


class TestFailClosedDuplicateIds:
    def test_duplicate_metric_id_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"].append(copy.deepcopy(payload["metrics"][0]))
        errors = validate_packet(payload)
        assert any("duplicate metric" in e or "non-unique" in e for e in errors)

    def test_duplicate_decision_id_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["decisions"].append(copy.deepcopy(payload["decisions"][0]))
        errors = validate_packet(payload)
        assert any("duplicate decision" in e or "non-unique" in e for e in errors)

    def test_duplicate_figure_link_id_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["figure_links"].append(copy.deepcopy(payload["figure_links"][0]))
        errors = validate_packet(payload)
        assert any("duplicate figure_link" in e or "non-unique" in e for e in errors)

    def test_duplicate_source_id_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["sources"].append(copy.deepcopy(payload["sources"][0]))
        errors = validate_packet(payload)
        assert any("duplicate source" in e or "non-unique" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: unsupported decision vocabulary
# ---------------------------------------------------------------------------


class TestFailClosedDecisionVocabulary:
    def test_invalid_decision_outcome_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["decisions"][0]["outcome"] = "probably_true"
        errors = validate_packet(payload)
        assert any("outcome" in e and "probably_true" in e for e in errors)

    @pytest.mark.parametrize(
        "outcome",
        ["supported", "not_supported", "inconclusive", "invalid", "unavailable"],
    )
    def test_valid_outcomes_accepted(self, outcome: str) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["decisions"][0]["outcome"] = outcome
        if outcome in ("supported",):
            payload["decisions"][0]["refusal_reason"] = None
        else:
            payload["decisions"][0]["refusal_reason"] = "test reason"
        errors = validate_packet(payload)
        assert errors == [], f"outcome {outcome!r} should be valid: {errors}"


# ---------------------------------------------------------------------------
# Fail-closed: source/packet digest drift after review
# ---------------------------------------------------------------------------


class TestFailClosedDigestDrift:
    def test_invalid_reviewed_packet_digest_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["reviewed_packet_digest"] = "not-a-valid-hex-digest"
        errors = validate_packet(payload)
        assert any("reviewed_packet_digest" in e for e in errors)

    def test_invalid_post_review_digest_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["post_review_digest"] = "zzz"
        errors = validate_packet(payload)
        assert any("post_review_digest" in e for e in errors)

    def test_valid_hex_digests_accepted(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["reviewer"] = {
            "actor_id": "reviewer_1",
            "commit": "ce9510737f87f6a52811c2257e9fd1e599cd6e46",
            "command": "review_result_interpretation_packet",
            "status": "draft",
        }
        draft_packet = build_and_validate_packet(payload)
        payload["reviewer"]["status"] = "reviewed"
        payload["reviewed_packet_digest"] = compute_packet_digest(draft_packet)
        reviewed_packet = replace(
            draft_packet,
            reviewer=replace(draft_packet.reviewer, status="reviewed"),
            reviewed_packet_digest=payload["reviewed_packet_digest"],
        )
        payload["post_review_digest"] = compute_post_review_digest(reviewed_packet)
        errors = validate_packet(payload)
        assert errors == [], f"hex digests should be valid: {errors}"

    def test_source_digest_mismatch_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["sources"][0]["sha256"] = "0" * 64
        errors = validate_packet(payload)
        assert any("digest mismatch" in e for e in errors)

    def test_tracked_source_commit_mismatch_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["sources"][0]["tracked_commit"] = payload["sources"][0]["commit"]
        errors = validate_packet(payload)
        assert any("tracked_commit" in e for e in errors)

    def test_nonfinite_uncertainty_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["uncertainty"]["ci_low"] = float("nan")
        errors = validate_packet(payload)
        assert any("ci_low" in e and "finite" in e for e in errors)

    def test_supported_fallback_rows_rejected_even_when_permitted(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["execution_mode"]["counts"] = {"native": 539, "fallback": 1}
        payload["execution_mode"]["fallback_permitted"] = True
        errors = validate_packet(payload)
        assert any("fallback/degraded rows" in e for e in errors)

    def test_supported_decision_below_threshold_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["support"] = 1
        errors = validate_packet(payload)
        assert any("below the declared support_threshold" in e for e in errors)

    def test_supported_partial_metric_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["missingness"] = "partial"
        errors = validate_packet(payload)
        assert any("complete metric data" in e for e in errors)

    def test_supported_uncertainty_without_observed_values_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["uncertainty"] = {
            "declared": True,
            "method": "percentile_bootstrap_95_ci",
            "ci_low": None,
            "ci_high": None,
            "p_value_raw": None,
            "p_value_adjusted": None,
        }
        errors = validate_packet(payload)
        assert any("observed uncertainty values" in e for e in errors)

    def test_supported_decision_requires_structured_contrast_result(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["decisions"][0].pop("contrast_result")
        errors = validate_packet(payload)
        assert any("decision-level contrast_result" in e for e in errors)

    def test_supported_decision_effect_must_match_structured_result(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["decisions"][0]["effect"] = 999999.0
        errors = validate_packet(payload)
        assert any("effect must match contrast_result.effect" in e for e in errors)

    def test_supported_decision_comparator_must_match_structured_result(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["decisions"][0]["comparator"]["comparison"] = "unrelated"
        errors = validate_packet(payload)
        assert any("comparator must match contrast_result.comparator" in e for e in errors)

    def test_execution_count_drift_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["execution_mode"]["counts"]["adapter"] = 359
        errors = validate_packet(payload)
        assert any("population included" in e for e in errors)

    def test_fallback_requires_explicit_permission(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["execution_mode"]["counts"] = {"native": 539, "fallback": 1}
        errors = validate_packet(payload)
        assert any("fallback_permitted" in e for e in errors)

    def test_review_digest_mismatch_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["reviewer"] = {
            "actor_id": "reviewer_1",
            "commit": "ce9510737f87f6a52811c2257e9fd1e599cd6e46",
            "command": "review_result_interpretation_packet",
            "status": "reviewed",
        }
        payload["reviewed_packet_digest"] = "a" * 64
        payload["post_review_digest"] = "b" * 64
        errors = validate_packet(payload)
        assert any("does not match" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: figure encoding with invalid sha256
# ---------------------------------------------------------------------------


class TestFailClosedFigureEncoding:
    def test_visual_contract_must_bind_packet_estimand(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["figure_links"][0]["visual_contract"]["estimand_id"] = "other_estimand"
        errors = validate_packet(payload)
        assert any("estimand_id" in e and "does not match" in e for e in errors)

    def test_visual_contract_requires_rationale(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["figure_links"][0]["visual_contract"].pop("rationale")
        errors = validate_packet(payload)
        assert errors

    def test_non_unavailable_figure_with_bad_sha256_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["figure_links"][0]["encoding"] = "png"
        payload["figure_links"][0]["sha256"] = "invalid"
        errors = validate_packet(payload)
        assert any("sha256" in e for e in errors)

    def test_unavailable_figure_with_any_sha256_accepted(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["figure_links"][0]["encoding"] = "unavailable"
        payload["figure_links"][0]["sha256"] = "0" * 64
        errors = validate_packet(payload)
        assert errors == [], f"unavailable encoding should allow any sha256: {errors}"

    def test_rendered_figure_must_exist(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["figure_links"][0]["encoding"] = "png"
        payload["figure_links"][0]["path"] = "tests/fixtures/does_not_exist.png"
        payload["figure_links"][0]["sha256"] = "0" * 64
        errors = validate_packet(payload)
        assert any("file does not exist" in e for e in errors)

    def test_readme_bytes_cannot_be_declared_as_pdf_figure(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        readme = FIXTURES_DIR / "README.md"
        payload["figure_links"][0]["encoding"] = "pdf"
        payload["figure_links"][0]["path"] = (
            "tests/fixtures/result_interpretation_packet/v1/README.md"
        )
        payload["figure_links"][0]["sha256"] = hashlib.sha256(readme.read_bytes()).hexdigest()
        errors = validate_packet(payload)
        assert any("suffix" in e or "bytes do not match" in e for e in errors)
        assert any("artifact_catalog" in e for e in errors)

    def test_available_figure_must_match_catalog_entry(self) -> None:
        payload = _available_ch7_payload()
        errors = validate_packet(payload)
        assert errors == [], f"catalog-bound fixture should validate: {errors}"


# ---------------------------------------------------------------------------
# Fail-closed: unsupported decision vocabulary (full enum)
# ---------------------------------------------------------------------------


class TestFailClosedUnsupportedDecisionVocabulary:
    @pytest.mark.parametrize(
        "outcome",
        ["probably_true", "likely", "maybe", "confirmed", "refuted"],
    )
    def test_non_vocabulary_outcome_rejected(self, outcome: str) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["decisions"][0]["outcome"] = outcome
        errors = validate_packet(payload)
        assert any("outcome" in e for e in errors), f"{outcome!r} should be rejected"


# ---------------------------------------------------------------------------
# Fail-closed: population accounting
# ---------------------------------------------------------------------------


class TestFailClosedPopulation:
    def test_included_plus_excluded_not_equal_total_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["population"]["included"] = 100
        payload["population"]["excluded"] = 200
        payload["population"]["total"] = 200
        errors = validate_packet(payload)
        assert any("included" in e and "excluded" in e and "total" in e for e in errors)

    def test_attrition_sum_not_equal_excluded_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["population"]["attrition"]["native"] = 999
        errors = validate_packet(payload)
        assert any("attrition sum" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: claim boundary
# ---------------------------------------------------------------------------


class TestFailClosedClaimBoundary:
    def test_empty_allowed_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["claim_boundary"]["allowed"] = []
        errors = validate_packet(payload)
        assert any("allowed" in e for e in errors)

    def test_empty_forbidden_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["claim_boundary"]["forbidden"] = []
        errors = validate_packet(payload)
        assert any("forbidden" in e for e in errors)

    def test_top_level_forbidden_claims_must_match_boundary(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["forbidden_claims"] = payload["forbidden_claims"][:-1]
        errors = validate_packet(payload)
        assert any("must exactly match" in e for e in errors)

    def test_allowed_claim_cannot_escalate_to_universal_superiority(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["claim_boundary"]["allowed"].append("The planner is universally superior.")
        errors = validate_packet(payload)
        assert any("universally superior" in e for e in errors)


class TestFailClosedCaptionBinding:
    def test_caption_binding_is_required(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["caption_assertions"][0]["bound_to_packet_fields"] = []
        errors = validate_packet(payload)
        assert any("bound_to_packet_fields" in e for e in errors)

    def test_caption_binding_must_name_known_packet_field(self) -> None:
        payload = copy.deepcopy(_VALID_CH7)
        payload["caption_assertions"][0]["bound_to_packet_fields"] = ["arbitrary_field"]
        errors = validate_packet(payload)
        assert any("unknown packet field" in e for e in errors)

    def test_caption_text_must_match_controlled_template(self) -> None:
        payload = _available_ch7_payload()
        payload["caption_assertions"][0]["assertion_text"] = (
            "PPO dominates every planner in every scenario and should be deployed."
        )
        errors = validate_packet(payload)
        assert any("generated observed_visualization.v1" in e for e in errors)


class TestFailClosedAdmission:
    def test_paper_grade_admission_requires_independent_review(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["evidence"]["tier"] = "paper_grade"
        payload["evidence"]["admission_state"] = "admitted"
        errors = validate_packet(payload)
        assert any("independent reviewer" in e for e in errors)

    def test_unknown_evidence_vocabulary_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["evidence"]["tier"] = "arbitrary_grade"
        errors = validate_packet(payload)
        assert any("evidence/tier" in e or "evidence.tier" in e for e in errors)

    def test_reviewer_must_be_independent(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["reviewer"] = {
            "actor_id": payload["producer"]["actor_id"],
            "commit": payload["producer"]["commit"],
            "command": "review_result_interpretation_packet",
            "status": "draft",
        }
        errors = validate_packet(payload)
        assert any("independent" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: decision references undeclared metric
# ---------------------------------------------------------------------------


class TestFailClosedUndeclaredMetric:
    def test_decision_references_undeclared_metric_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["decisions"][0]["metric_id"] = "nonexistent_metric"
        errors = validate_packet(payload)
        assert any("undeclared metric" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: support > denominator
# ---------------------------------------------------------------------------


class TestFailClosedSupportDenominator:
    def test_support_exceeding_denominator_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["support"] = 999
        payload["metrics"][0]["denominator"] = 100
        errors = validate_packet(payload)
        assert any("support" in e and "denominator" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: forbidden claim escalation via decision outcome
# ---------------------------------------------------------------------------


class TestFailClosedForbiddenClaimEscalation:
    def test_unsupported_metric_with_supported_outcome_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6944)
        payload["decisions"][0]["outcome"] = "supported"
        payload["decisions"][0]["rationale"] = "Test"
        payload["decisions"][0]["refusal_reason"] = None
        errors = validate_packet(payload)
        assert any("supported outcome requires" in e for e in errors)

    def test_non_supported_decision_rationale_cannot_escalate_claim(self) -> None:
        payload = copy.deepcopy(_VALID_6944)
        payload["decisions"][0]["rationale"] = "This proves a causal effect."
        errors = validate_packet(payload)
        assert any("decision" in e and "causal effect" in e for e in errors)


# ---------------------------------------------------------------------------
# Fail-closed: invalid desirability vocabulary
# ---------------------------------------------------------------------------


class TestFailClosedDesirability:
    def test_invalid_desirability_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["desirability"] = "good"
        errors = validate_packet(payload)
        assert any("desirability" in e for e in errors)


# ---------------------------------------------------------------------------
# Mutation tests for major failure classes
# ---------------------------------------------------------------------------


class TestMutationTests:
    """Comprehensive mutation tests for the major failure classes."""

    def test_mutate_question_id(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["question"]["question_id"] = ""
        errors = validate_packet(payload)
        assert errors  # JSON Schema should reject empty question_id

    def test_mutate_packet_id_pattern(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["packet_id"] = "123-bad-id!"
        errors = validate_packet(payload)
        assert errors

    def test_mutate_source_sha256(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["sources"][0]["sha256"] = "x" * 64
        errors = validate_packet(payload)
        assert any("sha256" in e for e in errors)

    def test_generation_command_must_bind_to_tracked_script(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["sources"][0]["command"] = "echo forged"
        errors = validate_packet(payload)
        assert any("command must name a tracked Python script" in e for e in errors)

    def test_generation_commit_must_contain_command_script(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["sources"][0]["command"] = (
            "uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py"
        )
        errors = validate_packet(payload)
        assert any("does not contain command script" in e for e in errors)

    def test_mutate_missingness_to_invalid_value(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["missingness"] = "imputed"
        errors = validate_packet(payload)
        assert any("missingness" in e for e in errors)

    def test_mutate_unavailable_handling_to_invalid(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["metrics"][0]["unavailable_handling"] = "ignore"
        errors = validate_packet(payload)
        assert any("unavailable_handling" in e for e in errors)

    def test_mutate_execution_mode_to_invalid(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["execution_mode"]["counts"]["hybrid"] = 1
        errors = validate_packet(payload)
        assert errors

    def test_local_only_source_path_rejected(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["sources"][0]["path"] = "results/derived.json"
        errors = validate_packet(payload)
        assert any("local-only" in e for e in errors)

    def test_mutate_actor_status_to_invalid(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["producer"]["status"] = "unknown"
        errors = validate_packet(payload)
        assert any("status" in e for e in errors)

    def test_mutate_schema_version(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["schema_version"] = "result_interpretation_packet.v99"
        errors = validate_packet(payload)
        assert errors

    def test_add_unexpected_field(self) -> None:
        payload = copy.deepcopy(_VALID_6474)
        payload["surprise_field"] = "hello"
        errors = validate_packet(payload)
        assert errors


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_validate_only(self, tmp_path: Path) -> None:
        from robot_sf.benchmark.result_interpretation_packet import main

        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_VALID_6474, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        rc = main(["--input", str(input_file), "--validate-only"])
        assert rc == 0

    def test_cli_invalid_input_returns_error(self, tmp_path: Path) -> None:
        from robot_sf.benchmark.result_interpretation_packet import main

        input_file = tmp_path / "bad.json"
        input_file.write_text('{"schema_version": "wrong"}', encoding="utf-8")
        rc = main(["--input", str(input_file)])
        assert rc == 1

    def test_cli_output_writes_file(self, tmp_path: Path) -> None:
        from robot_sf.benchmark.result_interpretation_packet import main

        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_VALID_6474, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        output_file = tmp_path / "output.json"
        rc = main(["--input", str(input_file), "--output", str(output_file)])
        assert rc == 0
        assert output_file.exists()

    def test_cli_show_digest(self, tmp_path: Path) -> None:
        from robot_sf.benchmark.result_interpretation_packet import main

        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_VALID_6474, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        rc = main(["--input", str(input_file), "--show-digest"])
        assert rc == 0


# ---------------------------------------------------------------------------
# Script CLI integration
# ---------------------------------------------------------------------------


class TestScriptCLI:
    def test_script_cli_validate_only(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_VALID_6474, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analysis/build_result_interpretation_packet.py",
                "--input",
                str(input_file),
                "--validate-only",
            ],
            capture_output=True,
            check=False,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert result.returncode == 0
        assert "is valid" in result.stdout

    def test_script_cli_writes_caption_review_and_checksum_outputs(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        input_file = tmp_path / "input.json"
        input_file.write_text(
            json.dumps(_VALID_CH7, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        output_file = tmp_path / "packet.json"
        caption_file = tmp_path / "caption.txt"
        review_file = tmp_path / "review.json"
        checksum_file = tmp_path / "SHA256SUMS"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/analysis/build_result_interpretation_packet.py",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--caption-output",
                str(caption_file),
                "--review-output",
                str(review_file),
                "--checksum-output",
                str(checksum_file),
            ],
            capture_output=True,
            check=False,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert result.returncode == 0, result.stderr
        assert output_file.exists()
        assert caption_file.exists()
        assert review_file.exists()
        assert checksum_file.exists()
        assert "admission=unavailable_causal_inference" in caption_file.read_text(encoding="utf-8")
        assert "packet.json" in checksum_file.read_text(encoding="utf-8")
        assert "source/portfolio_schema/" in checksum_file.read_text(encoding="utf-8")
        assert "post_review_digest: unavailable" in result.stdout
